#include "ncnn_frontend/importer.hpp"

#include "op_schema.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "ncnn_frontend/verifier.hpp"

namespace ncnn_frontend {
namespace {

class PendingValue {
 public:
  PendingValue(std::string name, TensorType type, ValueDef definition)
    : name_(std::move(name)),
      type_(std::move(type)),
      definition_(std::move(definition)),
      uses_() {}

  const TensorType& get_type() const noexcept { return type_; }
  void add_use(Use use) { uses_.push_back(use); }
  Value finish() {
    return {std::move(name_),
            std::move(type_),
            std::move(definition_),
            std::move(uses_)};
  }

 private:
  std::string name_;
  TensorType type_;
  ValueDef definition_;
  std::vector<Use> uses_;
};

struct LayerContext {
  std::size_t index;
  const ncnn_graph::Layer& layer;
};

ImportError make_error(const LayerContext& context, std::string message) {
  return {context.index,
          std::string(context.layer.get_type()),
          std::string(context.layer.get_name()),
          std::move(message)};
}

const ncnn_graph::ParamValue* find_param(const ncnn_graph::ParamDict& params,
                                         int id) {
  for (const auto& entry : params.get_entries()) {
    if (entry.first == id) {
      return &entry.second;
    }
  }
  return nullptr;
}

std::expected<void, std::string> validate_param_ids(
  const ncnn_graph::ParamDict& params, std::span<const int> allowed) {
  for (const auto& entry : params.get_entries()) {
    if ((entry.first == 30 || entry.first == 31) ||
        std::ranges::find(allowed, entry.first) != allowed.end()) {
      continue;
    }
    return std::unexpected(
      std::format("unsupported parameter id {}", entry.first));
  }
  return {};
}

std::expected<void, std::string> validate_shape_hint(
  const ncnn_graph::ParamDict& params,
  std::span<const TensorType> result_types,
  std::size_t output_count) {
  const auto* value = find_param(params, 30);
  if (value == nullptr) {
    return {};
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::IntArray) {
    return std::unexpected(
      "parameter 30 (shape hint) must be an integer array");
  }
  const auto hint = value->get_int_array();
  if (output_count == 0 || hint.empty() || hint.size() % output_count != 0) {
    return std::unexpected("shape hint length does not match outputs");
  }
  const std::size_t step = hint.size() / output_count;
  for (std::size_t output = 0; output < output_count; ++output) {
    const auto part = hint.subspan(output * step, step);
    const auto dims = part[0];
    std::vector<std::int64_t> shape;
    if (dims == 1 && step >= 2) {
      shape = {part[1]};
    } else if (dims == 2 && step >= 3) {
      shape = {part[2], part[1]};
    } else if (dims == 3 && step == 4) {
      shape = {part[3], part[2], part[1]};
    } else if (dims == 3 && step == 5) {
      shape = {part[4], part[2], part[1]};
    } else if (dims == 4 && step >= 5) {
      shape = {part[4], part[3], part[2], part[1]};
    } else {
      return std::unexpected("shape hint has invalid dimensions or stride");
    }
    const auto result_shape = result_types[output].get_shape();
    if (!std::ranges::equal(shape, result_shape)) {
      return std::unexpected(
        std::format("shape hint for output {} disagrees "
                    "with inferred result",
                    output));
    }
  }
  return {};
}

std::expected<void, std::string> validate_feature_mask(
  const ncnn_graph::ParamDict& params) {
  const auto* value = find_param(params, 31);
  if (value == nullptr) {
    return {};
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Int ||
      !std::in_range<int>(value->get_int())) {
    return std::unexpected(
      "parameter 31 (feature mask) must fit the ncnn int type");
  }
  return {};
}

std::expected<std::int64_t, std::string> get_int(
  const ncnn_graph::ParamDict& params,
  int id,
  std::int64_t default_value,
  std::string_view name) {
  const auto* value = find_param(params, id);
  if (value == nullptr) {
    return default_value;
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Int) {
    return std::unexpected(
      std::format("parameter {} ({}) must be integer", id, name));
  }
  return value->get_int();
}

std::expected<float, std::string> get_float(const ncnn_graph::ParamDict& params,
                                            int id,
                                            float default_value,
                                            std::string_view name) {
  const auto* value = find_param(params, id);
  if (value == nullptr) {
    return default_value;
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Float) {
    return std::unexpected(
      std::format("parameter {} ({}) must be float", id, name));
  }
  if (!std::isfinite(value->get_float())) {
    return std::unexpected(
      std::format("parameter {} ({}) must be finite", id, name));
  }
  return value->get_float();
}

std::expected<void, std::string> expect_boolean(std::int64_t value,
                                                std::string_view name) {
  if (value != 0 && value != 1) {
    return std::unexpected(std::format("{} must be 0 or 1", name));
  }
  return {};
}

std::expected<void, std::string> expect_source_arity(
  const ncnn_graph::Layer& layer, std::size_t inputs, std::size_t outputs) {
  if (layer.get_inputs().size() != inputs ||
      layer.get_outputs().size() != outputs) {
    return std::unexpected(
      std::format("requires {} inputs and {} outputs, got {} and {}",
                  inputs,
                  outputs,
                  layer.get_inputs().size(),
                  layer.get_outputs().size()));
  }
  return {};
}

std::expected<ElementType, std::string> convert_element_type(
  ncnn_graph::DataType type) {
  switch (type) {
    case ncnn_graph::DataType::Float32:
      return ElementType::Float32;
    case ncnn_graph::DataType::Float16:
      return ElementType::Float16;
    case ncnn_graph::DataType::Int8:
      return ElementType::Int8;
    case ncnn_graph::DataType::Unknown:
      return std::unexpected("constant has unknown element type");
  }
  return std::unexpected("constant has invalid element type");
}

class ImportState {
 public:
  ImportState()
    : operations_(), values_(), blob_values_(), inputs_(), outputs_() {}

  std::expected<void, ImportError> import_layer(const LayerContext& context) {
    const auto type = context.layer.get_type();
    if (type == "Input") {
      return import_input(context);
    }
    if (type == "Convolution") {
      return import_convolution(context);
    }
    if (type == "ReLU") {
      return import_relu(context);
    }
    if (type == "Pooling") {
      return import_pooling(context);
    }
    if (type == "Split") {
      return import_split(context);
    }
    if (type == "Concat") {
      return import_concat(context);
    }
    if (type == "Dropout") {
      return import_dropout(context);
    }
    if (type == "Softmax") {
      return import_softmax(context);
    }
    return std::unexpected(
      make_error(context, std::format("unsupported layer type {}", type)));
  }

  std::expected<Graph, ImportError> finish(const ncnn_graph::Graph& source) {
    if (source.get_input_blob_names().size() != inputs_.size()) {
      return std::unexpected(
        ImportError(source.get_layers().size(),
                    "GraphInput",
                    "typed_graph",
                    std::format("source lists {} inputs, imported {}",
                                source.get_input_blob_names().size(),
                                inputs_.size())));
    }
    for (std::size_t index = 0; index < inputs_.size(); ++index) {
      auto source_input = find_blob(source.get_input_blob_names()[index]);
      if (!source_input || *source_input != inputs_[index]) {
        return std::unexpected(
          ImportError(source.get_layers().size(),
                      "GraphInput",
                      source.get_input_blob_names()[index],
                      "input list disagrees with Input layers"));
      }
    }
    for (const auto& output_name : source.get_output_blob_names()) {
      auto output = find_blob(output_name);
      if (!output) {
        return std::unexpected(ImportError(source.get_layers().size(),
                                           "GraphOutput",
                                           output_name,
                                           output.error()));
      }
      outputs_.push_back(*output);
    }
    std::vector<Value> values;
    values.reserve(values_.size());
    for (auto& value : values_) {
      values.push_back(value.finish());
    }
    Graph graph(std::move(operations_),
                std::move(values),
                std::move(inputs_),
                std::move(outputs_));
    auto verified = verify_graph(graph);
    if (!verified) {
      return std::unexpected(ImportError(source.get_layers().size(),
                                         "Verifier",
                                         "typed_graph",
                                         verified.error()));
    }
    return graph;
  }

 private:
  std::expected<ValueId, std::string> find_blob(std::string_view name) const {
    auto iterator = std::ranges::find(
      blob_values_, name, [](const auto& entry) { return entry.first; });
    if (iterator == blob_values_.end()) {
      return std::unexpected(std::format("unresolved blob {}", name));
    }
    return iterator->second;
  }

  std::expected<void, std::string> bind_blob(std::string name, ValueId value) {
    if (find_blob(name)) {
      return std::unexpected(
        std::format("blob {} has multiple definitions", name));
    }
    blob_values_.emplace_back(std::move(name), value);
    return {};
  }

  std::expected<std::vector<ValueId>, std::string> resolve_operands(
    const ncnn_graph::Layer& layer) const {
    std::vector<ValueId> operands;
    operands.reserve(layer.get_inputs().size());
    for (const auto& input : layer.get_inputs()) {
      auto value = find_blob(input);
      if (!value) {
        return std::unexpected(value.error());
      }
      operands.push_back(*value);
    }
    return operands;
  }

  std::expected<ValueId, std::string> add_constant(
    const LayerContext& context,
    const ncnn_graph::Tensor& tensor,
    std::string role,
    TensorLayout layout) {
    auto element_type = convert_element_type(tensor.get_dtype());
    if (!element_type) {
      return std::unexpected(std::format("{}: {}", role, element_type.error()));
    }
    auto type =
      TensorType::create(std::vector<std::int64_t>(tensor.get_shape().begin(),
                                                   tensor.get_shape().end()),
                         *element_type,
                         layout);
    if (!type) {
      return std::unexpected(std::format("{} type: {}", role, type.error()));
    }
    auto literal =
      TensorLiteral::create(*type,
                            std::vector<std::byte>(tensor.get_data().begin(),
                                                   tensor.get_data().end()));
    if (!literal) {
      return std::unexpected(
        std::format("{} payload: {}", role, literal.error()));
    }
    const OpId op_id(operations_.size());
    const ValueId value_id(values_.size());
    const std::string name =
      std::format("{}.{}", context.layer.get_name(), role);
    values_.emplace_back(name, *type, OpResultDef(op_id, 0));
    operations_.emplace_back(name,
                             ConstOp(std::move(*literal)),
                             std::vector<ValueId>(),
                             std::vector<ValueId>{value_id},
                             context.index);
    return value_id;
  }

  std::expected<void, std::string> add_operation(
    const LayerContext& context,
    OperationAttributes attributes,
    std::vector<ValueId> operands) {
    std::vector<TensorType> operand_types;
    operand_types.reserve(operands.size());
    for (const ValueId operand : operands) {
      if (operand.get_index() >= values_.size()) {
        return std::unexpected("internal operand ValueId is out of range");
      }
      operand_types.push_back(values_[operand.get_index()].get_type());
    }
    auto inferred = infer_and_verify_operation(
      attributes, operand_types, context.layer.get_outputs().size());
    if (!inferred) {
      return std::unexpected(inferred.error());
    }
    auto shape_hint = validate_shape_hint(context.layer.get_params(),
                                          *inferred,
                                          context.layer.get_outputs().size());
    if (!shape_hint) {
      return std::unexpected(shape_hint.error());
    }
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(feature_mask.error());
    }
    const OpId op_id(operations_.size());
    std::vector<ValueId> results;
    results.reserve(inferred->size());
    for (std::size_t result_index = 0; result_index < inferred->size();
         ++result_index) {
      const ValueId value_id(values_.size());
      values_.emplace_back(context.layer.get_outputs()[result_index],
                           (*inferred)[result_index],
                           OpResultDef(op_id, result_index));
      results.push_back(value_id);
    }
    for (std::size_t operand_index = 0; operand_index < operands.size();
         ++operand_index) {
      values_[operands[operand_index].get_index()].add_use(
        Use(op_id, operand_index));
    }
    operations_.emplace_back(std::string(context.layer.get_name()),
                             std::move(attributes),
                             std::move(operands),
                             results,
                             context.index);
    for (std::size_t result_index = 0; result_index < results.size();
         ++result_index) {
      auto bound = bind_blob(context.layer.get_outputs()[result_index],
                             results[result_index]);
      if (!bound) {
        return bound;
      }
    }
    return {};
  }

  std::expected<void, ImportError> import_input(const LayerContext& context) {
    auto arity = expect_source_arity(context.layer, 0, 1);
    if (!arity) {
      return std::unexpected(make_error(context, arity.error()));
    }
    constexpr int kAllowed[] = {0, 1, 2, 11};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    if (!allowed) {
      return std::unexpected(make_error(context, allowed.error()));
    }
    auto width = get_int(context.layer.get_params(), 0, 0, "width");
    auto height = get_int(context.layer.get_params(), 1, 0, "height");
    auto channels = get_int(context.layer.get_params(), 2, 0, "channels");
    auto depth = get_int(context.layer.get_params(), 11, 0, "depth");
    if (!width || !height || !channels || !depth) {
      return std::unexpected(make_error(context,
                                        !width      ? width.error()
                                        : !height   ? height.error()
                                        : !channels ? channels.error()
                                                    : depth.error()));
    }
    if (*width <= 0 || *height <= 0 || *channels <= 0 || *depth != 0) {
      return std::unexpected(make_error(
        context,
        "Input requires positive w/h/c and unsupported depth must be 0"));
    }
    auto type = TensorType::create({*channels, *height, *width},
                                   ElementType::Float32,
                                   TensorLayout::NcnnCHW);
    if (!type) {
      return std::unexpected(make_error(context, type.error()));
    }
    const std::array result_types = {*type};
    auto shape_hint = validate_shape_hint(context.layer.get_params(),
                                          result_types,
                                          context.layer.get_outputs().size());
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!shape_hint || !feature_mask) {
      return std::unexpected(make_error(
        context, !shape_hint ? shape_hint.error() : feature_mask.error()));
    }
    const ValueId value_id(values_.size());
    const std::size_t input_index = inputs_.size();
    values_.emplace_back(context.layer.get_outputs()[0],
                         std::move(*type),
                         GraphInputDef(input_index));
    auto bound = bind_blob(context.layer.get_outputs()[0], value_id);
    if (!bound) {
      return std::unexpected(make_error(context, bound.error()));
    }
    inputs_.push_back(value_id);
    return {};
  }

  std::expected<void, ImportError> import_convolution(
    const LayerContext& context) {
    auto arity = expect_source_arity(context.layer, 1, 1);
    if (!arity) {
      return std::unexpected(make_error(context, arity.error()));
    }
    constexpr int kAllowed[] = {0,  1,  2,  3,  4,  5,  6,  8,  9,  10,
                                11, 12, 13, 14, 15, 16, 18, 19, 30, 31};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    if (!allowed) {
      return std::unexpected(make_error(context, allowed.error()));
    }
    const auto& params = context.layer.get_params();
    auto output_channels = get_int(params, 0, 0, "num_output");
    auto kernel_width = get_int(params, 1, 0, "kernel_w");
    if (!output_channels || !kernel_width) {
      return std::unexpected(make_error(
        context,
        !output_channels ? output_channels.error() : kernel_width.error()));
    }
    auto kernel_height = get_int(params, 11, *kernel_width, "kernel_h");
    auto dilation_width = get_int(params, 2, 1, "dilation_w");
    auto dilation_height = get_int(params, 12, *dilation_width, "dilation_h");
    auto stride_width = get_int(params, 3, 1, "stride_w");
    auto stride_height = get_int(params, 13, *stride_width, "stride_h");
    auto pad_left = get_int(params, 4, 0, "pad_left");
    auto pad_right = get_int(params, 15, *pad_left, "pad_right");
    auto pad_top = get_int(params, 14, *pad_left, "pad_top");
    auto pad_bottom = get_int(params, 16, *pad_top, "pad_bottom");
    auto bias = get_int(params, 5, 0, "bias_term");
    auto weight_count = get_int(params, 6, 0, "weight_data_size");
    auto int8_scale = get_int(params, 8, 0, "int8_scale_term");
    auto activation = get_int(params, 9, 0, "activation_type");
    auto dynamic = get_int(params, 19, 0, "dynamic_weight");
    auto pad_value = get_float(params, 18, 0.0f, "pad_value");
    if (!kernel_height || !dilation_width || !dilation_height ||
        !stride_width || !stride_height || !pad_left || !pad_right ||
        !pad_top || !pad_bottom || !bias || !weight_count || !int8_scale ||
        !activation || !dynamic || !pad_value) {
      const std::expected<std::int64_t, std::string>* integers[] = {
        &kernel_height,
        &dilation_width,
        &dilation_height,
        &stride_width,
        &stride_height,
        &pad_left,
        &pad_right,
        &pad_top,
        &pad_bottom,
        &bias,
        &weight_count,
        &int8_scale,
        &activation,
        &dynamic};
      for (const auto* result : integers) {
        if (!*result) {
          return std::unexpected(make_error(context, result->error()));
        }
      }
      return std::unexpected(make_error(context, pad_value.error()));
    }
    if (*output_channels <= 0 || *kernel_width <= 0 || *kernel_height <= 0 ||
        *dilation_width <= 0 || *dilation_height <= 0 || *stride_width <= 0 ||
        *stride_height <= 0 || *weight_count <= 0) {
      return std::unexpected(make_error(
        context,
        "output channels, kernel, dilation, stride, and weight count must be "
        "positive"));
    }
    auto bias_valid = expect_boolean(*bias, "bias_term");
    auto dynamic_valid = expect_boolean(*dynamic, "dynamic_weight");
    if (!bias_valid || !dynamic_valid) {
      return std::unexpected(make_error(
        context, !bias_valid ? bias_valid.error() : dynamic_valid.error()));
    }
    if (*dynamic != 0 || *activation != 0 ||
        find_param(params, 10) != nullptr || *pad_value != 0.0f) {
      return std::unexpected(
        make_error(context,
                   "dynamic weights, fused activation, and nonzero pad value "
                   "are unsupported"));
    }
    if (*int8_scale != 0 && *int8_scale != 1 && *int8_scale != 2 &&
        *int8_scale != 101 && *int8_scale != 102) {
      return std::unexpected(
        make_error(context, "int8_scale_term must be 0, 1, 2, 101, or 102"));
    }
    const std::size_t expected_weights = 1 + static_cast<std::size_t>(*bias) +
                                         (*int8_scale == 0 ? 0 : 2) +
                                         (*int8_scale > 100 ? 1 : 0);
    if (context.layer.get_weights().size() != expected_weights) {
      return std::unexpected(
        make_error(context,
                   std::format("expected {} weight tensors, got {}",
                               expected_weights,
                               context.layer.get_weights().size())));
    }
    const auto weight_dtype = context.layer.get_weights()[0].get_dtype();
    const bool quantized = *int8_scale != 0;
    if ((!quantized && weight_dtype == ncnn_graph::DataType::Int8) ||
        (quantized && weight_dtype == ncnn_graph::DataType::Float16)) {
      return std::unexpected(make_error(
        context, "convolution kernel element type does not match scale term"));
    }
    const auto weight_shape = context.layer.get_weights()[0].get_shape();
    if (weight_shape.size() != 4 || weight_shape[0] != *output_channels ||
        weight_shape[2] != *kernel_height || weight_shape[3] != *kernel_width ||
        context.layer.get_weights()[0].element_count() !=
          static_cast<std::size_t>(*weight_count)) {
      return std::unexpected(make_error(
        context, "kernel shape or weight_data_size is inconsistent"));
    }
    auto operands = resolve_operands(context.layer);
    if (!operands) {
      return std::unexpected(make_error(context, operands.error()));
    }
    const char* roles[] = {
      "weight", "bias", "weight_scale", "bottom_scale", "top_scale"};
    for (std::size_t index = 0; index < expected_weights; ++index) {
      const TensorLayout layout =
        index == 0 ? TensorLayout::OIHW : TensorLayout::NcnnW;
      std::size_t role_index = index;
      if (*bias == 0 && index > 0) {
        ++role_index;
      }
      auto constant = add_constant(
        context, context.layer.get_weights()[index], roles[role_index], layout);
      if (!constant) {
        return std::unexpected(make_error(context, constant.error()));
      }
      operands->push_back(*constant);
    }
    Conv2DOp attributes(*kernel_height,
                        *kernel_width,
                        *stride_height,
                        *stride_width,
                        *dilation_height,
                        *dilation_width,
                        *pad_top,
                        *pad_bottom,
                        *pad_left,
                        *pad_right,
                        *bias == 1,
                        *int8_scale);
    auto added =
      add_operation(context, std::move(attributes), std::move(*operands));
    if (!added) {
      return std::unexpected(make_error(context, added.error()));
    }
    return {};
  }

  std::expected<void, ImportError> import_relu(const LayerContext& context) {
    auto arity = expect_source_arity(context.layer, 1, 1);
    constexpr int kAllowed[] = {0};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    auto slope =
      get_float(context.layer.get_params(), 0, 0.0f, "negative_slope");
    if (!arity || !allowed || !slope) {
      return std::unexpected(make_error(context,
                                        !arity     ? arity.error()
                                        : !allowed ? allowed.error()
                                                   : slope.error()));
    }
    return import_simple(context, ReluOp(*slope));
  }

  std::expected<void, ImportError> import_pooling(const LayerContext& context) {
    auto arity = expect_source_arity(context.layer, 1, 1);
    if (!arity) {
      return std::unexpected(make_error(context, arity.error()));
    }
    constexpr int kAllowed[] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 30, 31};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    if (!allowed) {
      return std::unexpected(make_error(context, allowed.error()));
    }
    const auto& params = context.layer.get_params();
    auto kind = get_int(params, 0, 0, "pooling_type");
    auto kernel_width = get_int(params, 1, 0, "kernel_w");
    if (!kind || !kernel_width) {
      return std::unexpected(
        make_error(context, !kind ? kind.error() : kernel_width.error()));
    }
    auto kernel_height = get_int(params, 11, *kernel_width, "kernel_h");
    auto stride_width = get_int(params, 2, 1, "stride_w");
    auto stride_height = get_int(params, 12, *stride_width, "stride_h");
    auto pad_left = get_int(params, 3, 0, "pad_left");
    auto pad_right = get_int(params, 14, *pad_left, "pad_right");
    auto pad_top = get_int(params, 13, *pad_left, "pad_top");
    auto pad_bottom = get_int(params, 15, *pad_top, "pad_bottom");
    auto global = get_int(params, 4, 0, "global_pooling");
    auto pad_mode = get_int(params, 5, 0, "pad_mode");
    auto include_pad = get_int(params, 6, 0, "avgpool_count_include_pad");
    auto adaptive = get_int(params, 7, 0, "adaptive_pooling");
    auto output_width = get_int(params, 8, 0, "out_w");
    auto output_height = get_int(params, 18, *output_width, "out_h");
    const std::expected<std::int64_t, std::string>* results[] = {
      &kernel_height,
      &stride_width,
      &stride_height,
      &pad_left,
      &pad_right,
      &pad_top,
      &pad_bottom,
      &global,
      &pad_mode,
      &include_pad,
      &adaptive,
      &output_width,
      &output_height};
    for (const auto* result : results) {
      if (!*result) {
        return std::unexpected(make_error(context, result->error()));
      }
    }
    if ((*kind != 0 && *kind != 1) ||
        !expect_boolean(*global, "global_pooling") ||
        !expect_boolean(*adaptive, "adaptive_pooling") ||
        !expect_boolean(*include_pad, "avgpool_count_include_pad")) {
      return std::unexpected(make_error(
        context, "pooling kind or boolean mode parameter is invalid"));
    }
    if (*pad_mode < 0 || *pad_mode > 3) {
      return std::unexpected(
        make_error(context, "pooling pad_mode must be in [0, 3]"));
    }
    PoolMode mode = PoolMode::Regular;
    if (*global == 1) {
      mode = PoolMode::Global;
    } else if (*adaptive == 1) {
      mode = PoolMode::Adaptive;
      kernel_width = *output_width;
      kernel_height = *output_height;
    }
    Pool2DOp attributes(*kind == 0 ? PoolKind::Maximum : PoolKind::Average,
                        mode,
                        *kernel_height,
                        *kernel_width,
                        *stride_height,
                        *stride_width,
                        *pad_top,
                        *pad_bottom,
                        *pad_left,
                        *pad_right,
                        static_cast<int>(*pad_mode),
                        *include_pad == 1);
    return import_simple(context, std::move(attributes));
  }

  std::expected<void, ImportError> import_split(const LayerContext& context) {
    if (context.layer.get_inputs().size() != 1 ||
        context.layer.get_outputs().size() < 2) {
      return std::unexpected(make_error(
        context, "Split requires one input and at least two outputs"));
    }
    constexpr std::array<int, 0> kAllowed = {};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    if (!allowed) {
      return std::unexpected(make_error(context, allowed.error()));
    }
    return import_simple(context, SplitOp());
  }

  std::expected<void, ImportError> import_concat(const LayerContext& context) {
    if (context.layer.get_inputs().size() < 2 ||
        context.layer.get_outputs().size() != 1) {
      return std::unexpected(make_error(
        context, "Concat requires at least two inputs and one output"));
    }
    constexpr int kAllowed[] = {0};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    auto axis = get_int(context.layer.get_params(), 0, 0, "axis");
    if (!allowed || !axis) {
      return std::unexpected(
        make_error(context, !allowed ? allowed.error() : axis.error()));
    }
    return import_simple(context, ConcatOp(*axis));
  }

  std::expected<void, ImportError> import_dropout(const LayerContext& context) {
    auto arity = expect_source_arity(context.layer, 1, 1);
    constexpr int kAllowed[] = {0};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    auto scale = get_float(context.layer.get_params(), 0, 1.0f, "scale");
    if (!arity || !allowed || !scale) {
      return std::unexpected(make_error(context,
                                        !arity     ? arity.error()
                                        : !allowed ? allowed.error()
                                                   : scale.error()));
    }
    return import_simple(context, DropoutOp(*scale));
  }

  std::expected<void, ImportError> import_softmax(const LayerContext& context) {
    auto arity = expect_source_arity(context.layer, 1, 1);
    constexpr int kAllowed[] = {0, 1};
    auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
    auto axis = get_int(context.layer.get_params(), 0, 0, "axis");
    auto fixbug = get_int(context.layer.get_params(), 1, 0, "fixbug0");
    if (!arity || !allowed || !axis || !fixbug) {
      return std::unexpected(make_error(context,
                                        !arity     ? arity.error()
                                        : !allowed ? allowed.error()
                                        : !axis    ? axis.error()
                                                   : fixbug.error()));
    }
    if (*fixbug != 0 && *fixbug != 1) {
      return std::unexpected(make_error(context, "fixbug0 must be 0 or 1"));
    }
    if (*fixbug == 0 && *axis != 0) {
      return std::unexpected(make_error(
        context, "legacy Softmax only supports axis 0 unless fixbug0 is 1"));
    }
    return import_simple(context, SoftmaxOp(*axis));
  }

  std::expected<void, ImportError> import_simple(
    const LayerContext& context, OperationAttributes attributes) {
    auto operands = resolve_operands(context.layer);
    if (!operands) {
      return std::unexpected(make_error(context, operands.error()));
    }
    auto added =
      add_operation(context, std::move(attributes), std::move(*operands));
    if (!added) {
      return std::unexpected(make_error(context, added.error()));
    }
    return {};
  }

  std::vector<Operation> operations_;
  std::vector<PendingValue> values_;
  std::vector<std::pair<std::string, ValueId>> blob_values_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
};

}  // namespace

ImportError::ImportError(std::size_t layer_index,
                         std::string layer_type,
                         std::string layer_name,
                         std::string message)
  : layer_index_(layer_index),
    layer_type_(std::move(layer_type)),
    layer_name_(std::move(layer_name)),
    message_(std::move(message)) {}

std::size_t ImportError::get_layer_index() const noexcept {
  return layer_index_;
}

std::string_view ImportError::get_layer_type() const noexcept {
  return layer_type_;
}

std::string_view ImportError::get_layer_name() const noexcept {
  return layer_name_;
}

std::string_view ImportError::get_message() const noexcept {
  return message_;
}

std::string ImportError::to_string() const {
  return std::format(
    "layer {} ({}, {}): {}", layer_index_, layer_type_, layer_name_, message_);
}

std::expected<Graph, ImportError> import_graph(const ncnn_graph::Graph& graph) {
  ImportState state;
  for (std::size_t index = 0; index < graph.get_layers().size(); ++index) {
    auto imported = state.import_layer(
      LayerContext{.index = index, .layer = graph.get_layers()[index]});
    if (!imported) {
      return std::unexpected(imported.error());
    }
  }
  return state.finish(graph);
}

}  // namespace ncnn_frontend
