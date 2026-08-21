#include "ncnn-mlir/Importer/NCNNImporter.hpp"

#include "ImporterInternal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"

namespace ncnn_importer {

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

namespace detail {
namespace {

using ImportHandler = ImportResult (*)(ImportContext&, const LayerContext&);

std::optional<std::int64_t> infer_input_channels(
  const ncnn_graph::Graph& source, std::string_view output) {
  for (const ncnn_graph::Layer& layer : source.get_layers()) {
    if (std::ranges::find(layer.get_inputs(), output) ==
          layer.get_inputs().end() ||
        layer.get_type() != "Convolution" || layer.get_weights().empty()) {
      continue;
    }
    const auto shape = layer.get_weights().front().get_shape();
    if (shape.size() == 4 && shape[1] > 0) {
      return shape[1];
    }
  }
  return std::nullopt;
}

void infer_input_shapes(const ncnn_graph::Graph& source,
                        ImportOptions& options) {
  if (options.input_shape || !options.input_shapes.empty()) {
    return;
  }
  for (const ncnn_graph::Layer& layer : source.get_layers()) {
    if (layer.get_type() != "Input") {
      continue;
    }
    constexpr std::array<int, 4> dimension_ids{0, 1, 2, 11};
    const bool omitted = std::ranges::all_of(
      dimension_ids, [&](int id) { return layer.get_param_int(id) == 0; });
    if (!omitted) {
      continue;
    }
    for (std::string_view output : layer.get_outputs()) {
      const std::int64_t channels = infer_input_channels(source, output)
                                      .value_or(ncnn_importer::kDynamicExtent);
      options.input_shapes.push_back({channels,
                                      ncnn_importer::kDynamicExtent,
                                      ncnn_importer::kDynamicExtent});
    }
  }
}

struct ImportEntry {
  std::string_view type;
  ImportHandler handler;
};

std::span<const ImportEntry> importers() noexcept {
  static constexpr std::array kImporters{
    ImportEntry{.type = "Input", .handler = import_input},
    ImportEntry{.type = "Convolution", .handler = import_convolution},
    ImportEntry{.type = "ConvolutionDepthWise",
                .handler = import_convolution_depthwise},
    ImportEntry{.type = "Deconvolution", .handler = import_deconvolution},
    ImportEntry{.type = "Padding", .handler = import_padding},
    ImportEntry{.type = "Interp", .handler = import_interp},
    ImportEntry{.type = "GridSample", .handler = import_grid_sample},
    ImportEntry{.type = "Sigmoid", .handler = import_sigmoid},
    ImportEntry{.type = "TanH", .handler = import_tanh},
    ImportEntry{.type = "MemoryData", .handler = import_memory_data},
    ImportEntry{.type = "Swish", .handler = import_swish},
    ImportEntry{.type = "LayerNorm", .handler = import_layer_norm},
    ImportEntry{.type = "Embed", .handler = import_embed},
    ImportEntry{.type = "Eltwise", .handler = import_eltwise},
    ImportEntry{.type = "MultiHeadAttention",
                .handler = import_multi_head_attention},
    ImportEntry{.type = "SDPA", .handler = import_sdpa},
    ImportEntry{.type = "DetectionOutput", .handler = import_detection_output},
    ImportEntry{.type = "HardSigmoid", .handler = import_hard_sigmoid},
    ImportEntry{.type = "HardSwish", .handler = import_hard_swish},
    ImportEntry{.type = "Reshape", .handler = import_reshape},
    ImportEntry{.type = "Flatten", .handler = import_flatten},
    ImportEntry{.type = "BinaryOp", .handler = import_binary_op},
    ImportEntry{.type = "InnerProduct", .handler = import_inner_product},
    ImportEntry{.type = "ShuffleChannel", .handler = import_shuffle_channel},
    ImportEntry{.type = "Slice", .handler = import_slice},
    ImportEntry{.type = "Reduction", .handler = import_reduction},
    ImportEntry{.type = "ReLU", .handler = import_relu},
    ImportEntry{.type = "PReLU", .handler = import_prelu},
    ImportEntry{.type = "Pooling", .handler = import_pooling},
    ImportEntry{.type = "Split", .handler = import_split},
    ImportEntry{.type = "Concat", .handler = import_concat},
    ImportEntry{.type = "Dropout", .handler = import_dropout},
    ImportEntry{.type = "Softmax", .handler = import_softmax},
    ImportEntry{.type = "GELU", .handler = import_gelu},
    ImportEntry{.type = "Squeeze", .handler = import_squeeze},
    ImportEntry{.type = "BatchNorm", .handler = import_batch_norm},
    ImportEntry{.type = "ExpandDims", .handler = import_expand_dims},
    ImportEntry{.type = "Permute", .handler = import_permute},
    ImportEntry{.type = "Gemm", .handler = import_gemm},
    ImportEntry{.type = "Quantize", .handler = import_quantize},
    ImportEntry{.type = "Dequantize", .handler = import_dequantize},
    ImportEntry{.type = "Requantize", .handler = import_requantize},
    ImportEntry{.type = "Cast", .handler = import_cast},
  };
  return kImporters;
}

std::expected<mlir::DenseElementsAttr, std::string> make_dense_attr(
  mlir::MLIRContext* context,
  const ncnn_graph::Tensor& tensor,
  ncnn_mlir::PrecisionMode precision,
  bool precision_storage) {
  const std::span<const std::byte> raw = tensor.get_data();
  if (raw.size() != tensor.byte_size()) {
    return std::unexpected("constant payload size does not match its type");
  }
  llvm::SmallVector<std::int64_t> shape(tensor.get_shape().begin(),
                                        tensor.get_shape().end());
  const bool targetFloat16 =
    precision_storage && precision == ncnn_mlir::PrecisionMode::Float16;
  const bool targetBFloat16 =
    precision_storage && precision == ncnn_mlir::PrecisionMode::BFloat16;
  if ((tensor.get_dtype() == ncnn_graph::DataType::Float16 && !targetFloat16) ||
      (tensor.get_dtype() == ncnn_graph::DataType::Float32 &&
       (targetFloat16 || targetBFloat16))) {
    const llvm::fltSemantics& targetSemantics =
      targetFloat16    ? llvm::APFloat::IEEEhalf()
      : targetBFloat16 ? llvm::APFloat::BFloat()
                       : llvm::APFloat::IEEEsingle();
    mlir::Type targetElement =
      targetFloat16 ? static_cast<mlir::Type>(mlir::Float16Type::get(context))
      : targetBFloat16
        ? static_cast<mlir::Type>(mlir::BFloat16Type::get(context))
        : static_cast<mlir::Type>(mlir::Float32Type::get(context));
    std::vector<llvm::APFloat> values;
    values.reserve(tensor.element_count());
    const std::size_t width =
      tensor.get_dtype() == ncnn_graph::DataType::Float16 ? 2 : 4;
    for (std::size_t offset = 0; offset < raw.size(); offset += width) {
      std::uint64_t bits = 0;
      for (std::size_t byte = 0; byte < width; ++byte) {
        bits |= static_cast<std::uint64_t>(raw[offset + byte]) << (byte * 8);
      }
      llvm::APFloat value(tensor.get_dtype() == ncnn_graph::DataType::Float16
                            ? llvm::APFloat::IEEEhalf()
                            : llvm::APFloat::IEEEsingle(),
                          llvm::APInt(static_cast<unsigned>(width * 8), bits));
      bool losesInfo = false;
      value.convert(
        targetSemantics, llvm::RoundingMode::NearestTiesToEven, &losesInfo);
      values.push_back(value);
    }
    auto type = mlir::RankedTensorType::get(shape, targetElement);
    return mlir::DenseElementsAttr::get(type, llvm::ArrayRef(values));
  }

  mlir::Type element;
  std::size_t width = 0;
  switch (tensor.get_dtype()) {
    case ncnn_graph::DataType::Float32:
      element = mlir::Float32Type::get(context);
      width = 4;
      break;
    case ncnn_graph::DataType::Float16:
      element = mlir::Float16Type::get(context);
      width = 2;
      break;
    case ncnn_graph::DataType::BFloat16:
      element = mlir::BFloat16Type::get(context);
      width = 2;
      break;
    case ncnn_graph::DataType::Int8:
      element = mlir::IntegerType::get(context, 8);
      width = 1;
      break;
    case ncnn_graph::DataType::Unknown:
      return std::unexpected("constant has unknown element type");
  }
  auto type = mlir::RankedTensorType::get(shape, element);
  std::vector<char> native(raw.size());
  std::memcpy(native.data(), raw.data(), raw.size());
  if constexpr (std::endian::native == std::endian::big) {
    for (std::size_t offset = 0; offset + width <= native.size();
         offset += width) {
      std::reverse(
        native.begin() + static_cast<std::ptrdiff_t>(offset),
        native.begin() + static_cast<std::ptrdiff_t>(offset + width));
    }
  }
  auto attr = mlir::DenseElementsAttr::getFromRawBuffer(
    type, llvm::ArrayRef<char>(native));
  if (!attr) {
    return std::unexpected("constant payload is not a valid dense buffer");
  }
  return attr;
}

}  // namespace

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
  return *value->get_int();
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
  if (!std::isfinite(*value->get_float())) {
    return std::unexpected(
      std::format("parameter {} ({}) must be finite", id, name));
  }
  return *value->get_float();
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

std::expected<void, std::string> validate_feature_mask(
  const ncnn_graph::ParamDict& params) {
  const auto* value = find_param(params, 31);
  if (value == nullptr) {
    return {};
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Int ||
      !std::in_range<int>(*value->get_int())) {
    return std::unexpected(
      "parameter 31 (feature mask) must fit the ncnn int type");
  }
  return {};
}

ImportContext::ImportContext(mlir::MLIRContext& context, ImportOptions options)
  : context_(&context),
    options_(std::move(options)),
    builder_(&context),
    module_(mlir::ModuleOp::create(builder_.getUnknownLoc())),
    model_(),
    blobs_(),
    integer_input_blobs_(),
    captured_diag_(),
    imported_input_count_(0) {
  builder_.setInsertionPointToEnd(module_->getBody());
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
ImportContext::run(const ncnn_graph::Graph& source) {
  for (const ncnn_graph::Layer& layer : source.get_layers()) {
    if (layer.get_type() == "Embed") {
      for (std::string_view input : layer.get_inputs()) {
        integer_input_blobs_.insert(input);
      }
    }
  }
  std::size_t input_count = 0;
  std::size_t omitted_input_count = 0;
  for (const ncnn_graph::Layer& layer : source.get_layers()) {
    if (layer.get_type() != "Input") {
      continue;
    }
    input_count += layer.get_outputs().size();
    bool omitted = true;
    for (int id : {0, 1, 2, 11}) {
      const ncnn_graph::ParamValue* value = find_param(layer.get_params(), id);
      if (value != nullptr &&
          (value->get_kind() != ncnn_graph::ParamValue::Kind::Int ||
           *value->get_int() != 0)) {
        omitted = false;
        break;
      }
    }
    if (omitted) {
      omitted_input_count += layer.get_outputs().size();
    }
  }
  if (options_.dynamic_rank) {
    return std::unexpected(
      ImportError(0,
                  "Input",
                  "input-shape",
                  "dynamic rank must be specialized before import"));
  }
  if (options_.input_shape && input_count != 1) {
    return std::unexpected(
      ImportError(0,
                  "Input",
                  "input-shape",
                  "input shape override requires exactly one Input output"));
  }
  if (options_.input_shape && !options_.input_shapes.empty()) {
    return std::unexpected(ImportError(0,
                                       "Input",
                                       "input-shape",
                                       "legacy input_shape and input_shapes "
                                       "cannot both be specified"));
  }
  infer_input_shapes(source, options_);
  if (!options_.input_shapes.empty() &&
      options_.input_shapes.size() != input_count &&
      options_.input_shapes.size() != omitted_input_count) {
    return std::unexpected(ImportError(
      0,
      "Input",
      "input-shape",
      std::format("input shape override count {} matches neither {} Input "
                  "outputs nor {} outputs with omitted dimensions",
                  options_.input_shapes.size(),
                  input_count,
                  omitted_input_count)));
  }
  sparse_input_shapes_ = !options_.input_shapes.empty() &&
                         options_.input_shapes.size() == omitted_input_count &&
                         omitted_input_count != input_count;
  auto prepared = prepare_model();
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  for (std::size_t index = 0; index < source.get_layers().size(); ++index) {
    auto imported =
      import_layer({.index = index, .layer = source.get_layers()[index]});
    if (!imported) {
      return std::unexpected(imported.error());
    }
  }
  return finish(source);
}

mlir::OpBuilder& ImportContext::builder() noexcept {
  return builder_;
}

const std::optional<ncnn_importer::InputShape>& ImportContext::input_shape()
  const noexcept {
  return options_.input_shape;
}

const std::vector<ncnn_importer::InputShape>& ImportContext::input_shapes()
  const noexcept {
  return options_.input_shapes;
}

mlir::ncnn::ShapeMode ImportContext::shape_mode(
  bool has_dynamic_input) const noexcept {
  return mlir::ncnn::classifyShapeMode(options_.rank_specialization.has_value(),
                                       has_dynamic_input);
}

std::optional<ncnn_importer::InputShape> ImportContext::next_input_shape(
  bool dimensions_omitted) noexcept {
  if (options_.input_shapes.empty()) {
    ++imported_input_count_;
    return options_.input_shape;
  }
  if (sparse_input_shapes_ && !dimensions_omitted) {
    return std::nullopt;
  }
  return options_.input_shapes[imported_input_count_++];
}

bool ImportContext::input_uses_integer_storage(
  std::string_view blob_name) const noexcept {
  return integer_input_blobs_.contains(blob_name);
}

std::expected<mlir::Value, ImportError> ImportContext::find_blob(
  const LayerContext& context, std::string_view name) {
  auto iterator = blobs_.find(name);
  if (iterator == blobs_.end()) {
    return std::unexpected(
      make_error(context, std::format("unresolved blob {}", name)));
  }
  return iterator->second;
}

ImportResult ImportContext::bind_blob(const LayerContext& context,
                                      std::string name,
                                      mlir::Value value) {
  auto inserted = blobs_.insert({name, value});
  if (!inserted.second) {
    return std::unexpected(make_error(
      context, std::format("blob {} has multiple definitions", name)));
  }
  return {};
}

std::expected<mlir::Value, ImportError> ImportContext::make_constant(
  const LayerContext& context,
  const ncnn_graph::Tensor& tensor,
  std::size_t weight_index,
  bool precision_storage) {
  auto attr = make_dense_attr(
    context_, tensor, options_.precision.mode, precision_storage);
  if (!attr) {
    return std::unexpected(make_error(context, attr.error()));
  }
  auto constant = builder_.create<mlir::ncnn::ConstOp>(
    builder_.getUnknownLoc(),
    attr->getType(),
    builder_.getStringAttr(
      std::format("{}.weight.{}", context.layer.get_name(), weight_index)),
    *attr);
  tag_source(constant.getOperation(), context);
  return constant.getOutput();
}

void ImportContext::tag_source(mlir::Operation* operation,
                               const LayerContext& context) {
  operation->setAttr("ncnn.name",
                     builder_.getStringAttr(context.layer.get_name()));
  operation->setAttr(
    "ncnn.source_layer",
    builder_.getI64IntegerAttr(static_cast<std::int64_t>(context.index)));
}

const std::string& ImportContext::captured_diagnostic() const noexcept {
  return captured_diag_;
}

ImportResult ImportContext::prepare_model() {
  model_ = builder_.create<mlir::ncnn::ModelOp>(
    builder_.getUnknownLoc(), builder_.getStringAttr("model"));
  model_->setAttr(
    "ncnn.precision",
    builder_.getStringAttr(precision_mode_name(options_.precision.mode)));
  model_->setAttr("ncnn.fp16_accumulator",
                  builder_.getStringAttr(fp16_accumulator_mode_name(
                    options_.precision.fp16_accumulator)));
  if (options_.precision.used_fallback) {
    model_->setAttr("ncnn.precision_fallback", builder_.getUnitAttr());
  }
  if (!options_.input_dim_constraints.empty()) {
    llvm::SmallVector<mlir::Attribute> constraints;
    constraints.reserve(options_.input_dim_constraints.size());
    for (const InputDimConstraint& constraint :
         options_.input_dim_constraints) {
      constraints.push_back(
        mlir::ncnn::DimConstraintAttr::get(context_,
                                           constraint.input,
                                           constraint.dimension,
                                           constraint.minimum,
                                           constraint.multiple_of));
    }
    model_->setAttr("ncnn.shape_constraints",
                    builder_.getArrayAttr(constraints));
  }
  if (options_.rank_specialization) {
    const std::uint32_t rank = *options_.rank_specialization;
    model_.setSymName(std::format("model_rank{}", rank));
    model_->setAttr("ncnn.rank_variant", builder_.getI32IntegerAttr(rank));
    model_->setAttr("ncnn.dynamic_rank", builder_.getUnitAttr());
  }
  mlir::Block* block = &model_.getBody().emplaceBlock();
  builder_.setInsertionPointToStart(block);
  return {};
}

ImportResult infer_shape_constraints(mlir::ncnn::ModelOp model) {
  std::map<std::pair<std::uint32_t, std::uint32_t>,
           ncnn_importer::InputDimConstraint>
    constraints;
  if (auto existing =
        model->getAttrOfType<mlir::ArrayAttr>("ncnn.shape_constraints")) {
    for (mlir::Attribute attribute : existing) {
      auto constraint = mlir::cast<mlir::ncnn::DimConstraintAttr>(attribute);
      constraints[{constraint.getInput(), constraint.getDim()}] = {
        .input = constraint.getInput(),
        .dimension = constraint.getDim(),
        .minimum = constraint.getMin(),
        .multiple_of = constraint.getMultipleOf()};
    }
  }

  mlir::WalkResult result = model.walk([&](mlir::ncnn::SliceOp operation) {
    auto requirement = mlir::ncnn::inferInputDimensionRequirement(operation);
    if (mlir::failed(requirement)) {
      return mlir::WalkResult::interrupt();
    }
    if (!*requirement) {
      return mlir::WalkResult::advance();
    }
    auto& constraint =
      constraints[{(*requirement)->input, (*requirement)->dimension}];
    constraint.input = (*requirement)->input;
    constraint.dimension = (*requirement)->dimension;
    constraint.minimum = std::max(constraint.minimum, (*requirement)->minimum);
    constraint.multiple_of = std::max<std::int64_t>(constraint.multiple_of, 1);
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return std::unexpected(
      ImportError(0, "ShapeInference", "constraints", "inference failed"));
  }
  if (constraints.empty()) {
    return {};
  }
  mlir::Builder builder(model.getContext());
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(constraints.size());
  for (const auto& [key, constraint] : constraints) {
    (void)key;
    attributes.push_back(
      mlir::ncnn::DimConstraintAttr::get(model.getContext(),
                                         constraint.input,
                                         constraint.dimension,
                                         constraint.minimum,
                                         constraint.multiple_of));
  }
  model->setAttr("ncnn.shape_constraints", builder.getArrayAttr(attributes));
  return {};
}

ImportResult ImportContext::import_layer(const LayerContext& context) {
  const auto entries = importers();
  const auto importer =
    std::ranges::find(entries, context.layer.get_type(), &ImportEntry::type);
  if (importer == entries.end()) {
    return std::unexpected(make_error(
      context,
      std::format("unsupported layer type {}", context.layer.get_type())));
  }
  return importer->handler(*this, context);
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
ImportContext::finish(const ncnn_graph::Graph& source) {
  llvm::SmallVector<mlir::Value> outputs;
  for (const auto& name : source.get_output_blob_names()) {
    auto value = blobs_.find(name);
    if (value == blobs_.end()) {
      return std::unexpected(ImportError(source.get_layers().size(),
                                         "GraphOutput",
                                         name,
                                         "unresolved output blob"));
    }
    outputs.push_back(value->second);
  }
  builder_.setInsertionPointToEnd(&model_.getBody().front());
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    builder_.create<mlir::ncnn::OutputOp>(
      builder_.getUnknownLoc(),
      outputs[index],
      builder_.getStringAttr(source.get_output_blob_names()[index]));
  }
  if (auto inferred = infer_shape_constraints(model_); !inferred) {
    return std::unexpected(inferred.error());
  }
  captured_diag_.clear();
  mlir::ScopedDiagnosticHandler handler(context_,
                                        [this](mlir::Diagnostic& diagnostic) {
                                          captured_diag_ = diagnostic.str();
                                          return mlir::success();
                                        });
  if (mlir::failed(mlir::verify(module_.get().getOperation()))) {
    return std::unexpected(ImportError(
      source.get_layers().size(),
      "Verifier",
      "module",
      captured_diag_.empty() ? "module verification failed" : captured_diag_));
  }
  return std::move(module_);
}

}  // namespace detail

bool has_layer_importer(std::string_view layer_type) noexcept {
  const auto entries = detail::importers();
  return std::ranges::find(entries, layer_type, &detail::ImportEntry::type) !=
         entries.end();
}

std::size_t get_layer_importer_count() noexcept {
  return detail::importers().size();
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> import_graph(
  const ncnn_graph::Graph& graph, mlir::MLIRContext& context) {
  return import_graph(graph, context, ImportOptions{});
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> import_graph(
  const ncnn_graph::Graph& graph,
  mlir::MLIRContext& context,
  const ImportOptions& options) {
  if (options.dynamic_rank) {
    std::size_t inputCount = 0;
    for (const ncnn_graph::Layer& layer : graph.get_layers()) {
      if (layer.get_type() == "Input") {
        inputCount += layer.get_outputs().size();
      }
    }
    if (inputCount != 1 || options.input_shape ||
        !options.input_shapes.empty() || options.rank_specialization ||
        !options.input_dim_constraints.empty()) {
      return std::unexpected(ImportError(
        0,
        "Input",
        "input-shape",
        "dynamic rank requires exactly one Input output and no fixed shape "
        "override"));
    }
    for (const ncnn_graph::Layer& layer : graph.get_layers()) {
      if (layer.get_type() != "Input" && layer.get_type() != "ReLU") {
        return std::unexpected(ImportError(
          0,
          std::string(layer.get_type()),
          std::string(layer.get_name()),
          "dynamic rank supports only identity/shape-preserving ReLU models"));
      }
      if (layer.get_type() == "Input" &&
          !layer.get_params().get_entries().empty()) {
        return std::unexpected(
          ImportError(0,
                      "Input",
                      std::string(layer.get_name()),
                      "dynamic rank requires omitted Input dimensions"));
      }
    }
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    for (std::uint32_t rank = 1; rank <= 4; ++rank) {
      ImportOptions specialized;
      specialized.precision = options.precision;
      specialized.input_shapes = {InputShape(rank, kDynamicExtent)};
      specialized.rank_specialization = rank;
      detail::ImportContext importer(context, specialized);
      auto imported = importer.run(graph);
      if (!imported) {
        return std::unexpected(ImportError(
          imported.error().get_layer_index(),
          std::string(imported.error().get_layer_type()),
          std::string(imported.error().get_layer_name()),
          std::format("dynamic-rank specialization {} is unsupported: {}",
                      rank,
                      imported.error().get_message())));
      }
      module.getBody()->getOperations().splice(
        module.getBody()->end(), imported->get().getBody()->getOperations());
    }
    return mlir::OwningOpRef<mlir::ModuleOp>(module);
  }
  detail::ImportContext importer(context, options);
  return importer.run(graph);
}

}  // namespace ncnn_importer
