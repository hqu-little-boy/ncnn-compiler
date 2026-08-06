#include "ncnn-mlir/Importer/NCNNImporter.hpp"

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
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace ncnn_importer {
namespace {

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

// 从 ncnn 权重张量（原始小端字节）构造 DenseElementsAttr。宿主为大端时按元素
// 宽度逐元素字节翻转，保证跨端正确。
std::expected<mlir::DenseElementsAttr, std::string> make_dense_attr(
  mlir::MLIRContext* context, const ncnn_graph::Tensor& tensor) {
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
    case ncnn_graph::DataType::Int8:
      element = mlir::IntegerType::get(context, 8);
      width = 1;
      break;
    case ncnn_graph::DataType::Unknown:
      return std::unexpected("constant has unknown element type");
  }
  llvm::SmallVector<std::int64_t> shape(tensor.get_shape().begin(),
                                        tensor.get_shape().end());
  auto type = mlir::RankedTensorType::get(shape, element);
  const std::span<const std::byte> raw = tensor.get_data();
  if (raw.size() != tensor.byte_size()) {
    return std::unexpected("constant payload size does not match its type");
  }
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

namespace {

class ImportState {
 public:
  explicit ImportState(mlir::MLIRContext& context)
    : context_(&context),
      builder_(&context),
      module_(mlir::ModuleOp::create(builder_.getUnknownLoc())),
      model_(),
      blobs_(),
      captured_diag_() {
    builder_.setInsertionPointToEnd(module_->getBody());
  }

  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> run(
    const ncnn_graph::Graph& source) {
    auto prepared = prepare_model();
    if (!prepared) {
      return std::unexpected(prepared.error());
    }
    for (std::size_t index = 0; index < source.get_layers().size(); ++index) {
      const auto& layer = source.get_layers()[index];
      auto imported =
        import_layer(LayerContext{.index = index, .layer = layer});
      if (!imported) {
        return std::unexpected(imported.error());
      }
    }
    return finish(source);
  }

 private:
  using ImportHandler =
    std::expected<void, ImportError> (ImportState::*)(const LayerContext&);

  struct ImportEntry {
    std::string_view type;
    ImportHandler handler;
  };

  std::expected<void, ImportError> prepare_model() {
    model_ = builder_.create<mlir::ncnn::ModelOp>(
      builder_.getUnknownLoc(), builder_.getStringAttr("model"));
    mlir::Block* block = &model_.getBody().emplaceBlock();
    builder_.setInsertionPointToStart(block);
    return {};
  }

  std::expected<mlir::Type, ImportError> make_input_type(
    const LayerContext& context) {
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
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    return mlir::RankedTensorType::get(
      llvm::SmallVector<std::int64_t>{*channels, *height, *width},
      mlir::Float32Type::get(context_));
  }

  std::expected<void, ImportError> import_layer(const LayerContext& context) {
    static constexpr std::array kImporters{
      ImportEntry{.type = "Input", .handler = &ImportState::import_input},
      ImportEntry{.type = "Convolution",
                  .handler = &ImportState::import_convolution},
      ImportEntry{.type = "ReLU", .handler = &ImportState::import_relu},
      ImportEntry{.type = "Pooling", .handler = &ImportState::import_pooling},
      ImportEntry{.type = "Split", .handler = &ImportState::import_split},
      ImportEntry{.type = "Concat", .handler = &ImportState::import_concat},
      ImportEntry{.type = "Dropout", .handler = &ImportState::import_dropout},
      ImportEntry{.type = "Softmax", .handler = &ImportState::import_softmax},
    };

    const auto type = context.layer.get_type();
    const ImportEntry* const importer =
      std::ranges::find(kImporters, type, &ImportEntry::type);
    if (importer != kImporters.end()) {
      return (this->*importer->handler)(context);
    }
    return std::unexpected(
      make_error(context, std::format("unsupported layer type {}", type)));
  }

  std::expected<void, ImportError> import_input(const LayerContext& context) {
    auto type = make_input_type(context);
    if (!type) {
      return std::unexpected(type.error());
    }
    auto input = builder_.create<mlir::ncnn::InputOp>(
      builder_.getUnknownLoc(),
      *type,
      builder_.getStringAttr(context.layer.get_name()),
      builder_.getStringAttr(context.layer.get_outputs()[0]));
    tag_source(input.getOperation(), context);
    return bind_blob(
      context, std::string(context.layer.get_outputs()[0]), input.getOutput());
  }

  std::expected<mlir::Value, ImportError> find_blob(const LayerContext& context,
                                                    std::string_view name) {
    auto it = blobs_.find(name);
    if (it == blobs_.end()) {
      return std::unexpected(
        make_error(context, std::format("unresolved blob {}", name)));
    }
    return it->second;
  }

  std::expected<void, ImportError> bind_blob(const LayerContext& context,
                                             std::string name,
                                             mlir::Value value) {
    auto inserted = blobs_.insert({name, value});
    if (!inserted.second) {
      return std::unexpected(make_error(
        context, std::format("blob {} has multiple definitions", name)));
    }
    return {};
  }

  // 将 .bin 中隐式挂在 layer 上的权重显式化为模型内 SSA 定义。
  std::expected<mlir::Value, ImportError> make_constant(
    const LayerContext& context,
    const ncnn_graph::Tensor& tensor,
    std::size_t weight_index) {
    auto attr = make_dense_attr(context_, tensor);
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

  // 在算子上保留来源 ncnn 层名/层号（discardable 属性），便于回溯与调试，
  // 对应原自定义 IR 的 name / source_layer 字段。
  void tag_source(mlir::Operation* op, const LayerContext& context) {
    op->setAttr("ncnn.name", builder_.getStringAttr(context.layer.get_name()));
    op->setAttr(
      "ncnn.source_layer",
      builder_.getI64IntegerAttr(static_cast<std::int64_t>(context.index)));
  }

  // 在捕获 MLIR 诊断的情况下运行 fallible 推断，失败时把诊断写进
  // captured_diag_。
  template <typename Fn>
  auto capturing(Fn&& fn) {
    captured_diag_.clear();
    mlir::ScopedDiagnosticHandler handler(context_,
                                          [this](mlir::Diagnostic& diagnostic) {
                                            captured_diag_ = diagnostic.str();
                                            return mlir::success();
                                          });
    return fn();
  }

  template <typename Op>
  mlir::FailureOr<mlir::RankedTensorType> inferSingleTensorResult(
    mlir::Location location,
    mlir::ValueRange operands,
    typename Op::Properties& properties) {
    llvm::SmallVector<mlir::Type, 1> inferredTypes;
    if (mlir::failed(Op::inferReturnTypes(context_,
                                          location,
                                          operands,
                                          mlir::DictionaryAttr{},
                                          mlir::OpaqueProperties(&properties),
                                          mlir::RegionRange{},
                                          inferredTypes)) ||
        inferredTypes.size() != 1) {
      return mlir::failure();
    }
    auto result = llvm::dyn_cast<mlir::RankedTensorType>(inferredTypes.front());
    if (!result) {
      return mlir::failure();
    }
    return result;
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
    auto decoded = ncnn_graph::decode_convolution_params(params);
    if (!decoded) {
      return std::unexpected(make_error(context, decoded.error()));
    }
    const ncnn_graph::ConvolutionParams& convolution = *decoded;
    if (convolution.dynamic_weight || convolution.activation_type != 0 ||
        convolution.has_activation_params || convolution.pad_value != 0.0F) {
      return std::unexpected(
        make_error(context,
                   "dynamic weights, fused activation, and nonzero pad value "
                   "are unsupported"));
    }
    const std::size_t expected_weights = convolution.expected_weight_tensors();
    if (context.layer.get_weights().size() != expected_weights) {
      return std::unexpected(
        make_error(context,
                   std::format("expected {} weight tensors, got {}",
                               expected_weights,
                               context.layer.get_weights().size())));
    }
    const auto weight_dtype = context.layer.get_weights()[0].get_dtype();
    const bool quantized = convolution.int8_scale_term != 0;
    if ((!quantized && weight_dtype == ncnn_graph::DataType::Int8) ||
        (quantized && weight_dtype == ncnn_graph::DataType::Float16)) {
      return std::unexpected(make_error(
        context, "convolution kernel element type does not match scale term"));
    }
    const auto weight_shape = context.layer.get_weights()[0].get_shape();
    if (weight_shape.size() != 4 ||
        weight_shape[0] != convolution.output_channels ||
        weight_shape[2] != convolution.kernel_h ||
        weight_shape[3] != convolution.kernel_w ||
        context.layer.get_weights()[0].element_count() !=
          static_cast<std::size_t>(convolution.weight_count)) {
      return std::unexpected(make_error(
        context, "kernel shape or weight_data_size is inconsistent"));
    }
    auto feature_mask = validate_feature_mask(params);
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    auto input = find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    llvm::SmallVector<mlir::Value> tail;
    mlir::Value weight_value;
    for (std::size_t i = 0; i < expected_weights; ++i) {
      auto constant = make_constant(context, context.layer.get_weights()[i], i);
      if (!constant) {
        return std::unexpected(constant.error());
      }
      if (i == 0) {
        // weight 单独作为 convolution 的第二操作数，不进 tail。
        weight_value = *constant;
      } else {
        tail.push_back(*constant);
      }
    }
    auto i64 = [this](std::int64_t value) {
      return builder_.getI64IntegerAttr(value);
    };
    const mlir::Location location = builder_.getUnknownLoc();
    llvm::SmallVector<mlir::Value> operands{*input, weight_value};
    operands.append(tail);
    mlir::ncnn::ConvolutionOp::Properties properties;
    properties.kernel_h = i64(convolution.kernel_h);
    properties.kernel_w = i64(convolution.kernel_w);
    properties.stride_h = i64(convolution.stride_h);
    properties.stride_w = i64(convolution.stride_w);
    properties.dilation_h = i64(convolution.dilation_h);
    properties.dilation_w = i64(convolution.dilation_w);
    properties.pad_top = i64(convolution.pad_top);
    properties.pad_bottom = i64(convolution.pad_bottom);
    properties.pad_left = i64(convolution.pad_left);
    properties.pad_right = i64(convolution.pad_right);
    properties.has_bias = builder_.getBoolAttr(convolution.has_bias);
    properties.int8_scale_term = i64(convolution.int8_scale_term);
    auto result_type = capturing([&] {
      return inferSingleTensorResult<mlir::ncnn::ConvolutionOp>(
        location, operands, properties);
    });
    if (mlir::failed(result_type)) {
      return std::unexpected(make_error(context,
                                        captured_diag_.empty()
                                          ? "convolution shape inference failed"
                                          : captured_diag_));
    }
    auto conv = builder_.create<mlir::ncnn::ConvolutionOp>(
      location,
      *result_type,
      *input,
      weight_value,
      mlir::ValueRange(tail),
      i64(convolution.kernel_h),
      i64(convolution.kernel_w),
      i64(convolution.stride_h),
      i64(convolution.stride_w),
      i64(convolution.dilation_h),
      i64(convolution.dilation_w),
      i64(convolution.pad_top),
      i64(convolution.pad_bottom),
      i64(convolution.pad_left),
      i64(convolution.pad_right),
      builder_.getBoolAttr(convolution.has_bias),
      i64(convolution.int8_scale_term));
    tag_source(conv.getOperation(), context);
    return bind_blob(
      context, std::string(context.layer.get_outputs()[0]), conv.getResult());
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
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    auto input = find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    auto relu =
      builder_.create<mlir::ncnn::ReluOp>(builder_.getUnknownLoc(),
                                          input->getType(),
                                          *input,
                                          builder_.getF32FloatAttr(*slope));
    tag_source(relu.getOperation(), context);
    return bind_blob(
      context, std::string(context.layer.get_outputs()[0]), relu.getResult());
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
    if (!kernel_height || !stride_width || !stride_height || !pad_left ||
        !pad_right || !pad_top || !pad_bottom || !global || !pad_mode ||
        !include_pad || !adaptive || !output_width || !output_height) {
      return std::unexpected(
        make_error(context, "invalid pooling parameter type"));
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
    auto feature_mask = validate_feature_mask(params);
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    std::int64_t mode = 0;  // Regular
    std::int64_t effective_kernel_h = *kernel_height;
    std::int64_t effective_kernel_w = *kernel_width;
    if (*global == 1) {
      mode = 1;  // Global
    } else if (*adaptive == 1) {
      mode = 2;  // Adaptive
      effective_kernel_w = *output_width;
      effective_kernel_h = *output_height;
    }
    auto input = find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    auto i64 = [this](std::int64_t value) {
      return builder_.getI64IntegerAttr(value);
    };
    const mlir::Location location = builder_.getUnknownLoc();
    mlir::ncnn::PoolingOp::Properties properties;
    properties.kind = i64(*kind);
    properties.mode = i64(mode);
    properties.kernel_h = i64(effective_kernel_h);
    properties.kernel_w = i64(effective_kernel_w);
    properties.stride_h = i64(*stride_height);
    properties.stride_w = i64(*stride_width);
    properties.pad_top = i64(*pad_top);
    properties.pad_bottom = i64(*pad_bottom);
    properties.pad_left = i64(*pad_left);
    properties.pad_right = i64(*pad_right);
    properties.pad_mode = i64(*pad_mode);
    properties.include_pad = builder_.getBoolAttr(*include_pad == 1);
    auto result_type = capturing([&] {
      return inferSingleTensorResult<mlir::ncnn::PoolingOp>(
        location, mlir::ValueRange{*input}, properties);
    });
    if (mlir::failed(result_type)) {
      return std::unexpected(make_error(context,
                                        captured_diag_.empty()
                                          ? "pooling shape inference failed"
                                          : captured_diag_));
    }
    auto pool = builder_.create<mlir::ncnn::PoolingOp>(
      location,
      *result_type,
      *input,
      i64(*kind),
      i64(mode),
      i64(effective_kernel_h),
      i64(effective_kernel_w),
      i64(*stride_height),
      i64(*stride_width),
      i64(*pad_top),
      i64(*pad_bottom),
      i64(*pad_left),
      i64(*pad_right),
      i64(*pad_mode),
      builder_.getBoolAttr(*include_pad == 1));
    tag_source(pool.getOperation(), context);
    return bind_blob(
      context, std::string(context.layer.get_outputs()[0]), pool.getResult());
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
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    auto input = find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    const std::size_t result_count = context.layer.get_outputs().size();
    llvm::SmallVector<mlir::Type> result_types(result_count, input->getType());
    auto split = builder_.create<mlir::ncnn::SplitOp>(
      builder_.getUnknownLoc(), result_types, *input);
    tag_source(split.getOperation(), context);
    for (std::size_t i = 0; i < result_count; ++i) {
      auto bound = bind_blob(context,
                             std::string(context.layer.get_outputs()[i]),
                             split->getResult(i));
      if (!bound) {
        return std::unexpected(bound.error());
      }
    }
    return {};
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
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    llvm::SmallVector<mlir::Value> inputs;
    for (const auto& name : context.layer.get_inputs()) {
      auto value = find_blob(context, name);
      if (!value) {
        return std::unexpected(value.error());
      }
      inputs.push_back(*value);
    }
    const mlir::Location location = builder_.getUnknownLoc();
    mlir::ncnn::ConcatOp::Properties properties;
    properties.axis = builder_.getI64IntegerAttr(*axis);
    auto result_type = capturing([&] {
      return inferSingleTensorResult<mlir::ncnn::ConcatOp>(
        location, mlir::ValueRange(inputs), properties);
    });
    if (mlir::failed(result_type)) {
      return std::unexpected(make_error(context,
                                        captured_diag_.empty()
                                          ? "concat shape inference failed"
                                          : captured_diag_));
    }
    auto concat =
      builder_.create<mlir::ncnn::ConcatOp>(location,
                                            *result_type,
                                            mlir::ValueRange(inputs),
                                            builder_.getI64IntegerAttr(*axis));
    tag_source(concat.getOperation(), context);
    return bind_blob(
      context, std::string(context.layer.get_outputs()[0]), concat.getResult());
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
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    auto input = find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    auto dropout =
      builder_.create<mlir::ncnn::DropoutOp>(builder_.getUnknownLoc(),
                                             input->getType(),
                                             *input,
                                             builder_.getF32FloatAttr(*scale));
    tag_source(dropout.getOperation(), context);
    return bind_blob(context,
                     std::string(context.layer.get_outputs()[0]),
                     dropout.getResult());
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
    auto feature_mask = validate_feature_mask(context.layer.get_params());
    if (!feature_mask) {
      return std::unexpected(make_error(context, feature_mask.error()));
    }
    auto input = find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    auto softmax =
      builder_.create<mlir::ncnn::SoftmaxOp>(builder_.getUnknownLoc(),
                                             input->getType(),
                                             *input,
                                             builder_.getI64IntegerAttr(*axis));
    tag_source(softmax.getOperation(), context);
    return bind_blob(context,
                     std::string(context.layer.get_outputs()[0]),
                     softmax.getResult());
  }

  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> finish(
    const ncnn_graph::Graph& source) {
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
    captured_diag_.clear();
    mlir::ScopedDiagnosticHandler handler(context_,
                                          [this](mlir::Diagnostic& diagnostic) {
                                            captured_diag_ = diagnostic.str();
                                            return mlir::success();
                                          });
    if (mlir::failed(mlir::verify(module_.get().getOperation()))) {
      return std::unexpected(ImportError(source.get_layers().size(),
                                         "Verifier",
                                         "module",
                                         captured_diag_.empty()
                                           ? "module verification failed"
                                           : captured_diag_));
    }
    return std::move(module_);
  }

  mlir::MLIRContext* context_;
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::ncnn::ModelOp model_;
  llvm::StringMap<mlir::Value> blobs_;
  std::string captured_diag_;
};

}  // namespace

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> import_graph(
  const ncnn_graph::Graph& graph, mlir::MLIRContext& context) {
  ImportState state(context);
  return state.run(graph);
}

}  // namespace ncnn_importer
