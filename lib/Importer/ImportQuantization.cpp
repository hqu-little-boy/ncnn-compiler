#include "ImporterInternal.hpp"

#include <array>

namespace ncnn_importer::detail {
namespace {

ImportResult validate_layer(const LayerContext& context,
                            std::span<const int> allowed,
                            std::size_t weights) {
  auto arity = expect_source_arity(context.layer, 1, 1);
  if (!arity) {
    return std::unexpected(make_error(context, arity.error()));
  }
  auto params = validate_param_ids(context.layer.get_params(), allowed);
  if (!params) {
    return std::unexpected(make_error(context, params.error()));
  }
  if (context.layer.get_weights().size() != weights) {
    return std::unexpected(make_error(context, "unexpected weight count"));
  }
  return {};
}

std::expected<mlir::Value, ImportError> make_parameter(
  ImportContext& importer, const LayerContext& context, std::size_t index) {
  if (context.layer.get_weights()[index].get_dtype() !=
      ncnn_graph::DataType::Float32) {
    return std::unexpected(
      make_error(context, "quantization parameters must be FP32"));
  }
  return importer.make_constant(
    context, context.layer.get_weights()[index], index);
}

}  // namespace

ImportResult import_quantize(ImportContext& importer,
                             const LayerContext& context) {
  constexpr int kAllowed[] = {0};
  if (auto valid = validate_layer(context, kAllowed, 1); !valid) {
    return valid;
  }
  auto params = ncnn_graph::decode_quantize_params(context.layer.get_params());
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  auto scale = make_parameter(importer, context, 0);
  if (!params || !input || !scale) {
    return std::unexpected(!params ? make_error(context, params.error())
                                   : (!input ? input.error() : scale.error()));
  }
  mlir::ncnn::QuantizeOp::Properties properties;
  llvm::SmallVector<mlir::Value> operands{*input, *scale};
  auto type = importer.infer_single_tensor_result<mlir::ncnn::QuantizeOp>(
    importer.builder().getUnknownLoc(), operands, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation = importer.builder().create<mlir::ncnn::QuantizeOp>(
    importer.builder().getUnknownLoc(), *type, *input, *scale);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

ImportResult import_dequantize(ImportContext& importer,
                               const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1};
  auto params =
    ncnn_graph::decode_dequantize_params(context.layer.get_params());
  if (!params) {
    return std::unexpected(make_error(context, params.error()));
  }
  const std::size_t weightCount = params->bias_count == 0 ? 1 : 2;
  if (auto valid = validate_layer(context, kAllowed, weightCount); !valid) {
    return valid;
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  auto scale = make_parameter(importer, context, 0);
  if (!input || !scale) {
    return std::unexpected(!input ? input.error() : scale.error());
  }
  llvm::SmallVector<mlir::Value> bias;
  if (weightCount == 2) {
    auto value = make_parameter(importer, context, 1);
    if (!value) {
      return std::unexpected(value.error());
    }
    bias.push_back(*value);
  }
  llvm::SmallVector<mlir::Value> operands{*input, *scale};
  operands.append(bias);
  mlir::ncnn::DequantizeOp::Properties properties;
  auto type = importer.infer_single_tensor_result<mlir::ncnn::DequantizeOp>(
    importer.builder().getUnknownLoc(), operands, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation = importer.builder().create<mlir::ncnn::DequantizeOp>(
    importer.builder().getUnknownLoc(), *type, *input, *scale, bias);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

ImportResult import_requantize(ImportContext& importer,
                               const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 4};
  auto params =
    ncnn_graph::decode_requantize_params(context.layer.get_params());
  if (!params) {
    return std::unexpected(make_error(context, params.error()));
  }
  if (params->activation_type != 0 && params->activation_type != 1) {
    return std::unexpected(
      make_error(context, "Requantize supports no activation or ReLU only"));
  }
  const std::size_t weightCount = params->bias_count == 0 ? 2 : 3;
  if (auto valid = validate_layer(context, kAllowed, weightCount); !valid) {
    return valid;
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  auto inputScale = make_parameter(importer, context, 0);
  auto outputScale = make_parameter(importer, context, 1);
  if (!input || !inputScale || !outputScale) {
    return std::unexpected(!input        ? input.error()
                           : !inputScale ? inputScale.error()
                                         : outputScale.error());
  }
  llvm::SmallVector<mlir::Value> bias;
  if (weightCount == 3) {
    auto value = make_parameter(importer, context, 2);
    if (!value) {
      return std::unexpected(value.error());
    }
    bias.push_back(*value);
  }
  auto& builder = importer.builder();
  mlir::ncnn::RequantizeOp::Properties properties;
  properties.activation_type =
    builder.getI64IntegerAttr(params->activation_type);
  properties.activation_params = builder.getDenseF32ArrayAttr({});
  llvm::SmallVector<mlir::Value> operands{*input, *inputScale, *outputScale};
  operands.append(bias);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::RequantizeOp>(
    builder.getUnknownLoc(), operands, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation =
    builder.create<mlir::ncnn::RequantizeOp>(builder.getUnknownLoc(),
                                             *type,
                                             *input,
                                             *inputScale,
                                             *outputScale,
                                             bias,
                                             properties.activation_type,
                                             properties.activation_params);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

ImportResult import_cast(ImportContext& importer, const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1};
  if (auto valid = validate_layer(context, kAllowed, 0); !valid) {
    return valid;
  }
  auto params = ncnn_graph::decode_cast_params(context.layer.get_params());
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!params || !input) {
    return std::unexpected(!params ? make_error(context, params.error())
                                   : input.error());
  }
  auto& builder = importer.builder();
  mlir::ncnn::CastOp::Properties properties;
  properties.type_from = builder.getI64IntegerAttr(params->type_from);
  properties.type_to = builder.getI64IntegerAttr(params->type_to);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::CastOp>(
    builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation = builder.create<mlir::ncnn::CastOp>(builder.getUnknownLoc(),
                                                      *type,
                                                      *input,
                                                      properties.type_from,
                                                      properties.type_to);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

}  // namespace ncnn_importer::detail
