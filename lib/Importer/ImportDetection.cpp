#include "ImporterInternal.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"

namespace ncnn_importer::detail {
namespace {

ImportResult validate_layer(const LayerContext& context,
                            std::span<const int> allowed) {
  auto arity = expect_source_arity(context.layer, 1, 1);
  if (!arity) {
    return std::unexpected(make_error(context, arity.error()));
  }
  auto params = validate_param_ids(context.layer.get_params(), allowed);
  if (!params) {
    return std::unexpected(make_error(context, params.error()));
  }
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!mask) {
    return std::unexpected(make_error(context, mask.error()));
  }
  return {};
}

}  // namespace

ImportResult import_padding(ImportContext& importer,
                            const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 4, 5, 6};
  auto valid = validate_layer(context, kAllowed);
  if (!valid) {
    return valid;
  }
  const auto& params = context.layer.get_params();
  auto top = get_int(params, 0, 0, "top");
  auto bottom = get_int(params, 1, 0, "bottom");
  auto left = get_int(params, 2, 0, "left");
  auto right = get_int(params, 3, 0, "right");
  auto type = get_int(params, 4, 0, "type");
  auto value = get_float(params, 5, 0.0F, "value");
  auto perChannel = get_int(params, 6, 0, "per_channel_pad_data_size");
  if (!top || !bottom || !left || !right || !type || !value || !perChannel ||
      *type != 0 || *perChannel != 0) {
    return std::unexpected(make_error(
      context,
      "Padding only supports constant spatial padding without per-channel "
      "data"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto i64 = [&builder](int64_t value) {
    return builder.getI64IntegerAttr(value);
  };
  mlir::ncnn::PaddingOp::Properties properties;
  properties.top = i64(*top);
  properties.bottom = i64(*bottom);
  properties.left = i64(*left);
  properties.right = i64(*right);
  properties.value = builder.getF32FloatAttr(*value);
  auto result = importer.infer_single_tensor_result<mlir::ncnn::PaddingOp>(
    builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(result)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation =
    builder.create<mlir::ncnn::PaddingOp>(builder.getUnknownLoc(),
                                          *result,
                                          *input,
                                          properties.top,
                                          properties.bottom,
                                          properties.left,
                                          properties.right,
                                          properties.value);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

ImportResult import_interp(ImportContext& importer,
                           const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 4, 6};
  auto valid = validate_layer(context, kAllowed);
  if (!valid) {
    return valid;
  }
  const auto& params = context.layer.get_params();
  auto resizeType = get_int(params, 0, 0, "resize_type");
  auto heightScale = get_float(params, 1, 1.0F, "height_scale");
  auto widthScale = get_float(params, 2, 1.0F, "width_scale");
  auto outputH = get_int(params, 3, 0, "output_h");
  auto outputW = get_int(params, 4, 0, "output_w");
  auto alignCorner = get_int(params, 6, 0, "align_corner");
  if (!resizeType || !heightScale || !widthScale || !outputH || !outputW ||
      !alignCorner || *resizeType != 1 || *alignCorner != 0 ||
      *heightScale < 1.0F || *widthScale < 1.0F ||
      std::trunc(*heightScale) != *heightScale ||
      std::trunc(*widthScale) != *widthScale) {
    return std::unexpected(make_error(
      context,
      "Interp only supports nearest resize by positive integer scale"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
  const int64_t heightFactor = static_cast<int64_t>(*heightScale);
  const int64_t widthFactor = static_cast<int64_t>(*widthScale);
  if (!inputType || inputType.getRank() != 3 ||
      (*outputH != 0 && (*outputH % heightFactor != 0 ||
                         *outputH / heightFactor != inputType.getShape()[1])) ||
      (*outputW != 0 && (*outputW % widthFactor != 0 ||
                         *outputW / widthFactor != inputType.getShape()[2]))) {
    return std::unexpected(
      make_error(context, "Interp explicit output size must match its scale"));
  }
  auto& builder = importer.builder();
  mlir::ncnn::InterpOp::Properties properties;
  properties.height_scale = builder.getI64IntegerAttr(heightFactor);
  properties.width_scale = builder.getI64IntegerAttr(widthFactor);
  auto result = importer.infer_single_tensor_result<mlir::ncnn::InterpOp>(
    builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(result)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation = builder.create<mlir::ncnn::InterpOp>(builder.getUnknownLoc(),
                                                        *result,
                                                        *input,
                                                        properties.height_scale,
                                                        properties.width_scale);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

ImportResult import_sigmoid(ImportContext& importer,
                            const LayerContext& context) {
  auto valid = validate_layer(context, {});
  if (!valid) {
    return valid;
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto operation = builder.create<mlir::ncnn::SigmoidOp>(
    builder.getUnknownLoc(), input->getType(), *input);
  importer.tag_source(operation, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], operation.getOutput());
}

ImportResult import_deconvolution(ImportContext& importer,
                                  const LayerContext& context) {
  constexpr int kAllowed[] = {0,  1,  2,  3,  4,  5,  6,  9,  10, 11,
                              12, 13, 14, 15, 16, 18, 19, 20, 21, 28};
  auto valid = validate_layer(context, kAllowed);
  if (!valid) {
    return valid;
  }
  auto decoded =
    ncnn_graph::decode_deconvolution_params(context.layer.get_params());
  if (!decoded) {
    return std::unexpected(make_error(context, decoded.error()));
  }
  const auto& params = *decoded;
  if (params.dynamic_weight || params.kernel_h != 2 || params.kernel_w != 2 ||
      params.stride_h != 2 || params.stride_w != 2 || params.dilation_h != 1 ||
      params.dilation_w != 1 || params.pad_top != 0 || params.pad_bottom != 0 ||
      params.pad_left != 0 || params.pad_right != 0 ||
      params.output_pad_bottom != 0 || params.output_pad_right != 0 ||
      params.output_h != 0 || params.output_w != 0 ||
      (params.activation_type != 0 && params.activation_type != 1) ||
      params.has_activation_params ||
      context.layer.get_weights().size() != params.expected_weight_tensors()) {
    return std::unexpected(make_error(
      context, "only static FP32 2x2 stride-2 Deconvolution is supported"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
  const auto weightShape = context.layer.get_weights()[0].get_shape();
  if (!inputType || inputType.getRank() != 3 || weightShape.size() != 4 ||
      weightShape[0] != params.output_channels ||
      weightShape[1] != inputType.getShape()[0] || weightShape[2] != 2 ||
      weightShape[3] != 2) {
    return std::unexpected(
      make_error(context, "Deconvolution weight must have shape [O,I,2,2]"));
  }
  for (const auto& weight : context.layer.get_weights()) {
    if (weight.get_dtype() != ncnn_graph::DataType::Float32) {
      return std::unexpected(
        make_error(context, "Deconvolution weights must be FP32"));
    }
  }
  auto weight =
    importer.make_constant(context, context.layer.get_weights()[0], 0);
  if (!weight) {
    return std::unexpected(weight.error());
  }
  llvm::SmallVector<mlir::Value> bias;
  if (params.has_bias) {
    auto value =
      importer.make_constant(context, context.layer.get_weights()[1], 1);
    if (!value) {
      return std::unexpected(value.error());
    }
    bias.push_back(*value);
  }
  auto& builder = importer.builder();
  auto i64 = [&builder](int64_t value) {
    return builder.getI64IntegerAttr(value);
  };
  mlir::ncnn::DeconvolutionOp::Properties properties;
  properties.kernel_h = i64(params.kernel_h);
  properties.kernel_w = i64(params.kernel_w);
  properties.stride_h = i64(params.stride_h);
  properties.stride_w = i64(params.stride_w);
  properties.dilation_h = i64(params.dilation_h);
  properties.dilation_w = i64(params.dilation_w);
  properties.pad_top = i64(params.pad_top);
  properties.pad_bottom = i64(params.pad_bottom);
  properties.pad_left = i64(params.pad_left);
  properties.pad_right = i64(params.pad_right);
  properties.output_pad_bottom = i64(params.output_pad_bottom);
  properties.output_pad_right = i64(params.output_pad_right);
  properties.has_bias = builder.getBoolAttr(params.has_bias);
  llvm::SmallVector<mlir::Value> operands{*input, *weight};
  operands.append(bias);
  auto result =
    importer.infer_single_tensor_result<mlir::ncnn::DeconvolutionOp>(
      builder.getUnknownLoc(), operands, properties);
  if (mlir::failed(result)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto operation =
    builder.create<mlir::ncnn::DeconvolutionOp>(builder.getUnknownLoc(),
                                                *result,
                                                *input,
                                                *weight,
                                                bias,
                                                properties.kernel_h,
                                                properties.kernel_w,
                                                properties.stride_h,
                                                properties.stride_w,
                                                properties.dilation_h,
                                                properties.dilation_w,
                                                properties.pad_top,
                                                properties.pad_bottom,
                                                properties.pad_left,
                                                properties.pad_right,
                                                properties.output_pad_bottom,
                                                properties.output_pad_right,
                                                properties.has_bias);
  importer.tag_source(operation, context);
  mlir::Value output = operation.getOutput();
  if (params.activation_type == 1) {
    auto relu = builder.create<mlir::ncnn::ReluOp>(
      builder.getUnknownLoc(), *result, output, builder.getF32FloatAttr(0));
    importer.tag_source(relu, context);
    output = relu.getOutput();
  }
  return importer.bind_blob(context, context.layer.get_outputs()[0], output);
}

}  // namespace ncnn_importer::detail
