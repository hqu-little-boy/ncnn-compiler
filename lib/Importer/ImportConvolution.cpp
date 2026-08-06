#include "ImporterInternal.hpp"

#include <cstddef>
#include <cstdint>
#include <format>

#include "llvm/ADT/SmallVector.h"

namespace ncnn_importer::detail {

ImportResult import_convolution(ImportContext& importer,
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
    return std::unexpected(
      make_error(context, "kernel shape or weight_data_size is inconsistent"));
  }
  auto feature_mask = validate_feature_mask(params);
  if (!feature_mask) {
    return std::unexpected(make_error(context, feature_mask.error()));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }

  llvm::SmallVector<mlir::Value> tail;
  mlir::Value weight_value;
  for (std::size_t index = 0; index < expected_weights; ++index) {
    auto constant = importer.make_constant(
      context, context.layer.get_weights()[index], index);
    if (!constant) {
      return std::unexpected(constant.error());
    }
    if (index == 0) {
      weight_value = *constant;
    } else {
      tail.push_back(*constant);
    }
  }

  auto& builder = importer.builder();
  auto i64 = [&builder](std::int64_t value) {
    return builder.getI64IntegerAttr(value);
  };
  const mlir::Location location = builder.getUnknownLoc();
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
  properties.has_bias = builder.getBoolAttr(convolution.has_bias);
  properties.int8_scale_term = i64(convolution.int8_scale_term);
  auto result_type =
    importer.infer_single_tensor_result<mlir::ncnn::ConvolutionOp>(
      location, operands, properties);
  if (mlir::failed(result_type)) {
    return std::unexpected(make_error(context,
                                      importer.captured_diagnostic().empty()
                                        ? "convolution shape inference failed"
                                        : importer.captured_diagnostic()));
  }
  auto convolution_op = builder.create<mlir::ncnn::ConvolutionOp>(
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
    builder.getBoolAttr(convolution.has_bias),
    i64(convolution.int8_scale_term));
  importer.tag_source(convolution_op.getOperation(), context);
  return importer.bind_blob(context,
                            std::string(context.layer.get_outputs()[0]),
                            convolution_op.getResult());
}

}  // namespace ncnn_importer::detail
