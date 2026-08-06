#include "ImporterInternal.hpp"

#include <cstdint>

namespace ncnn_importer::detail {

ImportResult import_pooling(ImportContext& importer,
                            const LayerContext& context) {
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
    return std::unexpected(
      make_error(context, "pooling kind or boolean mode parameter is invalid"));
  }
  if (*pad_mode < 0 || *pad_mode > 3) {
    return std::unexpected(
      make_error(context, "pooling pad_mode must be in [0, 3]"));
  }
  auto feature_mask = validate_feature_mask(params);
  if (!feature_mask) {
    return std::unexpected(make_error(context, feature_mask.error()));
  }

  std::int64_t mode = 0;
  std::int64_t effective_kernel_h = *kernel_height;
  std::int64_t effective_kernel_w = *kernel_width;
  if (*global == 1) {
    mode = 1;
  } else if (*adaptive == 1) {
    mode = 2;
    effective_kernel_w = *output_width;
    effective_kernel_h = *output_height;
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }

  auto& builder = importer.builder();
  auto i64 = [&builder](std::int64_t value) {
    return builder.getI64IntegerAttr(value);
  };
  const mlir::Location location = builder.getUnknownLoc();
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
  properties.include_pad = builder.getBoolAttr(*include_pad == 1);
  auto result_type = importer.infer_single_tensor_result<mlir::ncnn::PoolingOp>(
    location, mlir::ValueRange{*input}, properties);
  if (mlir::failed(result_type)) {
    return std::unexpected(make_error(context,
                                      importer.captured_diagnostic().empty()
                                        ? "pooling shape inference failed"
                                        : importer.captured_diagnostic()));
  }
  auto pool = builder.create<mlir::ncnn::PoolingOp>(
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
    builder.getBoolAttr(*include_pad == 1));
  importer.tag_source(pool.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), pool.getResult());
}

}  // namespace ncnn_importer::detail
