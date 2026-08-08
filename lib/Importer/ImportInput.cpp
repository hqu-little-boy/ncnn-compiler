#include "ImporterInternal.hpp"

#include <cstdint>

#include "llvm/ADT/SmallVector.h"

namespace ncnn_importer::detail {

ImportResult import_input(ImportContext& importer,
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
  const bool dimensions_omitted =
    *width == 0 && *height == 0 && *channels == 0 && *depth == 0;
  auto shape_override = importer.next_input_shape();
  auto valid_extent = [](std::int64_t extent) {
    return extent > 0 || extent == ncnn_importer::kDynamicExtent;
  };
  if (dimensions_omitted && shape_override) {
    const auto& shape = *shape_override;
    const bool specialized_dynamic_rank =
      importer.input_shapes().size() == 1 && !shape.empty() &&
      shape.size() <= 4 && std::ranges::all_of(shape, [](std::int64_t extent) {
        return extent == ncnn_importer::kDynamicExtent;
      });
    if ((!specialized_dynamic_rank && shape.size() != 3) ||
        !std::ranges::all_of(shape, valid_extent)) {
      return std::unexpected(
        make_error(context,
                   "input shape override must have rank 1..4 with positive or "
                   "dynamic extents"));
    }
    if (specialized_dynamic_rank && shape.size() != 3) {
      auto& builder = importer.builder();
      llvm::SmallVector<std::int64_t> dimensions(shape.size(),
                                                 mlir::ShapedType::kDynamic);
      auto type = mlir::RankedTensorType::get(dimensions, builder.getF32Type());
      auto input = builder.create<mlir::ncnn::InputOp>(
        builder.getUnknownLoc(),
        type,
        builder.getStringAttr(context.layer.get_name()),
        builder.getStringAttr(context.layer.get_outputs()[0]));
      importer.tag_source(input.getOperation(), context);
      return importer.bind_blob(context,
                                std::string(context.layer.get_outputs()[0]),
                                input.getOutput());
    }
    *channels = shape[0];
    *height = shape[1];
    *width = shape[2];
  }
  if ((!valid_extent(*width) || !valid_extent(*height) ||
       !valid_extent(*channels)) ||
      *depth != 0) {
    return std::unexpected(make_error(
      context,
      "Input requires positive w/h/c and unsupported depth must be 0"));
  }
  auto feature_mask = validate_feature_mask(context.layer.get_params());
  if (!feature_mask) {
    return std::unexpected(make_error(context, feature_mask.error()));
  }

  auto& builder = importer.builder();
  auto mlir_extent = [](std::int64_t extent) {
    return extent == ncnn_importer::kDynamicExtent ? mlir::ShapedType::kDynamic
                                                   : extent;
  };
  auto type = mlir::RankedTensorType::get(
    llvm::SmallVector<std::int64_t>{
      mlir_extent(*channels), mlir_extent(*height), mlir_extent(*width)},
    builder.getF32Type());
  auto input = builder.create<mlir::ncnn::InputOp>(
    builder.getUnknownLoc(),
    type,
    builder.getStringAttr(context.layer.get_name()),
    builder.getStringAttr(context.layer.get_outputs()[0]));
  importer.tag_source(input.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), input.getOutput());
}

}  // namespace ncnn_importer::detail
