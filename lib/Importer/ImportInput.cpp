#include "ImporterInternal.hpp"

#include <cstdint>
#include <format>

#include "llvm/ADT/SmallVector.h"

namespace ncnn_importer::detail {

ImportResult import_input(ImportContext& importer,
                          const LayerContext& context) {
  if (!context.layer.get_inputs().empty() ||
      context.layer.get_outputs().empty()) {
    return std::unexpected(
      make_error(context, "Input requires no inputs and at least one output"));
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
  auto valid_extent = [](std::int64_t extent) {
    return extent > 0 || extent == ncnn_importer::kDynamicExtent;
  };
  if (!dimensions_omitted &&
      ((!valid_extent(*width) || !valid_extent(*height) ||
        !valid_extent(*channels)) ||
       *depth != 0)) {
    return std::unexpected(make_error(
      context,
      "Input requires positive w/h/c and unsupported depth must be 0"));
  }
  auto feature_mask = validate_feature_mask(context.layer.get_params());
  if (!feature_mask) {
    return std::unexpected(make_error(context, feature_mask.error()));
  }

  auto mlir_extent = [](std::int64_t extent) {
    return extent == ncnn_importer::kDynamicExtent ? mlir::ShapedType::kDynamic
                                                   : extent;
  };
  auto& builder = importer.builder();
  const bool multipleOutputs = context.layer.get_outputs().size() > 1;
  for (std::string_view output : context.layer.get_outputs()) {
    auto shapeOverride = importer.next_input_shape(dimensions_omitted);
    llvm::SmallVector<std::int64_t> dimensions;
    if (dimensions_omitted && shapeOverride) {
      if ((shapeOverride->empty() || shapeOverride->size() > 4) ||
          !std::ranges::all_of(*shapeOverride, valid_extent)) {
        return std::unexpected(make_error(
          context,
          "input shape override must have rank 1..4 with positive or dynamic "
          "extents"));
      }
      dimensions.assign(shapeOverride->begin(), shapeOverride->end());
    } else if (dimensions_omitted) {
      return std::unexpected(make_error(
        context, "Input dimensions are omitted without an override"));
    } else {
      dimensions = {
        mlir_extent(*channels), mlir_extent(*height), mlir_extent(*width)};
    }
    std::ranges::replace(
      dimensions, ncnn_importer::kDynamicExtent, mlir::ShapedType::kDynamic);
    mlir::Type elementType = importer.input_uses_integer_storage(output)
                               ? static_cast<mlir::Type>(builder.getI32Type())
                               : static_cast<mlir::Type>(builder.getF32Type());
    auto type = mlir::RankedTensorType::get(dimensions, elementType);
    const std::string layerName =
      multipleOutputs ? std::format("{}.{}", context.layer.get_name(), output)
                      : std::string(context.layer.get_name());
    auto input =
      builder.create<mlir::ncnn::InputOp>(builder.getUnknownLoc(),
                                          type,
                                          builder.getStringAttr(layerName),
                                          builder.getStringAttr(output));
    importer.tag_source(input.getOperation(), context);
    auto bound =
      importer.bind_blob(context, std::string(output), input.getOutput());
    if (!bound) {
      return bound;
    }
  }
  return {};
}

}  // namespace ncnn_importer::detail
