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
  if (*width <= 0 || *height <= 0 || *channels <= 0 || *depth != 0) {
    return std::unexpected(make_error(
      context,
      "Input requires positive w/h/c and unsupported depth must be 0"));
  }
  auto feature_mask = validate_feature_mask(context.layer.get_params());
  if (!feature_mask) {
    return std::unexpected(make_error(context, feature_mask.error()));
  }

  auto& builder = importer.builder();
  auto type = mlir::RankedTensorType::get(
    llvm::SmallVector<std::int64_t>{*channels, *height, *width},
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
