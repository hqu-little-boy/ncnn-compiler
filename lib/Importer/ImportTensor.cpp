#include "ImporterInternal.hpp"

#include <array>
#include <cstddef>

#include "llvm/ADT/SmallVector.h"

namespace ncnn_importer::detail {

ImportResult import_split(ImportContext& importer,
                          const LayerContext& context) {
  if (context.layer.get_inputs().size() != 1 ||
      context.layer.get_outputs().size() < 2) {
    return std::unexpected(
      make_error(context, "Split requires one input and at least two outputs"));
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
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  const std::size_t result_count = context.layer.get_outputs().size();
  llvm::SmallVector<mlir::Type> result_types(result_count, input->getType());
  auto& builder = importer.builder();
  auto split = builder.create<mlir::ncnn::SplitOp>(
    builder.getUnknownLoc(), result_types, *input);
  importer.tag_source(split.getOperation(), context);
  for (std::size_t index = 0; index < result_count; ++index) {
    auto bound =
      importer.bind_blob(context,
                         std::string(context.layer.get_outputs()[index]),
                         split->getResult(index));
    if (!bound) {
      return std::unexpected(bound.error());
    }
  }
  return {};
}

ImportResult import_concat(ImportContext& importer,
                           const LayerContext& context) {
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
    auto value = importer.find_blob(context, name);
    if (!value) {
      return std::unexpected(value.error());
    }
    inputs.push_back(*value);
  }
  auto& builder = importer.builder();
  const mlir::Location location = builder.getUnknownLoc();
  mlir::ncnn::ConcatOp::Properties properties;
  properties.axis = builder.getI64IntegerAttr(*axis);
  auto result_type = importer.infer_single_tensor_result<mlir::ncnn::ConcatOp>(
    location, mlir::ValueRange(inputs), properties);
  if (mlir::failed(result_type)) {
    return std::unexpected(make_error(context,
                                      importer.captured_diagnostic().empty()
                                        ? "concat shape inference failed"
                                        : importer.captured_diagnostic()));
  }
  auto concat =
    builder.create<mlir::ncnn::ConcatOp>(location,
                                         *result_type,
                                         mlir::ValueRange(inputs),
                                         builder.getI64IntegerAttr(*axis));
  importer.tag_source(concat.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), concat.getResult());
}

}  // namespace ncnn_importer::detail
