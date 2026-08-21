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

ImportResult import_shuffle_channel(ImportContext& importer,
                                    const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto featureMask = validate_feature_mask(context.layer.get_params());
  auto group = get_int(context.layer.get_params(), 0, 1, "group");
  auto reverse = get_int(context.layer.get_params(), 1, 0, "reverse");
  if (!arity || !allowed || !featureMask || !group || !reverse) {
    return std::unexpected(make_error(context,
                                      !arity         ? arity.error()
                                      : !allowed     ? allowed.error()
                                      : !featureMask ? featureMask.error()
                                      : !group       ? group.error()
                                                     : reverse.error()));
  }
  auto boolean = expect_boolean(*reverse, "reverse");
  if (!boolean || *group <= 0) {
    return std::unexpected(make_error(
      context, !boolean ? boolean.error() : "group must be positive"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto op = builder.create<mlir::ncnn::ShuffleChannelOp>(
    builder.getUnknownLoc(),
    input->getType(),
    *input,
    builder.getI64IntegerAttr(*group),
    builder.getBoolAttr(*reverse));
  if (mlir::failed(op.verify())) {
    return std::unexpected(make_error(context, "invalid channel shuffle"));
  }
  importer.tag_source(op.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op.getResult());
}

ImportResult import_slice(ImportContext& importer,
                          const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1};
  if (context.layer.get_inputs().size() != 1 ||
      context.layer.get_outputs().size() < 2) {
    return std::unexpected(
      make_error(context, "Slice requires one input and at least two outputs"));
  }
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto featureMask = validate_feature_mask(context.layer.get_params());
  auto axis = get_int(context.layer.get_params(), 1, 0, "axis");
  const auto* slicesValue = find_param(context.layer.get_params(), 0);
  if (!allowed || !featureMask || !axis || slicesValue == nullptr ||
      slicesValue->get_kind() != ncnn_graph::ParamValue::Kind::IntArray) {
    return std::unexpected(
      make_error(context,
                 !allowed       ? allowed.error()
                 : !featureMask ? featureMask.error()
                 : !axis        ? axis.error()
                         : "parameter 0 (slices) must be an integer array"));
  }
  auto slices = *slicesValue->get_int_array();
  if (slices.size() != context.layer.get_outputs().size()) {
    return std::unexpected(
      make_error(context, "slice count must match output count"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  mlir::ncnn::SliceOp::Properties properties;
  properties.slices = builder.getDenseI64ArrayAttr(slices);
  properties.axis = builder.getI64IntegerAttr(*axis);
  llvm::SmallVector<mlir::Type> resultTypes;
  if (mlir::failed(importer.infer_tensor_results<mlir::ncnn::SliceOp>(
        builder.getUnknownLoc(), *input, properties, resultTypes)) ||
      resultTypes.size() != slices.size()) {
    return std::unexpected(make_error(context,
                                      importer.captured_diagnostic().empty()
                                        ? "slice shape inference failed"
                                        : importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::SliceOp>(builder.getUnknownLoc(),
                                                resultTypes,
                                                *input,
                                                properties.slices,
                                                properties.axis);
  importer.tag_source(op.getOperation(), context);
  for (std::size_t index = 0; index < resultTypes.size(); ++index) {
    auto bound = importer.bind_blob(
      context, context.layer.get_outputs()[index], op->getResult(index));
    if (!bound) {
      return bound;
    }
  }
  return {};
}

ImportResult import_reduction(ImportContext& importer,
                              const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 4, 5};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto featureMask = validate_feature_mask(context.layer.get_params());
  auto operation = get_int(context.layer.get_params(), 0, 0, "operation");
  auto reduceAll = get_int(context.layer.get_params(), 1, 1, "reduce_all");
  auto coeff = get_float(context.layer.get_params(), 2, 1.0F, "coeff");
  auto keepDims = get_int(context.layer.get_params(), 4, 0, "keepdims");
  auto fixbug = get_int(context.layer.get_params(), 5, 0, "fixbug0");
  if (!arity || !allowed || !featureMask || !operation || !reduceAll ||
      !coeff || !keepDims || !fixbug) {
    return std::unexpected(make_error(context, "invalid Reduction parameters"));
  }
  auto reduceBoolean = expect_boolean(*reduceAll, "reduce_all");
  auto keepBoolean = expect_boolean(*keepDims, "keepdims");
  const auto* axesValue = find_param(context.layer.get_params(), 3);
  if (!reduceBoolean || !keepBoolean || *operation != 3 ||
      (axesValue != nullptr &&
       axesValue->get_kind() != ncnn_graph::ParamValue::Kind::IntArray)) {
    return std::unexpected(make_error(
      context, "Reduction supports static FP32 mean with integer axes only"));
  }
  std::span<const int64_t> axes;
  if (axesValue != nullptr) {
    axes = *axesValue->get_int_array();
  }
  if (!*reduceAll && (axes.empty() || *fixbug != 1)) {
    return std::unexpected(
      make_error(context, "explicit Reduction axes require fixbug0=1"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  mlir::ncnn::ReductionOp::Properties properties;
  properties.kind = builder.getI64IntegerAttr(*operation);
  properties.reduce_all = builder.getBoolAttr(*reduceAll);
  properties.coeff = builder.getF32FloatAttr(*coeff);
  properties.axes = builder.getDenseI64ArrayAttr(axes);
  properties.keepdims = builder.getBoolAttr(*keepDims);
  auto resultType =
    importer.infer_single_tensor_result<mlir::ncnn::ReductionOp>(
      builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(resultType)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::ReductionOp>(builder.getUnknownLoc(),
                                                    *resultType,
                                                    *input,
                                                    properties.kind,
                                                    properties.reduce_all,
                                                    properties.coeff,
                                                    properties.axes,
                                                    properties.keepdims);
  importer.tag_source(op.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op.getResult());
}

namespace {
std::expected<std::span<const std::int64_t>, ImportError> get_axes(
  const LayerContext& context) {
  const auto* value = find_param(context.layer.get_params(), 3);
  if (value == nullptr ||
      value->get_kind() != ncnn_graph::ParamValue::Kind::IntArray) {
    return std::unexpected(
      make_error(context, "parameter 3 (axes) must be an integer array"));
  }
  return *value->get_int_array();
}
}  // namespace

ImportResult import_squeeze(ImportContext& importer,
                            const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 11};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto axes = get_axes(context);
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !axes || !mask ||
      context.layer.get_params().has(0) || context.layer.get_params().has(1) ||
      context.layer.get_params().has(2) || context.layer.get_params().has(11)) {
    return std::unexpected(
      make_error(context, "Squeeze requires explicit axes only"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  if (auto padding = input->getDefiningOp<mlir::ncnn::PaddingOp>();
      padding && padding->hasAttr("ncnn.normalized_rank4_reflection") &&
      axes->size() == 1 && (*axes)[0] == 1) {
    return importer.bind_blob(
      context, context.layer.get_outputs()[0], padding.getOutput());
  }
  auto& builder = importer.builder();
  mlir::ncnn::SqueezeOp::Properties properties;
  properties.axes = builder.getDenseI64ArrayAttr(*axes);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::SqueezeOp>(
    builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::SqueezeOp>(
    builder.getUnknownLoc(), *type, *input, properties.axes);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_expand_dims(ImportContext& importer,
                                const LayerContext& context) {
  constexpr int kAllowed[] = {3};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto axes = get_axes(context);
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !axes || !mask) {
    return std::unexpected(
      make_error(context, "invalid ExpandDims parameters"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  mlir::ncnn::ExpandDimsOp::Properties properties;
  properties.axes = builder.getDenseI64ArrayAttr(*axes);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::ExpandDimsOp>(
    builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::ExpandDimsOp>(
    builder.getUnknownLoc(), *type, *input, properties.axes);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_permute(ImportContext& importer,
                            const LayerContext& context) {
  constexpr int kAllowed[] = {0};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto order = get_int(context.layer.get_params(), 0, 0, "order_type");
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !order || !mask) {
    return std::unexpected(make_error(context, "invalid Permute parameters"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
  if (!inputType ||
      !((inputType.getRank() == 2 && (*order == 0 || *order == 1)) ||
        (inputType.getRank() == 3 && (*order == 2 || *order == 4)) ||
        (inputType.getRank() == 4 && *order == 3))) {
    return std::unexpected(make_error(
      context,
      "Permute supports rank-2 order 0/1, rank-3 order 2/4, and rank-4 order "
      "3 only"));
  }
  const std::array<int64_t, 2> identity{0, 1};
  const std::array<int64_t, 2> transpose{1, 0};
  const std::array<int64_t, 3> chwToHcw{1, 0, 2};
  const std::array<int64_t, 3> hcw{2, 0, 1};
  // ncnn rank-4 order 3 (`d w h c`) maps output (c,d,h,w) to input
  // (c,h,w,d); the MLIR tensor keeps the native [C,D,H,W] order.
  const std::array<int64_t, 4> cdhwToChwd{0, 2, 3, 1};
  std::span<const int64_t> permutation =
    inputType.getRank() == 4 ? std::span<const int64_t>(cdhwToChwd)
    : inputType.getRank() == 3 && *order == 2
      ? std::span<const int64_t>(chwToHcw)
    : inputType.getRank() == 3 ? std::span<const int64_t>(hcw)
    : *order == 0              ? std::span<const int64_t>(identity)
                               : std::span<const int64_t>(transpose);
  auto& builder = importer.builder();
  mlir::ncnn::PermuteOp::Properties properties;
  properties.permutation = builder.getDenseI64ArrayAttr(permutation);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::PermuteOp>(
    builder.getUnknownLoc(), *input, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::PermuteOp>(
    builder.getUnknownLoc(), *type, *input, properties.permutation);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

}  // namespace ncnn_importer::detail
