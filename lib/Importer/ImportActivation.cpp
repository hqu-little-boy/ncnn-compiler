#include "ImporterInternal.hpp"

namespace ncnn_importer::detail {

ImportResult import_relu(ImportContext& importer, const LayerContext& context) {
  auto arity = expect_source_arity(context.layer, 1, 1);
  constexpr int kAllowed[] = {0};
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto slope = get_float(context.layer.get_params(), 0, 0.0F, "negative_slope");
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
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto relu =
    builder.create<mlir::ncnn::ReluOp>(builder.getUnknownLoc(),
                                       input->getType(),
                                       *input,
                                       builder.getF32FloatAttr(*slope));
  importer.tag_source(relu.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), relu.getResult());
}

ImportResult import_gelu(ImportContext& importer, const LayerContext& context) {
  auto arity = expect_source_arity(context.layer, 1, 1);
  constexpr int kAllowed[] = {0};
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto fast = get_int(context.layer.get_params(), 0, 0, "fast_gelu");
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !fast || !mask || *fast != 0) {
    return std::unexpected(make_error(context, "invalid GELU parameters"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto op = builder.create<mlir::ncnn::GELUOp>(builder.getUnknownLoc(),
                                               input->getType(),
                                               *input,
                                               builder.getBoolAttr(*fast));
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_batch_norm(ImportContext& importer,
                               const LayerContext& context) {
  auto arity = expect_source_arity(context.layer, 1, 1);
  constexpr int kAllowed[] = {0, 1};
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params =
    ncnn_graph::decode_batch_norm_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !params || !mask ||
      context.layer.get_weights().size() != 4) {
    return std::unexpected(make_error(context, "invalid BatchNorm parameters"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  llvm::SmallVector<mlir::Value> values{*input};
  for (std::size_t index = 0; index < 4; ++index) {
    if (context.layer.get_weights()[index].get_dtype() !=
        ncnn_graph::DataType::Float32) {
      return std::unexpected(
        make_error(context, "BatchNorm parameters must be FP32"));
    }
    auto value = importer.make_constant(
      context, context.layer.get_weights()[index], index);
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(*value);
  }
  auto& builder = importer.builder();
  mlir::ncnn::BatchNormOp::Properties properties;
  properties.epsilon = builder.getF32FloatAttr(params->epsilon);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::BatchNormOp>(
    builder.getUnknownLoc(), values, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::BatchNormOp>(builder.getUnknownLoc(),
                                                    *type,
                                                    values[0],
                                                    values[1],
                                                    values[2],
                                                    values[3],
                                                    values[4],
                                                    properties.epsilon);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_dropout(ImportContext& importer,
                            const LayerContext& context) {
  auto arity = expect_source_arity(context.layer, 1, 1);
  constexpr int kAllowed[] = {0};
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto scale = get_float(context.layer.get_params(), 0, 1.0F, "scale");
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
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto dropout =
    builder.create<mlir::ncnn::DropoutOp>(builder.getUnknownLoc(),
                                          input->getType(),
                                          *input,
                                          builder.getF32FloatAttr(*scale));
  importer.tag_source(dropout.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), dropout.getResult());
}

ImportResult import_softmax(ImportContext& importer,
                            const LayerContext& context) {
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
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto softmax =
    builder.create<mlir::ncnn::SoftmaxOp>(builder.getUnknownLoc(),
                                          input->getType(),
                                          *input,
                                          builder.getI64IntegerAttr(*axis));
  importer.tag_source(softmax.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), softmax.getResult());
}

}  // namespace ncnn_importer::detail
