#include "ImporterInternal.hpp"

#include <array>
#include <cmath>

namespace ncnn_importer::detail {

namespace {

std::optional<unsigned> resolved_input_index(mlir::Value value) {
  while (auto split = value.getDefiningOp<mlir::ncnn::SplitOp>()) {
    value = split.getInput();
  }
  auto input = value.getDefiningOp<mlir::ncnn::InputOp>();
  if (!input) {
    return std::nullopt;
  }
  auto model = input->getParentOfType<mlir::ncnn::ModelOp>();
  unsigned index = 0;
  for (mlir::ncnn::InputOp candidate : model.getOps<mlir::ncnn::InputOp>()) {
    if (candidate == input) {
      return index;
    }
    ++index;
  }
  return std::nullopt;
}

void add_input_dim_relation(mlir::Value lhs,
                            unsigned lhs_dim,
                            mlir::Value rhs,
                            unsigned rhs_dim,
                            std::int64_t offset) {
  auto lhs_index = resolved_input_index(lhs);
  auto rhs_index = resolved_input_index(rhs);
  if (!lhs_index || !rhs_index) {
    return;
  }
  auto model = lhs.getDefiningOp()->getParentOfType<mlir::ncnn::ModelOp>();
  mlir::Builder builder(model.getContext());
  llvm::SmallVector<mlir::Attribute> relations;
  if (auto existing =
        model->getAttrOfType<mlir::ArrayAttr>("ncnn.input_dim_relations")) {
    relations.append(existing.begin(), existing.end());
  }
  auto relation =
    builder.getDenseI64ArrayAttr({static_cast<std::int64_t>(*lhs_index),
                                  static_cast<std::int64_t>(lhs_dim),
                                  static_cast<std::int64_t>(*rhs_index),
                                  static_cast<std::int64_t>(rhs_dim),
                                  offset});
  if (!llvm::is_contained(relations, relation)) {
    relations.push_back(relation);
    model->setAttr("ncnn.input_dim_relations", builder.getArrayAttr(relations));
  }
}

}  // namespace

ImportResult import_embed(ImportContext& importer,
                          const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 18};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params = ncnn_graph::decode_embed_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !params || !mask) {
    return std::unexpected(make_error(context,
                                      !arity     ? arity.error()
                                      : !allowed ? allowed.error()
                                      : !params  ? params.error()
                                                 : mask.error()));
  }
  const std::size_t expectedWeights = params->has_bias ? 2U : 1U;
  if (params->int8_scale_term != 0 ||
      context.layer.get_weights().size() != expectedWeights) {
    return std::unexpected(make_error(
      context, "Embed supports unquantized table and optional bias only"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto weight =
    importer.make_constant(context, context.layer.get_weights()[0], 0);
  if (!weight) {
    return std::unexpected(weight.error());
  }
  llvm::SmallVector<mlir::Value> bias;
  if (params->has_bias) {
    auto value =
      importer.make_constant(context, context.layer.get_weights()[1], 1);
    if (!value) {
      return std::unexpected(value.error());
    }
    bias.push_back(*value);
  }
  auto& builder = importer.builder();
  mlir::ncnn::EmbedOp::Properties properties;
  properties.input_dim = builder.getI64IntegerAttr(params->input_dim);
  properties.num_output = builder.getI64IntegerAttr(params->output_channels);
  llvm::SmallVector<mlir::Value> operands{*input, *weight};
  operands.append(bias);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::EmbedOp>(
    builder.getUnknownLoc(), operands, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::EmbedOp>(builder.getUnknownLoc(),
                                                *type,
                                                *input,
                                                *weight,
                                                bias,
                                                properties.input_dim,
                                                properties.num_output);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_tanh(ImportContext& importer, const LayerContext& context) {
  constexpr std::array<int, 0> kAllowed{};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !mask || !context.layer.get_weights().empty()) {
    return std::unexpected(make_error(
      context, "TanH requires one input, one output, and no parameters"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto op = builder.create<mlir::ncnn::TanHOp>(
    builder.getUnknownLoc(), input->getType(), *input);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_memory_data(ImportContext& importer,
                                const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 11, 21};
  auto arity = expect_source_arity(context.layer, 0, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params =
    ncnn_graph::decode_memory_data_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !params || !mask ||
      context.layer.get_weights().size() != 1) {
    return std::unexpected(make_error(context, "invalid MemoryData layer"));
  }
  auto value =
    importer.make_constant(context, context.layer.get_weights()[0], 0);
  if (!value) {
    return std::unexpected(value.error());
  }
  return importer.bind_blob(context, context.layer.get_outputs()[0], *value);
}

ImportResult import_swish(ImportContext& importer,
                          const LayerContext& context) {
  constexpr std::array<int, 0> kAllowed{};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !mask || !context.layer.get_weights().empty()) {
    return std::unexpected(make_error(
      context, "Swish requires one input, one output, and no parameters"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& builder = importer.builder();
  auto op = builder.create<mlir::ncnn::SwishOp>(
    builder.getUnknownLoc(), input->getType(), *input);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_layer_norm(ImportContext& importer,
                               const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params =
    ncnn_graph::decode_layer_norm_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !params || !mask) {
    return std::unexpected(make_error(context,
                                      !arity     ? arity.error()
                                      : !allowed ? allowed.error()
                                      : !params  ? params.error()
                                                 : mask.error()));
  }
  const std::size_t expectedWeights = params->affine ? 2 : 0;
  if (context.layer.get_weights().size() != expectedWeights) {
    return std::unexpected(make_error(
      context, "LayerNorm affine mode does not match gamma/beta weights"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  int64_t affineSize = params->affine_size;
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
  if (!params->affine && affineSize == 0 && inputType &&
      inputType.getRank() >= 1 && inputType.getRank() <= 2 &&
      !inputType.isDynamicDim(inputType.getRank() - 1)) {
    affineSize = inputType.getShape().back();
  }
  llvm::SmallVector<mlir::Value> affineParameters;
  for (std::size_t index = 0; index < expectedWeights; ++index) {
    if (context.layer.get_weights()[index].get_dtype() !=
        ncnn_graph::DataType::Float32) {
      return std::unexpected(
        make_error(context, "LayerNorm gamma and beta must be FP32"));
    }
    auto value = importer.make_constant(
      context, context.layer.get_weights()[index], index);
    if (!value) {
      return std::unexpected(value.error());
    }
    affineParameters.push_back(*value);
  }
  auto& builder = importer.builder();
  mlir::ncnn::LayerNormOp::Properties properties;
  properties.affine_size = builder.getI64IntegerAttr(affineSize);
  properties.epsilon = builder.getF32FloatAttr(params->epsilon);
  properties.affine = builder.getBoolAttr(params->affine);
  llvm::SmallVector<mlir::Value> operands{*input};
  operands.append(affineParameters);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::LayerNormOp>(
    builder.getUnknownLoc(), operands, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::LayerNormOp>(builder.getUnknownLoc(),
                                                    *type,
                                                    *input,
                                                    affineParameters,
                                                    properties.affine_size,
                                                    properties.epsilon,
                                                    properties.affine);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_multi_head_attention(ImportContext& importer,
                                         const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1, 2, 3, 4, 5, 6, 7, 18};
  auto arity = expect_source_arity(context.layer, 1, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params =
    ncnn_graph::decode_multi_head_attention_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!arity || !allowed || !params || !mask) {
    return std::unexpected(make_error(context,
                                      !arity     ? arity.error()
                                      : !allowed ? allowed.error()
                                      : !params  ? params.error()
                                                 : mask.error()));
  }
  if (params->has_attention_mask || params->kv_cache ||
      params->int8_scale_term != 0 || params->query_dim != params->key_dim ||
      params->query_dim != params->value_dim ||
      context.layer.get_weights().size() != 8) {
    return std::unexpected(make_error(
      context,
      "MultiHeadAttention supports unmasked, unquantized one-input self "
      "attention with matching qdim/kdim/vdim and eight weights only"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  llvm::SmallVector<mlir::Value> operands{*input};
  for (std::size_t index = 0; index < 8; ++index) {
    if (context.layer.get_weights()[index].get_dtype() !=
        ncnn_graph::DataType::Float32) {
      return std::unexpected(
        make_error(context, "MultiHeadAttention weights must be FP32"));
    }
    auto value = importer.make_constant(
      context, context.layer.get_weights()[index], index);
    if (!value) {
      return std::unexpected(value.error());
    }
    operands.push_back(*value);
  }
  auto& builder = importer.builder();
  mlir::ncnn::MultiHeadAttentionOp::Properties properties;
  properties.embed_dim = builder.getI64IntegerAttr(params->embed_dim);
  properties.num_heads = builder.getI64IntegerAttr(params->num_heads);
  properties.weight_data_size = builder.getI64IntegerAttr(params->weight_count);
  properties.qdim = builder.getI64IntegerAttr(params->query_dim);
  properties.kdim = builder.getI64IntegerAttr(params->key_dim);
  properties.vdim = builder.getI64IntegerAttr(params->value_dim);
  properties.scale = builder.getF32FloatAttr(params->scale);
  auto type =
    importer.infer_single_tensor_result<mlir::ncnn::MultiHeadAttentionOp>(
      builder.getUnknownLoc(), operands, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = builder.create<mlir::ncnn::MultiHeadAttentionOp>(
    builder.getUnknownLoc(),
    *type,
    operands[0],
    operands[1],
    operands[2],
    operands[3],
    operands[4],
    operands[5],
    operands[6],
    operands[7],
    operands[8],
    properties.embed_dim,
    properties.num_heads,
    properties.weight_data_size,
    properties.qdim,
    properties.kdim,
    properties.vdim,
    properties.scale);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_sdpa(ImportContext& importer, const LayerContext& context) {
  constexpr int kAllowed[] = {5, 6, 7, 18};
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params = ncnn_graph::decode_sdpa_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!allowed || !params || !mask || !context.layer.get_weights().empty()) {
    return std::unexpected(make_error(context,
                                      !allowed  ? allowed.error()
                                      : !params ? params.error()
                                      : !mask   ? mask.error()
                                              : "SDPA must not have weights"));
  }
  const bool cacheForm = params->has_attention_mask && params->kv_cache;
  const std::size_t expectedInputs = cacheForm ? 6U : 3U;
  const std::size_t expectedOutputs = cacheForm ? 3U : 1U;
  if ((!cacheForm && (params->has_attention_mask || params->kv_cache)) ||
      context.layer.get_inputs().size() != expectedInputs ||
      context.layer.get_outputs().size() != expectedOutputs) {
    return std::unexpected(make_error(
      context,
      "SDPA decoder supports exactly unmasked 3-to-1 or masked-cache 6-to-3"));
  }

  llvm::SmallVector<mlir::Value> operands;
  operands.reserve(context.layer.get_inputs().size());
  for (std::string_view name : context.layer.get_inputs()) {
    auto value = importer.find_blob(context, name);
    if (!value) {
      return std::unexpected(value.error());
    }
    operands.push_back(*value);
  }
  auto queryType =
    mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
  if (!queryType || queryType.getRank() != 3 || queryType.isDynamicDim(2) ||
      queryType.getShape()[2] <= 0) {
    return std::unexpected(make_error(
      context, "SDPA query feature width must be statically positive"));
  }
  const float scale =
    params->scale == 0.0F
      ? 1.0F / std::sqrt(static_cast<float>(queryType.getShape()[2]))
      : params->scale;
  auto& builder = importer.builder();
  mlir::ncnn::SDPAOp::Properties properties;
  properties.scale = builder.getF32FloatAttr(scale);
  properties.has_mask = builder.getBoolAttr(params->has_attention_mask);
  properties.kv_cache = builder.getBoolAttr(params->kv_cache);
  llvm::SmallVector<mlir::Type> resultTypes;
  if (mlir::failed(importer.infer_tensor_results<mlir::ncnn::SDPAOp>(
        builder.getUnknownLoc(), operands, properties, resultTypes))) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op =
    builder.create<mlir::ncnn::SDPAOp>(builder.getUnknownLoc(),
                                       resultTypes,
                                       operands[0],
                                       operands[1],
                                       operands[2],
                                       mlir::ValueRange(operands).drop_front(3),
                                       properties.scale,
                                       properties.has_mask,
                                       properties.kv_cache);
  importer.tag_source(op, context);
  if (cacheForm) {
    auto keyType = mlir::cast<mlir::RankedTensorType>(operands[1].getType());
    if (!keyType.isDynamicDim(1)) {
      add_input_dim_relation(
        operands[3], 1, operands[4], 1, keyType.getShape()[1]);
    }
    add_input_dim_relation(operands[4], 1, operands[5], 1, 0);
  }
  for (auto [name, result] :
       llvm::zip(context.layer.get_outputs(), op.getResults())) {
    auto bound = importer.bind_blob(context, name, result);
    if (!bound) {
      return bound;
    }
  }
  return {};
}

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
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
  if (inputType && inputType.getRank() == 1) {
    *axis = 0;
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
