#include "ImporterInternal.hpp"

#include <cmath>
#include <cstring>
#include <format>

namespace ncnn_importer::detail {
namespace {

std::expected<void, std::string> validate_scale(
  const ncnn_graph::Tensor& tensor,
  std::size_t expectedSize,
  std::string_view role,
  bool allowEmptyChannel = false) {
  if (tensor.get_dtype() != ncnn_graph::DataType::Float32 ||
      tensor.get_shape().size() != 1 ||
      tensor.get_shape()[0] !=
        static_cast<decltype(tensor.get_shape()[0])>(expectedSize)) {
    return std::unexpected(
      std::format("{} scale must be FP32 [{}]", role, expectedSize));
  }
  const auto data = tensor.get_data();
  for (std::size_t offset = 0; offset < data.size(); offset += sizeof(float)) {
    float value;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    const bool invalid = allowEmptyChannel
                           ? std::isnan(value) || value < 0.0F
                           : !std::isfinite(value) || value <= 0.0F;
    if (invalid) {
      return std::unexpected(std::format(
        "{} scale values must be {}",
        role,
        allowEmptyChannel ? "nonnegative and not NaN" : "finite and positive"));
    }
  }
  return {};
}

std::expected<llvm::SmallVector<int64_t>, std::string> parse_shape_expression(
  std::string_view expression, mlir::ValueRange inputs) {
  llvm::SmallVector<int64_t> sources;
  std::size_t begin = 0;
  while (begin < expression.size()) {
    const std::size_t end = expression.find(',', begin);
    const std::string_view token = expression.substr(
      begin,
      end == std::string_view::npos ? expression.size() - begin : end - begin);
    if (token.size() != 2 || token[0] < '0' || token[0] > '9') {
      return std::unexpected(
        "reshape shape expression currently requires dimension references");
    }
    const auto inputIndex = static_cast<unsigned>(token[0] - '0');
    if (inputIndex >= inputs.size()) {
      return std::unexpected("reshape shape expression input is out of range");
    }
    auto type =
      mlir::dyn_cast<mlir::RankedTensorType>(inputs[inputIndex].getType());
    if (!type) {
      return std::unexpected("reshape shape expression input must be ranked");
    }
    int64_t dimension = -1;
    switch (token[1]) {
      case 'w':
        dimension = type.getRank() - 1;
        break;
      case 'h':
        dimension = type.getRank() - 2;
        break;
      case 'd':
        dimension = type.getRank() == 4 ? 1 : -1;
        break;
      case 'c':
        dimension = type.getRank() >= 3 ? 0 : -1;
        break;
      default:
        break;
    }
    if (dimension < 0 || dimension >= type.getRank()) {
      return std::unexpected(
        "reshape shape expression references a missing dimension");
    }
    sources.push_back(inputIndex);
    sources.push_back(dimension);
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  if (sources.empty() || sources.size() > 8) {
    return std::unexpected(
      "reshape shape expression must produce rank 1 through 4");
  }

  llvm::SmallVector<int64_t> ncnnOrder(sources);
  sources.clear();
  for (std::size_t index = ncnnOrder.size(); index > 0; index -= 2) {
    sources.push_back(ncnnOrder[index - 2]);
    sources.push_back(ncnnOrder[index - 1]);
  }
  return sources;
}
ImportResult arity_params(const LayerContext& context,
                          std::size_t in,
                          std::size_t out,
                          std::span<const int> ids) {
  auto arity = expect_source_arity(context.layer, in, out);
  auto allowed = validate_param_ids(context.layer.get_params(), ids);
  if (!arity) {
    return std::unexpected(make_error(context, arity.error()));
  }
  if (!allowed) {
    return std::unexpected(make_error(context, allowed.error()));
  }
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!mask) {
    return std::unexpected(make_error(context, mask.error()));
  }
  return {};
}

ImportResult import_unary_activation(ImportContext& importer,
                                     const LayerContext& context,
                                     bool swish) {
  constexpr int ids[] = {0, 1};
  auto valid = arity_params(context, 1, 1, ids);
  auto alpha = get_float(context.layer.get_params(), 0, 0.2F, "alpha");
  auto beta = get_float(context.layer.get_params(), 1, 0.5F, "beta");
  if (!valid) {
    return std::unexpected(valid.error());
  }
  if (!alpha || !beta || *alpha <= 0.0F) {
    return std::unexpected(make_error(context,
                                      !alpha  ? alpha.error()
                                      : !beta ? beta.error()
                                              : "alpha must be positive"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto& b = importer.builder();
  mlir::Operation* op =
    swish ? b.create<mlir::ncnn::HardSwishOp>(b.getUnknownLoc(),
                                              (*input).getType(),
                                              *input,
                                              b.getF32FloatAttr(*alpha),
                                              b.getF32FloatAttr(*beta))
          : b.create<mlir::ncnn::HardSigmoidOp>(b.getUnknownLoc(),
                                                (*input).getType(),
                                                *input,
                                                b.getF32FloatAttr(*alpha),
                                                b.getF32FloatAttr(*beta));
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op->getResult(0));
}
}  // namespace

ImportResult import_hard_sigmoid(ImportContext& i, const LayerContext& c) {
  return import_unary_activation(i, c, false);
}
ImportResult import_hard_swish(ImportContext& i, const LayerContext& c) {
  return import_unary_activation(i, c, true);
}

ImportResult import_reshape(ImportContext& importer,
                            const LayerContext& context) {
  constexpr int ids[] = {0, 1, 2, 6, 11};
  auto validParams = validate_param_ids(context.layer.get_params(), ids);
  if (!validParams) {
    return std::unexpected(make_error(context, validParams.error()));
  }
  auto featureMask = validate_feature_mask(context.layer.get_params());
  if (!featureMask) {
    return std::unexpected(make_error(context, featureMask.error()));
  }
  if (context.layer.get_outputs().size() != 1 ||
      context.layer.get_inputs().empty()) {
    return std::unexpected(make_error(
      context, "reshape requires at least one input and one output"));
  }
  const auto& p = context.layer.get_params();
  const auto expression = p.get_string(6);
  if (expression && expression->get().empty()) {
    return std::unexpected(
      make_error(context, "reshape shape expression must not be empty"));
  }
  if (!expression && context.layer.get_inputs().size() != 1) {
    return std::unexpected(make_error(
      context, "reshape without a shape expression requires one input"));
  }
  auto w = get_int(p, 0, -233, "w");
  auto h = get_int(p, 1, -233, "h");
  auto d = get_int(p, 11, -233, "d");
  auto c = get_int(p, 2, -233, "c");
  if (!w || !h || !d || !c) {
    return std::unexpected(
      make_error(context, "reshape parameter type is invalid"));
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (const std::string& name : context.layer.get_inputs()) {
    auto value = importer.find_blob(context, name);
    if (!value) {
      return std::unexpected(value.error());
    }
    inputs.push_back(*value);
  }
  mlir::Value input = inputs.front();
  auto input_type = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!input_type) {
    return std::unexpected(make_error(context, "reshape input must be ranked"));
  }
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> shapeSources;
  if (expression) {
    auto parsed = parse_shape_expression(expression->get(), inputs);
    if (!parsed) {
      return std::unexpected(make_error(context, parsed.error()));
    }
    shapeSources = std::move(*parsed);
    const int64_t sourceInput = shapeSources.front();
    int64_t dimension = 0;
    for (std::size_t index = 0; index < shapeSources.size(); index += 2) {
      if (shapeSources[index] != sourceInput ||
          shapeSources[index + 1] != dimension) {
        return std::unexpected(
          make_error(context,
                     "reshape shape expression must preserve one input's "
                     "dimension order"));
      }
      ++dimension;
    }
    for (std::size_t index = 0; index < shapeSources.size(); index += 2) {
      auto sourceType = mlir::cast<mlir::RankedTensorType>(
        inputs[shapeSources[index]].getType());
      shape.push_back(sourceType.getShape()[shapeSources[index + 1]]);
    }
  }
  const auto input_shape = input_type.getShape();
  llvm::SmallVector<int64_t> rawShape;
  llvm::SmallVector<int64_t> zeroSources;
  if (!expression) {
    const int64_t raw[] = {*c, *d, *h, *w};
    const int64_t source[] = {
      0, 1, input_type.getRank() - 2, input_type.getRank() - 1};
    for (int index = 0; index < 4; ++index) {
      const int64_t dimension = raw[index];
      if (dimension != -233) {
        rawShape.push_back(dimension);
        zeroSources.push_back(dimension == 0 ? source[index] : -1);
      }
    }
    if (llvm::count(rawShape, -1) > 1) {
      return std::unexpected(
        make_error(context, "reshape allows at most one -1 dimension"));
    }
  }
  if (!expression && *w == 0) {
    *w = input_shape.back();
  }
  if (!expression && *h == 0) {
    if (input_type.getRank() < 2) {
      return std::unexpected(
        make_error(context, "reshape cannot copy missing height dimension"));
    }
    *h = input_shape[input_type.getRank() - 2];
  }
  if (!expression && *d == 0) {
    if (input_type.getRank() != 4) {
      return std::unexpected(
        make_error(context, "reshape cannot copy missing depth dimension"));
    }
    *d = input_shape[1];
  }
  if (!expression && *c == 0) {
    if (input_type.getRank() < 3) {
      return std::unexpected(
        make_error(context, "reshape cannot copy missing channel dimension"));
    }
    *c = input_shape.front();
  }
  if (!expression) {
    for (int64_t dimension : {*c, *d, *h, *w}) {
      if (dimension != -233) {
        shape.push_back(dimension);
      }
    }
  }
  auto& b = importer.builder();
  mlir::ncnn::ReshapeOp::Properties props;
  props.shape = b.getDenseI64ArrayAttr(shape);
  if (!expression) {
    props.shape_spec = b.getDenseI64ArrayAttr(rawShape);
    props.shape_zero_sources = b.getDenseI64ArrayAttr(zeroSources);
  }
  if (expression) {
    props.shape_sources = b.getDenseI64ArrayAttr(shapeSources);
    props.shape_expression = b.getStringAttr(expression->get());
  }
  auto type = importer.infer_single_tensor_result<mlir::ncnn::ReshapeOp>(
    b.getUnknownLoc(), inputs, props);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op =
    b.create<mlir::ncnn::ReshapeOp>(b.getUnknownLoc(),
                                    *type,
                                    input,
                                    mlir::ValueRange(inputs).drop_front(),
                                    props.shape,
                                    props.shape_spec,
                                    props.shape_zero_sources,
                                    props.shape_sources,
                                    props.shape_expression);
  importer.tag_source(op.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op.getResult());
}

ImportResult import_binary_op(ImportContext& importer,
                              const LayerContext& context) {
  constexpr int ids[] = {0, 1, 2};
  auto valid = arity_params(context, context.layer.get_inputs().size(), 1, ids);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  if (context.layer.get_inputs().size() != 1 &&
      context.layer.get_inputs().size() != 2) {
    return std::unexpected(
      make_error(context, "BinaryOp requires one or two inputs"));
  }
  auto type = get_int(context.layer.get_params(), 0, 0, "op_type");
  auto scalar = get_float(context.layer.get_params(), 2, 0.0F, "scalar");
  auto ws = get_int(context.layer.get_params(), 1, 0, "with_scalar");
  if (!type || !scalar || !ws ||
      (*type != 0 && *type != 1 && *type != 2 && *type != 4) ||
      (*ws != 0 && *ws != 1) ||
      ((*ws == 1) != (context.layer.get_inputs().size() == 1))) {
    return std::unexpected(make_error(
      context,
      "BinaryOp supports add/subtract/multiply/max with scalar or two inputs "
      "only"));
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (const auto& n : context.layer.get_inputs()) {
    auto v = importer.find_blob(context, n);
    if (!v) {
      return std::unexpected(v.error());
    }
    inputs.push_back(*v);
  }
  auto& b = importer.builder();
  mlir::ncnn::BinaryOp::Properties props;
  props.scalar = b.getF32FloatAttr(*scalar);
  props.with_scalar = b.getBoolAttr(*ws);
  props.op_type = b.getI64IntegerAttr(*type);
  auto result = importer.infer_single_tensor_result<mlir::ncnn::BinaryOp>(
    b.getUnknownLoc(), inputs, props);
  if (mlir::failed(result)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = b.create<mlir::ncnn::BinaryOp>(b.getUnknownLoc(),
                                           *result,
                                           inputs,
                                           props.scalar,
                                           props.with_scalar,
                                           props.op_type);
  importer.tag_source(op.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op.getResult());
}

ImportResult import_eltwise(ImportContext& importer,
                            const LayerContext& context) {
  constexpr int kAllowed[] = {0, 1};
  auto arity = expect_source_arity(context.layer, 2, 1);
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto operation = get_int(context.layer.get_params(), 0, 1, "op_type");
  auto mask = validate_feature_mask(context.layer.get_params());
  const ncnn_graph::ParamValue* coefficientValue =
    find_param(context.layer.get_params(), 1);
  auto coefficients =
    coefficientValue == nullptr
      ? std::optional<std::span<const float>>(std::span<const float>())
      : coefficientValue->get_float_array();
  if (!arity || !allowed || !operation || !mask || *operation != 1 ||
      !coefficients || (!coefficients->empty() && coefficients->size() != 2) ||
      !context.layer.get_weights().empty()) {
    return std::unexpected(
      make_error(context,
                 "Eltwise supports unweighted or coefficient-weighted SUM of "
                 "two inputs only"));
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (std::string_view name : context.layer.get_inputs()) {
    auto input = importer.find_blob(context, name);
    if (!input) {
      return std::unexpected(input.error());
    }
    inputs.push_back(*input);
  }
  auto& builder = importer.builder();
  auto createBinary =
    [&](mlir::ValueRange values,
        float scalar,
        bool withScalar,
        std::int64_t opType) -> std::expected<mlir::Value, ImportError> {
    mlir::ncnn::BinaryOp::Properties properties;
    properties.scalar = builder.getF32FloatAttr(scalar);
    properties.with_scalar = builder.getBoolAttr(withScalar);
    properties.op_type = builder.getI64IntegerAttr(opType);
    auto type = importer.infer_single_tensor_result<mlir::ncnn::BinaryOp>(
      builder.getUnknownLoc(), values, properties);
    if (mlir::failed(type)) {
      return std::unexpected(
        make_error(context, importer.captured_diagnostic()));
    }
    auto binary = builder.create<mlir::ncnn::BinaryOp>(builder.getUnknownLoc(),
                                                       *type,
                                                       values,
                                                       properties.scalar,
                                                       properties.with_scalar,
                                                       properties.op_type);
    importer.tag_source(binary, context);
    return binary.getOutput();
  };
  if (!coefficients->empty()) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if ((*coefficients)[index] != 1.0F) {
        auto scaled =
          createBinary(inputs[index], (*coefficients)[index], true, 2);
        if (!scaled) {
          return std::unexpected(scaled.error());
        }
        inputs[index] = *scaled;
      }
    }
  }
  auto sum = createBinary(inputs, 0.0F, false, 0);
  if (!sum) {
    return std::unexpected(sum.error());
  }
  return importer.bind_blob(context, context.layer.get_outputs()[0], *sum);
}

ImportResult import_gemm(ImportContext& importer, const LayerContext& context) {
  constexpr int kAllowed[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 20, 21, 22};
  auto allowed = validate_param_ids(context.layer.get_params(), kAllowed);
  auto params = ncnn_graph::decode_gemm_params(context.layer.get_params());
  auto mask = validate_feature_mask(context.layer.get_params());
  if (!allowed || !params || !mask) {
    return std::unexpected(make_error(context,
                                      !allowed  ? allowed.error()
                                      : !params ? params.error()
                                                : mask.error()));
  }
  const bool dynamicMatrices =
    !params->constant_a && !params->constant_b && !params->constant_c;
  if (dynamicMatrices) {
    auto arity = expect_source_arity(context.layer, 2, 1);
    if (!arity || !context.layer.get_weights().empty() || params->transpose_a ||
        params->transpose_b || params->output_n1m != 0 ||
        params->output_elempack != 0 || params->output_elemtype != 0 ||
        params->output_transpose != 0 || params->int8_scale_term != 0) {
      return std::unexpected(make_error(
        context, "dynamic Gemm requires plain rank-2 A and B inputs"));
    }
    auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
    auto weight = importer.find_blob(context, context.layer.get_inputs()[1]);
    if (!input || !weight) {
      return std::unexpected(!input ? input.error() : weight.error());
    }
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight->getType());
    if (!inputType || !weightType || inputType.getRank() != 2 ||
        weightType.getRank() != 2 || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape() ||
        inputType.getShape()[1] != weightType.getShape()[0]) {
      return std::unexpected(make_error(
        context, "dynamic Gemm A [M,K] and B [K,N] shapes must match"));
    }
    auto& builder = importer.builder();
    auto transposedType = mlir::RankedTensorType::get(
      {weightType.getShape()[1], weightType.getShape()[0]},
      weightType.getElementType());
    auto transposed = builder.create<mlir::ncnn::PermuteOp>(
      builder.getUnknownLoc(),
      transposedType,
      *weight,
      builder.getDenseI64ArrayAttr({1, 0}));
    importer.tag_source(transposed, context);
    auto biasType = mlir::RankedTensorType::get({weightType.getShape()[1]},
                                                builder.getF32Type());
    auto biasValue =
      mlir::DenseElementsAttr::get(biasType, builder.getF32FloatAttr(0.0F));
    auto bias = builder.create<mlir::ncnn::ConstOp>(
      builder.getUnknownLoc(),
      biasType,
      builder.getStringAttr(
        std::format("{}.zero_bias", context.layer.get_name())),
      biasValue);
    importer.tag_source(bias, context);
    mlir::ncnn::GemmOp::Properties properties;
    properties.alpha = builder.getF32FloatAttr(params->alpha);
    properties.beta = builder.getF32FloatAttr(params->beta);
    properties.int8_scale_term = builder.getI64IntegerAttr(0);
    llvm::SmallVector<mlir::Value> values{
      *input, transposed.getOutput(), bias.getOutput()};
    auto type = importer.infer_single_tensor_result<mlir::ncnn::GemmOp>(
      builder.getUnknownLoc(), values, properties);
    if (mlir::failed(type)) {
      return std::unexpected(
        make_error(context, importer.captured_diagnostic()));
    }
    auto op = builder.create<mlir::ncnn::GemmOp>(builder.getUnknownLoc(),
                                                 *type,
                                                 values[0],
                                                 values[1],
                                                 values[2],
                                                 mlir::ValueRange{},
                                                 properties.alpha,
                                                 properties.beta,
                                                 properties.int8_scale_term);
    importer.tag_source(op, context);
    return importer.bind_blob(
      context, context.layer.get_outputs()[0], op.getOutput());
  }
  if (!params->constant_a && params->constant_b && params->constant_c &&
      !params->transpose_a && params->transpose_b &&
      params->broadcast_c == -1 && params->output_n1m == 0 &&
      params->output_elempack == 0 && params->output_elemtype == 0 &&
      params->output_transpose == 0 && params->int8_scale_term == 0) {
    auto arity = expect_source_arity(context.layer, 1, 1);
    const auto dtype = context.layer.get_weights().empty()
                         ? ncnn_graph::DataType::Unknown
                         : context.layer.get_weights()[0].get_dtype();
    if (!arity || context.layer.get_weights().size() != 1 ||
        (dtype != ncnn_graph::DataType::Float16 &&
         dtype != ncnn_graph::DataType::Float32)) {
      return std::unexpected(make_error(
        context, "constant Gemm without C requires one FP16 or FP32 B tensor"));
    }
    auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
    if (!input) {
      return std::unexpected(input.error());
    }
    auto weight =
      importer.make_constant(context, context.layer.get_weights()[0], 0, true);
    if (!weight) {
      return std::unexpected(weight.error());
    }
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input->getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight->getType());
    if (!inputType || !weightType || inputType.getRank() != 2 ||
        weightType.getRank() != 2 || inputType.isDynamicDim(1) ||
        !weightType.hasStaticShape()) {
      return std::unexpected(make_error(
        context,
        std::format("constant Gemm requires rank-2 input with static K and "
                    "static rank-2 weight; got ranks {} and {} (static K {} "
                    "and weight {})",
                    inputType ? inputType.getRank() : -1,
                    weightType ? weightType.getRank() : -1,
                    inputType && !inputType.isDynamicDim(1),
                    weightType && weightType.hasStaticShape())));
    }
    if (inputType.getShape()[1] != weightType.getShape()[1]) {
      return std::unexpected(make_error(
        context,
        std::format("constant Gemm K mismatch: input {} vs weight {}",
                    inputType.getShape()[1],
                    weightType.getShape()[1])));
    }
    auto& builder = importer.builder();
    auto biasType =
      mlir::RankedTensorType::get({params->constant_n}, builder.getF32Type());
    auto biasValue =
      mlir::DenseElementsAttr::get(biasType, builder.getF32FloatAttr(0.0F));
    auto bias = builder.create<mlir::ncnn::ConstOp>(
      builder.getUnknownLoc(),
      biasType,
      builder.getStringAttr(
        std::format("{}.zero_bias", context.layer.get_name())),
      biasValue);
    importer.tag_source(bias, context);
    mlir::ncnn::GemmOp::Properties properties;
    properties.alpha = builder.getF32FloatAttr(params->alpha);
    properties.beta = builder.getF32FloatAttr(params->beta);
    properties.int8_scale_term = builder.getI64IntegerAttr(0);
    llvm::SmallVector<mlir::Value> values{*input, *weight, bias.getOutput()};
    auto type = importer.infer_single_tensor_result<mlir::ncnn::GemmOp>(
      builder.getUnknownLoc(), values, properties);
    if (mlir::failed(type)) {
      return std::unexpected(
        make_error(context, importer.captured_diagnostic()));
    }
    auto op = builder.create<mlir::ncnn::GemmOp>(builder.getUnknownLoc(),
                                                 *type,
                                                 values[0],
                                                 values[1],
                                                 values[2],
                                                 mlir::ValueRange{},
                                                 properties.alpha,
                                                 properties.beta,
                                                 properties.int8_scale_term);
    importer.tag_source(op, context);
    return importer.bind_blob(
      context, context.layer.get_outputs()[0], op.getOutput());
  }
  auto arity = expect_source_arity(context.layer, 1, 1);
  if (!arity ||
      context.layer.get_weights().size() !=
        (params->int8_scale_term != 0 ? 3U : 2U) ||
      params->constant_a || !params->constant_b || !params->constant_c ||
      params->transpose_a || !params->transpose_b || params->broadcast_c != 4 ||
      params->output_n1m != 0 || params->output_elempack != 0 ||
      params->output_elemtype != 0 || params->output_transpose != 0 ||
      (params->int8_scale_term != 0 && params->int8_scale_term != 1 &&
       params->int8_scale_term != 2)) {
    return std::unexpected(
      make_error(context,
                 "only dynamic-A, transposed constant-B Gemm with row bias "
                 "and optional int8 B is supported"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  llvm::SmallVector<mlir::Value> values{*input};
  const bool quantized = params->int8_scale_term != 0;
  for (std::size_t index = 0; index < context.layer.get_weights().size();
       ++index) {
    const auto dtype = context.layer.get_weights()[index].get_dtype();
    const bool validDtype =
      index == 0 ? quantized ? dtype == ncnn_graph::DataType::Int8
                             : dtype == ncnn_graph::DataType::Float16 ||
                                 dtype == ncnn_graph::DataType::Float32
                 : dtype == ncnn_graph::DataType::Float32;
    if (!validDtype) {
      return std::unexpected(
        make_error(context, "Gemm weight, bias, or scale has invalid dtype"));
    }
    auto value = importer.make_constant(context,
                                        context.layer.get_weights()[index],
                                        index,
                                        index == 0 && !quantized);
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(*value);
  }
  auto& builder = importer.builder();
  mlir::ncnn::GemmOp::Properties properties;
  properties.alpha = builder.getF32FloatAttr(params->alpha);
  properties.beta = builder.getF32FloatAttr(params->beta);
  properties.int8_scale_term =
    builder.getI64IntegerAttr(params->int8_scale_term);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::GemmOp>(
    builder.getUnknownLoc(), values, properties);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op =
    builder.create<mlir::ncnn::GemmOp>(builder.getUnknownLoc(),
                                       *type,
                                       values[0],
                                       values[1],
                                       values[2],
                                       mlir::ValueRange(values).drop_front(3),
                                       properties.alpha,
                                       properties.beta,
                                       properties.int8_scale_term);
  importer.tag_source(op, context);
  return importer.bind_blob(
    context, context.layer.get_outputs()[0], op.getOutput());
}

ImportResult import_inner_product(ImportContext& importer,
                                  const LayerContext& context) {
  constexpr int ids[] = {0, 1, 2, 8, 9, 10};
  auto valid = arity_params(context, 1, 1, ids);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  auto decoded =
    ncnn_graph::decode_inner_product_params(context.layer.get_params());
  auto act = get_int(context.layer.get_params(), 9, 0, "activation_type");
  if (!decoded) {
    return std::unexpected(make_error(context, decoded.error()));
  }
  const auto& params = *decoded;
  const bool quantized = params.int8_scale_term != 0;
  if (!act || (*act != 0 && *act != 1 && *act != 4) ||
      (*act != 0 && context.layer.get_params().has(10)) ||
      context.layer.get_weights().size() != params.expected_weight_tensors()) {
    return std::unexpected(
      make_error(context, "unsupported InnerProduct configuration"));
  }
  const auto kernelType = context.layer.get_weights()[0].get_dtype();
  if ((!quantized && kernelType != ncnn_graph::DataType::Float32) ||
      (quantized && kernelType != ncnn_graph::DataType::Float32 &&
       kernelType != ncnn_graph::DataType::Int8)) {
    return std::unexpected(make_error(
      context, "InnerProduct kernel element type does not match scale term"));
  }
  if (params.has_bias && context.layer.get_weights()[1].get_dtype() !=
                           ncnn_graph::DataType::Float32) {
    return std::unexpected(
      make_error(context, "InnerProduct bias must be FP32"));
  }
  const std::size_t scaleOffset = 1 + static_cast<std::size_t>(params.has_bias);
  if (quantized) {
    auto weightScale =
      validate_scale(context.layer.get_weights()[scaleOffset],
                     static_cast<std::size_t>(params.output_channels),
                     "InnerProduct weight",
                     true);
    auto inputScale = validate_scale(
      context.layer.get_weights()[scaleOffset + 1], 1, "InnerProduct input");
    if (!weightScale || !inputScale) {
      return std::unexpected(make_error(
        context, !weightScale ? weightScale.error() : inputScale.error()));
    }
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto weight = importer.make_constant(
    context, context.layer.get_weights()[0], 0, !quantized);
  if (!weight) {
    return std::unexpected(weight.error());
  }
  llvm::SmallVector<mlir::Value> tail;
  for (std::size_t index = 1; index < params.expected_weight_tensors();
       ++index) {
    auto v = importer.make_constant(
      context, context.layer.get_weights()[index], index);
    if (!v) {
      return std::unexpected(v.error());
    }
    tail.push_back(*v);
  }
  auto& b = importer.builder();
  mlir::ncnn::InnerProductOp::Properties props;
  props.has_bias = b.getBoolAttr(params.has_bias);
  props.int8_scale_term = b.getI64IntegerAttr(params.int8_scale_term);
  llvm::SmallVector<mlir::Value> operands{*input, *weight};
  operands.append(tail);
  auto result = importer.infer_single_tensor_result<mlir::ncnn::InnerProductOp>(
    b.getUnknownLoc(), operands, props);
  if (mlir::failed(result)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = b.create<mlir::ncnn::InnerProductOp>(b.getUnknownLoc(),
                                                 *result,
                                                 *input,
                                                 *weight,
                                                 tail,
                                                 props.has_bias,
                                                 props.int8_scale_term);
  importer.tag_source(op.getOperation(), context);
  mlir::Value output = op.getResult();
  if (*act == 1) {
    auto relu = b.create<mlir::ncnn::ReluOp>(
      b.getUnknownLoc(), *result, output, b.getF32FloatAttr(0.0F));
    output = relu.getResult();
  } else if (*act == 4) {
    auto sigmoid =
      b.create<mlir::ncnn::SigmoidOp>(b.getUnknownLoc(), *result, output);
    output = sigmoid.getOutput();
  }
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), output);
}

ImportResult import_convolution_depthwise(ImportContext& importer,
                                          const LayerContext& context) {
  constexpr int ids[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                         11, 12, 13, 14, 15, 16, 18, 19, 30, 31};
  auto valid = arity_params(context, 1, 1, ids);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  auto p =
    ncnn_graph::decode_convolution_depthwise_params(context.layer.get_params());
  auto activation =
    get_int(context.layer.get_params(), 9, 0, "activation_type");
  auto padValue = get_float(context.layer.get_params(), 18, 0.0F, "pad_value");
  if (!p) {
    return std::unexpected(make_error(context, p.error()));
  }
  if (!activation || !padValue) {
    return std::unexpected(
      make_error(context, !activation ? activation.error() : padValue.error()));
  }
  if (p->dynamic_weight || *activation < 0 || *activation > 1 ||
      context.layer.get_params().has(10) || *padValue != 0.0F) {
    return std::unexpected(make_error(
      context,
      "only static pure depthwise convolution with FP32 computation is "
      "supported"));
  }
  const auto kernelType = context.layer.get_weights().empty()
                            ? ncnn_graph::DataType::Unknown
                            : context.layer.get_weights()[0].get_dtype();
  const bool quantized = p->int8_scale_term != 0;
  if (context.layer.get_weights().size() != p->expected_weight_tensors() ||
      context.layer.get_weights().empty() ||
      (!quantized && kernelType != ncnn_graph::DataType::Float32 &&
       kernelType != ncnn_graph::DataType::Float16 &&
       kernelType != ncnn_graph::DataType::BFloat16) ||
      (quantized && kernelType != ncnn_graph::DataType::Float32 &&
       kernelType != ncnn_graph::DataType::Int8)) {
    return std::unexpected(make_error(
      context,
      "depthwise weights must use static FP32 or FP16-storage kernel and "
      "optional FP32 bias"));
  }
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto inputType = llvm::dyn_cast<mlir::RankedTensorType>(input->getType());
  if (!inputType || inputType.getRank() != 3 ||
      p->group != inputType.getShape()[0] ||
      p->output_channels != inputType.getShape()[0] ||
      context.layer.get_weights()[0].get_shape().size() != 4 ||
      context.layer.get_weights()[0].get_shape()[0] != p->output_channels ||
      context.layer.get_weights()[0].get_shape()[1] != 1 ||
      context.layer.get_weights()[0].get_shape()[2] != p->kernel_h ||
      context.layer.get_weights()[0].get_shape()[3] != p->kernel_w) {
    return std::unexpected(
      make_error(context, "depthwise group must equal input channels"));
  }
  auto weight =
    importer.make_constant(context, context.layer.get_weights()[0], 0);
  if (!weight) {
    return std::unexpected(weight.error());
  }
  llvm::SmallVector<mlir::Value> tail;
  if (p->has_bias) {
    if (context.layer.get_weights()[1].get_dtype() !=
        ncnn_graph::DataType::Float32) {
      return std::unexpected(
        make_error(context, "depthwise bias must be FP32"));
    }
    auto value =
      importer.make_constant(context, context.layer.get_weights()[1], 1);
    if (!value) {
      return std::unexpected(value.error());
    }
    tail.push_back(*value);
  }
  if (quantized) {
    const std::size_t scaleOffset = 1 + static_cast<std::size_t>(p->has_bias);
    const std::size_t weightScaleSize =
      p->int8_scale_term == 1 || p->int8_scale_term == 101
        ? static_cast<std::size_t>(p->group)
        : 1;
    auto weightScale = validate_scale(context.layer.get_weights()[scaleOffset],
                                      weightScaleSize,
                                      "depthwise weight",
                                      true);
    auto inputScale = validate_scale(
      context.layer.get_weights()[scaleOffset + 1], 1, "depthwise input");
    if (!weightScale || !inputScale) {
      return std::unexpected(make_error(
        context, !weightScale ? weightScale.error() : inputScale.error()));
    }
    if (p->int8_scale_term > 100) {
      auto outputScale = validate_scale(
        context.layer.get_weights()[scaleOffset + 2], 1, "depthwise output");
      if (!outputScale) {
        return std::unexpected(make_error(context, outputScale.error()));
      }
    }
    for (std::size_t index = scaleOffset; index < p->expected_weight_tensors();
         ++index) {
      auto value = importer.make_constant(
        context, context.layer.get_weights()[index], index);
      if (!value) {
        return std::unexpected(value.error());
      }
      tail.push_back(*value);
    }
  }
  auto& b = importer.builder();
  auto i64 = [&b](int64_t x) {
    return b.getI64IntegerAttr(x);
  };
  mlir::ncnn::ConvolutionDepthWiseOp::Properties props;
  props.kernel_h = i64(p->kernel_h);
  props.kernel_w = i64(p->kernel_w);
  props.stride_h = i64(p->stride_h);
  props.stride_w = i64(p->stride_w);
  props.dilation_h = i64(p->dilation_h);
  props.dilation_w = i64(p->dilation_w);
  props.pad_top = i64(p->pad_top);
  props.pad_bottom = i64(p->pad_bottom);
  props.pad_left = i64(p->pad_left);
  props.pad_right = i64(p->pad_right);
  props.has_bias = b.getBoolAttr(p->has_bias);
  props.int8_scale_term = i64(p->int8_scale_term);
  llvm::SmallVector<mlir::Value> operands{*input, *weight};
  operands.append(tail);
  auto type =
    importer.infer_single_tensor_result<mlir::ncnn::ConvolutionDepthWiseOp>(
      b.getUnknownLoc(), operands, props);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = b.create<mlir::ncnn::ConvolutionDepthWiseOp>(b.getUnknownLoc(),
                                                         *type,
                                                         *input,
                                                         *weight,
                                                         tail,
                                                         props.kernel_h,
                                                         props.kernel_w,
                                                         props.stride_h,
                                                         props.stride_w,
                                                         props.dilation_h,
                                                         props.dilation_w,
                                                         props.pad_top,
                                                         props.pad_bottom,
                                                         props.pad_left,
                                                         props.pad_right,
                                                         props.has_bias,
                                                         props.int8_scale_term);
  importer.tag_source(op.getOperation(), context);
  mlir::Value output = op.getResult();
  if (*activation == 1) {
    auto relu = b.create<mlir::ncnn::ReluOp>(
      b.getUnknownLoc(), *type, output, b.getF32FloatAttr(0.0F));
    importer.tag_source(relu.getOperation(), context);
    output = relu.getResult();
  }
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), output);
}
}  // namespace ncnn_importer::detail
