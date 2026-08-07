#include "ImporterInternal.hpp"

#include <format>

namespace ncnn_importer::detail {
namespace {
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
  auto valid = arity_params(context, 1, 1, ids);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  const auto& p = context.layer.get_params();
  auto w = get_int(p, 0, -233, "w");
  auto h = get_int(p, 1, -233, "h");
  auto d = get_int(p, 11, -233, "d");
  auto c = get_int(p, 2, -233, "c");
  if (!w || !h || !d || !c) {
    return std::unexpected(
      make_error(context, "reshape parameter type is invalid"));
  }
  llvm::SmallVector<int64_t> shape;
  auto input = importer.find_blob(context, context.layer.get_inputs()[0]);
  if (!input) {
    return std::unexpected(input.error());
  }
  auto input_type = llvm::dyn_cast<mlir::RankedTensorType>(input->getType());
  if (!input_type) {
    return std::unexpected(make_error(context, "reshape input must be ranked"));
  }
  const auto input_shape = input_type.getShape();
  if (*w == 0) {
    *w = input_shape.back();
  }
  if (*h == 0) {
    if (input_type.getRank() < 2) {
      return std::unexpected(
        make_error(context, "reshape cannot copy missing height dimension"));
    }
    *h = input_shape[input_type.getRank() - 2];
  }
  if (*d == 0) {
    if (input_type.getRank() != 4) {
      return std::unexpected(
        make_error(context, "reshape cannot copy missing depth dimension"));
    }
    *d = input_shape[1];
  }
  if (*c == 0) {
    if (input_type.getRank() < 3) {
      return std::unexpected(
        make_error(context, "reshape cannot copy missing channel dimension"));
    }
    *c = input_shape.front();
  }
  for (int64_t dimension : {*c, *d, *h, *w}) {
    if (dimension != -233) {
      shape.push_back(dimension);
    }
  }
  auto& b = importer.builder();
  mlir::ncnn::ReshapeOp::Properties props;
  props.shape = b.getDenseI64ArrayAttr(shape);
  auto type = importer.infer_single_tensor_result<mlir::ncnn::ReshapeOp>(
    b.getUnknownLoc(), *input, props);
  if (mlir::failed(type)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = b.create<mlir::ncnn::ReshapeOp>(
    b.getUnknownLoc(), *type, *input, props.shape);
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
  if (!type || !scalar || !ws || (*type != 2) || (*ws != 0 && *ws != 1) ||
      ((*ws == 1) != (context.layer.get_inputs().size() == 1))) {
    return std::unexpected(make_error(
      context, "BinaryOp supports multiply with scalar or two inputs only"));
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

ImportResult import_inner_product(ImportContext& importer,
                                  const LayerContext& context) {
  constexpr int ids[] = {0, 1, 2, 8, 9, 10};
  auto valid = arity_params(context, 1, 1, ids);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  auto out = get_int(context.layer.get_params(), 0, 0, "num_output");
  auto bias = get_int(context.layer.get_params(), 1, 0, "bias_term");
  auto count = get_int(context.layer.get_params(), 2, 0, "weight_data_size");
  auto term = get_int(context.layer.get_params(), 8, 0, "int8_scale_term");
  auto act = get_int(context.layer.get_params(), 9, 0, "activation_type");
  if (!out || !bias || !count || !term || !act || *out <= 0 || *count <= 0 ||
      *bias < 0 || *bias > 1 || *term != 0 || *act != 0 ||
      context.layer.get_weights().size() != static_cast<size_t>(1 + *bias)) {
    return std::unexpected(
      make_error(context, "only static FP32 InnerProduct is supported"));
  }
  for (const auto& w : context.layer.get_weights()) {
    if (w.get_dtype() != ncnn_graph::DataType::Float32) {
      return std::unexpected(
        make_error(context, "InnerProduct weights must be FP32"));
    }
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
  llvm::SmallVector<mlir::Value> tail;
  if (*bias) {
    auto v = importer.make_constant(context, context.layer.get_weights()[1], 1);
    if (!v) {
      return std::unexpected(v.error());
    }
    tail.push_back(*v);
  }
  auto& b = importer.builder();
  mlir::ncnn::InnerProductOp::Properties props;
  props.has_bias = b.getBoolAttr(*bias);
  llvm::SmallVector<mlir::Value> operands{*input, *weight};
  operands.append(tail);
  auto result = importer.infer_single_tensor_result<mlir::ncnn::InnerProductOp>(
    b.getUnknownLoc(), operands, props);
  if (mlir::failed(result)) {
    return std::unexpected(make_error(context, importer.captured_diagnostic()));
  }
  auto op = b.create<mlir::ncnn::InnerProductOp>(
    b.getUnknownLoc(), *result, *input, *weight, tail, props.has_bias);
  importer.tag_source(op.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op.getResult());
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
  if (!p) {
    return std::unexpected(make_error(context, p.error()));
  }
  if (p->dynamic_weight || p->int8_scale_term != 0 ||
      context.layer.get_params().get_int(9, 0) != 0 ||
      context.layer.get_params().has(10) ||
      context.layer.get_params().get_float(18, 0.0F) != 0.0F) {
    return std::unexpected(make_error(
      context, "only static pure depthwise FP32 convolution is supported"));
  }
  if (context.layer.get_weights().size() != p->expected_weight_tensors() ||
      context.layer.get_weights().empty() ||
      context.layer.get_weights()[0].get_dtype() !=
        ncnn_graph::DataType::Float32) {
    return std::unexpected(make_error(
      context,
      "depthwise weights must be static FP32 kernel and optional bias"));
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
  llvm::SmallVector<mlir::Value> bias;
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
    bias.push_back(*value);
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
  llvm::SmallVector<mlir::Value> operands{*input, *weight};
  operands.append(bias);
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
                                                         bias,
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
                                                         props.has_bias);
  importer.tag_source(op.getOperation(), context);
  return importer.bind_blob(
    context, std::string(context.layer.get_outputs()[0]), op.getResult());
}
}  // namespace ncnn_importer::detail
