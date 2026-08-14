//===- NCNNOps.td - ncnn dialect ops ----------------------*- tablegen -*-===//
//
// ncnn 方言算子定义。解析期由 model/input/const/output 表达完整网络，计算层
// 保持 ncnn 原生语义；后续显式函数化 pass 才建立 func.func 与 ABI。
//
//===----------------------------------------------------------------------===//

#ifndef NCNN_OPS
#define NCNN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.td"

//===----------------------------------------------------------------------===//
// 基类
//===----------------------------------------------------------------------===//

class NCNN_Op<string mnemonic, list<Trait> traits = []>
  : Op<NCNN_Dialect, mnemonic, traits> {
  // 统一汇编格式：打印全部操作数 + 属性字典 + 函数式类型签名。
  // 对「定长 + 可变长」操作数混排的算子（convolution）也能避免悬空逗号。
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

// 无副作用的 ncnn 计算层；模型容器和输入/输出边界不属于此类。
class NCNN_LayerOp<string mnemonic, list<Trait> traits = []>
  : NCNN_Op<mnemonic, !listconcat([Pure], traits)>;

def NCNN_RankedTensor
  : RankedTensorOf<
      [AnyType],
      [CPred<"!::llvm::cast<::mlir::RankedTensorType>($_self).getEncoding()">],
      "ranked tensor without encoding">;

class NCNN_AtLeastNOperands<int n>
  : ParamNativeOpTrait<"AtLeastNOperands", !cast<string>(n)>;
class NCNN_AtLeastNResults<int n>
  : ParamNativeOpTrait<"AtLeastNResults", !cast<string>(n)>;

//===----------------------------------------------------------------------===//
// 模型边界
//===----------------------------------------------------------------------===//

def NCNN_ModelOp : NCNN_Op<"model",
                           [Symbol,
                            IsolatedFromAbove,
                            SingleBlock,
                            NoTerminator,
                            NoRegionArguments]> {
  let summary = "An imported ncnn network before function ABI construction.";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasRegionVerifier = 1;
}

def NCNN_InputOp : NCNN_Op<"input", [HasParent<"ModelOp">]> {
  let summary = "A source ncnn Input layer.";
  let arguments = (ins StrAttr:$layer_name, StrAttr:$blob_name);
  let results = (outs NCNN_RankedTensor:$output);
  let assemblyFormat = "attr-dict `:` type($output)";
}

def NCNN_ConstOp : NCNN_Op<"const",
                           [ConstantLike,
                            Pure,
                            AllTypesMatch<["value", "output"]>,
                            HasParent<"ModelOp">]> {
  let summary = "An embedded weight tensor from the ncnn .bin file.";
  let arguments = (ins StrAttr:$name, ElementsAttr:$value);
  let results = (outs NCNN_RankedTensor:$output);
  let assemblyFormat = "attr-dict `:` type($output)";
  let hasFolder = 1;
}

def NCNN_OutputOp : NCNN_Op<"output", [HasParent<"ModelOp">]> {
  let summary = "A blob selected as an exported model output.";
  let arguments = (ins NCNN_RankedTensor:$input, StrAttr:$blob_name);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// convolution
//===----------------------------------------------------------------------===//

def NCNN_ConvolutionOp : NCNN_LayerOp<"convolution", [InferTensorTypeAdaptor]> {
  let summary =
    "Cross-correlation convolution (1:1 mirrors ncnn `Convolution`).";
  let description =
    [{
    Faithfully mirrors the ncnn `Convolution` layer; this dialect does NOT judge
    dimensionality/layout — realizing it as `tosa.conv2d` (vs 1d/3d), the
    CHW->NHWC layout conversion, and resolving SAME padding to explicit pads are
    all deferred to the ncnn->tosa lowering.

    Operands are `input` `[C_in, H, W]`, `weight` `[O, I, H, W]`, then optional
    `bias` `[O]` and quantization scales (per ncnn `int8_scale_term`). The
    inferred result is `[O, H_out, W_out]`. SAME padding is carried with the
    ncnn sentinels `-233` (SAME_UPPER) / `-234` (SAME_LOWER) in the pad attrs.
  }];
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$weight,
    Variadic<NCNN_RankedTensor>:$bias_and_scales,
    I64Attr:$kernel_h,
    I64Attr:$kernel_w,
    I64Attr:$stride_h,
    I64Attr:$stride_w,
    I64Attr:$dilation_h,
    I64Attr:$dilation_w,
    I64Attr:$pad_top,
    I64Attr:$pad_bottom,
    I64Attr:$pad_left,
    I64Attr:$pad_right,
    BoolAttr:$has_bias,
    DefaultValuedOptionalAttr<I64Attr, "0">:$int8_scale_term);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_ConvolutionDepthWiseOp
  : NCNN_LayerOp<"convolution_depthwise", [InferTensorTypeAdaptor]> {
  let summary = "Pure depthwise convolution with explicit quantization data.";
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$weight,
    Variadic<NCNN_RankedTensor>:$bias_and_scales,
    I64Attr:$kernel_h,
    I64Attr:$kernel_w,
    I64Attr:$stride_h,
    I64Attr:$stride_w,
    I64Attr:$dilation_h,
    I64Attr:$dilation_w,
    I64Attr:$pad_top,
    I64Attr:$pad_bottom,
    I64Attr:$pad_left,
    I64Attr:$pad_right,
    BoolAttr:$has_bias,
    DefaultValuedOptionalAttr<I64Attr, "0">:$int8_scale_term);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_DeconvolutionOp
  : NCNN_LayerOp<"deconvolution", [InferTensorTypeAdaptor]> {
  let summary = "FP32 ncnn transposed convolution with static weights.";
  let description =
    [{
    Operands use ncnn's model layout: input `[I,H,W]`, static weight
    `[O,I,K_h,K_w]`, and optional bias `[O]`. The result is `[O,H_out,W_out]`.
  }];
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$weight,
    Variadic<NCNN_RankedTensor>:$bias,
    I64Attr:$kernel_h,
    I64Attr:$kernel_w,
    I64Attr:$stride_h,
    I64Attr:$stride_w,
    I64Attr:$dilation_h,
    I64Attr:$dilation_w,
    I64Attr:$pad_top,
    I64Attr:$pad_bottom,
    I64Attr:$pad_left,
    I64Attr:$pad_right,
    I64Attr:$output_pad_bottom,
    I64Attr:$output_pad_right,
    BoolAttr:$has_bias);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_PaddingOp : NCNN_LayerOp<"padding", [InferTensorTypeAdaptor]> {
  let summary = "Constant spatial padding of a CHW tensor.";
  let arguments = (ins NCNN_RankedTensor:$input,
    I64Attr:$top,
    I64Attr:$bottom,
    I64Attr:$left,
    I64Attr:$right,
    F32Attr:$value);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_InterpOp : NCNN_LayerOp<"interp", [InferTensorTypeAdaptor]> {
  let summary = "Nearest-neighbor resize of a CHW tensor.";
  let arguments = (ins NCNN_RankedTensor:$input,
    I64Attr:$height_scale,
    I64Attr:$width_scale,
    DefaultValuedAttr<I64Attr, "0">:$output_h,
    DefaultValuedAttr<I64Attr, "0">:$output_w);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_SigmoidOp : NCNN_LayerOp<"sigmoid", [SameOperandsAndResultType]> {
  let arguments = (ins NCNN_RankedTensor:$input);
  let results = (outs NCNN_RankedTensor:$output);
  let hasVerifier = 1;
}

def NCNN_SwishOp : NCNN_LayerOp<"swish", [SameOperandsAndResultType]> {
  let summary = "Elementwise x * sigmoid(x).";
  let arguments = (ins NCNN_RankedTensor:$input);
  let results = (outs NCNN_RankedTensor:$output);
  let hasVerifier = 1;
}

def NCNN_LayerNormOp
  : NCNN_LayerOp<"layer_norm", [InferTensorTypeAdaptor]> {
  let summary = "Layer normalization over the last source dimension.";
  let arguments = (ins NCNN_RankedTensor:$input,
    Variadic<NCNN_RankedTensor>:$affine_parameters,
    I64Attr:$affine_size,
    F32Attr:$epsilon,
    BoolAttr:$affine);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_MultiHeadAttentionOp
  : NCNN_LayerOp<"multi_head_attention", [InferTensorTypeAdaptor]> {
  let summary = "Static-sequence FP32 self attention.";
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$q_weight,
    NCNN_RankedTensor:$q_bias,
    NCNN_RankedTensor:$k_weight,
    NCNN_RankedTensor:$k_bias,
    NCNN_RankedTensor:$v_weight,
    NCNN_RankedTensor:$v_bias,
    NCNN_RankedTensor:$out_weight,
    NCNN_RankedTensor:$out_bias,
    I64Attr:$embed_dim,
    I64Attr:$num_heads,
    I64Attr:$weight_data_size,
    I64Attr:$qdim,
    I64Attr:$kdim,
    I64Attr:$vdim,
    F32Attr:$scale);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_DetectionOutputOp
  : NCNN_LayerOp<"detection_output", [InferTensorTypeAdaptor]> {
  let summary = "Decode SSD boxes, apply per-class NMS, and emit detections.";
  let description =
    [{
    Implements the ncnn DetectionOutput contract for Caffe-style SSD inputs.
    The result storage is `[maximum_detections, 6]`; only a data-dependent
    prefix is logically present. Rows contain label, confidence, xmin, ymin,
    xmax, and ymax. The exact logical extent must be returned by lowering.
  }];
  let arguments = (ins NCNN_RankedTensor:$location,
    NCNN_RankedTensor:$confidence,
    NCNN_RankedTensor:$priorbox,
    I64Attr:$num_class,
    F32Attr:$nms_threshold,
    I64Attr:$nms_top_k,
    I64Attr:$keep_top_k,
    F32Attr:$confidence_threshold,
    F32Attr:$variance_x,
    F32Attr:$variance_y,
    F32Attr:$variance_w,
    F32Attr:$variance_h);
  let results = (outs NCNN_RankedTensor:$output,
    NCNN_RankedTensor:$actual_shape);
}

def NCNN_HardSigmoidOp
  : NCNN_LayerOp<"hard_sigmoid", [SameOperandsAndResultType]> {
  let arguments = (ins NCNN_RankedTensor:$input, F32Attr:$alpha, F32Attr:$beta);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_HardSwishOp : NCNN_LayerOp<"hard_swish", [SameOperandsAndResultType]> {
  let arguments = (ins NCNN_RankedTensor:$input, F32Attr:$alpha, F32Attr:$beta);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_GELUOp : NCNN_LayerOp<"gelu", [SameOperandsAndResultType]> {
  let summary = "Gaussian error linear unit.";
  let arguments = (ins NCNN_RankedTensor:$input, BoolAttr:$fast);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_BatchNormOp : NCNN_LayerOp<"batch_norm", [InferTensorTypeAdaptor]> {
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$slope,
    NCNN_RankedTensor:$mean,
    NCNN_RankedTensor:$variance,
    NCNN_RankedTensor:$bias,
    F32Attr:$epsilon);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_ReshapeOp : NCNN_LayerOp<"reshape", [InferTensorTypeAdaptor]> {
  let arguments = (ins NCNN_RankedTensor:$input,
    Variadic<NCNN_RankedTensor>:$shape_references,
    DenseI64ArrayAttr:$shape,
    OptionalAttr<DenseI64ArrayAttr>:$shape_spec,
    OptionalAttr<DenseI64ArrayAttr>:$shape_zero_sources,
    OptionalAttr<DenseI64ArrayAttr>:$shape_sources,
    OptionalAttr<StrAttr>:$shape_expression);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_SqueezeOp : NCNN_LayerOp<"squeeze", [InferTensorTypeAdaptor]> {
  let arguments = (ins NCNN_RankedTensor:$input, DenseI64ArrayAttr:$axes);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_ExpandDimsOp : NCNN_LayerOp<"expand_dims", [InferTensorTypeAdaptor]> {
  let arguments = (ins NCNN_RankedTensor:$input, DenseI64ArrayAttr:$axes);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_PermuteOp : NCNN_LayerOp<"permute", [InferTensorTypeAdaptor]> {
  let arguments = (ins NCNN_RankedTensor:$input,
    DenseI64ArrayAttr:$permutation);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_BinaryOp : NCNN_LayerOp<"binary", [InferTensorTypeAdaptor]> {
  let arguments = (ins Variadic<NCNN_RankedTensor>:$inputs,
    F32Attr:$scalar,
    BoolAttr:$with_scalar,
    I64Attr:$op_type);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_InnerProductOp
  : NCNN_LayerOp<"inner_product", [InferTensorTypeAdaptor]> {
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$weight,
    Variadic<NCNN_RankedTensor>:$bias_and_scales,
    BoolAttr:$has_bias,
    DefaultValuedOptionalAttr<I64Attr, "0">:$int8_scale_term);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_GemmOp : NCNN_LayerOp<"gemm", [InferTensorTypeAdaptor]> {
  let summary = "Dynamic-A Gemm with transposed constant B and row bias.";
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$weight,
    NCNN_RankedTensor:$bias,
    Variadic<NCNN_RankedTensor>:$scales,
    F32Attr:$alpha,
    F32Attr:$beta,
    DefaultValuedOptionalAttr<I64Attr, "0">:$int8_scale_term);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_QuantizeOp : NCNN_LayerOp<"quantize", [InferTensorTypeAdaptor]> {
  let summary = "ncnn FP32 to signed INT8 quantization.";
  let arguments = (ins NCNN_RankedTensor:$input, NCNN_RankedTensor:$scale);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_DequantizeOp
  : NCNN_LayerOp<"dequantize", [InferTensorTypeAdaptor]> {
  let summary = "ncnn INT32 accumulator dequantization with optional bias.";
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$scale,
    Variadic<NCNN_RankedTensor>:$bias);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_RequantizeOp
  : NCNN_LayerOp<"requantize", [InferTensorTypeAdaptor]> {
  let summary = "ncnn INT32 accumulator requantization to signed INT8.";
  let arguments = (ins NCNN_RankedTensor:$input,
    NCNN_RankedTensor:$input_scale,
    NCNN_RankedTensor:$output_scale,
    Variadic<NCNN_RankedTensor>:$bias,
    I64Attr:$activation_type,
    DefaultValuedOptionalAttr<DenseF32ArrayAttr, "{}">:$activation_params);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_CastOp : NCNN_LayerOp<"cast", [InferTensorTypeAdaptor]> {
  let summary = "Explicit ncnn storage cast.";
  let arguments = (ins NCNN_RankedTensor:$input,
    I64Attr:$type_from,
    I64Attr:$type_to);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_ZeroPointCastOp
  : NCNN_LayerOp<"zero_point_cast", [InferTensorTypeAdaptor]> {
  let summary = "Explicit saturating rebias between unsigned and signed INT8.";
  let arguments = (ins NCNN_RankedTensor:$input,
    I64Attr:$zero_point,
    BoolAttr:$to_unsigned);
  let results = (outs NCNN_RankedTensor:$output);
}

def NCNN_ShuffleChannelOp
  : NCNN_LayerOp<"shuffle_channel", [SameOperandsAndResultType]> {
  let summary = "Shuffle channels between groups.";
  let arguments =
    (ins NCNN_RankedTensor:$input, I64Attr:$group, BoolAttr:$reverse);
  let results = (outs NCNN_RankedTensor:$output);
  let hasVerifier = 1;
}

def NCNN_SliceOp
  : NCNN_LayerOp<"slice", [InferTensorTypeAdaptor, NCNN_AtLeastNResults<2>]> {
  let summary = "Split a tensor into slices along one axis.";
  let arguments =
    (ins NCNN_RankedTensor:$input, DenseI64ArrayAttr:$slices, I64Attr:$axis);
  let results = (outs Variadic<NCNN_RankedTensor>:$results);
  let hasVerifier = 1;
}

def NCNN_ReductionOp : NCNN_LayerOp<"reduction", [InferTensorTypeAdaptor]> {
  let summary = "Reduce a tensor along explicit axes.";
  let arguments = (ins NCNN_RankedTensor:$input,
    I64Attr:$kind,
    BoolAttr:$reduce_all,
    F32Attr:$coeff,
    DenseI64ArrayAttr:$axes,
    BoolAttr:$keepdims);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// relu
//===----------------------------------------------------------------------===//

def NCNN_ReluOp : NCNN_LayerOp<"relu", [SameOperandsAndResultType]> {
  let summary = "ReLU / LeakyReLU activation.";
  let description =
    [{
    `y = x` for `x >= 0`, else `y = negative_slope * x`. `negative_slope = 0`
    is the standard ReLU.
  }];
  let arguments = (ins NCNN_RankedTensor:$input,
    DefaultValuedOptionalAttr<F32Attr, "0.0">:$negative_slope);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// pooling
//===----------------------------------------------------------------------===//

def NCNN_PoolingOp : NCNN_LayerOp<"pooling", [InferTensorTypeAdaptor]> {
  let summary = "Max/average pooling (1:1 mirrors ncnn `Pooling`).";
  let description =
    [{
    Faithfully mirrors the ncnn `Pooling` layer; dimensionality/layout and the
    choice of target op (`tosa.max_pool2d`/`avg_pool2d`, vs 1d/3d) are deferred
    to the ncnn->tosa lowering.

    `kind` selects max/average; `mode` selects regular sliding-window, global
    (reduce all spatial), or adaptive (kernel attrs carry the target output
    size, `-233` = keep input dim). `pad_mode`: 0=full(+tail), 1=valid,
    2=SAME_UPPER, 3=SAME_LOWER.
  }];
  let arguments = (ins NCNN_RankedTensor:$input,
    I64Attr:$kind,
    I64Attr:$mode,
    I64Attr:$kernel_h,
    I64Attr:$kernel_w,
    I64Attr:$stride_h,
    I64Attr:$stride_w,
    I64Attr:$pad_top,
    I64Attr:$pad_bottom,
    I64Attr:$pad_left,
    I64Attr:$pad_right,
    I64Attr:$pad_mode,
    BoolAttr:$include_pad);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// split
//===----------------------------------------------------------------------===//

def NCNN_SplitOp
  : NCNN_LayerOp<"split",
                 [SameOperandsAndResultType, NCNN_AtLeastNResults<2>]> {
  let summary = "Fan one value out to N identical copies (ncnn Split).";
  let description =
    [{
    ncnn `Split` duplicates a blob into N consumers; it is pure routing. In SSA
    the fan-out is implicit, but this op preserves the ncnn layer structure.
    All results must have the same type as the input.
  }];
  let arguments = (ins NCNN_RankedTensor:$input);
  let results = (outs Variadic<NCNN_RankedTensor>:$results);
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// concat
//===----------------------------------------------------------------------===//

def NCNN_ConcatOp
  : NCNN_LayerOp<"concat", [InferTensorTypeAdaptor, NCNN_AtLeastNOperands<2>]> {
  let summary = "Concatenate along an axis (ncnn Concat).";
  let arguments = (ins Variadic<NCNN_RankedTensor>:$inputs, I64Attr:$axis);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// dropout
//===----------------------------------------------------------------------===//

def NCNN_DropoutOp : NCNN_LayerOp<"dropout", [SameOperandsAndResultType]> {
  let summary = "Inference-time dropout: identity or uniform scale.";
  let arguments = (ins NCNN_RankedTensor:$input,
    DefaultValuedOptionalAttr<F32Attr, "1.0">:$scale);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// softmax
//===----------------------------------------------------------------------===//

def NCNN_SoftmaxOp : NCNN_LayerOp<"softmax", [SameOperandsAndResultType]> {
  let summary = "Numerically-stable softmax along an axis (ncnn Softmax).";
  let arguments = (ins NCNN_RankedTensor:$input, I64Attr:$axis);
  let results = (outs NCNN_RankedTensor:$output);
  let hasVerifier = 1;
}

#endif  // NCNN_OPS
