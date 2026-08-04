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

//===----------------------------------------------------------------------===//
// 模型边界
//===----------------------------------------------------------------------===//

def NCNN_ModelOp : NCNN_Op<"model", [Symbol, IsolatedFromAbove, SingleBlock,
                                      NoTerminator, NoRegionArguments]> {
  let summary = "An imported ncnn network before function ABI construction.";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasRegionVerifier = 1;
}

def NCNN_InputOp : NCNN_Op<"input", [HasParent<"ModelOp">]> {
  let summary = "A source ncnn Input layer.";
  let arguments = (ins StrAttr:$layer_name, StrAttr:$blob_name);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "attr-dict `:` type($output)";
}

def NCNN_ConstOp : NCNN_Op<"const", [Pure, HasParent<"ModelOp">]> {
  let summary = "An embedded weight tensor from the ncnn .bin file.";
  let arguments = (ins StrAttr:$name, ElementsAttr:$value);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "attr-dict `:` type($output)";
  let hasVerifier = 1;
}

def NCNN_OutputOp : NCNN_Op<"output", [HasParent<"ModelOp">]> {
  let summary = "A blob selected as an exported model output.";
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$blob_name);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// convolution
//===----------------------------------------------------------------------===//

def NCNN_ConvolutionOp : NCNN_Op<"convolution", [InferTensorTypeAdaptor]> {
  let summary = "Cross-correlation convolution (1:1 mirrors ncnn `Convolution`).";
  let description = [{
    Faithfully mirrors the ncnn `Convolution` layer; this dialect does NOT judge
    dimensionality/layout — realizing it as `tosa.conv2d` (vs 1d/3d), the
    CHW->NHWC layout conversion, and resolving SAME padding to explicit pads are
    all deferred to the ncnn->tosa lowering.

    Operands are `input` `[C_in, H, W]`, `weight` `[O, I, H, W]`, then optional
    `bias` `[O]` and quantization scales (per ncnn `int8_scale_term`). The
    inferred result is `[O, H_out, W_out]`. SAME padding is carried with the
    ncnn sentinels `-233` (SAME_UPPER) / `-234` (SAME_LOWER) in the pad attrs.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Variadic<AnyRankedTensor>:$bias_and_scales,
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
  let results = (outs AnyRankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// relu
//===----------------------------------------------------------------------===//

def NCNN_ReluOp : NCNN_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU / LeakyReLU activation.";
  let description = [{
    `y = x` for `x >= 0`, else `y = negative_slope * x`. `negative_slope = 0`
    is the standard ReLU.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedOptionalAttr<F32Attr, "0.0">:$negative_slope);
  let results = (outs AnyRankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// pooling
//===----------------------------------------------------------------------===//

def NCNN_PoolingOp : NCNN_Op<"pooling", [InferTensorTypeAdaptor]> {
  let summary = "Max/average pooling (1:1 mirrors ncnn `Pooling`).";
  let description = [{
    Faithfully mirrors the ncnn `Pooling` layer; dimensionality/layout and the
    choice of target op (`tosa.max_pool2d`/`avg_pool2d`, vs 1d/3d) are deferred
    to the ncnn->tosa lowering.

    `kind` selects max/average; `mode` selects regular sliding-window, global
    (reduce all spatial), or adaptive (kernel attrs carry the target output
    size, `-233` = keep input dim). `pad_mode`: 0=full(+tail), 1=valid,
    2=SAME_UPPER, 3=SAME_LOWER.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
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
  let results = (outs AnyRankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// split
//===----------------------------------------------------------------------===//

def NCNN_SplitOp : NCNN_Op<"split", [Pure, SameOperandsAndResultType]> {
  let summary = "Fan one value out to N identical copies (ncnn Split).";
  let description = [{
    ncnn `Split` duplicates a blob into N consumers; it is pure routing. In SSA
    the fan-out is implicit, but this op preserves the ncnn layer structure.
    All results must have the same type as the input.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// concat
//===----------------------------------------------------------------------===//

def NCNN_ConcatOp : NCNN_Op<"concat", [InferTensorTypeAdaptor]> {
  let summary = "Concatenate along an axis (ncnn Concat).";
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$axis);
  let results = (outs AnyRankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// dropout
//===----------------------------------------------------------------------===//

def NCNN_DropoutOp : NCNN_Op<"dropout", [Pure, SameOperandsAndResultType]> {
  let summary = "Inference-time dropout: identity or uniform scale.";
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedOptionalAttr<F32Attr, "1.0">:$scale);
  let results = (outs AnyRankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// softmax
//===----------------------------------------------------------------------===//

def NCNN_SoftmaxOp : NCNN_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Numerically-stable softmax along an axis (ncnn Softmax).";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
}

#endif  // NCNN_OPS
