//===- NCNNOps.td - ncnn dialect ops ----------------------*- tablegen -*-===//
//
// ncnn 方言算子定义。算子集与原自定义前端 IR 的 8 个 kind 对应：
//   const   -> 用 arith.constant（不在此方言内）
//   conv2d / relu / pool2d / split / concat / dropout / softmax -> 本文件
// Input 层表示为 func.func 的 block argument，也不是算子。
//
//===----------------------------------------------------------------------===//

#ifndef NCNN_OPS
#define NCNN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.td"

//===----------------------------------------------------------------------===//
// 基类
//===----------------------------------------------------------------------===//

class NCNN_Op<string mnemonic, list<Trait> traits = []>
    : Op<NCNN_Dialect, mnemonic, traits> {
  // 统一汇编格式：打印全部操作数 + 属性字典 + 函数式类型签名。
  // 对「定长 + 可变长」操作数混排的算子（conv2d）也能避免悬空逗号。
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// conv2d
//===----------------------------------------------------------------------===//

def NCNN_Conv2DOp : NCNN_Op<"conv2d", [InferShapedTypeOpAdaptor]> {
  let summary = "2D cross-correlation convolution (ncnn Convolution).";
  let description = [{
    Operands are `input` `[C_in, H, W]`, `weight` `[O, I, H, W]`, then optional
    `bias` `[O]` and quantization scales (per ncnn `int8_scale_term`). The
    result is `[O, H_out, W_out]`. SAME padding is encoded with the ncnn
    sentinels `-233` (SAME_UPPER) / `-234` (SAME_LOWER) in the pad attributes.
  }];
  let arguments = (ins
    NCNN_RankedTensor:$input,
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
  let hasVerifier = 1;
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
    NCNN_RankedTensor:$input,
    DefaultValuedOptionalAttr<F32Attr, "0.0">:$negative_slope);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// pool2d
//===----------------------------------------------------------------------===//

def NCNN_Pool2DOp : NCNN_Op<"pool2d", [InferShapedTypeOpAdaptor]> {
  let summary = "2D max/average pooling (ncnn Pooling).";
  let description = [{
    `kind` selects max/average; `mode` selects regular sliding-window, global
    (reduce all spatial), or adaptive (kernel attrs carry the target output
    size, `-233` = keep input dim). `pad_mode`: 0=full(+tail), 1=valid,
    2=SAME_UPPER, 3=SAME_LOWER.
  }];
  let arguments = (ins
    NCNN_RankedTensor:$input,
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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// split
//===----------------------------------------------------------------------===//

def NCNN_SplitOp : NCNN_Op<"split", [Pure]> {
  let summary = "Fan one value out to N identical copies (ncnn Split).";
  let description = [{
    ncnn `Split` duplicates a blob into N consumers; it is pure routing. In SSA
    the fan-out is implicit, but this op preserves the ncnn layer structure.
    All results must have the same type as the input.
  }];
  let arguments = (ins NCNN_RankedTensor:$input);
  let results = (outs Variadic<NCNN_RankedTensor>:$results);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// concat
//===----------------------------------------------------------------------===//

def NCNN_ConcatOp : NCNN_Op<"concat", [InferShapedTypeOpAdaptor]> {
  let summary = "Concatenate along an axis (ncnn Concat).";
  let arguments = (ins
    Variadic<NCNN_RankedTensor>:$inputs,
    I64Attr:$axis);
  let results = (outs NCNN_RankedTensor:$output);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// dropout
//===----------------------------------------------------------------------===//

def NCNN_DropoutOp : NCNN_Op<"dropout", [Pure, SameOperandsAndResultType]> {
  let summary = "Inference-time dropout: identity or uniform scale.";
  let arguments = (ins
    NCNN_RankedTensor:$input,
    DefaultValuedOptionalAttr<F32Attr, "1.0">:$scale);
  let results = (outs NCNN_RankedTensor:$output);
}

//===----------------------------------------------------------------------===//
// softmax
//===----------------------------------------------------------------------===//

def NCNN_SoftmaxOp : NCNN_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Numerically-stable softmax along an axis (ncnn Softmax).";
  let arguments = (ins
    NCNN_RankedTensor:$input,
    I64Attr:$axis);
  let results = (outs NCNN_RankedTensor:$output);
  let hasVerifier = 1;
}

#endif  // NCNN_OPS
