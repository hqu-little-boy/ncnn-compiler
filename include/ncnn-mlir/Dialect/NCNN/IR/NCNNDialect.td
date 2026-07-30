//===- NCNNDialect.td - ncnn dialect definition -----------*- tablegen -*-===//
//
// ncnn 高层方言：1:1 镜像 ncnn 层，作为 lowering 到 tosa 的输入。
// 对应原自定义前端 IR（include/ncnn_frontend/）的方言化重写。
//
//===----------------------------------------------------------------------===//

#ifndef NCNN_DIALECT
#define NCNN_DIALECT

include "mlir/IR/OpBase.td"

def NCNN_Dialect : Dialect {
  let name = "ncnn";
  let cppNamespace = "::mlir::ncnn";
  let summary = "High-level dialect mirroring ncnn layers.";
  let description = [{
    The `ncnn` dialect represents ncnn network layers as a typed SSA graph.
    Feature tensors use the ncnn-native `[C, H, W]` dimension order (the
    conversion to NHWC happens later, at the ncnn -> tosa boundary). Convolution
    weights are `[O, I, H, W]`; biases and other 1-D constants are `[N]`.
  }];
}

// 秩已知的张量约束。ncnn 方言只处理 ranked tensor；元素类型（f32/f16/i8）
// 由各算子的 verifier / 形状推断进一步约束。
def NCNN_RankedTensor
    : Type<CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">,
           "ranked tensor">;

#endif  // NCNN_DIALECT
