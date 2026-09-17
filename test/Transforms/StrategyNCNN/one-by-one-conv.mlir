// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize %s | FileCheck %s
// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize %s | FileCheck --check-prefix=DYN %s

// 1×1 s1 无条件改写为折叠视图 matmul（对齐 ncnn 的恒 GEMM 分支）；
// 权重 collapse 是常量重排，由 canonicalizer 折叠为二维 .rodata 常量。

func.func @one_by_one(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  return %conv : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @one_by_one
// CHECK: arith.constant dense{{.*}} : tensor<3x16xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<1x8x8x3xf32> into tensor<64x3xf32>
// CHECK: linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// 1×1 s2 不满足路径一；通道数未超阈值时保留直接卷积。
func.func @one_by_one_strided(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x4x4x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x4x4x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  return %conv : tensor<1x4x4x16xf32>
}

// CHECK-LABEL: func.func @one_by_one_strided
// CHECK: linalg.conv_2d_nhwc_hwcf

// -----

// 动态空间维的 1×1 s1：行折叠改走 tensor.reshape（重联组含多个动态维会
// 触发下游 ExpandStridedMetadata 的上游崩溃），matmul 行维取 ?，还原以
// 运行时 shape 张量完成——LiteOCR 类动态产物无需特殊处理。

func.func @dynamic_one_by_one(%arg0: tensor<1x?x?x3xf32>) -> tensor<1x?x?x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h = tensor.dim %arg0, %c1 : tensor<1x?x?x3xf32>
  %w = tensor.dim %arg0, %c2 : tensor<1x?x?x3xf32>
  %init = tensor.empty(%h, %w) : tensor<1x?x?x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x?x?x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32>
  return %conv : tensor<1x?x?x16xf32>
}

// DYN-LABEL: func.func @dynamic_one_by_one
// DYN: arith.muli {{.*}} : index
// DYN: tensor.from_elements {{.*}} : tensor<2xindex>
// DYN: tensor.reshape {{.*}}) : (tensor<1x?x?x3xf32>, tensor<2xindex>) -> tensor<?x3xf32>
// DYN: linalg.matmul {{.*}} ins({{.*}} : tensor<?x3xf32>, tensor<3x16xf32>)
// DYN: tensor.reshape {{.*}} : (tensor<?x16xf32>, tensor<4xindex>) -> tensor<1x?x?x16xf32>
