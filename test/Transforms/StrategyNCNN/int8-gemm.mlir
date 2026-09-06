// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize %s | FileCheck %s

// int8 量化卷积（P4）：无直接卷积内核可用，任何 k×k 静态形状一律走
// im2col+GEMM；权重常量直接重排为 [N,K]（B 面转置物化，k 序 kh,kw,ic
// 与 gather 折叠序一致），收缩形态为 matmul_transpose_b——A1b 的
// i8 row-dot 内核依赖 B 沿 k 连续。

// 小通道 3×3（浮点侧 prefer_gemm 会拒绝的形态）也走 GEMM。
func.func @int8_small_conv(%arg0: tensor<1x10x10x8xi8>) -> tensor<1x8x8x8xi32> {
  %cst = arith.constant dense<1> : tensor<3x3x8x8xi8>
  %init = tensor.empty() : tensor<1x8x8x8xi32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x10x10x8xi8>, tensor<3x3x8x8xi8>) outs(%init : tensor<1x8x8x8xi32>) -> tensor<1x8x8x8xi32>
  return %conv : tensor<1x8x8x8xi32>
}

// CHECK-LABEL: func.func @int8_small_conv
// 权重 [3,3,8,8] → [8, 72]（转置 + 折叠一次到位，无中间 transpose op）
// CHECK: arith.constant dense{{.*}} : tensor<8x72xi8>
// CHECK: linalg.generic {{.*}}ins({{.*}} : tensor<1x10x10x8xi8>) outs({{.*}} : tensor<8x8x3x3x8xi8>)
// CHECK: tensor.collapse_shape {{.*}} tensor<64x72xi8>
// CHECK: linalg.matmul_transpose_b ins({{.*}} : tensor<64x72xi8>, tensor<8x72xi8>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 8]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.matmul ins

// -----

// 1×1 s1：视图折叠（无 gather），同样走 matmul_transpose_b。
func.func @int8_one_by_one(%arg0: tensor<1x4x4x8xi8>) -> tensor<1x4x4x4xi32> {
  %cst = arith.constant dense<2> : tensor<1x1x8x4xi8>
  %init = tensor.empty() : tensor<1x4x4x4xi32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x4x4x8xi8>, tensor<1x1x8x4xi8>) outs(%init : tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi32>
  return %conv : tensor<1x4x4x4xi32>
}

// CHECK-LABEL: func.func @int8_one_by_one
// CHECK: arith.constant dense{{.*}} : tensor<4x8xi8>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul_transpose_b ins({{.*}} : tensor<16x8xi8>, tensor<4x8xi8>)
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// 动态空间维：OH·OW 不可知，im2col 不可用，保持直接卷积路径
// （与浮点侧同款门控；batch 维在本管线中恒为 1）。
func.func @int8_dynamic_spatial(%arg0: tensor<1x?x?x8xi8>) -> tensor<1x?x?x8xi32> {
  %cst = arith.constant dense<1> : tensor<3x3x8x8xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h = tensor.dim %arg0, %c1 : tensor<1x?x?x8xi8>
  %w = tensor.dim %arg0, %c2 : tensor<1x?x?x8xi8>
  %oh = arith.subi %h, %c2 : index
  %ow = arith.subi %w, %c2 : index
  %init = tensor.empty(%oh, %ow) : tensor<1x?x?x8xi32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x?x?x8xi8>, tensor<3x3x8x8xi8>) outs(%init : tensor<1x?x?x8xi32>) -> tensor<1x?x?x8xi32>
  return %conv : tensor<1x?x?x8xi32>
}

// CHECK-LABEL: func.func @int8_dynamic_spatial
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.matmul_transpose_b
