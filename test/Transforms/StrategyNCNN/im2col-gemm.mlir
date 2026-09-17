// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize %s | FileCheck %s

// k×k 卷积按 ncnn prefer_gemm 启发式分侧：通道数超限或权重工作集超过
// L2 预算的实例落 im2col+matmul，小实例保留直接卷积。

// IC=64 > 16 → GEMM：gather 构造 [OH,OW,KH,KW,IC]，折叠为 im2col 行，
// 权重常量折叠为 [KH·KW·IC, OC]。
func.func @above_threshold(%arg0: tensor<1x10x10x64xf32>) -> tensor<1x8x8x128xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x64x128xf32>
  %init = tensor.empty() : tensor<1x8x8x128xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x10x10x64xf32>, tensor<3x3x64x128xf32>) outs(%init : tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32>
  return %conv : tensor<1x8x8x128xf32>
}

// CHECK-LABEL: func.func @above_threshold
// CHECK: arith.constant dense{{.*}} : tensor<576x128xf32>
// CHECK: linalg.generic {{.*}}ins({{.*}} : tensor<1x10x10x64xf32>) outs({{.*}} : tensor<8x8x3x3x64xf32>)
// CHECK: tensor.collapse_shape {{.*}} tensor<8x8x3x3x64xf32> into tensor<64x576xf32>
// CHECK: linalg.matmul {{.*ncnn.implementation = "gemm".*ncnn.operation_family = "conv".*}} ins({{.*}} : tensor<64x576xf32>, tensor<576x128xf32>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 128]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// IC=OC=8、K=3：工作集 8·8·9·4·2=4608 字节 ≤ L2 且无通道超限 → 保留 conv。
func.func @below_threshold(%arg0: tensor<1x10x10x8xf32>) -> tensor<1x8x8x8xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x8x8xf32>
  %init = tensor.empty() : tensor<1x8x8x8xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x10x10x8xf32>, tensor<3x3x8x8xf32>) outs(%init : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %conv : tensor<1x8x8x8xf32>
}

// CHECK-LABEL: func.func @below_threshold
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.matmul

// -----

// 显式 --strategy=gemm 强制低于阈值的静态实例走 im2col+matmul。
// RUN: ncnn-mlir-opt --strategy-ncnn=strategy=gemm --canonicalize %s | FileCheck --check-prefix=GEMM %s

// GEMM-LABEL: func.func @below_threshold
// GEMM: linalg.matmul {{.*}} ins({{.*}} : tensor<64x72xf32>, tensor<72x8xf32>)
// GEMM-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// 显式 --strategy=conv 关闭全部改写。
// RUN: ncnn-mlir-opt --strategy-ncnn=strategy=conv %s | FileCheck --check-prefix=CONV %s

// CONV-LABEL: func.func @above_threshold
// CONV: linalg.conv_2d_nhwc_hwcf
// CONV-NOT: linalg.matmul

// -----

// 非法策略值在 pass 入口即失败。
// RUN: not ncnn-mlir-opt --strategy-ncnn=strategy=bogus %s 2>&1 | FileCheck --check-prefix=INVALID %s

// INVALID: invalid strategy-ncnn strategy 'bogus'
