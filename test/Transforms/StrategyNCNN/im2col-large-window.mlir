// RUN: ncnn-mlir-opt --strategy-ncnn=strategy=gemm %s | FileCheck %s

func.func @large_but_bounded_window(%input: tensor<1x168x168x256xf32>,
                                   %weights: tensor<9x9x256x64xf32>,
                                   %init: tensor<1x160x160x64xf32>) -> tensor<1x160x160x64xf32> {
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %weights : tensor<1x168x168x256xf32>, tensor<9x9x256x64xf32>) outs(%init : tensor<1x160x160x64xf32>) -> tensor<1x160x160x64xf32>
  return %conv : tensor<1x160x160x64xf32>
}

// 160*160*9*9*256 = 530,841,600 elements: above the general strategy cap,
// but below the bounded im2col window cap (2^29).
// CHECK-LABEL: func.func @large_but_bounded_window
// CHECK: linalg.matmul
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

func.func @window_over_cap(%input: tensor<1x170x170x256xf32>,
                           %weights: tensor<9x9x256x64xf32>,
                           %init: tensor<1x162x162x64xf32>) -> tensor<1x162x162x64xf32> {
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %weights : tensor<1x170x170x256xf32>, tensor<9x9x256x64xf32>) outs(%init : tensor<1x162x162x64xf32>) -> tensor<1x162x162x64xf32>
  return %conv : tensor<1x162x162x64xf32>
}

// 162*162*9*9*256 = 544,195,584 elements: above 2^29, so forced GEMM still fails closed.
// CHECK-LABEL: func.func @window_over_cap
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.matmul
