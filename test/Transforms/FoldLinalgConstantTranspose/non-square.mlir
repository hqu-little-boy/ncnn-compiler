// RUN: ncnn-mlir-opt --fold-linalg-constant-transpose %s | FileCheck %s

// 非方形张量：折叠按 permutation 逐元素重排，不要求各维等长。
// 源 [OC=3, KH=2, KW=2, IC=2] --permutation [1,2,3,0]--> [KH=2, KW=2, IC=2, OC=3]，
// 即 conv_2d_nhwc_hwcf 权重布局 [KH, KW, IC, OC]。

func.func @folds_non_square(%arg0 : tensor<1x2x2x2xf32>) -> tensor<1x1x1x3xf32> {
  %cst = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], [[[17.0, 18.0], [19.0, 20.0]], [[21.0, 22.0], [23.0, 24.0]]]]> : tensor<3x2x2x2xf32>
  %0 = tensor.empty() : tensor<2x2x2x3xf32>
  %t = linalg.transpose ins(%cst : tensor<3x2x2x2xf32>) outs(%0 : tensor<2x2x2x3xf32>) permutation = [1, 2, 3, 0]
  %1 = tensor.empty() : tensor<1x1x1x3xf32>
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %t : tensor<1x2x2x2xf32>, tensor<2x2x2x3xf32>) outs(%1 : tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
  return %r : tensor<1x1x1x3xf32>
}

// result[i][j][k][l] = source[l][i][j][k]，因此 result[0][0][0][*] =
// source[*][0][0][0] = 1, 9, 17。
// CHECK-LABEL: func.func @folds_non_square
// CHECK: %[[FOLDED:.*]] = arith.constant dense<{{.*}}1.000000e+00, 9.000000e+00, 1.700000e+01{{.*}}> : tensor<2x2x2x3xf32>
// CHECK-NOT: linalg.transpose
// CHECK: linalg.conv_2d_nhwc_hwcf {{.*}} ins(%arg0, %[[FOLDED]]
