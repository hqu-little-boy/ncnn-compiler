// RUN: not ncnn-mlir-opt %s 2>&1 | FileCheck %s

// 校验失败用例：声明的结果类型与形状推断不一致，verifier 应拒绝。

// CHECK: inferred type(s) {{.*}} are incompatible with return type(s)
func.func @bad_conv_result(%arg0: tensor<3x8x8xf32>) -> tensor<64x8x8xf32> {
  %w = arith.constant dense<0.000000e+00> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.000000e+00> : tensor<64xf32>
  // kernel 3 / stride 1 / pad 0 作用于 8x8 应得 6x6，这里故意声明 8x8。
  %0 = ncnn.convolution %arg0, %w, %b {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x8x8xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<64x8x8xf32>
  return %0 : tensor<64x8x8xf32>
}
