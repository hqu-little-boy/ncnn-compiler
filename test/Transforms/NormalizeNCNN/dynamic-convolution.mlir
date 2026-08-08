// RUN: ncnn-mlir-opt --normalize-ncnn %s | FileCheck %s

func.func @dynamic_same(%arg0: tensor<3x?x?xf32>) -> tensor<8x?x?xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<8x3x2x3xf32>
  %upper = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>, tensor<8x3x2x3xf32>) -> tensor<8x?x?xf32>
  %lower = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 3 : i64, pad_bottom = -234 : i64, pad_left = -234 : i64, pad_right = -234 : i64, pad_top = -234 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>, tensor<8x3x2x3xf32>) -> tensor<8x?x?xf32>
  return %lower : tensor<8x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_same
// CHECK: ncnn.convolution
// CHECK-SAME: pad_bottom = 1 : i64
// CHECK-SAME: pad_left = 1 : i64
// CHECK-SAME: pad_right = 1 : i64
// CHECK-SAME: pad_top = 0 : i64
// CHECK: ncnn.convolution
// CHECK-SAME: pad_bottom = 0 : i64
// CHECK-SAME: pad_left = 1 : i64
// CHECK-SAME: pad_right = 1 : i64
// CHECK-SAME: pad_top = 1 : i64
// CHECK-NOT: -233
// CHECK-NOT: -234
