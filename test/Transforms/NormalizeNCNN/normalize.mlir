// RUN: ncnn-mlir-opt --normalize-ncnn %s | FileCheck %s

func.func @normalize(%arg0: tensor<3x4x5xf32>) -> tensor<16x2x3xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<8x3x3x3xf32>
  %upper = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x4x5xf32>, tensor<8x3x3x3xf32>) -> tensor<8x2x3xf32>
  %lower = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -234 : i64, pad_left = -234 : i64, pad_right = -234 : i64, pad_top = -234 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x4x5xf32>, tensor<8x3x3x3xf32>) -> tensor<8x2x3xf32>
  %pooled = ncnn.pooling %arg0 {include_pad = false, kernel_h = 3 : i64, kernel_w = 3 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 2 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x4x5xf32>) -> tensor<3x2x3xf32>
  %split0, %split1 = ncnn.split %upper : (tensor<8x2x3xf32>) -> (tensor<8x2x3xf32>, tensor<8x2x3xf32>)
  %concat = ncnn.concat %split0, %split1 {axis = -3 : i64} : (tensor<8x2x3xf32>, tensor<8x2x3xf32>) -> tensor<16x2x3xf32>
  %softmax = ncnn.softmax %pooled {axis = -1 : i64} : (tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
  return %concat : tensor<16x2x3xf32>
}

// CHECK-LABEL: func.func @normalize
// CHECK: %[[UPPER:.*]] = ncnn.convolution
// CHECK-SAME: pad_bottom = 1 : i64
// CHECK-SAME: pad_left = 1 : i64
// CHECK-SAME: pad_right = 1 : i64
// CHECK-SAME: pad_top = 0 : i64
// CHECK: ncnn.convolution
// CHECK-SAME: pad_bottom = 0 : i64
// CHECK-SAME: pad_left = 1 : i64
// CHECK-SAME: pad_right = 1 : i64
// CHECK-SAME: pad_top = 1 : i64
// CHECK: ncnn.pooling
// CHECK-SAME: pad_bottom = 1 : i64
// CHECK-SAME: pad_left = 1 : i64
// CHECK-SAME: pad_mode = 1 : i64
// CHECK-SAME: pad_right = 1 : i64
// CHECK-SAME: pad_top = 0 : i64
// CHECK-NOT: ncnn.split
// CHECK: ncnn.concat %[[UPPER]], %[[UPPER]] {axis = 0 : i64}
// CHECK: ncnn.softmax {{.*}} {axis = 2 : i64}

// CHECK-NOT: -233
// CHECK-NOT: -234
