// RUN: not ncnn-mlir-opt --normalize-ncnn %s 2>&1 | FileCheck %s

func.func @dynamic_stride(%arg0: tensor<3x?x?xf32>) -> tensor<8x?x?xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<8x3x3x3xf32>
  %result = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 2 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>, tensor<8x3x3x3xf32>) -> tensor<8x?x?xf32>
  return %result : tensor<8x?x?xf32>
}

// CHECK: error: 'ncnn.convolution' op cannot resolve SAME padding; dynamic dimensions require stride 1
