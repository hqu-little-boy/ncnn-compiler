// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

module {
  func.func @dynamic_depthwise_lowering(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
    %weight = arith.constant dense<0.000000e+00> : tensor<4x1x3x3xf32>
    %0 = ncnn.convolution_depthwise %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x?x?xf32>, tensor<4x1x3x3xf32>) -> tensor<4x?x?xf32>
    return %0 : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_depthwise_lowering
// CHECK: tensor.dim
// CHECK: tosa.depthwise_conv2d
