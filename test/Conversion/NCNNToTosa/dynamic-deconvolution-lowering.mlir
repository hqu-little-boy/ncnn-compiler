// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

module {
  func.func @dynamic_deconvolution_lowering(%arg0: tensor<4x?x?xf32>) -> tensor<2x?x?xf32> {
    %weight = arith.constant dense<0.000000e+00> : tensor<2x4x2x2xf32>
    %0 = ncnn.deconvolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 2 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>, tensor<2x4x2x2xf32>) -> tensor<2x?x?xf32>
    return %0 : tensor<2x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_deconvolution_lowering
// CHECK: tensor.dim
// CHECK: linalg.generic
