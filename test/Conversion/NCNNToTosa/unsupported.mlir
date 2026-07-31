// RUN: not ncnn-mlir-opt --split-input-file --convert-ncnn-to-tosa %s 2>&1 | FileCheck %s

func.func @same_without_normalize(%arg0: tensor<2x4x4xf32>) -> tensor<3x4x4xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<3x2x3x3xf32>
  %conv = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>, tensor<3x2x3x3xf32>) -> tensor<3x4x4xf32>
  return %conv : tensor<3x4x4xf32>
}

// CHECK: error: 'ncnn.convolution' op requires explicit non-negative padding; run normalize-ncnn first
