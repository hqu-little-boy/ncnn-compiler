// RUN: not ncnn-mlir-opt --split-input-file --convert-ncnn-to-tosa %s 2>&1 | FileCheck %s

func.func @same_without_normalize(%arg0: tensor<2x4x4xf32>) -> tensor<3x4x4xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<3x2x3x3xf32>
  %conv = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>, tensor<3x2x3x3xf32>) -> tensor<3x4x4xf32>
  return %conv : tensor<3x4x4xf32>
}

// CHECK: error: 'ncnn.convolution' op requires explicit non-negative padding; run normalize-ncnn first

// -----

func.func @deconvolution_crop_exceeds_tosa_limit(%arg0: tensor<1x3x3xf32>) -> tensor<1x4x7xf32> {
  %weight = arith.constant dense<1.000000e+00> : tensor<1x1x3x3xf32>
  %result = "ncnn.deconvolution"(%arg0, %weight) {kernel_h = 3 : i64, kernel_w = 3 : i64, stride_h = 2 : i64, stride_w = 2 : i64, dilation_h = 1 : i64, dilation_w = 1 : i64, pad_top = 3 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, has_bias = false} : (tensor<1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x4x7xf32>
  return %result : tensor<1x4x7xf32>
}

// CHECK: error: 'ncnn.deconvolution' op crop exceeds the TOSA transpose_conv2d out_pad range
