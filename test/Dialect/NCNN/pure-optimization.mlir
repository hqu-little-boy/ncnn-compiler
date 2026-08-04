// RUN: ncnn-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=CANONICALIZE
// RUN: ncnn-mlir-opt --cse %s | FileCheck %s --check-prefix=CSE

func.func @dead_ops(%input: tensor<2x4x4xf32>, %weight: tensor<3x2x1x1xf32>) {
  %conv = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>, tensor<3x2x1x1xf32>) -> tensor<3x4x4xf32>
  %pool = ncnn.pooling %conv {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x4x4xf32>) -> tensor<3x2x2xf32>
  %concat = ncnn.concat %pool, %pool {axis = 0 : i64} : (tensor<3x2x2xf32>, tensor<3x2x2xf32>) -> tensor<6x2x2xf32>
  return
}

// CANONICALIZE-LABEL: func.func @dead_ops
// CANONICALIZE-NOT: ncnn.convolution
// CANONICALIZE-NOT: ncnn.pooling
// CANONICALIZE-NOT: ncnn.concat
// CANONICALIZE: return

func.func @common_ops(%input: tensor<2x4x4xf32>, %weight: tensor<3x2x1x1xf32>) -> (tensor<3x4x4xf32>, tensor<3x4x4xf32>, tensor<3x2x2xf32>, tensor<3x2x2xf32>, tensor<6x4x4xf32>, tensor<6x4x4xf32>) {
  %conv0 = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>, tensor<3x2x1x1xf32>) -> tensor<3x4x4xf32>
  %conv1 = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>, tensor<3x2x1x1xf32>) -> tensor<3x4x4xf32>
  %pool0 = ncnn.pooling %conv0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x4x4xf32>) -> tensor<3x2x2xf32>
  %pool1 = ncnn.pooling %conv0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x4x4xf32>) -> tensor<3x2x2xf32>
  %concat0 = ncnn.concat %conv0, %conv0 {axis = 0 : i64} : (tensor<3x4x4xf32>, tensor<3x4x4xf32>) -> tensor<6x4x4xf32>
  %concat1 = ncnn.concat %conv0, %conv0 {axis = 0 : i64} : (tensor<3x4x4xf32>, tensor<3x4x4xf32>) -> tensor<6x4x4xf32>
  return %conv0, %conv1, %pool0, %pool1, %concat0, %concat1 : tensor<3x4x4xf32>, tensor<3x4x4xf32>, tensor<3x2x2xf32>, tensor<3x2x2xf32>, tensor<6x4x4xf32>, tensor<6x4x4xf32>
}

// CSE-LABEL: func.func @common_ops
// CSE-COUNT-1: ncnn.convolution
// CSE-COUNT-1: ncnn.pooling
// CSE-COUNT-1: ncnn.concat
