// RUN: ncnn-mlir-opt %s | FileCheck %s

func.func @dynamic_spatial(%arg0: tensor<32x?x?xf32>) -> tensor<8x?x?xf32> {
  %depthwise_weight = arith.constant dense<0.000000e+00> : tensor<32x1x3x3xf32>
  %depthwise = ncnn.convolution_depthwise %arg0, %depthwise_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<32x?x?xf32>, tensor<32x1x3x3xf32>) -> tensor<32x?x?xf32>
  %pool = ncnn.pooling %depthwise {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<32x?x?xf32>) -> tensor<32x?x?xf32>
  %deconv_weight = arith.constant dense<0.000000e+00> : tensor<8x32x2x2xf32>
  %deconv = ncnn.deconvolution %pool, %deconv_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 2 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<32x?x?xf32>, tensor<8x32x2x2xf32>) -> tensor<8x?x?xf32>
  %sigmoid = ncnn.sigmoid %deconv : (tensor<8x?x?xf32>) -> tensor<8x?x?xf32>
  return %sigmoid : tensor<8x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_spatial
// CHECK: ncnn.convolution_depthwise
// CHECK: ncnn.pooling
// CHECK: ncnn.deconvolution
// CHECK: ncnn.sigmoid

func.func @static_pooling(%arg0: tensor<32x33x65xf32>) -> tensor<32x32x64xf32> {
  %pool = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<32x33x65xf32>) -> tensor<32x32x64xf32>
  return %pool : tensor<32x32x64xf32>
}

// CHECK-LABEL: func.func @static_pooling
// CHECK: ncnn.pooling
