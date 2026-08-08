// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @dynamic_channel(%arg0: tensor<?x32x32xf32>) {
  // expected-error@+2 {{pooling input channels must be static and positive}}
  // expected-error@+1 {{'ncnn.pooling' op failed to infer returned types}}
  %pool = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<?x32x32xf32>) -> tensor<?x31x31xf32>
  return
}

// -----

func.func @invalid_pooling(%arg0: tensor<32x?x?xf32>) {
  // expected-error@+2 {{pooling height kernel must be positive}}
  // expected-error@+1 {{'ncnn.pooling' op failed to infer returned types}}
  %pool = ncnn.pooling %arg0 {include_pad = false, kernel_h = 0 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<32x?x?xf32>) -> tensor<32x?x?xf32>
  return
}

// -----

func.func @dynamic_depthwise_channel(%arg0: tensor<?x?x?xf32>, %weight: tensor<32x1x3x3xf32>) {
  // expected-error@+2 {{ConvolutionDepthWise requires FP32 pure depthwise weights}}
  // expected-error@+1 {{'ncnn.convolution_depthwise' op failed to infer returned types}}
  %result = ncnn.convolution_depthwise %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<?x?x?xf32>, tensor<32x1x3x3xf32>) -> tensor<32x?x?xf32>
  return
}
