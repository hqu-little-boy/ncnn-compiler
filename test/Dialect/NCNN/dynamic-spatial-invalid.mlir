// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @dynamic_inner_product_weight(%arg0: tensor<?x4xf32>, %weight: tensor<3x?xf32>) {
  // expected-error@+2 {{InnerProduct input elements must match weight [O,I]}}
  // expected-error@+1 {{'ncnn.inner_product' op failed to infer returned types}}
  %product = ncnn.inner_product %arg0, %weight {has_bias = false} : (tensor<?x4xf32>, tensor<3x?xf32>) -> tensor<?x3xf32>
  return
}

// -----

func.func @dynamic_static_binary_broadcast(%left: tensor<?x4xf32>, %right: tensor<2x4xf32>) {
  // expected-error@+2 {{BinaryOp dynamic extent cannot be broadcast to a static extent}}
  // expected-error@+1 {{'ncnn.binary' op failed to infer returned types}}
  %max = ncnn.binary %left, %right {op_type = 4 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<?x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return
}

// -----

func.func @inexact_dynamic_reshape(%input: tensor<?x6xf32>) {
  // expected-error@+2 {{Reshape cannot prove exact inferred dimension}}
  // expected-error@+1 {{'ncnn.reshape' op failed to infer returned types}}
  %result = "ncnn.reshape"(%input) {shape = array<i64: -1, 4>, shape_spec = array<i64: -1, 4>, shape_zero_sources = array<i64: -1, -1>} : (tensor<?x6xf32>) -> tensor<?x4xf32>
  return
}

// -----

func.func @inexact_copied_dynamic_reshape(%input: tensor<?x6xf32>) {
  // expected-error@+2 {{Reshape cannot prove exact element count}}
  // expected-error@+1 {{'ncnn.reshape' op failed to infer returned types}}
  %result = "ncnn.reshape"(%input) {shape = array<i64: -9223372036854775808, 5>, shape_spec = array<i64: 0, 5>, shape_zero_sources = array<i64: 0, -1>} : (tensor<?x6xf32>) -> tensor<?x5xf32>
  return
}

// -----

func.func @fixed_slices_on_dynamic_axis(%input: tensor<?x4xf32>) {
  // expected-error@+2 {{Slice dynamic axis requires a trailing -233 remainder slice}}
  // expected-error@+1 {{'ncnn.slice' op failed to infer returned types}}
  %first, %second = ncnn.slice %input {axis = 0 : i64, slices = array<i64: 2, 2>} : (tensor<?x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
  return
}

// -----

func.func @nontrailing_dynamic_remainder(%input: tensor<?x4xf32>) {
  // expected-error@+2 {{Slice dynamic axis requires a trailing -233 remainder slice}}
  // expected-error@+1 {{'ncnn.slice' op failed to infer returned types}}
  %first, %second = ncnn.slice %input {axis = 0 : i64, slices = array<i64: -233, 2>} : (tensor<?x4xf32>) -> (tensor<?x4xf32>, tensor<2x4xf32>)
  return
}

// -----

func.func @unconstrained_dynamic_slice(%input: tensor<?x4xf32>) {
  // expected-error@+1 {{dynamic axis requires an input minimum extent of at least 3}}
  %first, %second = ncnn.slice %input {axis = 0 : i64, slices = array<i64: 2, -233>} : (tensor<?x4xf32>) -> (tensor<2x4xf32>, tensor<?x4xf32>)
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
  // expected-error@+2 {{ConvolutionDepthWise requires pure depthwise weights [C,1,H,W]}}
  // expected-error@+1 {{'ncnn.convolution_depthwise' op failed to infer returned types}}
  %result = ncnn.convolution_depthwise %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<?x?x?xf32>, tensor<32x1x3x3xf32>) -> tensor<32x?x?xf32>
  return
}
