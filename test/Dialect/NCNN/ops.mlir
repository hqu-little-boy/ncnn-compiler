// RUN: ncnn-mlir-opt %s | ncnn-mlir-opt | FileCheck %s

// 验证 ncnn 方言各算子可解析、通过校验并稳定 round-trip。

// CHECK-LABEL: func.func @all_ops
func.func @all_ops(%arg0: tensor<3x227x227xf32>) -> tensor<128xf32> {
  %w = arith.constant dense<0.000000e+00> : tensor<64x3x3x3xf32>
  %b = arith.constant dense<0.000000e+00> : tensor<64xf32>
  // CHECK: ncnn.convolution
  // CHECK-SAME: kernel_h = 3 : i64
  // CHECK-SAME: stride_h = 2 : i64
  %c = ncnn.convolution %arg0, %w, %b {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x227x227xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<64x113x113xf32>
  // CHECK: ncnn.relu
  %r = ncnn.relu %c : (tensor<64x113x113xf32>) -> tensor<64x113x113xf32>
  // CHECK: ncnn.pooling
  // CHECK-SAME: kind = 0 : i64
  %p = ncnn.pooling %r {include_pad = false, kernel_h = 3 : i64, kernel_w = 3 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<64x113x113xf32>) -> tensor<64x56x56xf32>
  // CHECK: ncnn.split
  %s0, %s1 = ncnn.split %p : (tensor<64x56x56xf32>) -> (tensor<64x56x56xf32>, tensor<64x56x56xf32>)
  // CHECK: ncnn.concat
  // CHECK-SAME: axis = 0 : i64
  %cat = ncnn.concat %s0, %s1 {axis = 0 : i64} : (tensor<64x56x56xf32>, tensor<64x56x56xf32>) -> tensor<128x56x56xf32>
  // CHECK: ncnn.dropout
  %d = ncnn.dropout %cat : (tensor<128x56x56xf32>) -> tensor<128x56x56xf32>
  // CHECK: ncnn.pooling
  // CHECK-SAME: mode = 1 : i64
  %g = ncnn.pooling %d {include_pad = false, kernel_h = 0 : i64, kernel_w = 0 : i64, kind = 1 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<128x56x56xf32>) -> tensor<128xf32>
  // CHECK: ncnn.softmax
  %sm = ncnn.softmax %g {axis = 0 : i64} : (tensor<128xf32>) -> tensor<128xf32>
  return %sm : tensor<128xf32>
}

// CHECK-LABEL: func.func @anglenet_ops
func.func @anglenet_ops(%arg0: tensor<4x3x4xf32>) -> tensor<4xf32> {
  // CHECK: ncnn.shuffle_channel
  %shuffle = ncnn.shuffle_channel %arg0 {group = 2 : i64, reverse = false} : (tensor<4x3x4xf32>) -> tensor<4x3x4xf32>
  // CHECK: ncnn.slice
  %left, %right = ncnn.slice %shuffle {axis = 0 : i64, slices = array<i64: 2, 2>} : (tensor<4x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>)
  %joined = ncnn.concat %left, %right {axis = 0 : i64} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<4x3x4xf32>
  // CHECK: ncnn.reduction
  // CHECK-SAME: axes = array<i64: 1, 2>
  %mean = ncnn.reduction %joined {axes = array<i64: 1, 2>, coeff = 1.000000e+00 : f32, keepdims = false, kind = 3 : i64, reduce_all = false} : (tensor<4x3x4xf32>) -> tensor<4xf32>
  return %mean : tensor<4xf32>
}

// CHECK-LABEL: func.func @ppocr_rec_ops
func.func @ppocr_rec_ops(%arg0: tensor<2x1x3xf32>) -> tensor<3x4xf32> {
  %slope = arith.constant dense<1.000000e+00> : tensor<2xf32>
  %mean = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %variance = arith.constant dense<1.000000e+00> : tensor<2xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %weight = arith.constant dense<0.000000e+00> : tensor<4x2xf32>
  %gemm_bias = arith.constant dense<0.000000e+00> : tensor<4xf32>
  // CHECK: ncnn.gelu
  %gelu = ncnn.gelu %arg0 {fast = false} : (tensor<2x1x3xf32>) -> tensor<2x1x3xf32>
  // CHECK: ncnn.squeeze
  %squeezed = ncnn.squeeze %gelu {axes = array<i64: 1>} : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  // CHECK: ncnn.batch_norm
  %normalized = ncnn.batch_norm %squeezed, %slope, %mean, %variance, %bias {epsilon = 1.000000e-05 : f32} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  // CHECK: ncnn.expand_dims
  %expanded = ncnn.expand_dims %normalized {axes = array<i64: 1>} : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %again = ncnn.squeeze %expanded {axes = array<i64: 1>} : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  // CHECK: ncnn.permute
  %transposed = ncnn.permute %again {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: ncnn.gemm
  %output = ncnn.gemm %transposed, %weight, %gemm_bias {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32} : (tensor<3x2xf32>, tensor<4x2xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %output : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @static_inner_product_flattens
func.func @static_inner_product_flattens(%arg0: tensor<2x4xf32>) -> tensor<3xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<3x8xf32>
  // CHECK: ncnn.inner_product
  // CHECK-SAME: (tensor<2x4xf32>, tensor<3x8xf32>) -> tensor<3xf32>
  %output = ncnn.inner_product %arg0, %weight {has_bias = false} : (tensor<2x4xf32>, tensor<3x8xf32>) -> tensor<3xf32>
  return %output : tensor<3xf32>
}
