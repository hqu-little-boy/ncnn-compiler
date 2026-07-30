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
