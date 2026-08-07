// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

func.func @activations_binary(%arg0: tensor<4x3x3xf32>, %scale: tensor<4x1x1xf32>) -> tensor<4x3x3xf32> {
  %sigmoid = "ncnn.hard_sigmoid"(%arg0) {alpha = 1.66666672E-1 : f32, beta = 5.000000E-1 : f32} : (tensor<4x3x3xf32>) -> tensor<4x3x3xf32>
  %swish = "ncnn.hard_swish"(%sigmoid) {alpha = 1.66666672E-1 : f32, beta = 5.000000E-1 : f32} : (tensor<4x3x3xf32>) -> tensor<4x3x3xf32>
  %scaled = "ncnn.binary"(%swish, %scale) {op_type = 2 : i64, scalar = 0.0 : f32, with_scalar = false} : (tensor<4x3x3xf32>, tensor<4x1x1xf32>) -> tensor<4x3x3xf32>
  return %scaled : tensor<4x3x3xf32>
}

// CHECK-LABEL: func.func @activations_binary
// CHECK: %[[SCALE_T:.*]] = tosa.transpose %arg1 {perms = array<i32: 1, 2, 0>}
// CHECK: %[[SCALE:.*]] = tosa.reshape %[[SCALE_T]]
// CHECK-SAME: -> tensor<1x1x1x4xf32>
// CHECK: tosa.mul
// CHECK: tosa.add
// CHECK: tosa.clamp
// CHECK: tosa.clamp
// CHECK: tosa.mul
// CHECK: tosa.mul {{.*}}, %[[SCALE]]

func.func @depthwise(%arg0: tensor<4x5x5xf32>) -> tensor<4x5x5xf32> {
  %weight = arith.constant dense<1.000000e+00> : tensor<4x1x3x3xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %result = "ncnn.convolution_depthwise"(%arg0, %weight, %bias) {dilation_h = 1 : i64, dilation_w = 1 : i64, kernel_h = 3 : i64, kernel_w = 3 : i64, has_bias = true, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x5x5xf32>, tensor<4x1x3x3xf32>, tensor<4xf32>) -> tensor<4x5x5xf32>
  return %result : tensor<4x5x5xf32>
}

// CHECK-LABEL: func.func @depthwise
// CHECK: %[[GROUPED:.*]] = tosa.reshape %cst, {{.*}} : (tensor<4x1x3x3xf32>, !tosa.shape<4>) -> tensor<4x1x3x3xf32>
// CHECK: %[[WEIGHT:.*]] = tosa.transpose %[[GROUPED]] {perms = array<i32: 2, 3, 0, 1>}
// CHECK-SAME: -> tensor<3x3x4x1xf32>
// CHECK: tosa.depthwise_conv2d
// CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
// CHECK-SAME: -> tensor<1x5x5x4xf32>

func.func @depthwise_asymmetric(%arg0: tensor<2x10x8xf32>) -> tensor<2x4x4xf32> {
  %weight = arith.constant dense<1.000000e+00> : tensor<2x1x2x3xf32>
  %result = "ncnn.convolution_depthwise"(%arg0, %weight) {dilation_h = 1 : i64, dilation_w = 2 : i64, kernel_h = 2 : i64, kernel_w = 3 : i64, has_bias = false, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 2 : i64, pad_top = 0 : i64, stride_h = 3 : i64, stride_w = 2 : i64} : (tensor<2x10x8xf32>, tensor<2x1x2x3xf32>) -> tensor<2x4x4xf32>
  return %result : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @depthwise_asymmetric
// CHECK: tosa.depthwise_conv2d
// CHECK-SAME: dilation = array<i64: 1, 2>
// CHECK-SAME: pad = array<i64: 0, 1, 1, 2>
// CHECK-SAME: stride = array<i64: 3, 2>
// CHECK-SAME: -> tensor<1x4x4x2xf32>

func.func @reshape_inner_product(%arg0: tensor<2x2x2xf32>) -> tensor<3xf32> {
  %flat = "ncnn.reshape"(%arg0) {shape = array<i64: 8>} : (tensor<2x2x2xf32>) -> tensor<8xf32>
  %weight = arith.constant dense<1.000000e+00> : tensor<3x8xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<3xf32>
  %result = "ncnn.inner_product"(%flat, %weight, %bias) {has_bias = true} : (tensor<8xf32>, tensor<3x8xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: func.func @reshape_inner_product
// CHECK: %[[FLAT:.*]] = tosa.reshape {{.*}} : (tensor<2x2x2xf32>, !tosa.shape<1>) -> tensor<8xf32>
// CHECK: %[[FC_INPUT:.*]] = tosa.reshape %[[FLAT]]
// CHECK-SAME: -> tensor<1x1x8xf32>
// CHECK: %[[FC_WEIGHT:.*]] = tosa.transpose {{.*}} {perms = array<i32: 1, 0>}
// CHECK: %[[MATMUL:.*]] = tosa.matmul %[[FC_INPUT]],
// CHECK-SAME: -> tensor<1x1x3xf32>
// CHECK: %[[BIASED:.*]] = tosa.add %[[MATMUL]],
// CHECK: tosa.reshape %[[BIASED]]
// CHECK-SAME: -> tensor<3xf32>

func.func @reshape_to_chw(%arg0: tensor<8xf32>) -> tensor<2x2x2xf32> {
  %result = "ncnn.reshape"(%arg0) {shape = array<i64: 2, 2, 2>} : (tensor<8xf32>) -> tensor<2x2x2xf32>
  return %result : tensor<2x2x2xf32>
}

// CHECK-LABEL: func.func @reshape_to_chw
// CHECK: %[[CHW:.*]] = tosa.reshape %arg0
// CHECK-SAME: -> tensor<2x2x2xf32>
// CHECK: %[[HWC:.*]] = tosa.transpose %[[CHW]] {perms = array<i32: 1, 2, 0>}
// CHECK: %[[NHWC:.*]] = tosa.reshape %[[HWC]]
// CHECK-SAME: -> tensor<1x2x2x2xf32>
// CHECK: %[[OUT_HWC:.*]] = tosa.reshape %[[NHWC]]
// CHECK: %[[OUT:.*]] = tosa.transpose %[[OUT_HWC]] {perms = array<i32: 2, 0, 1>}
// CHECK: return %[[OUT]]

// CHECK-NOT: ncnn.
