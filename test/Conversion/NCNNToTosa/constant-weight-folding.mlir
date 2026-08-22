// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

// Constant weights must be folded at compile time: the OIHW->OHWI transpose,
// the depthwise HWCM rearrangement, and the [O,K]->[K,O] contraction weight
// permutation all become pre-materialized constants instead of per-inference
// tensor transposes.

func.func @conv_folds_weight_transpose(%arg0: tensor<2x5x5xf32>) -> tensor<2x3x3xf32> {
  %weight = arith.constant dense<[[
      [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
      [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]]],
      [[[2.1, 2.2, 2.3], [2.4, 2.5, 2.6], [2.7, 2.8, 2.9]],
      [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6], [3.7, 3.8, 3.9]]]]> : tensor<2x2x3x3xf32>
  %conv = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x5x5xf32>, tensor<2x2x3x3xf32>) -> tensor<2x3x3xf32>
  return %conv : tensor<2x3x3xf32>
}

// CHECK-LABEL: func.func @conv_folds_weight_transpose
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{.*}}> : tensor<2x3x3x2xf32>
// CHECK-NOT: tosa.transpose
// CHECK: tosa.conv2d {{.*}}, %[[WEIGHT]],

func.func @depthwise_folds_weight(%arg0: tensor<2x6x6xf32>) -> tensor<2x6x5xf32> {
  %weight = arith.constant dense<[[[[0.1, 0.2]]], [[[0.3, 0.4]]]]> : tensor<2x1x1x2xf32>
  %depthwise = "ncnn.convolution_depthwise"(%arg0, %weight) {dilation_h = 1 : i64, dilation_w = 1 : i64, kernel_h = 1 : i64, kernel_w = 2 : i64, has_bias = false, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x6x6xf32>, tensor<2x1x1x2xf32>) -> tensor<2x6x5xf32>
  return %depthwise : tensor<2x6x5xf32>
}

// CHECK-LABEL: func.func @depthwise_folds_weight
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{.*}}> : tensor<1x2x2x1xf32>
// CHECK-NOT: tosa.transpose
// CHECK: tosa.depthwise_conv2d {{.*}}, %[[WEIGHT]],

func.func @inner_product_folds_weight(%arg0: tensor<8xf32>) -> tensor<3xf32> {
  %weight = arith.constant dense<[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                 [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                                 [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]]> : tensor<3x8xf32>
  %bias = arith.constant dense<0.5> : tensor<3xf32>
  %ip = "ncnn.inner_product"(%arg0, %weight, %bias) {has_bias = true} : (tensor<8xf32>, tensor<3x8xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %ip : tensor<3xf32>
}

// CHECK-LABEL: func.func @inner_product_folds_weight
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{.*}}> : tensor<1x8x3xf32>
// CHECK-NOT: tosa.transpose
// CHECK: tosa.matmul {{.*}}, %[[WEIGHT]],
// CHECK: tosa.add {{.*}} -> tensor<1x1x3xf32>

func.func @gemm_folds_weight(%arg0: tensor<3x2xf32>) -> tensor<3x4xf32> {
  %weight = arith.constant dense<[[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]> : tensor<4x2xf32>
  %bias = arith.constant dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>
  %gemm = ncnn.gemm %arg0, %weight, %bias {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32} : (tensor<3x2xf32>, tensor<4x2xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %gemm : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @gemm_folds_weight
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{.*}}> : tensor<1x2x4xf32>
// CHECK-NOT: tosa.transpose
// CHECK: tosa.matmul {{.*}}, %[[WEIGHT]],

// Non-constant weights keep the runtime transpose.
func.func @dynamic_weight_keeps_transpose(%arg0: tensor<2x5x5xf32>, %weight: tensor<2x2x3x3xf32>) -> tensor<2x3x3xf32> {
  %conv = ncnn.convolution %arg0, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x5x5xf32>, tensor<2x2x3x3xf32>) -> tensor<2x3x3xf32>
  return %conv : tensor<2x3x3xf32>
}

// CHECK-LABEL: func.func @dynamic_weight_keeps_transpose
// CHECK: %[[WEIGHT:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 3, 1>}
// CHECK-SAME: -> tensor<2x3x3x2xf32>
// CHECK: tosa.conv2d {{.*}}, %[[WEIGHT]],
