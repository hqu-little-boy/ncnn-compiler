// RUN: ncnn-mlir-opt --split-input-file --lower-ncnn-vulkan-graph %s | FileCheck %s --check-prefix=IR

func.func @max_pool_tail(%input: tensor<1x56x56x2xf32>) -> tensor<1x28x28x2xf32> {
  %pooled = "tosa.max_pool2d"(%input) <{kernel = array<i64: 3, 3>, nan_mode = "PROPAGATE", pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>}> : (tensor<1x56x56x2xf32>) -> tensor<1x28x28x2xf32> loc("pool_tail"("ncnn-layer":10:0))
  return %pooled : tensor<1x28x28x2xf32>
}
// IR: gpu.module @max_pool_tail_vulkan_k0
// IR: arith.constant 0xFF800000 : f32
// IR: arith.maximumf
// IR: gpu.module @max_pool_tail_vulkan_k1
// IR-NOT: tosa.

// -----

func.func @global_average(%input: tensor<1x14x14x2xf32>) -> tensor<1x1x1x2xf32> {
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %pooled = "tosa.avg_pool2d"(%input, %zero, %zero) <{acc_type = f32, kernel = array<i64: 14, 14>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x14x14x2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1x2xf32> loc("global_avg"("ncnn-layer":11:0))
  return %pooled : tensor<1x1x1x2xf32>
}
// IR: gpu.module @global_average_vulkan_k0
// IR: arith.addf
// IR: arith.divf
// IR: gpu.module @global_average_vulkan_k1
// IR-NOT: tosa.

// -----

func.func @stable_softmax(%input: tensor<8xf32>) -> tensor<8xf32> {
  %integer_zero = arith.constant dense<0> : tensor<1xi8>
  %maximum = "tosa.reduce_max"(%input) <{axis = 0 : i32}> : (tensor<8xf32>) -> tensor<1xf32>
  %shifted = "tosa.sub"(%input, %maximum) : (tensor<8xf32>, tensor<1xf32>) -> tensor<8xf32>
  %exponent = "tosa.exp"(%shifted) : (tensor<8xf32>) -> tensor<8xf32>
  %sum = "tosa.reduce_sum"(%exponent) <{axis = 0 : i32}> : (tensor<8xf32>) -> tensor<1xf32>
  %reciprocal = "tosa.reciprocal"(%sum) : (tensor<1xf32>) -> tensor<1xf32>
  %probability = "tosa.mul"(%exponent, %reciprocal, %integer_zero) : (tensor<8xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<8xf32>
  return %probability : tensor<8xf32>
}
// IR: gpu.module @stable_softmax_vulkan_k0
// IR: arith.maximumf
// IR: gpu.module @stable_softmax_vulkan_k1
// IR: arith.subf
// IR: gpu.module @stable_softmax_vulkan_k2
// IR: math.exp
// IR: gpu.module @stable_softmax_vulkan_k3
// IR: arith.addf
// IR: gpu.module @stable_softmax_vulkan_k4
// IR: arith.divf
// IR: gpu.module @stable_softmax_vulkan_k5
// IR: arith.mulf
// IR: gpu.module @stable_softmax_vulkan_k6
// IR-NOT: tosa.

// -----

func.func @concat_spatial(%top: tensor<1x2x2x3xf32>, %bottom: tensor<1x3x2x3xf32>) -> tensor<1x5x2x3xf32> {
  %joined = "tosa.concat"(%top, %bottom) <{axis = 1 : i32}> : (tensor<1x2x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x5x2x3xf32>
  return %joined : tensor<1x5x2x3xf32>
}
// IR: gpu.module @concat_spatial_vulkan_k0
// IR: scf.if
// IR: gpu.module @concat_spatial_vulkan_k1
// IR-NOT: tosa.

// -----

func.func @conv1x1_pad1(%input: tensor<1x3x3x2xf32>) -> tensor<1x5x5x1xf32> {
  %weight = arith.constant dense<0.5> : tensor<1x1x1x2xf32>
  %bias = arith.constant dense<0.125> : tensor<1xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %result = "tosa.conv2d"(%input, %weight, %bias, %zero, %zero) <{acc_type = f32, dilation = array<i64: 1, 1>, local_bound = false, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x3x3x2xf32>, tensor<1x1x1x2xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32> loc("conv1x1_pad1"("ncnn-layer":12:0))
  return %result : tensor<1x5x5x1xf32>
}
// IR: gpu.module @conv1x1_pad1_vulkan_k0
// IR: scf.if
// IR: gpu.module @conv1x1_pad1_vulkan_k1
// IR-NOT: tosa.

// -----

func.func @broadcast_matrix(%lhs: tensor<2x3xf32>, %rhs: tensor<2x3xf32>, %row: tensor<1x3xf32>, %column: tensor<2x1xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %zero = arith.constant dense<0> : tensor<1xi8>
  %sub_equal = "tosa.sub"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %sub_rhs_row = "tosa.sub"(%lhs, %row) : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  %sub_lhs_row = "tosa.sub"(%row, %lhs) : (tensor<1x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %sub_rhs_column = "tosa.sub"(%lhs, %column) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  %sub_lhs_column = "tosa.sub"(%column, %lhs) : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %mul_equal = "tosa.mul"(%lhs, %rhs, %zero) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  %mul_rhs_row = "tosa.mul"(%lhs, %row, %zero) : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  %mul_lhs_row = "tosa.mul"(%row, %lhs, %zero) : (tensor<1x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  %mul_rhs_column = "tosa.mul"(%lhs, %column, %zero) : (tensor<2x3xf32>, tensor<2x1xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  %mul_lhs_column = "tosa.mul"(%column, %lhs, %zero) : (tensor<2x1xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %sub_equal, %sub_rhs_row, %sub_lhs_row, %sub_rhs_column, %sub_lhs_column, %mul_equal, %mul_rhs_row, %mul_lhs_row, %mul_rhs_column, %mul_lhs_column : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// IR: gpu.module @broadcast_matrix_vulkan_k0
// IR: arith.subf
// IR: gpu.module @broadcast_matrix_vulkan_k1
// IR: arith.subf
// IR: gpu.module @broadcast_matrix_vulkan_k2
// IR: arith.subf
// IR: gpu.module @broadcast_matrix_vulkan_k3
// IR: arith.subf
// IR: gpu.module @broadcast_matrix_vulkan_k4
// IR: arith.subf
// IR: gpu.module @broadcast_matrix_vulkan_k5
// IR: arith.mulf
// IR: gpu.module @broadcast_matrix_vulkan_k6
// IR: arith.mulf
// IR: gpu.module @broadcast_matrix_vulkan_k7
// IR: arith.mulf
// IR: gpu.module @broadcast_matrix_vulkan_k8
// IR: arith.mulf
// IR: gpu.module @broadcast_matrix_vulkan_k9
// IR: arith.mulf
// IR: gpu.module @broadcast_matrix_vulkan_k19
// IR-NOT: tosa.
