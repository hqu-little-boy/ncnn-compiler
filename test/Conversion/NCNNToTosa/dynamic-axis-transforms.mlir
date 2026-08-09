// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa --split-input-file %s | FileCheck %s

func.func @rank_changes(%arg0: tensor<1x?x?xf32>) -> tensor<1x?x?xf32> {
  %0 = ncnn.squeeze %arg0 {axes = array<i64: 0>} : (tensor<1x?x?xf32>) -> tensor<?x?xf32>
  %1 = ncnn.expand_dims %0 {axes = array<i64: 0>} : (tensor<?x?xf32>) -> tensor<1x?x?xf32>
  return %1 : tensor<1x?x?xf32>
}

// CHECK-LABEL: func.func @rank_changes
// CHECK: tensor.reshape
// CHECK: tensor.reshape
// CHECK-NOT: ncnn.squeeze
// CHECK-NOT: ncnn.expand_dims

// -----

func.func @permute(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = ncnn.permute %arg0 {permutation = array<i64: 1, 0>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @permute
// CHECK: tosa.transpose

// -----

func.func @gemm(%arg0: tensor<?x4xf32>, %weight: tensor<2x4xf32>, %bias: tensor<2xf32>) -> tensor<?x2xf32> {
  %0 = ncnn.gemm %arg0, %weight, %bias {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32} : (tensor<?x4xf32>, tensor<2x4xf32>, tensor<2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: func.func @gemm
// CHECK: tensor.expand_shape
// CHECK: tosa.matmul
// CHECK: tensor.collapse_shape

// -----

func.func @slice(%arg0: tensor<4x?x?xf32>) -> (tensor<2x?x?xf32>, tensor<2x?x?xf32>) {
  %0, %1 = ncnn.slice %arg0 {axis = 0 : i64, slices = array<i64: 2, 2>} : (tensor<4x?x?xf32>) -> (tensor<2x?x?xf32>, tensor<2x?x?xf32>)
  return %0, %1 : tensor<2x?x?xf32>, tensor<2x?x?xf32>
}

// CHECK-LABEL: func.func @slice
// CHECK-COUNT-2: tensor.extract_slice

// -----

func.func @reduction(%arg0: tensor<4x?x?xf32>) -> tensor<?x?xf32> {
  %0 = ncnn.reduction %arg0 {axes = array<i64: 0>, coeff = 1.000000e+00 : f32, keepdims = false, kind = 3 : i64, reduce_all = false} : (tensor<4x?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @reduction
// CHECK: tosa.reduce_sum
// CHECK: tensor.reshape
