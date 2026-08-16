// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

func.func @grouped_masked_cache(
    %query: tensor<4x1x8xf32>, %key: tensor<2x1x8xf32>,
    %value: tensor<2x1x6xf32>, %mask: tensor<1x?xf32>,
    %past_key: tensor<2x?x8xf32>, %past_value: tensor<2x?x6xf32>)
    -> (tensor<4x1x6xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>) {
  %context, %updated_key, %updated_value = ncnn.sdpa %query, %key, %value, %mask, %past_key, %past_value {has_mask = true, kv_cache = true, scale = 0.35355338 : f32} : (tensor<4x1x8xf32>, tensor<2x1x8xf32>, tensor<2x1x6xf32>, tensor<1x?xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>) -> (tensor<4x1x6xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>)
  return %context, %updated_key, %updated_value : tensor<4x1x6xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>
}

// CHECK-DAG: affine_map<(d0, d1, d2, d3) -> (d0 floordiv 2, d2, d3)>
// CHECK-DAG: affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: affine_map<(d0, d1, d2, d3) -> (d0 floordiv 2, d3, d2)>
// CHECK-LABEL: func.func @grouped_masked_cache
// CHECK-COUNT-2: tensor.concat
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: arith.divf
// CHECK-NOT: ncnn.sdpa

func.func @cross_attention(
    %query: tensor<4x?x8xf32>, %key: tensor<2x?x8xf32>,
    %value: tensor<2x?x6xf32>) -> tensor<4x?x6xf32> {
  %context = ncnn.sdpa %query, %key, %value {has_mask = false, kv_cache = false, scale = 0.5 : f32} : (tensor<4x?x8xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>) -> tensor<4x?x6xf32>
  return %context : tensor<4x?x6xf32>
}

// CHECK-LABEL: func.func @cross_attention
// CHECK: tensor.dim
// CHECK: linalg.generic
// CHECK-NOT: tensor.concat
// CHECK-NOT: ncnn.sdpa
