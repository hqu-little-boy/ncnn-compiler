// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @sdpa_shape_provenance attributes {ncnn.input_dim_relations = [array<i64: 3, 1, 4, 1, 1>, array<i64: 4, 1, 5, 1, 0>]} {
    %query = ncnn.input {blob_name = "query", layer_name = "query"} : tensor<4x?x8xf32>
    %key = ncnn.input {blob_name = "key", layer_name = "key"} : tensor<2x1x8xf32>
    %value = ncnn.input {blob_name = "value", layer_name = "value"} : tensor<2x1x6xf32>
    %mask = ncnn.input {blob_name = "mask", layer_name = "mask"} : tensor<?x?xf32>
    %past_key = ncnn.input {blob_name = "past_key", layer_name = "past_key"} : tensor<2x?x8xf32>
    %past_value = ncnn.input {blob_name = "past_value", layer_name = "past_value"} : tensor<2x?x6xf32>
    %context, %updated_key, %updated_value = ncnn.sdpa %query, %key, %value, %mask, %past_key, %past_value {has_mask = true, kv_cache = true, scale = 0.35355338 : f32} : (tensor<4x?x8xf32>, tensor<2x1x8xf32>, tensor<2x1x6xf32>, tensor<?x?xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>) -> (tensor<4x?x6xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>)
    ncnn.output %context {blob_name = "context"} : tensor<4x?x6xf32>
    ncnn.output %updated_key {blob_name = "updated_key"} : tensor<2x?x8xf32>
    ncnn.output %updated_value {blob_name = "updated_value"} : tensor<2x?x6xf32>
  }
}

// CHECK-LABEL: func.func @sdpa_shape_provenance
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 1, 0, 1>, array<i64: 1, 2, 2>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
// CHECK-SAME: ncnn.shape_program = [array<i64: 0, 2>, array<i64: 2, 1, 4, 1, 0, 1>, array<i64: 0, 8>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
// CHECK-SAME: ncnn.shape_program = [array<i64: 0, 2>, array<i64: 2, 1, 5, 1, 0, 1>, array<i64: 0, 6>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
// CHECK-SAME: ncnn.input_dim_relations = [array<i64: 3, 1, 4, 1, 1>, array<i64: 4, 1, 5, 1, 0>]
