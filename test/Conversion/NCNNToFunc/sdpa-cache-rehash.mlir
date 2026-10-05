// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

// LLVM 21 DenseMap grows from 64 buckets at the 48th entry. The 46 input
// transforms and context transform put the key-cache insertion at that
// boundary. Both cache transforms must be copied before either insertion;
// otherwise the value-cache expression reads an invalidated iterator.
module {
  ncnn.model @sdpa_cache_rehash {
    %query = ncnn.input {blob_name = "query", layer_name = "query"} : tensor<4x?x8xf32>
    %key = ncnn.input {blob_name = "key", layer_name = "key"} : tensor<2x1x8xf32>
    %value = ncnn.input {blob_name = "value", layer_name = "value"} : tensor<2x1x6xf32>
    %mask = ncnn.input {blob_name = "mask", layer_name = "mask"} : tensor<?x?xf32>
    %past_key = ncnn.input {blob_name = "past_key", layer_name = "past_key"} : tensor<2x?x8xf32>
    %past_value = ncnn.input {blob_name = "past_value", layer_name = "past_value"} : tensor<2x?x6xf32>
    %extra00 = ncnn.input {blob_name = "extra00", layer_name = "extra00"} : tensor<1xf32>
    %extra01 = ncnn.input {blob_name = "extra01", layer_name = "extra01"} : tensor<1xf32>
    %extra02 = ncnn.input {blob_name = "extra02", layer_name = "extra02"} : tensor<1xf32>
    %extra03 = ncnn.input {blob_name = "extra03", layer_name = "extra03"} : tensor<1xf32>
    %extra04 = ncnn.input {blob_name = "extra04", layer_name = "extra04"} : tensor<1xf32>
    %extra05 = ncnn.input {blob_name = "extra05", layer_name = "extra05"} : tensor<1xf32>
    %extra06 = ncnn.input {blob_name = "extra06", layer_name = "extra06"} : tensor<1xf32>
    %extra07 = ncnn.input {blob_name = "extra07", layer_name = "extra07"} : tensor<1xf32>
    %extra08 = ncnn.input {blob_name = "extra08", layer_name = "extra08"} : tensor<1xf32>
    %extra09 = ncnn.input {blob_name = "extra09", layer_name = "extra09"} : tensor<1xf32>
    %extra10 = ncnn.input {blob_name = "extra10", layer_name = "extra10"} : tensor<1xf32>
    %extra11 = ncnn.input {blob_name = "extra11", layer_name = "extra11"} : tensor<1xf32>
    %extra12 = ncnn.input {blob_name = "extra12", layer_name = "extra12"} : tensor<1xf32>
    %extra13 = ncnn.input {blob_name = "extra13", layer_name = "extra13"} : tensor<1xf32>
    %extra14 = ncnn.input {blob_name = "extra14", layer_name = "extra14"} : tensor<1xf32>
    %extra15 = ncnn.input {blob_name = "extra15", layer_name = "extra15"} : tensor<1xf32>
    %extra16 = ncnn.input {blob_name = "extra16", layer_name = "extra16"} : tensor<1xf32>
    %extra17 = ncnn.input {blob_name = "extra17", layer_name = "extra17"} : tensor<1xf32>
    %extra18 = ncnn.input {blob_name = "extra18", layer_name = "extra18"} : tensor<1xf32>
    %extra19 = ncnn.input {blob_name = "extra19", layer_name = "extra19"} : tensor<1xf32>
    %extra20 = ncnn.input {blob_name = "extra20", layer_name = "extra20"} : tensor<1xf32>
    %extra21 = ncnn.input {blob_name = "extra21", layer_name = "extra21"} : tensor<1xf32>
    %extra22 = ncnn.input {blob_name = "extra22", layer_name = "extra22"} : tensor<1xf32>
    %extra23 = ncnn.input {blob_name = "extra23", layer_name = "extra23"} : tensor<1xf32>
    %extra24 = ncnn.input {blob_name = "extra24", layer_name = "extra24"} : tensor<1xf32>
    %extra25 = ncnn.input {blob_name = "extra25", layer_name = "extra25"} : tensor<1xf32>
    %extra26 = ncnn.input {blob_name = "extra26", layer_name = "extra26"} : tensor<1xf32>
    %extra27 = ncnn.input {blob_name = "extra27", layer_name = "extra27"} : tensor<1xf32>
    %extra28 = ncnn.input {blob_name = "extra28", layer_name = "extra28"} : tensor<1xf32>
    %extra29 = ncnn.input {blob_name = "extra29", layer_name = "extra29"} : tensor<1xf32>
    %extra30 = ncnn.input {blob_name = "extra30", layer_name = "extra30"} : tensor<1xf32>
    %extra31 = ncnn.input {blob_name = "extra31", layer_name = "extra31"} : tensor<1xf32>
    %extra32 = ncnn.input {blob_name = "extra32", layer_name = "extra32"} : tensor<1xf32>
    %extra33 = ncnn.input {blob_name = "extra33", layer_name = "extra33"} : tensor<1xf32>
    %extra34 = ncnn.input {blob_name = "extra34", layer_name = "extra34"} : tensor<1xf32>
    %extra35 = ncnn.input {blob_name = "extra35", layer_name = "extra35"} : tensor<1xf32>
    %extra36 = ncnn.input {blob_name = "extra36", layer_name = "extra36"} : tensor<1xf32>
    %extra37 = ncnn.input {blob_name = "extra37", layer_name = "extra37"} : tensor<1xf32>
    %extra38 = ncnn.input {blob_name = "extra38", layer_name = "extra38"} : tensor<1xf32>
    %extra39 = ncnn.input {blob_name = "extra39", layer_name = "extra39"} : tensor<1xf32>
    %context, %updated_key, %updated_value = ncnn.sdpa %query, %key, %value, %mask, %past_key, %past_value {has_mask = true, kv_cache = true, scale = 0.35355338 : f32} : (tensor<4x?x8xf32>, tensor<2x1x8xf32>, tensor<2x1x6xf32>, tensor<?x?xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>) -> (tensor<4x?x6xf32>, tensor<2x?x8xf32>, tensor<2x?x6xf32>)
    ncnn.output %context {blob_name = "context"} : tensor<4x?x6xf32>
    ncnn.output %updated_key {blob_name = "updated_key"} : tensor<2x?x8xf32>
    ncnn.output %updated_value {blob_name = "updated_value"} : tensor<2x?x6xf32>
  }
}

// CHECK-LABEL: func.func @sdpa_cache_rehash
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 1, 0, 1>, array<i64: 1, 2, 2>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
// CHECK-SAME: ncnn.shape_program = [array<i64: 0, 2>, array<i64: 2, 1, 4, 1, 0, 1>, array<i64: 0, 8>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
// CHECK-SAME: ncnn.shape_program = [array<i64: 0, 2>, array<i64: 2, 1, 5, 1, 0, 1>, array<i64: 0, 6>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
