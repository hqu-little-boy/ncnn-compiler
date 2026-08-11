// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_multi_source {
    %first = ncnn.input {blob_name = "first", layer_name = "first"} : tensor<?x1x?xf32>
    %second = ncnn.input {blob_name = "second", layer_name = "second"} : tensor<1x?x1xf32>
    %sum = ncnn.binary %first, %second {op_type = 0 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<?x1x?xf32>, tensor<1x?x1xf32>) -> tensor<?x?x?xf32>
    ncnn.output %sum {blob_name = "output"} : tensor<?x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_multi_source
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 1, 1, 1>, array<i64: 1, 0, 2>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
// CHECK-NOT: ncnn.shape_source_input

// -----

module {
  ncnn.model @dynamic_spatial_concat_same_source {
    %input = ncnn.input {blob_name = "input", layer_name = "input"} : tensor<4x?x?xf32>
    %first, %second = ncnn.split %input : (tensor<4x?x?xf32>) -> (tensor<4x?x?xf32>, tensor<4x?x?xf32>)
    %joined = ncnn.concat %first, %second {axis = -2 : i64} : (tensor<4x?x?xf32>, tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %joined {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_spatial_concat_same_source
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 2, 1, 0, 1, 1, 0, 1>, array<i64: 1, 0, 2>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32

// -----

module {
  ncnn.model @dynamic_spatial_concat {
    %first = ncnn.input {blob_name = "first", layer_name = "first"} : tensor<4x?x8xf32>
    %second = ncnn.input {blob_name = "second", layer_name = "second"} : tensor<4x?x8xf32>
    %joined = ncnn.concat %first, %second {axis = -2 : i64} : (tensor<4x?x8xf32>, tensor<4x?x8xf32>) -> tensor<4x?x8xf32>
    ncnn.output %joined {blob_name = "output"} : tensor<4x?x8xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_spatial_concat
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 2, 1, 0, 1, 1, 1, 1>, array<i64: 1, 0, 2>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32

// -----
