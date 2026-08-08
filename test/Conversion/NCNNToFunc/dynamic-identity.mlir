// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @identity attributes {
    ncnn.shape_constraints = [
      #ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>,
      #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>
    ]
  } {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<3x?x?xf32>
    ncnn.output %input {blob_name = "output"} : tensor<3x?x?xf32>
  }
}

// CHECK-LABEL: func.func @identity(%arg0: tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32
// CHECK-SAME: attributes {{.*}}ncnn.shape_constraints = [#ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>, #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>]
// CHECK: return %arg0 : tensor<3x?x?xf32>
