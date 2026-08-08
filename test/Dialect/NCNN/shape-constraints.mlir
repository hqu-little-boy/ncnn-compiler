// RUN: ncnn-mlir-opt %s | FileCheck %s

module {
  ncnn.model @constrained attributes {
    ncnn.shape_constraints = [
      #ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>,
      #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>
    ]
  } {
    %input = ncnn.input {blob_name = "data", layer_name = "input"}
      : tensor<3x?x?xf32>
    ncnn.output %input {blob_name = "output"} : tensor<3x?x?xf32>
  }
}

// CHECK: #ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>
// CHECK: #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>
