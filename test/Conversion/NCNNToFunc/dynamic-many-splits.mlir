// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @many_splits attributes {
    ncnn.shape_constraints = [
      #ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>,
      #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>
    ]
  } {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<3x?x?xf32>
    %s0, %s1 = ncnn.split %input : (tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>, tensor<3x?x?xf32>)
    %s2, %s3 = ncnn.split %s0 : (tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>, tensor<3x?x?xf32>)
    %s4, %s5 = ncnn.split %s1 : (tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>, tensor<3x?x?xf32>)
    %s6, %s7 = ncnn.split %s2 : (tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>, tensor<3x?x?xf32>)
    %s8, %s9 = ncnn.split %s3 : (tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>, tensor<3x?x?xf32>)
    %cat = ncnn.concat %s4, %s5 {axis = 0 : i64} : (tensor<3x?x?xf32>, tensor<3x?x?xf32>) -> tensor<6x?x?xf32>
    ncnn.output %cat {blob_name = "output"} : tensor<6x?x?xf32>
  }
}

// CHECK-LABEL: func.func @many_splits
// CHECK: return
