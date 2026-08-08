// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @identity {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<3x?x?xf32>
    ncnn.output %input {blob_name = "output"} : tensor<3x?x?xf32>
  }
}

// CHECK-LABEL: func.func @identity(%arg0: tensor<3x?x?xf32>) -> (tensor<3x?x?xf32>
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32
// CHECK: return %arg0 : tensor<3x?x?xf32>
