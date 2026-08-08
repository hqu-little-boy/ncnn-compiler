// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_sigmoid {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.sigmoid %input : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_sigmoid
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
