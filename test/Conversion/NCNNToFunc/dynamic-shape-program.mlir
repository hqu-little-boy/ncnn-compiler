// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  "ncnn.model"() <{sym_name = "shape_chain"}> ({
    %input = "ncnn.input"() {blob_name = "input", layer_name = "input"}
      : () -> tensor<3x?x?xf32>
    %padded = "ncnn.padding"(%input) {top = 1 : i64, bottom = 2 : i64,
      left = 3 : i64, right = 4 : i64, value = 0.0 : f32}
      : (tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
    %resized = "ncnn.interp"(%padded) {height_scale = 2 : i64,
      width_scale = 3 : i64}
      : (tensor<3x?x?xf32>) -> tensor<3x?x?xf32>
    "ncnn.output"(%resized) {blob_name = "output"}
      : (tensor<3x?x?xf32>) -> ()
  }) : () -> ()
}

// CHECK-LABEL: func.func @shape_chain
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 0, 3, 1, 2>, array<i64: 0, 7, 1, 3>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32
