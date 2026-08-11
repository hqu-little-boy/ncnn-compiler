// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func --split-input-file %s | FileCheck %s

module {
  ncnn.model @static_global_pooling {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.pooling %input {include_pad = false, kernel_h = 0 : i64, kernel_w = 0 : i64, kind = 0 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x?x?xf32>) -> tensor<4xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @static_global_pooling
// CHECK-NOT: ncnn.shape_program

// -----

module {
  ncnn.model @adaptive_pooling {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.pooling %input {include_pad = false, kernel_h = 3 : i64, kernel_w = -233 : i64, kind = 0 : i64, mode = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x?x?xf32>) -> tensor<4x3x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x3x?xf32>
  }
}

// CHECK-LABEL: func.func @adaptive_pooling
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 0, 3>, array<i64: 1, 0, 2>]
// CHECK-SAME: ncnn.shape_program_version = 2 : i32

// -----

module {
  ncnn.model @reshape_original_shape {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<?x4xf32>
    %output = "ncnn.reshape"(%input) {shape = array<i64: -9223372036854775808, -1>, shape_spec = array<i64: 0, -1>, shape_zero_sources = array<i64: 0, -1>} : (tensor<?x4xf32>) -> tensor<?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @reshape_original_shape
// CHECK-SAME: ncnn.shape_program = [array<i64: 1, 0, 0>, array<i64: 4,
// CHECK-SAME: ncnn.shape_program_version = 2 : i32
