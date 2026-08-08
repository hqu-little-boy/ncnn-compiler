// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_pooling_inference {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.pooling %input {include_pad = false, kernel_h = 3 : i64, kernel_w = 5 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 2 : i64, pad_left = 1 : i64, pad_mode = 1 : i64, pad_right = 3 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 4 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_pooling_inference
// CHECK: ncnn.pooling
