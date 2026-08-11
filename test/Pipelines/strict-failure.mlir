// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %s | FileCheck %s

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "images", layer_name = "input"} : tensor<2x4x4xf32>
    %pool = ncnn.pooling %input {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 2 : i64, ncnn.name = "adaptive_pool", ncnn.source_layer = 7 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>) -> tensor<2x2x2xf32>
    ncnn.output %pool {blob_name = "result"} : tensor<2x2x2xf32>
  }
}

// CHECK-LABEL: func.func @network
// CHECK: linalg.generic
// CHECK-NOT: ncnn.pooling
