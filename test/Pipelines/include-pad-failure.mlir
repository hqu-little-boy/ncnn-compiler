// RUN: not ncnn-mlir-opt --ncnn-to-tosa-pipeline %s 2>&1 | FileCheck %s

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "images", layer_name = "input"} : tensor<2x4x4xf32>
    %pool = ncnn.pooling %input {include_pad = true, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 1 : i64, mode = 0 : i64, ncnn.name = "average_include_pad", ncnn.source_layer = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>) -> tensor<2x3x3xf32>
    ncnn.output %pool {blob_name = "result"} : tensor<2x3x3xf32>
  }
}

// CHECK: error: 'ncnn.pooling' op remains after lowering; op=ncnn.pooling, ncnn.name="average_include_pad", ncnn.source_layer=1
