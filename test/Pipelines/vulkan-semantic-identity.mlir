// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline --ncnn-vulkan-semantic-pipeline %s | FileCheck %s

module {
  ncnn.model @identity {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<1x2x3xf32>
    %identity = ncnn.dropout %input {ncnn.name = "inference_dropout", ncnn.source_layer = 1 : i64, scale = 1.0 : f32} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    ncnn.output %identity {blob_name = "result"} : tensor<1x2x3xf32>
  }
}

// CHECK-LABEL: func.func @identity
// CHECK: return %arg0 : tensor<1x2x3xf32>
// CHECK-NOT: linalg.
// CHECK-NOT: tosa.
// CHECK-NOT: ncnn.
