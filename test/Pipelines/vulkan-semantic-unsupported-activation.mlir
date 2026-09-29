// RUN: not ncnn-mlir-opt --ncnn-to-tosa-pipeline --ncnn-vulkan-semantic-pipeline %s 2>&1 | FileCheck %s

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<1x2x3xf32>
    %relu = ncnn.relu %input {negative_slope = 0.25 : f32, ncnn.name = "leaky_relu", ncnn.source_layer = 5 : i64} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32> loc("leaky_relu"("ncnn-layer":5:0))
    ncnn.output %relu {blob_name = "result"} : tensor<1x2x3xf32>
  }
}

// CHECK: error: 'linalg.generic' op is not supported by Vulkan V1
// CHECK-SAME: ncnn.name="leaky_relu"
// CHECK-SAME: ncnn.source_layer=5
