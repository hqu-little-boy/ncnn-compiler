// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline --ncnn-vulkan-semantic-pipeline %s | FileCheck %s

module {
  ncnn.model @network {
    %first = ncnn.input {blob_name = "first", layer_name = "input_first"} : tensor<3x4x5xf32>
    %second = ncnn.input {blob_name = "second", layer_name = "input_second"} : tensor<3x4x5xf32>
    %sum = ncnn.binary %first, %second {ncnn.name = "add", ncnn.source_layer = 2 : i64, op_type = 0 : i64, scalar = 0.0 : f32, with_scalar = false} : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
    %relu = ncnn.relu %sum {negative_slope = 0.0 : f32, ncnn.name = "relu", ncnn.source_layer = 3 : i64} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
    ncnn.output %relu {blob_name = "result"} : tensor<3x4x5xf32>
  }
}

// CHECK-LABEL: func.func @network
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK-NOT: tosa.
// CHECK-NOT: ncnn.
