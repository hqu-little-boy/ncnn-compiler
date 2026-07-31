// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %s | FileCheck %s
// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %s | mlir-opt-21 --tosa-validate

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "images", layer_name = "input"} : tensor<2x4x4xf32>
    %weight = ncnn.const {name = "conv.weight", value = dense<0.000000e+00> : tensor<3x2x1x1xf32>} : tensor<3x2x1x1xf32>
    %conv = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, ncnn.name = "conv", ncnn.source_layer = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32>, tensor<3x2x1x1xf32>) -> tensor<3x4x4xf32>
    %relu = ncnn.relu %conv {ncnn.name = "relu", ncnn.source_layer = 2 : i64} : (tensor<3x4x4xf32>) -> tensor<3x4x4xf32>
    ncnn.output %relu {blob_name = "result"} : tensor<3x4x4xf32>
  }
}

// CHECK-LABEL: func.func @network
// CHECK: tosa.conv2d
// CHECK: tosa.clamp
// CHECK-NOT: ncnn.
