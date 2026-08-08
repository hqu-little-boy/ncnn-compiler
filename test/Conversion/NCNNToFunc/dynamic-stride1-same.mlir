// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_stride1_same {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %weight = ncnn.const {name = "weight", value = dense<0.000000e+00> : tensor<4x4x3x3xf32>} : tensor<4x4x3x3xf32>
    %output = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x?x?xf32>, tensor<4x4x3x3xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_stride1_same
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
