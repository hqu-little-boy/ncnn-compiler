// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_deconvolution_inference {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %weight = ncnn.const {name = "weight", value = dense<0.000000e+00> : tensor<2x4x2x2xf32>} : tensor<2x4x2x2xf32>
    %output = ncnn.deconvolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 2 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>, tensor<2x4x2x2xf32>) -> tensor<2x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<2x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_deconvolution_inference
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 0, -1, 1, 2, 0, 2>, array<i64: 0, -1, 1, 2, 0, 2>]
