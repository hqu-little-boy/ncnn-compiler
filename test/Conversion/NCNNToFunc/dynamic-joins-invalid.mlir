// RUN: not ncnn-mlir-opt --convert-ncnn-model-to-func %s 2>&1 | FileCheck %s

module {
  ncnn.model @unproven_join {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<3x?x?xf32>
    %half_weight = ncnn.const {name = "half.weight", value = dense<0.000000e+00> : tensor<8x3x1x1xf32>} : tensor<8x3x1x1xf32>
    %half = ncnn.convolution %input, %half_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x?x?xf32>, tensor<8x3x1x1xf32>) -> tensor<8x?x?xf32>
    %quarter_weight = ncnn.const {name = "quarter.weight", value = dense<0.000000e+00> : tensor<8x3x1x1xf32>} : tensor<8x3x1x1xf32>
    %quarter = ncnn.convolution %input, %quarter_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 4 : i64, stride_w = 4 : i64} : (tensor<3x?x?xf32>, tensor<8x3x1x1xf32>) -> tensor<8x?x?xf32>
    %upsampled = ncnn.interp %quarter {height_scale = 2 : i64, width_scale = 2 : i64} : (tensor<8x?x?xf32>) -> tensor<8x?x?xf32>
    %concat = ncnn.concat %half, %upsampled {axis = 0 : i64} : (tensor<8x?x?xf32>, tensor<8x?x?xf32>) -> tensor<16x?x?xf32>
    ncnn.output %concat {blob_name = "output"} : tensor<16x?x?xf32>
  }
}

// CHECK: error: 'ncnn.concat' op cannot prove input dimension 1 equal under input shape constraints
