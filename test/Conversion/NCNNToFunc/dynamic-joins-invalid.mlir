// RUN: not ncnn-mlir-opt --convert-ncnn-model-to-func --split-input-file %s 2>&1 | FileCheck %s

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

// -----

module {
  ncnn.model @unproven_spatial_concat {
    %first = ncnn.input {blob_name = "first", layer_name = "first"} : tensor<4x?x?xf32>
    %second = ncnn.input {blob_name = "second", layer_name = "second"} : tensor<4x?x?xf32>
    %joined = ncnn.concat %first, %second {axis = -2 : i64} : (tensor<4x?x?xf32>, tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %joined {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK: error: 'ncnn.concat' op cannot prove input dimension 2 equal under input shape constraints

// -----

module {
  ncnn.model @unproven_binary_broadcast {
    %first = ncnn.input {blob_name = "first", layer_name = "first"} : tensor<?x?x?xf32>
    %second = ncnn.input {blob_name = "second", layer_name = "second"} : tensor<?x?x?xf32>
    %maximum = ncnn.binary %first, %second {op_type = 4 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    ncnn.output %maximum {blob_name = "output"} : tensor<?x?x?xf32>
  }
}

// CHECK: error: 'ncnn.binary' op cannot prove input dimension 0 equal under input shape constraints
