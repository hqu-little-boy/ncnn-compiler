// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_joins attributes {
    ncnn.shape_constraints = [
      #ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>,
      #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>
    ]
  } {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<3x?x?xf32>
    %half_weight = ncnn.const {name = "half.weight", value = dense<0.000000e+00> : tensor<8x3x1x1xf32>} : tensor<8x3x1x1xf32>
    %half = ncnn.convolution %input, %half_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x?x?xf32>, tensor<8x3x1x1xf32>) -> tensor<8x?x?xf32>
    %quarter_weight = ncnn.const {name = "quarter.weight", value = dense<0.000000e+00> : tensor<8x3x1x1xf32>} : tensor<8x3x1x1xf32>
    %quarter = ncnn.convolution %input, %quarter_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 4 : i64, stride_w = 4 : i64} : (tensor<3x?x?xf32>, tensor<8x3x1x1xf32>) -> tensor<8x?x?xf32>
    %upsampled = ncnn.interp %quarter {height_scale = 2 : i64, width_scale = 2 : i64} : (tensor<8x?x?xf32>) -> tensor<8x?x?xf32>
    %sum = ncnn.binary %half, %upsampled {op_type = 0 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<8x?x?xf32>, tensor<8x?x?xf32>) -> tensor<8x?x?xf32>
    %pointwise_weight = ncnn.const {name = "pointwise.weight", value = dense<0.000000e+00> : tensor<8x8x1x1xf32>} : tensor<8x8x1x1xf32>
    %pointwise = ncnn.convolution %half, %pointwise_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<8x?x?xf32>, tensor<8x8x1x1xf32>) -> tensor<8x?x?xf32>
    %depthwise_weight = ncnn.const {name = "depthwise.weight", value = dense<0.000000e+00> : tensor<8x1x5x5xf32>} : tensor<8x1x5x5xf32>
    %depthwise = ncnn.convolution_depthwise %pointwise, %depthwise_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 5 : i64, kernel_w = 5 : i64, pad_bottom = 2 : i64, pad_left = 2 : i64, pad_right = 2 : i64, pad_top = 2 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<8x?x?xf32>, tensor<8x1x5x5xf32>) -> tensor<8x?x?xf32>
    %concat = ncnn.concat %half, %upsampled, %depthwise {axis = 0 : i64} : (tensor<8x?x?xf32>, tensor<8x?x?xf32>, tensor<8x?x?xf32>) -> tensor<24x?x?xf32>
    ncnn.output %concat {blob_name = "output"} : tensor<24x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_joins
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 0, -1, 2, 2, 0, 1>, array<i64: 0, -1, 2, 2, 0, 1>]
// CHECK: ncnn.binary
// CHECK: ncnn.concat
