// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @depthwise_after_shape_preserving {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %relu = ncnn.relu %input : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %sigmoid = ncnn.sigmoid %relu : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %first, %second = ncnn.split %sigmoid : (tensor<4x?x?xf32>) -> (tensor<4x?x?xf32>, tensor<4x?x?xf32>)
    %weight = ncnn.const {name = "depthwise.weight", value = dense<0.000000e+00> : tensor<4x1x3x3xf32>} : tensor<4x1x3x3xf32>
    %half = ncnn.convolution_depthwise %second, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>, tensor<4x1x3x3xf32>) -> tensor<4x?x?xf32>
    ncnn.output %half {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @depthwise_after_shape_preserving
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 0, -1, 2, 2, 0, 1>, array<i64: 0, -1, 2, 2, 0, 1>]
// CHECK: ncnn.relu
// CHECK: ncnn.sigmoid
// CHECK: ncnn.split

// -----

module {
  ncnn.model @deconvolution_program {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %weight = ncnn.const {name = "deconvolution.weight", value = dense<0.000000e+00> : tensor<2x4x2x2xf32>} : tensor<2x4x2x2xf32>
    %output = ncnn.deconvolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 2 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>, tensor<2x4x2x2xf32>) -> tensor<2x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<2x?x?xf32>
  }
}

// CHECK-LABEL: func.func @deconvolution_program
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 0, -1, 1, 2, 0, 2>, array<i64: 0, -1, 1, 2, 0, 2>]

// -----

module {
  ncnn.model @pooling_program {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.pooling %input {include_pad = false, kernel_h = 3 : i64, kernel_w = 5 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 2 : i64, pad_left = 1 : i64, pad_mode = 1 : i64, pad_right = 3 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 4 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @pooling_program
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 2, 2, 0, 1>, array<i64: 0, -1, 2, 4, 0, 1>]

// -----

module {
  ncnn.model @same_convolution_program {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %weight = ncnn.const {name = "convolution.weight", value = dense<0.000000e+00> : tensor<4x4x3x3xf32>} : tensor<4x4x3x3xf32>
    %output = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = -233 : i64, pad_left = -233 : i64, pad_right = -233 : i64, pad_top = -233 : i64, stride_h = 2 : i64, stride_w = 4 : i64} : (tensor<4x?x?xf32>, tensor<4x4x3x3xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @same_convolution_program
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 0, -1, 2, 2, 0, 1>, array<i64: 0, -1, 2, 4, 0, 1>]

// -----

module {
  ncnn.model @same_type_is_provenance {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.hard_sigmoid %input {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @same_type_is_provenance
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32

// -----

module {
  ncnn.model @static_unit_broadcast {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %scale = ncnn.const {name = "scale", value = dense<1.000000e+00> : tensor<4x1x1xf32>} : tensor<4x1x1xf32>
    %output = ncnn.binary %input, %scale {op_type = 2 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<4x?x?xf32>, tensor<4x1x1xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @static_unit_broadcast
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32

// -----

module {
  ncnn.model @provenance_unit_broadcast {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %pooled = ncnn.pooling %input {include_pad = false, kernel_h = 0 : i64, kernel_w = 0 : i64, kind = 1 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x?x?xf32>) -> tensor<4xf32>
    %scale = ncnn.reshape %pooled {shape = array<i64: 4, 1, 1>} : (tensor<4xf32>) -> tensor<4x1x1xf32>
    %output = ncnn.binary %input, %scale {op_type = 2 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<4x?x?xf32>, tensor<4x1x1xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @provenance_unit_broadcast
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32
