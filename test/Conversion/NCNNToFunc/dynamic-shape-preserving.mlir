// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @dynamic_shape_preserving {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %hard_sigmoid = ncnn.hard_sigmoid %input {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %hard_swish = ncnn.hard_swish %hard_sigmoid {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %gelu = ncnn.gelu %hard_swish {fast = false} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %dropout = ncnn.dropout %gelu {scale = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %softmax = ncnn.softmax %dropout {axis = 0 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    %slope = ncnn.const {name = "slope", value = dense<1.000000e+00> : tensor<4xf32>} : tensor<4xf32>
    %mean = ncnn.const {name = "mean", value = dense<0.000000e+00> : tensor<4xf32>} : tensor<4xf32>
    %variance = ncnn.const {name = "variance", value = dense<1.000000e+00> : tensor<4xf32>} : tensor<4xf32>
    %bias = ncnn.const {name = "bias", value = dense<0.000000e+00> : tensor<4xf32>} : tensor<4xf32>
    %normalized = ncnn.batch_norm %softmax, %slope, %mean, %variance, %bias {epsilon = 1.000000e-05 : f32} : (tensor<4x?x?xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4x?x?xf32>
    %output = ncnn.shuffle_channel %normalized {group = 2 : i64, reverse = false} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_shape_preserving
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32
