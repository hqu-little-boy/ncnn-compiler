// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "images", layer_name = "input"} : tensor<3x4x4xf32>
    %weight = ncnn.const {name = "conv.weight", value = dense<0.000000e+00> : tensor<2x3x1x1xf32>} : tensor<2x3x1x1xf32>
    %conv = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x4x4xf32>, tensor<2x3x1x1xf32>) -> tensor<2x4x4xf32>
    ncnn.output %conv {blob_name = "result"} : tensor<2x4x4xf32>
  }
}

// CHECK-NOT: ncnn.model
// CHECK-NOT: ncnn.input
// CHECK-NOT: ncnn.const
// CHECK-NOT: ncnn.output
// CHECK-LABEL: func.func @network(%arg0: tensor<3x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK-SAME: attributes {llvm.emit_c_interface, ncnn.entry_point}
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<0.000000e+00> : tensor<2x3x1x1xf32>
// CHECK: %[[CONV:.*]] = ncnn.convolution %arg0, %[[WEIGHT]]
// CHECK: return %[[CONV]] : tensor<2x4x4xf32>
