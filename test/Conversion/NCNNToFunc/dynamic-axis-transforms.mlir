// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func --split-input-file %s | FileCheck %s

module {
  ncnn.model @permute {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<?x?xf32>
    %output = ncnn.permute %input {permutation = array<i64: 1, 0>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @permute
// CHECK-SAME: ncnn.shape_program = [array<i64: 3, 1>, array<i64: 3, 0>]

// -----

module {
  ncnn.model @rank_changes {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<1x?x?xf32>
    %squeezed = ncnn.squeeze %input {axes = array<i64: 0>} : (tensor<1x?x?xf32>) -> tensor<?x?xf32>
    %output = ncnn.expand_dims %squeezed {axes = array<i64: 0>} : (tensor<?x?xf32>) -> tensor<1x?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<1x?x?xf32>
  }
}

// CHECK-LABEL: func.func @rank_changes
// CHECK-SAME: ncnn.shape_program = [array<i64: 3, 1>, array<i64>, array<i64>]

// -----

module {
  ncnn.model @reduction {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %output = ncnn.reduction %input {axes = array<i64: 0>, coeff = 1.000000e+00 : f32, keepdims = false, kind = 3 : i64, reduce_all = false} : (tensor<4x?x?xf32>) -> tensor<?x?xf32>
    ncnn.output %output {blob_name = "output"} : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @reduction
// CHECK-SAME: ncnn.shape_program = [array<i64: 3, 1>, array<i64: 3, 2>]

// -----

module {
  ncnn.model @gemm {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<?x4xf32>
    %weight = ncnn.const {name = "weight", value = dense<0.000000e+00> : tensor<2x4xf32>} : tensor<2x4xf32>
    %bias = ncnn.const {name = "bias", value = dense<0.000000e+00> : tensor<2xf32>} : tensor<2xf32>
    %output = ncnn.gemm %input, %weight, %bias {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32} : (tensor<?x4xf32>, tensor<2x4xf32>, tensor<2xf32>) -> tensor<?x2xf32>
    ncnn.output %output {blob_name = "output"} : tensor<?x2xf32>
  }
}

// CHECK-LABEL: func.func @gemm
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64: 3, 0>]

// -----

module {
  ncnn.model @slice {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<4x?x?xf32>
    %left, %right = ncnn.slice %input {axis = 0 : i64, slices = array<i64: 2, 2>} : (tensor<4x?x?xf32>) -> (tensor<2x?x?xf32>, tensor<2x?x?xf32>)
    ncnn.output %right {blob_name = "output"} : tensor<2x?x?xf32>
  }
}

// CHECK-LABEL: func.func @slice
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>, array<i64>]
