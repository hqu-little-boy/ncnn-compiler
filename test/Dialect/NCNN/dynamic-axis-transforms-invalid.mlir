// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @squeeze_dynamic_axis(%input: tensor<?x?x?xf32>) {
  // expected-error@+2 {{Squeeze axis must have unit extent}}
  // expected-error@+1 {{'ncnn.squeeze' op failed to infer returned types}}
  %0 = ncnn.squeeze %input {axes = array<i64: 0>} : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
  return
}

// -----

func.func @slice_invalid_size(%input: tensor<?x?x?xf32>) {
  // expected-error@+2 {{Slice dynamic axis requires a trailing -233 remainder slice}}
  // expected-error@+1 {{'ncnn.slice' op failed to infer returned types}}
  %0, %1 = ncnn.slice %input {axis = 0 : i64, slices = array<i64: 2, -2>} : (tensor<?x?x?xf32>) -> (tensor<2x?x?xf32>, tensor<?x?x?xf32>)
  return
}

// -----

func.func @reduction_dynamic_axis(%input: tensor<?x?x?xf32>) {
  // expected-error@+2 {{Reduction axes must have static extents}}
  // expected-error@+1 {{'ncnn.reduction' op failed to infer returned types}}
  %0 = ncnn.reduction %input {axes = array<i64: 0>, coeff = 1.000000e+00 : f32, keepdims = false, kind = 3 : i64, reduce_all = false} : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
  return
}

// -----

func.func @gemm_dynamic_k(%input: tensor<?x?xf32>, %weight: tensor<2x?xf32>, %bias: tensor<2xf32>) {
  // expected-error@+2 {{Gemm expects input [M,K], weight [N,K], and bias [N] with static K/N}}
  // expected-error@+1 {{'ncnn.gemm' op failed to infer returned types}}
  %0 = ncnn.gemm %input, %weight, %bias {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32} : (tensor<?x?xf32>, tensor<2x?xf32>, tensor<2xf32>) -> tensor<?x2xf32>
  return
}
