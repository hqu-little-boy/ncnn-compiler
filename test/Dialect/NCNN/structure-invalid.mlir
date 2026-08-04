// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @short_split(%input: tensor<2xf32>) {
  // expected-error@+1 {{'ncnn.split' op expected 2 or more results}}
  %result = ncnn.split %input : (tensor<2xf32>) -> tensor<2xf32>
  return
}

// -----

func.func @short_concat(%input: tensor<2xf32>) {
  // expected-error@+1 {{'ncnn.concat' op expected 2 or more operands, but found 1}}
  %result = ncnn.concat %input {axis = 0 : i64} : (tensor<2xf32>) -> tensor<2xf32>
  return
}

// -----

ncnn.model @bad_const {
  // expected-error@+1 {{'ncnn.const' op failed to verify that all of {value, output} have same type}}
  %value = "ncnn.const"() <{name = "value", value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<2xf32>
  ncnn.output %value {blob_name = "output"} : tensor<2xf32>
}
