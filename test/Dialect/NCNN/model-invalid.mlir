// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @double_entry(%arg0: tensor<1xf32>) {
  // expected-error@+1 {{'ncnn.input' op expects parent op 'ncnn.model'}}
  %0 = ncnn.input {blob_name = "input", layer_name = "input"} : tensor<1xf32>
  return
}

// -----

// expected-error@+1 {{'ncnn.model' op body can only contain ncnn dialect operations}}
ncnn.model @foreign_operation {
  %value = arith.constant dense<0.000000e+00> : tensor<1xf32>
  ncnn.output %value {blob_name = "output"} : tensor<1xf32>
}

// -----

ncnn.model @duplicate_inputs {
  %first = ncnn.input {blob_name = "input", layer_name = "first"} : tensor<1xf32>
  // expected-error@+1 {{'ncnn.input' op duplicates input blob 'input'}}
  %second = ncnn.input {blob_name = "input", layer_name = "second"} : tensor<1xf32>
  ncnn.output %first {blob_name = "output"} : tensor<1xf32>
}

// -----

// expected-error@+1 {{'ncnn.model' op requires at least one ncnn.output}}
ncnn.model @missing_output {
  %input = ncnn.input {blob_name = "input", layer_name = "input"} : tensor<1xf32>
}

// -----

ncnn.model @malformed_input {
  // expected-error@+1 {{'ncnn.input' op requires attribute 'blob_name'}}
  %input = "ncnn.input"() <{layer_name = "input"}> : () -> tensor<1xf32>
  ncnn.output %input {blob_name = "output"} : tensor<1xf32>
}

// -----

func.func @encoded_relu(%input: tensor<2x4x4xf32, "test.encoding">) {
  // expected-error@+1 {{'ncnn.relu' op operand #0 must be ranked tensor without encoding}}
  %result = ncnn.relu %input : (tensor<2x4x4xf32, "test.encoding">) -> tensor<2x4x4xf32, "test.encoding">
  return
}

// -----

func.func @encoded_convolution(%input: tensor<2x4x4xf32, "test.encoding">, %weight: tensor<3x2x1x1xf32>) {
  // expected-error@+1 {{'ncnn.convolution' op operand #0 must be ranked tensor without encoding}}
  %result = ncnn.convolution %input, %weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x4x4xf32, "test.encoding">, tensor<3x2x1x1xf32>) -> tensor<3x4x4xf32>
  return
}
