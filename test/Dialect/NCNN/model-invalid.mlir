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
