// RUN: ncnn-mlir-opt --verify-diagnostics %s

func.func @double_entry(%arg0: tensor<1xf32>) {
  // expected-error@+1 {{'ncnn.input' op expects parent op 'ncnn.model'}}
  %0 = ncnn.input {blob_name = "input", layer_name = "input"} : tensor<1xf32>
  return
}
