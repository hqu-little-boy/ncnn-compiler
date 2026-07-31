// RUN: not ncnn-mlir-opt --normalize-ncnn %s 2>&1 | FileCheck %s

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "images", layer_name = "input"} : tensor<3x4x5xf32>
    ncnn.output %input {blob_name = "result"} : tensor<3x4x5xf32>
  }
}

// CHECK: error: normalize-ncnn requires ncnn.model to be converted to func.func first
