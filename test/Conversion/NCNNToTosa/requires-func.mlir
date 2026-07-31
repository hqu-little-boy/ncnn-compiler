// RUN: not ncnn-mlir-opt --convert-ncnn-to-tosa %s 2>&1 | FileCheck %s

module {
  ncnn.model @network {
    %input = ncnn.input {blob_name = "images", layer_name = "input"} : tensor<3x4x5xf32>
    ncnn.output %input {blob_name = "result"} : tensor<3x4x5xf32>
  }
}

// CHECK: error: convert-ncnn-to-tosa requires ncnn.model to be converted to func.func first
