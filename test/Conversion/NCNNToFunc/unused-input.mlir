// RUN: ncnn-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=CANONICALIZE
// RUN: ncnn-mlir-opt --canonicalize --convert-ncnn-model-to-func %s | FileCheck %s --check-prefix=CONVERT

module {
  ncnn.model @network {
    %unused = ncnn.input {blob_name = "unused", layer_name = "unused"} : tensor<2xf32>
    %used = ncnn.input {blob_name = "used", layer_name = "used"} : tensor<3xf32>
    ncnn.output %used {blob_name = "result"} : tensor<3xf32>
  }
}

// CANONICALIZE-LABEL: ncnn.model @network
// CANONICALIZE: %[[UNUSED:.*]] = ncnn.input {blob_name = "unused"
// CANONICALIZE: %[[USED:.*]] = ncnn.input {blob_name = "used"
// CANONICALIZE: ncnn.output %[[USED]]

// CONVERT-LABEL: func.func @network(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32>
// CONVERT: return %arg1 : tensor<3xf32>
