// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  func.func private @helper() {
    return
  }

  ncnn.model @network {
    %first = ncnn.input {blob_name = "first", layer_name = "first"} : tensor<2xf32>
    %second = ncnn.input {blob_name = "second", layer_name = "second"} : tensor<3xf32>
    %scaled = ncnn.dropout %second {scale = 5.000000e-01 : f32} : (tensor<3xf32>) -> tensor<3xf32>
    ncnn.output %scaled {blob_name = "scaled"} : tensor<3xf32>
    ncnn.output %first {blob_name = "first_result"} : tensor<2xf32>
  }
}

// CHECK-LABEL: func.func private @helper()
// CHECK-LABEL: func.func @network(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xf32>)
// CHECK-SAME: attributes {llvm.emit_c_interface, ncnn.entry_point}
// CHECK: %[[SCALED:.*]] = ncnn.dropout %arg1
// CHECK: return %[[SCALED]], %arg0 : tensor<3xf32>, tensor<2xf32>
// CHECK-NOT: ncnn.model
// CHECK-NOT: ncnn.input
// CHECK-NOT: ncnn.output
