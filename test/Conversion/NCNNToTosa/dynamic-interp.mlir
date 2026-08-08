// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

module {
  func.func @dynamic_interp(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
    %0 = "ncnn.interp"(%arg0) {height_scale = 2 : i64, width_scale = 3 : i64}
      : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    return %0 : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_interp
// CHECK: tensor.dim
// CHECK: arith.muli
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK: linalg.index 1
// CHECK: arith.divui
// CHECK: tensor.extract
// CHECK-NOT: tosa.resize
// CHECK-NOT: ncnn.interp
