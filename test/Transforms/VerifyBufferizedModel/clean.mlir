// RUN: ncnn-mlir-opt --verify-bufferized-model %s | FileCheck %s

module {
  func.func @model(%input: memref<4xf32>, %output: memref<2xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    return
  }
}

// CHECK-LABEL: func.func @model(
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: memref<2xf32> {bufferize.result}
