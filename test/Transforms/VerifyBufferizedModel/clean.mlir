// RUN: ncnn-mlir-opt --verify-bufferized-model %s | FileCheck %s

module {
  func.func @model(%input: memref<4xf32>, %output: memref<2xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    return
  }
}

func.func private @branch_carried_allocation() {
  %allocation = memref.alloc() : memref<4xf32>
  cf.br ^release(%allocation : memref<4xf32>)
^release(%alias: memref<4xf32>):
  memref.dealloc %alias : memref<4xf32>
  return
}

func.func private @automatic_allocation() {
  %allocation = memref.alloca() : memref<4xf32>
  return
}

// CHECK-LABEL: func.func @model(
// CHECK-SAME: memref<4xf32>
// CHECK-SAME: memref<2xf32> {bufferize.result}
// CHECK-LABEL: func.func private @branch_carried_allocation()
// CHECK-LABEL: func.func private @automatic_allocation()
