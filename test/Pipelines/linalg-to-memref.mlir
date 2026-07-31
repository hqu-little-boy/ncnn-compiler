// RUN: ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline %s | FileCheck %s

module {
  func.func @model(%input: tensor<4xf32>) -> tensor<4xf32> attributes {llvm.emit_c_interface, ncnn.entry_point} {
    %output = tensor.empty() : tensor<4xf32>
    %result = linalg.copy ins(%input : tensor<4xf32>) outs(%output : tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @model(
// CHECK-SAME: %[[INPUT:.*]]: memref<4xf32>,
// CHECK-SAME: %[[OUTPUT:.*]]: memref<4xf32> {bufferize.result})
// CHECK-NOT: ->
// CHECK: linalg.copy ins(%[[INPUT]] : memref<4xf32>) outs(%[[OUTPUT]] : memref<4xf32>)
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK-NOT: memref.dealloc %[[INPUT]]
// CHECK-NOT: memref.dealloc %[[OUTPUT]]
// CHECK: return
