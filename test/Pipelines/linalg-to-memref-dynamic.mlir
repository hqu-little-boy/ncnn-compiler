// RUN: ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline %s | FileCheck %s

module {
  func.func @dynamic_generic(
      %input: tensor<1x?x?x4xf32>) ->
      (tensor<1x?x?x4xf32> {ncnn.shape_program = [array<i64>, array<i64>, array<i64>, array<i64>], ncnn.shape_source_input = 0 : i32})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    %c1 = arith.constant 1 : index
    %h = tensor.dim %input, %c1 : tensor<1x?x?x4xf32>
    %c2 = arith.constant 2 : index
    %w = tensor.dim %input, %c2 : tensor<1x?x?x4xf32>
    %empty = tensor.empty(%h, %w) : tensor<1x?x?x4xf32>
    %result = linalg.copy ins(%input : tensor<1x?x?x4xf32>) outs(%empty : tensor<1x?x?x4xf32>) -> tensor<1x?x?x4xf32>
    return %result : tensor<1x?x?x4xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_generic(
// CHECK-SAME: %[[INPUT:.*]]: memref<1x?x?x4xf32>,
// CHECK-SAME: %[[OUTPUT:.*]]: memref<1x?x?x4xf32> {bufferize.result, ncnn.shape_program = [array<i64>, array<i64>, array<i64>, array<i64>], ncnn.shape_source_input = 0 : i32})
// CHECK-NOT: ->
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: linalg.copy ins(%[[INPUT]] : memref<1x?x?x4xf32>) outs(%[[ALLOC]] : memref<1x?x?x4xf32>)
// CHECK: memref.copy %[[ALLOC]], %[[OUTPUT]]
// CHECK: memref.dealloc %[[ALLOC]]
// CHECK-NOT: memref<1x?x?x4xf32, strided
// CHECK-NOT: tensor.
// CHECK-NOT: bufferization.
