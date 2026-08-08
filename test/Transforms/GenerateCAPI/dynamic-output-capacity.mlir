// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_capacity' %s -o %t.mlir
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s

module {
  func.func @model(%input: memref<3x?x?xf32>, %output: memref<1x?x?xf32> {bufferize.result, ncnn.shape_source_input = 0 : i32, ncnn.shape_program = [array<i64>, array<i64>, array<i64>]}) attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// CHECK: llvm.func @dynamic_capacity(
// CHECK-SAME: i64) -> i32
// CHECK: llvm.intr.umul.with.overflow
