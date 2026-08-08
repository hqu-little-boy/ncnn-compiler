// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_constraints manifest-path=%t.json' %s -o %t.mlir
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir -o %t.llvm.mlir
// RUN: FileCheck %s --check-prefix=LLVM < %t.llvm.mlir

module {
  func.func @model(%input: memref<3x?x?xf32>, %output: memref<1x?x?xf32> {bufferize.result, ncnn.shape_source_input = 0 : i32, ncnn.shape_program = [array<i64>, array<i64>, array<i64>]}) attributes {llvm.emit_c_interface, ncnn.entry_point, ncnn.shape_constraints = [#ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>, #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>]} {
    return
  }
}

// LLVM: llvm.icmp "sge"
// LLVM: llvm.srem
