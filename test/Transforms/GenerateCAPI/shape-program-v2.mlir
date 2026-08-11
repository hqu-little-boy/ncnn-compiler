// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=shape_v2 manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(
      %first: memref<?x?xf32>,
      %second: memref<?x?xf32>,
      %output: memref<?x?x?x?x?xf32> {bufferize.result,
        ncnn.shape_program_version = 2 : i32,
        ncnn.shape_program = [
          array<i64: 2, 1, 0, 0, 0, 3>,
          array<i64: 3, 1, 1, 1, 0, 2>,
          array<i64: 4, 2, 1, 0, 1, 0, -1, 2, 1, 1, 0, 0, -7>,
          array<i64: 5, 1, 1, 0, 0, 3>,
          array<i64: 6, 1, 0, 0, 1, 1, 0>
        ]})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// MANIFEST: "name": "output1"
// MANIFEST: "shape_program": [
// MANIFEST: "shape_program_version": 2
// MANIFEST-NOT: "shape_source_input"

// LLVM-LABEL: llvm.func @shape_v2(
// LLVM: llvm.intr.sadd.with.overflow
// LLVM: llvm.intr.smul.with.overflow
// LLVM: llvm.sdiv
// LLVM: llvm.srem
// LLVM: llvm.select
// LLVM-LABEL: llvm.func @shape_v2_infer_output_shapes(
// LLVM: llvm.intr.sadd.with.overflow
// LLVM: llvm.intr.smul.with.overflow
// LLVM: llvm.sdiv
// LLVM: llvm.srem
// LLVM: llvm.select
