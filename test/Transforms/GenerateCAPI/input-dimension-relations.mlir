// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=related manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(
      %mask: memref<1x?xf32>,
      %past_key: memref<2x?x4xf32>,
      %past_value: memref<2x?x4xf32>,
      %present_key: memref<2x?x4xf32> {bufferize.result,
        ncnn.shape_program = [array<i64: 0, 2>, array<i64: 1, 1, 1>, array<i64: 0, 4>],
        ncnn.shape_program_version = 2 : i32})
      attributes {llvm.emit_c_interface, ncnn.entry_point,
        ncnn.input_dim_relations = [
          array<i64: 0, 1, 1, 1, 1>,
          array<i64: 1, 1, 2, 1, 0>,
          array<i64: 1, 0, 2, 0, 0>
        ]} {
    return
  }
}

// MANIFEST: "input_dimension_relations": [
// MANIFEST: "lhs_dimension": 1
// MANIFEST: "lhs_input": 0
// MANIFEST: "offset": 1
// MANIFEST: "rhs_dimension": 1
// MANIFEST: "rhs_input": 1
// MANIFEST: "lhs_input": 1
// MANIFEST: "offset": 0
// MANIFEST: "rhs_input": 2
// MANIFEST: "lhs_dimension": 0
// MANIFEST: "rhs_dimension": 0

// LLVM-LABEL: llvm.func @related(
// LLVM: llvm.intr.sadd.with.overflow
// LLVM: llvm.icmp "ne"
// LLVM: llvm.intr.sadd.with.overflow
// LLVM: llvm.icmp "ne"
// LLVM: llvm.call @__ncnn_internal_related
// LLVM-LABEL: llvm.func @related_infer_output_shapes(
// LLVM: llvm.intr.sadd.with.overflow
// LLVM: llvm.icmp "ne"
// LLVM: llvm.intr.sadd.with.overflow
// LLVM: llvm.icmp "ne"
