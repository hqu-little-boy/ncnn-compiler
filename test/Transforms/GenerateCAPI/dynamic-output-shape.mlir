// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_identity manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(
      %input: memref<3x?x?xf32>,
      %output: memref<3x?x?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>, array<i64>, array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {llvm.emit_c_interface, ncnn.entry_point,
        ncnn.shape_constraints = [
          #ncnn.dim_constraint<input = 0, dim = 1, min = 32, multiple_of = 32>,
          #ncnn.dim_constraint<input = 0, dim = 2, min = 32, multiple_of = 32>
        ]} {
    memref.copy %input, %output : memref<3x?x?xf32> to memref<3x?x?xf32>
    return
  }
}

// MANIFEST: "dimension_constraints": [
// MANIFEST: "dimension": 1
// MANIFEST: "minimum": 32
// MANIFEST: "multiple_of": 32
// MANIFEST: "name": "output1"
// MANIFEST: "shape_source_input": 0

// LLVM-LABEL: llvm.func @dynamic_identity(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// LLVM: llvm.icmp "sge"
// LLVM: llvm.srem
// LLVM: llvm.call @__ncnn_internal_dynamic_identity(
// LLVM-LABEL: llvm.func @dynamic_identity_infer_output_shapes(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// LLVM: llvm.load
// LLVM: llvm.icmp "sge"
// LLVM: llvm.srem
// LLVM: llvm.store
// LLVM: llvm.icmp "sgt"
