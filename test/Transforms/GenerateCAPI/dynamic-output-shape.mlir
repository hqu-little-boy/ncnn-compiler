// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_identity manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(
      %input: memref<3x?x?xf32>,
      %output: memref<3x?x?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>, array<i64>, array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    memref.copy %input, %output : memref<3x?x?xf32> to memref<3x?x?xf32>
    return
  }
}

// MANIFEST: "name": "output1"
// MANIFEST: "shape_source_input": 0

// LLVM-LABEL: llvm.func @dynamic_identity(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// LLVM: llvm.call @__ncnn_internal_dynamic_identity(
// LLVM-LABEL: llvm.func @dynamic_identity_infer_output_shapes(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// LLVM: llvm.load
// LLVM: llvm.store
// LLVM: llvm.icmp "sgt"
