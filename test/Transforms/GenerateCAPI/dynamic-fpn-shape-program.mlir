// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_fpn manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json

module {
  func.func @model(%input: memref<3x?x?xf32>, %output: memref<1x?x?xf32> {bufferize.result, ncnn.shape_source_input = 0 : i32, ncnn.shape_program = [array<i64>, array<i64>, array<i64>]}) attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// MANIFEST: "dynamic_dim_mask": 6
// MANIFEST: "shape_program": [
// MANIFEST: "shape_source_input": 0
