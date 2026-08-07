// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_typed manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(%input1: memref<?x4xi32>,
                   %output1: memref<8xi64> {bufferize.result})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// MANIFEST: "dynamic_dim_mask": 1
// MANIFEST: "element_type": "i32"
// MANIFEST: "name": "input1"
// MANIFEST: -1
// MANIFEST: 4
// MANIFEST: "dynamic_dim_mask": 0
// MANIFEST: "element_type": "i64"
// MANIFEST: "name": "output1"
// MANIFEST: "shape_depends_on_data": false

// LLVM-LABEL: llvm.func @dynamic_typed(
// LLVM-SAME: %[[DATA:.*]]: !llvm.ptr, %[[SHAPE:.*]]: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr) -> i32
// LLVM: llvm.getelementptr %[[SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// LLVM: llvm.load
// LLVM: llvm.icmp "sgt"
// LLVM: llvm.icmp "eq"
// LLVM: llvm.cond_br
// LLVM: llvm.mul
// LLVM: llvm.call @__ncnn_internal_dynamic_typed(
