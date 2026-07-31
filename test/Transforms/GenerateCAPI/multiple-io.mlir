// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=multi_io manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=PREPARED < %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(%output1: memref<2xf32> {bufferize.result},
                   %input1: memref<2x3xf32>,
                   %output2: memref<4xf32> {bufferize.result},
                   %input2: memref<5xf32>) attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// PREPARED-LABEL: func.func private @__ncnn_internal_multi_io(
// PREPARED-NOT: llvm.emit_c_interface
// PREPARED-NOT: ncnn.entry_point

// MANIFEST: "function": "multi_io"
// MANIFEST: "name": "input1"
// MANIFEST: 2
// MANIFEST: 3
// MANIFEST: "name": "input2"
// MANIFEST: 5
// MANIFEST: "name": "output1"
// MANIFEST: 2
// MANIFEST: "name": "output2"
// MANIFEST: 4

// LLVM-LABEL: llvm.func @multi_io(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// LLVM: llvm.call @__ncnn_internal_multi_io(
// LLVM: llvm.func @__ncnn_internal_multi_io(
// LLVM-SAME: attributes {sym_visibility = "private"}
// LLVM-NOT: _mlir_ciface
