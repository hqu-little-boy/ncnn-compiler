// RUN: not ncnn-mlir-opt --mlir-print-ir-after-failure --generate-ncnn-c-api='export-name=exported manifest-path=/dev/null/manifest.json' %s 2>&1 | FileCheck %s
// RUN: rm -f %t.full
// RUN: ln -s /dev/full %t.full
// RUN: not ncnn-mlir-opt --mlir-print-ir-after-failure --generate-ncnn-c-api='export-name=exported manifest-path=%t.full' %s 2>&1 | FileCheck %s --check-prefix=WRITE
// RUN: rm -f %t.full
// UNSUPPORTED: system-windows

module {
  func.func @model(%input: memref<2xf32>, %output: memref<2xf32> {bufferize.result}) attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// CHECK: error: cannot open ABI manifest '/dev/null/manifest.json'
// CHECK: IR Dump After {{.*}}GenerateCAPIPass Failed (generate-ncnn-c-api)
// CHECK: func.func @model
// CHECK-SAME: attributes {llvm.emit_c_interface, ncnn.entry_point}
// CHECK-NOT: __ncnn_internal_exported

// WRITE: error: cannot write ABI manifest '{{.*}}.full': No space left on device
// WRITE: IR Dump After {{.*}}GenerateCAPIPass Failed (generate-ncnn-c-api)
// WRITE: func.func @model
// WRITE-SAME: attributes {llvm.emit_c_interface, ncnn.entry_point}
// WRITE-NOT: __ncnn_internal_exported
