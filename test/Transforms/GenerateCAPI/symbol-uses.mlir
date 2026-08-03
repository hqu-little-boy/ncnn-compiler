// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=exported' %s | FileCheck %s

module {
  func.func @model(%input: memref<2xf32>,
                   %output: memref<2xf32> {bufferize.result}) attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }

  func.func @helper(%input: memref<2xf32>,
                    %output: memref<2xf32>) attributes {ncnn.target = @model} {
    call @model(%input, %output) : (memref<2xf32>, memref<2xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @__ncnn_internal_exported(
// CHECK-LABEL: func.func @helper(
// CHECK-SAME: attributes {ncnn.target = @__ncnn_internal_exported}
// CHECK: call @__ncnn_internal_exported(
// CHECK-NOT: call @model(
