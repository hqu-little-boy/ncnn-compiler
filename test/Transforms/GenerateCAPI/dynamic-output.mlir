// RUN: not ncnn-mlir-opt --generate-ncnn-c-api='export-name=model' %s 2>&1 | FileCheck %s

module {
  func.func @model(%input: memref<4xf32>,
                   %output: memref<?xf32> {bufferize.result})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// CHECK: output 1 has dynamic extents; output shape inference is required before this ABI can be generated
