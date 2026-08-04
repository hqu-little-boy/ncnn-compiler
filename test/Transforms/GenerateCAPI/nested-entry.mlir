// RUN: not ncnn-mlir-opt --generate-ncnn-c-api='export-name=exported' %s 2>&1 | FileCheck %s

module {
  module @nested {
    func.func @model(%input: memref<2xf32>, %output: memref<2xf32> {bufferize.result}) attributes {ncnn.entry_point} {
      return
    }
  }
}

// CHECK: error: 'func.func' op must be a top-level function in the pass module
