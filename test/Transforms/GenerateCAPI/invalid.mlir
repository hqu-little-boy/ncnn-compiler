// RUN: not ncnn-mlir-opt --generate-ncnn-c-api='export-name=bad-name' %s 2>&1 | FileCheck %s

module {
  func.func @model(%input: memref<2xf32>, %output: memref<2xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    return
  }
}

// CHECK: error: 'func.func' op requires export-name to be a valid C identifier
