// RUN: not ncnn-mlir-opt --split-input-file --generate-ncnn-c-api='export-name=exported' %s 2>&1 | FileCheck %s

module {
  func.func private @exported()
  func.func @model(%input: memref<2xf32>, %output: memref<2xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    return
  }
}

// CHECK: error: 'func.func' op cannot export duplicate symbol 'exported'

// -----

module {
  func.func private @__ncnn_internal_exported()
  func.func @model(%input: memref<2xf32>, %output: memref<2xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    return
  }
}

// CHECK: error: 'func.func' op cannot create duplicate internal symbol '__ncnn_internal_exported'
