// RUN: not ncnn-mlir-opt --split-input-file --verify-bufferized-model %s 2>&1 | FileCheck %s

module {
  func.func @tensor_result(%input: tensor<4xf32>) -> tensor<4xf32> attributes {ncnn.entry_point} {
    return %input : tensor<4xf32>
  }
}

// CHECK: error: 'func.func' op has a tensor-typed argument after bufferization
// CHECK: error: 'func.func' op has a tensor-typed function result after bufferization
// CHECK: error: 'func.func' op must not return values after bufferization
// CHECK: error: 'func.func' op has no bufferize.result output parameter

// -----

module {
  func.func @missing_output(%input: memref<4xf32>) attributes {ncnn.entry_point} {
    return
  }
}

// CHECK: error: 'func.func' op has no bufferize.result output parameter

// -----

module {
  func.func @releases_output(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    memref.dealloc %output : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op must not release a caller-owned function argument
