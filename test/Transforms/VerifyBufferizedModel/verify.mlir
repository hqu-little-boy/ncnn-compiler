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
  func.func @residual_bufferization(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %clone = bufferization.clone %output : memref<4xf32> to memref<4xf32>
    memref.dealloc %clone : memref<4xf32>
    return
  }
}

// CHECK: error: 'bufferization.clone' op belongs to forbidden residual dialect 'bufferization'

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

// CHECK: error: 'memref.dealloc' op must not release a caller-owned function argument or its alias

// -----

module {
  func.func @releases_output_subview(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %alias = memref.subview %output[0] [4] [1] : memref<4xf32> to memref<4xf32, strided<[1]>>
    memref.dealloc %alias : memref<4xf32, strided<[1]>>
    return
  }
}

// CHECK: error: 'memref.dealloc' op must not release a caller-owned function argument or its alias

// -----

module {
  func.func private @may_release_argument(%condition: i1, %argument: memref<4xf32>) {
    %allocation = memref.alloc() : memref<4xf32>
    %selected = arith.select %condition, %argument, %allocation : memref<4xf32>
    memref.dealloc %selected : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op must not release a caller-owned function argument or its alias
// CHECK: error: 'memref.alloc' op has no matching deallocation

// -----

module {
  func.func private @identity(%input: memref<4xf32>) -> memref<4xf32> {
    return %input : memref<4xf32>
  }

  func.func private @unproven_call_alias() {
    %allocation = memref.alloc() : memref<4xf32>
    %alias = call @identity(%allocation) : (memref<4xf32>) -> memref<4xf32>
    memref.dealloc %alias : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op does not resolve to a unique heap allocation root
// CHECK: error: 'memref.alloc' op has no matching deallocation

// -----

module {
  func.func private @reallocation_use_after_free() {
    %allocation = memref.alloc() : memref<4xf32>
    %resized = memref.realloc %allocation : memref<4xf32> to memref<8xf32>
    %index = arith.constant 0 : index
    %value = memref.load %allocation[%index] : memref<4xf32>
    memref.dealloc %resized : memref<8xf32>
    return
  }
}

// CHECK: error: 'memref.load' op may execute after or without the allocation deallocation

// -----

module {
  func.func @releases_branch_carried_output(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    cf.br ^release(%output : memref<4xf32>)
  ^release(%alias: memref<4xf32>):
    memref.dealloc %alias : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op must not release a caller-owned function argument or its alias

// -----

module {
  func.func private @private_leak() {
    %allocation = memref.alloc() : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.alloc' op has no matching deallocation

// -----

module {
  func.func @alias_double_free(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %allocation = memref.alloc() : memref<4xf32>
    %alias = memref.cast %allocation : memref<4xf32> to memref<?xf32>
    memref.dealloc %allocation : memref<4xf32>
    memref.dealloc %alias : memref<?xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op has a matching deallocation that may execute after it; expected exactly one to avoid double-free

// -----

module {
  func.func @use_after_free(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %allocation = memref.alloc() : memref<4xf32>
    %alias = memref.cast %allocation : memref<4xf32> to memref<?xf32>
    memref.dealloc %allocation : memref<4xf32>
    %index = arith.constant 0 : index
    %value = memref.load %alias[%index] : memref<?xf32>
    return
  }
}

// CHECK: error: 'memref.load' op may execute after or without the allocation deallocation

// -----

module {
  func.func @control_flow_double_free(%condition: i1, %output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %allocation = memref.alloc() : memref<4xf32>
    cf.cond_br %condition, ^free, ^exit
  ^free:
    memref.dealloc %allocation : memref<4xf32>
    cf.br ^exit
  ^exit:
    memref.dealloc %allocation : memref<4xf32>
    return
  }
}

// -----

module {
  func.func @leaks(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %allocation = memref.alloc() : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.alloc' op has no matching deallocation

// -----

module {
  func.func @double_free(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %allocation = memref.alloc() : memref<4xf32>
    memref.dealloc %allocation : memref<4xf32>
    memref.dealloc %allocation : memref<4xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op has a matching deallocation that may execute after it; expected exactly one to avoid double-free

// -----

module {
  func.func @releases_output_alias(%output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %alias = memref.cast %output : memref<4xf32> to memref<?xf32>
    memref.dealloc %alias : memref<?xf32>
    return
  }
}

// CHECK: error: 'memref.dealloc' op must not release a caller-owned function argument or its alias
