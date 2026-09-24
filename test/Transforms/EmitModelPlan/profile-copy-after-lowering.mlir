// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" --instrument-ncnn-profile %s | FileCheck %s

module {
  func.func @model() {
    %source = memref.alloc() : memref<10xf32>
    %target = memref.alloc() : memref<10xf32>
    memref.copy %source, %target : memref<10xf32> to memref<10xf32>
    memref.dealloc %source : memref<10xf32>
    memref.dealloc %target : memref<10xf32>
    return
  }
}

// CHECK-LABEL: func.func @model
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: scf.for
// CHECK: call @__ncnn_profile_copy
// CHECK-NOT: memref.copy
