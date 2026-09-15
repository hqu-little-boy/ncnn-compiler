// RUN: ncnn-mlir-opt --reuse-ncnn-workspace-slots --verify-bufferized-model %s | FileCheck %s

module {
  func.func @model(%input: memref<4xf32>, %output: memref<4xf32> {bufferize.result}) attributes {ncnn.entry_point} {
    %a = memref.alloc() : memref<4xf32>
    %c0 = arith.constant 0 : index
    %value = arith.constant 1.0 : f32
    memref.store %value, %a[%c0] : memref<4xf32>
    memref.dealloc %a : memref<4xf32>
    %b = memref.alloc() : memref<4xf32>
    memref.store %value, %b[%c0] : memref<4xf32>
    memref.dealloc %b : memref<4xf32>
    return
  }
}

// CHECK-LABEL: func.func @model
// CHECK: memref.alloc() {ncnn.workspace_reuse_count = 2 : i64, ncnn.workspace_reuse_status = "reused"
// CHECK-NOT: memref.alloc
// CHECK-COUNT-1: memref.dealloc
