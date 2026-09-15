// RUN: ncnn-mlir-opt --reuse-ncnn-workspace-slots %s | FileCheck %s

func.func @dynamic(%n : index) {
  %buffer = memref.alloc(%n) : memref<?xf32>
  memref.dealloc %buffer : memref<?xf32>
  return
}

func.func @alias() {
  %buffer = memref.alloc() : memref<4xf32>
  %view = memref.subview %buffer[0] [4] [1] : memref<4xf32> to memref<4xf32, strided<[1]>>
  memref.dealloc %view : memref<4xf32, strided<[1], offset: 0>>
  return
}

func.func @overlap() {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  %c0 = arith.constant 0 : index
  %value = arith.constant 1.0 : f32
  memref.store %value, %a[%c0] : memref<4xf32>
  memref.store %value, %b[%c0] : memref<4xf32>
  memref.dealloc %a : memref<4xf32>
  memref.dealloc %b : memref<4xf32>
  return
}

func.func @different_types() {
  %float = memref.alloc() : memref<4xf32>
  %c0 = arith.constant 0 : index
  %fvalue = arith.constant 1.0 : f32
  memref.store %fvalue, %float[%c0] : memref<4xf32>
  memref.dealloc %float : memref<4xf32>
  %byte = memref.alloc() : memref<4xi8>
  %ivalue = arith.constant 1 : i8
  memref.store %ivalue, %byte[%c0] : memref<4xi8>
  memref.dealloc %byte : memref<4xi8>
  return
}

func.func @consume(%buffer: memref<4xf32>) {
  return
}

func.func @call_boundary() {
  %buffer = memref.alloc() : memref<4xf32>
  func.call @consume(%buffer) : (memref<4xf32>) -> ()
  memref.dealloc %buffer : memref<4xf32>
  return
}

func.func @control_flow(%condition: i1) {
  %buffer = memref.alloc() : memref<4xf32>
  %c0 = arith.constant 0 : index
  %value = arith.constant 1.0 : f32
  scf.if %condition {
    memref.store %value, %buffer[%c0] : memref<4xf32>
  } else {
    memref.store %value, %buffer[%c0] : memref<4xf32>
  }
  memref.dealloc %buffer : memref<4xf32>
  return
}

func.func @non_identity_layout() {
  %buffer = memref.alloc() : memref<4xf32, strided<[2], offset: 0>>
  %c0 = arith.constant 0 : index
  %value = arith.constant 1.0 : f32
  memref.store %value, %buffer[%c0] : memref<4xf32, strided<[2], offset: 0>>
  memref.dealloc %buffer : memref<4xf32, strided<[2], offset: 0>>
  return
}

// CHECK-LABEL: func.func @dynamic
// CHECK: %{{.*}} = memref.alloc({{.*}}) {ncnn.workspace_fallback_reason = "dynamic_shape", ncnn.workspace_reuse_status = "fallback"} : memref<?xf32>
// CHECK-LABEL: func.func @alias
// CHECK: %{{.*}} = memref.alloc() {ncnn.workspace_fallback_reason = "alias_not_proven", ncnn.workspace_reuse_status = "fallback"} : memref<4xf32>
// CHECK-LABEL: func.func @overlap
// CHECK-COUNT-2: ncnn.workspace_reuse_status = "dedicated"
// CHECK-NOT: ncnn.workspace_reuse_status = "reused"
// CHECK-LABEL: func.func @different_types
// CHECK-COUNT-2: ncnn.workspace_reuse_status = "dedicated"
// CHECK-LABEL: func.func @call_boundary
// CHECK: %{{.*}} = memref.alloc() {ncnn.workspace_fallback_reason = "call_boundary", ncnn.workspace_reuse_status = "fallback"} : memref<4xf32>
// CHECK-LABEL: func.func @control_flow
// CHECK: %{{.*}} = memref.alloc() {ncnn.workspace_fallback_reason = "control_flow_or_region", ncnn.workspace_reuse_status = "fallback"} : memref<4xf32>
// CHECK-LABEL: func.func @non_identity_layout
// CHECK: %{{.*}} = memref.alloc() {ncnn.workspace_fallback_reason = "non_identity_layout", ncnn.workspace_reuse_status = "fallback"} : memref<4xf32, strided<[2]>>
