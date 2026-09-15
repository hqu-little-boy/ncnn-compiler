// RUN: ncnn-mlir-opt --reuse-ncnn-workspace-slots %s | FileCheck %s

func.func @reuse() {
  %a = memref.alloc() {alignment = 16 : i64} : memref<4xf32>
  %ca = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  memref.store %ca, %a[%c0] : memref<4xf32>
  %va = memref.load %a[%c0] : memref<4xf32>
  memref.dealloc %a : memref<4xf32>
  %b = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
  memref.store %va, %b[%c0] : memref<4xf32>
  %vb = memref.load %b[%c0] : memref<4xf32>
  memref.dealloc %b : memref<4xf32>
  return
}

// CHECK-LABEL: func.func @reuse
// CHECK: %[[SLOT:.*]] = memref.alloc() {alignment = 64 : i64, ncnn.workspace_reuse_count = 2 : i64, ncnn.workspace_reuse_status = "reused", ncnn.workspace_slot = 0 : i64, ncnn.workspace_slot_alignment = 64 : i64, ncnn.workspace_slot_bytes = 16 : i64, ncnn.workspace_slot_lifetime_begin = 0 : i64, ncnn.workspace_slot_lifetime_end = 9 : i64, ncnn.workspace_slot_owner = "reuse", ncnn.workspace_slot_thread_visibility = "function_serial"} : memref<4xf32>
// CHECK-NOT: memref.alloc
// CHECK: memref.store %{{.*}}, %[[SLOT]][%{{.*}}] : memref<4xf32>
// CHECK: memref.load %[[SLOT]][%{{.*}}] : memref<4xf32>
// CHECK-NOT: memref.dealloc
// CHECK: memref.dealloc %[[SLOT]] : memref<4xf32>
