// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --reuse-ncnn-workspace-slots --emit-ncnn-execution-plan="path=%t.plan.json model=workspace_slots target-triple=x86_64-pc-linux-gnu threads=1" %s -o %t.out
// RUN: ncnn-mlir-opt --reuse-ncnn-workspace-slots --emit-ncnn-execution-plan="path=%t.plan2.json model=workspace_slots target-triple=x86_64-pc-linux-gnu threads=1" %s -o %t.out
// RUN: cmp %t.plan.json %t.plan2.json
// RUN: FileCheck %s --input-file=%t.plan.json

module {
  func.func @model() {
    %a = memref.alloc() : memref<4xf32>
    %c0 = arith.constant 0 : index
    %value = arith.constant 1.0 : f32
    memref.store %value, %a[%c0] : memref<4xf32>
    %loaded = memref.load %a[%c0] : memref<4xf32>
    memref.dealloc %a : memref<4xf32>
    %b = memref.alloc() : memref<4xf32>
    memref.store %loaded, %b[%c0] : memref<4xf32>
    memref.dealloc %b : memref<4xf32>
    return
  }
}

// CHECK: "workspace_reuse_count": 2
// CHECK: "workspace_reuse_status": "reused"
// CHECK: "workspace_slot": 0
// CHECK: "workspace_slot_alignment": 0
// CHECK: "workspace_slot_bytes": 16
// CHECK: "workspace_slot_lifetime_begin": 0
// CHECK: "workspace_slot_lifetime_end": 8
// CHECK: "workspace_slot_owner": "model"
// CHECK: "workspace_slot_thread_visibility": "function_serial"
// CHECK: "contract_revision": "layout-kernel-v1|workspace-slot-v1|fusion-v2|copy-v1|attention-segment-v1|conv-depthwise-v1|packed-gemm-v1|layout-island-v1|int8-target-v1|tuning-v1|attribution-v2"
// CHECK: "plan_revision": "static-v1|workspace-slot-v1|fusion-v2|copy-v1|attention-segment-v1|conv-depthwise-v1|packed-gemm-v1|layout-island-v1|int8-target-v1|tuning-v1|attribution-v2"
// CHECK: "allocation_count": 1
// CHECK: "deallocation_count": 1
// CHECK: "workspace_fallback_count": 0
// CHECK: "workspace_reused_allocation_count": 1
// CHECK: "workspace_slot_bytes": 16
// CHECK: "workspace_slot_bytes_known": true
// CHECK: "workspace_slot_count": 1
