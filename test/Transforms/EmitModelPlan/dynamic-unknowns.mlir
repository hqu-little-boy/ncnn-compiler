// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=dynamic_plan target-triple=x86_64-pc-linux-gnu" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module {
  func.func @dynamic(%input: memref<?x4xf32>) {
    %four = arith.constant 4 : index
    %buffer = memref.alloc(%four) : memref<?x4xf32>
    memref.copy %input, %buffer : memref<?x4xf32> to memref<?x4xf32>
    memref.dealloc %buffer : memref<?x4xf32>
    return
  }
}

// CHECK: "bytes": null
// CHECK: "shape": [
// CHECK-NEXT: null,
// CHECK-NEXT: 4
// CHECK-NEXT: ],
// CHECK: "runtime_counters_not_collected"
// CHECK: "prepared_runner_not_supported"
// CHECK: "kind": "ncnn.model_execution_plan"
// CHECK: "model": "dynamic_plan"
// CHECK: "allocation_count": 1
// CHECK: "copy_count": 1
// CHECK: "dynamic_buffer_count": 2
// CHECK: "peak_workspace_bytes": null
// CHECK: "static_buffer_bytes": null
