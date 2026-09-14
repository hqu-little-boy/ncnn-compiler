// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=call_escape target-triple=x86_64-pc-linux-gnu threads=1" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module {
  func.func private @retain(%arg: memref<4x4xf32>)
  func.func @escape(%input: memref<4x4xf32>) {
    %buffer = memref.alloc() : memref<4x4xf32>
    func.call @retain(%buffer) : (memref<4x4xf32>) -> ()
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }
}

// CHECK: "liveness_status": "unknown"
// CHECK: "buffer_liveness_unknown"
// CHECK: "kind": "ncnn.model_execution_plan"
// CHECK: "model": "call_escape"
// CHECK: "peak_workspace_bytes": null
// CHECK: "peak_workspace_proven": false
