// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=static_plan target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module {
  func.func @model(%input: memref<4x8xf32>, %weights: memref<8x4xf32>) {
    %cst = arith.constant 1.0 : f32
    %vector = vector.splat %cst : vector<4xf32>
    %fma = vector.fma %vector, %vector, %vector : vector<4xf32>
    %output = memref.alloc() : memref<4x4xf32>
    %copy = memref.alloc() : memref<4x8xf32>
    linalg.matmul ins(%input, %weights : memref<4x8xf32>, memref<8x4xf32>) outs(%output : memref<4x4xf32>)
    memref.copy %input, %copy : memref<4x8xf32> to memref<4x8xf32>
    omp.parallel {
      omp.terminator
    }
    memref.dealloc %output : memref<4x4xf32>
    memref.dealloc %copy : memref<4x8xf32>
    return
  }
}

// CHECK: "kind": "ncnn.model_execution_plan"
// CHECK: "model": "static_plan"
// CHECK: "model/memref.alloc#0"
// CHECK: "model/memref.dealloc#0"
// CHECK: "allocation_count": 2
// CHECK: "copy_count": 1
// CHECK: "deallocation_count": 2
// CHECK: "dynamic_buffer_count": 0
// CHECK: "matmul_count": 1
// CHECK: "parallel_region_count": 1
// CHECK: "peak_workspace_bytes": null
// CHECK: "static_buffer_bytes": 192
// CHECK: "vector_fma_count": 1
// CHECK: "threads": 4
// CHECK: "triple": "x86_64-pc-linux-gnu"
// CHECK: "vector_lanes": 8
// CHECK: "vector_tail": true
