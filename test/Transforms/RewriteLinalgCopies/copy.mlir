// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" %s | FileCheck %s --check-prefix=VECTOR
// RUN: ncnn-mlir-opt --pass-pipeline='builtin.module(rewrite-linalg-copies{vectorize=true vector-lanes=4},rewrite-linalg-copies{vectorize=true vector-lanes=4},emit-ncnn-execution-plan{path=%t.plan.json model=copy target-triple=x86_64-pc-linux-gnu threads=1 vector-lanes=4 vector-tail=true})' %s -o %t.plan.mlir
// RUN: FileCheck %s --check-prefix=PLAN --input-file=%t.plan.json

#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @copies() {
    %source = memref.alloc() : memref<2x10xf32>
    %target = memref.alloc() : memref<2x10xf32>
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%source : memref<2x10xf32>) outs(%target : memref<2x10xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.copy %target, %target : memref<2x10xf32> to memref<2x10xf32>
    linalg.copy ins(%target : memref<2x10xf32>) outs(%target : memref<2x10xf32>)
    memref.dealloc %source : memref<2x10xf32>
    memref.dealloc %target : memref<2x10xf32>
    return
  }

  func.func @explicit_copies() {
    %source = memref.alloc() : memref<2x10xf32>
    %target = memref.alloc() : memref<2x10xf32>
    linalg.copy ins(%source : memref<2x10xf32>) outs(%target : memref<2x10xf32>)
    memref.copy %source, %target : memref<2x10xf32> to memref<2x10xf32>
    memref.dealloc %source : memref<2x10xf32>
    memref.dealloc %target : memref<2x10xf32>
    return
  }

  func.func @dynamic_fallback(%source: memref<?x10xf32>, %target: memref<?x10xf32>) {
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%source : memref<?x10xf32>) outs(%target : memref<?x10xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @unsupported_cast_copy() {
    %source = memref.alloc() : memref<4xi32>
    %target = memref.alloc() : memref<4xi8>
    linalg.copy {cast = #linalg.type_fn<cast_signed>}
      ins(%source : memref<4xi32>) outs(%target : memref<4xi8>)
    return
  }
}

// VECTOR: module attributes
// VECTOR-SAME: ncnn.copy_eliminated_count = 2 : i64
// VECTOR-SAME: ncnn.copy_fallback_count = 1 : i64
// VECTOR-SAME: ncnn.copy_vectorized_count = 3 : i64
// VECTOR-LABEL: func.func @copies
// VECTOR: vector.transfer_read
// VECTOR: vector.transfer_write
// VECTOR: scf.for
// VECTOR: ncnn.copy_kind = "vectorized"
// VECTOR: ncnn.copy_vector_lanes = 4 : i64
// VECTOR-LABEL: func.func @explicit_copies
// VECTOR: vector.transfer_read
// VECTOR: vector.transfer_write
// VECTOR: ncnn.copy_kind = "vectorized"
// VECTOR-LABEL: func.func @dynamic_fallback
// VECTOR: memref.load
// VECTOR: memref.store
// VECTOR: ncnn.copy_kind = "fallback"
// VECTOR-NOT: linalg.generic
// VECTOR-LABEL: func.func @unsupported_cast_copy
// VECTOR: linalg.copy {cast = #linalg.type_fn<cast_signed>}

// PLAN-DAG: "copy_contract_count": 6
// PLAN-DAG: "copy_contract_status": "collected"
// PLAN-DAG: "copy_eliminated_count": 2
// PLAN-DAG: "copy_fallback_count": 1
// PLAN-DAG: "copy_vectorized_count": 3
// PLAN-DAG: "static_copy_count": 5
// PLAN-DAG: "runtime_bytes": null
// PLAN-DAG: "runtime_status": "not_collected"
