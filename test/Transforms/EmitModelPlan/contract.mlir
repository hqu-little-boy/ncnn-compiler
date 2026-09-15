// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=contract target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module {
  func.func @model(%input: memref<4x8xf32>) {
    %output = memref.alloc() : memref<4x8xf32>
    "omp.parallel"() ({
      ^bb0:
        omp.terminator
    }) {
      ncnn.contract = "selected",
      ncnn.kernel = "f32_mxn_fma",
      ncnn.input_layout = "identity",
      ncnn.weight_layout = "row_major_kxn",
      ncnn.output_layout = "identity",
      ncnn.packing = "unpacked",
      ncnn.pack_factor = 1 : i64,
      ncnn.pack_bytes = 0 : i64,
      ncnn.unpack_bytes = 0 : i64,
      ncnn.tile_m = 4 : i64,
      ncnn.tile_n = 8 : i64,
      ncnn.tile_k = 8 : i64,
      ncnn.parallel = "outer_tile+inner_simd",
      ncnn.simd_lanes = 8 : i64,
      ncnn.simd_chunk = 8 : i64,
      ncnn.fma = "vector.fma",
      ncnn.tail = "none",
      ncnn.alignment = "unknown",
      ncnn.alias = "disjoint_output_proven"
    } : () -> ()
    memref.dealloc %output : memref<4x8xf32>
    return
  }
}

// CHECK: "contract_revision": "layout-kernel-v1|workspace-slot-v1"
// CHECK-DAG: "kernel_contract": {
// CHECK-DAG: "kernel": "f32_mxn_fma"
// CHECK-DAG: "parallel": "outer_tile+inner_simd"
// CHECK-DAG: "simd_lanes": 8
// CHECK-DAG: "tile_m": 4
// CHECK-DAG: "contracts": [
// CHECK-DAG: "kernel_contract_count": 1
// CHECK-DAG: "kernel_contract_fallback_count": 0
// CHECK-DAG: "nested_openmp_count": 0
// CHECK-DAG: "packed_buffer_bytes": 0
