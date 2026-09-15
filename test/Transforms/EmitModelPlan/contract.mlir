// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=contract target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module attributes {
  ncnn.fusion_enabled = true,
  ncnn.fusion_selected_count = 1 : i64,
  ncnn.fusion_residual_count = 1 : i64,
  ncnn.fusion_rejected_count = 2 : i64,
  ncnn.fusion_rejection_reasons = "multi_consumer=1,non_identity_map=1"
} {
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
      ncnn.alias = "disjoint_output_proven",
      ncnn.fusion = "selected",
      ncnn.fusion_kind = "matmul_epilogue",
      ncnn.fusion_producer = "linalg.matmul",
      ncnn.fusion_residual_inputs = 1 : i64,
      ncnn.fusion_tile_width = 16 : i64,
      ncnn.fusion_intermediate_bytes = 1024 : i64,
      ncnn.fusion_saved_bytes = 1024 : i64
    } : () -> ()
    memref.dealloc %output : memref<4x8xf32>
    return
  }
}

// CHECK: "contract_revision": "layout-kernel-v1|workspace-slot-v1|fusion-v1"
// CHECK-DAG: "fusion": {
// CHECK-DAG: "enabled": true
// CHECK-DAG: "selected_count": 1
// CHECK-DAG: "residual_count": 1
// CHECK-DAG: "rejected_count": 2
// CHECK-DAG: "rejection_reasons": "multi_consumer=1,non_identity_map=1"
// CHECK-DAG: "fusions": [
// CHECK-DAG: "fusion_kind": "matmul_epilogue"
// CHECK-DAG: "fusion_residual_inputs": 1
// CHECK-DAG: "fusion_intermediate_bytes": 1024
// CHECK-DAG: "fusion_saved_bytes": 1024
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
