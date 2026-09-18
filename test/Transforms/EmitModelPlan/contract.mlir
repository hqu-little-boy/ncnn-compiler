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
      ncnn.operation_family = "conv",
      ncnn.implementation = "gather_free",
      ncnn.kernel_static = true,
      ncnn.kernel_h = 3 : i64,
      ncnn.kernel_w = 3 : i64,
      ncnn.stride_h = 1 : i64,
      ncnn.stride_w = 1 : i64,
      ncnn.dilation_h = 1 : i64,
      ncnn.dilation_w = 1 : i64,
      ncnn.input_channels = 32 : i64,
      ncnn.output_channels = 16 : i64,
      ncnn.source_layer = 7 : i64,
      ncnn.name = "conv7",
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

// CHECK-DAG: "low_precision": {
// CHECK-DAG: "revision": "int8-target-v1"
// CHECK-DAG: "requested_policy": "unknown"
// CHECK-DAG: "requested_policy_status": "unknown"
// CHECK-DAG: "capability": "unknown"
// CHECK-DAG: "depthwise_enabled": false
// CHECK-DAG: "cast_chain_enabled": false
// CHECK-DAG: "tuning": {
// CHECK-DAG: "revision": "tuning-v1"
// CHECK-DAG: "profile": "stable"
// CHECK-DAG: "status": "stable"
// CHECK-DAG: "matmul_m_rows": 4
// CHECK-DAG: "matmul_acc_columns": 16
// CHECK-DAG: "row_chunk_lanes": 8
// CHECK-DAG: "matmul_i8_rows": 2
// CHECK-DAG: "matmul_i8_acc_columns": 4
// CHECK-DAG: "operations": [
// CHECK-DAG: "contract_revision": "layout-kernel-v1|workspace-slot-v1|fusion-v1|attention-segment-v1|conv-depthwise-v1|int8-target-v1|tuning-v1"
// CHECK-DAG: "plan_revision": "static-v1|workspace-slot-v1|fusion-v1|attention-segment-v1|conv-depthwise-v1|int8-target-v1|tuning-v1"
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
// CHECK-DAG: "conv_depthwise_operations": [
// CHECK-DAG: "family": "conv"
// CHECK-DAG: "implementation": "gather_free"
// CHECK-DAG: "source_layer": 7
// CHECK-DAG: "conv_depthwise": {
// CHECK-DAG: "conv_operation_count": 1
// CHECK-DAG: "conv_fallback_count": 0
// CHECK-DAG: "gather_free": 1
// CHECK-DAG: "nested_openmp_count": 0
// CHECK-DAG: "packed_buffer_bytes": 0
