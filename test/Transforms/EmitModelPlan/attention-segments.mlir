// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=attention target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module attributes {
  ncnn.attention_segment_revision = "attention-segment-v1",
  ncnn.attention_segments = [
    {id = "model/attention#0/score", function = "model",
     source_operation = "ncnn.multi_head_attention", attention_ordinal = 0 : i64,
     phase = "score", heads = 8 : i64, M = 40 : i64, K = 16 : i64,
     N = 40 : i64, kernel_status = "selected", transpose_count = 0 : i64,
     transpose_bytes = 0 : i64, transpose_bytes_known = true,
     copy_bytes_known = false, layout = "sequence_major",
     layout_producer = "projection_reshape",
     layout_consumer = "indexed_attention_contraction",
     parallel_policy = "head_query_key",
     tile_policy = "shape_driven_linalg_parallel",
     transpose_elided_reason = "direct_indexed_attention"},
    {id = "model/attention#0/softmax", function = "model",
     source_operation = "ncnn.multi_head_attention", attention_ordinal = 0 : i64,
     phase = "softmax", heads = 8 : i64, M = 40 : i64, K = 40 : i64,
     kernel_status = "selected", reason = "stable_two_pass",
     softmax_strategy = "stable_two_pass", transpose_count = 0 : i64,
     transpose_bytes_known = false, copy_bytes_known = false}
  ]
} {
  func.func @model(%input: memref<4x8xf32>) {
    %output = memref.alloc() : memref<4x8xf32>
    memref.dealloc %output : memref<4x8xf32>
    return
  }
}

// CHECK: "attention_revision": "attention-segment-v1"
// CHECK: "attention_segments": [
// CHECK-DAG: "copy_count": null
// CHECK: "id": "model/attention#0/score"
// CHECK: "kernel_status": "selected"
// CHECK: "layout": "sequence_major"
// CHECK: "layout_consumer": "indexed_attention_contraction"
// CHECK: "layout_producer": "projection_reshape"
// CHECK: "parallel_policy": "head_query_key"
// CHECK: "phase": "score"
// CHECK: "tile_policy": "shape_driven_linalg_parallel"
// CHECK: "transpose_bytes": 0
// CHECK: "transpose_bytes_known": true
// CHECK: "transpose_elided_reason": "direct_indexed_attention"
// CHECK: "kernel_status": "selected"
// CHECK: "phase": "softmax"
// CHECK: "reason": "stable_two_pass"
// CHECK: "softmax_strategy": "stable_two_pass"
// CHECK: "transpose_bytes": null
// CHECK-DAG: "attention_segment_count": 2
// CHECK-DAG: "attention_selected_count": 2
// CHECK-DAG: "attention_fallback_count": 0
// CHECK-DAG: "attention_unknown_count": 0
