// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=attention target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module attributes {
  ncnn.attention_segment_revision = "attention-segment-v1",
  ncnn.attention_segments = [
    {id = "model/attention#0/score", function = "model",
     source_operation = "ncnn.multi_head_attention", attention_ordinal = 0 : i64,
     phase = "score", heads = 8 : i64, M = 40 : i64, K = 16 : i64,
     N = 40 : i64, kernel_status = "selected", transpose_count = 1 : i64,
     transpose_bytes = 20480 : i64, transpose_bytes_known = true,
     copy_bytes_known = false},
    {id = "model/attention#0/softmax", function = "model",
     source_operation = "ncnn.multi_head_attention", attention_ordinal = 0 : i64,
     phase = "softmax", heads = 8 : i64, M = 40 : i64, K = 40 : i64,
     kernel_status = "unknown", reason = "reduction_generic",
     transpose_count = 0 : i64, transpose_bytes_known = false,
     copy_bytes_known = false}
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
// CHECK: "phase": "score"
// CHECK: "transpose_bytes": 20480
// CHECK: "transpose_bytes_known": true
// CHECK: "kernel_status": "unknown"
// CHECK: "phase": "softmax"
// CHECK: "reason": "reduction_generic"
// CHECK: "transpose_bytes": null
// CHECK-DAG: "attention_segment_count": 2
// CHECK-DAG: "attention_selected_count": 1
// CHECK-DAG: "attention_fallback_count": 0
// CHECK-DAG: "attention_unknown_count": 1
