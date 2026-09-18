// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=int8_audit target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module attributes {
  ncnn.int8_target = "avx-vnni",
  ncnn.int8_kernel = "auto",
  ncnn.int8_depthwise = true,
  ncnn.int8_cast_chain = false
} {
  func.func @int8_audit(%input: memref<4x8xi8>, %weights: memref<8x4xi8>) {
    %output = memref.alloc() : memref<4x4xi32>
    "omp.parallel"() ({
      ^bb0:
        omp.terminator
    }) {
      ncnn.contract = "selected",
      ncnn.kernel = "int8_vnni_row_dot",
      ncnn.int8_isa = "avx-vnni",
      ncnn.tile_m = 4 : i64,
      ncnn.tile_n = 4 : i64,
      ncnn.tile_k = 67 : i64,
      ncnn.tail = "none",
      ncnn.fma = "vpdpbusd_signed_correction"
    } : () -> ()
    memref.dealloc %output : memref<4x4xi32>
    return
  }
}

// CHECK-DAG: "low_precision": {
// CHECK-DAG: "requested_policy": "auto"
// CHECK-DAG: "requested_policy_fallback_reason": null
// CHECK-DAG: "requested_policy_status": "pending_defaultization"
// CHECK-DAG: "capability": "avx-vnni"
// CHECK-DAG: "depthwise_enabled": true
// CHECK-DAG: "cast_chain_enabled": false
// CHECK-DAG: "operations": [
// CHECK-DAG: "id": "int8_audit/omp.parallel#0"
// CHECK-DAG: "kernel": "int8_vnni_row_dot"
// CHECK-DAG: "required_isa": "avx-vnni"
// CHECK-DAG: "signedness": "signed8*signed8"
// CHECK-DAG: "signedness_correction": "xor_0x80_subtract_128_times_rhs_sum"
// CHECK-DAG: "accumulator": "modulo32"
// CHECK-DAG: "reduction_elements": 3
// CHECK-DAG: "output_policy": "none"
// CHECK-DAG: "fallback_reason": null

// CHECK-NOT: "runtime_low_precision"
