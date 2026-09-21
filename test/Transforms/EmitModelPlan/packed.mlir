// RUN: rm -f %t.json
// RUN: ncnn-mlir-opt '--ncnn-linalg-to-memref-pipeline=vector-tail=true matmul-packing=auto execution-plan-path=%t.json execution-plan-model=packed-test' %s -o /dev/null
// RUN: FileCheck %s --input-file=%t.json --check-prefix=PLAN

module {
  func.func @packed_plan(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
    %out = tensor.empty() : tensor<64x64xf32>
    %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }
}

// PLAN-DAG: "packed_constant_count": 1
// PLAN-DAG: "packed_constant_bytes": 8192
// PLAN-DAG: "packed_constant_bytes_known": true
// PLAN-DAG: "packed_buffer_bytes": 8192
// PLAN-DAG: "static_pack_bytes": 8192
// PLAN-DAG: "kernel": "f32_packed_mxn_fma"
// PLAN-DAG: "pack_runtime": "compile_time_B"
// PLAN-DAG: "runtime_counters_not_collected"
