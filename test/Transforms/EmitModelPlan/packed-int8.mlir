// RUN: rm -f %t.json
// RUN: ncnn-mlir-opt '--ncnn-linalg-to-memref-pipeline=vector-tail=true int8-kernel=vnni int8-target=avx-vnni tuning-profile=native-int8 execution-plan-path=%t.json execution-plan-model=packed-int8-test' %s -o /dev/null
// RUN: FileCheck %s --input-file=%t.json --check-prefix=PLAN

module {
  func.func @packed_int8_plan(%lhs: tensor<64x36xi8>) -> tensor<64x64xi32> {
    %rhs = arith.constant dense<1> : tensor<64x36xi8>
    %out = tensor.empty() : tensor<64x64xi32>
    %mm = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<64x36xi8>, tensor<64x36xi8>)
                                    outs(%out : tensor<64x64xi32>) -> tensor<64x64xi32>
    return %mm : tensor<64x64xi32>
  }
}

// PLAN-DAG: "packed_constant_count": 1
// PLAN-DAG: "packed_constant_bytes": 4096
// PLAN-DAG: "packed_constant_bytes_known": true
// PLAN-DAG: "static_pack_bytes": 4096
// PLAN-DAG: "packing": "prepacked_B"
// PLAN-DAG: "pack_schema": "p23-int8-panel-row-kpad64-v1"
// PLAN-DAG: "pack_raw_bytes": 2304
// PLAN-DAG: "pack_bytes": 4096
// PLAN-DAG: "pack_runtime": "compile_time_B"
// PLAN-DAG: "required_isa": "avx-vnni"
// PLAN-DAG: "emitted_intrinsic": "llvm.x86.avx512.vpdpbusd.256"
// PLAN-DAG: "k_alignment": 64
// PLAN-DAG: "reduction_tail": 4
