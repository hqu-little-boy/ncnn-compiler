// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=enabled=false int8-enabled=true max-total-bytes=1024' %s | FileCheck %s

module {
  memref.global constant @int8_weights : memref<16x36xi8> = dense<1>

  func.func @shares_p20_budget(%f32_input: tensor<1x1xf32>,
                               %f32_output: tensor<1x1xf32>,
                               %lhs: memref<4x36xi8>,
                               %out: memref<4x16xi32>) {
    %f32_rhs = arith.constant dense<1.0> : tensor<1x1xf32>
    %f32_mm = linalg.matmul {
      ncnn.pack_bytes = 1 : i64,
      ncnn.pack_schema = "p20-panel-nk-v1"
    } ins(%f32_input, %f32_rhs : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%f32_output : tensor<1x1xf32>) -> tensor<1x1xf32>
    %rhs = memref.get_global @int8_weights : memref<16x36xi8>
    linalg.matmul_transpose_b ins(%lhs, %rhs : memref<4x36xi8>, memref<16x36xi8>)
                              outs(%out : memref<4x16xi32>)
    return
  }
}

// CHECK-LABEL: func.func @shares_p20_budget
// CHECK: linalg.matmul_transpose_b {ncnn.contract = "fallback"
// CHECK-SAME: ncnn.fallback_reason = "packing_rejected_budget"
// CHECK-SAME: ncnn.packing = "packing_rejected_budget"
