// RUN: ncnn-mlir-opt --pack-static-matmul-ncnn %s | FileCheck %s

// Packing resolves the RHS constant through the same view-aware lookup the
// batchnorm folder uses, so a constant wrapped in tensor.cast views must still
// pack — the shared helper has to keep this capability that the old local
// findConstant had.

func.func @packs_through_cast_views(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %rhs_dyn = tensor.cast %rhs : tensor<64x64xf32> to tensor<?x?xf32>
  %rhs_view = tensor.cast %rhs_dyn : tensor<?x?xf32> to tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs_view : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @packs_through_cast_views
// CHECK: %[[PACKED:.*]] = arith.constant {ncnn.alignment = "64"{{.*}}ncnn.packed_weight = true{{.*}}ncnn.weight_layout = "panel_nk"} dense<1.000000e+00> : tensor<64x64xf32>
// CHECK: linalg.matmul {ncnn.alignment = "64"{{.*}}ncnn.contract = "selected"{{.*}}ncnn.packing = "prepacked_B"{{.*}}ins(%arg0, %[[PACKED]]

// -----

// The two rejection reasons are distinct plan fields and must not collapse: a
// non-constant RHS reports packing_rejected_dynamic, while a constant of the
// wrong shape or element type reports packing_rejected_layout.

func.func @rejects_non_constant_rhs(%arg0: tensor<64x64xf32>, %rhs: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @rejects_non_constant_rhs
// CHECK: linalg.matmul {ncnn.contract = "fallback"{{.*}}ncnn.fallback_reason = "packing_rejected_dynamic"
