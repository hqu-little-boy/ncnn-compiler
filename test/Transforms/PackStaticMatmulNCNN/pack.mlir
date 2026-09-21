// RUN: ncnn-mlir-opt --pack-static-matmul-ncnn %s | FileCheck %s
// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=enabled=false' %s | FileCheck %s --check-prefix=OFF
// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=max-total-bytes=4096' %s | FileCheck %s --check-prefix=BUDGET

func.func @packs_static_rhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @packs_static_rhs
// CHECK: %[[PACKED:.*]] = arith.constant {ncnn.alignment = "64"{{.*}}ncnn.packed_weight = true{{.*}}ncnn.weight_layout = "panel_nk"} dense<1.000000e+00> : tensor<64x64xf32>
// CHECK: linalg.matmul {ncnn.alignment = "64"{{.*}}ncnn.pack_bytes = 16384{{.*}}ncnn.pack_factor = 16{{.*}}ncnn.pack_raw_bytes = 16384{{.*}}ncnn.pack_runtime = "compile_time_B"{{.*}}ncnn.pack_schema = "p20-panel-nk-v1"{{.*}}ncnn.packing = "prepacked_B"{{.*}}ncnn.weight_layout = "panel_nk"} ins(%arg0, %[[PACKED]]

// OFF-LABEL: func.func @packs_static_rhs
// OFF: linalg.matmul {ncnn.contract = "fallback"{{.*}}ncnn.fallback_reason = "unpacked_direct"{{.*}}ncnn.packing = "unpacked_direct"{{.*}}} ins(%arg0, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>)

// -----

func.func @rejects_dynamic_rhs(%arg0: tensor<?x64xf32>, %rhs: tensor<64x64xf32>) -> tensor<?x64xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x64xf32>
  %out = tensor.empty(%dim) : tensor<?x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<?x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %mm : tensor<?x64xf32>
}

// CHECK-LABEL: func.func @rejects_dynamic_rhs
// CHECK: linalg.matmul {ncnn.contract = "fallback", ncnn.fallback_reason = "packing_rejected_dynamic", ncnn.pack_bytes = 0{{.*}}ncnn.packing = "packing_rejected_dynamic"

// BUDGET-LABEL: func.func @packs_static_rhs
// BUDGET: linalg.matmul {ncnn.contract = "fallback"
// BUDGET-SAME: ncnn.fallback_reason = "packing_rejected_budget"
// BUDGET-SAME: ncnn.packing = "packing_rejected_budget"

// -----

func.func @skips_untileable_single_row(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<1x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<1x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %mm : tensor<1x64xf32>
}

// CHECK-LABEL: func.func @skips_untileable_single_row
// CHECK: linalg.matmul {ncnn.contract = "fallback"{{.*}}ncnn.fallback_reason = "packing_skipped_small_shape"{{.*}}ncnn.packing = "packing_skipped_small_shape"

// -----

func.func @skips_untileable_extent(%arg0: tensor<2x64xf32>) -> tensor<2x32xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<2x64xf32>, tensor<64x32xf32>)
                   outs(%out : tensor<2x32xf32>) -> tensor<2x32xf32>
  return %mm : tensor<2x32xf32>
}

func.func @skips_untileable_prime(%arg0: tensor<37x64xf32>) -> tensor<37x32xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x32xf32>
  %out = tensor.empty() : tensor<37x32xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<37x64xf32>, tensor<64x32xf32>)
                   outs(%out : tensor<37x32xf32>) -> tensor<37x32xf32>
  return %mm : tensor<37x32xf32>
}

// CHECK-LABEL: func.func @skips_untileable_extent
// CHECK: linalg.matmul {ncnn.contract = "fallback"{{.*}}ncnn.fallback_reason = "packing_skipped_small_shape"{{.*}}ncnn.packing = "packing_skipped_small_shape"
// CHECK-LABEL: func.func @skips_untileable_prime
// CHECK: linalg.matmul {ncnn.contract = "fallback"{{.*}}ncnn.fallback_reason = "packing_skipped_small_shape"{{.*}}ncnn.packing = "packing_skipped_small_shape"

// -----

func.func @rejects_unsupported_consumer(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul {ncnn.fallback_reason = "unsupported_consumer"}
                   ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @rejects_unsupported_consumer
// CHECK: linalg.matmul {ncnn.contract = "fallback"{{.*}}ncnn.fallback_reason = "packing_rejected_layout"{{.*}}ncnn.packing = "packing_rejected_layout"
