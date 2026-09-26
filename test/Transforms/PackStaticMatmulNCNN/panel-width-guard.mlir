// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=pack-n=8' %s | FileCheck %s --check-prefix=N8
// RUN: ncnn-mlir-opt --pack-static-matmul-ncnn %s | FileCheck %s --check-prefix=N16

// P20-v1 使用固定 16-lane 物理面板：bufferized 内核按面板边界寻址，自定义
// 面板宽不能保证不跨面板。因此
//   * pack-n != 16              → packing_rejected_budget（f32 路径）；
//   * columns % 32 != 0         → packing_rejected_budget（面板/向量对齐）。
// 注意 int8 路径对 pack-n != 16 报的是 packing_rejected_layout，见
// int8-gates.mlir 的 @panel_width_guard。

func.func @custom_panel_width(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// N8-LABEL: func.func @custom_panel_width
// N8: linalg.matmul {ncnn.contract = "fallback"
// N8-SAME: ncnn.fallback_reason = "packing_rejected_budget"
// N8-SAME: ncnn.packing = "packing_rejected_budget"

// N16-LABEL: func.func @custom_panel_width
// N16: linalg.matmul
// N16-SAME: ncnn.pack_factor = 16
// N16-SAME: ncnn.packing = "prepacked_B"

// -----

// columns 不是 32 的倍数：面板/向量对齐不成立，同样落到 budget 拒绝。
func.func @unaligned_columns(%arg0: tensor<64x64xf32>) -> tensor<64x48xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x48xf32>
  %out = tensor.empty() : tensor<64x48xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x48xf32>)
                   outs(%out : tensor<64x48xf32>) -> tensor<64x48xf32>
  return %mm : tensor<64x48xf32>
}

// N16-LABEL: func.func @unaligned_columns
// N16: linalg.matmul {ncnn.contract = "fallback"
// N16-SAME: ncnn.fallback_reason = "packing_rejected_budget"
// N16-SAME: ncnn.packing = "packing_rejected_budget"
