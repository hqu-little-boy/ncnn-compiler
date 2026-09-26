// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=min-n=128 min-k=128' %s | FileCheck %s --check-prefix=MIN
// RUN: ncnn-mlir-opt --pack-static-matmul-ncnn %s | FileCheck %s --check-prefix=DEF

// 预打包下限：N < minN 或 K < minK 时不做物理重排（收益不抵搬运），落到
// packing_skipped_small_shape；单行 GEMM（rows < 2）下游 forall 切不动，
// 保持行主序以免未内核化的 fallback 吃到面板布局。

func.func @below_min_n(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// MIN-LABEL: func.func @below_min_n
// MIN: linalg.matmul {ncnn.contract = "fallback"
// MIN-SAME: ncnn.fallback_reason = "packing_skipped_small_shape"
// MIN-SAME: ncnn.packing = "packing_skipped_small_shape"

// DEF-LABEL: func.func @below_min_n
// DEF: ncnn.packing = "prepacked_B"

// -----

func.func @single_row(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<1x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<1x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %mm : tensor<1x64xf32>
}

// MIN-LABEL: func.func @single_row
// MIN: ncnn.fallback_reason = "packing_skipped_small_shape"

// DEF-LABEL: func.func @single_row
// DEF: ncnn.fallback_reason = "packing_skipped_small_shape"

// -----

// 两维都无可行 tile 因子时下游 forall 保持顶层，物理面板会被通用行主序
// 下降吃错布局，同样跳过。
func.func @no_tile_divisor(%arg0: tensor<2x64xf32>) -> tensor<2x32xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<2x64xf32>, tensor<64x32xf32>)
                   outs(%out : tensor<2x32xf32>) -> tensor<2x32xf32>
  return %mm : tensor<2x32xf32>
}

// DEF-LABEL: func.func @no_tile_divisor
// DEF: ncnn.fallback_reason = "packing_skipped_small_shape"
