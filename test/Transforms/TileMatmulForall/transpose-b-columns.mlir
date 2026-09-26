// RUN: ncnn-mlir-opt '--tile-matmul-forall=tile-rows=4 tile-columns=4' %s | FileCheck %s

// MatmulTransposeBOp 的 B 面板是 [N, K] 转置布局：切分会物化部分常量
// 面板并丢失动态 N 偏移，因此列维固定取全 columns（columnChunk = columns），
// 只切 M。行维仍按因子切。

func.func @int8_row_dot(%a: tensor<16x32xi8>, %w: tensor<8x32xi8>) -> tensor<16x8xi32> {
  %empty = tensor.empty() : tensor<16x8xi32>
  %mm = linalg.matmul_transpose_b ins(%a, %w : tensor<16x32xi8>, tensor<8x32xi8>) outs(%empty : tensor<16x8xi32>) -> tensor<16x8xi32>
  return %mm : tensor<16x8xi32>
}

// CHECK-LABEL: func.func @int8_row_dot
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (16, 8) step (4, 8)
// CHECK: linalg.matmul_transpose_b ins(%{{.*}}: tensor<4x32xi8>, tensor<8x32xi8>)

// -----

// N 只有 1 个块时仍按全 columns 输出 size 向量；对照：同等尺寸的普通
// matmul 会把 N 切成 4。
func.func @transpose_b_vs_matmul(%a: tensor<16x32xi8>, %w: tensor<8x32xi8>, %b: tensor<16x32xf32>, %v: tensor<32x8xf32>) -> (tensor<16x8xi32>, tensor<16x8xf32>) {
  %e0 = tensor.empty() : tensor<16x8xi32>
  %m0 = linalg.matmul_transpose_b ins(%a, %w : tensor<16x32xi8>, tensor<8x32xi8>) outs(%e0 : tensor<16x8xi32>) -> tensor<16x8xi32>
  %e1 = tensor.empty() : tensor<16x8xf32>
  %m1 = linalg.matmul ins(%b, %v : tensor<16x32xf32>, tensor<32x8xf32>) outs(%e1 : tensor<16x8xf32>) -> tensor<16x8xf32>
  return %m0, %m1 : tensor<16x8xi32>, tensor<16x8xf32>
}

// CHECK-LABEL: func.func @transpose_b_vs_matmul
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (16, 8) step (4, 8)
// CHECK: linalg.matmul_transpose_b ins(%{{.*}}: tensor<4x32xi8>, tensor<8x32xi8>)
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (16, 8) step (4, 4)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<4x32xf32>, tensor<32x4xf32>)
