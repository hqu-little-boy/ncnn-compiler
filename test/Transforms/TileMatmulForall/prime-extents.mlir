// RUN: ncnn-mlir-opt '--tile-matmul-forall=tile-rows=4 tile-columns=4' %s | FileCheck %s

// 切分尺寸必须是该维 extent 的因子：选不到大于 1 的因子时该维取 extent
// 本身（单次迭代等效不切）。两维都无可行切分时整个 matmul 保持原样。
// 上游对非整除尾块会生成动态尺寸切片（affine.min），违反静态形状契约，
// 因此这里绝不允许出现 tensor<?x...> 或 affine.min。

func.func @both_dims_prime(%a: tensor<7x3xf32>, %w: tensor<3x7xf32>) -> tensor<7x7xf32> {
  %empty = tensor.empty() : tensor<7x7xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<7x3xf32>, tensor<3x7xf32>) outs(%empty : tensor<7x7xf32>) -> tensor<7x7xf32>
  return %mm : tensor<7x7xf32>
}

// CHECK-LABEL: func.func @both_dims_prime
// CHECK-NOT: scf.forall
// CHECK: linalg.matmul ins(%{{.*}}: tensor<7x3xf32>, tensor<3x7xf32>) outs(%{{.*}}: tensor<7x7xf32>)

// -----

func.func @prime_rows_only(%a: tensor<7x3xf32>, %w: tensor<3x8xf32>) -> tensor<7x8xf32> {
  %empty = tensor.empty() : tensor<7x8xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<7x3xf32>, tensor<3x8xf32>) outs(%empty : tensor<7x8xf32>) -> tensor<7x8xf32>
  return %mm : tensor<7x8xf32>
}

// CHECK-LABEL: func.func @prime_rows_only
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (7, 8) step (7, 4)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<7x3xf32>, tensor<3x4xf32>)
// CHECK-NOT: affine.min
// CHECK-NOT: tensor<?x

// -----

func.func @prime_columns_only(%a: tensor<8x3xf32>, %w: tensor<3x7xf32>) -> tensor<8x7xf32> {
  %empty = tensor.empty() : tensor<8x7xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<8x3xf32>, tensor<3x7xf32>) outs(%empty : tensor<8x7xf32>) -> tensor<8x7xf32>
  return %mm : tensor<8x7xf32>
}

// CHECK-LABEL: func.func @prime_columns_only
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (8, 7) step (4, 7)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<4x3xf32>, tensor<3x7xf32>)
// CHECK-NOT: affine.min
// CHECK-NOT: tensor<?x

// -----

// 请求尺寸大于等于 extent 时该维不切（requested >= extent → extent）。
func.func @requested_exceeds_extent(%a: tensor<16x3xf32>, %w: tensor<3x2xf32>) -> tensor<16x2xf32> {
  %empty = tensor.empty() : tensor<16x2xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<16x3xf32>, tensor<3x2xf32>) outs(%empty : tensor<16x2xf32>) -> tensor<16x2xf32>
  return %mm : tensor<16x2xf32>
}

// CHECK-LABEL: func.func @requested_exceeds_extent
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (16, 2) step (4, 2)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<4x3xf32>, tensor<3x2xf32>)
