// RUN: ncnn-mlir-opt '--tile-matmul-forall=tile-rows=8 tile-columns=4' %s | FileCheck %s

// 切分尺寸向量按迭代空间位置对应 M/N，不按大小或声明顺序。交换 M/N
// 尺寸后 step 必须仍以 (rowChunk, columnChunk) 的顺序输出两个条目。

func.func @tall(%a: tensor<16x3xf32>, %w: tensor<3x8xf32>) -> tensor<16x8xf32> {
  %empty = tensor.empty() : tensor<16x8xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<16x3xf32>, tensor<3x8xf32>) outs(%empty : tensor<16x8xf32>) -> tensor<16x8xf32>
  return %mm : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @tall
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (16, 8) step (8, 4)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<8x3xf32>, tensor<3x4xf32>)

// -----

func.func @wide(%a: tensor<8x3xf32>, %w: tensor<3x16xf32>) -> tensor<8x16xf32> {
  %empty = tensor.empty() : tensor<8x16xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<8x3xf32>, tensor<3x16xf32>) outs(%empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %mm : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @wide
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (8, 16) step (8, 4)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<8x3xf32>, tensor<3x4xf32>)

// -----

// M 小于请求、N 大于请求：M 保持不切（requested >= extent → extent），
// N 切；size 向量仍输出两个条目且顺序为 (M, N)。
func.func @mixed(%a: tensor<4x3xf32>, %w: tensor<3x16xf32>) -> tensor<4x16xf32> {
  %empty = tensor.empty() : tensor<4x16xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<4x3xf32>, tensor<3x16xf32>) outs(%empty : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %mm : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @mixed
// CHECK: scf.forall (%{{.*}}, %{{.*}}) = (0, 0) to (4, 16) step (4, 4)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<4x3xf32>, tensor<3x4xf32>)
