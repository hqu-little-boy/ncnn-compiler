// RUN: ncnn-mlir-opt '--tile-matmul-forall=tile-rows=32 tile-columns=32' %s | FileCheck %s

// A1 形态：折叠二维域的 matmul + 唯一恒等逐元素消费者。切消费者并融合
// matmul 生产者：两者同入一个 forall，激活随分块并行；K 维（此处 3）
// 不在切分维度内，tile 内完整归约。
func.func @mm_relu(%a: tensor<64x3xf32>, %w: tensor<3x16xf32>) -> tensor<64x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x16xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<64x3xf32>, tensor<3x16xf32>) outs(%empty : tensor<64x16xf32>) -> tensor<64x16xf32>
  %out = tensor.empty() : tensor<64x16xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<64x16xf32>) outs(%out : tensor<64x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<64x16xf32>
  return %r : tensor<64x16xf32>
}

// CHECK-LABEL: func.func @mm_relu
// CHECK: scf.forall (%[[I:.*]], %[[J:.*]]) = (0, 0) to (64, 16) step (32, 16) shared_outs
// CHECK: linalg.matmul ins({{.*}}: tensor<32x3xf32>, tensor<3x16xf32>) outs({{.*}}: tensor<32x16xf32>
// CHECK: linalg.generic {{.*}}tensor<32x16xf32>
// CHECK: arith.maximumf
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice
// CHECK-NOT: tensor<?

// -----

// 裸 matmul（多消费者）：回退为仅切分 matmul 本身。
func.func @mm_two_users(%a: tensor<64x3xf32>, %w: tensor<3x16xf32>) -> (tensor<64x16xf32>, tensor<64x16xf32>) {
  %empty = tensor.empty() : tensor<64x16xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<64x3xf32>, tensor<3x16xf32>) outs(%empty : tensor<64x16xf32>) -> tensor<64x16xf32>
  return %mm, %mm : tensor<64x16xf32>, tensor<64x16xf32>
}

// CHECK-LABEL: func.func @mm_two_users
// CHECK: scf.forall
// CHECK: linalg.matmul ins({{.*}}tensor<32x3xf32>
// CHECK: tensor.parallel_insert_slice

// -----

// 非整除维退化为因子或整维：M=10 的可行因子中 ≤4 的最大值是 2；
// N=8 无可行切分（请求即全域）。不得出现动态尺寸切片。
func.func @mm_odd(%a: tensor<10x3xf32>, %w: tensor<3x8xf32>) -> tensor<10x8xf32> {
  %empty = tensor.empty() : tensor<10x8xf32>
  %mm = linalg.matmul ins(%a, %w : tensor<10x3xf32>, tensor<3x8xf32>) outs(%empty : tensor<10x8xf32>) -> tensor<10x8xf32>
  return %mm : tensor<10x8xf32>
}

// RUN: ncnn-mlir-opt '--tile-matmul-forall=tile-rows=4 tile-columns=4' %s | FileCheck %s --check-prefix=ODD

// ODD-LABEL: func.func @mm_odd
// ODD: scf.forall {{.*}} step (2, 4)
// ODD: linalg.matmul ins({{.*}}tensor<2x3xf32>
// ODD-NOT: tensor<?x
