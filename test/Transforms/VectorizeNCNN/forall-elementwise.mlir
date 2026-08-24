// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=4 %s | FileCheck %s

// 多输入逐元素：除最内维外的输出维组成 scf.forall(shared_outs) 网格，
// 各输入整行 rank-1 transfer_read，向量化 body 后写入私有行张量
// （tensor.empty 在 forall 体内，bufferize 后每次执行私有——提出体外
// 会成为跨线程共享临时缓冲），经 tensor.parallel_insert_slice 落回
// 共享输出的不相交行片。

func.func @add(%a: tensor<4x16xf32>, %b: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %empty = tensor.empty() : tensor<4x16xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<4x16xf32>, tensor<4x16xf32>) outs(%empty : tensor<4x16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %added = arith.addf %x, %y : f32
    linalg.yield %added : f32
  } -> tensor<4x16xf32>
  return %sum : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @add
// CHECK-NOT: linalg.generic
// CHECK: scf.forall (%[[I:[^)]*]]) in (4) shared_outs
// CHECK: vector.transfer_read %{{[A-Za-z0-9_]+}}[%[[I]], %{{c[0-9_]*}}], {{.*}} vector<16xf32>
// CHECK: vector.transfer_read %{{[A-Za-z0-9_]+}}[%[[I]], %{{c[0-9_]*}}], {{.*}} vector<16xf32>
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: tensor.empty() : tensor<1x16xf32>
// CHECK: vector.transfer_write {{.*}} : vector<16xf32>, tensor<1x16xf32>
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice {{.*}} into {{%.*}}[%[[I]], 0] [1, 16] [1, 1]

// rank-1 没有可并行的外层输出维：保持串行直写形态。
func.func @scale(%x: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %half = arith.constant 0.5 : f32
  %scaled = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%v: f32, %o: f32):
    %mul = arith.mulf %v, %half : f32
    linalg.yield %mul : f32
  } -> tensor<32xf32>
  return %scaled : tensor<32xf32>
}

// CHECK-LABEL: func.func @scale
// CHECK-NOT: scf.forall
// CHECK: vector.transfer_read {{.*}} vector<32xf32>
// CHECK: arith.mulf {{.*}} : vector<32xf32>
// CHECK: vector.transfer_write {{.*}} : vector<32xf32>, tensor<32xf32>
