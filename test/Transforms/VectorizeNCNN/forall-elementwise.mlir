// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=4 %s | FileCheck %s

// 多输入逐元素：除最内维外的输出维组成 scf.forall(shared_outs) 网格，
// 各输入按分块预算（≤32 位元素 4×lanes）rank-1 transfer_read，向量化
// body 经分块 scf.for（iter_args 链）写入私有行张量（tensor.empty 在
// forall 体内，bufferize 后每次执行私有——提出体外会成为跨线程共享
// 临时缓冲），再经 tensor.parallel_insert_slice 落回共享输出的不相交
// 行片。行宽 40 > 16（4×lanes）→ 2 个整分块 + 8 元素标量尾。

func.func @add(%a: tensor<4x40xf32>, %b: tensor<4x40xf32>) -> tensor<4x40xf32> {
  %empty = tensor.empty() : tensor<4x40xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<4x40xf32>, tensor<4x40xf32>) outs(%empty : tensor<4x40xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %added = arith.addf %x, %y : f32
    linalg.yield %added : f32
  } -> tensor<4x40xf32>
  return %sum : tensor<4x40xf32>
}

// CHECK-LABEL: func.func @add
// CHECK-NOT: linalg.generic
// CHECK: scf.forall (%[[I:[^)]*]]) in (4) shared_outs
// CHECK: tensor.empty() : tensor<1x40xf32>
// CHECK: scf.for {{.*}} iter_args
// CHECK: vector.transfer_read {{.*}} vector<16xf32>
// CHECK: vector.transfer_read {{.*}} vector<16xf32>
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.transfer_write {{.*}} : vector<16xf32>, tensor<1x40xf32>
// CHECK: scf.yield
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice {{.*}} into {{%.*}}[%[[I]], 0] [1, 40] [1, 1]

// rank-1 没有可并行的外层维：保持串行直写形态（分块 scf.for + iter_args）。
func.func @scale(%x: tensor<64xf32>) -> tensor<64xf32> {
  %empty = tensor.empty() : tensor<64xf32>
  %half = arith.constant 0.5 : f32
  %scaled = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%empty : tensor<64xf32>) {
  ^bb0(%v: f32, %o: f32):
    %mul = arith.mulf %v, %half : f32
    linalg.yield %mul : f32
  } -> tensor<64xf32>
  return %scaled : tensor<64xf32>
}

// CHECK-LABEL: func.func @scale
// CHECK-NOT: scf.forall
// CHECK: tensor.empty() : tensor<64xf32>
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} vector<16xf32>
// CHECK: arith.mulf {{.*}} : vector<16xf32>
// CHECK: vector.transfer_write {{.*}} : vector<16xf32>, tensor<64xf32>
// CHECK: scf.yield

// 行宽非分块预算整数倍：满分块循环之后余数（< 4×lanes）以标量
// tensor.extract/tensor.insert 兜底，不能整除的行不留超宽或越界向量。
func.func @tail(%x: tensor<2x50xf32>) -> tensor<2x50xf32> {
  %empty = tensor.empty() : tensor<2x50xf32>
  %neg = arith.constant -1.0 : f32
  %clamped = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<2x50xf32>) outs(%empty : tensor<2x50xf32>) {
  ^bb0(%v: f32, %o: f32):
    %mul = arith.mulf %v, %neg : f32
    linalg.yield %mul : f32
  } -> tensor<2x50xf32>
  return %clamped : tensor<2x50xf32>
}

// CHECK-LABEL: func.func @tail
// CHECK: scf.for {{.*}} iter_args
// CHECK: vector.transfer_read {{.*}} vector<16xf32>
// CHECK: vector.transfer_write {{.*}} : vector<16xf32>, tensor<1x50xf32>
// CHECK: scf.for {{.*}} iter_args
// CHECK: tensor.extract {{.*}} : tensor<2x50xf32>
// CHECK: arith.mulf {{.*}} : f32
// CHECK: tensor.insert {{.*}} into {{.*}} : tensor<1x50xf32>
// CHECK: scf.yield

// 整行不超过分块预算：保持整行向量直写，不引入分块循环。
func.func @narrow(%x: tensor<4x12xf32>) -> tensor<4x12xf32> {
  %empty = tensor.empty() : tensor<4x12xf32>
  %neg = arith.constant -1.0 : f32
  %clamped = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<4x12xf32>) outs(%empty : tensor<4x12xf32>) {
  ^bb0(%v: f32, %o: f32):
    %mul = arith.mulf %v, %neg : f32
    linalg.yield %mul : f32
  } -> tensor<4x12xf32>
  return %clamped : tensor<4x12xf32>
}

// CHECK-LABEL: func.func @narrow
// CHECK-NOT: scf.for {{.*}} iter_args
// CHECK: vector.transfer_read {{.*}} vector<12xf32>
// CHECK: arith.mulf {{.*}} : vector<12xf32>
// CHECK: vector.transfer_write {{.*}} : vector<12xf32>, tensor<1x12xf32>
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice {{.*}} into {{%.*}}[%{{.*}}, 0] [1, 12] [1, 1]
