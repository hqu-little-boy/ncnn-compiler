// RUN: ncnn-mlir-opt --forallize-disjoint-tile-loops %s | FileCheck %s

// 安全拒绝用例：任一硬约束不满足即保持原样（误判等于数据竞争）。

// ---- 拒绝 1：体内存在对 acc 的读取（跨迭代耦合未知）。----
func.func @acc_read(%lhs: tensor<32x32xf32>, %init: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<32x64xf32> {
    %off = arith.muli %i, %c32 : index
    %peek = tensor.extract_slice %acc[0, %off] [32, 32] [1, 1] : tensor<32x64xf32> to tensor<32x32xf32>
    %rhs = tensor.extract_slice %lhs[0, %off] [32, 32] [1, 1] : tensor<32x32xf32> to tensor<32x32xf32>
    %mm = linalg.matmul ins(%rhs, %peek : tensor<32x32xf32>, tensor<32x32xf32>) outs(%peek : tensor<32x32xf32>) -> tensor<32x32xf32>
    %ins = tensor.insert_slice %mm into %acc[0, %off] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<32x64xf32>
    scf.yield %ins : tensor<32x64xf32>
  }
  return %r : tensor<32x64xf32>
}
// CHECK-LABEL: func.func @acc_read
// CHECK: scf.for
// CHECK-NOT: scf.forall

// -----

// ---- 拒绝 2：窗口偏移非步长倍数（iv*48 与 size 64 不一致 → 窗口重叠）。----
func.func @bad_stride(%lhs: tensor<32x32xf32>, %init: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c48 = arith.constant 48 : index
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<32x128xf32> {
    %off = arith.muli %i, %c48 : index
    %rhs = tensor.extract_slice %lhs[0, %off] [32, 64] [1, 1] : tensor<32x32xf32> to tensor<32x64xf32>
    %ini = tensor.extract_slice %init[0, %off] [32, 64] [1, 1] : tensor<32x128xf32> to tensor<32x64xf32>
    %neg = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%rhs : tensor<32x64xf32>) outs(%ini : tensor<32x64xf32>) {
    ^bb0(%in: f32, %o: f32):
      %n = arith.negf %in : f32
      linalg.yield %n : f32
    } -> tensor<32x64xf32>
    %ins = tensor.insert_slice %neg into %acc[0, %off] [32, 64] [1, 1] : tensor<32x64xf32> into tensor<32x128xf32>
    scf.yield %ins : tensor<32x128xf32>
  }
  return %r : tensor<32x128xf32>
}
// CHECK-LABEL: func.func @bad_stride
// CHECK: scf.for
// CHECK-NOT: scf.forall

// -----

// ---- 拒绝 3：两个 iter_args（多链串联语义未证明不相交）。----
func.func @two_accs(%lhs: tensor<16x16xf32>, %init0: tensor<16x32xf32>,
                    %init1: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r0:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%a0 = %init0, %a1 = %init1) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
    %off = arith.muli %i, %c16 : index
    %rhs = tensor.extract_slice %lhs[0, %off] [16, 16] [1, 1] : tensor<16x16xf32> to tensor<16x16xf32>
    %ini0 = tensor.extract_slice %a0[0, %off] [16, 16] [1, 1] : tensor<16x32xf32> to tensor<16x16xf32>
    %mm = linalg.matmul ins(%rhs, %rhs : tensor<16x16xf32>, tensor<16x16xf32>) outs(%ini0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %ins0 = tensor.insert_slice %mm into %a0[0, %off] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<16x32xf32>
    %ins1 = tensor.insert_slice %mm into %a1[0, %off] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<16x32xf32>
    scf.yield %ins0, %ins1 : tensor<16x32xf32>, tensor<16x32xf32>
  }
  return %r0#0, %r0#1 : tensor<16x32xf32>, tensor<16x32xf32>
}
// CHECK-LABEL: func.func @two_accs
// CHECK: scf.for
// CHECK-NOT: scf.forall
