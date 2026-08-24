// RUN: ncnn-mlir-opt --forallize-disjoint-tile-loops %s | FileCheck %s

// fuse-linalg-epilogue 的分块循环产物形态：单 tensor iter_args、体内零
// acc 引用、终结 insert_slice 窗口偏移 = iv*T 且 size = T。改写为
// shared_outs forall；K 维（64）整体保留在 tile 计算内。

func.func @tiled_mm(%lhs: tensor<128x64xf32>, %weight: tensor<64x256xf32>,
                    %init: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<128x256xf32> {
    %off = arith.muli %i, %c64 : index
    %rhs = tensor.extract_slice %weight[0, %off] [64, 64] [1, 1] : tensor<64x256xf32> to tensor<64x64xf32>
    %ini = tensor.extract_slice %init[0, %off] [128, 64] [1, 1] : tensor<128x256xf32> to tensor<128x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x64xf32>) outs(%ini : tensor<128x64xf32>) -> tensor<128x64xf32>
    %ins = tensor.insert_slice %mm into %acc[0, %off] [128, 64] [1, 1] : tensor<128x64xf32> into tensor<128x256xf32>
    scf.yield %ins : tensor<128x256xf32>
  }
  return %r : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @tiled_mm
// CHECK-NOT: scf.for
// CHECK: scf.forall (%[[I:.*]]) in (4) shared_outs
// CHECK: %[[OFF:.*]] = arith.muli %[[I]],
// CHECK: linalg.matmul ins({{.*}}tensor<128x64xf32>{{.*}}tensor<64x64xf32>
// CHECK: scf.forall.in_parallel {
// CHECK: tensor.parallel_insert_slice {{.*}} into {{%.*}}[0, %[[OFF]]]

// -----

// 循环外的尾块残片（写窗与主循环不相交）保持原样串行。
func.func @tiled_with_tail(%lhs: tensor<96x32xf32>, %weight: tensor<32x96xf32>,
                           %init: tensor<96x96xf32>) -> tensor<96x96xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c96 = arith.constant 96 : index
  %main = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %init) -> tensor<96x96xf32> {
    %off = arith.muli %i, %c32 : index
    %rhs = tensor.extract_slice %weight[0, %off] [32, 32] [1, 1] : tensor<32x96xf32> to tensor<32x32xf32>
    %ini = tensor.extract_slice %init[0, %off] [96, 32] [1, 1] : tensor<96x96xf32> to tensor<96x32xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<96x32xf32>, tensor<32x32xf32>) outs(%ini : tensor<96x32xf32>) -> tensor<96x32xf32>
    %ins = tensor.insert_slice %mm into %acc[0, %off] [96, 32] [1, 1] : tensor<96x32xf32> into tensor<96x96xf32>
    scf.yield %ins : tensor<96x96xf32>
  }
  %tailOff = arith.muli %c3, %c32 : index
  %tailRhs = tensor.extract_slice %weight[0, %tailOff] [32, 32] [1, 1] : tensor<32x96xf32> to tensor<32x32xf32>
  %tailIni = tensor.extract_slice %main[0, %tailOff] [96, 32] [1, 1] : tensor<96x96xf32> to tensor<96x32xf32>
  %tailMm = linalg.matmul ins(%lhs, %tailRhs : tensor<96x32xf32>, tensor<32x32xf32>) outs(%tailIni : tensor<96x32xf32>) -> tensor<96x32xf32>
  %tailIns = tensor.insert_slice %tailMm into %main[0, %tailOff] [96, 32] [1, 1] : tensor<96x32xf32> into tensor<96x96xf32>
  return %tailIns : tensor<96x96xf32>
}

// CHECK-LABEL: func.func @tiled_with_tail
// CHECK: scf.forall (%[[I:.*]]) in (3) shared_outs
// CHECK: tensor.parallel_insert_slice
// 主循环改写后尾块仍以串行 insert_slice 收尾：
// CHECK: tensor.insert_slice
// CHECK: return
