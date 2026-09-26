// RUN: ncnn-mlir-opt '--tile-matmul-forall=tile-rows=4 tile-columns=4' %s | FileCheck %s

// 只处理顶层实例：嵌套实例再切会制造嵌套并行。scf.for / scf.forall /
// scf.parallel 之内的 matmul 一律保持原样。

func.func @inside_scf_for(%a: tensor<16x3xf32>, %w: tensor<3x16xf32>) -> tensor<16x16xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = linalg.fill ins(%f0 : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) {
    %mm = linalg.matmul ins(%a, %w : tensor<16x3xf32>, tensor<3x16xf32>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mm : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @inside_scf_for
// CHECK: scf.for
// CHECK-NOT: scf.forall
// CHECK: linalg.matmul ins(%{{.*}}: tensor<16x3xf32>, tensor<3x16xf32>) outs(%{{.*}}: tensor<16x16xf32>)

// -----

func.func @inside_scf_forall(%a: tensor<16x3xf32>, %w: tensor<3x16xf32>) -> tensor<16x16xf32> {
  %f0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = linalg.fill ins(%f0 : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %r = scf.forall (%i) = (0) to (2) step (1) shared_outs(%o = %init) -> tensor<16x16xf32> {
    %mm = linalg.matmul ins(%a, %w : tensor<16x3xf32>, tensor<3x16xf32>) outs(%o : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %mm into %o[0, 0] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<16x16xf32>
    }
  }
  return %r : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @inside_scf_forall
// CHECK: scf.forall
// CHECK-NOT: step (4, 4)
// CHECK: linalg.matmul ins(%{{.*}}: tensor<16x3xf32>, tensor<3x16xf32>)

// -----

func.func @inside_scf_parallel(%a: tensor<16x3xf32>, %w: tensor<3x16xf32>) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = linalg.fill ins(%f0 : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %r = scf.parallel (%i) = (%c0) to (%c2) step (%c1) init(%f0) -> f32 {
    %mm = linalg.matmul ins(%a, %w : tensor<16x3xf32>, tensor<3x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %s = tensor.extract %mm[%i, %i] : tensor<16x16xf32>
    scf.reduce(%s : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      scf.reduce.return %sum : f32
    }
  }
  return %r : f32
}

// CHECK-LABEL: func.func @inside_scf_parallel
// CHECK: scf.parallel
// CHECK-NOT: scf.forall
// CHECK: linalg.matmul ins(%{{.*}}: tensor<16x3xf32>, tensor<3x16xf32>) outs(%{{.*}}: tensor<16x16xf32>)
