// RUN: ncnn-mlir-opt --fold-linalg-constant-transpose %s | FileCheck %s

// 拒绝与快路径：
//   1. 元素类型不可折叠（位宽非 8 的倍数，如 i1）→ transpose 保持原样；
//   2. splat 常量走 splat 快路径折叠（不必逐元素重排）；
//   3. 运行期值（非常量）→ 保持原样。

func.func @non_foldable_element_type() -> tensor<2x2x2x3xi1> {
  %cst = arith.constant dense<true> : tensor<3x2x2x2xi1>
  %0 = tensor.empty() : tensor<2x2x2x3xi1>
  %t = linalg.transpose ins(%cst : tensor<3x2x2x2xi1>) outs(%0 : tensor<2x2x2x3xi1>) permutation = [1, 2, 3, 0]
  return %t : tensor<2x2x2x3xi1>
}

// CHECK-LABEL: func.func @non_foldable_element_type
// CHECK: linalg.transpose ins(%{{.*}} : tensor<3x2x2x2xi1>)

// -----

func.func @splat_constant() -> tensor<2x2x2x3xf32> {
  %cst = arith.constant dense<7.0> : tensor<3x2x2x2xf32>
  %0 = tensor.empty() : tensor<2x2x2x3xf32>
  %t = linalg.transpose ins(%cst : tensor<3x2x2x2xf32>) outs(%0 : tensor<2x2x2x3xf32>) permutation = [1, 2, 3, 0]
  return %t : tensor<2x2x2x3xf32>
}

// CHECK-LABEL: func.func @splat_constant
// CHECK: %[[FOLDED:.*]] = arith.constant dense<7.000000e+00> : tensor<2x2x2x3xf32>
// CHECK-NOT: linalg.transpose
// CHECK: return %[[FOLDED]]

// -----

func.func @runtime_value(%src : tensor<3x2x2x2xf32>) -> tensor<2x2x2x3xf32> {
  %0 = tensor.empty() : tensor<2x2x2x3xf32>
  %t = linalg.transpose ins(%src : tensor<3x2x2x2xf32>) outs(%0 : tensor<2x2x2x3xf32>) permutation = [1, 2, 3, 0]
  return %t : tensor<2x2x2x3xf32>
}

// CHECK-LABEL: func.func @runtime_value
// CHECK: linalg.transpose ins(%arg0
