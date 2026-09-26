// RUN: ncnn-mlir-opt --fold-linalg-constant-transpose %s | FileCheck %s

// 负向：本 pass 只认 transpose 的输入直接定义于 arith.constant，
// 不像 findConstantElements 那样穿透 tensor.cast / collapse / expand 视图。
// 被视图包着的常量保持原样（linalg.transpose 不被折叠）。
//
// 这与 docs/code-quality-refactor-roadmap.md §4.4 T-M1 列的「常量穿透」
// 预期相反：实测该 pass 未接入 P1 的统一常量查找，穿透不成立。
// 若日后收敛到 findConstantElements，本用例需反转为正向。
// 记录见 docs/refactor/p4-2026-09-26/backlog.md。

func.func @constant_behind_cast() -> tensor<2x2x2x3xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x2x2x2xf32>
  %cast = tensor.cast %cst : tensor<3x2x2x2xf32> to tensor<3x2x2x2xf32>
  %0 = tensor.empty() : tensor<2x2x2x3xf32>
  %t = linalg.transpose ins(%cast : tensor<3x2x2x2xf32>) outs(%0 : tensor<2x2x2x3xf32>) permutation = [1, 2, 3, 0]
  return %t : tensor<2x2x2x3xf32>
}

// CHECK-LABEL: func.func @constant_behind_cast
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<3x2x2x2xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[CST]]
// CHECK: linalg.transpose ins(%[[CAST]] : tensor<3x2x2x2xf32>) outs(%{{.*}}: tensor<2x2x2x3xf32>) permutation = [1, 2, 3, 0]

// -----

func.func @constant_behind_collapse() -> tensor<2x2x2x3xf32> {
  %cst = arith.constant dense<1.0> : tensor<1x3x2x2x2xf32>
  %collapsed = tensor.collapse_shape %cst [[0, 1], [2], [3], [4]] : tensor<1x3x2x2x2xf32> into tensor<3x2x2x2xf32>
  %0 = tensor.empty() : tensor<2x2x2x3xf32>
  %t = linalg.transpose ins(%collapsed : tensor<3x2x2x2xf32>) outs(%0 : tensor<2x2x2x3xf32>) permutation = [1, 2, 3, 0]
  return %t : tensor<2x2x2x3xf32>
}

// CHECK-LABEL: func.func @constant_behind_collapse
// CHECK: tensor.collapse_shape
// CHECK: linalg.transpose ins(%{{.*}} : tensor<3x2x2x2xf32>)
