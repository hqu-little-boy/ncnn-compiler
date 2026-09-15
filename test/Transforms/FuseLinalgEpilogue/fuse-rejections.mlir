#identity = affine_map<(d0, d1) -> (d0, d1)>
#permuted = affine_map<(d0, d1) -> (d1, d0)>

// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s --check-prefix=REJECT
// RUN: ncnn-mlir-opt --fuse-linalg-epilogue="tile-width=0" %s | FileCheck %s --check-prefix=WIDTH

func.func @rejects_permuted_residual(%lhs: tensor<32x4xf32>, %rhs: tensor<4x32xf32>, %residual: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<32x4xf32>, tensor<4x32xf32>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  %out = tensor.empty() : tensor<32x32xf32>
  %sum = linalg.generic {indexing_maps = [#identity, #permuted, #identity], iterator_types = ["parallel", "parallel"]} ins(%matmul, %residual : tensor<32x32xf32>, tensor<32x32xf32>) outs(%out : tensor<32x32xf32>) {
  ^bb0(%in: f32, %res: f32, %unused: f32):
    %add = arith.addf %in, %res : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  return %sum : tensor<32x32xf32>
}

func.func @rejects_residual_view(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>, %residual: tensor<2x32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %view = tensor.cast %residual : tensor<2x32xf32> to tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %sum = linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel", "parallel"]} ins(%matmul, %view : tensor<2x32xf32>, tensor<2x32xf32>) outs(%out : tensor<2x32xf32>) {
  ^bb0(%in: f32, %res: f32, %unused: f32):
    %add = arith.addf %in, %res : f32
    linalg.yield %add : f32
  } -> tensor<2x32xf32>
  return %sum : tensor<2x32xf32>
}

// REJECT: ncnn.fusion_rejection_reasons = "non_identity_map=1,tensor_view_or_alias=1"
// REJECT: ncnn.fusion_selected_count = 0 : i64
// REJECT-LABEL: func.func @rejects_permuted_residual
// REJECT: linalg.matmul
// REJECT-NOT: scf.for
// REJECT-LABEL: func.func @rejects_residual_view
// REJECT: linalg.matmul
// REJECT-NOT: scf.for

// WIDTH: ncnn.fusion_rejection_reasons = "invalid_tile_width=1,non_identity_map=1,tensor_view_or_alias=1"
// WIDTH: ncnn.fusion_selected_count = 0 : i64
// WIDTH-LABEL: func.func @rejects_permuted_residual
// WIDTH: linalg.matmul
// WIDTH-NOT: scf.for
