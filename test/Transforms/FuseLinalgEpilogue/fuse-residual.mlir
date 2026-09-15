#map = affine_map<(d0, d1) -> (d0, d1)>

// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s --check-prefix=SELECT
// RUN: ncnn-mlir-opt --fuse-linalg-epilogue="allow-residual=false" %s | FileCheck %s --check-prefix=DISABLE

func.func @fuses_matmul_residual(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>, %residual: tensor<2x32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%matmul, %residual : tensor<2x32xf32>, tensor<2x32xf32>) outs(%out : tensor<2x32xf32>) {
  ^bb0(%in: f32, %res: f32, %unused: f32):
    %add = arith.addf %in, %res : f32
    linalg.yield %add : f32
  } -> tensor<2x32xf32>
  return %sum : tensor<2x32xf32>
}

// SELECT: ncnn.fusion_residual_count = 1 : i64
// SELECT: ncnn.fusion_selected_count = 1 : i64
// SELECT-LABEL: func.func @fuses_matmul_residual
// SELECT: tensor.extract_slice %{{.*}}[{{.*}}] [2, 16] [1, 1] : tensor<2x32xf32> to tensor<2x16xf32>
// SELECT: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<2x16xf32>, tensor<2x16xf32>)
// SELECT: arith.addf
// SELECT: tensor.insert_slice {{.*}} tensor<2x16xf32> into tensor<2x32xf32>
// SELECT-NOT: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<2x32xf32>, tensor<2x32xf32>)

// DISABLE: ncnn.fusion_rejection_reasons = "residual_disabled=1"
// DISABLE: ncnn.fusion_selected_count = 0 : i64
// DISABLE-LABEL: func.func @fuses_matmul_residual
// DISABLE: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<2x32xf32>, tensor<2x32xf32>)
// DISABLE-NOT: scf.for
