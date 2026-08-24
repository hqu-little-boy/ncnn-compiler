// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize %s | FileCheck %s

// 方案 a epilogue 衔接：conv 结果的唯一用户是恒等逐元素 generic 时，
// consumer 整体搬进折叠二维域（matmul → 2D generic → expand_shape），
// 下游继续看到原四维形状；fuse-linalg-epilogue 的视图守卫保证不会二次
// 融合本 pass 产物。

func.func @conv_relu(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x8x8x16xf32>
  return %relu : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_relu
// CHECK: arith.constant dense{{.*}} : tensor<3x16xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<1x8x8x3xf32> into tensor<64x3xf32>
// CHECK: linalg.matmul ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: linalg.generic {{.*}}ins({{.*}} : tensor<64x16xf32>) outs({{.*}} : tensor<64x16xf32>)
// CHECK: arith.maximumf
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// 多消费者 / 非逐元素用户不融合：matmul 直接 expand 回四维，consumer 原样保留。
func.func @conv_two_users(%arg0: tensor<1x8x8x3xf32>) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  return %conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_two_users
// CHECK: linalg.matmul ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: arith.maximumf
