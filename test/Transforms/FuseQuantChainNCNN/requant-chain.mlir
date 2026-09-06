// RUN: ncnn-mlir-opt --fuse-quant-chain-ncnn %s | FileCheck %s

// int8 requant 尾部（P4）：sitofp map → scale mulf（广播）→ bias addf →
// quantize generic（f32→i8）收拢为单一 generic——body 依原序拼接，混合
// 元素类型输入（i32 主值 + f32 广播参数）保留各自仿射映射。

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#mapb = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>

// CHECK-LABEL: func.func @requant_chain
func.func @requant_chain(%arg0: tensor<1x3x4x3xi32>) -> tensor<1x3x4x3xi8> {
  %cst_scale = arith.constant dense<[[[[5.0, 6.5, 8.0]]]]> : tensor<1x1x1x3xf32>
  %cst_bias = arith.constant dense<[[[[0.5, -0.5, 1.0]]]]> : tensor<1x1x1x3xf32>
  // CHECK-NOT: linalg.map
  // CHECK-NOT: linalg.generic{{.*}}f32{{.*}}outs({{.*}}f32
  // CHECK: linalg.generic
  // CHECK-SAME: ins({{.*}}tensor<1x1x1x3xf32>{{.*}}tensor<1x1x1x3xf32>{{.*}}tensor<1x3x4x3xi32>
  // CHECK-SAME: outs({{.*}}tensor<1x3x4x3xi8>
  // body 原序：sitofp → mulf(scale) → addf(bias) → fptosi
  // CHECK: arith.sitofp
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.fptosi
  // CHECK-NOT: linalg.yield{{.*}}linalg
  %0 = tensor.empty() : tensor<1x3x4x3xf32>
  %mapped = linalg.map { arith.sitofp } ins(%arg0 : tensor<1x3x4x3xi32>) outs(%0 : tensor<1x3x4x3xf32>)
  %1 = tensor.empty() : tensor<1x3x4x3xf32>
  %2 = linalg.generic {indexing_maps = [#map, #mapb, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mapped, %cst_scale : tensor<1x3x4x3xf32>, tensor<1x1x1x3xf32>) outs(%1 : tensor<1x3x4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.mulf %in, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<1x3x4x3xf32>
  %3 = tensor.empty() : tensor<1x3x4x3xf32>
  %4 = linalg.generic {indexing_maps = [#map, #mapb, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst_bias : tensor<1x3x4x3xf32>, tensor<1x1x1x3xf32>) outs(%3 : tensor<1x3x4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.addf %in, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<1x3x4x3xf32>
  %6 = tensor.empty() : tensor<1x3x4x3xi8>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x3x4x3xf32>) outs(%6 : tensor<1x3x4x3xi8>) {
  ^bb0(%in: f32, %out: i8):
    %5 = arith.fptosi %in : f32 to i8
    linalg.yield %5 : i8
  } -> tensor<1x3x4x3xi8>
  // CHECK: return {{.*}} : tensor<1x3x4x3xi8>
  return %7 : tensor<1x3x4x3xi8>
}

// -----

// 多用户 producer 不融合（融合仅吸收单用户链条节点）。
// CHECK-LABEL: func.func @multi_user_no_fuse
func.func @multi_user_no_fuse(%arg0: tensor<1x3x4x3xi32>) -> (tensor<1x3x4x3xf32>, tensor<1x3x4x3xi8>) {
  %cst_scale = arith.constant dense<[[[[5.0, 6.5, 8.0]]]]> : tensor<1x1x1x3xf32>
  %0 = tensor.empty() : tensor<1x3x4x3xf32>
  %mapped = linalg.map { arith.sitofp } ins(%arg0 : tensor<1x3x4x3xi32>) outs(%0 : tensor<1x3x4x3xf32>)
  // mulf 结果多用户（返回 + quantize），只吸收单用户上游：sitofp 并入
  // mulf，mulf 与 quantize 保持分离。
  // CHECK-NOT: linalg.map
  // CHECK: linalg.generic
  // CHECK: arith.sitofp
  // CHECK: arith.mulf
  // CHECK: linalg.generic
  // CHECK: arith.fptosi
  %1 = tensor.empty() : tensor<1x3x4x3xf32>
  %2 = linalg.generic {indexing_maps = [#map, #mapb, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mapped, %cst_scale : tensor<1x3x4x3xf32>, tensor<1x1x1x3xf32>) outs(%1 : tensor<1x3x4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.mulf %in, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<1x3x4x3xf32>
  %6 = tensor.empty() : tensor<1x3x4x3xi8>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x3x4x3xf32>) outs(%6 : tensor<1x3x4x3xi8>) {
  ^bb0(%in: f32, %out: i8):
    %5 = arith.fptosi %in : f32 to i8
    linalg.yield %5 : i8
  } -> tensor<1x3x4x3xi8>
  // CHECK: return {{.*}}tensor<1x3x4x3xf32>{{.*}}tensor<1x3x4x3xi8>
  return %2, %7 : tensor<1x3x4x3xf32>, tensor<1x3x4x3xi8>
}
