// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s

#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @deferred_repeated_producer_input(
    %input: tensor<1x8x8x3xf32>, %weights: tensor<1x1x3x16xf32>)
    -> tensor<1x8x8x16xf32> {
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>}
    ins(%input, %weights : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>)
    outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %result = linalg.generic {
      indexing_maps = [#identity4, #identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>)
    outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %same_value: f32, %unused: f32):
    %negative = arith.negf %same_value : f32
    %exp = math.exp %negative : f32
    %one = arith.constant 1.000000e+00 : f32
    %denominator = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denominator : f32
    %swish = arith.mulf %value, %sigmoid : f32
    linalg.yield %swish : f32
  } -> tensor<1x8x8x16xf32>
  return %result : tensor<1x8x8x16xf32>
}

func.func @rejects_external_producer_use(
    %input: tensor<1x8x8x3xf32>, %weights: tensor<1x1x3x16xf32>)
    -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>}
    ins(%input, %weights : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>)
    outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %result = linalg.generic {
      indexing_maps = [#identity4, #identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>)
    outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %same_value: f32, %unused: f32):
    %negative = arith.negf %same_value : f32
    %exp = math.exp %negative : f32
    %one = arith.constant 1.000000e+00 : f32
    %denominator = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denominator : f32
    %swish = arith.mulf %value, %sigmoid : f32
    linalg.yield %swish : f32
  } -> tensor<1x8x8x16xf32>
  return %conv, %result : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
}

// CHECK-DAG: ncnn.fusion_selected_count = 0 : i64
// CHECK-DAG: ncnn.fusion_residual_count = 0 : i64
// CHECK-DAG: ncnn.fusion_rejected_count = 2 : i64
// CHECK-DAG: ncnn.fusion_rejection_reasons = "multi_consumer=1,repeated_producer_input=1"
// CHECK-LABEL: func.func @deferred_repeated_producer_input
// CHECK: %[[CONV:.*]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: ncnn.fallback_reason = "repeated_producer_input"
// CHECK: linalg.generic {{.*}} ins(%[[CONV]], %[[CONV]] :
// CHECK-LABEL: func.func @rejects_external_producer_use
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: ncnn.fallback_reason = "multi_consumer"
// CHECK: linalg.generic
// CHECK: return %{{.*}}, %{{.*}} : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
