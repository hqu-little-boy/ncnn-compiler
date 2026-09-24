// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline %s | FileCheck %s
// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline='selective-fusion=false' %s | FileCheck %s --check-prefix=NO-FUSION
// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline='profile-materialized-sites=true' --ncnn-linalg-to-memref-pipeline='profile-instrumentation=true vector-lanes=8 vector-tail=true' %s | FileCheck %s --check-prefix=PROFILE

#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @fuses_original_conv_epilogue(%input: tensor<1x8x36x3xf32>,
                                        %conv_init: tensor<1x6x34x4xf32>,
                                        %relu_init: tensor<1x6x34x4xf32>) -> tensor<1x6x34x4xf32> {
  %weights = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %weights : tensor<1x8x36x3xf32>, tensor<3x3x3x4xf32>) outs(%conv_init : tensor<1x6x34x4xf32>) -> tensor<1x6x34x4xf32>
  %zero = arith.constant 0.0 : f32
  %relu = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x6x34x4xf32>) outs(%relu_init : tensor<1x6x34x4xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x6x34x4xf32>
  return %relu : tensor<1x6x34x4xf32>
}

func.func @keeps_gemm_for_rejected_epilogue(%input: tensor<1x8x8x3xf32>,
                                           %residual: tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32> {
  %weights = arith.constant dense<1.0> : tensor<1x1x3x16xf32>
  %conv_init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %weights : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%conv_init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %result = linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %residual : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %unused: f32, %output: f32):
    %root = math.sqrt %value : f32
    linalg.yield %root : f32
  } -> tensor<1x8x8x16xf32>
  return %result : tensor<1x8x8x16xf32>
}

func.func @lifts_repeated_producer_input(%input: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
  %weights = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %weights : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %one = arith.constant 1.0 : f32
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %silu = linalg.generic {indexing_maps = [#identity, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %same: f32, %unused: f32):
    %negative = arith.negf %same : f32
    %exp = math.exp %negative : f32
    %denominator = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denominator : f32
    %result = arith.mulf %value, %sigmoid : f32
    linalg.yield %result : f32
  } -> tensor<1x8x8x16xf32>
  return %silu : tensor<1x8x8x16xf32>
}

// CHECK: ncnn.fusion_selected_count = 1 : i64
// CHECK-LABEL: func.func @fuses_original_conv_epilogue
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK: arith.maximumf
// CHECK: tensor.insert_slice
// CHECK-NOT: linalg.generic {{.*}} ins({{.*}} : tensor<1x6x34x4xf32>) outs({{.*}} : tensor<1x6x34x4xf32>)
// CHECK-LABEL: func.func @keeps_gemm_for_rejected_epilogue
// CHECK: linalg.matmul
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: math.sqrt
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// NO-FUSION: ncnn.fusion_selected_count = 0 : i64
// NO-FUSION-LABEL: func.func @fuses_original_conv_epilogue
// NO-FUSION: linalg.conv_2d_nhwc_hwcf
// NO-FUSION: linalg.generic
// NO-FUSION: arith.maximumf
// NO-FUSION-LABEL: func.func @keeps_gemm_for_rejected_epilogue
// NO-FUSION: linalg.matmul
// NO-FUSION: linalg.generic
// NO-FUSION: math.sqrt
// NO-FUSION: return

// PROFILE: profile_id = [[SITE:[0-9]+]] : i64
// PROFILE: tail_profile_id = [[TAIL:[0-9]+]] : i64
// PROFILE-LABEL: func.func @fuses_original_conv_epilogue
// PROFILE: call @__ncnn_profile_event_begin(%c[[SITE]]_i64,
// PROFILE: scf.for
// PROFILE: call @__ncnn_profile_event_end(%c[[SITE]]_i64
// PROFILE: call @__ncnn_profile_event_begin(%c[[TAIL]]_i64,
// PROFILE: call @__ncnn_profile_event_end(%c[[TAIL]]_i64

// CHECK-LABEL: func.func @lifts_repeated_producer_input
// CHECK: %[[PIPELINE_MM:.*]] = linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: linalg.generic {{.*}} ins(%[[PIPELINE_MM]], %[[PIPELINE_MM]] : tensor<64x16xf32>, tensor<64x16xf32>)
// CHECK: arith.negf
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// NO-FUSION-LABEL: func.func @lifts_repeated_producer_input
// NO-FUSION: %[[PIPELINE_NOFUSE_MM:.*]] = linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// NO-FUSION: linalg.generic {{.*}} ins(%[[PIPELINE_NOFUSE_MM]], %[[PIPELINE_NOFUSE_MM]] : tensor<64x16xf32>, tensor<64x16xf32>)
// NO-FUSION: arith.negf
// NO-FUSION: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// NO-FUSION-NOT: linalg.conv_2d_nhwc_hwcf
