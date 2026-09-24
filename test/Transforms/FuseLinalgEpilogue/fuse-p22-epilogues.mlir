// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s --check-prefix=P22
// RUN: ncnn-mlir-opt --fuse-linalg-epilogue="allow-broadcast=false" %s | FileCheck %s --check-prefix=REJECT
// RUN: ncnn-mlir-opt --fuse-linalg-epilogue --instrument-ncnn-profile %s | FileCheck %s --check-prefix=PROFILE

#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>

func.func @clamp(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>, %bias: tensor<32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %result = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel", "parallel"]} ins(%mm, %bias : tensor<2x32xf32>, tensor<32xf32>) outs(%out : tensor<2x32xf32>) {
  ^bb0(%x: f32, %b: f32, %unused: f32):
    %sum = arith.addf %x, %b : f32
    %min = arith.constant 0.000000e+00 : f32
    %max = arith.constant 6.000000e+00 : f32
    %lower = arith.maximumf %sum, %min : f32
    %clamped = arith.minimumf %lower, %max : f32
    linalg.yield %clamped : f32
  } -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

func.func @swish(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2x32xf32>) outs(%out : tensor<2x32xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %neg = arith.negf %x : f32
    %exp = math.exp %neg : f32
    %one = arith.constant 1.000000e+00 : f32
    %denom = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denom : f32
    %swish = arith.mulf %x, %sigmoid : f32
    linalg.yield %swish : f32
  } -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

func.func @gelu(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2x32xf32>) outs(%out : tensor<2x32xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %half = arith.constant 5.000000e-01 : f32
    %scaled = arith.mulf %x, %half : f32
    %erf = math.erf %scaled : f32
    %one = arith.constant 1.000000e+00 : f32
    %sum = arith.addf %one, %erf : f32
    %gelu = arith.mulf %x, %sum : f32
    %result = arith.mulf %gelu, %half : f32
    linalg.yield %result : f32
  } -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

func.func @cast_chain(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2x32xf32>) outs(%out : tensor<2x32xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %narrow = arith.truncf %x : f32 to f16
    %wide = arith.extf %narrow : f16 to f32
    linalg.yield %wide : f32
  } -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

func.func @tail_profile_site(%lhs: tensor<2x4xf32>, %rhs: tensor<4x18xf32>) -> tensor<2x18xf32> {
  %init = tensor.empty() : tensor<2x18xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x18xf32>) outs(%init : tensor<2x18xf32>) -> tensor<2x18xf32>
  %out = tensor.empty() : tensor<2x18xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2x18xf32>) outs(%out : tensor<2x18xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %root = arith.negf %x : f32
    linalg.yield %root : f32
  } -> tensor<2x18xf32>
  return %result : tensor<2x18xf32>
}

// P22-DAG: ncnn.fusion_revision = "fusion-v2"
// P22-DAG: ncnn.fusion_site_id =
// P22-DAG: profile_id =
// P22-DAG: ncnn.fusion_broadcast_inputs = "projected_broadcast"
// P22-DAG: ncnn.fusion_chain = "ordered_elementwise"
// P22-DAG: math.exp
// P22-DAG: math.erf
// P22-DAG: arith.truncf
// P22-DAG: tensor.extract_slice %{{.*}}[{{.*}}] [16] [1] : tensor<32xf32> to tensor<16xf32>
// P22-NOT: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<2x32xf32>, tensor<32xf32>)
// REJECT: ncnn.fusion_rejection_reasons = "broadcast_disabled=1"
// REJECT: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<2x32xf32>, tensor<32xf32>)

// PROFILE-LABEL: func.func @tail_profile_site
// PROFILE: call @__ncnn_profile_event_begin
// PROFILE: scf.for
// PROFILE: call @__ncnn_profile_event_end
// PROFILE: call @__ncnn_profile_event_begin
// PROFILE: linalg.generic
// PROFILE: call @__ncnn_profile_event_end
