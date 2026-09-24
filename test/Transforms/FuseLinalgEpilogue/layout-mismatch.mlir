// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

func.func @rejects_layout_mismatch(%lhs: tensor<2x4xf32>, %rhs: tensor<4x32xf32>) -> tensor<2x32xf32> {
  %init = tensor.empty() : tensor<2x32xf32>
  %mm = linalg.matmul {ncnn.output_layout = "packed"} ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x32xf32>) outs(%init : tensor<2x32xf32>) -> tensor<2x32xf32>
  %out = tensor.empty() : tensor<2x32xf32>
  %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2x32xf32>) outs(%out : tensor<2x32xf32>) attrs = {ncnn.input_layout = "identity"} {
  ^bb0(%x: f32, %unused: f32):
    linalg.yield %x : f32
  } -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

// CHECK: ncnn.fusion_allow_broadcast = true
// CHECK: ncnn.fusion_layout_aware = true
// CHECK: ncnn.fusion_rejection_reasons = "layout_mismatch=1"
// CHECK: ncnn.fusion_selected_count = 0 : i64
// CHECK: ncnn.fallback_reason = "layout_mismatch"
// CHECK-NOT: ncnn.fusion = "selected"
