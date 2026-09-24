// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline='selective-fusion=false profile-materialized-sites=true' %s | FileCheck %s
// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline='selective-fusion=false' %s | FileCheck %s --check-prefix=NO-PROFILE

#identity = affine_map<(d0, d1) -> (d0, d1)>

func.func @unfused_intermediate(%lhs: tensor<4x4xf32>,
                                %rhs: tensor<4x4xf32>,
                                %matmul_init: tensor<4x4xf32>,
                                %epilogue_init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%matmul_init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul : tensor<4x4xf32>) outs(%epilogue_init : tensor<4x4xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %root = math.sqrt %value : f32
    linalg.yield %root : f32
  } -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

func.func private @nested_side_effect(i32)

func.func @nested_fusion_site(%condition: i1) {
  %value = arith.constant 1 : i32
  scf.if %condition {
    func.call @nested_side_effect(%value) {ncnn.fusion_site_id = 123 : i64} : (i32) -> ()
  }
  return
}

func.func @parallel_fusion_site(%trip_count: index) {
  scf.forall (%iv) in (%trip_count) {
    %value = arith.constant 1 : i32
    func.call @nested_side_effect(%value) {ncnn.fusion_site_id = 124 : i64} : (i32) -> ()
  }
  return
}

// CHECK-LABEL: func.func @unfused_intermediate
// CHECK: linalg.matmul
// CHECK: call @__ncnn_profile_materialized
// CHECK: call @__ncnn_profile_materialized
// CHECK: linalg.generic
// CHECK: math.sqrt
// CHECK: return

// CHECK-LABEL: func.func @nested_fusion_site
// CHECK: scf.if
// CHECK: call @__ncnn_profile_event_begin(%c123_i64, %c10_i64)
// CHECK: call @nested_side_effect
// CHECK: call @__ncnn_profile_event_end(%c123_i64{{(_[0-9]+)?}})

// CHECK-LABEL: func.func @parallel_fusion_site
// CHECK: scf.forall
// CHECK: call @__ncnn_profile_event_begin(%c124_i64, %c10_i64)
// CHECK: call @nested_side_effect
// CHECK: call @__ncnn_profile_event_end(%c124_i64{{(_[0-9]+)?}})

// NO-PROFILE-LABEL: func.func @unfused_intermediate
// NO-PROFILE-NOT: __ncnn_profile_materialized
// NO-PROFILE: return
// NO-PROFILE-LABEL: func.func @nested_fusion_site
// NO-PROFILE-NOT: __ncnn_profile_event_begin
// NO-PROFILE: call @nested_side_effect
// NO-PROFILE-LABEL: func.func @parallel_fusion_site
// NO-PROFILE-NOT: __ncnn_profile_event_begin
// NO-PROFILE: call @nested_side_effect
