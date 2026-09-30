// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t-plan.json

module {
  func.func @add(%lhs: tensor<257x1x1xf32>, %rhs: tensor<257x1x1xf32>) -> tensor<257x1x1xf32> {
    %empty = tensor.empty() : tensor<257x1x1xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<257x1x1xf32>, tensor<257x1x1xf32>) outs(%empty : tensor<257x1x1xf32>) {
    ^bb0(%left: f32, %right: f32, %output: f32):
      %sum = arith.addf %left, %right : f32
      linalg.yield %sum : f32
    } -> tensor<257x1x1xf32>
    return %result : tensor<257x1x1xf32>
  }
}

// IR: gpu.container_module
// IR: spirv.target_env = #spirv.target_env
// IR: gpu.module @add_vulkan
// IR: gpu.func @add
// IR-SAME: kernel
// IR: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: arith.addf
// IR: memref.store
// IR: gpu.return

// PLAN: "inputs": [
// PLAN: "binding": 0
// PLAN: "binding": 1
// PLAN: "binding": 2
// PLAN: "entry_point": "add"
// PLAN: "local_size": [
// PLAN: "operation": "add"
// PLAN: "outputs": [
// PLAN: "schema_version": 1
