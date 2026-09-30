// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t-plan.json

module {
  func.func @relu(%input: tensor<257x1x1xf32>) -> tensor<257x1x1xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<257x1x1xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<257x1x1xf32>) outs(%empty : tensor<257x1x1xf32>) {
    ^bb0(%value: f32, %output: f32):
      %relu = arith.maximumf %value, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<257x1x1xf32>
    return %result : tensor<257x1x1xf32>
  }
}

// IR: gpu.container_module
// IR: spirv.target_env = #spirv.target_env
// IR: gpu.module @relu_vulkan
// IR: gpu.func @relu
// IR-SAME: kernel
// IR: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: memref.load
// IR: arith.maximumf
// IR: memref.store
// IR: gpu.return

// PLAN: "inputs": [
// PLAN: "binding": 0
// PLAN: "binding": 1
// PLAN: "set": 0
// PLAN: "entry_point": "relu"
// PLAN: "local_size": [
// PLAN: "operation": "relu"
// PLAN: "outputs": [
// PLAN: "schema_version": 1
