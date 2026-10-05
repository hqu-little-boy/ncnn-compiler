// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-graph="plan-path=%t.plan.json weights-path=%t.weights.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan.json
// RUN: FileCheck %s --check-prefix=WEIGHTS < %t.weights.json

module {
  func.func @add_relu(%lhs: tensor<3x5x7xf32>, %rhs: tensor<3x5x7xf32>) -> (tensor<3x5x7xf32>, tensor<3x5x7xf32>) {
    %sum = "tosa.add"(%lhs, %rhs) : (tensor<3x5x7xf32>, tensor<3x5x7xf32>) -> tensor<3x5x7xf32>
    %relu = "tosa.clamp"(%sum) <{min_val = 0.0 : f32, max_val = 0x7F800000 : f32, nan_mode = "PROPAGATE"}> : (tensor<3x5x7xf32>) -> tensor<3x5x7xf32>
    return %sum, %relu : tensor<3x5x7xf32>, tensor<3x5x7xf32>
  }
}

// IR: gpu.container_module
// IR: gpu.module @add_relu_vulkan_k0
// IR: gpu.func @kernel_0
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: arith.addf
// IR: gpu.module @add_relu_vulkan_k1
// IR: gpu.func @kernel_1
// IR: arith.maximumf
// IR: gpu.module @add_relu_vulkan_k2
// IR: gpu.func @kernel_2
// IR: gpu.module @add_relu_vulkan_k3
// IR: gpu.func @kernel_3
// IR-NOT: func.func
// IR-NOT: tosa.
// IR-NOT: tensor.
// IR-NOT: unrealized_conversion_cast

// PLAN: "entry_point": "kernel_0"
// PLAN: "operation": "add"
// PLAN: "entry_point": "kernel_1"
// PLAN: "operation": "relu"
// PLAN: "entry_point": "kernel_2"
// PLAN: "operation": "copy"
// PLAN: "entry_point": "kernel_3"
// PLAN: "operation": "copy"
// PLAN: "kind": "ncnn.vulkan_execution_plan"
// PLAN: "schema_version": 1
// WEIGHTS: "kind": "ncnn.vulkan_weight_payloads"
// WEIGHTS: "weights": []
