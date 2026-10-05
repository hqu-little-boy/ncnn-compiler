// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-graph="plan-path=%t.plan.json weights-path=%t.weights.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan.json

module {
  func.func @unused_input(%input: tensor<3x5x7xf32>, %unused: tensor<3x5x7xf32>) -> tensor<3x5x7xf32> {
    %sum = "tosa.add"(%input, %input) : (tensor<3x5x7xf32>, tensor<3x5x7xf32>) -> tensor<3x5x7xf32>
    return %sum : tensor<3x5x7xf32>
  }
}

// IR: gpu.module @unused_input_vulkan_k0
// IR: gpu.module @unused_input_vulkan_k1
// IR-NOT: func.func
// IR-NOT: tosa.

// PLAN: "first_kernel": 4294967295
// PLAN: "last_kernel": 4294967295
// PLAN: "name": "input2"
