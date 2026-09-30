// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t-plan.json

module {
  func.func @reshape(%input: tensor<2x2x2xf32>) -> tensor<8x1x1xf32> {
    %shape = arith.constant dense<[8, 1, 1]> : tensor<3xindex>
    %result = tensor.reshape %input(%shape) : (tensor<2x2x2xf32>, tensor<3xindex>) -> tensor<8x1x1xf32>
    return %result : tensor<8x1x1xf32>
  }
}

// IR: gpu.module @reshape_vulkan
// IR: gpu.func @reshape
// IR-SAME: kernel
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: memref.load
// IR: memref.store
// IR: gpu.return
// IR-NOT: tensor.reshape

// PLAN: "entry_point": "reshape"
// PLAN: "operation": "reshape"
// PLAN: "shape": [
