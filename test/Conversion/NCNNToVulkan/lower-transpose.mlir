// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t-plan.json

module {
  func.func @transpose(%input: tensor<51x5x1xf32>) -> tensor<5x51x1xf32> {
    %empty = tensor.empty() : tensor<5x51x1xf32>
    %result = linalg.transpose ins(%input : tensor<51x5x1xf32>) outs(%empty : tensor<5x51x1xf32>) permutation = [1, 0, 2]
    return %result : tensor<5x51x1xf32>
  }
}

// IR: gpu.module @transpose_vulkan
// IR: gpu.func @transpose
// IR-SAME: kernel
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: arith.remui
// IR: arith.divui
// IR: memref.load
// IR: memref.store
// IR: gpu.return

// PLAN: "entry_point": "transpose"
// PLAN: "local_size": [
// PLAN: "operation": "transpose"
// PLAN: "shape": [
// PLAN: "shape": [
