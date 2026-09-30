// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t-plan.json

module {
  func.func @copy(%input: tensor<257x1x1xf32>) -> tensor<257x1x1xf32> {
    %empty = tensor.empty() : tensor<257x1x1xf32>
    %result = linalg.copy ins(%input : tensor<257x1x1xf32>) outs(%empty : tensor<257x1x1xf32>) -> tensor<257x1x1xf32>
    return %result : tensor<257x1x1xf32>
  }
}

// IR: gpu.module @copy_vulkan
// IR: gpu.func @copy
// IR-SAME: kernel
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: memref.load
// IR: memref.store
// IR: gpu.return

// PLAN: "entry_point": "copy"
// PLAN: "operation": "copy"
// PLAN: "reads": [
// PLAN: "writes": [
