// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t-plan.json

module {
  func.func @identity(%input: tensor<257x1x1xf32>) -> tensor<257x1x1xf32> {
    return %input : tensor<257x1x1xf32>
  }
}

// IR: gpu.module @identity_vulkan
// IR: gpu.func @identity
// IR-SAME: kernel
// IR: gpu.global_id x
// IR: arith.cmpi ult
// IR: memref.load
// IR: memref.store
// IR: gpu.return
// IR-NOT: func.func
// IR-NOT: tensor.

// PLAN: "entry_point": "identity"
// PLAN: "operation": "identity"
// PLAN: "reads": [
// PLAN: "writes": [
