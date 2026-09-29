// RUN: ncnn-mlir-opt --ncnn-vulkan-semantic-pipeline --dump-pass-pipeline %s 2>&1 | FileCheck %s

module {
  func.func @model(%input: tensor<1xf32>) -> tensor<1xf32> {
    return %input : tensor<1xf32>
  }
}

// CHECK: verify-no-ncnn-ops
// CHECK: tosa-optional-decompositions
// CHECK: tosa-to-linalg-named{{.*}}prefer-conv2d-kernel-layout-hwcf=true
// CHECK: tosa-to-linalg
// CHECK: fold-linalg-constant-transpose
// CHECK: tosa-to-tensor
// CHECK: tosa-to-arith
// CHECK: verify-no-tosa-ops
// CHECK: verify-vulkan-supported-model
// CHECK-NOT: fuse-linalg-epilogue
// CHECK-NOT: strategy-ncnn
// CHECK-NOT: vectorize-ncnn
// CHECK-NOT: emit-ncnn-execution-plan
