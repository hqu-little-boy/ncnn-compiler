// RUN: not ncnn-mlir-opt --split-input-file --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s 2>&1 | FileCheck %s
// RUN: test ! -e %t-plan.json

func.func @dynamic(%input: tensor<?xf32>) -> tensor<?xf32> {
  return %input : tensor<?xf32>
}

// CHECK: error: Vulkan V2 single-op lowering requires one static f32 tensor result

// -----

func.func @fp16(%input: tensor<8xf16>) -> tensor<8xf16> {
  return %input : tensor<8xf16>
}

// CHECK: error: Vulkan V2 single-op lowering requires one static f32 tensor result

// -----

func.func @rank5(%input: tensor<1x1x1x1x1xf32>) -> tensor<1x1x1x1x1xf32> {
  return %input : tensor<1x1x1x1x1xf32>
}

// CHECK: error: Vulkan V2 single-op lowering requires one static f32 tensor result
