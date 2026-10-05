// RUN: not ncnn-mlir-opt --lower-ncnn-vulkan-graph="plan-path=%t-collision.json weights-path=%t-collision.json" %s 2>&1 | FileCheck %s --check-prefix=COLLISION
// RUN: test ! -e %t-collision.json

func.func @identity(%input: tensor<4xf32>) -> tensor<4xf32> {
  return %input : tensor<4xf32>
}

// COLLISION: Vulkan plan and weights artifacts must use distinct output paths
