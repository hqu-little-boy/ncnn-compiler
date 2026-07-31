// RUN: ncnn-mlir-opt --verify-no-ncnn-ops %s | FileCheck %s

module {
  func.func @clean(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @clean
// CHECK-NOT: ncnn.
