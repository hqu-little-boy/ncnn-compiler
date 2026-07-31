// RUN: not ncnn-mlir-opt --verify-no-tosa-ops %s 2>&1 | FileCheck %s

module {
  func.func @residual() -> tensor<4xf32> {
    %0 = "tosa.const"() <{values = dense<0.0> : tensor<4xf32>}> : () -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: error: 'tosa.const' op remains after lowering to Linalg; op=tosa.const
