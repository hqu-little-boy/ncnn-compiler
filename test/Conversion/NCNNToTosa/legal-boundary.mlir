// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

func.func private @identity(tensor<2x3x4xf32>) -> tensor<2x3x4xf32>

func.func @legal_boundary(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %before = ncnn.relu %arg0 : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %call = func.call @identity(%before) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %after = ncnn.relu %call : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %after : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @legal_boundary
// CHECK: %[[INPUT:.*]] = tosa.reshape
// CHECK: %[[BEFORE:.*]] = tosa.clamp %[[INPUT]]
// CHECK: %[[TO_CHW:.*]] = tosa.transpose
// CHECK: %[[CALL:.*]] = call @identity(%[[TO_CHW]])
// CHECK: %[[FROM_CHW:.*]] = tosa.transpose %[[CALL]]
// CHECK: %[[AFTER:.*]] = tosa.clamp
// CHECK: return
// CHECK-NOT: ncnn.relu
