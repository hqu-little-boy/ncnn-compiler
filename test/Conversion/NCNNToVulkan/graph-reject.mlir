// RUN: not ncnn-mlir-opt --split-input-file --lower-ncnn-vulkan-graph="plan-path=%t.plan.json weights-path=%t.weights.json" %s 2>&1 | FileCheck %s
// RUN: test ! -e %t.plan.json
// RUN: test ! -e %t.weights.json

func.func @dynamic(%input: tensor<?x4xf32>) -> tensor<?x4xf32> attributes {ncnn.name = "dynamic_input", ncnn.source_layer = 11 : i64} {
  return %input : tensor<?x4xf32>
}
// CHECK: is not supported by Vulkan graph: requires static tensor dimensions
// CHECK-SAME: ncnn.name="dynamic_input", ncnn.source_layer=11

// -----

func.func @fp16(%input: tensor<4xf16>) -> tensor<4xf16> {
  return %input : tensor<4xf16>
}
// CHECK: is not supported by Vulkan graph: requires f32 elements

// -----

func.func @rank5(%input: tensor<1x1x1x1x1xf32>) -> tensor<1x1x1x1x1xf32> {
  return %input : tensor<1x1x1x1x1xf32>
}
// CHECK: is not supported by Vulkan graph: supports tensor ranks 1 through 4

// -----

func.func @zero_extent(%input: tensor<0xf32>) -> tensor<0xf32> {
  return %input : tensor<0xf32>
}
// CHECK: is not supported by Vulkan graph: requires positive tensor dimensions

// -----

func.func @oversized(%input: tensor<33554433xf32>) -> tensor<33554433xf32> {
  return %input : tensor<33554433xf32>
}
// CHECK: is not supported by Vulkan graph: tensor exceeds the Vulkan 1.1 minimum storage-buffer range

// -----

func.func @overflow(%input: tensor<2147483648x2147483648x4xf32>) -> tensor<2147483648x2147483648x4xf32> {
  return %input : tensor<2147483648x2147483648x4xf32>
}
// CHECK: is not supported by Vulkan graph: tensor element count overflows uint64

// -----

func.func @unsupported(%input: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %result = "tosa.sigmoid"(%input) : (tensor<3x4xf32>) -> tensor<3x4xf32> loc("negative"("ncnn-layer":14:0))
  return %result : tensor<3x4xf32>
}
// CHECK: is not supported by Vulkan graph: unsupported operation in the static FP32 graph subset
// CHECK-SAME: ncnn.name="negative", ncnn.source_layer=14

// -----

func.func @broadcast_add(%lhs: tensor<1x4xf32>, %rhs: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %sum = "tosa.add"(%lhs, %rhs) : (tensor<1x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %sum : tensor<2x4xf32>
}
// CHECK: is not supported by Vulkan graph: Add requires identical static tensor shapes

// -----

func.func @bounded_clamp(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = "tosa.clamp"(%input) <{min_val = 0.0 : f32, max_val = 6.0 : f32, nan_mode = "PROPAGATE"}> : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
// CHECK: is not supported by Vulkan graph: only FP32 ReLU clamp [0,+inf] with propagating NaNs is supported

// -----

func.func @integer_data(%input: tensor<8xf32>) -> tensor<1xi8> {
  %zero = arith.constant dense<0> : tensor<1xi8>
  return %zero : tensor<1xi8>
}
// CHECK: is not supported by Vulkan graph: integer metadata cannot be used as tensor data

// -----

func.func private @external(%input: tensor<8xf32>) -> tensor<8xf32>
// CHECK: is not supported by Vulkan graph: requires one straight-line model with static tensor inputs/outputs

// -----

func.func @uninitialized(%input: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  return %empty : tensor<8xf32>
}
// CHECK: is not supported by Vulkan graph: unsupported operation in the static FP32 graph subset

// -----

func.func @first(%input: tensor<8xf32>) -> tensor<8xf32> {
  return %input : tensor<8xf32>
}
func.func @second(%input: tensor<8xf32>) -> tensor<8xf32> {
  return %input : tensor<8xf32>
}
// CHECK: Vulkan graph lowering requires exactly one model function
