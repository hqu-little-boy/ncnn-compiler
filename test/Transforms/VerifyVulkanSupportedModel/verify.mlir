// RUN: not ncnn-mlir-opt --split-input-file --verify-vulkan-supported-model %s 2>&1 | FileCheck %s

func.func @dynamic(%input: tensor<?x4xf32>) -> tensor<?x4xf32> attributes {ncnn.name = "dynamic_input", ncnn.source_layer = 11 : i64} {
  return %input : tensor<?x4xf32>
}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: requires static tensor dimensions
// CHECK-SAME: ncnn.name="dynamic_input"
// CHECK-SAME: ncnn.source_layer=11

// -----

func.func @fp16(%input: tensor<4xf16>) -> tensor<4xf16> attributes {ncnn.name = "half_input", ncnn.source_layer = 12 : i64} {
  return %input : tensor<4xf16>
}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: requires f32 elements
// CHECK-SAME: ncnn.name="half_input"
// CHECK-SAME: ncnn.source_layer=12

// -----

func.func @rank5(%input: tensor<1x1x1x1x1xf32>) -> tensor<1x1x1x1x1xf32> attributes {ncnn.name = "rank5", ncnn.source_layer = 13 : i64} {
  return %input : tensor<1x1x1x1x1xf32>
}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: supports tensor ranks 1 through 4
// CHECK-SAME: ncnn.name="rank5"
// CHECK-SAME: ncnn.source_layer=13

// -----

func.func @zero_extent(%input: tensor<0xf32>) -> tensor<0xf32> attributes {ncnn.name = "zero_extent", ncnn.source_layer = 14 : i64} {
  return %input : tensor<0xf32>
}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: requires positive tensor dimensions
// CHECK-SAME: ncnn.name="zero_extent"
// CHECK-SAME: ncnn.source_layer=14

// -----

func.func @oversized(%input: tensor<33554433xf32>) -> tensor<33554433xf32> attributes {ncnn.name = "oversized", ncnn.source_layer = 13 : i64} {
  return %input : tensor<33554433xf32>
}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: tensor exceeds the Vulkan 1.1 minimum storage-buffer range
// CHECK-SAME: ncnn.name="oversized"
// CHECK-SAME: ncnn.source_layer=13

// -----

func.func @overflow(%input: tensor<2147483648x2147483648x4xf32>) -> tensor<2147483648x2147483648x4xf32> attributes {ncnn.name = "overflow", ncnn.source_layer = 14 : i64} {
  return %input : tensor<2147483648x2147483648x4xf32>
}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: tensor element count overflows uint64
// CHECK-SAME: ncnn.name="overflow"
// CHECK-SAME: ncnn.source_layer=14

// -----

func.func @unsupported_matmul(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x4xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>) outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32> loc("matmul"("ncnn-layer":14:0))
  return %result : tensor<2x4xf32>
}

// CHECK: error: 'linalg.matmul' op is not supported by Vulkan V1
// CHECK-SAME: ncnn.name="matmul"
// CHECK-SAME: ncnn.source_layer=14

// -----

func.func private @external_entry(%input: tensor<4xf32>) -> tensor<4xf32> attributes {ncnn.name = "external_entry", ncnn.source_layer = 15 : i64}

// CHECK: error: 'func.func' op is not supported by Vulkan V1: function declarations are not supported
// CHECK-SAME: ncnn.name="external_entry"
// CHECK-SAME: ncnn.source_layer=15

// -----

func.func @returns_empty(%input: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  return %empty : tensor<4xf32>
}

// CHECK: error: 'tensor.empty' op is not supported by Vulkan V1: tensor.empty may only initialize a written output

// -----

func.func @reads_empty(%input: tensor<4xf32>) -> tensor<4xf32> {
  %source = tensor.empty() : tensor<4xf32>
  %destination = tensor.empty() : tensor<4xf32>
  %copied = linalg.copy ins(%source : tensor<4xf32>) outs(%destination : tensor<4xf32>) -> tensor<4xf32>
  return %copied : tensor<4xf32>
}

// CHECK: error: 'tensor.empty' op is not supported by Vulkan V1: tensor.empty may only initialize a written output

// -----

func.func @negative_zero_relu(%input: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant -0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%value: f32, %output: f32):
    %relu = arith.maximumf %value, %zero : f32
    linalg.yield %relu : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK: error: 'linalg.generic' op is not supported by Vulkan V1: ReLU must compute max(input, 0.0)

// -----

module {}

// CHECK: error: Vulkan V1 legality requires a model function with inputs and outputs
