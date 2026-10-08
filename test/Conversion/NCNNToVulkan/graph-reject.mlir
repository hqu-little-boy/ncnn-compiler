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

// -----

func.func @max_pool_unsupported_padding(%input: tensor<1x57x56x1xf32>) -> tensor<1x29x28x1xf32> {
  %pool = "tosa.max_pool2d"(%input) <{kernel = array<i64: 3, 3>, nan_mode = "PROPAGATE", pad = array<i64: 1, 1, 0, 1>, stride = array<i64: 2, 2>}> : (tensor<1x57x56x1xf32>) -> tensor<1x29x28x1xf32> loc("max_pool_parameter"("ncnn-layer":13:0))
  return %pool : tensor<1x29x28x1xf32>
}
// CHECK: is not supported by Vulkan graph: max_pool2d requires zero top and left padding
// CHECK-SAME: ncnn.name="max_pool_parameter", ncnn.source_layer=13

// -----

func.func @regular_average_pool(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %pool = "tosa.avg_pool2d"(%input, %zero, %zero) <{acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x4x4x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x2x1xf32>
  return %pool : tensor<1x2x2x1xf32>
}
// CHECK: is not supported by Vulkan graph: avg_pool2d requires a full-spatial kernel and 1x1 output

// -----

func.func @average_pool_nonzero_zero_point(%input: tensor<1x4x4x1xf32>) -> tensor<1x1x1x1xf32> {
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %nonzero = arith.constant dense<1.0> : tensor<1xf32>
  %pool = "tosa.avg_pool2d"(%input, %nonzero, %zero) <{acc_type = f32, kernel = array<i64: 4, 4>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x4x4x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1x1xf32>
  return %pool : tensor<1x1x1x1xf32>
}
// CHECK: error: 'tosa.avg_pool2d' op input zero point must be zero for non-int8 integer types

// -----

func.func @mul_nonzero_shift(%input: tensor<8xf32>) -> tensor<8xf32> {
  %one = arith.constant dense<1> : tensor<1xi8>
  %result = "tosa.mul"(%input, %input, %one) : (tensor<8xf32>, tensor<8xf32>, tensor<1xi8>) -> tensor<8xf32> loc("mul_shift"("ncnn-layer":17:0))
  return %result : tensor<8xf32>
}
// CHECK: ncnn-layer:17:0: error: 'tosa.mul' op require shift to be 0 for float type

// -----

func.func @concat_bad_axis(%lhs: tensor<1x2x2x1xf32>, %rhs: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32> {
  %out = "tosa.concat"(%lhs, %rhs) <{axis = 4 : i32}> : (tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %out : tensor<1x2x2x1xf32>
}
// CHECK: error: 'tosa.concat' op expect axis to be within range 0 < axis < rank(input1[firstRankedTensorIdx]), got 4

// -----

func.func @concat_mismatched_non_axis(%lhs: tensor<1x2x2x1xf32>, %rhs: tensor<1x2x3x1xf32>) -> tensor<1x4x2x1xf32> {
  %out = "tosa.concat"(%lhs, %rhs) <{axis = 1 : i32}> : (tensor<1x2x2x1xf32>, tensor<1x2x3x1xf32>) -> tensor<1x4x2x1xf32>
  return %out : tensor<1x4x2x1xf32>
}
// CHECK: error: 'tosa.concat' op expect all operand shapes to have the same sizes on non-axis dimensions

// -----

func.func @sub_crossed_broadcast(%lhs: tensor<2x1xf32>, %rhs: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %out = "tosa.sub"(%lhs, %rhs) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32> loc("sub_crossed"("ncnn-layer":18:0))
  return %out : tensor<2x3xf32>
}
// CHECK: is not supported by Vulkan graph: broadcast may expand at most one tensor axis
// CHECK-SAME: ncnn.name="sub_crossed", ncnn.source_layer=18

// -----

func.func @mul_crossed_broadcast(%lhs: tensor<2x1xf32>, %rhs: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %zero = arith.constant dense<0> : tensor<1xi8>
  %out = "tosa.mul"(%lhs, %rhs, %zero) : (tensor<2x1xf32>, tensor<1x3xf32>, tensor<1xi8>) -> tensor<2x3xf32> loc("mul_crossed"("ncnn-layer":19:0))
  return %out : tensor<2x3xf32>
}
// CHECK: is not supported by Vulkan graph: broadcast may expand at most one tensor axis
// CHECK-SAME: ncnn.name="mul_crossed", ncnn.source_layer=19

// -----

func.func @sub_one_operand_multiaxis(%lhs: tensor<1x1x3xf32>, %rhs: tensor<2x4x3xf32>) -> tensor<2x4x3xf32> {
  %out = "tosa.sub"(%lhs, %rhs) : (tensor<1x1x3xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32> loc("sub_multiaxis"("ncnn-layer":20:0))
  return %out : tensor<2x4x3xf32>
}
// CHECK: is not supported by Vulkan graph: broadcast may expand at most one tensor axis
// CHECK-SAME: ncnn.name="sub_multiaxis", ncnn.source_layer=20

// -----

func.func @max_pool_ignore_nan(%input: tensor<1x57x57x1xf32>) -> tensor<1x28x28x1xf32> {
  %pool = "tosa.max_pool2d"(%input) <{kernel = array<i64: 3, 3>, nan_mode = "IGNORE", pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x57x57x1xf32>) -> tensor<1x28x28x1xf32> loc("max_pool_nan"("ncnn-layer":21:0))
  return %pool : tensor<1x28x28x1xf32>
}
// CHECK: is not supported by Vulkan graph: max_pool2d requires propagating NaN semantics
// CHECK-SAME: ncnn.name="max_pool_nan", ncnn.source_layer=21

// -----

func.func @max_pool_bad_kernel(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %pool = "tosa.max_pool2d"(%input) <{kernel = array<i64: 2, 2>, nan_mode = "PROPAGATE", pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32>
  return %pool : tensor<1x2x2x1xf32>
}
// CHECK: is not supported by Vulkan graph: max_pool2d supports only a 3x3 kernel

// -----

func.func @max_pool_bad_stride(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %pool = "tosa.max_pool2d"(%input) <{kernel = array<i64: 3, 3>, nan_mode = "PROPAGATE", pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32>
  return %pool : tensor<1x2x2x1xf32>
}
// CHECK: is not supported by Vulkan graph: max_pool2d supports only stride 2x2

// -----

// Shape tensor constants are compile-time metadata but this static-graph path
// currently only admits zero integer metadata, not arbitrary index vectors.
func.func @nonzero_index_shape_metadata(%input: tensor<512xf32>) -> tensor<512xf32> {
  %shape = arith.constant dense<[512, 1, 1]> : tensor<3xindex>
  return %input : tensor<512xf32>
}
// CHECK: is not supported by Vulkan graph: non-FP32 constants must be scalar zero metadata
// CHECK-SAME: ncnn.name=<missing>, ncnn.source_layer=<missing>

// -----

func.func @slice_out_of_bounds(%input: tensor<4x4xf32>) -> tensor<2x2xf32> {
  %start = "tosa.const_shape"() <{values = dense<[3, 3]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %size = "tosa.const_shape"() <{values = dense<[2, 2]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %out = "tosa.slice"(%input, %start, %size) : (tensor<4x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}
// CHECK: is not supported by Vulkan graph: slice is outside the source tensor

// -----

func.func @duplicate_transpose_axis(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %out = "tosa.transpose"(%input) <{perms = array<i32: 0, 0>}> : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %out : tensor<2x3xf32>
}
// CHECK: 'tosa.transpose' op expected valid permutation indices
