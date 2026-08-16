// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

func.func @reflection_padding(%input: tensor<1x4x5xf32>)
    -> tensor<1x8x9xf32> {
  %output = ncnn.padding %input {bottom = 2 : i64, left = 2 : i64,
    padding_type = 2 : i64, right = 2 : i64, top = 2 : i64,
    value = 0.0 : f32} : (tensor<1x4x5xf32>) -> tensor<1x8x9xf32>
  return %output : tensor<1x8x9xf32>
}

// CHECK-LABEL: func.func @reflection_padding
// CHECK: linalg.generic
// CHECK: arith.select
// CHECK-NOT: ncnn.padding

func.func @dynamic_grid_sample(
    %input: tensor<3x?x?xf32>, %grid: tensor<2x5x7xf32>)
    -> tensor<3x?x?xf32> {
  %resized = ncnn.interp %grid, %input {align_corner = true,
    height_scale = 1 : i64, resize_type = 2 : i64,
    width_scale = 1 : i64} : (tensor<2x5x7xf32>, tensor<3x?x?xf32>)
    -> tensor<2x?x?xf32>
  %output = ncnn.grid_sample %input, %resized {align_corner = true,
    padding_mode = 1 : i64, permute_fusion = true,
    sample_type = 1 : i64} : (tensor<3x?x?xf32>, tensor<2x?x?xf32>)
    -> tensor<3x?x?xf32>
  return %output : tensor<3x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_grid_sample
// CHECK: tensor.dim
// CHECK: linalg.generic
// CHECK: math.floor
// CHECK: arith.select
// CHECK-NOT: ncnn.interp
// CHECK-NOT: ncnn.grid_sample
