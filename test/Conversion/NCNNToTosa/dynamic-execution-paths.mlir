// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s --check-prefix=NCNN
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline | FileCheck %s --check-prefix=LINALG

module {
  func.func @concat_and_max(%lhs: tensor<2x?x4xf32>, %rhs: tensor<2x?x4xf32>, %bias: tensor<2x1x4xf32>) -> tensor<2x?x4xf32> {
    %joined = ncnn.concat %lhs, %rhs {axis = -2 : i64} : (tensor<2x?x4xf32>, tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
    %maximum = "ncnn.binary"(%joined, %bias) {op_type = 4 : i64, scalar = 0.000000e+00 : f32, with_scalar = false} : (tensor<2x?x4xf32>, tensor<2x1x4xf32>) -> tensor<2x?x4xf32>
    return %maximum : tensor<2x?x4xf32>
  }

  func.func @reshape_spec(%arg0: tensor<?x6xf32>) -> tensor<?x3xf32> {
    %0 = "ncnn.reshape"(%arg0) {shape = array<i64: -1, 3>, shape_spec = array<i64: -1, 3>, shape_zero_sources = array<i64: -1, -1>} : (tensor<?x6xf32>) -> tensor<?x3xf32>
    return %0 : tensor<?x3xf32>
  }

  func.func @ordered_slice(%arg0: tensor<?x4xf32>) -> (tensor<?x4xf32>, tensor<2x4xf32>, tensor<?x4xf32>) attributes {ncnn.shape_constraints = [#ncnn.dim_constraint<input = 0, dim = 0, min = 4, multiple_of = 1>]} {
    %0, %1, %2 = ncnn.slice %arg0 {axis = 0 : i64, slices = array<i64: -233, 2, -233>} : (tensor<?x4xf32>) -> (tensor<?x4xf32>, tensor<2x4xf32>, tensor<?x4xf32>)
    return %0, %1, %2 : tensor<?x4xf32>, tensor<2x4xf32>, tensor<?x4xf32>
  }

  func.func @global_pool(%arg0: tensor<3x?x?xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
    %max = ncnn.pooling %arg0 {include_pad = false, kernel_h = 1 : i64, kernel_w = 1 : i64, kind = 0 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>) -> tensor<3xf32>
    %avg = ncnn.pooling %arg0 {include_pad = false, kernel_h = 1 : i64, kernel_w = 1 : i64, kind = 1 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>) -> tensor<3xf32>
    return %max, %avg : tensor<3xf32>, tensor<3xf32>
  }

  func.func @adaptive_pool(%arg0: tensor<3x?x?xf32>) -> (tensor<3x2x3xf32>, tensor<3x2x3xf32>) {
    %max = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 3 : i64, kind = 0 : i64, mode = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>) -> tensor<3x2x3xf32>
    %avg = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 3 : i64, kind = 1 : i64, mode = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>) -> tensor<3x2x3xf32>
    return %max, %avg : tensor<3x2x3xf32>, tensor<3x2x3xf32>
  }

  func.func @dynamic_channel_pool(%arg0: tensor<?x?x?xf32>) -> (tensor<?xf32>, tensor<?x2x3xf32>) {
    %global = ncnn.pooling %arg0 {include_pad = false, kernel_h = 1 : i64, kernel_w = 1 : i64, kind = 0 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<?x?x?xf32>) -> tensor<?xf32>
    %adaptive = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 3 : i64, kind = 0 : i64, mode = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<?x?x?xf32>) -> tensor<?x2x3xf32>
    return %global, %adaptive : tensor<?xf32>, tensor<?x2x3xf32>
  }

  func.func @adaptive_pool_passthrough(%arg0: tensor<3x?x?xf32>) -> tensor<3x2x?xf32> {
    %adaptive = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = -233 : i64, kind = 0 : i64, mode = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x?x?xf32>) -> tensor<3x2x?xf32>
    return %adaptive : tensor<3x2x?xf32>
  }

  func.func @inner_product_dynamic_m(%arg0: tensor<?x4xf32>, %weight: tensor<3x4xf32>, %bias: tensor<3xf32>) -> tensor<?x3xf32> {
    %0 = "ncnn.inner_product"(%arg0, %weight, %bias) {has_bias = true} : (tensor<?x4xf32>, tensor<3x4xf32>, tensor<3xf32>) -> tensor<?x3xf32>
    return %0 : tensor<?x3xf32>
  }
}

// NCNN-LABEL: func.func @concat_and_max
// NCNN: tosa.concat
// NCNN: tosa.maximum
// NCNN-LABEL: func.func @reshape_spec
// NCNN: arith.divui
// NCNN: tensor.from_elements
// NCNN: tensor.reshape
// NCNN-LABEL: func.func @ordered_slice
// NCNN-COUNT-3: tensor.extract_slice
// NCNN-LABEL: func.func @global_pool
// NCNN: linalg.generic
// NCNN: scf.for
// NCNN-LABEL: func.func @adaptive_pool
// NCNN: arith.index_castui {{.*}} : index to i128
// NCNN: arith.muli {{.*}} : i128
// NCNN: arith.divui
// NCNN: linalg.generic
// NCNN-LABEL: func.func @dynamic_channel_pool
// NCNN-COUNT-2: tensor.empty
// NCNN-LABEL: func.func @adaptive_pool_passthrough
// NCNN: linalg.index 2
// NCNN: arith.constant 1 : index
// NCNN: arith.addi
// NCNN: arith.index_castui {{.*}} : index to i128
// NCNN: arith.divui {{.*}} : i128
// NCNN-LABEL: func.func @inner_product_dynamic_m
// NCNN: tensor.expand_shape
// NCNN: tosa.matmul
// NCNN: tensor.collapse_shape

// LINALG-LABEL: func.func @concat_and_max
// LINALG: tensor.insert_slice
// LINALG: arith.maximumf
// LINALG-LABEL: func.func @reshape_spec
// LINALG: arith.divui
// LINALG-LABEL: func.func @ordered_slice
// LINALG-COUNT-3: tensor.extract_slice
// LINALG-LABEL: func.func @global_pool
// LINALG: scf.for
// LINALG-LABEL: func.func @adaptive_pool
// LINALG: scf.for
// LINALG-LABEL: func.func @dynamic_channel_pool
// LINALG-COUNT-2: linalg.generic
// LINALG-LABEL: func.func @adaptive_pool_passthrough
// LINALG: scf.for
// LINALG-LABEL: func.func @inner_product_dynamic_m
// LINALG: linalg.batch_matmul
// LINALG-NOT: tosa.
