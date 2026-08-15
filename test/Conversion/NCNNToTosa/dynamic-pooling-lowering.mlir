// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

module {
  func.func @dynamic_pooling_lowering(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
    %0 = ncnn.pooling %arg0 {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
    return %0 : tensor<4x?x?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_pooling_lowering
// CHECK: tensor.dim
// CHECK: linalg.generic
// CHECK: scf.for
// CHECK-NOT: tosa.max_pool2d

func.func @dynamic_average_pooling(%arg0: tensor<4x3x?xf32>) -> tensor<4x1x?xf32> {
  %0 = ncnn.pooling %arg0 {include_pad = false, kernel_h = 3 : i64, kernel_w = 2 : i64, kind = 1 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 3 : i64, stride_w = 2 : i64} : (tensor<4x3x?xf32>) -> tensor<4x1x?xf32>
  return %0 : tensor<4x1x?xf32>
}

// CHECK-LABEL: func.func @dynamic_average_pooling
// CHECK: linalg.generic
// CHECK: arith.divf
// CHECK-NOT: tosa.avg_pool2d

func.func @dynamic_padded_max_pooling(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
  %0 = ncnn.pooling %arg0 {include_pad = false, kernel_h = 3 : i64, kernel_w = 3 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_mode = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  return %0 : tensor<4x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_padded_max_pooling
// CHECK: arith.subi
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: scf.for

func.func @dynamic_padded_average_excluding_pad(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
  %0 = ncnn.pooling %arg0 {include_pad = false, kernel_h = 3 : i64, kernel_w = 3 : i64, kind = 1 : i64, mode = 0 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_mode = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  return %0 : tensor<4x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_padded_average_excluding_pad
// CHECK: arith.subi
// CHECK: arith.index_cast
// CHECK: arith.divf

func.func @dynamic_padded_average_including_pad(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
  %0 = ncnn.pooling %arg0 {include_pad = true, kernel_h = 3 : i64, kernel_w = 3 : i64, kind = 1 : i64, mode = 0 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_mode = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  return %0 : tensor<4x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_padded_average_including_pad
// CHECK: arith.constant 3 : index
// CHECK: arith.index_cast
// CHECK: arith.divf
