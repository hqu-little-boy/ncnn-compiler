// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

func.func @quantize(%input: tensor<2x3xf32>) -> tensor<2x3xi8> {
  %scale = arith.constant dense<[2.0, 4.0]> : tensor<2xf32>
  %result = ncnn.quantize %input, %scale : (tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x3xi8>
  return %result : tensor<2x3xi8>
}

// CHECK-LABEL: func.func @quantize
// CHECK: linalg.generic
// CHECK: math.floor
// CHECK: math.ceil
// CHECK: arith.constant -1.270000e+02 : f32
// CHECK: arith.constant 1.270000e+02 : f32
// CHECK: arith.fptosi {{.*}} : f32 to i8

func.func @dequantize(%input: tensor<2x3xi32>) -> tensor<2x3xf32> {
  %scale = arith.constant dense<[0.5, 0.25]> : tensor<2xf32>
  %bias = arith.constant dense<[1.0, -1.0]> : tensor<2xf32>
  %result = ncnn.dequantize %input, %scale, %bias : (tensor<2x3xi32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @dequantize
// CHECK: linalg.generic
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: arith.addf

func.func @requantize(%input: tensor<2x3xi32>) -> tensor<2x3xi8> {
  %input_scale = arith.constant dense<[0.5, 0.25]> : tensor<2xf32>
  %output_scale = arith.constant dense<2.0> : tensor<1xf32>
  %result = ncnn.requantize %input, %input_scale, %output_scale {activation_type = 1 : i64} : (tensor<2x3xi32>, tensor<2xf32>, tensor<1xf32>) -> tensor<2x3xi8>
  return %result : tensor<2x3xi8>
}

// CHECK-LABEL: func.func @requantize
// CHECK: arith.sitofp
// CHECK: tosa.clamp
// CHECK: arith.fptosi {{.*}} : f32 to i8

func.func @cast_round_trip(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %half = ncnn.cast %input {type_from = 1 : i64, type_to = 2 : i64} : (tensor<2x3xf32>) -> tensor<2x3xf16>
  %result = ncnn.cast %half {type_from = 2 : i64, type_to = 1 : i64} : (tensor<2x3xf16>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @cast_round_trip
// CHECK: arith.truncf
// CHECK: arith.extf

func.func @scale_preserving_i8(%left: tensor<2x4x4xi8>, %right: tensor<2x4x4xi8>) -> tensor<4x2x2xi8> {
  %maximum = ncnn.binary %left, %right {op_type = 4 : i64, scalar = 0.0 : f32, with_scalar = false} : (tensor<2x4x4xi8>, tensor<2x4x4xi8>) -> tensor<2x4x4xi8>
  %pooled = ncnn.pooling %maximum {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x4x4xi8>) -> tensor<2x2x2xi8>
  %result = ncnn.concat %pooled, %pooled {axis = 0 : i64} : (tensor<2x2x2xi8>, tensor<2x2x2xi8>) -> tensor<4x2x2xi8>
  return %result : tensor<4x2x2xi8>
}

// CHECK-LABEL: func.func @scale_preserving_i8
// CHECK: tosa.maximum
// CHECK: tosa.max_pool2d
// CHECK: tosa.concat
// CHECK-NOT: ncnn.

func.func @quantized_gemm(%input: tensor<?x4xf32>) -> tensor<?x3xf32> {
  %weight = arith.constant dense<1> : tensor<3x4xi8>
  %bias = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %weight_scale = arith.constant dense<4.0> : tensor<1xf32>
  %result = ncnn.gemm %input, %weight, %bias, %weight_scale {alpha = 1.0 : f32, beta = 1.0 : f32, int8_scale_term = 2 : i64} : (tensor<?x4xf32>, tensor<3x4xi8>, tensor<3xf32>, tensor<1xf32>) -> tensor<?x3xf32>
  return %result : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @quantized_gemm
// CHECK: linalg.generic
// CHECK: math.absf
// CHECK: arith.maximumf
// CHECK: arith.divf
// CHECK: arith.fptosi {{.*}} : f32 to i8
// CHECK: tosa.matmul
// CHECK-SAME: -> tensor<1x?x3xi32>
// CHECK: arith.sitofp
// CHECK: arith.divf
// CHECK: tosa.add

func.func @unsigned_boundary(%input: tensor<4xui8>) -> tensor<4xi8> {
  %result = ncnn.zero_point_cast %input {to_unsigned = false, zero_point = 128 : i64} : (tensor<4xui8>) -> tensor<4xi8>
  return %result : tensor<4xi8>
}

// CHECK-LABEL: func.func @unsigned_boundary
// CHECK: arith.extui
// CHECK: arith.subi
// CHECK: arith.minsi
// CHECK: arith.maxsi
// CHECK: arith.trunci
