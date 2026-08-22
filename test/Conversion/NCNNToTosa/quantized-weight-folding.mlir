// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

// Constant f32 weights with an INT8 scale term are pre-quantized at compile
// time with the exact runtime semantics: scale, round half away from zero,
// clamp to [-127, 127], convert to i8. The OHWI transpose folds as well.

func.func @quantizes_constant_weight(%arg0: tensor<3x5x5xf32>) -> tensor<2x2x2xf32> {
  %weight = arith.constant dense<[[[[0.25, -0.25], [0.4999999, 100.0]], [[0.0, -200.0], [0.05, -0.05]], [[1.0, 1.0], [1.0, 1.0]]],
                                 [[[0.5, -0.5], [0.3, -0.3]], [[0.6, -0.6], [0.7, -0.7]], [[-1.0, -1.0], [-1.0, -1.0]]]]> : tensor<2x3x2x2xf32>
  %bias = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %wscale = arith.constant dense<[2.0, 4.0]> : tensor<2xf32>
  %iscale = arith.constant dense<[1.0]> : tensor<1xf32>
  %conv = ncnn.convolution %arg0, %weight, %bias, %wscale, %iscale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, int8_scale_term = 1 : i64, kernel_h = 2 : i64, kernel_w = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x5x5xf32>, tensor<2x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<1xf32>) -> tensor<2x2x2xf32>
  return %conv : tensor<2x2x2xf32>
}

// CHECK-LABEL: func.func @quantizes_constant_weight
// CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<1x5x5x3xf32>, tensor<1xf32>) outs({{.*}} : tensor<1x5x5x3xi8>)
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{.*}}> : tensor<2x3x2x2xi8>
// CHECK: tosa.conv2d {{.*}}, %{{.*}}, {{.*}} : (tensor<1x5x5x3xi8>, tensor<2x2x2x3xi8>

func.func @nonconstant_weight_keeps_runtime_quantize(%arg0: tensor<3x5x5xf32>, %weight: tensor<2x3x1x1xf32>) -> tensor<2x5x5xf32> {
  %bias = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %wscale = arith.constant dense<[2.0, 4.0]> : tensor<2xf32>
  %iscale = arith.constant dense<[1.0]> : tensor<1xf32>
  %conv = ncnn.convolution %arg0, %weight, %bias, %wscale, %iscale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, int8_scale_term = 1 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x5x5xf32>, tensor<2x3x1x1xf32>, tensor<2xf32>, tensor<2xf32>, tensor<1xf32>) -> tensor<2x5x5xf32>
  return %conv : tensor<2x5x5xf32>
}

// CHECK-LABEL: func.func @nonconstant_weight_keeps_runtime_quantize
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<2x3x1x1xf32>
