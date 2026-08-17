// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s --check-prefix=LOWERING

func.func @quantized_convolution(%input: tensor<2x3x3xf32>) -> tensor<3x3x3xf32> {
  %weight = arith.constant dense<1> : tensor<3x2x1x1xi8>
  %bias = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %weight_scale = arith.constant dense<[2.0, 4.0, 8.0]> : tensor<3xf32>
  %input_scale = arith.constant dense<2.0> : tensor<1xf32>
  %0 = ncnn.convolution %input, %weight, %bias, %weight_scale, %input_scale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, int8_scale_term = 1 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x3x3xf32>, tensor<3x2x1x1xi8>, tensor<3xf32>, tensor<3xf32>, tensor<1xf32>) -> tensor<3x3x3xf32>
  return %0 : tensor<3x3x3xf32>
}

// LOWERING-LABEL: func.func @quantized_convolution
// LOWERING: linalg.generic
// LOWERING: math.floor
// LOWERING: math.ceil
// LOWERING: arith.maximumf
// LOWERING: arith.fptosi {{.*}} : f32 to i8
// LOWERING: %[[INPUT_ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// LOWERING: %[[WEIGHT_ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// LOWERING: %[[BIAS:.*]] = "tosa.const"() <{values = dense<0> : tensor<3xi32>}>
// LOWERING: %[[CONV:.*]] = tosa.conv2d {{.*}}, %[[BIAS]], %[[INPUT_ZERO]], %[[WEIGHT_ZERO]]
// LOWERING-SAME: acc_type = i32
// LOWERING-SAME: -> tensor<1x3x3x3xi32>
// LOWERING: tosa.reciprocal
// LOWERING: linalg.map { arith.sitofp }
// LOWERING: tosa.add
// LOWERING-NOT: ncnn.convolution

func.func @requantized_convolution(%input: tensor<2x3x3xi8>) -> tensor<3x3x3xi8> {
  %weight = arith.constant dense<1> : tensor<3x2x1x1xi8>
  %weight_scale = arith.constant dense<[2.0, 4.0, 8.0]> : tensor<3xf32>
  %input_scale = arith.constant dense<2.0> : tensor<1xf32>
  %output_scale = arith.constant dense<4.0> : tensor<1xf32>
  %0 = ncnn.convolution %input, %weight, %weight_scale, %input_scale, %output_scale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, int8_scale_term = 101 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x3x3xi8>, tensor<3x2x1x1xi8>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x3x3xi8>
  return %0 : tensor<3x3x3xi8>
}

// LOWERING-LABEL: func.func @requantized_convolution
// LOWERING: tosa.conv2d
// LOWERING: arith.sitofp
// LOWERING: arith.mulf
// LOWERING: math.floor
// LOWERING: math.ceil
// LOWERING: arith.constant -1.270000e+02 : f32
// LOWERING: arith.constant 1.270000e+02 : f32
// LOWERING: arith.fptosi {{.*}} : f32 to i8
// LOWERING: return {{.*}} : tensor<3x3x3xi8>

func.func @quantized_depthwise(%input: tensor<2x3x3xf32>) -> tensor<2x3x3xf32> {
  %weight = arith.constant dense<1.0> : tensor<2x1x1x1xf32>
  %weight_scale = arith.constant dense<4.0> : tensor<1xf32>
  %input_scale = arith.constant dense<2.0> : tensor<1xf32>
  %0 = ncnn.convolution_depthwise %input, %weight, %weight_scale, %input_scale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, int8_scale_term = 2 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x3x3xf32>, tensor<2x1x1x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x3x3xf32>
  return %0 : tensor<2x3x3xf32>
}

// LOWERING-LABEL: func.func @quantized_depthwise
// LOWERING: linalg.generic
// LOWERING: arith.fptosi
// LOWERING: tosa.depthwise_conv2d
// LOWERING-SAME: acc_type = i32
// LOWERING-SAME: -> tensor<1x3x3x2xi32>
// LOWERING: tosa.reciprocal
// LOWERING: arith.sitofp
// LOWERING-NOT: ncnn.convolution_depthwise

func.func @quantized_inner_product_dynamic(%input: tensor<?x4xf32>) -> tensor<?x3xf32> {
  %weight = arith.constant dense<1> : tensor<3x4xi8>
  %bias = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %weight_scale = arith.constant dense<[2.0, 4.0, 8.0]> : tensor<3xf32>
  %input_scale = arith.constant dense<2.0> : tensor<1xf32>
  %0 = ncnn.inner_product %input, %weight, %bias, %weight_scale, %input_scale {has_bias = true, int8_scale_term = 1 : i64} : (tensor<?x4xf32>, tensor<3x4xi8>, tensor<3xf32>, tensor<3xf32>, tensor<1xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// LOWERING-LABEL: func.func @quantized_inner_product_dynamic
// LOWERING: tensor.dim
// LOWERING: tensor.expand_shape {{.*}} output_shape [1, %{{.*}}, 4]
// LOWERING: %[[MATMUL:.*]] = tosa.matmul
// LOWERING-SAME: -> tensor<1x?x3xi32>
// LOWERING: tosa.reciprocal
// LOWERING: arith.sitofp
// LOWERING: tosa.add
// LOWERING: tensor.collapse_shape
// LOWERING-NOT: ncnn.inner_product

func.func @quantized_convolution_dynamic(%input: tensor<2x?x?xf32>) -> tensor<3x?x?xf32> {
  %weight = arith.constant dense<1> : tensor<3x2x3x3xi8>
  %weight_scale = arith.constant dense<[2.0, 4.0, 8.0]> : tensor<3xf32>
  %input_scale = arith.constant dense<2.0> : tensor<1xf32>
  %0 = ncnn.convolution %input, %weight, %weight_scale, %input_scale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, int8_scale_term = 2 : i64, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x?x?xf32>, tensor<3x2x3x3xi8>, tensor<3xf32>, tensor<1xf32>) -> tensor<3x?x?xf32>
  return %0 : tensor<3x?x?xf32>
}

// LOWERING-LABEL: func.func @quantized_convolution_dynamic
// LOWERING: tosa.pad
// LOWERING: arith.divui
// LOWERING: linalg.fill
// LOWERING: linalg.conv_2d_nhwc_hwcf
// LOWERING: arith.sitofp
// LOWERING-NOT: tosa.conv2d
// LOWERING-NOT: ncnn.convolution

func.func @quantized_depthwise_dynamic(%input: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> {
  %weight = arith.constant dense<1> : tensor<2x1x3x3xi8>
  %weight_scale = arith.constant dense<2.0> : tensor<1xf32>
  %input_scale = arith.constant dense<2.0> : tensor<1xf32>
  %0 = ncnn.convolution_depthwise %input, %weight, %weight_scale, %input_scale {dilation_h = 1 : i64, dilation_w = 1 : i64, group = 2 : i64, has_bias = false, int8_scale_term = 2 : i64, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x?x?xf32>, tensor<2x1x3x3xi8>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x?x?xf32>
  return %0 : tensor<2x?x?xf32>
}

// LOWERING-LABEL: func.func @quantized_depthwise_dynamic
// LOWERING: tosa.pad
// LOWERING: arith.divui
// LOWERING: linalg.depthwise_conv_2d_nhwc_hwcm
// LOWERING: tensor.collapse_shape
// LOWERING: arith.sitofp
// LOWERING-NOT: tosa.depthwise_conv2d
// LOWERING-NOT: ncnn.convolution_depthwise
