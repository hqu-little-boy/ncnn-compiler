// RUN: not ncnn-mlir-opt --split-input-file --lower-ncnn-vulkan-model="plan-path=%t-plan.json" %s 2>&1 | FileCheck %s
// RUN: test ! -e %t-plan.json

func.func @kernel_5x5(%input: tensor<1x7x7xf32>) -> tensor<1x3x3xf32> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 7, 7, 1] : tensor<1x7x7xf32> into tensor<1x7x7x1xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<5x5x1x1xf32>
  %init = tensor.empty() : tensor<1x3x3x1xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%expanded, %weights : tensor<1x7x7x1xf32>, tensor<5x5x1x1xf32>) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x3x3x1xf32> into tensor<3x3x1xf32>
  %out_init = tensor.empty() : tensor<1x3x3xf32>
  %result = linalg.transpose ins(%collapsed : tensor<3x3x1xf32>) outs(%out_init : tensor<1x3x3xf32>) permutation = [2, 0, 1]
  return %result : tensor<1x3x3xf32>
}

// CHECK: error: 'linalg.conv_2d_nhwc_hwcf' op is not supported by Vulkan V2.2: supports only 1x1 and 3x3 kernels

// -----

func.func @dilation_2(%input: tensor<1x7x7xf32>) -> tensor<1x3x3xf32> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 7, 7, 1] : tensor<1x7x7xf32> into tensor<1x7x7x1xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<3x3x1x1xf32>
  %init = tensor.empty() : tensor<1x3x3x1xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%expanded, %weights : tensor<1x7x7x1xf32>, tensor<3x3x1x1xf32>) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x3x3x1xf32> into tensor<3x3x1xf32>
  %out_init = tensor.empty() : tensor<1x3x3xf32>
  %result = linalg.transpose ins(%collapsed : tensor<3x3x1xf32>) outs(%out_init : tensor<1x3x3xf32>) permutation = [2, 0, 1]
  return %result : tensor<1x3x3xf32>
}

// CHECK: error: 'linalg.conv_2d_nhwc_hwcf' op is not supported by Vulkan V2.2: dilation must be one

// -----

func.func @stride_3(%input: tensor<1x7x7xf32>) -> tensor<1x2x2xf32> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 7, 7, 1] : tensor<1x7x7xf32> into tensor<1x7x7x1xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<3x3x1x1xf32>
  %init = tensor.empty() : tensor<1x2x2x1xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<3> : tensor<2xi64>} ins(%expanded, %weights : tensor<1x7x7x1xf32>, tensor<3x3x1x1xf32>) outs(%init : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x2x2x1xf32> into tensor<2x2x1xf32>
  %out_init = tensor.empty() : tensor<1x2x2xf32>
  %result = linalg.transpose ins(%collapsed : tensor<2x2x1xf32>) outs(%out_init : tensor<1x2x2xf32>) permutation = [2, 0, 1]
  return %result : tensor<1x2x2xf32>
}

// CHECK: error: 'linalg.conv_2d_nhwc_hwcf' op is not supported by Vulkan V2.2: stride must be one or two on each axis

// -----

func.func @padding_2(%input: tensor<1x4x4xf32>) -> tensor<1x6x6xf32> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 4, 4, 1] : tensor<1x4x4xf32> into tensor<1x4x4x1xf32>
  %zero = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %expanded low[0, 2, 2, 0] high[0, 2, 2, 0] {
  ^bb0(%n: index, %h: index, %w: index, %c: index):
    tensor.yield %zero : f32
  } : tensor<1x4x4x1xf32> to tensor<1x8x8x1xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<3x3x1x1xf32>
  %init = tensor.empty() : tensor<1x6x6x1xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %weights : tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>) outs(%init : tensor<1x6x6x1xf32>) -> tensor<1x6x6x1xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x6x6x1xf32> into tensor<6x6x1xf32>
  %out_init = tensor.empty() : tensor<1x6x6xf32>
  %result = linalg.transpose ins(%collapsed : tensor<6x6x1xf32>) outs(%out_init : tensor<1x6x6xf32>) permutation = [2, 0, 1]
  return %result : tensor<1x6x6xf32>
}

// CHECK: error: 'linalg.conv_2d_nhwc_hwcf' op is not supported by Vulkan V2.2: padding is outside the supported 1x1/3x3 subset

// -----

func.func @nonzero_padding(%input: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 4, 4, 1] : tensor<1x4x4xf32> into tensor<1x4x4x1xf32>
  %pad_value = arith.constant 1.000000e+00 : f32
  %padded = tensor.pad %expanded low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%n: index, %h: index, %w: index, %c: index):
    tensor.yield %pad_value : f32
  } : tensor<1x4x4x1xf32> to tensor<1x6x6x1xf32>
  %weights = arith.constant dense<1.000000e+00> : tensor<3x3x1x1xf32>
  %init = tensor.empty() : tensor<1x4x4x1xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %weights : tensor<1x6x6x1xf32>, tensor<3x3x1x1xf32>) outs(%init : tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x4x4x1xf32> into tensor<4x4x1xf32>
  %out_init = tensor.empty() : tensor<1x4x4xf32>
  %result = linalg.transpose ins(%collapsed : tensor<4x4x1xf32>) outs(%out_init : tensor<1x4x4xf32>) permutation = [2, 0, 1]
  return %result : tensor<1x4x4xf32>
}

// CHECK: error: 'linalg.conv_2d_nhwc_hwcf' op is not supported by Vulkan V2.2: padding value must be positive zero

// -----

func.func @fp16(%input: tensor<1x4x4xf16>) -> tensor<1x2x2xf16> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 4, 4, 1] : tensor<1x4x4xf16> into tensor<1x4x4x1xf16>
  %weights = arith.constant dense<1.000000e+00> : tensor<3x3x1x1xf16>
  %init = tensor.empty() : tensor<1x2x2x1xf16>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%expanded, %weights : tensor<1x4x4x1xf16>, tensor<3x3x1x1xf16>) outs(%init : tensor<1x2x2x1xf16>) -> tensor<1x2x2x1xf16>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x2x2x1xf16> into tensor<2x2x1xf16>
  %out_init = tensor.empty() : tensor<1x2x2xf16>
  %result = linalg.transpose ins(%collapsed : tensor<2x2x1xf16>) outs(%out_init : tensor<1x2x2xf16>) permutation = [2, 0, 1]
  return %result : tensor<1x2x2xf16>
}

// CHECK: error: 'linalg.conv_2d_nhwc_hwcf' op is not supported by Vulkan V2.2: requires static rank-three FP32 CHW input/output

// -----

func.func @dynamic_weight(%input: tensor<1x4x4xf32>, %weights: tensor<3x3x1x1xf32>) -> tensor<1x2x2xf32> {
  %expanded = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [1, 4, 4, 1] : tensor<1x4x4xf32> into tensor<1x4x4x1xf32>
  %init = tensor.empty() : tensor<1x2x2x1xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%expanded, %weights : tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>) outs(%init : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  %collapsed = tensor.collapse_shape %conv [[0, 1], [2], [3]] : tensor<1x2x2x1xf32> into tensor<2x2x1xf32>
  %out_init = tensor.empty() : tensor<1x2x2xf32>
  %result = linalg.transpose ins(%collapsed : tensor<2x2x1xf32>) outs(%out_init : tensor<1x2x2xf32>) permutation = [2, 0, 1]
  return %result : tensor<1x2x2xf32>
}

// CHECK: error: Vulkan V2.2 Conv2D requires one model input and one output

// -----

func.func @depthwise(%input: tensor<1x7x9x4xf32>) -> tensor<1x3x7x4xf32> {
  %weights = arith.constant dense<0.5> : tensor<3x3x4x1xf32>
  %empty = tensor.empty() : tensor<1x3x7x4x1xf32>
  %conv = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} ins(%input, %weights : tensor<1x7x9x4xf32>, tensor<3x3x4x1xf32>) outs(%empty : tensor<1x3x7x4x1xf32>) -> tensor<1x3x7x4x1xf32>
  %result = tensor.collapse_shape %conv [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x1xf32> into tensor<1x3x7x4xf32>
  return %result : tensor<1x3x7x4xf32>
}

// CHECK: error: Vulkan V2 reshape requires one direct function input
