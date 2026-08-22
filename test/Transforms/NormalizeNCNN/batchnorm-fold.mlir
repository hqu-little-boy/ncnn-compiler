// RUN: ncnn-mlir-opt --normalize-ncnn %s | FileCheck %s

// Constant BatchNorm parameters fold into the preceding convolution: the
// channel scale is slope * (variance + eps)^-0.5 (with the 10000 fallback when
// the variance term is zero), and the folded bias absorbs the original bias.

func.func @conv_bn(%arg0: tensor<3x5x5xf32>) -> tensor<2x5x5xf32> {
  %weight = arith.constant dense<[[[[0.5]], [[1.0]], [[2.0]]], [[[4.0]], [[8.0]], [[16.0]]]]> : tensor<2x3x1x1xf32>
  %conv_bias = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %slope = arith.constant dense<[2.0, 1.0]> : tensor<2xf32>
  %mean = arith.constant dense<[3.0, 0.0]> : tensor<2xf32>
  %variance = arith.constant dense<[1.0, 4.0]> : tensor<2xf32>
  %bn_bias = arith.constant dense<[0.5, -1.0]> : tensor<2xf32>
  %conv = ncnn.convolution %arg0, %weight, %conv_bias {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, int8_scale_term = 0 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x5x5xf32>, tensor<2x3x1x1xf32>, tensor<2xf32>) -> tensor<2x5x5xf32>
  %bn = ncnn.batch_norm %conv, %slope, %mean, %variance, %bn_bias {epsilon = 0.000000e+00 : f32} : (tensor<2x5x5xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x5x5xf32>
  return %bn : tensor<2x5x5xf32>
}

// CHECK-LABEL: func.func @conv_bn
// CHECK-NOT: ncnn.batch_norm
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{\[\[\[\[1\.000000e\+00\]\], \[\[2\.000000e\+00\]\], \[\[4\.000000e\+00\]\]\], \[\[\[2\.000000e\+00\]\], \[\[4\.000000e\+00\]\], \[\[8\.000000e\+00\]\]\]\]}}> : tensor<2x3x1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<[-3.500000e+00, 0.000000e+00]> : tensor<2xf32>
// CHECK: ncnn.convolution {{.*}}, %[[WEIGHT]], %[[BIAS]]
// CHECK-SAME: has_bias = true

func.func @depthwise_bn_no_bias(%arg0: tensor<2x6x6xf32>) -> tensor<2x6x6xf32> {
  %weight = arith.constant dense<[[[[2.0]]], [[[4.0]]]]> : tensor<2x1x1x1xf32>
  %slope = arith.constant dense<[1.0, 1.0]> : tensor<2xf32>
  %mean = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %variance = arith.constant dense<[1.0, 0.0]> : tensor<2xf32>
  %bn_bias = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %dw = "ncnn.convolution_depthwise"(%arg0, %weight) {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, int8_scale_term = 0 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x6x6xf32>, tensor<2x1x1x1xf32>) -> tensor<2x6x6xf32>
  %bn = ncnn.batch_norm %dw, %slope, %mean, %variance, %bn_bias {epsilon = 0.000000e+00 : f32} : (tensor<2x6x6xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x6x6xf32>
  return %bn : tensor<2x6x6xf32>
}

// CHECK-LABEL: func.func @depthwise_bn_no_bias
// CHECK-NOT: ncnn.batch_norm
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<{{.*2\.000000e\+00.*4\.000000e\+04.*}}> : tensor<2x1x1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK: ncnn.convolution_depthwise {{.*}}, %[[WEIGHT]], %[[BIAS]]
// CHECK-SAME: has_bias = true

// Non-constant BN parameters keep the explicit batch norm.
func.func @dynamic_bn_keeps_batch_norm(%arg0: tensor<3x5x5xf32>) -> tensor<2x5x5xf32> {
  %weight = arith.constant dense<[[[[0.5]], [[1.0]], [[2.0]]], [[[4.0]], [[8.0]], [[16.0]]]]> : tensor<2x3x1x1xf32>
  %conv_bias = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %param = arith.constant dense<[1.0, 1.0]> : tensor<2xf32>
  %pslope = ncnn.relu %param : (tensor<2xf32>) -> tensor<2xf32>
  %pmean = ncnn.relu %param : (tensor<2xf32>) -> tensor<2xf32>
  %pvariance = ncnn.relu %param : (tensor<2xf32>) -> tensor<2xf32>
  %pbias = ncnn.relu %param : (tensor<2xf32>) -> tensor<2xf32>
  %conv = ncnn.convolution %arg0, %weight, %conv_bias {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, int8_scale_term = 0 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x5x5xf32>, tensor<2x3x1x1xf32>, tensor<2xf32>) -> tensor<2x5x5xf32>
  %bn = ncnn.batch_norm %conv, %pslope, %pmean, %pvariance, %pbias {epsilon = 1.000000e-05 : f32} : (tensor<2x5x5xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x5x5xf32>
  return %bn : tensor<2x5x5xf32>
}

// CHECK-LABEL: func.func @dynamic_bn_keeps_batch_norm
// CHECK: ncnn.batch_norm

// Quantized convolutions keep the explicit batch norm.
func.func @quantized_conv_keeps_batch_norm(%arg0: tensor<3x5x5xf32>) -> tensor<2x5x5xf32> {
  %weight = arith.constant dense<[[[[0.5]], [[1.0]], [[2.0]]], [[[4.0]], [[8.0]], [[16.0]]]]> : tensor<2x3x1x1xf32>
  %conv_bias = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %wscale = arith.constant dense<[1.0, 1.0]> : tensor<2xf32>
  %iscale = arith.constant dense<[1.0]> : tensor<1xf32>
  %slope = arith.constant dense<[1.0, 1.0]> : tensor<2xf32>
  %mean = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %variance = arith.constant dense<[1.0, 1.0]> : tensor<2xf32>
  %bn_bias = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %conv = ncnn.convolution %arg0, %weight, %conv_bias, %wscale, %iscale {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, int8_scale_term = 1 : i64, kernel_h = 1 : i64, kernel_w = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x5x5xf32>, tensor<2x3x1x1xf32>, tensor<2xf32>, tensor<2xf32>, tensor<1xf32>) -> tensor<2x5x5xf32>
  %bn = ncnn.batch_norm %conv, %slope, %mean, %variance, %bn_bias {epsilon = 1.000000e-05 : f32} : (tensor<2x5x5xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x5x5xf32>
  return %bn : tensor<2x5x5xf32>
}

// CHECK-LABEL: func.func @quantized_conv_keeps_batch_norm
// CHECK: ncnn.batch_norm
