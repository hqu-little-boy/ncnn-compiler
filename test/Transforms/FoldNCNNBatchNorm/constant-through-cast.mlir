// RUN: ncnn-mlir-opt --fold-ncnn-batchnorm %s | FileCheck %s

// Constant lookup must see through pure views (tensor.cast here) so that a
// batchnorm whose parameters arrive wrapped in importer-produced casts is still
// folded into the preceding convolution.  tensor.cast is the view the importer
// emits when a static parameter is carried through a dynamic-shaped region.

func.func @folds_through_cast_views(%arg0: tensor<3x8x8xf32>) -> tensor<2x6x6xf32> {
  %weight = arith.constant dense<1.0> : tensor<2x3x3x3xf32>
  %conv_bias = arith.constant dense<0.0> : tensor<2xf32>
  %slope = arith.constant dense<1.0> : tensor<2xf32>
  %mean = arith.constant dense<0.0> : tensor<2xf32>
  %variance = arith.constant dense<1.0> : tensor<2xf32>
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %slope_dyn = tensor.cast %slope : tensor<2xf32> to tensor<?xf32>
  %slope_view = tensor.cast %slope_dyn : tensor<?xf32> to tensor<2xf32>
  %mean_dyn = tensor.cast %mean : tensor<2xf32> to tensor<?xf32>
  %mean_view = tensor.cast %mean_dyn : tensor<?xf32> to tensor<2xf32>
  %variance_dyn = tensor.cast %variance : tensor<2xf32> to tensor<?xf32>
  %variance_view = tensor.cast %variance_dyn : tensor<?xf32> to tensor<2xf32>
  %bias_dyn = tensor.cast %bias : tensor<2xf32> to tensor<?xf32>
  %bias_view = tensor.cast %bias_dyn : tensor<?xf32> to tensor<2xf32>
  %weight_dyn = tensor.cast %weight : tensor<2x3x3x3xf32> to tensor<?x?x?x?xf32>
  %weight_view = tensor.cast %weight_dyn : tensor<?x?x?x?xf32> to tensor<2x3x3x3xf32>
  %conv = ncnn.convolution %arg0, %weight_view, %conv_bias {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x8x8xf32>, tensor<2x3x3x3xf32>, tensor<2xf32>) -> tensor<2x6x6xf32>
  %normalized = ncnn.batch_norm %conv, %slope_view, %mean_view, %variance_view, %bias_view {epsilon = 1.000000e-05 : f32} : (tensor<2x6x6xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x6x6xf32>
  return %normalized : tensor<2x6x6xf32>
}

// CHECK-LABEL: func.func @folds_through_cast_views
// CHECK-NOT: ncnn.batch_norm
// The folded weight is scaled by (variance + eps)^-0.5, so anchoring on that
// value pins the *folded* constant rather than the original dense<1.0>.
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<0.99999{{.*}}> : tensor<2x3x3x3xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: ncnn.convolution %arg0, %[[WEIGHT]], %[[BIAS]]
// CHECK-NOT: ncnn.batch_norm

// -----

// tensor.extract_slice is NOT a view: it selects a subset of elements, so the
// source constant does not hold the slice's contents.  Handing it to the folder
// would compute against the wrong parameters, so the lookup must stop here and
// the batchnorm must survive untouched.

func.func @does_not_cross_extract_slice(%arg0: tensor<3x8x8xf32>) -> tensor<2x6x6xf32> {
  %weight = arith.constant dense<1.0> : tensor<2x3x3x3xf32>
  %conv_bias = arith.constant dense<0.0> : tensor<2xf32>
  %slope_full = arith.constant dense<1.0> : tensor<4xf32>
  %slope = tensor.extract_slice %slope_full[0] [2] [1] : tensor<4xf32> to tensor<2xf32>
  %mean = arith.constant dense<0.0> : tensor<2xf32>
  %variance = arith.constant dense<1.0> : tensor<2xf32>
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %conv = ncnn.convolution %arg0, %weight, %conv_bias {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<3x8x8xf32>, tensor<2x3x3x3xf32>, tensor<2xf32>) -> tensor<2x6x6xf32>
  %normalized = ncnn.batch_norm %conv, %slope, %mean, %variance, %bias {epsilon = 1.000000e-05 : f32} : (tensor<2x6x6xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x6x6xf32>
  return %normalized : tensor<2x6x6xf32>
}

// CHECK-LABEL: func.func @does_not_cross_extract_slice
// CHECK: ncnn.batch_norm
