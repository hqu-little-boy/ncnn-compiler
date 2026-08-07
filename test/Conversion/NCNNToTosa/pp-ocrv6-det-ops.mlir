// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

func.func @padding(%arg0: tensor<3x4x5xf32>) -> tensor<3x7x12xf32> {
  %result = "ncnn.padding"(%arg0) {top = 1 : i64, bottom = 2 : i64, left = 3 : i64, right = 4 : i64, value = -3.40282347E+38 : f32} : (tensor<3x4x5xf32>) -> tensor<3x7x12xf32>
  return %result : tensor<3x7x12xf32>
}

// CHECK-LABEL: func.func @padding
// CHECK: %[[PAD_VALUE:.*]] = "tosa.const"() <{values = dense<-3.40282347E+38> : tensor<1xf32>}>
// CHECK: %[[PAD:.*]] = tosa.pad {{.*}}, {{.*}}, %[[PAD_VALUE]]
// CHECK-SAME: -> tensor<1x7x12x3xf32>
// CHECK: return

func.func @interp_scales(%arg0: tensor<4x2x3xf32>) -> (tensor<4x4x6xf32>, tensor<4x8x12xf32>, tensor<4x16x24xf32>) {
  %x2 = "ncnn.interp"(%arg0) {height_scale = 2 : i64, width_scale = 2 : i64} : (tensor<4x2x3xf32>) -> tensor<4x4x6xf32>
  %x4 = "ncnn.interp"(%arg0) {height_scale = 4 : i64, width_scale = 4 : i64} : (tensor<4x2x3xf32>) -> tensor<4x8x12xf32>
  %x8 = "ncnn.interp"(%arg0) {height_scale = 8 : i64, width_scale = 8 : i64} : (tensor<4x2x3xf32>) -> tensor<4x16x24xf32>
  return %x2, %x4, %x8 : tensor<4x4x6xf32>, tensor<4x8x12xf32>, tensor<4x16x24xf32>
}

// CHECK-LABEL: func.func @interp_scales
// CHECK: %[[SCALE2:.*]] = tosa.const_shape {values = dense<[2, 1, 2, 1]>
// CHECK: %[[OFFSET2:.*]] = tosa.const_shape {values = dense<-1>
// CHECK: %[[BORDER2:.*]] = tosa.const_shape {values = dense<0>
// CHECK: tosa.resize {{.*}}, %[[SCALE2]], %[[OFFSET2]], %[[BORDER2]] {mode = "NEAREST_NEIGHBOR"}
// CHECK-SAME: -> tensor<1x4x6x4xf32>
// CHECK: %[[SCALE4:.*]] = tosa.const_shape {values = dense<[4, 1, 4, 1]>
// CHECK: %[[OFFSET4:.*]] = tosa.const_shape {values = dense<-2>
// CHECK: %[[BORDER4:.*]] = tosa.const_shape {values = dense<1>
// CHECK: tosa.resize {{.*}}, %[[SCALE4]], %[[OFFSET4]], %[[BORDER4]] {mode = "NEAREST_NEIGHBOR"}
// CHECK-SAME: -> tensor<1x8x12x4xf32>
// CHECK: %[[SCALE8:.*]] = tosa.const_shape {values = dense<[8, 1, 8, 1]>
// CHECK: %[[OFFSET8:.*]] = tosa.const_shape {values = dense<-4>
// CHECK: %[[BORDER8:.*]] = tosa.const_shape {values = dense<3>
// CHECK: tosa.resize {{.*}}, %[[SCALE8]], %[[OFFSET8]], %[[BORDER8]] {mode = "NEAREST_NEIGHBOR"}
// CHECK-SAME: -> tensor<1x16x24x4xf32>

func.func @deconvolution_bias_relu(%arg0: tensor<2x3x4xf32>) -> tensor<3x6x10xf32> {
  %weight = arith.constant dense<1.000000e+00> : tensor<3x2x3x3xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<3xf32>
  %result = "ncnn.deconvolution"(%arg0, %weight, %bias) {kernel_h = 3 : i64, kernel_w = 3 : i64, stride_h = 2 : i64, stride_w = 3 : i64, dilation_h = 1 : i64, dilation_w = 1 : i64, pad_top = 1 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, output_pad_bottom = 1 : i64, output_pad_right = 0 : i64, has_bias = true, activation_type = 1 : i64} : (tensor<2x3x4xf32>, tensor<3x2x3x3xf32>, tensor<3xf32>) -> tensor<3x6x10xf32>
  return %result : tensor<3x6x10xf32>
}

// CHECK-LABEL: func.func @deconvolution_bias_relu
// CHECK: %[[WEIGHT:.*]] = tosa.transpose %{{.*}} {perms = array<i32: 0, 2, 3, 1>}
// CHECK-SAME: -> tensor<3x3x3x2xf32>
// CHECK: %[[DECONV:.*]] = tosa.transpose_conv2d %{{[^,]+}}, %[[WEIGHT]], %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}} {
// CHECK-SAME: out_pad = array<i64: -1, 0, -1, -1>
// CHECK-SAME: stride = array<i64: 2, 3>
// CHECK-SAME: -> tensor<1x6x10x3xf32>
// CHECK: %[[RELU:.*]] = tosa.clamp %[[DECONV]]
// CHECK: return

func.func @deconvolution_no_bias(%arg0: tensor<2x2x2xf32>) -> tensor<4x4x4xf32> {
  %weight = arith.constant dense<1.000000e+00> : tensor<4x2x2x2xf32>
  %result = "ncnn.deconvolution"(%arg0, %weight) {kernel_h = 2 : i64, kernel_w = 2 : i64, stride_h = 2 : i64, stride_w = 2 : i64, dilation_h = 1 : i64, dilation_w = 1 : i64, pad_top = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, has_bias = false, activation_type = 0 : i64} : (tensor<2x2x2xf32>, tensor<4x2x2x2xf32>) -> tensor<4x4x4xf32>
  return %result : tensor<4x4x4xf32>
}

// CHECK-LABEL: func.func @deconvolution_no_bias
// CHECK: %[[ZERO_BIAS:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}>
// CHECK: tosa.transpose_conv2d %{{[^,]+}}, %{{[^,]+}}, %[[ZERO_BIAS]],
// CHECK-SAME: out_pad = array<i64: 0, 0, 0, 0>
// CHECK-SAME: -> tensor<1x4x4x4xf32>

func.func @sigmoid(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %result = "ncnn.sigmoid"(%arg0) : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %result : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @sigmoid
// CHECK: %[[CLAMPED:.*]] = tosa.clamp {{.*}} {max_val = 88.3762589 : f32, min_val = -88.3762589 : f32}
// CHECK: %[[SIGMOID:.*]] = tosa.sigmoid %[[CLAMPED]] : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: return
// CHECK-NOT: ncnn.padding
// CHECK-NOT: ncnn.interp
// CHECK-NOT: ncnn.deconvolution
// CHECK-NOT: ncnn.sigmoid
