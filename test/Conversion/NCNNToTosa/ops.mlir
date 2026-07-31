// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

func.func @spatial(%arg0: tensor<2x5x5xf32>) -> tensor<6x2x2xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<3x2x3x3xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<3xf32>
  %conv = ncnn.convolution %arg0, %weight, %bias {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x5x5xf32>, tensor<3x2x3x3xf32>, tensor<3xf32>) -> tensor<3x3x3xf32>
  %relu = ncnn.relu %conv : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
  %pool = ncnn.pooling %relu {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x3x3xf32>) -> tensor<3x2x2xf32>
  %split0, %split1 = ncnn.split %pool : (tensor<3x2x2xf32>) -> (tensor<3x2x2xf32>, tensor<3x2x2xf32>)
  %concat = ncnn.concat %split0, %split1 {axis = 0 : i64} : (tensor<3x2x2xf32>, tensor<3x2x2xf32>) -> tensor<6x2x2xf32>
  %dropout = ncnn.dropout %concat : (tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
  return %dropout : tensor<6x2x2xf32>
}

// CHECK-LABEL: func.func @spatial
// CHECK: %[[INPUT_TRANSPOSE:.*]] = tosa.transpose %arg0 {perms = array<i32: 1, 2, 0>} : (tensor<2x5x5xf32>) -> tensor<5x5x2xf32>
// CHECK: %[[INPUT:.*]] = tosa.reshape %[[INPUT_TRANSPOSE]]
// CHECK-SAME: -> tensor<1x5x5x2xf32>
// CHECK: %[[WEIGHT:.*]] = tosa.transpose %{{.*}} {perms = array<i32: 0, 2, 3, 1>} : (tensor<3x2x3x3xf32>) -> tensor<3x3x3x2xf32>
// CHECK: %[[CONV:.*]] = tosa.conv2d %[[INPUT]], %[[WEIGHT]],
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
// CHECK-SAME: stride = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x3x3x3xf32>
// CHECK: %[[RELU:.*]] = tosa.clamp %[[CONV]]
// CHECK: %[[POOL:.*]] = tosa.max_pool2d %[[RELU]]
// CHECK-SAME: pad = array<i64: 0, 1, 0, 1>
// CHECK: %[[CONCAT:.*]] = tosa.concat %[[POOL]], %[[POOL]] {axis = 3 : i32}
// CHECK-NOT: ncnn.split
// CHECK-NOT: ncnn.dropout
// CHECK: %[[HWC:.*]] = tosa.reshape %[[CONCAT]]
// CHECK-SAME: -> tensor<2x2x6xf32>
// CHECK: %[[OUTPUT:.*]] = tosa.transpose %[[HWC]] {perms = array<i32: 2, 0, 1>}
// CHECK: return %[[OUTPUT]] : tensor<6x2x2xf32>

func.func @global_softmax(%arg0: tensor<4x2x2xf32>) -> tensor<4xf32> {
  %pool = ncnn.pooling %arg0 {include_pad = false, kernel_h = 0 : i64, kernel_w = 0 : i64, kind = 1 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x2x2xf32>) -> tensor<4xf32>
  %softmax = ncnn.softmax %pool {axis = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
  %scaled = ncnn.dropout %softmax {scale = 5.000000e-01 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %scaled : tensor<4xf32>
}

// CHECK-LABEL: func.func @global_softmax
// CHECK: %[[AVG:.*]] = tosa.avg_pool2d
// CHECK-SAME: kernel = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x1x1x4xf32>
// CHECK: %[[FLAT:.*]] = tosa.reshape %[[AVG]]
// CHECK-SAME: -> tensor<4xf32>
// CHECK: %[[MAX:.*]] = tosa.reduce_max %[[FLAT]] {axis = 0 : i32}
// CHECK: %[[SUB:.*]] = tosa.sub %[[FLAT]], %[[MAX]]
// CHECK: %[[EXP:.*]] = tosa.exp %[[SUB]]
// CHECK: %[[SUM:.*]] = tosa.reduce_sum %[[EXP]] {axis = 0 : i32}
// CHECK: %[[RECIP:.*]] = tosa.reciprocal %[[SUM]]
// CHECK: %[[SOFTMAX:.*]] = tosa.mul %[[EXP]], %[[RECIP]],
// CHECK: %[[SCALE:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1xf32>}>
// CHECK: %[[SCALED:.*]] = tosa.mul %[[SOFTMAX]], %[[SCALE]],
// CHECK: return %[[SCALED]] : tensor<4xf32>

func.func @leaky_relu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %relu = ncnn.relu %arg0 {negative_slope = 2.500000e-01 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %relu : tensor<4xf32>
}

// CHECK-LABEL: func.func @leaky_relu
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[SLOPE:.*]] = "tosa.const"() <{values = dense<2.500000e-01> : tensor<1xf32>}>
// CHECK: %[[NEGATIVE:.*]] = tosa.mul %arg0, %[[SLOPE]],
// CHECK: %[[CONDITION:.*]] = tosa.greater_equal %arg0, %[[ZERO]]
// CHECK: %[[LEAKY:.*]] = tosa.select %[[CONDITION]], %arg0, %[[NEGATIVE]]
// CHECK: return %[[LEAKY]] : tensor<4xf32>

// CHECK-NOT: ncnn.convolution
// CHECK-NOT: ncnn.relu
// CHECK-NOT: ncnn.pooling
// CHECK-NOT: ncnn.split
// CHECK-NOT: ncnn.concat
// CHECK-NOT: ncnn.dropout
// CHECK-NOT: ncnn.softmax
