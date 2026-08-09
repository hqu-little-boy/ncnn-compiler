// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

func.func @dynamic_shape_preserving(%arg0: tensor<4x?x?xf32>) -> tensor<4x?x?xf32> {
  %slope = arith.constant dense<1.000000e+00> : tensor<4xf32>
  %mean = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %variance = arith.constant dense<1.000000e+00> : tensor<4xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %hard_sigmoid = ncnn.hard_sigmoid %arg0 {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %hard_swish = ncnn.hard_swish %hard_sigmoid {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %gelu = ncnn.gelu %hard_swish {fast = false} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %dropout = ncnn.dropout %gelu {scale = 5.000000e-01 : f32} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %softmax = ncnn.softmax %dropout {axis = 0 : i64} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %batch_norm = ncnn.batch_norm %softmax, %slope, %mean, %variance, %bias {epsilon = 1.000000e-05 : f32} : (tensor<4x?x?xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4x?x?xf32>
  %shuffle = ncnn.shuffle_channel %batch_norm {group = 2 : i64, reverse = false} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  return %shuffle : tensor<4x?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_shape_preserving
// CHECK: tosa.mul
// CHECK: tosa.clamp
// CHECK: tensor.empty
// CHECK: math.erfc
// CHECK: tosa.reduce_max
// CHECK: tosa.reduce_sum
// CHECK: tosa.pow
// CHECK: tensor.expand_shape
// CHECK: tosa.transpose
// CHECK: tensor.collapse_shape
// CHECK-NOT: ncnn.hard_sigmoid
// CHECK-NOT: ncnn.hard_swish
// CHECK-NOT: ncnn.gelu
// CHECK-NOT: ncnn.dropout
// CHECK-NOT: ncnn.softmax
// CHECK-NOT: ncnn.batch_norm
// CHECK-NOT: ncnn.shuffle_channel
