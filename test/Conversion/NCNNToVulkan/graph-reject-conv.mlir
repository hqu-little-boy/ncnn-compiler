// RUN: not ncnn-mlir-opt --split-input-file --lower-ncnn-vulkan-graph="plan-path=%t.plan.json weights-path=%t.weights.json" %s 2>&1 | FileCheck %s
// RUN: test ! -e %t.plan.json
// RUN: test ! -e %t.weights.json

func.func @kernel5(%input: tensor<1x7x7x1xf32>) -> tensor<1x3x3x2xf32> {
  %weight = arith.constant dense<0.25> : tensor<2x5x5x1xf32>
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %result = "tosa.conv2d"(%input, %weight, %bias, %zero, %zero) <{pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = f32, local_bound = false}> : (tensor<1x7x7x1xf32>, tensor<2x5x5x1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x2xf32> loc("kernel5"("ncnn-layer":2:0))
  return %result : tensor<1x3x3x2xf32>
}
// CHECK: is not supported by Vulkan graph: conv2d supports only 1x1 and 3x3 kernels
// CHECK-SAME: ncnn.name="kernel5", ncnn.source_layer=2

// -----

func.func @dilation2(%input: tensor<1x7x7x1xf32>) -> tensor<1x3x3x2xf32> {
  %weight = arith.constant dense<0.25> : tensor<2x3x3x1xf32>
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %result = "tosa.conv2d"(%input, %weight, %bias, %zero, %zero) <{pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 2, 2>, acc_type = f32, local_bound = false}> : (tensor<1x7x7x1xf32>, tensor<2x3x3x1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x2xf32>
  return %result : tensor<1x3x3x2xf32>
}
// CHECK: is not supported by Vulkan graph: conv2d requires static padding/stride and dilation one

// -----

func.func @stride3(%input: tensor<1x6x6x1xf32>) -> tensor<1x2x2x2xf32> {
  %weight = arith.constant dense<0.25> : tensor<2x3x3x1xf32>
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %result = "tosa.conv2d"(%input, %weight, %bias, %zero, %zero) <{pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 3>, dilation = array<i64: 1, 1>, acc_type = f32, local_bound = false}> : (tensor<1x6x6x1xf32>, tensor<2x3x3x1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x2x2xf32>
  return %result : tensor<1x2x2x2xf32>
}
// CHECK: is not supported by Vulkan graph: conv2d stride must be one or two

// -----

func.func @padding2(%input: tensor<1x5x5x1xf32>) -> tensor<1x7x7x2xf32> {
  %weight = arith.constant dense<0.25> : tensor<2x3x3x1xf32>
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %result = "tosa.conv2d"(%input, %weight, %bias, %zero, %zero) <{pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = f32, local_bound = false}> : (tensor<1x5x5x1xf32>, tensor<2x3x3x1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x2xf32>
  return %result : tensor<1x7x7x2xf32>
}
// CHECK: is not supported by Vulkan graph: conv2d padding is outside the supported 1x1/3x3 subset

// -----

func.func @dynamic_weight(%input: tensor<1x5x5x1xf32>, %weight: tensor<2x3x3x1xf32>) -> tensor<1x3x3x2xf32> {
  %bias = arith.constant dense<0.0> : tensor<2xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %result = "tosa.conv2d"(%input, %weight, %bias, %zero, %zero) <{pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = f32, local_bound = false}> : (tensor<1x5x5x1xf32>, tensor<2x3x3x1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x2xf32>
  return %result : tensor<1x3x3x2xf32>
}
// CHECK: is not supported by Vulkan graph: conv2d weights/bias must be constant and zero points must be zero
