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

// -----

// Extracted from ResNet18's normal NCNN-to-TOSA output: convrelu_0 is the
// first computational convolution after static input-shape metadata is known.
func.func @resnet18_first_blocked_conv(%input: tensor<1x224x224x3xf32>) -> tensor<1x113x113x64xf32> {
  %filter = arith.constant dense<0.0> : tensor<64x7x7x3xf32>
  %bias = arith.constant dense<0.0> : tensor<64xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %output = "tosa.conv2d"(%input, %filter, %bias, %zero, %zero) <{acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 3, 4, 3, 4>, stride = array<i64: 2, 2>}> : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x113x113x64xf32> loc("convrelu_0"("ncnn-layer":1:0))
  return %output : tensor<1x113x113x64xf32>
}
// CHECK: is not supported by Vulkan graph: conv2d supports only 1x1 and 3x3 kernels
// CHECK-SAME: ncnn.name="convrelu_0", ncnn.source_layer=1
