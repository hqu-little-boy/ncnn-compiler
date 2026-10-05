// RUN: ncnn-mlir-opt --lower-ncnn-vulkan-graph="plan-path=%t.plan.json weights-path=%t.weights.json" %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan.json
// RUN: FileCheck %s --check-prefix=WEIGHTS < %t.weights.json

func.func @conv_chain(%input: tensor<1x5x7x2xf32>) -> (tensor<1x5x7x3xf32>, tensor<1x5x7x4xf32>) {
  %w1 = arith.constant dense<0.25> : tensor<3x3x3x2xf32>
  %b1 = arith.constant dense<0.125> : tensor<3xf32>
  %w2 = arith.constant dense<-0.125> : tensor<4x1x1x3xf32>
  %b2 = arith.constant dense<0.0> : tensor<4xf32>
  %zero = arith.constant dense<0.0> : tensor<1xf32>
  %first = "tosa.conv2d"(%input, %w1, %b1, %zero, %zero) <{pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = f32, local_bound = false}> : (tensor<1x5x7x2xf32>, tensor<3x3x3x2xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x7x3xf32>
  %relu = "tosa.clamp"(%first) <{min_val = 0.0 : f32, max_val = 0x7F800000 : f32, nan_mode = "PROPAGATE"}> : (tensor<1x5x7x3xf32>) -> tensor<1x5x7x3xf32>
  %second = "tosa.conv2d"(%relu, %w2, %b2, %zero, %zero) <{pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = f32, local_bound = false}> : (tensor<1x5x7x3xf32>, tensor<4x1x1x3xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x7x4xf32>
  return %first, %second : tensor<1x5x7x3xf32>, tensor<1x5x7x4xf32>
}

// IR: gpu.module @conv_chain_vulkan_k0
// IR: arith.cmpi ult
// IR: scf.for
// IR: arith.mulf
// IR: arith.addf
// IR: gpu.module @conv_chain_vulkan_k1
// IR: arith.maximumf
// IR: gpu.module @conv_chain_vulkan_k2
// IR: scf.for
// IR: gpu.module @conv_chain_vulkan_k3
// IR: gpu.module @conv_chain_vulkan_k4
// IR-NOT: tosa.
// IR-NOT: tensor.
// IR-NOT: func.func

// PLAN: "operation": "conv2d"
// PLAN: "operation": "relu"
// PLAN: "operation": "conv2d"
// PLAN: "logical_layout": "OIHW"
// PLAN: "physical_layout": "OHWI"
// PLAN: "weight_sha256":
// PLAN: "logical_layout": "OIHW"
// PLAN: "physical_layout": "OHWI"
// PLAN: "schema_version": 1
// WEIGHTS: "kind": "ncnn.vulkan_weight_payloads"
// WEIGHTS: "bytes":
// WEIGHTS: "resource_id":
// WEIGHTS: "bytes":
// WEIGHTS: "resource_id":
