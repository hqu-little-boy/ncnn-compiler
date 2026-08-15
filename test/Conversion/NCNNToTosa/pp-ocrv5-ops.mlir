// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

func.func @swish_layer_norm(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %gamma = arith.constant dense<1.0> : tensor<4xf32>
  %beta = arith.constant dense<0.0> : tensor<4xf32>
  %normalized = ncnn.layer_norm %input, %gamma, %beta {affine = true, affine_size = 4 : i64, epsilon = 1.0e-5 : f32} : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  %result = ncnn.swish %normalized : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @swish_layer_norm
// CHECK: tosa.reduce_sum {{.*}} {axis = 1 : i32}
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK: tosa.reduce_sum {{.*}} {axis = 1 : i32}
// CHECK: tosa.pow
// CHECK: tosa.sigmoid
// CHECK: tosa.mul
// CHECK-NOT: ncnn.layer_norm
// CHECK-NOT: ncnn.swish

func.func @layer_norm_without_affine(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %result = ncnn.layer_norm %input {affine = false, affine_size = 4 : i64, epsilon = 1.0e-5 : f32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @layer_norm_without_affine
// CHECK: tosa.reduce_sum
// CHECK: tosa.pow
// CHECK-NOT: ncnn.layer_norm

func.func @permute_hcw(%input: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  %result = ncnn.permute %input {permutation = array<i64: 2, 0, 1>} : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %result : tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @permute_hcw
// CHECK: tosa.transpose {{.*}} {perms = array<i32: 2, 0, 1>} : (tensor<3x4x2xf32>) -> tensor<2x3x4xf32>
// CHECK: tosa.transpose {{.*}} {perms = array<i32: 2, 0, 1>} : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
// CHECK: tosa.transpose {{.*}} {perms = array<i32: 1, 2, 0>} : (tensor<4x2x3xf32>) -> tensor<2x3x4xf32>
// CHECK-NOT: ncnn.permute

func.func @self_attention(%input: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %qw = arith.constant dense<0.0> : tensor<4x4xf32>
  %qb = arith.constant dense<0.0> : tensor<4xf32>
  %kw = arith.constant dense<0.0> : tensor<4x4xf32>
  %kb = arith.constant dense<0.0> : tensor<4xf32>
  %vw = arith.constant dense<0.0> : tensor<4x4xf32>
  %vb = arith.constant dense<0.0> : tensor<4xf32>
  %ow = arith.constant dense<0.0> : tensor<4x4xf32>
  %ob = arith.constant dense<0.0> : tensor<4xf32>
  %result = ncnn.multi_head_attention %input, %qw, %qb, %kw, %kb, %vw, %vb, %ow, %ob {embed_dim = 4 : i64, kdim = 4 : i64, num_heads = 2 : i64, qdim = 4 : i64, scale = 0.5 : f32, vdim = 4 : i64, weight_data_size = 16 : i64} : (tensor<3x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %result : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @self_attention
// CHECK-COUNT-3: tosa.matmul
// CHECK: tosa.matmul
// CHECK: tosa.reduce_max {{.*}} {axis = 2 : i32}
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum {{.*}} {axis = 2 : i32}
// CHECK: tosa.reciprocal
// CHECK-COUNT-2: tosa.matmul
// CHECK-NOT: ncnn.multi_head_attention

func.func @dynamic_self_attention(%input: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %qw = arith.constant dense<0.0> : tensor<4x4xf32>
  %qb = arith.constant dense<0.0> : tensor<4xf32>
  %kw = arith.constant dense<0.0> : tensor<4x4xf32>
  %kb = arith.constant dense<0.0> : tensor<4xf32>
  %vw = arith.constant dense<0.0> : tensor<4x4xf32>
  %vb = arith.constant dense<0.0> : tensor<4xf32>
  %ow = arith.constant dense<0.0> : tensor<4x4xf32>
  %ob = arith.constant dense<0.0> : tensor<4xf32>
  %result = ncnn.multi_head_attention %input, %qw, %qb, %kw, %kb, %vw, %vb, %ow, %ob {embed_dim = 4 : i64, kdim = 4 : i64, num_heads = 2 : i64, qdim = 4 : i64, scale = 0.5 : f32, vdim = 4 : i64, weight_data_size = 16 : i64} : (tensor<?x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
  return %result : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @dynamic_self_attention
// CHECK: tensor.dim
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: tensor.collapse_shape
// CHECK-NOT: tosa.matmul
// CHECK-NOT: ncnn.multi_head_attention
