// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @dynamic_attention(%input: tensor<?x4xf32>, %weight: tensor<4x4xf32>, %bias: tensor<4xf32>) {
  // expected-error@+2 {{MultiHeadAttention requires a static positive sequence dimension}}
  // expected-error@+1 {{'ncnn.multi_head_attention' op failed to infer returned types}}
  %0 = ncnn.multi_head_attention %input, %weight, %bias, %weight, %bias, %weight, %bias, %weight, %bias {embed_dim = 4 : i64, kdim = 4 : i64, num_heads = 2 : i64, qdim = 4 : i64, scale = 0.5 : f32, vdim = 4 : i64, weight_data_size = 16 : i64} : (tensor<?x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
  return
}

// -----

func.func @empty_attention(%input: tensor<0x4xf32>, %weight: tensor<4x4xf32>, %bias: tensor<4xf32>) {
  // expected-error@+2 {{MultiHeadAttention requires a static positive sequence dimension}}
  // expected-error@+1 {{'ncnn.multi_head_attention' op failed to infer returned types}}
  %0 = ncnn.multi_head_attention %input, %weight, %bias, %weight, %bias, %weight, %bias, %weight, %bias {embed_dim = 4 : i64, kdim = 4 : i64, num_heads = 2 : i64, qdim = 4 : i64, scale = 0.5 : f32, vdim = 4 : i64, weight_data_size = 16 : i64} : (tensor<0x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>) -> tensor<0x4xf32>
  return
}

// -----

func.func @bad_layer_norm_affine(%input: tensor<2x4xf32>, %gamma: tensor<4xf32>) {
  // expected-error@+2 {{LayerNorm affine mode requires exactly gamma and beta}}
  // expected-error@+1 {{'ncnn.layer_norm' op failed to infer returned types}}
  %0 = ncnn.layer_norm %input, %gamma {affine = true, affine_size = 4 : i64, epsilon = 1.0e-5 : f32} : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return
}

// -----

func.func @bad_layer_norm_size(%input: tensor<2x4xf32>) {
  // expected-error@+2 {{LayerNorm affine_size must equal the last input dimension}}
  // expected-error@+1 {{'ncnn.layer_norm' op failed to infer returned types}}
  %0 = ncnn.layer_norm %input {affine = false, affine_size = 3 : i64, epsilon = 1.0e-5 : f32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return
}
