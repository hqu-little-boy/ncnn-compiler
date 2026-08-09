// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @batch_norm_dynamic_channel(
    %input: tensor<?x?x?xf32>,
    %slope: tensor<4xf32>,
    %mean: tensor<4xf32>,
    %variance: tensor<4xf32>,
    %bias: tensor<4xf32>) {
  // expected-error@+2 {{BatchNorm requires a ranked f32 input with static positive channels}}
  // expected-error@+1 {{'ncnn.batch_norm' op failed to infer returned types}}
  %result = ncnn.batch_norm %input, %slope, %mean, %variance, %bias {epsilon = 1.000000e-05 : f32} : (tensor<?x?x?xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<?x?x?xf32>
  return
}

// -----

func.func @shuffle_channel_dynamic_channel(%input: tensor<?x?x?xf32>) {
  // expected-error@+1 {{input must be a positive-channel CHW f32 tensor with static channels}}
  %result = ncnn.shuffle_channel %input {group = 2 : i64, reverse = false} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return
}

// -----

func.func @shuffle_channel_invalid_group(%input: tensor<4x?x?xf32>) {
  // expected-error@+1 {{group must be positive and divide channel count}}
  %result = ncnn.shuffle_channel %input {group = 3 : i64, reverse = false} : (tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  return
}
