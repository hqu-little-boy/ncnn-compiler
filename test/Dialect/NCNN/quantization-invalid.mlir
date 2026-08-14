// RUN: ncnn-mlir-opt --verify-diagnostics --split-input-file %s

func.func @implicit_unsigned_to_signed(%input: tensor<4xui8>) {
  // expected-error@+2 {{Cast requires explicit valid source and destination types}}
  // expected-error@+1 {{'ncnn.cast' op failed to infer returned types}}
  %result = ncnn.cast %input {type_from = 3 : i64, type_to = 1 : i64} : (tensor<4xui8>) -> tensor<4xf32>
  return
}

// -----

func.func @integer_average_pool(%input: tensor<2x4x4xi8>) {
  // expected-error@+2 {{signed i8 pooling only supports maximum}}
  // expected-error@+1 {{'ncnn.pooling' op failed to infer returned types}}
  %result = ncnn.pooling %input {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 1 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x4x4xi8>) -> tensor<2x2x2xi8>
  return
}

// -----

func.func @integer_binary_add(%left: tensor<2x4xi8>, %right: tensor<2x4xi8>) {
  // expected-error@+2 {{signed i8 BinaryOp only supports two-input maximum}}
  // expected-error@+1 {{'ncnn.binary' op failed to infer returned types}}
  %result = ncnn.binary %left, %right {op_type = 0 : i64, scalar = 0.0 : f32, with_scalar = false} : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>
  return
}

// -----

func.func @invalid_zero_point(%input: tensor<4xui8>) {
  // expected-error@+2 {{ZeroPointCast requires an 8-bit zero point}}
  // expected-error@+1 {{'ncnn.zero_point_cast' op failed to infer returned types}}
  %result = ncnn.zero_point_cast %input {to_unsigned = false, zero_point = 256 : i64} : (tensor<4xui8>) -> tensor<4xi8>
  return
}
