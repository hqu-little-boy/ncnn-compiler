// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s --check-prefix=NCNN
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline | FileCheck %s --check-prefix=LINALG

module {
  func.func @dynamic_lowering(%arg0: tensor<32x?x?xf32>) -> tensor<8x?x?xf32> {
    %depthwise_weight = arith.constant dense<0.000000e+00> : tensor<32x1x3x3xf32>
    %depthwise = ncnn.convolution_depthwise %arg0, %depthwise_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<32x?x?xf32>, tensor<32x1x3x3xf32>) -> tensor<32x?x?xf32>
    %pool = ncnn.pooling %depthwise {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 1 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<32x?x?xf32>) -> tensor<32x?x?xf32>
    %deconv_weight = arith.constant dense<0.000000e+00> : tensor<8x32x2x2xf32>
    %deconv = ncnn.deconvolution %pool, %deconv_weight {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = false, kernel_h = 2 : i64, kernel_w = 2 : i64, output_pad_bottom = 0 : i64, output_pad_right = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<32x?x?xf32>, tensor<8x32x2x2xf32>) -> tensor<8x?x?xf32>
    %sigmoid = ncnn.sigmoid %deconv : (tensor<8x?x?xf32>) -> tensor<8x?x?xf32>
    return %sigmoid : tensor<8x?x?xf32>
  }
}

// NCNN-LABEL: func.func @dynamic_lowering
// NCNN: tosa.depthwise_conv2d
// NCNN: tensor.dim
// NCNN: linalg.generic
// NCNN: scf.for
// NCNN: arith.muli
// NCNN: tensor.empty
// NCNN: linalg.generic
// NCNN-NOT: tosa.transpose_conv2d
// NCNN: tosa.sigmoid

// LINALG-LABEL: func.func @dynamic_lowering
// LINALG: linalg.depthwise_conv_2d_nhwc_hwcm
// LINALG: linalg.generic
// LINALG: scf.for
// LINALG-NOT: tosa.
