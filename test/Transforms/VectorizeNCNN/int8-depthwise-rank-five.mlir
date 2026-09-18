// RUN: ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline='vector-lanes=4 vector-tail=true int8-depthwise=true' %s | FileCheck %s
// CHECK-LABEL: func.func @rank_five_result
// CHECK: scf.forall
// CHECK: arith.extsi
// CHECK: ncnn.kernel = "depthwise_simd"
// CHECK: return
module {
  func.func @rank_five_result(%a: tensor<1x5x6x5xi8>,
                             %b: tensor<3x2x5x1xi8>,
                             %c: tensor<1x3x5x5x1xi32>) -> tensor<1x3x5x5x1xi32> {
    %r = linalg.depthwise_conv_2d_nhwc_hwcm
      {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x5x6x5xi8>, tensor<3x2x5x1xi8>)
      outs(%c : tensor<1x3x5x5x1xi32>) -> tensor<1x3x5x5x1xi32>
    return %r : tensor<1x3x5x5x1xi32>
  }
}
