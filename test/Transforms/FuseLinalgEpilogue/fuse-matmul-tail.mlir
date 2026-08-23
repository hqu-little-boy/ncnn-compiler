#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s

// Matmul epilogues are tiled along the column axis; a remainder tile that does
// not fill the last block is emitted as straight-line code after the loop.

func.func @fuses_matmul_relu_with_tail(%arg0: tensor<8x16xf32>, %arg1: tensor<16x40xf32>) -> tensor<8x40xf32> {
  %empty = tensor.empty() : tensor<8x40xf32>
  %matmul = linalg.matmul ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x40xf32>) outs(%empty : tensor<8x40xf32>) -> tensor<8x40xf32>
  %zero = arith.constant 0.0 : f32
  %relu_out = tensor.empty() : tensor<8x40xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%matmul : tensor<8x40xf32>) outs(%relu_out : tensor<8x40xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<8x40xf32>
  return %relu : tensor<8x40xf32>
}

// CHECK-LABEL: func.func @fuses_matmul_relu_with_tail
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<8x40xf32>)
// CHECK: linalg.matmul
// CHECK: arith.maximumf
// CHECK: scf.yield
// CHECK: tensor.extract_slice {{.*}} tensor<8x40xf32> to tensor<8x8xf32>
// CHECK: linalg.matmul
// CHECK: tensor.insert_slice {{.*}} into tensor<8x40xf32>
// CHECK-NOT: outs({{.*}} : tensor<8x40xf32>)
// CHECK: return

func.func @fuses_conv_leaky_relu(%arg0: tensor<1x8x18x3xf32>) -> tensor<1x6x16x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %empty = tensor.empty() : tensor<1x6x16x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x18x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x6x16x4xf32>) -> tensor<1x6x16x4xf32>
  %zero = arith.constant 0.0 : f32
  %slope = arith.constant 0.1 : f32
  %out = tensor.empty() : tensor<1x6x16x4xf32>
  %leaky = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x6x16x4xf32>) outs(%out : tensor<1x6x16x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    %scaled = arith.mulf %in, %slope : f32
    %cond = arith.cmpf ogt, %in, %zero : f32
    %sel = arith.select %cond, %in, %scaled : f32
    linalg.yield %sel : f32
  } -> tensor<1x6x16x4xf32>
  return %leaky : tensor<1x6x16x4xf32>
}

// CHECK-LABEL: func.func @fuses_conv_leaky_relu
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: arith.select
// CHECK-NOT: linalg.generic

func.func @fuses_conv_relu_with_tail(%arg0: tensor<1x10x20x3xf32>) -> tensor<1x8x18x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %empty = tensor.empty() : tensor<1x8x18x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x10x20x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x8x18x4xf32>) -> tensor<1x8x18x4xf32>
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<1x8x18x4xf32>
  %relu = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x8x18x4xf32>) outs(%out : tensor<1x8x18x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x8x18x4xf32>
  return %relu : tensor<1x8x18x4xf32>
}

// CHECK-LABEL: func.func @fuses_conv_relu_with_tail
// CHECK-COUNT-2: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: outs({{.*}} : tensor<1x8x18x4xf32>)
