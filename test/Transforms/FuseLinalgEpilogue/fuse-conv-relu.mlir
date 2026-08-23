#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// RUN: ncnn-mlir-opt --fuse-linalg-epilogue %s | FileCheck %s

// A single-user elementwise ReLU epilogue is fused into the tiled convolution
// loop; the standalone generic disappears and all tile types stay static.

func.func @fuses_conv_relu(%arg0: tensor<1x8x34x3xf32>) -> tensor<1x6x32x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %empty = tensor.empty() : tensor<1x6x32x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x34x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) -> tensor<1x6x32x4xf32>
  %zero = arith.constant 0.0 : f32
  %relu_out = tensor.empty() : tensor<1x6x32x4xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x6x32x4xf32>) outs(%relu_out : tensor<1x6x32x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x6x32x4xf32>
  return %relu : tensor<1x6x32x4xf32>
}

// CHECK-LABEL: func.func @fuses_conv_relu
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<1x6x32x4xf32>)
// CHECK: tensor.extract_slice {{.*}} tensor<1x6x32x4xf32> to tensor<1x6x16x4xf32>
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: arith.maximumf
// CHECK: linalg.yield
// CHECK: tensor.insert_slice {{.*}} tensor<1x6x16x4xf32> into tensor<1x6x32x4xf32>
// CHECK-NOT: outs({{.*}} : tensor<1x6x32x4xf32>)

func.func @keeps_multi_user_conv(%arg0: tensor<1x8x34x3xf32>, %arg1: tensor<1x6x32x4xf32>) -> tensor<1x6x32x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %empty = tensor.empty() : tensor<1x6x32x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x34x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) -> tensor<1x6x32x4xf32>
  %first = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %arg1 : tensor<1x6x32x4xf32>, tensor<1x6x32x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %add = arith.addf %in, %in_0 : f32
    linalg.yield %add : f32
  } -> tensor<1x6x32x4xf32>
  %second = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x6x32x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x6x32x4xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%first, %second : tensor<1x6x32x4xf32>, tensor<1x6x32x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %add = arith.addf %in, %in_0 : f32
    linalg.yield %add : f32
  } -> tensor<1x6x32x4xf32>
  return %sum : tensor<1x6x32x4xf32>
}

// CHECK-LABEL: func.func @keeps_multi_user_conv
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: scf.for

func.func @rejects_dynamic_shape(%arg0: tensor<1x?x?x3xf32>) -> tensor<1x?x?x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty(%c4, %c8) : tensor<1x?x?x4xf32>
  %empty2 = tensor.empty(%c4, %c8) : tensor<1x?x?x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x?x?x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x?x?x4xf32>) -> tensor<1x?x?x4xf32>
  %zero = arith.constant 0.0 : f32
  %relu_out = tensor.empty(%c4, %c8) : tensor<1x?x?x4xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x?x?x4xf32>) outs(%relu_out : tensor<1x?x?x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x?x?x4xf32>
  return %relu : tensor<1x?x?x4xf32>
}

// CHECK-LABEL: func.func @rejects_dynamic_shape
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: scf.for

func.func @rejects_unrecognized_body(%arg0: tensor<1x8x34x3xf32>) -> tensor<1x6x32x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %empty = tensor.empty() : tensor<1x6x32x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x34x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) -> tensor<1x6x32x4xf32>
  %tanh_out = tensor.empty() : tensor<1x6x32x4xf32>
  %tanh = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x6x32x4xf32>) outs(%tanh_out : tensor<1x6x32x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %result = math.tanh %in : f32
    linalg.yield %result : f32
  } -> tensor<1x6x32x4xf32>
  return %tanh : tensor<1x6x32x4xf32>
}

// CHECK-LABEL: func.func @rejects_unrecognized_body
// CHECK: math.tanh
// CHECK-NOT: scf.for
