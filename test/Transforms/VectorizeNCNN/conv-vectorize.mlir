// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=4 %s | FileCheck %s

// 静态逐元素 generic 行级向量化：外层标量循环 + 最内维整行 rank-1 向量
// （rank-1 连续 transfer 是 VectorToLLVM 的可靠下降形态）；标量 linalg
// generic 消失。卷积的窗口仿射映射不满足前置条件，保持 Linalg 形式。

func.func @relu(%arg0: tensor<6x8xf32>) -> tensor<6x8xf32> {
  %empty = tensor.empty() : tensor<6x8xf32>
  %zero = arith.constant 0.0 : f32
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<6x8xf32>) outs(%empty : tensor<6x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<6x8xf32>
  return %result : tensor<6x8xf32>
}

// CHECK-LABEL: func.func @relu
// CHECK-NOT: linalg.generic
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} vector<8xf32>
// CHECK: arith.maximumf {{.*}} vector<8xf32>
// CHECK: vector.transfer_write {{.*}} vector<8xf32>

func.func @conv2d(%arg0: tensor<1x8x34x3xf32>) -> tensor<1x6x32x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
  %empty = tensor.empty() : tensor<1x6x32x4xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x34x3xf32>, tensor<3x3x3x4xf32>) outs(%empty : tensor<1x6x32x4xf32>) -> tensor<1x6x32x4xf32>
  return %conv : tensor<1x6x32x4xf32>
}

// CHECK-LABEL: func.func @conv2d
// CHECK: linalg.conv_2d_nhwc_hwcf
