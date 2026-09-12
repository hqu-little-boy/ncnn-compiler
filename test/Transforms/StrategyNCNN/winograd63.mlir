// RUN: ncnn-mlir-opt --strategy-ncnn=strategy=winograd --canonicalize %s | FileCheck %s

// P7 Winograd F(6,3)：3×3 s1 d1 且 IC=64>8 的实例改写为
// pad → 输入变换两段 generic → 常量权重变换 [64,OC,IC] →
// linalg.batch_matmul → 输出变换两段 → 裁剪回原形状。

// CHECK-LABEL: func.func @conv3x3_large_channels
func.func @conv3x3_large_channels(%arg0: tensor<1x10x10x64xf32>) -> tensor<1x8x8x128xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x64x128xf32>
  %init = tensor.empty() : tensor<1x8x8x128xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x10x10x64xf32>, tensor<3x3x64x128xf32>) outs(%init : tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32>
  return %conv : tensor<1x8x8x128xf32>
}

// OH=OW=8 → 2×2 tiles，pad 到 1x14x14（2·6+2）
// CHECK: tensor.pad {{.*}} low[0, 0, 0, 0] high[0, 4, 4, 0]
// 输入变换两段（B 与 Bᵀ）
// CHECK: linalg.generic
// CHECK: linalg.generic
// 中心批量收缩（批维 64 = 变换域元素；权重常量 [64,OC=128,IC=64] 经
// batch_matmul 的 ins 类型断言覆盖）
// CHECK: linalg.batch_matmul ins({{.*}} : tensor<64x128x64xf32>, tensor<64x64x4xf32>) outs({{.*}} : tensor<64x128x4xf32>)
// 输出变换两段（A 与 Aᵀ）
// CHECK: linalg.generic
// CHECK: linalg.generic
// tail tile 裁剪（12×12 → 8×8）
// CHECK: tensor.extract_slice {{.*}} : tensor<1x12x12x128xf32> to tensor<1x8x8x128xf32>
// bias 补加（conv init 逐元素相加，最后一层 generic）
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>)
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// 判据外（IC=OC=8 均 ≤8）保持常规 dispatch：IC≤16 无通道超限且工作集
// 小于 L2 → 直接卷积。winograd 策略不比 auto 更少优化。
// CHECK-LABEL: func.func @below_channel_threshold
func.func @below_channel_threshold(%arg0: tensor<1x10x10x8xf32>) -> tensor<1x8x8x8xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x8x8xf32>
  %init = tensor.empty() : tensor<1x8x8x8xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x10x10x8xf32>, tensor<3x3x8x8xf32>) outs(%init : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %conv : tensor<1x8x8x8xf32>
}

// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.batch_matmul

// -----

// 非 3×3（5×5）不在 F(6,3) 判据内；IC=64>16 命中 prefer_gemm → im2col。
// CHECK-LABEL: func.func @conv5x5_falls_to_gemm
func.func @conv5x5_falls_to_gemm(%arg0: tensor<1x12x12x64xf32>) -> tensor<1x8x8x128xf32> {
  %cst = arith.constant dense<0.5> : tensor<5x5x64x128xf32>
  %init = tensor.empty() : tensor<1x8x8x128xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x12x12x64xf32>, tensor<5x5x64x128xf32>) outs(%init : tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32>
  return %conv : tensor<1x8x8x128xf32>
}

// CHECK: linalg.matmul
// CHECK-NOT: linalg.batch_matmul

// -----

// s2 不在判据内 → 常规 dispatch。
// CHECK-LABEL: func.func @stride2_falls_to_gemm
func.func @stride2_falls_to_gemm(%arg0: tensor<1x17x17x64xf32>) -> tensor<1x8x8x128xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x64x128xf32>
  %init = tensor.empty() : tensor<1x8x8x128xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x17x17x64xf32>, tensor<3x3x64x128xf32>) outs(%init : tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32>
  return %conv : tensor<1x8x8x128xf32>
}

// CHECK: linalg.matmul
// CHECK-NOT: linalg.batch_matmul
