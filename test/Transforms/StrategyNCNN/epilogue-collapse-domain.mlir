// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize %s | FileCheck %s

// 方案 a epilogue 衔接：conv 结果的唯一用户是恒等逐元素 generic 时，
// consumer 整体搬进折叠二维域（matmul → 2D generic → expand_shape），
// 下游继续看到原四维形状；fuse-linalg-epilogue 的视图守卫保证不会二次
// 融合本 pass 产物。

func.func @conv_relu(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x8x8x16xf32>
  return %relu : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_relu
// CHECK: arith.constant dense{{.*}} : tensor<3x16xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<1x8x8x3xf32> into tensor<64x3xf32>
// CHECK: linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: linalg.generic {{.*}}ins({{.*}} : tensor<64x16xf32>) outs({{.*}} : tensor<64x16xf32>)
// CHECK: arith.maximumf
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// -----

// 多消费者 / 非逐元素用户不融合：matmul 直接 expand 回四维，consumer 原样保留。
func.func @conv_two_users(%arg0: tensor<1x8x8x3xf32>) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  return %conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_two_users
// CHECK: linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: arith.maximumf

// P22：同形 residual 可证明使用相同的二维 collapse 域，
// 因此 producer 与 consumer 一起转为 matmul + 2D generic，保留 residual 输入。
// -----
func.func @conv_residual(%arg0: tensor<1x8x8x3xf32>,
                         %residual: tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %add = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %residual : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %res: f32, %o: f32):
    %sum = arith.addf %in, %res : f32
    linalg.yield %sum : f32
  } -> tensor<1x8x8x16xf32>
  return %add : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_residual
// CHECK: linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: linalg.generic {{.*}}ins({{.*}} : tensor<64x16xf32>, tensor<64x16xf32>) outs({{.*}} : tensor<64x16xf32>)
// CHECK: arith.addf
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// When Strategy cannot lift a multi-input epilogue, keep the known GEMM
// implementation and leave the unsupported generic as a separate operation.
// Do not defer the convolution to a fusion pass that may reject the consumer.
// -----
func.func @conv_unsupported_epilogue(%arg0: tensor<1x8x8x3xf32>,
                                     %residual: tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %residual : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %res: f32, %o: f32):
    %root = math.sqrt %in : f32
    linalg.yield %root : f32
  } -> tensor<1x8x8x16xf32>
  return %result : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_unsupported_epilogue
// CHECK: linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK: linalg.generic
// CHECK: math.sqrt
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

// Repeated references to the same producer in one DPS consumer are one
// distinct consumer and can be lifted into the same flattened GEMM domain.
func.func @conv_repeated_silu(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %one = arith.constant 1.0 : f32
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %silu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %same: f32, %unused: f32):
    %negative = arith.negf %same : f32
    %exp = math.exp %negative : f32
    %denominator = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denominator : f32
    %result = arith.mulf %value, %sigmoid : f32
    linalg.yield %result : f32
  } -> tensor<1x8x8x16xf32>
  return %silu : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_repeated_silu
// CHECK: %[[REPEATED_MM:.*]] = linalg.matmul {{.*}} ins({{.*}} : tensor<64x3xf32>, tensor<3x16xf32>)
// CHECK: linalg.generic {{.*}} ins(%[[REPEATED_MM]], %[[REPEATED_MM]] : tensor<64x16xf32>, tensor<64x16xf32>)
// CHECK: arith.negf
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf

func.func @conv_repeated_silu_external_use(%arg0: tensor<1x8x8x3xf32>) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %one = arith.constant 1.0 : f32
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %silu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %same: f32, %unused: f32):
    %negative = arith.negf %same : f32
    %exp = math.exp %negative : f32
    %denominator = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denominator : f32
    %result = arith.mulf %value, %sigmoid : f32
    linalg.yield %result : f32
  } -> tensor<1x8x8x16xf32>
  return %conv, %silu : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_repeated_silu_external_use
// CHECK: linalg.matmul
// CHECK: %[[EXTERNAL_EXPANDED:.*]] = tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK: linalg.generic {{.*}} ins(%[[EXTERNAL_EXPANDED]], %[[EXTERNAL_EXPANDED]] : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>)
// CHECK: return %[[EXTERNAL_EXPANDED]],

func.func @conv_repeated_nonidentity_map(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
  %init = tensor.empty() : tensor<1x8x8x16xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %one = arith.constant 1.0 : f32
  %out = tensor.empty() : tensor<1x8x8x16xf32>
  %silu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
  ^bb0(%value: f32, %same: f32, %unused: f32):
    %negative = arith.negf %same : f32
    %exp = math.exp %negative : f32
    %denominator = arith.addf %one, %exp : f32
    %sigmoid = arith.divf %one, %denominator : f32
    %result = arith.mulf %value, %sigmoid : f32
    linalg.yield %result : f32
  } -> tensor<1x8x8x16xf32>
  return %silu : tensor<1x8x8x16xf32>
}

// CHECK-LABEL: func.func @conv_repeated_nonidentity_map
// CHECK: linalg.matmul
// CHECK: %[[NONIDENTITY_EXPANDED:.*]] = tensor.expand_shape {{.*}} output_shape [1, 8, 8, 16]
// CHECK: linalg.generic {{.*}} ins(%[[NONIDENTITY_EXPANDED]], %[[NONIDENTITY_EXPANDED]] : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>)
// CHECK-NOT: ncnn.strategy_lifted_epilogue
