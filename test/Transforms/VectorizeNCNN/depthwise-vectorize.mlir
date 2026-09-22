// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=4 %s | FileCheck %s
// RUN: ncnn-mlir-opt --vectorize-ncnn='lanes=4 packed-conv-depthwise=true' %s | FileCheck --check-prefix=PACKED %s

// P5 depthwise 行向量化：multiplier=1 的 depthwise_conv_2d_nhwc_hwcm
// 改写为 forall(n,oh,ow) 网格 + C 维分块 rank-1 transfer——权重折叠为
// [KH*KW,C] 行视图（常量权重由 canonicalizer 直接折叠出折叠形态）、
// init 折叠 4D 作累加种子，窗口 kh 外 kw 内逐个 vector.fma 累加
// （named op body acc=addf(mulf(in,w),acc) 的归约序），结果行经
// parallel_insert_slice 落回共享输出。替换值先 expand 回 5D，与下游
// collapse_shape 消费者在 canonicalizer 对消，forall 共享输出直接呈
// 折叠 4D 形态。分块预算 ≤32 位元素 4×lanes，非 2 幂通道退 lanes；
// multiplier≠1、动态 shape、仍非 2 幂的窄通道保持标量 named op
// （多通道 multiplier 变体留待 P8）。

func.func @depthwise_basic(%arg0: tensor<1x7x9x4xf32>) -> tensor<1x3x7x4xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x4x1xf32>
  %empty = tensor.empty() : tensor<1x3x7x4x1xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x4xf32>, tensor<3x3x4x1xf32>) outs(%empty : tensor<1x3x7x4x1xf32>) -> tensor<1x3x7x4x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x1xf32> into tensor<1x3x7x4xf32>
  return %1 : tensor<1x3x7x4xf32>
}

// C=4 ≤ 16（4×lanes）整行成块：无分块循环、无标量尾，kh 外 kw 内
// 双层窗口循环做 vector.fma；stride 高维 2 保留 muli、宽维 1 被折叠。
// CHECK-LABEL: func.func @depthwise_basic
// CHECK: arith.constant {{.*}} : tensor<9x4xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<1x3x7x4x1xf32> into tensor<1x3x7x4xf32>
// CHECK: scf.forall {{.*}} in (1, 3, 7) shared_outs
// CHECK: tensor.empty() : tensor<1x1x1x4xf32>
// CHECK: vector.transfer_read {{.*}} tensor<1x3x7x4xf32>, vector<4xf32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} tensor<1x7x9x4xf32>, vector<4xf32>
// CHECK: vector.transfer_read {{.*}} tensor<9x4xf32>, vector<4xf32>
// CHECK: vector.fma {{.*}} : vector<4xf32>
// CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<1x1x1x4xf32>
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice {{.*}} [1, 1, 1, 4]
// CHECK: ncnn.implementation = "depthwise_simd"
// CHECK: ncnn.multiplier = 1 : i64
// CHECK: ncnn.operation_family = "depthwise"
// CHECK: return {{.*}} : tensor<1x3x7x4xf32>
// CHECK-NOT: linalg.depthwise

func.func @depthwise_chunk_tail(%arg0: tensor<1x7x9x10xf32>) -> tensor<1x3x7x10xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x10x1xf32>
  %empty = tensor.empty() : tensor<1x3x7x10x1xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x10xf32>, tensor<3x3x10x1xf32>) outs(%empty : tensor<1x3x7x10x1xf32>) -> tensor<1x3x7x10x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x10x1xf32> into tensor<1x3x7x10xf32>
  return %1 : tensor<1x3x7x10xf32>
}

// C=10：整行 10 非 2 幂 → 退 lanes=4 分块，2 个整分块 + 2 元素标量尾
// （逐位复刻 mulf+addf，不经 FMA 收缩）。
// CHECK-LABEL: func.func @depthwise_chunk_tail
// CHECK: scf.for {{.*}} iter_args
// CHECK: vector.transfer_read {{.*}} vector<4xf32>
// CHECK: scf.for
// CHECK: vector.fma {{.*}} : vector<4xf32>
// CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<1x1x1x10xf32>
// CHECK: scf.for {{.*}} iter_args
// CHECK: tensor.extract
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.insert {{.*}} tensor<1x1x1x10xf32>
// CHECK: return {{.*}} : tensor<1x3x7x10xf32>
// CHECK-NOT: linalg.depthwise

func.func @depthwise_multiplier2(%arg0: tensor<1x7x9x4xf32>) -> tensor<1x3x7x8xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x4x2xf32>
  %empty = tensor.empty() : tensor<1x3x7x4x2xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x4xf32>, tensor<3x3x4x2xf32>) outs(%empty : tensor<1x3x7x4x2xf32>) -> tensor<1x3x7x4x2xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x2xf32> into tensor<1x3x7x8xf32>
  return %1 : tensor<1x3x7x8xf32>
}

// multiplier=2 不改写（P8 评估）。
// CHECK-LABEL: func.func @depthwise_multiplier2
// CHECK: linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-NOT: vector.fma

func.func @depthwise_narrow_channels(%arg0: tensor<1x7x9x3xf32>) -> tensor<1x3x7x3xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x3x1xf32>
  %empty = tensor.empty() : tensor<1x3x7x3x1xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x3xf32>, tensor<3x3x3x1xf32>) outs(%empty : tensor<1x3x7x3x1xf32>) -> tensor<1x3x7x3x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x3x1xf32> into tensor<1x3x7x3xf32>
  return %1 : tensor<1x3x7x3xf32>
}

// C=3 整行/lanes 均非 2 幂（vector<3> 宽度教训），保持标量。
// CHECK-LABEL: func.func @depthwise_narrow_channels
// CHECK: linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-NOT: vector.fma

func.func @depthwise_dynamic(%arg0: tensor<1x?x9x4xf32>) -> tensor<1x?x7x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<3x3x4x1xf32>
  %c1 = arith.constant 1 : index
  %height = tensor.dim %arg0, %c1 : tensor<1x?x9x4xf32>
  %empty = tensor.empty(%height) : tensor<1x?x7x4x1xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x?x9x4xf32>, tensor<3x3x4x1xf32>) outs(%empty : tensor<1x?x7x4x1xf32>) -> tensor<1x?x7x4x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x?x7x4x1xf32> into tensor<1x?x7x4xf32>
  return %1 : tensor<1x?x7x4xf32>
}

// 动态空间维不满足静态 shape 前置条件，保持标量。
// CHECK-LABEL: func.func @depthwise_dynamic
// CHECK: linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-NOT: vector.fma

// CHECK-LABEL: func.func @depthwise_packed_pack8
func.func @depthwise_packed_pack8(%arg0: tensor<1x7x9x8xf32>) -> tensor<1x3x7x8xf32> {
  %cst = arith.constant dense<0.5> : tensor<3x3x8x1xf32>
  %empty = tensor.empty() : tensor<1x3x7x8x1xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x8xf32>, tensor<3x3x8x1xf32>) outs(%empty : tensor<1x3x7x8x1xf32>) -> tensor<1x3x7x8x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x8x1xf32> into tensor<1x3x7x8xf32>
  return %1 : tensor<1x3x7x8xf32>
}

// PACKED-LABEL: func.func @depthwise_packed_pack8
// PACKED: arith.constant dense<{{.*}}> : tensor<9x1x8xf32>
// PACKED: tensor.expand_shape {{.*}} tensor<1x7x9x8xf32> into tensor<1x7x9x1x8xf32>
// PACKED: tensor.expand_shape {{.*}} tensor<1x3x7x8x1xf32> into tensor<1x3x7x1x8x1xf32>
// PACKED: vector.transfer_read {{.*}} vector<8xf32>
// PACKED: ncnn.implementation = "depthwise_packed"
// PACKED-SAME: ncnn.layout_island_id = "depthwise-packed"
// PACKED-SAME: ncnn.layout_pack_factor = 8 : i64
// PACKED-SAME: ncnn.packing = "depthwise_packed"
// PACKED: tensor.collapse_shape {{.*}} tensor<1x3x7x1x8xf32> into tensor<1x3x7x8xf32>
// PACKED-NOT: linalg.depthwise_conv_2d_nhwc_hwcm
