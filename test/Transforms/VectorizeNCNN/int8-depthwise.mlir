// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=4 --split-input-file %s | FileCheck %s

// P16 静态 INT8 depthwise SIMD（module 属性 ncnn.int8_depthwise 选择进入）：
// multiplier=1、i8 输入/权重 + i32 累加器共享 P5 的 forall(n,oh,ow) 网格与
// 窗口/索引几何，但算术独立——每 lane 独立 extsi→muli→addi（i32 wrap 语义，
// 无 nsw/nuw），不做通道点积、无 vector.fma；输入（vector<Nxi8>）与累加
// （vector<Nxi32>）的 tensor/vector 类型与 poison 全程区分；标量尾同样
// extsi→muli→addi 逐位复刻。仅接受规范零零点 region（命名 op regionBuilder
// 的 extsi/extsi/muli/addi 链）；linalg verifier 接受自定义 region，非零
// zero-point/clamp 变体（规范上属 _q）必须拒绝。multiplier>1、动态 shape、
// 门未开保持标量。注意负 stride/dilation 由 linalg verifier 前置拒绝，无
// IR 可达；in-pass windowFits 校验为纵深防御。

// CHECK-LABEL: func.func @int8_basic
module attributes {ncnn.int8_depthwise = true} {
  func.func @int8_basic(%arg0: tensor<1x7x9x4xi8>) -> tensor<1x3x7x4xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x4x1xi8>
    %empty = tensor.empty() : tensor<1x3x7x4x1xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x4xi8>, tensor<3x3x4x1xi8>) outs(%empty : tensor<1x3x7x4x1xi32>) -> tensor<1x3x7x4x1xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x1xi32> into tensor<1x3x7x4xi32>
    return %1 : tensor<1x3x7x4xi32>
  }
  // C=4 ≤ 4×lanes 整行单分块：i8 权重折叠 [KH*KW,C]、init 折叠 4D 作累加
  // 种子，kh 外 kw 内逐窗口 extsi→muli→addi；无 channel 点积、无 fma。
  // CHECK: arith.constant {{.*}} : tensor<9x4xi8>
  // CHECK: ub.poison : i32
  // CHECK: ub.poison : i8
  // CHECK: tensor.collapse_shape {{.*}} : tensor<1x3x7x4x1xi32> into tensor<1x3x7x4xi32>
  // CHECK: scf.forall {{.*}} in (1, 3, 7) shared_outs
  // CHECK: tensor.empty() : tensor<1x1x1x4xi32>
  // CHECK: vector.transfer_read {{.*}} tensor<1x3x7x4xi32>, vector<4xi32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: vector.transfer_read {{.*}} tensor<1x7x9x4xi8>, vector<4xi8>
  // CHECK: vector.transfer_read {{.*}} tensor<9x4xi8>, vector<4xi8>
  // CHECK: arith.extsi {{.*}} : vector<4xi8> to vector<4xi32>
  // CHECK: arith.extsi {{.*}} : vector<4xi8> to vector<4xi32>
  // CHECK: arith.muli {{.*}} : vector<4xi32>
  // CHECK: arith.addi {{.*}} : vector<4xi32>
  // CHECK-NOT: vector.fma
  // CHECK-NOT: arith.mulf
  // CHECK: vector.transfer_write {{.*}} : vector<4xi32>, tensor<1x1x1x4xi32>
  // CHECK: scf.forall.in_parallel
  // CHECK: tensor.parallel_insert_slice {{.*}} [1, 1, 1, 4]
  // CHECK: ncnn.fma = "extsi_muli_addi"
  // CHECK: ncnn.implementation = "depthwise_simd"
  // CHECK: ncnn.multiplier = 1 : i64
  // CHECK-NOT: linalg.depthwise

  // CHECK-LABEL: func.func @int8_wide
  func.func @int8_wide(%arg0: tensor<1x7x9x16xi8>) -> tensor<1x3x7x16xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x16x1xi8>
    %empty = tensor.empty() : tensor<1x3x7x16x1xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x16xi8>, tensor<3x3x16x1xi8>) outs(%empty : tensor<1x3x7x16x1xi32>) -> tensor<1x3x7x16x1xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x16x1xi32> into tensor<1x3x7x16xi32>
    return %1 : tensor<1x3x7x16xi32>
  }
  // C=16 = 4×lanes 整行单分块：vector<16xi8> 输入读独立于 vector<16xi32>
  // 累加读，无分块循环、无标量尾。
  // CHECK: arith.constant {{.*}} : tensor<9x16xi8>
  // CHECK: tensor.empty() : tensor<1x1x1x16xi32>
  // CHECK: vector.transfer_read {{.*}} tensor<1x3x7x16xi32>, vector<16xi32>
  // CHECK: vector.transfer_read {{.*}} tensor<1x7x9x16xi8>, vector<16xi8>
  // CHECK: vector.transfer_read {{.*}} tensor<9x16xi8>, vector<16xi8>
  // CHECK: arith.extsi {{.*}} : vector<16xi8> to vector<16xi32>
  // CHECK: arith.extsi {{.*}} : vector<16xi8> to vector<16xi32>
  // CHECK: arith.muli {{.*}} : vector<16xi32>
  // CHECK: arith.addi {{.*}} : vector<16xi32>
  // CHECK: vector.transfer_write {{.*}} : vector<16xi32>, tensor<1x1x1x16xi32>
  // CHECK: ncnn.simd_chunk = 16 : i64
  // CHECK: ncnn.tail = "none"

  // CHECK-LABEL: func.func @int8_chunk_tail
  func.func @int8_chunk_tail(%arg0: tensor<1x7x9x10xi8>) -> tensor<1x3x7x10xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x10x1xi8>
    %empty = tensor.empty() : tensor<1x3x7x10x1xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x10xi8>, tensor<3x3x10x1xi8>) outs(%empty : tensor<1x3x7x10x1xi32>) -> tensor<1x3x7x10x1xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x10x1xi32> into tensor<1x3x7x10xi32>
    return %1 : tensor<1x3x7x10xi32>
  }
  // C=10：2 个 vector<4xi8> 整分块 + 2 lane 标量尾（逐位 extsi→muli→addi，
  // wrap 语义不收缩）。
  // CHECK: scf.for {{.*}} iter_args
  // CHECK: vector.transfer_read {{.*}} tensor<1x3x7x10xi32>, vector<4xi32>
  // CHECK: vector.transfer_read {{.*}} tensor<1x7x9x10xi8>, vector<4xi8>
  // CHECK: vector.transfer_read {{.*}} tensor<9x10xi8>, vector<4xi8>
  // CHECK: arith.extsi {{.*}} : vector<4xi8> to vector<4xi32>
  // CHECK: arith.muli {{.*}} : vector<4xi32>
  // CHECK: arith.addi {{.*}} : vector<4xi32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xi32>, tensor<1x1x1x10xi32>
  // CHECK: scf.for {{.*}} iter_args
  // CHECK: tensor.extract {{.*}} tensor<1x3x7x10xi32>
  // CHECK: arith.extsi {{.*}} : i8 to i32
  // CHECK: arith.muli {{.*}} : i32
  // CHECK: arith.addi {{.*}} : i32
  // CHECK: tensor.insert {{.*}} tensor<1x1x1x10xi32>
  // CHECK: ncnn.tail = "scalar_tail"
  // CHECK-NOT: vector.fma

  // CHECK-LABEL: func.func @int8_multiplier2
  func.func @int8_multiplier2(%arg0: tensor<1x7x9x4xi8>) -> tensor<1x3x7x8xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x4x2xi8>
    %empty = tensor.empty() : tensor<1x3x7x4x2xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x4xi8>, tensor<3x3x4x2xi8>) outs(%empty : tensor<1x3x7x4x2xi32>) -> tensor<1x3x7x4x2xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x2xi32> into tensor<1x3x7x8xi32>
    return %1 : tensor<1x3x7x8xi32>
  }
  // multiplier=2 不满足 weight [KH,KW,C,1] 前置，保持标量 named op。
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwcm {{.*}} tensor<3x3x4x2xi8>
  // CHECK-NOT: vector.transfer_read

  // CHECK-LABEL: func.func @int8_dynamic
  func.func @int8_dynamic(%arg0: tensor<1x?x9x4xi8>, %height: index) -> tensor<1x?x7x4xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x4x1xi8>
    %empty = tensor.empty(%height) : tensor<1x?x7x4x1xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x?x9x4xi8>, tensor<3x3x4x1xi8>) outs(%empty : tensor<1x?x7x4x1xi32>) -> tensor<1x?x7x4x1xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x?x7x4x1xi32> into tensor<1x?x7x4xi32>
    return %1 : tensor<1x?x7x4xi32>
  }
  // 动态空间维无静态 shape 前置，保持标量 named op。
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwcm {{.*}} tensor<1x?x9x4xi8>
  // CHECK-NOT: vector.transfer_read

  // CHECK-LABEL: func.func @int8_noncanonical_region
  func.func @int8_noncanonical_region(%arg0: tensor<1x7x9x4xi8>) -> tensor<1x3x7x4xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x4x1xi8>
    %empty = tensor.empty() : tensor<1x3x7x4x1xi32>
    %0 = "linalg.depthwise_conv_2d_nhwc_hwcm"(%arg0, %cst, %empty) ({
    ^bb0(%in: i8, %w: i8, %out: i32):
      %1 = arith.extsi %in : i8 to i32
      %2 = arith.extsi %w : i8 to i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.subi %3, %out : i32
      linalg.yield %4 : i32
    }) {dilations = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<1x7x9x4xi8>, tensor<3x3x4x1xi8>, tensor<1x3x7x4x1xi32>) -> tensor<1x3x7x4x1xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x1xi32> into tensor<1x3x7x4xi32>
    return %1 : tensor<1x3x7x4xi32>
  }
  // 自定义 region（zero-point 风格 subi 变体）不在规范零零点形态内，保持
  // 标量 named op：向量化会改变归约语义，绝不可静默改写。linalg 命名 op
  // 打印会省略 region 体（含 subi），故以"本函数内无 scf.forall"为锚：
  // CHECK-NOT 窗口从 LABEL 覆盖到 fallback 标注行，若该 op 被向量化，
  // scf.forall 必然出现在窗口内而失败。
  // CHECK-NOT: scf.forall
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwcm {{.*}} ncnn.fallback_reason = "not_vectorized"
  // CHECK-NOT: vector.transfer_read
}

// -----

// CHECK-LABEL: func.func @int8_overflow_flags
module attributes {ncnn.int8_depthwise = true} {
  func.func @int8_overflow_flags(%arg0: tensor<1x7x9x4xi8>) -> tensor<1x3x7x4xi32> {
    %cst = arith.constant dense<5> : tensor<3x3x4x1xi8>
    %empty = tensor.empty() : tensor<1x3x7x4x1xi32>
    %0 = "linalg.depthwise_conv_2d_nhwc_hwcm"(%arg0, %cst, %empty) ({
    ^bb0(%in: i8, %w: i8, %out: i32):
      %1 = arith.extsi %in : i8 to i32
      %2 = arith.extsi %w : i8 to i32
      %3 = arith.muli %1, %2 overflow<nsw> : i32
      %4 = arith.addi %out, %3 overflow<nsw> : i32
      linalg.yield %4 : i32
    }) {dilations = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<1> : tensor<2xi64>} : (tensor<1x7x9x4xi8>, tensor<3x3x4x1xi8>, tensor<1x3x7x4x1xi32>) -> tensor<1x3x7x4x1xi32>
    %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x1xi32> into tensor<1x3x7x4xi32>
    return %1 : tensor<1x3x7x4xi32>
  }
  // Overflow flags are part of the integer semantics; the wrapping SIMD
  // emitter must not rewrite this body or silently drop nsw/nuw.
  // CHECK-NOT: scf.forall
  // CHECK: linalg.depthwise_conv_2d_nhwc_hwcm {{.*}} ncnn.fallback_reason = "not_vectorized"
  // CHECK-NOT: vector.transfer_read
}

// -----

// CHECK-LABEL: func.func @int8_gate_off
func.func @int8_gate_off(%arg0: tensor<1x7x9x4xi8>) -> tensor<1x3x7x4xi32> {
  %cst = arith.constant dense<5> : tensor<3x3x4x1xi8>
  %empty = tensor.empty() : tensor<1x3x7x4x1xi32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x7x9x4xi8>, tensor<3x3x4x1xi8>) outs(%empty : tensor<1x3x7x4x1xi32>) -> tensor<1x3x7x4x1xi32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2], [3, 4]] : tensor<1x3x7x4x1xi32> into tensor<1x3x7x4xi32>
  return %1 : tensor<1x3x7x4xi32>
}
// 门未开（无 module 属性，本 chunk 为独立 split 段）：f32 前置不满足，
// int8 op 保持标量并落 fallback 标注。
// CHECK: linalg.depthwise_conv_2d_nhwc_hwcm {{.*}} ncnn.fallback_reason = "not_vectorized"
// CHECK-NOT: vector.transfer_read
