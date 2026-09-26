// Winograd F(6,3)（P7）：ncnn conv3x3s1_winograd63 的编译期化。3×3 s1
// 卷积改写为 输入变换 → 批量 GEMM → 输出变换 三段，算术强度从每权重
// 1 次 MAC/输出提升到 36/输出。矩阵抄录自 vendored ncnn
// convolution_3x3_winograd.h（G=ktm、Bᵀ=itm、Aᵀ=otm，interpolation 点
// ±1, ±2, ±1/2, ∞ 的有理系数）。数值上变换改变累加结构（不再是逐 k 串
// 行 FMA 链），f32 相对误差 ~1e-3 量级，由逐模型数值预算对账把关
// （docs/ncnn-performance-parity-plan.md §3-P7：预算过不了的模型不启
// 用，不做全局默认）。
//
// 本头文件只放变换矩阵与判据常量；IR 发射在
// lib/Transforms/Winograd63NCNN/Winograd63NCNN.cpp。系数是有理数的 f32
// 字面量，逐位相等于 vendored ncnn 的 ktm/itm/otm，改写它们等于改写
// 数值契约。
#pragma once

#include <cstdint>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ncnn::winograd63 {

// G（kernel transform，ktm[8][3]）。
constexpr float kG[8][3] = {{1.0F, 0.0F, 0.0F},
                            {-2.0F / 9, -2.0F / 9, -2.0F / 9},
                            {-2.0F / 9, 2.0F / 9, -2.0F / 9},
                            {1.0F / 90, 1.0F / 45, 2.0F / 45},
                            {1.0F / 90, -1.0F / 45, 2.0F / 45},
                            {1.0F / 45, 1.0F / 90, 1.0F / 180},
                            {1.0F / 45, -1.0F / 90, 1.0F / 180},
                            {0.0F, 0.0F, 1.0F}};

// Bᵀ（input transform，itm[8][8]）。
constexpr float kBt[8][8] = {
  {1.0F, 0.0F, -5.25F, 0.0F, 5.25F, 0.0F, -1.0F, 0.0F},
  {0.0F, 1.0F, 1.0F, -4.25F, -4.25F, 1.0F, 1.0F, 0.0F},
  {0.0F, -1.0F, 1.0F, 4.25F, -4.25F, -1.0F, 1.0F, 0.0F},
  {0.0F, 0.5F, 0.25F, -2.5F, -1.25F, 2.0F, 1.0F, 0.0F},
  {0.0F, -0.5F, 0.25F, 2.5F, -1.25F, -2.0F, 1.0F, 0.0F},
  {0.0F, 2.0F, 4.0F, -2.5F, -5.0F, 0.5F, 1.0F, 0.0F},
  {0.0F, -2.0F, 4.0F, 2.5F, -5.0F, -0.5F, 1.0F, 0.0F},
  {0.0F, -1.0F, 0.0F, 5.25F, 0.0F, -5.25F, 0.0F, 1.0F}};

// Aᵀ（output transform，otm[6][8]）。
constexpr float kAt[6][8] = {
  {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 32.0F, 32.0F, 0.0F},
  {0.0F, 1.0F, -1.0F, 2.0F, -2.0F, 16.0F, -16.0F, 0.0F},
  {0.0F, 1.0F, 1.0F, 4.0F, 4.0F, 8.0F, 8.0F, 0.0F},
  {0.0F, 1.0F, -1.0F, 8.0F, -8.0F, 4.0F, -4.0F, 0.0F},
  {0.0F, 1.0F, 1.0F, 16.0F, 16.0F, 2.0F, 2.0F, 0.0F},
  {0.0F, 1.0F, -1.0F, 32.0F, -32.0F, 1.0F, -1.0F, 1.0F}};

constexpr int64_t kAlpha = 8;   // 变换域边长（tile 6 + 界 2）
constexpr int64_t kTile = 6;    // 输出 tile 边长
constexpr int64_t kBatch = 64;  // 变换域批数 α²

// ncnn dispatch 判据的编译期化（convolution_x86.cpp:640）：3×3 s1 d1 且
// IC>8 或 OC>8 才值得 Winograd——小通道层的变换开销吃不掉 GEMM 收益。
// 策略分派（StrategyNCNN）与 IR 发射（Winograd63NCNN）共用同一判据，
// 避免"选了却不改写"或"改写了却没选上"。
inline bool eligible(int64_t kernelHeight,
                     int64_t kernelWidth,
                     int64_t dilationHeight,
                     int64_t dilationWidth,
                     int64_t strideHeight,
                     int64_t strideWidth,
                     int64_t inputChannels,
                     int64_t outputChannels) {
  return kernelHeight == 3 && kernelWidth == 3 && dilationHeight == 1 &&
         dilationWidth == 1 && strideHeight == 1 && strideWidth == 1 &&
         (inputChannels > 8 || outputChannels > 8);
}

// 编译期权重变换：G·w·Gᵀ 折叠 [64, OC, IC] 常量（ncnn create_pipeline
// 预变换的编译期等价物）。输入 [3,3,IC,OC]（MLIR 布局，kh-major），
// 输出批维最外与 batch_matmul 的 A 面板一致（行内 oc*IC+ic）。
//
// 实现刻意保留 APFloat 算术（multiply/add + 显式 RN-even 舍入）：改写成
// float 运算会令 -ffp-contract 把 a*b+c 收缩成 fmuladd，改变权重常量的
// 比特，违反产物逐字节不变（见 docs/refactor/p3-2026-09-26/backlog.md §2）。
mlir::DenseFPElementsAttr transformWeight(mlir::RankedTensorType weightType,
                                          mlir::DenseFPElementsAttr weights);

}  // namespace mlir::ncnn::winograd63
