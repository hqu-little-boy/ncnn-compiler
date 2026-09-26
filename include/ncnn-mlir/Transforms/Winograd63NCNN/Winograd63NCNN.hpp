// Winograd F(6,3) 改写的 IR 发射入口。矩阵与判据见
// ncnn-mlir/Support/Winograd63.hpp。
//
// 这不是独立的 MLIR pass，而是 StrategyNCNN 的可选改写路径之一：
// StrategyNCNN 负责 "conv → gemm/direct/winograd" 分派，本模块只负责
// 命中判据后的 IR 构造。两者共用 conv 契约注解（KernelContract.hpp），
// 使下游 MatmulKernelNCNN 的 kernelizeBatchMatmul 能识别中央的
// linalg.batch_matmul。
#pragma once

#include <cstdint>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::ncnn::winograd63 {

// 把 conv2d_nhwc_hwcf 改写为 pad → 两段输入变换 → batch_matmul →
// 两段输出变换 → 裁剪 + bias 逐元素加。返回 false 表示判据不满足或
// 尺寸证明失败，此时 convolution 保持原样（改写是事务性的）。
//
// maxElements 是策略层的单张量元素预算（StrategyNCNN 的
// kMaxStrategyElements）：Winograd 的临时缓冲共享该预算，超出即放弃。
// 由调用方传入，本模块不持有策略常量。
bool rewriteWinograd(mlir::RewriterBase& rewriter,
                     mlir::linalg::Conv2DNhwcHwcfOp convolution,
                     int64_t maxElements);

}  // namespace mlir::ncnn::winograd63
