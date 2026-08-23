#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace ncnn_mlir {

// 目标向量 ABI 描述：由 target triple、march 与显式 feature 列表推导。
// FixedWidth 表示编译期确定的 lane 数（x86 SSE/AVX/AVX512、AArch64 NEON）；
// Scalable 表示运行时可变 VL（AArch64 SVE、RISC-V RVV），lanes 为 vscale=1
// 时的最小 f32 lane 数；Scalar 表示无向量收益目标，保持标量下降。
struct TargetVectorInfo {
  enum class Mode {
    Scalar,
    FixedWidth,
    Scalable,
  };

  Mode mode = Mode::Scalar;
  unsigned lanes = 0;

  static TargetVectorInfo resolve(llvm::StringRef triple,
                                  llvm::StringRef march,
                                  llvm::ArrayRef<llvm::StringRef> features);
};

}  // namespace ncnn_mlir
