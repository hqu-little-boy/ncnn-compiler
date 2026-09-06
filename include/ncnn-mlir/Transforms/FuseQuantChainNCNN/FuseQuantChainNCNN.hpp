#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerFuseQuantChainNCNNPasses() {
  registerFuseQuantChainNCNNPass();
}
}  // namespace mlir::ncnn
