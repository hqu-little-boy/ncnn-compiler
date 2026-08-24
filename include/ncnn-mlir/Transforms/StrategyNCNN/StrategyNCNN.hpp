#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerStrategyNCNNPasses() {
  registerStrategyNCNNPass();
}
}  // namespace mlir::ncnn
