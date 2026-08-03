#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerNormalizeNCNNPasses() {
  registerNormalizeNCNNPass();
}
}  // namespace mlir::ncnn
