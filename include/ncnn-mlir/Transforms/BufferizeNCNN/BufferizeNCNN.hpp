#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerBufferizeNCNNPasses() {
  registerBufferizeNCNNPass();
}
}  // namespace mlir::ncnn
