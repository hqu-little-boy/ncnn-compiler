#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVectorizeNCNNPasses() {
  registerVectorizeNCNNPass();
}
}  // namespace mlir::ncnn
