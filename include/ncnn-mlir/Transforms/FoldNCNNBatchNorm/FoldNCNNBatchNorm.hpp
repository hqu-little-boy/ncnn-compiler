#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerFoldNCNNBatchNormPasses() {
  registerFoldNCNNBatchNormPass();
}
}  // namespace mlir::ncnn
