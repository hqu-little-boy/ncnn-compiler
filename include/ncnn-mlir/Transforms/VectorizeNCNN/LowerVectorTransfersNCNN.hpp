#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerLowerVectorTransfersNCNNPasses() {
  registerLowerVectorTransfersNCNNPass();
}
}  // namespace mlir::ncnn
