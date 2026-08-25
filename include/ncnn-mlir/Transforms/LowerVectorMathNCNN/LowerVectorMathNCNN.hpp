#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerLowerVectorMathNCNNPasses() {
  registerLowerVectorMathNCNNPass();
}
}  // namespace mlir::ncnn
