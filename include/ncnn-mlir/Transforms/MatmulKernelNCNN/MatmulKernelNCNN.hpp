#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerMatmulKernelNCNNPasses() {
  registerMatmulKernelNCNNPass();
}
}  // namespace mlir::ncnn
