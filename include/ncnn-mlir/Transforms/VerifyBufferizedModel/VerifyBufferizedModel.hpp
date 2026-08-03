#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyBufferizedModelPasses() {
  registerVerifyBufferizedModelPass();
}
}  // namespace mlir::ncnn
