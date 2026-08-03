#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyNoNCNNOpsPasses() {
  registerVerifyNoNCNNOpsPass();
}
}  // namespace mlir::ncnn
