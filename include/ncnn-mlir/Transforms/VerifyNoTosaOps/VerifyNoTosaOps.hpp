#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyNoTosaOpsPasses() {
  registerVerifyNoTosaOpsPass();
}
}  // namespace mlir::ncnn
