#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyNoNestedOpenMPPasses() {
  registerVerifyNoNestedOpenMPPass();
}
}  // namespace mlir::ncnn
