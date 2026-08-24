#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyNoSCFForallPasses() {
  registerVerifyNoSCFForallPass();
}
}  // namespace mlir::ncnn
