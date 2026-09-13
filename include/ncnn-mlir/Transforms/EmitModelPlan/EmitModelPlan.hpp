#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerEmitModelPlanPasses() {
  registerEmitModelPlanPass();
}
}  // namespace mlir::ncnn
