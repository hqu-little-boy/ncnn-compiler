#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerForallizeDisjointTileLoopsPasses() {
  registerForallizeDisjointTileLoopsPass();
}
}  // namespace mlir::ncnn
