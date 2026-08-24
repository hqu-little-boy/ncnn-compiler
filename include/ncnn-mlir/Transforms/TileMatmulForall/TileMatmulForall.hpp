#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerTileMatmulForallPasses() {
  registerTileMatmulForallPass();
}
}  // namespace mlir::ncnn
