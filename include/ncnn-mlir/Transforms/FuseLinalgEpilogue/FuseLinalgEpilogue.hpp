#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerFuseLinalgEpiloguePasses() {
  registerFuseLinalgEpiloguePass();
}
}  // namespace mlir::ncnn
