#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerRewriteLinalgCopiesPasses() {
  registerRewriteLinalgCopiesPass();
}
}  // namespace mlir::ncnn
