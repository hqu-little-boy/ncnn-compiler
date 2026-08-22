#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerFoldLinalgConstantTransposePasses() {
  registerFoldLinalgConstantTransposePass();
}
}  // namespace mlir::ncnn
