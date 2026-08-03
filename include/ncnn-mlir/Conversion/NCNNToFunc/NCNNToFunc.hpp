#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerNCNNToFuncPasses() {
  registerConvertNCNNModelToFuncPass();
}
}  // namespace mlir::ncnn
