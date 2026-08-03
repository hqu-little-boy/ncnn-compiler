#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerNCNNToTosaPasses() {
  registerConvertNCNNToTosaPass();
}
}  // namespace mlir::ncnn
