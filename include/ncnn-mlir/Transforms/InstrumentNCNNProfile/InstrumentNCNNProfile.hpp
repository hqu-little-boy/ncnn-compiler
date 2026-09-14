#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerInstrumentNCNNProfilePasses() {
  registerInstrumentNCNNProfilePass();
}
}  // namespace mlir::ncnn
