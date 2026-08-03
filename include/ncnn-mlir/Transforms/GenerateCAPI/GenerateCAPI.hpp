#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerGenerateCAPIPasses() {
  registerGenerateCAPIPass();
  registerFinalizeCAPIPass();
}
}  // namespace mlir::ncnn
