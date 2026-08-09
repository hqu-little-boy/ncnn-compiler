#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyModelShapeContractsPasses() {
  registerVerifyModelShapeContractsPass();
}
}  // namespace mlir::ncnn
