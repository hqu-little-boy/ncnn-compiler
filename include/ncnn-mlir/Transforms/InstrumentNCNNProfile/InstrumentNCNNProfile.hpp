#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
std::unique_ptr<Pass> createInstrumentNCNNFusionSitesPass();
std::unique_ptr<Pass> createInstrumentNCNNMaterializedSitesPass();

inline void registerInstrumentNCNNProfilePasses() {
  registerInstrumentNCNNProfilePass();
}
}  // namespace mlir::ncnn
