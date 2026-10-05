#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {

inline void registerNCNNToVulkanPasses() {
  registerLowerNCNNVulkanModelPass();
  registerLowerNCNNVulkanGraphPass();
}

}  // namespace mlir::ncnn
