#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {
inline void registerVerifyVulkanSupportedModelPasses() {
  registerVerifyVulkanSupportedModelPass();
}
}  // namespace mlir::ncnn
