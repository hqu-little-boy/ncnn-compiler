#pragma once

#include "ncnn-mlir/Passes.hpp"

namespace mlir::ncnn {

inline void registerReuseWorkspaceSlotsPasses() {
  registerReuseWorkspaceSlotsPass();
}

}  // namespace mlir::ncnn
