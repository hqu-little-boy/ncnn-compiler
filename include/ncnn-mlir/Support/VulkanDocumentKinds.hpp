#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir::ncnn::contract {

inline constexpr llvm::StringLiteral kVulkanWeightPayloadKind =
  "ncnn.vulkan_weight_payloads";
inline constexpr llvm::StringLiteral kVulkanExecutionPlanKind =
  "ncnn.vulkan_execution_plan";

}  // namespace mlir::ncnn::contract
