#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace ncnn_mlir::vulkan::test {

enum class TestVulkanShaderMutation : std::uint8_t {
  None,
  TruncateFinalInstruction,
  WrongVersion,
  WrongMemoryModel,
  UnsupportedCapability,
  WrongExecutionModel,
  WrongEntryPoint,
  WrongLocalSize,
  WrongBinding,
  DuplicateBinding,
  WrongFloatWidth,
  WrongStoreValueType,
  WrongActualAccess,
  UseRuntimeArray,
  WrongArrayExtent,
  OutOfBoundsArrayIndex,
  UnsupportedPointerFlow,
  AccessesOnlyInDeadFunction,
  AccessesOnlyInDeadBlock,
  AccessesOnlyInConstantFalseBranch,
  MixedLocalStore
};

[[nodiscard]] std::expected<VulkanModelShader, std::string>
makeTestVulkanShader(const VulkanExecutionPlan& plan,
                     const KernelExecution& kernel,
                     std::string binary_symbol = "");
[[nodiscard]] std::expected<VulkanModelShader, std::string>
makeTestVulkanShader(const VulkanExecutionPlan& plan,
                     const KernelExecution& kernel,
                     TestVulkanShaderMutation mutation,
                     std::string binary_symbol = "");

}  // namespace ncnn_mlir::vulkan::test
