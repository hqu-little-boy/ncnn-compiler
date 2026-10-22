#pragma once

#include <expected>
#include <span>
#include <string>

#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace ncnn_compile {

// Called only after source verification by VulkanProduct. No compiler-side
// program or payload types appear in the emitted implementation.
[[nodiscard]] std::expected<std::string, std::string>
generateVulkanLifecycleImplementation(
  const ncnn_mlir::vulkan::VulkanHostProgram& program,
  std::span<const ncnn_mlir::vulkan::VulkanModelShader> shaders,
  std::span<const ncnn_mlir::vulkan::VulkanModelWeight> weights);

[[nodiscard]] std::expected<std::string, std::string>
generateVulkanHostImplementation(
  const ncnn_mlir::vulkan::VulkanHostProgram& program,
  std::span<const ncnn_mlir::vulkan::VulkanModelShader> shaders,
  std::span<const ncnn_mlir::vulkan::VulkanModelWeight> weights);

}  // namespace ncnn_compile
