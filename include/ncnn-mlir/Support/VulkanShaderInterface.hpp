#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace ncnn_mlir::vulkan {

struct VulkanShaderBinding final {
  std::uint32_t set;
  std::uint32_t binding;
  ResourceAccess access;
  std::uint64_t byte_size;
};

struct VulkanShaderPushRange final {
  std::uint32_t offset;
  std::uint32_t byte_size;
};

struct VulkanShaderInterface final {
  std::string entry_point;
  std::array<std::uint32_t, 3> local_size;
  std::vector<VulkanShaderBinding> bindings;
  std::vector<VulkanShaderPushRange> push_constants;
};

[[nodiscard]] std::expected<VulkanShaderInterface, std::string>
reflectVulkanShaderInterface(const VulkanExecutionPlan& plan,
                             const KernelExecution& kernel,
                             const VulkanModelShader& shader);

}  // namespace ncnn_mlir::vulkan
