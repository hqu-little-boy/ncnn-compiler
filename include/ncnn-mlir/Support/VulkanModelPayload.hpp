#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace ncnn_mlir::vulkan {

struct VulkanModelShader final {
  std::uint32_t kernel_id;
  std::string binary_symbol;
  std::vector<std::uint32_t> words;
};

struct VulkanModelWeight final {
  std::uint32_t resource_id;
  std::vector<std::uint8_t> bytes;
};

[[nodiscard]] std::expected<std::string, std::string>
serializeVulkanModelWeights(std::span<const VulkanModelWeight> weights);

[[nodiscard]] std::expected<std::vector<VulkanModelWeight>, std::string>
deserializeVulkanModelWeights(std::string_view json);

[[nodiscard]] std::expected<void, std::string> verifyVulkanModelWeights(
  const VulkanExecutionPlan& plan, std::span<const VulkanModelWeight> weights);

[[nodiscard]] std::expected<std::string, std::string> vulkanWeightSHA256(
  const TensorResource& resource, std::span<const std::uint8_t> bytes);

}  // namespace ncnn_mlir::vulkan
