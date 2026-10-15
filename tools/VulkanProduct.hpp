#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace ncnn_compile {

[[nodiscard]] std::expected<void, std::string> validateVulkanSpirvInterface(
  const ncnn_mlir::vulkan::VulkanExecutionPlan& plan,
  std::string_view binary_ir,
  std::string_view disassembly);

[[nodiscard]] std::expected<void, std::string> validateVulkanSpirvInterface(
  const ncnn_mlir::vulkan::VulkanExecutionPlan& plan,
  const ncnn_mlir::vulkan::KernelExecution& kernel,
  std::string_view binary_symbol,
  std::string_view binary_ir,
  std::string_view disassembly);

struct VulkanModelPackage final {
  ncnn_mlir::vulkan::VulkanExecutionPlan plan;
  ncnn_mlir::vulkan::VulkanHostProgram host_program;
  std::vector<std::uint32_t> shader_words;
  std::vector<ncnn_mlir::vulkan::VulkanModelShader> shaders;
  std::vector<ncnn_mlir::vulkan::VulkanModelWeight> weights;
  std::vector<std::string> shader_sha256s;
  std::string binary_symbol;
  std::string plan_json;
  std::string header;
  std::string wrapper_source;
  std::string manifest_json;
  std::string build_identity;
  std::string shader_sha256;
  std::string plan_sha256;
};

[[nodiscard]] std::expected<VulkanModelPackage, std::string>
makeVulkanModelPackage(
  std::string_view model_name,
  std::string_view host_triple,
  std::string_view plan_json,
  std::span<const ncnn_mlir::vulkan::VulkanModelShader> shaders,
  std::span<const ncnn_mlir::vulkan::VulkanModelWeight> weights,
  std::string_view param_bytes,
  std::string_view bin_bytes);

[[nodiscard]] std::expected<VulkanModelPackage, std::string>
makeVulkanModelPackage(std::string_view model_name,
                       std::string_view host_triple,
                       std::string_view plan_json,
                       std::string_view spirv_bytes,
                       std::string_view param_bytes,
                       std::string_view bin_bytes);

[[nodiscard]] std::expected<std::string, std::string>
updateVulkanManifestDependencies(std::string_view manifest_json,
                                 std::vector<std::string> dependencies);

}  // namespace ncnn_compile
