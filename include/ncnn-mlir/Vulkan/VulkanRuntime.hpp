#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Vulkan/ModelABI.h"
#include "ncnn-mlir/Vulkan/VulkanDeviceTypes.hpp"

namespace ncnn_mlir::vulkan {

struct VulkanKernelBinary final {
  std::uint32_t kernel_id;
  std::span<const std::uint32_t> words;
};

struct VulkanResourcePayload final {
  std::uint32_t resource_id;
  std::span<const std::byte> bytes;
};

struct VulkanInputTensor final {
  std::uint32_t resource_id;
  std::span<const std::byte> bytes;
};

struct VulkanOutputTensor final {
  std::uint32_t resource_id;
  std::span<std::byte> bytes;
};

using VulkanRuntimeOperation = VulkanDeviceOperation;

struct VulkanRuntimeOptions final {
  std::uint32_t device_index;
  std::uint64_t max_context_memory_bytes;
  std::string device_name_match;
  bool enable_validation;
  bool force_mapped_memory_sync;  // Test hook for flush/invalidate coverage.
  std::function<std::optional<ncnn_vk_status>(VulkanRuntimeOperation)>
    failure_injector;
};

using VulkanRuntimeError = VulkanDeviceError;
using VulkanRuntimeDeviceInfo = VulkanDeviceInfo;

class VulkanRuntimeContext final {
 public:
  VulkanRuntimeContext(VulkanRuntimeContext&& other) noexcept;
  VulkanRuntimeContext& operator=(VulkanRuntimeContext&& other) noexcept;
  ~VulkanRuntimeContext();

  VulkanRuntimeContext(const VulkanRuntimeContext&) = delete;
  VulkanRuntimeContext& operator=(const VulkanRuntimeContext&) = delete;

  [[nodiscard]] static std::expected<VulkanRuntimeContext, VulkanRuntimeError>
  create(const VulkanExecutionPlan& plan,
         std::span<const VulkanKernelBinary> kernels,
         std::span<const VulkanResourcePayload> weights,
         const VulkanRuntimeOptions& options);

  [[nodiscard]] std::expected<void, VulkanRuntimeError> run(
    std::span<const VulkanInputTensor> inputs,
    std::span<const VulkanOutputTensor> outputs);
  [[nodiscard]] std::expected<void, VulkanRuntimeError> destroy();

  [[nodiscard]] VulkanRuntimeDeviceInfo get_device_info() const;
  [[nodiscard]] std::uint64_t get_peak_memory_bytes() const noexcept;
  [[nodiscard]] std::uint64_t get_weight_upload_count() const noexcept;
  [[nodiscard]] std::uint32_t get_validation_error_count() const noexcept;

 private:
  class Impl;
  explicit VulkanRuntimeContext(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] VulkanRuntimeOptions makeVulkanRuntimeOptions() noexcept;
[[nodiscard]] std::string_view getVulkanRuntimeStatusName(
  ncnn_vk_status status) noexcept;

}  // namespace ncnn_mlir::vulkan
