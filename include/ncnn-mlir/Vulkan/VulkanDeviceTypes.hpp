#pragma once

#include <cstdint>
#include <string>

#include "ncnn-mlir/Vulkan/ModelABI.h"

namespace ncnn_mlir::vulkan {

inline constexpr std::uint64_t kVulkanDeviceHelperMaxBufferBytes =
  128ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kVulkanDeviceHelperDefaultBudgetBytes =
  512ULL * 1024ULL * 1024ULL;

enum class VulkanDeviceOperation : std::uint8_t {
  CreateInstance,
  SelectDevice,
  CreateDevice,
  CreateBuffer,
  AllocateMemory,
  CreateDescriptor,
  CreateShaderModule,
  CreatePipeline,
  CreateCommandResources,
  Submit,
  Wait,
  FlushMappedMemory,
  InvalidateMappedMemory,
};

struct VulkanDeviceError final {
  ncnn_vk_status status;
  std::string diagnostic;
};

struct VulkanDeviceInfo final {
  std::string name;
  std::uint32_t api_version;
  std::uint32_t driver_version;
  std::uint32_t vendor_id;
  std::uint32_t device_id;
};

}  // namespace ncnn_mlir::vulkan
