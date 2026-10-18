#pragma once

#include <cstdint>
#include <string_view>

#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"

namespace ncnn_mlir::vulkan::test {

struct LifecycleTestSettings final {
  std::string_view device_name_match;
  bool enable_validation;
  bool force_mapped_memory_sync;
  std::uint64_t max_context_memory_bytes;
  detail::VulkanOperationHook operation_hook;
};

[[nodiscard]] LifecycleTestSettings getLifecycleTestSettings();

}  // namespace ncnn_mlir::vulkan::test
