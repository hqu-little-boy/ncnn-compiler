#pragma once

#include <cstdint>
#include <string_view>

#include <vulkan/vulkan.h>

namespace fake_vulkan {

void reset();
void enable();
void fail_next(std::string_view function, VkResult result);
void set_max_memory_allocation_count(std::uint32_t count);
[[nodiscard]] std::uint64_t call_count(std::string_view function);
[[nodiscard]] std::uint64_t live_count(std::string_view object_type);
[[nodiscard]] bool saw_whole_allocation_range(std::string_view function);
[[nodiscard]] bool saw_whole_allocation_map();
[[nodiscard]] std::uint64_t last_mapped_allocation_size();
[[nodiscard]] bool saw_destruction_before_wait_resolution();

}  // namespace fake_vulkan
