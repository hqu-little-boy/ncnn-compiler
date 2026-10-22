#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

namespace fake_vulkan {

struct DescriptorWrite final {
  VkStructureType structure_type;
  bool has_next;
  VkDescriptorSet set;
  std::uint32_t binding;
  std::uint32_t array_element;
  std::uint32_t descriptor_count;
  VkDescriptorType descriptor_type;
  bool has_image_info;
  bool has_buffer_info;
  bool has_texel_buffer_view;
  VkBuffer buffer;
  VkDeviceSize offset;
  VkDeviceSize range;
};

void reset();
void enable();
void fail_next(std::string_view function, VkResult result);
void set_max_memory_allocation_count(std::uint32_t count);
void return_null_mapping_on_call(std::uint64_t call_index);
void inject_validation_error_on_invalidate_call(std::uint64_t call_index);
[[nodiscard]] std::uint64_t injected_validation_error_count();
[[nodiscard]] std::uint64_t call_count(std::string_view function);
[[nodiscard]] std::uint64_t live_count(std::string_view object_type);
[[nodiscard]] std::optional<std::uint32_t> buffer_slot(VkBuffer buffer);
[[nodiscard]] std::optional<std::uint32_t> command_buffer_slot(
  VkCommandBuffer command_buffer);
[[nodiscard]] std::optional<std::uint32_t> pipeline_slot(VkPipeline pipeline);
[[nodiscard]] std::optional<std::uint32_t> pipeline_layout_slot(
  VkPipelineLayout layout);
[[nodiscard]] std::optional<std::uint32_t> descriptor_set_slot(
  VkDescriptorSet descriptor_set);
[[nodiscard]] std::vector<DescriptorWrite> descriptor_writes();
[[nodiscard]] std::uint64_t descriptor_copy_count();
[[nodiscard]] bool saw_whole_allocation_range(std::string_view function);
[[nodiscard]] bool saw_whole_allocation_map();
[[nodiscard]] std::uint64_t last_mapped_allocation_size();
[[nodiscard]] bool saw_destruction_before_wait_resolution();

}  // namespace fake_vulkan
