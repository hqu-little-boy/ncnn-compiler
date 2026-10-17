#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "ncnn-mlir/Vulkan/VulkanDeviceTypes.hpp"
#include <vulkan/vulkan.h>

namespace ncnn_mlir::vulkan {

namespace detail {
using VulkanOperationHook =
  std::function<std::optional<ncnn_vk_status>(VulkanDeviceOperation)>;
}

struct VulkanDeviceOptions final {
  std::uint32_t device_index;
  std::string_view device_name_match;
  bool enable_validation;
};

// Limits for one compute pipeline. Descriptor counts are storage-buffer-only
// and therefore apply equally to descriptor-set and per-stage resource caps.
struct VulkanDeviceLimitRequirements final {
  std::uint32_t local_size_x;
  std::uint32_t dispatch_count_x;
  std::uint64_t storage_buffer_range;
  std::uint32_t push_constant_bytes;
  std::uint32_t descriptor_set_count;
  std::uint32_t storage_buffer_descriptor_count;
};

struct VulkanAllocationBudget final {
  std::uint64_t persistent_bytes;
  std::uint64_t temporary_staging_bytes;
  std::uint64_t allocation_count;
};

struct VulkanMemoryTypeSelection final {
  std::uint32_t index;
  bool is_coherent;
};

[[nodiscard]] std::expected<void, VulkanDeviceError> validateDeviceLimits(
  const VkPhysicalDeviceProperties& properties,
  std::span<const VulkanDeviceLimitRequirements> requirements);
[[nodiscard]] std::expected<void, VulkanDeviceError> validateAllocationBudget(
  const VulkanAllocationBudget& budget,
  std::uint64_t max_memory_allocation_count,
  std::uint64_t max_context_memory_bytes);

// Move operations preserve the implementation address used by validation
// callbacks. Destroy all child resources before destroying their context.
class VulkanDeviceContext final {
 public:
  VulkanDeviceContext(VulkanDeviceContext&& other) noexcept;
  VulkanDeviceContext& operator=(VulkanDeviceContext&& other) noexcept;
  VulkanDeviceContext(const VulkanDeviceContext&) = delete;
  VulkanDeviceContext& operator=(const VulkanDeviceContext&) = delete;
  ~VulkanDeviceContext();

  [[nodiscard]] static std::expected<VulkanDeviceContext, VulkanDeviceError>
  create(const VulkanDeviceOptions& options,
         const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> destroy();

  [[nodiscard]] VkInstance get_instance() const noexcept;
  [[nodiscard]] VkPhysicalDevice get_physical_device() const noexcept;
  [[nodiscard]] VkDevice get_device() const noexcept;
  [[nodiscard]] VkQueue get_queue() const noexcept;
  [[nodiscard]] std::uint32_t get_queue_family_index() const noexcept;
  [[nodiscard]] const VkPhysicalDeviceProperties& get_properties()
    const noexcept;
  [[nodiscard]] const VkPhysicalDeviceMemoryProperties& get_memory_properties()
    const noexcept;
  [[nodiscard]] VulkanDeviceInfo get_device_info() const;
  [[nodiscard]] std::uint32_t get_validation_error_count() const noexcept;
  [[nodiscard]] std::expected<VulkanMemoryTypeSelection, VulkanDeviceError>
  get_memory_type(std::uint32_t allowed_types,
                  VkMemoryPropertyFlags required,
                  VkMemoryPropertyFlags preferred,
                  bool prefer_non_coherent = false) const;
  [[nodiscard]] bool has_memory_type(
    std::uint32_t allowed_types, VkMemoryPropertyFlags required) const noexcept;
  [[nodiscard]] std::expected<void, VulkanDeviceError> wait_idle();

 private:
  class Impl;
  explicit VulkanDeviceContext(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

// Buffers, command resources, descriptor pools, and pipelines require their
// device context to outlive them; explicit destroy supports diagnostic cleanup.
struct VulkanBufferCreateInfo final {
  VkDeviceSize size;
  VkBufferUsageFlags usage;
  bool host_visible;
  bool force_mapped_memory_sync;
};

class VulkanBuffer final {
 public:
  VulkanBuffer(VulkanBuffer&& other) noexcept;
  VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;
  ~VulkanBuffer();

  [[nodiscard]] static std::expected<VulkanBuffer, VulkanDeviceError> prepare(
    VulkanDeviceContext& context,
    const VulkanBufferCreateInfo& create_info,
    const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> allocate_and_bind(
    bool map, const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> flush(
    const detail::VulkanOperationHook& operation_hook = {}) const;
  [[nodiscard]] std::expected<void, VulkanDeviceError> invalidate(
    const detail::VulkanOperationHook& operation_hook = {}) const;
  [[nodiscard]] std::expected<void, VulkanDeviceError> destroy() noexcept;

  [[nodiscard]] VkBuffer get_handle() const noexcept;
  [[nodiscard]] VkDeviceMemory get_memory() const noexcept;
  [[nodiscard]] VkDeviceSize get_logical_size() const noexcept;
  [[nodiscard]] VkDeviceSize get_allocation_size() const noexcept;
  [[nodiscard]] std::uint32_t get_memory_type_index() const noexcept;
  [[nodiscard]] std::span<std::byte> get_mapped() const noexcept;
  [[nodiscard]] bool is_coherent() const noexcept;
  [[nodiscard]] bool is_prepared() const noexcept;
  [[nodiscard]] bool is_allocated() const noexcept;

 private:
  class Impl;
  explicit VulkanBuffer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

class VulkanCommandResources final {
 public:
  VulkanCommandResources(VulkanCommandResources&& other) noexcept;
  VulkanCommandResources& operator=(VulkanCommandResources&& other) noexcept;
  VulkanCommandResources(const VulkanCommandResources&) = delete;
  VulkanCommandResources& operator=(const VulkanCommandResources&) = delete;
  ~VulkanCommandResources();

  [[nodiscard]] static std::expected<VulkanCommandResources, VulkanDeviceError>
  create(VulkanDeviceContext& context,
         const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> reset();
  [[nodiscard]] std::expected<void, VulkanDeviceError> begin();
  [[nodiscard]] std::expected<void, VulkanDeviceError> end();
  [[nodiscard]] std::expected<void, VulkanDeviceError> submit_and_wait(
    const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> wait(
    const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> destroy();

  [[nodiscard]] VkCommandPool get_command_pool() const noexcept;
  [[nodiscard]] VkCommandBuffer get_command_buffer() const noexcept;
  [[nodiscard]] VkFence get_fence() const noexcept;

 private:
  class Impl;
  explicit VulkanCommandResources(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

// Owns descriptor sets and set layouts used by pipelines; destroy pipelines
// and wait for submissions before destroying the shared pool.
class VulkanDescriptorPool final {
 public:
  VulkanDescriptorPool(VulkanDescriptorPool&& other) noexcept;
  VulkanDescriptorPool& operator=(VulkanDescriptorPool&& other) noexcept;
  VulkanDescriptorPool(const VulkanDescriptorPool&) = delete;
  VulkanDescriptorPool& operator=(const VulkanDescriptorPool&) = delete;
  ~VulkanDescriptorPool();

  [[nodiscard]] static std::expected<VulkanDescriptorPool, VulkanDeviceError>
  create(VulkanDeviceContext& context,
         std::uint32_t max_sets,
         std::uint32_t storage_buffer_descriptor_count,
         const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> destroy() noexcept;

  [[nodiscard]] VkDescriptorPool get_handle() const noexcept;
  [[nodiscard]] std::uint32_t get_max_sets() const noexcept;
  [[nodiscard]] std::uint32_t get_storage_buffer_descriptor_count()
    const noexcept;
  [[nodiscard]] bool is_for(const VulkanDeviceContext& context) const noexcept;

 private:
  friend class VulkanComputePipeline;

  class Impl;
  explicit VulkanDescriptorPool(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

struct VulkanPipelineBinding final {
  std::uint32_t set;
  std::uint32_t binding;
  VkBuffer buffer;
  VkDeviceSize buffer_size;
  VkDeviceSize offset;
  VkDeviceSize range;
};

struct VulkanPushConstantRange final {
  VkShaderStageFlags stage_flags;
  std::uint32_t offset;
  std::uint32_t size;
};

class VulkanComputePipeline final {
 public:
  VulkanComputePipeline(VulkanComputePipeline&& other) noexcept;
  VulkanComputePipeline& operator=(VulkanComputePipeline&& other) noexcept;
  VulkanComputePipeline(const VulkanComputePipeline&) = delete;
  VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;
  ~VulkanComputePipeline();

  [[nodiscard]] static std::expected<VulkanComputePipeline, VulkanDeviceError>
  create(VulkanDeviceContext& context,
         VulkanDescriptorPool& descriptor_pool,
         std::span<const std::uint32_t> spirv_words,
         std::string_view entry_point,
         std::span<const VulkanPipelineBinding> bindings,
         std::span<const VulkanPushConstantRange> push_constants,
         const detail::VulkanOperationHook& operation_hook = {});
  [[nodiscard]] std::expected<void, VulkanDeviceError> destroy() noexcept;

  [[nodiscard]] VkPipeline get_pipeline() const noexcept;
  [[nodiscard]] VkPipelineLayout get_layout() const noexcept;
  [[nodiscard]] std::span<const VkDescriptorSet> get_descriptor_sets()
    const noexcept;

 private:
  class Impl;
  explicit VulkanComputePipeline(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ncnn_mlir::vulkan
