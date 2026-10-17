#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "fakeVulkan/fake_vulkan.hpp"
#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"

namespace {

using ncnn_mlir::vulkan::validateAllocationBudget;
using ncnn_mlir::vulkan::validateDeviceLimits;
using ncnn_mlir::vulkan::VulkanAllocationBudget;
using ncnn_mlir::vulkan::VulkanBuffer;
using ncnn_mlir::vulkan::VulkanBufferCreateInfo;
using ncnn_mlir::vulkan::VulkanCommandResources;
using ncnn_mlir::vulkan::VulkanComputePipeline;
using ncnn_mlir::vulkan::VulkanDescriptorPool;
using ncnn_mlir::vulkan::VulkanDeviceContext;
using ncnn_mlir::vulkan::VulkanDeviceError;
using ncnn_mlir::vulkan::VulkanDeviceLimitRequirements;
using ncnn_mlir::vulkan::VulkanDeviceOperation;
using ncnn_mlir::vulkan::VulkanDeviceOptions;
using ncnn_mlir::vulkan::VulkanPipelineBinding;
using ncnn_mlir::vulkan::detail::VulkanOperationHook;

int failures = 0;

void expect(bool condition, std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

VkPhysicalDeviceProperties limits_fixture() {
  VkPhysicalDeviceProperties properties{};
  properties.apiVersion = VK_API_VERSION_1_1;
  properties.limits.maxStorageBufferRange = 128U * 1024U * 1024U;
  properties.limits.maxComputeWorkGroupInvocations = 256;
  properties.limits.maxComputeWorkGroupSize[0] = 256;
  properties.limits.maxComputeWorkGroupCount[0] = 65535;
  properties.limits.maxPushConstantsSize = 128;
  properties.limits.maxBoundDescriptorSets = 4;
  properties.limits.maxPerStageDescriptorStorageBuffers = 16;
  properties.limits.maxDescriptorSetStorageBuffers = 16;
  properties.limits.maxPerStageResources = 32;
  return properties;
}

VulkanDeviceOptions fake_options() {
  return VulkanDeviceOptions{.device_index = NCNN_VK_DEVICE_AUTO,
                             .device_name_match = {},
                             .enable_validation = false};
}

std::expected<VulkanDeviceContext, VulkanDeviceError> create_fake_context(
  VulkanOperationHook hook = {}) {
  return VulkanDeviceContext::create(fake_options(), std::move(hook));
}

void expect_no_live_fake_objects(std::string_view description) {
  const bool empty = fake_vulkan::live_count("instance") == 0 &&
                     fake_vulkan::live_count("device") == 0 &&
                     fake_vulkan::live_count("buffer") == 0 &&
                     fake_vulkan::live_count("memory") == 0 &&
                     fake_vulkan::live_count("command_pool") == 0 &&
                     fake_vulkan::live_count("command_buffer") == 0 &&
                     fake_vulkan::live_count("fence") == 0 &&
                     fake_vulkan::live_count("descriptor_pool") == 0 &&
                     fake_vulkan::live_count("descriptor_set") == 0 &&
                     fake_vulkan::live_count("descriptor_set_layout") == 0 &&
                     fake_vulkan::live_count("shader_module") == 0 &&
                     fake_vulkan::live_count("pipeline_layout") == 0 &&
                     fake_vulkan::live_count("pipeline") == 0;
  expect(empty, description);
}

void test_limits() {
  constexpr std::uint32_t max_storage = 128U * 1024U * 1024U;
  const VulkanDeviceLimitRequirements exact{
    .local_size_x = 256,
    .dispatch_count_x = 65535,
    .storage_buffer_range = max_storage,
    .push_constant_bytes = 128,
    .descriptor_set_count = 4,
    .storage_buffer_descriptor_count = 16,
  };
  const std::array requirements{exact};
  VkPhysicalDeviceProperties properties = limits_fixture();
  expect(validateDeviceLimits(properties, requirements).has_value(),
         "device-limit exact boundaries are accepted");
  expect(validateDeviceLimits(properties, {}).has_value(),
         "empty limit requirements are accepted");

  auto rejected = [&](VulkanDeviceLimitRequirements invalid,
                      std::string_view description) {
    const std::array invalid_requirements{invalid};
    expect(!validateDeviceLimits(properties, invalid_requirements),
           description);
  };
  auto invalid = exact;
  ++invalid.local_size_x;
  rejected(invalid, "workgroup size above the device limit is rejected");
  invalid = exact;
  ++invalid.dispatch_count_x;
  rejected(invalid, "dispatch count above the device limit is rejected");
  invalid = exact;
  ++invalid.storage_buffer_range;
  rejected(invalid, "storage range above 128 MiB is rejected");
  invalid = exact;
  ++invalid.push_constant_bytes;
  rejected(invalid, "push constant above the device limit is rejected");
  invalid = exact;
  ++invalid.descriptor_set_count;
  rejected(invalid, "descriptor set count above the device limit is rejected");
  invalid = exact;
  ++invalid.storage_buffer_descriptor_count;
  rejected(invalid,
           "storage descriptor count above the device limit is rejected");

  const VulkanDeviceLimitRequirements five_descriptors{
    .local_size_x = 1,
    .dispatch_count_x = 1,
    .storage_buffer_range = 16,
    .push_constant_bytes = 0,
    .descriptor_set_count = 1,
    .storage_buffer_descriptor_count = 5};
  VkPhysicalDeviceProperties set_limit_properties = limits_fixture();
  set_limit_properties.limits.maxPerStageDescriptorStorageBuffers = 8;
  set_limit_properties.limits.maxDescriptorSetStorageBuffers = 4;
  set_limit_properties.limits.maxPerStageResources = 8;
  const std::array five_descriptor_requirements{five_descriptors};
  const auto set_limit_error =
    validateDeviceLimits(set_limit_properties, five_descriptor_requirements);
  expect(
    !set_limit_error &&
      set_limit_error.error().status == NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
    "maxDescriptorSetStorageBuffers below descriptor count is unsupported");
  set_limit_properties.limits.maxDescriptorSetStorageBuffers = 8;
  set_limit_properties.limits.maxPerStageResources = 4;
  const auto resource_limit_error =
    validateDeviceLimits(set_limit_properties, five_descriptor_requirements);
  expect(!resource_limit_error && resource_limit_error.error().status ==
                                    NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
         "maxPerStageResources below descriptor count is unsupported");
  expect(
    fake_vulkan::call_count("vkCreateShaderModule") == 0 &&
      fake_vulkan::call_count("vkAllocateMemory") == 0,
    "synthetic descriptor-limit rejections perform no shader or memory calls");

  properties.limits.maxStorageBufferRange = max_storage - 1;
  expect(!validateDeviceLimits(properties, requirements),
         "a device limit below the 128 MiB storage contract is rejected");
}

void test_allocation_budget() {
  constexpr std::uint64_t mib = 1024U * 1024U;
  constexpr std::uint64_t context_budget = 512U * mib;
  expect(validateAllocationBudget(
           VulkanAllocationBudget{.persistent_bytes = 384U * mib,
                                  .temporary_staging_bytes = 128U * mib,
                                  .allocation_count = 4096},
           4096,
           context_budget)
           .has_value(),
         "512 MiB context and allocation-count exact boundaries are accepted");
  expect(validateAllocationBudget(
           VulkanAllocationBudget{.persistent_bytes = 128U * mib,
                                  .temporary_staging_bytes = 0,
                                  .allocation_count = 1},
           1,
           context_budget)
           .has_value(),
         "one 128 MiB allocation fits the context budget");
  expect(!validateAllocationBudget(
           VulkanAllocationBudget{.persistent_bytes = 384U * mib,
                                  .temporary_staging_bytes = (128U * mib) + 1,
                                  .allocation_count = 1},
           4096,
           context_budget),
         "one byte above the configured context budget is rejected");
  expect(!validateAllocationBudget(
           VulkanAllocationBudget{.persistent_bytes = 1,
                                  .temporary_staging_bytes = 0,
                                  .allocation_count = 4097},
           4096,
           context_budget),
         "allocation count above maxMemoryAllocationCount is rejected");
  expect(!validateAllocationBudget(
           VulkanAllocationBudget{
             .persistent_bytes = std::numeric_limits<std::uint64_t>::max(),
             .temporary_staging_bytes = 1,
             .allocation_count = 1},
           4096,
           context_budget),
         "allocation byte-count overflow is rejected");
  expect(!validateAllocationBudget(
           VulkanAllocationBudget{.persistent_bytes = 0,
                                  .temporary_staging_bytes = 0,
                                  .allocation_count = 1},
           0,
           context_budget),
         "allocation request is rejected when device allocation count is zero");
}

void test_context_hooks_move_and_memory_types() {
  for (VulkanDeviceOperation operation :
       {VulkanDeviceOperation::CreateInstance,
        VulkanDeviceOperation::SelectDevice,
        VulkanDeviceOperation::CreateDevice}) {
    fake_vulkan::reset();
    auto result = create_fake_context([operation](VulkanDeviceOperation current)
                                        -> std::optional<ncnn_vk_status> {
      return current == operation
               ? std::optional<ncnn_vk_status>{NCNN_VK_STATUS_DEVICE_LOST}
               : std::nullopt;
    });
    expect(!result, "context create hook reports injected failures");
    if (!result) {
      expect(result.error().status == NCNN_VK_STATUS_DEVICE_LOST,
             "injected context failure preserves device-lost status");
    }
    expect_no_live_fake_objects("context hook failure cleans partial objects");
  }

  fake_vulkan::reset();
  {
    auto created = create_fake_context();
    expect(created.has_value(), "fake Vulkan context initializes");
    if (created) {
      const VkInstance original_instance = created->get_instance();
      const auto coherent =
        created->get_memory_type(0x7U,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      expect(coherent && coherent->index == 1 && coherent->is_coherent,
             "preferred coherent host-visible memory type is selected");
      const auto noncoherent =
        created->get_memory_type(0x1U,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      expect(
        noncoherent && noncoherent->index == 0 && !noncoherent->is_coherent,
        "required-only memory type reports non-coherent selection");
      expect(
        created->has_memory_type(0x4U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        "memory-type predicate finds required device-local memory");
      expect(!created->get_memory_type(0x4U,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
             "memory-type selection fails when no type satisfies requirements");

      VulkanDeviceContext moved(std::move(*created));
      expect(created->get_instance() == VK_NULL_HANDLE,
             "moved-from context relinquishes its instance");
      expect(moved.get_instance() == original_instance,
             "moved-to context retains the original instance");
      expect(moved.destroy().has_value(), "explicit context destroy succeeds");
      expect(moved.destroy().has_value(), "context destroy is idempotent");
    }
  }
  {
    auto destination_result = create_fake_context();
    auto source_result = create_fake_context();
    expect(destination_result.has_value() && source_result.has_value(),
           "contexts for move assignment initialize");
    if (destination_result && source_result) {
      VulkanDeviceContext destination(std::move(*destination_result));
      VulkanDeviceContext source(std::move(*source_result));
      const VkInstance replacement_instance = source.get_instance();
      const std::uint64_t destroys_before =
        fake_vulkan::call_count("vkDestroyDevice");
      destination = std::move(source);
      expect(destination.get_instance() == replacement_instance,
             "move assignment transfers source device ownership");
      expect(source.get_instance() == VK_NULL_HANDLE,
             "move-assigned source context relinquishes ownership");
      expect(fake_vulkan::call_count("vkDestroyDevice") == destroys_before + 1,
             "move assignment releases the destination's previous device once");
    }
  }
  expect_no_live_fake_objects("context move/destroy releases ownership once");
}

void test_buffer_failures_and_noncoherent_sync() {
  fake_vulkan::reset();
  auto context_result = create_fake_context();
  expect(context_result.has_value(), "buffer test context initializes");
  if (!context_result) {
    return;
  }
  VulkanDeviceContext& context = *context_result;
  constexpr VkDeviceSize logical_size = 257U * sizeof(float);
  const VulkanBufferCreateInfo info{.size = logical_size,
                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    .host_visible = true,
                                    .force_mapped_memory_sync = true};

  fake_vulkan::fail_next("vkCreateBuffer", VK_ERROR_OUT_OF_HOST_MEMORY);
  auto create_failure = VulkanBuffer::prepare(context, info);
  expect(!create_failure &&
           create_failure.error().status == NCNN_VK_STATUS_OUT_OF_MEMORY,
         "vkCreateBuffer failure maps to out-of-memory");

  auto prepared_result = VulkanBuffer::prepare(context, info);
  expect(prepared_result.has_value(),
         "buffer prepare creates and queries buffer");
  if (!prepared_result) {
    return;
  }
  VulkanBuffer prepared(std::move(*prepared_result));
  expect(prepared.is_prepared() && !prepared.is_allocated(),
         "buffer prepare does not allocate memory");
  expect(prepared.get_allocation_size() > prepared.get_logical_size(),
         "actual Vulkan allocation requirement exceeds logical buffer size");

  auto staging_result = VulkanBuffer::prepare(context, info);
  expect(staging_result.has_value(),
         "temporary staging requirement is queried");
  if (!staging_result) {
    return;
  }
  VulkanBuffer staging(std::move(*staging_result));
  expect(staging.get_allocation_size() > staging.get_logical_size(),
         "staging requirement includes Vulkan allocation padding");
  const std::uint64_t persistent_requirement = prepared.get_allocation_size();
  const std::uint64_t staging_requirement = staging.get_allocation_size();
  const VulkanAllocationBudget aggregate_budget{
    .persistent_bytes = persistent_requirement,
    .temporary_staging_bytes = staging_requirement,
    .allocation_count = 2};
  const std::uint64_t total_requirement =
    persistent_requirement + staging_requirement;
  const std::uint64_t allocations_before_budget_check =
    fake_vulkan::call_count("vkAllocateMemory");
  expect(!validateAllocationBudget(
           aggregate_budget,
           context.get_properties().limits.maxMemoryAllocationCount,
           total_requirement - 1),
         "actual persistent plus temporary staging requirements exceed budget");
  expect(fake_vulkan::call_count("vkAllocateMemory") ==
           allocations_before_budget_check,
         "failed requirements budget check occurs before vkAllocateMemory");
  expect(!validateAllocationBudget(aggregate_budget, 1, total_requirement),
         "combined persistent/staging allocation count is checked preallocate");
  expect(fake_vulkan::call_count("vkAllocateMemory") ==
           allocations_before_budget_check,
         "allocation-count budget rejection occurs before vkAllocateMemory");

  fake_vulkan::fail_next("vkAllocateMemory", VK_ERROR_OUT_OF_DEVICE_MEMORY);
  auto allocation_failure = prepared.allocate_and_bind(true);
  expect(!allocation_failure &&
           allocation_failure.error().status == NCNN_VK_STATUS_OUT_OF_MEMORY,
         "vkAllocateMemory failure maps to out-of-memory");
  expect(!prepared.is_allocated(), "allocation failure owns no memory object");
  expect(prepared.destroy().has_value(),
         "prepared buffer can be destroyed after allocation failure");

  auto bind_result = VulkanBuffer::prepare(context, info);
  expect(bind_result.has_value(), "buffer can be prepared before bind fault");
  if (bind_result) {
    VulkanBuffer buffer(std::move(*bind_result));
    fake_vulkan::fail_next("vkBindBufferMemory", VK_ERROR_DEVICE_LOST);
    auto bind_failure = buffer.allocate_and_bind(true);
    expect(!bind_failure &&
             bind_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "vkBindBufferMemory device-lost failure is preserved");
    expect(buffer.destroy().has_value(),
           "partially bound buffer cleanup succeeds");
  }

  auto map_result = VulkanBuffer::prepare(context, info);
  expect(map_result.has_value(), "buffer can be prepared before map fault");
  if (map_result) {
    VulkanBuffer buffer(std::move(*map_result));
    fake_vulkan::fail_next("vkMapMemory", VK_ERROR_MEMORY_MAP_FAILED);
    auto map_failure = buffer.allocate_and_bind(true);
    expect(!map_failure, "vkMapMemory failure is returned");
    expect(buffer.destroy().has_value(),
           "mapped buffer partial cleanup succeeds");
  }

  auto noncoherent_result = VulkanBuffer::prepare(context, info);
  expect(noncoherent_result.has_value(),
         "buffer can be prepared for sync coverage");
  if (noncoherent_result) {
    VulkanBuffer noncoherent(std::move(*noncoherent_result));
    expect(noncoherent.allocate_and_bind(true).has_value(),
           "forced-sync buffer allocates and maps");
    const auto mapped = noncoherent.get_mapped();
    expect(mapped.size() == logical_size,
           "mapped observer span exposes only the logical buffer range");
    expect(fake_vulkan::saw_whole_allocation_range("vkMapMemory") &&
             fake_vulkan::last_mapped_allocation_size() ==
               noncoherent.get_allocation_size() &&
             noncoherent.get_allocation_size() > logical_size,
           "Vulkan maps the complete actual non-coherent allocation");
    expect(
      !noncoherent.is_coherent() && noncoherent.get_memory_type_index() == 0,
      "forced-sync buffer selects the actual non-coherent fake memory type");
    if (!mapped.empty()) {
      mapped.front() = std::byte{0x5A};
    }
    expect(noncoherent.flush().has_value(), "non-coherent flush succeeds");
    expect(fake_vulkan::saw_whole_allocation_range("vkFlushMappedMemoryRanges"),
           "flush uses offset zero and VK_WHOLE_SIZE");
    expect(noncoherent.invalidate().has_value(),
           "non-coherent invalidate succeeds");
    expect(
      fake_vulkan::saw_whole_allocation_range("vkInvalidateMappedMemoryRanges"),
      "invalidate uses offset zero and VK_WHOLE_SIZE");

    fake_vulkan::fail_next("vkFlushMappedMemoryRanges", VK_ERROR_DEVICE_LOST);
    auto flush_failure = noncoherent.flush();
    expect(!flush_failure &&
             flush_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "flush failure reports device-lost status");
    fake_vulkan::fail_next("vkInvalidateMappedMemoryRanges",
                           VK_ERROR_DEVICE_LOST);
    auto invalidate_failure = noncoherent.invalidate();
    expect(!invalidate_failure &&
             invalidate_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "invalidate failure reports device-lost status");
    expect(noncoherent.destroy().has_value(), "non-coherent buffer destroys");
    expect(noncoherent.destroy().has_value(), "buffer destroy is idempotent");
  }

  expect(staging.destroy().has_value(), "temporary staging buffer destroys");
  expect(context.destroy().has_value(), "buffer test context destroys");
  expect_no_live_fake_objects(
    "buffer failures and cleanup release all objects");
}

void test_pipeline_failures() {
  constexpr std::array<std::uint32_t, 5> shader_words{
    0x07230203U, 0x00010000U, 0U, 1U, 0U};
  fake_vulkan::reset();
  auto context_result = create_fake_context();
  expect(context_result.has_value(), "pipeline test context initializes");
  if (!context_result) {
    return;
  }
  VulkanDeviceContext& context = *context_result;
  const VulkanBufferCreateInfo buffer_info{
    .size = 257U * sizeof(float),
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .host_visible = false,
    .force_mapped_memory_sync = false};
  auto buffer_result = VulkanBuffer::prepare(context, buffer_info);
  expect(buffer_result.has_value(), "pipeline storage buffer prepares");
  if (!buffer_result) {
    return;
  }
  VulkanBuffer buffer(std::move(*buffer_result));
  expect(buffer.allocate_and_bind(false).has_value(),
         "pipeline storage buffer allocates");
  const std::array bindings{
    VulkanPipelineBinding{.set = 0,
                          .binding = 0,
                          .buffer = buffer.get_handle(),
                          .buffer_size = buffer_info.size,
                          .offset = 0,
                          .range = buffer_info.size}};

  fake_vulkan::fail_next("vkCreateDescriptorPool", VK_ERROR_OUT_OF_HOST_MEMORY);
  auto pool_failure = VulkanDescriptorPool::create(context, 1, 1);
  expect(!pool_failure &&
           pool_failure.error().status == NCNN_VK_STATUS_OUT_OF_MEMORY,
         "descriptor-pool Vulkan allocation failure is reported");
  expect(fake_vulkan::live_count("descriptor_pool") == 0,
         "failed descriptor-pool create owns no pool");
  auto pool_hook_failure = VulkanDescriptorPool::create(
    context,
    1,
    1,
    [](VulkanDeviceOperation operation) -> std::optional<ncnn_vk_status> {
      return operation == VulkanDeviceOperation::CreateDescriptor
               ? std::optional<ncnn_vk_status>{NCNN_VK_STATUS_DEVICE_LOST}
               : std::nullopt;
    });
  expect(
    !pool_hook_failure &&
      pool_hook_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
    "descriptor-pool operation hook is testable without production backend");

  const std::array failure_points{
    std::pair{"vkCreateShaderModule", VK_ERROR_OUT_OF_HOST_MEMORY},
    std::pair{"vkCreateDescriptorSetLayout", VK_ERROR_OUT_OF_HOST_MEMORY},
    std::pair{"vkAllocateDescriptorSets", VK_ERROR_OUT_OF_HOST_MEMORY},
    std::pair{"vkCreatePipelineLayout", VK_ERROR_OUT_OF_HOST_MEMORY},
    std::pair{"vkCreateComputePipelines", VK_ERROR_OUT_OF_HOST_MEMORY},
  };
  for (const auto& [function, result] : failure_points) {
    auto pool_result = VulkanDescriptorPool::create(context, 1, 1);
    expect(pool_result.has_value(), "per-failure descriptor pool creates");
    if (!pool_result) {
      continue;
    }
    VulkanDescriptorPool pool(std::move(*pool_result));
    fake_vulkan::fail_next(function, result);
    auto pipeline = VulkanComputePipeline::create(
      context, pool, shader_words, "add", bindings, {});
    expect(!pipeline, "pipeline construction returns injected Vulkan failure");
    const bool pool_allocated_layout =
      std::string_view(function) == "vkAllocateDescriptorSets" ||
      std::string_view(function) == "vkCreatePipelineLayout" ||
      std::string_view(function) == "vkCreateComputePipelines";
    const bool pool_allocated_set =
      std::string_view(function) == "vkCreatePipelineLayout" ||
      std::string_view(function) == "vkCreateComputePipelines";
    expect(fake_vulkan::live_count("shader_module") == 0 &&
             fake_vulkan::live_count("pipeline_layout") == 0 &&
             fake_vulkan::live_count("pipeline") == 0,
           "pipeline partial construction cleans pipeline-owned objects");
    expect(fake_vulkan::live_count("descriptor_pool") == 1,
           "pipeline leaves the caller-owned descriptor pool alive");
    expect(fake_vulkan::live_count("descriptor_set_layout") ==
             static_cast<std::uint64_t>(pool_allocated_layout),
           "created set layout remains owned by the shared descriptor pool");
    expect(fake_vulkan::live_count("descriptor_set") ==
             static_cast<std::uint64_t>(pool_allocated_set),
           "allocated descriptor set remains owned by the shared pool");
    expect(pool.destroy().has_value(),
           "caller destroys shared pool after pipeline failure");
    expect(
      fake_vulkan::live_count("descriptor_set") == 0 &&
        fake_vulkan::live_count("descriptor_set_layout") == 0,
      "pool destruction releases any partially allocated sets and layouts");
  }

  auto capacity_result = VulkanDescriptorPool::create(context, 0, 0);
  expect(capacity_result.has_value(),
         "zero-capacity pool is a valid empty owner");
  if (capacity_result) {
    VulkanDescriptorPool empty_pool(std::move(*capacity_result));
    const std::uint64_t shader_calls =
      fake_vulkan::call_count("vkCreateShaderModule");
    auto undersized = VulkanComputePipeline::create(
      context, empty_pool, shader_words, "add", bindings, {});
    expect(!undersized &&
             undersized.error().status == NCNN_VK_STATUS_INVALID_ARGUMENT,
           "undersized descriptor pool is rejected before Vulkan creation");
    expect(fake_vulkan::call_count("vkCreateShaderModule") == shader_calls,
           "pool capacity rejection precedes shader-module creation");
    expect(empty_pool.destroy().has_value(), "empty descriptor pool destroys");
  }

  auto success_pool_result = VulkanDescriptorPool::create(context, 1, 1);
  expect(success_pool_result.has_value(), "success descriptor pool creates");
  if (success_pool_result) {
    VulkanDescriptorPool pool(std::move(*success_pool_result));
    auto success = VulkanComputePipeline::create(
      context, pool, shader_words, "add", bindings, {});
    expect(success.has_value(),
           "pipeline succeeds after injected failures clear");
    if (success) {
      VulkanComputePipeline pipeline(std::move(*success));
      expect(pipeline.get_pipeline() != VK_NULL_HANDLE &&
               pipeline.get_layout() != VK_NULL_HANDLE &&
               pipeline.get_descriptor_sets().size() == 1,
             "pipeline exposes explicit pipeline/layout/descriptor handles");
      expect(pipeline.destroy().has_value(), "pipeline destroy succeeds");
    }
    expect(fake_vulkan::live_count("descriptor_set") == 1 &&
             fake_vulkan::live_count("descriptor_set_layout") == 1,
           "sets and layouts remain owned by the shared pool after pipeline "
           "destroy");
    expect(pool.destroy().has_value(),
           "shared descriptor pool destroys explicitly");
  }
  expect(buffer.destroy().has_value(), "pipeline test buffer destroys");
  expect(context.destroy().has_value(), "pipeline test context destroys");
  expect_no_live_fake_objects(
    "pipeline failures and cleanup release all objects");
}

void test_command_failures_and_device_lost() {
  fake_vulkan::reset();
  auto context_result = create_fake_context();
  expect(context_result.has_value(), "command test context initializes");
  if (!context_result) {
    return;
  }
  VulkanDeviceContext& context = *context_result;

  for (const auto* function :
       {"vkCreateCommandPool", "vkAllocateCommandBuffers", "vkCreateFence"}) {
    fake_vulkan::fail_next(function, VK_ERROR_OUT_OF_HOST_MEMORY);
    auto commands = VulkanCommandResources::create(context);
    expect(!commands, "command-resource construction returns Vulkan failure");
    expect(fake_vulkan::live_count("command_pool") == 0 &&
             fake_vulkan::live_count("fence") == 0,
           "partial command-resource creation is cleaned");
  }

  auto command_result = VulkanCommandResources::create(context);
  expect(command_result.has_value(), "command resources create after failures");
  if (command_result) {
    VulkanCommandResources commands(std::move(*command_result));
    fake_vulkan::fail_next("vkResetFences", VK_ERROR_DEVICE_LOST);
    auto reset_fence_failure = commands.reset();
    expect(!reset_fence_failure &&
             reset_fence_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "fence reset failure is reported");
    fake_vulkan::fail_next("vkResetCommandPool", VK_ERROR_DEVICE_LOST);
    auto reset_pool_failure = commands.reset();
    expect(!reset_pool_failure &&
             reset_pool_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "command-pool reset failure is reported");
    expect(
      commands.reset().has_value(),
      "command resources reset the initially signaled fence before submission");
    fake_vulkan::fail_next("vkBeginCommandBuffer", VK_ERROR_DEVICE_LOST);
    auto begin_failure = commands.begin();
    expect(!begin_failure &&
             begin_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "begin failure retains device-lost status");
    expect(commands.reset().has_value(), "command reset succeeds");
    expect(commands.begin().has_value(), "command begin succeeds");
    fake_vulkan::fail_next("vkEndCommandBuffer", VK_ERROR_DEVICE_LOST);
    auto end_failure = commands.end();
    expect(
      !end_failure && end_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
      "end failure retains device-lost status");

    fake_vulkan::fail_next("vkQueueSubmit", VK_ERROR_DEVICE_LOST);
    auto submit_failure = commands.submit_and_wait();
    expect(!submit_failure &&
             submit_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "submit failure retains device-lost status");
    expect(commands.reset().has_value(), "commands reset before wait fault");
    expect(commands.begin().has_value(), "commands begin before wait fault");
    expect(commands.end().has_value(), "commands end before wait fault");
    fake_vulkan::fail_next("vkWaitForFences", VK_ERROR_DEVICE_LOST);
    auto wait_failure = commands.submit_and_wait();
    expect(!wait_failure &&
             wait_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
           "wait failure retains device-lost status");
    expect(commands.wait().has_value(),
           "pending submission can be explicitly waited after a transient wait "
           "failure");
    expect(commands.destroy().has_value(),
           "command destroy succeeds after pending wait recovery");
    expect(commands.destroy().has_value(), "command destroy is idempotent");
  }

  fake_vulkan::fail_next("vkDeviceWaitIdle", VK_ERROR_DEVICE_LOST);
  auto destroy_failure = context.destroy();
  expect(!destroy_failure &&
           destroy_failure.error().status == NCNN_VK_STATUS_DEVICE_LOST,
         "device-lost idle failure is reported during explicit destroy");
  const auto device_destroy_calls = fake_vulkan::call_count("vkDestroyDevice");
  expect(!context.destroy(), "repeat destroy retains the prior cleanup error");
  expect(fake_vulkan::call_count("vkDestroyDevice") == device_destroy_calls,
         "failed destroy does not destroy the device twice");
  expect_no_live_fake_objects(
    "command and device-lost cleanup releases all objects");
}

}  // namespace

int main() {
  fake_vulkan::reset();
  test_limits();
  test_allocation_budget();
  test_context_hooks_move_and_memory_types();
  test_buffer_failures_and_noncoherent_sync();
  test_pipeline_failures();
  test_command_failures_and_device_lost();
  if (failures != 0) {
    std::cerr << failures << " native helper unit assertion(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "Vulkan device helper native/fake Vulkan tests passed\n";
  return EXIT_SUCCESS;
}
