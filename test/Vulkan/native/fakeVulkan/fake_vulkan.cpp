#include "fake_vulkan.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

template <typename Handle>
std::uintptr_t raw(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<std::uintptr_t>(handle);
  } else {
    return static_cast<std::uintptr_t>(handle);
  }
}

template <typename Handle>
Handle make_handle(std::uintptr_t value) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(value);
  } else {
    return static_cast<Handle>(value);
  }
}

struct BufferRecord final {
  VkDeviceSize size = 0;
  VkBufferUsageFlags usage = 0;
};

struct MemoryRecord final {
  VkDeviceSize size = 0;
  std::vector<std::byte> bytes;
};

struct State final {
  bool enabled = false;
  std::uintptr_t next_handle = 0x1000;
  std::string fail_function;
  VkResult fail_result = VK_SUCCESS;
  std::uint32_t max_memory_allocation_count = 4096;
  std::unordered_map<std::string, std::uint64_t> calls;
  std::unordered_map<std::string, std::int64_t> created;
  std::unordered_map<std::string, std::int64_t> destroyed;
  std::unordered_map<std::uintptr_t, BufferRecord> buffers;
  std::unordered_map<std::uintptr_t, MemoryRecord> memories;
  std::unordered_map<std::uintptr_t, std::uint64_t> command_pool_buffers;
  std::unordered_map<std::uintptr_t, std::uint64_t> descriptor_pool_sets;
  bool whole_flush = false;
  bool whole_invalidate = false;
  bool whole_map = false;
  VkDeviceSize last_map_offset = 0;
  VkDeviceSize last_map_size = 0;
};

State state;

std::uintptr_t next_handle() {
  return state.next_handle++;
}

VkResult called(std::string_view function) {
  ++state.calls[std::string(function)];
  if (state.fail_function == function) {
    state.fail_function.clear();
    return std::exchange(state.fail_result, VK_SUCCESS);
  }
  return VK_SUCCESS;
}

void created(std::string_view object_type) {
  ++state.created[std::string(object_type)];
}

void destroyed(std::string_view object_type) {
  ++state.destroyed[std::string(object_type)];
}

std::uint64_t count(const auto& values, std::string_view key) {
  auto found = values.find(std::string(key));
  return found == values.end() ? 0 : static_cast<std::uint64_t>(found->second);
}

}  // namespace

namespace fake_vulkan {

void reset() {
  state = State{};
  state.enabled = true;
}

void enable() {
  state.enabled = true;
}

void fail_next(std::string_view function, VkResult result) {
  state.fail_function = function;
  state.fail_result = result;
}

void set_max_memory_allocation_count(std::uint32_t count) {
  state.max_memory_allocation_count = count;
}

std::uint64_t call_count(std::string_view function) {
  return count(state.calls, function);
}

std::uint64_t live_count(std::string_view object_type) {
  return count(state.created, object_type) -
         count(state.destroyed, object_type);
}

bool saw_whole_allocation_range(std::string_view function) {
  if (function == "vkFlushMappedMemoryRanges") {
    return state.whole_flush;
  }
  if (function == "vkInvalidateMappedMemoryRanges") {
    return state.whole_invalidate;
  }
  if (function == "vkMapMemory") {
    return state.whole_map && state.last_map_offset == 0;
  }
  return false;
}

bool saw_whole_allocation_map() {
  return state.whole_map && state.last_map_offset == 0;
}

std::uint64_t last_mapped_allocation_size() {
  return state.last_map_size;
}

}  // namespace fake_vulkan

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateInstance(const VkInstanceCreateInfo*,
                        const VkAllocationCallbacks*,
                        VkInstance* instance) {
  if (const VkResult result = called("vkCreateInstance");
      result != VK_SUCCESS) {
    *instance = VK_NULL_HANDLE;
    return result;
  }
  *instance = make_handle<VkInstance>(next_handle());
  created("instance");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkEnumerateInstanceLayerProperties(
  std::uint32_t* count_out, VkLayerProperties* properties) {
  called("vkEnumerateInstanceLayerProperties");
  *count_out = 0;
  (void)properties;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkEnumerateInstanceExtensionProperties(
  const char*, std::uint32_t* count_out, VkExtensionProperties* properties) {
  called("vkEnumerateInstanceExtensionProperties");
  *count_out = 0;
  (void)properties;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkEnumeratePhysicalDevices(
  VkInstance, std::uint32_t* count_out, VkPhysicalDevice* devices) {
  called("vkEnumeratePhysicalDevices");
  if (devices == nullptr) {
    *count_out = 1;
  } else if (*count_out == 0) {
    return VK_INCOMPLETE;
  } else {
    devices[0] = make_handle<VkPhysicalDevice>(0x55);
    *count_out = 1;
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkGetPhysicalDeviceProperties(
  VkPhysicalDevice, VkPhysicalDeviceProperties* properties) {
  called("vkGetPhysicalDeviceProperties");
  *properties = VkPhysicalDeviceProperties{};
  properties->apiVersion = VK_API_VERSION_1_1;
  properties->driverVersion = 1;
  properties->vendorID = 0xCAFE;
  properties->deviceID = 0x1234;
  std::strncpy(properties->deviceName,
               "fakeVulkan test device",
               VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
  properties->limits.maxStorageBufferRange = 128U * 1024U * 1024U;
  properties->limits.maxComputeWorkGroupInvocations = 256;
  properties->limits.maxComputeWorkGroupSize[0] = 256;
  properties->limits.maxComputeWorkGroupCount[0] = 65535;
  properties->limits.maxPushConstantsSize = 128;
  properties->limits.maxBoundDescriptorSets = 4;
  properties->limits.maxPerStageDescriptorStorageBuffers = 16;
  properties->limits.maxDescriptorSetStorageBuffers = 16;
  properties->limits.maxPerStageResources = 32;
  properties->limits.maxMemoryAllocationCount =
    state.max_memory_allocation_count;
  properties->limits.minStorageBufferOffsetAlignment = 64;
  properties->limits.nonCoherentAtomSize = 64;
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkGetPhysicalDeviceMemoryProperties(
  VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* properties) {
  called("vkGetPhysicalDeviceMemoryProperties");
  *properties = VkPhysicalDeviceMemoryProperties{};
  properties->memoryTypeCount = 3;
  properties->memoryTypes[0].propertyFlags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  properties->memoryTypes[0].heapIndex = 0;
  properties->memoryTypes[1].propertyFlags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  properties->memoryTypes[1].heapIndex = 0;
  properties->memoryTypes[2].propertyFlags =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  properties->memoryTypes[2].heapIndex = 1;
  properties->memoryHeapCount = 2;
  properties->memoryHeaps[0].size = 1ULL << 30U;
  properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT;
  properties->memoryHeaps[1].size = 1ULL << 30U;
  properties->memoryHeaps[1].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkGetPhysicalDeviceQueueFamilyProperties(
  VkPhysicalDevice,
  std::uint32_t* count_out,
  VkQueueFamilyProperties* properties) {
  called("vkGetPhysicalDeviceQueueFamilyProperties");
  if (properties == nullptr) {
    *count_out = 1;
  } else if (*count_out > 0) {
    properties[0] = VkQueueFamilyProperties{};
    properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    properties[0].queueCount = 1;
    properties[0].timestampValidBits = 64;
    *count_out = 1;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateDevice(VkPhysicalDevice,
                      const VkDeviceCreateInfo*,
                      const VkAllocationCallbacks*,
                      VkDevice* device) {
  if (const VkResult result = called("vkCreateDevice"); result != VK_SUCCESS) {
    *device = VK_NULL_HANDLE;
    return result;
  }
  *device = make_handle<VkDevice>(next_handle());
  created("device");
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
__wrap_vkDestroyDevice(VkDevice, const VkAllocationCallbacks*) {
  called("vkDestroyDevice");
  destroyed("device");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkGetDeviceQueue(VkDevice,
                                                   std::uint32_t,
                                                   std::uint32_t,
                                                   VkQueue* queue) {
  called("vkGetDeviceQueue");
  *queue = make_handle<VkQueue>(next_handle());
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkDeviceWaitIdle(VkDevice) {
  return called("vkDeviceWaitIdle");
}

VKAPI_ATTR void VKAPI_CALL
__wrap_vkDestroyInstance(VkInstance, const VkAllocationCallbacks*) {
  called("vkDestroyInstance");
  destroyed("instance");
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateBuffer(VkDevice,
                      const VkBufferCreateInfo* info,
                      const VkAllocationCallbacks*,
                      VkBuffer* buffer) {
  if (const VkResult result = called("vkCreateBuffer"); result != VK_SUCCESS) {
    *buffer = VK_NULL_HANDLE;
    return result;
  }
  *buffer = make_handle<VkBuffer>(next_handle());
  state.buffers.emplace(raw(*buffer),
                        BufferRecord{.size = info->size, .usage = info->usage});
  created("buffer");
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkGetBufferMemoryRequirements(
  VkDevice, VkBuffer buffer, VkMemoryRequirements* requirements) {
  called("vkGetBufferMemoryRequirements");
  const auto found = state.buffers.find(raw(buffer));
  const VkDeviceSize logical_size =
    found == state.buffers.end() ? 16 : found->second.size;
  const VkDeviceSize aligned = (logical_size + 63U) & ~VkDeviceSize{63U};
  requirements->size = aligned + 64U;
  requirements->alignment = 64;
  const bool storage =
    found != state.buffers.end() &&
    (found->second.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0;
  requirements->memoryTypeBits = storage ? (1U << 2U) : 0x3U;
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkAllocateMemory(VkDevice,
                        const VkMemoryAllocateInfo* info,
                        const VkAllocationCallbacks*,
                        VkDeviceMemory* memory) {
  if (const VkResult result = called("vkAllocateMemory");
      result != VK_SUCCESS) {
    *memory = VK_NULL_HANDLE;
    return result;
  }
  *memory = make_handle<VkDeviceMemory>(next_handle());
  MemoryRecord record;
  record.size = info->allocationSize;
  record.bytes.resize(static_cast<std::size_t>(info->allocationSize));
  state.memories.emplace(raw(*memory), std::move(record));
  created("memory");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkBindBufferMemory(VkDevice,
                                                         VkBuffer,
                                                         VkDeviceMemory,
                                                         VkDeviceSize) {
  return called("vkBindBufferMemory");
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkMapMemory(VkDevice,
                                                  VkDeviceMemory memory,
                                                  VkDeviceSize offset,
                                                  VkDeviceSize size,
                                                  VkMemoryMapFlags,
                                                  void** data) {
  if (const VkResult result = called("vkMapMemory"); result != VK_SUCCESS) {
    *data = nullptr;
    return result;
  }
  auto found = state.memories.find(raw(memory));
  if (found == state.memories.end() || offset != 0 ||
      (size != VK_WHOLE_SIZE && size > found->second.size)) {
    *data = nullptr;
    return VK_ERROR_MEMORY_MAP_FAILED;
  }
  *data = found->second.bytes.data();
  state.whole_map = size == VK_WHOLE_SIZE;
  state.last_map_offset = offset;
  state.last_map_size = found->second.size;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkUnmapMemory(VkDevice, VkDeviceMemory) {
  called("vkUnmapMemory");
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkFlushMappedMemoryRanges(
  VkDevice, std::uint32_t count_in, const VkMappedMemoryRange* ranges) {
  const VkResult result = called("vkFlushMappedMemoryRanges");
  if (result != VK_SUCCESS) {
    return result;
  }
  state.whole_flush = count_in == 1 && ranges != nullptr &&
                      ranges[0].offset == 0 && ranges[0].size == VK_WHOLE_SIZE;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkInvalidateMappedMemoryRanges(
  VkDevice, std::uint32_t count_in, const VkMappedMemoryRange* ranges) {
  const VkResult result = called("vkInvalidateMappedMemoryRanges");
  if (result != VK_SUCCESS) {
    return result;
  }
  state.whole_invalidate = count_in == 1 && ranges != nullptr &&
                           ranges[0].offset == 0 &&
                           ranges[0].size == VK_WHOLE_SIZE;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyBuffer(
  VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
  called("vkDestroyBuffer");
  state.buffers.erase(raw(buffer));
  destroyed("buffer");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkFreeMemory(VkDevice,
                                               VkDeviceMemory memory,
                                               const VkAllocationCallbacks*) {
  called("vkFreeMemory");
  state.memories.erase(raw(memory));
  destroyed("memory");
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateCommandPool(VkDevice,
                           const VkCommandPoolCreateInfo*,
                           const VkAllocationCallbacks*,
                           VkCommandPool* pool) {
  if (const VkResult result = called("vkCreateCommandPool");
      result != VK_SUCCESS) {
    *pool = VK_NULL_HANDLE;
    return result;
  }
  *pool = make_handle<VkCommandPool>(next_handle());
  created("command_pool");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkAllocateCommandBuffers(
  VkDevice, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers) {
  if (const VkResult result = called("vkAllocateCommandBuffers");
      result != VK_SUCCESS) {
    return result;
  }
  for (std::uint32_t index = 0; index < info->commandBufferCount; ++index) {
    buffers[index] = make_handle<VkCommandBuffer>(next_handle());
    ++state.command_pool_buffers[raw(info->commandPool)];
    created("command_buffer");
  }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateFence(VkDevice,
                     const VkFenceCreateInfo*,
                     const VkAllocationCallbacks*,
                     VkFence* fence) {
  if (const VkResult result = called("vkCreateFence"); result != VK_SUCCESS) {
    *fence = VK_NULL_HANDLE;
    return result;
  }
  *fence = make_handle<VkFence>(next_handle());
  created("fence");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkResetFences(VkDevice,
                                                    std::uint32_t,
                                                    const VkFence*) {
  return called("vkResetFences");
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkResetCommandPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags) {
  return called("vkResetCommandPool");
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
  return called("vkBeginCommandBuffer");
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkEndCommandBuffer(VkCommandBuffer) {
  return called("vkEndCommandBuffer");
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkQueueSubmit(VkQueue,
                                                    std::uint32_t,
                                                    const VkSubmitInfo*,
                                                    VkFence) {
  return called("vkQueueSubmit");
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkWaitForFences(
  VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
  return called("vkWaitForFences");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyFence(VkDevice,
                                                 VkFence,
                                                 const VkAllocationCallbacks*) {
  called("vkDestroyFence");
  destroyed("fence");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyCommandPool(
  VkDevice, VkCommandPool pool, const VkAllocationCallbacks*) {
  called("vkDestroyCommandPool");
  const auto key = raw(pool);
  const auto found = state.command_pool_buffers.find(key);
  if (found != state.command_pool_buffers.end()) {
    for (std::uint64_t index = 0; index < found->second; ++index) {
      destroyed("command_buffer");
    }
    state.command_pool_buffers.erase(found);
  }
  destroyed("command_pool");
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateDescriptorSetLayout(VkDevice,
                                   const VkDescriptorSetLayoutCreateInfo*,
                                   const VkAllocationCallbacks*,
                                   VkDescriptorSetLayout* layout) {
  if (const VkResult result = called("vkCreateDescriptorSetLayout");
      result != VK_SUCCESS) {
    *layout = VK_NULL_HANDLE;
    return result;
  }
  *layout = make_handle<VkDescriptorSetLayout>(next_handle());
  created("descriptor_set_layout");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateDescriptorPool(VkDevice,
                              const VkDescriptorPoolCreateInfo*,
                              const VkAllocationCallbacks*,
                              VkDescriptorPool* pool) {
  if (const VkResult result = called("vkCreateDescriptorPool");
      result != VK_SUCCESS) {
    *pool = VK_NULL_HANDLE;
    return result;
  }
  *pool = make_handle<VkDescriptorPool>(next_handle());
  created("descriptor_pool");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL __wrap_vkAllocateDescriptorSets(
  VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
  if (const VkResult result = called("vkAllocateDescriptorSets");
      result != VK_SUCCESS) {
    return result;
  }
  for (std::uint32_t index = 0; index < info->descriptorSetCount; ++index) {
    sets[index] = make_handle<VkDescriptorSet>(next_handle());
    ++state.descriptor_pool_sets[raw(info->descriptorPool)];
    created("descriptor_set");
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
__wrap_vkUpdateDescriptorSets(VkDevice,
                              std::uint32_t,
                              const VkWriteDescriptorSet*,
                              std::uint32_t,
                              const VkCopyDescriptorSet*) {
  called("vkUpdateDescriptorSets");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyDescriptorPool(
  VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
  called("vkDestroyDescriptorPool");
  const auto key = raw(pool);
  const auto found = state.descriptor_pool_sets.find(key);
  if (found != state.descriptor_pool_sets.end()) {
    for (std::uint64_t index = 0; index < found->second; ++index) {
      destroyed("descriptor_set");
    }
    state.descriptor_pool_sets.erase(found);
  }
  destroyed("descriptor_pool");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyDescriptorSetLayout(
  VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {
  called("vkDestroyDescriptorSetLayout");
  destroyed("descriptor_set_layout");
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateShaderModule(VkDevice,
                            const VkShaderModuleCreateInfo*,
                            const VkAllocationCallbacks*,
                            VkShaderModule* module) {
  if (const VkResult result = called("vkCreateShaderModule");
      result != VK_SUCCESS) {
    *module = VK_NULL_HANDLE;
    return result;
  }
  *module = make_handle<VkShaderModule>(next_handle());
  created("shader_module");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreatePipelineLayout(VkDevice,
                              const VkPipelineLayoutCreateInfo*,
                              const VkAllocationCallbacks*,
                              VkPipelineLayout* layout) {
  if (const VkResult result = called("vkCreatePipelineLayout");
      result != VK_SUCCESS) {
    *layout = VK_NULL_HANDLE;
    return result;
  }
  *layout = make_handle<VkPipelineLayout>(next_handle());
  created("pipeline_layout");
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
__wrap_vkCreateComputePipelines(VkDevice,
                                VkPipelineCache,
                                std::uint32_t count_in,
                                const VkComputePipelineCreateInfo*,
                                const VkAllocationCallbacks*,
                                VkPipeline* pipelines) {
  if (const VkResult result = called("vkCreateComputePipelines");
      result != VK_SUCCESS) {
    return result;
  }
  for (std::uint32_t index = 0; index < count_in; ++index) {
    pipelines[index] = make_handle<VkPipeline>(next_handle());
    created("pipeline");
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
__wrap_vkDestroyPipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {
  called("vkDestroyPipeline");
  destroyed("pipeline");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyPipelineLayout(
  VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {
  called("vkDestroyPipelineLayout");
  destroyed("pipeline_layout");
}

VKAPI_ATTR void VKAPI_CALL __wrap_vkDestroyShaderModule(
  VkDevice, VkShaderModule, const VkAllocationCallbacks*) {
  called("vkDestroyShaderModule");
  destroyed("shader_module");
}

}  // extern "C"
