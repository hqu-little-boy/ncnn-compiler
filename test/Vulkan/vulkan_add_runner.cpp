#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

namespace {

constexpr uint32_t kElementCount = 257;
constexpr uint32_t kLocalSize = 64;
constexpr float kAbsTolerance = 1.0e-6F;
constexpr float kRelTolerance = 1.0e-6F;
std::atomic<uint32_t> g_validation_errors = 0;

void check(VkResult result, std::string_view operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                             std::to_string(result));
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
validation_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                    VkDebugUtilsMessageTypeFlagsEXT,
                    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                    void*) {
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    ++g_validation_errors;
    std::cerr << "Vulkan validation error: "
              << (callback_data->pMessage ? callback_data->pMessage : "")
              << '\n';
  }
  return VK_FALSE;
}

struct Options {
  std::string shader_path;
  std::string cpu_library;
  std::string cpu_symbol = "v0_add";
  std::string device_match;
  std::string evidence_path;
  VkDeviceSize probe_buffer_bytes = 0;
  bool validation = false;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto value = [&](std::string_view name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(name));
      }
      return argv[++i];
    };
    if (arg == "--shader") {
      options.shader_path = value(arg);
    } else if (arg == "--cpu-library") {
      options.cpu_library = value(arg);
    } else if (arg == "--cpu-symbol") {
      options.cpu_symbol = value(arg);
    } else if (arg == "--device-match") {
      options.device_match = value(arg);
    } else if (arg == "--evidence") {
      options.evidence_path = value(arg);
    } else if (arg == "--probe-buffer-bytes") {
      options.probe_buffer_bytes = std::stoull(value(arg));
    } else if (arg == "--validation") {
      options.validation = true;
    } else {
      throw std::runtime_error("unknown argument: " + std::string(arg));
    }
  }
  if (options.shader_path.empty() || options.cpu_library.empty() ||
      options.device_match.empty() || options.evidence_path.empty()) {
    throw std::runtime_error(
      "usage: vulkan_add_runner --shader FILE --cpu-library FILE "
      "--device-match NAME --evidence FILE [--cpu-symbol NAME] [--validation]");
  }
  return options;
}

std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open SPIR-V file: " + path);
  }
  const std::streamsize byte_count = input.tellg();
  if (byte_count <= 0 || byte_count % sizeof(uint32_t) != 0) {
    throw std::runtime_error("SPIR-V file has an invalid byte length");
  }
  input.seekg(0);
  std::vector<uint32_t> words(static_cast<size_t>(byte_count) /
                              sizeof(uint32_t));
  if (!input.read(reinterpret_cast<char*>(words.data()), byte_count)) {
    throw std::runtime_error("cannot read SPIR-V file: " + path);
  }
  if (words.front() != 0x07230203U) {
    throw std::runtime_error("SPIR-V file has an invalid magic number");
  }
  return words;
}

class VulkanAddRunner {
 public:
  explicit VulkanAddRunner(const Options& options)
    : options_(options), shader_(read_spirv(options.shader_path)) {}

  ~VulkanAddRunner() { destroy(); }

  VulkanAddRunner(const VulkanAddRunner&) = delete;
  VulkanAddRunner& operator=(const VulkanAddRunner&) = delete;

  std::string run(const std::array<std::vector<float>, 2>& inputs,
                  std::vector<float>& output) {
    create_instance();
    select_device();
    create_buffers();
    write_buffer(0, inputs[0].data());
    write_buffer(1, inputs[1].data());
    create_descriptors();
    create_pipeline();
    dispatch();
    read_buffer(output.data());
    return device_name_;
  }

  const VkPhysicalDeviceProperties& properties() const { return properties_; }

 private:
  struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    bool coherent = false;
  };

  void create_instance() {
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    if (options_.validation) {
      uint32_t layer_count = 0;
      check(vkEnumerateInstanceLayerProperties(&layer_count, nullptr),
            "vkEnumerateInstanceLayerProperties");
      std::vector<VkLayerProperties> available(layer_count);
      check(vkEnumerateInstanceLayerProperties(&layer_count, available.data()),
            "vkEnumerateInstanceLayerProperties");
      const bool found = std::ranges::any_of(available, [](const auto& layer) {
        return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0;
      });
      if (!found) {
        throw std::runtime_error("VK_LAYER_KHRONOS_validation is unavailable");
      }
      layers.push_back("VK_LAYER_KHRONOS_validation");

      uint32_t extension_count = 0;
      check(vkEnumerateInstanceExtensionProperties(
              "VK_LAYER_KHRONOS_validation", &extension_count, nullptr),
            "vkEnumerateInstanceExtensionProperties");
      std::vector<VkExtensionProperties> available_extensions(extension_count);
      check(
        vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation",
                                               &extension_count,
                                               available_extensions.data()),
        "vkEnumerateInstanceExtensionProperties");
      const bool debug_utils =
        std::ranges::any_of(available_extensions, [](const auto& extension) {
          return std::strcmp(extension.extensionName,
                             VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        });
      if (!debug_utils) {
        throw std::runtime_error(
          "VK_EXT_debug_utils is unavailable from validation layer");
      }
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "ncnn-mlir-vulkan-v0";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName = "ncnn-mlir-vulkan-v0";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkDebugUtilsMessengerCreateInfoEXT debug_info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_info.pfnUserCallback = validation_callback;

    VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();
    create_info.enabledExtensionCount =
      static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.pNext = options_.validation ? &debug_info : nullptr;
    check(vkCreateInstance(&create_info, nullptr, &instance_),
          "vkCreateInstance");

    if (options_.validation) {
      const auto create_messenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
      if (!create_messenger) {
        throw std::runtime_error(
          "vkCreateDebugUtilsMessengerEXT is unavailable");
      }
      check(create_messenger(instance_, &debug_info, nullptr, &messenger_),
            "vkCreateDebugUtilsMessengerEXT");
    }
  }

  void select_device() {
    uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
          "vkEnumeratePhysicalDevices");
    if (device_count == 0) {
      throw std::runtime_error("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices");
    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(candidate, &properties);
      if (std::string_view(properties.deviceName).find(options_.device_match) ==
          std::string_view::npos) {
        continue;
      }
      vkGetPhysicalDeviceMemoryProperties(candidate, &memory_properties_);
      vkGetPhysicalDeviceProperties(candidate, &properties_);
      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(
        candidate, &queue_count, nullptr);
      std::vector<VkQueueFamilyProperties> queues(queue_count);
      vkGetPhysicalDeviceQueueFamilyProperties(
        candidate, &queue_count, queues.data());
      for (uint32_t index = 0; index < queue_count; ++index) {
        if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
          physical_device_ = candidate;
          queue_family_ = index;
          device_name_ = properties.deviceName;
          break;
        }
      }
      if (physical_device_ != VK_NULL_HANDLE) {
        break;
      }
    }
    if (physical_device_ == VK_NULL_HANDLE) {
      throw std::runtime_error("no compute device matched '" +
                               options_.device_match + "'");
    }
    if (properties_.limits.maxStorageBufferRange <
        kElementCount * sizeof(float)) {
      throw std::runtime_error("device maxStorageBufferRange is too small");
    }
    const VkDeviceSize requested_bytes = options_.probe_buffer_bytes == 0
                                           ? kElementCount * sizeof(float)
                                           : options_.probe_buffer_bytes;
    if (requested_bytes < kElementCount * sizeof(float)) {
      throw std::runtime_error(
        "requested storage buffer is smaller than the fixture");
    }
    if (requested_bytes > properties_.limits.maxStorageBufferRange) {
      throw std::runtime_error(
        "requested storage buffer exceeds maxStorageBufferRange");
    }
    if (properties_.limits.maxComputeWorkGroupInvocations < kLocalSize ||
        properties_.limits.maxComputeWorkGroupSize[0] < kLocalSize) {
      throw std::runtime_error("device workgroup limits are too small");
    }

    constexpr float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
          "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
  }

  uint32_t find_memory_type(uint32_t allowed_bits, bool& coherent) const {
    std::optional<uint32_t> host_visible;
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
      if ((allowed_bits & (1U << i)) == 0) {
        continue;
      }
      const VkMemoryPropertyFlags flags =
        memory_properties_.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        continue;
      }
      if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        coherent = true;
        return i;
      }
      host_visible = i;
    }
    if (host_visible) {
      coherent = false;
      return *host_visible;
    }
    throw std::runtime_error("no host-visible memory type for storage buffer");
  }

  void create_buffers() {
    for (Buffer& buffer : buffers_) {
      VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      buffer_info.size = kElementCount * sizeof(float);
      buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      check(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer.handle),
            "vkCreateBuffer");

      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
      bool coherent = false;
      const uint32_t memory_type =
        find_memory_type(requirements.memoryTypeBits, coherent);
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = memory_type;
      check(vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory),
            "vkAllocateMemory");
      buffer.coherent = coherent;
      check(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0),
            "vkBindBufferMemory");
      check(vkMapMemory(
              device_, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped),
            "vkMapMemory");
    }
  }

  void write_buffer(size_t index, const float* values) {
    Buffer& buffer = buffers_[index];
    std::memcpy(buffer.mapped, values, kElementCount * sizeof(float));
    if (!buffer.coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      check(vkFlushMappedMemoryRanges(device_, 1, &range),
            "vkFlushMappedMemoryRanges");
    }
  }

  void create_descriptors() {
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(
            device_, &layout_info, nullptr, &descriptor_layout_),
          "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pool_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(
      vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
      "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo set_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool_;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &descriptor_layout_;
    check(vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_),
          "vkAllocateDescriptorSets");

    std::array<VkDescriptorBufferInfo, 3> buffer_infos{};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < buffer_infos.size(); ++i) {
      buffer_infos[i].buffer = buffers_[i].handle;
      buffer_infos[i].offset = 0;
      buffer_infos[i].range = kElementCount * sizeof(float);
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = descriptor_set_;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(
      device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void create_pipeline() {
    VkShaderModuleCreateInfo shader_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = shader_.size() * sizeof(uint32_t);
    shader_info.pCode = shader_.data();
    check(vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_),
          "vkCreateShaderModule");

    VkPipelineLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_layout_;
    check(
      vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_),
      "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module_;
    stage_info.pName = "add";
    VkComputePipelineCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout_;
    check(vkCreateComputePipelines(
            device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_),
          "vkCreateComputePipelines");
  }

  void dispatch() {
    VkCommandPoolCreateInfo pool_info{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = queue_family_;
    check(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
          "vkCreateCommandPool");
    VkCommandBufferAllocateInfo allocation{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = command_pool_;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(device_, &allocation, &command_buffer),
          "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_buffer, &begin_info),
          "vkBeginCommandBuffer");
    std::array<VkBufferMemoryBarrier, 3> before{};
    for (size_t i = 0; i < before.size(); ++i) {
      before[i] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      before[i].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      before[i].dstAccessMask =
        i < 2 ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT;
      before[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      before[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      before[i].buffer = buffers_[i].handle;
      before[i].offset = 0;
      before[i].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<uint32_t>(before.size()),
                         before.data(),
                         0,
                         nullptr);
    vkCmdBindPipeline(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_,
                            0,
                            1,
                            &descriptor_set_,
                            0,
                            nullptr);
    vkCmdDispatch(
      command_buffer, (kElementCount + kLocalSize - 1) / kLocalSize, 1, 1);
    VkBufferMemoryBarrier after{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.buffer = buffers_[2].handle;
    after.offset = 0;
    after.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &after,
                         0,
                         nullptr);
    check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(device_, &fence_info, nullptr, &fence),
          "vkCreateFence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    check(vkQueueSubmit(queue_, 1, &submit, fence), "vkQueueSubmit");
    check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    vkDestroyFence(device_, fence, nullptr);
  }

  void read_buffer(float* output) {
    Buffer& buffer = buffers_[2];
    if (!buffer.coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      check(vkInvalidateMappedMemoryRanges(device_, 1, &range),
            "vkInvalidateMappedMemoryRanges");
    }
    std::memcpy(output, buffer.mapped, kElementCount * sizeof(float));
  }

  void destroy() noexcept {
    if (device_ != VK_NULL_HANDLE) {
      if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
      }
      if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
      }
      if (shader_module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, shader_module_, nullptr);
      }
      if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
      }
      if (descriptor_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
      }
      if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
      }
      for (Buffer& buffer : buffers_) {
        if (buffer.mapped != nullptr) {
          vkUnmapMemory(device_, buffer.memory);
        }
        if (buffer.handle != VK_NULL_HANDLE) {
          vkDestroyBuffer(device_, buffer.handle, nullptr);
        }
        if (buffer.memory != VK_NULL_HANDLE) {
          vkFreeMemory(device_, buffer.memory, nullptr);
        }
      }
      vkDestroyDevice(device_, nullptr);
    }
    if (messenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
      const auto destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
      if (destroy_messenger) {
        destroy_messenger(instance_, messenger_, nullptr);
      }
    }
    if (instance_ != VK_NULL_HANDLE) {
      vkDestroyInstance(instance_, nullptr);
    }
  }

  const Options& options_;
  std::vector<uint32_t> shader_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties_{};
  VkPhysicalDeviceMemoryProperties memory_properties_{};
  uint32_t queue_family_ = 0;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  std::string device_name_;
  std::array<Buffer, 3> buffers_{};
  VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkShaderModule shader_module_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
};

void write_inputs(const std::string& path,
                  const std::array<std::vector<float>, 2>& inputs) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("cannot create input evidence file: " + path);
  }
  for (const auto& input : inputs) {
    output.write(reinterpret_cast<const char*>(input.data()),
                 static_cast<std::streamsize>(input.size() * sizeof(float)));
  }
  if (!output) {
    throw std::runtime_error("cannot write input evidence file: " + path);
  }
}

void write_outputs(const std::string& path,
                   const std::vector<float>& cpu,
                   const std::vector<float>& gpu,
                   const std::vector<float>& golden) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("cannot create output evidence file: " + path);
  }
  for (const std::vector<float>* values : {&cpu, &gpu, &golden}) {
    output.write(reinterpret_cast<const char*>(values->data()),
                 static_cast<std::streamsize>(values->size() * sizeof(float)));
  }
  if (!output) {
    throw std::runtime_error("cannot write output evidence file: " + path);
  }
}

struct Difference {
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  bool passed = true;
};

Difference compare(const std::vector<float>& actual,
                   const std::vector<float>& expected) {
  Difference difference;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float absolute = std::abs(actual[i] - expected[i]);
    const float relative = absolute / std::max(std::abs(expected[i]), 1.0e-30F);
    difference.max_abs = std::max(difference.max_abs, absolute);
    difference.max_rel = std::max(difference.max_rel, relative);
    difference.passed &=
      absolute <= kAbsTolerance + kRelTolerance * std::abs(expected[i]);
  }
  return difference;
}

void write_evidence(const std::string& path,
                    const Options& options,
                    const std::string& device_name,
                    const VkPhysicalDeviceProperties& properties,
                    const Difference& cpu_diff,
                    const Difference& gpu_diff,
                    const Difference& cross_diff) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot write evidence JSON: " + path);
  }
  output << "{\n"
         << "  \"device\": \"" << device_name << "\",\n"
         << "  \"device_match\": \"" << options.device_match << "\",\n"
         << "  \"api_version\": \"" << VK_VERSION_MAJOR(properties.apiVersion)
         << '.' << VK_VERSION_MINOR(properties.apiVersion) << '.'
         << VK_VERSION_PATCH(properties.apiVersion) << "\",\n"
         << "  \"driver_version\": " << properties.driverVersion << ",\n"
         << "  \"vendor_id\": " << properties.vendorID << ",\n"
         << "  \"device_id\": " << properties.deviceID << ",\n"
         << "  \"max_storage_buffer_range\": "
         << properties.limits.maxStorageBufferRange << ",\n"
         << "  \"max_compute_workgroup_invocations\": "
         << properties.limits.maxComputeWorkGroupInvocations << ",\n"
         << "  \"max_compute_workgroup_size_x\": "
         << properties.limits.maxComputeWorkGroupSize[0] << ",\n"
         << "  \"elements\": " << kElementCount << ",\n"
         << "  \"local_size\": " << kLocalSize << ",\n"
         << "  \"dispatch_groups_x\": "
         << (kElementCount + kLocalSize - 1) / kLocalSize << ",\n"
         << "  \"validation_enabled\": "
         << (options.validation ? "true" : "false") << ",\n"
         << "  \"validation_errors\": " << g_validation_errors.load() << ",\n"
         << "  \"tolerance\": {\"atol\": " << kAbsTolerance
         << ", \"rtol\": " << kRelTolerance << "},\n"
         << "  \"cpu_vs_golden\": {\"passed\": "
         << (cpu_diff.passed ? "true" : "false")
         << ", \"max_abs\": " << cpu_diff.max_abs
         << ", \"max_rel\": " << cpu_diff.max_rel << "},\n"
         << "  \"gpu_vs_golden\": {\"passed\": "
         << (gpu_diff.passed ? "true" : "false")
         << ", \"max_abs\": " << gpu_diff.max_abs
         << ", \"max_rel\": " << gpu_diff.max_rel << "},\n"
         << "  \"gpu_vs_cpu\": {\"passed\": "
         << (cross_diff.passed ? "true" : "false")
         << ", \"max_abs\": " << cross_diff.max_abs
         << ", \"max_rel\": " << cross_diff.max_rel << "}\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::array<std::vector<float>, 2> inputs;
    std::vector<float> golden(kElementCount);
    std::vector<float> cpu_output(kElementCount,
                                  std::numeric_limits<float>::quiet_NaN());
    std::vector<float> gpu_output(kElementCount,
                                  std::numeric_limits<float>::quiet_NaN());
    for (uint32_t i = 0; i < kElementCount; ++i) {
      inputs[0].push_back((static_cast<int>(i % 17) - 8) * 0.25F);
      inputs[1].push_back((static_cast<int>(i % 13) - 6) * 0.125F);
      golden[i] = inputs[0][i] + inputs[1][i];
    }
    write_inputs(options.evidence_path + ".inputs.bin", inputs);

    void* cpu_library =
      dlopen(options.cpu_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!cpu_library) {
      throw std::runtime_error("dlopen CPU AOT library failed: " +
                               std::string(dlerror()));
    }
    using CpuModel = int (*)(const float*, const float*, float*);
    dlerror();
    auto cpu_model = reinterpret_cast<CpuModel>(
      dlsym(cpu_library, options.cpu_symbol.c_str()));
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || cpu_model == nullptr) {
      dlclose(cpu_library);
      throw std::runtime_error(
        "CPU AOT symbol lookup failed: " +
        std::string(symbol_error ? symbol_error : "null"));
    }
    const int cpu_status =
      cpu_model(inputs[0].data(), inputs[1].data(), cpu_output.data());
    dlclose(cpu_library);
    if (cpu_status != 0) {
      throw std::runtime_error("CPU AOT returned status " +
                               std::to_string(cpu_status));
    }

    std::string device_name;
    VkPhysicalDeviceProperties device_properties{};
    {
      VulkanAddRunner runner(options);
      device_name = runner.run(inputs, gpu_output);
      device_properties = runner.properties();
    }
    const Difference cpu_diff = compare(cpu_output, golden);
    const Difference gpu_diff = compare(gpu_output, golden);
    const Difference cross_diff = compare(gpu_output, cpu_output);
    write_evidence(options.evidence_path,
                   options,
                   device_name,
                   device_properties,
                   cpu_diff,
                   gpu_diff,
                   cross_diff);
    write_outputs(
      options.evidence_path + ".outputs.bin", cpu_output, gpu_output, golden);
    std::cout << "device=" << device_name << " elements=" << kElementCount
              << " groups=" << (kElementCount + kLocalSize - 1) / kLocalSize
              << " cpu_abs=" << cpu_diff.max_abs
              << " gpu_abs=" << gpu_diff.max_abs
              << " cpu_gpu_abs=" << cross_diff.max_abs
              << " validation_errors=" << g_validation_errors.load() << '\n';
    if (!cpu_diff.passed || !gpu_diff.passed || !cross_diff.passed ||
        (options.validation && g_validation_errors.load() != 0)) {
      throw std::runtime_error(
        "CPU/GPU/golden comparison or validation failed");
    }
  } catch (const std::exception& error) {
    std::cerr << "vulkan_add_runner: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
