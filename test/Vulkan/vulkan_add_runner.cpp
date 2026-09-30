#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

namespace {

constexpr uint32_t kElementCount = 257;
constexpr uint32_t kLocalSize = 64;
constexpr uint64_t kMaximumStorageBufferBytes = 128ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMaximumStorageBufferElements =
  kMaximumStorageBufferBytes / sizeof(float);
constexpr float kAbsTolerance = 1.0e-6F;
constexpr float kRelTolerance = 1.0e-6F;
std::atomic<uint32_t> g_validation_errors = 0;

void check(VkResult result, std::string_view operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                             std::to_string(result));
  }
}

uint64_t checked_product(std::initializer_list<uint64_t> factors,
                         std::string_view description) {
  uint64_t product = 1;
  for (uint64_t factor : factors) {
    if (factor == 0 || product > UINT64_MAX / factor) {
      throw std::runtime_error(std::string(description) +
                               " is zero or overflows uint64");
    }
    product *= factor;
  }
  if (product > kMaximumStorageBufferElements) {
    throw std::runtime_error(
      std::string(description) +
      " exceeds the minimum Vulkan storage-buffer range");
  }
  return product;
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
  std::string operation = "add";
  std::string weight_file;
  std::string device_match;
  std::string evidence_path;
  uint32_t elements = kElementCount;
  uint32_t seed = 0;
  std::vector<uint32_t> input_shape;
  std::vector<uint32_t> permutation;
  std::vector<uint32_t> kernel_size;
  std::vector<uint32_t> stride;
  std::vector<uint32_t> padding;
  uint32_t output_channels = 0;
  VkDeviceSize probe_buffer_bytes = 0;
  bool convolution_has_bias = false;
  bool validation = false;
};

uint32_t parse_u32(std::string_view text, std::string_view field) {
  if (text.empty()) {
    throw std::runtime_error(std::string(field) + " must not be empty");
  }
  std::size_t parsed_chars = 0;
  const unsigned long long parsed =
    std::stoull(std::string(text), &parsed_chars, 10);
  if (parsed_chars != text.size() ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string(field) + " exceeds uint32 range");
  }
  return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> parse_u32_list(const std::string& text, char separator) {
  std::vector<uint32_t> values;
  std::stringstream stream(text);
  std::string value;
  while (std::getline(stream, value, separator)) {
    if (value.empty()) {
      throw std::runtime_error("list arguments must not contain empty values");
    }
    values.push_back(parse_u32(value, "list argument"));
  }
  if (values.empty()) {
    throw std::runtime_error("list argument must contain at least one value");
  }
  return values;
}

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
    } else if (arg == "--operation") {
      options.operation = value(arg);
    } else if (arg == "--weight-file") {
      options.weight_file = value(arg);
    } else if (arg == "--elements") {
      options.elements = parse_u32(value(arg), "elements");
    } else if (arg == "--seed") {
      options.seed = parse_u32(value(arg), "seed");
    } else if (arg == "--input-shape") {
      options.input_shape = parse_u32_list(value(arg), 'x');
    } else if (arg == "--permutation") {
      options.permutation = parse_u32_list(value(arg), ',');
    } else if (arg == "--kernel-size") {
      options.kernel_size = parse_u32_list(value(arg), 'x');
    } else if (arg == "--stride") {
      options.stride = parse_u32_list(value(arg), 'x');
    } else if (arg == "--padding") {
      options.padding = parse_u32_list(value(arg), ',');
    } else if (arg == "--output-channels") {
      options.output_channels = parse_u32(value(arg), "output-channels");
    } else if (arg == "--conv-bias") {
      options.convolution_has_bias = true;
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
      options.device_match.empty() || options.evidence_path.empty() ||
      options.elements == 0 ||
      options.elements > kMaximumStorageBufferElements ||
      (options.operation != "add" && options.operation != "relu" &&
       options.operation != "copy" && options.operation != "identity" &&
       options.operation != "reshape" && options.operation != "transpose" &&
       options.operation != "conv2d")) {
    throw std::runtime_error(
      "usage: vulkan_add_runner --shader FILE --cpu-library FILE "
      "--device-match NAME --evidence FILE [--cpu-symbol NAME] "
      "[--operation add|relu|copy|identity|reshape|transpose|conv2d] "
      "[--elements N] [--input-shape CxHxW --permutation AXES] "
      "[--kernel-size HxW --stride HxW --padding T,L,B,R "
      "--output-channels N --weight-file FILE] [--validation]");
  }
  if (options.operation == "transpose") {
    if (options.input_shape.empty() ||
        options.input_shape.size() != options.permutation.size()) {
      throw std::runtime_error(
        "transpose requires matching input-shape and permutation ranks");
    }
    uint64_t input_elements = 1;
    std::vector<bool> seen(options.permutation.size(), false);
    for (size_t axis = 0; axis < options.input_shape.size(); ++axis) {
      const uint32_t extent = options.input_shape[axis];
      const uint32_t source_axis = options.permutation[axis];
      if (extent == 0 || input_elements > UINT64_MAX / extent ||
          input_elements > kMaximumStorageBufferElements / extent ||
          source_axis >= seen.size() || seen[source_axis]) {
        throw std::runtime_error("invalid static transpose shape/permutation");
      }
      input_elements *= extent;
      seen[source_axis] = true;
    }
    if (input_elements != options.elements) {
      throw std::runtime_error("transpose shape does not match element count");
    }
  } else if (options.operation == "conv2d") {
    if (options.input_shape.size() != 3 || options.kernel_size.size() != 2 ||
        options.stride.size() != 2 || options.padding.size() != 4 ||
        options.output_channels == 0 || options.weight_file.empty()) {
      throw std::runtime_error(
        "Conv2D requires static shape, kernel, stride, padding, channels, and "
        "weights");
    }
    if (!((options.kernel_size[0] == 1 && options.kernel_size[1] == 1) ||
          (options.kernel_size[0] == 3 && options.kernel_size[1] == 3)) ||
        (options.stride[0] != 1 && options.stride[0] != 2) ||
        (options.stride[1] != 1 && options.stride[1] != 2) ||
        std::ranges::any_of(options.padding,
                            [](uint32_t pad) { return pad > 1; }) ||
        (options.kernel_size[0] == 1 &&
         std::ranges::any_of(options.padding,
                             [](uint32_t pad) { return pad != 0; }))) {
      throw std::runtime_error(
        "unsupported Conv2D kernel/stride/padding parameters");
    }
    const uint64_t input_channels = options.input_shape[0];
    const uint64_t height = options.input_shape[1];
    const uint64_t width = options.input_shape[2];
    const uint64_t input_elements =
      checked_product({input_channels, height, width}, "Conv2D input buffer");
    const uint64_t weight_elements = checked_product({options.output_channels,
                                                      input_channels,
                                                      options.kernel_size[0],
                                                      options.kernel_size[1]},
                                                     "Conv2D weight buffer");
    (void)input_elements;
    (void)weight_elements;
    const uint64_t padded_height =
      height + options.padding[0] + options.padding[2];
    const uint64_t padded_width =
      width + options.padding[1] + options.padding[3];
    if (padded_height < options.kernel_size[0] ||
        padded_width < options.kernel_size[1]) {
      throw std::runtime_error("Conv2D kernel exceeds padded input dimensions");
    }
    const uint64_t output_height =
      ((padded_height - options.kernel_size[0]) / options.stride[0]) + 1;
    const uint64_t output_width =
      ((padded_width - options.kernel_size[1]) / options.stride[1]) + 1;
    const uint64_t output_elements =
      checked_product({options.output_channels, output_height, output_width},
                      "Conv2D output buffer");
    if (output_elements != options.elements) {
      throw std::runtime_error(
        "Conv2D output shape does not match element count");
    }
  } else if (!options.input_shape.empty() || !options.permutation.empty() ||
             !options.kernel_size.empty() || !options.stride.empty() ||
             !options.padding.empty() || options.output_channels != 0 ||
             !options.weight_file.empty() || options.convolution_has_bias) {
    throw std::runtime_error(
      "shape, permutation, and Conv2D options are operation-specific");
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
    : options_(options),
      shader_(read_spirv(options.shader_path)),
      buffers_(options.operation == "conv2d" ? 4
               : options.operation == "add"  ? 3
                                             : 2) {}

  ~VulkanAddRunner() { destroy(); }

  VulkanAddRunner(const VulkanAddRunner&) = delete;
  VulkanAddRunner& operator=(const VulkanAddRunner&) = delete;

  std::string run(const std::vector<std::vector<float>>& inputs,
                  std::vector<float>& output) {
    if (inputs.size() + 1 != buffers_.size() ||
        output.size() != options_.elements) {
      throw std::runtime_error(
        "input/output counts do not match the operation");
    }
    buffer_sizes_.clear();
    for (const std::vector<float>& input : inputs) {
      buffer_sizes_.push_back(input.size() * sizeof(float));
    }
    buffer_sizes_.push_back(output.size() * sizeof(float));
    for (VkDeviceSize size : buffer_sizes_) {
      if (size == 0 || size > kMaximumStorageBufferBytes) {
        throw std::runtime_error(
          "input/output storage buffer violates the static size contract");
      }
    }
    create_instance();
    select_device();
    create_buffers();
    for (size_t index = 0; index < inputs.size(); ++index) {
      write_buffer(index, inputs[index].data());
    }
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
    for (VkDeviceSize buffer_size : buffer_sizes_) {
      if (buffer_size > properties_.limits.maxStorageBufferRange) {
        throw std::runtime_error("device maxStorageBufferRange is too small");
      }
    }
    const VkDeviceSize input_bytes = buffer_sizes_.front();
    const VkDeviceSize requested_bytes = options_.probe_buffer_bytes == 0
                                           ? input_bytes
                                           : options_.probe_buffer_bytes;
    if (requested_bytes < input_bytes) {
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
    const uint64_t group_count =
      (static_cast<uint64_t>(options_.elements) + kLocalSize - 1) / kLocalSize;
    if (group_count > properties_.limits.maxComputeWorkGroupCount[0]) {
      throw std::runtime_error(
        "device dispatch exceeds maxComputeWorkGroupCount");
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
    for (size_t index = 0; index < buffers_.size(); ++index) {
      Buffer& buffer = buffers_[index];
      VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      buffer_info.size = buffer_sizes_[index];
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
    std::memcpy(buffer.mapped, values, buffer_sizes_[index]);
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
    const auto buffer_count = static_cast<uint32_t>(buffers_.size());
    std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_count);
    for (uint32_t i = 0; i < buffer_count; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = buffer_count;
    layout_info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(
            device_, &layout_info, nullptr, &descriptor_layout_),
          "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   buffer_count};
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

    std::vector<VkDescriptorBufferInfo> buffer_infos(buffer_count);
    std::vector<VkWriteDescriptorSet> writes(buffer_count);
    for (uint32_t i = 0; i < buffer_count; ++i) {
      buffer_infos[i].buffer = buffers_[i].handle;
      buffer_infos[i].offset = 0;
      buffer_infos[i].range = buffer_sizes_[i];
      writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = descriptor_set_;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device_, buffer_count, writes.data(), 0, nullptr);
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
    stage_info.pName = options_.operation.c_str();
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
    std::vector<VkBufferMemoryBarrier> before(buffers_.size());
    const size_t output_index = buffers_.size() - 1;
    for (size_t i = 0; i < before.size(); ++i) {
      before[i] = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      before[i].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      before[i].dstAccessMask = i < output_index ? VK_ACCESS_SHADER_READ_BIT
                                                 : VK_ACCESS_SHADER_WRITE_BIT;
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
      command_buffer, (options_.elements + kLocalSize - 1) / kLocalSize, 1, 1);
    VkBufferMemoryBarrier after{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.buffer = buffers_[output_index].handle;
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
    Buffer& buffer = buffers_.back();
    if (!buffer.coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      check(vkInvalidateMappedMemoryRanges(device_, 1, &range),
            "vkInvalidateMappedMemoryRanges");
    }
    std::memcpy(output, buffer.mapped, options_.elements * sizeof(float));
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
  std::vector<Buffer> buffers_;
  std::vector<VkDeviceSize> buffer_sizes_;
  VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkShaderModule shader_module_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
};

struct ConvWeightData {
  std::vector<float> hwcf_weights;
  std::vector<float> bias;
};

ConvWeightData load_conv_weights(const Options& options) {
  std::ifstream source(options.weight_file, std::ios::binary | std::ios::ate);
  if (!source) {
    throw std::runtime_error("cannot open Conv2D weight file: " +
                             options.weight_file);
  }
  const std::streamsize file_size = source.tellg();
  if (file_size < 0 ||
      static_cast<std::uint64_t>(file_size) < sizeof(std::uint32_t)) {
    throw std::runtime_error("Conv2D weight file is truncated");
  }
  source.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(file_size));
  if (!source.read(bytes.data(), file_size)) {
    throw std::runtime_error("cannot read Conv2D weight file");
  }
  std::uint32_t flag = 0;
  std::memcpy(&flag, bytes.data(), sizeof(flag));
  if (flag != 0) {
    throw std::runtime_error(
      "Conv2D test runner accepts FP32 weight streams only");
  }

  const std::uint64_t input_channels = options.input_shape[0];
  const std::uint64_t kernel_height = options.kernel_size[0];
  const std::uint64_t kernel_width = options.kernel_size[1];
  const std::uint64_t output_channels = options.output_channels;
  const std::uint64_t weight_count =
    output_channels * input_channels * kernel_height * kernel_width;
  const std::uint64_t bias_count =
    options.convolution_has_bias ? output_channels : 0;
  const std::uint64_t expected_bytes =
    sizeof(flag) + ((weight_count + bias_count) * sizeof(float));
  if (bytes.size() != expected_bytes) {
    throw std::runtime_error(
      "Conv2D weight file size does not match parameters");
  }

  std::vector<float> source_weights(weight_count);
  std::memcpy(source_weights.data(),
              bytes.data() + sizeof(flag),
              weight_count * sizeof(float));
  ConvWeightData result;
  result.hwcf_weights.resize(weight_count);
  result.bias.assign(output_channels, 0.0F);
  if (bias_count != 0) {
    std::memcpy(result.bias.data(),
                bytes.data() + sizeof(flag) + (weight_count * sizeof(float)),
                bias_count * sizeof(float));
  }
  for (std::uint64_t output = 0; output < output_channels; ++output) {
    for (std::uint64_t input = 0; input < input_channels; ++input) {
      for (std::uint64_t y = 0; y < kernel_height; ++y) {
        for (std::uint64_t x = 0; x < kernel_width; ++x) {
          const std::uint64_t source_index =
            (((output * input_channels + input) * kernel_height + y) *
             kernel_width) +
            x;
          const std::uint64_t device_index =
            (((y * kernel_width + x) * input_channels + input) *
             output_channels) +
            output;
          result.hwcf_weights[device_index] = source_weights[source_index];
        }
      }
    }
  }
  return result;
}

void write_inputs(const std::string& path,
                  const std::vector<std::vector<float>>& inputs) {
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
      absolute <= kAbsTolerance + (kRelTolerance * std::abs(expected[i]));
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
         << R"(  "operation": ")" << options.operation << "\",\n"
         << "  \"seed\": " << options.seed << ",\n"
         << R"(  "device": ")" << device_name << "\",\n"
         << R"(  "device_match": ")" << options.device_match << "\",\n"
         << R"(  "api_version": ")" << VK_VERSION_MAJOR(properties.apiVersion)
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
         << "  \"elements\": " << options.elements << ",\n"
         << "  \"local_size\": " << kLocalSize << ",\n"
         << "  \"dispatch_groups_x\": "
         << (options.elements + kLocalSize - 1) / kLocalSize << ",\n"
         << "  \"validation_enabled\": "
         << (options.validation ? "true" : "false") << ",\n"
         << "  \"validation_errors\": " << g_validation_errors.load() << ",\n"
         << R"(  "tolerance": {"atol": )" << kAbsTolerance
         << ", \"rtol\": " << kRelTolerance << "},\n"
         << R"(  "cpu_vs_golden": {"passed": )"
         << (cpu_diff.passed ? "true" : "false")
         << ", \"max_abs\": " << cpu_diff.max_abs
         << ", \"max_rel\": " << cpu_diff.max_rel << "},\n"
         << R"(  "gpu_vs_golden": {"passed": )"
         << (gpu_diff.passed ? "true" : "false")
         << ", \"max_abs\": " << gpu_diff.max_abs
         << ", \"max_rel\": " << gpu_diff.max_rel << "},\n"
         << R"(  "gpu_vs_cpu": {"passed": )"
         << (cross_diff.passed ? "true" : "false")
         << ", \"max_abs\": " << cross_diff.max_abs
         << ", \"max_rel\": " << cross_diff.max_rel << "}\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const size_t input_count = options.operation == "add"      ? 2
                               : options.operation == "conv2d" ? 3
                                                               : 1;
    std::vector<std::vector<float>> inputs(input_count);
    if (options.operation == "conv2d") {
      const size_t input_elements =
        static_cast<size_t>(options.input_shape[0]) * options.input_shape[1] *
        options.input_shape[2];
      inputs[0].resize(input_elements);
      ConvWeightData weights = load_conv_weights(options);
      inputs[1] = std::move(weights.hwcf_weights);
      inputs[2] = std::move(weights.bias);
    } else {
      for (std::vector<float>& input : inputs) {
        input.resize(options.elements);
      }
    }
    std::vector<float> golden(options.elements);
    std::vector<float> cpu_output(options.elements,
                                  std::numeric_limits<float>::quiet_NaN());
    std::vector<float> gpu_output(options.elements,
                                  std::numeric_limits<float>::quiet_NaN());
    std::mt19937 generator(options.seed);
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    auto input_value = [&](uint32_t index, size_t operand) {
      if (index == 0) {
        return operand == 0 ? -1.0F : 1.0F;
      }
      if (index == 1) {
        return 0.0F;
      }
      if (index == 2) {
        return operand == 0 ? 1.0F : -1.0F;
      }
      return distribution(generator);
    };
    if (options.operation == "conv2d") {
      for (uint32_t i = 0; i < inputs[0].size(); ++i) {
        inputs[0][i] = input_value(i, 0);
      }
      const std::int64_t inputChannels = options.input_shape[0];
      const std::int64_t inputHeight = options.input_shape[1];
      const std::int64_t inputWidth = options.input_shape[2];
      const std::int64_t outputHeight =
        ((inputHeight + options.padding[0] + options.padding[2] -
          options.kernel_size[0]) /
         options.stride[0]) +
        1;
      const std::int64_t outputWidth =
        ((inputWidth + options.padding[1] + options.padding[3] -
          options.kernel_size[1]) /
         options.stride[1]) +
        1;
      for (std::int64_t outputChannel = 0;
           std::cmp_less(outputChannel, options.output_channels);
           ++outputChannel) {
        for (std::int64_t outputY = 0; outputY < outputHeight; ++outputY) {
          for (std::int64_t outputX = 0; outputX < outputWidth; ++outputX) {
            float sum = inputs[2][outputChannel];
            for (std::int64_t kernelY = 0;
                 std::cmp_less(kernelY, options.kernel_size[0]);
                 ++kernelY) {
              for (std::int64_t kernelX = 0;
                   std::cmp_less(kernelX, options.kernel_size[1]);
                   ++kernelX) {
                const std::int64_t inputY =
                  (outputY * options.stride[0]) + kernelY - options.padding[0];
                const std::int64_t inputX =
                  (outputX * options.stride[1]) + kernelX - options.padding[1];
                if (inputY < 0 || inputY >= inputHeight || inputX < 0 ||
                    inputX >= inputWidth) {
                  continue;
                }
                for (std::int64_t inputChannel = 0;
                     inputChannel < inputChannels;
                     ++inputChannel) {
                  const auto inputIndex = static_cast<std::size_t>(
                    ((inputChannel * inputHeight + inputY) * inputWidth) +
                    inputX);
                  const auto weightIndex = static_cast<std::size_t>(
                    (((kernelY * options.kernel_size[1] + kernelX) *
                        inputChannels +
                      inputChannel) *
                     options.output_channels) +
                    outputChannel);
                  sum += inputs[0][inputIndex] * inputs[1][weightIndex];
                }
              }
            }
            const auto outputIndex = static_cast<std::size_t>(
              ((outputChannel * outputHeight + outputY) * outputWidth) +
              outputX);
            golden[outputIndex] = sum;
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < options.elements; ++i) {
        inputs[0][i] = input_value(i, 0);
        if (options.operation == "add") {
          inputs[1][i] = input_value(i, 1);
          golden[i] = inputs[0][i] + inputs[1][i];
        } else if (options.operation == "relu") {
          golden[i] = std::max(inputs[0][i], 0.0F);
        } else if (options.operation != "transpose") {
          golden[i] = inputs[0][i];
        }
      }
    }
    if (options.operation == "transpose") {
      std::vector<uint32_t> output_shape(options.input_shape.size());
      for (size_t axis = 0; axis < options.permutation.size(); ++axis) {
        output_shape[axis] = options.input_shape[options.permutation[axis]];
      }
      for (uint32_t output_index = 0; output_index < options.elements;
           ++output_index) {
        uint32_t quotient = output_index;
        std::vector<uint32_t> output_coordinates(output_shape.size());
        for (size_t axis = output_shape.size(); axis-- > 0;) {
          output_coordinates[axis] = quotient % output_shape[axis];
          quotient /= output_shape[axis];
        }
        std::vector<uint32_t> input_coordinates(options.input_shape.size());
        for (size_t axis = 0; axis < output_shape.size(); ++axis) {
          input_coordinates[options.permutation[axis]] =
            output_coordinates[axis];
        }
        uint32_t input_index = 0;
        for (size_t axis = 0; axis < options.input_shape.size(); ++axis) {
          input_index =
            (input_index * options.input_shape[axis]) + input_coordinates[axis];
        }
        golden[output_index] = inputs[0][input_index];
      }
    }
    write_inputs(options.evidence_path + ".inputs.bin", inputs);

    void* cpu_library =
      dlopen(options.cpu_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!cpu_library) {
      throw std::runtime_error("dlopen CPU AOT library failed: " +
                               std::string(dlerror()));
    }
    dlerror();
    void* cpu_symbol = dlsym(cpu_library, options.cpu_symbol.c_str());
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || cpu_symbol == nullptr) {
      dlclose(cpu_library);
      throw std::runtime_error(
        "CPU AOT symbol lookup failed: " +
        std::string(symbol_error ? symbol_error : "null"));
    }
    int cpu_status = 0;
    if (options.operation == "add") {
      using CpuBinaryModel = int (*)(const float*, const float*, float*);
      cpu_status = reinterpret_cast<CpuBinaryModel>(cpu_symbol)(
        inputs[0].data(), inputs[1].data(), cpu_output.data());
    } else {
      using CpuUnaryModel = int (*)(const float*, float*);
      cpu_status = reinterpret_cast<CpuUnaryModel>(cpu_symbol)(
        inputs[0].data(), cpu_output.data());
    }
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
    std::cout << "device=" << device_name << " operation=" << options.operation
              << " elements=" << options.elements
              << " groups=" << (options.elements + kLocalSize - 1) / kLocalSize
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
