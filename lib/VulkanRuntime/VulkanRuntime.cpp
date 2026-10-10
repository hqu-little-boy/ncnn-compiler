#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <expected>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ncnn-mlir/Support/VulkanModelPayload.hpp"
#include <vulkan/vulkan.h>

namespace ncnn_mlir::vulkan {
namespace {

constexpr std::uint64_t kDefaultContextMemoryBytes = 512ULL * 1024ULL * 1024ULL;

class RuntimeFailure final : public std::exception {
 public:
  explicit RuntimeFailure(VulkanRuntimeError error)
    : error_(std::move(error)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return error_.diagnostic.c_str();
  }

  [[nodiscard]] const VulkanRuntimeError& get_error() const noexcept {
    return error_;
  }

 private:
  VulkanRuntimeError error_;
};

[[noreturn]] void fail(ncnn_vk_status status, std::string diagnostic) {
  throw RuntimeFailure(
    VulkanRuntimeError{.status = status, .diagnostic = std::move(diagnostic)});
}

ncnn_vk_status mapVkResult(VkResult result) noexcept {
  if (result == VK_ERROR_OUT_OF_HOST_MEMORY ||
      result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
      result == VK_ERROR_OUT_OF_POOL_MEMORY ||
      result == VK_ERROR_FRAGMENTED_POOL ||
      result == VK_ERROR_TOO_MANY_OBJECTS) {
    return NCNN_VK_STATUS_OUT_OF_MEMORY;
  }
  if (result == VK_ERROR_FEATURE_NOT_PRESENT ||
      result == VK_ERROR_EXTENSION_NOT_PRESENT ||
      result == VK_ERROR_INCOMPATIBLE_DRIVER) {
    return NCNN_VK_STATUS_UNSUPPORTED_DEVICE;
  }
  if (result == VK_ERROR_DEVICE_LOST) {
    return NCNN_VK_STATUS_DEVICE_LOST;
  }
  return NCNN_VK_STATUS_API_ERROR;
}

void checkVk(VkResult result, std::string_view operation) {
  if (result != VK_SUCCESS) {
    fail(mapVkResult(result),
         std::string(operation) + " failed with VkResult " +
           std::to_string(result));
  }
}

template <typename Structure>
Structure makeVkStructure(VkStructureType type) {
  Structure structure{};
  structure.sType = type;
  return structure;
}

bool checkedAdd(std::uint64_t lhs,
                std::uint64_t rhs,
                std::uint64_t& result) noexcept {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

std::uint64_t resourceElements(const TensorResource& resource) {
  if (resource.byte_size == 0 || resource.byte_size % sizeof(float) != 0) {
    fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
         "resource " + std::to_string(resource.id) +
           " has invalid FP32 byte size");
  }
  return resource.byte_size / sizeof(float);
}

}  // namespace

class VulkanRuntimeContext::Impl final {
 public:
  struct Buffer final {
    VkBuffer handle;
    VkDeviceMemory memory;
    VkDeviceSize logical_size;
    VkDeviceSize allocation_size;
    void* mapped;
    bool coherent;
  };

  struct Resource final {
    const TensorResource* spec;
    Buffer device;
    Buffer staging;
  };

  struct Kernel final {
    const KernelExecution* spec;
    VkShaderModule module;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    std::vector<VkDescriptorSetLayout> set_layouts;
    std::vector<VkDescriptorSet> sets;
  };

  Impl(VulkanExecutionPlan plan,
       VulkanRuntimeOptions options,
       std::vector<VulkanKernelBinary> kernels,
       std::vector<VulkanResourcePayload> weights)
    : plan_(std::move(plan)),
      options_(std::move(options)),
      kernels_(std::move(kernels)),
      weights_(std::move(weights)),
      instance_(VK_NULL_HANDLE),
      messenger_(VK_NULL_HANDLE),
      physical_device_(VK_NULL_HANDLE),
      properties_(),
      memory_properties_(),
      device_(VK_NULL_HANDLE),
      queue_(VK_NULL_HANDLE),
      queue_family_(0),
      command_pool_(VK_NULL_HANDLE),
      command_buffer_(VK_NULL_HANDLE),
      fence_(VK_NULL_HANDLE),
      descriptor_pool_(VK_NULL_HANDLE),
      kernels_state_(),
      resources_(),
      resource_requirements_(),
      resource_memory_types_(),
      staging_requirements_(),
      staging_memory_types_(),
      weight_staging_(),
      weight_staging_requirements_(),
      weight_staging_memory_types_(),
      peak_memory_bytes_(0),
      weight_upload_count_(0),
      validation_errors_(0),
      run_mutex_(),
      destroyed_(false),
      unusable_status_(NCNN_VK_STATUS_SUCCESS),
      destroy_status_(NCNN_VK_STATUS_SUCCESS) {}

  ~Impl() { (void)destroy(); }

  void initialize() {
    const auto verified = verifyExecutionPlan(plan_);
    if (!verified) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "invalid Vulkan execution plan: " + verified.error());
    }
    validatePackage();
    createInstance();
    selectDevice();
    createDevice();
    createCommandResources();
    createBuffers();
    allocateBuffers();
    createDescriptorsAndPipelines();
    uploadWeights();
  }

  std::expected<void, VulkanRuntimeError> run(
    std::span<const VulkanInputTensor> inputs,
    std::span<const VulkanOutputTensor> outputs) {
    std::scoped_lock lock(run_mutex_);
    try {
      if (unusable_status_ != NCNN_VK_STATUS_SUCCESS) {
        fail(unusable_status_,
             "Vulkan context cannot run after a previous execution failure");
      }
      validateHostTensors(inputs, outputs);
      try {
        prepareInputStaging(inputs);
        recordAndSubmit(inputs, outputs);
        invalidateOutputs(outputs);
        publishOutputs(outputs);
        return {};
      } catch (const RuntimeFailure& error) {
        unusable_status_ = error.get_error().status;
        throw;
      } catch (const std::bad_alloc&) {
        unusable_status_ = NCNN_VK_STATUS_OUT_OF_MEMORY;
        throw;
      } catch (const std::exception&) {
        unusable_status_ = NCNN_VK_STATUS_API_ERROR;
        throw;
      } catch (...) {
        unusable_status_ = NCNN_VK_STATUS_API_ERROR;
        fail(NCNN_VK_STATUS_API_ERROR, "unknown exception during Vulkan run");
      }
    } catch (const RuntimeFailure& error) {
      return std::unexpected(error.get_error());
    } catch (const std::bad_alloc&) {
      return std::unexpected(VulkanRuntimeError{
        .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
        .diagnostic = "host allocation failed during Vulkan run"});
    } catch (const std::exception& error) {
      return std::unexpected(VulkanRuntimeError{
        .status = NCNN_VK_STATUS_API_ERROR, .diagnostic = error.what()});
    }
  }

  ncnn_vk_status destroy() noexcept {
    if (destroyed_) {
      return destroy_status_;
    }
    destroyed_ = true;
    if (device_ != VK_NULL_HANDLE) {
      const VkResult wait_result = vkDeviceWaitIdle(device_);
      if (wait_result != VK_SUCCESS) {
        destroy_status_ = mapVkResult(wait_result);
      }
      for (Kernel& kernel : kernels_state_) {
        if (kernel.pipeline != VK_NULL_HANDLE) {
          vkDestroyPipeline(device_, kernel.pipeline, nullptr);
        }
        if (kernel.layout != VK_NULL_HANDLE) {
          vkDestroyPipelineLayout(device_, kernel.layout, nullptr);
        }
        if (kernel.module != VK_NULL_HANDLE) {
          vkDestroyShaderModule(device_, kernel.module, nullptr);
        }
      }
      if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
      }
      for (const Kernel& kernel : kernels_state_) {
        for (VkDescriptorSetLayout layout : kernel.set_layouts) {
          if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout, nullptr);
          }
        }
      }
      for (auto& [id, resource] : resources_) {
        (void)id;
        destroyBuffer(resource.staging);
        destroyBuffer(resource.device);
      }
      for (auto& [id, buffer] : weight_staging_) {
        (void)id;
        destroyBuffer(buffer);
      }
      if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
      }
      if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
      }
      vkDestroyDevice(device_, nullptr);
      device_ = VK_NULL_HANDLE;
      if (destroy_status_ == NCNN_VK_STATUS_SUCCESS &&
          validation_errors_.load() != 0) {
        destroy_status_ = NCNN_VK_STATUS_API_ERROR;
      }
    }
    if (messenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
      const auto destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
      if (destroy_messenger != nullptr) {
        destroy_messenger(instance_, messenger_, nullptr);
      }
      messenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
      vkDestroyInstance(instance_, nullptr);
      instance_ = VK_NULL_HANDLE;
    }
    return destroy_status_;
  }

  [[nodiscard]] VulkanRuntimeDeviceInfo get_device_info() const {
    return VulkanRuntimeDeviceInfo{.name = properties_.deviceName,
                                   .api_version = properties_.apiVersion,
                                   .driver_version = properties_.driverVersion,
                                   .vendor_id = properties_.vendorID,
                                   .device_id = properties_.deviceID};
  }

  [[nodiscard]] std::uint64_t get_peak_memory_bytes() const noexcept {
    return peak_memory_bytes_;
  }

  [[nodiscard]] std::uint64_t get_weight_upload_count() const noexcept {
    return weight_upload_count_;
  }

  [[nodiscard]] std::uint32_t get_validation_error_count() const noexcept {
    return validation_errors_.load();
  }

  void releasePackageViews() noexcept {
    kernels_.clear();
    weights_.clear();
  }

 private:
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  validationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                     VkDebugUtilsMessageTypeFlagsEXT,
                     const VkDebugUtilsMessengerCallbackDataEXT*,
                     void* user_data) {
    auto* self = static_cast<Impl*>(user_data);
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
      self->validation_errors_.fetch_add(1);
    }
    return VK_FALSE;
  }

  void inject(VulkanRuntimeOperation operation) const {
    if (!options_.failure_injector) {
      return;
    }
    const std::optional<ncnn_vk_status> status =
      options_.failure_injector(operation);
    if (status) {
      fail(*status,
           "injected Vulkan runtime failure at " +
             std::string(operationName(operation)));
    }
  }

  static std::string_view operationName(
    VulkanRuntimeOperation operation) noexcept {
    switch (operation) {
      case VulkanRuntimeOperation::CreateInstance:
        return "instance creation";
      case VulkanRuntimeOperation::SelectDevice:
        return "device selection";
      case VulkanRuntimeOperation::CreateDevice:
        return "device creation";
      case VulkanRuntimeOperation::CreateBuffer:
        return "buffer creation";
      case VulkanRuntimeOperation::AllocateMemory:
        return "memory allocation";
      case VulkanRuntimeOperation::CreateDescriptor:
        return "descriptor creation";
      case VulkanRuntimeOperation::CreateShaderModule:
        return "shader module creation";
      case VulkanRuntimeOperation::CreatePipeline:
        return "pipeline creation";
      case VulkanRuntimeOperation::CreateCommandResources:
        return "command resource creation";
      case VulkanRuntimeOperation::Submit:
        return "queue submission";
      case VulkanRuntimeOperation::Wait:
        return "queue wait";
      case VulkanRuntimeOperation::FlushMappedMemory:
        return "mapped-memory flush";
      case VulkanRuntimeOperation::InvalidateMappedMemory:
        return "mapped-memory invalidate";
    }
    return "unknown operation";
  }

  void validatePackage() {
    if (options_.max_context_memory_bytes == 0) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "context memory budget must be nonzero");
    }
    if (plan_.device_target != "vulkan1.1-spirv1.0") {
      fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
           "runtime supports only Vulkan 1.1 / SPIR-V 1.0 plans");
    }
    if (plan_.kernels.empty()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "execution plan contains no kernels");
    }

    std::unordered_set<std::uint32_t> expected_kernel_ids;
    for (const KernelExecution& kernel : plan_.kernels) {
      expected_kernel_ids.insert(kernel.id);
    }
    if (expected_kernel_ids.size() != kernels_.size()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "SPIR-V package does not match the plan kernel count");
    }
    std::unordered_set<std::uint32_t> found_kernel_ids;
    for (const VulkanKernelBinary& kernel : kernels_) {
      if (!expected_kernel_ids.contains(kernel.kernel_id) ||
          !found_kernel_ids.insert(kernel.kernel_id).second ||
          kernel.words.size() < 5 || kernel.words.front() != 0x07230203U) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "SPIR-V package has a missing, duplicate, or invalid kernel");
      }
    }

    std::unordered_set<std::uint32_t> expected_weights;
    for (const TensorResource& resource : plan_.resources) {
      (void)resourceElements(resource);
      if (resource.kind == ResourceKind::Weight) {
        expected_weights.insert(resource.id);
      }
    }
    if (expected_weights.size() != weights_.size()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "weight payload count does not match the plan");
    }
    std::unordered_set<std::uint32_t> found_weights;
    for (const VulkanResourcePayload& weight : weights_) {
      const TensorResource* resource = findResourceSpec(weight.resource_id);
      if (resource == nullptr || resource->kind != ResourceKind::Weight ||
          !found_weights.insert(weight.resource_id).second ||
          weight.bytes.size() != resource->byte_size) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "weight payload does not match the plan resource identity");
      }
      const auto* data =
        reinterpret_cast<const std::uint8_t*>(weight.bytes.data());
      const auto digest = vulkanWeightSHA256(
        *resource, std::span<const std::uint8_t>(data, weight.bytes.size()));
      if (!digest || *digest != resource->weight_sha256) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             digest ? "weight payload does not match the plan resource identity"
                    : digest.error());
      }
    }
    for (const TensorResource& resource : plan_.resources) {
      if (resource.byte_size > kMinimumStorageBufferRange) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "resource exceeds Vulkan 1.1 minimum storage-buffer range");
      }
    }
  }

  const TensorResource* findResourceSpec(std::uint32_t id) const noexcept {
    const auto found = std::ranges::find_if(
      plan_.resources,
      [id](const TensorResource& resource) { return resource.id == id; });
    return found == plan_.resources.end() ? nullptr : &*found;
  }

  void createInstance() {
    inject(VulkanRuntimeOperation::CreateInstance);
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    auto debug_info = makeVkStructure<VkDebugUtilsMessengerCreateInfoEXT>(
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
    if (options_.enable_validation) {
      std::uint32_t layer_count = 0;
      checkVk(vkEnumerateInstanceLayerProperties(&layer_count, nullptr),
              "vkEnumerateInstanceLayerProperties");
      std::vector<VkLayerProperties> available_layers(layer_count);
      checkVk(vkEnumerateInstanceLayerProperties(&layer_count,
                                                 available_layers.data()),
              "vkEnumerateInstanceLayerProperties");
      const bool validation_available = std::ranges::any_of(
        available_layers, [](const VkLayerProperties& layer) {
          return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") ==
                 0;
        });
      if (!validation_available) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "VK_LAYER_KHRONOS_validation is unavailable");
      }
      layers.push_back("VK_LAYER_KHRONOS_validation");
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      debug_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      debug_info.pfnUserCallback = validationCallback;
      debug_info.pUserData = this;
    }

    auto application =
      makeVkStructure<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.pApplicationName = "ncnn-mlir-vulkan-v0";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "ncnn-mlir-vulkan-v0";
    application.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_1;

    auto create_info = makeVkStructure<VkInstanceCreateInfo>(
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    create_info.pNext = options_.enable_validation ? &debug_info : nullptr;
    create_info.pApplicationInfo = &application;
    create_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();
    create_info.enabledExtensionCount =
      static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    checkVk(vkCreateInstance(&create_info, nullptr, &instance_),
            "vkCreateInstance");

    if (options_.enable_validation) {
      const auto create_messenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
      if (create_messenger == nullptr) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "VK_EXT_debug_utils is unavailable");
      }
      checkVk(create_messenger(instance_, &debug_info, nullptr, &messenger_),
              "vkCreateDebugUtilsMessengerEXT");
    }
  }

  void selectDevice() {
    inject(VulkanRuntimeOperation::SelectDevice);
    std::uint32_t count = 0;
    checkVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
            "vkEnumeratePhysicalDevices");
    if (count == 0) {
      fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
           "no Vulkan physical devices were found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    checkVk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
            "vkEnumeratePhysicalDevices");
    if (options_.device_index != NCNN_VK_DEVICE_AUTO &&
        options_.device_index >= devices.size()) {
      fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
           "requested Vulkan device index is unavailable");
    }

    std::ostringstream summary;
    for (std::size_t device_index = 0; device_index < devices.size();
         ++device_index) {
      if (options_.device_index != NCNN_VK_DEVICE_AUTO &&
          device_index != options_.device_index) {
        continue;
      }
      VkPhysicalDevice candidate = devices[device_index];
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(candidate, &properties);
      summary << properties.deviceName << " (Vulkan "
              << VK_VERSION_MAJOR(properties.apiVersion) << '.'
              << VK_VERSION_MINOR(properties.apiVersion) << ")";
      if (VK_VERSION_MAJOR(properties.apiVersion) < 1 ||
          (VK_VERSION_MAJOR(properties.apiVersion) == 1 &&
           VK_VERSION_MINOR(properties.apiVersion) < 1)) {
        summary << " below Vulkan 1.1; ";
        continue;
      }
      if (!options_.device_name_match.empty() &&
          std::string_view(properties.deviceName)
              .find(options_.device_name_match) == std::string_view::npos) {
        summary << " did not match '" << options_.device_name_match << "'; ";
        continue;
      }
      std::uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(
        candidate, &queue_count, nullptr);
      std::vector<VkQueueFamilyProperties> queues(queue_count);
      vkGetPhysicalDeviceQueueFamilyProperties(
        candidate, &queue_count, queues.data());
      for (std::uint32_t family = 0; family < queue_count; ++family) {
        if ((queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
          physical_device_ = candidate;
          queue_family_ = family;
          properties_ = properties;
          vkGetPhysicalDeviceMemoryProperties(candidate, &memory_properties_);
          return;
        }
      }
      summary << " has no compute queue; ";
    }
    fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
         "no compatible Vulkan 1.1 compute queue was found (devices=" +
           std::to_string(count) + ", requested_index=" +
           std::to_string(options_.device_index) + "): " + summary.str());
  }

  void createDevice() {
    inject(VulkanRuntimeOperation::CreateDevice);
    constexpr float priority = 1.0F;
    auto queue_info = makeVkStructure<VkDeviceQueueCreateInfo>(
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    auto device_info =
      makeVkStructure<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    checkVk(vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
            "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    if (queue_ == VK_NULL_HANDLE) {
      fail(NCNN_VK_STATUS_API_ERROR, "vkGetDeviceQueue returned a null queue");
    }
  }

  void validateDeviceLimits() const {
    if (properties_.limits.maxStorageBufferRange == 0 ||
        properties_.limits.maxPushConstantsSize == 0) {
      fail(
        NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
        "Vulkan device lacks required storage-buffer or push-constant limits");
    }
    for (const KernelExecution& kernel : plan_.kernels) {
      const auto local = kernel.local_size;
      if (local[0] == 0 || local[1] != 1 || local[2] != 1 ||
          local[0] > properties_.limits.maxComputeWorkGroupSize[0] ||
          local[0] > properties_.limits.maxComputeWorkGroupInvocations) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "plan workgroup size is unsupported by the selected device");
      }
      std::uint64_t output_elements = 0;
      for (std::uint32_t id : kernel.writes) {
        const TensorResource* resource = findResourceSpec(id);
        if (resource == nullptr) {
          fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
               "kernel writes an unknown resource");
        }
        output_elements =
          std::max(output_elements, resourceElements(*resource));
      }
      const std::uint64_t groups = (output_elements + local[0] - 1U) / local[0];
      if (groups == 0 ||
          groups > properties_.limits.maxComputeWorkGroupCount[0]) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "plan dispatch exceeds the selected device workgroup limit");
      }
      std::uint32_t max_push_end = 0;
      for (const PushConstantBlock& block : kernel.push_constants) {
        if (block.bytes.empty() || block.offset % 4 != 0 ||
            block.bytes.size() % 4 != 0 ||
            block.bytes.size() >
              std::numeric_limits<std::uint32_t>::max() - block.offset) {
          fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
               "kernel push-constant block has invalid alignment or size");
        }
        max_push_end = std::max(
          max_push_end,
          block.offset + static_cast<std::uint32_t>(block.bytes.size()));
      }
      if (max_push_end > properties_.limits.maxPushConstantsSize) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "plan push constants exceed the selected device limit");
      }
      std::uint32_t max_set = 0;
      for (const DescriptorBinding& binding : kernel.bindings) {
        max_set = std::max(max_set, binding.set);
      }
      if (!kernel.bindings.empty() &&
          max_set >= properties_.limits.maxBoundDescriptorSets) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "plan descriptor sets exceed the selected device limit");
      }
      if (kernel.bindings.size() >
          properties_.limits.maxPerStageDescriptorStorageBuffers) {
        fail(
          NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
          "plan storage-buffer descriptors exceed the selected device limit");
      }
      for (const TensorResource& resource : plan_.resources) {
        if (resource.byte_size > properties_.limits.maxStorageBufferRange) {
          fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
               "plan resource exceeds selected device maxStorageBufferRange");
        }
      }
    }
  }

  std::uint32_t findMemoryType(std::uint32_t allowed,
                               VkMemoryPropertyFlags required,
                               VkMemoryPropertyFlags preferred,
                               bool* coherent) const {
    if (options_.force_mapped_memory_sync && coherent != nullptr &&
        (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      std::optional<std::uint32_t> coherent_fallback;
      for (std::uint32_t index = 0; index < memory_properties_.memoryTypeCount;
           ++index) {
        if ((allowed & (1U << index)) == 0) {
          continue;
        }
        const VkMemoryPropertyFlags flags =
          memory_properties_.memoryTypes[index].propertyFlags;
        if ((flags & required) != required) {
          continue;
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
          *coherent = false;
          return index;
        }
        if (!coherent_fallback) {
          coherent_fallback = index;
        }
      }
      if (coherent_fallback) {
        *coherent = false;
        return *coherent_fallback;
      }
    }
    std::optional<std::uint32_t> fallback;
    for (std::uint32_t index = 0; index < memory_properties_.memoryTypeCount;
         ++index) {
      if ((allowed & (1U << index)) == 0) {
        continue;
      }
      const VkMemoryPropertyFlags flags =
        memory_properties_.memoryTypes[index].propertyFlags;
      if ((flags & required) != required) {
        continue;
      }
      if (!fallback) {
        fallback = index;
      }
      if ((flags & preferred) == preferred) {
        if (coherent != nullptr) {
          *coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        }
        return index;
      }
    }
    if (!fallback) {
      fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
           "no Vulkan memory type satisfies buffer requirements");
    }
    const VkMemoryPropertyFlags flags =
      memory_properties_.memoryTypes[*fallback].propertyFlags;
    if (coherent != nullptr) {
      *coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
    return *fallback;
  }

  Buffer createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      bool host_visible,
                      VkMemoryRequirements& requirements,
                      std::uint32_t& memory_type) {
    if (size == 0 || size > std::numeric_limits<VkDeviceSize>::max()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT, "invalid Vulkan buffer size");
    }
    inject(VulkanRuntimeOperation::CreateBuffer);
    auto buffer_info =
      makeVkStructure<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Buffer buffer{.handle = VK_NULL_HANDLE,
                  .memory = VK_NULL_HANDLE,
                  .logical_size = size,
                  .allocation_size = 0,
                  .mapped = nullptr,
                  .coherent = false};
    checkVk(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer.handle),
            "vkCreateBuffer");
    try {
      vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
      const VkMemoryPropertyFlags required =
        host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      const VkMemoryPropertyFlags preferred =
        host_visible ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      memory_type = findMemoryType(
        requirements.memoryTypeBits, required, preferred, &buffer.coherent);
      if (host_visible && options_.force_mapped_memory_sync) {
        buffer.coherent = false;
      }
      buffer.allocation_size = requirements.size;
      return buffer;
    } catch (...) {
      vkDestroyBuffer(device_, buffer.handle, nullptr);
      throw;
    }
  }

  void createCommandResources() {
    inject(VulkanRuntimeOperation::CreateCommandResources);
    auto pool_info = makeVkStructure<VkCommandPoolCreateInfo>(
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_;
    checkVk(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
            "vkCreateCommandPool");
    auto command_info = makeVkStructure<VkCommandBufferAllocateInfo>(
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = command_pool_;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    inject(VulkanRuntimeOperation::CreateCommandResources);
    checkVk(vkAllocateCommandBuffers(device_, &command_info, &command_buffer_),
            "vkAllocateCommandBuffers");
    inject(VulkanRuntimeOperation::CreateCommandResources);
    auto fence_info =
      makeVkStructure<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    checkVk(vkCreateFence(device_, &fence_info, nullptr, &fence_),
            "vkCreateFence");
  }

  void createBuffers() {
    validateDeviceLimits();
    for (const TensorResource& spec : plan_.resources) {
      const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      VkMemoryRequirements requirements{};
      std::uint32_t memory_type = 0;
      Buffer device_buffer =
        createBuffer(spec.byte_size, usage, false, requirements, memory_type);
      Resource resource{.spec = &spec,
                        .device = device_buffer,
                        .staging = {.handle = VK_NULL_HANDLE,
                                    .memory = VK_NULL_HANDLE,
                                    .logical_size = spec.byte_size,
                                    .allocation_size = 0,
                                    .mapped = nullptr,
                                    .coherent = false}};
      bool buffer_owned = false;
      try {
        const auto [iterator, inserted] = resources_.emplace(spec.id, resource);
        (void)iterator;
        if (!inserted) {
          fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
               "execution plan contains duplicate resource IDs");
        }
        buffer_owned = true;
      } catch (...) {
        if (!buffer_owned) {
          vkDestroyBuffer(device_, device_buffer.handle, nullptr);
        }
        throw;
      }
      resource_memory_types_[spec.id] = memory_type;
      resource_requirements_[spec.id] = requirements;
    }

    for (const TensorResource& spec : plan_.resources) {
      if (spec.kind != ResourceKind::Input &&
          spec.kind != ResourceKind::Output) {
        continue;
      }
      VkMemoryRequirements requirements{};
      std::uint32_t memory_type = 0;
      Buffer staging = createBuffer(
        spec.byte_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        true,
        requirements,
        memory_type);
      resources_.at(spec.id).staging = staging;
      staging_memory_types_[spec.id] = memory_type;
      staging_requirements_[spec.id] = requirements;
    }

    std::uint64_t total_bytes = 0;
    std::uint64_t total_weight_staging = 0;
    for (const auto& [id, resource] : resources_) {
      (void)id;
      if (!checkedAdd(
            total_bytes, resource.device.allocation_size, total_bytes)) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "Vulkan context allocation size overflows uint64");
      }
      if (resource.staging.handle != VK_NULL_HANDLE &&
          !checkedAdd(
            total_bytes, resource.staging.allocation_size, total_bytes)) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "Vulkan staging allocation size overflows uint64");
      }
    }
    for (const TensorResource& spec : plan_.resources) {
      if (spec.kind == ResourceKind::Weight) {
        VkMemoryRequirements requirements{};
        std::uint32_t memory_type = 0;
        Buffer staging = createBuffer(spec.byte_size,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      true,
                                      requirements,
                                      memory_type);
        bool staging_owned = false;
        try {
          if (!checkedAdd(total_weight_staging,
                          requirements.size,
                          total_weight_staging)) {
            fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
                 "weight staging allocation size overflows uint64");
          }
          const auto [iterator, inserted] =
            weight_staging_.emplace(spec.id, staging);
          (void)iterator;
          if (!inserted) {
            fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
                 "execution plan contains duplicate weight resources");
          }
          staging_owned = true;
        } catch (...) {
          if (!staging_owned) {
            vkDestroyBuffer(device_, staging.handle, nullptr);
          }
          throw;
        }
        weight_staging_memory_types_[spec.id] = memory_type;
        weight_staging_requirements_[spec.id] = requirements;
      }
    }
    std::uint64_t allocation_count = 0;
    if (!checkedAdd(allocation_count,
                    static_cast<std::uint64_t>(resources_.size()),
                    allocation_count) ||
        !checkedAdd(allocation_count,
                    static_cast<std::uint64_t>(staging_requirements_.size()),
                    allocation_count) ||
        !checkedAdd(allocation_count,
                    static_cast<std::uint64_t>(weight_staging_.size()),
                    allocation_count)) {
      fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
           "Vulkan context allocation count overflows uint64");
    }
    if (allocation_count > properties_.limits.maxMemoryAllocationCount) {
      fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
           "Vulkan context exceeds the device allocation-count limit");
    }
    std::uint64_t peak = 0;
    if (!checkedAdd(total_bytes, total_weight_staging, peak) ||
        peak > options_.max_context_memory_bytes) {
      fail(
        NCNN_VK_STATUS_OUT_OF_MEMORY,
        "Vulkan context exceeds configured memory budget (required at most " +
          std::to_string(peak) + " bytes, limit " +
          std::to_string(options_.max_context_memory_bytes) + " bytes)");
    }
    peak_memory_bytes_ = peak;
  }

  void allocateOne(Buffer& buffer,
                   VkMemoryRequirements requirements,
                   std::uint32_t memory_type,
                   bool map) {
    inject(VulkanRuntimeOperation::AllocateMemory);
    auto allocation = makeVkStructure<VkMemoryAllocateInfo>(
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    checkVk(vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory),
            "vkAllocateMemory");
    checkVk(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0),
            "vkBindBufferMemory");
    if (map) {
      checkVk(vkMapMemory(
                device_, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped),
              "vkMapMemory");
    }
  }

  void allocateBuffers() {
    for (auto& [id, resource] : resources_) {
      allocateOne(resource.device,
                  resource_requirements_.at(id),
                  resource_memory_types_.at(id),
                  false);
      if (resource.staging.handle != VK_NULL_HANDLE) {
        allocateOne(resource.staging,
                    staging_requirements_.at(id),
                    staging_memory_types_.at(id),
                    true);
      }
    }
    for (auto& [id, buffer] : weight_staging_) {
      allocateOne(buffer,
                  weight_staging_requirements_.at(id),
                  weight_staging_memory_types_.at(id),
                  true);
    }
  }

  void createDescriptorsAndPipelines() {
    std::uint64_t descriptor_count = 0;
    std::uint64_t set_count = 0;
    for (const KernelExecution& spec : plan_.kernels) {
      if (!checkedAdd(descriptor_count,
                      static_cast<std::uint64_t>(spec.bindings.size()),
                      descriptor_count)) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "execution plan descriptor count overflows uint64");
      }
      std::uint32_t max_set = 0;
      for (const DescriptorBinding& binding : spec.bindings) {
        max_set = std::max(max_set, binding.set);
      }
      if (!spec.bindings.empty() &&
          !checkedAdd(
            set_count, static_cast<std::uint64_t>(max_set) + 1, set_count)) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "execution plan descriptor-set count overflows uint64");
      }
    }
    if (descriptor_count > std::numeric_limits<std::uint32_t>::max() ||
        set_count > std::numeric_limits<std::uint32_t>::max()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "execution plan descriptor counts exceed Vulkan limits");
    }
    if (descriptor_count > 0) {
      VkDescriptorPoolSize pool_size{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        static_cast<std::uint32_t>(descriptor_count)};
      auto pool_info = makeVkStructure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
      pool_info.maxSets = static_cast<std::uint32_t>(set_count);
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes = &pool_size;
      inject(VulkanRuntimeOperation::CreateDescriptor);
      checkVk(
        vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
    }

    for (const KernelExecution& spec : plan_.kernels) {
      const VulkanKernelBinary& binary = findKernelBinary(spec.id);
      Kernel& kernel =
        kernels_state_.emplace_back(Kernel{.spec = &spec,
                                           .module = VK_NULL_HANDLE,
                                           .layout = VK_NULL_HANDLE,
                                           .pipeline = VK_NULL_HANDLE,
                                           .set_layouts = {},
                                           .sets = {}});
      auto shader_info = makeVkStructure<VkShaderModuleCreateInfo>(
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
      shader_info.codeSize = binary.words.size_bytes();
      shader_info.pCode = binary.words.data();
      inject(VulkanRuntimeOperation::CreateShaderModule);
      checkVk(
        vkCreateShaderModule(device_, &shader_info, nullptr, &kernel.module),
        "vkCreateShaderModule");

      std::uint32_t max_set = 0;
      for (const DescriptorBinding& binding : spec.bindings) {
        max_set = std::max(max_set, binding.set);
      }
      if (!spec.bindings.empty()) {
        kernel.set_layouts.resize(max_set + 1, VK_NULL_HANDLE);
        for (std::uint32_t set = 0; set <= max_set; ++set) {
          std::vector<VkDescriptorSetLayoutBinding> bindings;
          for (const DescriptorBinding& descriptor : spec.bindings) {
            if (descriptor.set != set) {
              continue;
            }
            bindings.push_back(VkDescriptorSetLayoutBinding{
              .binding = descriptor.binding,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1,
              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
              .pImmutableSamplers = nullptr});
          }
          auto layout_info = makeVkStructure<VkDescriptorSetLayoutCreateInfo>(
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
          layout_info.bindingCount =
            static_cast<std::uint32_t>(bindings.size());
          layout_info.pBindings = bindings.data();
          inject(VulkanRuntimeOperation::CreateDescriptor);
          checkVk(vkCreateDescriptorSetLayout(
                    device_, &layout_info, nullptr, &kernel.set_layouts[set]),
                  "vkCreateDescriptorSetLayout");
        }

        std::vector<VkDescriptorSetLayout> layout_handles = kernel.set_layouts;
        auto set_info = makeVkStructure<VkDescriptorSetAllocateInfo>(
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        set_info.descriptorPool = descriptor_pool_;
        set_info.descriptorSetCount =
          static_cast<std::uint32_t>(layout_handles.size());
        set_info.pSetLayouts = layout_handles.data();
        kernel.sets.resize(layout_handles.size(), VK_NULL_HANDLE);
        inject(VulkanRuntimeOperation::CreateDescriptor);
        checkVk(
          vkAllocateDescriptorSets(device_, &set_info, kernel.sets.data()),
          "vkAllocateDescriptorSets");

        std::vector<VkDescriptorBufferInfo> buffer_infos;
        std::vector<VkWriteDescriptorSet> writes;
        buffer_infos.reserve(spec.bindings.size());
        writes.reserve(spec.bindings.size());
        for (const DescriptorBinding& descriptor : spec.bindings) {
          const Resource& resource = resources_.at(descriptor.resource_id);
          buffer_infos.push_back(
            VkDescriptorBufferInfo{.buffer = resource.device.handle,
                                   .offset = 0,
                                   .range = resource.device.logical_size});
          writes.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = kernel.sets[descriptor.set],
            .dstBinding = descriptor.binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_infos.back(),
            .pTexelBufferView = nullptr});
        }
        vkUpdateDescriptorSets(device_,
                               static_cast<std::uint32_t>(writes.size()),
                               writes.data(),
                               0,
                               nullptr);
      }

      std::uint32_t max_push_end = 0;
      for (const PushConstantBlock& block : spec.push_constants) {
        max_push_end = std::max(
          max_push_end,
          block.offset + static_cast<std::uint32_t>(block.bytes.size()));
      }
      VkPushConstantRange push_range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                     .offset = 0,
                                     .size = max_push_end};
      auto pipeline_layout_info = makeVkStructure<VkPipelineLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
      pipeline_layout_info.setLayoutCount =
        static_cast<std::uint32_t>(kernel.set_layouts.size());
      pipeline_layout_info.pSetLayouts = kernel.set_layouts.data();
      pipeline_layout_info.pushConstantRangeCount = max_push_end == 0 ? 0 : 1;
      pipeline_layout_info.pPushConstantRanges =
        max_push_end == 0 ? nullptr : &push_range;
      checkVk(vkCreatePipelineLayout(
                device_, &pipeline_layout_info, nullptr, &kernel.layout),
              "vkCreatePipelineLayout");
      auto stage = makeVkStructure<VkPipelineShaderStageCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = kernel.module;
      stage.pName = spec.entry_point.c_str();
      auto pipeline_info = makeVkStructure<VkComputePipelineCreateInfo>(
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
      pipeline_info.stage = stage;
      pipeline_info.layout = kernel.layout;
      inject(VulkanRuntimeOperation::CreatePipeline);
      checkVk(vkCreateComputePipelines(device_,
                                       VK_NULL_HANDLE,
                                       1,
                                       &pipeline_info,
                                       nullptr,
                                       &kernel.pipeline),
              "vkCreateComputePipelines");
    }
  }

  const VulkanKernelBinary& findKernelBinary(std::uint32_t id) const {
    const auto found =
      std::ranges::find_if(kernels_, [id](const VulkanKernelBinary& binary) {
        return binary.kernel_id == id;
      });
    if (found == kernels_.end()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "execution plan kernel has no SPIR-V binary");
    }
    return *found;
  }

  void uploadWeights() {
    if (weights_.empty()) {
      return;
    }
    checkVk(vkResetFences(device_, 1, &fence_), "vkResetFences");
    checkVk(vkResetCommandPool(device_, command_pool_, 0),
            "vkResetCommandPool");
    auto begin = makeVkStructure<VkCommandBufferBeginInfo>(
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(command_buffer_, &begin),
            "vkBeginCommandBuffer");
    std::vector<VkBufferMemoryBarrier> host_barriers;
    std::vector<VkBufferMemoryBarrier> device_barriers;
    for (const VulkanResourcePayload& payload : weights_) {
      Buffer& staging = weight_staging_.at(payload.resource_id);
      std::memcpy(staging.mapped, payload.bytes.data(), payload.bytes.size());
      flushBuffer(staging);
      host_barriers.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = staging.handle,
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    vkCmdPipelineBarrier(command_buffer_,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(host_barriers.size()),
                         host_barriers.data(),
                         0,
                         nullptr);
    for (const VulkanResourcePayload& payload : weights_) {
      const Buffer& staging = weight_staging_.at(payload.resource_id);
      Resource& resource = resources_.at(payload.resource_id);
      const VkBufferCopy copy{
        .srcOffset = 0, .dstOffset = 0, .size = resource.device.logical_size};
      vkCmdCopyBuffer(
        command_buffer_, staging.handle, resource.device.handle, 1, &copy);
      device_barriers.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = resource.device.handle,
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    if (!device_barriers.empty()) {
      vkCmdPipelineBarrier(command_buffer_,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0,
                           0,
                           nullptr,
                           static_cast<std::uint32_t>(device_barriers.size()),
                           device_barriers.data(),
                           0,
                           nullptr);
    }
    checkVk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");
    submitAndWait();
    weight_upload_count_ += weights_.size();
    for (auto& [id, buffer] : weight_staging_) {
      (void)id;
      destroyBuffer(buffer);
    }
    weight_staging_.clear();
  }

  void validateHostTensors(std::span<const VulkanInputTensor> inputs,
                           std::span<const VulkanOutputTensor> outputs) const {
    if (inputs.size() != plan_.inputs.size() ||
        outputs.size() != plan_.outputs.size()) {
      fail(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
           "input/output count does not match the execution plan");
    }
    std::unordered_set<std::uint32_t> input_ids;
    for (const VulkanInputTensor& input : inputs) {
      const TensorResource* spec = findResourceSpec(input.resource_id);
      if (spec == nullptr || spec->kind != ResourceKind::Input ||
          !input_ids.insert(input.resource_id).second ||
          input.bytes.data() == nullptr ||
          input.bytes.size() != spec->byte_size) {
        fail(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
             "input pointer, identity, or byte count violates the plan");
      }
    }
    for (std::uint32_t id : plan_.inputs) {
      if (!input_ids.contains(id)) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "an execution-plan input was not provided");
      }
    }
    std::unordered_set<std::uint32_t> output_ids;
    for (const VulkanOutputTensor& output : outputs) {
      const TensorResource* spec = findResourceSpec(output.resource_id);
      if (spec == nullptr || spec->kind != ResourceKind::Output ||
          !output_ids.insert(output.resource_id).second ||
          output.bytes.data() == nullptr ||
          output.bytes.size() != spec->byte_size) {
        fail(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
             "output pointer, identity, or byte count violates the plan");
      }
    }
    for (std::uint32_t id : plan_.outputs) {
      if (!output_ids.contains(id)) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "an execution-plan output was not provided");
      }
    }
  }

  void flushBuffer(const Buffer& buffer) {
    if (buffer.coherent) {
      return;
    }
    inject(VulkanRuntimeOperation::FlushMappedMemory);
    auto range = makeVkStructure<VkMappedMemoryRange>(
      VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    checkVk(vkFlushMappedMemoryRanges(device_, 1, &range),
            "vkFlushMappedMemoryRanges");
  }

  void invalidateBuffer(const Buffer& buffer) {
    if (buffer.coherent) {
      return;
    }
    inject(VulkanRuntimeOperation::InvalidateMappedMemory);
    auto range = makeVkStructure<VkMappedMemoryRange>(
      VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    checkVk(vkInvalidateMappedMemoryRanges(device_, 1, &range),
            "vkInvalidateMappedMemoryRanges");
  }

  void prepareInputStaging(std::span<const VulkanInputTensor> inputs) {
    for (const VulkanInputTensor& input : inputs) {
      Buffer& staging = resources_.at(input.resource_id).staging;
      std::memcpy(staging.mapped, input.bytes.data(), input.bytes.size());
      flushBuffer(staging);
    }
  }

  void recordAndSubmit(std::span<const VulkanInputTensor> inputs,
                       std::span<const VulkanOutputTensor> outputs) {
    checkVk(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");
    checkVk(vkResetFences(device_, 1, &fence_), "vkResetFences");
    checkVk(vkResetCommandPool(device_, command_pool_, 0),
            "vkResetCommandPool");
    auto begin = makeVkStructure<VkCommandBufferBeginInfo>(
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(command_buffer_, &begin),
            "vkBeginCommandBuffer");

    std::vector<VkBufferMemoryBarrier> host_to_transfer;
    std::vector<VkBufferMemoryBarrier> transfer_to_compute;
    for (const VulkanInputTensor& input : inputs) {
      Resource& resource = resources_.at(input.resource_id);
      host_to_transfer.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = resource.staging.handle,
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    if (!host_to_transfer.empty()) {
      vkCmdPipelineBarrier(command_buffer_,
                           VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0,
                           0,
                           nullptr,
                           static_cast<std::uint32_t>(host_to_transfer.size()),
                           host_to_transfer.data(),
                           0,
                           nullptr);
    }
    for (const VulkanInputTensor& input : inputs) {
      Resource& resource = resources_.at(input.resource_id);
      const VkBufferCopy copy{
        .srcOffset = 0, .dstOffset = 0, .size = resource.device.logical_size};
      vkCmdCopyBuffer(command_buffer_,
                      resource.staging.handle,
                      resource.device.handle,
                      1,
                      &copy);
      transfer_to_compute.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = resource.device.handle,
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    if (!transfer_to_compute.empty()) {
      vkCmdPipelineBarrier(
        command_buffer_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<std::uint32_t>(transfer_to_compute.size()),
        transfer_to_compute.data(),
        0,
        nullptr);
    }

    std::unordered_set<std::uint32_t> shader_written;
    for (std::size_t kernel_index = 0; kernel_index < kernels_state_.size();
         ++kernel_index) {
      const Kernel& kernel = kernels_state_[kernel_index];
      std::vector<VkBufferMemoryBarrier> barriers;
      std::unordered_set<std::uint32_t> current_ids(kernel.spec->reads.begin(),
                                                    kernel.spec->reads.end());
      current_ids.insert(kernel.spec->writes.begin(),
                         kernel.spec->writes.end());
      for (std::uint32_t id : current_ids) {
        if (!shader_written.contains(id)) {
          continue;
        }
        const Resource& resource = resources_.at(id);
        barriers.push_back(VkBufferMemoryBarrier{
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = resource.device.handle,
          .offset = 0,
          .size = VK_WHOLE_SIZE});
      }
      if (!barriers.empty()) {
        vkCmdPipelineBarrier(command_buffer_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             static_cast<std::uint32_t>(barriers.size()),
                             barriers.data(),
                             0,
                             nullptr);
      }

      vkCmdBindPipeline(
        command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
      if (!kernel.sets.empty()) {
        vkCmdBindDescriptorSets(command_buffer_,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                kernel.layout,
                                0,
                                static_cast<std::uint32_t>(kernel.sets.size()),
                                kernel.sets.data(),
                                0,
                                nullptr);
      }
      for (const PushConstantBlock& block : kernel.spec->push_constants) {
        vkCmdPushConstants(command_buffer_,
                           kernel.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           block.offset,
                           static_cast<std::uint32_t>(block.bytes.size()),
                           block.bytes.data());
      }
      std::uint64_t output_elements = 0;
      for (std::uint32_t id : kernel.spec->writes) {
        output_elements =
          std::max(output_elements, resourceElements(*resources_.at(id).spec));
      }
      const auto groups = static_cast<std::uint32_t>(
        (output_elements + kernel.spec->local_size[0] - 1U) /
        kernel.spec->local_size[0]);
      vkCmdDispatch(command_buffer_, groups, 1, 1);
      shader_written.insert(kernel.spec->writes.begin(),
                            kernel.spec->writes.end());
      (void)kernel_index;
    }

    std::vector<VkBufferMemoryBarrier> compute_to_transfer;
    for (const VulkanOutputTensor& output : outputs) {
      Resource& resource = resources_.at(output.resource_id);
      compute_to_transfer.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = resource.device.handle,
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    vkCmdPipelineBarrier(command_buffer_,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(compute_to_transfer.size()),
                         compute_to_transfer.data(),
                         0,
                         nullptr);
    std::vector<VkBufferMemoryBarrier> transfer_to_host;
    for (const VulkanOutputTensor& output : outputs) {
      Resource& resource = resources_.at(output.resource_id);
      const VkBufferCopy copy{
        .srcOffset = 0, .dstOffset = 0, .size = resource.device.logical_size};
      vkCmdCopyBuffer(command_buffer_,
                      resource.device.handle,
                      resource.staging.handle,
                      1,
                      &copy);
      transfer_to_host.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = resource.staging.handle,
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    vkCmdPipelineBarrier(command_buffer_,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(transfer_to_host.size()),
                         transfer_to_host.data(),
                         0,
                         nullptr);
    checkVk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");
    submitAndWait();
  }

  void submitAndWait() {
    inject(VulkanRuntimeOperation::Submit);
    auto submit = makeVkStructure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    checkVk(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
    inject(VulkanRuntimeOperation::Wait);
    checkVk(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");
  }

  void invalidateOutputs(std::span<const VulkanOutputTensor> outputs) {
    for (const VulkanOutputTensor& output : outputs) {
      invalidateBuffer(resources_.at(output.resource_id).staging);
    }
  }

  void publishOutputs(std::span<const VulkanOutputTensor> outputs) {
    std::vector<std::reference_wrapper<const Buffer>> staging_buffers;
    staging_buffers.reserve(outputs.size());
    for (const VulkanOutputTensor& output : outputs) {
      const Buffer& staging = resources_.at(output.resource_id).staging;
      if (staging.mapped == nullptr) {
        fail(NCNN_VK_STATUS_API_ERROR, "output staging memory is not mapped");
      }
      staging_buffers.emplace_back(staging);
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      std::memcpy(outputs[index].bytes.data(),
                  staging_buffers[index].get().mapped,
                  outputs[index].bytes.size());
    }
  }

  void destroyBuffer(Buffer& buffer) noexcept {
    if (device_ == VK_NULL_HANDLE) {
      return;
    }
    if (buffer.mapped != nullptr) {
      vkUnmapMemory(device_, buffer.memory);
      buffer.mapped = nullptr;
    }
    if (buffer.handle != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer.handle, nullptr);
      buffer.handle = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, buffer.memory, nullptr);
      buffer.memory = VK_NULL_HANDLE;
    }
  }

  VulkanExecutionPlan plan_;
  VulkanRuntimeOptions options_;
  std::vector<VulkanKernelBinary> kernels_;
  std::vector<VulkanResourcePayload> weights_;
  VkInstance instance_;
  VkDebugUtilsMessengerEXT messenger_;
  VkPhysicalDevice physical_device_;
  VkPhysicalDeviceProperties properties_;
  VkPhysicalDeviceMemoryProperties memory_properties_;
  VkDevice device_;
  VkQueue queue_;
  std::uint32_t queue_family_;
  VkCommandPool command_pool_;
  VkCommandBuffer command_buffer_;
  VkFence fence_;
  VkDescriptorPool descriptor_pool_;
  std::vector<Kernel> kernels_state_;
  std::unordered_map<std::uint32_t, Resource> resources_;
  std::unordered_map<std::uint32_t, VkMemoryRequirements>
    resource_requirements_;
  std::unordered_map<std::uint32_t, std::uint32_t> resource_memory_types_;
  std::unordered_map<std::uint32_t, VkMemoryRequirements> staging_requirements_;
  std::unordered_map<std::uint32_t, std::uint32_t> staging_memory_types_;
  std::unordered_map<std::uint32_t, Buffer> weight_staging_;
  std::unordered_map<std::uint32_t, VkMemoryRequirements>
    weight_staging_requirements_;
  std::unordered_map<std::uint32_t, std::uint32_t> weight_staging_memory_types_;
  std::uint64_t peak_memory_bytes_;
  std::uint64_t weight_upload_count_;
  std::atomic<std::uint32_t> validation_errors_;
  std::mutex run_mutex_;
  bool destroyed_;
  ncnn_vk_status unusable_status_;
  ncnn_vk_status destroy_status_;
};

VulkanRuntimeContext::VulkanRuntimeContext(std::unique_ptr<Impl> impl) noexcept
  : impl_(std::move(impl)) {}

VulkanRuntimeContext::VulkanRuntimeContext(
  VulkanRuntimeContext&& other) noexcept
  : impl_(std::move(other.impl_)) {}

VulkanRuntimeContext& VulkanRuntimeContext::operator=(
  VulkanRuntimeContext&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

VulkanRuntimeContext::~VulkanRuntimeContext() = default;

std::expected<VulkanRuntimeContext, VulkanRuntimeError>
VulkanRuntimeContext::create(const VulkanExecutionPlan& plan,
                             std::span<const VulkanKernelBinary> kernels,
                             std::span<const VulkanResourcePayload> weights,
                             const VulkanRuntimeOptions& options) {
  std::unique_ptr<Impl> impl;
  auto cleanup_failed_create = [&impl](std::string& diagnostic) noexcept {
    if (!impl) {
      return;
    }
    const ncnn_vk_status cleanup_status = impl->destroy();
    const std::uint32_t validation_errors = impl->get_validation_error_count();
    if (cleanup_status != NCNN_VK_STATUS_SUCCESS) {
      try {
        diagnostic += "; partial cleanup returned " +
                      std::string(getVulkanRuntimeStatusName(cleanup_status));
      } catch (...) {}
    }
    if (validation_errors != 0) {
      try {
        diagnostic +=
          "; validation_error_count=" + std::to_string(validation_errors) +
          " during partial cleanup";
      } catch (...) {}
    }
  };
  try {
    std::vector<VulkanKernelBinary> kernel_copies(kernels.begin(),
                                                  kernels.end());
    std::vector<VulkanResourcePayload> weight_copies(weights.begin(),
                                                     weights.end());
    impl = std::make_unique<Impl>(
      plan, options, std::move(kernel_copies), std::move(weight_copies));
    impl->initialize();
    impl->releasePackageViews();
    return VulkanRuntimeContext(std::move(impl));
  } catch (const RuntimeFailure& error) {
    VulkanRuntimeError failure = error.get_error();
    cleanup_failed_create(failure.diagnostic);
    return std::unexpected(std::move(failure));
  } catch (const std::bad_alloc&) {
    VulkanRuntimeError failure{
      .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
      .diagnostic = "host allocation failed while creating Vulkan context"};
    cleanup_failed_create(failure.diagnostic);
    return std::unexpected(std::move(failure));
  } catch (const std::exception& error) {
    VulkanRuntimeError failure{.status = NCNN_VK_STATUS_API_ERROR,
                               .diagnostic = error.what()};
    cleanup_failed_create(failure.diagnostic);
    return std::unexpected(std::move(failure));
  }
}

std::expected<void, VulkanRuntimeError> VulkanRuntimeContext::run(
  std::span<const VulkanInputTensor> inputs,
  std::span<const VulkanOutputTensor> outputs) {
  if (!impl_) {
    return std::unexpected(
      VulkanRuntimeError{.status = NCNN_VK_STATUS_INVALID_ARGUMENT,
                         .diagnostic = "Vulkan context is empty"});
  }
  return impl_->run(inputs, outputs);
}

std::expected<void, VulkanRuntimeError> VulkanRuntimeContext::destroy() {
  if (!impl_) {
    return {};
  }
  ncnn_vk_status status = impl_->destroy();
  const std::uint32_t validation_errors = impl_->get_validation_error_count();
  impl_.reset();
  if (status == NCNN_VK_STATUS_SUCCESS && validation_errors != 0) {
    status = NCNN_VK_STATUS_API_ERROR;
  }
  if (status != NCNN_VK_STATUS_SUCCESS) {
    std::string diagnostic = "Vulkan context cleanup failed: " +
                             std::string(getVulkanRuntimeStatusName(status));
    if (validation_errors != 0) {
      diagnostic += "; validation errors=" + std::to_string(validation_errors);
    }
    return std::unexpected(VulkanRuntimeError{
      .status = status, .diagnostic = std::move(diagnostic)});
  }
  return {};
}

VulkanRuntimeDeviceInfo VulkanRuntimeContext::get_device_info() const {
  return impl_ ? impl_->get_device_info()
               : VulkanRuntimeDeviceInfo{.name = "",
                                         .api_version = 0,
                                         .driver_version = 0,
                                         .vendor_id = 0,
                                         .device_id = 0};
}

std::uint64_t VulkanRuntimeContext::get_peak_memory_bytes() const noexcept {
  return impl_ ? impl_->get_peak_memory_bytes() : 0;
}

std::uint64_t VulkanRuntimeContext::get_weight_upload_count() const noexcept {
  return impl_ ? impl_->get_weight_upload_count() : 0;
}

std::uint32_t VulkanRuntimeContext::get_validation_error_count()
  const noexcept {
  return impl_ ? impl_->get_validation_error_count() : 0;
}

VulkanRuntimeOptions makeVulkanRuntimeOptions() noexcept {
  return VulkanRuntimeOptions{
    .device_index = NCNN_VK_DEVICE_AUTO,
    .max_context_memory_bytes = kDefaultContextMemoryBytes,
    .device_name_match = "",
    .enable_validation = false,
    .force_mapped_memory_sync = false,
    .failure_injector = {}};
}

std::string_view getVulkanRuntimeStatusName(ncnn_vk_status status) noexcept {
  switch (status) {
    case NCNN_VK_STATUS_SUCCESS:
      return "success";
    case NCNN_VK_STATUS_INVALID_ARGUMENT:
      return "invalid_argument";
    case NCNN_VK_STATUS_UNSUPPORTED_DEVICE:
      return "unsupported_device";
    case NCNN_VK_STATUS_OUT_OF_MEMORY:
      return "out_of_memory";
    case NCNN_VK_STATUS_DEVICE_LOST:
      return "device_lost";
    case NCNN_VK_STATUS_API_ERROR:
      return "api_error";
    case NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH:
      return "buffer_size_mismatch";
    default:
      return "unknown_status";
  }
}

}  // namespace ncnn_mlir::vulkan
