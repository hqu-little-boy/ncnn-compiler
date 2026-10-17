#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ncnn_mlir::vulkan {
namespace {

class HelperFailure final : public std::exception {
 public:
  explicit HelperFailure(VulkanDeviceError error) : error_(std::move(error)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return error_.diagnostic.c_str();
  }

  [[nodiscard]] const VulkanDeviceError& get_error() const noexcept {
    return error_;
  }

 private:
  VulkanDeviceError error_;
};

[[noreturn]] void fail(ncnn_vk_status status, std::string diagnostic) {
  throw HelperFailure(
    VulkanDeviceError{.status = status, .diagnostic = std::move(diagnostic)});
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

std::string_view operationName(VulkanDeviceOperation operation) noexcept {
  switch (operation) {
    case VulkanDeviceOperation::CreateInstance:
      return "instance creation";
    case VulkanDeviceOperation::SelectDevice:
      return "device selection";
    case VulkanDeviceOperation::CreateDevice:
      return "device creation";
    case VulkanDeviceOperation::CreateBuffer:
      return "buffer creation";
    case VulkanDeviceOperation::AllocateMemory:
      return "memory allocation";
    case VulkanDeviceOperation::CreateDescriptor:
      return "descriptor creation";
    case VulkanDeviceOperation::CreateShaderModule:
      return "shader module creation";
    case VulkanDeviceOperation::CreatePipeline:
      return "pipeline creation";
    case VulkanDeviceOperation::CreateCommandResources:
      return "command resource creation";
    case VulkanDeviceOperation::Submit:
      return "queue submission";
    case VulkanDeviceOperation::Wait:
      return "queue wait";
    case VulkanDeviceOperation::FlushMappedMemory:
      return "mapped-memory flush";
    case VulkanDeviceOperation::InvalidateMappedMemory:
      return "mapped-memory invalidate";
  }
  return "unknown operation";
}

void inject(const detail::VulkanOperationHook& hook,
            VulkanDeviceOperation operation) {
  if (!hook) {
    return;
  }
  const std::optional<ncnn_vk_status> status = hook(operation);
  if (status) {
    fail(*status,
         "injected Vulkan runtime failure at " +
           std::string(operationName(operation)));
  }
}

template <typename Structure>
Structure makeVkStructure(VkStructureType type) {
  Structure structure{};
  structure.sType = type;
  return structure;
}

template <typename Result, typename Callable>
std::expected<Result, VulkanDeviceError> capture(Callable&& callable) {
  try {
    return std::forward<Callable>(callable)();
  } catch (const HelperFailure& error) {
    return std::unexpected(error.get_error());
  } catch (const std::bad_alloc&) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
      .diagnostic = "host allocation failed in Vulkan device helper"});
  } catch (const std::exception& error) {
    return std::unexpected(VulkanDeviceError{.status = NCNN_VK_STATUS_API_ERROR,
                                             .diagnostic = error.what()});
  } catch (...) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_API_ERROR,
      .diagnostic = "unknown failure in Vulkan device helper"});
  }
}

template <typename Callable>
std::expected<void, VulkanDeviceError> captureVoid(Callable&& callable) {
  try {
    std::forward<Callable>(callable)();
    return {};
  } catch (const HelperFailure& error) {
    return std::unexpected(error.get_error());
  } catch (const std::bad_alloc&) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
      .diagnostic = "host allocation failed in Vulkan device helper"});
  } catch (const std::exception& error) {
    return std::unexpected(VulkanDeviceError{.status = NCNN_VK_STATUS_API_ERROR,
                                             .diagnostic = error.what()});
  } catch (...) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_API_ERROR,
      .diagnostic = "unknown failure in Vulkan device helper"});
  }
}

VkBool32 VKAPI_CALL
validationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                   VkDebugUtilsMessageTypeFlagsEXT,
                   const VkDebugUtilsMessengerCallbackDataEXT*,
                   void* user_data) {
  auto* errors = static_cast<std::atomic<std::uint32_t>*>(user_data);
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    errors->fetch_add(1);
  }
  return VK_FALSE;
}

bool addChecked(std::uint64_t lhs,
                std::uint64_t rhs,
                std::uint64_t& result) noexcept {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

const VkPhysicalDeviceProperties& emptyProperties() noexcept {
  static const VkPhysicalDeviceProperties properties{};
  return properties;
}

const VkPhysicalDeviceMemoryProperties& emptyMemoryProperties() noexcept {
  static const VkPhysicalDeviceMemoryProperties properties{};
  return properties;
}

}  // namespace

std::expected<void, VulkanDeviceError> validateDeviceLimits(
  const VkPhysicalDeviceProperties& properties,
  std::span<const VulkanDeviceLimitRequirements> requirements) {
  if (properties.limits.maxStorageBufferRange == 0 ||
      properties.limits.maxPushConstantsSize == 0) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
      .diagnostic = "device lacks required storage or push-constant limits"});
  }
  for (const VulkanDeviceLimitRequirements& required : requirements) {
    if (required.local_size_x == 0 ||
        required.local_size_x > properties.limits.maxComputeWorkGroupSize[0] ||
        required.local_size_x >
          properties.limits.maxComputeWorkGroupInvocations) {
      return std::unexpected(VulkanDeviceError{
        .status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
        .diagnostic = "compute local size exceeds device limits"});
    }
    if (required.dispatch_count_x == 0 ||
        required.dispatch_count_x >
          properties.limits.maxComputeWorkGroupCount[0]) {
      return std::unexpected(VulkanDeviceError{
        .status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
        .diagnostic = "compute dispatch count exceeds device limits"});
    }
    if (required.storage_buffer_range >
        properties.limits.maxStorageBufferRange) {
      return std::unexpected(VulkanDeviceError{
        .status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
        .diagnostic = "storage buffer range exceeds device limits"});
    }
    if (required.push_constant_bytes > properties.limits.maxPushConstantsSize) {
      return std::unexpected(
        VulkanDeviceError{.status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
                          .diagnostic = "push constants exceed device limits"});
    }
    if (required.descriptor_set_count >
          properties.limits.maxBoundDescriptorSets ||
        required.storage_buffer_descriptor_count >
          properties.limits.maxDescriptorSetStorageBuffers) {
      return std::unexpected(VulkanDeviceError{
        .status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
        .diagnostic =
          "descriptor-set storage buffer use exceeds device limits"});
    }
    if (required.storage_buffer_descriptor_count >
          properties.limits.maxPerStageDescriptorStorageBuffers ||
        required.storage_buffer_descriptor_count >
          properties.limits.maxPerStageResources) {
      return std::unexpected(VulkanDeviceError{
        .status = NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
        .diagnostic =
          "per-stage descriptor resource use exceeds device limits"});
    }
  }
  return {};
}

std::expected<void, VulkanDeviceError> validateAllocationBudget(
  const VulkanAllocationBudget& budget,
  std::uint64_t max_memory_allocation_count,
  std::uint64_t max_context_memory_bytes) {
  std::uint64_t total_bytes = 0;
  if (!addChecked(
        budget.persistent_bytes, budget.temporary_staging_bytes, total_bytes)) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
      .diagnostic = "Vulkan allocation budget byte count overflows uint64"});
  }
  if (budget.allocation_count > max_memory_allocation_count) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
      .diagnostic =
        "Vulkan allocation budget exceeds device allocation count"});
  }
  if (max_context_memory_bytes == 0 || total_bytes > max_context_memory_bytes) {
    return std::unexpected(VulkanDeviceError{
      .status = max_context_memory_bytes == 0 ? NCNN_VK_STATUS_INVALID_ARGUMENT
                                              : NCNN_VK_STATUS_OUT_OF_MEMORY,
      .diagnostic =
        "Vulkan allocation budget exceeds configured memory limit"});
  }
  return {};
}

class VulkanDeviceContext::Impl final {
  friend class VulkanDeviceContext;

 public:
  Impl()
    : instance_(VK_NULL_HANDLE),
      messenger_(VK_NULL_HANDLE),
      physical_device_(VK_NULL_HANDLE),
      properties_(),
      memory_properties_(),
      device_(VK_NULL_HANDLE),
      queue_(VK_NULL_HANDLE),
      queue_family_index_(0),
      validation_errors_(0),
      destroyed_(false),
      destroy_status_(NCNN_VK_STATUS_SUCCESS) {}

  ~Impl() { (void)destroy(); }

  void initialize(const VulkanDeviceOptions& options,
                  const detail::VulkanOperationHook& operation_hook) {
    createInstance(options.enable_validation, operation_hook);
    selectDevice(options, operation_hook);
    createDevice(operation_hook);
  }

  void createInstance(bool enable_validation,
                      const detail::VulkanOperationHook& operation_hook) {
    inject(operation_hook, VulkanDeviceOperation::CreateInstance);
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    auto debug_info = makeVkStructure<VkDebugUtilsMessengerCreateInfoEXT>(
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
    if (enable_validation) {
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
      debug_info.pUserData = &validation_errors_;
    }

    auto application =
      makeVkStructure<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.pApplicationName = "ncnn-mlir-vulkan-device-helper";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "ncnn-mlir-vulkan-device-helper";
    application.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_1;

    auto create_info = makeVkStructure<VkInstanceCreateInfo>(
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    create_info.pNext = enable_validation ? &debug_info : nullptr;
    create_info.pApplicationInfo = &application;
    create_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();
    create_info.enabledExtensionCount =
      static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    checkVk(vkCreateInstance(&create_info, nullptr, &instance_),
            "vkCreateInstance");

    if (enable_validation) {
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

  void selectDevice(const VulkanDeviceOptions& options,
                    const detail::VulkanOperationHook& operation_hook) {
    inject(operation_hook, VulkanDeviceOperation::SelectDevice);
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
    if (options.device_index != NCNN_VK_DEVICE_AUTO &&
        options.device_index >= devices.size()) {
      fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
           "requested Vulkan device index is unavailable");
    }

    std::ostringstream summary;
    for (std::size_t index = 0; index < devices.size(); ++index) {
      if (options.device_index != NCNN_VK_DEVICE_AUTO &&
          index != options.device_index) {
        continue;
      }
      const VkPhysicalDevice candidate = devices[index];
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
      if (!options.device_name_match.empty() &&
          std::string_view(properties.deviceName)
              .find(options.device_name_match) == std::string_view::npos) {
        summary << " did not match '" << options.device_name_match << "'; ";
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
          queue_family_index_ = family;
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
           std::to_string(options.device_index) + "): " + summary.str());
  }

  void createDevice(const detail::VulkanOperationHook& operation_hook) {
    inject(operation_hook, VulkanDeviceOperation::CreateDevice);
    constexpr float priority = 1.0F;
    auto queue_info = makeVkStructure<VkDeviceQueueCreateInfo>(
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue_info.queueFamilyIndex = queue_family_index_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    auto device_info =
      makeVkStructure<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    checkVk(vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
            "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);
    if (queue_ == VK_NULL_HANDLE) {
      fail(NCNN_VK_STATUS_API_ERROR, "vkGetDeviceQueue returned a null queue");
    }
  }

  ncnn_vk_status destroy() noexcept {
    if (destroyed_) {
      return destroy_status_;
    }
    destroyed_ = true;
    if (device_ != VK_NULL_HANDLE) {
      const VkResult result = vkDeviceWaitIdle(device_);
      if (result != VK_SUCCESS) {
        destroy_status_ = mapVkResult(result);
      }
      vkDestroyDevice(device_, nullptr);
      device_ = VK_NULL_HANDLE;
      queue_ = VK_NULL_HANDLE;
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

 private:
  VkInstance instance_;
  VkDebugUtilsMessengerEXT messenger_;
  VkPhysicalDevice physical_device_;
  VkPhysicalDeviceProperties properties_;
  VkPhysicalDeviceMemoryProperties memory_properties_;
  VkDevice device_;
  VkQueue queue_;
  std::uint32_t queue_family_index_;
  std::atomic<std::uint32_t> validation_errors_;
  bool destroyed_;
  ncnn_vk_status destroy_status_;
};

VulkanDeviceContext::VulkanDeviceContext(std::unique_ptr<Impl> impl) noexcept
  : impl_(std::move(impl)) {}

VulkanDeviceContext::VulkanDeviceContext(VulkanDeviceContext&& other) noexcept =
  default;

VulkanDeviceContext& VulkanDeviceContext::operator=(
  VulkanDeviceContext&& other) noexcept = default;

VulkanDeviceContext::~VulkanDeviceContext() = default;

std::expected<VulkanDeviceContext, VulkanDeviceError>
VulkanDeviceContext::create(const VulkanDeviceOptions& options,
                            const detail::VulkanOperationHook& operation_hook) {
  return capture<VulkanDeviceContext>([&]() {
    auto impl = std::make_unique<Impl>();
    impl->initialize(options, operation_hook);
    return VulkanDeviceContext(std::move(impl));
  });
}

std::expected<void, VulkanDeviceError> VulkanDeviceContext::destroy() {
  if (!impl_) {
    return {};
  }
  const ncnn_vk_status status = impl_->destroy();
  if (status != NCNN_VK_STATUS_SUCCESS) {
    return std::unexpected(VulkanDeviceError{
      .status = status,
      .diagnostic = "vkDeviceWaitIdle failed during Vulkan device cleanup"});
  }
  if (impl_->validation_errors_.load() != 0) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_API_ERROR,
      .diagnostic =
        "Vulkan validation reported errors during device lifetime"});
  }
  return {};
}

VkInstance VulkanDeviceContext::get_instance() const noexcept {
  return impl_ ? impl_->instance_ : VK_NULL_HANDLE;
}

VkPhysicalDevice VulkanDeviceContext::get_physical_device() const noexcept {
  return impl_ ? impl_->physical_device_ : VK_NULL_HANDLE;
}

VkDevice VulkanDeviceContext::get_device() const noexcept {
  return impl_ ? impl_->device_ : VK_NULL_HANDLE;
}

VkQueue VulkanDeviceContext::get_queue() const noexcept {
  return impl_ ? impl_->queue_ : VK_NULL_HANDLE;
}

std::uint32_t VulkanDeviceContext::get_queue_family_index() const noexcept {
  return impl_ ? impl_->queue_family_index_ : 0;
}

const VkPhysicalDeviceProperties& VulkanDeviceContext::get_properties()
  const noexcept {
  return impl_ ? impl_->properties_ : emptyProperties();
}

const VkPhysicalDeviceMemoryProperties&
VulkanDeviceContext::get_memory_properties() const noexcept {
  return impl_ ? impl_->memory_properties_ : emptyMemoryProperties();
}

VulkanDeviceInfo VulkanDeviceContext::get_device_info() const {
  if (!impl_) {
    return VulkanDeviceInfo{.name = "",
                            .api_version = 0,
                            .driver_version = 0,
                            .vendor_id = 0,
                            .device_id = 0};
  }
  return VulkanDeviceInfo{.name = impl_->properties_.deviceName,
                          .api_version = impl_->properties_.apiVersion,
                          .driver_version = impl_->properties_.driverVersion,
                          .vendor_id = impl_->properties_.vendorID,
                          .device_id = impl_->properties_.deviceID};
}

std::uint32_t VulkanDeviceContext::get_validation_error_count() const noexcept {
  return impl_ ? impl_->validation_errors_.load() : 0;
}

std::expected<VulkanMemoryTypeSelection, VulkanDeviceError>
VulkanDeviceContext::get_memory_type(std::uint32_t allowed_types,
                                     VkMemoryPropertyFlags required,
                                     VkMemoryPropertyFlags preferred,
                                     bool prefer_non_coherent) const {
  if (!impl_) {
    return std::unexpected(
      VulkanDeviceError{.status = NCNN_VK_STATUS_INVALID_ARGUMENT,
                        .diagnostic = "Vulkan device context is empty"});
  }
  if (prefer_non_coherent &&
      (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    for (std::uint32_t index = 0;
         index < impl_->memory_properties_.memoryTypeCount;
         ++index) {
      if ((allowed_types & (1U << index)) == 0) {
        continue;
      }
      const VkMemoryPropertyFlags flags =
        impl_->memory_properties_.memoryTypes[index].propertyFlags;
      if ((flags & required) == required &&
          (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
        return VulkanMemoryTypeSelection{.index = index, .is_coherent = false};
      }
    }
  }
  std::optional<VulkanMemoryTypeSelection> fallback;
  for (std::uint32_t index = 0;
       index < impl_->memory_properties_.memoryTypeCount;
       ++index) {
    if ((allowed_types & (1U << index)) == 0) {
      continue;
    }
    const VkMemoryPropertyFlags flags =
      impl_->memory_properties_.memoryTypes[index].propertyFlags;
    if ((flags & required) != required) {
      continue;
    }
    const VulkanMemoryTypeSelection selection{
      .index = index,
      .is_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0};
    if (!fallback) {
      fallback = selection;
    }
    if ((flags & preferred) == preferred) {
      return selection;
    }
  }
  if (fallback) {
    return *fallback;
  }
  return std::unexpected(VulkanDeviceError{
    .status = NCNN_VK_STATUS_OUT_OF_MEMORY,
    .diagnostic = "no Vulkan memory type satisfies buffer requirements"});
}

bool VulkanDeviceContext::has_memory_type(
  std::uint32_t allowed_types, VkMemoryPropertyFlags required) const noexcept {
  if (!impl_) {
    return false;
  }
  for (std::uint32_t index = 0;
       index < impl_->memory_properties_.memoryTypeCount;
       ++index) {
    if ((allowed_types & (1U << index)) != 0 &&
        (impl_->memory_properties_.memoryTypes[index].propertyFlags &
         required) == required) {
      return true;
    }
  }
  return false;
}

std::expected<void, VulkanDeviceError> VulkanDeviceContext::wait_idle() {
  if (!impl_ || impl_->device_ == VK_NULL_HANDLE) {
    return std::unexpected(VulkanDeviceError{
      .status = NCNN_VK_STATUS_INVALID_ARGUMENT,
      .diagnostic = "Vulkan device context is not initialized"});
  }
  const VkResult result = vkDeviceWaitIdle(impl_->device_);
  if (result != VK_SUCCESS) {
    return std::unexpected(VulkanDeviceError{
      .status = mapVkResult(result),
      .diagnostic =
        "vkDeviceWaitIdle failed with VkResult " + std::to_string(result)});
  }
  return {};
}

class VulkanBuffer::Impl final {
  friend class VulkanBuffer;

 public:
  Impl(VkDevice device,
       VkBuffer handle,
       VkDeviceSize logical_size,
       VkDeviceSize allocation_size,
       std::uint32_t memory_type_index,
       bool coherent,
       bool force_mapped_memory_sync)
    : device_(device),
      handle_(handle),
      memory_(VK_NULL_HANDLE),
      logical_size_(logical_size),
      allocation_size_(allocation_size),
      memory_type_index_(memory_type_index),
      mapped_(nullptr),
      coherent_(coherent),
      force_mapped_memory_sync_(force_mapped_memory_sync),
      prepared_(true) {}

  ~Impl() { destroy(); }

  void destroy() noexcept {
    if (device_ == VK_NULL_HANDLE) {
      return;
    }
    if (mapped_ != nullptr && memory_ != VK_NULL_HANDLE) {
      vkUnmapMemory(device_, memory_);
      mapped_ = nullptr;
    }
    if (handle_ != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, handle_, nullptr);
      handle_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory_, nullptr);
      memory_ = VK_NULL_HANDLE;
    }
  }

 private:
  VkDevice device_;
  VkBuffer handle_;
  VkDeviceMemory memory_;
  VkDeviceSize logical_size_;
  VkDeviceSize allocation_size_;
  std::uint32_t memory_type_index_;
  void* mapped_;
  bool coherent_;
  bool force_mapped_memory_sync_;
  bool prepared_;
};

VulkanBuffer::VulkanBuffer(std::unique_ptr<Impl> impl) noexcept
  : impl_(std::move(impl)) {}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept = default;
VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept = default;
VulkanBuffer::~VulkanBuffer() = default;

std::expected<VulkanBuffer, VulkanDeviceError> VulkanBuffer::prepare(
  VulkanDeviceContext& context,
  const VulkanBufferCreateInfo& create_info,
  const detail::VulkanOperationHook& operation_hook) {
  return capture<VulkanBuffer>([&]() {
    if (context.get_device() == VK_NULL_HANDLE || create_info.size == 0 ||
        create_info.size > kVulkanDeviceHelperMaxBufferBytes) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "invalid Vulkan buffer size or device context");
    }
    if (create_info.size >
        context.get_properties().limits.maxStorageBufferRange) {
      fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
           "Vulkan buffer exceeds the selected device storage-buffer limit");
    }
    inject(operation_hook, VulkanDeviceOperation::CreateBuffer);
    auto buffer_info =
      makeVkStructure<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    buffer_info.size = create_info.size;
    buffer_info.usage = create_info.usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer handle = VK_NULL_HANDLE;
    checkVk(
      vkCreateBuffer(context.get_device(), &buffer_info, nullptr, &handle),
      "vkCreateBuffer");
    try {
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(
        context.get_device(), handle, &requirements);
      const VkMemoryPropertyFlags required =
        create_info.host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      const VkMemoryPropertyFlags preferred =
        create_info.host_visible ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      auto memory_type =
        context.get_memory_type(requirements.memoryTypeBits,
                                required,
                                preferred,
                                create_info.force_mapped_memory_sync);
      if (!memory_type) {
        fail(memory_type.error().status, memory_type.error().diagnostic);
      }
      auto impl = std::make_unique<Impl>(context.get_device(),
                                         handle,
                                         create_info.size,
                                         requirements.size,
                                         memory_type->index,
                                         memory_type->is_coherent,
                                         create_info.force_mapped_memory_sync);
      return VulkanBuffer(std::move(impl));
    } catch (...) {
      vkDestroyBuffer(context.get_device(), handle, nullptr);
      throw;
    }
  });
}

std::expected<void, VulkanDeviceError> VulkanBuffer::allocate_and_bind(
  bool map, const detail::VulkanOperationHook& operation_hook) {
  return captureVoid([&]() {
    if (!impl_ || !impl_->prepared_ || impl_->handle_ == VK_NULL_HANDLE ||
        impl_->memory_ != VK_NULL_HANDLE) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan buffer is not prepared for allocation");
    }
    inject(operation_hook, VulkanDeviceOperation::AllocateMemory);
    auto allocation = makeVkStructure<VkMemoryAllocateInfo>(
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = impl_->allocation_size_;
    allocation.memoryTypeIndex = impl_->memory_type_index_;
    checkVk(
      vkAllocateMemory(impl_->device_, &allocation, nullptr, &impl_->memory_),
      "vkAllocateMemory");
    checkVk(
      vkBindBufferMemory(impl_->device_, impl_->handle_, impl_->memory_, 0),
      "vkBindBufferMemory");
    if (map) {
      checkVk(
        vkMapMemory(
          impl_->device_, impl_->memory_, 0, VK_WHOLE_SIZE, 0, &impl_->mapped_),
        "vkMapMemory");
    }
  });
}

std::expected<void, VulkanDeviceError> VulkanBuffer::flush(
  const detail::VulkanOperationHook& operation_hook) const {
  return captureVoid([&]() {
    if (!impl_ || impl_->mapped_ == nullptr) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan buffer is not mapped for flush");
    }
    if (impl_->coherent_ && !impl_->force_mapped_memory_sync_) {
      return;
    }
    inject(operation_hook, VulkanDeviceOperation::FlushMappedMemory);
    auto range = makeVkStructure<VkMappedMemoryRange>(
      VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = impl_->memory_;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    checkVk(vkFlushMappedMemoryRanges(impl_->device_, 1, &range),
            "vkFlushMappedMemoryRanges");
  });
}

std::expected<void, VulkanDeviceError> VulkanBuffer::invalidate(
  const detail::VulkanOperationHook& operation_hook) const {
  return captureVoid([&]() {
    if (!impl_ || impl_->mapped_ == nullptr) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan buffer is not mapped for invalidate");
    }
    if (impl_->coherent_ && !impl_->force_mapped_memory_sync_) {
      return;
    }
    inject(operation_hook, VulkanDeviceOperation::InvalidateMappedMemory);
    auto range = makeVkStructure<VkMappedMemoryRange>(
      VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
    range.memory = impl_->memory_;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    checkVk(vkInvalidateMappedMemoryRanges(impl_->device_, 1, &range),
            "vkInvalidateMappedMemoryRanges");
  });
}

std::expected<void, VulkanDeviceError> VulkanBuffer::destroy() noexcept {
  if (impl_) {
    impl_->destroy();
    impl_.reset();
  }
  return {};
}

VkBuffer VulkanBuffer::get_handle() const noexcept {
  return impl_ ? impl_->handle_ : VK_NULL_HANDLE;
}

VkDeviceMemory VulkanBuffer::get_memory() const noexcept {
  return impl_ ? impl_->memory_ : VK_NULL_HANDLE;
}

VkDeviceSize VulkanBuffer::get_logical_size() const noexcept {
  return impl_ ? impl_->logical_size_ : 0;
}

VkDeviceSize VulkanBuffer::get_allocation_size() const noexcept {
  return impl_ ? impl_->allocation_size_ : 0;
}

std::uint32_t VulkanBuffer::get_memory_type_index() const noexcept {
  return impl_ ? impl_->memory_type_index_ : 0;
}

std::span<std::byte> VulkanBuffer::get_mapped() const noexcept {
  if (!impl_ || impl_->mapped_ == nullptr) {
    return {};
  }
  if (impl_->logical_size_ > std::numeric_limits<std::size_t>::max()) {
    return {};
  }
  return {static_cast<std::byte*>(impl_->mapped_),
          static_cast<std::size_t>(impl_->logical_size_)};
}

bool VulkanBuffer::is_coherent() const noexcept {
  return impl_ && impl_->coherent_;
}

bool VulkanBuffer::is_prepared() const noexcept {
  return impl_ && impl_->prepared_;
}

bool VulkanBuffer::is_allocated() const noexcept {
  return impl_ && impl_->memory_ != VK_NULL_HANDLE;
}

class VulkanCommandResources::Impl final {
  friend class VulkanCommandResources;

 public:
  explicit Impl(VkDevice device,
                VkQueue queue,
                std::uint32_t queue_family_index)
    : device_(device),
      queue_(queue),
      queue_family_index_(queue_family_index),
      command_pool_(VK_NULL_HANDLE),
      command_buffer_(VK_NULL_HANDLE),
      fence_(VK_NULL_HANDLE),
      submission_pending_(false) {}

  ~Impl() { (void)destroy(); }

  ncnn_vk_status destroy() noexcept {
    if (device_ == VK_NULL_HANDLE) {
      return NCNN_VK_STATUS_SUCCESS;
    }
    ncnn_vk_status status = NCNN_VK_STATUS_SUCCESS;
    if (submission_pending_) {
      const VkResult wait_result = vkDeviceWaitIdle(device_);
      if (wait_result != VK_SUCCESS) {
        status = mapVkResult(wait_result);
      }
      submission_pending_ = false;
    }
    if (fence_ != VK_NULL_HANDLE) {
      vkDestroyFence(device_, fence_, nullptr);
      fence_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
      command_pool_ = VK_NULL_HANDLE;
      command_buffer_ = VK_NULL_HANDLE;
    }
    return status;
  }

 private:
  VkDevice device_;
  VkQueue queue_;
  std::uint32_t queue_family_index_;
  VkCommandPool command_pool_;
  VkCommandBuffer command_buffer_;
  VkFence fence_;
  bool submission_pending_;
};

VulkanCommandResources::VulkanCommandResources(
  std::unique_ptr<Impl> impl) noexcept
  : impl_(std::move(impl)) {}

VulkanCommandResources::VulkanCommandResources(
  VulkanCommandResources&& other) noexcept = default;
VulkanCommandResources& VulkanCommandResources::operator=(
  VulkanCommandResources&& other) noexcept = default;
VulkanCommandResources::~VulkanCommandResources() = default;

std::expected<VulkanCommandResources, VulkanDeviceError>
VulkanCommandResources::create(
  VulkanDeviceContext& context,
  const detail::VulkanOperationHook& operation_hook) {
  return capture<VulkanCommandResources>([&]() {
    if (context.get_device() == VK_NULL_HANDLE ||
        context.get_queue() == VK_NULL_HANDLE) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan command resources need an initialized device");
    }
    auto impl = std::make_unique<Impl>(context.get_device(),
                                       context.get_queue(),
                                       context.get_queue_family_index());
    auto pool_info = makeVkStructure<VkCommandPoolCreateInfo>(
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = impl->queue_family_index_;
    inject(operation_hook, VulkanDeviceOperation::CreateCommandResources);
    checkVk(vkCreateCommandPool(
              impl->device_, &pool_info, nullptr, &impl->command_pool_),
            "vkCreateCommandPool");
    auto command_info = makeVkStructure<VkCommandBufferAllocateInfo>(
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = impl->command_pool_;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    inject(operation_hook, VulkanDeviceOperation::CreateCommandResources);
    checkVk(vkAllocateCommandBuffers(
              impl->device_, &command_info, &impl->command_buffer_),
            "vkAllocateCommandBuffers");
    auto fence_info =
      makeVkStructure<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    inject(operation_hook, VulkanDeviceOperation::CreateCommandResources);
    checkVk(vkCreateFence(impl->device_, &fence_info, nullptr, &impl->fence_),
            "vkCreateFence");
    return VulkanCommandResources(std::move(impl));
  });
}

std::expected<void, VulkanDeviceError> VulkanCommandResources::reset() {
  return captureVoid([&]() {
    if (!impl_ || impl_->submission_pending_) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan command resources are empty or still in flight");
    }
    checkVk(vkResetFences(impl_->device_, 1, &impl_->fence_), "vkResetFences");
    checkVk(vkResetCommandPool(impl_->device_, impl_->command_pool_, 0),
            "vkResetCommandPool");
  });
}

std::expected<void, VulkanDeviceError> VulkanCommandResources::begin() {
  return captureVoid([&]() {
    if (!impl_) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan command resources are empty");
    }
    auto begin_info = makeVkStructure<VkCommandBufferBeginInfo>(
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(impl_->command_buffer_, &begin_info),
            "vkBeginCommandBuffer");
  });
}

std::expected<void, VulkanDeviceError> VulkanCommandResources::end() {
  return captureVoid([&]() {
    if (!impl_) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan command resources are empty");
    }
    checkVk(vkEndCommandBuffer(impl_->command_buffer_), "vkEndCommandBuffer");
  });
}

std::expected<void, VulkanDeviceError> VulkanCommandResources::submit_and_wait(
  const detail::VulkanOperationHook& operation_hook) {
  return captureVoid([&]() {
    if (!impl_) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan command resources are empty");
    }
    inject(operation_hook, VulkanDeviceOperation::Submit);
    auto submit = makeVkStructure<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &impl_->command_buffer_;
    checkVk(vkQueueSubmit(impl_->queue_, 1, &submit, impl_->fence_),
            "vkQueueSubmit");
    impl_->submission_pending_ = true;
    inject(operation_hook, VulkanDeviceOperation::Wait);
    checkVk(
      vkWaitForFences(impl_->device_, 1, &impl_->fence_, VK_TRUE, UINT64_MAX),
      "vkWaitForFences");
    impl_->submission_pending_ = false;
  });
}

std::expected<void, VulkanDeviceError> VulkanCommandResources::wait(
  const detail::VulkanOperationHook& operation_hook) {
  return captureVoid([&]() {
    if (!impl_) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "Vulkan command resources are empty");
    }
    inject(operation_hook, VulkanDeviceOperation::Wait);
    checkVk(
      vkWaitForFences(impl_->device_, 1, &impl_->fence_, VK_TRUE, UINT64_MAX),
      "vkWaitForFences");
    impl_->submission_pending_ = false;
  });
}

std::expected<void, VulkanDeviceError> VulkanCommandResources::destroy() {
  if (!impl_) {
    return {};
  }
  const ncnn_vk_status status = impl_->destroy();
  impl_.reset();
  if (status != NCNN_VK_STATUS_SUCCESS) {
    return std::unexpected(VulkanDeviceError{
      .status = status,
      .diagnostic = "Vulkan command resources cleanup wait failed"});
  }
  return {};
}

VkCommandPool VulkanCommandResources::get_command_pool() const noexcept {
  return impl_ ? impl_->command_pool_ : VK_NULL_HANDLE;
}

VkCommandBuffer VulkanCommandResources::get_command_buffer() const noexcept {
  return impl_ ? impl_->command_buffer_ : VK_NULL_HANDLE;
}

VkFence VulkanCommandResources::get_fence() const noexcept {
  return impl_ ? impl_->fence_ : VK_NULL_HANDLE;
}

class VulkanDescriptorPool::Impl final {
  friend class VulkanDescriptorPool;

 public:
  Impl(VkDevice device,
       std::uint32_t max_sets,
       std::uint32_t storage_buffer_descriptor_count)
    : device_(device),
      handle_(VK_NULL_HANDLE),
      max_sets_(max_sets),
      storage_buffer_descriptor_count_(storage_buffer_descriptor_count),
      set_layouts_() {}

  ~Impl() { destroy(); }

  [[nodiscard]] bool can_add_set_layouts(std::uint32_t count) const noexcept {
    return set_layouts_.size() <= max_sets_ &&
           count <= max_sets_ - set_layouts_.size();
  }

  void reserve_next_set_layout() {
    set_layouts_.reserve(set_layouts_.size() + 1);
  }

  void add_set_layout(VkDescriptorSetLayout layout) {
    try {
      set_layouts_.push_back(layout);
    } catch (...) {
      vkDestroyDescriptorSetLayout(device_, layout, nullptr);
      throw;
    }
  }

  void destroy() noexcept {
    if (device_ == VK_NULL_HANDLE) {
      return;
    }
    if (handle_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, handle_, nullptr);
      handle_ = VK_NULL_HANDLE;
    }
    for (VkDescriptorSetLayout layout : set_layouts_) {
      if (layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout, nullptr);
      }
    }
    set_layouts_.clear();
  }

 private:
  VkDevice device_;
  VkDescriptorPool handle_;
  std::uint32_t max_sets_;
  std::uint32_t storage_buffer_descriptor_count_;
  std::vector<VkDescriptorSetLayout> set_layouts_;
};

VulkanDescriptorPool::VulkanDescriptorPool(std::unique_ptr<Impl> impl) noexcept
  : impl_(std::move(impl)) {}

VulkanDescriptorPool::VulkanDescriptorPool(
  VulkanDescriptorPool&& other) noexcept = default;
VulkanDescriptorPool& VulkanDescriptorPool::operator=(
  VulkanDescriptorPool&& other) noexcept = default;
VulkanDescriptorPool::~VulkanDescriptorPool() = default;

std::expected<VulkanDescriptorPool, VulkanDeviceError>
VulkanDescriptorPool::create(
  VulkanDeviceContext& context,
  std::uint32_t max_sets,
  std::uint32_t storage_buffer_descriptor_count,
  const detail::VulkanOperationHook& operation_hook) {
  return capture<VulkanDescriptorPool>([&]() {
    if (context.get_device() == VK_NULL_HANDLE ||
        ((max_sets == 0) != (storage_buffer_descriptor_count == 0))) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "invalid Vulkan descriptor-pool device or capacity");
    }
    auto impl = std::make_unique<Impl>(
      context.get_device(), max_sets, storage_buffer_descriptor_count);
    if (max_sets != 0) {
      const VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = storage_buffer_descriptor_count};
      auto pool_info = makeVkStructure<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
      pool_info.maxSets = max_sets;
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes = &pool_size;
      inject(operation_hook, VulkanDeviceOperation::CreateDescriptor);
      checkVk(vkCreateDescriptorPool(
                impl->device_, &pool_info, nullptr, &impl->handle_),
              "vkCreateDescriptorPool");
    }
    return VulkanDescriptorPool(std::move(impl));
  });
}

std::expected<void, VulkanDeviceError>
VulkanDescriptorPool::destroy() noexcept {
  if (impl_) {
    impl_->destroy();
    impl_.reset();
  }
  return {};
}

VkDescriptorPool VulkanDescriptorPool::get_handle() const noexcept {
  return impl_ ? impl_->handle_ : VK_NULL_HANDLE;
}

std::uint32_t VulkanDescriptorPool::get_max_sets() const noexcept {
  return impl_ ? impl_->max_sets_ : 0;
}

std::uint32_t VulkanDescriptorPool::get_storage_buffer_descriptor_count()
  const noexcept {
  return impl_ ? impl_->storage_buffer_descriptor_count_ : 0;
}

bool VulkanDescriptorPool::is_for(
  const VulkanDeviceContext& context) const noexcept {
  return impl_ && impl_->device_ == context.get_device();
}

class VulkanComputePipeline::Impl final {
  friend class VulkanComputePipeline;

 public:
  explicit Impl(VkDevice device)
    : device_(device),
      module_(VK_NULL_HANDLE),
      set_layouts_(),
      descriptor_sets_(),
      layout_(VK_NULL_HANDLE),
      pipeline_(VK_NULL_HANDLE) {}

  ~Impl() { destroy(); }

  void destroy() noexcept {
    if (device_ == VK_NULL_HANDLE) {
      return;
    }
    if (pipeline_ != VK_NULL_HANDLE) {
      vkDestroyPipeline(device_, pipeline_, nullptr);
      pipeline_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device_, layout_, nullptr);
      layout_ = VK_NULL_HANDLE;
    }
    if (module_ != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device_, module_, nullptr);
      module_ = VK_NULL_HANDLE;
    }
    set_layouts_.clear();
    descriptor_sets_.clear();
  }

 private:
  VkDevice device_;
  VkShaderModule module_;
  std::vector<VkDescriptorSetLayout> set_layouts_;
  std::vector<VkDescriptorSet> descriptor_sets_;
  VkPipelineLayout layout_;
  VkPipeline pipeline_;
};

VulkanComputePipeline::VulkanComputePipeline(
  std::unique_ptr<Impl> impl) noexcept
  : impl_(std::move(impl)) {}

VulkanComputePipeline::VulkanComputePipeline(
  VulkanComputePipeline&& other) noexcept = default;
VulkanComputePipeline& VulkanComputePipeline::operator=(
  VulkanComputePipeline&& other) noexcept = default;
VulkanComputePipeline::~VulkanComputePipeline() = default;

std::expected<VulkanComputePipeline, VulkanDeviceError>
VulkanComputePipeline::create(
  VulkanDeviceContext& context,
  VulkanDescriptorPool& descriptor_pool,
  std::span<const std::uint32_t> spirv_words,
  std::string_view entry_point,
  std::span<const VulkanPipelineBinding> bindings,
  std::span<const VulkanPushConstantRange> push_constants,
  const detail::VulkanOperationHook& operation_hook) {
  return capture<VulkanComputePipeline>([&]() {
    if (context.get_device() == VK_NULL_HANDLE || spirv_words.size() < 5 ||
        spirv_words.front() != 0x07230203U || entry_point.empty() ||
        entry_point.find('\0') != std::string_view::npos) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "invalid device, SPIR-V module, or compute entry point");
    }

    const VkPhysicalDeviceProperties& properties = context.get_properties();
    if (bindings.size() > std::numeric_limits<std::uint32_t>::max()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "compute pipeline descriptor count exceeds Vulkan limits");
    }
    std::uint32_t max_set = 0;
    std::uint64_t max_storage_range = 0;
    std::unordered_set<std::uint64_t> binding_keys;
    for (const VulkanPipelineBinding& binding : bindings) {
      if (binding.buffer == VK_NULL_HANDLE || binding.buffer_size == 0 ||
          binding.range == 0 ||
          binding.set == std::numeric_limits<std::uint32_t>::max() ||
          binding.binding == std::numeric_limits<std::uint32_t>::max() ||
          binding.offset > binding.buffer_size ||
          binding.range > binding.buffer_size - binding.offset ||
          binding.range > kVulkanDeviceHelperMaxBufferBytes ||
          binding.range > properties.limits.maxStorageBufferRange ||
          (properties.limits.minStorageBufferOffsetAlignment != 0 &&
           binding.offset % properties.limits.minStorageBufferOffsetAlignment !=
             0)) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "compute pipeline has an invalid descriptor buffer range");
      }
      const std::uint64_t key =
        (static_cast<std::uint64_t>(binding.set) << 32U) | binding.binding;
      if (!binding_keys.insert(key).second) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "compute pipeline has duplicate descriptor bindings");
      }
      max_storage_range =
        std::max(max_storage_range, static_cast<std::uint64_t>(binding.range));
      max_set = std::max(max_set, binding.set);
    }
    const std::uint32_t set_count = bindings.empty() ? 0 : max_set + 1;
    const auto descriptor_count = static_cast<std::uint32_t>(bindings.size());
    if (!descriptor_pool.is_for(context) ||
        (descriptor_count != 0 &&
         (descriptor_pool.get_handle() == VK_NULL_HANDLE ||
          descriptor_pool.get_max_sets() < set_count ||
          descriptor_pool.get_storage_buffer_descriptor_count() <
            descriptor_count ||
          !descriptor_pool.impl_->can_add_set_layouts(set_count)))) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "compute pipeline descriptor pool is incompatible or undersized");
    }
    std::uint32_t max_push_constant_end = 0;
    for (std::size_t index = 0; index < push_constants.size(); ++index) {
      const VulkanPushConstantRange& range = push_constants[index];
      if (range.stage_flags != VK_SHADER_STAGE_COMPUTE_BIT || range.size == 0 ||
          range.offset % 4 != 0 || range.size % 4 != 0 ||
          range.size >
            std::numeric_limits<std::uint32_t>::max() - range.offset) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "compute pipeline has an invalid push-constant range");
      }
      const std::uint32_t range_end = range.offset + range.size;
      for (std::size_t prior_index = 0; prior_index < index; ++prior_index) {
        const VulkanPushConstantRange& prior = push_constants[prior_index];
        const std::uint32_t prior_end = prior.offset + prior.size;
        if (range.offset < prior_end && prior.offset < range_end) {
          fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
               "compute pipeline push-constant ranges overlap");
        }
      }
      max_push_constant_end = std::max(max_push_constant_end, range_end);
    }
    const VulkanDeviceLimitRequirements limits{
      .local_size_x = 1,
      .dispatch_count_x = 1,
      .storage_buffer_range = max_storage_range,
      .push_constant_bytes = max_push_constant_end,
      .descriptor_set_count = set_count,
      .storage_buffer_descriptor_count =
        static_cast<std::uint32_t>(bindings.size())};
    const auto validated_limits =
      validateDeviceLimits(properties, std::span(&limits, 1));
    if (!validated_limits) {
      fail(validated_limits.error().status,
           validated_limits.error().diagnostic);
    }

    auto impl = std::make_unique<Impl>(context.get_device());
    auto shader_info = makeVkStructure<VkShaderModuleCreateInfo>(
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    shader_info.codeSize = spirv_words.size_bytes();
    shader_info.pCode = spirv_words.data();
    inject(operation_hook, VulkanDeviceOperation::CreateShaderModule);
    checkVk(vkCreateShaderModule(
              impl->device_, &shader_info, nullptr, &impl->module_),
            "vkCreateShaderModule");

    if (!bindings.empty()) {
      impl->set_layouts_.resize(set_count, VK_NULL_HANDLE);
      impl->descriptor_sets_.resize(set_count, VK_NULL_HANDLE);
      for (std::uint32_t set = 0; set < set_count; ++set) {
        std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
        for (const VulkanPipelineBinding& binding : bindings) {
          if (binding.set == set) {
            layout_bindings.push_back(VkDescriptorSetLayoutBinding{
              .binding = binding.binding,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1,
              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
              .pImmutableSamplers = nullptr});
          }
        }
        auto layout_info = makeVkStructure<VkDescriptorSetLayoutCreateInfo>(
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
        layout_info.bindingCount =
          static_cast<std::uint32_t>(layout_bindings.size());
        layout_info.pBindings = layout_bindings.data();
        descriptor_pool.impl_->reserve_next_set_layout();
        inject(operation_hook, VulkanDeviceOperation::CreateDescriptor);
        VkDescriptorSetLayout layout_handle = VK_NULL_HANDLE;
        const VkResult layout_result = vkCreateDescriptorSetLayout(
          impl->device_, &layout_info, nullptr, &layout_handle);
        if (layout_result != VK_SUCCESS) {
          if (layout_handle != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(impl->device_, layout_handle, nullptr);
          }
          checkVk(layout_result, "vkCreateDescriptorSetLayout");
        }
        if (layout_handle == VK_NULL_HANDLE) {
          fail(NCNN_VK_STATUS_API_ERROR,
               "vkCreateDescriptorSetLayout returned a null handle");
        }
        descriptor_pool.impl_->add_set_layout(layout_handle);
        impl->set_layouts_[set] = layout_handle;
      }
      auto allocate_info = makeVkStructure<VkDescriptorSetAllocateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
      allocate_info.descriptorPool = descriptor_pool.get_handle();
      allocate_info.descriptorSetCount = set_count;
      allocate_info.pSetLayouts = impl->set_layouts_.data();
      inject(operation_hook, VulkanDeviceOperation::CreateDescriptor);
      checkVk(vkAllocateDescriptorSets(
                impl->device_, &allocate_info, impl->descriptor_sets_.data()),
              "vkAllocateDescriptorSets");

      std::vector<VkDescriptorBufferInfo> buffer_infos;
      std::vector<VkWriteDescriptorSet> writes;
      buffer_infos.reserve(bindings.size());
      writes.reserve(bindings.size());
      for (const VulkanPipelineBinding& binding : bindings) {
        buffer_infos.push_back(VkDescriptorBufferInfo{.buffer = binding.buffer,
                                                      .offset = binding.offset,
                                                      .range = binding.range});
        writes.push_back(VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .pNext = nullptr,
          .dstSet = impl->descriptor_sets_[binding.set],
          .dstBinding = binding.binding,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pImageInfo = nullptr,
          .pBufferInfo = &buffer_infos.back(),
          .pTexelBufferView = nullptr});
      }
      vkUpdateDescriptorSets(impl->device_,
                             static_cast<std::uint32_t>(writes.size()),
                             writes.data(),
                             0,
                             nullptr);
    }

    for (const VulkanPushConstantRange& range : push_constants) {
      if (range.stage_flags == 0 || range.size == 0 || range.offset % 4 != 0 ||
          range.size % 4 != 0 ||
          range.size >
            std::numeric_limits<std::uint32_t>::max() - range.offset) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "compute pipeline has an invalid push-constant range");
      }
    }
    auto pipeline_layout_info = makeVkStructure<VkPipelineLayoutCreateInfo>(
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    pipeline_layout_info.setLayoutCount =
      static_cast<std::uint32_t>(impl->set_layouts_.size());
    pipeline_layout_info.pSetLayouts = impl->set_layouts_.data();
    std::vector<VkPushConstantRange> vk_push_constants;
    vk_push_constants.reserve(push_constants.size());
    for (const VulkanPushConstantRange& range : push_constants) {
      vk_push_constants.push_back(
        VkPushConstantRange{.stageFlags = range.stage_flags,
                            .offset = range.offset,
                            .size = range.size});
    }
    pipeline_layout_info.pushConstantRangeCount =
      static_cast<std::uint32_t>(vk_push_constants.size());
    pipeline_layout_info.pPushConstantRanges = vk_push_constants.data();
    checkVk(vkCreatePipelineLayout(
              impl->device_, &pipeline_layout_info, nullptr, &impl->layout_),
            "vkCreatePipelineLayout");
    auto stage = makeVkStructure<VkPipelineShaderStageCreateInfo>(
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = impl->module_;
    const std::string entry_name(entry_point);
    stage.pName = entry_name.c_str();
    auto pipeline_info = makeVkStructure<VkComputePipelineCreateInfo>(
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    pipeline_info.stage = stage;
    pipeline_info.layout = impl->layout_;
    inject(operation_hook, VulkanDeviceOperation::CreatePipeline);
    checkVk(vkCreateComputePipelines(impl->device_,
                                     VK_NULL_HANDLE,
                                     1,
                                     &pipeline_info,
                                     nullptr,
                                     &impl->pipeline_),
            "vkCreateComputePipelines");
    return VulkanComputePipeline(std::move(impl));
  });
}

std::expected<void, VulkanDeviceError>
VulkanComputePipeline::destroy() noexcept {
  if (impl_) {
    impl_->destroy();
    impl_.reset();
  }
  return {};
}

VkPipeline VulkanComputePipeline::get_pipeline() const noexcept {
  return impl_ ? impl_->pipeline_ : VK_NULL_HANDLE;
}

VkPipelineLayout VulkanComputePipeline::get_layout() const noexcept {
  return impl_ ? impl_->layout_ : VK_NULL_HANDLE;
}

std::span<const VkDescriptorSet> VulkanComputePipeline::get_descriptor_sets()
  const noexcept {
  return impl_ ? std::span<const VkDescriptorSet>(impl_->descriptor_sets_)
               : std::span<const VkDescriptorSet>();
}

}  // namespace ncnn_mlir::vulkan
