#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"
#include <vulkan/vulkan.h>

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
using ncnn_mlir::vulkan::VulkanDeviceInfo;
using ncnn_mlir::vulkan::VulkanDeviceLimitRequirements;
using ncnn_mlir::vulkan::VulkanDeviceOperation;
using ncnn_mlir::vulkan::VulkanDeviceOptions;
using ncnn_mlir::vulkan::VulkanPipelineBinding;

constexpr std::string_view kFixtureId = "va2-f32-add-257-gpu-func-v1";
constexpr std::uint32_t kElementCount = 257;
constexpr std::uint32_t kLocalSize = 64;
constexpr std::size_t kMaximumShaderBytes = 64U * 1024U * 1024U;
constexpr float kTolerance = 1.0e-6F;
std::atomic<std::uint32_t> g_validation_messages = 0;

VKAPI_ATTR VkBool32 VKAPI_CALL test_validation_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT severity,
  VkDebugUtilsMessageTypeFlagsEXT,
  const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
  void*) {
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    g_validation_messages.fetch_add(1);
    std::cerr << "Vulkan validation error: "
              << (callback_data != nullptr && callback_data->pMessage != nullptr
                    ? callback_data->pMessage
                    : "<no message>")
              << '\n';
  }
  return VK_FALSE;
}

class ValidationObserver final {
 public:
  ValidationObserver(VkInstance instance, bool enabled) : instance_(instance) {
    if (!enabled) {
      return;
    }
    const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create == nullptr) {
      throw std::runtime_error(
        "validation layer did not expose vkCreateDebugUtilsMessengerEXT");
    }
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = test_validation_callback;
    const VkResult result =
      create(instance_, &create_info, nullptr, &messenger_);
    if (result != VK_SUCCESS) {
      throw std::runtime_error(
        "vkCreateDebugUtilsMessengerEXT failed with VkResult " +
        std::to_string(result));
    }
  }
  ValidationObserver(const ValidationObserver&) = delete;
  ValidationObserver& operator=(const ValidationObserver&) = delete;
  ~ValidationObserver() { destroy(); }

  void destroy() noexcept {
    if (messenger_ == VK_NULL_HANDLE || instance_ == VK_NULL_HANDLE) {
      return;
    }
    const auto destroy_messenger =
      reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_messenger != nullptr) {
      destroy_messenger(instance_, messenger_, nullptr);
    }
    messenger_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
  }

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

struct Options final {
  std::string mode;
  std::filesystem::path shader;
  std::filesystem::path cpu_library;
  std::string cpu_symbol = "v0_add";
  std::string device_match;
  std::filesystem::path lhs;
  std::filesystem::path rhs;
  std::filesystem::path cpu_output;
  std::filesystem::path gpu_output;
  std::filesystem::path gpu_repeat_output;
  std::filesystem::path golden;
  std::filesystem::path result;
  bool validation = false;
};

std::runtime_error helper_error(const VulkanDeviceError& error) {
  return std::runtime_error("Vulkan helper status " +
                            std::to_string(error.status) + ": " +
                            error.diagnostic);
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::string json_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20U) {
          escaped += '?';
        } else {
          escaped += character;
        }
    }
  }
  return escaped;
}

void write_text(const std::filesystem::path& path, std::string_view value) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output ||
      !output.write(value.data(), static_cast<std::streamsize>(value.size()))) {
    throw std::runtime_error("cannot write evidence file: " + path.string());
  }
}

void write_f32_file(const std::filesystem::path& path,
                    std::span<const float> values) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create FP32 evidence file: " +
                             path.string());
  }
  for (const float value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::array<unsigned char, 4> bytes{
      static_cast<unsigned char>(bits & 0xFFU),
      static_cast<unsigned char>((bits >> 8U) & 0xFFU),
      static_cast<unsigned char>((bits >> 16U) & 0xFFU),
      static_cast<unsigned char>((bits >> 24U) & 0xFFU),
    };
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  if (!output) {
    throw std::runtime_error("cannot write FP32 evidence file: " +
                             path.string());
  }
}

std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open SPIR-V file: " + path.string());
  }
  const std::streamoff byte_count = input.tellg();
  if (byte_count < 20 || std::cmp_greater(byte_count, kMaximumShaderBytes) ||
      byte_count % 4 != 0) {
    throw std::runtime_error("SPIR-V file has an invalid byte length");
  }
  input.seekg(0);
  std::vector<unsigned char> bytes(static_cast<std::size_t>(byte_count));
  if (!input.read(reinterpret_cast<char*>(bytes.data()), byte_count)) {
    throw std::runtime_error("cannot read SPIR-V file: " + path.string());
  }
  std::vector<std::uint32_t> words(bytes.size() / 4U);
  for (std::size_t index = 0; index < words.size(); ++index) {
    const std::size_t offset = index * 4U;
    words[index] = static_cast<std::uint32_t>(bytes[offset]) |
                   (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
                   (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
                   (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
  }
  if (words.front() != 0x07230203U) {
    throw std::runtime_error(
      "SPIR-V file has an invalid little-endian magic number");
  }
  return words;
}

Options parse_options(int argc, char** argv) {
  Options options;
  auto next_value = [&](int& index, std::string_view name) -> std::string {
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + std::string(name));
    }
    ++index;
    return argv[index];
  };
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--mode") {
      options.mode = next_value(index, arg);
    } else if (arg == "--shader") {
      options.shader = next_value(index, arg);
    } else if (arg == "--cpu-library") {
      options.cpu_library = next_value(index, arg);
    } else if (arg == "--cpu-symbol") {
      options.cpu_symbol = next_value(index, arg);
    } else if (arg == "--device-match") {
      options.device_match = next_value(index, arg);
    } else if (arg == "--lhs") {
      options.lhs = next_value(index, arg);
    } else if (arg == "--rhs") {
      options.rhs = next_value(index, arg);
    } else if (arg == "--cpu-output") {
      options.cpu_output = next_value(index, arg);
    } else if (arg == "--gpu-output") {
      options.gpu_output = next_value(index, arg);
    } else if (arg == "--gpu-repeat-output") {
      options.gpu_repeat_output = next_value(index, arg);
    } else if (arg == "--golden") {
      options.golden = next_value(index, arg);
    } else if (arg == "--result") {
      options.result = next_value(index, arg);
    } else if (arg == "--validation") {
      options.validation = true;
    } else {
      throw std::runtime_error("unknown argument: " + std::string(arg));
    }
  }
  if (options.mode != "add" && options.mode != "no-device") {
    throw std::runtime_error("mode must be add or no-device");
  }
  if (options.result.empty()) {
    throw std::runtime_error("--result is required");
  }
  if (options.mode == "add" &&
      (options.shader.empty() || options.cpu_library.empty() ||
       options.device_match.empty() || options.lhs.empty() ||
       options.rhs.empty() || options.cpu_output.empty() ||
       options.gpu_output.empty() || options.gpu_repeat_output.empty() ||
       options.golden.empty())) {
    throw std::runtime_error(
      "add mode is missing a required path or device match");
  }
  return options;
}

std::vector<float> make_input(bool lhs) {
  std::vector<float> values(kElementCount);
  for (std::uint32_t index = 0; index < kElementCount; ++index) {
    const auto residue = static_cast<std::int32_t>(index % (lhs ? 17U : 11U));
    const std::int32_t centered =
      residue - static_cast<std::int32_t>((lhs ? 17U : 11U) / 2U);
    const float divisor = lhs ? 8.0F : 4.0F;
    values[index] = static_cast<float>(centered) / divisor;
  }
  return values;
}

class SharedLibrary final {
 public:
  explicit SharedLibrary(const std::filesystem::path& path) {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw std::runtime_error(
        "dlopen CPU AOT library failed: " +
        std::string(error != nullptr ? error : "unknown"));
    }
  }
  SharedLibrary(const SharedLibrary&) = delete;
  SharedLibrary& operator=(const SharedLibrary&) = delete;
  ~SharedLibrary() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  void* symbol(std::string_view name) const {
    dlerror();
    void* address = dlsym(handle_, std::string(name).c_str());
    const char* error = dlerror();
    if (error != nullptr || address == nullptr) {
      throw std::runtime_error("CPU AOT symbol lookup failed: " +
                               std::string(error != nullptr ? error : "null"));
    }
    return address;
  }

 private:
  void* handle_ = nullptr;
};

void check_result(const std::expected<void, VulkanDeviceError>& result) {
  if (!result) {
    throw helper_error(result.error());
  }
}

template <typename T>
T take(std::expected<T, VulkanDeviceError>&& result) {
  if (!result) {
    throw helper_error(result.error());
  }
  return std::move(*result);
}

class DeviceResourceOwner final {
 public:
  explicit DeviceResourceOwner(VulkanDeviceContext& context)
    : context_(context) {}
  DeviceResourceOwner(const DeviceResourceOwner&) = delete;
  DeviceResourceOwner& operator=(const DeviceResourceOwner&) = delete;
  ~DeviceResourceOwner() { (void)context_.wait_idle(); }

  std::vector<VulkanBuffer> buffers;
  std::optional<VulkanDescriptorPool> descriptor_pool;
  std::optional<VulkanComputePipeline> pipeline;
  std::optional<VulkanCommandResources> commands;

 private:
  VulkanDeviceContext& context_;
};

struct AddExecution final {
  std::vector<float> output;
  bool wait_failure_recovered;
};

AddExecution execute_add(VulkanDeviceContext& context,
                         std::span<const std::uint32_t> shader_words,
                         std::span<const float> lhs,
                         std::span<const float> rhs,
                         bool inject_wait_failure) {
  constexpr VkDeviceSize byte_count = kElementCount * sizeof(float);
  const std::array<VulkanDeviceLimitRequirements, 1> limits{
    VulkanDeviceLimitRequirements{
      .local_size_x = kLocalSize,
      .dispatch_count_x = (kElementCount + kLocalSize - 1U) / kLocalSize,
      .storage_buffer_range = byte_count,
      .push_constant_bytes = 0,
      .descriptor_set_count = 1,
      .storage_buffer_descriptor_count = 3}};
  check_result(validateDeviceLimits(context.get_properties(), limits));

  constexpr VkBufferUsageFlags host_input_usage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  constexpr VkBufferUsageFlags host_output_usage =
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  constexpr VkBufferUsageFlags device_usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  DeviceResourceOwner owner(context);
  std::vector<VulkanBuffer>& buffers = owner.buffers;
  buffers.reserve(6);
  auto prepare = [&](VkBufferUsageFlags usage, bool host_visible) {
    buffers.push_back(take(VulkanBuffer::prepare(
      context,
      VulkanBufferCreateInfo{.size = byte_count,
                             .usage = usage,
                             .host_visible = host_visible,
                             .force_mapped_memory_sync = false})));
  };
  prepare(host_input_usage, true);
  prepare(host_input_usage, true);
  prepare(device_usage, false);
  prepare(device_usage, false);
  prepare(device_usage, false);
  prepare(host_output_usage, true);

  std::uint64_t persistent_bytes = 0;
  std::uint64_t staging_bytes = 0;
  for (std::size_t index = 0; index < buffers.size(); ++index) {
    const std::uint64_t allocation_size = buffers[index].get_allocation_size();
    std::uint64_t& total =
      index == 2 || index == 3 || index == 4 ? persistent_bytes : staging_bytes;
    if (allocation_size > std::numeric_limits<std::uint64_t>::max() - total) {
      throw std::runtime_error(
        "Vulkan requirements allocation size overflows uint64");
    }
    total += allocation_size;
  }
  const VulkanAllocationBudget budget{.persistent_bytes = persistent_bytes,
                                      .temporary_staging_bytes = staging_bytes,
                                      .allocation_count = buffers.size()};
  check_result(validateAllocationBudget(
    budget,
    context.get_properties().limits.maxMemoryAllocationCount,
    512U * 1024U * 1024U));

  for (std::size_t index = 0; index < buffers.size(); ++index) {
    check_result(buffers[index].allocate_and_bind(index < 2 || index == 5));
  }
  for (const auto& [index, values] :
       {std::pair<std::size_t, std::span<const float>>{0, lhs}, {1, rhs}}) {
    const auto mapped = buffers[index].get_mapped();
    require(mapped.size() >= byte_count,
            "input staging map is shorter than 257 FP32 values");
    std::memcpy(
      mapped.data(), values.data(), static_cast<std::size_t>(byte_count));
    check_result(buffers[index].flush());
  }

  const std::array<VulkanPipelineBinding, 3> bindings{
    VulkanPipelineBinding{.set = 0,
                          .binding = 0,
                          .buffer = buffers[2].get_handle(),
                          .buffer_size = byte_count,
                          .offset = 0,
                          .range = byte_count},
    VulkanPipelineBinding{.set = 0,
                          .binding = 1,
                          .buffer = buffers[3].get_handle(),
                          .buffer_size = byte_count,
                          .offset = 0,
                          .range = byte_count},
    VulkanPipelineBinding{.set = 0,
                          .binding = 2,
                          .buffer = buffers[4].get_handle(),
                          .buffer_size = byte_count,
                          .offset = 0,
                          .range = byte_count},
  };
  owner.descriptor_pool.emplace(
    take(VulkanDescriptorPool::create(context, 1, bindings.size())));
  owner.pipeline.emplace(take(VulkanComputePipeline::create(
    context, *owner.descriptor_pool, shader_words, "add", bindings, {})));
  owner.commands.emplace(take(VulkanCommandResources::create(context)));
  VulkanComputePipeline& pipeline = *owner.pipeline;
  VulkanCommandResources& commands = *owner.commands;
  check_result(commands.reset());
  check_result(commands.begin());
  const VkCommandBuffer command_buffer = commands.get_command_buffer();
  const VkBufferCopy copy_region{
    .srcOffset = 0, .dstOffset = 0, .size = byte_count};
  vkCmdCopyBuffer(command_buffer,
                  buffers[0].get_handle(),
                  buffers[2].get_handle(),
                  1,
                  &copy_region);
  vkCmdCopyBuffer(command_buffer,
                  buffers[1].get_handle(),
                  buffers[3].get_handle(),
                  1,
                  &copy_region);

  const std::array<VkBufferMemoryBarrier, 2> input_barriers{
    VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                          .pNext = nullptr,
                          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .buffer = buffers[2].get_handle(),
                          .offset = 0,
                          .size = VK_WHOLE_SIZE},
    VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                          .pNext = nullptr,
                          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                          .buffer = buffers[3].get_handle(),
                          .offset = 0,
                          .size = VK_WHOLE_SIZE},
  };
  vkCmdPipelineBarrier(command_buffer,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0,
                       0,
                       nullptr,
                       static_cast<std::uint32_t>(input_barriers.size()),
                       input_barriers.data(),
                       0,
                       nullptr);
  vkCmdBindPipeline(
    command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.get_pipeline());
  const auto descriptor_sets = pipeline.get_descriptor_sets();
  require(descriptor_sets.size() == 1,
          "Add pipeline must expose exactly one descriptor set");
  vkCmdBindDescriptorSets(command_buffer,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.get_layout(),
                          0,
                          static_cast<std::uint32_t>(descriptor_sets.size()),
                          descriptor_sets.data(),
                          0,
                          nullptr);
  vkCmdDispatch(
    command_buffer, (kElementCount + kLocalSize - 1U) / kLocalSize, 1, 1);
  const VkBufferMemoryBarrier output_barrier{
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .buffer = buffers[4].get_handle(),
    .offset = 0,
    .size = VK_WHOLE_SIZE};
  vkCmdPipelineBarrier(command_buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0,
                       0,
                       nullptr,
                       1,
                       &output_barrier,
                       0,
                       nullptr);
  vkCmdCopyBuffer(command_buffer,
                  buffers[4].get_handle(),
                  buffers[5].get_handle(),
                  1,
                  &copy_region);
  const VkBufferMemoryBarrier readback_barrier{
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .buffer = buffers[5].get_handle(),
    .offset = 0,
    .size = VK_WHOLE_SIZE};
  vkCmdPipelineBarrier(command_buffer,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT,
                       0,
                       0,
                       nullptr,
                       1,
                       &readback_barrier,
                       0,
                       nullptr);
  check_result(commands.end());
  bool wait_failure_recovered = false;
  if (inject_wait_failure) {
    auto injected_wait = commands.submit_and_wait(
      [](VulkanDeviceOperation operation) -> std::optional<ncnn_vk_status> {
        if (operation == VulkanDeviceOperation::Wait) {
          return NCNN_VK_STATUS_DEVICE_LOST;
        }
        return std::nullopt;
      });
    require(!injected_wait &&
              injected_wait.error().status == NCNN_VK_STATUS_DEVICE_LOST,
            "wait-failure hook did not report the injected device-lost status");
    check_result(commands.wait());
    wait_failure_recovered = true;
  } else {
    check_result(commands.submit_and_wait());
  }
  check_result(buffers[5].invalidate());
  const auto mapped_output = buffers[5].get_mapped();
  require(mapped_output.size() >= byte_count,
          "readback staging map is shorter than 257 FP32 values");
  std::vector<float> output(kElementCount);
  std::memcpy(
    output.data(), mapped_output.data(), static_cast<std::size_t>(byte_count));

  check_result(commands.destroy());
  check_result(pipeline.destroy());
  check_result(owner.descriptor_pool->destroy());
  for (VulkanBuffer& buffer : buffers) {
    check_result(buffer.destroy());
  }
  check_result(context.wait_idle());
  return AddExecution{.output = std::move(output),
                      .wait_failure_recovered = wait_failure_recovered};
}

void run_no_device(const Options& options) {
  const VulkanDeviceOptions device_options{.device_index = NCNN_VK_DEVICE_AUTO,
                                           .device_name_match = {},
                                           .enable_validation = false};
  auto context = VulkanDeviceContext::create(device_options);
  if (context) {
    check_result(context->destroy());
    write_text(options.result,
               "{\"schema_version\":1,\"kind\":\"ncnn.vulkan_device_helper_no_"
               "device_result\","
               "\"helper_create_succeeded\":true,\"diagnostic\":\"an ICD "
               "exposed a device\"}\n");
    throw std::runtime_error(
      "helper unexpectedly created a device under the unavailable ICD");
  }
  const std::string diagnostic = context.error().diagnostic;
  write_text(options.result,
             "{\"schema_version\":1,\"kind\":\"ncnn.vulkan_device_helper_no_"
             "device_result\","
             "\"helper_create_succeeded\":false,\"status\":" +
               std::to_string(context.error().status) + R"(,"diagnostic":")" +
               json_escape(diagnostic) + "\"}\n");
  std::cout << "helper create failed without a Vulkan device: " << diagnostic
            << '\n';
}

void run_add(const Options& options) {
  const std::vector<std::uint32_t> shader = read_spirv(options.shader);
  std::vector<float> lhs = make_input(true);
  std::vector<float> rhs = make_input(false);
  std::vector<float> golden(kElementCount);
  std::vector<float> cpu_output(kElementCount,
                                std::numeric_limits<float>::quiet_NaN());
  for (std::size_t index = 0; index < golden.size(); ++index) {
    golden[index] = lhs[index] + rhs[index];
  }
  {
    SharedLibrary library(options.cpu_library);
    using CpuAdd = int (*)(const float*, const float*, float*);
    auto* add = reinterpret_cast<CpuAdd>(library.symbol(options.cpu_symbol));
    const int status = add(lhs.data(), rhs.data(), cpu_output.data());
    require(status == 0,
            "CPU AOT Add returned non-zero status " + std::to_string(status));
  }

  const VulkanDeviceOptions device_options{
    .device_index = NCNN_VK_DEVICE_AUTO,
    .device_name_match = options.device_match,
    .enable_validation = options.validation};
  VulkanDeviceContext first = take(VulkanDeviceContext::create(device_options));
  VulkanDeviceContext second =
    take(VulkanDeviceContext::create(device_options));
  require(first.get_instance() != second.get_instance(),
          "independent contexts unexpectedly share the same VkInstance");
  require(first.get_device() != second.get_device(),
          "independent contexts unexpectedly share the same VkDevice");
  g_validation_messages.store(0);
  ValidationObserver first_validation(first.get_instance(), options.validation);
  ValidationObserver second_validation(second.get_instance(),
                                       options.validation);
  const VulkanDeviceInfo device = first.get_device_info();
  const AddExecution first_execution =
    execute_add(first, shader, lhs, rhs, false);
  const AddExecution repeated_execution =
    execute_add(second, shader, lhs, rhs, true);
  const std::vector<float>& first_output = first_execution.output;
  const std::vector<float>& repeated_output = repeated_execution.output;
  float max_cpu_abs = 0.0F;
  float max_gpu_abs = 0.0F;
  float max_repeat_abs = 0.0F;
  for (std::size_t index = 0; index < golden.size(); ++index) {
    max_cpu_abs =
      std::max(max_cpu_abs, std::abs(cpu_output[index] - golden[index]));
    max_gpu_abs =
      std::max(max_gpu_abs, std::abs(first_output[index] - golden[index]));
    max_repeat_abs = std::max(
      max_repeat_abs, std::abs(first_output[index] - repeated_output[index]));
  }
  const bool wait_failure_recovered = repeated_execution.wait_failure_recovered;
  const bool passed = max_cpu_abs <= kTolerance && max_gpu_abs <= kTolerance &&
                      max_repeat_abs == 0.0F && wait_failure_recovered;
  write_f32_file(options.lhs, lhs);
  write_f32_file(options.rhs, rhs);
  write_f32_file(options.cpu_output, cpu_output);
  write_f32_file(options.gpu_output, first_output);
  write_f32_file(options.gpu_repeat_output, repeated_output);
  write_f32_file(options.golden, golden);

  first_validation.destroy();
  second_validation.destroy();
  check_result(first.destroy());
  check_result(second.destroy());
  const std::uint32_t validation_errors = first.get_validation_error_count() +
                                          second.get_validation_error_count() +
                                          g_validation_messages.load();
  require(validation_errors == 0,
          "Vulkan validation callback reported an error");

  const std::string result =
    "{\n"
    "  \"schema_version\": 1,\n"
    "  \"kind\": \"ncnn.vulkan_device_helper_add_result\",\n"
    "  \"fixture_id\": \"" +
    std::string(kFixtureId) +
    "\",\n"
    "  \"status\": \"" +
    (passed ? "passed" : "failed") +
    "\",\n"
    "  \"device\": \"" +
    (options.device_match == "NVIDIA" ? "nvidia" : "llvmpipe") +
    "\",\n"
    "  \"device_name\": \"" +
    json_escape(device.name) +
    "\",\n"
    "  \"api_version\": " +
    std::to_string(device.api_version) +
    ",\n"
    "  \"driver_version\": " +
    std::to_string(device.driver_version) +
    ",\n"
    "  \"vendor_id\": " +
    std::to_string(device.vendor_id) +
    ",\n"
    "  \"device_id\": " +
    std::to_string(device.device_id) +
    ",\n"
    "  \"elements\": 257,\n"
    "  \"local_size\": 64,\n"
    "  \"dispatch_groups_x\": 5,\n"
    "  \"validation_enabled\": " +
    (options.validation ? "true" : "false") +
    ",\n"
    "  \"validation_errors\": " +
    std::to_string(validation_errors) +
    ",\n"
    "  \"wait_failure_recovered\": " +
    (wait_failure_recovered ? "true" : "false") +
    ",\n"
    "  \"max_cpu_golden_abs\": " +
    std::to_string(max_cpu_abs) +
    ",\n"
    "  \"max_gpu_golden_abs\": " +
    std::to_string(max_gpu_abs) +
    ",\n"
    "  \"max_repeat_abs\": " +
    std::to_string(max_repeat_abs) +
    "\n"
    "}\n";
  write_text(options.result, result);
  require(passed, "CPU AOT/helper GPU/scalar Add comparison failed");
  std::cout << "device=" << device.name << " elements=257 groups=5"
            << " cpu_abs=" << max_cpu_abs << " gpu_abs=" << max_gpu_abs
            << " repeat_abs=" << max_repeat_abs
            << " validation_errors=" << validation_errors << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path result_path;
  try {
    const Options options = parse_options(argc, argv);
    result_path = options.result;
    if (options.mode == "no-device") {
      run_no_device(options);
    } else {
      run_add(options);
    }
  } catch (const std::exception& error) {
    if (!result_path.empty()) {
      try {
        write_text(result_path,
                   "{\"schema_version\":1,\"kind\":\"ncnn.vulkan_device_helper_"
                   "runner_error\","
                   "\"status\":\"failed\",\"diagnostic\":\"" +
                     json_escape(error.what()) + "\"}\n");
      } catch (...) {}
    }
    std::cerr << "vulkan_device_helper_runner: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
