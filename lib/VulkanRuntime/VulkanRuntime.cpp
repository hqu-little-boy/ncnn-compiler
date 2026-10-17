#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <expected>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ncnn-mlir/Support/VulkanModelPayload.hpp"
#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"
#include <vulkan/vulkan.h>

namespace ncnn_mlir::vulkan {
namespace {

constexpr std::uint64_t kDefaultContextMemoryBytes =
  kVulkanDeviceHelperDefaultBudgetBytes;
static_assert(kVulkanDeviceHelperMaxBufferBytes == kMinimumStorageBufferRange);

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
  using Buffer = VulkanBuffer;

  class Resource final {
   public:
    Resource(const TensorResource& spec, Buffer device)
      : spec_(&spec), device_(std::move(device)), staging_() {}

    [[nodiscard]] const TensorResource& get_spec() const noexcept {
      return *spec_;
    }
    [[nodiscard]] Buffer& get_device_buffer() noexcept { return device_; }
    [[nodiscard]] const Buffer& get_device_buffer() const noexcept {
      return device_;
    }
    [[nodiscard]] bool has_staging_buffer() const noexcept {
      return staging_.has_value();
    }
    [[nodiscard]] Buffer& get_staging_buffer() { return staging_.value(); }
    [[nodiscard]] const Buffer& get_staging_buffer() const {
      return staging_.value();
    }
    void set_staging_buffer(Buffer staging) {
      staging_.emplace(std::move(staging));
    }

   private:
    const TensorResource* spec_;
    Buffer device_;
    std::optional<Buffer> staging_;
  };

  class Kernel final {
   public:
    Kernel(const KernelExecution& spec, VulkanComputePipeline pipeline)
      : spec_(&spec), pipeline_(std::move(pipeline)) {}

    [[nodiscard]] const KernelExecution& get_spec() const noexcept {
      return *spec_;
    }
    [[nodiscard]] const VulkanComputePipeline& get_pipeline() const noexcept {
      return pipeline_;
    }

   private:
    const KernelExecution* spec_;
    VulkanComputePipeline pipeline_;
  };

  Impl(VulkanExecutionPlan plan,
       VulkanRuntimeOptions options,
       std::vector<VulkanKernelBinary> kernels,
       std::vector<VulkanResourcePayload> weights)
    : plan_(std::move(plan)),
      options_(std::move(options)),
      kernels_(std::move(kernels)),
      weights_(std::move(weights)),
      device_info_{.name = "",
                   .api_version = 0,
                   .driver_version = 0,
                   .vendor_id = 0,
                   .device_id = 0},
      device_context_(),
      command_resources_(),
      descriptor_pool_(),
      kernels_state_(),
      resources_(),
      weight_staging_(),
      peak_memory_bytes_(0),
      weight_upload_count_(0),
      validation_error_count_(0),
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
    createDeviceContext();
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
    try {
      if (device_context_) {
        const auto waited = device_context_->wait_idle();
        if (!waited) {
          destroy_status_ = waited.error().status;
        }
      }
      kernels_state_.clear();
      if (descriptor_pool_) {
        (void)descriptor_pool_->destroy();
        descriptor_pool_.reset();
      }
      resources_.clear();
      weight_staging_.clear();
      if (command_resources_) {
        const auto destroyed_commands = command_resources_->destroy();
        if (!destroyed_commands && destroy_status_ == NCNN_VK_STATUS_SUCCESS) {
          destroy_status_ = destroyed_commands.error().status;
        }
        command_resources_.reset();
      }
      if (device_context_) {
        if (destroy_status_ == NCNN_VK_STATUS_SUCCESS &&
            device_context_->get_validation_error_count() != 0) {
          destroy_status_ = NCNN_VK_STATUS_API_ERROR;
        }
        const auto destroyed = device_context_->destroy();
        validation_error_count_ = device_context_->get_validation_error_count();
        if (!destroyed && destroy_status_ == NCNN_VK_STATUS_SUCCESS) {
          destroy_status_ = destroyed.error().status;
        }
        if (validation_error_count_ != 0 &&
            destroy_status_ == NCNN_VK_STATUS_SUCCESS) {
          destroy_status_ = NCNN_VK_STATUS_API_ERROR;
        }
        device_context_.reset();
      }
    } catch (...) {
      destroy_status_ = NCNN_VK_STATUS_API_ERROR;
    }
    return destroy_status_;
  }

  [[nodiscard]] VulkanRuntimeDeviceInfo get_device_info() const {
    return device_info_;
  }

  [[nodiscard]] std::uint64_t get_peak_memory_bytes() const noexcept {
    return peak_memory_bytes_;
  }

  [[nodiscard]] std::uint64_t get_weight_upload_count() const noexcept {
    return weight_upload_count_;
  }

  [[nodiscard]] std::uint32_t get_validation_error_count() const noexcept {
    return device_context_ ? device_context_->get_validation_error_count()
                           : validation_error_count_;
  }

  void releasePackageViews() noexcept {
    kernels_.clear();
    weights_.clear();
  }

 private:
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

  void createDeviceContext() {
    const VulkanDeviceOptions device_options{
      .device_index = options_.device_index,
      .device_name_match = options_.device_name_match,
      .enable_validation = options_.enable_validation};
    auto created =
      VulkanDeviceContext::create(device_options, options_.failure_injector);
    if (!created) {
      fail(created.error().status, created.error().diagnostic);
    }
    device_context_.emplace(std::move(*created));
    device_info_ = device_context_->get_device_info();
  }

  void validateDeviceLimits() const {
    const VkPhysicalDeviceProperties& properties =
      device_context_->get_properties();
    std::uint64_t max_storage_range = 0;
    for (const TensorResource& resource : plan_.resources) {
      max_storage_range = std::max(
        max_storage_range, static_cast<std::uint64_t>(resource.byte_size));
    }
    std::vector<VulkanDeviceLimitRequirements> requirements;
    requirements.reserve(plan_.kernels.size());
    for (const KernelExecution& kernel : plan_.kernels) {
      const auto local = kernel.local_size;
      if (local[0] == 0 || local[1] != 1 || local[2] != 1) {
        fail(NCNN_VK_STATUS_UNSUPPORTED_DEVICE,
             "plan workgroup shape is unsupported by the selected device");
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
      std::uint32_t max_set = 0;
      for (const DescriptorBinding& binding : kernel.bindings) {
        if (binding.set == std::numeric_limits<std::uint32_t>::max() ||
            binding.binding == std::numeric_limits<std::uint32_t>::max()) {
          fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
               "plan descriptor binding exceeds representable limits");
        }
        max_set = std::max(max_set, binding.set);
      }
      const std::uint32_t set_count =
        kernel.bindings.empty() ? 0 : max_set + 1U;
      requirements.push_back(VulkanDeviceLimitRequirements{
        .local_size_x = local[0],
        .dispatch_count_x = groups > std::numeric_limits<std::uint32_t>::max()
                              ? std::numeric_limits<std::uint32_t>::max()
                              : static_cast<std::uint32_t>(groups),
        .storage_buffer_range = max_storage_range,
        .push_constant_bytes = max_push_end,
        .descriptor_set_count = set_count,
        .storage_buffer_descriptor_count =
          kernel.bindings.size() > std::numeric_limits<std::uint32_t>::max()
            ? std::numeric_limits<std::uint32_t>::max()
            : static_cast<std::uint32_t>(kernel.bindings.size())});
    }
    const auto validated =
      ncnn_mlir::vulkan::validateDeviceLimits(properties, requirements);
    if (!validated) {
      fail(validated.error().status, validated.error().diagnostic);
    }
  }

  void createCommandResources() {
    auto created = VulkanCommandResources::create(*device_context_,
                                                  options_.failure_injector);
    if (!created) {
      fail(created.error().status, created.error().diagnostic);
    }
    command_resources_.emplace(std::move(*created));
  }

  void createBuffers() {
    validateDeviceLimits();
    const auto prepare_buffer =
      [this](VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible) {
        const VulkanBufferCreateInfo create_info{
          .size = size,
          .usage = usage,
          .host_visible = host_visible,
          .force_mapped_memory_sync = options_.force_mapped_memory_sync};
        auto prepared = VulkanBuffer::prepare(
          *device_context_, create_info, options_.failure_injector);
        if (!prepared) {
          fail(prepared.error().status, prepared.error().diagnostic);
        }
        return std::move(*prepared);
      };

    for (const TensorResource& spec : plan_.resources) {
      const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      Buffer device_buffer = prepare_buffer(spec.byte_size, usage, false);
      Resource resource(spec, std::move(device_buffer));
      const auto [iterator, inserted] =
        resources_.emplace(spec.id, std::move(resource));
      (void)iterator;
      if (!inserted) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "execution plan contains duplicate resource IDs");
      }
    }

    for (const TensorResource& spec : plan_.resources) {
      if (spec.kind != ResourceKind::Input &&
          spec.kind != ResourceKind::Output) {
        continue;
      }
      resources_.at(spec.id).set_staging_buffer(prepare_buffer(
        spec.byte_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        true));
    }

    for (const TensorResource& spec : plan_.resources) {
      if (spec.kind != ResourceKind::Weight) {
        continue;
      }
      Buffer staging =
        prepare_buffer(spec.byte_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
      const auto [iterator, inserted] =
        weight_staging_.emplace(spec.id, std::move(staging));
      (void)iterator;
      if (!inserted) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "execution plan contains duplicate weight resources");
      }
    }

    std::uint64_t persistent_bytes = 0;
    std::uint64_t temporary_staging_bytes = 0;
    std::uint64_t allocation_count = 0;
    for (const auto& [id, resource] : resources_) {
      (void)id;
      if (!checkedAdd(persistent_bytes,
                      resource.get_device_buffer().get_allocation_size(),
                      persistent_bytes)) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "Vulkan context allocation size overflows uint64");
      }
      if (resource.has_staging_buffer() &&
          !checkedAdd(persistent_bytes,
                      resource.get_staging_buffer().get_allocation_size(),
                      persistent_bytes)) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "Vulkan staging allocation size overflows uint64");
      }
    }
    for (const auto& [id, staging] : weight_staging_) {
      (void)id;
      if (!checkedAdd(temporary_staging_bytes,
                      staging.get_allocation_size(),
                      temporary_staging_bytes)) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "weight staging allocation size overflows uint64");
      }
    }
    if (!checkedAdd(allocation_count,
                    static_cast<std::uint64_t>(resources_.size()),
                    allocation_count)) {
      fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
           "Vulkan context allocation count overflows uint64");
    }
    for (const auto& [id, resource] : resources_) {
      (void)id;
      if (resource.has_staging_buffer() &&
          !checkedAdd(allocation_count, 1, allocation_count)) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "Vulkan context allocation count overflows uint64");
      }
    }
    if (!checkedAdd(allocation_count,
                    static_cast<std::uint64_t>(weight_staging_.size()),
                    allocation_count)) {
      fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
           "Vulkan context allocation count overflows uint64");
    }
    std::uint64_t peak = 0;
    if (!checkedAdd(persistent_bytes, temporary_staging_bytes, peak)) {
      fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
           "Vulkan context allocation size overflows uint64");
    }
    const VulkanAllocationBudget budget{
      .persistent_bytes = persistent_bytes,
      .temporary_staging_bytes = temporary_staging_bytes,
      .allocation_count = allocation_count};
    const auto validated = validateAllocationBudget(
      budget,
      device_context_->get_properties().limits.maxMemoryAllocationCount,
      options_.max_context_memory_bytes);
    if (!validated) {
      if (validated.error().status == NCNN_VK_STATUS_OUT_OF_MEMORY &&
          peak > options_.max_context_memory_bytes) {
        fail(NCNN_VK_STATUS_OUT_OF_MEMORY,
             "Vulkan context exceeds configured memory budget (required " +
               std::to_string(peak) + " bytes, limit " +
               std::to_string(options_.max_context_memory_bytes) + " bytes)");
      }
      fail(validated.error().status, validated.error().diagnostic);
    }
    peak_memory_bytes_ = peak;
  }

  void allocateBuffers() {
    const auto allocate_buffer = [this](Buffer& buffer, bool map) {
      auto allocated = buffer.allocate_and_bind(map, options_.failure_injector);
      if (!allocated) {
        fail(allocated.error().status, allocated.error().diagnostic);
      }
    };
    for (auto& [id, resource] : resources_) {
      (void)id;
      allocate_buffer(resource.get_device_buffer(), false);
      if (resource.has_staging_buffer()) {
        allocate_buffer(resource.get_staging_buffer(), true);
      }
    }
    for (auto& [id, buffer] : weight_staging_) {
      (void)id;
      allocate_buffer(buffer, true);
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
      if (spec.bindings.empty()) {
        continue;
      }
      std::uint32_t max_set = 0;
      for (const DescriptorBinding& binding : spec.bindings) {
        max_set = std::max(max_set, binding.set);
      }
      if (max_set == std::numeric_limits<std::uint32_t>::max() ||
          !checkedAdd(
            set_count, static_cast<std::uint64_t>(max_set) + 1U, set_count)) {
        fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
             "execution plan descriptor-set count overflows uint64");
      }
    }
    if (descriptor_count > std::numeric_limits<std::uint32_t>::max() ||
        set_count > std::numeric_limits<std::uint32_t>::max()) {
      fail(NCNN_VK_STATUS_INVALID_ARGUMENT,
           "execution plan descriptor counts exceed Vulkan limits");
    }
    auto pool =
      VulkanDescriptorPool::create(*device_context_,
                                   static_cast<std::uint32_t>(set_count),
                                   static_cast<std::uint32_t>(descriptor_count),
                                   options_.failure_injector);
    if (!pool) {
      fail(pool.error().status, pool.error().diagnostic);
    }
    descriptor_pool_.emplace(std::move(*pool));

    for (const KernelExecution& spec : plan_.kernels) {
      const VulkanKernelBinary& binary = findKernelBinary(spec.id);
      std::vector<VulkanPipelineBinding> bindings;
      bindings.reserve(spec.bindings.size());
      for (const DescriptorBinding& descriptor : spec.bindings) {
        const Resource& resource = resources_.at(descriptor.resource_id);
        bindings.push_back(VulkanPipelineBinding{
          .set = descriptor.set,
          .binding = descriptor.binding,
          .buffer = resource.get_device_buffer().get_handle(),
          .buffer_size = resource.get_device_buffer().get_logical_size(),
          .offset = 0,
          .range = resource.get_device_buffer().get_logical_size()});
      }
      std::uint32_t max_push_end = 0;
      for (const PushConstantBlock& block : spec.push_constants) {
        max_push_end = std::max(
          max_push_end,
          block.offset + static_cast<std::uint32_t>(block.bytes.size()));
      }
      std::vector<VulkanPushConstantRange> push_constants;
      if (max_push_end != 0) {
        push_constants.push_back(
          VulkanPushConstantRange{.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT,
                                  .offset = 0,
                                  .size = max_push_end});
      }
      auto created = VulkanComputePipeline::create(*device_context_,
                                                   *descriptor_pool_,
                                                   binary.words,
                                                   spec.entry_point,
                                                   bindings,
                                                   push_constants,
                                                   options_.failure_injector);
      if (!created) {
        fail(created.error().status, created.error().diagnostic);
      }
      kernels_state_.emplace_back(spec, std::move(*created));
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

  void checkHelper(std::expected<void, VulkanDeviceError> result) {
    if (!result) {
      fail(result.error().status, result.error().diagnostic);
    }
  }

  void uploadWeights() {
    if (weights_.empty()) {
      return;
    }
    checkHelper(command_resources_->reset());
    checkHelper(command_resources_->begin());
    const VkCommandBuffer command_buffer =
      command_resources_->get_command_buffer();
    std::vector<VkBufferMemoryBarrier> host_barriers;
    std::vector<VkBufferMemoryBarrier> device_barriers;
    for (const VulkanResourcePayload& payload : weights_) {
      Buffer& staging = weight_staging_.at(payload.resource_id);
      const std::span<std::byte> mapped = staging.get_mapped();
      if (mapped.size() < payload.bytes.size()) {
        fail(NCNN_VK_STATUS_API_ERROR,
             "weight staging allocation is not mapped to the required size");
      }
      std::memcpy(mapped.data(), payload.bytes.data(), payload.bytes.size());
      checkHelper(staging.flush(options_.failure_injector));
      host_barriers.push_back(
        VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                              .pNext = nullptr,
                              .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .buffer = staging.get_handle(),
                              .offset = 0,
                              .size = VK_WHOLE_SIZE});
    }
    vkCmdPipelineBarrier(command_buffer,
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
        .srcOffset = 0,
        .dstOffset = 0,
        .size = resource.get_device_buffer().get_logical_size()};
      vkCmdCopyBuffer(command_buffer,
                      staging.get_handle(),
                      resource.get_device_buffer().get_handle(),
                      1,
                      &copy);
      device_barriers.push_back(VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = resource.get_device_buffer().get_handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE});
    }
    if (!device_barriers.empty()) {
      vkCmdPipelineBarrier(command_buffer,
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
    checkHelper(command_resources_->end());
    checkHelper(command_resources_->submit_and_wait(options_.failure_injector));
    weight_upload_count_ += weights_.size();
    for (auto& [id, buffer] : weight_staging_) {
      (void)id;
      (void)buffer.destroy();
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
    checkHelper(buffer.flush(options_.failure_injector));
  }

  void invalidateBuffer(const Buffer& buffer) {
    checkHelper(buffer.invalidate(options_.failure_injector));
  }

  void prepareInputStaging(std::span<const VulkanInputTensor> inputs) {
    for (const VulkanInputTensor& input : inputs) {
      Buffer& staging = resources_.at(input.resource_id).get_staging_buffer();
      const std::span<std::byte> mapped = staging.get_mapped();
      if (mapped.size() < input.bytes.size()) {
        fail(NCNN_VK_STATUS_API_ERROR,
             "input staging allocation is not mapped to the required size");
      }
      std::memcpy(mapped.data(), input.bytes.data(), input.bytes.size());
      flushBuffer(staging);
    }
  }

  void recordAndSubmit(std::span<const VulkanInputTensor> inputs,
                       std::span<const VulkanOutputTensor> outputs) {
    checkHelper(command_resources_->wait());
    checkHelper(command_resources_->reset());
    checkHelper(command_resources_->begin());
    const VkCommandBuffer command_buffer =
      command_resources_->get_command_buffer();

    std::vector<VkBufferMemoryBarrier> host_to_transfer;
    std::vector<VkBufferMemoryBarrier> transfer_to_compute;
    for (const VulkanInputTensor& input : inputs) {
      Resource& resource = resources_.at(input.resource_id);
      host_to_transfer.push_back(VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = resource.get_staging_buffer().get_handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE});
    }
    if (!host_to_transfer.empty()) {
      vkCmdPipelineBarrier(command_buffer,
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
        .srcOffset = 0,
        .dstOffset = 0,
        .size = resource.get_device_buffer().get_logical_size()};
      vkCmdCopyBuffer(command_buffer,
                      resource.get_staging_buffer().get_handle(),
                      resource.get_device_buffer().get_handle(),
                      1,
                      &copy);
      transfer_to_compute.push_back(VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = resource.get_device_buffer().get_handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE});
    }
    if (!transfer_to_compute.empty()) {
      vkCmdPipelineBarrier(
        command_buffer,
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
      std::unordered_set<std::uint32_t> current_ids(
        kernel.get_spec().reads.begin(), kernel.get_spec().reads.end());
      current_ids.insert(kernel.get_spec().writes.begin(),
                         kernel.get_spec().writes.end());
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
          .buffer = resource.get_device_buffer().get_handle(),
          .offset = 0,
          .size = VK_WHOLE_SIZE});
      }
      if (!barriers.empty()) {
        vkCmdPipelineBarrier(command_buffer,
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

      vkCmdBindPipeline(command_buffer,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        kernel.get_pipeline().get_pipeline());
      if (!kernel.get_pipeline().get_descriptor_sets().empty()) {
        vkCmdBindDescriptorSets(
          command_buffer,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          kernel.get_pipeline().get_layout(),
          0,
          static_cast<std::uint32_t>(
            kernel.get_pipeline().get_descriptor_sets().size()),
          kernel.get_pipeline().get_descriptor_sets().data(),
          0,
          nullptr);
      }
      for (const PushConstantBlock& block : kernel.get_spec().push_constants) {
        vkCmdPushConstants(command_buffer,
                           kernel.get_pipeline().get_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           block.offset,
                           static_cast<std::uint32_t>(block.bytes.size()),
                           block.bytes.data());
      }
      std::uint64_t output_elements = 0;
      for (std::uint32_t id : kernel.get_spec().writes) {
        output_elements = std::max(
          output_elements, resourceElements(resources_.at(id).get_spec()));
      }
      const auto groups = static_cast<std::uint32_t>(
        (output_elements + kernel.get_spec().local_size[0] - 1U) /
        kernel.get_spec().local_size[0]);
      vkCmdDispatch(command_buffer, groups, 1, 1);
      shader_written.insert(kernel.get_spec().writes.begin(),
                            kernel.get_spec().writes.end());
      (void)kernel_index;
    }

    std::vector<VkBufferMemoryBarrier> compute_to_transfer;
    for (const VulkanOutputTensor& output : outputs) {
      Resource& resource = resources_.at(output.resource_id);
      compute_to_transfer.push_back(VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = resource.get_device_buffer().get_handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE});
    }
    vkCmdPipelineBarrier(command_buffer,
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
        .srcOffset = 0,
        .dstOffset = 0,
        .size = resource.get_device_buffer().get_logical_size()};
      vkCmdCopyBuffer(command_buffer,
                      resource.get_device_buffer().get_handle(),
                      resource.get_staging_buffer().get_handle(),
                      1,
                      &copy);
      transfer_to_host.push_back(VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = resource.get_staging_buffer().get_handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE});
    }
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(transfer_to_host.size()),
                         transfer_to_host.data(),
                         0,
                         nullptr);
    checkHelper(command_resources_->end());
    checkHelper(command_resources_->submit_and_wait(options_.failure_injector));
  }

  void invalidateOutputs(std::span<const VulkanOutputTensor> outputs) {
    for (const VulkanOutputTensor& output : outputs) {
      invalidateBuffer(resources_.at(output.resource_id).get_staging_buffer());
    }
  }

  void publishOutputs(std::span<const VulkanOutputTensor> outputs) {
    std::vector<std::span<std::byte>> mapped_outputs;
    mapped_outputs.reserve(outputs.size());
    for (const VulkanOutputTensor& output : outputs) {
      const std::span<std::byte> mapped =
        resources_.at(output.resource_id).get_staging_buffer().get_mapped();
      if (mapped.empty()) {
        fail(NCNN_VK_STATUS_API_ERROR, "output staging memory is not mapped");
      }
      mapped_outputs.push_back(mapped);
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      std::memcpy(outputs[index].bytes.data(),
                  mapped_outputs[index].data(),
                  outputs[index].bytes.size());
    }
  }

  VulkanExecutionPlan plan_;
  VulkanRuntimeOptions options_;
  std::vector<VulkanKernelBinary> kernels_;
  std::vector<VulkanResourcePayload> weights_;
  VulkanRuntimeDeviceInfo device_info_;
  std::optional<VulkanDeviceContext> device_context_;
  std::optional<VulkanCommandResources> command_resources_;
  std::optional<VulkanDescriptorPool> descriptor_pool_;
  std::vector<Kernel> kernels_state_;
  std::unordered_map<std::uint32_t, Resource> resources_;
  std::unordered_map<std::uint32_t, Buffer> weight_staging_;
  std::uint64_t peak_memory_bytes_;
  std::uint64_t weight_upload_count_;
  std::uint32_t validation_error_count_;
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
