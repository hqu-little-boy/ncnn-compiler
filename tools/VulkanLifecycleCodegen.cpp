#include "VulkanLifecycleCodegen.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace ncnn_compile {
namespace {

namespace vk = ncnn_mlir::vulkan;

void replaceToken(std::string& source,
                  std::string_view token,
                  std::string_view replacement) {
  std::size_t position = 0;
  while ((position = source.find(token, position)) != std::string::npos) {
    source.replace(position, token.size(), replacement);
    position += replacement.size();
  }
}

std::string_view stageToken(vk::HostStage stage) {
  switch (stage) {
    case vk::HostStage::Host:
      return "VK_PIPELINE_STAGE_HOST_BIT";
    case vk::HostStage::Transfer:
      return "VK_PIPELINE_STAGE_TRANSFER_BIT";
    case vk::HostStage::Compute:
      return "VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT";
  }
  return {};
}

std::string_view accessToken(vk::HostAccess access) {
  switch (access) {
    case vk::HostAccess::HostWrite:
      return "VK_ACCESS_HOST_WRITE_BIT";
    case vk::HostAccess::TransferRead:
      return "VK_ACCESS_TRANSFER_READ_BIT";
    case vk::HostAccess::TransferWrite:
      return "VK_ACCESS_TRANSFER_WRITE_BIT";
    case vk::HostAccess::ShaderRead:
      return "VK_ACCESS_SHADER_READ_BIT";
    case vk::HostAccess::ShaderWrite:
      return "VK_ACCESS_SHADER_WRITE_BIT";
    case vk::HostAccess::ShaderReadWrite:
      return "VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT";
    case vk::HostAccess::HostRead:
      return "VK_ACCESS_HOST_READ_BIT";
  }
  return {};
}

std::string emitPorts(std::string_view name,
                      std::span<const vk::HostPort> ports) {
  std::string result =
    std::format("[[maybe_unused]] constexpr std::array<Port, {}> {}{{{{\n",
                ports.size(),
                name);
  for (const auto& port : ports) {
    result += std::format(
      "  Port{{.device_buffer = {}, .staging_buffer = {}, "
      ".elements = UINT64_C({})}},\n",
      port.device_buffer.value,
      port.staging_buffer.value,
      port.elements);
  }
  result += "}};\n";
  return result;
}

std::string emitCopyOperation(const vk::HostCopyOp& op) {
  return std::format(
    "    {{\n"
    "      const VkBufferCopy copy{{.srcOffset = UINT64_C({0}), "
    ".dstOffset = UINT64_C({1}), .size = UINT64_C({2})}};\n"
    "      vkCmdCopyBuffer(commands_->get_command_buffer(), "
    "buffers_[{3}]->get_handle(), buffers_[{4}]->get_handle(), "
    "1, &copy);\n"
    "    }}\n",
    op.source_offset,
    op.destination_offset,
    op.byte_size,
    op.source.value,
    op.destination.value);
}

std::expected<std::string, std::string> emitBarrierOperation(
  const vk::HostBarrierOp& op) {
  const std::string_view source = stageToken(op.source_stage);
  const std::string_view destination = stageToken(op.destination_stage);
  if (source.empty() || destination.empty() || op.buffers.empty() ||
      !std::in_range<std::uint32_t>(op.buffers.size())) {
    return std::unexpected("invalid host barrier stage or count");
  }
  std::string text = std::format(
    "    {{\n      const std::array<VkBufferMemoryBarrier, {}> "
    "barriers{{{{\n",
    op.buffers.size());
  for (const auto& barrier : op.buffers) {
    const auto source_access = accessToken(barrier.source_access);
    const auto destination_access = accessToken(barrier.destination_access);
    if (source_access.empty() || destination_access.empty()) {
      return std::unexpected("invalid host barrier access");
    }
    text += std::format(
      "        VkBufferMemoryBarrier{{.sType = "
      "VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .pNext = nullptr, "
      ".srcAccessMask = {0}, .dstAccessMask = {1}, "
      ".srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, "
      ".dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, "
      ".buffer = buffers_[{2}]->get_handle(), "
      ".offset = UINT64_C({3}), .size = UINT64_C({4})}},\n",
      source_access,
      destination_access,
      barrier.buffer.value,
      barrier.offset,
      barrier.byte_size);
  }
  text += std::format(
    "      }}}};\n"
    "      vkCmdPipelineBarrier(commands_->get_command_buffer(), "
    "{}, {}, 0, 0, nullptr, {}, barriers.data(), 0, nullptr);\n"
    "    }}\n",
    source,
    destination,
    op.buffers.size());
  return text;
}

std::expected<std::string, std::string> emitCreateOperations(
  const vk::VulkanHostProgram& program,
  const std::map<std::uint32_t, std::size_t>& weight_slots) {
  std::string result;
  std::uint64_t pending_uploads = 0;
  for (std::size_t index = 0; index < program.create_operations.size();
       ++index) {
    const auto* control =
      std::get_if<vk::HostControlOp>(&program.create_operations[index]);
    if (control != nullptr) {
      const auto nextControlIs = [&](vk::HostControl action) {
        if (index + 1 >= program.create_operations.size()) {
          return false;
        }
        const auto* next =
          std::get_if<vk::HostControlOp>(&program.create_operations[index + 1]);
        return next != nullptr && next->action == action;
      };
      switch (control->action) {
        case vk::HostControl::ResetFence:
          if (!nextControlIs(vk::HostControl::ResetCommandPool)) {
            return std::unexpected("lifecycle reset must reset fence and pool");
          }
          // The helper implements these two adjacent, already-planned
          // primitives together; it does not choose command ordering.
          result += "    require(commands_->reset());\n";
          ++index;
          break;
        case vk::HostControl::BeginCommands:
          result += "    require(commands_->begin());\n";
          break;
        case vk::HostControl::EndCommands:
          result += "    require(commands_->end());\n";
          break;
        case vk::HostControl::Submit:
          if (!nextControlIs(vk::HostControl::WaitFence)) {
            return std::unexpected("lifecycle submission must wait for fence");
          }
          result +=
            "    "
            "require(commands_->submit_and_wait(settings_.operation_hook));\n";
          result += std::format("    weight_upload_count_ += UINT64_C({});\n",
                                pending_uploads);
          pending_uploads = 0;
          ++index;
          break;
        default:
          return std::unexpected("unsupported lifecycle control operation");
      }
      continue;
    }
    auto emitted = std::visit(
      [&](const auto& op) -> std::expected<std::string, std::string> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, vk::HostUploadOp>) {
          const auto found = weight_slots.find(op.source_id);
          if (op.source != vk::HostUploadSource::Weight ||
              found == weight_slots.end()) {
            return std::unexpected("lifecycle upload has no embedded weight");
          }
          return std::format(
            "    std::memcpy(buffers_[{0}]->get_mapped().data(), "
            "kWeightBytes{1}.data(), kWeightBytes{1}.size());\n",
            op.staging_buffer.value,
            found->second);
        } else if constexpr (std::is_same_v<T, vk::HostMappedSyncOp>) {
          if (op.action != vk::HostMappedSync::Flush) {
            return std::unexpected("lifecycle only flushes uploaded weights");
          }
          return std::format(
            "    require(buffers_[{}]->flush(settings_.operation_hook));\n",
            op.buffer.value);
        } else if constexpr (std::is_same_v<T, vk::HostCopyOp>) {
          if (program.buffers[op.source.value].kind !=
              vk::HostBufferKind::WeightStaging) {
            return std::unexpected("lifecycle copy must upload a weight");
          }
          ++pending_uploads;
          return emitCopyOperation(op);
        } else if constexpr (std::is_same_v<T, vk::HostBarrierOp>) {
          return emitBarrierOperation(op);
        } else if constexpr (std::is_same_v<T, vk::HostReleaseBufferOp>) {
          return std::format(
            "    require(buffers_[{0}]->destroy());\n"
            "    buffers_[{0}].reset();\n",
            op.buffer.value);
        } else {
          return std::unexpected("non-lifecycle operation in create program");
        }
      },
      program.create_operations[index]);
    if (!emitted) {
      return std::unexpected(emitted.error());
    }
    result += *emitted;
  }
  if (pending_uploads != 0) {
    return std::unexpected(
      "lifecycle upload is missing a completed submission");
  }
  return result;
}

std::expected<std::string, std::string> emitRunOperations(
  const vk::VulkanHostProgram& program,
  const std::map<std::uint32_t, std::size_t>& pipeline_slots) {
  std::string result;
  for (std::size_t index = 0; index < program.run_operations.size(); ++index) {
    const auto* control =
      std::get_if<vk::HostControlOp>(&program.run_operations[index]);
    if (control != nullptr) {
      const auto nextControlIs = [&](vk::HostControl action) {
        if (index + 1 >= program.run_operations.size()) {
          return false;
        }
        const auto* next =
          std::get_if<vk::HostControlOp>(&program.run_operations[index + 1]);
        return next != nullptr && next->action == action;
      };
      switch (control->action) {
        case vk::HostControl::WaitFence:
          result += "    require(commands_->wait(settings_.operation_hook));\n";
          break;
        case vk::HostControl::ResetFence:
          if (!nextControlIs(vk::HostControl::ResetCommandPool)) {
            return std::unexpected("run reset must reset fence and pool");
          }
          result += "    require(commands_->reset());\n";
          ++index;
          break;
        case vk::HostControl::BeginCommands:
          result += "    require(commands_->begin());\n";
          break;
        case vk::HostControl::EndCommands:
          result += "    require(commands_->end());\n";
          break;
        case vk::HostControl::Submit:
          if (!nextControlIs(vk::HostControl::WaitFence)) {
            return std::unexpected("run submission must wait for fence");
          }
          result +=
            "    "
            "require(commands_->submit_and_wait(settings_.operation_hook));\n";
          ++index;
          break;
        case vk::HostControl::ResetCommandPool:
          return std::unexpected("run pool reset has no preceding fence reset");
      }
      continue;
    }
    auto emitted = std::visit(
      [&](const auto& op) -> std::expected<std::string, std::string> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, vk::HostUploadOp>) {
          if (op.source != vk::HostUploadSource::Input ||
              op.source_id >= program.inputs.size()) {
            return std::unexpected("run upload has no public input");
          }
          return std::format(
            "    std::memcpy(mappedBuffer({}, UINT64_C({})).data(), "
            "inputs[{}].data(), UINT64_C({}));\n",
            op.staging_buffer.value,
            op.byte_size,
            op.source_id,
            op.byte_size);
        } else if constexpr (std::is_same_v<T, vk::HostMappedSyncOp>) {
          return std::format(
            "    require(buffers_[{}]->{}(settings_.operation_hook));\n",
            op.buffer.value,
            op.action == vk::HostMappedSync::Flush ? "flush" : "invalidate");
        } else if constexpr (std::is_same_v<T, vk::HostCopyOp>) {
          return emitCopyOperation(op);
        } else if constexpr (std::is_same_v<T, vk::HostBarrierOp>) {
          return emitBarrierOperation(op);
        } else if constexpr (std::is_same_v<T, vk::HostBindPipelineOp>) {
          return std::format(
            "    vkCmdBindPipeline(commands_->get_command_buffer(), "
            "VK_PIPELINE_BIND_POINT_COMPUTE, "
            "pipelines_[{}]->get_pipeline());\n",
            pipeline_slots.at(op.pipeline.value));
        } else if constexpr (std::is_same_v<T, vk::HostBindDescriptorsOp>) {
          const auto slot = pipeline_slots.at(op.pipeline.value);
          std::uint32_t set_count = 0;
          for (const auto& binding : program.pipelines[slot].bindings) {
            set_count = std::max(set_count, binding.set + 1);
          }
          if (op.set_count == 0 || op.first_set > set_count ||
              op.set_count > set_count - op.first_set) {
            return std::unexpected("run descriptor set range is invalid");
          }
          return std::format(
            "    vkCmdBindDescriptorSets(commands_->get_command_buffer(), "
            "VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[{0}]->get_layout(), "
            "{1}, {2}, pipelines_[{0}]->get_descriptor_sets().data() + {1}, "
            "0, nullptr);\n",
            slot,
            op.first_set,
            op.set_count);
        } else if constexpr (std::is_same_v<T, vk::HostPushConstantsOp>) {
          if (op.bytes.empty() ||
              !std::in_range<std::uint32_t>(op.bytes.size())) {
            return std::unexpected("run push constant size is invalid");
          }
          std::string text = std::format(
            "    {{\n      alignas(std::uint32_t) constexpr "
            "std::array<std::uint8_t, {}> push_constants{{{{\n",
            op.bytes.size());
          for (std::uint8_t byte : op.bytes) {
            text += std::format("        UINT8_C(0x{:02x}),\n", byte);
          }
          text += std::format(
            "      }}}};\n"
            "      vkCmdPushConstants(commands_->get_command_buffer(), "
            "pipelines_[{}]->get_layout(), VK_SHADER_STAGE_COMPUTE_BIT, "
            "{}, {}, push_constants.data());\n    }}\n",
            pipeline_slots.at(op.pipeline.value),
            op.offset,
            op.bytes.size());
          return text;
        } else if constexpr (std::is_same_v<T, vk::HostDispatchOp>) {
          return std::format(
            "    vkCmdDispatch(commands_->get_command_buffer(), {}, {}, {});\n",
            op.groups[0],
            op.groups[1],
            op.groups[2]);
        } else if constexpr (std::is_same_v<T, vk::HostPrepareOutputsOp>) {
          if (op.buffers.size() != program.outputs.size()) {
            return std::unexpected("run output preparation count is invalid");
          }
          std::string text = std::format(
            "    const std::array<std::span<std::byte>, {}> readbacks{{{{\n",
            op.buffers.size());
          for (std::size_t port = 0; port < op.buffers.size(); ++port) {
            text +=
              std::format("      mappedBuffer({}, UINT64_C({})),\n",
                          op.buffers[port].value,
                          program.buffers[op.buffers[port].value].byte_size);
          }
          text += "    }};\n    checkValidation();\n";
          return text;
        } else if constexpr (std::is_same_v<T, vk::HostPublishOp>) {
          return std::format(
            "    std::memcpy(outputs[{0}].data(), readbacks[{0}].data(), "
            "UINT64_C({1}));\n",
            op.output_index,
            op.byte_size);
        } else {
          return std::unexpected("non-run operation in run program");
        }
      },
      program.run_operations[index]);
    if (!emitted) {
      return std::unexpected(emitted.error());
    }
    result += *emitted;
  }
  return result;
}

std::expected<std::string, std::string> generateVulkanImplementation(
  const vk::VulkanHostProgram& program,
  std::span<const vk::VulkanModelShader> shaders,
  std::span<const vk::VulkanModelWeight> weights,
  bool fixed_run) {
  std::map<std::uint32_t, std::size_t> pipeline_slots;
  std::map<std::uint32_t, std::size_t> weight_slots;
  std::string constants;
  std::string requirements = std::format(
    "constexpr std::array<VulkanDeviceLimitRequirements, {}> "
    "kDeviceRequirements{{{{\n",
    program.pipelines.size());
  std::uint64_t storage_range = 0;
  for (const auto& buffer : program.buffers) {
    if (buffer.kind == vk::HostBufferKind::Device) {
      storage_range = std::max(storage_range, buffer.byte_size);
    }
  }
  std::uint64_t total_sets = 0;
  std::uint64_t total_descriptors = 0;
  for (std::size_t slot = 0; slot < program.pipelines.size(); ++slot) {
    const auto& pipeline = program.pipelines[slot];
    pipeline_slots.emplace(pipeline.id.value, slot);
    if (pipeline.interface.entry_point.size() ==
        std::numeric_limits<std::size_t>::max()) {
      return std::unexpected("lifecycle entry point length overflows size_t");
    }
    const auto shader = std::ranges::find(
      shaders, pipeline.kernel_id, &vk::VulkanModelShader::kernel_id);
    if (shader == shaders.end() || !std::in_range<std::uint32_t>(slot) ||
        !std::in_range<std::uint32_t>(pipeline.bindings.size()) ||
        pipeline.interface.local_size[1] != 1 ||
        pipeline.interface.local_size[2] != 1 || pipeline.dispatch[1] != 1 ||
        pipeline.dispatch[2] != 1) {
      return std::unexpected(
        "lifecycle pipeline has invalid payload or limits");
    }
    constants += std::format(
      "constexpr std::array<std::uint32_t, {}> kShaderWords{}{{{{\n",
      shader->words.size(),
      slot);
    for (std::uint32_t word : shader->words) {
      constants += std::format("  UINT32_C(0x{:08x}),\n", word);
    }
    constants += "}};\n";
    // A numeric byte array cannot introduce C++ tokens or extend a hex escape.
    // Explicit string_view length preserves every UTF-8 byte without strlen.
    constants +=
      std::format("constexpr std::array<char, {}> kEntryPoint{}{{{{\n",
                  pipeline.interface.entry_point.size() + 1,
                  slot);
    for (unsigned char byte : pipeline.interface.entry_point) {
      constants += std::format("  static_cast<char>(0x{:02x}),\n", byte);
    }
    constants += "  '\\0'\n}};\n";
    std::uint32_t sets = 0;
    for (const auto& binding : pipeline.bindings) {
      if (binding.set == std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
          "lifecycle descriptor set count overflows uint32");
      }
      sets = std::max(sets, binding.set + 1);
    }
    if (total_sets > std::numeric_limits<std::uint32_t>::max() - sets ||
        pipeline.bindings.size() >
          std::numeric_limits<std::uint32_t>::max() - total_descriptors) {
      return std::unexpected(
        "lifecycle descriptor pool count overflows uint32");
    }
    total_sets += sets;
    total_descriptors += pipeline.bindings.size();
    std::uint32_t push_end = 0;
    for (const auto& block : pipeline.push_constants) {
      if (block.bytes.size() >
          std::numeric_limits<std::uint32_t>::max() - block.offset) {
        return std::unexpected(
          "lifecycle push constant range overflows uint32");
      }
      push_end =
        std::max(push_end,
                 block.offset + static_cast<std::uint32_t>(block.bytes.size()));
    }
    requirements += std::format(
      "  VulkanDeviceLimitRequirements{{.local_size_x = {}, "
      ".dispatch_count_x = {}, .storage_buffer_range = UINT64_C({}), "
      ".push_constant_bytes = {}, .descriptor_set_count = {}, "
      ".storage_buffer_descriptor_count = {}}},\n",
      pipeline.interface.local_size[0],
      pipeline.dispatch[0],
      storage_range,
      push_end,
      sets,
      pipeline.bindings.size());
  }
  requirements += "}};\n";
  constants += requirements;
  for (std::size_t slot = 0; slot < weights.size(); ++slot) {
    weight_slots.emplace(weights[slot].resource_id, slot);
    constants +=
      std::format("constexpr std::array<std::byte, {}> kWeightBytes{}{{{{\n",
                  weights[slot].bytes.size(),
                  slot);
    for (std::uint8_t byte : weights[slot].bytes) {
      constants += std::format("  std::byte{{0x{:02x}}},\n", byte);
    }
    constants += "}};\n";
  }
  constants += emitPorts("kInputs", program.inputs);
  constants += emitPorts("kOutputs", program.outputs);
  auto create_operations = emitCreateOperations(program, weight_slots);
  if (!create_operations) {
    return std::unexpected(create_operations.error());
  }
  std::string run_operations;
  if (fixed_run) {
    auto emitted = emitRunOperations(program, pipeline_slots);
    if (!emitted) {
      return std::unexpected(emitted.error());
    }
    run_operations = std::move(*emitted);
  }
  const std::string_view run_method = fixed_run ? R"run(
            ncnn_vk_status run(
              const std::array<std::span<const float>, @INPUT_COUNT@>& inputs,
              const std::array<std::span<float>, @OUTPUT_COUNT@>& outputs,
              char* diagnostic,
              std::size_t capacity) {
              const std::scoped_lock lock(run_mutex_);
              if (unusable_status_ != NCNN_VK_STATUS_SUCCESS) {
                return report(
                  unusable_status_, diagnostic, capacity,
                  "Vulkan context cannot run after a previous execution failure");
              }
              try {
                @RUN_OPERATIONS@
                return NCNN_VK_STATUS_SUCCESS;
              } catch (const LifecycleFailure& failure) {
                unusable_status_ = failure.get_error().status;
                return report(unusable_status_, diagnostic, capacity,
                              failure.get_error().diagnostic);
              } catch (const std::bad_alloc&) {
                unusable_status_ = NCNN_VK_STATUS_OUT_OF_MEMORY;
                return report(unusable_status_, diagnostic, capacity,
                              "host allocation failed during Vulkan run");
              } catch (const std::exception& failure) {
                unusable_status_ = NCNN_VK_STATUS_API_ERROR;
                return report(unusable_status_, diagnostic, capacity,
                              failure.what());
              } catch (...) {
                unusable_status_ = NCNN_VK_STATUS_API_ERROR;
                return report(unusable_status_, diagnostic, capacity,
                              "unknown Vulkan run failure");
              }
            }
    )run"
                                                : R"run(
            ncnn_vk_status rejectRun(char* diagnostic, std::size_t capacity) {
              const std::scoped_lock lock(run_mutex_);
              return report(
                NCNN_VK_STATUS_API_ERROR,
                diagnostic,
                capacity,
                "private lifecycle prototype does not implement model run");
            }
    )run";
  const std::string_view run_helpers = !fixed_run ? "" : R"run(
            std::span<std::byte> mappedBuffer(std::size_t slot,
                                               std::uint64_t bytes) {
              if (slot >= buffers_.size() || !buffers_[slot] ||
                  bytes > std::numeric_limits<std::size_t>::max()) {
                throw LifecycleFailure(
                  {.status = NCNN_VK_STATUS_API_ERROR,
                   .diagnostic = "invalid Vulkan staging buffer extent"});
              }
              auto mapped = buffers_[slot]->get_mapped();
              if (mapped.data() == nullptr || bytes > mapped.size()) {
                throw LifecycleFailure(
                  {.status = NCNN_VK_STATUS_API_ERROR,
                   .diagnostic = "Vulkan staging buffer is not fully mapped"});
              }
              return mapped.first(static_cast<std::size_t>(bytes));
            }

            void checkValidation() const {
              if (device_->get_validation_error_count() != 0) {
                throw LifecycleFailure(
                  {.status = NCNN_VK_STATUS_API_ERROR,
                   .diagnostic = "Vulkan run validation failed"});
              }
            }
    )run";

  // Raw templates keep the emitted implementation reviewable as C++, while
  // model-derived text is restricted to checked numeric values and identifiers.
  // The distinct raw delimiter keeps clang-format from parsing placeholders as
  // incomplete C++ and splitting their replacement tokens.
  std::string source =
    R"lifecycle(#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <expected>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>

#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"
#include "@HEADER@"
#ifdef NCNN_VULKAN_LIFECYCLE_TEST
#include "lifecycle_test_hooks.hpp"
#endif

          namespace {
          using ncnn_mlir::vulkan::VulkanAllocationBudget;
          using ncnn_mlir::vulkan::VulkanBuffer;
          using ncnn_mlir::vulkan::VulkanBufferCreateInfo;
          using ncnn_mlir::vulkan::VulkanCommandResources;
          using ncnn_mlir::vulkan::VulkanComputePipeline;
          using ncnn_mlir::vulkan::VulkanDescriptorPool;
          using ncnn_mlir::vulkan::VulkanDeviceContext;
          using ncnn_mlir::vulkan::VulkanDeviceError;
          using ncnn_mlir::vulkan::VulkanDeviceLimitRequirements;
          using ncnn_mlir::vulkan::VulkanDeviceOptions;
          using ncnn_mlir::vulkan::VulkanPipelineBinding;
          using ncnn_mlir::vulkan::VulkanPushConstantRange;

          struct Port final {
            std::uint32_t device_buffer;
            std::uint32_t staging_buffer;
            std::uint64_t elements;
          };
          @CONSTANTS@


          class LifecycleFailure final : public std::exception {
           public:
            explicit LifecycleFailure(VulkanDeviceError error)
              : error_(std::move(error)) {}
            const char* what() const noexcept override {
              return error_.diagnostic.c_str();
            }
            const VulkanDeviceError& get_error() const noexcept { return error_; }

           private:
            VulkanDeviceError error_;
          };

          void require(std::expected<void, VulkanDeviceError> result) {
            if (!result) {
              throw LifecycleFailure(std::move(result.error()));
            }
          }

          template <class T>
          T take(std::expected<T, VulkanDeviceError> result) {
            if (!result) {
              throw LifecycleFailure(std::move(result.error()));
            }
            return std::move(*result);
          }

          ncnn_vk_status report(ncnn_vk_status status,
                                char* diagnostic,
                                std::size_t capacity,
                                std::string_view message,
                                std::string_view suffix = {}) noexcept {
            if (diagnostic != nullptr && capacity != 0) {
              const auto count = std::min(message.size(), capacity - 1);
              std::copy_n(message.data(), count, diagnostic);
              const auto trailing = std::min(suffix.size(), capacity - 1 - count);
              if (trailing != 0) {
                std::copy_n(suffix.data(), trailing, diagnostic + count);
              }
              diagnostic[count + trailing] = '\0';
            }
            return status;
          }

          ncnn_vk_status validateDiagnostic(char* diagnostic,
                                            std::size_t capacity) noexcept {
            if (capacity != 0 && diagnostic == nullptr) {
              return NCNN_VK_STATUS_INVALID_ARGUMENT;
            }
            if (capacity != 0) {
              diagnostic[0] = '\0';
            }
            return NCNN_VK_STATUS_SUCCESS;
          }
          }  // namespace

          struct @CONTEXT@ final {
           public:
            struct CreateSettings final {
              std::string_view device_name_match;
              bool enable_validation;
              bool force_mapped_memory_sync;
              std::uint64_t max_context_memory_bytes;
              ncnn_mlir::vulkan::detail::VulkanOperationHook operation_hook;
            };

            explicit @CONTEXT@ (CreateSettings settings)
              : settings_(std::move(settings)),
                device_(),
                commands_(),
                descriptor_pool_(),
                buffers_(),
                pipelines_(),
                peak_memory_bytes_(0),
                weight_upload_count_(0),
                validation_error_count_(0),
                destroyed_(false),
                destroy_status_(NCNN_VK_STATUS_SUCCESS),
                run_mutex_()@RUN_STATE_INITIALIZER@ {}
            @CONTEXT@ (const @CONTEXT@ &) = delete;
            @CONTEXT@ & operator=(const @CONTEXT@ &) = delete;
            ~@CONTEXT@ () {
              (void)destroy();
            }

            void initialize(std::uint32_t device_index) {
              if (settings_.max_context_memory_bytes == 0) {
                throw LifecycleFailure(
                  {.status = NCNN_VK_STATUS_INVALID_ARGUMENT,
                   .diagnostic = "context memory budget is zero"});
              }
              device_.emplace(take(VulkanDeviceContext::create(
                VulkanDeviceOptions{
                  .device_index = device_index,
                  .device_name_match = settings_.device_name_match,
                  .enable_validation = settings_.enable_validation},
                settings_.operation_hook)));
              require(ncnn_mlir::vulkan::validateDeviceLimits(
                device_->get_properties(), kDeviceRequirements));
              commands_.emplace(take(VulkanCommandResources::create(
                *device_, settings_.operation_hook)));
              @PREPARE_BUFFERS@ checkAllocationBudget();
              @ALLOCATE_BUFFERS@ descriptor_pool_.emplace(
                take(VulkanDescriptorPool::create(*device_,
                                                  @POOL_SETS@ ,
                                                  @POOL_DESCRIPTORS@ ,
                                                  settings_.operation_hook)));
              @CREATE_PIPELINES@  @CREATE_OPERATIONS@  if (device_->get_validation_error_count() != 0) {
                throw LifecycleFailure(
                  {.status = NCNN_VK_STATUS_API_ERROR,
                   .diagnostic = "Vulkan lifecycle validation failed"});
              }
            }

            @RUN_METHOD@

            ncnn_vk_status destroy() noexcept {
              if (destroyed_) {
                return destroy_status_;
              }
              destroyed_ = true;
              if (device_) {
                // Never release an in-flight allocation merely because a wait
                // failed. With an infinite wait, only completion or device loss
                // permits teardown.
                for (;;) {
                  try {
                    auto waited = device_->wait_idle();
                    if (waited) {
                      break;
                    }
                    remember(waited.error().status);
                    if (waited.error().status == NCNN_VK_STATUS_DEVICE_LOST) {
                      break;
                    }
                  } catch (...) {
                    remember(NCNN_VK_STATUS_API_ERROR);
                  }
                  std::this_thread::yield();
                }
              }
              @DESTROY_PIPELINES@ release(descriptor_pool_);
              @DESTROY_BUFFERS@ release(commands_);
              if (device_) {
                validation_error_count_ = device_->get_validation_error_count();
                if (validation_error_count_ != 0) {
                  remember(NCNN_VK_STATUS_API_ERROR);
                }
                try {
                  auto result = device_->destroy();
                  if (!result) {
                    remember(result.error().status);
                  }
                  validation_error_count_ = device_->get_validation_error_count();
                  if (validation_error_count_ != 0) {
                    remember(NCNN_VK_STATUS_API_ERROR);
                  }
                } catch (...) {
                  remember(NCNN_VK_STATUS_API_ERROR);
                }
                device_.reset();
              }
              return destroy_status_;
            }

            std::uint64_t get_weight_upload_count() const noexcept {
              return weight_upload_count_;
            }
            std::uint64_t get_peak_memory_bytes() const noexcept {
              return peak_memory_bytes_;
            }
            std::uint32_t get_validation_error_count() const noexcept {
              return device_ ? device_->get_validation_error_count()
                             : validation_error_count_;
            }

           private:
            @RUN_HELPERS@
            void prepareBuffer(std::size_t slot,
                               VkDeviceSize size,
                               VkBufferUsageFlags usage,
                               bool host_visible) {
              buffers_[slot].emplace(take(VulkanBuffer::prepare(
                *device_,
                VulkanBufferCreateInfo{.size = size,
                                       .usage = usage,
                                       .host_visible = host_visible,
                                       .force_mapped_memory_sync =
                                         settings_.force_mapped_memory_sync},
                settings_.operation_hook)));
            }

            void addAllocation(std::uint64_t& total, std::uint64_t amount) {
              if (amount > std::numeric_limits<std::uint64_t>::max() - total) {
                throw LifecycleFailure(
                  {.status = NCNN_VK_STATUS_OUT_OF_MEMORY,
                   .diagnostic = "allocation requirements overflow uint64"});
              }
              total += amount;
            }

            void checkAllocationBudget() {
              std::uint64_t persistent = 0;
              std::uint64_t temporary = 0;
              @COUNT_ALLOCATIONS@ std::uint64_t peak = persistent;
              addAllocation(peak, temporary);
              require(ncnn_mlir::vulkan::validateAllocationBudget(
                VulkanAllocationBudget {
                  .persistent_bytes = persistent,
                  .temporary_staging_bytes = temporary,
                  .allocation_count = @BUFFER_COUNT@
                },
                device_->get_properties().limits.maxMemoryAllocationCount,
                settings_.max_context_memory_bytes));
              peak_memory_bytes_ = peak;
            }

            void remember(ncnn_vk_status status) noexcept {
              if (destroy_status_ == NCNN_VK_STATUS_SUCCESS) {
                destroy_status_ = status;
              }
            }

            template <class T>
            void release(std::optional<T> & object) noexcept {
              if (object) {
                try {
                  auto result = object->destroy();
                  if (!result) {
                    remember(result.error().status);
                  }
                } catch (...) {
                  remember(NCNN_VK_STATUS_API_ERROR);
                }
                object.reset();
              }
            }

            CreateSettings settings_;
            std::optional<VulkanDeviceContext> device_;
            std::optional<VulkanCommandResources> commands_;
            std::optional<VulkanDescriptorPool> descriptor_pool_;
            std::array<std::optional<VulkanBuffer>, @BUFFER_COUNT@ > buffers_;
            std::array<std::optional<VulkanComputePipeline>, @PIPELINE_COUNT@ >
              pipelines_;
            std::uint64_t peak_memory_bytes_;
            std::uint64_t weight_upload_count_;
            std::uint32_t validation_error_count_;
            bool destroyed_;
            ncnn_vk_status destroy_status_;
            std::mutex run_mutex_;
            @RUN_STATE@
          };

          namespace {
          @CONTEXT@::CreateSettings getCreateSettings() {
            @CONTEXT@::CreateSettings result{
              .device_name_match = "",
#ifdef NCNN_VULKAN_LIFECYCLE_VALIDATION
              .enable_validation = true,
#else
              .enable_validation = false,
#endif
              .force_mapped_memory_sync = false,
              .max_context_memory_bytes =
                ncnn_mlir::vulkan::kVulkanDeviceHelperDefaultBudgetBytes,
              .operation_hook = {}};
#ifdef NCNN_VULKAN_LIFECYCLE_TEST
            auto settings = ncnn_mlir::vulkan::test::getLifecycleTestSettings();
            result.device_name_match = settings.device_name_match;
            result.enable_validation = settings.enable_validation;
            result.force_mapped_memory_sync = settings.force_mapped_memory_sync;
            result.max_context_memory_bytes = settings.max_context_memory_bytes;
            result.operation_hook = std::move(settings.operation_hook);
#endif
            return result;
          }

          ncnn_vk_status failedCreate(ncnn_vk_status status,
                                      std::string_view message,
                                      std::unique_ptr<@CONTEXT@ >& context,
                                      char* diagnostic,
                                      std::size_t capacity) noexcept {
            const auto cleanup = context ? context->destroy()
                               : ncnn_vk_status{NCNN_VK_STATUS_SUCCESS};
            return report(status,
                          diagnostic,
                          capacity,
                          message,
                          cleanup == NCNN_VK_STATUS_SUCCESS
                            ? ""
                            : "; partial cleanup failed");
          }
          }  // namespace

          extern "C" ncnn_vk_status @CREATE@ (const ncnn_vk_options* options,
                                              @CONTEXT@ ** context,
                                              char* diagnostic,
                                              std::size_t capacity) {
            if (context == nullptr) {
              return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                            diagnostic,
                            capacity,
                            "missing context output");
            }
            *context = nullptr;
            const auto diagnostic_status = validateDiagnostic(diagnostic, capacity);
            if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
              return diagnostic_status;
            }
            if (options == nullptr ||
                options->struct_size < NCNN_VK_OPTIONS_V1_SIZE ||
                options->abi_version != NCNN_VK_ABI_VERSION ||
                options->flags != 0) {
              return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                            diagnostic,
                            capacity,
                            "invalid Vulkan options");
            }
            std::unique_ptr<@CONTEXT@ > created;
            try {
              created = std::make_unique<@CONTEXT@ >(getCreateSettings());
              created->initialize(options->device_index);
              *context = created.release();
              return NCNN_VK_STATUS_SUCCESS;
            } catch (const LifecycleFailure& failure) {
              return failedCreate(failure.get_error().status,
                                  failure.get_error().diagnostic,
                                  created,
                                  diagnostic,
                                  capacity);
            } catch (const std::bad_alloc&) {
              return failedCreate(
                NCNN_VK_STATUS_OUT_OF_MEMORY,
                "host allocation failed while creating Vulkan context",
                created,
                diagnostic,
                capacity);
            } catch (const std::exception& failure) {
              return failedCreate(NCNN_VK_STATUS_API_ERROR,
                                  failure.what(),
                                  created,
                                  diagnostic,
                                  capacity);
            } catch (...) {
              return failedCreate(NCNN_VK_STATUS_API_ERROR,
                                  "unknown Vulkan create failure",
                                  created,
                                  diagnostic,
                                  capacity);
            }
          }

          extern "C" ncnn_vk_status @RUN@ (@RUN_ARGUMENTS@ ) {
            const auto diagnostic_status =
              validateDiagnostic(diagnostic, diagnostic_capacity);
            if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
              return diagnostic_status;
            }
            if (context == nullptr @NULL_CHECKS@ ) {
              return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                            diagnostic,
                            diagnostic_capacity,
                            "invalid run argument");
            }
            if (@COUNT_CHECKS@ ) {
              return report(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
                            diagnostic,
                            diagnostic_capacity,
                            "element count does not match static shape");
            }
            try {
              @RUN_INVOKE@
            } catch (const std::bad_alloc&) {
              return report(NCNN_VK_STATUS_OUT_OF_MEMORY,
                            diagnostic,
                            diagnostic_capacity,
                            "host allocation failed before Vulkan run");
            } catch (const std::exception& failure) {
              return report(NCNN_VK_STATUS_API_ERROR,
                            diagnostic,
                            diagnostic_capacity,
                            failure.what());
            } catch (...) {
              return report(NCNN_VK_STATUS_API_ERROR,
                            diagnostic,
                            diagnostic_capacity,
                            "unknown failure before Vulkan run");
            }
          }

          extern "C" ncnn_vk_status @DESTROY@ (@CONTEXT@ * context) {
            const std::unique_ptr<@CONTEXT@ > owned(context);
            return owned ? owned->destroy() : ncnn_vk_status{NCNN_VK_STATUS_SUCCESS};
          }
    )lifecycle";

  std::string prepare;
  std::string allocate;
  std::string count_allocations;
  for (const auto& object : program.initialize) {
    if (object.kind != vk::HostObjectKind::Buffer) {
      continue;
    }
    const auto& buffer = program.buffers[object.id];
    const bool device = buffer.kind == vk::HostBufferKind::Device;
    const std::string_view usage =
      device ? "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | "
               "VK_BUFFER_USAGE_TRANSFER_SRC_BIT "
               "| VK_BUFFER_USAGE_TRANSFER_DST_BIT"
      : buffer.kind == vk::HostBufferKind::WeightStaging
        ? "VK_BUFFER_USAGE_TRANSFER_SRC_BIT"
        : "VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT";
    prepare += std::format("    prepareBuffer({}, UINT64_C({}), {}, {});\n",
                           object.id,
                           buffer.byte_size,
                           usage,
                           device ? "false" : "true");
    allocate += std::format(
      "    require(buffers_[{}]->allocate_and_bind({}, "
      "settings_.operation_hook));\n",
      object.id,
      device ? "false" : "true");
    count_allocations += std::format(
      "    addAllocation({}, buffers_[{}]->get_allocation_size());\n",
      buffer.scope == vk::HostBufferScope::CreateUpload ? "temporary"
                                                        : "persistent",
      object.id);
  }
  std::string create_pipelines;
  for (const auto& object : program.initialize) {
    if (object.kind != vk::HostObjectKind::Pipeline) {
      continue;
    }
    const std::size_t slot = pipeline_slots.at(object.id);
    const auto& pipeline = program.pipelines[slot];
    create_pipelines += std::format(
      "    {{\n      const std::array<VulkanPipelineBinding, {}> "
      "bindings{{{{\n",
      pipeline.bindings.size());
    for (const auto& binding : pipeline.bindings) {
      create_pipelines += std::format(
        "        VulkanPipelineBinding{{.set = {}, .binding = {}, "
        ".buffer = buffers_[{}]->get_handle(), .buffer_size = UINT64_C({}), "
        ".offset = 0, .range = UINT64_C({})}},\n",
        binding.set,
        binding.binding,
        binding.buffer.value,
        program.buffers[binding.buffer.value].byte_size,
        program.buffers[binding.buffer.value].byte_size);
    }
    create_pipelines += "      }};\n";
    std::uint32_t push_end = 0;
    for (const auto& block : pipeline.push_constants) {
      push_end =
        std::max(push_end,
                 block.offset + static_cast<std::uint32_t>(block.bytes.size()));
    }
    create_pipelines += std::format(
      "      const std::array<VulkanPushConstantRange, {}> "
      "push_constants{{{{\n",
      push_end == 0 ? 0 : 1);
    if (push_end != 0) {
      create_pipelines += std::format(
        "        VulkanPushConstantRange{{.stage_flags = "
        "VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = {}}},\n",
        push_end);
    }
    create_pipelines += "      }};\n";
    create_pipelines += std::format(
      "      pipelines_[{0}].emplace(take(VulkanComputePipeline::create(\n"
      "        *device_, *descriptor_pool_, kShaderWords{0}, "
      "std::string_view(kEntryPoint{0}.data(), {1}),\n"
      "        bindings, push_constants, settings_.operation_hook)));\n"
      "    }}\n",
      slot,
      pipeline.interface.entry_point.size());
  }
  std::string destroy_pipelines;
  for (const auto& object : program.destroy) {
    if (object.kind == vk::HostObjectKind::Pipeline) {
      destroy_pipelines += std::format("    release(pipelines_[{}]);\n",
                                       pipeline_slots.at(object.id));
    }
  }
  std::string destroy_buffers;
  // Temporary upload buffers are absent from the steady-state destroy list,
  // but must also be owned during partial-create failure.
  for (const auto& object : std::views::reverse(program.initialize)) {
    if (object.kind == vk::HostObjectKind::Buffer) {
      destroy_buffers += std::format("    release(buffers_[{}]);\n", object.id);
    }
  }
  std::string arguments = program.model + "_vk_context* context";
  std::string null_checks;
  std::string count_checks;
  for (const bool input : {true, false}) {
    const auto& ports = input ? program.inputs : program.outputs;
    for (std::size_t index = 0; index < ports.size(); ++index) {
      const auto parameter =
        std::format("{}{}", input ? "input" : "output", index + 1);
      arguments += std::format(", {0}float* {1}, std::size_t {1}_elements",
                               input ? "const " : "",
                               parameter);
      null_checks += std::format(" || {} == nullptr", parameter);
      if (!count_checks.empty()) {
        count_checks += " || ";
      }
      count_checks += std::format("{}_elements != {}[{}].elements",
                                  parameter,
                                  input ? "kInputs" : "kOutputs",
                                  index);
    }
  }
  arguments += ", char* diagnostic, std::size_t diagnostic_capacity";
  std::string run_invoke =
    "return context->rejectRun(diagnostic, diagnostic_capacity);";
  if (fixed_run) {
    run_invoke.clear();
    for (const bool input : {true, false}) {
      const auto& ports = input ? program.inputs : program.outputs;
      const std::string_view name = input ? "inputs" : "outputs";
      const std::string_view type = input ? "const float" : "float";
      run_invoke += std::format("const std::array<std::span<{}>, {}> {}{{{{\n",
                                type,
                                ports.size(),
                                name);
      for (std::size_t index = 0; index < ports.size(); ++index) {
        const auto parameter =
          std::format("{}{}", input ? "input" : "output", index + 1);
        run_invoke += std::format(
          "  std::span<{0}>({1}, {1}_elements),\n", type, parameter);
      }
      run_invoke += "}};\n";
    }
    run_invoke +=
      "return context->run(inputs, outputs, diagnostic, diagnostic_capacity);";
  }
  replaceToken(source, "@RUN_METHOD@", run_method);
  replaceToken(source, "@RUN_HELPERS@", run_helpers);
  replaceToken(source, "@RUN_OPERATIONS@", run_operations);
  replaceToken(source, "@RUN_INVOKE@", run_invoke);
  replaceToken(source, "@INPUT_COUNT@", std::to_string(program.inputs.size()));
  replaceToken(
    source, "@OUTPUT_COUNT@", std::to_string(program.outputs.size()));
  replaceToken(source,
               "@RUN_STATE_INITIALIZER@",
               fixed_run ? ", unusable_status_(NCNN_VK_STATUS_SUCCESS)" : "");
  replaceToken(
    source, "@RUN_STATE@", fixed_run ? "ncnn_vk_status unusable_status_;" : "");
  replaceToken(source, "@HEADER@", program.model + "_vulkan.h");
  replaceToken(source, "@CONSTANTS@", constants);
  replaceToken(source, "@PREPARE_BUFFERS@", prepare);
  replaceToken(source, "@ALLOCATE_BUFFERS@", allocate);
  replaceToken(source, "@COUNT_ALLOCATIONS@", count_allocations);
  replaceToken(source, "@POOL_SETS@", std::to_string(total_sets));
  replaceToken(source, "@POOL_DESCRIPTORS@", std::to_string(total_descriptors));
  replaceToken(source, "@CREATE_PIPELINES@", create_pipelines);
  replaceToken(source, "@CREATE_OPERATIONS@", *create_operations);
  replaceToken(source, "@DESTROY_PIPELINES@", destroy_pipelines);
  replaceToken(source, "@DESTROY_BUFFERS@", destroy_buffers);
  replaceToken(
    source, "@BUFFER_COUNT@", std::to_string(program.buffers.size()));
  replaceToken(
    source, "@PIPELINE_COUNT@", std::to_string(program.pipelines.size()));
  replaceToken(source, "@CONTEXT@", program.model + "_vk_context");
  replaceToken(source, "@CREATE@", program.model + "_vk_create");
  replaceToken(source, "@RUN@", program.model + "_vk_run");
  replaceToken(source, "@DESTROY@", program.model + "_vk_destroy");
  replaceToken(source, "@RUN_ARGUMENTS@", arguments);
  replaceToken(source, "@NULL_CHECKS@", null_checks);
  replaceToken(
    source, "@COUNT_CHECKS@", count_checks.empty() ? "false" : count_checks);
  if (source.find('@') != std::string::npos) {
    return std::unexpected("unresolved lifecycle code generation token");
  }
  return source;
}

}  // namespace

std::expected<std::string, std::string> generateVulkanLifecycleImplementation(
  const vk::VulkanHostProgram& program,
  std::span<const vk::VulkanModelShader> shaders,
  std::span<const vk::VulkanModelWeight> weights) {
  return generateVulkanImplementation(program, shaders, weights, false);
}

std::expected<std::string, std::string> generateVulkanHostImplementation(
  const vk::VulkanHostProgram& program,
  std::span<const vk::VulkanModelShader> shaders,
  std::span<const vk::VulkanModelWeight> weights) {
  return generateVulkanImplementation(program, shaders, weights, true);
}

}  // namespace ncnn_compile
