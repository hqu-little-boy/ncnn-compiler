#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"
#include "ncnn-mlir/Support/VulkanShaderInterface.hpp"

namespace ncnn_mlir::vulkan {

inline constexpr std::string_view kHostProgramSchema =
  "ncnn-vulkan-host-program/v1";
inline constexpr std::string_view kHostProgramCodegenVersion =
  "ncnn-vulkan-host-codegen/v1";
inline constexpr std::string_view kHostProgramHelperContract =
  "ncnn-vulkan-device-helper-contract/v1";
inline constexpr std::string_view kHostProgramSynchronizationVersion =
  "ncnn-vulkan-host-sync/v1";
inline constexpr std::string_view kHostProgramCanonicalEncoding =
  "length-prefixed-canonical-json/v1";

// Compiler-only IDs refer to declarations, never to Vulkan handles or memory
// types. Staging is distinct from device storage even for the same tensor.
struct HostBufferId final {
  std::uint32_t value;
  bool operator==(const HostBufferId&) const = default;
};
struct HostPipelineId final {
  std::uint32_t value;
  bool operator==(const HostPipelineId&) const = default;
};
enum class HostBufferKind : std::uint8_t {
  Device,
  InputStaging,
  OutputStaging,
  WeightStaging
};
enum class HostBufferScope : std::uint8_t { Context, CreateUpload };
struct HostBuffer final {
  HostBufferId id;
  std::uint32_t resource_id;
  HostBufferKind kind;
  HostBufferScope scope;
  std::uint64_t byte_size;
};
struct HostPort final {
  std::uint32_t resource_id;
  HostBufferId device_buffer;
  HostBufferId staging_buffer;
  std::uint64_t elements;
};
struct HostBinding final {
  std::uint32_t set;
  std::uint32_t binding;
  HostBufferId buffer;
  ResourceAccess access;
};
struct HostPipeline final {
  HostPipelineId id;
  std::uint32_t kernel_id;
  std::string binary_symbol;
  std::string shader_sha256;
  std::uint64_t shader_size_bytes;
  VulkanShaderInterface interface;
  std::vector<HostBinding> bindings;
  std::vector<PushConstantBlock> push_constants;
  std::array<std::uint32_t, 3> dispatch;
};
struct HostWeight final {
  std::uint32_t resource_id;
  std::uint64_t byte_size;
  std::string logical_sha256;
  std::string physical_sha256;
};
enum class HostStage : std::uint8_t { Host, Transfer, Compute };
enum class HostAccess : std::uint8_t {
  HostWrite,
  TransferRead,
  TransferWrite,
  ShaderRead,
  ShaderWrite,
  ShaderReadWrite,
  HostRead
};
struct HostBufferBarrier final {
  HostBufferId buffer;
  HostAccess source_access;
  HostAccess destination_access;
  std::uint64_t offset;
  std::uint64_t byte_size;
};
struct HostBarrierOp final {
  HostStage source_stage;
  HostStage destination_stage;
  std::vector<HostBufferBarrier> buffers;
};
struct HostCopyOp final {
  HostBufferId source;
  HostBufferId destination;
  std::uint64_t source_offset;
  std::uint64_t destination_offset;
  std::uint64_t byte_size;
};
enum class HostUploadSource : std::uint8_t { Input, Weight };
struct HostUploadOp final {
  HostBufferId staging_buffer;
  HostUploadSource source;
  std::uint32_t source_id;
  std::uint64_t byte_size;
};
enum class HostMappedSync : std::uint8_t { Flush, Invalidate };
struct HostMappedSyncOp final {
  HostBufferId buffer;
  HostMappedSync action;
  // Mapping and non-coherent synchronization cover the whole allocation. Its
  // actual size and nonCoherentAtomSize remain create-time device queries.
};
enum class HostControl : std::uint8_t {
  WaitFence,
  ResetFence,
  ResetCommandPool,
  BeginCommands,
  EndCommands,
  Submit
};
struct HostControlOp final {
  HostControl action;
};
struct HostBindPipelineOp final {
  HostPipelineId pipeline;
};
struct HostBindDescriptorsOp final {
  HostPipelineId pipeline;
  std::uint32_t first_set;
  std::uint32_t set_count;
};
struct HostPushConstantsOp final {
  HostPipelineId pipeline;
  std::uint32_t offset;
  std::vector<std::uint8_t> bytes;
};
struct HostDispatchOp final {
  HostPipelineId pipeline;
  std::array<std::uint32_t, 3> groups;
};
struct HostPrepareOutputsOp final {
  std::vector<HostBufferId> buffers;
};
struct HostPublishOp final {
  HostBufferId staging_buffer;
  std::uint32_t output_index;
  std::uint64_t byte_size;
};
struct HostReleaseBufferOp final {
  HostBufferId buffer;
};
using HostOperation = std::variant<HostBarrierOp,
                                   HostCopyOp,
                                   HostUploadOp,
                                   HostMappedSyncOp,
                                   HostControlOp,
                                   HostBindPipelineOp,
                                   HostBindDescriptorsOp,
                                   HostPushConstantsOp,
                                   HostDispatchOp,
                                   HostPrepareOutputsOp,
                                   HostPublishOp,
                                   HostReleaseBufferOp>;
enum class HostObjectKind : std::uint8_t { Buffer, Pipeline };
struct HostObject final {
  HostObjectKind kind;
  std::uint32_t id;
};

// A value DTO, not an execution API. Only compiler packaging and tests may use
// it; the device helper must never interpret this program at inference time.
struct VulkanHostProgram final {
  std::string schema;
  std::string codegen_version;
  std::string helper_contract;
  std::string synchronization_version;
  std::string canonical_encoding;
  std::uint32_t abi_version;
  std::string model;
  std::string host_triple;
  std::string device_target;
  std::vector<TensorResource> resources;
  std::vector<HostBuffer> buffers;
  std::vector<HostPort> inputs;
  std::vector<HostPort> outputs;
  std::vector<HostPipeline> pipelines;
  std::vector<HostWeight> weights;
  std::vector<HostObject> initialize;
  std::vector<HostObject> destroy;
  std::vector<HostOperation> create_operations;
  std::vector<HostOperation> run_operations;
  std::string identity;
};

[[nodiscard]] std::expected<VulkanHostProgram, std::string>
makeVulkanHostProgram(const VulkanExecutionPlan& plan,
                      std::span<const VulkanModelShader> shaders,
                      std::span<const VulkanModelWeight> weights);
[[nodiscard]] std::expected<void, std::string> verifyVulkanHostProgram(
  const VulkanHostProgram& program,
  const VulkanExecutionPlan& plan,
  std::span<const VulkanModelShader> shaders,
  std::span<const VulkanModelWeight> weights);
// Identity excludes the identity field itself and both ExecutionPlan hashes.
// Encoding: u64le(domain length), domain, u64le(canonical JSON length), JSON.
[[nodiscard]] std::expected<std::string, std::string>
canonicalVulkanHostProgram(const VulkanHostProgram& program);
[[nodiscard]] std::expected<std::string, std::string>
calculateVulkanHostProgramIdentity(const VulkanHostProgram& program);
[[nodiscard]] std::expected<std::string, std::string>
serializeVulkanHostProgram(const VulkanHostProgram& program);

}  // namespace ncnn_mlir::vulkan
