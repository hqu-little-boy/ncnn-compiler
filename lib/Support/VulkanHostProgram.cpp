#include "ncnn-mlir/Support/VulkanHostProgram.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Triple.h"

namespace ncnn_mlir::vulkan {
namespace {

using Object = llvm::json::Object;
using Array = llvm::json::Array;
using Value = llvm::json::Value;
constexpr std::string_view kCanonicalDomain =
  "ncnn-vulkan-host-program-canonical/v1";

std::string hexBytes(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (std::uint8_t byte : bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

std::string sha256(std::string_view bytes) {
  llvm::SHA256 hash;
  hash.update(llvm::StringRef(bytes.data(), bytes.size()));
  return hexBytes(hash.final());
}

std::string shaderHash(std::span<const std::uint32_t> words) {
  llvm::SHA256 hash;
  for (std::uint32_t word : words) {
    std::array<std::uint8_t, 4> bytes{};
    for (unsigned index = 0; index < bytes.size(); ++index) {
      bytes[index] = static_cast<std::uint8_t>(word >> (index * 8));
    }
    hash.update(bytes);
  }
  return hexBytes(hash.final());
}

std::string weightHash(std::span<const std::uint8_t> bytes) {
  return hexBytes(llvm::SHA256::hash(bytes));
}

std::string name(ResourceKind kind) {
  switch (kind) {
    case ResourceKind::Input:
      return "input";
    case ResourceKind::Output:
      return "output";
    case ResourceKind::Intermediate:
      return "intermediate";
    case ResourceKind::Weight:
      return "weight";
  }
  return "invalid";
}
std::string name(ResourceAccess access) {
  switch (access) {
    case ResourceAccess::Read:
      return "read";
    case ResourceAccess::Write:
      return "write";
  }
  return "invalid";
}
std::string name(HostBufferKind kind) {
  switch (kind) {
    case HostBufferKind::Device:
      return "device";
    case HostBufferKind::InputStaging:
      return "input-staging";
    case HostBufferKind::OutputStaging:
      return "output-staging";
    case HostBufferKind::WeightStaging:
      return "weight-staging";
  }
  return "invalid";
}
std::string name(HostBufferScope scope) {
  switch (scope) {
    case HostBufferScope::Context:
      return "context";
    case HostBufferScope::CreateUpload:
      return "create-upload";
  }
  return "invalid";
}
std::string name(HostStage stage) {
  switch (stage) {
    case HostStage::Host:
      return "host";
    case HostStage::Transfer:
      return "transfer";
    case HostStage::Compute:
      return "compute";
  }
  return "invalid";
}
std::string name(HostAccess access) {
  switch (access) {
    case HostAccess::HostWrite:
      return "host-write";
    case HostAccess::TransferRead:
      return "transfer-read";
    case HostAccess::TransferWrite:
      return "transfer-write";
    case HostAccess::ShaderRead:
      return "shader-read";
    case HostAccess::ShaderWrite:
      return "shader-write";
    case HostAccess::ShaderReadWrite:
      return "shader-read-write";
    case HostAccess::HostRead:
      return "host-read";
  }
  return "invalid";
}
std::string name(HostUploadSource source) {
  switch (source) {
    case HostUploadSource::Input:
      return "input";
    case HostUploadSource::Weight:
      return "weight";
  }
  return "invalid";
}
std::string name(HostMappedSync action) {
  switch (action) {
    case HostMappedSync::Flush:
      return "flush";
    case HostMappedSync::Invalidate:
      return "invalidate";
  }
  return "invalid";
}
std::string name(HostControl action) {
  switch (action) {
    case HostControl::WaitFence:
      return "wait-fence";
    case HostControl::ResetFence:
      return "reset-fence";
    case HostControl::ResetCommandPool:
      return "reset-command-pool";
    case HostControl::BeginCommands:
      return "begin-commands";
    case HostControl::EndCommands:
      return "end-commands";
    case HostControl::Submit:
      return "submit";
  }
  return "invalid";
}
std::string name(HostObjectKind kind) {
  switch (kind) {
    case HostObjectKind::Buffer:
      return "buffer";
    case HostObjectKind::Pipeline:
      return "pipeline";
  }
  return "invalid";
}

Value dimensions(const auto& values) {
  Array result;
  for (auto value : values) {
    result.push_back(value);
  }
  return result;
}

Value resourceJson(const TensorResource& resource) {
  Object result{{.K = "id", .V = resource.id},
                {.K = "kind", .V = name(resource.kind)},
                {.K = "name", .V = resource.name},
                {.K = "shape", .V = dimensions(resource.shape)},
                {.K = "element_type", .V = "f32"},
                {.K = "logical_layout", .V = resource.logical_layout},
                {.K = "physical_layout", .V = resource.physical_layout},
                {.K = "host_layout", .V = resource.host_layout},
                {.K = "weight_sha256", .V = resource.weight_sha256},
                {.K = "byte_size", .V = resource.byte_size},
                {.K = "first_kernel", .V = resource.first_kernel},
                {.K = "last_kernel", .V = resource.last_kernel}};
  if (resource.layout_transform) {
    const auto& transform = *resource.layout_transform;
    result["layout_transform"] =
      Object{{.K = "from_layout", .V = transform.from_layout},
             {.K = "to_layout", .V = transform.to_layout},
             {.K = "operation", .V = transform.operation}};
  } else {
    result["layout_transform"] = nullptr;
  }
  return result;
}

Value interfaceJson(const VulkanShaderInterface& interface) {
  Array bindings;
  for (const auto& binding : interface.bindings) {
    bindings.push_back(Object{{.K = "set", .V = binding.set},
                              {.K = "binding", .V = binding.binding},
                              {.K = "access", .V = name(binding.access)},
                              {.K = "byte_size", .V = binding.byte_size}});
  }
  Array pushes;
  for (const auto& block : interface.push_constants) {
    pushes.push_back(Object{{.K = "offset", .V = block.offset},
                            {.K = "byte_size", .V = block.byte_size}});
  }
  return Object{{.K = "entry_point", .V = interface.entry_point},
                {.K = "local_size", .V = dimensions(interface.local_size)},
                {.K = "bindings", .V = std::move(bindings)},
                {.K = "push_constants", .V = std::move(pushes)}};
}

Value operationJson(const HostOperation& operation) {
  return std::visit(
    [](const auto& op) -> Value {
      using T = std::decay_t<decltype(op)>;
      if constexpr (std::is_same_v<T, HostBarrierOp>) {
        Array buffers;
        for (const auto& barrier : op.buffers) {
          buffers.push_back(
            Object{{"buffer", barrier.buffer.value},
                   {"source_access", name(barrier.source_access)},
                   {"destination_access", name(barrier.destination_access)},
                   {"offset", barrier.offset},
                   {"byte_size", barrier.byte_size}});
        }
        return Object{{"op", "barrier"},
                      {"source_stage", name(op.source_stage)},
                      {"destination_stage", name(op.destination_stage)},
                      {"buffers", std::move(buffers)}};
      } else if constexpr (std::is_same_v<T, HostCopyOp>) {
        return Object{{"op", "copy"},
                      {"source", op.source.value},
                      {"destination", op.destination.value},
                      {"source_offset", op.source_offset},
                      {"destination_offset", op.destination_offset},
                      {"byte_size", op.byte_size}};
      } else if constexpr (std::is_same_v<T, HostUploadOp>) {
        return Object{{"op", "upload"},
                      {"staging_buffer", op.staging_buffer.value},
                      {"source", name(op.source)},
                      {"source_id", op.source_id},
                      {"byte_size", op.byte_size}};
      } else if constexpr (std::is_same_v<T, HostMappedSyncOp>) {
        return Object{{"op", "mapped-sync"},
                      {"buffer", op.buffer.value},
                      {"action", name(op.action)},
                      {"range", "whole-allocation"}};
      } else if constexpr (std::is_same_v<T, HostControlOp>) {
        return Object{{"op", "control"}, {"action", name(op.action)}};
      } else if constexpr (std::is_same_v<T, HostBindPipelineOp>) {
        return Object{{"op", "bind-pipeline"}, {"pipeline", op.pipeline.value}};
      } else if constexpr (std::is_same_v<T, HostBindDescriptorsOp>) {
        return Object{{"op", "bind-descriptors"},
                      {"pipeline", op.pipeline.value},
                      {"first_set", op.first_set},
                      {"set_count", op.set_count}};
      } else if constexpr (std::is_same_v<T, HostPushConstantsOp>) {
        return Object{{"op", "push-constants"},
                      {"pipeline", op.pipeline.value},
                      {"offset", op.offset},
                      {"bytes", hexBytes(op.bytes)}};
      } else if constexpr (std::is_same_v<T, HostDispatchOp>) {
        return Object{{"op", "dispatch"},
                      {"pipeline", op.pipeline.value},
                      {"groups", dimensions(op.groups)}};
      } else if constexpr (std::is_same_v<T, HostPrepareOutputsOp>) {
        Array buffers;
        for (const auto buffer : op.buffers) {
          buffers.push_back(buffer.value);
        }
        return Object{{.K = "op", .V = "prepare-outputs"},
                      {.K = "buffers", .V = std::move(buffers)}};
      } else if constexpr (std::is_same_v<T, HostPublishOp>) {
        return Object{{"op", "publish"},
                      {"staging_buffer", op.staging_buffer.value},
                      {"output_index", op.output_index},
                      {"byte_size", op.byte_size}};
      } else {
        return Object{{"op", "release-buffer"}, {"buffer", op.buffer.value}};
      }
    },
    operation);
}

Value programJson(const VulkanHostProgram& program, bool include_identity) {
  Object root{
    {.K = "schema", .V = program.schema},
    {.K = "codegen_version", .V = program.codegen_version},
    {.K = "helper_contract", .V = program.helper_contract},
    {.K = "synchronization_version", .V = program.synchronization_version},
    {.K = "canonical_encoding", .V = program.canonical_encoding},
    {.K = "abi_version", .V = program.abi_version},
    {.K = "model", .V = program.model},
    {.K = "host_triple", .V = program.host_triple},
    {.K = "device_target", .V = program.device_target}};
  root["options"] = Object{{.K = "allocation", .V = "independent-resources"},
                           {.K = "command_recording", .V = "per-run"},
                           {.K = "mapped_sync_range", .V = "whole-allocation"},
                           {.K = "output_publication",
                            .V = "after-all-invalidates-and-mapping-checks"},
                           {.K = "synchronize_same_context", .V = true},
                           {.K = "weight_upload", .V = "once-per-context"}};
  Array resources;
  for (const auto& resource : program.resources) {
    resources.push_back(resourceJson(resource));
  }
  root["resources"] = std::move(resources);
  Array buffers;
  for (const auto& buffer : program.buffers) {
    buffers.push_back(Object{{.K = "id", .V = buffer.id.value},
                             {.K = "resource_id", .V = buffer.resource_id},
                             {.K = "kind", .V = name(buffer.kind)},
                             {.K = "scope", .V = name(buffer.scope)},
                             {.K = "byte_size", .V = buffer.byte_size}});
  }
  root["buffers"] = std::move(buffers);
  auto ports = [](const std::vector<HostPort>& values) {
    Array result;
    for (const auto& port : values) {
      result.push_back(
        Object{{.K = "resource_id", .V = port.resource_id},
               {.K = "device_buffer", .V = port.device_buffer.value},
               {.K = "staging_buffer", .V = port.staging_buffer.value},
               {.K = "elements", .V = port.elements}});
    }
    return result;
  };
  root["inputs"] = ports(program.inputs);
  root["outputs"] = ports(program.outputs);
  Array pipelines;
  for (const auto& pipeline : program.pipelines) {
    Array bindings;
    for (const auto& binding : pipeline.bindings) {
      bindings.push_back(Object{{.K = "set", .V = binding.set},
                                {.K = "binding", .V = binding.binding},
                                {.K = "buffer", .V = binding.buffer.value},
                                {.K = "access", .V = name(binding.access)}});
    }
    Array pushes;
    for (const auto& push : pipeline.push_constants) {
      pushes.push_back(Object{{.K = "offset", .V = push.offset},
                              {.K = "bytes", .V = hexBytes(push.bytes)}});
    }
    pipelines.push_back(
      Object{{.K = "id", .V = pipeline.id.value},
             {.K = "kernel_id", .V = pipeline.kernel_id},
             {.K = "binary_symbol", .V = pipeline.binary_symbol},
             {.K = "shader_sha256", .V = pipeline.shader_sha256},
             {.K = "shader_size_bytes", .V = pipeline.shader_size_bytes},
             {.K = "interface", .V = interfaceJson(pipeline.interface)},
             {.K = "bindings", .V = std::move(bindings)},
             {.K = "push_constants", .V = std::move(pushes)},
             {.K = "dispatch", .V = dimensions(pipeline.dispatch)}});
  }
  root["pipelines"] = std::move(pipelines);
  Array weights;
  for (const auto& weight : program.weights) {
    weights.push_back(
      Object{{.K = "resource_id", .V = weight.resource_id},
             {.K = "byte_size", .V = weight.byte_size},
             {.K = "logical_sha256", .V = weight.logical_sha256},
             {.K = "physical_sha256", .V = weight.physical_sha256}});
  }
  root["weights"] = std::move(weights);
  auto objects = [](const std::vector<HostObject>& values) {
    Array result;
    for (const auto& object : values) {
      result.push_back(Object{{.K = "kind", .V = name(object.kind)},
                              {.K = "id", .V = object.id}});
    }
    return result;
  };
  root["initialize"] = objects(program.initialize);
  root["destroy"] = objects(program.destroy);
  auto operations = [](const std::vector<HostOperation>& values) {
    Array result;
    for (const auto& operation : values) {
      result.push_back(operationJson(operation));
    }
    return result;
  };
  root["create_operations"] = operations(program.create_operations);
  root["run_operations"] = operations(program.run_operations);
  if (include_identity) {
    root["identity"] = program.identity;
  }
  return root;
}

// This spelling matches JSON's standard UTF-8 string encoding, including the
// lowercase control escapes used by the independent Python evidence checker.
void appendString(std::string& output, llvm::StringRef value) {
  constexpr std::string_view digits = "0123456789abcdef";
  output.push_back('"');
  for (unsigned char byte : value.bytes()) {
    switch (byte) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 32) {
          output += "\\u00";
          output.push_back(digits[byte >> 4]);
          output.push_back(digits[byte & 15]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
    }
  }
  output.push_back('"');
}

void appendCanonicalJson(std::string& output, const Value& value) {
  if (auto string = value.getAsString()) {
    appendString(output, *string);
  } else if (const auto* array = value.getAsArray()) {
    output.push_back('[');
    bool first = true;
    for (const auto& element : *array) {
      if (!first) {
        output.push_back(',');
      }
      first = false;
      appendCanonicalJson(output, element);
    }
    output.push_back(']');
  } else if (const auto* object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto& item : *object) {
      keys.push_back(item.first);
    }
    std::ranges::sort(keys);
    output.push_back('{');
    bool first = true;
    for (auto key : keys) {
      if (!first) {
        output.push_back(',');
      }
      first = false;
      appendString(output, key);
      output.push_back(':');
      appendCanonicalJson(output, *object->get(key));
    }
    output.push_back('}');
  } else {
    output += llvm::formatv("{0}", value).str();
  }
}

void appendLength(std::string& output, std::uint64_t length) {
  for (unsigned index = 0; index < 8; ++index) {
    output.push_back(static_cast<char>((length >> (index * 8)) & 255));
  }
}

bool validText(std::string_view value) {
  return llvm::json::isUTF8(llvm::StringRef(value.data(), value.size()));
}

bool validResourceText(const TensorResource& resource) {
  if (!validText(resource.name) || !validText(resource.logical_layout) ||
      !validText(resource.physical_layout) ||
      !validText(resource.host_layout) || !validText(resource.weight_sha256)) {
    return false;
  }
  if (resource.layout_transform) {
    const auto& transform = *resource.layout_transform;
    return validText(transform.from_layout) && validText(transform.to_layout) &&
           validText(transform.operation);
  }
  return true;
}

// Validate before constructing llvm::json::Value: that constructor asserts on
// malformed UTF-8 in assertion builds and replaces bytes in Release builds.
std::expected<void, std::string> checkProgramText(
  const VulkanHostProgram& program) {
  if (!validText(program.schema) || !validText(program.codegen_version) ||
      !validText(program.helper_contract) ||
      !validText(program.synchronization_version) ||
      !validText(program.canonical_encoding) || !validText(program.identity) ||
      !validText(program.model) || !validText(program.host_triple) ||
      !validText(program.device_target) ||
      !std::ranges::all_of(program.resources, validResourceText)) {
    return std::unexpected("host program contains invalid UTF-8 metadata");
  }
  for (const auto& pipeline : program.pipelines) {
    if (!validText(pipeline.binary_symbol) ||
        !validText(pipeline.shader_sha256) ||
        !validText(pipeline.interface.entry_point)) {
      return std::unexpected("host pipeline contains invalid UTF-8 metadata");
    }
  }
  for (const auto& weight : program.weights) {
    if (!validText(weight.logical_sha256) ||
        !validText(weight.physical_sha256)) {
      return std::unexpected("host weight contains invalid UTF-8 metadata");
    }
  }
  return {};
}

template <typename E>
bool validEnum(E value, E last) {
  return std::to_underlying(value) <= std::to_underlying(last);
}

std::expected<void, std::string> checkEnums(const VulkanHostProgram& program) {
  for (const auto& resource : program.resources) {
    if (!validEnum(resource.kind, ResourceKind::Weight)) {
      return std::unexpected("host resource has an invalid role");
    }
  }
  for (const auto& buffer : program.buffers) {
    if (!validEnum(buffer.kind, HostBufferKind::WeightStaging) ||
        !validEnum(buffer.scope, HostBufferScope::CreateUpload)) {
      return std::unexpected("host buffer has an invalid kind or scope");
    }
  }
  for (const auto& pipeline : program.pipelines) {
    for (const auto& binding : pipeline.bindings) {
      if (!validEnum(binding.access, ResourceAccess::Write)) {
        return std::unexpected("host binding has invalid access");
      }
    }
    for (const auto& binding : pipeline.interface.bindings) {
      if (!validEnum(binding.access, ResourceAccess::Write)) {
        return std::unexpected("shader binding has invalid access");
      }
    }
  }
  for (const auto objects : {std::span<const HostObject>(program.initialize),
                             std::span<const HostObject>(program.destroy)}) {
    for (const auto& object : objects) {
      if (!validEnum(object.kind, HostObjectKind::Pipeline)) {
        return std::unexpected("host object has invalid kind");
      }
    }
  }
  for (const auto operations :
       {std::span<const HostOperation>(program.create_operations),
        std::span<const HostOperation>(program.run_operations)}) {
    for (const auto& operation : operations) {
      if (operation.valueless_by_exception()) {
        return std::unexpected("host operation has no active value");
      }
      const bool valid = std::visit(
        [](const auto& op) {
          using T = std::decay_t<decltype(op)>;
          if constexpr (std::is_same_v<T, HostBarrierOp>) {
            if (!validEnum(op.source_stage, HostStage::Compute) ||
                !validEnum(op.destination_stage, HostStage::Compute)) {
              return false;
            }
            return std::ranges::all_of(op.buffers, [](const auto& buffer) {
              return validEnum(buffer.source_access, HostAccess::HostRead) &&
                     validEnum(buffer.destination_access, HostAccess::HostRead);
            });
          } else if constexpr (std::is_same_v<T, HostUploadOp>) {
            return validEnum(op.source, HostUploadSource::Weight);
          } else if constexpr (std::is_same_v<T, HostMappedSyncOp>) {
            return validEnum(op.action, HostMappedSync::Invalidate);
          } else if constexpr (std::is_same_v<T, HostControlOp>) {
            return validEnum(op.action, HostControl::Submit);
          } else {
            return true;
          }
        },
        operation);
      if (!valid) {
        return std::unexpected("host operation has an invalid enum");
      }
    }
  }
  return {};
}

void appendBarrier(std::vector<HostOperation>& operations,
                   HostBarrierOp barrier) {
  // A buffer-barrier batch has no internal order because storage never aliases.
  std::ranges::sort(barrier.buffers, {}, [](const auto& buffer) {
    return buffer.buffer.value;
  });
  operations.emplace_back(std::move(barrier));
}

std::expected<VulkanHostProgram, std::string> buildProgram(
  const VulkanExecutionPlan& plan,
  std::span<const VulkanModelShader> shaders,
  std::span<const VulkanModelWeight> weights) {
  if (!validText(plan.model) || !validText(plan.host_triple) ||
      !validText(plan.device_target) || !validText(plan.build_identity) ||
      !validText(plan.plan_hash) ||
      !std::ranges::all_of(plan.resources, validResourceText) ||
      !std::ranges::all_of(plan.kernels, [](const auto& kernel) {
        return validText(kernel.entry_point) && validText(kernel.operation);
      })) {
    return std::unexpected("host program source contains invalid UTF-8");
  }
  if (auto checked = verifyExecutionPlan(plan); !checked) {
    return std::unexpected("host program: " + checked.error());
  }
  if (auto checked = verifyVulkanModelWeights(plan, weights); !checked) {
    return std::unexpected("host program: " + checked.error());
  }
  const llvm::Triple triple(llvm::Triple::normalize(plan.host_triple));
  if (triple.getArch() != llvm::Triple::x86_64 || !triple.isOSLinux() ||
      plan.device_target != "vulkan1.1-spirv1.0") {
    return std::unexpected(
      "host program requires x86-64 Linux / Vulkan 1.1 SPIR-V 1.0");
  }
  if (!std::in_range<std::uint32_t>(plan.resources.size()) ||
      !std::in_range<std::uint32_t>(plan.kernels.size()) ||
      !std::in_range<std::uint32_t>(plan.inputs.size()) ||
      !std::in_range<std::uint32_t>(plan.outputs.size())) {
    return std::unexpected("host declaration count overflows uint32");
  }
  if (shaders.size() != plan.kernels.size()) {
    return std::unexpected("host shader coverage differs from the schedule");
  }
  std::map<std::uint32_t, std::size_t> shader_indices;
  std::set<std::string> shader_symbols;
  for (std::size_t index = 0; index < shaders.size(); ++index) {
    const auto& shader = shaders[index];
    if (!shader_indices.emplace(shader.kernel_id, index).second ||
        shader.binary_symbol.empty() ||
        !shader_symbols.insert(shader.binary_symbol).second) {
      return std::unexpected(
        "host shader IDs/symbols are invalid or duplicated");
    }
  }
  std::map<std::uint32_t, std::size_t> resource_indices;
  for (std::size_t index = 0; index < plan.resources.size(); ++index) {
    resource_indices.emplace(plan.resources[index].id, index);
  }
  VulkanHostProgram program{
    .schema = std::string(kHostProgramSchema),
    .codegen_version = std::string(kHostProgramCodegenVersion),
    .helper_contract = std::string(kHostProgramHelperContract),
    .synchronization_version = std::string(kHostProgramSynchronizationVersion),
    .canonical_encoding = std::string(kHostProgramCanonicalEncoding),
    .abi_version = 1,
    .model = plan.model,
    .host_triple = plan.host_triple,
    .device_target = plan.device_target,
    .resources = plan.resources,
    .buffers = {},
    .inputs = {},
    .outputs = {},
    .pipelines = {},
    .weights = {},
    .initialize = {},
    .destroy = {},
    .create_operations = {},
    .run_operations = {},
    .identity = {}};
  auto addBuffer =
    [&](const TensorResource& resource,
        HostBufferKind kind,
        HostBufferScope scope) -> std::expected<HostBufferId, std::string> {
    if (program.buffers.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return std::unexpected("host buffer count overflows uint32");
    }
    HostBufferId id{static_cast<std::uint32_t>(program.buffers.size())};
    program.buffers.push_back(
      {id, resource.id, kind, scope, resource.byte_size});
    program.initialize.push_back({HostObjectKind::Buffer, id.value});
    return id;
  };
  std::map<std::uint32_t, HostBufferId> device_buffers;
  for (const auto& resource : plan.resources) {
    auto id =
      addBuffer(resource, HostBufferKind::Device, HostBufferScope::Context);
    if (!id) {
      return std::unexpected(id.error());
    }
    device_buffers.emplace(resource.id, *id);
  }
  auto addPorts =
    [&](const std::vector<std::uint32_t>& ids,
        HostBufferKind kind,
        std::vector<HostPort>& ports) -> std::expected<void, std::string> {
    for (auto id : ids) {
      const auto& resource = plan.resources[resource_indices.at(id)];
      if (resource.host_layout != resource.physical_layout) {
        return std::unexpected(
          "host program public I/O requires a materialized layout conversion");
      }
      auto staging = addBuffer(resource, kind, HostBufferScope::Context);
      if (!staging) {
        return std::unexpected(staging.error());
      }
      ports.push_back(
        {id, device_buffers.at(id), *staging, resource.byte_size / 4});
    }
    return {};
  };
  if (auto checked =
        addPorts(plan.inputs, HostBufferKind::InputStaging, program.inputs);
      !checked) {
    return std::unexpected(checked.error());
  }
  if (auto checked =
        addPorts(plan.outputs, HostBufferKind::OutputStaging, program.outputs);
      !checked) {
    return std::unexpected(checked.error());
  }
  std::map<std::uint32_t, HostBufferId> weight_staging;
  for (const auto& resource : plan.resources) {
    if (resource.kind != ResourceKind::Weight) {
      continue;
    }
    auto staging = addBuffer(
      resource, HostBufferKind::WeightStaging, HostBufferScope::CreateUpload);
    if (!staging) {
      return std::unexpected(staging.error());
    }
    weight_staging.emplace(resource.id, *staging);
    const auto found =
      std::ranges::find(weights, resource.id, &VulkanModelWeight::resource_id);
    program.weights.push_back({resource.id,
                               resource.byte_size,
                               resource.weight_sha256,
                               weightHash(found->bytes)});
  }
  for (const auto& kernel : plan.kernels) {
    auto shader_index = shader_indices.find(kernel.id);
    if (shader_index == shader_indices.end()) {
      return std::unexpected(
        std::format("host shader for kernel {} is missing", kernel.id));
    }
    const auto& shader = shaders[shader_index->second];
    auto interface = reflectVulkanShaderInterface(plan, kernel, shader);
    if (!interface) {
      return std::unexpected(
        std::format("host kernel {}: {}", kernel.id, interface.error()));
    }
    if (kernel.local_size[1] != 1 || kernel.local_size[2] != 1) {
      return std::unexpected(
        "host dispatch requires a one-dimensional workgroup");
    }
    std::uint64_t elements = 0;
    for (auto id : kernel.writes) {
      elements = std::max(
        elements, plan.resources[resource_indices.at(id)].byte_size / 4);
    }
    const std::uint64_t groups = (elements / kernel.local_size[0]) +
                                 (elements % kernel.local_size[0] != 0 ? 1 : 0);
    if (groups == 0 || !std::in_range<std::uint32_t>(groups) ||
        shader.words.size() > std::numeric_limits<std::uint64_t>::max() / 4) {
      return std::unexpected("host shader size or dispatch count overflows");
    }
    std::vector<HostBinding> bindings;
    for (const auto& binding : kernel.bindings) {
      if (binding.set == std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected("host descriptor set count overflows uint32");
      }
      bindings.push_back({binding.set,
                          binding.binding,
                          device_buffers.at(binding.resource_id),
                          binding.access});
    }
    program.pipelines.push_back(
      {HostPipelineId{kernel.id},
       kernel.id,
       shader.binary_symbol,
       shaderHash(shader.words),
       static_cast<std::uint64_t>(shader.words.size()) * 4,
       std::move(*interface),
       std::move(bindings),
       kernel.push_constants,
       {static_cast<std::uint32_t>(groups), 1, 1}});
    program.initialize.push_back({HostObjectKind::Pipeline, kernel.id});
  }
  for (auto it = program.initialize.rbegin(); it != program.initialize.rend();
       ++it) {
    if (it->kind == HostObjectKind::Buffer &&
        program.buffers[it->id].scope == HostBufferScope::CreateUpload) {
      continue;
    }
    program.destroy.push_back(*it);
  }
  auto makeBarrier = [&](HostBufferId id,
                         HostAccess source,
                         HostAccess destination) {
    return HostBufferBarrier{.buffer = id,
                             .source_access = source,
                             .destination_access = destination,
                             .offset = 0,
                             .byte_size = program.buffers[id.value].byte_size};
  };
  auto copy = [&](HostBufferId source, HostBufferId destination) {
    return HostCopyOp{.source = source,
                      .destination = destination,
                      .source_offset = 0,
                      .destination_offset = 0,
                      .byte_size = program.buffers[source.value].byte_size};
  };
  auto begin = [](std::vector<HostOperation>& operations) {
    operations.emplace_back(HostControlOp{HostControl::ResetFence});
    operations.emplace_back(HostControlOp{HostControl::ResetCommandPool});
    operations.emplace_back(HostControlOp{HostControl::BeginCommands});
  };
  auto submit = [](std::vector<HostOperation>& operations) {
    operations.emplace_back(HostControlOp{HostControl::EndCommands});
    operations.emplace_back(HostControlOp{HostControl::Submit});
    operations.emplace_back(HostControlOp{HostControl::WaitFence});
  };
  if (!program.weights.empty()) {
    auto& operations = program.create_operations;
    begin(operations);
    HostBarrierOp host_barrier{.source_stage = HostStage::Host,
                               .destination_stage = HostStage::Transfer,
                               .buffers = {}};
    HostBarrierOp device_barrier{.source_stage = HostStage::Transfer,
                                 .destination_stage = HostStage::Compute,
                                 .buffers = {}};
    for (const auto& weight : program.weights) {
      const auto staging = weight_staging.at(weight.resource_id);
      operations.emplace_back(HostUploadOp{.staging_buffer = staging,
                                           .source = HostUploadSource::Weight,
                                           .source_id = weight.resource_id,
                                           .byte_size = weight.byte_size});
      operations.emplace_back(
        HostMappedSyncOp{.buffer = staging, .action = HostMappedSync::Flush});
      host_barrier.buffers.push_back(
        makeBarrier(staging, HostAccess::HostWrite, HostAccess::TransferRead));
      device_barrier.buffers.push_back(
        makeBarrier(device_buffers.at(weight.resource_id),
                    HostAccess::TransferWrite,
                    HostAccess::ShaderRead));
    }
    appendBarrier(operations, std::move(host_barrier));
    for (const auto& weight : program.weights) {
      operations.emplace_back(copy(weight_staging.at(weight.resource_id),
                                   device_buffers.at(weight.resource_id)));
    }
    appendBarrier(operations, std::move(device_barrier));
    submit(operations);
    for (const auto& weight : program.weights) {
      operations.emplace_back(
        HostReleaseBufferOp{weight_staging.at(weight.resource_id)});
    }
  }
  auto& operations = program.run_operations;
  HostBarrierOp host_inputs{.source_stage = HostStage::Host,
                            .destination_stage = HostStage::Transfer,
                            .buffers = {}};
  HostBarrierOp device_inputs{.source_stage = HostStage::Transfer,
                              .destination_stage = HostStage::Compute,
                              .buffers = {}};
  for (std::size_t index = 0; index < program.inputs.size(); ++index) {
    const auto& port = program.inputs[index];
    const auto bytes = program.buffers[port.staging_buffer.value].byte_size;
    operations.emplace_back(
      HostUploadOp{.staging_buffer = port.staging_buffer,
                   .source = HostUploadSource::Input,
                   .source_id = static_cast<std::uint32_t>(index),
                   .byte_size = bytes});
    operations.emplace_back(HostMappedSyncOp{.buffer = port.staging_buffer,
                                             .action = HostMappedSync::Flush});
    host_inputs.buffers.push_back(makeBarrier(
      port.staging_buffer, HostAccess::HostWrite, HostAccess::TransferRead));
    device_inputs.buffers.push_back(makeBarrier(
      port.device_buffer, HostAccess::TransferWrite, HostAccess::ShaderRead));
  }
  operations.emplace_back(HostControlOp{HostControl::WaitFence});
  begin(operations);
  if (!host_inputs.buffers.empty()) {
    appendBarrier(operations, std::move(host_inputs));
    for (const auto& port : program.inputs) {
      operations.emplace_back(copy(port.staging_buffer, port.device_buffer));
    }
    appendBarrier(operations, std::move(device_inputs));
  }
  // Match the legacy recorder, not a last-access optimization: even a second
  // reader is conservatively synchronized with all earlier shader writes. The
  // compute source stage also orders earlier reads before a subsequent rewrite.
  std::set<std::uint32_t> shader_written;
  for (std::size_t index = 0; index < plan.kernels.size(); ++index) {
    const auto& kernel = plan.kernels[index];
    const auto& pipeline = program.pipelines[index];
    std::set<std::uint32_t> accessed(kernel.reads.begin(), kernel.reads.end());
    accessed.insert(kernel.writes.begin(), kernel.writes.end());
    HostBarrierOp barrier{.source_stage = HostStage::Compute,
                          .destination_stage = HostStage::Compute,
                          .buffers = {}};
    for (auto id : accessed) {
      if (shader_written.contains(id)) {
        barrier.buffers.push_back(makeBarrier(device_buffers.at(id),
                                              HostAccess::ShaderWrite,
                                              HostAccess::ShaderReadWrite));
      }
    }
    if (!barrier.buffers.empty()) {
      appendBarrier(operations, std::move(barrier));
    }
    operations.emplace_back(HostBindPipelineOp{pipeline.id});
    std::uint32_t sets = 0;
    for (const auto& binding : pipeline.bindings) {
      sets = std::max(sets, binding.set + 1);
    }
    if (sets != 0) {
      operations.emplace_back(HostBindDescriptorsOp{
        .pipeline = pipeline.id, .first_set = 0, .set_count = sets});
    }
    for (const auto& push : pipeline.push_constants) {
      operations.emplace_back(HostPushConstantsOp{
        .pipeline = pipeline.id, .offset = push.offset, .bytes = push.bytes});
    }
    operations.emplace_back(
      HostDispatchOp{.pipeline = pipeline.id, .groups = pipeline.dispatch});
    shader_written.insert(kernel.writes.begin(), kernel.writes.end());
  }
  HostBarrierOp device_outputs{.source_stage = HostStage::Compute,
                               .destination_stage = HostStage::Transfer,
                               .buffers = {}};
  HostBarrierOp host_outputs{.source_stage = HostStage::Transfer,
                             .destination_stage = HostStage::Host,
                             .buffers = {}};
  for (const auto& port : program.outputs) {
    device_outputs.buffers.push_back(makeBarrier(
      port.device_buffer, HostAccess::ShaderWrite, HostAccess::TransferRead));
    host_outputs.buffers.push_back(makeBarrier(
      port.staging_buffer, HostAccess::TransferWrite, HostAccess::HostRead));
  }
  if (!device_outputs.buffers.empty()) {
    appendBarrier(operations, std::move(device_outputs));
    for (const auto& port : program.outputs) {
      operations.emplace_back(copy(port.device_buffer, port.staging_buffer));
    }
    appendBarrier(operations, std::move(host_outputs));
  }
  submit(operations);
  std::vector<HostBufferId> output_staging;
  for (const auto& port : program.outputs) {
    operations.emplace_back(HostMappedSyncOp{
      .buffer = port.staging_buffer, .action = HostMappedSync::Invalidate});
    output_staging.push_back(port.staging_buffer);
  }
  operations.emplace_back(HostPrepareOutputsOp{std::move(output_staging)});
  for (std::size_t index = 0; index < program.outputs.size(); ++index) {
    const auto& port = program.outputs[index];
    operations.emplace_back(HostPublishOp{
      .staging_buffer = port.staging_buffer,
      .output_index = static_cast<std::uint32_t>(index),
      .byte_size = program.buffers[port.staging_buffer.value].byte_size});
  }
  return program;
}

// Validate the ordering-sensitive operation references independently of the
// planner, then compare the complete program against its verified source. This
// catches a rehashed but altered program as well as a stale identity.
std::expected<void, std::string> verifyOperationReferences(
  const VulkanHostProgram& program) {
  if (auto enums = checkEnums(program); !enums) {
    return enums;
  }
  std::map<std::uint32_t, std::size_t> buffers;
  for (std::size_t index = 0; index < program.buffers.size(); ++index) {
    if (!buffers.emplace(program.buffers[index].id.value, index).second) {
      return std::unexpected("duplicate host buffer ID");
    }
  }
  std::map<std::uint32_t, std::size_t> pipelines;
  for (std::size_t index = 0; index < program.pipelines.size(); ++index) {
    if (!pipelines.emplace(program.pipelines[index].id.value, index).second) {
      return std::unexpected("duplicate host pipeline ID");
    }
  }
  for (const auto& pipeline : program.pipelines) {
    for (const auto& binding : pipeline.bindings) {
      auto found = buffers.find(binding.buffer.value);
      if (found == buffers.end() ||
          program.buffers[found->second].kind != HostBufferKind::Device) {
        return std::unexpected(
          "host descriptor references an undefined device buffer");
      }
    }
  }
  auto checkOperations = [&](const std::vector<HostOperation>& operations,
                             bool is_run) -> std::expected<void, std::string> {
    std::set<std::uint32_t> alive;
    for (const auto& buffer : program.buffers) {
      if (!is_run || buffer.scope == HostBufferScope::Context) {
        alive.insert(buffer.id.value);
      }
    }
    bool recording = false;
    bool ended = false;
    bool completed = true;
    bool prepared = false;
    std::optional<HostPipelineId> bound;
    std::set<std::uint32_t> invalidated;
    std::uint32_t published = 0;
    auto hasBuffer = [&](HostBufferId id) {
      return alive.contains(id.value);
    };
    auto rangeFits =
      [&](HostBufferId id, std::uint64_t offset, std::uint64_t size) {
        if (!hasBuffer(id)) {
          return false;
        }
        const auto capacity = program.buffers[buffers.at(id.value)].byte_size;
        return size != 0 && offset <= capacity && size <= capacity - offset;
      };
    for (std::size_t index = 0; index < operations.size(); ++index) {
      const auto checked = std::visit(
        [&](const auto& op) -> bool {
          using T = std::decay_t<decltype(op)>;
          if constexpr (std::is_same_v<T, HostControlOp>) {
            switch (op.action) {
              case HostControl::BeginCommands:
                if (recording || !completed) {
                  return false;
                }
                recording = true;
                ended = false;
                return true;
              case HostControl::EndCommands:
                if (!recording) {
                  return false;
                }
                recording = false;
                ended = true;
                return true;
              case HostControl::Submit:
                if (recording || !ended || !completed) {
                  return false;
                }
                completed = false;
                ended = false;
                return true;
              case HostControl::WaitFence:
                if (recording) {
                  return false;
                }
                completed = true;
                return true;
              case HostControl::ResetFence:
              case HostControl::ResetCommandPool:
                return !recording && completed;
            }
            return false;
          } else if constexpr (std::is_same_v<T, HostCopyOp>) {
            return recording && op.source != op.destination &&
                   rangeFits(op.source, op.source_offset, op.byte_size) &&
                   rangeFits(
                     op.destination, op.destination_offset, op.byte_size);
          } else if constexpr (std::is_same_v<T, HostBarrierOp>) {
            return recording && !op.buffers.empty() &&
                   std::ranges::all_of(op.buffers, [&](const auto& buffer) {
                     return rangeFits(
                       buffer.buffer, buffer.offset, buffer.byte_size);
                   });
          } else if constexpr (std::is_same_v<T, HostUploadOp>) {
            return rangeFits(op.staging_buffer, 0, op.byte_size) &&
                   program.buffers[buffers.at(op.staging_buffer.value)].kind !=
                     HostBufferKind::Device;
          } else if constexpr (std::is_same_v<T, HostMappedSyncOp>) {
            if (!hasBuffer(op.buffer) ||
                program.buffers[buffers.at(op.buffer.value)].kind ==
                  HostBufferKind::Device) {
              return false;
            }
            if (op.action == HostMappedSync::Invalidate) {
              if (recording || !completed) {
                return false;
              }
              invalidated.insert(op.buffer.value);
            }
            return true;
          } else if constexpr (std::is_same_v<T, HostBindPipelineOp>) {
            if (!recording || !pipelines.contains(op.pipeline.value)) {
              return false;
            }
            bound = op.pipeline;
            return true;
          } else if constexpr (std::is_same_v<T, HostBindDescriptorsOp>) {
            return recording && bound == op.pipeline && op.set_count != 0 &&
                   op.first_set <= std::numeric_limits<std::uint32_t>::max() -
                                     (op.set_count - 1);
          } else if constexpr (std::is_same_v<T, HostPushConstantsOp>) {
            return recording && bound == op.pipeline && !op.bytes.empty() &&
                   op.offset <= 128 && op.bytes.size() <= 128 - op.offset &&
                   op.offset % 4 == 0 && op.bytes.size() % 4 == 0;
          } else if constexpr (std::is_same_v<T, HostDispatchOp>) {
            return recording && bound == op.pipeline &&
                   std::ranges::all_of(op.groups,
                                       [](auto group) { return group != 0; });
          } else if constexpr (std::is_same_v<T, HostPrepareOutputsOp>) {
            if (!is_run || recording || !completed ||
                op.buffers.size() != program.outputs.size()) {
              return false;
            }
            for (std::size_t port = 0; port < op.buffers.size(); ++port) {
              if (op.buffers[port] != program.outputs[port].staging_buffer ||
                  !hasBuffer(op.buffers[port]) ||
                  !invalidated.contains(op.buffers[port].value)) {
                return false;
              }
            }
            prepared = true;
            return true;
          } else if constexpr (std::is_same_v<T, HostPublishOp>) {
            if (!prepared || !completed || recording ||
                op.output_index != published ||
                published >= program.outputs.size() ||
                op.staging_buffer !=
                  program.outputs[published].staging_buffer ||
                !rangeFits(op.staging_buffer, 0, op.byte_size)) {
              return false;
            }
            ++published;
            return true;
          } else {
            if (is_run || recording || !completed || !hasBuffer(op.buffer) ||
                program.buffers[buffers.at(op.buffer.value)].scope !=
                  HostBufferScope::CreateUpload) {
              return false;
            }
            alive.erase(op.buffer.value);
            return true;
          }
        },
        operations[index]);
      if (!checked) {
        return std::unexpected(
          std::format("host {} operation {} has invalid reference/order/range",
                      is_run ? "run" : "create",
                      index));
      }
    }
    if (recording || !completed ||
        (is_run && published != program.outputs.size())) {
      return std::unexpected(
        "host commands or output publication are incomplete");
    }
    return {};
  };
  if (auto checked = checkOperations(program.create_operations, false);
      !checked) {
    return checked;
  }
  return checkOperations(program.run_operations, true);
}

}  // namespace

std::expected<std::string, std::string> canonicalVulkanHostProgram(
  const VulkanHostProgram& program) {
  if (auto checked = checkEnums(program); !checked) {
    return std::unexpected(checked.error());
  }
  if (auto checked = checkProgramText(program); !checked) {
    return std::unexpected(checked.error());
  }
  auto json = programJson(program, false);
  std::string canonical_json;
  appendCanonicalJson(canonical_json, json);
  std::string result;
  appendLength(result, kCanonicalDomain.size());
  result += kCanonicalDomain;
  appendLength(result, canonical_json.size());
  result += canonical_json;
  return result;
}

std::expected<std::string, std::string> calculateVulkanHostProgramIdentity(
  const VulkanHostProgram& program) {
  auto canonical = canonicalVulkanHostProgram(program);
  if (!canonical) {
    return std::unexpected(canonical.error());
  }
  return sha256(*canonical);
}

std::expected<std::string, std::string> serializeVulkanHostProgram(
  const VulkanHostProgram& program) {
  auto identity = calculateVulkanHostProgramIdentity(program);
  if (!identity) {
    return std::unexpected(identity.error());
  }
  if (program.identity != *identity) {
    return std::unexpected("host program identity mismatch");
  }
  std::string result;
  appendCanonicalJson(result, programJson(program, true));
  return result + "\n";
}

std::expected<VulkanHostProgram, std::string> makeVulkanHostProgram(
  const VulkanExecutionPlan& plan,
  std::span<const VulkanModelShader> shaders,
  std::span<const VulkanModelWeight> weights) {
  auto program = buildProgram(plan, shaders, weights);
  if (!program) {
    return program;
  }
  if (auto checked = verifyOperationReferences(*program); !checked) {
    return std::unexpected(checked.error());
  }
  auto identity = calculateVulkanHostProgramIdentity(*program);
  if (!identity) {
    return std::unexpected(identity.error());
  }
  program->identity = std::move(*identity);
  return program;
}

std::expected<void, std::string> verifyVulkanHostProgram(
  const VulkanHostProgram& program,
  const VulkanExecutionPlan& plan,
  std::span<const VulkanModelShader> shaders,
  std::span<const VulkanModelWeight> weights) {
  if (program.schema != kHostProgramSchema ||
      program.codegen_version != kHostProgramCodegenVersion ||
      program.helper_contract != kHostProgramHelperContract ||
      program.synchronization_version != kHostProgramSynchronizationVersion ||
      program.canonical_encoding != kHostProgramCanonicalEncoding ||
      program.abi_version != 1) {
    return std::unexpected(
      "unsupported host program/schema/codegen/helper/synchronization/ABI "
      "version");
  }
  if (auto checked = verifyOperationReferences(program); !checked) {
    return checked;
  }
  auto identity = calculateVulkanHostProgramIdentity(program);
  if (!identity || *identity != program.identity) {
    return std::unexpected("host program identity mismatch");
  }
  auto expected = buildProgram(plan, shaders, weights);
  if (!expected) {
    return std::unexpected(expected.error());
  }
  auto actual_bytes = canonicalVulkanHostProgram(program);
  auto expected_bytes = canonicalVulkanHostProgram(*expected);
  if (!actual_bytes || !expected_bytes || *actual_bytes != *expected_bytes) {
    return std::unexpected(
      "host program declarations/operations differ from verified source");
  }
  return {};
}

}  // namespace ncnn_mlir::vulkan
