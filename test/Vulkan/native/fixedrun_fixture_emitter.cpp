#include "VulkanProduct.hpp"
#include "vulkan_shader_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"
#include <vulkan/vulkan.h>

namespace {

namespace fs = std::filesystem;
namespace vk = ncnn_mlir::vulkan;

constexpr std::string_view kModelName = "va4_fixedrun";
constexpr std::string_view kHostTriple = "x86_64-linux-gnu";
constexpr std::uint64_t kElementCount = 17;

vk::TensorResource makeResource(std::uint32_t id,
                                vk::ResourceKind kind,
                                std::string name,
                                std::uint32_t first_kernel,
                                std::uint32_t last_kernel) {
  const bool public_io =
    kind == vk::ResourceKind::Input || kind == vk::ResourceKind::Output;
  return {.id = id,
          .kind = kind,
          .name = std::move(name),
          .shape = {static_cast<std::int64_t>(kElementCount)},
          .logical_layout = "linear",
          .physical_layout = "linear",
          .host_layout = public_io ? "linear" : "",
          .layout_transform = std::nullopt,
          .weight_sha256 = "",
          .byte_size = kElementCount * sizeof(float),
          .first_kernel = first_kernel,
          .last_kernel = last_kernel};
}

ncnn_compile::VulkanModelPackage makePackage() {
  std::vector<vk::TensorResource> resources{
    makeResource(7, vk::ResourceKind::Input, "input", 0, 0),
    vk::TensorResource{.id = 103,
                       .kind = vk::ResourceKind::Weight,
                       .name = "weight",
                       .shape = {4},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 4 * sizeof(float),
                       .first_kernel = 0,
                       .last_kernel = 0},
    makeResource(307, vk::ResourceKind::Intermediate, "intermediate", 0, 1),
    makeResource(41, vk::ResourceKind::Output, "output_a", 1, 1),
    makeResource(59, vk::ResourceKind::Output, "output_b", 1, 1),
  };

  std::vector<std::uint8_t> weight_bytes{0x00,
                                         0x00,
                                         0x80,
                                         0x3f,
                                         0x00,
                                         0x00,
                                         0x80,
                                         0x3f,
                                         0x00,
                                         0x00,
                                         0x80,
                                         0x3f,
                                         0x00,
                                         0x00,
                                         0x80,
                                         0x3f};
  auto weight_hash = vk::vulkanWeightSHA256(resources[1], weight_bytes);
  if (!weight_hash) {
    throw std::runtime_error("cannot hash fixed-run test weight: " +
                             weight_hash.error());
  }
  resources[1].weight_sha256 = *weight_hash;

  std::vector<vk::KernelExecution> kernels{
    {.id = 101,
     .operation = "fixedrun_stage0",
     .entry_point = "compute_main",
     .local_size = {64, 1, 1},
     .reads = {7, 103},
     .writes = {307},
     .bindings = {{.set = 1,
                   .binding = 2,
                   .resource_id = 7,
                   .access = vk::ResourceAccess::Read},
                  {.set = 1,
                   .binding = 7,
                   .resource_id = 103,
                   .access = vk::ResourceAccess::Read},
                  {.set = 1,
                   .binding = 11,
                   .resource_id = 307,
                   .access = vk::ResourceAccess::Write}},
     .push_constants = {{.offset = 0, .bytes = {17, 0, 0, 0}}}},
    {.id = 909,
     .operation = "fixedrun_stage1",
     .entry_point = "compute_main",
     .local_size = {64, 1, 1},
     .reads = {307},
     .writes = {41, 59},
     .bindings = {{.set = 2,
                   .binding = 0,
                   .resource_id = 307,
                   .access = vk::ResourceAccess::Read},
                  {.set = 2,
                   .binding = 3,
                   .resource_id = 41,
                   .access = vk::ResourceAccess::Write},
                  {.set = 2,
                   .binding = 9,
                   .resource_id = 59,
                   .access = vk::ResourceAccess::Write}},
     .push_constants = {}}};

  vk::VulkanExecutionPlan plan{
    .schema_version = vk::kExecutionPlanSchemaVersion,
    .model = std::string(kModelName),
    .build_identity = "va4-fixedrun-fake-fixture",
    .host_triple = std::string(kHostTriple),
    .device_target = "vulkan1.1-spirv1.0",
    .resources = std::move(resources),
    .kernels = std::move(kernels),
    .inputs = {7},
    .outputs = {41, 59},
    .plan_hash = ""};

  std::vector<vk::VulkanModelShader> shaders;
  shaders.reserve(plan.kernels.size());
  for (const auto& kernel : plan.kernels) {
    auto shader = vk::test::makeTestVulkanShader(
      plan, kernel, "va4_fixedrun_shader_" + std::to_string(kernel.id));
    if (!shader) {
      throw std::runtime_error("cannot build fixed-run test shader: " +
                               shader.error());
    }
    shaders.push_back(std::move(*shader));
  }
  std::vector<vk::VulkanModelWeight> weights;
  weights.push_back({.resource_id = 103, .bytes = std::move(weight_bytes)});
  auto serialized = vk::serializeExecutionPlan(plan);
  if (!serialized) {
    throw std::runtime_error("cannot serialize fixed-run test plan: " +
                             serialized.error());
  }
  auto package = ncnn_compile::makeVulkanModelPackage(kModelName,
                                                      kHostTriple,
                                                      *serialized,
                                                      shaders,
                                                      weights,
                                                      "fixedrun-fake-param",
                                                      "fixedrun-fake-bin");
  if (!package) {
    throw std::runtime_error("cannot create fixed-run test package: " +
                             package.error());
  }
  return std::move(*package);
}

std::uint32_t expectedStageMask(vk::HostStage stage) {
  switch (stage) {
    case vk::HostStage::Host:
      return VK_PIPELINE_STAGE_HOST_BIT;
    case vk::HostStage::Transfer:
      return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case vk::HostStage::Compute:
      return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  throw std::runtime_error("unknown host stage in fake fixture expectations");
}

std::uint32_t expectedAccessMask(vk::HostAccess access) {
  switch (access) {
    case vk::HostAccess::HostWrite:
      return VK_ACCESS_HOST_WRITE_BIT;
    case vk::HostAccess::TransferRead:
      return VK_ACCESS_TRANSFER_READ_BIT;
    case vk::HostAccess::TransferWrite:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    case vk::HostAccess::ShaderRead:
      return VK_ACCESS_SHADER_READ_BIT;
    case vk::HostAccess::ShaderWrite:
      return VK_ACCESS_SHADER_WRITE_BIT;
    case vk::HostAccess::ShaderReadWrite:
      return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    case vk::HostAccess::HostRead:
      return VK_ACCESS_HOST_READ_BIT;
  }
  throw std::runtime_error("unknown host access in fake fixture expectations");
}

struct ExpectedCommandData final {
  std::string_view kind;
  std::uint32_t source_stage = 0;
  std::uint32_t destination_stage = 0;
  std::uint32_t barrier_begin = 0;
  std::uint32_t barrier_count = 0;
  std::uint32_t source_buffer = 0;
  std::uint32_t destination_buffer = 0;
  std::uint64_t source_offset = 0;
  std::uint64_t destination_offset = 0;
  std::uint64_t byte_size = 0;
  std::uint32_t copy_region_count = 0;
  std::uint32_t pipeline_slot = 0;
  std::uint32_t first_set = 0;
  std::uint32_t set_count = 0;
  std::uint32_t descriptor_first_slot = 0;
  std::uint32_t push_offset = 0;
  std::uint32_t push_begin = 0;
  std::uint32_t push_count = 0;
  std::array<std::uint32_t, 3> groups{};
};

std::string emitFixtureExpectations(const vk::VulkanHostProgram& program) {
  std::map<std::uint32_t, std::uint32_t> pipeline_slots;
  std::map<std::uint32_t, std::uint32_t> descriptor_bases;
  std::vector<std::uint32_t> pipeline_ids;
  std::vector<std::uint32_t> descriptor_pipeline_slots;
  std::vector<std::string> descriptor_lines;
  std::uint32_t descriptor_count = 0;
  for (std::size_t slot = 0; slot < program.pipelines.size(); ++slot) {
    const auto& pipeline = program.pipelines[slot];
    pipeline_slots.emplace(pipeline.id.value, static_cast<std::uint32_t>(slot));
    pipeline_ids.push_back(pipeline.id.value);
    descriptor_bases.emplace(pipeline.id.value, descriptor_count);
    std::uint32_t set_count = 0;
    for (const auto& binding : pipeline.bindings) {
      set_count = std::max(set_count, binding.set + 1);
    }
    for (std::uint32_t set = 0; set < set_count; ++set) {
      descriptor_pipeline_slots.push_back(static_cast<std::uint32_t>(slot));
    }
    if (set_count >
        std::numeric_limits<std::uint32_t>::max() - descriptor_count) {
      throw std::runtime_error("fake descriptor set expectation overflows");
    }
    descriptor_count += set_count;
    for (const auto& binding : pipeline.bindings) {
      descriptor_lines.push_back(std::format(
        "  ExpectedDescriptorWrite{{.structure_type = "
        "VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .has_next = false, "
        ".descriptor_slot = {}, .binding = {}, .buffer_slot = {}, "
        ".resource_access = {}, .array_element = 0, .descriptor_count = 1, "
        ".descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, "
        ".has_image_info = false, .has_buffer_info = true, "
        ".has_texel_buffer_view = false, .offset = 0, "
        ".range = UINT64_C({})}},\n",
        descriptor_bases.at(pipeline.id.value) + binding.set,
        binding.binding,
        binding.buffer.value,
        static_cast<std::uint32_t>(binding.access),
        program.buffers[binding.buffer.value].byte_size));
    }
  }

  std::vector<std::uint32_t> buffer_resources;
  buffer_resources.reserve(program.buffers.size());
  for (const auto& buffer : program.buffers) {
    buffer_resources.push_back(buffer.resource_id);
  }

  std::vector<std::string> command_lines;
  std::vector<std::string> barrier_lines;
  std::vector<std::uint8_t> push_bytes;
  for (const auto& operation : program.run_operations) {
    auto emitted = std::visit(
      [&](const auto& op) -> std::optional<ExpectedCommandData> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, vk::HostBarrierOp>) {
          const auto barrier_begin =
            static_cast<std::uint32_t>(barrier_lines.size());
          for (const auto& barrier : op.buffers) {
            barrier_lines.push_back(std::format(
              "  ExpectedBarrier{{.structure_type = "
              "VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .has_next = false, "
              ".buffer_slot = {}, .source_access = {}, "
              ".destination_access = {}, .offset = UINT64_C({}), "
              ".size = UINT64_C({})}},\n",
              barrier.buffer.value,
              expectedAccessMask(barrier.source_access),
              expectedAccessMask(barrier.destination_access),
              barrier.offset,
              barrier.byte_size));
          }
          return ExpectedCommandData{
            .kind = "Barrier",
            .source_stage = expectedStageMask(op.source_stage),
            .destination_stage = expectedStageMask(op.destination_stage),
            .barrier_begin = barrier_begin,
            .barrier_count = static_cast<std::uint32_t>(op.buffers.size())};
        } else if constexpr (std::is_same_v<T, vk::HostCopyOp>) {
          return ExpectedCommandData{
            .kind = "Copy",
            .source_buffer = op.source.value,
            .destination_buffer = op.destination.value,
            .source_offset = op.source_offset,
            .destination_offset = op.destination_offset,
            .byte_size = op.byte_size,
            .copy_region_count = 1};
        } else if constexpr (std::is_same_v<T, vk::HostBindPipelineOp>) {
          return ExpectedCommandData{
            .kind = "BindPipeline",
            .pipeline_slot = pipeline_slots.at(op.pipeline.value)};
        } else if constexpr (std::is_same_v<T, vk::HostBindDescriptorsOp>) {
          return ExpectedCommandData{
            .kind = "BindDescriptors",
            .pipeline_slot = pipeline_slots.at(op.pipeline.value),
            .first_set = op.first_set,
            .set_count = op.set_count,
            .descriptor_first_slot =
              descriptor_bases.at(op.pipeline.value) + op.first_set};
        } else if constexpr (std::is_same_v<T, vk::HostPushConstantsOp>) {
          const auto first = static_cast<std::uint32_t>(push_bytes.size());
          push_bytes.insert(push_bytes.end(), op.bytes.begin(), op.bytes.end());
          return ExpectedCommandData{
            .kind = "PushConstants",
            .pipeline_slot = pipeline_slots.at(op.pipeline.value),
            .push_offset = op.offset,
            .push_begin = first,
            .push_count = static_cast<std::uint32_t>(op.bytes.size())};
        } else if constexpr (std::is_same_v<T, vk::HostDispatchOp>) {
          return ExpectedCommandData{
            .kind = "Dispatch",
            .pipeline_slot = pipeline_slots.at(op.pipeline.value),
            .groups = op.groups};
        }
        return std::nullopt;
      },
      operation);
    if (!emitted) {
      continue;
    }
    command_lines.push_back(std::format(
      "  ExpectedCommand{{.kind = CommandKind::{}, .source_stage = {}, "
      ".destination_stage = {}, .barrier_begin = {}, .barrier_count = {}, "
      ".source_buffer = {}, .destination_buffer = {}, "
      ".source_offset = UINT64_C({}), .destination_offset = UINT64_C({}), "
      ".byte_size = UINT64_C({}), .copy_region_count = {}, "
      ".pipeline_slot = {}, .first_set = {}, .set_count = {}, "
      ".descriptor_first_slot = {}, .push_offset = {}, .push_begin = {}, "
      ".push_count = {}, .groups = {{{}, {}, {}}}}},\n",
      emitted->kind,
      emitted->source_stage,
      emitted->destination_stage,
      emitted->barrier_begin,
      emitted->barrier_count,
      emitted->source_buffer,
      emitted->destination_buffer,
      emitted->source_offset,
      emitted->destination_offset,
      emitted->byte_size,
      emitted->copy_region_count,
      emitted->pipeline_slot,
      emitted->first_set,
      emitted->set_count,
      emitted->descriptor_first_slot,
      emitted->push_offset,
      emitted->push_begin,
      emitted->push_count,
      emitted->groups[0],
      emitted->groups[1],
      emitted->groups[2]));
  }

  const auto appendArray = [](std::string& output,
                              std::string_view element_type,
                              std::string_view name,
                              const std::vector<std::string>& lines) {
    output += std::format("inline constexpr std::array<{}, {}> {}{{{{\n",
                          element_type,
                          lines.size(),
                          name);
    for (const auto& line : lines) {
      output += line;
    }
    output += "}};\n";
  };
  std::string output =
    "#pragma once\n"
    "#include <array>\n"
    "#include <cstddef>\n"
    "#include <cstdint>\n"
    "#include <vulkan/vulkan.h>\n\n"
    "namespace fixedrun_expectations {\n"
    "enum class CommandKind : std::uint32_t { Barrier, Copy, BindPipeline, "
    "BindDescriptors, PushConstants, Dispatch };\n"
    "struct ExpectedCommand final {\n"
    "  CommandKind kind;\n"
    "  std::uint32_t source_stage;\n"
    "  std::uint32_t destination_stage;\n"
    "  std::uint32_t barrier_begin;\n"
    "  std::uint32_t barrier_count;\n"
    "  std::uint32_t source_buffer;\n"
    "  std::uint32_t destination_buffer;\n"
    "  std::uint64_t source_offset;\n"
    "  std::uint64_t destination_offset;\n"
    "  std::uint64_t byte_size;\n"
    "  std::uint32_t copy_region_count;\n"
    "  std::uint32_t pipeline_slot;\n"
    "  std::uint32_t first_set;\n"
    "  std::uint32_t set_count;\n"
    "  std::uint32_t descriptor_first_slot;\n"
    "  std::uint32_t push_offset;\n"
    "  std::uint32_t push_begin;\n"
    "  std::uint32_t push_count;\n"
    "  std::array<std::uint32_t, 3> groups;\n"
    "};\n"
    "struct ExpectedBarrier final {\n"
    "  std::uint32_t structure_type;\n"
    "  bool has_next;\n"
    "  std::uint32_t buffer_slot;\n"
    "  std::uint32_t source_access;\n"
    "  std::uint32_t destination_access;\n"
    "  std::uint64_t offset;\n"
    "  std::uint64_t size;\n"
    "};\n"
    "struct ExpectedDescriptorWrite final {\n"
    "  std::uint32_t structure_type;\n"
    "  bool has_next;\n"
    "  std::uint32_t descriptor_slot;\n"
    "  std::uint32_t binding;\n"
    "  std::uint32_t buffer_slot;\n"
    "  std::uint32_t resource_access;\n"
    "  std::uint32_t array_element;\n"
    "  std::uint32_t descriptor_count;\n"
    "  std::uint32_t descriptor_type;\n"
    "  bool has_image_info;\n"
    "  bool has_buffer_info;\n"
    "  bool has_texel_buffer_view;\n"
    "  std::uint64_t offset;\n"
    "  std::uint64_t range;\n"
    "};\n"
    "inline constexpr std::uint32_t kCommandBufferSlot = 0;\n"
    "inline constexpr std::uint32_t kDescriptorCopyCount = 0;\n"
    "inline constexpr std::uint32_t kLiteralResourceRead = 0;\n"
    "inline constexpr std::uint32_t kLiteralResourceWrite = 1;\n"
    "inline constexpr std::uint32_t kLiteralHostStage = " +
    std::to_string(expectedStageMask(vk::HostStage::Host)) + ";\n";
  output += std::format(
    "inline constexpr std::uint32_t kLiteralTransferStage = {};\n"
    "inline constexpr std::uint32_t kLiteralComputeStage = {};\n"
    "inline constexpr std::uint32_t kLiteralHostWriteAccess = {};\n"
    "inline constexpr std::uint32_t kLiteralTransferReadAccess = {};\n"
    "inline constexpr std::uint32_t kLiteralTransferWriteAccess = {};\n"
    "inline constexpr std::uint32_t kLiteralShaderReadAccess = {};\n"
    "inline constexpr std::uint32_t kLiteralShaderWriteAccess = {};\n"
    "inline constexpr std::uint32_t kLiteralShaderReadWriteAccess = {};\n"
    "inline constexpr std::uint32_t kLiteralHostReadAccess = {};\n",
    expectedStageMask(vk::HostStage::Transfer),
    expectedStageMask(vk::HostStage::Compute),
    expectedAccessMask(vk::HostAccess::HostWrite),
    expectedAccessMask(vk::HostAccess::TransferRead),
    expectedAccessMask(vk::HostAccess::TransferWrite),
    expectedAccessMask(vk::HostAccess::ShaderRead),
    expectedAccessMask(vk::HostAccess::ShaderWrite),
    expectedAccessMask(vk::HostAccess::ShaderReadWrite),
    expectedAccessMask(vk::HostAccess::HostRead));

  const auto appendNumericArray = [&output](std::string_view name,
                                            const auto& values) {
    output += std::format(
      "inline constexpr std::array<std::uint32_t, {}> "
      "{}{{{{\n",
      values.size(),
      name);
    for (const auto value : values) {
      output += std::format("  {},\n", value);
    }
    output += "}};\n";
  };
  appendNumericArray("kBufferResourceIds", buffer_resources);
  appendNumericArray("kPipelineIds", pipeline_ids);
  appendNumericArray("kDescriptorSetPipelineSlots", descriptor_pipeline_slots);
  appendArray(output, "ExpectedCommand", "kCommands", command_lines);
  appendArray(output, "ExpectedBarrier", "kBarriers", barrier_lines);
  appendArray(
    output, "ExpectedDescriptorWrite", "kDescriptorWrites", descriptor_lines);
  output += std::format(
    "inline constexpr std::array<std::uint8_t, {}> "
    "kPushBytes{{{{\n",
    push_bytes.size());
  for (const auto byte : push_bytes) {
    output += std::format("  UINT8_C({}),\n", byte);
  }
  output += "}};\n}  // namespace fixedrun_expectations\n";
  return output;
}

void writeFile(const fs::path& path, std::string_view bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output ||
      !output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("cannot write fixed-run fixture: " +
                             path.string());
  }
}

std::string encodeSpirv(std::span<const std::uint32_t> words) {
  std::string bytes;
  if (words.size() > bytes.max_size() / sizeof(std::uint32_t)) {
    throw std::runtime_error("SPIR-V fixture size exceeds string capacity");
  }
  bytes.reserve(words.size() * sizeof(std::uint32_t));
  for (std::uint32_t word : words) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<char>((word >> shift) & 0xffU));
    }
  }
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--output-dir") {
    std::cerr << "usage: vulkan_fixedrun_fixture_emitter --output-dir PATH\n";
    return 2;
  }
  try {
    const fs::path output_dir(argv[2]);
    fs::create_directories(output_dir);
    auto package = makePackage();
    auto generated = ncnn_compile::makeVulkanHostSource(package);
    if (!generated) {
      throw std::runtime_error("cannot generate fixed-run fixture: " +
                               generated.error());
    }
    auto host_program_json =
      vk::serializeVulkanHostProgram(package.host_program);
    if (!host_program_json) {
      throw std::runtime_error("cannot serialize fixed-run host program: " +
                               host_program_json.error());
    }
    writeFile(output_dir / "candidate_source.cpp", generated->source);
    writeFile(output_dir / "va4_fixedrun_vulkan.h", generated->header);
    writeFile(output_dir / "host-program-identity.txt",
              generated->host_program_identity + "\n");
    writeFile(output_dir / "codegen-identity.txt",
              generated->codegen_identity + "\n");
    writeFile(output_dir / "codegen-version.txt",
              generated->codegen_version + "\n");
    const std::string metadata = std::format(
      "{{\n  \"schema_version\": 1,\n"
      "  \"kind\": \"ncnn.vulkan_fixedrun_fake_fixture\",\n"
      "  \"execution_role\": \"synthetic-fault-only\",\n"
      "  \"real_model_oracle\": false,\n"
      "  \"host_program_identity\": \"{}\",\n"
      "  \"codegen_identity\": \"{}\",\n"
      "  \"codegen_version\": \"{}\"\n}}\n",
      generated->host_program_identity,
      generated->codegen_identity,
      generated->codegen_version);
    writeFile(output_dir / "metadata.json", metadata);
    writeFile(output_dir / "host_program.json", *host_program_json);
    writeFile(output_dir / "fixture_expectations.hpp",
              emitFixtureExpectations(package.host_program));
    writeFile(output_dir / "plan.json", package.plan_json);
    writeFile(output_dir / "package_manifest.json", package.manifest_json);
    for (std::size_t index = 0; index < package.shaders.size(); ++index) {
      writeFile(output_dir / ("shader_" + std::to_string(index) + ".spv"),
                encodeSpirv(package.shaders[index].words));
    }
  } catch (const std::exception& error) {
    std::cerr << "vulkan_fixedrun_fixture_emitter: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
