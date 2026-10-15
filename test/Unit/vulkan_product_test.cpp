#include "VulkanProduct.hpp"
#include "vulkan_shader_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace ncnn_compile {
namespace {

using ncnn_mlir::vulkan::DescriptorBinding;
using ncnn_mlir::vulkan::KernelExecution;
using ncnn_mlir::vulkan::ResourceAccess;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;
using ncnn_mlir::vulkan::VulkanModelShader;
using ncnn_mlir::vulkan::VulkanModelWeight;

VulkanExecutionPlan makeReluPlan() {
  return VulkanExecutionPlan{
    .schema_version = ncnn_mlir::vulkan::kExecutionPlanSchemaVersion,
    .model = "model",
    .build_identity = "lowering-fixture",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources =
      {
        TensorResource{.id = 1,
                       .kind = ResourceKind::Input,
                       .name = "input_0",
                       .shape = {1, 1, 2},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "linear",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 8,
                       .first_kernel = 0,
                       .last_kernel = 0},
        TensorResource{.id = 2,
                       .kind = ResourceKind::Output,
                       .name = "output",
                       .shape = {1, 1, 2},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "linear",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 8,
                       .first_kernel = 0,
                       .last_kernel = 0},
      },
    .kernels = {KernelExecution{
      .id = 0,
      .operation = "relu",
      .entry_point = "relu",
      .local_size = {64, 1, 1},
      .reads = {1},
      .writes = {2},
      .bindings = {{.set = 0,
                    .binding = 0,
                    .resource_id = 1,
                    .access = ResourceAccess::Read},
                   {.set = 0,
                    .binding = 1,
                    .resource_id = 2,
                    .access = ResourceAccess::Write}},
      .push_constants = {},
    }},
    .inputs = {1},
    .outputs = {2},
    .plan_hash = "",
  };
}

VulkanModelShader makeTestShader(const VulkanExecutionPlan& plan,
                                 const KernelExecution& kernel,
                                 std::string binary_symbol) {
  auto shader = ncnn_mlir::vulkan::test::makeTestVulkanShader(
    plan, kernel, std::move(binary_symbol));
  if (!shader) {
    throw std::runtime_error("cannot build Vulkan shader test fixture: " +
                             shader.error());
  }
  return std::move(*shader);
}

bool changeFixtureFloatLiteral(VulkanModelShader& shader) {
  std::uint32_t float_type = 0;
  for (std::size_t cursor = 5; cursor < shader.words.size();) {
    const std::uint32_t first = shader.words[cursor];
    const std::size_t count = first >> 16;
    const std::uint32_t opcode = first & 0xffffU;
    if (count == 0 || count > shader.words.size() - cursor) {
      return false;
    }
    if (opcode == 22 && count == 3) {
      float_type = shader.words[cursor + 1];
    } else if (opcode == 43 && count == 4 && float_type != 0 &&
               shader.words[cursor + 1] == float_type) {
      shader.words[cursor + 3] = UINT32_C(0x3f800000);
      return true;
    }
    cursor += count;
  }
  return false;
}

std::string encodeSpirvWords(std::span<const std::uint32_t> words) {
  std::string bytes;
  bytes.reserve(words.size() * sizeof(std::uint32_t));
  for (std::uint32_t word : words) {
    bytes.push_back(static_cast<char>(word & 0xff));
    bytes.push_back(static_cast<char>((word >> 8) & 0xff));
    bytes.push_back(static_cast<char>((word >> 16) & 0xff));
    bytes.push_back(static_cast<char>((word >> 24) & 0xff));
  }
  return bytes;
}

std::string makeSpirvBytes(const VulkanExecutionPlan& plan) {
  const VulkanModelShader shader =
    makeTestShader(plan, plan.kernels.front(), plan.model + "_vulkan");
  return encodeSpirvWords(shader.words);
}

std::string sha256Hex(std::string_view bytes) {
  llvm::SHA256 hash;
  hash.update(llvm::StringRef(bytes.data(), bytes.size()));
  return llvm::toHex(hash.final(), true);
}

void appendU64(std::string& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void appendIdentityField(std::string& output,
                         std::string_view tag,
                         std::string_view value) {
  appendU64(output, tag.size());
  output.append(tag);
  appendU64(output, value.size());
  output.append(value);
}

std::string semanticPlanSHA256(VulkanExecutionPlan plan,
                               std::string_view model,
                               std::string_view host_triple) {
  plan.model = model;
  plan.host_triple = host_triple;
  plan.build_identity = "vulkan-product-semantic-plan-v1";
  plan.plan_hash.clear();
  auto canonical = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  if (!canonical) {
    throw std::runtime_error("cannot serialize semantic plan fixture: " +
                             canonical.error());
  }
  return sha256Hex(*canonical);
}

std::string reorderPlanRootKeys(std::string_view json,
                                std::string_view first_key,
                                std::string_view second_key) {
  std::istringstream input{std::string(json)};
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    lines.push_back(std::move(line));
  }
  const auto find_line = [&lines](std::string_view key) {
    const std::string marker = std::format("\"{}\":", key);
    return std::ranges::find_if(lines, [&marker](const std::string& line) {
      return line.find(marker) != std::string::npos;
    });
  };
  const auto first = find_line(first_key);
  const auto second = find_line(second_key);
  if (first == lines.end() || second == lines.end() || first == second) {
    throw std::runtime_error("cannot find plan keys to reorder");
  }
  std::iter_swap(first, second);
  std::ostringstream output;
  for (std::size_t index = 0; index < lines.size(); ++index) {
    if (index != 0) {
      output << '\n';
    }
    output << lines[index];
  }
  output << '\n';
  return output.str();
}

VulkanExecutionPlan makeTwoKernelPlan(const VulkanModelWeight& weight) {
  const auto add_binding = [](std::uint32_t binding,
                              std::uint32_t resource_id,
                              ResourceAccess access) {
    return DescriptorBinding{.set = 0,
                             .binding = binding,
                             .resource_id = resource_id,
                             .access = access};
  };
  VulkanExecutionPlan plan{
    .schema_version = ncnn_mlir::vulkan::kExecutionPlanSchemaVersion,
    .model = "model",
    .build_identity = "static-plan-fixture",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources =
      {
        TensorResource{.id = 1,
                       .kind = ResourceKind::Input,
                       .name = "input_0",
                       .shape = {1, 1, 2},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "linear",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 8,
                       .first_kernel = 0,
                       .last_kernel = 0},
        TensorResource{.id = 2,
                       .kind = ResourceKind::Input,
                       .name = "input_1",
                       .shape = {1, 1, 2},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "linear",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 8,
                       .first_kernel = 0,
                       .last_kernel = 0},
        TensorResource{.id = weight.resource_id,
                       .kind = ResourceKind::Weight,
                       .name = "weight_0",
                       .shape = {1, 1, 1, 1},
                       .logical_layout = "OIHW",
                       .physical_layout = "OHWI",
                       .host_layout = "",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 4,
                       .first_kernel = 0,
                       .last_kernel = 0},
        TensorResource{.id = 4,
                       .kind = ResourceKind::Intermediate,
                       .name = "intermediate_0",
                       .shape = {1, 1, 2},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 8,
                       .first_kernel = 0,
                       .last_kernel = 1},
        TensorResource{.id = 5,
                       .kind = ResourceKind::Output,
                       .name = "output_0",
                       .shape = {1, 1, 2},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "linear",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 8,
                       .first_kernel = 1,
                       .last_kernel = 1},
      },
    .kernels =
      {KernelExecution{
         .id = 10,
         .operation = "conv2d",
         .entry_point = "kernel_10",
         .local_size = {8, 1, 1},
         .reads = {1, 2, weight.resource_id},
         .writes = {4},
         .bindings = {add_binding(0, 1, ResourceAccess::Read),
                      add_binding(1, 2, ResourceAccess::Read),
                      add_binding(2, weight.resource_id, ResourceAccess::Read),
                      add_binding(3, 4, ResourceAccess::Write)},
         .push_constants = {}},
       KernelExecution{.id = 20,
                       .operation = "relu",
                       .entry_point = "kernel_20",
                       .local_size = {8, 1, 1},
                       .reads = {4},
                       .writes = {5},
                       .bindings = {add_binding(0, 4, ResourceAccess::Read),
                                    add_binding(1, 5, ResourceAccess::Write)},
                       .push_constants = {}}},
    .inputs = {1, 2},
    .outputs = {5},
    .plan_hash = "",
  };
  const auto weight_hash =
    ncnn_mlir::vulkan::vulkanWeightSHA256(plan.resources[2], weight.bytes);
  if (weight_hash) {
    plan.resources[2].weight_sha256 = *weight_hash;
  }
  return plan;
}

TEST(VulkanProductTest, PackagesReproduciblyAndEmbedsBothIdentities) {
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(makeReluPlan());
  ASSERT_TRUE(serialized) << serialized.error();
  const std::string shader = makeSpirvBytes(makeReluPlan());
  auto first = makeVulkanModelPackage("tiny_relu",
                                      "x86_64-linux-gnu",
                                      *serialized,
                                      shader,
                                      "param-fixture",
                                      "bin-fixture");
  ASSERT_TRUE(first) << first.error();
  auto second = makeVulkanModelPackage("tiny_relu",
                                       "x86_64-linux-gnu",
                                       *serialized,
                                       shader,
                                       "param-fixture",
                                       "bin-fixture");
  ASSERT_TRUE(second) << second.error();

  EXPECT_EQ(first->plan_json, second->plan_json);
  EXPECT_EQ(first->header, second->header);
  EXPECT_EQ(first->wrapper_source, second->wrapper_source);
  EXPECT_EQ(first->manifest_json, second->manifest_json);
  EXPECT_EQ(first->binary_symbol, "model_vulkan");
  EXPECT_EQ(first->plan.model, "tiny_relu");
  EXPECT_EQ(first->shader_words.front(), UINT32_C(0x07230203));
  EXPECT_NE(first->manifest_json.find(first->shader_sha256), std::string::npos);
  EXPECT_NE(first->manifest_json.find(first->plan_sha256), std::string::npos);
  EXPECT_NE(first->header.find("TINY_RELU_VULKAN_INPUT1_ELEMENTS UINT64_C(2)"),
            std::string::npos);
  EXPECT_NE(first->wrapper_source.find("tiny_relu_vk_create"),
            std::string::npos);
}

TEST(VulkanProductTest, SemanticPlanIdentityIgnoresJsonFormattingAndKeyOrder) {
  const VulkanExecutionPlan relu_plan = makeReluPlan();
  auto relu_json = ncnn_mlir::vulkan::serializeExecutionPlan(relu_plan);
  ASSERT_TRUE(relu_json) << relu_json.error();
  const std::string relu_shader = makeSpirvBytes(relu_plan);
  const std::string relu_reordered =
    reorderPlanRootKeys(*relu_json, "model", "host_triple");
  const std::string relu_whitespace = " \n\t" + *relu_json + " \n";
  auto legacy_base = makeVulkanModelPackage("semantic_relu",
                                            "x86_64-linux-gnu",
                                            *relu_json,
                                            relu_shader,
                                            "param",
                                            "bin");
  auto legacy_reordered = makeVulkanModelPackage("semantic_relu",
                                                 "x86_64-linux-gnu",
                                                 relu_reordered,
                                                 relu_shader,
                                                 "param",
                                                 "bin");
  auto legacy_whitespace = makeVulkanModelPackage("semantic_relu",
                                                  "x86_64-linux-gnu",
                                                  relu_whitespace,
                                                  relu_shader,
                                                  "param",
                                                  "bin");
  ASSERT_TRUE(legacy_base) << legacy_base.error();
  ASSERT_TRUE(legacy_reordered) << legacy_reordered.error();
  ASSERT_TRUE(legacy_whitespace) << legacy_whitespace.error();
  EXPECT_EQ(legacy_base->build_identity, legacy_reordered->build_identity);
  EXPECT_EQ(legacy_base->build_identity, legacy_whitespace->build_identity);
  EXPECT_EQ(legacy_base->host_program.identity,
            legacy_reordered->host_program.identity);
  EXPECT_EQ(legacy_base->plan_json, legacy_reordered->plan_json);

  const VulkanModelWeight weight{
    .resource_id = 3,
    .bytes = {0x00, 0x00, 0x80, 0x3f},
  };
  const VulkanExecutionPlan general_plan = makeTwoKernelPlan(weight);
  auto general_json = ncnn_mlir::vulkan::serializeExecutionPlan(general_plan);
  ASSERT_TRUE(general_json) << general_json.error();
  const std::array<VulkanModelShader, 2> shaders{
    makeTestShader(general_plan, general_plan.kernels[1], "model_vulkan_k20"),
    makeTestShader(general_plan, general_plan.kernels[0], "model_vulkan_k10")};
  const std::array<VulkanModelWeight, 1> weights{weight};
  const std::string general_reordered =
    reorderPlanRootKeys(*general_json, "model", "host_triple");
  const std::string general_whitespace = " \n\t" + *general_json + " \n";
  auto general_base = makeVulkanModelPackage("semantic_general",
                                             "x86_64-linux-gnu",
                                             *general_json,
                                             shaders,
                                             weights,
                                             "param",
                                             "bin");
  auto general_reordered_package = makeVulkanModelPackage("semantic_general",
                                                          "x86_64-linux-gnu",
                                                          general_reordered,
                                                          shaders,
                                                          weights,
                                                          "param",
                                                          "bin");
  auto general_whitespace_package = makeVulkanModelPackage("semantic_general",
                                                           "x86_64-linux-gnu",
                                                           general_whitespace,
                                                           shaders,
                                                           weights,
                                                           "param",
                                                           "bin");
  ASSERT_TRUE(general_base) << general_base.error();
  ASSERT_TRUE(general_reordered_package) << general_reordered_package.error();
  ASSERT_TRUE(general_whitespace_package) << general_whitespace_package.error();
  EXPECT_EQ(general_base->build_identity,
            general_reordered_package->build_identity);
  EXPECT_EQ(general_base->build_identity,
            general_whitespace_package->build_identity);
  EXPECT_EQ(general_base->host_program.identity,
            general_reordered_package->host_program.identity);
  EXPECT_EQ(general_base->plan_json, general_reordered_package->plan_json);
}

TEST(VulkanProductTest, ManifestPinsHostExecutionContractAndIdentityDomain) {
  const auto expect_host_contract = [](const VulkanModelPackage& package) {
    EXPECT_NE(package.manifest_json.find(
                "\"host_implementation\": \"generic-plan-runtime\""),
              std::string::npos);
    EXPECT_NE(package.manifest_json.find("\"host_codegen_version\": 1"),
              std::string::npos);
    EXPECT_NE(package.manifest_json.find(
                "\"execution_plan_usage\": \"embedded-runtime-required\""),
              std::string::npos);
    EXPECT_NE(package.manifest_json.find(
                "\"host_program_usage\": \"compiler-only-diagnostic\""),
              std::string::npos);
    EXPECT_NE(package.manifest_json.find("\"host_program\": {"),
              std::string::npos);
    EXPECT_NE(package.manifest_json.find("\"host_program_identity\": \"" +
                                         package.host_program.identity + "\""),
              std::string::npos);
    EXPECT_NE(
      package.manifest_json.find("\"schema\": \"ncnn-vulkan-host-program/v1\""),
      std::string::npos);
    EXPECT_NE(package.manifest_json.find(
                "\"codegen_version\": \"ncnn-vulkan-host-codegen/v1\""),
              std::string::npos);
    EXPECT_NE(
      package.manifest_json.find("\"helper_contract\": "
                                 "\"ncnn-vulkan-device-helper-contract/v1\""),
      std::string::npos);
    EXPECT_NE(package.manifest_json.find(
                "\"synchronization_version\": \"ncnn-vulkan-host-sync/v1\""),
              std::string::npos);
    EXPECT_NE(
      package.manifest_json.find("\"canonical_encoding\": "
                                 "\"length-prefixed-canonical-json/v1\""),
      std::string::npos);
    EXPECT_EQ(package.manifest_json.find("\"host_implementation\": \"aot\""),
              std::string::npos);
    EXPECT_EQ(package.host_program.schema,
              ncnn_mlir::vulkan::kHostProgramSchema);
    EXPECT_EQ(package.host_program.codegen_version,
              ncnn_mlir::vulkan::kHostProgramCodegenVersion);
    EXPECT_EQ(package.host_program.helper_contract,
              ncnn_mlir::vulkan::kHostProgramHelperContract);
    EXPECT_EQ(package.host_program.synchronization_version,
              ncnn_mlir::vulkan::kHostProgramSynchronizationVersion);
    EXPECT_EQ(package.host_program.canonical_encoding,
              ncnn_mlir::vulkan::kHostProgramCanonicalEncoding);
  };

  const VulkanExecutionPlan legacy_source_plan = makeReluPlan();
  const std::string shader = makeSpirvBytes(legacy_source_plan);
  auto legacy_plan =
    ncnn_mlir::vulkan::serializeExecutionPlan(legacy_source_plan);
  ASSERT_TRUE(legacy_plan) << legacy_plan.error();
  auto legacy_first = makeVulkanModelPackage(
    "host_contract", "x86_64-linux-gnu", *legacy_plan, shader, "param", "bin");
  ASSERT_TRUE(legacy_first) << legacy_first.error();
  auto legacy_repeat = makeVulkanModelPackage(
    "host_contract", "x86_64-linux-gnu", *legacy_plan, shader, "param", "bin");
  ASSERT_TRUE(legacy_repeat) << legacy_repeat.error();
  auto legacy_changed = makeVulkanModelPackage("host_contract",
                                               "x86_64-linux-gnu",
                                               *legacy_plan,
                                               shader,
                                               "changed-param",
                                               "bin");
  ASSERT_TRUE(legacy_changed) << legacy_changed.error();
  expect_host_contract(*legacy_first);
  std::string legacy_identity_input;
  appendIdentityField(
    legacy_identity_input, "domain", "ncnn-vulkan-product-v2");
  appendIdentityField(legacy_identity_input, "model", "host_contract");
  appendIdentityField(legacy_identity_input, "host_triple", "x86_64-linux-gnu");
  appendIdentityField(
    legacy_identity_input, "param_sha256", sha256Hex("param"));
  appendIdentityField(legacy_identity_input, "bin_sha256", sha256Hex("bin"));
  appendIdentityField(
    legacy_identity_input,
    "semantic_plan_sha256",
    semanticPlanSHA256(
      legacy_source_plan, "host_contract", "x86_64-linux-gnu"));
  appendIdentityField(
    legacy_identity_input, "host_implementation", "generic-plan-runtime");
  appendIdentityField(legacy_identity_input, "host_codegen_version", "1");
  appendIdentityField(legacy_identity_input,
                      "host_program_identity",
                      legacy_first->host_program.identity);
  appendIdentityField(legacy_identity_input, "shader_count", "1");
  appendIdentityField(legacy_identity_input,
                      "shader_kernel_id",
                      std::to_string(legacy_first->plan.kernels.front().id));
  appendIdentityField(
    legacy_identity_input, "shader_binary_symbol", legacy_first->binary_symbol);
  appendIdentityField(
    legacy_identity_input, "shader_sha256", sha256Hex(shader));
  appendIdentityField(legacy_identity_input, "weight_count", "0");
  EXPECT_EQ(legacy_first->build_identity,
            "ncnn-vulkan-product-v2-" + sha256Hex(legacy_identity_input));
  EXPECT_EQ(legacy_first->build_identity, legacy_repeat->build_identity);
  EXPECT_NE(legacy_first->build_identity, legacy_changed->build_identity);

  const VulkanModelWeight weight{
    .resource_id = 3,
    .bytes = {0x00, 0x00, 0x80, 0x3f},
  };
  const VulkanExecutionPlan general_plan = makeTwoKernelPlan(weight);
  auto general_json = ncnn_mlir::vulkan::serializeExecutionPlan(general_plan);
  ASSERT_TRUE(general_json) << general_json.error();
  const std::array<VulkanModelShader, 2> shaders = {
    makeTestShader(general_plan, general_plan.kernels[0], "model_vulkan_k10"),
    makeTestShader(general_plan, general_plan.kernels[1], "model_vulkan_k20"),
  };
  const std::array<VulkanModelWeight, 1> weights = {weight};
  auto general_first = makeVulkanModelPackage("host_contract_general",
                                              "x86_64-linux-gnu",
                                              *general_json,
                                              shaders,
                                              weights,
                                              "param",
                                              "bin");
  ASSERT_TRUE(general_first) << general_first.error();
  auto general_repeat = makeVulkanModelPackage("host_contract_general",
                                               "x86_64-linux-gnu",
                                               *general_json,
                                               shaders,
                                               weights,
                                               "param",
                                               "bin");
  ASSERT_TRUE(general_repeat) << general_repeat.error();
  auto general_changed = makeVulkanModelPackage("host_contract_general",
                                                "x86_64-linux-gnu",
                                                *general_json,
                                                shaders,
                                                weights,
                                                "changed-param",
                                                "bin");
  ASSERT_TRUE(general_changed) << general_changed.error();
  expect_host_contract(*general_first);
  std::string general_identity_input;
  appendIdentityField(
    general_identity_input, "domain", "ncnn-vulkan-static-plan-v3");
  appendIdentityField(general_identity_input, "model", "host_contract_general");
  appendIdentityField(
    general_identity_input, "host_triple", "x86_64-linux-gnu");
  appendIdentityField(
    general_identity_input, "param_sha256", sha256Hex("param"));
  appendIdentityField(general_identity_input, "bin_sha256", sha256Hex("bin"));
  appendIdentityField(
    general_identity_input,
    "semantic_plan_sha256",
    semanticPlanSHA256(
      general_plan, "host_contract_general", "x86_64-linux-gnu"));
  appendIdentityField(
    general_identity_input, "host_implementation", "generic-plan-runtime");
  appendIdentityField(general_identity_input, "host_codegen_version", "1");
  appendIdentityField(general_identity_input,
                      "host_program_identity",
                      general_first->host_program.identity);
  appendIdentityField(general_identity_input, "shader_count", "2");
  for (std::size_t index = 0; index < general_first->shaders.size(); ++index) {
    appendIdentityField(
      general_identity_input,
      "shader_kernel_id",
      std::to_string(general_first->shaders[index].kernel_id));
    appendIdentityField(general_identity_input,
                        "shader_binary_symbol",
                        general_first->shaders[index].binary_symbol);
    appendIdentityField(general_identity_input,
                        "shader_sha256",
                        general_first->shader_sha256s[index]);
  }
  appendIdentityField(general_identity_input, "weight_count", "1");
  appendIdentityField(general_identity_input, "weight_resource_id", "3");
  appendIdentityField(general_identity_input,
                      "weight_sha256",
                      sha256Hex(std::string_view("\0\0\x80\x3f", 4)));
  EXPECT_EQ(general_first->build_identity,
            "ncnn-vulkan-static-plan-v3-" + sha256Hex(general_identity_input));
  EXPECT_EQ(general_first->build_identity, general_repeat->build_identity);
  EXPECT_NE(general_first->build_identity, general_changed->build_identity);

  auto changed_shader = shaders;
  ASSERT_TRUE(changeFixtureFloatLiteral(changed_shader.front()));
  auto shader_changed = makeVulkanModelPackage("host_contract_general",
                                               "x86_64-linux-gnu",
                                               *general_json,
                                               changed_shader,
                                               weights,
                                               "param",
                                               "bin");
  ASSERT_TRUE(shader_changed) << shader_changed.error();
  EXPECT_NE(general_first->build_identity, shader_changed->build_identity);
  EXPECT_NE(general_first->host_program.identity,
            shader_changed->host_program.identity);

  const VulkanModelWeight changed_weight{
    .resource_id = 3,
    .bytes = {0x00, 0x00, 0x00, 0x3f},
  };
  const VulkanExecutionPlan changed_weight_plan =
    makeTwoKernelPlan(changed_weight);
  auto changed_weight_json =
    ncnn_mlir::vulkan::serializeExecutionPlan(changed_weight_plan);
  ASSERT_TRUE(changed_weight_json) << changed_weight_json.error();
  const std::array<VulkanModelShader, 2> changed_weight_shaders{
    makeTestShader(
      changed_weight_plan, changed_weight_plan.kernels[0], "model_vulkan_k10"),
    makeTestShader(
      changed_weight_plan, changed_weight_plan.kernels[1], "model_vulkan_k20")};
  const std::array<VulkanModelWeight, 1> changed_weights{changed_weight};
  auto weight_changed = makeVulkanModelPackage("host_contract_general",
                                               "x86_64-linux-gnu",
                                               *changed_weight_json,
                                               changed_weight_shaders,
                                               changed_weights,
                                               "param",
                                               "bin");
  ASSERT_TRUE(weight_changed) << weight_changed.error();
  EXPECT_NE(general_first->build_identity, weight_changed->build_identity);
  EXPECT_NE(general_first->host_program.identity,
            weight_changed->host_program.identity);
}

TEST(VulkanProductTest, RejectsMalformedShaderAndUnsupportedGraph) {
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(makeReluPlan());
  ASSERT_TRUE(serialized) << serialized.error();
  auto malformed = makeVulkanModelPackage(
    "tiny_relu", "x86_64-linux-gnu", *serialized, "not-spirv", "param", "bin");
  EXPECT_FALSE(malformed);

  VulkanExecutionPlan unsupported = makeReluPlan();
  unsupported.kernels.front().operation = "conv2d";
  unsupported.kernels.front().entry_point = "conv2d";
  unsupported.plan_hash.clear();
  auto unsupported_json =
    ncnn_mlir::vulkan::serializeExecutionPlan(unsupported);
  ASSERT_TRUE(unsupported_json) << unsupported_json.error();
  auto rejected = makeVulkanModelPackage("tiny_conv",
                                         "x86_64-linux-gnu",
                                         *unsupported_json,
                                         makeSpirvBytes(makeReluPlan()),
                                         "param",
                                         "weights");
  EXPECT_FALSE(rejected);
}

TEST(VulkanProductTest, ChecksSpirvEntryPointWorkgroupAndDescriptors) {
  VulkanExecutionPlan plan = makeReluPlan();
  const std::string converted = "gpu.binary @model_vulkan { object }";
  const std::string disassembly =
    "; Version: 1.0\n"
    "OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n"
    "OpEntryPoint GLCompute %main \"relu\"\n"
    "OpExecutionMode %main LocalSize 64 1 1\n"
    "OpDecorate %input DescriptorSet 0\n"
    "OpDecorate %input Binding 0\n"
    "OpDecorate %output DescriptorSet 0\n"
    "OpDecorate %output Binding 1\n";
  EXPECT_TRUE(validateVulkanSpirvInterface(plan, converted, disassembly));
  EXPECT_TRUE(validateVulkanSpirvInterface(
    plan,
    converted + "\ngpu.binary @model_vulkan_k10 { object }",
    disassembly));
  EXPECT_FALSE(validateVulkanSpirvInterface(
    plan, "gpu.binary @model_vulkan_k10 { object }", disassembly));
  EXPECT_FALSE(validateVulkanSpirvInterface(
    plan, converted + "\n" + converted, disassembly));

  const std::string wrong_entry_size =
    "; Version: 1.0\n"
    "OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n"
    "OpEntryPoint GLCompute %main \"relu\"\n"
    "OpExecutionMode %main LocalSize 32 1 1\n"
    "OpEntryPoint GLCompute %other \"other\"\n"
    "OpExecutionMode %other LocalSize 64 1 1\n"
    "OpDecorate %input DescriptorSet 0\n"
    "OpDecorate %input Binding 0\n"
    "OpDecorate %output DescriptorSet 0\n"
    "OpDecorate %output Binding 1\n";
  EXPECT_FALSE(validateVulkanSpirvInterface(plan, converted, wrong_entry_size));

  const std::string wrong_binding =
    "; Version: 1.0\n"
    "OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n"
    "OpEntryPoint GLCompute %main \"relu\"\n"
    "OpExecutionMode %main LocalSize 64 1 1\n"
    "OpDecorate %input DescriptorSet 0\n"
    "OpDecorate %input Binding 0\n"
    "OpDecorate %output DescriptorSet 0\n"
    "OpDecorate %output Binding 2\n";
  EXPECT_FALSE(validateVulkanSpirvInterface(plan, converted, wrong_binding));
}

TEST(VulkanProductTest, PackagesGeneralPlanByKernelAndResourceIdentity) {
  const VulkanModelWeight weight{
    .resource_id = 3,
    .bytes = {0x00, 0x00, 0x80, 0x3f},
  };
  const VulkanExecutionPlan plan = makeTwoKernelPlan(weight);
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  ASSERT_TRUE(serialized) << serialized.error();
  const std::array<VulkanModelShader, 2> shaders = {
    makeTestShader(plan, plan.kernels[1], "model_vulkan_k20"),
    makeTestShader(plan, plan.kernels[0], "model_vulkan_k10"),
  };
  const std::array<VulkanModelWeight, 1> weights = {weight};
  auto package = makeVulkanModelPackage("multi_layer",
                                        "x86_64-linux-gnu",
                                        *serialized,
                                        shaders,
                                        weights,
                                        "param-fixture",
                                        "bin-fixture");
  ASSERT_TRUE(package) << package.error();
  EXPECT_EQ(package->shaders.size(), 2);
  EXPECT_EQ(package->shaders[0].kernel_id, 10);
  EXPECT_EQ(package->shaders[1].kernel_id, 20);
  EXPECT_EQ(package->weights.size(), 1);
  EXPECT_EQ(package->binary_symbol, "model_vulkan_k10");
  EXPECT_NE(package->header.find("MULTI_LAYER_VULKAN_INPUT_COUNT UINT32_C(2)"),
            std::string::npos);
  EXPECT_NE(package->header.find("MULTI_LAYER_VULKAN_INPUT2_ELEMENTS"),
            std::string::npos);
  EXPECT_NE(package->header.find("#include \"ncnn-mlir/Vulkan/ModelABI.h\""),
            std::string::npos);
  EXPECT_NE(package->wrapper_source.find("#include \"multi_layer_vulkan.h\""),
            std::string::npos);
  EXPECT_NE(package->header.find("multi_layer_vk_run"), std::string::npos);
  EXPECT_NE(package->wrapper_source.find("kShaderWords0"), std::string::npos);
  EXPECT_NE(package->wrapper_source.find("kShaderWords1"), std::string::npos);
  EXPECT_NE(package->wrapper_source.find("kWeightBytes0"), std::string::npos);
  EXPECT_NE(package->manifest_json.find("\"shader_sha256s\""),
            std::string::npos);
  EXPECT_NE(package->manifest_json.find("\"weight_sha256s\""),
            std::string::npos);
  EXPECT_NE(package->manifest_json.find("model_vulkan_k10"), std::string::npos);
  EXPECT_NE(package->manifest_json.find("model_vulkan_k20"), std::string::npos);

  std::array<VulkanModelShader, 2> duplicate_shaders = shaders;
  duplicate_shaders[1].kernel_id = 20;
  EXPECT_FALSE(makeVulkanModelPackage("multi_layer",
                                      "x86_64-linux-gnu",
                                      *serialized,
                                      duplicate_shaders,
                                      weights,
                                      "param-fixture",
                                      "bin-fixture"));
  EXPECT_FALSE(makeVulkanModelPackage(
    "multi_layer",
    "x86_64-linux-gnu",
    *serialized,
    std::span<const VulkanModelShader>(shaders.data(), 1),
    weights,
    "param-fixture",
    "bin-fixture"));

  const std::string first_binary = "gpu.binary @model_vulkan_k10 { object }";
  const std::string first_disassembly =
    "; Version: 1.0\n"
    "OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n"
    "OpEntryPoint GLCompute %main \"kernel_10\"\n"
    "OpExecutionMode %main LocalSize 8 1 1\n"
    "OpDecorate %input0 DescriptorSet 0\n"
    "OpDecorate %input0 Binding 0\n"
    "OpDecorate %input1 DescriptorSet 0\n"
    "OpDecorate %input1 Binding 1\n"
    "OpDecorate %weight DescriptorSet 0\n"
    "OpDecorate %weight Binding 2\n"
    "OpDecorate %intermediate DescriptorSet 0\n"
    "OpDecorate %intermediate Binding 3\n";
  EXPECT_TRUE(validateVulkanSpirvInterface(plan,
                                           plan.kernels[0],
                                           "model_vulkan_k10",
                                           first_binary,
                                           first_disassembly));
  EXPECT_FALSE(validateVulkanSpirvInterface(plan,
                                            plan.kernels[1],
                                            "model_vulkan_k20",
                                            first_binary,
                                            first_disassembly));
}

TEST(VulkanProductTest, GeneralSingleTensorAbiKeepsPointerAndCountSignature) {
  const VulkanExecutionPlan plan = makeReluPlan();
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  ASSERT_TRUE(serialized) << serialized.error();
  const std::array<VulkanModelShader, 1> shaders = {
    makeTestShader(plan, plan.kernels.front(), "model_vulkan_k0"),
  };
  const std::array<VulkanModelWeight, 0> weights{};
  auto package = makeVulkanModelPackage("tiny_relu",
                                        "x86_64-linux-gnu",
                                        *serialized,
                                        shaders,
                                        weights,
                                        "param-fixture",
                                        "bin-fixture");
  ASSERT_TRUE(package) << package.error();
  EXPECT_NE(package->header.find("const float *input1, size_t input1_elements"),
            std::string::npos);
  EXPECT_NE(package->header.find("float *output1, size_t output1_elements"),
            std::string::npos);
  EXPECT_NE(
    package->wrapper_source.find("tiny_relu_vk_run(tiny_relu_vk_context* "
                                 "context, const float* input1"),
    std::string::npos);
}

TEST(VulkanProductTest, ManifestHashesPhysicalWeightsNotLogicalLayout) {
  VulkanModelWeight weight{.resource_id = 3,
                           .bytes = std::vector<std::uint8_t>(32)};
  for (std::size_t index = 0; index < weight.bytes.size(); ++index) {
    weight.bytes[index] = static_cast<std::uint8_t>(index);
  }
  VulkanExecutionPlan plan = makeTwoKernelPlan(
    VulkanModelWeight{.resource_id = 3, .bytes = {0x00, 0x00, 0x80, 0x3f}});
  TensorResource& resource = plan.resources[2];
  resource.shape = {2, 2, 2, 1};
  resource.byte_size = weight.bytes.size();
  auto logical_hash =
    ncnn_mlir::vulkan::vulkanWeightSHA256(resource, weight.bytes);
  ASSERT_TRUE(logical_hash) << logical_hash.error();
  resource.weight_sha256 = *logical_hash;
  TensorResource physical = resource;
  physical.logical_layout = physical.physical_layout;
  auto physical_hash =
    ncnn_mlir::vulkan::vulkanWeightSHA256(physical, weight.bytes);
  ASSERT_TRUE(physical_hash) << physical_hash.error();
  ASSERT_NE(*logical_hash, *physical_hash);
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  ASSERT_TRUE(serialized) << serialized.error();
  const std::array<VulkanModelShader, 2> shaders{
    makeTestShader(plan, plan.kernels[0], "model_vulkan_k10"),
    makeTestShader(plan, plan.kernels[1], "model_vulkan_k20")};
  const std::array<VulkanModelWeight, 1> weights{weight};
  auto package = makeVulkanModelPackage("physical_weights",
                                        "x86_64-linux-gnu",
                                        *serialized,
                                        shaders,
                                        weights,
                                        "param",
                                        "bin");
  ASSERT_TRUE(package) << package.error();
  EXPECT_EQ(package->plan.resources[2].weight_sha256, *logical_hash);
  auto manifest_json = llvm::json::parse(package->manifest_json);
  ASSERT_TRUE(static_cast<bool>(manifest_json));
  const llvm::json::Object* manifest = manifest_json->getAsObject();
  ASSERT_NE(manifest, nullptr);
  const llvm::json::Array* weight_records = manifest->getArray("weights");
  ASSERT_NE(weight_records, nullptr);
  ASSERT_EQ(weight_records->size(), 1);
  const llvm::json::Object* weight_record =
    weight_records->front().getAsObject();
  ASSERT_NE(weight_record, nullptr);
  const auto recorded_hash = weight_record->getString("sha256");
  ASSERT_TRUE(recorded_hash);
  EXPECT_EQ(*recorded_hash, *physical_hash);
  EXPECT_EQ(weight_record->get("logical_sha256"), nullptr);
  EXPECT_NE(package->manifest_json.find("\"logical_sha256\": \"" +
                                        *logical_hash + "\""),
            std::string::npos);
  EXPECT_NE(package->wrapper_source.find(
              "const std::array<VulkanResourcePayload, 1> kWeights"),
            std::string::npos);
  EXPECT_EQ(
    package->wrapper_source.find("constexpr std::array<VulkanResourcePayload"),
    std::string::npos);
}

TEST(VulkanProductTest, BuildIdentityIncludesStaticPublicPlanContract) {
  VulkanExecutionPlan plan = makeReluPlan();
  const std::array<VulkanModelShader, 1> shaders{
    makeTestShader(plan, plan.kernels.front(), "model_vulkan_k0")};
  auto first_json = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  ASSERT_TRUE(first_json) << first_json.error();
  auto first = makeVulkanModelPackage("plan_identity",
                                      "x86_64-linux-gnu",
                                      *first_json,
                                      shaders,
                                      {},
                                      "param",
                                      "bin");
  ASSERT_TRUE(first) << first.error();
  plan.resources[0].shape = {1, 2};
  plan.plan_hash.clear();
  auto second_json = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  ASSERT_TRUE(second_json) << second_json.error();
  auto second = makeVulkanModelPackage("plan_identity",
                                       "x86_64-linux-gnu",
                                       *second_json,
                                       shaders,
                                       {},
                                       "param",
                                       "bin");
  ASSERT_TRUE(second) << second.error();
  EXPECT_EQ(first->shader_sha256s, second->shader_sha256s);
  EXPECT_NE(first->header, second->header);
  EXPECT_NE(first->build_identity, second->build_identity);
}

TEST(VulkanProductTest, WeightSidecarIsStrictAndVerifiesEveryPayload) {
  const VulkanModelWeight weight{
    .resource_id = 3,
    .bytes = {0x00, 0x00, 0x80, 0x3f},
  };
  const VulkanExecutionPlan plan = makeTwoKernelPlan(weight);
  const std::array<VulkanModelWeight, 1> weights = {weight};
  auto json = ncnn_mlir::vulkan::serializeVulkanModelWeights(weights);
  ASSERT_TRUE(json) << json.error();
  EXPECT_NE(json->find("ncnn.vulkan_weight_payloads"), std::string::npos);
  EXPECT_NE(json->find("0000803f"), std::string::npos);
  auto decoded = ncnn_mlir::vulkan::deserializeVulkanModelWeights(*json);
  ASSERT_TRUE(decoded) << decoded.error();
  ASSERT_EQ(decoded->size(), 1);
  EXPECT_EQ(decoded->front().resource_id, 3);
  EXPECT_EQ(decoded->front().bytes, weight.bytes);
  EXPECT_TRUE(ncnn_mlir::vulkan::verifyVulkanModelWeights(plan, *decoded));

  EXPECT_FALSE(ncnn_mlir::vulkan::deserializeVulkanModelWeights("not-json"));
  EXPECT_FALSE(updateVulkanManifestDependencies("{", {}));
  std::array<VulkanModelWeight, 2> duplicate = {weight, weight};
  EXPECT_FALSE(ncnn_mlir::vulkan::serializeVulkanModelWeights(duplicate));
  EXPECT_FALSE(ncnn_mlir::vulkan::deserializeVulkanModelWeights(
    R"({"kind":"ncnn.vulkan_weight_payloads","schema_version":1,"weights":[{"resource_id":3,"bytes":"0000803F"}]})"));
  EXPECT_FALSE(ncnn_mlir::vulkan::deserializeVulkanModelWeights(
    R"({"kind":"ncnn.vulkan_weight_payloads","schema_version":1,"weights":[{"resource_id":3,"bytes":"0000803f"},{"resource_id":3,"bytes":"0000803f"}]})"));
  EXPECT_FALSE(ncnn_mlir::vulkan::deserializeVulkanModelWeights(
    R"({"kind":"ncnn.vulkan_weight_payloads","schema_version":1,"weights":[{"resource_id":3,"bytes":"0000"}]})"));
  EXPECT_FALSE(ncnn_mlir::vulkan::deserializeVulkanModelWeights(
    R"({"kind":"ncnn.vulkan_weight_payloads","schema_version":1,"weights":[{"resource_id":3,"bytes":"00"}]})"));
  EXPECT_FALSE(ncnn_mlir::vulkan::verifyVulkanModelWeights(
    plan, std::span<const VulkanModelWeight>{}));

  auto altered = weights;
  altered[0].bytes[0] ^= 1;
  EXPECT_FALSE(ncnn_mlir::vulkan::verifyVulkanModelWeights(plan, altered));
}

TEST(VulkanProductTest, LogicalWeightHashNormalizesBothConvLayouts) {
  TensorResource resource =
    makeTwoKernelPlan(
      VulkanModelWeight{.resource_id = 3, .bytes = {0x00, 0x00, 0x80, 0x3f}})
      .resources[2];
  resource.shape = {2, 3, 2, 1};
  resource.byte_size = 48;
  std::vector<std::uint8_t> logical(resource.byte_size);
  std::vector<std::uint8_t> ohwi(resource.byte_size);
  std::vector<std::uint8_t> hwcf(resource.byte_size);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    logical[index] = static_cast<std::uint8_t>(index);
  }
  constexpr std::size_t output_channels = 2;
  constexpr std::size_t input_channels = 3;
  constexpr std::size_t kernel_height = 2;
  constexpr std::size_t kernel_width = 1;
  for (std::size_t output = 0; output < output_channels; ++output) {
    for (std::size_t input = 0; input < input_channels; ++input) {
      for (std::size_t y = 0; y < kernel_height; ++y) {
        for (std::size_t x = 0; x < kernel_width; ++x) {
          const std::size_t logical_index =
            (((output * input_channels + input) * kernel_height + y) *
             kernel_width) +
            x;
          const std::size_t ohwi_index =
            (((output * kernel_height + y) * kernel_width + x) *
             input_channels) +
            input;
          const std::size_t hwcf_index =
            (((y * kernel_width + x) * input_channels + input) *
             output_channels) +
            output;
          std::copy_n(logical.begin() + logical_index * sizeof(float),
                      sizeof(float),
                      ohwi.begin() + ohwi_index * sizeof(float));
          std::copy_n(logical.begin() + logical_index * sizeof(float),
                      sizeof(float),
                      hwcf.begin() + hwcf_index * sizeof(float));
        }
      }
    }
  }
  resource.physical_layout = "OHWI";
  auto ohwi_hash = ncnn_mlir::vulkan::vulkanWeightSHA256(resource, ohwi);
  ASSERT_TRUE(ohwi_hash) << ohwi_hash.error();
  resource.physical_layout = "HWCF";
  auto hwcf_hash = ncnn_mlir::vulkan::vulkanWeightSHA256(resource, hwcf);
  ASSERT_TRUE(hwcf_hash) << hwcf_hash.error();
  EXPECT_EQ(*ohwi_hash, *hwcf_hash);
  resource.physical_layout = "unsupported";
  EXPECT_FALSE(ncnn_mlir::vulkan::vulkanWeightSHA256(resource, hwcf));
  resource.physical_layout = "OHWI";
  resource.shape[0] = 0;
  EXPECT_FALSE(ncnn_mlir::vulkan::vulkanWeightSHA256(resource, ohwi));
}

}  // namespace
}  // namespace ncnn_compile
