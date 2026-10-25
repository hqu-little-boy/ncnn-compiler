#include "VulkanProduct.hpp"
#include "vulkan_shader_fixture.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace ncnn_compile {
namespace {

namespace vk = ncnn_mlir::vulkan;

vk::TensorResource resource(std::uint32_t id, vk::ResourceKind kind) {
  const bool public_io =
    kind == vk::ResourceKind::Input || kind == vk::ResourceKind::Output;
  return {.id = id,
          .kind = kind,
          .name = "resource_" + std::to_string(id),
          .shape = {17},
          .logical_layout = "linear",
          .physical_layout = "linear",
          .host_layout = public_io ? "linear" : "",
          .layout_transform = std::nullopt,
          .weight_sha256 = "",
          .byte_size = 68,
          .first_kernel = 0,
          .last_kernel = 0};
}

vk::VulkanExecutionPlan makePlan() {
  return {.schema_version = vk::kExecutionPlanSchemaVersion,
          .model = "lifecycle_unit",
          .build_identity = "lifecycle-unit-source",
          .host_triple = "x86_64-unknown-linux-gnu",
          .device_target = "vulkan1.1-spirv1.0",
          .resources = {resource(7, vk::ResourceKind::Input),
                        resource(41, vk::ResourceKind::Output)},
          .kernels = {{.id = 97,
                       .operation = "relu",
                       .entry_point = "relu",
                       .local_size = {64, 1, 1},
                       .reads = {7},
                       .writes = {41},
                       .bindings = {{.set = 2,
                                     .binding = 5,
                                     .resource_id = 7,
                                     .access = vk::ResourceAccess::Read},
                                    {.set = 2,
                                     .binding = 9,
                                     .resource_id = 41,
                                     .access = vk::ResourceAccess::Write}},
                       .push_constants = {}}},
          .inputs = {7},
          .outputs = {41},
          .plan_hash = ""};
}

std::expected<VulkanModelPackage, std::string> packageFromPlan(
  const vk::VulkanExecutionPlan& plan,
  std::span<const vk::VulkanModelWeight> weights = {},
  std::string_view binary_symbol = "fixture_shader") {
  std::vector<vk::VulkanModelShader> shaders;
  for (const auto& kernel : plan.kernels) {
    auto shader = vk::test::makeTestVulkanShader(
      plan, kernel, std::string(binary_symbol) + std::to_string(kernel.id));
    if (!shader) {
      return std::unexpected(shader.error());
    }
    shaders.push_back(std::move(*shader));
  }
  auto serialized = vk::serializeExecutionPlan(plan);
  if (!serialized) {
    return std::unexpected(serialized.error());
  }
  return makeVulkanModelPackage(plan.model,
                                plan.host_triple,
                                *serialized,
                                shaders,
                                weights,
                                "lifecycle-unit-param",
                                "lifecycle-unit-bin");
}

std::expected<VulkanModelPackage, std::string> weightedPackage(
  std::uint32_t count = 1) {
  auto plan = makePlan();
  plan.kernels.front().operation = "copy";
  std::vector<vk::VulkanModelWeight> weights;
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint32_t id = 100003 + index;
    auto weight_resource = resource(id, vk::ResourceKind::Weight);
    std::vector<std::uint8_t> bytes(68, 0);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 4) {
      bytes[offset] = static_cast<std::uint8_t>((offset + index) & 0xffU);
      bytes[offset + 2] = 0x80;
      bytes[offset + 3] = 0x3f;
    }
    auto digest = vk::vulkanWeightSHA256(weight_resource, bytes);
    if (!digest) {
      return std::unexpected(digest.error());
    }
    weight_resource.weight_sha256 = *digest;
    plan.resources.push_back(std::move(weight_resource));
    weights.push_back({.resource_id = id, .bytes = std::move(bytes)});
    plan.kernels.front().reads.push_back(id);
    plan.kernels.front().bindings.push_back(
      {.set = 2,
       .binding = 12 + index,
       .resource_id = id,
       .access = vk::ResourceAccess::Read});
  }
  return packageFromPlan(plan, weights);
}

TEST(VulkanLifecycleCodegenTest, ProducesSameHeaderAndDeterministicSource) {
  auto package = packageFromPlan(makePlan());
  ASSERT_TRUE(package) << package.error();
  auto first = makeVulkanLifecycleSource(*package);
  auto second = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(first) << first.error();
  ASSERT_TRUE(second) << second.error();
  EXPECT_EQ(first->header, package->header);
  EXPECT_EQ(first->header, second->header);
  EXPECT_EQ(first->source, second->source);
  EXPECT_EQ(first->source.find('@'), std::string::npos);
  EXPECT_EQ(first->host_program_identity, package->host_program.identity);
  EXPECT_EQ(first->host_program_identity, second->host_program_identity);
}

TEST(VulkanLifecycleCodegenTest, CandidateDoesNotChangeLegacyProductSource) {
  auto package = packageFromPlan(makePlan());
  ASSERT_TRUE(package) << package.error();
  const auto legacy_source = package->wrapper_source;
  const auto legacy_manifest = package->manifest_json;
  auto candidate = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(candidate) << candidate.error();
  EXPECT_EQ(package->wrapper_source, legacy_source);
  EXPECT_EQ(package->manifest_json, legacy_manifest);
  EXPECT_NE(legacy_source.find("VulkanRuntimeContext::create"),
            std::string::npos);
  EXPECT_NE(legacy_manifest.find("model-fixed-run"), std::string::npos);
  EXPECT_EQ(legacy_manifest.find("generic-plan-runtime"), std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, UsesOnlyDeviceHelperInDeploymentSource) {
  auto package = weightedPackage();
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  for (const std::string_view forbidden : {"VulkanRuntime",
                                           "VulkanExecutionPlan",
                                           "VulkanHostProgram",
                                           "VulkanModelPayload",
                                           "deserializeExecutionPlan",
                                           "kPlanJsonBytes",
                                           "llvm/",
                                           "mlir/Support",
                                           "runtime.run(",
                                           "vkCmdDispatch"}) {
    EXPECT_EQ(generated->source.find(forbidden), std::string::npos)
      << forbidden;
  }
  EXPECT_NE(generated->source.find("VulkanDeviceHelper.hpp"),
            std::string::npos);
  EXPECT_NE(generated->source.find("does not implement model run"),
            std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, SparseIdsBecomeDenseOwnedSlots) {
  auto package = weightedPackage();
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_NE(generated->source.find("pipelines_[0].emplace"), std::string::npos);
  EXPECT_EQ(generated->source.find("pipelines_[97]"), std::string::npos);
  EXPECT_EQ(generated->source.find("buffers_[100003]"), std::string::npos);
  EXPECT_NE(generated->source.find(".set = 2, .binding = 5"),
            std::string::npos);
  EXPECT_NE(generated->source.find(".set = 2, .binding = 9"),
            std::string::npos);
  EXPECT_NE(generated->source.find(".descriptor_set_count = 3"),
            std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, EmptyWeightsAndPushRangesUseStandardArrays) {
  auto package = packageFromPlan(makePlan());
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_NE(generated->source.find("std::array<VulkanPushConstantRange, 0>"),
            std::string::npos);
  EXPECT_EQ(generated->source.find("kWeightBytes"), std::string::npos);
  EXPECT_EQ(generated->source.find("vkCmdCopyBuffer"), std::string::npos);
  EXPECT_EQ(generated->source.find("submit_and_wait"), std::string::npos);
  EXPECT_EQ(generated->source.find("[] = {}"), std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, BudgetPrecedesAllBufferAllocations) {
  auto package = weightedPackage();
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  const auto budget = generated->source.find("checkAllocationBudget();");
  const auto allocation = generated->source.find("->allocate_and_bind(");
  ASSERT_NE(budget, std::string::npos);
  ASSERT_NE(allocation, std::string::npos);
  EXPECT_LT(budget, allocation);
  EXPECT_NE(generated->source.find("->get_allocation_size()"),
            std::string::npos);
  EXPECT_NE(generated->source.find("validateAllocationBudget("),
            std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, MultipleWeightsHaveFixedCopiesAndRelease) {
  auto package = weightedPackage(2);
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_NE(generated->source.find("kWeightBytes0"), std::string::npos);
  EXPECT_NE(generated->source.find("kWeightBytes1"), std::string::npos);
  EXPECT_NE(generated->source.find("weight_upload_count_ += UINT64_C(2)"),
            std::string::npos);
  const auto waited = generated->source.find("commands_->submit_and_wait(");
  ASSERT_NE(waited, std::string::npos);
  for (const auto& buffer : package->host_program.buffers) {
    if (buffer.kind != vk::HostBufferKind::WeightStaging) {
      continue;
    }
    const auto release = generated->source.find(
      "require(buffers_[" + std::to_string(buffer.id.value) + "]->destroy());");
    ASSERT_NE(release, std::string::npos);
    EXPECT_LT(waited, release);
  }
}

TEST(VulkanLifecycleCodegenTest, MultiplePublicPortsRemainDistinct) {
  auto plan = makePlan();
  plan.resources.push_back(resource(300, vk::ResourceKind::Input));
  plan.inputs.push_back(300);
  plan.resources.push_back(resource(92, vk::ResourceKind::Output));
  plan.outputs.push_back(92);
  plan.kernels.front().writes.push_back(92);
  plan.kernels.front().bindings.push_back(
    {.set = 2,
     .binding = 13,
     .resource_id = 92,
     .access = vk::ResourceAccess::Write});
  plan.kernels.front().reads.push_back(300);
  plan.kernels.front().bindings.push_back({.set = 2,
                                           .binding = 11,
                                           .resource_id = 300,
                                           .access = vk::ResourceAccess::Read});
  auto package = packageFromPlan(plan);
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_NE(generated->header.find("INPUT_COUNT UINT32_C(2)"),
            std::string::npos);
  EXPECT_NE(generated->header.find("OUTPUT_COUNT UINT32_C(2)"),
            std::string::npos);
  EXPECT_NE(generated->source.find("input2_elements != kInputs[1].elements"),
            std::string::npos);
  EXPECT_NE(generated->source.find("output2_elements != kOutputs[1].elements"),
            std::string::npos);
  ASSERT_EQ(package->host_program.outputs.size(), 2);
  EXPECT_NE(package->host_program.outputs[0].device_buffer,
            package->host_program.outputs[1].device_buffer);
  EXPECT_NE(package->host_program.outputs[0].staging_buffer,
            package->host_program.outputs[1].staging_buffer);
}

TEST(VulkanLifecycleCodegenTest,
     EntryPointBytesAreNotCppSourceOrCStringLengths) {
  auto plan = makePlan();
  const std::string entry =
    "entry_@\"\\\n"
    "\xc3\xa9"
    "af";
  plan.kernels.front().entry_point = entry;
  auto package = packageFromPlan(plan, {}, "binary_\"\\\n");
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_EQ(generated->source.find(entry), std::string::npos);
  EXPECT_EQ(generated->source.find("binary_\"\\\n"), std::string::npos);
  EXPECT_NE(generated->source.find("static_cast<char>(0xc3)"),
            std::string::npos);
  EXPECT_NE(generated->source.find("static_cast<char>(0xa9)"),
            std::string::npos);
  EXPECT_NE(generated->source.find("std::string_view(kEntryPoint0.data(), " +
                                   std::to_string(entry.size()) + ")"),
            std::string::npos);
  EXPECT_EQ(generated->source.find("strlen("), std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, UsesReversePipelineAndResourceOwnership) {
  auto plan = makePlan();
  auto temporary = resource(39, vk::ResourceKind::Intermediate);
  temporary.last_kernel = 1;
  plan.resources.push_back(std::move(temporary));
  plan.resources[1].first_kernel = 1;
  plan.resources[1].last_kernel = 1;
  plan.kernels.front().writes = {39};
  plan.kernels.front().bindings[1].resource_id = 39;
  plan.kernels.push_back({.id = 198,
                          .operation = "copy",
                          .entry_point = "copy",
                          .local_size = {64, 1, 1},
                          .reads = {39},
                          .writes = {41},
                          .bindings = {{0, 0, 39, vk::ResourceAccess::Read},
                                       {0, 1, 41, vk::ResourceAccess::Write}},
                          .push_constants = {}});
  auto package = packageFromPlan(plan);
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  const auto second = generated->source.find("release(pipelines_[1]);");
  const auto first = generated->source.find("release(pipelines_[0]);");
  const auto pool = generated->source.find("release(descriptor_pool_);");
  const auto buffer = generated->source.find("release(buffers_[");
  ASSERT_NE(second, std::string::npos);
  ASSERT_NE(first, std::string::npos);
  ASSERT_NE(pool, std::string::npos);
  ASSERT_NE(buffer, std::string::npos);
  EXPECT_LT(second, first);
  EXPECT_LT(first, pool);
  EXPECT_LT(pool, buffer);
}

TEST(VulkanLifecycleCodegenTest, RejectsUnsafeOrReservedModelIdentifiers) {
  for (const std::string_view model : {"",
                                       "_model",
                                       "__model",
                                       "a__b",
                                       "model_",
                                       "a-b",
                                       "a\nb",
                                       "x\";unexpected();"}) {
    auto package = packageFromPlan(makePlan());
    ASSERT_TRUE(package) << package.error();
    package->plan.model = model;
    auto generated = makeVulkanLifecycleSource(*package);
    ASSERT_FALSE(generated) << model;
    EXPECT_NE(generated.error().find("lifecycle model name"),
              std::string::npos);
  }
}

TEST(VulkanLifecycleCodegenTest,
     ReservedPrefixesFailBeforeHeaderOrSourceChecks) {
  for (const std::string_view model : {"_model", "a__b", "model_"}) {
    auto plan = makePlan();
    plan.model = model;
    auto package = packageFromPlan(plan);
    ASSERT_TRUE(package) << package.error();
    auto generated = makeVulkanLifecycleSource(*package);
    ASSERT_FALSE(generated) << model;
    EXPECT_NE(generated.error().find("lifecycle model name"),
              std::string::npos);
  }
}

TEST(VulkanLifecycleCodegenTest,
     KeywordPrefixProducesLegalSuffixedIdentifiers) {
  auto plan = makePlan();
  plan.model = "class";
  auto package = packageFromPlan(plan);
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_NE(generated->source.find("struct class_vk_context final"),
            std::string::npos);
  EXPECT_NE(generated->header.find("CLASS_VULKAN_H"), std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, HeaderGuardUsesAsciiCaseMapping) {
  auto plan = makePlan();
  plan.model = "abcdefghijklmnopqrstuvwxyz_0123456789";
  auto package = packageFromPlan(plan);
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_NE(
    generated->header.find("ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789_VULKAN_H"),
    std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, RejectsMutatedHeaderAndHostIdentity) {
  auto package = packageFromPlan(makePlan());
  ASSERT_TRUE(package) << package.error();
  auto bad_header = *package;
  bad_header.header += "\n#error injected header\n";
  EXPECT_FALSE(makeVulkanLifecycleSource(bad_header));
  package->host_program.identity[0] ^= 1;
  EXPECT_FALSE(makeVulkanLifecycleSource(*package));
}

TEST(VulkanLifecycleCodegenTest, RejectsIllegalReferencesAndLifetimeChanges) {
  auto package = weightedPackage();
  ASSERT_TRUE(package) << package.error();
  auto bad_reference = *package;
  bad_reference.host_program.pipelines.front().bindings.front().buffer.value =
    std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(makeVulkanLifecycleSource(bad_reference));
  auto bad_initialize = *package;
  bad_initialize.host_program.initialize.erase(
    bad_initialize.host_program.initialize.begin());
  EXPECT_FALSE(makeVulkanLifecycleSource(bad_initialize));
  package->host_program.destroy.clear();
  EXPECT_FALSE(makeVulkanLifecycleSource(*package));
}

TEST(VulkanLifecycleCodegenTest, RejectsShaderAndWeightPayloadMutation) {
  auto package = weightedPackage();
  ASSERT_TRUE(package) << package.error();
  auto bad_shader = *package;
  bad_shader.shaders.front().words.pop_back();
  EXPECT_FALSE(makeVulkanLifecycleSource(bad_shader));
  package->weights.front().bytes.front() ^= 0xffU;
  EXPECT_FALSE(makeVulkanLifecycleSource(*package));
}

TEST(VulkanLifecycleCodegenTest, RejectsByteAndPushRangeOverflow) {
  auto package = packageFromPlan(makePlan());
  ASSERT_TRUE(package) << package.error();
  auto bad_bytes = *package;
  bad_bytes.host_program.buffers.front().byte_size =
    std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(makeVulkanLifecycleSource(bad_bytes));
  package->plan.kernels.front().push_constants.push_back(
    {.offset = std::numeric_limits<std::uint32_t>::max() - 3,
     .bytes = std::vector<std::uint8_t>(8, 0)});
  EXPECT_FALSE(makeVulkanLifecycleSource(*package));
}

TEST(VulkanLifecycleCodegenTest, RejectsRuntimeCommandsInCreateOperations) {
  auto package = weightedPackage();
  ASSERT_TRUE(package) << package.error();
  package->host_program.create_operations.emplace_back(
    vk::HostDispatchOp{.pipeline = {97}, .groups = {1, 1, 1}});
  EXPECT_FALSE(makeVulkanLifecycleSource(*package));
}

TEST(VulkanLifecycleCodegenTest, PreservesOriginalSingleShaderPackagingHeader) {
  auto plan = makePlan();
  auto shader = vk::test::makeTestVulkanShader(plan, plan.kernels.front());
  ASSERT_TRUE(shader) << shader.error();
  auto serialized = vk::serializeExecutionPlan(plan);
  ASSERT_TRUE(serialized) << serialized.error();
  std::string bytes;
  for (std::uint32_t word : shader->words) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<char>((word >> shift) & 0xffU));
    }
  }
  auto package = makeVulkanModelPackage(plan.model,
                                        plan.host_triple,
                                        *serialized,
                                        bytes,
                                        "lifecycle-unit-param",
                                        "lifecycle-unit-bin");
  ASSERT_TRUE(package) << package.error();
  ASSERT_TRUE(package->shaders.empty());
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  EXPECT_EQ(generated->header, package->header);
  EXPECT_NE(generated->source.find("kShaderWords0"), std::string::npos);
  EXPECT_EQ(generated->source.find("kPlanJsonBytes"), std::string::npos);
}

TEST(VulkanLifecycleCodegenTest, EmbeddedWeightsMatchVerifiedPayloadBytes) {
  auto package = weightedPackage(2);
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanLifecycleSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  constexpr std::string_view byte_token = "std::byte{0x";
  for (std::size_t index = 0; index < package->weights.size(); ++index) {
    const auto& expected = package->weights[index].bytes;
    const auto declaration = "std::array<std::byte, " +
                             std::to_string(expected.size()) +
                             "> kWeightBytes" + std::to_string(index) + "{{";
    auto cursor = generated->source.find(declaration);
    ASSERT_NE(cursor, std::string::npos);
    const auto end = generated->source.find("}};", cursor);
    ASSERT_NE(end, std::string::npos);
    std::vector<std::uint8_t> actual;
    while ((cursor = generated->source.find(byte_token, cursor)) !=
             std::string::npos &&
           cursor < end) {
      const auto digits = cursor + byte_token.size();
      ASSERT_LE(digits, end);
      ASSERT_GE(end - digits, 2U);
      std::uint32_t value = 0;
      const auto* begin = generated->source.data() + digits;
      const auto [next, error] = std::from_chars(begin, begin + 2, value, 16);
      ASSERT_EQ(error, std::errc{});
      ASSERT_EQ(next, begin + 2);
      ASSERT_TRUE(std::in_range<std::uint8_t>(value));
      actual.push_back(static_cast<std::uint8_t>(value));
      cursor = digits + 2;
    }
    EXPECT_EQ(actual, expected) << "weight " << index;
  }
}

TEST(VulkanLifecycleCodegenTest, RepeatedPublicResourceIdsAreRejected) {
  auto package = packageFromPlan(makePlan());
  ASSERT_TRUE(package) << package.error();
  // Repeated SSA outputs must first be materialized as distinct plan resources;
  // a duplicated public resource ID is not a proven alias.
  package->plan.outputs.push_back(41);
  EXPECT_FALSE(makeVulkanLifecycleSource(*package));
}

}  // namespace
}  // namespace ncnn_compile
