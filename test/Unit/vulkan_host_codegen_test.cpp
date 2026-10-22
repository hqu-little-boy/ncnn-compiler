#include "VulkanProduct.hpp"
#include "vulkan_shader_fixture.hpp"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace ncnn_compile {
namespace {

namespace vk = ncnn_mlir::vulkan;

vk::TensorResource makeResource(std::uint32_t id, vk::ResourceKind kind) {
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

vk::VulkanExecutionPlan makeReluPlan() {
  return {.schema_version = vk::kExecutionPlanSchemaVersion,
          .model = "fixedrun_unit",
          .build_identity = "fixedrun-unit-source",
          .host_triple = "x86_64-unknown-linux-gnu",
          .device_target = "vulkan1.1-spirv1.0",
          .resources = {makeResource(7, vk::ResourceKind::Input),
                        makeResource(41, vk::ResourceKind::Output)},
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

vk::VulkanExecutionPlan makeMultiOutputPlan() {
  auto plan = makeReluPlan();
  plan.model = "fixedrun_multi_output";
  plan.resources[1] = {.id = 19,
                       .kind = vk::ResourceKind::Intermediate,
                       .name = "intermediate",
                       .shape = {17},
                       .logical_layout = "linear",
                       .physical_layout = "linear",
                       .host_layout = "",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 68,
                       .first_kernel = 0,
                       .last_kernel = 1};
  plan.resources.push_back({.id = 41,
                            .kind = vk::ResourceKind::Output,
                            .name = "output_a",
                            .shape = {17},
                            .logical_layout = "linear",
                            .physical_layout = "linear",
                            .host_layout = "linear",
                            .layout_transform = std::nullopt,
                            .weight_sha256 = "",
                            .byte_size = 68,
                            .first_kernel = 1,
                            .last_kernel = 1});
  plan.resources.push_back({.id = 59,
                            .kind = vk::ResourceKind::Output,
                            .name = "output_b",
                            .shape = {17},
                            .logical_layout = "linear",
                            .physical_layout = "linear",
                            .host_layout = "linear",
                            .layout_transform = std::nullopt,
                            .weight_sha256 = "",
                            .byte_size = 68,
                            .first_kernel = 1,
                            .last_kernel = 1});
  plan.outputs.push_back(59);
  plan.kernels = {{.id = 101,
                   .operation = "fixedrun_stage0",
                   .entry_point = "compute_main",
                   .local_size = {64, 1, 1},
                   .reads = {7},
                   .writes = {19},
                   .bindings = {{.set = 1,
                                 .binding = 2,
                                 .resource_id = 7,
                                 .access = vk::ResourceAccess::Read},
                                {.set = 1,
                                 .binding = 11,
                                 .resource_id = 19,
                                 .access = vk::ResourceAccess::Write}},
                   .push_constants = {{.offset = 0, .bytes = {17, 0, 0, 0}}}},
                  {.id = 909,
                   .operation = "fixedrun_stage1",
                   .entry_point = "compute_main",
                   .local_size = {64, 1, 1},
                   .reads = {19},
                   .writes = {41, 59},
                   .bindings = {{.set = 2,
                                 .binding = 0,
                                 .resource_id = 19,
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
  plan.build_identity = "fixedrun-multi-output-source";
  return plan;
}

std::expected<VulkanModelPackage, std::string> makePackage(
  const vk::VulkanExecutionPlan& plan) {
  std::vector<vk::VulkanModelShader> shaders;
  for (const auto& kernel : plan.kernels) {
    auto shader = vk::test::makeTestVulkanShader(
      plan, kernel, "fixedrun_shader_" + std::to_string(kernel.id));
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
                                {},
                                "fixedrun-unit-param",
                                "fixedrun-unit-bin");
}

std::size_t positionOf(std::string_view source, std::string_view token) {
  const auto position = source.find(token);
  EXPECT_NE(position, std::string_view::npos) << token;
  return position;
}

std::size_t countOf(std::string_view source, std::string_view token) {
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = source.find(token, position)) != std::string_view::npos) {
    ++count;
    position += token.size();
  }
  return count;
}

TEST(VulkanFixedRunCodegenTest, EmitsFixedOperationsInVerifiedOrder) {
  auto package = makePackage(makeReluPlan());
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanHostSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  auto repeated = makeVulkanHostSource(*package);
  ASSERT_TRUE(repeated) << repeated.error();

  EXPECT_EQ(generated->source, repeated->source);
  EXPECT_EQ(generated->codegen_identity, repeated->codegen_identity);
  EXPECT_EQ(generated->codegen_version, kVulkanHostSourceCodegenVersion);
  EXPECT_NE(generated->codegen_identity, generated->host_program_identity);
  EXPECT_NE(generated->source.find("Fixed-run codegen:"), std::string::npos);
  EXPECT_EQ(generated->header, package->header);
  EXPECT_EQ(generated->host_program_identity, package->host_program.identity);
  EXPECT_FALSE(generated->codegen_identity.empty());
  EXPECT_FALSE(generated->codegen_version.empty());
  EXPECT_NE(generated->source.find("vkCmdCopyBuffer"), std::string::npos);
  EXPECT_NE(generated->source.find("vkCmdPipelineBarrier"), std::string::npos);
  EXPECT_NE(generated->source.find("vkCmdBindPipeline"), std::string::npos);
  EXPECT_NE(generated->source.find("vkCmdBindDescriptorSets"),
            std::string::npos);
  EXPECT_NE(generated->source.find("vkCmdDispatch"), std::string::npos);
  EXPECT_NE(generated->source.find("->flush("), std::string::npos);
  EXPECT_NE(generated->source.find("->invalidate("), std::string::npos);

  const auto input_copy = positionOf(generated->source, "vkCmdCopyBuffer");
  const auto dispatch = positionOf(generated->source, "vkCmdDispatch");
  const auto output_invalidate = positionOf(generated->source, "->invalidate(");
  const auto publish = positionOf(generated->source, "std::memcpy(outputs[0]");
  EXPECT_LT(input_copy, dispatch);
  EXPECT_LT(dispatch, output_invalidate);
  EXPECT_LT(output_invalidate, publish);

  for (const std::string_view forbidden : {"VulkanRuntimeContext",
                                           "ExecutionPlan",
                                           "HostProgram",
                                           "deserializeExecutionPlan",
                                           "kernel_schedule",
                                           "reads.begin()",
                                           "writes.begin()",
                                           "does not implement model run"}) {
    EXPECT_EQ(generated->source.find(forbidden), std::string::npos)
      << forbidden;
  }
}

TEST(VulkanFixedRunCodegenTest, UsesDenseSlotsForSparseModelIds) {
  auto package = makePackage(makeReluPlan());
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanHostSource(*package);
  ASSERT_TRUE(generated) << generated.error();

  EXPECT_NE(generated->source.find("buffers_[0]"), std::string::npos);
  EXPECT_NE(generated->source.find("pipelines_[0]"), std::string::npos);
  EXPECT_EQ(generated->source.find("buffers_[7]"), std::string::npos);
  EXPECT_EQ(generated->source.find("pipelines_[97]"), std::string::npos);
}

TEST(VulkanFixedRunCodegenTest,
     EmitsMultipleDispatchesAndPreparesEveryOutputBeforePublishing) {
  auto package = makePackage(makeMultiOutputPlan());
  ASSERT_TRUE(package) << package.error();
  auto generated = makeVulkanHostSource(*package);
  ASSERT_TRUE(generated) << generated.error();
  const auto& source = generated->source;

  const auto first_dispatch = positionOf(source, "vkCmdDispatch");
  const auto second_dispatch = source.find("vkCmdDispatch", first_dispatch + 1);
  ASSERT_NE(second_dispatch, std::string::npos);
  EXPECT_EQ(source.find("vkCmdDispatch", second_dispatch + 1),
            std::string::npos);
  EXPECT_NE(source.find("UINT8_C(0x11)"), std::string::npos);
  EXPECT_NE(source.find("VK_ACCESS_SHADER_WRITE_BIT"), std::string::npos);
  EXPECT_NE(
    source.find("VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT"),
    std::string::npos);
  EXPECT_NE(source.find("VK_PIPELINE_STAGE_HOST_BIT"), std::string::npos);
  EXPECT_NE(source.find("VK_PIPELINE_STAGE_TRANSFER_BIT"), std::string::npos);
  EXPECT_NE(source.find("VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"),
            std::string::npos);
  EXPECT_NE(source.find(".offset = UINT64_C(0), .size = UINT64_C(68)"),
            std::string::npos);
  EXPECT_EQ(countOf(source, "vkCmdCopyBuffer("), 3);
  EXPECT_EQ(countOf(source, "vkCmdPipelineBarrier("), 5);

  const auto readbacks =
    positionOf(source, "std::array<std::span<std::byte>, 2> readbacks");
  const auto validation = positionOf(source, "checkValidation();");
  const auto first_publish = positionOf(source, "std::memcpy(outputs[0]");
  const auto second_publish = positionOf(source, "std::memcpy(outputs[1]");
  EXPECT_LT(second_dispatch, readbacks);
  EXPECT_LT(readbacks, validation);
  EXPECT_LT(validation, first_publish);
  EXPECT_LT(validation, second_publish);
  EXPECT_LT(first_publish, second_publish);

  const auto run_entry = positionOf(source, "ncnn_vk_status run(");
  const auto run_body_open = source.find('{', run_entry);
  const auto run_lock =
    source.find("const std::scoped_lock lock(run_mutex_);", run_entry);
  const auto run_try = source.find("try {", run_lock);
  const auto first_operation = source.find("std::memcpy(mappedBuffer", run_try);
  const auto run_catch =
    source.find("} catch (const LifecycleFailure& failure)", second_publish);
  const auto run_end = source.find("ncnn_vk_status destroy()", run_entry);
  ASSERT_NE(run_body_open, std::string::npos);
  ASSERT_NE(run_lock, std::string::npos);
  ASSERT_NE(run_try, std::string::npos);
  ASSERT_NE(first_operation, std::string::npos);
  ASSERT_NE(run_catch, std::string::npos);
  ASSERT_NE(run_end, std::string::npos);
  EXPECT_LT(run_entry, run_body_open);
  EXPECT_LT(run_body_open, run_lock);
  EXPECT_LT(run_lock, run_try);
  EXPECT_LT(run_try, first_operation);
  EXPECT_LT(first_operation, first_dispatch);
  EXPECT_LT(second_publish, run_catch);
  EXPECT_LT(run_catch, run_end);

  // The scoped lock must be in the outer run() body, not a nested block, and
  // that body must stay open through the last caller-output publication.
  std::size_t brace_depth = 1;
  for (std::size_t position = run_body_open + 1; position < run_lock;
       ++position) {
    if (source[position] == '{') {
      ++brace_depth;
    } else if (source[position] == '}') {
      ASSERT_GT(brace_depth, 1U);
      --brace_depth;
    }
  }
  EXPECT_EQ(brace_depth, 1U);
  for (std::size_t position = run_lock; position < second_publish; ++position) {
    if (source[position] == '{') {
      ++brace_depth;
    } else if (source[position] == '}') {
      ASSERT_GT(brace_depth, 1U);
      --brace_depth;
    }
  }
  EXPECT_GE(brace_depth, 2U);
}

TEST(VulkanFixedRunCodegenTest,
     RejectsMutatedProgramAndPreservesLegacyProduct) {
  auto package = makePackage(makeReluPlan());
  ASSERT_TRUE(package) << package.error();
  const auto legacy_wrapper = package->wrapper_source;
  const auto legacy_manifest = package->manifest_json;
  auto generated = makeVulkanHostSource(*package);
  ASSERT_TRUE(generated) << generated.error();

  auto mutated = *package;
  std::ranges::reverse(mutated.host_program.run_operations);
  EXPECT_FALSE(makeVulkanHostSource(mutated));

  mutated = *package;
  mutated.host_program.identity.front() ^= 1;
  EXPECT_FALSE(makeVulkanHostSource(mutated));

  EXPECT_EQ(package->wrapper_source, legacy_wrapper);
  EXPECT_EQ(package->manifest_json, legacy_manifest);
  EXPECT_NE(legacy_wrapper.find("VulkanRuntimeContext::create"),
            std::string::npos);
  EXPECT_NE(legacy_manifest.find("generic-plan-runtime"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_compile
