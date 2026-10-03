#include "VulkanProduct.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace ncnn_compile {
namespace {

using ncnn_mlir::vulkan::DescriptorBinding;
using ncnn_mlir::vulkan::KernelExecution;
using ncnn_mlir::vulkan::ResourceAccess;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;

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

std::string makeSpirvBytes() {
  const std::string bytes(
    "\x03\x02\x23\x07\x00\x00\x01\x00"
    "\x00\x00\x00\x00\x01\x00\x00\x00"
    "\x00\x00\x00\x00",
    20);
  return bytes;
}

TEST(VulkanProductTest, PackagesReproduciblyAndEmbedsBothIdentities) {
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(makeReluPlan());
  ASSERT_TRUE(serialized) << serialized.error();
  const std::string shader = makeSpirvBytes();
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
                                         makeSpirvBytes(),
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

}  // namespace
}  // namespace ncnn_compile
