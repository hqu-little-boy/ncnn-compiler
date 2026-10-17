#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"
#include "ncnn-mlir/Vulkan/VulkanDeviceTypes.hpp"
#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"
#include "test/Vulkan/native/fakeVulkan/fake_vulkan.hpp"
#include <vulkan/vulkan.h>

namespace ncnn_mlir::vulkan {
namespace {

TensorResource makeResource(std::uint32_t id,
                            ResourceKind kind,
                            std::string_view name) {
  std::optional<LayoutTransform> transform;
  if (kind == ResourceKind::Input) {
    transform = LayoutTransform{
      .from_layout = "CHW", .to_layout = "linear", .operation = "linearize"};
  } else if (kind == ResourceKind::Output) {
    transform = LayoutTransform{
      .from_layout = "linear", .to_layout = "CHW", .operation = "de-linearize"};
  }
  const bool is_public =
    kind == ResourceKind::Input || kind == ResourceKind::Output;
  return TensorResource{.id = id,
                        .kind = kind,
                        .name = std::string(name),
                        .shape = {1, 1, 4},
                        .logical_layout = "CHW",
                        .physical_layout = "linear",
                        .host_layout = is_public ? "CHW" : "",
                        .layout_transform = std::move(transform),
                        .weight_sha256 = "",
                        .byte_size = 16,
                        .first_kernel = 0,
                        .last_kernel = 0};
}

VulkanExecutionPlan makePlan() {
  return VulkanExecutionPlan{
    .schema_version = kExecutionPlanSchemaVersion,
    .model = "runtime_budget_fixture",
    .build_identity = "runtime-budget-fixture",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources = {makeResource(1, ResourceKind::Input, "input"),
                  makeResource(2, ResourceKind::Output, "output")},
    .kernels = {KernelExecution{.id = 10,
                                .operation = "identity",
                                .entry_point = "identity",
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
                                .push_constants = {}}},
    .inputs = {1},
    .outputs = {2},
    .plan_hash = ""};
}

constexpr std::array<std::uint32_t, 5> kPlaceholderSpirv{
  0x07230203U, 0x00010000U, 0U, 1U, 0U};

// This target intentionally links the old executor and its plan support. The
// helper-only native target proves independence; this one catches moving actual
// runtime allocations ahead of the complete requirements/staging preflight.
class VulkanRuntimeBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fake_vulkan::reset();
    fake_vulkan::enable();
  }

  void TearDown() override {
    constexpr std::array<std::string_view, 6> kOwnedObjects{
      "instance", "device", "buffer", "memory", "command_pool", "fence"};
    for (std::string_view kind : kOwnedObjects) {
      EXPECT_EQ(fake_vulkan::live_count(kind), 0U) << kind;
    }
  }
};

TEST_F(VulkanRuntimeBudgetTest, DefaultBudgetAndStorageContractAreShared) {
  static_assert(kVulkanDeviceHelperMaxBufferBytes == 128ULL * 1024 * 1024);
  static_assert(kVulkanDeviceHelperDefaultBudgetBytes == 512ULL * 1024 * 1024);
  EXPECT_EQ(makeVulkanRuntimeOptions().max_context_memory_bytes,
            kVulkanDeviceHelperDefaultBudgetBytes);
}

TEST_F(VulkanRuntimeBudgetTest,
       ChecksActualIoStagingBudgetBeforeAnyAllocation) {
  const VulkanExecutionPlan plan = makePlan();
  ASSERT_TRUE(verifyExecutionPlan(plan));
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = kPlaceholderSpirv};
  VulkanRuntimeOptions options = makeVulkanRuntimeOptions();
  // Four logical 16-byte buffers require 128 bytes each in this fake driver.
  // The device-only subtotal fits, but persistent I/O staging takes it over.
  options.max_context_memory_bytes = 400;

  auto context = VulkanRuntimeContext::create(plan, {&kernel, 1}, {}, options);

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_OUT_OF_MEMORY);
  EXPECT_NE(context.error().diagnostic.find("budget"), std::string::npos);
  EXPECT_EQ(fake_vulkan::call_count("vkGetBufferMemoryRequirements"), 4U);
  EXPECT_EQ(fake_vulkan::call_count("vkAllocateMemory"), 0U);
}

TEST_F(VulkanRuntimeBudgetTest, CountsTemporaryWeightStagingBeforeAllocation) {
  VulkanExecutionPlan plan = makePlan();
  TensorResource weight = makeResource(3, ResourceKind::Weight, "weight");
  weight.logical_layout = "linear";
  constexpr std::array<std::uint8_t, 16> kWeightBytes{};
  const auto digest = vulkanWeightSHA256(weight, kWeightBytes);
  ASSERT_TRUE(digest) << (digest ? "" : digest.error());
  weight.weight_sha256 = *digest;
  plan.resources.push_back(std::move(weight));
  plan.kernels.front().reads.push_back(3);
  plan.kernels.front().bindings.push_back(DescriptorBinding{
    .set = 0, .binding = 2, .resource_id = 3, .access = ResourceAccess::Read});
  ASSERT_TRUE(verifyExecutionPlan(plan));
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = kPlaceholderSpirv};
  const VulkanResourcePayload payload{
    .resource_id = 3, .bytes = std::as_bytes(std::span(kWeightBytes))};
  VulkanRuntimeOptions options = makeVulkanRuntimeOptions();
  // Persistent requirements total 640 bytes; temporary weight staging makes
  // the create peak 768, despite just 96 logical bytes across all six buffers.
  options.max_context_memory_bytes = 700;

  auto context =
    VulkanRuntimeContext::create(plan, {&kernel, 1}, {&payload, 1}, options);

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_OUT_OF_MEMORY);
  EXPECT_EQ(fake_vulkan::call_count("vkGetBufferMemoryRequirements"), 6U);
  EXPECT_EQ(fake_vulkan::call_count("vkAllocateMemory"), 0U);
}

TEST_F(VulkanRuntimeBudgetTest, ChecksActualAllocationCountBeforeAllocation) {
  const VulkanExecutionPlan plan = makePlan();
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = kPlaceholderSpirv};
  fake_vulkan::set_max_memory_allocation_count(3);

  auto context = VulkanRuntimeContext::create(
    plan, {&kernel, 1}, {}, makeVulkanRuntimeOptions());

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_OUT_OF_MEMORY);
  EXPECT_EQ(fake_vulkan::call_count("vkGetBufferMemoryRequirements"), 4U);
  EXPECT_EQ(fake_vulkan::call_count("vkAllocateMemory"), 0U);
}

TEST_F(VulkanRuntimeBudgetTest, PartialBindFailureReleasesEveryOwnedObject) {
  const VulkanExecutionPlan plan = makePlan();
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = kPlaceholderSpirv};
  fake_vulkan::fail_next("vkBindBufferMemory", VK_ERROR_DEVICE_LOST);

  auto context = VulkanRuntimeContext::create(
    plan, {&kernel, 1}, {}, makeVulkanRuntimeOptions());

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_DEVICE_LOST);
  EXPECT_GT(fake_vulkan::call_count("vkAllocateMemory"), 0U);
}

}  // namespace
}  // namespace ncnn_mlir::vulkan
