#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"

namespace ncnn_mlir::vulkan {
namespace {

VulkanExecutionPlan makePlan() {
  return VulkanExecutionPlan{
    .schema_version = kExecutionPlanSchemaVersion,
    .model = "runtime_fixture",
    .build_identity = "runtime-fixture-build",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources = {TensorResource{.id = 1,
                                 .kind = ResourceKind::Input,
                                 .name = "input",
                                 .shape = {1, 1, 4},
                                 .logical_layout = "CHW",
                                 .physical_layout = "linear",
                                 .host_layout = "CHW",
                                 .layout_transform =
                                   LayoutTransform{.from_layout = "CHW",
                                                   .to_layout = "linear",
                                                   .operation = "linearize"},
                                 .weight_sha256 = "",
                                 .byte_size = 16,
                                 .first_kernel = 0,
                                 .last_kernel = 0},
                  TensorResource{.id = 2,
                                 .kind = ResourceKind::Output,
                                 .name = "output",
                                 .shape = {1, 1, 4},
                                 .logical_layout = "CHW",
                                 .physical_layout = "linear",
                                 .host_layout = "CHW",
                                 .layout_transform =
                                   LayoutTransform{.from_layout = "linear",
                                                   .to_layout = "CHW",
                                                   .operation = "de-linearize"},
                                 .weight_sha256 = "",
                                 .byte_size = 16,
                                 .first_kernel = 0,
                                 .last_kernel = 0}},
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
    .plan_hash = "",
  };
}

std::array<std::uint32_t, 5> makePlaceholderSpirv() {
  return {0x07230203U, 0x00010000U, 0U, 1U, 0U};
}

TEST(VulkanRuntimeTest, RejectsInvalidPlanBeforeCreatingDevice) {
  VulkanExecutionPlan plan = makePlan();
  plan.device_target = "vulkan1.2-spirv1.5";
  const auto spirv = makePlaceholderSpirv();
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = spirv};
  const VulkanRuntimeOptions options = makeVulkanRuntimeOptions();

  auto context = VulkanRuntimeContext::create(plan, {&kernel, 1}, {}, options);

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_UNSUPPORTED_DEVICE);
  EXPECT_NE(context.error().diagnostic.find("Vulkan 1.1"), std::string::npos);
}

TEST(VulkanRuntimeTest, RejectsMissingKernelPackage) {
  const VulkanExecutionPlan plan = makePlan();
  const VulkanRuntimeOptions options = makeVulkanRuntimeOptions();

  auto context = VulkanRuntimeContext::create(plan, {}, {}, options);

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(context.error().diagnostic.find("kernel count"), std::string::npos);
}

TEST(VulkanRuntimeTest, RejectsZeroMemoryBudgetBeforeCreatingDevice) {
  const VulkanExecutionPlan plan = makePlan();
  const auto spirv = makePlaceholderSpirv();
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = spirv};
  VulkanRuntimeOptions options = makeVulkanRuntimeOptions();
  options.max_context_memory_bytes = 0;

  auto context = VulkanRuntimeContext::create(plan, {&kernel, 1}, {}, options);

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(context.error().diagnostic.find("budget"), std::string::npos);
}

TEST(VulkanRuntimeTest, InjectedInstanceFailureReturnsStatusAndDiagnostic) {
  const VulkanExecutionPlan plan = makePlan();
  const auto spirv = makePlaceholderSpirv();
  const VulkanKernelBinary kernel{.kernel_id = 10, .words = spirv};
  VulkanRuntimeOptions options = makeVulkanRuntimeOptions();
  options.failure_injector =
    [](VulkanRuntimeOperation operation) -> std::optional<ncnn_vk_status> {
    if (operation == VulkanRuntimeOperation::CreateInstance) {
      return NCNN_VK_STATUS_OUT_OF_MEMORY;
    }
    return std::nullopt;
  };

  auto context = VulkanRuntimeContext::create(plan, {&kernel, 1}, {}, options);

  ASSERT_FALSE(context);
  EXPECT_EQ(context.error().status, NCNN_VK_STATUS_OUT_OF_MEMORY);
  EXPECT_NE(context.error().diagnostic.find("injected"), std::string::npos);
}

TEST(VulkanRuntimeTest, StatusNamesAreStableAndTotal) {
  EXPECT_EQ(getVulkanRuntimeStatusName(NCNN_VK_STATUS_SUCCESS), "success");
  EXPECT_EQ(getVulkanRuntimeStatusName(NCNN_VK_STATUS_DEVICE_LOST),
            "device_lost");
  EXPECT_EQ(getVulkanRuntimeStatusName(999), "unknown_status");
}

}  // namespace
}  // namespace ncnn_mlir::vulkan
