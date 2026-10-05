#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace ncnn_mlir::vulkan {
namespace {

VulkanExecutionPlan makePlan() {
  return VulkanExecutionPlan{
    .schema_version = kExecutionPlanSchemaVersion,
    .model = "relu_add_fixture",
    .build_identity = "fixture-build-identity",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources =
      {
        TensorResource{
          .id = 1,
          .kind = ResourceKind::Input,
          .name = "input",
          .shape = {1, 1, 4},
          .logical_layout = "CHW",
          .physical_layout = "linear",
          .host_layout = "CHW",
          .layout_transform = LayoutTransform{.from_layout = "CHW",
                                              .to_layout = "linear",
                                              .operation = "linearize"},
          .weight_sha256 = "",
          .byte_size = 16,
          .first_kernel = 0,
          .last_kernel = 0},
        TensorResource{.id = 3,
                       .kind = ResourceKind::Intermediate,
                       .name = "relu_output",
                       .shape = {1, 1, 4},
                       .logical_layout = "CHW",
                       .physical_layout = "linear",
                       .host_layout = "",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = "",
                       .byte_size = 16,
                       .first_kernel = 0,
                       .last_kernel = 1},
        TensorResource{.id = 4,
                       .kind = ResourceKind::Weight,
                       .name = "scale",
                       .shape = {1, 1, 4},
                       .logical_layout = "CHW",
                       .physical_layout = "linear",
                       .host_layout = "",
                       .layout_transform = std::nullopt,
                       .weight_sha256 = std::string(64, 'a'),
                       .byte_size = 16,
                       .first_kernel = 1,
                       .last_kernel = 1},
        TensorResource{
          .id = 2,
          .kind = ResourceKind::Output,
          .name = "output",
          .shape = {1, 1, 4},
          .logical_layout = "CHW",
          .physical_layout = "linear",
          .host_layout = "CHW",
          .layout_transform = LayoutTransform{.from_layout = "linear",
                                              .to_layout = "CHW",
                                              .operation = "de-linearize"},
          .weight_sha256 = "",
          .byte_size = 16,
          .first_kernel = 1,
          .last_kernel = 1},
      },
    .kernels =
      {
        KernelExecution{
          .id = 10,
          .operation = "relu",
          .entry_point = "relu_main",
          .local_size = {64, 1, 1},
          .reads = {1},
          .writes = {3},
          .bindings = {{.set = 0,
                        .binding = 0,
                        .resource_id = 1,
                        .access = ResourceAccess::Read},
                       {.set = 0,
                        .binding = 1,
                        .resource_id = 3,
                        .access = ResourceAccess::Write}},
          .push_constants = {{.offset = 0, .bytes = {4, 0, 0, 0}}}},
        KernelExecution{.id = 11,
                        .operation = "add",
                        .entry_point = "add_main",
                        .local_size = {64, 1, 1},
                        .reads = {3, 4},
                        .writes = {2},
                        .bindings = {{.set = 0,
                                      .binding = 0,
                                      .resource_id = 3,
                                      .access = ResourceAccess::Read},
                                     {.set = 0,
                                      .binding = 1,
                                      .resource_id = 4,
                                      .access = ResourceAccess::Read},
                                     {.set = 0,
                                      .binding = 2,
                                      .resource_id = 2,
                                      .access = ResourceAccess::Write}},
                        .push_constants = {}},
      },
    .inputs = {1},
    .outputs = {2},
    .plan_hash = "",
  };
}

TEST(VulkanExecutionPlanTest, SerializesDeterministicallyAndRoundTrips) {
  const VulkanExecutionPlan plan = makePlan();
  auto first = serializeExecutionPlan(plan);
  ASSERT_TRUE(first) << first.error();
  auto second = serializeExecutionPlan(plan);
  ASSERT_TRUE(second) << second.error();
  EXPECT_EQ(*first, *second);
  EXPECT_NE(first->find("\"schema_version\": 1"), std::string::npos);
  EXPECT_NE(first->find("\"weight_sha256\""), std::string::npos);
  EXPECT_NE(first->find("\"plan_hash\""), std::string::npos);

  auto decoded = deserializeExecutionPlan(*first);
  ASSERT_TRUE(decoded) << decoded.error() << '\n' << *first;
  auto roundTrip = serializeExecutionPlan(*decoded);
  ASSERT_TRUE(roundTrip) << roundTrip.error();
  EXPECT_EQ(*first, *roundTrip);
}

TEST(VulkanExecutionPlanTest, PreservesUnusedPublicInputLifetimeSentinel) {
  VulkanExecutionPlan plan = makePlan();
  TensorResource unused = plan.resources.front();
  unused.id = 5;
  unused.name = "unused_input";
  unused.first_kernel = UINT32_MAX;
  unused.last_kernel = UINT32_MAX;
  plan.resources.push_back(unused);
  plan.inputs.push_back(unused.id);

  auto encoded = serializeExecutionPlan(plan);
  ASSERT_TRUE(encoded) << encoded.error();
  auto decoded = deserializeExecutionPlan(*encoded);
  ASSERT_TRUE(decoded) << decoded.error();
  EXPECT_EQ(decoded->inputs, plan.inputs);
  EXPECT_EQ(decoded->resources.back().first_kernel, UINT32_MAX);
  EXPECT_EQ(decoded->resources.back().last_kernel, UINT32_MAX);
  auto roundTrip = serializeExecutionPlan(*decoded);
  ASSERT_TRUE(roundTrip) << roundTrip.error();
  EXPECT_EQ(*encoded, *roundTrip);

  plan.resources.back().first_kernel = 0;
  plan.resources.back().last_kernel = 0;
  auto ordinaryLifetime = verifyExecutionPlan(plan);
  ASSERT_FALSE(ordinaryLifetime);
  EXPECT_NE(ordinaryLifetime.error().find("must be used"), std::string::npos);
}

TEST(VulkanExecutionPlanTest,
     RejectsUnusedSentinelForAccessedOrNonInputResource) {
  VulkanExecutionPlan accessed = makePlan();
  accessed.resources.front().first_kernel = UINT32_MAX;
  accessed.resources.front().last_kernel = UINT32_MAX;
  auto usedInput = verifyExecutionPlan(accessed);
  ASSERT_FALSE(usedInput);
  EXPECT_NE(usedInput.error().find("lifetime"), std::string::npos);

  for (unsigned index : {1U, 2U, 3U}) {
    VulkanExecutionPlan nonInput = makePlan();
    nonInput.resources[index].first_kernel = UINT32_MAX;
    nonInput.resources[index].last_kernel = UINT32_MAX;
    auto invalid = verifyExecutionPlan(nonInput);
    ASSERT_FALSE(invalid);
    EXPECT_NE(invalid.error().find("lifetime"), std::string::npos);
  }
}

TEST(VulkanExecutionPlanTest, RejectsDuplicateDescriptorBindings) {
  VulkanExecutionPlan plan = makePlan();
  plan.kernels[0].bindings.push_back(plan.kernels[0].bindings.front());
  EXPECT_FALSE(verifyExecutionPlan(plan));
}

TEST(VulkanExecutionPlanTest, RejectsReadBeforeWrite) {
  VulkanExecutionPlan plan = makePlan();
  plan.kernels[0].reads = {3};
  plan.kernels[0].writes = {2};
  plan.kernels[0].bindings = {
    {.set = 0, .binding = 0, .resource_id = 3, .access = ResourceAccess::Read},
    {.set = 0, .binding = 1, .resource_id = 2, .access = ResourceAccess::Write},
  };
  plan.kernels[1].reads = {1};
  plan.kernels[1].writes = {3};
  plan.kernels[1].bindings = {
    {.set = 0, .binding = 0, .resource_id = 1, .access = ResourceAccess::Read},
    {.set = 0, .binding = 1, .resource_id = 3, .access = ResourceAccess::Write},
  };
  plan.resources[0].first_kernel = 1;
  plan.resources[0].last_kernel = 1;
  plan.resources[1].first_kernel = 0;
  plan.resources[1].last_kernel = 1;
  plan.resources[2].first_kernel = 1;
  plan.resources[2].last_kernel = 1;
  plan.resources[3].first_kernel = 0;
  plan.resources[3].last_kernel = 0;

  auto result = verifyExecutionPlan(plan);
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().find("reads a resource before"), std::string::npos);
}

TEST(VulkanExecutionPlanTest, RejectsInvalidDispatchAndPushConstants) {
  VulkanExecutionPlan localSize = makePlan();
  localSize.kernels[0].local_size = {0, 1, 1};
  auto localSizeResult = verifyExecutionPlan(localSize);
  ASSERT_FALSE(localSizeResult);
  EXPECT_NE(localSizeResult.error().find("local_size"), std::string::npos);

  VulkanExecutionPlan overlap = makePlan();
  overlap.kernels[0].push_constants.push_back(
    {.offset = 0, .bytes = {8, 0, 0, 0}});
  auto overlapResult = verifyExecutionPlan(overlap);
  ASSERT_FALSE(overlapResult);
  EXPECT_NE(overlapResult.error().find("must not overlap"), std::string::npos);
}

TEST(VulkanExecutionPlanTest, RejectsOverflowAndByteSizeMismatch) {
  VulkanExecutionPlan overflow = makePlan();
  overflow.resources[0].shape = {UINT64_MAX, 4};
  auto overflowResult = verifyExecutionPlan(overflow);
  ASSERT_FALSE(overflowResult);
  EXPECT_NE(overflowResult.error().find("overflows"), std::string::npos);

  VulkanExecutionPlan mismatch = makePlan();
  mismatch.resources[0].byte_size = 20;
  auto mismatchResult = verifyExecutionPlan(mismatch);
  ASSERT_FALSE(mismatchResult);
  EXPECT_NE(mismatchResult.error().find("byte_size"), std::string::npos);
}

TEST(VulkanExecutionPlanTest, RejectsInvalidHostDeviceLayoutMappings) {
  VulkanExecutionPlan missingMapping = makePlan();
  missingMapping.resources[0].layout_transform.reset();
  auto missingResult = verifyExecutionPlan(missingMapping);
  ASSERT_FALSE(missingResult);
  EXPECT_NE(missingResult.error().find("layout mismatch"), std::string::npos);

  VulkanExecutionPlan wrongDirection = makePlan();
  wrongDirection.resources[0].layout_transform->to_layout = "CHW";
  auto directionResult = verifyExecutionPlan(wrongDirection);
  ASSERT_FALSE(directionResult);
  EXPECT_NE(directionResult.error().find("layout mismatch"), std::string::npos);
}

TEST(VulkanExecutionPlanTest, RejectsInvalidLifetimeAndSchemaVersion) {
  VulkanExecutionPlan lifetime = makePlan();
  lifetime.resources[0].last_kernel = 1;
  auto lifetimeResult = verifyExecutionPlan(lifetime);
  ASSERT_FALSE(lifetimeResult);
  EXPECT_NE(lifetimeResult.error().find("lifetime"), std::string::npos);

  VulkanExecutionPlan version = makePlan();
  version.schema_version = kExecutionPlanSchemaVersion + 1;
  auto versionResult = verifyExecutionPlan(version);
  ASSERT_FALSE(versionResult);
  EXPECT_NE(versionResult.error().find("schema_version"), std::string::npos);
}

TEST(VulkanExecutionPlanTest, RejectsUnsupportedJsonFieldsAndHashTampering) {
  VulkanExecutionPlan plan = makePlan();
  auto serialized = serializeExecutionPlan(plan);
  ASSERT_TRUE(serialized) << serialized.error();
  std::string extraField = *serialized;
  const auto position = extraField.find("\"kind\"");
  ASSERT_NE(position, std::string::npos);
  extraField.insert(position, "\"unexpected\": true,\n  ");
  EXPECT_FALSE(deserializeExecutionPlan(extraField));

  std::string tampered = *serialized;
  const auto hashKey = tampered.find("\"plan_hash\"");
  ASSERT_NE(hashKey, std::string::npos);
  const auto hashColon = tampered.find(':', hashKey);
  ASSERT_NE(hashColon, std::string::npos);
  const auto hashValue =
    tampered.find_first_of("0123456789abcdef", hashColon + 1);
  ASSERT_NE(hashValue, std::string::npos);
  tampered[hashValue] = tampered[hashValue] == '0' ? '1' : '0';
  auto tamperedPlan = deserializeExecutionPlan(tampered);
  ASSERT_FALSE(tamperedPlan);
  EXPECT_NE(tamperedPlan.error().find("plan_hash"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_mlir::vulkan
