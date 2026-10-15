#include "vulkan_shader_fixture.hpp"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include <gtest/gtest.h>

namespace ncnn_mlir::vulkan {
namespace {

struct HostCase final {
  VulkanExecutionPlan plan;
  std::vector<VulkanModelShader> shaders;
  std::vector<VulkanModelWeight> weights;
};

TensorResource resource(std::uint32_t id,
                        ResourceKind kind,
                        std::uint64_t elements = 65) {
  const bool host = kind == ResourceKind::Input || kind == ResourceKind::Output;
  return {.id = id,
          .kind = kind,
          .name = "resource_" + std::to_string(id),
          .shape = {elements},
          .logical_layout = "linear",
          .physical_layout = "linear",
          .host_layout = host ? "linear" : "",
          .layout_transform = std::nullopt,
          .weight_sha256 = "",
          .byte_size = elements * 4,
          .first_kernel = std::numeric_limits<std::uint32_t>::max(),
          .last_kernel = std::numeric_limits<std::uint32_t>::max()};
}

KernelExecution kernel(std::uint32_t id,
                       std::vector<std::uint32_t> reads,
                       std::vector<std::uint32_t> writes) {
  std::vector<DescriptorBinding> bindings;
  for (auto resource_id : reads) {
    bindings.push_back({0,
                        static_cast<std::uint32_t>(bindings.size()),
                        resource_id,
                        ResourceAccess::Read});
  }
  for (auto resource_id : writes) {
    bindings.push_back({0,
                        static_cast<std::uint32_t>(bindings.size()),
                        resource_id,
                        ResourceAccess::Write});
  }
  return {.id = id,
          .operation = "copy",
          .entry_point = "kernel_" + std::to_string(id),
          .local_size = {64, 1, 1},
          .reads = std::move(reads),
          .writes = std::move(writes),
          .bindings = std::move(bindings),
          .push_constants = {}};
}

std::expected<HostCase, std::string> fixture(
  std::vector<TensorResource> resources,
  std::vector<KernelExecution> kernels,
  std::vector<VulkanModelWeight> weights = {}) {
  VulkanExecutionPlan plan{.schema_version = kExecutionPlanSchemaVersion,
                           .model = "host_fixture",
                           .build_identity = "unit-source-not-host-identity",
                           .host_triple = "x86_64-unknown-linux-gnu",
                           .device_target = "vulkan1.1-spirv1.0",
                           .resources = std::move(resources),
                           .kernels = std::move(kernels),
                           .inputs = {},
                           .outputs = {},
                           .plan_hash = ""};
  for (auto& value : plan.resources) {
    if (value.kind == ResourceKind::Input) {
      plan.inputs.push_back(value.id);
    } else if (value.kind == ResourceKind::Output) {
      plan.outputs.push_back(value.id);
    } else if (value.kind == ResourceKind::Weight) {
      auto weight =
        std::ranges::find(weights, value.id, &VulkanModelWeight::resource_id);
      if (weight == weights.end()) {
        return std::unexpected("unit weight payload is missing");
      }
      auto hash = vulkanWeightSHA256(value, weight->bytes);
      if (!hash) {
        return std::unexpected(hash.error());
      }
      value.weight_sha256 = *hash;
    }
    for (std::size_t index = 0; index < plan.kernels.size(); ++index) {
      const auto& bindings = plan.kernels[index].bindings;
      if (std::ranges::any_of(bindings, [&](const auto& binding) {
            return binding.resource_id == value.id;
          })) {
        value.first_kernel =
          std::min(value.first_kernel, static_cast<std::uint32_t>(index));
        value.last_kernel = static_cast<std::uint32_t>(index);
      }
    }
  }
  if (auto valid = verifyExecutionPlan(plan); !valid) {
    return std::unexpected(valid.error());
  }
  std::vector<VulkanModelShader> shaders;
  for (const auto& value : plan.kernels) {
    auto shader = test::makeTestVulkanShader(plan, value);
    if (!shader) {
      return std::unexpected(shader.error());
    }
    shaders.push_back(std::move(*shader));
  }
  return HostCase{.plan = std::move(plan),
                  .shaders = std::move(shaders),
                  .weights = std::move(weights)};
}

std::expected<HostCase, std::string> chain(std::uint64_t elements = 65) {
  return fixture({resource(0, ResourceKind::Input, elements),
                  resource(1, ResourceKind::Intermediate, elements),
                  resource(2, ResourceKind::Output, elements)},
                 {kernel(10, {0}, {1}), kernel(20, {1}, {2})});
}

// Frozen recorder oracle: search earlier writers for each current access. This
// deliberately does not reuse the planner's running shader_written set.
std::vector<std::set<std::uint32_t>> legacyComputeHazards(
  const VulkanExecutionPlan& plan) {
  std::vector<std::set<std::uint32_t>> result;
  for (std::size_t index = 0; index < plan.kernels.size(); ++index) {
    std::set<std::uint32_t> current;
    for (const auto& binding : plan.kernels[index].bindings) {
      for (std::size_t earlier = 0; earlier < index; ++earlier) {
        const auto& writes = plan.kernels[earlier].writes;
        if (std::ranges::find(writes, binding.resource_id) != writes.end()) {
          current.insert(binding.resource_id);
        }
      }
    }
    result.push_back(std::move(current));
  }
  return result;
}

std::vector<std::set<std::uint32_t>> computeHazards(
  const VulkanHostProgram& program) {
  std::vector<std::set<std::uint32_t>> result;
  std::set<std::uint32_t> pending;
  for (const auto& operation : program.run_operations) {
    if (const auto* barrier = std::get_if<HostBarrierOp>(&operation)) {
      if (barrier->source_stage == HostStage::Compute &&
          barrier->destination_stage == HostStage::Compute) {
        for (const auto& value : barrier->buffers) {
          const auto& buffer = program.buffers[value.buffer.value];
          EXPECT_EQ(value.source_access, HostAccess::ShaderWrite);
          EXPECT_EQ(value.destination_access, HostAccess::ShaderReadWrite);
          EXPECT_EQ(value.offset, 0);
          EXPECT_EQ(value.byte_size, buffer.byte_size);
          pending.insert(buffer.resource_id);
        }
      }
    } else if (std::holds_alternative<HostDispatchOp>(operation)) {
      result.push_back(pending);
      pending.clear();
    }
  }
  EXPECT_TRUE(pending.empty());
  return result;
}

TEST(VulkanHostProgramTest, ChainMatchesLegacyTrace) {
  auto source = chain();
  ASSERT_TRUE(source) << source.error();
  auto program =
    makeVulkanHostProgram(source->plan, source->shaders, source->weights);
  ASSERT_TRUE(program) << program.error();
  EXPECT_TRUE(verifyVulkanHostProgram(
    *program, source->plan, source->shaders, source->weights));
  EXPECT_EQ(computeHazards(*program), legacyComputeHazards(source->plan));
  ASSERT_EQ(program->pipelines.size(), 2);
  EXPECT_EQ(program->pipelines[0].dispatch,
            (std::array<std::uint32_t, 3>{2, 1, 1}));
  EXPECT_EQ(program->pipelines[1].dispatch,
            (std::array<std::uint32_t, 3>{2, 1, 1}));
  EXPECT_TRUE(program->create_operations.empty());
  std::vector<std::pair<HostStage, HostStage>> transfers;
  std::size_t copies = 0;
  for (const auto& operation : program->run_operations) {
    if (const auto* barrier = std::get_if<HostBarrierOp>(&operation)) {
      if (barrier->source_stage != barrier->destination_stage) {
        transfers.emplace_back(barrier->source_stage,
                               barrier->destination_stage);
      }
    } else if (std::holds_alternative<HostCopyOp>(operation)) {
      ++copies;
    }
  }
  EXPECT_EQ(copies, 2);
  const std::vector<std::pair<HostStage, HostStage>> expected{
    {HostStage::Host, HostStage::Transfer},
    {HostStage::Transfer, HostStage::Compute},
    {HostStage::Compute, HostStage::Transfer},
    {HostStage::Transfer, HostStage::Host}};
  EXPECT_EQ(transfers, expected);
}

TEST(VulkanHostProgramTest, BranchAndRepeatedOutputs) {
  auto source = fixture({resource(0, ResourceKind::Input),
                         resource(1, ResourceKind::Intermediate),
                         resource(2, ResourceKind::Intermediate),
                         resource(3, ResourceKind::Output),
                         resource(4, ResourceKind::Output),
                         resource(5, ResourceKind::Output)},
                        {kernel(0, {0}, {1}),
                         kernel(1, {0}, {2}),
                         kernel(2, {1, 2}, {3}),
                         kernel(3, {1}, {4}),
                         kernel(4, {1}, {5})});
  ASSERT_TRUE(source) << source.error();
  auto program = makeVulkanHostProgram(source->plan, source->shaders, {});
  ASSERT_TRUE(program) << program.error();
  EXPECT_EQ(computeHazards(*program), legacyComputeHazards(source->plan));
  ASSERT_EQ(program->outputs.size(), 3);
  EXPECT_NE(program->outputs[1].device_buffer,
            program->outputs[2].device_buffer);
  EXPECT_NE(program->outputs[1].staging_buffer,
            program->outputs[2].staging_buffer);
  std::size_t invalidates = 0;
  std::size_t publications = 0;
  bool prepared = false;
  for (const auto& operation : program->run_operations) {
    if (const auto* sync = std::get_if<HostMappedSyncOp>(&operation)) {
      if (sync->action == HostMappedSync::Invalidate) {
        EXPECT_FALSE(prepared);
        ++invalidates;
      }
    } else if (std::holds_alternative<HostPrepareOutputsOp>(operation)) {
      EXPECT_EQ(invalidates, 3);
      prepared = true;
    } else if (const auto* publish = std::get_if<HostPublishOp>(&operation)) {
      EXPECT_TRUE(prepared);
      EXPECT_EQ(publish->output_index, publications);
      ++publications;
    }
  }
  EXPECT_EQ(publications, 3);
  // A read-only SSA view uses the same resource; duplicate public IDs are not
  // aliases, and must still fail rather than silently merge output ports.
  auto duplicate = source->plan;
  duplicate.outputs[2] = duplicate.outputs[1];
  EXPECT_FALSE(makeVulkanHostProgram(duplicate, source->shaders, {}));
}

TEST(VulkanHostProgramTest, WeightsAndUnusedInput) {
  auto source = fixture({resource(0, ResourceKind::Input, 4),
                         resource(1, ResourceKind::Weight, 4),
                         resource(2, ResourceKind::Output, 4),
                         resource(3, ResourceKind::Input, 4)},
                        {kernel(0, {0, 1}, {2})},
                        {{1, std::vector<std::uint8_t>(16, 0)}});
  ASSERT_TRUE(source) << source.error();
  auto program =
    makeVulkanHostProgram(source->plan, source->shaders, source->weights);
  ASSERT_TRUE(program) << program.error();
  EXPECT_EQ(program->inputs.size(), 2);
  EXPECT_EQ(program->resources[3].first_kernel,
            std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(computeHazards(*program), legacyComputeHazards(source->plan));
  std::size_t uploads = 0;
  std::size_t releases = 0;
  for (const auto& operation : program->create_operations) {
    if (const auto* upload = std::get_if<HostUploadOp>(&operation)) {
      EXPECT_EQ(upload->source, HostUploadSource::Weight);
      EXPECT_EQ(upload->source_id, 1);
      ++uploads;
    } else if (const auto* release =
                 std::get_if<HostReleaseBufferOp>(&operation)) {
      const auto& buffer = program->buffers[release->buffer.value];
      EXPECT_EQ(buffer.scope, HostBufferScope::CreateUpload);
      EXPECT_EQ(buffer.kind, HostBufferKind::WeightStaging);
      ++releases;
    }
  }
  EXPECT_EQ(uploads, 1);
  EXPECT_EQ(releases, 1);
  for (const auto& object : program->destroy) {
    if (object.kind == HostObjectKind::Buffer) {
      EXPECT_EQ(program->buffers[object.id].scope, HostBufferScope::Context);
    }
  }
  auto bad_weights = source->weights;
  bad_weights[0].bytes[0] = 1;
  EXPECT_FALSE(
    makeVulkanHostProgram(source->plan, source->shaders, bad_weights));
}

TEST(VulkanHostProgramTest, WawWarAndReadRead) {
  // Resource 1 is produced, read twice, and rewritten. Its WAR execution
  // dependency is covered by the same conservative compute stage barrier used
  // by the legacy recorder; no in-place or input/weight writes are introduced.
  auto source = fixture({resource(0, ResourceKind::Input),
                         resource(1, ResourceKind::Intermediate),
                         resource(2, ResourceKind::Output),
                         resource(3, ResourceKind::Output),
                         resource(4, ResourceKind::Output)},
                        {kernel(0, {0}, {1}),
                         kernel(1, {1}, {2}),
                         kernel(2, {1}, {3}),
                         kernel(3, {0}, {1}),
                         kernel(4, {0}, {1}),
                         kernel(5, {1}, {4})});
  ASSERT_TRUE(source) << source.error();
  auto program = makeVulkanHostProgram(source->plan, source->shaders, {});
  ASSERT_TRUE(program) << program.error();
  const auto hazards = computeHazards(*program);
  EXPECT_EQ(hazards, legacyComputeHazards(source->plan));
  ASSERT_EQ(hazards.size(), 6);
  EXPECT_TRUE(hazards[0].empty());
  for (std::size_t index = 1; index < hazards.size(); ++index) {
    EXPECT_EQ(hazards[index], (std::set<std::uint32_t>{1}));
  }
}

TEST(VulkanHostProgramTest, RejectsReferencesLifetimesAndBindings) {
  auto source = chain();
  ASSERT_TRUE(source) << source.error();
  auto missing = source->plan;
  missing.kernels[1].reads[0] = 99;
  EXPECT_FALSE(makeVulkanHostProgram(missing, source->shaders, {}));
  auto lifetime = source->plan;
  lifetime.resources[1].last_kernel = 0;
  EXPECT_FALSE(makeVulkanHostProgram(lifetime, source->shaders, {}));
  auto binding = source->plan;
  binding.kernels[0].bindings[1].binding = 0;
  EXPECT_FALSE(makeVulkanHostProgram(binding, source->shaders, {}));
  auto before_producer = source->plan;
  std::swap(before_producer.kernels[0], before_producer.kernels[1]);
  EXPECT_FALSE(makeVulkanHostProgram(before_producer, source->shaders, {}));
  auto in_place = source->plan;
  in_place.kernels[1].writes = {1};
  EXPECT_FALSE(makeVulkanHostProgram(in_place, source->shaders, {}));
  auto missing_shader = source->shaders;
  missing_shader.pop_back();
  EXPECT_FALSE(makeVulkanHostProgram(source->plan, missing_shader, {}));
  auto duplicated_shader = source->shaders;
  duplicated_shader[1] = duplicated_shader[0];
  EXPECT_FALSE(makeVulkanHostProgram(source->plan, duplicated_shader, {}));
}

TEST(VulkanHostProgramTest, RejectsDispatchAndOverflow) {
  auto source = chain();
  ASSERT_TRUE(source) << source.error();
  auto zero = source->plan;
  zero.kernels[0].local_size[0] = 0;
  EXPECT_FALSE(makeVulkanHostProgram(zero, source->shaders, {}));
  auto overflow = source->plan;
  overflow.resources[0].shape = {std::numeric_limits<std::uint64_t>::max(), 2};
  EXPECT_FALSE(makeVulkanHostProgram(overflow, source->shaders, {}));
  auto oversized = source->plan;
  oversized.resources[0].shape = {(kMinimumStorageBufferRange / 4) + 1};
  oversized.resources[0].byte_size = kMinimumStorageBufferRange + 4;
  EXPECT_FALSE(makeVulkanHostProgram(oversized, source->shaders, {}));
  auto target = source->plan;
  target.device_target = "vulkan1.2-spirv1.5";
  EXPECT_FALSE(makeVulkanHostProgram(target, source->shaders, {}));
  target = source->plan;
  target.host_triple = "aarch64-unknown-linux-gnu";
  EXPECT_FALSE(makeVulkanHostProgram(target, source->shaders, {}));
  auto sparse_set = source->plan;
  sparse_set.kernels[0].bindings[0].set =
    std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(makeVulkanHostProgram(sparse_set, source->shaders, {}));
  // A target's guaranteed minimum is not its actual max group count. Keep the
  // declared resource contract intact; actual device limits remain late-bound.
  auto large = chain(kMinimumStorageBufferRange / 4);
  ASSERT_TRUE(large) << large.error();
  auto program = makeVulkanHostProgram(large->plan, large->shaders, {});
  ASSERT_TRUE(program) << program.error();
  EXPECT_GT(program->pipelines[0].dispatch[0], 65535U);
}

TEST(VulkanHostProgramTest, RejectsTamperedOperations) {
  auto source = chain();
  ASSERT_TRUE(source) << source.error();
  auto program = makeVulkanHostProgram(source->plan, source->shaders, {});
  ASSERT_TRUE(program) << program.error();
  auto altered = *program;
  for (auto& operation : altered.run_operations) {
    if (auto* copy = std::get_if<HostCopyOp>(&operation)) {
      copy->source.value = 999;
      break;
    }
  }
  auto identity = calculateVulkanHostProgramIdentity(altered);
  ASSERT_TRUE(identity) << identity.error();
  altered.identity = *identity;
  EXPECT_FALSE(
    verifyVulkanHostProgram(altered, source->plan, source->shaders, {}));
  altered = *program;
  std::erase_if(altered.run_operations, [](const auto& operation) {
    const auto* barrier = std::get_if<HostBarrierOp>(&operation);
    return barrier != nullptr && barrier->source_stage == HostStage::Compute &&
           barrier->destination_stage == HostStage::Compute;
  });
  identity = calculateVulkanHostProgramIdentity(altered);
  ASSERT_TRUE(identity) << identity.error();
  altered.identity = *identity;
  EXPECT_FALSE(
    verifyVulkanHostProgram(altered, source->plan, source->shaders, {}));
  altered = *program;
  std::erase_if(altered.run_operations, [](const auto& operation) {
    return std::holds_alternative<HostPrepareOutputsOp>(operation);
  });
  EXPECT_FALSE(
    verifyVulkanHostProgram(altered, source->plan, source->shaders, {}));
  altered = *program;
  altered.run_operations.emplace_back(HostReleaseBufferOp{HostBufferId{0}});
  EXPECT_FALSE(
    verifyVulkanHostProgram(altered, source->plan, source->shaders, {}));
  altered = *program;
  altered.synchronization_version += "-unknown";
  EXPECT_FALSE(
    verifyVulkanHostProgram(altered, source->plan, source->shaders, {}));
}

TEST(VulkanHostProgramTest, IdentityIsCanonicalAndSensitive) {
  auto source = chain();
  ASSERT_TRUE(source) << source.error();
  source->plan.resources[0].name = "输入\n\"\\\t";
  auto program = makeVulkanHostProgram(source->plan, source->shaders, {});
  ASSERT_TRUE(program) << program.error();
  auto repeated = makeVulkanHostProgram(source->plan, source->shaders, {});
  ASSERT_TRUE(repeated) << repeated.error();
  EXPECT_EQ(program->identity, repeated->identity);
  EXPECT_EQ(serializeVulkanHostProgram(*program),
            serializeVulkanHostProgram(*repeated));
  EXPECT_EQ(program->identity.size(), 64);
  auto changed_source = source->plan;
  changed_source.build_identity = "different-unrelated-intermediate-hash";
  changed_source.plan_hash.clear();
  auto changed_json = serializeExecutionPlan(changed_source);
  ASSERT_TRUE(changed_json) << changed_json.error();
  auto checked_source = deserializeExecutionPlan(*changed_json);
  ASSERT_TRUE(checked_source) << checked_source.error();
  repeated = makeVulkanHostProgram(*checked_source, source->shaders, {});
  ASSERT_TRUE(repeated) << repeated.error();
  EXPECT_EQ(program->identity, repeated->identity);
  auto reordered = source->shaders;
  std::ranges::reverse(reordered);
  repeated = makeVulkanHostProgram(source->plan, reordered, {});
  ASSERT_TRUE(repeated) << repeated.error();
  EXPECT_EQ(program->identity, repeated->identity);
  auto changed = *program;
  changed.codegen_version += "-different";
  auto hash = calculateVulkanHostProgramIdentity(changed);
  ASSERT_TRUE(hash) << hash.error();
  EXPECT_NE(program->identity, *hash);
  changed = *program;
  changed.pipelines[0].bindings[0].binding += 2;
  hash = calculateVulkanHostProgramIdentity(changed);
  ASSERT_TRUE(hash) << hash.error();
  EXPECT_NE(program->identity, *hash);
  changed = *program;
  changed.identity = std::string(64, '0');
  EXPECT_FALSE(serializeVulkanHostProgram(changed));
  changed = *program;
  changed.resources[0].name = std::string(1, static_cast<char>(0xff));
  EXPECT_FALSE(calculateVulkanHostProgramIdentity(changed));
  auto invalid_text = source->plan;
  invalid_text.resources[0].name = changed.resources[0].name;
  EXPECT_FALSE(makeVulkanHostProgram(invalid_text, source->shaders, {}));
}

}  // namespace
}  // namespace ncnn_mlir::vulkan
