#include "fixture_expectations.hpp"
#include "lifecycle_test_hooks.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include "candidate_source.cpp"
#include "fakeVulkan/fake_vulkan.hpp"
#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"
#include "ncnn-mlir/Vulkan/VulkanDeviceTypes.hpp"
#include "va4_fixedrun_vulkan.h"

namespace {

using ncnn_mlir::vulkan::VulkanDeviceOperation;
using ncnn_mlir::vulkan::test::LifecycleTestSettings;

LifecycleTestSettings g_settings{
  .device_name_match = "",
  .enable_validation = false,
  .force_mapped_memory_sync = true,
  .max_context_memory_bytes = 512U * 1024U * 1024U,
  .operation_hook = {}};

struct ActualBarrier final {
  std::uint32_t structure_type;
  bool has_next;
  std::uint32_t buffer_slot;
  std::uint32_t source_access;
  std::uint32_t destination_access;
  std::uint32_t source_queue_family;
  std::uint32_t destination_queue_family;
  std::uint64_t offset;
  std::uint64_t size;
};

struct ActualCopyRegion final {
  std::uint64_t source_offset;
  std::uint64_t destination_offset;
  std::uint64_t size;
};

struct RecordedCommand final {
  std::uint64_t run_id;
  fixedrun_expectations::CommandKind kind;
  std::uint32_t command_buffer_slot = UINT32_MAX;
  std::uint32_t source_stage = 0;
  std::uint32_t destination_stage = 0;
  std::uint32_t dependency_flags = 0;
  std::uint32_t memory_barrier_count = 0;
  std::uint32_t image_barrier_count = 0;
  bool memory_barriers_null = true;
  bool buffer_barriers_null = true;
  bool image_barriers_null = true;
  std::vector<ActualBarrier> barriers;
  std::uint32_t source_buffer = 0;
  std::uint32_t destination_buffer = 0;
  std::uint32_t copy_region_count = 0;
  bool copy_regions_null = true;
  std::vector<ActualCopyRegion> copy_regions;
  VkPipelineBindPoint pipeline_bind_point = VK_PIPELINE_BIND_POINT_MAX_ENUM;
  std::uint32_t pipeline_slot = UINT32_MAX;
  VkPipelineBindPoint descriptor_bind_point = VK_PIPELINE_BIND_POINT_MAX_ENUM;
  std::uint32_t first_set = 0;
  std::uint32_t set_count = 0;
  bool descriptor_sets_null = true;
  std::vector<std::uint32_t> descriptor_slots;
  std::uint32_t dynamic_offset_count = 0;
  bool dynamic_offsets_null = true;
  std::vector<std::uint32_t> dynamic_offsets;
  std::uint32_t push_stage_flags = 0;
  bool push_data_null = true;
  std::uint32_t push_offset = 0;
  std::uint32_t push_size = 0;
  std::vector<std::uint8_t> push_bytes;
  std::array<std::uint32_t, 3> groups{};
};

thread_local std::uint64_t g_run_id = 0;
thread_local std::uint32_t g_bound_pipeline_slot = UINT32_MAX;
std::mutex g_command_mutex;
std::vector<RecordedCommand> g_commands;
std::atomic<std::uint64_t> g_copy_command_count{0};
std::atomic<std::uint64_t> g_dispatch_count{0};
std::mutex g_check_mutex;
std::string_view g_failed_check;

void check(bool condition, std::string_view description) {
  if (!condition) {
    std::scoped_lock lock(g_check_mutex);
    if (g_failed_check.empty()) {
      g_failed_check = description;
    }
    std::cerr << "FAIL: " << description << '\n';
  }
}

void recordCommand(RecordedCommand command) {
  std::scoped_lock lock(g_command_mutex);
  command.run_id = g_run_id;
  g_commands.push_back(std::move(command));
}

void resetCommands() {
  std::scoped_lock lock(g_command_mutex);
  g_commands.clear();
}

bool commandsAreGroupedByRun() {
  std::scoped_lock lock(g_command_mutex);
  if (g_commands.empty()) {
    return false;
  }
  std::set<std::uint64_t> completed_runs;
  std::uint64_t current_run = g_commands.front().run_id;
  for (std::size_t index = 1; index < g_commands.size(); ++index) {
    const std::uint64_t next_run = g_commands[index].run_id;
    if (next_run == current_run) {
      continue;
    }
    completed_runs.insert(current_run);
    if (completed_runs.contains(next_run)) {
      return false;
    }
    current_run = next_run;
  }
  return !completed_runs.empty();
}

bool commandsMatchFixtureExpectations() {
  bool passed = true;
  const auto expect = [&](bool condition, std::string_view description) {
    check(condition, description);
    passed = passed && condition;
  };
  std::vector<RecordedCommand> actual;
  {
    std::scoped_lock lock(g_command_mutex);
    actual = g_commands;
  }
  const auto& expected = fixedrun_expectations::kCommands;
  expect(actual.size() == expected.size(),
         "fake Vulkan records exactly the typed command count");
  if (actual.size() != expected.size()) {
    return false;
  }

  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& wanted = expected[index];
    const auto& observed = actual[index];
    expect(observed.kind == wanted.kind,
           "fake Vulkan command kind matches the typed operation order");
    expect(
      observed.command_buffer_slot == fixedrun_expectations::kCommandBufferSlot,
      "Vulkan command uses the context-owned command buffer slot");
    if (observed.kind != wanted.kind) {
      continue;
    }
    switch (wanted.kind) {
      case fixedrun_expectations::CommandKind::Barrier:
        expect(observed.source_stage == wanted.source_stage &&
                 observed.destination_stage == wanted.destination_stage &&
                 observed.dependency_flags == 0 &&
                 observed.memory_barrier_count == 0 &&
                 observed.memory_barriers_null &&
                 observed.image_barrier_count == 0 &&
                 observed.image_barriers_null &&
                 observed.buffer_barriers_null == (wanted.barrier_count == 0) &&
                 observed.barriers.size() == wanted.barrier_count,
               "pipeline barrier stages and barrier counts match the plan");
        if (wanted.barrier_begin + wanted.barrier_count >
            fixedrun_expectations::kBarriers.size()) {
          expect(false, "typed barrier expectation indices are in range");
          break;
        }
        for (std::size_t barrier_index = 0;
             barrier_index < observed.barriers.size() &&
             barrier_index < wanted.barrier_count;
             ++barrier_index) {
          const auto& expected_barrier =
            fixedrun_expectations::kBarriers[wanted.barrier_begin +
                                             barrier_index];
          const auto& actual_barrier = observed.barriers[barrier_index];
          expect(
            actual_barrier.structure_type == expected_barrier.structure_type &&
              !actual_barrier.has_next && !expected_barrier.has_next &&
              actual_barrier.buffer_slot == expected_barrier.buffer_slot &&
              actual_barrier.source_access == expected_barrier.source_access &&
              actual_barrier.destination_access ==
                expected_barrier.destination_access &&
              actual_barrier.source_queue_family == VK_QUEUE_FAMILY_IGNORED &&
              actual_barrier.destination_queue_family ==
                VK_QUEUE_FAMILY_IGNORED &&
              actual_barrier.offset == expected_barrier.offset &&
              actual_barrier.size == expected_barrier.size,
            "buffer barrier IDs, access masks, and ranges match the plan");
        }
        break;
      case fixedrun_expectations::CommandKind::Copy:
        expect(observed.source_buffer == wanted.source_buffer &&
                 observed.destination_buffer == wanted.destination_buffer &&
                 !observed.copy_regions_null &&
                 observed.copy_region_count == wanted.copy_region_count &&
                 observed.copy_regions.size() == wanted.copy_region_count,
               "buffer copy slots and region count match the plan");
        if (observed.copy_regions.size() == wanted.copy_region_count &&
            wanted.copy_region_count == 1) {
          const auto& region = observed.copy_regions.front();
          expect(region.source_offset == wanted.source_offset &&
                   region.destination_offset == wanted.destination_offset &&
                   region.size == wanted.byte_size,
                 "buffer copy offsets and size match the plan");
        }
        break;
      case fixedrun_expectations::CommandKind::BindPipeline:
        expect(observed.pipeline_bind_point == VK_PIPELINE_BIND_POINT_COMPUTE &&
                 observed.pipeline_slot == wanted.pipeline_slot,
               "pipeline bind point and dense pipeline slot match the plan");
        break;
      case fixedrun_expectations::CommandKind::BindDescriptors:
        expect(
          observed.pipeline_slot == wanted.pipeline_slot &&
            observed.descriptor_bind_point == VK_PIPELINE_BIND_POINT_COMPUTE &&
            observed.first_set == wanted.first_set &&
            observed.set_count == wanted.set_count &&
            !observed.descriptor_sets_null &&
            observed.dynamic_offset_count == 0 &&
            observed.dynamic_offsets_null && observed.dynamic_offsets.empty() &&
            observed.descriptor_slots.size() == wanted.set_count,
          "descriptor set range and pipeline layout match the plan");
        if (observed.descriptor_slots.size() == wanted.set_count) {
          for (std::size_t set_index = 0;
               set_index < observed.descriptor_slots.size();
               ++set_index) {
            const std::uint32_t descriptor_slot =
              observed.descriptor_slots[set_index];
            expect(
              descriptor_slot == wanted.descriptor_first_slot + set_index &&
                descriptor_slot <
                  fixedrun_expectations::kDescriptorSetPipelineSlots.size() &&
                fixedrun_expectations::kDescriptorSetPipelineSlots
                    [descriptor_slot] == wanted.pipeline_slot,
              "descriptor handles map to the planned set slots");
          }
        }
        break;
      case fixedrun_expectations::CommandKind::PushConstants:
        expect(observed.pipeline_slot == wanted.pipeline_slot &&
                 !observed.push_data_null &&
                 observed.push_stage_flags == VK_SHADER_STAGE_COMPUTE_BIT &&
                 observed.push_offset == wanted.push_offset &&
                 observed.push_size == wanted.push_count &&
                 wanted.push_begin + wanted.push_count <=
                   fixedrun_expectations::kPushBytes.size() &&
                 observed.push_bytes.size() == wanted.push_count,
               "push constant stage, range, and size match the plan");
        if (observed.push_bytes.size() == wanted.push_count &&
            wanted.push_begin + wanted.push_count <=
              fixedrun_expectations::kPushBytes.size()) {
          for (std::size_t byte = 0; byte < observed.push_bytes.size();
               ++byte) {
            expect(
              observed.push_bytes[byte] ==
                fixedrun_expectations::kPushBytes[wanted.push_begin + byte],
              "push constant bytes match the plan");
          }
        }
        break;
      case fixedrun_expectations::CommandKind::Dispatch:
        expect(observed.pipeline_slot == wanted.pipeline_slot &&
                 observed.groups == wanted.groups,
               "dispatch pipeline and group counts match the plan");
        break;
    }
  }

  struct DescriptorTuple final {
    std::uint32_t structure_type;
    bool has_next;
    std::uint32_t descriptor_slot;
    std::uint32_t binding;
    std::uint32_t array_element;
    std::uint32_t descriptor_count;
    std::uint32_t descriptor_type;
    bool has_image_info;
    bool has_buffer_info;
    bool has_texel_buffer_view;
    std::uint32_t buffer_slot;
    std::uint64_t offset;
    std::uint64_t range;
  };
  std::vector<DescriptorTuple> actual_writes;
  for (const auto& write : fake_vulkan::descriptor_writes()) {
    const auto descriptor_slot = fake_vulkan::descriptor_set_slot(write.set);
    const auto buffer_slot = fake_vulkan::buffer_slot(write.buffer);
    actual_writes.push_back(
      {.structure_type = write.structure_type,
       .has_next = write.has_next,
       .descriptor_slot = descriptor_slot.value_or(UINT32_MAX),
       .binding = write.binding,
       .array_element = write.array_element,
       .descriptor_count = write.descriptor_count,
       .descriptor_type = write.descriptor_type,
       .has_image_info = write.has_image_info,
       .has_buffer_info = write.has_buffer_info,
       .has_texel_buffer_view = write.has_texel_buffer_view,
       .buffer_slot = buffer_slot.value_or(UINT32_MAX),
       .offset = write.offset,
       .range = write.range});
  }
  std::ranges::sort(actual_writes, [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.descriptor_slot, lhs.binding, lhs.array_element) <
           std::tie(rhs.descriptor_slot, rhs.binding, rhs.array_element);
  });
  std::vector<fixedrun_expectations::ExpectedDescriptorWrite> expected_writes(
    fixedrun_expectations::kDescriptorWrites.begin(),
    fixedrun_expectations::kDescriptorWrites.end());
  std::ranges::sort(expected_writes, [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.descriptor_slot, lhs.binding, lhs.array_element) <
           std::tie(rhs.descriptor_slot, rhs.binding, rhs.array_element);
  });
  expect(fake_vulkan::descriptor_copy_count() ==
           fixedrun_expectations::kDescriptorCopyCount,
         "descriptor update copy arguments match the typed fixture");
  expect(actual_writes.size() == expected_writes.size(),
         "fake Vulkan observes every typed descriptor write");
  if (actual_writes.size() == expected_writes.size()) {
    for (std::size_t index = 0; index < actual_writes.size(); ++index) {
      const auto& lhs = actual_writes[index];
      const auto& rhs = expected_writes[index];
      expect(
        lhs.structure_type == rhs.structure_type && !lhs.has_next &&
          !rhs.has_next && lhs.descriptor_slot == rhs.descriptor_slot &&
          lhs.binding == rhs.binding &&
          lhs.array_element == rhs.array_element &&
          lhs.descriptor_count == rhs.descriptor_count &&
          lhs.descriptor_type == rhs.descriptor_type && !lhs.has_image_info &&
          !rhs.has_image_info && lhs.has_buffer_info && rhs.has_buffer_info &&
          !lhs.has_texel_buffer_view && !rhs.has_texel_buffer_view &&
          lhs.buffer_slot == rhs.buffer_slot && lhs.offset == rhs.offset &&
          lhs.range == rhs.range,
        "descriptor writes match typed set/binding/buffer/range expectations");
    }
  }

  expect(fixedrun_expectations::kLiteralResourceRead == 0 &&
           fixedrun_expectations::kLiteralResourceWrite == 1,
         "independent descriptor access enum literals are fixed");
  const std::array<std::uint32_t, 6> expected_resource_accesses{
    0, 0, 1, 0, 1, 1};
  expect(expected_writes.size() == expected_resource_accesses.size(),
         "typed descriptor access expectations cover every fixture binding");
  if (expected_writes.size() == expected_resource_accesses.size()) {
    for (std::size_t index = 0; index < expected_resource_accesses.size();
         ++index) {
      expect(expected_writes[index].resource_access ==
               expected_resource_accesses[index],
             "fixture resource read/write access matches literal plan intent");
    }
  }
  expect(fixedrun_expectations::kBufferResourceIds ==
           std::array<std::uint32_t, 9>{7, 103, 307, 41, 59, 7, 41, 59, 103},
         "fake buffer creation order matches typed dense resource slots");
  expect(fixedrun_expectations::kPipelineIds ==
           std::array<std::uint32_t, 2>{101, 909},
         "fake pipeline creation order matches typed kernel slots");
  expect(fixedrun_expectations::kDescriptorSetPipelineSlots ==
           std::array<std::uint32_t, 5>{0, 0, 1, 1, 1},
         "fake descriptor set creation order matches pipeline ownership");
  expect(
    fixedrun_expectations::kLiteralHostStage == VK_PIPELINE_STAGE_HOST_BIT &&
      fixedrun_expectations::kLiteralTransferStage ==
        VK_PIPELINE_STAGE_TRANSFER_BIT &&
      fixedrun_expectations::kLiteralComputeStage ==
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    "independent stage mapping matches Vulkan literal flags");
  expect(
    fixedrun_expectations::kLiteralHostWriteAccess ==
        VK_ACCESS_HOST_WRITE_BIT &&
      fixedrun_expectations::kLiteralTransferReadAccess ==
        VK_ACCESS_TRANSFER_READ_BIT &&
      fixedrun_expectations::kLiteralTransferWriteAccess ==
        VK_ACCESS_TRANSFER_WRITE_BIT &&
      fixedrun_expectations::kLiteralShaderReadAccess ==
        VK_ACCESS_SHADER_READ_BIT &&
      fixedrun_expectations::kLiteralShaderWriteAccess ==
        VK_ACCESS_SHADER_WRITE_BIT &&
      fixedrun_expectations::kLiteralShaderReadWriteAccess ==
        (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) &&
      fixedrun_expectations::kLiteralHostReadAccess == VK_ACCESS_HOST_READ_BIT,
    "independent access mapping matches Vulkan literal flags");
  return passed;
}

bool noLiveObjects() {
  constexpr std::array<std::string_view, 14> object_types{
    "instance",
    "debug_messenger",
    "device",
    "buffer",
    "memory",
    "command_pool",
    "command_buffer",
    "fence",
    "descriptor_pool",
    "descriptor_set",
    "descriptor_set_layout",
    "shader_module",
    "pipeline_layout",
    "pipeline",
  };
  for (const auto type : object_types) {
    if (fake_vulkan::live_count(type) != 0) {
      return false;
    }
  }
  return true;
}

va4_fixedrun_vk_context* createContext() {
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  va4_fixedrun_vk_context* context = nullptr;
  const auto status = va4_fixedrun_vk_create(&options, &context, nullptr, 0);
  check(status == NCNN_VK_STATUS_SUCCESS && context != nullptr,
        "fake fixed-run candidate creates a context");
  return context;
}

ncnn_vk_status run(va4_fixedrun_vk_context* context,
                   const std::array<float, 17>& input,
                   std::array<float, 17>& first,
                   std::array<float, 17>& second) {
  return va4_fixedrun_vk_run(context,
                             input.data(),
                             input.size(),
                             first.data(),
                             first.size(),
                             second.data(),
                             second.size(),
                             nullptr,
                             0);
}

ncnn_vk_status runWithDiagnostic(va4_fixedrun_vk_context* context,
                                 const std::array<float, 17>& input,
                                 std::array<float, 17>& first,
                                 std::array<float, 17>& second,
                                 std::array<char, 256>& diagnostic) {
  return va4_fixedrun_vk_run(context,
                             input.data(),
                             input.size(),
                             first.data(),
                             first.size(),
                             second.data(),
                             second.size(),
                             diagnostic.data(),
                             diagnostic.size());
}

struct HookFailure final {
  HookFailure(VulkanDeviceOperation operation_value,
              std::uint32_t fail_on_call_value,
              ncnn_vk_status status_value)
    : operation(operation_value),
      fail_on_call(fail_on_call_value),
      status(status_value),
      call_count(0),
      armed(false) {}

  VulkanDeviceOperation operation;
  std::uint32_t fail_on_call;
  ncnn_vk_status status;
  std::atomic<std::uint32_t> call_count;
  std::atomic<bool> armed;
};

std::shared_ptr<HookFailure> installHookFailure(VulkanDeviceOperation operation,
                                                std::uint32_t fail_on_call,
                                                ncnn_vk_status status) {
  auto failure = std::make_shared<HookFailure>(operation, fail_on_call, status);
  g_settings.operation_hook =
    [failure](VulkanDeviceOperation current) -> std::optional<ncnn_vk_status> {
    if (failure->armed.load() && current == failure->operation &&
        failure->call_count.fetch_add(1) + 1 == failure->fail_on_call) {
      return failure->status;
    }
    return std::nullopt;
  };
  return failure;
}

void clearHook() {
  g_settings.operation_hook = {};
}

void fillOutputs(std::array<float, 17>& first, std::array<float, 17>& second) {
  first.fill(123.25F);
  second.fill(-456.5F);
}

bool sentinelsIntact(const std::array<float, 17>& first,
                     const std::array<float, 17>& second) {
  return std::ranges::all_of(first,
                             [](float value) { return value == 123.25F; }) &&
         std::ranges::all_of(second,
                             [](float value) { return value == -456.5F; });
}

void testArgumentFailuresRemainRecoverable() {
  fake_vulkan::reset();
  clearHook();
  auto* context = createContext();
  if (context == nullptr) {
    return;
  }
  g_copy_command_count.store(0);
  g_dispatch_count.store(0);
  std::array<float, 17> input{};
  std::array<float, 17> first{};
  std::array<float, 17> second{};
  fillOutputs(first, second);
  const auto null_status = va4_fixedrun_vk_run(context,
                                               nullptr,
                                               input.size(),
                                               first.data(),
                                               first.size(),
                                               second.data(),
                                               second.size(),
                                               nullptr,
                                               0);
  check(null_status == NCNN_VK_STATUS_INVALID_ARGUMENT,
        "null input is rejected before execution");
  const auto count_status = va4_fixedrun_vk_run(context,
                                                input.data(),
                                                input.size() - 1,
                                                first.data(),
                                                first.size(),
                                                second.data(),
                                                second.size(),
                                                nullptr,
                                                0);
  check(count_status == NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
        "wrong static element count is rejected before execution");
  const auto run_status = run(context, input, first, second);
  check(run_status == NCNN_VK_STATUS_SUCCESS,
        "corrected arguments can run on the same context");
  check(g_dispatch_count.load() == 2,
        "fake dispatch is recorded by the test wrapper");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_SUCCESS,
        "argument-validation context destroys");
  check(noLiveObjects(), "argument-validation context releases all objects");
}

struct HookFailureCase final {
  VulkanDeviceOperation operation;
  std::uint32_t call;
  ncnn_vk_status status;
  std::string_view name;
};

void runHookFailureCase(const HookFailureCase& failure) {
  fake_vulkan::reset();
  clearHook();
  g_settings.force_mapped_memory_sync = true;
  auto hook_failure =
    installHookFailure(failure.operation, failure.call, failure.status);
  auto* context = createContext();
  if (context == nullptr) {
    clearHook();
    return;
  }
  hook_failure->armed.store(true);
  std::array<float, 17> input{};
  std::array<float, 17> first{};
  std::array<float, 17> second{};
  fillOutputs(first, second);
  const auto status = run(context, input, first, second);
  check(status == failure.status, failure.name);
  if (failure.operation == VulkanDeviceOperation::InvalidateMappedMemory) {
    check(fake_vulkan::call_count("vkInvalidateMappedMemoryRanges") == 1,
          "first output invalidate succeeds before second output failure");
  }
  check(sentinelsIntact(first, second),
        "execution failure leaves every output sentinel unchanged");
  clearHook();
  const auto poisoned_status = run(context, input, first, second);
  check(poisoned_status == failure.status,
        "execution failure poisons the affected context");
  check(sentinelsIntact(first, second),
        "poisoned context does not publish caller outputs");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_SUCCESS,
        "failed-run context destroys after injected failure");
  check(noLiveObjects(), "failed-run context releases all fake objects");

  fake_vulkan::reset();
  clearHook();
  auto* healthy = createContext();
  if (healthy != nullptr) {
    fillOutputs(first, second);
    check(run(healthy, input, first, second) == NCNN_VK_STATUS_SUCCESS,
          "a new context remains healthy after another context fails");
    check(va4_fixedrun_vk_destroy(healthy) == NCNN_VK_STATUS_SUCCESS,
          "healthy replacement context destroys");
  }
  check(noLiveObjects(), "replacement context releases all fake objects");
}

void runVulkanFailureCase(std::string_view function,
                          VkResult result,
                          ncnn_vk_status expected,
                          std::string_view name) {
  fake_vulkan::reset();
  clearHook();
  auto* context = createContext();
  if (context == nullptr) {
    return;
  }
  std::array<float, 17> input{};
  std::array<float, 17> first{};
  std::array<float, 17> second{};
  fillOutputs(first, second);
  fake_vulkan::fail_next(function, result);
  const auto status = run(context, input, first, second);
  check(status == expected, name);
  check(sentinelsIntact(first, second),
        "recording failure leaves all output sentinels unchanged");
  check(run(context, input, first, second) == expected,
        "Vulkan recording failure poisons the context");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_SUCCESS,
        "Vulkan recording failure context destroys");
  check(noLiveObjects(), "Vulkan recording failure cleans all objects");
}

void testExecutionFailuresAreAtomicAndPoison() {
  const std::array hook_failures{
    HookFailureCase{.operation = VulkanDeviceOperation::FlushMappedMemory,
                    .call = 1,
                    .status = NCNN_VK_STATUS_API_ERROR,
                    .name = "input flush failure"},
    HookFailureCase{.operation = VulkanDeviceOperation::Submit,
                    .call = 1,
                    .status = NCNN_VK_STATUS_API_ERROR,
                    .name = "submit failure"},
    HookFailureCase{.operation = VulkanDeviceOperation::Wait,
                    .call = 2,
                    .status = NCNN_VK_STATUS_DEVICE_LOST,
                    .name = "submitted fence wait device loss"},
    // The first output invalidation succeeds; the second fails before either
    // caller buffer is published.
    HookFailureCase{.operation = VulkanDeviceOperation::InvalidateMappedMemory,
                    .call = 2,
                    .status = NCNN_VK_STATUS_API_ERROR,
                    .name = "second output invalidate failure"},
  };
  for (const auto& failure : hook_failures) {
    runHookFailureCase(failure);
  }
  runVulkanFailureCase("vkResetCommandPool",
                       VK_ERROR_DEVICE_LOST,
                       NCNN_VK_STATUS_DEVICE_LOST,
                       "command-pool reset device loss is propagated");
  runVulkanFailureCase("vkBeginCommandBuffer",
                       VK_ERROR_UNKNOWN,
                       NCNN_VK_STATUS_API_ERROR,
                       "command-buffer begin failure is propagated");
  runVulkanFailureCase("vkEndCommandBuffer",
                       VK_ERROR_UNKNOWN,
                       NCNN_VK_STATUS_API_ERROR,
                       "command-buffer end failure is propagated");
}

void testSecondOutputMappingFailureIsAtomic() {
  fake_vulkan::reset();
  clearHook();
  fake_vulkan::return_null_mapping_on_call(3);
  auto* context = createContext();
  if (context == nullptr) {
    return;
  }
  std::array<float, 17> input{};
  std::array<float, 17> first{};
  std::array<float, 17> second{};
  fillOutputs(first, second);

  const auto status = run(context, input, first, second);
  check(status == NCNN_VK_STATUS_INVALID_ARGUMENT,
        "a null second-output mapping fails in VulkanBuffer::invalidate");
  check(fake_vulkan::call_count("vkMapMemory") == 4,
        "input, both outputs, and weight staging mappings are attempted");
  check(fake_vulkan::call_count("vkInvalidateMappedMemoryRanges") == 1,
        "first output invalidates before the second mapped output is rejected");
  check(sentinelsIntact(first, second),
        "second-output mapping failure preserves both caller buffers");
  const auto dispatches = g_dispatch_count.load();
  const auto copies = g_copy_command_count.load();
  const auto invalidations =
    fake_vulkan::call_count("vkInvalidateMappedMemoryRanges");
  const auto submissions = fake_vulkan::call_count("vkQueueSubmit");
  std::array<char, 256> diagnostic{};
  check(runWithDiagnostic(context, input, first, second, diagnostic) ==
          NCNN_VK_STATUS_INVALID_ARGUMENT,
        "a failed output mapping poisons the context");
  check(
    std::string_view(diagnostic.data()).find("previous execution failure") !=
      std::string_view::npos,
    "mapping failure retry reports the poisoned-context diagnostic");
  check(g_dispatch_count.load() == dispatches &&
          g_copy_command_count.load() == copies &&
          fake_vulkan::call_count("vkInvalidateMappedMemoryRanges") ==
            invalidations &&
          fake_vulkan::call_count("vkQueueSubmit") == submissions,
        "poison rejects mapping-failure retry before command submission");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_SUCCESS,
        "mapping-failure context destroys");
  check(noLiveObjects(), "mapping-failure context releases all objects");
}

void testValidationErrorAfterAllReadbacksIsAtomic() {
  fake_vulkan::reset();
  clearHook();
  g_settings.enable_validation = true;
  auto* context = createContext();
  g_settings.enable_validation = false;
  if (context == nullptr) {
    return;
  }
  fake_vulkan::inject_validation_error_on_invalidate_call(2);
  std::array<float, 17> input{};
  std::array<float, 17> first{};
  std::array<float, 17> second{};
  fillOutputs(first, second);

  const auto status = run(context, input, first, second);
  check(status == NCNN_VK_STATUS_API_ERROR,
        "a Vulkan validation callback error rejects the completed run");
  check(fake_vulkan::injected_validation_error_count() == 1,
        "the fake debug-messenger callback reports one validation error");
  check(
    fake_vulkan::call_count("vkInvalidateMappedMemoryRanges") == 2,
    "both output mappings invalidate before validation rejects publication");
  check(sentinelsIntact(first, second),
        "validation failure after readback preserves both caller buffers");
  const auto dispatches = g_dispatch_count.load();
  const auto copies = g_copy_command_count.load();
  const auto invalidations =
    fake_vulkan::call_count("vkInvalidateMappedMemoryRanges");
  const auto submissions = fake_vulkan::call_count("vkQueueSubmit");
  std::array<char, 256> diagnostic{};
  check(runWithDiagnostic(context, input, first, second, diagnostic) ==
          NCNN_VK_STATUS_API_ERROR,
        "validation failure poisons the context");
  check(
    std::string_view(diagnostic.data()).find("previous execution failure") !=
      std::string_view::npos,
    "validation failure retry reports the poisoned-context diagnostic");
  check(g_dispatch_count.load() == dispatches &&
          g_copy_command_count.load() == copies &&
          fake_vulkan::call_count("vkInvalidateMappedMemoryRanges") ==
            invalidations &&
          fake_vulkan::call_count("vkQueueSubmit") == submissions,
        "poison rejects validation-failure retry before command submission");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_API_ERROR,
        "validation error is reported during context destruction");
  check(noLiveObjects(), "validation-failure context releases all objects");
}

void testWeightUploadAndFixedDispatchCounts() {
  fake_vulkan::reset();
  clearHook();
  g_copy_command_count.store(0);
  g_dispatch_count.store(0);
  auto* context = createContext();
  if (context == nullptr) {
    return;
  }
  const auto create_copy_count = g_copy_command_count.load();
  check(context->get_weight_upload_count() == 1,
        "embedded weight is uploaded once during context creation");
  check(create_copy_count == 1, "weight creation emits one fixed upload copy");
  resetCommands();
  g_copy_command_count.store(0);
  g_dispatch_count.store(0);
  std::array<float, 17> input{};
  std::array<float, 17> first{};
  std::array<float, 17> second{};
  check(run(context, input, first, second) == NCNN_VK_STATUS_SUCCESS,
        "fixed-run candidate executes the synthetic graph");
  check(g_dispatch_count.load() == 2,
        "one run records both statically planned dispatches");
  check(g_copy_command_count.load() == 3,
        "one run records input and both output copies, not weight upload");
  check(commandsMatchFixtureExpectations(),
        "fake records every typed Vulkan command argument exactly");
  check(context->get_weight_upload_count() == 1,
        "run does not re-upload the embedded weight");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_SUCCESS,
        "fixed dispatch context destroys");
  check(noLiveObjects(), "fixed dispatch context releases all objects");
}

void testSameContextRunsAreSerialized() {
  fake_vulkan::reset();
  clearHook();
  auto* context = createContext();
  if (context == nullptr) {
    return;
  }

  constexpr std::uint32_t kConcurrentRounds = 24;
  std::barrier run_start(3);
  std::barrier run_finished(3);
  std::atomic<std::uint32_t> failed_runs{0};
  std::array<float, 17> input_a{};
  std::array<float, 17> input_b{};
  std::array<float, 17> first_a{};
  std::array<float, 17> second_a{};
  std::array<float, 17> first_b{};
  std::array<float, 17> second_b{};

  auto worker = [&](std::uint32_t worker_index) {
    auto& input = worker_index == 0 ? input_a : input_b;
    auto& first = worker_index == 0 ? first_a : first_b;
    auto& second = worker_index == 0 ? second_a : second_b;
    for (std::uint32_t round = 0; round < kConcurrentRounds; ++round) {
      input.fill(static_cast<float>((round * 2) + worker_index + 1));
      g_run_id = (static_cast<std::uint64_t>(round) * 2) + worker_index + 1;
      run_start.arrive_and_wait();
      if (run(context, input, first, second) != NCNN_VK_STATUS_SUCCESS) {
        failed_runs.fetch_add(1);
      }
      run_finished.arrive_and_wait();
    }
  };

  g_copy_command_count.store(0);
  g_dispatch_count.store(0);
  resetCommands();
  std::thread first_thread(worker, 0);
  std::thread second_thread(worker, 1);
  for (std::uint32_t round = 0; round < kConcurrentRounds; ++round) {
    run_start.arrive_and_wait();
    run_finished.arrive_and_wait();
  }
  first_thread.join();
  second_thread.join();
  check(failed_runs.load() == 0,
        "barrier-contended same-context calls all complete successfully");
  check(g_dispatch_count.load() == 4 * kConcurrentRounds &&
          g_copy_command_count.load() == 6 * kConcurrentRounds,
        "repeated contending calls each record their full command sequence");
  check(commandsAreGroupedByRun(),
        "command recording for same-context runs is not interleaved");
  check(va4_fixedrun_vk_destroy(context) == NCNN_VK_STATUS_SUCCESS,
        "concurrent-run context destroys after both callers join");
  check(noLiveObjects(), "concurrent-run context releases all fake objects");
}

void testIndependentContextsDoNotSharePoisonState() {
  fake_vulkan::reset();
  clearHook();
  auto hook_failure = installHookFailure(
    VulkanDeviceOperation::FlushMappedMemory, 1, NCNN_VK_STATUS_API_ERROR);
  auto* failing = createContext();
  auto* healthy = createContext();
  if (failing == nullptr || healthy == nullptr) {
    if (failing != nullptr) {
      (void)va4_fixedrun_vk_destroy(failing);
    }
    if (healthy != nullptr) {
      (void)va4_fixedrun_vk_destroy(healthy);
    }
    clearHook();
    return;
  }
  hook_failure->armed.store(true);
  std::array<float, 17> input{};
  std::array<float, 17> failed_a{};
  std::array<float, 17> failed_b{};
  std::array<float, 17> healthy_a{};
  std::array<float, 17> healthy_b{};
  fillOutputs(failed_a, failed_b);
  check(run(failing, input, failed_a, failed_b) == NCNN_VK_STATUS_API_ERROR,
        "injected failure poisons only its owning context");
  clearHook();
  check(run(healthy, input, healthy_a, healthy_b) == NCNN_VK_STATUS_SUCCESS,
        "independent context remains executable after neighboring failure");
  check(run(failing, input, failed_a, failed_b) == NCNN_VK_STATUS_API_ERROR,
        "poison remains local while contexts are alternated");
  check(run(healthy, input, healthy_a, healthy_b) == NCNN_VK_STATUS_SUCCESS,
        "interleaved healthy context remains executable");
  check(va4_fixedrun_vk_destroy(failing) == NCNN_VK_STATUS_SUCCESS,
        "poisoned context destroys independently");
  check(va4_fixedrun_vk_destroy(healthy) == NCNN_VK_STATUS_SUCCESS,
        "healthy context destroys independently");
  check(noLiveObjects(), "independent contexts release all fake objects");
}

}  // namespace

namespace ncnn_mlir::vulkan::test {

LifecycleTestSettings getLifecycleTestSettings() {
  return g_settings;
}

}  // namespace ncnn_mlir::vulkan::test

extern "C" VKAPI_ATTR void VKAPI_CALL
__wrap_vkCmdCopyBuffer(VkCommandBuffer command_buffer,
                       VkBuffer source,
                       VkBuffer destination,
                       std::uint32_t region_count,
                       const VkBufferCopy* regions) {
  RecordedCommand command{};
  command.kind = fixedrun_expectations::CommandKind::Copy;
  command.command_buffer_slot =
    fake_vulkan::command_buffer_slot(command_buffer).value_or(UINT32_MAX);
  command.source_buffer = fake_vulkan::buffer_slot(source).value_or(UINT32_MAX);
  command.destination_buffer =
    fake_vulkan::buffer_slot(destination).value_or(UINT32_MAX);
  command.copy_region_count = region_count;
  command.copy_regions_null = regions == nullptr;
  if (regions != nullptr) {
    for (std::uint32_t index = 0; index < region_count; ++index) {
      command.copy_regions.push_back(
        {.source_offset = regions[index].srcOffset,
         .destination_offset = regions[index].dstOffset,
         .size = regions[index].size});
    }
  }
  ++g_copy_command_count;
  recordCommand(std::move(command));
}

extern "C" VKAPI_ATTR void VKAPI_CALL
__wrap_vkCmdBindPipeline(VkCommandBuffer command_buffer,
                         VkPipelineBindPoint bind_point,
                         VkPipeline pipeline) {
  g_bound_pipeline_slot =
    fake_vulkan::pipeline_slot(pipeline).value_or(UINT32_MAX);
  RecordedCommand command{};
  command.kind = fixedrun_expectations::CommandKind::BindPipeline;
  command.command_buffer_slot =
    fake_vulkan::command_buffer_slot(command_buffer).value_or(UINT32_MAX);
  command.pipeline_bind_point = bind_point;
  command.pipeline_slot = g_bound_pipeline_slot;
  recordCommand(std::move(command));
}

extern "C" VKAPI_ATTR void VKAPI_CALL
__wrap_vkCmdBindDescriptorSets(VkCommandBuffer command_buffer,
                               VkPipelineBindPoint bind_point,
                               VkPipelineLayout layout,
                               std::uint32_t first_set,
                               std::uint32_t descriptor_set_count,
                               const VkDescriptorSet* descriptor_sets,
                               std::uint32_t dynamic_offset_count,
                               const std::uint32_t* dynamic_offsets) {
  RecordedCommand command{};
  command.kind = fixedrun_expectations::CommandKind::BindDescriptors;
  command.command_buffer_slot =
    fake_vulkan::command_buffer_slot(command_buffer).value_or(UINT32_MAX);
  command.descriptor_bind_point = bind_point;
  command.pipeline_slot =
    fake_vulkan::pipeline_layout_slot(layout).value_or(UINT32_MAX);
  command.first_set = first_set;
  command.set_count = descriptor_set_count;
  command.descriptor_sets_null = descriptor_sets == nullptr;
  command.dynamic_offset_count = dynamic_offset_count;
  command.dynamic_offsets_null = dynamic_offsets == nullptr;
  if (dynamic_offsets != nullptr) {
    command.dynamic_offsets.assign(dynamic_offsets,
                                   dynamic_offsets + dynamic_offset_count);
  }
  if (descriptor_sets != nullptr) {
    for (std::uint32_t index = 0; index < descriptor_set_count; ++index) {
      command.descriptor_slots.push_back(
        fake_vulkan::descriptor_set_slot(descriptor_sets[index])
          .value_or(UINT32_MAX));
    }
  }
  recordCommand(std::move(command));
}

extern "C" VKAPI_ATTR void VKAPI_CALL
__wrap_vkCmdPushConstants(VkCommandBuffer command_buffer,
                          VkPipelineLayout layout,
                          VkShaderStageFlags stage_flags,
                          std::uint32_t offset,
                          std::uint32_t size,
                          const void* values) {
  RecordedCommand command{};
  command.kind = fixedrun_expectations::CommandKind::PushConstants;
  command.command_buffer_slot =
    fake_vulkan::command_buffer_slot(command_buffer).value_or(UINT32_MAX);
  command.pipeline_slot =
    fake_vulkan::pipeline_layout_slot(layout).value_or(UINT32_MAX);
  command.push_stage_flags = stage_flags;
  command.push_offset = offset;
  command.push_size = size;
  command.push_data_null = values == nullptr;
  if (values != nullptr) {
    const auto* bytes = static_cast<const std::uint8_t*>(values);
    command.push_bytes.assign(bytes, bytes + size);
  }
  recordCommand(std::move(command));
}

extern "C" VKAPI_ATTR void VKAPI_CALL
__wrap_vkCmdDispatch(VkCommandBuffer command_buffer,
                     std::uint32_t group_count_x,
                     std::uint32_t group_count_y,
                     std::uint32_t group_count_z) {
  ++g_dispatch_count;
  RecordedCommand command{};
  command.kind = fixedrun_expectations::CommandKind::Dispatch;
  command.command_buffer_slot =
    fake_vulkan::command_buffer_slot(command_buffer).value_or(UINT32_MAX);
  command.pipeline_slot = g_bound_pipeline_slot;
  command.groups = {group_count_x, group_count_y, group_count_z};
  recordCommand(std::move(command));
}

extern "C" VKAPI_ATTR void VKAPI_CALL
__wrap_vkCmdPipelineBarrier(VkCommandBuffer command_buffer,
                            VkPipelineStageFlags source_stage,
                            VkPipelineStageFlags destination_stage,
                            VkDependencyFlags dependencies,
                            std::uint32_t memory_barrier_count,
                            const VkMemoryBarrier* memory_barriers,
                            std::uint32_t buffer_barrier_count,
                            const VkBufferMemoryBarrier* buffer_barriers,
                            std::uint32_t image_barrier_count,
                            const VkImageMemoryBarrier* image_barriers) {
  RecordedCommand command{};
  command.kind = fixedrun_expectations::CommandKind::Barrier;
  command.command_buffer_slot =
    fake_vulkan::command_buffer_slot(command_buffer).value_or(UINT32_MAX);
  command.source_stage = source_stage;
  command.destination_stage = destination_stage;
  command.dependency_flags = dependencies;
  command.memory_barrier_count = memory_barrier_count;
  command.image_barrier_count = image_barrier_count;
  command.memory_barriers_null = memory_barriers == nullptr;
  command.buffer_barriers_null = buffer_barriers == nullptr;
  command.image_barriers_null = image_barriers == nullptr;
  if (buffer_barriers != nullptr) {
    for (std::uint32_t index = 0; index < buffer_barrier_count; ++index) {
      const auto& barrier = buffer_barriers[index];
      command.barriers.push_back(
        {barrier.sType,
         barrier.pNext != nullptr,
         fake_vulkan::buffer_slot(barrier.buffer).value_or(UINT32_MAX),
         barrier.srcAccessMask,
         barrier.dstAccessMask,
         barrier.srcQueueFamilyIndex,
         barrier.dstQueueFamilyIndex,
         barrier.offset,
         barrier.size});
    }
  }
  recordCommand(std::move(command));
}

int main() {
  try {
    fake_vulkan::reset();
    testArgumentFailuresRemainRecoverable();
    testExecutionFailuresAreAtomicAndPoison();
    testSecondOutputMappingFailureIsAtomic();
    testValidationErrorAfterAllReadbacksIsAtomic();
    testWeightUploadAndFixedDispatchCounts();
    testSameContextRunsAreSerialized();
    testIndependentContextsDoNotSharePoisonState();
  } catch (const std::exception& error) {
    std::cerr << "fixed-run fake test threw: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  if (!g_failed_check.empty()) {
    std::cerr << "fixed-run fake test failed: " << g_failed_check << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "Vulkan fixed-run fake tests passed\n";
  return EXIT_SUCCESS;
}
