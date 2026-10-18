#include "lifecycle_test_hooks.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "candidate_source.cpp"
#include "fakeVulkan/fake_vulkan.hpp"
#include "lifecycle_fixture_vulkan.h"
#include "ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"
#include "ncnn-mlir/Vulkan/VulkanDeviceTypes.hpp"

namespace {

using ncnn_mlir::vulkan::VulkanDeviceOperation;
using ncnn_mlir::vulkan::test::LifecycleTestSettings;

LifecycleTestSettings g_settings{
  .device_name_match = "",
  .enable_validation = false,
  .force_mapped_memory_sync = false,
  .max_context_memory_bytes = 512U * 1024U * 1024U,
  .operation_hook = {}};
constexpr VkDeviceSize kExpectedWeightUploadBytes = kWeightBytes0.size();

struct CopyCommandRecord final {
  bool source_non_null;
  bool destination_non_null;
  bool source_destination_distinct;
  std::uint32_t region_count;
  std::vector<VkBufferCopy> regions;
};

std::uint64_t g_copy_command_count = 0;
std::vector<CopyCommandRecord> g_copy_commands;
int g_failures = 0;

void reset_copy_recording() {
  g_copy_command_count = 0;
  g_copy_commands.clear();
}

bool copy_arguments_valid(const std::vector<CopyCommandRecord>& records,
                          std::uint64_t call_count) {
  if (call_count != 1 || records.size() != 1) {
    return false;
  }
  const CopyCommandRecord& record = records.front();
  if (!record.source_non_null || !record.destination_non_null ||
      !record.source_destination_distinct || record.region_count != 1 ||
      record.regions.size() != record.region_count) {
    return false;
  }
  const VkBufferCopy& region = record.regions.front();
  return region.srcOffset == 0 && region.dstOffset == 0 &&
         region.size == kExpectedWeightUploadBytes;
}

void expect(bool condition, std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++g_failures;
  }
}

bool no_live_objects() {
  constexpr std::array<std::string_view, 13> kObjectTypes{
    "instance",
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
  for (const std::string_view type : kObjectTypes) {
    if (fake_vulkan::live_count(type) != 0) {
      return false;
    }
  }
  return true;
}

struct FailureObservation final {
  std::string_view function;
  VkResult result;
  ncnn_vk_status status;
  bool cleaned;
};

FailureObservation inject_create_failure(std::string_view function,
                                         VkResult result,
                                         ncnn_vk_status expected_status) {
  fake_vulkan::reset();
  reset_copy_recording();
  fake_vulkan::fail_next(function, result);
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  lifecycle_fixture_vk_context* context = nullptr;
  const ncnn_vk_status status =
    lifecycle_fixture_vk_create(&options, &context, nullptr, 0);
  if (context != nullptr) {
    (void)lifecycle_fixture_vk_destroy(context);
  }
  const bool cleaned = no_live_objects();
  expect(status == expected_status,
         "injected Vulkan create failure preserves its ABI status");
  expect(context == nullptr,
         "failed candidate create does not return a partial context");
  expect(cleaned, "partial candidate create releases all fake Vulkan objects");
  return FailureObservation{.function = function,
                            .result = result,
                            .status = status,
                            .cleaned = cleaned};
}

bool test_budget_preflight(ncnn_vk_status* status_out,
                           std::uint64_t* allocation_count_out) {
  fake_vulkan::reset();
  reset_copy_recording();
  g_settings.max_context_memory_bytes = 1;
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  lifecycle_fixture_vk_context* context = nullptr;
  *status_out = lifecycle_fixture_vk_create(&options, &context, nullptr, 0);
  *allocation_count_out = fake_vulkan::call_count("vkAllocateMemory");
  const bool cleaned = no_live_objects();
  g_settings.max_context_memory_bytes = 512U * 1024U * 1024U;
  expect(*status_out == NCNN_VK_STATUS_OUT_OF_MEMORY,
         "actual requirements above the configured budget fail create");
  expect(*allocation_count_out == 0,
         "actual budget rejection happens before any memory allocation");
  expect(context == nullptr, "budget rejection returns no candidate context");
  expect(cleaned, "budget rejection releases every prepared object");
  return cleaned;
}

bool test_weighted_success(std::uint64_t* upload_count_out,
                           std::uint64_t* copy_count_out,
                           std::vector<CopyCommandRecord>* copy_records_out,
                           bool* copy_arguments_valid_out,
                           std::uint64_t* live_buffers_out,
                           std::uint64_t* released_staging_out,
                           std::uint64_t* peak_memory_out) {
  fake_vulkan::reset();
  reset_copy_recording();
  g_settings.force_mapped_memory_sync = true;
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  lifecycle_fixture_vk_context* context = nullptr;
  const ncnn_vk_status create_status =
    lifecycle_fixture_vk_create(&options, &context, nullptr, 0);
  expect(create_status == NCNN_VK_STATUS_SUCCESS && context != nullptr,
         "weighted candidate initializes against fake Vulkan");
  if (create_status != NCNN_VK_STATUS_SUCCESS || context == nullptr) {
    g_settings.force_mapped_memory_sync = false;
    return false;
  }

  *upload_count_out = context->get_weight_upload_count();
  *copy_count_out = g_copy_command_count;
  *copy_records_out = g_copy_commands;
  *copy_arguments_valid_out =
    copy_arguments_valid(g_copy_commands, g_copy_command_count);
  *live_buffers_out = fake_vulkan::live_count("buffer");
  *released_staging_out = fake_vulkan::call_count("vkDestroyBuffer");
  *peak_memory_out = context->get_peak_memory_bytes();
  const std::uint64_t flush_count =
    fake_vulkan::call_count("vkFlushMappedMemoryRanges");
  expect(*upload_count_out == 1,
         "the weighted fixture uploads its constant exactly once per context");
  expect(*copy_arguments_valid_out,
         "weight upload records one bounded buffer copy with the exact payload "
         "range");
  expect(*live_buffers_out == 5,
         "create-upload staging buffer is released after upload");
  expect(*released_staging_out >= 1,
         "staging buffer is destroyed before create returns");
  expect(flush_count >= 1,
         "forced non-coherent weight staging is flushed before submission");
  expect(*peak_memory_out > 0,
         "context records actual allocation requirements as peak memory");

  const ncnn_vk_status destroy_status = lifecycle_fixture_vk_destroy(context);
  expect(destroy_status == NCNN_VK_STATUS_SUCCESS,
         "weighted candidate destroys successfully");
  expect(no_live_objects(),
         "weighted success releases all fake Vulkan objects");
  g_settings.force_mapped_memory_sync = false;
  return no_live_objects();
}

bool test_wait_idle_retry(ncnn_vk_status* destroy_status_out,
                          std::uint64_t* wait_count_out,
                          bool* no_early_destroy_out) {
  fake_vulkan::reset();
  reset_copy_recording();
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  lifecycle_fixture_vk_context* context = nullptr;
  const ncnn_vk_status create_status =
    lifecycle_fixture_vk_create(&options, &context, nullptr, 0);
  expect(create_status == NCNN_VK_STATUS_SUCCESS && context != nullptr,
         "candidate initializes before a transient idle-wait failure");
  if (create_status != NCNN_VK_STATUS_SUCCESS || context == nullptr) {
    return false;
  }
  fake_vulkan::fail_next("vkDeviceWaitIdle", VK_ERROR_UNKNOWN);
  *destroy_status_out = lifecycle_fixture_vk_destroy(context);
  *wait_count_out = fake_vulkan::call_count("vkDeviceWaitIdle");
  *no_early_destroy_out =
    !fake_vulkan::saw_destruction_before_wait_resolution();
  expect(*destroy_status_out == NCNN_VK_STATUS_API_ERROR,
         "transient idle failure is preserved after safe cleanup");
  expect(*wait_count_out >= 2,
         "idle wait retries until the device reports completion");
  expect(*no_early_destroy_out,
         "no device object is released before wait completion is confirmed");
  expect(no_live_objects(), "retried idle wait ends with complete cleanup");
  return no_live_objects();
}

bool write_result(const std::filesystem::path& path,
                  ncnn_vk_status budget_status,
                  std::uint64_t allocations_after_budget,
                  bool budget_cleanup,
                  const std::vector<FailureObservation>& failures,
                  ncnn_vk_status operation_wait_status,
                  std::uint64_t operation_wait_copy_count,
                  bool operation_wait_copy_arguments_valid,
                  bool operation_wait_cleanup,
                  std::uint64_t upload_count,
                  std::uint64_t copy_count,
                  const std::vector<CopyCommandRecord>& weighted_copy_records,
                  bool weighted_copy_arguments_valid,
                  std::uint64_t live_buffers,
                  std::uint64_t released_staging,
                  std::uint64_t peak_memory,
                  bool weighted_cleanup,
                  ncnn_vk_status wait_destroy_status,
                  std::uint64_t wait_count,
                  bool no_early_destroy,
                  bool wait_cleanup) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"kind\": \"ncnn.vulkan_lifecycle_fake_result\",\n"
         << R"(  "status": ")" << (g_failures == 0 ? "passed" : "failed")
         << "\",\n"
         << "  \"candidate_execution\": \"not_run\",\n"
         << "  \"budget_preflight\": {\n"
         << "    \"status\": " << budget_status << ",\n"
         << "    \"allocations_after_rejection\": " << allocations_after_budget
         << ",\n"
         << "    \"cleanup_all_objects_zero\": "
         << (budget_cleanup ? "true" : "false") << "\n"
         << "  },\n"
         << "  \"partial_create_failures\": [\n";
  for (std::size_t index = 0; index < failures.size(); ++index) {
    const FailureObservation& failure = failures[index];
    output << R"(    {"function": ")" << failure.function
           << R"(", "vk_result": )" << failure.result << R"(, "status": )"
           << failure.status << R"(, "cleanup_all_objects_zero": )"
           << (failure.cleaned ? "true" : "false") << "}"
           << (index + 1 == failures.size() ? "\n" : ",\n");
  }
  output << "  ],\n"
         << "  \"operation_wait_failure\": {\n"
         << "    \"status\": " << operation_wait_status << ",\n"
         << "    \"vkCmdCopyBuffer_call_count\": " << operation_wait_copy_count
         << ",\n"
         << "    \"copy_arguments_valid\": "
         << (operation_wait_copy_arguments_valid ? "true" : "false") << ",\n"
         << "    \"cleanup_all_objects_zero\": "
         << (operation_wait_cleanup ? "true" : "false") << "\n"
         << "  },\n"
         << "  \"weighted_upload\": {\n"
         << "    \"observed_weight_upload_count\": " << upload_count << ",\n"
         << "    \"expected_weight_upload_count\": 1,\n"
         << "    \"vkCmdCopyBuffer_call_count\": " << copy_count << ",\n"
         << "    \"expected_vkCmdCopyBuffer_call_count\": 1,\n"
         << "    \"copy_arguments_valid\": "
         << (weighted_copy_arguments_valid ? "true" : "false") << ",\n"
         << "    \"source_non_null\": "
         << (weighted_copy_records.size() == 1 &&
                 weighted_copy_records.front().source_non_null
               ? "true"
               : "false")
         << ",\n"
         << "    \"destination_non_null\": "
         << (weighted_copy_records.size() == 1 &&
                 weighted_copy_records.front().destination_non_null
               ? "true"
               : "false")
         << ",\n"
         << "    \"source_destination_distinct\": "
         << (weighted_copy_records.size() == 1 &&
                 weighted_copy_records.front().source_destination_distinct
               ? "true"
               : "false")
         << ",\n"
         << "    \"region_count\": "
         << (weighted_copy_records.size() == 1
               ? weighted_copy_records.front().region_count
               : 0)
         << ",\n"
         << "    \"recorded_region_count\": "
         << (weighted_copy_records.size() == 1
               ? weighted_copy_records.front().regions.size()
               : 0)
         << ",\n"
         << "    \"records\": [";
  if (weighted_copy_records.size() == 1) {
    const auto& regions = weighted_copy_records.front().regions;
    for (std::size_t index = 0; index < regions.size(); ++index) {
      const VkBufferCopy& region = regions[index];
      output << (index == 0 ? "" : ",")
             << "{\"src_offset\": " << region.srcOffset
             << ", \"dst_offset\": " << region.dstOffset
             << ", \"size\": " << region.size << "}";
    }
  }
  output
    << "],\n"
    << "    \"device_transfer\": \"not_executed_fake_command_recording\",\n"
    << "    \"live_buffer_count_after_create\": " << live_buffers << ",\n"
    << "    \"staging_buffers_destroyed_before_create_return\": "
    << released_staging << ",\n"
    << "    \"peak_memory_bytes\": " << peak_memory << ",\n"
    << "    \"cleanup_all_objects_zero\": "
    << (weighted_cleanup ? "true" : "false") << "\n"
    << "  },\n"
    << "  \"wait_idle_retry\": {\n"
    << "    \"destroy_status\": " << wait_destroy_status << ",\n"
    << "    \"wait_count\": " << wait_count << ",\n"
    << "    \"destruction_before_wait_resolution\": "
    << (no_early_destroy ? "false" : "true") << ",\n"
    << "    \"cleanup_all_objects_zero\": " << (wait_cleanup ? "true" : "false")
    << "\n"
    << "  }\n"
    << "}\n";
  return static_cast<bool>(output);
}

std::filesystem::path parse_result_path(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--result") {
    throw std::runtime_error(
      "usage: vulkan_lifecycle_fake_unit_test --result PATH");
  }
  return argv[2];
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
  (void)command_buffer;
  ++g_copy_command_count;
  CopyCommandRecord record{
    .source_non_null = source != VK_NULL_HANDLE,
    .destination_non_null = destination != VK_NULL_HANDLE,
    .source_destination_distinct = source != destination,
    .region_count = region_count,
    .regions = {}};
  constexpr std::uint32_t kMaximumRecordedRegions = 16;
  if (regions != nullptr && region_count <= kMaximumRecordedRegions) {
    record.regions.assign(regions, regions + region_count);
  }
  g_copy_commands.push_back(std::move(record));
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
  (void)command_buffer;
  (void)source_stage;
  (void)destination_stage;
  (void)dependencies;
  (void)memory_barrier_count;
  (void)memory_barriers;
  (void)buffer_barrier_count;
  (void)buffer_barriers;
  (void)image_barrier_count;
  (void)image_barriers;
}

int main(int argc, char** argv) {
  std::filesystem::path result_path;
  ncnn_vk_status budget_status = UINT32_MAX;
  std::uint64_t allocations_after_budget = UINT64_MAX;
  bool budget_cleanup = false;
  std::vector<FailureObservation> failure_results;
  ncnn_vk_status operation_wait_status = UINT32_MAX;
  std::uint64_t operation_wait_copy_count = 0;
  bool operation_wait_copy_arguments_valid = false;
  bool operation_wait_cleanup = false;
  std::uint64_t upload_count = 0;
  std::uint64_t copy_count = 0;
  std::vector<CopyCommandRecord> weighted_copy_records;
  bool weighted_copy_arguments_valid = false;
  std::uint64_t live_buffers = 0;
  std::uint64_t released_staging = 0;
  std::uint64_t peak_memory = 0;
  bool weighted_cleanup = false;
  ncnn_vk_status wait_destroy_status = UINT32_MAX;
  std::uint64_t wait_count = 0;
  bool no_early_destroy = false;
  bool wait_cleanup = false;

  try {
    result_path = parse_result_path(argc, argv);
    fake_vulkan::reset();
    budget_cleanup =
      test_budget_preflight(&budget_status, &allocations_after_budget);

    g_settings.operation_hook = {};
    g_settings.force_mapped_memory_sync = true;
    const std::array failure_points{
      std::tuple{"vkCreateBuffer",
                 VK_ERROR_OUT_OF_HOST_MEMORY,
                 NCNN_VK_STATUS_OUT_OF_MEMORY},
      std::tuple{"vkAllocateMemory",
                 VK_ERROR_OUT_OF_DEVICE_MEMORY,
                 NCNN_VK_STATUS_OUT_OF_MEMORY},
      std::tuple{
        "vkBindBufferMemory", VK_ERROR_DEVICE_LOST, NCNN_VK_STATUS_DEVICE_LOST},
      std::tuple{
        "vkMapMemory", VK_ERROR_MEMORY_MAP_FAILED, NCNN_VK_STATUS_API_ERROR},
      std::tuple{"vkFlushMappedMemoryRanges",
                 VK_ERROR_DEVICE_LOST,
                 NCNN_VK_STATUS_DEVICE_LOST},
      std::tuple{"vkCreateDescriptorPool",
                 VK_ERROR_OUT_OF_HOST_MEMORY,
                 NCNN_VK_STATUS_OUT_OF_MEMORY},
      std::tuple{"vkCreateShaderModule",
                 VK_ERROR_OUT_OF_HOST_MEMORY,
                 NCNN_VK_STATUS_OUT_OF_MEMORY},
      std::tuple{
        "vkQueueSubmit", VK_ERROR_DEVICE_LOST, NCNN_VK_STATUS_DEVICE_LOST},
      std::tuple{
        "vkWaitForFences", VK_ERROR_DEVICE_LOST, NCNN_VK_STATUS_DEVICE_LOST},
    };
    for (const auto& [function, vk_result, status] : failure_points) {
      failure_results.push_back(
        inject_create_failure(function, vk_result, status));
    }

    fake_vulkan::reset();
    reset_copy_recording();
    bool fail_wait_once = true;
    g_settings.operation_hook =
      [&fail_wait_once](
        VulkanDeviceOperation operation) -> std::optional<ncnn_vk_status> {
      if (operation == VulkanDeviceOperation::Wait && fail_wait_once) {
        fail_wait_once = false;
        return NCNN_VK_STATUS_DEVICE_LOST;
      }
      return std::nullopt;
    };
    ncnn_vk_options options;
    ncnn_vk_options_init(&options);
    lifecycle_fixture_vk_context* failed_wait_context = nullptr;
    operation_wait_status =
      lifecycle_fixture_vk_create(&options, &failed_wait_context, nullptr, 0);
    operation_wait_copy_count = g_copy_command_count;
    operation_wait_copy_arguments_valid =
      copy_arguments_valid(g_copy_commands, g_copy_command_count);
    if (failed_wait_context != nullptr) {
      (void)lifecycle_fixture_vk_destroy(failed_wait_context);
    }
    expect(operation_wait_status == NCNN_VK_STATUS_DEVICE_LOST,
           "operation hook injects a post-submit wait/device-lost failure");
    expect(operation_wait_copy_arguments_valid,
           "wait-hook failure follows a valid recorded weight-copy command");
    operation_wait_cleanup = no_live_objects();
    expect(operation_wait_cleanup,
           "wait-hook failure drains or safely abandons partial resources");
    g_settings.operation_hook = {};

    weighted_cleanup = test_weighted_success(&upload_count,
                                             &copy_count,
                                             &weighted_copy_records,
                                             &weighted_copy_arguments_valid,
                                             &live_buffers,
                                             &released_staging,
                                             &peak_memory);
    wait_cleanup = test_wait_idle_retry(
      &wait_destroy_status, &wait_count, &no_early_destroy);
  } catch (const std::exception& error) {
    std::cerr << "lifecycle fake test exception: " << error.what() << '\n';
    ++g_failures;
  }
  g_settings.operation_hook = {};
  const bool result_written =
    !result_path.empty() && write_result(result_path,
                                         budget_status,
                                         allocations_after_budget,
                                         budget_cleanup,
                                         failure_results,
                                         operation_wait_status,
                                         operation_wait_copy_count,
                                         operation_wait_copy_arguments_valid,
                                         operation_wait_cleanup,
                                         upload_count,
                                         copy_count,
                                         weighted_copy_records,
                                         weighted_copy_arguments_valid,
                                         live_buffers,
                                         released_staging,
                                         peak_memory,
                                         weighted_cleanup,
                                         wait_destroy_status,
                                         wait_count,
                                         no_early_destroy,
                                         wait_cleanup);
  expect(result_written, "raw fake lifecycle result is written");
  if (g_failures != 0) {
    std::cerr << g_failures << " lifecycle fake assertion(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "Vulkan lifecycle fake tests passed; result="
            << result_path.string() << '\n';
  return EXIT_SUCCESS;
}
