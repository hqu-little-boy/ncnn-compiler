#pragma once

#include "ncnn-mlir/Vulkan/ModelABI.h"

#ifdef __cplusplus
#include <cstdint>
#include <vector>

#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"

struct v3_add_relu_vk_context;

namespace ncnn_mlir::vulkan::test {

void configureAddReluRuntimeFixture(VulkanExecutionPlan plan,
                                    std::vector<std::uint32_t> add_shader,
                                    std::vector<std::uint32_t> relu_shader,
                                    VulkanRuntimeOptions options);

[[nodiscard]] std::uint64_t getFixturePeakMemoryBytes(
  const ::v3_add_relu_vk_context* context) noexcept;
[[nodiscard]] std::uint64_t getFixtureWeightUploadCount(
  const ::v3_add_relu_vk_context* context) noexcept;
[[nodiscard]] std::uint32_t getFixtureValidationErrorCount(
  const ::v3_add_relu_vk_context* context) noexcept;
[[nodiscard]] VulkanRuntimeDeviceInfo getFixtureDeviceInfo(
  const ::v3_add_relu_vk_context* context);

}  // namespace ncnn_mlir::vulkan::test

extern "C" {
struct v3_add_relu_vk_context;
#else
typedef struct v3_add_relu_vk_context v3_add_relu_vk_context;
#endif

ncnn_vk_status v3_add_relu_vk_create(const ncnn_vk_options* options,
                                     v3_add_relu_vk_context** context,
                                     char* diagnostic,
                                     size_t diagnostic_capacity);

ncnn_vk_status v3_add_relu_vk_run(v3_add_relu_vk_context* context,
                                  const float* input1,
                                  size_t input1_elements,
                                  const float* input2,
                                  size_t input2_elements,
                                  float* output,
                                  size_t output_elements,
                                  char* diagnostic,
                                  size_t diagnostic_capacity);

ncnn_vk_status v3_add_relu_vk_destroy(v3_add_relu_vk_context* context);

#ifdef __cplusplus
}
#endif
