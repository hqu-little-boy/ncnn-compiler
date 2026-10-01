#include "vulkan_runtime_abi_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace ncnn_mlir::vulkan::test {
namespace {

constexpr std::size_t kFixtureElements = 257;

struct FixturePackage final {
  VulkanExecutionPlan plan;
  std::vector<std::uint32_t> add_shader;
  std::vector<std::uint32_t> relu_shader;
  VulkanRuntimeOptions options;
};

std::optional<FixturePackage> g_fixture_package;

void writeDiagnostic(char* diagnostic,
                     std::size_t capacity,
                     std::string_view message) noexcept {
  if (capacity == 0 || diagnostic == nullptr) {
    return;
  }
  const std::size_t count = std::min(capacity - 1, message.size());
  std::memcpy(diagnostic, message.data(), count);
  diagnostic[count] = '\0';
}

ncnn_vk_status validateDiagnostic(char* diagnostic,
                                  std::size_t capacity) noexcept {
  if (capacity > 0 && diagnostic == nullptr) {
    return NCNN_VK_STATUS_INVALID_ARGUMENT;
  }
  if (capacity > 0) {
    diagnostic[0] = '\0';
  }
  return NCNN_VK_STATUS_SUCCESS;
}

}  // namespace

void configureAddReluRuntimeFixture(VulkanExecutionPlan plan,
                                    std::vector<std::uint32_t> add_shader,
                                    std::vector<std::uint32_t> relu_shader,
                                    VulkanRuntimeOptions options) {
  g_fixture_package.emplace(
    FixturePackage{.plan = std::move(plan),
                   .add_shader = std::move(add_shader),
                   .relu_shader = std::move(relu_shader),
                   .options = std::move(options)});
}

}  // namespace ncnn_mlir::vulkan::test

struct v3_add_relu_vk_context final {
  explicit v3_add_relu_vk_context(
    ncnn_mlir::vulkan::VulkanRuntimeContext runtime_context)
    : runtime(std::move(runtime_context)) {}

  ncnn_mlir::vulkan::VulkanRuntimeContext runtime;
};

namespace ncnn_mlir::vulkan::test {

std::uint64_t getFixturePeakMemoryBytes(
  const v3_add_relu_vk_context* context) noexcept {
  return context == nullptr ? 0 : context->runtime.get_peak_memory_bytes();
}

std::uint64_t getFixtureWeightUploadCount(
  const v3_add_relu_vk_context* context) noexcept {
  return context == nullptr ? 0 : context->runtime.get_weight_upload_count();
}

std::uint32_t getFixtureValidationErrorCount(
  const v3_add_relu_vk_context* context) noexcept {
  return context == nullptr ? 0 : context->runtime.get_validation_error_count();
}

VulkanRuntimeDeviceInfo getFixtureDeviceInfo(
  const v3_add_relu_vk_context* context) {
  return context == nullptr ? VulkanRuntimeDeviceInfo{}
                            : context->runtime.get_device_info();
}

}  // namespace ncnn_mlir::vulkan::test

using ncnn_mlir::vulkan::VulkanInputTensor;
using ncnn_mlir::vulkan::VulkanKernelBinary;
using ncnn_mlir::vulkan::VulkanOutputTensor;
using ncnn_mlir::vulkan::VulkanRuntimeContext;
using ncnn_mlir::vulkan::VulkanRuntimeOptions;
using ncnn_mlir::vulkan::test::g_fixture_package;
using ncnn_mlir::vulkan::test::kFixtureElements;
using ncnn_mlir::vulkan::test::validateDiagnostic;
using ncnn_mlir::vulkan::test::writeDiagnostic;

extern "C" ncnn_vk_status v3_add_relu_vk_create(
  const ncnn_vk_options* options,
  v3_add_relu_vk_context** context,
  char* diagnostic,
  size_t diagnostic_capacity) {
  if (context == nullptr) {
    writeDiagnostic(diagnostic, diagnostic_capacity, "missing context output");
    return NCNN_VK_STATUS_INVALID_ARGUMENT;
  }
  *context = nullptr;
  const ncnn_vk_status diagnostic_status =
    validateDiagnostic(diagnostic, diagnostic_capacity);
  if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
    return diagnostic_status;
  }
  try {
    if (options == nullptr || options->struct_size < NCNN_VK_OPTIONS_V1_SIZE ||
        options->abi_version != NCNN_VK_ABI_VERSION || options->flags != 0) {
      writeDiagnostic(
        diagnostic, diagnostic_capacity, "invalid Vulkan options");
      return NCNN_VK_STATUS_INVALID_ARGUMENT;
    }
    if (!g_fixture_package) {
      writeDiagnostic(diagnostic,
                      diagnostic_capacity,
                      "Vulkan test fixture package is not configured");
      return NCNN_VK_STATUS_API_ERROR;
    }

    VulkanRuntimeOptions runtime_options = g_fixture_package->options;
    runtime_options.device_index = options->device_index;
    std::array<VulkanKernelBinary, 2> kernels{
      VulkanKernelBinary{.kernel_id = 100,
                         .words = g_fixture_package->add_shader},
      VulkanKernelBinary{.kernel_id = 101,
                         .words = g_fixture_package->relu_shader}};
    auto runtime = VulkanRuntimeContext::create(
      g_fixture_package->plan, kernels, {}, runtime_options);
    if (!runtime) {
      writeDiagnostic(
        diagnostic, diagnostic_capacity, runtime.error().diagnostic);
      return runtime.error().status;
    }
    try {
      auto created =
        std::make_unique<v3_add_relu_vk_context>(std::move(*runtime));
      *context = created.release();
    } catch (const std::bad_alloc&) {
      writeDiagnostic(diagnostic,
                      diagnostic_capacity,
                      "cannot allocate Vulkan context wrapper");
      return NCNN_VK_STATUS_OUT_OF_MEMORY;
    }
    return NCNN_VK_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    writeDiagnostic(diagnostic,
                    diagnostic_capacity,
                    "host allocation failed while creating Vulkan context");
    return NCNN_VK_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    writeDiagnostic(diagnostic, diagnostic_capacity, error.what());
    return NCNN_VK_STATUS_API_ERROR;
  } catch (...) {
    writeDiagnostic(diagnostic,
                    diagnostic_capacity,
                    "unknown error while creating Vulkan context");
    return NCNN_VK_STATUS_API_ERROR;
  }
}

extern "C" ncnn_vk_status v3_add_relu_vk_run(v3_add_relu_vk_context* context,
                                             const float* input1,
                                             size_t input1_elements,
                                             const float* input2,
                                             size_t input2_elements,
                                             float* output,
                                             size_t output_elements,
                                             char* diagnostic,
                                             size_t diagnostic_capacity) {
  const ncnn_vk_status diagnostic_status =
    validateDiagnostic(diagnostic, diagnostic_capacity);
  if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
    return diagnostic_status;
  }
  try {
    if (context == nullptr || input1 == nullptr || input2 == nullptr ||
        output == nullptr) {
      writeDiagnostic(diagnostic, diagnostic_capacity, "invalid run argument");
      return NCNN_VK_STATUS_INVALID_ARGUMENT;
    }
    if (input1_elements != kFixtureElements ||
        input2_elements != kFixtureElements ||
        output_elements != kFixtureElements) {
      writeDiagnostic(diagnostic,
                      diagnostic_capacity,
                      "element count does not match static shape");
      return NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH;
    }
    const std::span<const float> first(input1, input1_elements);
    const std::span<const float> second(input2, input2_elements);
    const std::span<float> result(output, output_elements);
    const std::array<VulkanInputTensor, 2> inputs{
      VulkanInputTensor{.resource_id = 1, .bytes = std::as_bytes(first)},
      VulkanInputTensor{.resource_id = 2, .bytes = std::as_bytes(second)}};
    const std::array<VulkanOutputTensor, 1> outputs{VulkanOutputTensor{
      .resource_id = 4, .bytes = std::as_writable_bytes(result)}};
    auto run_result = context->runtime.run(inputs, outputs);
    if (!run_result) {
      writeDiagnostic(
        diagnostic, diagnostic_capacity, run_result.error().diagnostic);
      return run_result.error().status;
    }
    return NCNN_VK_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    writeDiagnostic(diagnostic,
                    diagnostic_capacity,
                    "host allocation failed during Vulkan run");
    return NCNN_VK_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    writeDiagnostic(diagnostic, diagnostic_capacity, error.what());
    return NCNN_VK_STATUS_API_ERROR;
  } catch (...) {
    writeDiagnostic(
      diagnostic, diagnostic_capacity, "unknown Vulkan run error");
    return NCNN_VK_STATUS_API_ERROR;
  }
}

extern "C" ncnn_vk_status v3_add_relu_vk_destroy(
  v3_add_relu_vk_context* context) {
  const std::unique_ptr<v3_add_relu_vk_context> owned(context);
  if (!owned) {
    return NCNN_VK_STATUS_SUCCESS;
  }
  try {
    const auto result = owned->runtime.destroy();
    return result ? NCNN_VK_STATUS_SUCCESS : result.error().status;
  } catch (...) {
    return NCNN_VK_STATUS_API_ERROR;
  }
}
