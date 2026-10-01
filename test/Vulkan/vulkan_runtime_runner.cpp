#include "vulkan_runtime_abi_fixture.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using ncnn_mlir::vulkan::KernelExecution;
using ncnn_mlir::vulkan::kExecutionPlanSchemaVersion;
using ncnn_mlir::vulkan::LayoutTransform;
using ncnn_mlir::vulkan::makeVulkanRuntimeOptions;
using ncnn_mlir::vulkan::ResourceAccess;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::serializeExecutionPlan;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::verifyExecutionPlan;
using ncnn_mlir::vulkan::VulkanExecutionPlan;
using ncnn_mlir::vulkan::VulkanRuntimeDeviceInfo;
using ncnn_mlir::vulkan::VulkanRuntimeOperation;
using ncnn_mlir::vulkan::VulkanRuntimeOptions;
using ncnn_mlir::vulkan::test::configureAddReluRuntimeFixture;
using ncnn_mlir::vulkan::test::getFixtureDeviceInfo;
using ncnn_mlir::vulkan::test::getFixturePeakMemoryBytes;
using ncnn_mlir::vulkan::test::getFixtureValidationErrorCount;
using ncnn_mlir::vulkan::test::getFixtureWeightUploadCount;

constexpr std::size_t kElementCount = 257;
constexpr float kTolerance = 1.0e-6F;

struct Options final {
  std::string add_shader;
  std::string relu_shader;
  std::string cpu_library;
  std::string cpu_symbol = "v3_add_relu";
  std::string device_match;
  std::string evidence;
  std::uint32_t calls = 3;
  std::uint32_t concurrent_calls = 4;
  bool validation = false;
};

struct Difference final {
  float max_absolute;
  bool passed;
};

struct RunCapture final {
  std::vector<float> input1;
  std::vector<float> input2;
  std::vector<float> cpu_output;
  std::vector<float> gpu_output;
  std::vector<float> golden;
};

ncnn_mlir::vulkan::VulkanExecutionPlan makePlan() {
  return VulkanExecutionPlan{
    .schema_version = kExecutionPlanSchemaVersion,
    .model = "v3_add_relu_runtime_fixture",
    .build_identity = "v3-runtime-fixture-v1",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources = {TensorResource{.id = 1,
                                 .kind = ResourceKind::Input,
                                 .name = "input1",
                                 .shape = {1, 1, kElementCount},
                                 .logical_layout = "CHW",
                                 .physical_layout = "linear",
                                 .host_layout = "CHW",
                                 .layout_transform =
                                   LayoutTransform{.from_layout = "CHW",
                                                   .to_layout = "linear",
                                                   .operation = "linearize"},
                                 .weight_sha256 = "",
                                 .byte_size = kElementCount * sizeof(float),
                                 .first_kernel = 0,
                                 .last_kernel = 0},
                  TensorResource{.id = 2,
                                 .kind = ResourceKind::Input,
                                 .name = "input2",
                                 .shape = {1, 1, kElementCount},
                                 .logical_layout = "CHW",
                                 .physical_layout = "linear",
                                 .host_layout = "CHW",
                                 .layout_transform =
                                   LayoutTransform{.from_layout = "CHW",
                                                   .to_layout = "linear",
                                                   .operation = "linearize"},
                                 .weight_sha256 = "",
                                 .byte_size = kElementCount * sizeof(float),
                                 .first_kernel = 0,
                                 .last_kernel = 0},
                  TensorResource{.id = 3,
                                 .kind = ResourceKind::Intermediate,
                                 .name = "sum",
                                 .shape = {1, 1, kElementCount},
                                 .logical_layout = "CHW",
                                 .physical_layout = "linear",
                                 .host_layout = "",
                                 .layout_transform = std::nullopt,
                                 .weight_sha256 = "",
                                 .byte_size = kElementCount * sizeof(float),
                                 .first_kernel = 0,
                                 .last_kernel = 1},
                  TensorResource{.id = 4,
                                 .kind = ResourceKind::Output,
                                 .name = "output",
                                 .shape = {1, 1, kElementCount},
                                 .logical_layout = "CHW",
                                 .physical_layout = "linear",
                                 .host_layout = "CHW",
                                 .layout_transform =
                                   LayoutTransform{.from_layout = "linear",
                                                   .to_layout = "CHW",
                                                   .operation = "de-linearize"},
                                 .weight_sha256 = "",
                                 .byte_size = kElementCount * sizeof(float),
                                 .first_kernel = 1,
                                 .last_kernel = 1}},
    .kernels = {KernelExecution{.id = 100,
                                .operation = "add",
                                .entry_point = "add",
                                .local_size = {64, 1, 1},
                                .reads = {1, 2},
                                .writes = {3},
                                .bindings = {{.set = 0,
                                              .binding = 0,
                                              .resource_id = 1,
                                              .access = ResourceAccess::Read},
                                             {.set = 0,
                                              .binding = 1,
                                              .resource_id = 2,
                                              .access = ResourceAccess::Read},
                                             {.set = 0,
                                              .binding = 2,
                                              .resource_id = 3,
                                              .access = ResourceAccess::Write}},
                                .push_constants = {}},
                KernelExecution{.id = 101,
                                .operation = "relu",
                                .entry_point = "relu",
                                .local_size = {64, 1, 1},
                                .reads = {3},
                                .writes = {4},
                                .bindings = {{.set = 0,
                                              .binding = 0,
                                              .resource_id = 3,
                                              .access = ResourceAccess::Read},
                                             {.set = 0,
                                              .binding = 1,
                                              .resource_id = 4,
                                              .access = ResourceAccess::Write}},
                                .push_constants = {}}},
    .inputs = {1, 2},
    .outputs = {4},
    .plan_hash = "",
  };
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    auto next = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value after " +
                                 std::string(argument));
      }
      return argv[++index];
    };
    if (argument == "--add-shader") {
      options.add_shader = next();
    } else if (argument == "--relu-shader") {
      options.relu_shader = next();
    } else if (argument == "--cpu-library") {
      options.cpu_library = next();
    } else if (argument == "--cpu-symbol") {
      options.cpu_symbol = next();
    } else if (argument == "--device-match") {
      options.device_match = next();
    } else if (argument == "--evidence") {
      options.evidence = next();
    } else if (argument == "--calls") {
      options.calls = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (argument == "--concurrent-calls") {
      options.concurrent_calls = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (argument == "--validation") {
      options.validation = true;
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argument));
    }
  }
  if (options.add_shader.empty() || options.relu_shader.empty() ||
      options.cpu_library.empty() || options.device_match.empty() ||
      options.evidence.empty() || options.calls == 0 ||
      options.concurrent_calls == 0) {
    throw std::runtime_error(
      "usage: vulkan_runtime_runner --add-shader FILE --relu-shader FILE "
      "--cpu-library FILE --device-match NAME --evidence FILE [--calls N] "
      "[--concurrent-calls N] [--validation]");
  }
  return options;
}

std::vector<std::uint32_t> readSpirv(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open SPIR-V file: " + path);
  }
  const std::streamoff byte_count = input.tellg();
  if (byte_count < 20 || byte_count % sizeof(std::uint32_t) != 0) {
    throw std::runtime_error("SPIR-V file has an invalid byte length");
  }
  input.seekg(0);
  std::vector<std::uint32_t> words(static_cast<std::size_t>(byte_count) /
                                   sizeof(std::uint32_t));
  if (!input.read(reinterpret_cast<char*>(words.data()), byte_count)) {
    throw std::runtime_error("cannot read SPIR-V file: " + path);
  }
  if (words.front() != 0x07230203U) {
    throw std::runtime_error("SPIR-V file has an invalid magic number");
  }
  return words;
}

Difference compare(std::span<const float> actual,
                   std::span<const float> expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("output length differs from reference");
  }
  Difference difference{.max_absolute = 0.0F, .passed = true};
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float error = std::abs(actual[index] - expected[index]);
    difference.max_absolute = std::max(difference.max_absolute, error);
    difference.passed &=
      error <= kTolerance + (kTolerance * std::abs(expected[index]));
  }
  return difference;
}

void makeInput(std::uint32_t seed,
               std::vector<float>& input1,
               std::vector<float>& input2) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
  for (std::size_t index = 0; index < input1.size(); ++index) {
    if (index == 0) {
      input1[index] = -1.0F;
      input2[index] = 1.0F;
    } else if (index == 1) {
      input1[index] = 0.0F;
      input2[index] = 0.0F;
    } else if (index == 2) {
      input1[index] = 1.0F;
      input2[index] = -1.0F;
    } else {
      input1[index] = distribution(generator);
      input2[index] = distribution(generator);
    }
  }
}

void runOnce(v3_add_relu_vk_context* context,
             std::uint32_t seed,
             int (*cpu_model)(const float*, const float*, float*),
             Difference& cpu_diff,
             Difference& gpu_diff,
             RunCapture* evidence_capture) {
  std::vector<float> input1(kElementCount);
  std::vector<float> input2(kElementCount);
  std::vector<float> cpu_output(kElementCount,
                                std::numeric_limits<float>::quiet_NaN());
  std::vector<float> gpu_output(kElementCount,
                                std::numeric_limits<float>::quiet_NaN());
  std::vector<float> golden(kElementCount);
  makeInput(seed, input1, input2);
  for (std::size_t index = 0; index < kElementCount; ++index) {
    golden[index] = std::max(input1[index] + input2[index], 0.0F);
  }
  if (cpu_model(input1.data(), input2.data(), cpu_output.data()) != 0) {
    throw std::runtime_error("CPU AOT fixture returned failure");
  }
  std::array<char, 256> diagnostic{};
  const ncnn_vk_status status = v3_add_relu_vk_run(context,
                                                   input1.data(),
                                                   input1.size(),
                                                   input2.data(),
                                                   input2.size(),
                                                   gpu_output.data(),
                                                   gpu_output.size(),
                                                   diagnostic.data(),
                                                   diagnostic.size());
  if (status != NCNN_VK_STATUS_SUCCESS) {
    throw std::runtime_error("Vulkan C ABI run failed: " +
                             std::string(diagnostic.data()));
  }
  cpu_diff = compare(cpu_output, golden);
  gpu_diff = compare(gpu_output, golden);
  const Difference cross_diff = compare(gpu_output, cpu_output);
  if (!cpu_diff.passed || !gpu_diff.passed || !cross_diff.passed) {
    throw std::runtime_error("CPU/Vulkan/golden comparison failed");
  }
  if (evidence_capture != nullptr) {
    *evidence_capture = RunCapture{.input1 = std::move(input1),
                                   .input2 = std::move(input2),
                                   .cpu_output = std::move(cpu_output),
                                   .gpu_output = std::move(gpu_output),
                                   .golden = std::move(golden)};
  }
}

void writeFloats(const std::string& path, std::span<const float> values) {
  std::ofstream output(path, std::ios::binary);
  if (!output ||
      !output.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size_bytes()))) {
    throw std::runtime_error("cannot write runtime binary evidence: " + path);
  }
}

void validateFailureCleanup(
  const ncnn_mlir::vulkan::VulkanExecutionPlan& plan,
  const std::vector<std::uint32_t>& add_shader,
  const std::vector<std::uint32_t>& relu_shader,
  const ncnn_mlir::vulkan::VulkanRuntimeOptions& base) {
  const std::array<std::pair<VulkanRuntimeOperation, std::uint32_t>, 11> points{
    {
      {VulkanRuntimeOperation::CreateInstance, 1},
      {VulkanRuntimeOperation::SelectDevice, 1},
      {VulkanRuntimeOperation::CreateDevice, 1},
      {VulkanRuntimeOperation::CreateCommandResources, 1},
      {VulkanRuntimeOperation::CreateCommandResources, 2},
      {VulkanRuntimeOperation::CreateCommandResources, 3},
      {VulkanRuntimeOperation::CreateBuffer, 2},
      {VulkanRuntimeOperation::AllocateMemory, 2},
      {VulkanRuntimeOperation::CreateDescriptor, 2},
      {VulkanRuntimeOperation::CreateShaderModule, 1},
      {VulkanRuntimeOperation::CreatePipeline, 1},
    }};
  for (const auto& [failure_point, fail_on_call] : points) {
    std::uint32_t seen = 0;
    VulkanRuntimeOptions options = base;
    options.failure_injector =
      [failure_point, fail_on_call, &seen](
        VulkanRuntimeOperation operation) -> std::optional<ncnn_vk_status> {
      if (operation == failure_point && ++seen == fail_on_call) {
        return failure_point == VulkanRuntimeOperation::AllocateMemory
                 ? NCNN_VK_STATUS_OUT_OF_MEMORY
                 : NCNN_VK_STATUS_API_ERROR;
      }
      return std::nullopt;
    };
    ncnn_mlir::vulkan::test::configureAddReluRuntimeFixture(
      plan, add_shader, relu_shader, options);
    ncnn_vk_options abi_options;
    ncnn_vk_options_init(&abi_options);
    abi_options.device_index = NCNN_VK_DEVICE_AUTO;
    v3_add_relu_vk_context* context = nullptr;
    std::array<char, 256> diagnostic{};
    const ncnn_vk_status status = v3_add_relu_vk_create(
      &abi_options, &context, diagnostic.data(), diagnostic.size());
    const ncnn_vk_status expected_status =
      failure_point == VulkanRuntimeOperation::AllocateMemory
        ? NCNN_VK_STATUS_OUT_OF_MEMORY
        : NCNN_VK_STATUS_API_ERROR;
    if (context != nullptr || status != expected_status ||
        diagnostic[0] == '\0' || diagnostic.back() != '\0' ||
        std::string_view(diagnostic.data()).find("validation_error_count") !=
          std::string_view::npos) {
      (void)v3_add_relu_vk_destroy(context);
      throw std::runtime_error(
        "injected runtime create failure was not cleaned");
    }
  }
  VulkanRuntimeOptions limited_options = base;
  limited_options.max_context_memory_bytes = 1;
  ncnn_mlir::vulkan::test::configureAddReluRuntimeFixture(
    plan, add_shader, relu_shader, limited_options);
  ncnn_vk_options abi_options;
  ncnn_vk_options_init(&abi_options);
  v3_add_relu_vk_context* context = nullptr;
  std::array<char, 256> diagnostic{};
  const ncnn_vk_status status = v3_add_relu_vk_create(
    &abi_options, &context, diagnostic.data(), diagnostic.size());
  if (context != nullptr || status != NCNN_VK_STATUS_OUT_OF_MEMORY ||
      std::string_view(diagnostic.data()).find("memory budget") ==
        std::string_view::npos ||
      std::string_view(diagnostic.data()).find("validation_error_count") !=
        std::string_view::npos) {
    (void)v3_add_relu_vk_destroy(context);
    throw std::runtime_error("context memory budget failure was not cleaned");
  }
}

int run(const Options& options) {
  const VulkanExecutionPlan plan = makePlan();
  const auto verified = verifyExecutionPlan(plan);
  if (!verified) {
    throw std::runtime_error("invalid V3 integration execution plan: " +
                             verified.error());
  }
  const auto serialized_plan = serializeExecutionPlan(plan);
  if (!serialized_plan) {
    throw std::runtime_error("cannot serialize V3 runtime plan: " +
                             serialized_plan.error());
  }
  std::ofstream plan_evidence(options.evidence + ".execution-plan.json");
  if (!plan_evidence || !plan_evidence.write(serialized_plan->data(),
                                             static_cast<std::streamsize>(
                                               serialized_plan->size()))) {
    throw std::runtime_error("cannot write V3 runtime plan evidence");
  }
  std::vector<std::uint32_t> add_shader = readSpirv(options.add_shader);
  std::vector<std::uint32_t> relu_shader = readSpirv(options.relu_shader);
  VulkanRuntimeOptions runtime_options = makeVulkanRuntimeOptions();
  runtime_options.enable_validation = options.validation;
  runtime_options.device_name_match = options.device_match;
  runtime_options.force_mapped_memory_sync = true;
  validateFailureCleanup(plan, add_shader, relu_shader, runtime_options);
  bool inject_failure = false;
  VulkanRuntimeOperation injected_operation = VulkanRuntimeOperation::Submit;
  runtime_options.failure_injector =
    [&inject_failure, &injected_operation](
      VulkanRuntimeOperation operation) -> std::optional<ncnn_vk_status> {
    if (inject_failure && operation == injected_operation) {
      inject_failure = false;
      return NCNN_VK_STATUS_DEVICE_LOST;
    }
    return std::nullopt;
  };
  configureAddReluRuntimeFixture(
    plan, add_shader, relu_shader, runtime_options);

  void* cpu_library =
    dlopen(options.cpu_library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (cpu_library == nullptr) {
    throw std::runtime_error("cannot load CPU AOT fixture: " +
                             std::string(dlerror()));
  }
  dlerror();
  void* symbol = dlsym(cpu_library, options.cpu_symbol.c_str());
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || symbol == nullptr) {
    dlclose(cpu_library);
    throw std::runtime_error(
      "cannot find CPU model symbol: " +
      std::string(symbol_error != nullptr ? symbol_error : "null symbol"));
  }
  using CpuModel = int (*)(const float*, const float*, float*);
  auto cpu_model = reinterpret_cast<CpuModel>(symbol);

  ncnn_vk_options abi_options;
  ncnn_vk_options_init(&abi_options);
  abi_options.device_index = NCNN_VK_DEVICE_AUTO;
  v3_add_relu_vk_context* context = nullptr;
  v3_add_relu_vk_context* other_context = nullptr;
  std::array<char, 256> diagnostic{};
  if (v3_add_relu_vk_destroy(nullptr) != NCNN_VK_STATUS_SUCCESS) {
    throw std::runtime_error("destroy(NULL) did not return success");
  }
  if (v3_add_relu_vk_create(
        nullptr, &context, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != nullptr ||
      v3_add_relu_vk_create(
        &abi_options, nullptr, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT) {
    throw std::runtime_error("ABI create null-pointer validation did not hold");
  }
  ncnn_vk_options invalid_options = abi_options;
  invalid_options.flags = 1;
  if (v3_add_relu_vk_create(
        &invalid_options, &context, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != nullptr || diagnostic.back() != '\0') {
    throw std::runtime_error("ABI options failure contract did not hold");
  }
  invalid_options = abi_options;
  invalid_options.struct_size = NCNN_VK_OPTIONS_V1_SIZE - 1;
  diagnostic.fill('x');
  if (v3_add_relu_vk_create(&invalid_options, &context, diagnostic.data(), 2) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != nullptr || diagnostic[1] != '\0') {
    throw std::runtime_error("ABI diagnostic truncation contract did not hold");
  }
  if (v3_add_relu_vk_create(&abi_options, &context, nullptr, 1) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      context != nullptr) {
    throw std::runtime_error("ABI diagnostic pointer validation did not hold");
  }
  ncnn_vk_options unsupported_device = abi_options;
  unsupported_device.device_index = 0xfffffffeU;
  if (v3_add_relu_vk_create(
        &unsupported_device, &context, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_UNSUPPORTED_DEVICE ||
      context != nullptr) {
    throw std::runtime_error("ABI unsupported-device status did not hold");
  }
  struct ExtendedOptions final {
    ncnn_vk_options options;
    std::uint64_t reserved;
  } extended_options{};
  ncnn_vk_options_init(&extended_options.options);
  extended_options.options.struct_size =
    static_cast<std::uint32_t>(sizeof(extended_options));
  extended_options.options.device_index = 0;
  v3_add_relu_vk_context* extended_context = nullptr;
  if (v3_add_relu_vk_create(&extended_options.options,
                            &extended_context,
                            diagnostic.data(),
                            diagnostic.size()) != NCNN_VK_STATUS_SUCCESS ||
      extended_context == nullptr ||
      v3_add_relu_vk_destroy(extended_context) != NCNN_VK_STATUS_SUCCESS) {
    (void)v3_add_relu_vk_destroy(extended_context);
    throw std::runtime_error("ABI forward-compatible options failed");
  }
  if (v3_add_relu_vk_create(
        &abi_options, &context, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_SUCCESS ||
      context == nullptr) {
    dlclose(cpu_library);
    throw std::runtime_error("Vulkan context creation failed: " +
                             std::string(diagnostic.data()));
  }
  if (v3_add_relu_vk_create(
        &abi_options, &other_context, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_SUCCESS ||
      other_context == nullptr || other_context == context) {
    (void)v3_add_relu_vk_destroy(context);
    dlclose(cpu_library);
    throw std::runtime_error("independent context creation failed: " +
                             std::string(diagnostic.data()));
  }
  const VulkanRuntimeDeviceInfo device_info = getFixtureDeviceInfo(context);
  const VulkanRuntimeDeviceInfo other_device_info =
    getFixtureDeviceInfo(other_context);
  if (device_info.name.empty() || device_info.name != other_device_info.name ||
      device_info.vendor_id != other_device_info.vendor_id ||
      device_info.device_id != other_device_info.device_id) {
    (void)v3_add_relu_vk_destroy(context);
    (void)v3_add_relu_vk_destroy(other_context);
    dlclose(cpu_library);
    throw std::runtime_error("separate contexts selected different devices");
  }

  std::vector<float> invalid_output(kElementCount, -17.0F);
  if (v3_add_relu_vk_run(context,
                         nullptr,
                         kElementCount,
                         nullptr,
                         kElementCount,
                         invalid_output.data(),
                         kElementCount,
                         diagnostic.data(),
                         diagnostic.size()) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      invalid_output.front() != -17.0F) {
    throw std::runtime_error("ABI null input validation did not hold");
  }
  std::vector<float> zeros(kElementCount, 0.0F);
  if (v3_add_relu_vk_run(nullptr,
                         zeros.data(),
                         zeros.size(),
                         zeros.data(),
                         zeros.size(),
                         invalid_output.data(),
                         invalid_output.size(),
                         diagnostic.data(),
                         diagnostic.size()) !=
        NCNN_VK_STATUS_INVALID_ARGUMENT ||
      invalid_output.front() != -17.0F) {
    throw std::runtime_error("ABI null-context validation did not hold");
  }
  if (v3_add_relu_vk_run(context,
                         zeros.data(),
                         zeros.size(),
                         zeros.data(),
                         zeros.size(),
                         nullptr,
                         kElementCount,
                         diagnostic.data(),
                         diagnostic.size()) !=
      NCNN_VK_STATUS_INVALID_ARGUMENT) {
    throw std::runtime_error("ABI null-output validation did not hold");
  }
  if (v3_add_relu_vk_run(context,
                         zeros.data(),
                         kElementCount - 1,
                         zeros.data(),
                         kElementCount,
                         invalid_output.data(),
                         kElementCount,
                         diagnostic.data(),
                         diagnostic.size()) !=
        NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH ||
      invalid_output.front() != -17.0F) {
    throw std::runtime_error("ABI buffer-capacity validation did not hold");
  }

  float max_cpu_error = 0.0F;
  float max_gpu_error = 0.0F;
  RunCapture reference_capture;
  for (std::uint32_t call = 0; call < options.calls; ++call) {
    Difference cpu_diff{};
    Difference gpu_diff{};
    runOnce(call % 2 == 0 ? context : other_context,
            call * 37U,
            cpu_model,
            cpu_diff,
            gpu_diff,
            call == 0 ? &reference_capture : nullptr);
    max_cpu_error = std::max(max_cpu_error, cpu_diff.max_absolute);
    max_gpu_error = std::max(max_gpu_error, gpu_diff.max_absolute);
  }

  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> worker_errors(options.concurrent_calls);
  std::vector<float> concurrent_cpu_errors(options.concurrent_calls);
  std::vector<float> concurrent_gpu_errors(options.concurrent_calls);
  for (std::uint32_t index = 0; index < options.concurrent_calls; ++index) {
    workers.emplace_back([&, index]() {
      try {
        Difference cpu_diff{};
        Difference gpu_diff{};
        runOnce(context, 9000U + index, cpu_model, cpu_diff, gpu_diff, nullptr);
        concurrent_cpu_errors[index] = cpu_diff.max_absolute;
        concurrent_gpu_errors[index] = gpu_diff.max_absolute;
      } catch (...) {
        worker_errors[index] = std::current_exception();
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  for (const std::exception_ptr& error : worker_errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  for (std::size_t index = 0; index < concurrent_cpu_errors.size(); ++index) {
    max_cpu_error = std::max(max_cpu_error, concurrent_cpu_errors[index]);
    max_gpu_error = std::max(max_gpu_error, concurrent_gpu_errors[index]);
  }
  std::uint32_t validation_errors =
    getFixtureValidationErrorCount(context) +
    getFixtureValidationErrorCount(other_context);
  if (validation_errors != 0) {
    throw std::runtime_error("validation layer reported a runtime error");
  }

  inject_failure = true;
  injected_operation = VulkanRuntimeOperation::Submit;
  std::vector<float> untouched(kElementCount, -23.0F);
  if (v3_add_relu_vk_run(context,
                         zeros.data(),
                         zeros.size(),
                         zeros.data(),
                         zeros.size(),
                         untouched.data(),
                         untouched.size(),
                         diagnostic.data(),
                         diagnostic.size()) != NCNN_VK_STATUS_DEVICE_LOST ||
      untouched.front() != -23.0F) {
    throw std::runtime_error("injected submit failure did not preserve output");
  }
  validation_errors += getFixtureValidationErrorCount(context);
  if (validation_errors != 0) {
    throw std::runtime_error(
      "validation layer reported a submit failure error");
  }
  if (v3_add_relu_vk_run(context,
                         zeros.data(),
                         zeros.size(),
                         zeros.data(),
                         zeros.size(),
                         untouched.data(),
                         untouched.size(),
                         diagnostic.data(),
                         diagnostic.size()) != NCNN_VK_STATUS_DEVICE_LOST) {
    throw std::runtime_error("failed context was not marked unusable");
  }
  if (v3_add_relu_vk_destroy(context) != NCNN_VK_STATUS_SUCCESS) {
    context = nullptr;
    throw std::runtime_error("submit-failed context cleanup reported an error");
  }
  context = nullptr;

  inject_failure = true;
  injected_operation = VulkanRuntimeOperation::Wait;
  if (v3_add_relu_vk_run(other_context,
                         zeros.data(),
                         zeros.size(),
                         zeros.data(),
                         zeros.size(),
                         untouched.data(),
                         untouched.size(),
                         diagnostic.data(),
                         diagnostic.size()) != NCNN_VK_STATUS_DEVICE_LOST ||
      untouched.front() != -23.0F) {
    throw std::runtime_error("injected wait failure did not preserve output");
  }
  validation_errors += getFixtureValidationErrorCount(other_context);
  if (validation_errors != 0) {
    throw std::runtime_error("validation layer reported an error in runtime");
  }
  if (v3_add_relu_vk_destroy(other_context) != NCNN_VK_STATUS_SUCCESS) {
    other_context = nullptr;
    throw std::runtime_error("wait-failed context cleanup reported an error");
  }
  other_context = nullptr;

  inject_failure = false;
  if (v3_add_relu_vk_create(
        &abi_options, &context, diagnostic.data(), diagnostic.size()) !=
        NCNN_VK_STATUS_SUCCESS ||
      context == nullptr ||
      v3_add_relu_vk_run(context,
                         zeros.data(),
                         zeros.size(),
                         zeros.data(),
                         zeros.size(),
                         invalid_output.data(),
                         invalid_output.size(),
                         diagnostic.data(),
                         diagnostic.size()) != NCNN_VK_STATUS_SUCCESS) {
    (void)v3_add_relu_vk_destroy(context);
    dlclose(cpu_library);
    throw std::runtime_error("runtime recovery after injected failures failed");
  }
  validation_errors += getFixtureValidationErrorCount(context);
  if (validation_errors != 0 || getFixturePeakMemoryBytes(context) == 0 ||
      getFixtureWeightUploadCount(context) != 0) {
    (void)v3_add_relu_vk_destroy(context);
    dlclose(cpu_library);
    throw std::runtime_error(
      "runtime resource or validation counters are invalid");
  }
  if (v3_add_relu_vk_destroy(context) != NCNN_VK_STATUS_SUCCESS) {
    context = nullptr;
    dlclose(cpu_library);
    throw std::runtime_error("healthy context cleanup reported an error");
  }
  context = nullptr;
  dlclose(cpu_library);

  std::vector<float> input_evidence;
  input_evidence.reserve(reference_capture.input1.size() +
                         reference_capture.input2.size());
  input_evidence.insert(input_evidence.end(),
                        reference_capture.input1.begin(),
                        reference_capture.input1.end());
  input_evidence.insert(input_evidence.end(),
                        reference_capture.input2.begin(),
                        reference_capture.input2.end());
  std::vector<float> output_evidence;
  output_evidence.reserve(reference_capture.cpu_output.size() * 3);
  output_evidence.insert(output_evidence.end(),
                         reference_capture.cpu_output.begin(),
                         reference_capture.cpu_output.end());
  output_evidence.insert(output_evidence.end(),
                         reference_capture.gpu_output.begin(),
                         reference_capture.gpu_output.end());
  output_evidence.insert(output_evidence.end(),
                         reference_capture.golden.begin(),
                         reference_capture.golden.end());
  writeFloats(options.evidence + ".inputs.bin", input_evidence);
  writeFloats(options.evidence + ".outputs.bin", output_evidence);

  std::ofstream evidence(options.evidence);
  if (!evidence) {
    throw std::runtime_error("cannot create runtime evidence file");
  }
  evidence << "{\n"
           << R"(  "device_match": ")" << options.device_match << R"(",)"
           << '\n'
           << R"(  "device_name": )" << std::quoted(device_info.name) << ",\n"
           << R"(  "api_version": )" << device_info.api_version << ",\n"
           << R"(  "driver_version": )" << device_info.driver_version << ",\n"
           << R"(  "vendor_id": )" << device_info.vendor_id << ",\n"
           << R"(  "device_id": )" << device_info.device_id << ",\n"
           << R"(  "operation": "add_then_relu",)" << '\n'
           << R"(  "kernel_count": 2,)" << '\n'
           << R"(  "calls": )" << options.calls << ",\n"
           << R"(  "concurrent_calls": )" << options.concurrent_calls << ",\n"
           << R"(  "validation_enabled": )"
           << (options.validation ? "true" : "false") << ",\n"
           << R"(  "validation_errors": )" << validation_errors << ",\n"
           << R"(  "cpu_max_abs_error": )" << max_cpu_error << ",\n"
           << R"(  "gpu_max_abs_error": )" << max_gpu_error << ",\n"
           << R"(  "allocation_budget_bytes": )"
           << makeVulkanRuntimeOptions().max_context_memory_bytes << "\n"
           << "}\n";
  if (!evidence) {
    throw std::runtime_error("cannot write runtime evidence file");
  }
  std::cout << "runtime=device-match:" << options.device_match
            << " kernels=2 calls=" << options.calls
            << " concurrent=" << options.concurrent_calls
            << " cpu_abs=" << max_cpu_error << " gpu_abs=" << max_gpu_error
            << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parseOptions(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "vulkan_runtime_runner: " << error.what() << '\n';
    return 1;
  }
}
