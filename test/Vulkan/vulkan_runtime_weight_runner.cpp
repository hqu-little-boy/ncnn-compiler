#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"

namespace {

using ncnn_mlir::vulkan::deserializeExecutionPlan;
using ncnn_mlir::vulkan::makeVulkanRuntimeOptions;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;
using ncnn_mlir::vulkan::VulkanInputTensor;
using ncnn_mlir::vulkan::VulkanKernelBinary;
using ncnn_mlir::vulkan::VulkanOutputTensor;
using ncnn_mlir::vulkan::VulkanResourcePayload;
using ncnn_mlir::vulkan::VulkanRuntimeContext;
using ncnn_mlir::vulkan::VulkanRuntimeDeviceInfo;
using ncnn_mlir::vulkan::VulkanRuntimeOptions;

constexpr std::uint32_t kInputChannels = 1;
constexpr std::uint32_t kInputHeight = 4;
constexpr std::uint32_t kInputWidth = 4;
constexpr std::uint32_t kOutputChannels = 2;
constexpr std::uint32_t kKernelSize = 3;
constexpr std::uint32_t kOutputHeight = 4;
constexpr std::uint32_t kOutputWidth = 4;
constexpr std::size_t kInputElements =
  kInputChannels * kInputHeight * kInputWidth;
constexpr std::size_t kOutputElements =
  kOutputChannels * kOutputHeight * kOutputWidth;
constexpr float kTolerance = 1.0e-6F;

struct Options final {
  std::string plan;
  std::string shader;
  std::string weight_file;
  std::string cpu_library;
  std::string cpu_symbol = "v2_conv3x3";
  std::string device_match;
  std::string evidence;
  bool validation = false;
};

struct Difference final {
  float maximum_absolute;
  bool passed;
};

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
    if (argument == "--plan") {
      options.plan = next();
    } else if (argument == "--shader") {
      options.shader = next();
    } else if (argument == "--weight-file") {
      options.weight_file = next();
    } else if (argument == "--cpu-library") {
      options.cpu_library = next();
    } else if (argument == "--cpu-symbol") {
      options.cpu_symbol = next();
    } else if (argument == "--device-match") {
      options.device_match = next();
    } else if (argument == "--evidence") {
      options.evidence = next();
    } else if (argument == "--validation") {
      options.validation = true;
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argument));
    }
  }
  if (options.plan.empty() || options.shader.empty() ||
      options.weight_file.empty() || options.cpu_library.empty() ||
      options.device_match.empty() || options.evidence.empty()) {
    throw std::runtime_error(
      "usage: vulkan_runtime_weight_runner --plan FILE --shader FILE "
      "--weight-file FILE --cpu-library FILE --device-match NAME "
      "--evidence FILE [--validation]");
  }
  return options;
}

std::vector<std::byte> readBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open file: " + path);
  }
  const std::streamoff length = input.tellg();
  if (length < 0 || static_cast<std::uint64_t>(length) >
                      std::vector<std::byte>().max_size()) {
    throw std::runtime_error("file has an invalid size: " + path);
  }
  input.seekg(0);
  std::vector<std::byte> bytes(static_cast<std::size_t>(length));
  if (!input.read(reinterpret_cast<char*>(bytes.data()), length)) {
    throw std::runtime_error("cannot read file: " + path);
  }
  return bytes;
}

std::vector<std::uint32_t> readSpirv(const std::string& path) {
  const std::vector<std::byte> bytes = readBytes(path);
  if (bytes.size() < 5 * sizeof(std::uint32_t) ||
      bytes.size() % sizeof(std::uint32_t) != 0) {
    throw std::runtime_error("SPIR-V file has an invalid byte length");
  }
  std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
  std::memcpy(words.data(), bytes.data(), bytes.size());
  if (words.front() != 0x07230203U) {
    throw std::runtime_error("SPIR-V file has an invalid magic number");
  }
  return words;
}

std::uint32_t readU32Le(std::span<const std::byte> bytes, std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4) {
    throw std::runtime_error("truncated weight stream");
  }
  return std::to_integer<std::uint32_t>(bytes[offset]) |
         (std::to_integer<std::uint32_t>(bytes[offset + 1]) << 8) |
         (std::to_integer<std::uint32_t>(bytes[offset + 2]) << 16) |
         (std::to_integer<std::uint32_t>(bytes[offset + 3]) << 24);
}

float readFloatLe(std::span<const std::byte> bytes, std::size_t offset) {
  const std::uint32_t bits = readU32Le(bytes, offset);
  return std::bit_cast<float>(bits);
}

std::vector<float> toHwcf(std::span<const std::byte> bytes) {
  const std::uint64_t weight_count =
    kOutputChannels * kInputChannels * kKernelSize * kKernelSize;
  const std::uint64_t bias_count = kOutputChannels;
  const std::uint64_t expected_bytes =
    sizeof(std::uint32_t) + ((weight_count + bias_count) * sizeof(float));
  if (bytes.size() != expected_bytes || readU32Le(bytes, 0) != 0) {
    throw std::runtime_error("FP32 convolution weight stream is invalid");
  }
  std::vector<float> hwcf(weight_count);
  for (std::uint32_t output = 0; output < kOutputChannels; ++output) {
    for (std::uint32_t input = 0; input < kInputChannels; ++input) {
      for (std::uint32_t y = 0; y < kKernelSize; ++y) {
        for (std::uint32_t x = 0; x < kKernelSize; ++x) {
          const std::size_t source =
            (((output * kInputChannels + input) * kKernelSize + y) *
             kKernelSize) +
            x;
          const std::size_t target =
            (((y * kKernelSize + x) * kInputChannels + input) *
             kOutputChannels) +
            output;
          hwcf[target] = readFloatLe(bytes, 4 + (source * sizeof(float)));
        }
      }
    }
  }
  return hwcf;
}

std::vector<float> readBias(std::span<const std::byte> bytes) {
  std::vector<float> bias(kOutputChannels);
  const std::size_t offset =
    4 + (kOutputChannels * kInputChannels * kKernelSize * kKernelSize * 4);
  for (std::uint32_t index = 0; index < kOutputChannels; ++index) {
    bias[index] = readFloatLe(bytes, offset + (index * sizeof(float)));
  }
  return bias;
}

std::vector<std::byte> asBytes(std::span<const float> values) {
  const std::span<const std::byte> source = std::as_bytes(values);
  return {source.begin(), source.end()};
}

void writeFloats(const std::string& path, std::span<const float> values) {
  std::ofstream output(path, std::ios::binary);
  if (!output ||
      !output.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size_bytes()))) {
    throw std::runtime_error("cannot write runtime evidence: " + path);
  }
}

Difference compare(std::span<const float> actual,
                   std::span<const float> expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("output length differs from reference");
  }
  Difference result{.maximum_absolute = 0.0F, .passed = true};
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float difference = std::abs(actual[index] - expected[index]);
    result.maximum_absolute = std::max(result.maximum_absolute, difference);
    result.passed &=
      difference <= kTolerance + (kTolerance * std::abs(expected[index]));
  }
  return result;
}

void fillInput(std::uint32_t seed, std::vector<float>& input) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  for (std::size_t index = 0; index < input.size(); ++index) {
    if (index == 0) {
      input[index] = -1.0F;
    } else if (index == 1) {
      input[index] = 0.0F;
    } else {
      input[index] = distribution(generator);
    }
  }
}

std::vector<float> makeGolden(std::span<const float> input,
                              std::span<const float> weights,
                              std::span<const float> bias) {
  std::vector<float> output(kOutputElements);
  for (std::uint32_t output_channel = 0; output_channel < kOutputChannels;
       ++output_channel) {
    for (std::uint32_t output_y = 0; output_y < kOutputHeight; ++output_y) {
      for (std::uint32_t output_x = 0; output_x < kOutputWidth; ++output_x) {
        float sum = bias[output_channel];
        for (std::uint32_t kernel_y = 0; kernel_y < kKernelSize; ++kernel_y) {
          for (std::uint32_t kernel_x = 0; kernel_x < kKernelSize; ++kernel_x) {
            const std::int32_t input_y =
              static_cast<std::int32_t>(output_y + kernel_y) - 1;
            const std::int32_t input_x =
              static_cast<std::int32_t>(output_x + kernel_x) - 1;
            if (input_y < 0 || std::cmp_greater_equal(input_y, kInputHeight) ||
                input_x < 0 || std::cmp_greater_equal(input_x, kInputWidth)) {
              continue;
            }
            const std::size_t input_index =
              (static_cast<std::size_t>(input_y) * kInputWidth) +
              static_cast<std::size_t>(input_x);
            const std::size_t weight_index =
              ((kernel_y * kKernelSize + kernel_x) * kOutputChannels) +
              output_channel;
            sum += input[input_index] * weights[weight_index];
          }
        }
        const std::size_t output_index =
          ((output_channel * kOutputHeight + output_y) * kOutputWidth) +
          output_x;
        output[output_index] = sum;
      }
    }
  }
  return output;
}

int run(const Options& options) {
  const auto plan_contents = readBytes(options.plan);
  const std::string plan_json(
    reinterpret_cast<const char*>(plan_contents.data()), plan_contents.size());
  auto plan_result = deserializeExecutionPlan(plan_json);
  if (!plan_result) {
    throw std::runtime_error("invalid Vulkan execution plan: " +
                             plan_result.error());
  }
  VulkanExecutionPlan plan = std::move(*plan_result);
  std::vector<std::uint32_t> shader = readSpirv(options.shader);
  const std::vector<std::byte> source_weights = readBytes(options.weight_file);
  const std::vector<float> hwcf_weights = toHwcf(source_weights);
  const std::vector<float> bias = readBias(source_weights);
  std::uint32_t weight_resource = 0;
  std::uint32_t bias_resource = 0;
  for (const TensorResource& resource : plan.resources) {
    if (resource.kind != ResourceKind::Weight) {
      continue;
    }
    if (resource.name == "weights") {
      weight_resource = resource.id;
    } else if (resource.name == "bias") {
      bias_resource = resource.id;
    }
  }
  if (weight_resource == 0 || bias_resource == 0 || plan.kernels.size() != 1) {
    throw std::runtime_error(
      "Conv2D plan lacks expected persistent weight resources");
  }
  const std::vector<std::byte> weight_bytes = asBytes(hwcf_weights);
  const std::vector<std::byte> bias_bytes = asBytes(bias);
  const std::array<VulkanResourcePayload, 2> weights{
    VulkanResourcePayload{.resource_id = weight_resource,
                          .bytes = weight_bytes},
    VulkanResourcePayload{.resource_id = bias_resource, .bytes = bias_bytes}};
  const std::array<VulkanKernelBinary, 1> kernels{
    VulkanKernelBinary{.kernel_id = plan.kernels.front().id, .words = shader}};
  VulkanRuntimeOptions runtime_options = makeVulkanRuntimeOptions();
  runtime_options.enable_validation = options.validation;
  runtime_options.device_name_match = options.device_match;
  runtime_options.force_mapped_memory_sync = true;
  auto context =
    VulkanRuntimeContext::create(plan, kernels, weights, runtime_options);
  if (!context) {
    throw std::runtime_error("Vulkan context creation failed: " +
                             context.error().diagnostic);
  }
  if (context->get_weight_upload_count() != weights.size()) {
    throw std::runtime_error("weights were not uploaded once during create");
  }
  const VulkanRuntimeDeviceInfo device_info = context->get_device_info();

  void* cpu_library =
    dlopen(options.cpu_library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (cpu_library == nullptr) {
    throw std::runtime_error("cannot load CPU Conv2D AOT: " +
                             std::string(dlerror()));
  }
  dlerror();
  void* symbol = dlsym(cpu_library, options.cpu_symbol.c_str());
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || symbol == nullptr) {
    dlclose(cpu_library);
    throw std::runtime_error(
      "cannot find CPU Conv2D symbol: " +
      std::string(symbol_error != nullptr ? symbol_error : "null symbol"));
  }
  using CpuModel = int (*)(const float*, float*);
  auto cpu_model = reinterpret_cast<CpuModel>(symbol);

  float max_cpu_error = 0.0F;
  float max_gpu_error = 0.0F;
  std::vector<float> captured_input;
  std::vector<float> captured_cpu;
  std::vector<float> captured_gpu;
  std::vector<float> captured_golden;
  for (std::uint32_t call = 0; call < 3; ++call) {
    std::vector<float> input(kInputElements);
    fillInput(call * 71U, input);
    std::vector<float> cpu_output(kOutputElements,
                                  std::numeric_limits<float>::quiet_NaN());
    std::vector<float> gpu_output(kOutputElements,
                                  std::numeric_limits<float>::quiet_NaN());
    const std::vector<float> golden = makeGolden(input, hwcf_weights, bias);
    if (cpu_model(input.data(), cpu_output.data()) != 0) {
      dlclose(cpu_library);
      throw std::runtime_error("CPU Conv2D AOT returned failure");
    }
    const VulkanInputTensor input_tensor{
      .resource_id = plan.inputs.front(),
      .bytes = std::as_bytes(std::span(input))};
    const VulkanOutputTensor output_tensor{
      .resource_id = plan.outputs.front(),
      .bytes = std::as_writable_bytes(std::span(gpu_output))};
    const std::array<VulkanInputTensor, 1> inputs{input_tensor};
    const std::array<VulkanOutputTensor, 1> outputs{output_tensor};
    auto run_result = context->run(inputs, outputs);
    if (!run_result) {
      dlclose(cpu_library);
      throw std::runtime_error("Vulkan Conv2D run failed: " +
                               run_result.error().diagnostic);
    }
    const Difference cpu_diff = compare(cpu_output, golden);
    const Difference gpu_diff = compare(gpu_output, golden);
    const Difference cross_diff = compare(gpu_output, cpu_output);
    if (!cpu_diff.passed || !gpu_diff.passed || !cross_diff.passed) {
      dlclose(cpu_library);
      throw std::runtime_error("Conv2D CPU/Vulkan/golden comparison failed");
    }
    max_cpu_error = std::max(max_cpu_error, cpu_diff.maximum_absolute);
    max_gpu_error = std::max(max_gpu_error, gpu_diff.maximum_absolute);
    if (call == 0) {
      captured_input = input;
      captured_cpu = cpu_output;
      captured_gpu = gpu_output;
      captured_golden = golden;
    }
  }
  dlclose(cpu_library);
  const std::uint64_t weight_upload_count = context->get_weight_upload_count();
  const std::uint64_t peak_memory_bytes = context->get_peak_memory_bytes();
  const std::uint32_t validation_error_count =
    context->get_validation_error_count();
  if (weight_upload_count != weights.size() || validation_error_count != 0 ||
      peak_memory_bytes == 0) {
    throw std::runtime_error("runtime weight/resource validation did not pass");
  }
  auto destroy_result = context->destroy();
  if (!destroy_result) {
    throw std::runtime_error("Vulkan context destroy failed: " +
                             destroy_result.error().diagnostic);
  }
  writeFloats(options.evidence + ".inputs.bin", captured_input);
  std::vector<float> output_capture;
  output_capture.reserve(kOutputElements * 3);
  output_capture.insert(
    output_capture.end(), captured_cpu.begin(), captured_cpu.end());
  output_capture.insert(
    output_capture.end(), captured_gpu.begin(), captured_gpu.end());
  output_capture.insert(
    output_capture.end(), captured_golden.begin(), captured_golden.end());
  writeFloats(options.evidence + ".outputs.bin", output_capture);

  std::ofstream evidence(options.evidence);
  if (!evidence) {
    throw std::runtime_error("cannot create V3 weight evidence file");
  }
  evidence << "{\n"
           << R"(  "device_match": ")" << options.device_match << R"(",)"
           << '\n'
           << R"(  "device_name": )" << std::quoted(device_info.name) << ",\n"
           << R"(  "api_version": )" << device_info.api_version << ",\n"
           << R"(  "driver_version": )" << device_info.driver_version << ",\n"
           << R"(  "vendor_id": )" << device_info.vendor_id << ",\n"
           << R"(  "device_id": )" << device_info.device_id << ",\n"
           << R"(  "operation": "conv2d",)" << '\n'
           << R"(  "kernel_count": 1,)" << '\n'
           << R"(  "repeated_calls": 3,)" << '\n'
           << R"(  "weight_payload_count": )" << weights.size() << ",\n"
           << R"(  "weight_upload_count": )" << weight_upload_count << ",\n"
           << R"(  "peak_memory_bytes": )" << peak_memory_bytes << ",\n"
           << R"(  "validation_enabled": )"
           << (options.validation ? "true" : "false") << ",\n"
           << R"(  "validation_errors": )" << validation_error_count << ",\n"
           << R"(  "cpu_max_abs_error": )" << max_cpu_error << ",\n"
           << R"(  "gpu_max_abs_error": )" << max_gpu_error << "\n"
           << "}\n";
  if (!evidence) {
    throw std::runtime_error("cannot write V3 weight evidence file");
  }
  std::cout << "runtime=device-match:" << options.device_match
            << " op=conv2d calls=3 weights_uploaded=" << weight_upload_count
            << " cpu_abs=" << max_cpu_error << " gpu_abs=" << max_gpu_error
            << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parseOptions(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "vulkan_runtime_weight_runner: " << error.what() << '\n';
    return 1;
  }
}
