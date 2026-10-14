#ifndef NCNN_PRODUCT_TEST_MODEL
#error "NCNN_PRODUCT_TEST_MODEL must name the generated model API"
#endif

#define NCNN_PRODUCT_JOIN_IMPL(left, right) left##right
#define NCNN_PRODUCT_JOIN(left, right) NCNN_PRODUCT_JOIN_IMPL(left, right)
#define NCNN_PRODUCT_STRINGIFY_IMPL(value) #value
#define NCNN_PRODUCT_STRINGIFY(value) NCNN_PRODUCT_STRINGIFY_IMPL(value)

#include NCNN_PRODUCT_STRINGIFY(NCNN_PRODUCT_TEST_MODEL.h)
#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
#include NCNN_PRODUCT_STRINGIFY(NCNN_PRODUCT_JOIN(NCNN_PRODUCT_TEST_MODEL, _vulkan).h)
#endif

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using CpuFunction = decltype(&NCNN_PRODUCT_TEST_MODEL);
#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
using VulkanContext = NCNN_PRODUCT_JOIN(NCNN_PRODUCT_TEST_MODEL, _vk_context);
using VulkanCreateFunction =
  decltype(&NCNN_PRODUCT_JOIN(NCNN_PRODUCT_TEST_MODEL, _vk_create));
using VulkanRunFunction =
  decltype(&NCNN_PRODUCT_JOIN(NCNN_PRODUCT_TEST_MODEL, _vk_run));
using VulkanDestroyFunction =
  decltype(&NCNN_PRODUCT_JOIN(NCNN_PRODUCT_TEST_MODEL, _vk_destroy));
#endif

namespace {

#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
constexpr std::size_t kDiagnosticCapacity = 1024;
#endif

struct Result {
  Result()
    : status("failure"),
      phase("arguments"),
      cpu_status(-1),
      vulkan_create_status(-1),
      vulkan_run_status(-1),
      vulkan_destroy_status(-1),
      vulkan_destroy_null_status(-1),
      context_null(false),
      diagnostic_bounded(false),
      diagnostic() {}

  std::string status;
  std::string phase;
  int cpu_status;
  int vulkan_create_status;
  int vulkan_run_status;
  int vulkan_destroy_status;
  int vulkan_destroy_null_status;
  bool context_null;
  bool diagnostic_bounded;
  std::string diagnostic;
};

struct DynamicLibraryCloser {
  void operator()(void* handle) const noexcept {
    if (handle != nullptr) {
      (void)dlclose(handle);
    }
  }
};

using DynamicLibrary = std::unique_ptr<void, DynamicLibraryCloser>;

#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
auto save_loaded_modules(const std::string& path) -> bool {
  std::ifstream input("/proc/self/maps", std::ios::binary);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!input || !output) {
    return false;
  }
  output << input.rdbuf();
  return static_cast<bool>(output);
}
#endif

auto json_escape(std::string_view value) -> std::string {
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (character < 0x20) {
          constexpr char kHex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped += kHex[(character >> 4) & 0xf];
          escaped += kHex[character & 0xf];
        } else {
          escaped += static_cast<char>(character);
        }
        break;
    }
  }
  return escaped;
}

auto write_result(const std::string& path, const Result& result) -> bool {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "cannot open result JSON: " << path << '\n';
    return false;
  }
  output << R"({"status":")" << json_escape(result.status) << R"(","phase":")"
         << json_escape(result.phase) << R"(","cpu_status":)"
         << result.cpu_status << R"(,"vulkan_create_status":)"
         << result.vulkan_create_status << R"(,"vulkan_run_status":)"
         << result.vulkan_run_status << R"(,"vulkan_destroy_status":)"
         << result.vulkan_destroy_status << R"(,"vulkan_destroy_null_status":)"
         << result.vulkan_destroy_null_status << R"(,"context_null":)"
         << (result.context_null ? "true" : "false")
         << R"(,"diagnostic_bounded":)"
         << (result.diagnostic_bounded ? "true" : "false")
         << R"(,"diagnostic":")" << json_escape(result.diagnostic) << R"("})"
         << '\n';
  if (!output) {
    std::cerr << "failed to write result JSON: " << path << '\n';
    return false;
  }
  return true;
}

auto fail(const std::string& result_path,
          Result& result,
          std::string phase,
          std::string diagnostic) -> int {
  result.status = "failure";
  result.phase = std::move(phase);
  result.diagnostic = std::move(diagnostic);
  std::cerr << result.phase << ": " << result.diagnostic << '\n';
  (void)write_result(result_path, result);
  return 1;
}

auto load_symbol(void* library,
                 const std::string& name,
                 std::string& diagnostic) -> void* {
  (void)dlerror();
  void* symbol = dlsym(library, name.c_str());
  const char* error = dlerror();
  if (error != nullptr) {
    diagnostic = error;
    return nullptr;
  }
  return symbol;
}

template <typename Function>
auto typed_symbol(void* library,
                  const std::string& name,
                  std::string& diagnostic) -> Function {
  return reinterpret_cast<Function>(load_symbol(library, name, diagnostic));
}

auto read_input(const std::string& path,
                std::size_t elements,
                std::vector<float>& input) -> bool {
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  input.resize(elements);
  const std::size_t bytes = elements * sizeof(float);
  file.read(reinterpret_cast<char*>(input.data()),
            static_cast<std::streamsize>(bytes));
  return file.gcount() == static_cast<std::streamsize>(bytes) &&
         file.peek() == std::char_traits<char>::eof();
}

auto write_floats(const std::string& path, const std::vector<float>& values)
  -> bool {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    return false;
  }
  const std::size_t bytes = values.size() * sizeof(float);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(bytes));
  return static_cast<bool>(file);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef NCNN_PRODUCT_TEST_CPU_ONLY
  constexpr int kArgumentCount = 7;
  constexpr int kElementsArgument = 3;
  constexpr int kInputArgument = 4;
  constexpr int kCpuOutputArgument = 5;
  constexpr int kResultArgument = 6;
  constexpr std::string_view kUsage =
    "usage: loader MODEL CPU_DSO ELEMENTS INPUT CPU_OUTPUT RESULT_JSON\n";
#else
  constexpr int kArgumentCount = 9;
  constexpr int kElementsArgument = 4;
  constexpr int kInputArgument = 5;
  constexpr int kCpuOutputArgument = 6;
  constexpr int kResultArgument = 8;
  constexpr std::string_view kUsage =
    "usage: loader MODEL CPU_DSO VULKAN_DSO ELEMENTS INPUT "
    "CPU_OUTPUT VULKAN_OUTPUT RESULT_JSON\n";
#endif
  if (argc != kArgumentCount) {
    std::cerr << kUsage;
    return 2;
  }

  Result result;
  const std::string model_name = argv[1];
  const std::string result_path = argv[kResultArgument];
  const std::string_view count_text(argv[kElementsArgument]);
  std::size_t elements = 0;
  const auto [end, error] = std::from_chars(
    count_text.data(), count_text.data() + count_text.size(), elements);
  if (error != std::errc{} || end != count_text.data() + count_text.size() ||
      elements == 0) {
    return fail(result_path, result, "arguments", "invalid element count");
  }

  std::vector<float> input;
  if (!read_input(argv[kInputArgument], elements, input)) {
    return fail(
      result_path, result, "input", "failed to read exact FP32 input buffer");
  }
  std::vector<float> cpu_output(elements);
#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
  std::vector<float> vulkan_output(elements);
#endif

#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
  if (!save_loaded_modules(result_path + ".before-load.maps")) {
    return fail(
      result_path, result, "module-evidence", "cannot save loader maps");
  }
#endif
  result.phase = "cpu-dlopen";
  DynamicLibrary cpu_library(dlopen(argv[2], RTLD_NOW | RTLD_LOCAL));
  if (!cpu_library) {
    const char* error = dlerror();
    return fail(result_path,
                result,
                result.phase,
                error == nullptr ? "dlopen failed" : error);
  }

#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
  result.phase = "vulkan-dlopen";
  DynamicLibrary vulkan_library(dlopen(argv[3], RTLD_NOW | RTLD_LOCAL));
  if (!vulkan_library) {
    const char* error = dlerror();
    return fail(result_path,
                result,
                result.phase,
                error == nullptr ? "dlopen failed" : error);
  }
  if (!save_loaded_modules(result_path + ".after-model-load.maps")) {
    return fail(
      result_path, result, "module-evidence", "cannot save model maps");
  }
#endif

  std::string symbol_error;
  const auto cpu_function =
    typed_symbol<CpuFunction>(cpu_library.get(), model_name, symbol_error);
  if (cpu_function == nullptr) {
    return fail(result_path, result, "cpu-symbol", symbol_error);
  }
  result.phase = "cpu-run";
  result.cpu_status = cpu_function(input.data(), cpu_output.data());
  if (result.cpu_status != 0) {
    return fail(result_path,
                result,
                result.phase,
                "generated CPU function returned nonzero status");
  }
  if (!write_floats(argv[kCpuOutputArgument], cpu_output)) {
    return fail(
      result_path, result, "cpu-output", "failed to write raw FP32 output");
  }

#ifndef NCNN_PRODUCT_TEST_CPU_ONLY
  const auto create = typed_symbol<VulkanCreateFunction>(
    vulkan_library.get(), model_name + "_vk_create", symbol_error);
  const auto run = typed_symbol<VulkanRunFunction>(
    vulkan_library.get(), model_name + "_vk_run", symbol_error);
  const auto destroy = typed_symbol<VulkanDestroyFunction>(
    vulkan_library.get(), model_name + "_vk_destroy", symbol_error);
  if (create == nullptr || run == nullptr || destroy == nullptr) {
    return fail(result_path, result, "vulkan-symbol", symbol_error);
  }

  ncnn_vk_options options{};
  ncnn_vk_options_init(&options);
  VulkanContext* context = nullptr;
  std::array<char, kDiagnosticCapacity + 2> diagnostic_storage{};
  diagnostic_storage.front() = '#';
  diagnostic_storage.back() = '#';
  char* diagnostic = diagnostic_storage.data() + 1;
  result.phase = "vulkan-create";
  result.vulkan_create_status =
    create(&options, &context, diagnostic, kDiagnosticCapacity);
  result.context_null = context == nullptr;
  result.diagnostic_bounded = diagnostic_storage.front() == '#' &&
                              diagnostic_storage.back() == '#' &&
                              std::find(diagnostic_storage.begin() + 1,
                                        diagnostic_storage.end() - 1,
                                        '\0') != diagnostic_storage.end() - 1;
  if (!save_loaded_modules(result_path + ".after-create.maps")) {
    (void)destroy(context);
    return fail(
      result_path, result, "module-evidence", "cannot save device maps");
  }
  if (!result.diagnostic_bounded) {
    (void)destroy(context);
    return fail(
      result_path, result, "diagnostic", "unbounded create diagnostic");
  }
#ifdef NCNN_PRODUCT_TEST_EXPECT_NO_DEVICE
  result.vulkan_destroy_null_status = destroy(nullptr);
  if ((result.vulkan_create_status != NCNN_VK_STATUS_API_ERROR &&
       result.vulkan_create_status != NCNN_VK_STATUS_UNSUPPORTED_DEVICE) ||
      !result.context_null || diagnostic[0] == '\0' ||
      result.vulkan_destroy_null_status != NCNN_VK_STATUS_SUCCESS) {
    (void)destroy(context);
    return fail(
      result_path, result, "no-device", "unexpected no-device lifecycle");
  }
  std::array<char, 3> tiny_diagnostic{'#', '#', '#'};
  VulkanContext* retry_context = nullptr;
  const auto retry_status =
    create(&options, &retry_context, tiny_diagnostic.data() + 1, 1);
  if (std::cmp_not_equal(retry_status, result.vulkan_create_status) ||
      retry_context != nullptr ||
      tiny_diagnostic != std::array<char, 3>{'#', '\0', '#'}) {
    (void)destroy(retry_context);
    return fail(
      result_path, result, "diagnostic", "one-byte diagnostic overflow");
  }
  result.status = "success";
  result.phase = "expected-no-device";
  result.diagnostic = diagnostic;
  return write_result(result_path, result) ? 0 : 1;
#else
  if (result.vulkan_create_status != NCNN_VK_STATUS_SUCCESS) {
    if (context != nullptr) {
      result.vulkan_destroy_status = destroy(context);
    }
    return fail(result_path,
                result,
                result.phase,
                diagnostic[0] == '\0' ? "Vulkan create returned nonzero status"
                                      : diagnostic);
  }

  result.phase = "vulkan-run";
  result.vulkan_run_status = run(context,
                                 input.data(),
                                 input.size(),
                                 vulkan_output.data(),
                                 vulkan_output.size(),
                                 diagnostic,
                                 kDiagnosticCapacity);
  result.vulkan_destroy_status = destroy(context);
  result.vulkan_destroy_null_status = destroy(nullptr);
  if (result.vulkan_run_status != NCNN_VK_STATUS_SUCCESS ||
      result.vulkan_destroy_status != NCNN_VK_STATUS_SUCCESS ||
      result.vulkan_destroy_null_status != NCNN_VK_STATUS_SUCCESS) {
    std::string message = "run/destroy returned statuses " +
                          std::to_string(result.vulkan_run_status) + "/" +
                          std::to_string(result.vulkan_destroy_status) + "/" +
                          std::to_string(result.vulkan_destroy_null_status);
    if (diagnostic[0] != '\0') {
      message += ": ";
      message += diagnostic;
    }
    return fail(result_path, result, result.phase, std::move(message));
  }
  if (!write_floats(argv[7], vulkan_output)) {
    return fail(
      result_path, result, "vulkan-output", "failed to write raw FP32 output");
  }
#endif
#endif

  result.status = "success";
  result.phase = "complete";
  result.diagnostic.clear();
  if (!write_result(result_path, result)) {
    return 1;
  }
  return 0;
}
