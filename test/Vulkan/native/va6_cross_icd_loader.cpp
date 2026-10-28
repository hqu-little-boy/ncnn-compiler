// VA6.3 cross-ICD loader: exercises device late binding on one fixed DSO.
//
// The loader is compiled against a deployed model header pair (CPU +
// Vulkan) and never sees the build directory.  It creates the model context
// with a caller-chosen device ordinal (NCNN_VK_DEVICE_AUTO, an explicit
// valid ordinal, or an invalid one), runs the model, destroys the context,
// and records the raw statuses as JSON.  Invalid ordinals must fail with
// UNSUPPORTED_DEVICE and publish nothing; the no-ICD environment must fail
// create explicitly and still report destroy(NULL) == SUCCESS.
#include <dlfcn.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef NCNN_VA6_CROSS_ICD_MODEL
#error "NCNN_VA6_CROSS_ICD_MODEL must name the deployed model API"
#endif

#define NCNN_VA6_JOIN_IMPL(left, right) left##right
#define NCNN_VA6_JOIN(left, right) NCNN_VA6_JOIN_IMPL(left, right)
#define NCNN_VA6_STRINGIFY_IMPL(value) #value
#define NCNN_VA6_STRINGIFY(value) NCNN_VA6_STRINGIFY_IMPL(value)

#define NCNN_VA6_MODEL_NAME NCNN_VA6_STRINGIFY(NCNN_VA6_CROSS_ICD_MODEL)

#ifndef NCNN_VA6_MODEL_HEADER
#error "NCNN_VA6_MODEL_HEADER must name the deployed CPU header"
#endif
#ifndef NCNN_VA6_VULKAN_HEADER
#error "NCNN_VA6_VULKAN_HEADER must name the deployed Vulkan header"
#endif

#include NCNN_VA6_MODEL_HEADER
#include NCNN_VA6_VULKAN_HEADER

namespace {

using CpuFunction = decltype(&NCNN_VA6_CROSS_ICD_MODEL);
using VulkanContext = NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_context);
using CreateFunction =
  decltype(&NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_create));
using RunFunction = decltype(&NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_run));
using DestroyFunction =
  decltype(&NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_destroy));

constexpr std::size_t kDiagnosticCapacity = 512;

struct LoaderResult final {
  std::string model;
  std::uint32_t requested_device_index;
  std::string create_status;
  bool context_null;
  bool diagnostic_bounded;
  std::string cpu_status;
  std::string destroy_status;
  std::string destroy_null_status;
  std::string run_status;
  std::string diagnostic;
  std::string phase;
  std::string status;
};

bool write_result(const std::string& path, const LoaderResult& result) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  auto quoted = [](const std::string& text) {
    std::string escaped;
    for (char character : text) {
      if (character == '"' || character == '\\') {
        escaped.push_back('\\');
      }
      if (static_cast<unsigned char>(character) >= 0x20) {
        escaped.push_back(character);
      }
    }
    return "\"" + escaped + "\"";
  };
  stream << "{"
         << "\"model\":" << quoted(result.model) << ","
         << "\"requested_device_index\":" << result.requested_device_index
         << ",\"create_status\":" << result.create_status
         << ",\"context_null\":" << (result.context_null ? "true" : "false")
         << ",\"diagnostic_bounded\":"
         << (result.diagnostic_bounded ? "true" : "false")
         << ",\"cpu_status\":" << result.cpu_status
         << ",\"run_status\":" << result.run_status
         << ",\"destroy_status\":" << result.destroy_status
         << ",\"destroy_null_status\":" << result.destroy_null_status
         << ",\"diagnostic\":" << quoted(result.diagnostic) << ","
         << "\"phase\":" << quoted(result.phase) << ","
         << "\"status\":" << quoted(result.status) << "}\n";
  return static_cast<bool>(stream);
}

int fail(const std::string& path,
         LoaderResult result,
         std::string phase,
         std::string message) {
  result.status = "failure";
  result.phase = std::move(phase);
  result.diagnostic = std::move(message);
  if (!write_result(path, result)) {
    return 2;
  }
  return 1;
}

bool write_floats(const std::string& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  return static_cast<bool>(stream);
}

}  // namespace

int main(int argc, char** argv) {
  // argv: model cpu_lib vk_lib device_index expect_create_success
  //       elements input cpu_output vulkan_output result_path
  if (argc != 11) {
    std::cerr << "usage: va6-cross-icd-loader MODEL CPU_LIB VK_LIB "
                 "DEVICE_INDEX EXPECT_SUCCESS ELEMENTS INPUT CPU_OUT VK_OUT "
                 "RESULT\n";
    return 2;
  }
  LoaderResult result;
  result.model = argv[1];
  result.requested_device_index =
    static_cast<std::uint32_t>(std::stoul(argv[4]));
  const bool expect_create_success = std::string(argv[5]) == "1";
  const std::size_t elements = std::stoul(argv[6]);
  const std::string input_path = argv[7];
  const std::string cpu_output_path = argv[8];
  const std::string vulkan_output_path = argv[9];
  const std::string result_path = argv[10];
  result.create_status = "-1";
  result.run_status = "-1";
  result.destroy_status = "-1";
  result.destroy_null_status = "-1";
  result.context_null = true;

  std::ifstream input_stream(input_path, std::ios::binary);
  if (!input_stream) {
    return fail(result_path, result, "input", "cannot open input tensor");
  }
  std::vector<float> input(elements);
  input_stream.read(reinterpret_cast<char*>(input.data()),
                    static_cast<std::streamsize>(elements * sizeof(float)));
  if (!input_stream || static_cast<std::size_t>(input_stream.gcount()) !=
                         elements * sizeof(float)) {
    return fail(
      result_path, result, "input", "input tensor has the wrong f32 size");
  }

  void* cpu_library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (cpu_library == nullptr) {
    return fail(result_path,
                result,
                "cpu-dlopen",
                dlerror() == nullptr ? "dlopen failed" : dlerror());
  }
  void* vulkan_library = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
  if (vulkan_library == nullptr) {
    dlclose(cpu_library);
    return fail(result_path,
                result,
                "vulkan-dlopen",
                dlerror() == nullptr ? "dlopen failed" : dlerror());
  }
  const auto cpu_function =
    reinterpret_cast<CpuFunction>(dlsym(cpu_library, NCNN_VA6_MODEL_NAME));
  const auto create = reinterpret_cast<CreateFunction>(dlsym(
    vulkan_library,
    NCNN_VA6_STRINGIFY(NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_create))));
  const auto run = reinterpret_cast<RunFunction>(dlsym(
    vulkan_library,
    NCNN_VA6_STRINGIFY(NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_run))));
  const auto destroy = reinterpret_cast<DestroyFunction>(dlsym(
    vulkan_library,
    NCNN_VA6_STRINGIFY(NCNN_VA6_JOIN(NCNN_VA6_CROSS_ICD_MODEL, _vk_destroy))));
  if (cpu_function == nullptr || create == nullptr || run == nullptr ||
      destroy == nullptr) {
    dlclose(vulkan_library);
    dlclose(cpu_library);
    return fail(result_path, result, "symbols", "typed symbol lookup failed");
  }

  std::vector<float> cpu_output(elements);
  result.phase = "cpu-run";
  const int cpu_status = cpu_function(input.data(), cpu_output.data());
  result.create_status = "-1";
  result.cpu_status = std::to_string(cpu_status);
  if (cpu_status != 0) {
    dlclose(vulkan_library);
    dlclose(cpu_library);
    return fail(
      result_path, result, "cpu-run", "CPU model call returned nonzero status");
  }
  if (!write_floats(cpu_output_path, cpu_output)) {
    dlclose(vulkan_library);
    dlclose(cpu_library);
    return fail(
      result_path, result, "cpu-output", "cannot write CPU output tensor");
  }

  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  options.device_index = result.requested_device_index;
  VulkanContext* context = nullptr;
  std::array<char, kDiagnosticCapacity + 2> diagnostic_storage{};
  char* diagnostic = diagnostic_storage.data() + 1;
  diagnostic_storage.front() = '#';
  diagnostic_storage.back() = '#';
  result.phase = "vulkan-create";
  const ncnn_vk_status create_status =
    create(&options, &context, diagnostic, kDiagnosticCapacity);
  result.create_status = std::to_string(create_status);
  result.context_null = context == nullptr;
  result.diagnostic_bounded =
    diagnostic_storage.front() == '#' && diagnostic_storage.back() == '#';
  if (create_status != NCNN_VK_STATUS_SUCCESS) {
    const ncnn_vk_status destroy_null = destroy(nullptr);
    result.destroy_null_status = std::to_string(destroy_null);
    const bool acceptable = !expect_create_success && result.context_null &&
                            result.diagnostic_bounded &&
                            diagnostic[0] != '\0' &&
                            destroy_null == NCNN_VK_STATUS_SUCCESS;
    dlclose(vulkan_library);
    dlclose(cpu_library);
    if (!acceptable) {
      result.diagnostic = "unexpected create failure path";
      result.status = "failure";
      result.phase = "vulkan-create";
      (void)write_result(result_path, result);
      return 1;
    }
    result.status = "success";
    result.phase = "expected-create-failure";
    result.diagnostic = diagnostic;
    return write_result(result_path, result) ? 0 : 1;
  }
  if (!expect_create_success) {
    (void)destroy(context);
    dlclose(vulkan_library);
    dlclose(cpu_library);
    return fail(result_path,
                result,
                "vulkan-create",
                "create succeeded where a failure was required");
  }
  if (!result.diagnostic_bounded) {
    (void)destroy(context);
    dlclose(vulkan_library);
    dlclose(cpu_library);
    return fail(
      result_path, result, "diagnostic", "unbounded create diagnostic");
  }

  std::vector<float> vulkan_output(elements);
  result.phase = "vulkan-run";
  const ncnn_vk_status run_status = run(context,
                                        input.data(),
                                        input.size(),
                                        vulkan_output.data(),
                                        vulkan_output.size(),
                                        diagnostic,
                                        kDiagnosticCapacity);
  result.run_status = std::to_string(run_status);
  const ncnn_vk_status destroy_status = destroy(context);
  const ncnn_vk_status destroy_null_status = destroy(nullptr);
  result.destroy_status = std::to_string(destroy_status);
  result.destroy_null_status = std::to_string(destroy_null_status);
  dlclose(vulkan_library);
  dlclose(cpu_library);
  if (run_status != NCNN_VK_STATUS_SUCCESS ||
      destroy_status != NCNN_VK_STATUS_SUCCESS ||
      destroy_null_status != NCNN_VK_STATUS_SUCCESS) {
    return fail(result_path,
                result,
                result.phase,
                "run/destroy returned nonzero statuses");
  }
  if (!write_floats(vulkan_output_path, vulkan_output)) {
    return fail(result_path,
                result,
                "vulkan-output",
                "cannot write Vulkan output tensor");
  }
  result.status = "success";
  result.phase = "complete";
  result.diagnostic.clear();
  return write_result(result_path, result) ? 0 : 1;
}
