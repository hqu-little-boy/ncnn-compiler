#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include NCNN_FIXEDRUN_MODEL_HEADER

#define NCNN_FIXEDRUN_JOIN2(lhs, rhs) lhs##rhs
#define NCNN_FIXEDRUN_JOIN(lhs, rhs) NCNN_FIXEDRUN_JOIN2(lhs, rhs)
#define NCNN_FIXEDRUN_JOIN3(lhs, middle, rhs) \
  NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_JOIN(lhs, middle), rhs)
#define NCNN_FIXEDRUN_JOIN4(first, second, third, fourth) \
  NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_JOIN3(first, second, third), fourth)
#define NCNN_FIXEDRUN_STRING_IMPL(value) #value
#define NCNN_FIXEDRUN_STRING(value) NCNN_FIXEDRUN_STRING_IMPL(value)

#if NCNN_FIXEDRUN_BACKEND_VULKAN
using FixedRunContext = NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_context);
using FixedRunCreate =
  decltype(&NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_create));
using FixedRunRun = decltype(&NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_run));
using FixedRunDestroy =
  decltype(&NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_destroy));
#define FIXEDRUN_INPUT_COUNT \
  NCNN_FIXEDRUN_JOIN3(NCNN_FIXEDRUN_MODEL_UPPER, _VULKAN_, INPUT_COUNT)
#define FIXEDRUN_OUTPUT_COUNT \
  NCNN_FIXEDRUN_JOIN3(NCNN_FIXEDRUN_MODEL_UPPER, _VULKAN_, OUTPUT_COUNT)
#define FIXEDRUN_INPUT_ELEMENTS(index) \
  NCNN_FIXEDRUN_JOIN4(                 \
    NCNN_FIXEDRUN_MODEL_UPPER, _VULKAN_INPUT, index, _ELEMENTS)
#define FIXEDRUN_OUTPUT_ELEMENTS(index) \
  NCNN_FIXEDRUN_JOIN4(                  \
    NCNN_FIXEDRUN_MODEL_UPPER, _VULKAN_OUTPUT, index, _ELEMENTS)
#else
using FixedRunCpu = decltype(&NCNN_FIXEDRUN_MODEL);
#define FIXEDRUN_INPUT_COUNT \
  NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL_UPPER, _INPUT_COUNT)
#define FIXEDRUN_OUTPUT_COUNT \
  NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL_UPPER, _OUTPUT_COUNT)
#define FIXEDRUN_INPUT_ELEMENTS(index) \
  NCNN_FIXEDRUN_JOIN4(NCNN_FIXEDRUN_MODEL_UPPER, _INPUT, index, _ELEMENTS)
#define FIXEDRUN_OUTPUT_ELEMENTS(index) \
  NCNN_FIXEDRUN_JOIN4(NCNN_FIXEDRUN_MODEL_UPPER, _OUTPUT, index, _ELEMENTS)
#endif

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kInputCount = FIXEDRUN_INPUT_COUNT;
constexpr std::size_t kOutputCount = FIXEDRUN_OUTPUT_COUNT;
static_assert(kInputCount > 0 && kInputCount <= 4);
static_assert(kOutputCount > 0 && kOutputCount <= 4);

constexpr std::array<std::size_t, kInputCount> kInputElements{
  FIXEDRUN_INPUT_ELEMENTS(1),
#if FIXEDRUN_INPUT_COUNT > 1
  FIXEDRUN_INPUT_ELEMENTS(2),
#endif
#if FIXEDRUN_INPUT_COUNT > 2
  FIXEDRUN_INPUT_ELEMENTS(3),
#endif
#if FIXEDRUN_INPUT_COUNT > 3
  FIXEDRUN_INPUT_ELEMENTS(4),
#endif
};
constexpr std::array<std::size_t, kOutputCount> kOutputElements{
  FIXEDRUN_OUTPUT_ELEMENTS(1),
#if FIXEDRUN_OUTPUT_COUNT > 1
  FIXEDRUN_OUTPUT_ELEMENTS(2),
#endif
#if FIXEDRUN_OUTPUT_COUNT > 2
  FIXEDRUN_OUTPUT_ELEMENTS(3),
#endif
#if FIXEDRUN_OUTPUT_COUNT > 3
  FIXEDRUN_OUTPUT_ELEMENTS(4),
#endif
};

struct Invocation final {
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> outputs;
  std::vector<fs::path> output_paths;
  std::uint32_t status;
  std::uint64_t elapsed_ns;
  std::string diagnostic;
};

#if NCNN_FIXEDRUN_BACKEND_VULKAN
struct LifecycleSnapshot final {
  std::vector<std::uint32_t> create_statuses;
  std::vector<std::uint32_t> run_statuses;
  std::vector<std::uint32_t> destroy_statuses;
  bool overflow;
};

class LifecycleTrace final {
 public:
  LifecycleTrace();
  void recordCreate(std::uint32_t status) noexcept;
  void recordRun(std::uint32_t status) noexcept;
  void recordDestroy(std::uint32_t status) noexcept;
  [[nodiscard]] LifecycleSnapshot get_snapshot() const;

 private:
  std::vector<std::uint32_t> create_statuses_;
  std::vector<std::uint32_t> run_statuses_;
  std::vector<std::uint32_t> destroy_statuses_;
  bool overflow_;
  mutable std::mutex mutex_;
};

LifecycleTrace::LifecycleTrace()
  : create_statuses_(),
    run_statuses_(),
    destroy_statuses_(),
    overflow_(false),
    mutex_() {
  create_statuses_.reserve(2);
  run_statuses_.reserve(16);
  destroy_statuses_.reserve(2);
}

void LifecycleTrace::recordCreate(std::uint32_t status) noexcept {
  const std::scoped_lock lock(mutex_);
  if (create_statuses_.size() == create_statuses_.capacity()) {
    overflow_ = true;
    return;
  }
  create_statuses_.push_back(status);
}

void LifecycleTrace::recordRun(std::uint32_t status) noexcept {
  const std::scoped_lock lock(mutex_);
  if (run_statuses_.size() == run_statuses_.capacity()) {
    overflow_ = true;
    return;
  }
  run_statuses_.push_back(status);
}

void LifecycleTrace::recordDestroy(std::uint32_t status) noexcept {
  const std::scoped_lock lock(mutex_);
  if (destroy_statuses_.size() == destroy_statuses_.capacity()) {
    overflow_ = true;
    return;
  }
  destroy_statuses_.push_back(status);
}

LifecycleSnapshot LifecycleTrace::get_snapshot() const {
  const std::scoped_lock lock(mutex_);
  return {.create_statuses = create_statuses_,
          .run_statuses = run_statuses_,
          .destroy_statuses = destroy_statuses_,
          .overflow = overflow_};
}
#endif

struct Arguments final {
  fs::path library;
  std::string mode;
  std::vector<fs::path> inputs_a;
  std::vector<fs::path> outputs_a;
  std::vector<fs::path> inputs_b;
  std::vector<fs::path> outputs_b;
};

std::vector<float> readFloats(const fs::path& path, std::size_t expected) {
  if (expected > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::runtime_error("tensor byte count overflows");
  }
  const std::size_t bytes = expected * sizeof(float);
  if (bytes >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("fixed-run raw input exceeds streamsize");
  }
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open tensor input " + path.string());
  }
  const std::streamoff length = stream.tellg();
  if (!std::cmp_equal(length, bytes)) {
    throw std::runtime_error("tensor byte size differs from its static shape");
  }
  std::vector<unsigned char> raw(bytes);
  stream.seekg(0);
  if (bytes != 0 && !stream.read(reinterpret_cast<char*>(raw.data()),
                                 static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("cannot read tensor input " + path.string());
  }
  std::vector<float> values(expected);
  for (std::size_t index = 0; index < expected; ++index) {
    const std::size_t offset = index * sizeof(float);
    const std::uint32_t bits =
      static_cast<std::uint32_t>(raw[offset]) |
      (static_cast<std::uint32_t>(raw[offset + 1]) << 8) |
      (static_cast<std::uint32_t>(raw[offset + 2]) << 16) |
      (static_cast<std::uint32_t>(raw[offset + 3]) << 24);
    values[index] = std::bit_cast<float>(bits);
  }
  return values;
}

void writeFloats(const fs::path& path, std::span<const float> values) {
  if (values.size_bytes() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("fixed-run raw output exceeds streamsize");
  }
  std::error_code error;
  fs::create_directories(path.parent_path(), error);
  if (error) {
    throw std::runtime_error("cannot create output directory: " +
                             error.message());
  }
  std::vector<unsigned char> raw;
  raw.reserve(values.size_bytes());
  for (float value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    raw.push_back(static_cast<unsigned char>(bits & 0xffU));
    raw.push_back(static_cast<unsigned char>((bits >> 8) & 0xffU));
    raw.push_back(static_cast<unsigned char>((bits >> 16) & 0xffU));
    raw.push_back(static_cast<unsigned char>((bits >> 24) & 0xffU));
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream || (!raw.empty() &&
                  !stream.write(reinterpret_cast<const char*>(raw.data()),
                                static_cast<std::streamsize>(raw.size())))) {
    throw std::runtime_error("cannot write tensor output " + path.string());
  }
}

Arguments parseArguments(int argc, char** argv) {
  Arguments result{.library = {},
                   .mode = "single",
                   .inputs_a = {},
                   .outputs_a = {},
                   .inputs_b = {},
                   .outputs_b = {}};
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    auto takePath = [&]() -> fs::path {
      if (++index >= argc) {
        throw std::runtime_error("missing value for " + std::string(option));
      }
      return argv[index];
    };
    if (option == "--library") {
      result.library = takePath();
    } else if (option == "--mode") {
      if (++index >= argc) {
        throw std::runtime_error("missing value for --mode");
      }
      result.mode = argv[index];
    } else if (option == "--input" || option == "--input-a") {
      result.inputs_a.push_back(takePath());
    } else if (option == "--output" || option == "--output-a") {
      result.outputs_a.push_back(takePath());
    } else if (option == "--input-b") {
      result.inputs_b.push_back(takePath());
    } else if (option == "--output-b") {
      result.outputs_b.push_back(takePath());
    } else {
      throw std::runtime_error("unexpected argument: " + std::string(option));
    }
  }
  if (result.library.empty() ||
      (result.mode != "single" && result.mode != "repeat" &&
       result.mode != "concurrent" && result.mode != "interleaved")) {
    throw std::runtime_error(
      "usage: fixedrun_native_loader --library DSO --mode "
      "single|concurrent|interleaved "
      "[--input FILE --output FILE]...");
  }
  return result;
}

std::vector<std::vector<float>> readTensorSet(
  std::span<const fs::path> paths,
  const std::array<std::size_t, kInputCount>& element_counts) {
  if (paths.size() != kInputCount) {
    throw std::runtime_error("input file count differs from the model ABI");
  }
  std::vector<std::vector<float>> inputs;
  inputs.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    inputs.push_back(readFloats(paths[index], element_counts[index]));
  }
  return inputs;
}

std::vector<std::vector<float>> makeOutputs() {
  std::vector<std::vector<float>> outputs;
  outputs.reserve(kOutputCount);
  for (std::size_t count : kOutputElements) {
    outputs.emplace_back(count, -9876.5F);
  }
  return outputs;
}

template <typename Run,
          typename Context,
          std::size_t... InputIndices,
          std::size_t... OutputIndices>
std::uint32_t invokeRun(Run run,
                        Context context,
                        std::vector<std::vector<float>>& inputs,
                        std::vector<std::vector<float>>& outputs,
                        char* diagnostic,
                        std::size_t diagnostic_capacity,
                        std::index_sequence<InputIndices...>,
                        std::index_sequence<OutputIndices...>) {
  auto input_arguments = std::tuple_cat(std::make_tuple(
    inputs[InputIndices].data(), inputs[InputIndices].size())...);
  auto output_arguments = std::tuple_cat(std::make_tuple(
    outputs[OutputIndices].data(), outputs[OutputIndices].size())...);
  auto arguments =
    std::tuple_cat(std::move(input_arguments), std::move(output_arguments));
  return std::apply(
    [&](auto... values) {
      return static_cast<std::uint32_t>(
        run(context, values..., diagnostic, diagnostic_capacity));
    },
    arguments);
}

template <typename Function,
          std::size_t... InputIndices,
          std::size_t... OutputIndices>
std::uint32_t invokeCpu(Function function,
                        std::vector<std::vector<float>>& inputs,
                        std::vector<std::vector<float>>& outputs,
                        std::index_sequence<InputIndices...>,
                        std::index_sequence<OutputIndices...>) {
  auto input_arguments = std::tuple_cat(
    std::make_tuple(static_cast<const float*>(inputs[InputIndices].data()))...);
  auto output_arguments =
    std::tuple_cat(std::make_tuple(outputs[OutputIndices].data())...);
  auto arguments =
    std::tuple_cat(std::move(input_arguments), std::move(output_arguments));
  return std::apply(
    [&](auto... values) {
      return static_cast<std::uint32_t>(function(values...));
    },
    arguments);
}

class DynamicLibrary final {
 public:
  explicit DynamicLibrary(const fs::path& path)
    : handle_(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
    if (handle_ == nullptr) {
      throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
  }
  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  ~DynamicLibrary() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  template <typename Function>
  Function get(std::string_view name) const {
    dlerror();
    void* symbol = dlsym(handle_, std::string(name).c_str());
    if (const char* error = dlerror(); error != nullptr) {
      throw std::runtime_error(std::string("dlsym failed: ") + error);
    }
    return reinterpret_cast<Function>(symbol);
  }

 private:
  void* handle_;
};

#if NCNN_FIXEDRUN_BACKEND_VULKAN
class ContextDeleter final {
 public:
  ContextDeleter(FixedRunDestroy destroy,
                 std::reference_wrapper<LifecycleTrace> trace) noexcept;
  void operator()(FixedRunContext* context) const noexcept;

 private:
  FixedRunDestroy destroy_;
  std::reference_wrapper<LifecycleTrace> trace_;
};

ContextDeleter::ContextDeleter(
  FixedRunDestroy destroy,
  std::reference_wrapper<LifecycleTrace> trace) noexcept
  : destroy_(destroy), trace_(trace) {}

void ContextDeleter::operator()(FixedRunContext* context) const noexcept {
  if (context != nullptr) {
    trace_.get().recordDestroy(destroy_(context));
  }
}
using ContextOwner = std::unique_ptr<FixedRunContext, ContextDeleter>;
#endif

std::uint64_t elapsedNanoseconds(
  std::chrono::steady_clock::time_point started) {
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started)
      .count());
}

std::string escapeJson(std::string_view value) {
  std::string escaped;
  constexpr char kHexDigits[] = "0123456789abcdef";
  for (unsigned char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
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
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      default:
        if (character < 0x20U) {
          escaped += "\\u00";
          escaped += kHexDigits[character >> 4];
          escaped += kHexDigits[character & 0x0fU];
        } else {
          escaped += static_cast<char>(character);
        }
        break;
    }
  }
  return escaped;
}

void printResult(std::string_view event,
                 std::uint32_t status,
                 std::uint64_t elapsed_ns,
                 std::string_view diagnostic,
                 std::span<const fs::path> output_paths = {}) {
  std::cout << R"({"event":")" << escapeJson(event) << R"(","status":)"
            << status << R"(,"elapsed_ns":)" << elapsed_ns
            << R"(,"diagnostic":")" << escapeJson(diagnostic)
            << R"(","output_paths":[)";
  for (std::size_t index = 0; index < output_paths.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << '\"' << escapeJson(output_paths[index].string()) << '\"';
  }
  std::cout << "]}\n";
}

#if NCNN_FIXEDRUN_BACKEND_VULKAN
void printLifecycle(const LifecycleTrace& trace) {
  const LifecycleSnapshot snapshot = trace.get_snapshot();
  std::cout << R"({"event":"lifecycle","overflow":)"
            << (snapshot.overflow ? "true" : "false") << R"(,"create_count":)"
            << snapshot.create_statuses.size() << R"(,"create_statuses":[)";
  for (std::size_t index = 0; index < snapshot.create_statuses.size();
       ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << snapshot.create_statuses[index];
  }
  std::cout << R"(],"run_count":)" << snapshot.run_statuses.size()
            << R"(,"run_statuses":[)";
  for (std::size_t index = 0; index < snapshot.run_statuses.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << snapshot.run_statuses[index];
  }
  std::cout << R"(],"destroy_count":)" << snapshot.destroy_statuses.size()
            << R"(,"destroy_statuses":[)";
  for (std::size_t index = 0; index < snapshot.destroy_statuses.size();
       ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << snapshot.destroy_statuses[index];
  }
  std::cout << "]}\n";
}

class LifecycleReporter final {
 public:
  explicit LifecycleReporter(std::reference_wrapper<const LifecycleTrace> trace)
    : trace_(trace) {}
  LifecycleReporter(const LifecycleReporter&) = delete;
  LifecycleReporter& operator=(const LifecycleReporter&) = delete;
  ~LifecycleReporter() { printLifecycle(trace_.get()); }

 private:
  std::reference_wrapper<const LifecycleTrace> trace_;
};
#else
void printCpuLifecycle(std::uint32_t run_status) {
  std::cout << R"({"event":"lifecycle","overflow":false,)"
               R"("create_count":0,"create_statuses":[],)"
               R"("run_count":1,"run_statuses":[)"
            << run_status << R"(],"destroy_count":0,"destroy_statuses":[]})"
            << '\n';
}
#endif

#if NCNN_FIXEDRUN_BACKEND_VULKAN
Invocation prepareVulkan(std::span<const fs::path> input_paths,
                         std::span<const fs::path> output_paths) {
  if (output_paths.size() != kOutputCount) {
    throw std::runtime_error("output file count differs from the model ABI");
  }
  return Invocation{.inputs = readTensorSet(input_paths, kInputElements),
                    .outputs = makeOutputs(),
                    .output_paths = {output_paths.begin(), output_paths.end()},
                    .status = 0,
                    .elapsed_ns = 0,
                    .diagnostic = std::string(1024, '\0')};
}

template <typename Run>
void invokeVulkan(Run run,
                  FixedRunContext* context,
                  Invocation& result,
                  LifecycleTrace& trace) {
  const auto started = std::chrono::steady_clock::now();
  result.status = invokeRun(run,
                            context,
                            result.inputs,
                            result.outputs,
                            result.diagnostic.data(),
                            result.diagnostic.size(),
                            std::make_index_sequence<kInputCount>{},
                            std::make_index_sequence<kOutputCount>{});
  trace.recordRun(result.status);
  result.elapsed_ns = elapsedNanoseconds(started);
  const std::size_t terminator = result.diagnostic.find('\0');
  if (terminator != std::string::npos) {
    result.diagnostic.resize(terminator);
  }
  if (result.status != 0) {
    return;
  }
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    writeFloats(result.output_paths[index], result.outputs[index]);
  }
}
#else
template <typename Function>
Invocation executeCpu(Function function,
                      std::span<const fs::path> input_paths,
                      std::span<const fs::path> output_paths) {
  if (output_paths.size() != kOutputCount) {
    throw std::runtime_error("output file count differs from the model ABI");
  }
  Invocation result{.inputs = readTensorSet(input_paths, kInputElements),
                    .outputs = makeOutputs(),
                    .output_paths = {output_paths.begin(), output_paths.end()},
                    .status = 0,
                    .elapsed_ns = 0,
                    .diagnostic = {}};
  const auto started = std::chrono::steady_clock::now();
  result.status = invokeCpu(function,
                            result.inputs,
                            result.outputs,
                            std::make_index_sequence<kInputCount>{},
                            std::make_index_sequence<kOutputCount>{});
  result.elapsed_ns = elapsedNanoseconds(started);
  if (result.status == 0) {
    for (std::size_t index = 0; index < kOutputCount; ++index) {
      writeFloats(output_paths[index], result.outputs[index]);
    }
  }
  return result;
}
#endif

void validatePaths(std::span<const fs::path> inputs,
                   std::span<const fs::path> outputs) {
  if (inputs.size() != kInputCount || outputs.size() != kOutputCount) {
    throw std::runtime_error(
      "input/output path counts differ from the model ABI");
  }
}

#if NCNN_FIXEDRUN_BACKEND_VULKAN
void emitInvocations(const std::vector<Invocation>& invocations) {
  for (const Invocation& invocation : invocations) {
    printResult("run",
                invocation.status,
                invocation.elapsed_ns,
                invocation.diagnostic,
                invocation.output_paths);
  }
}
#endif

}  // namespace

fs::path withSuffix(const fs::path& path, std::string_view suffix) {
  return path.parent_path() / (path.stem().string() + std::string(suffix) +
                               path.extension().string());
}

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
#if NCNN_FIXEDRUN_BACKEND_VULKAN
    DynamicLibrary library(arguments.library);
    const auto create = library.get<FixedRunCreate>(NCNN_FIXEDRUN_STRING(
      NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_create)));
    const auto run = library.get<FixedRunRun>(
      NCNN_FIXEDRUN_STRING(NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_run)));
    const auto destroy = library.get<FixedRunDestroy>(NCNN_FIXEDRUN_STRING(
      NCNN_FIXEDRUN_JOIN(NCNN_FIXEDRUN_MODEL, _vk_destroy)));
    if (arguments.mode == "single") {
      validatePaths(arguments.inputs_a, arguments.outputs_a);
    } else {
      validatePaths(arguments.inputs_a, arguments.outputs_a);
      validatePaths(arguments.inputs_b, arguments.outputs_b);
    }
    Invocation prepared_a =
      prepareVulkan(arguments.inputs_a, arguments.outputs_a);
    std::optional<Invocation> prepared_b;
    if (arguments.mode != "single") {
      prepared_b.emplace(
        prepareVulkan(arguments.inputs_b, arguments.outputs_b));
    }
    std::optional<Invocation> prepared_a_repeat;
    std::optional<Invocation> prepared_b_repeat;
    if (arguments.mode == "interleaved") {
      std::vector<fs::path> output_a_repeat;
      std::vector<fs::path> output_b_repeat;
      output_a_repeat.reserve(arguments.outputs_a.size());
      output_b_repeat.reserve(arguments.outputs_b.size());
      for (const fs::path& output : arguments.outputs_a) {
        output_a_repeat.push_back(withSuffix(output, ".interleaved-1"));
      }
      for (const fs::path& output : arguments.outputs_b) {
        output_b_repeat.push_back(withSuffix(output, ".interleaved-1"));
      }
      prepared_a_repeat.emplace(
        prepareVulkan(arguments.inputs_a, output_a_repeat));
      prepared_b_repeat.emplace(
        prepareVulkan(arguments.inputs_b, output_b_repeat));
    }

    LifecycleTrace trace;
    LifecycleReporter reporter(trace);
    ncnn_vk_options options;
    ncnn_vk_options_init(&options);
    options.device_index = 0;
    std::array<char, 1024> diagnostic{};
    FixedRunContext* raw_context_a = nullptr;
    const std::uint32_t create_a_status =
      create(&options, &raw_context_a, diagnostic.data(), diagnostic.size());
    trace.recordCreate(create_a_status);
    ContextOwner context_a(raw_context_a,
                           ContextDeleter(destroy, std::ref(trace)));
    if (create_a_status != NCNN_VK_STATUS_SUCCESS || raw_context_a == nullptr) {
      printResult("create", create_a_status, 0, diagnostic.data());
      return 2;
    }
    ContextOwner context_b(nullptr, ContextDeleter(destroy, std::ref(trace)));
    if (arguments.mode == "interleaved") {
      diagnostic.fill('\0');
      FixedRunContext* raw_context_b = nullptr;
      const std::uint32_t create_b_status =
        create(&options, &raw_context_b, diagnostic.data(), diagnostic.size());
      trace.recordCreate(create_b_status);
      context_b.reset(raw_context_b);
      if (create_b_status != NCNN_VK_STATUS_SUCCESS ||
          raw_context_b == nullptr) {
        printResult("create", create_b_status, 0, diagnostic.data());
        return 2;
      }
    }

    std::vector<Invocation> invocations;
    if (arguments.mode == "single" || arguments.mode == "repeat") {
      invokeVulkan(run, context_a.get(), prepared_a, trace);
      invocations.push_back(std::move(prepared_a));
      if (prepared_b) {
        invokeVulkan(run, context_a.get(), *prepared_b, trace);
        invocations.push_back(std::move(*prepared_b));
      }
    } else if (arguments.mode == "concurrent") {
      std::latch ready(2);
      std::latch start(1);
      std::array<std::exception_ptr, 2> worker_errors{};
      std::vector<std::jthread> workers;
      workers.reserve(2);
      auto worker = [&](std::size_t index, Invocation& invocation) {
        ready.count_down();
        start.wait();
        try {
          invokeVulkan(run, context_a.get(), invocation, trace);
        } catch (...) {
          worker_errors[index] = std::current_exception();
        }
      };
      try {
        workers.emplace_back(worker, 0U, std::ref(prepared_a));
        workers.emplace_back(worker, 1U, std::ref(*prepared_b));
      } catch (...) {
        start.count_down();
        workers.clear();
        throw;
      }
      ready.wait();
      start.count_down();
      workers.clear();
      for (const auto& error : worker_errors) {
        if (error != nullptr) {
          std::rethrow_exception(error);
        }
      }
      invocations.push_back(std::move(prepared_a));
      invocations.push_back(std::move(*prepared_b));
    } else {
      invokeVulkan(run, context_a.get(), prepared_a, trace);
      invokeVulkan(run, context_b.get(), *prepared_b, trace);
      invokeVulkan(run, context_a.get(), *prepared_a_repeat, trace);
      invokeVulkan(run, context_b.get(), *prepared_b_repeat, trace);
      invocations.push_back(std::move(prepared_a));
      invocations.push_back(std::move(*prepared_b));
      invocations.push_back(std::move(*prepared_a_repeat));
      invocations.push_back(std::move(*prepared_b_repeat));
    }
    const bool failed = std::ranges::any_of(invocations, [](const auto& item) {
      return item.status != NCNN_VK_STATUS_SUCCESS;
    });
    emitInvocations(invocations);
    context_b.reset();
    context_a.reset();
    const LifecycleSnapshot trace_snapshot = trace.get_snapshot();
    if (trace_snapshot.overflow ||
        trace_snapshot.destroy_statuses.size() !=
          trace_snapshot.create_statuses.size() ||
        trace_snapshot.run_statuses.size() != invocations.size()) {
      return 3;
    }
    const bool cleanup_failed = std::ranges::any_of(
      trace_snapshot.destroy_statuses,
      [](std::uint32_t status) { return status != NCNN_VK_STATUS_SUCCESS; });
    if (cleanup_failed) {
      return 3;
    }
    return failed ? 4 : 0;
#else
    if (arguments.mode != "single") {
      throw std::runtime_error("CPU AOT loader supports a single invocation");
    }
    DynamicLibrary library(arguments.library);
    const auto function =
      library.get<FixedRunCpu>(NCNN_FIXEDRUN_STRING(NCNN_FIXEDRUN_MODEL));
    validatePaths(arguments.inputs_a, arguments.outputs_a);
    Invocation result =
      executeCpu(function, arguments.inputs_a, arguments.outputs_a);
    printResult("run",
                result.status,
                result.elapsed_ns,
                result.diagnostic,
                result.output_paths);
    printCpuLifecycle(result.status);
    return result.status == 0 ? 0 : 4;
#endif
  } catch (const std::exception& error) {
    std::cerr << "fixedrun native loader: " << error.what() << '\n';
    return 1;
  }
}
