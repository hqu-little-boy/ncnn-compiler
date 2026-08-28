#include "performance_test_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <numeric>
#include <optional>
#include <print>
#include <utility>

#include "cpu.h"
#include "net.h"

namespace ncnn_compiler::test {
namespace {

// 环境变量整数解析：不存在或为空视为未设置（nullopt）；存在但不是完整
// 十进制整数视为配置错误。
std::expected<std::optional<int>, std::string>
parse_optional_integer_environment(const char* variable_name) {
  const char* raw = std::getenv(variable_name);
  if (raw == nullptr || *raw == '\0') {
    return std::optional<int>();
  }
  char* parse_end = nullptr;
  const long value = std::strtol(raw, &parse_end, 10);
  if (parse_end == raw || *parse_end != '\0') {
    return std::unexpected(
      std::format("{}='{}' is not an integer", variable_name, raw));
  }
  return std::optional<int>(static_cast<int>(value));
}

std::expected<ncnn::Mat, std::string> make_reference_input_mat(
  const ReferenceInput& input) {
  const TensorShape& shape = input.get_shape();
  auto input_elements = shape.element_count();
  if (!input_elements) {
    return std::unexpected(std::format(
      "input blob '{}': {}", input.get_blob_name(), input_elements.error()));
  }
  auto input_bytes = shape.byte_count(sizeof(float));
  if (!input_bytes) {
    return std::unexpected(std::format(
      "input blob '{}': {}", input.get_blob_name(), input_bytes.error()));
  }
  if (input.get_value_count() != *input_elements ||
      input.get_bytes().size() != *input_elements * sizeof(float)) {
    return std::unexpected(
      std::format("input blob '{}': value count does not match its shape",
                  input.get_blob_name()));
  }
  switch (shape.get_rank()) {
    case 1:
      return ncnn::Mat(shape.get_width());
    case 2:
      return ncnn::Mat(shape.get_width(), shape.get_height());
    case 3:
      return ncnn::Mat(
        shape.get_width(), shape.get_height(), shape.get_channels());
    default:
      return std::unexpected("input blob rank must be 1 through 3");
  }
  // ncnn::Mat 构造失败时是空 Mat，由调用方检查。
}

void fill_reference_input_mat(ncnn::Mat& mat, const ReferenceInput& input) {
  const TensorShape& shape = input.get_shape();
  auto input_bytes = shape.byte_count(sizeof(float));
  if (shape.get_rank() < 3) {
    std::memcpy(mat.data, input.get_bytes().data(), *input_bytes);
    return;
  }
  const std::size_t channel_bytes =
    *input_bytes / static_cast<std::size_t>(shape.get_channels());
  for (int channel = 0; channel < shape.get_channels(); ++channel) {
    std::memcpy(mat.channel(channel),
                input.get_bytes().data() +
                  (static_cast<std::size_t>(channel) * channel_bytes),
                channel_bytes);
  }
}

// 按 ncnn 输出 blob 的 channel-major 布局展平进调用方缓冲。
std::optional<std::string> flatten_ncnn_output(
  const ncnn::Mat& output,
  std::string_view blob_name,
  std::vector<float>& destination) {
  if (output.elempack != 1 || output.elemsize != sizeof(float)) {
    return std::format("ncnn output blob '{}' is not unpacked float32",
                       blob_name);
  }
  if (output.w < 0 || output.h < 0 || output.d < 0 || output.c < 0) {
    return std::format("ncnn output blob '{}' has a negative dimension",
                       blob_name);
  }
  const std::size_t channel_elements = static_cast<std::size_t>(output.w) *
                                       static_cast<std::size_t>(output.h) *
                                       static_cast<std::size_t>(output.d);
  const std::size_t logical_elements =
    channel_elements * static_cast<std::size_t>(output.c);
  if (destination.size() != logical_elements) {
    return std::format(
      "ncnn output blob '{}' has {} elements but caller provided {}",
      blob_name,
      logical_elements,
      destination.size());
  }
  if (output.dims >= 3) {
    std::size_t offset = 0;
    for (int channel = 0; channel < output.c; ++channel) {
      const float* begin = output.channel(channel);
      std::copy(begin,
                begin + channel_elements,
                destination.begin() + static_cast<std::ptrdiff_t>(offset));
      offset += channel_elements;
    }
  } else {
    const auto* begin = static_cast<const float*>(output.data);
    std::copy(begin, begin + logical_elements, destination.begin());
  }
  return std::nullopt;
}

TimingStats summarize_samples(const std::vector<double>& samples) {
  TimingStats stats;
  const auto count = static_cast<double>(samples.size());
  stats.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / count;
  stats.minimum_ms = *std::ranges::min_element(samples);
  std::vector<double> sorted(samples);
  const auto median_position =
    sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2);
  std::nth_element(sorted.begin(), median_position, sorted.end());
  stats.median_ms = *median_position;
  const double squared_deviation =
    std::accumulate(samples.begin(),
                    samples.end(),
                    0.0,
                    [mean = stats.mean_ms](double accumulator, double sample) {
                      const double deviation = sample - mean;
                      return accumulator + (deviation * deviation);
                    });
  stats.stddev_ms = std::sqrt(squared_deviation / count);
  stats.coefficient_of_variation =
    stats.mean_ms > 0.0 ? stats.stddev_ms / stats.mean_ms : 0.0;
  return stats;
}

}  // namespace

std::expected<int, std::string> resolve_benchmark_thread_count() {
  auto override_threads =
    parse_optional_integer_environment("NCNN_PERF_THREADS");
  if (!override_threads) {
    return std::unexpected(override_threads.error());
  }
  const int threads =
    override_threads->value_or(ncnn::get_physical_big_cpu_count());
  if (threads <= 0) {
    return std::unexpected(
      std::format("benchmark needs at least one thread (threads={})", threads));
  }
  return threads;
}

void apply_benchncnn_threading(int threads) {
  ncnn::set_cpu_powersave(2);
  ncnn::set_omp_dynamic(0);
  ncnn::set_omp_num_threads(threads);
  // 兜底：若 dlopen 进来的 .so 携带独立的 OpenMP 运行时副本，其初始化会读取
  // 该环境变量。主实例已通过 set_omp_num_threads 生效，此处尽力而为即可。
  const std::string thread_text = std::to_string(threads);
#ifdef _WIN32
  _putenv_s("OMP_NUM_THREADS", thread_text.c_str());
#else
  ::setenv("OMP_NUM_THREADS", thread_text.c_str(), 1);
#endif
}

std::expected<TimingPolicy, std::string> resolve_timing_policy(
  TimingPolicy default_policy) {
  auto warmup = parse_optional_integer_environment("NCNN_PERF_WARMUP");
  if (!warmup) {
    return std::unexpected(warmup.error());
  }
  auto iterations = parse_optional_integer_environment("NCNN_PERF_ITERS");
  if (!iterations) {
    return std::unexpected(iterations.error());
  }
  if (warmup->has_value()) {
    default_policy.warmup_iterations = **warmup;
  }
  if (iterations->has_value()) {
    default_policy.timed_iterations = **iterations;
  }
  if (default_policy.warmup_iterations < 0 ||
      default_policy.timed_iterations <= 0) {
    return std::unexpected(
      "timing policy needs warmup >= 0 and iterations > 0");
  }
  return default_policy;
}

std::expected<TimingStats, std::string> time_repeated_inference(
  const std::function<int()>& inference, const TimingPolicy& policy) {
  for (int iteration = 0; iteration < policy.warmup_iterations; ++iteration) {
    if (inference() != 0) {
      return std::unexpected("warmup inference iteration failed");
    }
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(policy.timed_iterations));
  for (int iteration = 0; iteration < policy.timed_iterations; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    const int status = inference();
    const auto end = std::chrono::steady_clock::now();
    if (status != 0) {
      return std::unexpected("timed inference iteration failed");
    }
    samples.push_back(
      std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return summarize_samples(samples);
}

struct NcnnBenchRunner::Impl final {
  Impl(std::string_view param_path,
       std::string_view bin_path,
       int num_threads,
       ReferenceInferenceMode mode) {
    // opt 块与 run_ncnn_reference 的对应推理路径逐字对齐，保证计时对象与
    // 数值金标参考完全同构。
    network.opt.lightmode = false;
    network.opt.use_vulkan_compute = false;
    network.opt.use_int8_inference = mode == ReferenceInferenceMode::Int8;
    network.opt.use_fp16_packed = false;
    network.opt.use_fp16_storage = false;
    network.opt.use_fp16_arithmetic = false;
    network.opt.use_bf16_packed = false;
    network.opt.use_bf16_storage = false;
    network.opt.flush_denormals = 0;
    network.opt.num_threads = num_threads;
    if (network.load_param(std::string(param_path).c_str()) != 0) {
      error = "ncnn failed to load param file";
      return;
    }
    if (network.load_model(std::string(bin_path).c_str()) != 0) {
      error = "ncnn failed to load bin file";
    }
  }

  ncnn::Net network;
  mutable std::string error;
};

NcnnBenchRunner::NcnnBenchRunner(std::string_view param_path,
                                 std::string_view bin_path,
                                 int num_threads,
                                 ReferenceInferenceMode mode)
  : impl_(new Impl(param_path, bin_path, num_threads, mode)) {}

NcnnBenchRunner::~NcnnBenchRunner() {
  delete impl_;
}

bool NcnnBenchRunner::valid() const noexcept {
  return impl_->error.empty();
}

std::string_view NcnnBenchRunner::error() const noexcept {
  return impl_->error;
}

int NcnnBenchRunner::run(std::span<const ReferenceInput> inputs,
                         std::span<const std::string_view> output_blob_names,
                         std::span<std::vector<float>> outputs) const {
  if (!valid()) {
    return -1;
  }
  impl_->error.clear();
  if (inputs.empty() || output_blob_names.empty() ||
      outputs.size() < output_blob_names.size()) {
    impl_->error =
      "bench runner requires inputs, output names and matching buffers";
    return -1;
  }
  ncnn::Extractor extractor = impl_->network.create_extractor();
  std::vector<ncnn::Mat> input_mats;
  input_mats.reserve(inputs.size());
  for (const ReferenceInput& input : inputs) {
    auto mat = make_reference_input_mat(input);
    if (!mat) {
      impl_->error = std::move(mat).error();
      return -1;
    }
    fill_reference_input_mat(*mat, input);
    if (extractor.input(std::string(input.get_blob_name()).c_str(), *mat) !=
        0) {
      impl_->error = std::format("ncnn failed to bind input blob '{}'",
                                 input.get_blob_name());
      return -1;
    }
    input_mats.push_back(std::move(*mat));
  }
  for (std::size_t index = 0; index < output_blob_names.size(); ++index) {
    const std::string_view blob_name = output_blob_names[index];
    ncnn::Mat output;
    if (extractor.extract(std::string(blob_name).c_str(), output) != 0) {
      impl_->error =
        std::format("ncnn failed to extract output blob '{}'", blob_name);
      return -1;
    }
    if (auto failure = flatten_ncnn_output(output, blob_name, outputs[index])) {
      impl_->error = std::move(*failure);
      return -1;
    }
  }
  return 0;
}

void emit_performance_report(std::string_view model,
                             int threads,
                             const TimingPolicy& policy,
                             const PairBenchmarkResult& result) {
  const double coefficient_of_variation =
    std::max(result.ncnn.coefficient_of_variation,
             result.compiled.coefficient_of_variation);
  std::println(
    "PERF model={} threads={} warmup={} iters={} "
    "ncnn_ms={:.3f} compiled_ms={:.3f} ratio={:.3f} "
    "ncnn_min_ms={:.3f} compiled_min_ms={:.3f} cv={:.4f}",
    model,
    threads,
    policy.warmup_iterations,
    policy.timed_iterations,
    result.ncnn.mean_ms,
    result.compiled.mean_ms,
    result.ratio,
    result.ncnn.minimum_ms,
    result.compiled.minimum_ms,
    coefficient_of_variation);
}

void append_performance_json_record(std::string_view model,
                                    int threads,
                                    const TimingPolicy& policy,
                                    const PairBenchmarkResult& result) {
  const char* path = std::getenv("NCNN_PERF_JSON");
  if (path == nullptr || *path == '\0') {
    return;
  }
  // 模型名均为 [A-Za-z0-9_] 标识符，无需 JSON 转义。
  std::ofstream stream(path, std::ios::app);
  stream << std::format(
    R"({{"model":"{}","threads":{},"warmup":{},"iterations":{},)"
    R"("ncnn_mean_ms":{:.3f},"ncnn_min_ms":{:.3f},"ncnn_median_ms":{:.3f},)"
    R"("compiled_mean_ms":{:.3f},"compiled_min_ms":{:.3f},)"
    R"("compiled_median_ms":{:.3f},"cv":{:.4f},"ratio":{:.3f}}})"
    "\n",
    model,
    threads,
    policy.warmup_iterations,
    policy.timed_iterations,
    result.ncnn.mean_ms,
    result.ncnn.minimum_ms,
    result.ncnn.median_ms,
    result.compiled.mean_ms,
    result.compiled.minimum_ms,
    result.compiled.median_ms,
    std::max(result.ncnn.coefficient_of_variation,
             result.compiled.coefficient_of_variation),
    result.ratio);
}

::testing::AssertionResult check_performance_gate(
  std::string_view model, const PairBenchmarkResult& result) {
  const char* raw_limit = std::getenv("NCNN_PERF_MAX_RATIO");
  if (raw_limit == nullptr || *raw_limit == '\0') {
    return ::testing::AssertionSuccess();
  }
  char* parse_end = nullptr;
  const double maximum_ratio = std::strtod(raw_limit, &parse_end);
  if (parse_end == raw_limit || *parse_end != '\0' ||
      !std::isfinite(maximum_ratio)) {
    return ::testing::AssertionFailure()
           << "NCNN_PERF_MAX_RATIO='" << raw_limit << "' is not a number";
  }
  if (result.compiled.mean_ms <= maximum_ratio * result.ncnn.mean_ms) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << model << ": compiled " << result.compiled.mean_ms
         << " ms exceeds gate " << maximum_ratio << " x ncnn "
         << result.ncnn.mean_ms << " ms (measured ratio=" << result.ratio
         << ", ncnn_min=" << result.ncnn.minimum_ms
         << " ms, compiled_min=" << result.compiled.minimum_ms << " ms)";
}

}  // namespace ncnn_compiler::test
