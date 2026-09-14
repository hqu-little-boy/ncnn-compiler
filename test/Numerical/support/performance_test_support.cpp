#include "performance_test_support.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <print>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "allocator.h"
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

std::string escape_json(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char character : value) {
    switch (character) {
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
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      default:
        if (character < 0x20U) {
          escaped += std::format("\\u{:04x}", character);
        } else {
          escaped.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  return escaped;
}

std::expected<std::string, std::string> read_json_string_field(
  std::string_view text, std::string_view field) {
  const std::string key = std::format("\"{}\"", field);
  const std::size_t key_position = text.find(key);
  if (key_position == std::string_view::npos) {
    return std::unexpected(std::format("JSON field '{}' is missing", field));
  }
  std::size_t position = key_position + key.size();
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (position >= text.size() || text[position] != ':') {
    return std::unexpected(std::format("JSON field '{}' has no value", field));
  }
  ++position;
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (position >= text.size() || text[position] != '"') {
    return std::unexpected(
      std::format("JSON field '{}' must be a string", field));
  }
  ++position;
  std::string result;
  while (position < text.size()) {
    const char character = text[position++];
    if (character == '"') {
      return result;
    }
    if (character != '\\' || position >= text.size()) {
      if (character == '\\') {
        break;
      }
      result.push_back(character);
      continue;
    }
    const char escaped = text[position++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        result.push_back(escaped);
        break;
      case 'b':
        result.push_back('\b');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      default:
        return std::unexpected(
          std::format("JSON field '{}' has unsupported escape", field));
    }
  }
  return std::unexpected(std::format("JSON field '{}' is unterminated", field));
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

class CountingAllocator final : public ncnn::Allocator {
 public:
  void* fastMalloc(std::size_t size) override {
    void* pointer = ncnn::fastMalloc(size);
    std::scoped_lock lock(mutex_);
    if (pointer == nullptr) {
      if (audit_active_) {
        known_ = false;
      }
      return nullptr;
    }
    allocations_[pointer] = size;
    if (live_bytes_ > std::numeric_limits<std::uint64_t>::max() - size) {
      known_ = false;
    } else {
      live_bytes_ += size;
    }
    if (audit_active_) {
      ++allocation_count_;
      if (allocation_bytes_ >
          std::numeric_limits<std::uint64_t>::max() - size) {
        known_ = false;
      } else {
        allocation_bytes_ += size;
      }
      const bool inserted = timed_allocations_.insert(pointer).second;
      if (!inserted || timed_live_bytes_ >
                         std::numeric_limits<std::uint64_t>::max() - size) {
        known_ = false;
      } else {
        timed_live_bytes_ += size;
        peak_live_bytes_ = std::max(peak_live_bytes_, timed_live_bytes_);
      }
    }
    return pointer;
  }

  void fastFree(void* pointer) override {
    if (pointer == nullptr) {
      return;
    }
    std::scoped_lock lock(mutex_);
    auto iterator = allocations_.find(pointer);
    if (iterator == allocations_.end()) {
      if (audit_active_) {
        known_ = false;
      }
    } else {
      const std::size_t size = iterator->second;
      if (live_bytes_ < size) {
        known_ = false;
        live_bytes_ = 0;
      } else {
        live_bytes_ -= size;
      }
      const bool timed =
        audit_active_ && timed_allocations_.erase(pointer) != 0;
      if (timed) {
        if (timed_live_bytes_ < size) {
          known_ = false;
          timed_live_bytes_ = 0;
        } else {
          timed_live_bytes_ -= size;
        }
        ++deallocation_count_;
        if (deallocation_bytes_ >
            std::numeric_limits<std::uint64_t>::max() - size) {
          known_ = false;
        } else {
          deallocation_bytes_ += size;
        }
      }
      allocations_.erase(iterator);
    }
    ncnn::fastFree(pointer);
  }

  void reset() {
    std::scoped_lock lock(mutex_);
    allocation_count_ = 0;
    allocation_bytes_ = 0;
    deallocation_count_ = 0;
    deallocation_bytes_ = 0;
    peak_live_bytes_ = 0;
    timed_live_bytes_ = 0;
    timed_allocations_.clear();
    audit_active_ = true;
    known_ = true;
  }

  AllocationAuditStats snapshot() const {
    std::scoped_lock lock(mutex_);
    return AllocationAuditStats{.allocation_count = allocation_count_,
                                .allocation_bytes = allocation_bytes_,
                                .deallocation_count = deallocation_count_,
                                .deallocation_bytes = deallocation_bytes_,
                                .peak_live_bytes = peak_live_bytes_,
                                .bytes_known = known_};
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<void*, std::size_t> allocations_;
  std::unordered_set<void*> timed_allocations_;
  std::uint64_t allocation_count_ = 0;
  std::uint64_t allocation_bytes_ = 0;
  std::uint64_t deallocation_count_ = 0;
  std::uint64_t deallocation_bytes_ = 0;
  std::uint64_t live_bytes_ = 0;
  std::uint64_t timed_live_bytes_ = 0;
  std::uint64_t peak_live_bytes_ = 0;
  bool audit_active_ = false;
  bool known_ = true;
};

}  // namespace

std::expected<PerformanceIdentity, std::string> read_performance_identity(
  std::string_view plan_path) {
  std::ifstream stream{std::string(plan_path)};
  if (!stream) {
    return std::unexpected(
      std::format("cannot open execution plan '{}'", plan_path));
  }
  const std::string text((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
  auto target = read_json_string_field(text, "triple");
  if (!target) {
    return std::unexpected(target.error());
  }
  auto plan_revision = read_json_string_field(text, "plan_revision");
  if (!plan_revision) {
    return std::unexpected(plan_revision.error());
  }
  auto plan_hash = read_json_string_field(text, "plan_hash");
  if (!plan_hash) {
    return std::unexpected(plan_hash.error());
  }
  auto build_identity = read_json_string_field(text, "build_identity");
  if (!build_identity) {
    return std::unexpected(build_identity.error());
  }
  if (target->empty() || plan_revision->empty() || plan_hash->empty() ||
      build_identity->empty()) {
    return std::unexpected("execution plan identity contains an empty field");
  }
  return PerformanceIdentity{.target = std::move(*target),
                             .plan_revision = std::move(*plan_revision),
                             .plan_hash = std::move(*plan_hash),
                             .build_identity = std::move(*build_identity)};
}

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

std::expected<PerformanceMode, std::string> resolve_performance_mode() {
  const char* raw = std::getenv("NCNN_PERF_MODE");
  if (raw == nullptr || *raw == '\0' || std::strcmp(raw, "end_to_end") == 0) {
    return PerformanceMode::EndToEnd;
  }
  if (std::strcmp(raw, "prepared") == 0) {
    return PerformanceMode::Prepared;
  }
  if (std::strcmp(raw, "allocation_audit") == 0) {
    return PerformanceMode::AllocationAudit;
  }
  return std::unexpected(std::format(
    "NCNN_PERF_MODE='{}' must be one of end_to_end, prepared, allocation_audit",
    raw));
}

std::string_view performance_mode_name(PerformanceMode mode) {
  switch (mode) {
    case PerformanceMode::EndToEnd:
      return "end_to_end";
    case PerformanceMode::Prepared:
      return "prepared";
    case PerformanceMode::AllocationAudit:
      return "allocation_audit";
  }
  return "unknown";
}

PerformanceMetadata make_performance_metadata(PerformanceMode mode) {
  switch (mode) {
    case PerformanceMode::EndToEnd:
      return PerformanceMetadata{};
    case PerformanceMode::Prepared:
      return PerformanceMetadata{
        .mode = mode,
        .status = "measured",
        .setup = "net_extractor_inputs_and_outputs_before_timing",
        .gate_eligible = false,
        .prepared_runner = "extractor_reset_and_compiled_library_reuse",
        .allocation_source = "not_collected",
        .runtime_counters = "not_collected",
        .reason = {},
      };
    case PerformanceMode::AllocationAudit:
      return PerformanceMetadata{
        .mode = mode,
        .status = "measured",
        .setup = "net_extractor_inputs_and_outputs_before_timing",
        .gate_eligible = false,
        .prepared_runner = "extractor_reset_and_compiled_library_reuse",
        .allocation_source = "ncnn_counting_allocator",
        .runtime_counters = "compiled_profile_not_collected",
        .reason = {},
      };
  }
  return PerformanceMetadata{
    .mode = mode,
    .status = "unsupported",
    .setup = "not_collected",
    .gate_eligible = false,
    .prepared_runner = "not_supported",
    .allocation_source = "not_collected",
    .runtime_counters = "not_collected",
    .reason = "Unknown performance mode",
  };
}

std::expected<TimingStats, std::string> time_repeated_inference(
  const std::function<int()>& inference,
  const TimingPolicy& policy,
  const std::function<void()>& before_timed) {
  for (int iteration = 0; iteration < policy.warmup_iterations; ++iteration) {
    if (inference() != 0) {
      return std::unexpected("warmup inference iteration failed");
    }
  }
  if (before_timed) {
    before_timed();
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

struct NcnnPreparedBenchRunner::Impl final {
  Impl(std::string_view param_path,
       std::string_view bin_path,
       int num_threads,
       std::span<const ReferenceInput> inputs,
       std::span<const std::string_view> output_blob_names,
       ReferenceInferenceMode mode,
       bool collect_allocation_audit)
    : audit_enabled(collect_allocation_audit) {
    network.opt.lightmode = false;
    network.opt.use_local_pool_allocator = false;
    network.opt.blob_allocator = audit_enabled ? &allocator : nullptr;
    network.opt.workspace_allocator = audit_enabled ? &allocator : nullptr;
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
      return;
    }
    extractor = std::make_unique<ncnn::Extractor>(network.create_extractor());
    clean_extractor =
      std::make_unique<ncnn::Extractor>(network.create_extractor());
    if (inputs.empty() || output_blob_names.empty()) {
      error = "prepared runner requires inputs and outputs";
      return;
    }
    // Prepare input Mats once.  Their data remains stable across reset calls;
    // the extractor only receives shallow Mat handles during the timed body.
    for (const ReferenceInput& input : inputs) {
      auto mat = make_reference_input_mat(input);
      if (!mat || mat->empty()) {
        error =
          mat ? std::format("input blob '{}' is empty", input.get_blob_name())
              : std::move(mat).error();
        return;
      }
      fill_reference_input_mat(*mat, input);
      input_names.emplace_back(input.get_blob_name());
      input_mats.push_back(std::move(*mat));
    }
    for (std::string_view name : output_blob_names) {
      output_names.emplace_back(name);
      output_mats.emplace_back();
    }
    // Validate output contracts and prime ncnn's allocator during setup.  The
    // clean extractor is restored before the first timed call, so this probe
    // is never included in prepared timing.
    for (std::size_t index = 0; index < input_names.size(); ++index) {
      if (extractor->input(input_names[index].c_str(), input_mats[index]) !=
          0) {
        error = std::format("ncnn failed to bind prepared input blob '{}'",
                            input_names[index]);
        return;
      }
    }
    for (std::size_t index = 0; index < output_names.size(); ++index) {
      if (extractor->extract(output_names[index].c_str(), output_mats[index]) !=
          0) {
        error = std::format("ncnn failed to extract prepared output blob '{}'",
                            output_names[index]);
        return;
      }
      output_mats[index].release();
    }
    *extractor = *clean_extractor;
  }

  bool audit_enabled = false;
  CountingAllocator allocator;
  ncnn::Net network;
  std::unique_ptr<ncnn::Extractor> extractor;
  std::unique_ptr<ncnn::Extractor> clean_extractor;
  std::vector<std::string> input_names;
  std::vector<ncnn::Mat> input_mats;
  std::vector<std::string> output_names;
  mutable std::vector<ncnn::Mat> output_mats;
  mutable std::string error;
};

NcnnPreparedBenchRunner::NcnnPreparedBenchRunner(
  std::string_view param_path,
  std::string_view bin_path,
  int num_threads,
  std::span<const ReferenceInput> inputs,
  std::span<const std::string_view> output_blob_names,
  ReferenceInferenceMode mode,
  bool collect_allocation_audit)
  : impl_(new Impl(param_path,
                   bin_path,
                   num_threads,
                   inputs,
                   output_blob_names,
                   mode,
                   collect_allocation_audit)) {}

NcnnPreparedBenchRunner::~NcnnPreparedBenchRunner() {
  delete impl_;
}

bool NcnnPreparedBenchRunner::valid() const noexcept {
  return impl_->error.empty();
}

std::string_view NcnnPreparedBenchRunner::error() const noexcept {
  return impl_->error;
}

int NcnnPreparedBenchRunner::run(std::span<std::vector<float>> outputs) const {
  if (!valid()) {
    return -1;
  }
  impl_->error.clear();
  if (outputs.size() != impl_->output_names.size()) {
    impl_->error = "prepared runner output count does not match setup";
    return -1;
  }
  if (!impl_->extractor || !impl_->clean_extractor) {
    impl_->error = "prepared runner extractor is not initialized";
    return -1;
  }
  *impl_->extractor = *impl_->clean_extractor;
  for (std::size_t index = 0; index < impl_->input_names.size(); ++index) {
    if (impl_->extractor->input(impl_->input_names[index].c_str(),
                                impl_->input_mats[index]) != 0) {
      impl_->error = std::format("ncnn failed to bind prepared input blob '{}'",
                                 impl_->input_names[index]);
      return -1;
    }
  }
  for (std::size_t index = 0; index < impl_->output_names.size(); ++index) {
    if (impl_->extractor->extract(impl_->output_names[index].c_str(),
                                  impl_->output_mats[index]) != 0) {
      impl_->error =
        std::format("ncnn failed to extract prepared output blob '{}'",
                    impl_->output_names[index]);
      return -1;
    }
    if (auto failure = flatten_ncnn_output(impl_->output_mats[index],
                                           impl_->output_names[index],
                                           outputs[index])) {
      impl_->error = std::move(*failure);
      impl_->output_mats[index].release();
      return -1;
    }
    impl_->output_mats[index].release();
  }
  return 0;
}

void NcnnPreparedBenchRunner::reset_allocation_audit() const {
  if (impl_->audit_enabled) {
    impl_->allocator.reset();
  }
}

AllocationAuditStats NcnnPreparedBenchRunner::allocation_audit() const {
  if (!impl_->audit_enabled) {
    return AllocationAuditStats{.bytes_known = false};
  }
  return impl_->allocator.snapshot();
}

void emit_performance_report(std::string_view model,
                             int threads,
                             const TimingPolicy& policy,
                             const PairBenchmarkResult& result,
                             const PerformanceMetadata& metadata) {
  const double coefficient_of_variation =
    std::max(result.ncnn.coefficient_of_variation,
             result.compiled.coefficient_of_variation);
  std::println(
    "PERF model={} mode={} status={} threads={} warmup={} iters={} "
    "ncnn_ms={:.3f} compiled_ms={:.3f} ratio={:.3f} "
    "ncnn_min_ms={:.3f} compiled_min_ms={:.3f} cv={:.4f} setup={}",
    model,
    performance_mode_name(metadata.mode),
    metadata.status,
    threads,
    policy.warmup_iterations,
    policy.timed_iterations,
    result.ncnn.mean_ms,
    result.compiled.mean_ms,
    result.ratio,
    result.ncnn.minimum_ms,
    result.compiled.minimum_ms,
    coefficient_of_variation,
    metadata.setup);
}

void emit_performance_diagnostic_report(std::string_view model,
                                        int threads,
                                        const TimingPolicy& policy,
                                        const PerformanceMetadata& metadata) {
  std::println(
    "PERF model={} mode={} status={} threads={} warmup={} iters={} "
    "gate_eligible={} setup={} reason={}",
    model,
    performance_mode_name(metadata.mode),
    metadata.status,
    threads,
    policy.warmup_iterations,
    policy.timed_iterations,
    metadata.gate_eligible,
    metadata.setup,
    metadata.reason);
}

std::expected<void, std::string> append_performance_json_record(
  std::string_view model,
  int threads,
  const TimingPolicy& policy,
  const PairBenchmarkResult& result,
  const PerformanceMetadata& metadata) {
  const char* path = std::getenv("NCNN_PERF_JSON");
  if (path == nullptr || *path == '\0') {
    return {};
  }
  std::ofstream stream(path, std::ios::app);
  if (!stream) {
    return std::unexpected(std::format(
      "cannot open NCNN_PERF_JSON '{}': {}", path, std::strerror(errno)));
  }
  std::string allocation_json = "null";
  if (result.ncnn_allocation.has_value()) {
    const AllocationAuditStats& allocation = *result.ncnn_allocation;
    const std::string allocation_bytes =
      allocation.bytes_known ? std::format("{}", allocation.allocation_bytes)
                             : "null";
    const std::string deallocation_bytes =
      allocation.bytes_known ? std::format("{}", allocation.deallocation_bytes)
                             : "null";
    const std::string peak_live_bytes =
      allocation.bytes_known ? std::format("{}", allocation.peak_live_bytes)
                             : "null";
    allocation_json =
      std::format(R"({{"allocation_count":{},"allocation_bytes":{},)"
                  R"("deallocation_count":{},"deallocation_bytes":{},)"
                  R"("peak_live_bytes":{},"bytes_known":{}}})",
                  allocation.allocation_count,
                  allocation_bytes,
                  allocation.deallocation_count,
                  deallocation_bytes,
                  peak_live_bytes,
                  allocation.bytes_known);
  }
  const auto identity_json = [](std::string_view value) {
    return value.empty() ? std::string("null")
                         : std::format("\"{}\"", escape_json(value));
  };
  stream << std::format(
    R"({{"model":"{}","mode":"{}","status":"{}","target":{},)"
    R"("plan_revision":"{}","plan_hash":{},"build_identity":{},"threads":{},)"
    R"("warmup":{},"iterations":{},"setup":{{"ncnn":"{}","compiled":"{}"}},)"
    R"("diagnostics":{{"gate_eligible":{},"prepared_runner":"{}",)"
    R"("allocation_source":"{}","runtime_counters":"{}"}},)"
    R"("ncnn_mean_ms":{:.3f},"ncnn_min_ms":{:.3f},"ncnn_median_ms":{:.3f},)"
    R"("compiled_mean_ms":{:.3f},"compiled_min_ms":{:.3f},)"
    R"("compiled_median_ms":{:.3f},"cv":{:.4f},"ratio":{:.3f},)"
    R"("ncnn_allocation":{}}})"
    "\n",
    escape_json(model),
    escape_json(performance_mode_name(metadata.mode)),
    escape_json(metadata.status),
    identity_json(metadata.target),
    escape_json(metadata.plan_revision),
    identity_json(metadata.plan_hash),
    identity_json(metadata.build_identity),
    threads,
    policy.warmup_iterations,
    policy.timed_iterations,
    escape_json(metadata.setup),
    "shared_library_loaded_before_timing",
    metadata.gate_eligible,
    escape_json(metadata.prepared_runner),
    escape_json(metadata.allocation_source),
    escape_json(metadata.runtime_counters),
    result.ncnn.mean_ms,
    result.ncnn.minimum_ms,
    result.ncnn.median_ms,
    result.compiled.mean_ms,
    result.compiled.minimum_ms,
    result.compiled.median_ms,
    std::max(result.ncnn.coefficient_of_variation,
             result.compiled.coefficient_of_variation),
    result.ratio,
    allocation_json);
  if (!stream) {
    return std::unexpected(std::format(
      "cannot write NCNN_PERF_JSON '{}': {}", path, std::strerror(errno)));
  }
  return {};
}

std::expected<void, std::string> append_performance_json_diagnostic(
  std::string_view model,
  int threads,
  const TimingPolicy& policy,
  const PerformanceMetadata& metadata) {
  const char* path = std::getenv("NCNN_PERF_JSON");
  if (path == nullptr || *path == '\0') {
    return {};
  }
  std::ofstream stream(path, std::ios::app);
  if (!stream) {
    return std::unexpected(std::format(
      "cannot open NCNN_PERF_JSON '{}': {}", path, std::strerror(errno)));
  }
  stream << std::format(
    R"({{"model":"{}","mode":"{}","status":"{}","target":null,)"
    R"("plan_revision":"static-v1","plan_hash":null,"build_identity":null,"threads":{},)"
    R"("warmup":{},"iterations":{},"setup":{{"ncnn":"{}","compiled":"{}"}},)"
    R"("diagnostics":{{"gate_eligible":{},"prepared_runner":"{}",)"
    R"("allocation_source":"{}","runtime_counters":"{}","reason":"{}"}},)"
    R"("ncnn_mean_ms":null,"compiled_mean_ms":null,"ratio":null}})"
    "\n",
    escape_json(model),
    escape_json(performance_mode_name(metadata.mode)),
    escape_json(metadata.status),
    threads,
    policy.warmup_iterations,
    policy.timed_iterations,
    escape_json(metadata.setup),
    "not_timed",
    metadata.gate_eligible,
    escape_json(metadata.prepared_runner),
    escape_json(metadata.allocation_source),
    escape_json(metadata.runtime_counters),
    escape_json(metadata.reason));
  if (!stream) {
    return std::unexpected(std::format(
      "cannot write NCNN_PERF_JSON '{}': {}", path, std::strerror(errno)));
  }
  return {};
}

::testing::AssertionResult check_performance_gate(
  std::string_view model,
  const PairBenchmarkResult& result,
  double default_limit) {
  const char* raw_limit = std::getenv("NCNN_PERF_MAX_RATIO");
  const bool has_env_limit = raw_limit != nullptr && *raw_limit != '\0';
  // 逐类默认门禁只对 6 线程正式口径生效：NCNN_PERF_THREADS pin 的单线程
  // /自定线程实验 ratio 量级完全不同，不设 env 时不判定，保持纯报告。
  const char* raw_threads = std::getenv("NCNN_PERF_THREADS");
  const bool threads_pinned = raw_threads != nullptr && *raw_threads != '\0';
  if (!has_env_limit && (threads_pinned || !(default_limit > 0.0))) {
    return ::testing::AssertionSuccess();
  }
  double maximum_ratio = default_limit;
  const char* limit_source = "per-class default";
  if (has_env_limit) {
    char* parse_end = nullptr;
    maximum_ratio = std::strtod(raw_limit, &parse_end);
    if (parse_end == raw_limit || *parse_end != '\0' ||
        !std::isfinite(maximum_ratio)) {
      return ::testing::AssertionFailure()
             << "NCNN_PERF_MAX_RATIO='" << raw_limit << "' is not a number";
    }
    limit_source = "NCNN_PERF_MAX_RATIO";
    if (!(maximum_ratio > 0.0)) {
      // 显式 0/负数 = 关闭门禁（含 per-class 默认）。
      return ::testing::AssertionSuccess();
    }
  }
  if (result.compiled.mean_ms <= maximum_ratio * result.ncnn.mean_ms) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << model << ": compiled " << result.compiled.mean_ms
         << " ms exceeds gate " << maximum_ratio << " (" << limit_source
         << ") x ncnn " << result.ncnn.mean_ms
         << " ms (measured ratio=" << result.ratio
         << ", ncnn_min=" << result.ncnn.minimum_ms
         << " ms, compiled_min=" << result.compiled.minimum_ms << " ms)";
}

}  // namespace ncnn_compiler::test
