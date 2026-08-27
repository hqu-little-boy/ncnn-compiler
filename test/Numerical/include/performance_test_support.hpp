#pragma once

#include "numerical_test_support.hpp"

#include <expected>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {

// benchncnn 约定的线程口径：物理大核数，参考侧与编译产物侧必须一致才公平。
// 环境变量 NCNN_PERF_THREADS 可覆盖；非数字值返回错误（配置错误要响亮失败）。
[[nodiscard]] std::expected<int, std::string> resolve_benchmark_thread_count();

// 按 vendored ncnn benchmark/benchncnn.cpp 的惯例配置线程：
// set_cpu_powersave(2)、set_omp_dynamic(0)、set_omp_num_threads(n)，并尽力
// 设置 OMP_NUM_THREADS。ncnn 与生成的 .so 在 clang 工具链下链接同一 libomp
// 实例，主线程上设置的 nthreads-var 对两者同时生效；环境变量只是为
// “晚加载的独立 OpenMP 运行时”兜底。必须在任何计时之前调用。
void apply_benchncnn_threading(int threads);

struct TimingPolicy final {
  int warmup_iterations = 0;
  int timed_iterations = 0;
};

// NCNN_PERF_WARMUP / NCNN_PERF_ITERS 以整数覆盖 default_policy 对应字段；
// 任一变量存在但不是完整整数时报错。
[[nodiscard]] std::expected<TimingPolicy, std::string> resolve_timing_policy(
  TimingPolicy default_policy);

struct TimingStats final {
  double mean_ms = 0.0;
  double minimum_ms = 0.0;
  double median_ms = 0.0;
  double stddev_ms = 0.0;
  double coefficient_of_variation = 0.0;
};

// 对一次完整推理闭包计时：先 warmup_iterations 次预热（每次都必须返回 0），
// 再做 timed_iterations 次计时采样，统计均值/最小/中位/标准差/变异系数。
[[nodiscard]] std::expected<TimingStats, std::string> time_repeated_inference(
  const std::function<int()>& inference, const TimingPolicy& policy);

// 复用已加载 ncnn::Net 的参考运行器（pimpl，避免在头文件暴露 ncnn 类型）。
// opt 块与 run_ncnn_reference 的 FP32 CPU 路径逐字对齐，仅追加
// opt.num_threads；每次 run() 新建 Extractor——与 benchncnn 计时体一致，
// Extractor 开销计入上游运行时成本是刻意为之。
class NcnnBenchRunner final {
 public:
  NcnnBenchRunner(std::string_view param_path,
                  std::string_view bin_path,
                  int num_threads);
  ~NcnnBenchRunner();

  NcnnBenchRunner(const NcnnBenchRunner&) = delete;
  NcnnBenchRunner& operator=(const NcnnBenchRunner&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  // 构造失败或最近一次 run() 失败的原因。
  [[nodiscard]] std::string_view error() const noexcept;

  // 一次完整推理：绑定 inputs，按 output_blob_names 顺序把各输出展平写入
  // outputs（调用方预先 resize 好每个缓冲）。
  [[nodiscard]] int run(std::span<const ReferenceInput> inputs,
                        std::span<const std::string_view> output_blob_names,
                        std::span<std::vector<float>> outputs) const;

 private:
  struct Impl;
  Impl* impl_;
};

struct PairBenchmarkResult final {
  TimingStats ncnn;
  TimingStats compiled;
  double ratio = 0.0;  // compiled.mean_ms / ncnn.mean_ms
};

// 固定格式单行报告，机器可 grep：
// PERF model=<name> threads=<n> warmup=<w> iters=<it>
//     ncnn_ms=<mean> compiled_ms=<mean> ratio=<r>
//     ncnn_min_ms=<..> compiled_min_ms=<..> cv=<两侧较大 CV>
void emit_performance_report(std::string_view model,
                             int threads,
                             const TimingPolicy& policy,
                             const PairBenchmarkResult& result);

// NCNN_PERF_JSON 已设置时向该文件追加一行 NDJSON；未设置时为空操作。
// 全部性能测试 RUN_SERIAL 执行，追加写安全。
void append_performance_json_record(std::string_view model,
                                    int threads,
                                    const TimingPolicy& policy,
                                    const PairBenchmarkResult& result);

// NCNN_PERF_MAX_RATIO 未设置 → 恒 Success（默认纯报告）；设置为正数 x 时，
// compiled.mean_ms > x * ncnn.mean_ms 即 Failure，消息携带全部实测数字；
// 变量存在但不是数字时直接 Failure（不许静默忽略错误配置）。
[[nodiscard]] ::testing::AssertionResult check_performance_gate(
  std::string_view model, const PairBenchmarkResult& result);

}  // namespace ncnn_compiler::test
