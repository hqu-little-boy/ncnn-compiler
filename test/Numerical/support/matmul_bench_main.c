// P3 隔离 matmul bench 驱动：调用 matmul_bench.mlir 生成的两个内核
// （@bench_shallow / @bench_deep），单线程计时并按 GFLOP/s 门禁判定。
// 门禁默认值取 P3 实测（浅 K ~104–112、深 K ~55）留 ≥25% 余量、显著
// 高于 P1 单行内核基线（36.7 / 16.2），慢机可经环境变量下调：
//   MATMUL_BENCH_MIN_GFLOPS_SHALLOW（默认 60）
//   MATMUL_BENCH_MIN_GFLOPS_DEEP（默认 40）
// 验收口径（1T 首轮 ≥ 45 GFLOP/s）见
// docs/ncnn-performance-parity-plan.md §3-P3。
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// MLIR memref 描述符（静态 2 维，llvm.emit_c_interface 包装约定）。
typedef struct MatmulBenchMemRef {
  float* allocated;
  float* aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} MatmulBenchMemRef;

void _mlir_ciface_bench_shallow(MatmulBenchMemRef*,
                                MatmulBenchMemRef*,
                                MatmulBenchMemRef*);
void _mlir_ciface_bench_deep(MatmulBenchMemRef*,
                             MatmulBenchMemRef*,
                             MatmulBenchMemRef*);

static double matmul_bench_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double matmul_bench_env_threshold(const char* name,
                                         double default_gflops) {
  const char* raw = getenv(name);
  if (raw == NULL || *raw == '\0') {
    return default_gflops;
  }
  char* parse_end = NULL;
  const double value = strtod(raw, &parse_end);
  if (parse_end == raw || *parse_end != '\0' || !(value > 0.0)) {
    fprintf(stderr, "matmul_bench: %s='%s' 不是正数\n", name, raw);
    exit(2);
  }
  return value;
}

// 返回：输出 checksum（用于人工比对新旧内核一致性）；*gflops 写入实测。
static double matmul_bench_run(void (*entry)(MatmulBenchMemRef*,
                                             MatmulBenchMemRef*,
                                             MatmulBenchMemRef*),
                               int m,
                               int k,
                               int n,
                               int iters,
                               double* gflops) {
  float* im2col = aligned_alloc(64, (size_t)m * k * sizeof(float));
  float* weight = aligned_alloc(64, (size_t)k * n * sizeof(float));
  float* out = aligned_alloc(64, (size_t)m * n * sizeof(float));
  if (im2col == NULL || weight == NULL || out == NULL) {
    fprintf(stderr, "matmul_bench: 缓冲区分配失败\n");
    exit(2);
  }
  for (int i = 0; i < m * k; ++i) {
    im2col[i] = (float)(i % 97) * 0.01f;
  }
  for (int i = 0; i < k * n; ++i) {
    weight[i] = (float)(i % 89) * 0.01f;
  }
  for (int i = 0; i < m * n; ++i) {
    out[i] = 0.0f;
  }
  MatmulBenchMemRef descriptor_im2col = {im2col, im2col, 0, {m, k}, {k, 1}};
  MatmulBenchMemRef descriptor_weight = {weight, weight, 0, {k, n}, {n, 1}};
  MatmulBenchMemRef descriptor_out = {out, out, 0, {m, n}, {n, 1}};

  entry(&descriptor_im2col, &descriptor_weight, &descriptor_out);  // 预热
  double best = 1e9;
  for (int round = 0; round < 3; ++round) {
    const double start = matmul_bench_now_seconds();
    for (int i = 0; i < iters; ++i) {
      entry(&descriptor_im2col, &descriptor_weight, &descriptor_out);
    }
    const double elapsed = matmul_bench_now_seconds() - start;
    if (elapsed < best) {
      best = elapsed;
    }
  }
  *gflops = 2.0 * m * n * k * (double)iters / best / 1e9;

  double checksum = 0.0f;
  for (int i = 0; i < m * n; ++i) {
    checksum += out[i];
  }
  free(im2col);
  free(weight);
  free(out);
  return checksum;
}

static int matmul_bench_case(const char* name,
                             void (*entry)(MatmulBenchMemRef*,
                                           MatmulBenchMemRef*,
                                           MatmulBenchMemRef*),
                             int m,
                             int k,
                             int n,
                             int iters,
                             const char* threshold_env,
                             double default_threshold) {
  const double threshold =
    matmul_bench_env_threshold(threshold_env, default_threshold);
  double gflops = 0.0;
  const double checksum = matmul_bench_run(entry, m, k, n, iters, &gflops);
  const int pass = gflops >= threshold;
  printf("%s: gflops=%.2f (threshold=%.2f, checksum=%.3f) -> %s\n",
         name,
         gflops,
         threshold,
         checksum,
         pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main(void) {
  int failures = 0;
  failures += matmul_bench_case("bench_shallow",
                                _mlir_ciface_bench_shallow,
                                1024,
                                576,
                                64,
                                50,
                                "MATMUL_BENCH_MIN_GFLOPS_SHALLOW",
                                60.0);
  failures += matmul_bench_case("bench_deep",
                                _mlir_ciface_bench_deep,
                                256,
                                2304,
                                256,
                                30,
                                "MATMUL_BENCH_MIN_GFLOPS_DEEP",
                                40.0);
  return failures == 0 ? 0 : 1;
}
