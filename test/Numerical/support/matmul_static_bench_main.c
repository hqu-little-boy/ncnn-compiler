// P20 static-RHS benchmark driver. Both packing=off and packing=auto
// executables use this exact driver and descriptor ABI for paired timing.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct MatmulStaticMemRef {
  float* allocated;
  float* aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} MatmulStaticMemRef;

void _mlir_ciface_bench_shallow(MatmulStaticMemRef*, MatmulStaticMemRef*);
void _mlir_ciface_bench_deep(MatmulStaticMemRef*, MatmulStaticMemRef*);

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + ((double)ts.tv_nsec / 1e9);
}

static int env_int(const char* name, int fallback) {
  const char* raw = getenv(name);
  if (raw == NULL || *raw == '\0') {
    return fallback;
  }
  char* end = NULL;
  long value = strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value <= 0 || value > 1000000) {
    fprintf(stderr, "invalid %s=%s\n", name, raw);
    exit(2);
  }
  return (int)value;
}

static int run_case(const char* name,
                    void (*entry)(MatmulStaticMemRef*, MatmulStaticMemRef*),
                    int m,
                    int k,
                    int n,
                    int iters) {
  float* input = aligned_alloc(64, (size_t)m * k * sizeof(float));
  float* output = aligned_alloc(64, (size_t)m * n * sizeof(float));
  if (input == NULL || output == NULL) {
    fprintf(stderr, "allocation failed for %s\n", name);
    return 2;
  }
  for (int i = 0; i < m * k; ++i) {
    input[i] = (float)(i % 97) * 0.01f;
  }
  for (int i = 0; i < m * n; ++i) {
    output[i] = 0.0f;
  }
  MatmulStaticMemRef in = {input, input, 0, {m, k}, {k, 1}};
  MatmulStaticMemRef out = {output, output, 0, {m, n}, {n, 1}};
  entry(&in, &out);
  double best = 1e30;
  for (int round = 0; round < 3; ++round) {
    double start = now_seconds();
    for (int i = 0; i < iters; ++i) {
      entry(&in, &out);
    }
    double elapsed = now_seconds() - start;
    if (elapsed < best) {
      best = elapsed;
    }
  }
  double gflops = 2.0 * m * n * k * (double)iters / best / 1e9;
  double checksum = 0.0;
  for (int i = 0; i < m * n; ++i) {
    checksum += output[i];
  }
  printf("%s: gflops=%.3f checksum=%.6f\n", name, gflops, checksum);
  free(input);
  free(output);
  return 0;
}

int main(void) {
  int iters = env_int("MATMUL_STATIC_BENCH_ITERS", 5);
  int failures = 0;
  failures += run_case(
    "static_shallow", _mlir_ciface_bench_shallow, 1024, 576, 64, iters);
  failures +=
    run_case("static_deep", _mlir_ciface_bench_deep, 256, 2304, 256, iters);
  return failures == 0 ? 0 : 1;
}
