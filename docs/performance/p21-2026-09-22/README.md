# P21 性能与验收证据（2026-09-22）

本目录保存 P21 `packed Conv/Depthwise + layout island` 的 fresh Release 验收、五轮 paired A/B↔B/A 性能数据和最终全模型 ncnn 对比 JSON。P21 packed Conv/Depthwise 是显式 opt-in；baseline 使用同一源码和同一 vendored ncnn、`NCNN_PACKED_CONV_DEPTHWISE=OFF`，candidate 使用 `NCNN_PACKED_CONV_DEPTHWISE=ON`。

## 构建身份

- baseline：`/tmp/ncnn-compiler-stage-p21-baseline-20260922-a`
- candidate：`/tmp/ncnn-compiler-stage-p21-repair-20260922-a`
- Release，LLVM/MLIR 21.1.8，x86_64，OpenMP，`COMPILER_INSTALL_RPATH=ON`
- 性能线程：6；warmup=10；iterations=20；end-to-end；`NCNN_PERF_MAX_RATIO=0`
- candidate 计划 revision 包含 `layout-island-v1`，并将 `packed_conv_depthwise=true` 纳入 code-generation identity

## 工程门禁

candidate fresh stage 完成：

- `format_check`：通过
- `tidy`：通过
- `numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests`：构建通过
- 完整 CTest：**449/449 通过，0 失败**；两个 upstream ncnn 已知 INT8 参考崩溃项保持真实 skip：`PPLCNetDocOriInt8`、`PPLCNetTextlineOriInt8`
- 完整 CTest 用时：1144.71 秒

对应记录见 `numerical-results.json`、`final-full-ctest.log`、`final-tidy.log` 和 `final-numerical-build.log`。

## 五轮 paired 性能结果

轮次顺序为：

1. baseline → candidate
2. candidate → baseline
3. baseline → candidate
4. candidate → baseline
5. baseline → candidate

每轮保留完整 45 条 measured NDJSON（44 个 official 模型 + `resnet18_winograd` 诊断模型），并保留对应 CTest 日志。正式 official 44 的五轮中位数汇总：

| 指标 | baseline | candidate | candidate 相对 baseline |
|---|---:|---:|---:|
| 模型 ratio p50 的中位数 | 1.927x | 1.978x | — |
| 模型 ratio p50 几何均值 | 2.119x | 2.150x | — |
| compiled 延迟 p50 | — | — | **+1.80%** |
| compiled 延迟几何均值 | — | — | **+2.35%** |
| candidate 更快的模型 | — | — | 12/44 |
| candidate 退化的模型 | — | — | 32/44 |

因此 P21 本批次没有满足默认产品化收益门槛，结论为 **opt-in / no-go for default**；不能将 layout island 被选中或 packed kernel 能生成解释为端到端收益。代表性结果和每个模型的 p50/delta 见 `paired-summary.json`、`paired-summary.txt`，原始数据见 `round-*-baseline.ndjson`、`round-*-candidate.ndjson` 和 `paired-rounds.ndjson`。

`final-all-model-ncnn-comparison.json` 与 `.ndjson` 是最后 candidate round 的完整模型→ncnn 对比，未覆盖既有 P18/P20 文件。

## 静态计划覆盖

candidate 的 45 个 measured plan 汇总出：

- layout island：359 个，selected 359 个；
- pack8：359 个；本模型批次没有 pack4 形状；
- `conv_packed_gemm`：196 个；
- `depthwise_packed`：359 个；
- packed constant/static pack bytes 为计划级汇总，不是 runtime 峰值，也不替代 profile 证据。

机器可读索引见 `plan-index.json` 和 `static-plan-coverage.json`。`efficientnet_b0-*-allocation-audit.ndjson` 仅为 selected allocation 诊断；其 `runtime_counters` 明确为 `compiled_profile_not_collected`，所以本阶段不声称 runtime packed-consumption 已闭环。

## 重放命令

```bash
cmake -S /mnt/ncnn-compiler/compiler \
  -B /tmp/ncnn-compiler-stage-p21-repair-20260922-a \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON \
  -DBUILD_TESTING=ON \
  -DNCNN_PACKED_CONV_DEPTHWISE=ON

cmake --build /tmp/ncnn-compiler-stage-p21-repair-20260922-a --parallel 16
cmake --build /tmp/ncnn-compiler-stage-p21-repair-20260922-a --target format_check --parallel 16
cmake --build /tmp/ncnn-compiler-stage-p21-repair-20260922-a --target tidy --parallel 16
cmake --build /tmp/ncnn-compiler-stage-p21-repair-20260922-a \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel 16
ctest --test-dir /tmp/ncnn-compiler-stage-p21-repair-20260922-a --output-on-failure

NCNN_PERF_THREADS=6 NCNN_PERF_WARMUP=10 NCNN_PERF_ITERS=20 \
NCNN_PERF_MAX_RATIO=0 NCNN_PERF_MODE=end_to_end \
NCNN_PERF_JSON=/path/to/round.ndjson \
ctest --test-dir /path/to/stage -R '^PerformanceModel\\.' --output-on-failure
```

完整命令和构建身份见 `build-identity.json`。大型生成 `.so` 保留在 fresh `/tmp` stage；证据目录保留 hash/plan/JSON/日志而不复制大型二进制。
