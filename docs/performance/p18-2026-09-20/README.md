# P18 完整模型性能归因闭环证据

日期：2026-09-20
阶段：`/tmp/ncnn-compiler-stage-p18-final-02`
工具链：LLVM/MLIR 21.1.8，Release，x86_64，OpenMP，`COMPILER_INSTALL_RPATH=ON`

## 结论

P18 的归因基础设施和验证门禁已完成，但 P18 不宣称性能收益或 parity。正式 fresh end-to-end 对照保留 45 条 measured 行：44 个 official 模型和 1 个 `resnet18_winograd` 诊断行；两个 upstream ncnn INT8 崩溃项保持真实 skip，不补造 ratio。

6 线程、warmup=10、iterations=20 的 official 44 统计：

| 指标 | 结果 |
|---|---:|
| p50 | 2.0705x |
| p90 | 3.8911x |
| max | 19.422x |
| 几何均值 | 2.2839x |
| ratio ≤1.5x | 5/44 |
| ncnn 总时间 | 9482.433 ms |
| compiled 总时间 | 29186.279 ms |
| 总超额 | 19703.846 ms |

绝对超额前五：`pp_ocrv5_server_rec`、`pp_ocrv5_server_det_static`、`pp_formulanet_plus_s_encoder`、`yolov5x_seg`、`yolov5x`。前两项仍是 P19 的主要止损对象。

正式 E2E 行的 `runtime_counters` 仍为 `not_collected`，所以 top5 runtime accounted coverage、runtime movement callback coverage 和 unknown exclusive-time 门槛尚未闭环。`prepared`/allocation diagnostic 和 `relu` profile fixture 只证明 schema/identity/bytes contract，不替代正式 E2E。

## Fresh 验证

```bash
cmake -S compiler -B /tmp/ncnn-compiler-stage-p18-final-02 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-p18-final-02 --target format_check --parallel 16
cmake --build /tmp/ncnn-compiler-stage-p18-final-02 --target tidy --parallel 16
cmake --build /tmp/ncnn-compiler-stage-p18-final-02 --parallel 16
cmake --build /tmp/ncnn-compiler-stage-p18-final-02 \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel 16
ctest --test-dir /tmp/ncnn-compiler-stage-p18-final-02 --output-on-failure
```

最终 CTest：447 tests，0 failed；`PPLCNetDocOriInt8` 和 `PPLCNetTextlineOriInt8` 为 upstream ncnn 真实 skip。中间 stage 发现并修复了 `contract.mlir` 的 stale FileCheck expectation（正向 packed contract 的 `packed_buffer_bytes=64`），修复后从全新 stage 重跑。

正式性能命令：

```bash
NCNN_PERF_THREADS=6 NCNN_PERF_WARMUP=10 NCNN_PERF_ITERS=20 \
NCNN_PERF_MAX_RATIO=0 NCNN_PERF_MODE=end_to_end \
NCNN_PERF_JSON=compiler/docs/performance/p18-2026-09-20/candidate-end-to-end.ndjson \
ctest --test-dir /tmp/ncnn-compiler-stage-p18-final-02 \
  -R '^PerformanceModel\\.' --output-on-failure
```

## 归档文件

- `candidate-end-to-end.ndjson`：最终 45-row 原始 NDJSON；不覆盖。
- `final-all-model-ncnn-comparison.ndjson` / `.json`：保留的全模型 ncnn comparison JSON。
- `official-summary.txt`：严格 official 44 汇总。
- `all-measured-summary.txt`：45 measured 汇总，含 Winograd 诊断行。
- `summary.json` / `top-excess.json`：机器可读统计和超额排序。
- `build-identity.json`：fresh stage、LLVM、target、线程、源码 diff hash、compiler hash。
- `plan-index.json` / `static-plan-coverage.json`：45 个模型的 plan identity 和静态覆盖索引。
- `compile-artifacts.json`：plan hash 和可重建的 `/tmp` library 路径；大型 `.so` 不复制进证据目录。
- `profile-sidecars/relu/`：schema-1 profile、plan、诊断 perf 行和 attribution report。
- `profile-sidecars/relu-schema2/`：schema-2 per-invocation NDJSON profile fixture。
- `perf-stat/`：top5 加两个对照模型的原始 `perf stat`/CTest 输出；不可用或 hybrid CPU 未计数事件保持原始 unknown。
- `numerical-results.json`：numerical target 与 CTest 验收记录。
- `performance-end-to-end.log`：正式 6T 性能命令日志。

## Attribution-v1 语义

Execution plan、profile runtime 和 `perf_attribution_report.py` 严格校验 model、plan hash、build identity、revision、target、threads、mode 和 schema。事件 ID 使用 function-local FNV-1a stable ID；event bytes 采用 `bytes`/`bytes_known` 不变量。allocation/live/peak、copy/transpose/pack/unpack、workspace join、top-20 runtime cost 和 top-10 allocation lifetime 均保留 unknown/incomplete 原因；parallel worker exclusive time、硬件计数和物理 RSS 不可证明时不填零。
