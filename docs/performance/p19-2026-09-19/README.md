# P19 性能证据（2026-09-19）

本目录保存 P19 `server_rec 长序列与布局专项` 的原始性能证据。正式性能口径使用全新 Release 构建产物、`NCNN_PERF_THREADS=6`、warmup `10`、timed iterations `20`，测试过滤器为 `PerformanceModel.*`。每个运行使用独立的 `NCNN_PERF_JSON` NDJSON 文件；`baseline` 是 P19 改动前的冻结 Release 构建，`candidate` 是 P19 最终 candidate-07 Release 构建。

- `baseline-6t-*.ndjson` / `candidate-6t-*.ndjson`：6 线程正式测量；奇数轮顺序为 baseline→candidate，偶数轮为 candidate→baseline。
- `baseline-1t-server-rec-focused.ndjson` / `candidate-1t-server-rec-focused.ndjson`：1 线程 `pp_ocrv5_server_rec` 专项诊断；candidate 文件是完整 raw run 中该模型的单行 extracted view，完整 raw 文件仍保留。
- `candidate-1t-full-suite-timeout.ndjson` 与对应 log：一次非正式 1 线程全套诊断保留的 44 条已测行；`PPOcrv5ServerDetStatic` 超过 CTest 1800 秒 timeout，因此不进入正式 6 线程门禁或 paired 汇总。
- 对应 `.ctest.log`：每次运行的完整测试日志。
- `run.log`：运行顺序和完成时间。
- `run-manifest.json`：构建目录、计时参数和配对顺序机器可读记录。
- `build-identity.json`：baseline/candidate source、Release stage、vendored ncnn revision 和代表 plan identity。
- `plan-index.json`：45 个 measured model 的 baseline/candidate plan hash、build identity 和 target join 索引。
- `paired-rounds.ndjson`、`paired-summary.json`、`paired-summary.txt`：由原始 NDJSON 聚合生成，不替代原始记录；single-run summary 也保留。
- `candidate-08-{configure,build,format,tidy,numerical-build,ctest}.log`：因 1T 诊断 timeout 后按要求从新 `/tmp` stage 重复的最终工程门禁证据；`candidate-08-status.txt` 记录结果。

每个正式 suite 应产生 45 条 measured NDJSON 记录。`PPLCNetDocOriInt8` 和 `PPLCNetTextlineOriInt8` 是 upstream ncnn 参考侧已知崩溃，保持显式 GTest skip，不伪造 ratio；`resnet18_winograd` 是诊断行，不计入 44 个正式模型的汇总门禁。

复现命令（构建目录需为新的 Release stage）：

```bash
NCNN_PERF_THREADS=6 NCNN_PERF_WARMUP=10 NCNN_PERF_ITERS=20 \
NCNN_PERF_MAX_RATIO=0 NCNN_PERF_MODE=end_to_end \
NCNN_PERF_JSON=/path/to/unique.ndjson \
ctest --test-dir /path/to/stage -R '^PerformanceModel\\.' --output-on-failure
```

paired 汇总结果：44 个正式模型的 baseline/candidate ratio p50 为 `1.935/2.040`，candidate compiled median 相对 baseline 为 `+0.78%`；含 `resnet18_winograd` 的 45 行口径为 `1.944/2.060`、`+0.97%`。`pp_ocrv5_server_rec` 6T 为 `8800.68/8795.07 ms`（`-0.06%`），1T 专项为 `52559.31/52506.60 ms`（`-0.10%`）。结论是 server_rec 专项无明显回归，但没有证据宣称全表 speedup。

`perf-stat` 硬件计数器不作为本批次的必需证据：本机没有可用的 `/usr/bin/perf` 时记录为 unknown，不把缺失计数器解释为零事件。运行时 profile 也只用于诊断，不能替代正式 end-to-end ratio。
