# P25 qualification-only / no-go 记录（2026-09-25）

## 结论

P25 的优化准入条件本轮**未能被证明**，因此没有选定或实现生产专项，不运行候选 paired A/B，也不提交代码。此结论不是“已证明候选算子不热”；而是当前 6 线程 runtime profile 对 OpenMP 并行算子缺少 exclusive-time 归因，无法按 P18 §12.2 的门槛将模型超额可靠地归属到具体算子。不能将包含嵌套/并行工作的 inclusive 时间当作 exclusive 时间，也不能由模型超额排序代替算子级证据。

针对五个当前最大超额模型、PP-OCRv6 medium-det 和 EfficientNet-B3 共采集七个 profile-on 模型。所有 schema-2 profile 的事件流自身标记 `complete=true`，事件没有 mismatch；但 attribution 报告对**七个模型均为 `runtime.complete=false`**，唯一不完整原因为 `runtime_exclusive_time_unknown`。目标模型 top-level runtime 中无法归属 exclusive time 的比例为 **80.1%–95.4%**，且 top-cost rows 主要是 `scf.forall` / `scf.parallel`，时间基准为 `inclusive_unknown_exclusive`。因此没有专项能够以该证据证明“占目标模型 compiled time ≥8%”或其它 P25 准入条件。

额外尝试对 `yolov5x_seg` 生成单线程 profile；编译在执行前失败，`--ncnn-memref-to-llvm-pipeline=threads=1` 后仍有 `scf.parallel`，无法降到 LLVM，详见 `profile-logs/yolov5x-seg-1t-compile.log`。这不是已通过的单线程 profile，也不作为任何热点结论。

P25 不更改编译器源码。没有 operator microbenchmark/golden/paired A/B 候选结果，故不宣称达成 P25 优化收益、M6 或 M7。后续应先闭合 OpenMP worker 的 exclusive-time attribution（或提供等价、可交叉验证的算子级 runtime 证据），再重新按 P18 §12 选择专项。

## 新鲜基线与统一工程门禁

- 源码基线：`05be0ae01c85828bcc350121071730b886ee9cf7`，分支 `p8-implementation`；vendored ncnn：`a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`。
- 全新构建目录：`/tmp/ncnn-compiler-stage-p25-baseline-20260925-01`，Release，LLVM/MLIR 21.1.8，`BUILD_TESTING=ON`、全局 format/tidy targets 开启、`COMPILER_INSTALL_RPATH=ON`，构建并行度 8。
- 全局 `format_check`、全局 `tidy`、完整 build、`numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests` targets 均通过。`tidy` 检查了 90 个 first-party translation units；日志没有 `warning:` 或 `error:` 诊断。
- 完整 CTest：449 registered，447 passed、0 failed、2 skipped，总计 766.13 s。唯一 skip 是既有 upstream ncnn INT8 reference 崩溃：`PerformanceModel.PPLCNetDocOriInt8`、`PerformanceModel.PPLCNetTextlineOriInt8`。无新增 skip。
- 完整日志：`full-build-baseline.log`、`format-check-baseline.log`、`tidy-baseline.log`、`numerical-build-baseline.log`、`full-ctest-baseline.log`；完整配置记录见 `configure-baseline.log` 和 `build-identity-baseline.json`。

## 当前版本与 ncnn 的全模型性能快照

正式 profile-off 基准运行使用 6 线程、10 warmups、20 iterations、`end_to_end`。有效记录 45 行：44 个 official 模型和 `resnet18_winograd` diagnostic；两项已知 CTest upstream skip 没有伪造测量行。这是当前版本快照，不是 P25 优化的前后因果 A/B。

| 集合 | 模型数 | ratio p50 | p90 | max | 几何均值 | ncnn 总时间 | compiled 总时间 | 总超额 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official | 44 | **1.803×** | **2.862×** | **5.015×** | 1.852× | 8,616.392 ms | 10,242.904 ms | 1,626.512 ms |
| heavy（ncnn ≥100 ms） | 11 | **1.340×** | **1.661×** | **1.664×** | 1.309× | 7,765.187 ms | 8,682.410 ms | 917.223 ms |

绝对超额前五为 `yolov5x_seg` +392.315 ms、`yolov5x` +319.137 ms、`yolov5l_seg` +250.976 ms、`pp_ocrv6_medium_det` +168.872 ms、`yolov5l` +147.804 ms。基于该排序采集了 profile，但这些数值本身不证明对应模型的某个尾部算子是热点。

最终完整模型对 ncnn JSON/NDJSON 保存在 `final-all-model-ncnn-comparison.json` 和 `final-all-model-ncnn-comparison.ndjson`；其原始构建批次文件为 `baseline-all-model-ncnn-comparison.json` 和 `.ndjson`。44 official 与含 diagnostic 的 45-row 表格分别见 `official-summary.txt` 和 `all-measured-summary.txt`，机器统计见 `comparison-summary.json`。

## P25 runtime profile 准入审计

| 模型 | 6T profile top-level | unknown exclusive time | unknown 比例 | attribution |
|---|---:|---:|---:|---|
| `yolov5x_seg` | 1,848.8 ms | 1,758.3 ms | 95.1% | 不完整 |
| `yolov5x` | 1,543.0 ms | 1,472.8 ms | 95.4% | 不完整 |
| `yolov5l_seg` | 1,361.7 ms | 1,287.7 ms | 94.6% | 不完整 |
| `yolov5l` | 1,061.4 ms | 1,011.8 ms | 95.3% | 不完整 |
| `yolov5m_seg` | 821.3 ms | 772.9 ms | 94.1% | 不完整 |
| `pp_ocrv6_medium_det` | 936.0 ms | 749.6 ms | 80.1% | 不完整 |
| `efficientnet_b3` | 231.8 ms | 212.8 ms | 91.8% | 不完整 |

Profile-on 是全零输入、2 次 warmup + 5 次采样的诊断运行，绝不作为正式 benchmark timing。每个模型的 execution plan、schema-2 per-invocation `profile.ndjson`、`diagnostic-perf.ndjson`、attribution report、编译/运行日志在 `profile-on/<model>/` 与 `profile-logs/` 中；大型 profile-on `.so` 已移除，可按 `profile-commands.txt` 和 `collect-profile-evidence.py` 重建。初次采集器把 `compile.log` 放进 ncnn-compile 要求空的输出目录而失败，失败目录及命令被保存在 `profile-on-attempt-01/` 和 `profile-commands-attempt-01.txt`；修正日志位置后七个 profile 均成功生成。

## 可复现命令和已知限制

基线 CMake configure/build/CTest/performance 命令及模型结果处理见 `commands.txt`。profile 生成脚本为 `collect-profile-evidence.py`，单库直接调用脚本为 `run-profiled-model.py`，统一入口参数和 profile output identity 均保存在该目录。

P25 本轮不生成候选代码、不运行候选全门禁/paired A/B、不提交。P18 计划文件位于 workspace 根目录、compiler Git 仓库之外；本轮仅在该指定位置更新 P25 的 qualification/no-go 结论，不声称此文件属于 compiler 仓库提交。
