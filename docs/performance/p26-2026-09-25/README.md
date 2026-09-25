# P26：有限调优、全量门禁和最终结案

日期：2026-09-25。源码基线 `05be0ae01c85828bcc350121071730b886ee9cf7`（分支 `p8-implementation`），vendored ncnn `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`。

## 结论

**P26 未找到具备既有热点证据和已通过产品化准入门槛的调优候选，因此本轮 bounded-search 候选数为 0，没有改动生产源码。** P20/P21 的 packing/layout 路径没有通过默认化门槛；P22 工程验收不等于性能门槛通过；P23 native INT8 为 `experiment / no-go for productization`；P24 workspace 和 P25 算子专项均为 no-go/unknown。P25 的 7 个 6T profile 有 80.1%–95.4% runtime exclusive time 未知，不能用 inclusive 时间或模型超额排序代替算子归因。无证据时不调阈值、tile 或并行维度，也不新建 model-specific 特例。

P26 的 fresh Release 工程门禁通过，全模型 `end_to_end` 对照已完成并保留 JSON/NDJSON。**最终结论：未达到 M7 最低门槛；不宣称 ncnn parity，也不把快照差异描述为本轮优化收益。** 生产源码没有变化；性能表是当前默认稳定 profile（`NCNN_NATIVE_INT8_PROFILE=OFF`）对同轮 ncnn 的快照。

## 工程门禁

fresh Release stage：`/tmp/ncnn-compiler-stage-p26-20260925-01`，GCC 14.2、LLVM/MLIR 21.1.8、Release、`BUILD_TESTING=ON`、format targets ON、`COMPILER_INSTALL_RPATH=ON`，所有 CMake build 均使用 `--parallel 8`。

| 门禁 | 结果 |
|---|---|
| 全局 `format_check`（clang-format） | 通过 |
| 全局 `tidy`（clang-tidy） | 通过；检查 90 个 first-party translation units |
| 完整 Release build | 通过；约 28 分 20 秒 |
| `numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests` | 均构建通过 |
| 完整 CTest | 449 注册；447 passed、0 failed、2 个既有 upstream skip；768.04 秒 |

两个 skip 是已知 upstream ncnn INT8 reference 崩溃：`PerformanceModel.PPLCNetDocOriInt8`、`PerformanceModel.PPLCNetTextlineOriInt8`。未伪造其性能行。仓库中原 `cmake-build-debug` 的 cache 指向已搬迁的 `/home/zeng/debian/...` 且 Ninja 不存在，旧目录上的两次 lint 尝试失败；已记录于 `format-check-existing-build.log` / `tidy-existing-build.log`，并在上述 fresh Release stage 重跑通过。门禁日志见 `configure.log.gz`、`format-check.log`、`tidy.log.gz`、`full-build.log.gz`、`numerical-build.log.gz`、`full-ctest.log.gz`。

## 与 ncnn 的全模型性能对比

正式口径为 profile-off `end_to_end`，6 线程、warmup=10、iterations=20；使用同一个新鲜 Release 产物及 ncnn revision。CTest 47 项中 45 项 measured（44 official + `resnet18_winograd` diagnostic），另 2 项为上述 upstream skips。finalizer 校验了行数、计时元数据、状态和 gate eligibility。官方统计排除 Winograd diagnostic。

| 集合 | 模型数 | ratio p50 | p90 | max | 几何均值 | ncnn 总时长 | compiled 总时长 | 总超额 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official | 44 | **1.901×** | **2.602×** | **5.066×** | 1.860× | 8,638.230 ms | 10,255.567 ms | 1,617.337 ms |
| Heavy（ncnn ≥100 ms） | 11 | **1.352×** | **1.674×** | **1.708×** | 1.301× | 7,774.884 ms | 8,688.083 ms | 913.199 ms |
| 原始 measured（含 diagnostic） | 45 | 1.950× | 3.490× | 5.200× | — | — | — | — |

Official ratio buckets：`<1.0` 2、`1.0–1.25` 6、`1.25–1.5` 4、`1.5–2.0` 12、`2.0–3.0` 16、`3.0–5.0` 3、`≥5.0` 1。44 official 中 2 个模型快于 ncnn、12 个不超过 1.5×。

绝对总超额前五：

| 模型 | ncnn ms | compiled ms | ratio | excess ms |
|---|---:|---:|---:|---:|
| `yolov5x_seg` | 592.362 | 991.424 | 1.674× | +399.062 |
| `yolov5x` | 471.141 | 804.729 | 1.708× | +333.588 |
| `yolov5l_seg` | 379.547 | 624.213 | 1.645× | +244.666 |
| `pp_ocrv6_medium_det` | 384.529 | 571.219 | 1.486× | +186.690 |
| `yolov5l` | 303.337 | 439.592 | 1.449× | +136.255 |

最高单项 ratio 为 `chineseocr_lite_anglenet` 5.066×；INT8 默认 portable 行中 `pp_ocrv6_medium_rec_int8` 为 4.918×。重模型多数接近 1.3–1.7×，但这不抵消全表 p50/p90 与 max 门槛未达。

模型族统计（44 official，以互斥主族分组）：

| 模型族 | n | ratio p50 | p90 | max | 总超额 |
|---|---:|---:|---:|---:|---:|
| CNN classification | 16 | 2.057× | 2.319× | 2.535× | +277.720 ms |
| Formula/Doc | 3 | 2.221× | 4.497× | 5.066× | +90.761 ms |
| OCR detection | 6 | 1.523× | 2.255× | 2.549× | −220.679 ms |
| OCR recognition | 9 | 2.292× | 4.271× | 4.918× | +178.996 ms |
| YOLO detection | 5 | 1.236× | 1.604× | 1.708× | +526.775 ms |
| YOLO segmentation | 5 | 1.352× | 1.662× | 1.674× | +763.764 ms |

INT8 是跨族正交子集：5 个 official INT8 模型 p50/p90/max = **4.071× / 4.594× / 4.918×**，总超额 +141.677 ms。完整机器统计见 `model-family-summary.json`；分组定义见 `summarize-model-families.py`。

与 P25（同 compiler/ncnn revisions 和 6T、10/20 设置）重复快照相比，p50 1.803→1.901，p90 2.862→2.602，max 5.015→5.066，总超额 1,626.512→1,617.337 ms（−0.56%）。本轮没有源码候选，差异属于跨运行快照，不作因果性能结论。

单线程仅作代表模型诊断（1T、warmup=2、iterations=5，不代表全表）：ResNet18 23.0→70.8 ms（3.08×）、YOLOv5n 69.2→197.4 ms（2.85×）、PP-OCRv6 tiny-rec INT8 4.0→30.0 ms（7.55×）、Formula encoder 132.8→356.3 ms（2.68×）。

## M7 与既有阶段判定

P18 §2.2 的 M7 最低门槛按 AND 关系判定：

| 指标 | P26 | 门槛 | 判定 |
|---|---:|---:|---|
| Official p50 | 1.901× | ≤1.5× | 未达 |
| Official p90 | 2.602× | ≤2.5× | 未达 |
| Heavy p50 | 1.352× | ≤1.4× | 达到 |
| Official max | 5.066× | ≤5× | 未达 |
| 总超额变化 | 1,617.337 ms | 较 P17 降低 ≥50% | 历史指示值约降低 90.46%，但不作严格门槛通过 |

P17 raw official excess 为 16,949.249 ms；该累计对比跨阶段、非 paired，且 P17 对 640² 重模型使用 5/10、P26 本表统一使用 10/20，因此只能记录为历史指示值。p50、p90 和 max 已足以判定 **M7 未达**。不宣称“接近/达到 ncnn”。

路径状态继续遵守既有证据：稳定默认 profile/portable INT8 保持默认；P20 packing 与 P21 layout island 保持原显式 guard/opt-in/fallback 策略；P22 只记已提交工程范围，不把工程验收写成性能通过（其部分 fusion 开关当前默认 true，不能笼统称为 opt-in）；P23 native INT8 保持实验/no-go，不默认化；P24 workspace、P25 hotspot-specialization 不启动。本轮未改任何上述策略。

P23 native-int8 是独立 track，不能与本轮 44-model stable/default portable 表混合：P23 五个目标对 ncnn ratio 为 tiny-rec 2.821×、mobile-rec 1.487×、medium-rec 3.026×、small-rec 2.768×、medium-det 2.189×；只有 mobile-rec 达到对应门槛，且 portable regression 与 Depthwise source-op join 门槛未过，故仍为 `experiment / no-go for productization`。原始独立产物在 `../p23-2026-09-24/native-int8-paired/`。

## 目标文件、sidecar、二进制和资源审计

- `final-all-model-ncnn-comparison.json` 与 `.ndjson`：完整 45 measured rows；44 official 单独统计，diagnostic 单独列出。`raw-all-model-ncnn-comparison.ndjson` 保存 CTest 原始行。每行都有 target、plan hash、build identity；45 行缺失字段为 0，model-specific plan/build identity 均可区分。
- `build-sidecar-integrity.json`：fresh stage 的 230 个 generated shared libraries 全部有可解析 manifest 与 execution-plan sidecar（230/230）；P26 performance row 仍是 profile-off，不把 sidecar 存在误称为 runtime profile coverage。
- `runtime-feature-coverage.json`：formal E2E 未收集 packing/workspace/fusion hit count 或 copy/transpose/allocation bytes，均标为 null/unknown 而非 0；P25 七个诊断 profile 的 event stream 完整，但 exclusive-time attribution 仍不完整，因此不满足新生产调优准入。
- `compiled-model-code-size.json`：45 个 E2E 对比库总计 3,059,869,096 bytes（约 2.850 GiB）。本轮无 candidate，不计算优化前后代码尺寸 delta。
- `build-rss-samples.csv` 是 full Release build 期间的工具进程抽样（639 点、覆盖约 1,280 秒，不等于全 build 全程）：进程 RSS 之和观测峰值 6,363,704 KiB（约 6.069 GiB），单个受监控编译工具观测峰值 3,004,376 KiB（约 2.865 GiB）。`serverdet-runtime-rss-samples.csv` 与 `serverdet-e2e-rss-samples.csv` 分别记录 CTest 默认性能用例、P26 6T E2E `PPOcrv5ServerDetStatic` 的进程 RSS 抽样；后者观测峰值约 11.36 GiB。采样 RSS 是进程级观察值，不等同于逻辑 allocation peak 或整机峰值。
- `binary-audit.txt` / `compiler-tool-audit.txt`：代表 Release 目标的 undefined symbols、FMA 反汇编计数、NEEDED、OpenMP runtime 和 RPATH/RUNPATH。模型 `.so` 依赖 `libomp.so.5`/libc/libm，未发现缺失依赖；MLIR driver/opt 的 RUNPATH 为 `/usr/lib/llvm-21/lib`，`ncnn-compile` 不需要 LLVM RUNPATH。代表库的未定义符号为 libomp/libc/libm 解析的运行时符号；ResNet18 与 server-det static 分别观测 964/5794 条 vfmadd 指令，INT8 tiny-rec 没有 vfmadd（整数路径）。
- `perf-stat-resnet18-retry.csv`：代表 ResNet18 hardware counter 运行 6T、2 warmup/5 iterations 成功；`cpu_core` 组约 5.187×10⁹ cycles、8.003×10⁹ instructions、42.16×10⁶ cache misses，单次诊断耗时不与正式全表混合。最初 wrapper 的后处理误用了 zsh 只读 `status` 参数，故以 exit=0 的 `*-retry.*` 为最终审计结果；初次运行文件仍保留以保持过程可追溯。

## Sanitizer 结果与已知 blocker

另建 Debug ASan+UBSan stage `/tmp/ncnn-compiler-stage-p26-sanitizer-20260925-02`，该 stage 的 format_check/tidy 通过。fresh sanitizer 全构建在 `convolution_int8_term1` fixture 编译阶段复现项目既有 LLVM/MLIR 21 blocker：`AddressSanitizer: use-after-poison`，`SmallVectorTemplateCommon<...>::begin()`，经 `mlir::Dialect::addType` → `BuiltinDialect::initialize` → `MLIRContext`。在 `unit_tests` 的 `NcnnImporterTest.ImportsDynamicInputShapeOverride` 中再次复现同一调用栈；因此 sanitizer full CTest 未运行，不能表述为 sanitizer 全绿。该 blocker 与 P17 已记录的 sanitizer `BuiltinDialect` 初始化问题相同（外部历史记录：`/mnt/ncnn-compiler/docs/ncnn-mlir-performance-optimization-plan.md:1410`），不是本轮生产代码改动引入。本轮 release 完整 build/CTest 结果仍如上，sanitizer blocker 作为明确例外保留。

第一 sanitizer 尝试曾有两个 build 命令意外并发写入 stage-01；该目录被判为无效并保留压缩日志 `sanitizer-build-concurrent-attempt.log.gz`。以上 sanitizer 结论只基于无并发的 fresh stage-02。

## 复现和归档

- 全部命令、CTest filter、6T/1T 测量环境：`commands.txt`。
- 构建配置、source/reference identity：`build-identity.json`。
- Finalizer/summary scripts 可在本目录独立复跑：`finalize-end-to-end.py`、`summarize-end-to-end.py`。
- 与 ncnn 完整性能比较 JSON/NDJSON、原始 logs、sidecar/二进制/内存和 sanitizer blocker 均归档于本目录。
- 计划主文档 `/mnt/ncnn-compiler/docs/ncnn-mlir-performance-optimization-plan-p18.md` 在 compiler Git 根之外；依用户选择，主文档会更新但不复制入仓库。提交仅包含 compiler 仓库 `docs/performance/p26-2026-09-25/`，不纳入既有未跟踪 P22/P24/P25 目录，不推送。
