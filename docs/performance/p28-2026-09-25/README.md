# P28：region 级 wall-exclusive、source-op 溯源与采样式 worker 计时（2026-09-25）

## 结论

本轮按三条预登记要求补完度量，**结论为 qualification-only / no-go for production candidate**：三条度量均闭合并给出可复算数值；但唯一通过 P18 §12.2 进入门槛的专项（YOLO Conv-origin GEMM packing）被既有 P20/P21 配对 A/B 反证为负收益，因此不实施任何生产候选、不默认化 P20/P21/P23 路径、保留全部 fallback。

本轮不是性能改进，也不是 ncnn parity 证据。profile-on 时间只作诊断，不替代 profile-off 正式结果。

## 1. region 级 wall-exclusive（要求 1）

**已闭合。** 修复前 P25 的 unknown exclusive time 占 profile top-level 的 **80.1%–95.4%**，YOLO 有 4.7–5.4 s 无法归属到操作。修复后 attribution report 的 `wall_partition` 是可相加的三分量分解，且 `accounted + unaccounted = top_level`：

```text
top_level_wall_ns = worker_ops_wall_attributed_ns
                  + sequential_ops_exclusive_ns
                  + unaccounted_wall_ns
```

逐操作 wall 采用 `equal_split_across_concurrent_worker_spans` 在并发 worker span 上均分，并按 `sampled_window` 时长做窗口归一；`wall_projection_factor = Σ region_wall / Σ sampled_window` 对逐操作归属值与覆盖时长统一缩放以保持可加性。region 级 wall-exclusive 只在 `sampling_duty <= 1` 且 `sampled_window_wall == region_wall` 且存在 worker span 时给出精确值（`exclusive_semantics = "region_wall_minus_worker_span_union"`），否则保持 `null` / `not_proven`，不用 inclusive 冒充 exclusive。

`run-03` 八模型、`NCNN_PERF_THREADS=6`、2 warmup + 5 timed、7 次 invocation 取中位（duty=1 为全量插桩）：

| 模型 | duty | top_level | worker 逐操作归属 | 顺序算子 exclusive | unaccounted | unacc% |
|---|---|---|---|---|---|---|
| yolov5x_seg | 1 | 1532.1 ms | 1195.3 ms (78.0%) | 83.2 ms (5.4%) | 54.0 ms | **3.5%** |
| yolov5x | 1 | 1276.0 ms | 1022.9 ms (80.2%) | 62.3 ms (4.9%) | 32.5 ms | **2.5%** |
| pp_ocrv6_medium_rec_int8 | 1 | 109.7 ms | 73.6 ms (67.1%) | 31.9 ms (29.1%) | 0.2 ms | **0.2%** |
| pp_ocrv6_small_rec_int8 | 1 | 49.1 ms | 27.0 ms (55.0%) | 18.0 ms (36.6%) | 0.2 ms | **0.4%** |
| pp_ocrv5_mobile_rec_int8 | 1 | 39.0 ms | 16.1 ms (41.4%) | 21.2 ms (54.3%) | 0.2 ms | **0.5%** |
| pp_ocrv6_tiny_rec_int8 | 1 | 13.1 ms | 5.9 ms (45.2%) | 5.6 ms (43.2%) | 0.2 ms | **1.2%** |
| pp_ocrv6_tiny_rec | 1 | 12.9 ms | 4.5 ms (34.8%) | 5.8 ms (44.7%) | 0.2 ms | **1.3%** |
| chineseocr_lite_anglenet | 1 | 6.6 ms | 0.7 ms (10.9%) | 1.0 ms (15.7%) | 2.0 ms | 30.5% |

YOLO 的 unknown 从 4.7–5.4 s 收敛到 **32.5 ms / 54.0 ms** 显式 `unaccounted` 桶（2.5% / 3.5%）；INT8 与小模型 0.2–1.3%。`chineseocr_lite_anglenet` 的 30.5% 是模型太小、并行 region 太碎（234 个 region 中 65 个无采样 span）造成的残余，不据此推断固定开销根因。

## 2. plan 生成侧的 source-op / 层 ID（要求 2）

**已闭合。** `NCNNImporter::tag_source` 现在同时写入 Location 载体：

```cpp
operation->setLoc(mlir::NameLoc::get(
    name, mlir::FileLineColLoc::get(builder_.getContext(),
                                    "ncnn-layer",
                                    static_cast<unsigned>(context.index), 0)));
```

`EmitModelPlan::sourceProvenanceFromLocation` 优先从 Location 反解（属性作 fallback），`isParallelWorkerSite` 判定 `scf.forall`/`scf.parallel`/`omp.parallel` 的直接子操作为 worker 站点，`collectRegionSourceProvenance` 汇总后写入 `operations[]`、`provenance[]`、plan-hash 与 family 条目。无法唯一归属的 worker 站点显式输出 `null` 并推送 `worker_site_source_ambiguous`，不伪造。

Location 必须跨序列化边界存活：`ncnn-mlir-driver` 用 `OpPrintingFlags().enableDebugInfo(true)` 打印，`ncnn-compile` 的 `--ncnn-to-tosa-pipeline` 与 tosa→linalg 中间阶段均带 `--mlir-print-debuginfo`（最终 LLVM-IR 阶段不加，避免 `.ll` 膨胀），lit 断言用 `--mlir-print-local-scope`。attribution revision 为 `attribution-v4`。

实测 join 率（duty=1 与 duty=8 均成立）：

| 模型 | duty=1 worker 站点带 source | duty=8 worker 站点带 source |
|---|---|---|
| yolov5x | **858 / 858** | 582 / 582 |
| yolov5x_seg | **874 / 874** | 573 / 573 |
| pp_ocrv6_medium_rec_int8 | **284 / 284** | 117 / 117 |
| pp_ocrv6_small_rec_int8 | **325 / 325** | 150 / 150 |
| pp_ocrv5_mobile_rec_int8 | **229 / 229** | 196 / 196 |
| pp_ocrv6_tiny_rec_int8 | **131 / 131** | 50 / 50 |
| pp_ocrv6_tiny_rec | **151 / 151** | 108 / 108 |
| chineseocr_lite_anglenet | **261 / 261** | 261 / 261 |

duty=8 的观测站点变少是采样的自然结果（未采样站点不产生 worker event），不是 join 退化：所有被观测到的站点都带 `source_layer`/`source_name`。P23 抱怨的「source_layer 未提供、无法稳定 join」已不成立。

## 3. 降低插桩扰动（要求 3）

**已实现并量化，但有明确代价，不能把占比直接转移到 profile-off 正式时间。**

实现是相干时间窗 duty 采样：门控用 `CLOCK_MONOTONIC_COARSE`（进程级一致且廉价），只有落在 ON 窗口内才用精确 `CLOCK_MONOTONIC` 记 span 时间戳；归一化按「采样窗时长」而不是 `× duty`（并发线程下 `1-(7/8)^6 ≈ 55%` 的并集偏差会让 `× duty` 严重高估，曾出现 `unaccounted_wall_ns = −2,096,784,613`）。`NCNN_PROFILE_SAMPLE_DUTY` 控制 duty。

同一 instrumented build、只改 duty 的 profile-on vs 同批 profile-off 扰动：

| 模型 | profile-off 正式 compiled | duty=1 扰动 | duty=8 扰动 |
|---|---|---|---|
| yolov5x | 743.12 ms | **+301.3%** | **+50.6%** |
| yolov5x_seg | 909.62 ms | **+265.9%** | **+48.9%** |
| pp_ocrv6_tiny_rec | 8.54 ms | +160.2% | +50.2% |
| pp_ocrv6_small_rec_int8 | 30.84 ms | +146.6% | +44.4% |
| pp_ocrv6_tiny_rec_int8 | 9.78 ms | +126.0% | **+19.8%** |
| chineseocr_lite_anglenet | 4.26 ms | +200.2% | +124.7% |
| pp_ocrv5_mobile_rec_int8 | 30.04 ms | +83.3% | **+24.6%** |
| pp_ocrv6_medium_rec_int8 | 100.68 ms | +37.7% | **−0.1%** |

duty=8 把扰动从 +37.7%…+301.3% 压到 −0.1%…+50.6%，但**完整性同步塌陷**：

| 模型 | duty=1 unacc% | duty=8 unacc% | duty=8 无采样 span 的 region |
|---|---|---|---|
| yolov5x | 2.5% | **67.3%** | 453 / 530 |
| yolov5x_seg | 3.5% | **60.9%** | 472 / 546 |
| pp_ocrv6_medium_rec_int8 | 0.2% | **61.1%** | 199 / 227 |
| chineseocr_lite_anglenet | 30.5% | **82.3%** | 140 / 234 |

duty=8 的逐操作 `event_join_status` 变为 `partial`，`joined_invocation_count` 常为 1/7，`calls_estimated`/`wall_attributed_ns` 多为 `null`。**结论：duty=8 只作排序 sanity 与扰动校准；逐操作占比仍以 duty=1 为准，且两者都属 profile-on 诊断，不得替换 profile-off 正式口径。**

## 4. 热点判定与 P18 §12.2 准入

duty=1 逐操作 top（`wall_attributed_share_of_top_level` 为该操作占目标模型 compiled top-level 的比例）：

| 模型 | top 站点 | source | 占 compiled | 占该模型超额上限 |
|---|---|---|---|---|
| pp_ocrv5_mobile_rec_int8 | `model/linalg.matmul#8` | 224:`gemm_4` | **13.74%** | 41.3% |
| pp_ocrv6_small_rec_int8 | `model/linalg.matmul#14` | 166:`gemm_4` | **12.58%** | 18.0% |
| pp_ocrv6_tiny_rec_int8 | `model/linalg.matmul#7` | 90:`gemm_1` | **11.27%** | 16.1% |
| pp_ocrv6_medium_rec_int8 | `model/linalg.matmul#16` | 178:`gemm_4` | 5.72% | 7.3% |
| yolov5x_seg | `model/scf.for#1128` | 347:`conv_133` | 4.91% | 12.6% |
| yolov5x | `model/scf.for#547` | 179:`conv_64` | 2.64% | 6.6% |
| pp_ocrv6_tiny_rec | `model/scf.for#241` | 90:`gemm_1` | 3.04% | 5.0% |
| chineseocr_lite_anglenet | `model/scf.for#8` | 4:`convdw_94` | 0.46% | 0.6% |

以 P27 正式全表（45 行，总超额 **2015.7 ms**，YOLOv5 家族 15 模型超额 **1554.0 ms = 77.1%**）为分母：

- INT8 `linalg.matmul` 家族（4 模型）绝对节省上限合计 14.87 ms = **0.738%** 全表超额 → §12.2 第 2 条失败；覆盖 4 个正式模型 → 第 3 条失败；不阻塞 M6/M7 → 第 4 条失败。第 1 条只在 duty=1（高扰动）下由 mobile/small/tiny 达到 8%，duty=8 下仅 small 保持 ≥8%。
- 小模型固定开销组：top 站点 0.46%/3.04%，`chineseocr_lite_anglenet` 4.79× 的 ratio 是弥散的，四条全部失败。
- YOLO 组 top 站点 `347:conv_133` 单点 2.22% 全表超额（第 2 条通过）、YOLOv5 家族覆盖 15 个正式模型且占 77.1% 全表超额（第 3 条通过）。

### 4.1 YOLO 桶的机制定位

把 plan `contracts[]` 与 attribution 逐操作 wall 按 `provenance` 的 `source_name` join（129/129 全部命中），`yolov5x_seg` duty=1：

| packing 状态 | fallback_reason | 归属 wall | 占 top-level | worker 站点 |
|---|---|---|---|---|
| `unpacked` | `packing_rejected_budget` | **547.7 ms** | **35.7%** | 268 |
| `unpacked` | `packing_rejected_layout` | 182.1 ms | 11.9% | 51 |
| `prepacked_B` | `producer_not_flowing_input` | 163.1 ms | 10.6% | 112 |
| `prepacked_B` | `repeated_producer_input` | 37.3 ms | 2.4% | 112 |

最热单层 `conv_133`（`model/scf.forall#248`，1×1 conv，`tile_k=2880`、`output_channels=320`）本身 6.87% top-level，属于 `packing_rejected_budget`。

根因是 **packed-B 代码尺寸预算被先到者占满**：`PackStaticMatmulNCNN` 的 `max-total-bytes` 默认 100663296（96 MiB），yolov5x_seg 实际 `Σ pack_bytes = 100659200`（99.996% 用满），62 个后续 conv-origin GEMM（还需约 193.5 MB）全部 `packing_rejected_budget`，走 unpacked + `row_major_kxn`。

这在 §12.2 上满足第 1/2/3 条（35.7% 目标模型 compiled、35.7%×909.62 ms ≈ 324.8 ms ≈ 16.1% 全表超额、YOLOv5 家族 77.1%），因此是本轮唯一有资格进入 A/B 的专项。

## 5. 为什么不做这个 A/B：packing 被既有配对证据反证

预登记的候选是「YOLO static Conv-origin matmul 局部 P20 RHS-panel / `conv_packed_gemm` 路径」。定位到的桶正是它，但把它打开会**变慢**：

**P20（`compiler/docs/performance/p20-2026-09-21/`）已做过 `matmul-packing=auto` vs `off` 的 5 轮配对 A/B**：

- `paired-all-model-summary.json`：`model_count = 45`，`auto_better_than_off_count = **2**`。
- 逐模型 `auto_vs_off_improvement_pct`：`resnet18` **−87.37%**、`resnet34` **−87.15%**、`squeezenet_v1_1` **−84.26%**、`efficientnet_b0` **−80.50%**、`b1` −69.74%、`b2` −60.75%、`b3` −65.82%、`resnet50` −9.20%、`yolov5s` **−10.30%**、`yolov5n` −7.61%。仅 `pp_formulanet_plus_s_encoder` +52.72% 与 `pp_ocrv6_medium_det_int8` +9.40% 变好。
- `paired-summary-final4.json`：static deep `improvement_pct = +3.95%` 但 `deep_improvement_gate = false`，`default_productization = "no-go"`。

**P21（`compiler/docs/performance/p21-2026-09-22/`）独立测过 packed Conv/Depthwise + layout island**：5 轮 44 official，candidate compiled delta **p50 +1.80% / GM +2.35%**，12 模型变快、32 个回退，结论 `opt-in / no-go for default`，并明确「不能将 layout island 被选中或 packed kernel 能生成解释为端到端收益」。

因此把 budget-rejected 的 35.7% 热 GEMM 改为 packed，等于把**已测得更慢的路径**套在最热的站点上。96 MiB 预算在这里是**保护性护栏**，不是漏掉的收益。这也直接否定「conv packing 是 YOLO 差距的杠杆」这一假设。

INT8 native 路径同理不作为本轮候选：P23（`compiler/docs/performance/p23-2026-09-24/`）native-int8 相对 portable 改善 tiny −12.61%、mobile −15.91%、medium −44.18%、small −26.00%、medium_det −5.34%，但只有 mobile_rec 达到其 §10.4 门槛，且 P22→P23 portable 全表在 `pp_ocrv6_medium_rec_int8` 上回退 +14.33%/+11.905 ms，结论是 `experiment / no-go for productization`。正式表默认 `NCNN_NATIVE_INT8_PROFILE=OFF`（`compiler/test/Numerical/CMakeLists.txt`）测的是 portable 回退；重跑这条不会产生新知识，且属于被否决路径。

## 6. 判定

- **不实施生产候选**，不提交、不推送；`PackStaticMatmulNCNN` 的 96 MiB `max-total-bytes`、`NCNN_PACKED_CONV_DEPTHWISE=OFF`、`NCNN_NATIVE_INT8_PROFILE=OFF` 均保持原状。
- P20/P21/P23 的 no-go 路径未被全局打开，也未被重新包装为收益。
- 归因度量本身通过验证：region wall 可加、source-op join 100%、duty=8 扰动可控且代价可量化。它们是后续任何专项的前置条件，本轮不宣称性能收益。
- 仍未知、未推断根因的项：`chineseocr_lite_anglenet` 30.5% unaccounted；duty=8 下 60.9%–82.3% unaccounted；`worker_attribution.complete_for_observed_events = false`（`observed_event_count` 与逐 invocation join 的口径差异，逐站点 `event_join_status` 已显式给出）；无 per-layer ncnn 对照，因此「哪里慢」不等于「哪里输给 ncnn」。

## 7. 目录

```text
collect-sampling-evidence.py   # 采样证据采集器（同一 instrumented build 跑 duty=1/8 + 同批 profile-off）
run-03/                        # 权威批次（run-02 因采集器缺陷作废，仅存档）
  summary.json                 # 扰动、identity、per-run wall_partition 与 top 站点
  profile-off/off.ndjson       # 同批 profile-off 正式对照
  profile-on/<model>/
    <model>.plan.json          # 含 contracts[]（packing/fallback_reason/tile_k/output_channels）
    duty1|duty8/
      profile.ndjson           # schema 3
      attribution-report.json  # attribution-v4，含 wall_partition
      run-summary.json
      diagnostic-perf.ndjson
commands.txt
```

## 8. 可复现

```bash
python3 compiler/tools/perf_attribution_report.py \
  --plan=<model>.plan.json --profile=profile.ndjson \
  --perf=diagnostic-perf.ndjson --mode=end_to_end \
  --output=attribution-report.json

python3 compiler/test/Native/check_profile_runtime.py --cc cc \
  --runtime compiler/lib/ProfileRuntime/profile_runtime.c
python3 compiler/test/Native/check_perf_attribution.py
```

全量 CTest 452 registered / 450 passed / 0 failed / 2 个既有 upstream INT8 reference skip
（`PerformanceModel.PPLCNetDocOriInt8`、`PerformanceModel.PPLCNetTextlineOriInt8`）。
