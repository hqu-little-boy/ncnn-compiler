# P23 native INT8：实现验收与全模型性能证据（2026-09-24）

## 结论

**工程实现和 fresh 正确性门禁通过；P23 不满足产品化 opt-in 验收，结论为 `experiment / no-go for productization`。** 不启用默认 VNNI，不添加 runtime dispatcher，也不宣称已达到 ncnn parity。

未通过/未闭环的硬门槛：

1. P22→P23 portable 五轮全表对照中，portable stable 回归门槛失败；`pp_ocrv6_medium_rec_int8` 的中位延迟回退 14.33%（+11.905 ms），另有两个轻模型超过 §3.4.1 的 light regression 条件。
2. P23 native-int8 对照 portable 虽改善 INT8 目标集，但五个目标中只有 mobile_rec 满足其性能门槛；tiny_rec、medium_rec、small_rec、medium_det 均未达到 P23 §10.4 的 ncnn ratio 阈值。
3. Profile-on native 运行确认所选静态 multiplier-1 Depthwise site 都实际执行，但 portable baseline 与 native plan 的 Depthwise ID 分别是 `linalg.depthwise_conv_2d_nhwc_hwcm#N` 和 `scf.forall#N`，且 `source_layer` 未提供。无法以稳定 source-op ID 完成“baseline eligible Depthwise exclusive-time ≥90%”严格 join；该门槛保持 **unknown/not passed**，不以静态选择数或未匹配事件当作覆盖率。

本机完成 AVX-VNNI（Alder Lake，Intel Core i5-12600KF）验证；本机没有 AVX512-VNNI，因此不能据此宣称 AVX512-VNNI 产品化。AVX-VNNI-INT8 继续作为独立 capability 并 fallback，当前没有匹配 backend。

## 实现范围

- 增加显式 target-bound `native-int8-v1` AOT profile；只有 target probe 报告 AVX-VNNI / AVX512-VNNI 且固定宽向量路径可用时选择 VNNI row-dot。稳定 `auto` 保持 portable，unsupported capability 保留可审计 fallback。
- 静态 INT8 RHS 使用 `p23-int8-panel-row-kpad64-v1` 物理格式（16 行 panel、逻辑 K-contiguous、物理 K stride 按 64 字节补齐），并把 P20 f32 packing 与 P23 INT8 packing 共同计入每模块预算。
- 修正 `linalg.matmul_transpose_b` 对 N 维常量 RHS 的错误分块：旧 `tileUsingSCF` 形态会把 64 行常量变成固定 32 行 panel，丢失动态 N offset。当前为该转置 B 形式保留完整 N RHS，避免 pack 和 portable fallback 读取错误权重。该 correctness 修正也改变 portable 的执行形状；其性能影响由本次 paired gate 实测，未通过 stable regression gate。
- 增加 packed INT8 plan/transform、预算、target fallback 测试和模型级 3 输入逐位 native-vs-portable 验证。模型级对照在同一输入上比较 native-int8 与同源 portable artifact，不把与 ncnn 的既有整图数值差异误当作 backend 差异。
- `NCNN_NATIVE_INT8_PROFILE` 是 Numerical/Performance fixture 的验收开关，默认 OFF；只在显式测试配置下将五个 P23 静态 INT8 fixture 编译为 native-int8，并在 native 测试阶段额外构建 portable comparator。它不改变稳定产物的 CLI 默认值。

## Fresh 工程门禁

三个 Release stage 均使用新的 `/tmp` 目录、LLVM/MLIR 21.1.8、`COMPILER_INSTALL_RPATH=ON`、`BUILD_TESTING=ON`；每条 `cmake --build` 均使用 `--parallel 8`。`format_check`、全局 `tidy`、完整构建、三个 numerical targets 和完整 CTest 全绿。CTest 中只有既有 upstream ncnn INT8 reference crash skip：`PPLCNetDocOriInt8`、`PPLCNetTextlineOriInt8`。

| 构建 | 目录 | CTest | 说明 |
|---|---|---:|---|
| clean P22 portable baseline | `/tmp/ncnn-compiler-stage-p23-portable-baseline-20260924-b` | 449/449，0 failed，2 known skips | clean archive：`1eea764`；ncnn submodule `a4d2ea1d` |
| P23 portable | `/tmp/ncnn-compiler-stage-p23-portable-final-20260924-b` | 449/449，0 failed，2 known skips | native fixture profile OFF |
| P23 native-int8 | `/tmp/ncnn-compiler-stage-p23-native-final-20260924-c` | 454/454，0 failed，2 known skips | `NCNN_NATIVE_INT8_PROFILE=ON`；包含五个 native-vs-portable bitwise test |

完整日志以 gzip 保存于 [`validation/`](validation/)。native target CLI、真实 VNNI object/assembly、signedness correction 与 modulo-2³² 数值 oracle 测试均在完整 CTest 中通过。五个 P23 模型各以三个固定随机输入比较 native 与 portable 输出的 float bit pattern，全部逐位一致。详细结果见 [`numerical-results.json`](numerical-results.json)。

## 五轮 6-thread paired 性能

每轮均为 B,C / C,B / B,C / C,B / B,C；每 arm 每次 45 行（44 official + `resnet18_winograd` diagnostic），6 threads、warmup=10、iterations=20、profile-off `end_to_end`。分母使用同批 ncnn 参考运行；候选相对 baseline 的 delta 与候选相对 ncnn 的 ratio 分开报告。

### Portable regression：clean P22 → P23 portable

| subset | compiled median delta（P23 vs P22） | 更快模型数 | candidate vs ncnn ratio p50 |
|---|---:|---:|---:|
| 44 official | **+0.29%**（回退） | 20/44 | 1.772× |
| 11 heavy | −0.07% | 6/11 | 1.372× |

candidate-vs-ncnn official 44：p50=`1.772×`、p90=`2.686×`、max=`5.526×`、GM=`1.841×`；heavy 11：p50=`1.372×`、p90=`1.655×`、max=`1.693×`。

P18 §3.4.1 阻断项：`pp_lcnet_x1_0_doc_ori` `+15.64%/+0.843 ms`、`pp_lcnet_x1_0_textline_ori` `+10.22%/+0.584 ms`、`pp_ocrv6_medium_rec_int8` `+14.33%/+11.905 ms`。因此 portable stable regression gate **失败**。不能因整体 median 很小而忽略逐模型 hard gate。

### P23 native-int8：P23 portable → native-int8

| subset | compiled median delta（native vs portable） | 更快模型数 | candidate vs ncnn ratio p50 |
|---|---:|---:|---:|
| 44 official | **−0.83%** | 29/44 | 1.786× |
| 11 heavy | +0.20% | 5/11 | 1.379× |

candidate-vs-ncnn official 44：p50=`1.786×`、p90=`2.757×`、max=`4.779×`、GM=`1.804×`；heavy 11：p50=`1.379×`、p90=`1.665×`、max=`1.706×`。

| P23 INT8 目标 | native compiled | ncnn | native/ncnn | §10.4 门槛 | 结论 |
|---|---:|---:|---:|---:|---|
| `pp_ocrv6_tiny_rec_int8` | 7.846 ms | 2.782 ms | **2.821×** | ≤2.5× | 未达 |
| `pp_ocrv5_mobile_rec_int8` | 27.084 ms | 18.208 ms | **1.487×** | 不退化 vs portable | 达到；native 较 portable 快 15.91% |
| `pp_ocrv6_medium_rec_int8` | 54.670 ms | 18.070 ms | **3.026×** | ≤2.5× | 未达 |
| `pp_ocrv6_small_rec_int8` | 24.998 ms | 9.032 ms | **2.768×** | ≤2.5× | 未达 |
| `pp_ocrv6_medium_det_int8` | 7.367 ms | 3.365 ms | **2.189×** | ≤1.8× | 未达 |

portable 与 native 的目标集变化中，tiny_rec `−12.61%`、mobile_rec `−15.91%`、medium_rec `−44.18%`、small_rec `−26.00%`、medium_det `−5.34%`；这些改善仍不足以达到 ncnn ratio 目标。不得以 candidate-vs-portable 改善替代 candidate-vs-ncnn 门槛。

所有模型原始 paired run、分块 CTest 日志、225 条 paired-round 记录、summary，以及最终 45-row candidate-vs-ncnn JSON/NDJSON 均保留在：

- [`portable-regression/`](portable-regression/) — 45-row 最终文件 `final-all-model-ncnn-comparison.json` / `.ndjson`；
- [`native-int8-paired/`](native-int8-paired/) — 45-row 最终文件 `final-all-model-ncnn-comparison.json` / `.ndjson`。

完整门槛汇总见 [`summary.json`](summary.json) 和 [`portable-regression/paired-summary.json`](portable-regression/paired-summary.json)、[`native-int8-paired/paired-summary.json`](native-int8-paired/paired-summary.json)。这些两个 JSON 是不同 profile track 的最终模型对 ncnn 比较，均完整保留 45 个 measured model 行，官方统计另排除诊断行。

## Depthwise runtime profile 诊断

[`profile-int8-depthwise.py`](profile-int8-depthwise.py) 为五个模型分别编译 profile-on portable/native 诊断产物并执行一次 direct-ABI invocation；plan、编译日志、schema-2 profile NDJSON 和 join 审计保存在 [`runtime-coverage/`](runtime-coverage/)。五个 profile 均 `complete=true`、`event_mismatch_count=0`。native static canonical multiplier-1 depthwise selected site 的实际 callback 调用数为：tiny `10/10`、mobile `14/14`、medium-rec `15/15`、small-rec `14/14`、medium-det `17/17`，总计 `70/70`。

portable fallback 的单次诊断也观测到 depthwise callback 与 exclusive time，但 P23 native 的选中 site 是 `scf.forall#N`，portable 侧是 `linalg.depthwise_conv_2d_nhwc_hwcm#N`；两侧 plan `source_layer` 为 null，medium-det 还存在 89 对 85 个 site 的数量差异。当前不能证明跨 profile 的逐 source-op 身份映射，因此 baseline eligible-depthwise exclusive-time 覆盖率仍为 **unknown**，不能报告成 0% 或 100%，P23 ≥90% runtime 权重门槛不通过。

## 可复现命令和状态

Fresh test stage 配置与门禁命令参见 `validation/` 中的日志及 P18 计划 §14。 paired benchmark 使用 `run-paired.py`、`aggregate-paired.py`，以本 README 同列的两个 stage 为输入。runtime profile 重建命令模板已实现在 `profile-int8-depthwise.py` 中；profile-on timing 仅作诊断，不替代 profile-off formal E2E。

P23 当前为 **实验性 opt-in / no-go for productization**：保持 portable/`auto` 默认，不做 dispatcher。后续需要先修复 native target 模型 ratio、portable full-table regression，并补齐稳定 source-op ID → baseline runtime exclusive-time join，再重新执行全新门禁与五轮 paired A/B。
