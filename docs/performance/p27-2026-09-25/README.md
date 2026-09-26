# P27：可否证归因闭环与三类 cohort 资格审计（2026-09-25）

## 结论

按“先修复归因、再取证、最后才谈候选”的顺序执行完毕。三步闭环产出：

1. **并行 worker 归因已修复并通过语义测试**：worker 执行时间不再被并行 wall span 吞掉，报告中
   worker CPU-exclusive 与按操作 ID 的 worker wall-union 分栏，parallel-region wall-exclusive
   仍保持 unknown（未伪造）。8 个 cohort 模型的 observed worker 事件 join 完整率 100%。
2. **比较边界已核对并量化**：ncnn/compiled 在 warmup 与计时阶段均交替执行（逐行记录实际次序），
   进程与全部观测 task 被限制在 ncnn big-core mask（0-11）并逐行验证，profile-on / profile-off /
   prepared 三条数据用同一 input hash 严格校验。`end_to_end` 的 ncnn 侧仍含 extractor/输入输出
   每轮构造——这已由 prepared 对照量化，而不是被“校正”掉。
3. **三类 cohort 无人达到预登记的专项准入门槛**，因此**没有实现任何性能候选、没有做 paired A/B**。
   这是预登记停止条件下的结论：不是“已证明候选不热”，而是“当前证据不足以提名一个合格候选”。

P20 packing / P21 packed Conv/Depthwise / P23 native INT8 / P24 workspace 的 opt-in、fallback
与 no-go 状态**未被触碰**，默认稳定路径不变。

## 源码与工程门禁

- 编译器仓库 `233a2ca`（分支 `p8-implementation`），vendored ncnn
  `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`。
- 全新 `/tmp/ncnn-compiler-stage-p27-20260925-01`，Release，GCC 14.2.0 / LLVM-MLIR 21.1.8，
  `BUILD_TESTING=ON`、`COMPILER_ENABLE_FORMAT_TARGETS=ON`、`COMPILER_INSTALL_RPATH=ON`，parallel=8。
- `format_check` 通过；`tidy` 通过（90 个 first-party translation units，无 user-code 诊断）；
  三个 numerical targets 与完整构建通过。
- 完整 CTest：**451 registered，449 passed、0 failed、2 个既有 upstream INT8 reference skip**
  （`PerformanceModel.PPLCNetDocOriInt8`、`PerformanceModel.PPLCNetTextlineOriInt8`，上游 ncnn
  INT8 参考路径崩溃，见 `docs/ncnn-suspected-issues.md`）。未新增 skip。
- 本轮 tracked 源码改动仅限归因与基准契约（见下节）；P20/P21/P23/P24 的产品化边界未改。

完整身份与环境记录见 [`build-identity.json`](build-identity.json)。

## 1. 并行 worker exclusive-time 归因修复

### 根因

`ProfileRuntime` 的事件栈是 `_Thread_local`：并行区事件由进入线程计时，worker 回调落在各自的
TLS 栈上，无法作为该 wall span 的可减去子时长；因此 `parallel` 记录被标 `exclusive_known=0`，
`perf_attribution_report.py` 如实把这部分计入 `unknown_time_ns`。P25 的 80.1%–95.4% unknown
即由此而来，而不是 join 或事件丢失。

### 语义（attribution-v3 / profile schema 3）

- 新增 `worker_operation` 事件类别与 `time_domain=worker_cpu`：worker 本地嵌套栈上计算
  **worker CPU-exclusive**（同一 worker 内减去子事件），跨线程聚合为 `exclusive_cpu_total_ns`。
  该量是各 worker 工作时间之和，**可以超过 wall 时间**，不得与顶层 wall 时间相减或当作 wall 占比。
- 同时按操作 ID 维护**跨 worker wall-union**（`worker_wall_union_ns`）：多 worker 同一操作的
  时间区间取并集，再跨调用累加；标注 `wall_union_semantics=per_operation_non_additive`。
  占比分母是 profile 顶层 wall 时间（`wall_union_share_of_top_level`），逐操作并集**不可相加**。
- **parallel-region wall-exclusive 保持 null**：没有实现跨线程 interval-union 的 region 级
  wall-exclusive 证明，就不改写 `exclusive_ns` 的含义。unknown 仍是 unknown，不是 0。
- schema/identity 版本化为 `attribution-v3`；新增 `input_hash`（perf 行与 profile 行必须一致，
  不一致直接拒绝 join）。未覆盖的 worker 事件计 `runtime_worker_operation_unattributed`，
  观察到但未完整证明的计 `runtime_worker_attribution_incomplete`。

### 插桩范围

`InstrumentNCNNProfile` 对 `scf.forall` / `scf.parallel` / `omp.parallel` 的**直接子作用域**
（`scf.for`、`linalg.*`）打 worker 回调，不逐 lane、不逐标量 op 插桩。生成的 MLIR 保留
`ncnn.profile_instrumented` 标记，profile-off 正式产物不带任何 runtime instrumentation。

### 测试

- `test/Native/check_profile_runtime.py`：真实多 worker、嵌套 worker 事件、并发聚合、
  wall-union 上界小于顶层 wall、`parallel.exclusive_ns` 仍为 null、schema-3 `input_hash`。
- `test/Native/check_perf_attribution.py`：worker CPU-exclusive 与 wall-union 分栏、
  wall-union 占比分母正确、worker CPU 时间不污染 wall exclusive unknown、input hash 不匹配拒绝。
- `test/Transforms/EmitModelPlan/profile-hooks.mlir`：worker 回调出现在并行作用域内，
  `__ncnn_profile_worker_event_{begin,end}` 声明齐全。
- `PerformanceOrder` 两个单测：ncnn/compiled 交替次序、input hash 确定性。

## 2. 比较边界核对

### 已修正

- **执行次序**：warmup 与计时均按 `ncnn_then_compiled` / `compiled_then_ncnn` 交替；
  实际次序写入每行 JSON `order.warmup` / `order.timed`，并由 cohort 脚本校验交替成立。
  （既有 baseline↔candidate 五轮 A/B↔B/A 交替保持不变；此前缺的是每轮内两后端的次序平衡。）
- **CPU 放置对称**：`apply_benchncnn_threading` 取 ncnn big-core mask 与进程允许集的交集，
  用 `sched_setaffinity` 限制整个进程，再让 ncnn `set_cpu_powersave(2)` 与 compiled OpenMP
  都继承同一允许集；`verify_benchmark_cpu_placement` 逐 task 读 `/proc/self/task/*/stat` 等价
  验证，全部观测 task 匹配才 `gate_eligible`。本机有效集 `0-11`，观测 task 全部匹配。
  仍记录在案的事实：ncnn 测试链 `libgomp.so.1`、compiled `.so` 链 `libomp.so.5`，是两个
  OpenMP runtime；本轮只统一 CPU 允许集与线程数（6T），不声称两个 runtime 的调度行为完全一致。

### 已量化、未“校正”

`end_to_end` 的计时边界不对称是刻意保留的历史口径：ncnn 每轮新建 Extractor/Input Mat、
bind/extract/flatten；compiled DSO 与输入输出缓冲在计时前就绪，直接调用导出函数。
同批 `prepared` 诊断把 ncnn 的 extractor/输入准备移到计时外，用于**量化**这部分成本：

| model | e2e ncnn (ms) | prepared ncnn (ms) | prepared/e2e |
|---|---:|---:|---:|
| yolov5x | 522.345 | 490.885 | 0.940 |
| yolov5x_seg | 583.437 | 603.708 | 1.035 |
| pp_ocrv6_medium_rec_int8 | 23.282 | 26.258 | 1.128 |
| pp_ocrv6_small_rec_int8 | 11.257 | 9.347 | 0.830 |
| pp_ocrv5_mobile_rec_int8 | 19.341 | 24.505 | 1.267 |
| pp_ocrv6_tiny_rec_int8 | 2.991 | 2.660 | 0.889 |
| pp_ocrv6_tiny_rec | 3.365 | 4.026 | 1.196 |
| chineseocr_lite_anglenet | 1.000 | 0.820 | 0.820 |

边界成本在 ±20% 带内且方向随模型漂移，**不构成把 ratio 解释成“测试假象”的依据**；它同时说明
ratio 不是纯内核速度。两条口径分开存档（`cohort-run-04/profile-off/{end-to-end,prepared}.ndjson`），
`prepared` 明确 `gate_eligible=false`。所有行的 input hash 在 end_to_end / prepared / profile-on
三者一致（脚本强制校验）。

## 3. 正式 profile-off 全模型基线（P27，不与历史批次混算因果）

6 线程、10 warmup、20 iterations、`end_to_end`、profile-off、后端次序交替。45 行 = 44 official
+ `resnet18_winograd` diagnostic；两个 upstream skip 不伪造行。

| 集合 | n | p50 | p90 | max | GM | ncnn 总时长 | compiled 总时长 | 总超额 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official | 44 | **1.844×** | **2.560×** | **4.790×** | 1.870× | 8,891.205 ms | 10,856.058 ms | 1,964.853 ms |
| heavy（ncnn ≥100 ms） | 11 | **1.437×** | **1.701×** | **1.703×** | 1.347× | 7,976.927 ms | 9,165.981 ms | 1,189.054 ms |
| 含 diagnostic | 45 | 1.85× | 3.32× | 4.96× | — | — | — | — |

42/44 慢于 ncnn、2/44 快于 ncnn（`pp_ocrv5_server_rec` 0.96×、`pp_ocrv5_server_det_static` 0.90×）。
绝对超额前五：`yolov5x_seg` +452.820 ms、`yolov5x` +346.878 ms、`yolov5l_seg` +258.176 ms、
`pp_ocrv6_medium_det` +210.227 ms、`yolov5l` +184.660 ms。最高 ratio 为
`chineseocr_lite_anglenet` 4.790×（绝对只多 ~3.2 ms），其次 `pp_ocrv6_medium_rec_int8` 4.633×。

本批是**计时边界与 CPU 放置修正后的新口径**，不与 P25/P26 数字做跨批次收益对账；三者的差异
应先按口径差理解。原始行与汇总见
[`final-all-model-ncnn-comparison.ndjson`](final-all-model-ncnn-comparison.ndjson)、
[`comparison-summary.json`](comparison-summary.json)、[`official-summary.txt`](official-summary.txt)。

## 4. 三类 cohort 归因资格审计

同一 source/target/输入的独立 instrumented 构建（`NCNN_PERF_PROFILED=1` + library override），
6T、2 warmup + 5 timed，profile-on 仅诊断；worker 归因对观察到的事件 100% join。

| model | 组 | e2e ratio | 绝对超额 | top worker site | wall-union 占比 | worker CPU | unknown wall | profile 扰动 | profile 二进制增长 | 编译墙钟 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| yolov5x | YOLO 绝对超额 | 1.897 | +468.6 ms | `model/scf.for#6` | 3.00% | 192.7 ms | 4,749.7 ms | +390.5% | +0.02% | 219.6 s |
| yolov5x_seg | YOLO 对照 | 1.686 | +400.3 ms | `model/scf.for#1125` | 4.09% | 238.7 ms | 5,372.6 ms | +458.0% | +0.02% | 222.6 s |
| pp_ocrv6_medium_rec_int8 | INT8 高 ratio | 4.347 | +77.9 ms | `model/linalg.matmul#16` | 6.21% | 41.3 ms | 131.1 ms | +62.6% | +0.35% | 13.3 s |
| pp_ocrv6_small_rec_int8 | INT8 高 ratio | 3.794 | +31.5 ms | `model/linalg.matmul#14` | 7.04% | 25.7 ms | 70.4 ms | +118.7% | +1.15% | 11.2 s |
| pp_ocrv5_mobile_rec_int8 | INT8 参考 | 1.798 | +15.4 ms | `model/linalg.matmul#8` | 15.73% | 24.9 ms | 38.4 ms | +81.7% | +1.22% | 8.7 s |
| pp_ocrv6_tiny_rec_int8 | INT8 高 ratio | 3.665 | +8.0 ms | `model/linalg.matmul#7` | 8.95% | 5.2 ms | 19.0 ms | +143.1% | +2.96% | 5.9 s |
| pp_ocrv6_tiny_rec | 小模型对照 | 2.282 | +4.3 ms | `model/scf.for#241` | 8.71% | 2.6 ms | 26.2 ms | +329.3% | +0.79% | 8.5 s |
| chineseocr_lite_anglenet | 小模型固定开销 | 4.688 | +3.7 ms | `model/scf.for#60` | 1.16% | 0.1 ms | 9.4 ms | +194.0% | +4.05% | 7.0 s |

（wall-union 占比分母是 profile-on 顶层 wall；worker CPU 是各 worker 工作时间之和，可超过 wall。
二者不可互换，也不可相加。`unknown wall` 是 parallel-region wall-exclusive 仍未知的部分。）

### 为何没有候选通过 P18 §12.2

- **YOLO（最大绝对超额面）**：修复后的归因显示超额**不集中**——`yolov5x` 有 858 个 worker
  站点，最大站点 wall-union 占比仅 3.00%，`yolov5x_seg` 4.09%；且 parallel wall-exclusive
  unknown 4.7–5.4 s，与 profile-on 顶层同量级。远未达 ≥8% 的单站点门槛，更谈不上“一个站点
  就能解释 +468 ms”。
- **INT8-rec 高 ratio 组（真正的目标组）**：`medium_rec_int8` 6.21%、`small_rec_int8` 7.04%、
  `tiny_rec_int8` 8.95% —— 只有 tiny 越过 8%，而其 profile 扰动 +143.1%，占比无法转移到
  profile-off 正式 compiled time。更重要的是全部 top worker 站点的 `source_layer` /
  `source_name` 为 **null**：无法建立 P18 §12.2 / P23 前置要求的“稳定 source-op ID →
  baseline runtime exclusive-time”映射，也就无法把候选限定到一个 source-op 而不是写死形状。
  `mobile_rec_int8` 的 15.73%（`model/linalg.matmul#8`，其 `kernel_contract` 记着
  `packing_rejected_layout` fallback）是**唯一看起来可操作的线索**，但它落在参考组而非目标组、
  正式 ratio 已是 1.798×、且在 +81.7% 扰动下测得、source-op 身份未闭环——按预登记条件不够格。
- **小模型固定开销**：`anglenet` top 站点 1.16%，`tiny_rec` 8.71% 但扰动 +329.3%。
  P24 的 alloc/dealloc 时间覆盖仍不完整，workspace/arena 方向维持 no-go。

预登记停止条件原文兑现：“profile-on 时间绝不替代正式结果”“任一目标组归因不完整则不进入该组
优化，报告未知而非推断根因”“若扰动过大，停止在 instrumentation validation，不开展性能 candidate”。
因此本轮**不实现候选、不做 paired A/B**；这等价于把“三类 cohort 中存在一个可提名的合格热点”
这一假设在当前插桩保真度下判为**未证实**。

## 5. 资源与工程预算（无 candidate，故为诊断记录）

- profile 二进制增长 0.02%–4.05%（预算 ≤20%）；profile 编译墙钟 5.9–222.6 s
  （yolov5x/x_seg 用登记的 900 s 大件预算，其余 300 s）。
- 批量 profile-off 进程树峰值 RSS（/proc 采样，含 ncnn+compiled 全部 task）：
  `end_to_end` 4,876,304 KiB、`prepared` 4,474,312 KiB；单模型 profile-on 峰值
  37.5 MiB（anglenet）– 4.87 GiB（yolov5x_seg）。这些是进程树 RSS 采样，不是逻辑
  allocation peak；两者在报告中分栏，不互相替代。**没有 candidate，就没有 candidate
  RSS 对比**——若日后做候选，需在同一采样器下预登记 baseline 波动上界与绝对上限。

## 6. 尝试记录

`cohort-run-02/`、`cohort-run-03/`、顶层 `profile-off/`、`profile-on/` 是三次失败尝试的残留，
按项目惯例保留而不是覆盖：

| 尝试 | 结果 | 原因 |
|---|---|---|
| attempt-01（顶层 `profile-off/`、`profile-on/`） | 无任何测量行 | `/usr/bin/time` 在本机不存在，采集器启动即失败；已改为 /proc 进程树 RSS 采样 |
| attempt-02（`cohort-run-02/`） | profile-off 8 行正常；profile-on yolov5x 编译失败 | 采集器把 `compile.log` 放进了 ncnn-compile 要求为空的输出目录（与 P25 attempt-01 同类）；已把编译日志移到 `profile-logs/` |
| attempt-03（`cohort-run-03/`） | yolov5x / yolov5x_seg 成功，OCR INT8 编译失败 | OCR 资产按 `.ncnn.param` 命名查找，实际为 `.param`/`.bin`；已按 asset group 分支 |

正式结论只引用 **`cohort-run-04/`**（8 模型完整、`cohort-summary.json` 为最终汇总）。

## 7. 产物与复现

```text
build-identity.json                  源码/工具链/运行时/门禁身份
raw-all-model-ncnn-comparison.ndjson 45 行正式 profile-off 原始数据
final-all-model-ncnn-comparison.{json,ndjson}
comparison-summary.json / official-summary.txt / all-measured-summary.txt
run-cohort-evidence.py               三类 cohort 采集器（可重跑，拒绝覆盖已有证据）
cohort-run-04/
  profile-off/{end-to-end,prepared}.{ndjson,ctest.log}   边界对照 + RSS 采样
  profile-on/<model>/                                    profile/plan/attribution/编译日志
  profile-logs/<model>-compile.log
  commands.txt                                          profile 编译命令
  cohort-summary.json                                   最终资格审计汇总
```

正式口径固定：`NCNN_PERF_THREADS=6 NCNN_PERF_WARMUP=10 NCNN_PERF_ITERS=20
NCNN_PERF_MODE=end_to_end NCNN_PERF_MAX_RATIO=0`；cohort 诊断口径 2/5 并显式标
`gate_eligible=false`。两批都禁止把 profile-on 时间当正式结果。

## 8. 下一步（按证据优先级，不预设收益）

1. **补 wall-exclusive 证据**：worker wall-union 是逐操作并集、不可相加，仍无法给出 region 级
   wall-exclusive；需要跨线程 interval-union 或等价的 critical-path 度量，才可能让 YOLO 的
   4.7–5.4 s unknown 归属到具体操作。
2. **补 source-op 身份**：worker 站点 `source_layer`/`source_name` 全为 null，是 INT8 专项无法
   限定到 source-op 的直接原因；在 plan 生成侧为并行 worker 站点保留源算子/层 ID。
3. **降低插桩扰动**（当前 +62.6%–+458.0%）：worker 粒度已粗化到 scf.for/linalg，但 YOLO 侧
   7.8M 回调/次仍把时间放大 4–5 倍；需要采样或更低频的 worker 计时，才谈得上把占比转移到
   profile-off 正式时间。
4. 上述三点未闭合前，不重启任何 P20/P21/P23 候选实验；`mobile_rec_int8` 的
   `linalg.matmul#8 / packing_rejected_layout` 只作为待验证线索记录，不作为结论。
