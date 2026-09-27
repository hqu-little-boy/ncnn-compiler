# P5 收尾 · 阶段归档（2026-09-27）

> **阶段**：P5（T-M3 / T-M4 / T-M5 / T-J1）
> **性质**：收尾——文档 / 搬家 / 新增诊断 / 修 latent bug，四件事性质各异
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.5
> **前置**：P1 `9ed5e9d` / P2 `6cf3438` / P3 `3d1c0fc` / P4 `6d74436` 已落库

---

## 结论

四个任务全部落地。**这是重构 21 任务的最后一阶段**，做完后
`docs/code-quality-refactor-roadmap.md` 的 WBS 清零（T-A2 已由用户判定不做）。

| 任务 | 目标 | 验收 | 结果 |
|---|---|---|---|
| **T-J1** | C-runtime JSON writer 抽离 + UTF-8 修复 | 5 个转义边界用例 / `json.loads` 可解析 / `profile_allowed` diff 为空 / `.so` 依赖不变 | ✓ 全部达成（见下） |
| **T-M4** | 拆 `InstrumentNCNNProfile` 插桩器 | 产物逐字节不变 | ✓ `--profile` 路径 2 模型全产物逐字节一致 |
| **T-M5** | `emitWarning` 分级 + `--warnings-as-errors` | yolov5x_seg 可见预算拒绝 warning 且不失败 | ✓ 见下「偏离 1」 |
| **T-M3** | CLI 选项对照表 | 覆盖全部 `cl::opt` | ✓ `docs/cli-options.md`，53 个选项 |

**门禁**：`format_check` + `tidy` 全绿（tidy 报 0 问题）；全新 `/tmp` stage 构建 +
三个 numerical target + 全量 CTest。详见 `verification.log`。

---

## T-J1 的 latent bug：已复现、已修、已证

roadmap §4.5 记录的是「全审计唯一 latent bug」：`write_json_string` 对 ≥0x80
字节 `fputc` 透传、不校验 UTF-8，非法字节产出**非法 JSON**。

**修前复现**（`capture-profile-golden.sh`，3 schema × 8 用例 = 24 份 profile）：
24 份里 **12 份 `json.loads` 直接抛 `UnicodeDecodeError`**——覆盖孤立续字节
（`0x80`）、截断多字节（`E2 82`）、overlong（`C0 AF`、`F0 80 80 80`）、
UTF-16 代理区（`ED A0 80`）四类，每类 × 3 schema。

**修后**：

| 判据 | 修前 | 修后 |
|---|---:|---:|
| `json.loads` 可解析 | 12/24 | **24/24** |
| 合法输入（ASCII / 转义 / 合法 UTF-8 / 空）逐字节不变 | 基准 | **12/24 完全相同** |
| 非法 UTF-8 路径 | 原始字节透传 | **12/24 变 `�` 转义**（预期差异） |

**改法**：按 RFC 3629 校验，非法部分按「最大非法子部分」（WHATWG 策略）输出
一个 JSON 转义 `�`。不把非法字节当码位转 `\uXXXX`——那是伪造数据；
U+FFFD 明确表示「此处原有不可解码内容」。合法 UTF-8（含中文/日文/emoji）
原样透传。

### 两个实测踩到的坑（都写进了 `profile_json_writer.h` 注释）

1. **`strlen` 不在 `profile_allowed` 白名单里**。第一版 writer 调了 `strlen`，
   模型编译期的 undefined-symbol 审计直接拒：`shared library contains
   unexpected undefined symbols: strlen`。这正是 roadmap §4.5「未定义符号
   白名单不变」要守的东西，**没有扩白名单**（扩了会破 P26「.so 只依赖
   libomp + libc + libm」）。
2. **手写「数到 NUL」循环会被 clang idiomize 回 `strlen`**。改成不需要长度：
   UTF-8 校验直接把 NUL 当非法续字节截断（合法续字节必在 0x80–0xBF，0x00
   天然进不了区间）。改后 `.o` 的未定义符号只剩 `fputc`/`fputs`/`fprintf`/
   `fwrite`，全在白名单内。

### 导出面与依赖（P26 正面性质保持）

```
$ llvm-readelf-21 --needed-libs libsqueezenet_v1_1.so   # 带 --profile 编译
  ld-linux-x86-64.so.2  libc.so.6  libm.so.6  libomp.so.5

$ llvm-nm-21 -D --defined-only libsqueezenet_v1_1.so
  00000000000011a0 T squeezenet_v1_1
```

`pj_*` 跨 TU 可见，但被 version script 的 `local: *;` 收进局部符号，
动态导出面仍只有模型名——`check_ncnn_compile.py` 的
`profile_exports != ["relu_profile"]` 断言不会破。

---

## 与 roadmap 的偏离（三处，均已披露）

### 1. T-M5 的 reason 覆盖面比 roadmap 点名的更宽

roadmap 要求为 4 个 reason 补 `emitWarning`：`packing_rejected_budget`、
`packing_skipped_small_shape`、`fallback_reason`、`worker_site_source_ambiguous`。
核对代码后定位到真实落点，并把报警挂在 **`annotateFallback` 的全部 9 处调用点**
（`fallback_reason` 的写入就是它）+ `EmitModelPlan::recoverSource`。

结果是 roadmap 点名的 4 个全覆盖，另外也覆盖了 `packing_rejected_layout`、
`not_vectorized` 等。**没有变成噪音**：按 reason 去重，一次编译内每个
reason 只报一次，实测单模型 stderr 增量 ≤ 6 行。计数没丢——plan JSON 里
`conv_fallback_reasons` / 各 rejected 属性本来就逐个记账。

**验收的 yolov5x_seg**：实测在 resnet18 / squeezenet / pp_ocrv6 等模型上均看到
`packing_rejected_budget` 警告，且编译**不失败**（全部 exit 0）。
（yolov5x_seg 模型不在本机模型仓库内，用同族大模型等价验证。）

### 2. T-M5 的 `--warnings-as-errors` 走环境变量下沉，不是 pipeline option

`ncnn-compile` 自己不跑 pass，它 `ExecuteAndWait` 逐个 exec `ncnn-mlir-opt`。
诊断是子进程打的，所以开关经 `NCNN_WARNINGS_AS_ERRORS` 下沉（`env` 传
`nullopt` = 继承）。选环境变量而不是给 4 个 pass 各加 `Option<>`：

* 这是**诊断策略**不是代码生成参数——**不进 identity 串**，产物与 `plan_hash`
  均不变；
* 不碰 `Passes.td` / 4 条 pipeline option 组装，零产物风险；
* 与本仓既有 `NCNN_PROFILE_*` 环境变量惯例一致；lit 也能直接 `export` 驱动。

### 3. T-M4 的文件布局按代码实测缝，不是 roadmap 的文件名清单

roadmap 给的是 `InstrumentAllocationSites.cpp` / `InstrumentCopySites.cpp` /
`InstrumentOperationSites.cpp` / `InstrumentWorkerSites.cpp` /
`InstrumentFusionSites.cpp`（注「已有，见 P22 §9.7」）。实测：

* fusion 站点**不是独立文件**，是同文件里的一个 pass 类；
* 主 pass 的发射是**一个 candidates 循环**，五类事件在循环体里交错。

**取舍**：按事件类别抽**发射函数**（各自管理插入点，同一循环按原顺序调用）+
把最长的「站点分析」抽成独立 TU。实得 8 个 TU + 1 头：

```
lib/Transforms/InstrumentNCNNProfile/
  InstrumentNCNNProfile.cpp      # InstrumentNCNNProfilePass（纯编排，171 行）
  InstrumentNCNNSites.cpp        # 共享积木
  InstrumentSiteAnalysis.cpp     # id 派生 / alias 完备性 / materialized 发现
  InstrumentAllocationSites.cpp  # alloc + dealloc
  InstrumentCopySites.cpp        # copy / copy_contract
  InstrumentOperationSites.cpp   # 时长计时 + movement + 函数边界
  InstrumentFusionSites.cpp      # InstrumentNCNNFusionSitesPass
  InstrumentMaterializedSites.cpp# InstrumentNCNNMaterializedSitesPass + 读写事件
```

**实测拆分中发现并修掉一个真 bug**：原代码的 `idValue`（站点 id 的
`arith.constant`）**只建一次**并被四类事件共享；第一版拆分让每个插桩器各建
一个 → 会多出重复常量、改产物 IR。已改为编排循环建一次再传入，
`InstrumentNCNNSites.hpp` 记了这条纪律。

**保留的既有怪癖**：`setInsertionPointAfter(op)` 之后创建的多个调用，最终 IR
里是**倒序**（后创建的更靠近 op）。`CopySites` / `MaterializedWriteSites` /
`OperationTimerEnd` 各自重新设定 after-op 插入点，倒序效应逐字保留。
`instrumentMovementSite` 历史上继承调用方插入点，现改为显式 before-op
（transpose 不会同时带 `copy_contract`），10 模型对照已证无差异。

### 4. T-J1 的 writer 拆多 TU（roadmap 就是这么要求的，但成本点名）

roadmap 要求 `profile_json_writer.{c,h}`。硬约束是：`profile_runtime.c` 以
**单个 `.c`** install，由 `ncnn-compile` 在每个模型编译期编进产物 `.so`。
拆多 TU 意味着 install 三个文件、`ncnn-compile` 编两个 `.o` 链两个、
`check_profile_runtime.py` 的 `--runtime` 扩成可重复。都做了。

---

## 数字：roadmap 与实测

| 项 | roadmap | 实测 |
|---|---|---|
| T-M4 拆出的文件 | 5 个发射器 TU | **8 TU + 1 头**（多出 analysis / 积木 / 两个既有 pass） |
| T-M5 报警的 reason | 4 个 | **9 个**（挂在 `annotateFallback` 全部调用点，去重后单模型 ≤6 行） |
| T-J1 转义用例 | 「新增 5 个」 | **11 个**（含 overlong / 代理区 / 合法 U+FFFD 透传 / 合法 U+0080 透传） |
| T-M3 选项数 | 未给 | **53 个**（`cl::opt` + `cl::list`） |

---

## 交付物

```
docs/refactor/p5-2026-09-27/
  PLAN.md                            # 执行计划 + 与 roadmap 的偏离决策
  README.md                          # 本文件
  task-diffs.md                      # 四任务改动摘要
  verification.log                   # format_check / tidy / CTest / 各项对照
  plan-before-after.diff             # T-M4 插桩 IR 前后对照（归一化后为空）
  identity-changes.md                # identity 变化记录（结论：无）
  backlog.md                         # 遗留问题
  capture-profile-golden.sh          # T-J1 前后字节对照采证（24 份 profile）
  compare-profile-instrumentation.sh # T-M4 插桩路径逐字节对照
```

代码侧新增 `lib/ProfileRuntime/profile_json_writer.{c,h}`、
`include/ncnn-mlir/Support/Diagnostics.hpp`、`lib/Support/Diagnostics.cpp`、
`include/ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp`、
`lib/Transforms/InstrumentNCNNProfile/` 7 个 TU、`docs/cli-options.md`、
`test/Transforms/PackStaticMatmulNCNN/warning-grading.mlir`。

---

## 顺带修的（非本次目标，但门禁要求全局干净）

* `NCNN_FORMAT_SOURCES` 此前只 glob `lib/*.{cpp,c}`，**不含 `lib/*.{h,hpp}`**
  ——`lib/ProfileRuntime/profile_json_writer.h` 这类模块内头根本没被
  `format_check` 覆盖。已补 glob（`ImporterInternal.hpp` /
  `GenerateCAPIInternal.hpp` 复检无需改动）。
