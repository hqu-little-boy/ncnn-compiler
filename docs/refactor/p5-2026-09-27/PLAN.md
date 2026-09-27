# P5：收尾 — 执行计划

> **阶段**：P5 = T-M3 + T-M4 + T-M5 + T-J1
> **日期**：2026-09-27
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.5 / §6 / §7
> **前置**：P1 `9ed5e9d`、P2 `6cf3438`、P3 `3d1c0fc`、P4 `6d74436` 已落库
> **分支**：`p8-implementation`

---

## 0. Context

代码质量重构（roadmap 21 任务 / 5 阶段）只剩最后阶段。P5 是**收尾**：
四件事性质各不相同——T-M3 纯文档、T-M4 纯搬家、T-M5 加诊断能力、
T-J1 修唯一 latent bug。因此本阶段**不是零行为变化**（P4 是），护栏要分开：

| 任务 | 性质 | 行为影响 | 护栏 |
|---|---|---|---|
| **T-M3** CLI 对照表 | 纯文档 | 零 | 文档自洽 + 选项齐全 |
| **T-M4** 拆 `InstrumentNCNNProfile` | 纯搬家 | **零（须证）** | 10 模型产物逐字节对照 |
| **T-M5** `emitWarning` 分级 | 新增诊断 | **stderr 多 warning**；产物不变 | 产物逐字节对照 + lit + CLI 用例 |
| **T-J1** JSON writer + UTF-8 | 重构 + 修 bug | ASCII 输入零变化；非法 UTF-8 从「非法 JSON」变合法 | round-trip 用例 + `profile_allowed` diff 为空 + `.so` 依赖审计 |

---

## 1. 执行决策（与 roadmap 的偏离，均已核实代码后定案）

### 决策 1：T-J1 拆多 TU（不走 header-only）

roadmap 要求 `profile_json_writer.{c,h}`。硬约束是：

* `tools/CMakeLists.txt:36` 以 **单个 `.c` 文件** install，由 `ncnn-compile`
  在每个模型编译期 `clang -c profile_runtime.c -o profile_runtime.o` 编进产物 `.so`；
* `test/Native/check_profile_runtime.py --runtime <单一路径>` 也是单文件假设。

**取舍**：走多 TU（`profile_json_writer.c` + `.h`，两份 `.c` 都编译、都链接），
理由是 roadmap 明确点名 `.c`，且改造成本可控：

1. `tools/CMakeLists.txt` install 三个文件（两 `.c` + 一 `.h`）；
2. 宏 `NCNN_PROFILE_RUNTIME_SOURCE`（单路径）→ 新增
   `NCNN_PROFILE_RUNTIME_WRITER_SOURCE`；`ncnn-compile` 编两个 `.o`、链接两个；
3. `check_profile_runtime.py` 的 `--runtime` 扩成可重复（`action="append"`）。

**导出面安全**：链接用 `-Wl,--version-script`，`{ global: <model>; local: *; }`
（`tools/ncnn-compile.cpp:2961`）。`pj_*` 非 static 跨 TU 可见，但会被
`local: *` 收进局部符号，`nm -D --defined-only` 仍只看到模型名——
`check_ncnn_compile.py:393` 的 `profile_exports != ["relu_profile"]` 断言不会破。

**不做的事**：不给 writer 加浮点输出（roadmap §4.5 明令保留「只出整数」设计）。

### 决策 2：T-J1 UTF-8 修复口径 = `�`（替换字符）

现状 `write_json_string`（`profile_runtime.c:1272`）对 ≥0x80 字节 `fputc` 直接透传，
不校验 UTF-8。非法字节 → 非法 JSON（`json.loads` 直接失败）。

改法按 roadmap「非法字节转 `�` 或 `\uXXXX`」取**前者**：

* 完整合法 UTF-8 序列原样透传（层名里的中文/日文仍是可读字符，不炸成 `\uXXXX`）；
* 非法字节 / 被截断的序列输出 `�`（即 `�`）；
* 控制字符（<0x20）沿用现有 `\\u%04x`；`"` `\\` 与既有转义不变。

不取「每个非法字节转 `\uXXXX`」的原因：那会把一个非法字节解读成一个码位，
语义上是**伪造**数据；`�` 明确表示「此处原有不可解码内容」。
**合法 ASCII 路径逐字节不变**，故不触发 identity 变化。

### 决策 3：T-M5 的 `--warnings-as-errors` 走**环境变量下沉**

`ncnn-compile` 自己不跑 pass——它 `llvm::sys::ExecuteAndWait` 逐个调
`ncnn-mlir-opt` 子进程（`tools/ncnn-compile.cpp:2670` 起）。诊断是子进程打的，
所以 CLI 开关必须下沉到 `ncnn-mlir-opt` 进程内。

两个候选：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. pass pipeline option | 4 个 pass 各加 `Option<"warningsAsErrors"...>`，`ncnn-compile` 往 4 条 pipeline option 字符串拼参数 | 改 `Passes.td` + 4 条 pipeline 组装；pipeline option 解析有回归风险 |
| B. 环境变量 `NCNN_WARNINGS_AS_ERRORS` | `Support/Diagnostics` 进程内读一次；`ncnn-compile` 的 `--warnings-as-errors` 用 `setenv` 下沉（`ExecuteAndWait` env 传 `std::nullopt` = 继承） | 零 pipeline 改动 |

**取 B**。理由：`warnings-as-errors` 是**诊断策略**不是代码生成参数——
它不改变产物，因此**不进 identity 串**（`tools/ncnn-compile.cpp:1900` 起的手工拼串
不含它，保持如此）。B 不碰 `Passes.td`、不碰 pipeline 组装，identity/IR 零风险；
且与本仓既有 `NCNN_PROFILE_*` 环境变量惯例一致。lit 用例也能直接 `export` 驱动。

### 决策 4：T-M5 只对 roadmap 点名的 4 个 reason 报警，且**按 reason 去重**

roadmap 列的四个：`packing_rejected_budget`、`packing_skipped_small_shape`、
`fallback_reason`、`worker_site_source_ambiguous`。核对代码后定位到真实落点：

| reason | 落点 | 现状 |
|---|---|---|
| `packing_rejected_budget` | `PackStaticMatmulNCNN.cpp` `annotateRejected` / `annotateInt8Rejected`（8 处） | 只写属性，静默 |
| `packing_skipped_small_shape` | 同上（3 处） | 同上 |
| `fallback_reason` | `contract::kFallback` 写入点：`MatmulKernelNCNN.cpp:2336`（conv 策略降级） | 同上 |
| `worker_site_source_ambiguous` | `EmitModelPlan.cpp:658` `add_unknown("worker_site_source_ambiguous")` | 同上 |

**不去重会淹掉 stderr**，故 `emitNcnnWarning` 内部按 reason 去重：
同一 reason 一次编译内只报一次，消息里注明后续重复已抑制。
计数没有丢——plan JSON 里 `conv_fallback_reasons` / 各 rejected 属性本来就逐个记账。

**明确不报警的**（roadmap 也没列，实测是常态路径，报了就是噪音）：

* `packing_rejected_layout` / `packing_rejected_dynamic` / `packing_rejected_nonconstant_rhs`
* `ncnn.workspace_fallback_reason`（`ReuseWorkspaceSlots::markFallback`，每个
  不可复用的 allocation 都会触发，与 roadmap 的 `fallback_reason` 是**两个键**）

### 决策 5：T-M4 的拆分缝按代码实测取，不照抄 roadmap 的文件名清单

roadmap 给的是 `InstrumentAllocationSites.cpp` / `InstrumentCopySites.cpp` /
`InstrumentOperationSites.cpp` / `InstrumentWorkerSites.cpp` /
`InstrumentFusionSites.cpp`（注「已有，见 P22 §9.7」）。实测：

* fusion 站点**不是独立文件**，是同文件里的 `InstrumentNCNNFusionSitesPass`；
* 主 pass 的发射是**一个 candidates 循环**，五类事件在循环体里交错
  （有的 `setInsertionPoint` 在 op 前、有的在 op 后），**不是五段平铺代码**。

**取舍**：按事件类别抽**发射函数**（各自管理自己的插入点，从同一循环按原顺序调用），
外加把最长的「准备阶段」（id 派生 / alias 完备性 / materialized 事件发现）抽成
独立 TU。发射顺序与插入点保持不变 ⇒ **产物 IR 逐字节不变**（用 P3 的
`emit-plans.sh` 10 模型对照证明）。

```
lib/Transforms/InstrumentNCNNProfile/
  InstrumentNCNNProfile.hpp      # 已有
  InstrumentNCNNSites.hpp        # 新：SiteContext + 各类 emit* 声明
  InstrumentNCNNProfile.cpp      # 三个 pass 定义 + 候选扫描 + 编排循环（瘦）
  InstrumentSiteAnalysis.cpp     # id 派生 / alias 完备性 / materialized 发现
  InstrumentAllocationSites.cpp  # alloc + dealloc 事件
  InstrumentCopySites.cpp        # copy / kCopyContract 事件
  InstrumentOperationSites.cpp   # 时长计时 + root/flush + transpose movement
  InstrumentWorkerSites.cpp      # worker 计时 + worker 候选判定
```

**不强求行数下降**（P4 的 T-C5 已证「去重换的是改一处生效三处，不是行数」）。
验收看的是**装配关系**：主文件只剩编排，类别逻辑各自可读。

### 决策 6：T-M3 放 `compiler/docs/cli-options.md`

仓库里已有 `compiler/docs/ncnn-compile-command-line.md`（1261 行散文参考）。
T-M3 要的是 roadmap §4.5 那种**紧凑对照表**（选项 / 取值 / 默认 / 所属 pass / 备注），
是另一份东西：查「这个选项归哪个 pass 管」用的索引。两份互链，不重写散文参考。

**不改任何选项名**（roadmap §8「不统一改 CLI 选项名」——公开契约）。

---

## 2. 任务分解（执行顺序）

顺序原则：先做行为面最大的 T-J1（latent bug，测试面最厚），再做纯搬家 T-M4
（需要逐字节护栏），再做 T-M5（会改 stderr，必须在 T-M4 逐字节基线之后），
最后纯文档 T-M3。

### T-J1（1）

1. `lib/ProfileRuntime/profile_json_writer.h`：`pj_object_start/end`、`pj_key`、
   `pj_string`、`pj_u64`、`pj_i64`、`pj_bool`、`pj_null`，附「为什么不用
   `strtoul`/`qsort`/`llvm::json`」注释。
2. `lib/ProfileRuntime/profile_json_writer.c`：实现 + `write_json_string` 收敛到
   `pj_string`（含 UTF-8 校验）。
3. `profile_runtime.c` 删本地 `write_json_string`，改调 `pj_*`；
   88 处 `fprintf` JSON 片段按段落换成 `pj_*` 调用（能等价替换的就替换，
   **数值仍只出整数**）。
4. `tools/CMakeLists.txt` + `tools/ncnn-compile.cpp`：多 TU 编译/链接/安装。
5. `test/Native/check_profile_runtime.py`：`--runtime` 可重复；新增 5 个转义边界
   round-trip 用例（`"`、`\`、控制字符、非 ASCII、截断 UTF-8）。

### T-M4（2）

按决策 5 拆 TU。每拆一步跑 lit（InstrumentNCNNProfile 相关用例）+
`emit-plans.sh` 10 模型逐字节对照（归一化 `/tmp/ncnn-compile-<随机>` scratch 路径）。

### T-M5（3）

1. `include/ncnn-mlir/Support/Diagnostics.hpp` + `lib/Support/Diagnostics.cpp`：
   `emitNcnnWarning(Operation*, StringRef reason)`、`setWarningsAsErrors(bool)`、
   `warningsAsErrors()`；按 reason 去重；escalate 时改发 error 并返回 `failure()`。
2. 四个落点接线（决策 4）。
3. `tools/ncnn-compile.cpp`：`--warnings-as-errors`（默认关）→ `setenv` 下沉。
4. lit：warning 文案出现；`NCNN_WARNINGS_AS_ERRORS=1` 时 pass 失败。
5. CLI 用例：`check_ncnn_compile.py` 加「默认不失败 + 开关后失败」两例。

### T-M3（4）

`compiler/docs/cli-options.md` 对照表 + 与散文参考互链。

---

## 3. 门禁（用户硬约束）

每个任务跑 format_check + tidy；**阶段收尾统一门禁**：

```bash
cmake --build <build-dir> --target format_check --parallel
cmake --build <build-dir> --target tidy --parallel
mkdir /tmp/ncnn-compiler-stage-p5-<n>
cmake -S compiler -B /tmp/ncnn-compiler-stage-p5-<n> \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
      -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
      -DCOMPILER_INSTALL_RPATH=ON \
      -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-p5-<n> --parallel
cmake --build /tmp/ncnn-compiler-stage-p5-<n> \
      --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
      --parallel
ctest --test-dir /tmp/ncnn-compiler-stage-p5-<n> --output-on-failure
```

* `format_check` / `tidy` 是**全局**的——非本次产生的问题一并修（用户约束）。
* 每次全量测试用**新 `/tmp` 目录**，不用旧产物推断。
* 验收失败**不提交**；修复后**另起全新 `/tmp` 目录**重建重测（P4 首轮教训）。
* `cmake --build` 一律 `--parallel`。
* 提交前：`git diff --check` / `git status --short` / `git diff --cached` /
  `git log --oneline -10`。
* **只 commit 不 push**；提交信息中文 `refactor(ncnn): ...` 格式。

## 4. 验收标准（逐任务）

| 任务 | 验收 |
|---|---|
| **T-J1** | 5 个转义边界 round-trip 全绿；profile 仍可 `json.loads`；`profile_allowed` diff 为空；`.so` NEEDED 仍是 libomp+libc+libm（P26 审计脚本复跑）；合法 ASCII 输入 profile 逐字节不变 |
| **T-M4** | 10 模型 × 7 产物逐字节不变（归一化 scratch 路径后 diff 为空）；InstrumentNCNNProfile lit 全绿；主文件只剩编排 |
| **T-M5** | yolov5x_seg 编译可见 `packing_rejected_budget` warning 且**不失败**；`--warnings-as-errors` 打开后同样输入失败；默认路径产物逐字节不变 |
| **T-M3** | 表覆盖 `ncnn-compile` 全部 `cl::opt`；取值/默认与代码一致；与散文参考互链 |
| **阶段** | format_check + tidy 全绿；全新 `/tmp` 构建 + 三个 numerical target + 全量 CTest 通过 |

## 5. 明确不做

* 不改任何 CLI 选项名（公开契约）。
* 不给 C runtime 加浮点 JSON 输出。
* 不改性能默认值（`NCNN_MATMUL_PACKING=auto`、`NCNN_PACKED_CONV_DEPTHWISE=OFF`、
  `NCNN_NATIVE_INT8_PROFILE=OFF`）。
* 不重跑性能基线——P5 无代码生成语义变更；roadmap §6「P5 复跑性能基线」按
  P4 先例省略（P4 亦未跑，产物逐字节不变即等价于零漂移）。若 T-M5 的诊断下沉
  被证明影响编译路径，再补跑。
* 不做「顺便优化」。

## 6. 交付物

```
docs/refactor/p5-2026-09-27/
  PLAN.md                 # 本文件
  README.md               # 做了什么、验收结果、偏离
  task-diffs.md           # 四任务改动摘要
  verification.log        # format_check / tidy / CTest / 逐字节对照
  plan-before-after.diff  # T-M4 产物逐字节对照（归一化后应为空）
  identity-changes.md     # identity 变化记录（预期：无）
  backlog.md              # 遗留问题
```

代码侧新增 `lib/ProfileRuntime/profile_json_writer.{c,h}`、
`include/ncnn-mlir/Support/Diagnostics.hpp`、`lib/Support/Diagnostics.cpp`、
`lib/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp` + 4 个 TU、
`compiler/docs/cli-options.md`。
