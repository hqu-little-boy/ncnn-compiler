# P5 任务改动摘要

> 对应提交一笔 `refactor(ncnn)`。本文件按任务列出改动面与验收证据。
> 全局数据：23 个既有文件修改（+803 / −1284），15 个新增文件。

---

## T-J1 · C-runtime JSON writer 抽离 + UTF-8 修复

**新增**

| 文件 | 行数 | 作用 |
|---|---:|---|
| `lib/ProfileRuntime/profile_json_writer.h` | 76 | 编码原语 API + 硬约束说明（不用 llvm::json / qsort / strtoul / strlen） |
| `lib/ProfileRuntime/profile_json_writer.c` | 168 | UTF-8 校验 + 转义 + 标量编码 |

**改动**

| 文件 | 改动 |
|---|---|
| `lib/ProfileRuntime/profile_runtime.c` | 979 行重排：删本地 `write_json_string`，88 处 `fprintf` JSON 片段 + 26 处 `fputs("null")` 收敛到 `pj_*` |
| `lib/ProfileRuntime/CMakeLists.txt` | 加 writer TU（进 `NCNNProfileRuntime`，供 tidy 覆盖） |
| `tools/CMakeLists.txt` | install 三文件（两 `.c` + 一 `.h`）；新增 `NCNN_PROFILE_RUNTIME_WRITER_{SOURCE,RELATIVE_PATH}` |
| `tools/CompileSession.hpp` | 新增 `profile_writer_object` / `profile_writer_source` |
| `tools/ncnn-compile.cpp` | 抽 `resolve_profile_source()`；编两个 `.o`、链两个 |
| `test/Native/check_profile_runtime.py` | `--runtime` 扩成 `action="append"`；新增 11 个转义边界用例（env 用 **bytes** 注入，str 会被 fsencode 编成合法 UTF-8 测不到本 bug） |
| `test/CMakeLists.txt` | 传两个 `--runtime` |

**验收**

| 判据 | 结果 |
|---|---|
| 转义边界 round-trip | 11/11 绿（引号/反斜杠/控制字符/合法 UTF-8/空/仅引号/字面 U+FFFD/合法 U+0080/孤立续字节/截断 3 字节/overlong/代理区） |
| `json.loads` 可解析 | 24/24（修前 12/24） |
| 合法输入逐字节不变 | 12/24 与修前完全相同（3 schema × 4 用例） |
| `profile_allowed` diff | **空**（`strlen` 被消除而非加入白名单） |
| `.so` NEEDED | `ld-linux` / `libc` / `libm` / `libomp`（P26 正面性质保持） |
| 动态导出面 | 只有模型名（`pj_*` 被 `local: *;` 收编） |

---

## T-M4 · 拆 `InstrumentNCNNProfile` 插桩器

**新增**（见 README 的 8 TU 清单）

| 文件 | 行数 | 作用 |
|---|---:|---|
| `include/ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp` | 185 | 契约：runtime ABI 符号名、`SiteContext`、`SiteAnalysis`、插入点纪律 |
| `lib/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.cpp` | 240 | 共享积木（declare/constant/call/字节尺寸/候选判定） |
| `.../InstrumentSiteAnalysis.cpp` | 190 | id 派生 / alias 完备性 / materialized 事件发现 |
| `.../InstrumentAllocationSites.cpp` | 106 | alloc + dealloc（含 `resolveAllocation` 的 SSA 转发） |
| `.../InstrumentCopySites.cpp` | 45 | copy / `copy_contract` |
| `.../InstrumentOperationSites.cpp` | 143 | 时长计时 + movement + 函数边界 |
| `.../InstrumentFusionSites.cpp` | 96 | `InstrumentNCNNFusionSitesPass` |
| `.../InstrumentMaterializedSites.cpp` | 215 | `InstrumentNCNNMaterializedSitesPass` + 读写事件 |

**改动**：`InstrumentNCNNProfile.cpp` **883 → 171 行**（只剩编排）；
`CMakeLists.txt` 挂 8 个源文件。

**验收**：`compare-profile-instrumentation.sh` 用 HEAD 参考构建 vs 本次构建，
带 `--profile --emit=all` 编 squeezenet_v1_1 + resnet18，**归一化
`/tmp/ncnn-compile-<随机>` scratch 路径后全部产物逐字节一致**
（`.ncnn/.tosa/.linalg/.memref/.capi/.llvm.mlir` + `model.ll` + `model.s` +
`.h` + manifest + `.plan.json`）。`.so`/`.o` 不参与比对——它们含
`profile_runtime` 机器码，两侧源不同（T-J1），是另一个对照面。

---

## T-M5 · `emitWarning` 分级 + `--warnings-as-errors`

**新增**

| 文件 | 作用 |
|---|---|
| `include/ncnn-mlir/Support/Diagnostics.hpp` | 三级诊断契约 + 去重策略 + 开关下沉说明 |
| `lib/Support/Diagnostics.cpp` | `emitNcnnWarning` / `emitNcnnFallbackWarning` / `consumeDiagnosticFailure` |
| `test/Transforms/PackStaticMatmulNCNN/warning-grading.mlir` | 三处纪律：只报少数 reason / 同 reason 只报一次 / 默认不失败，升级后失败 |

**改动**

| 文件 | 改动 |
|---|---|
| `TileMatmulForall` / `PackStaticMatmulNCNN` / `VectorizeNCNN` / `StrategyNCNN` / `MatmulKernelNCNN` | `contract::annotateFallback(...)` → `emitNcnnFallbackWarning(...)`（9 处） |
| `EmitModelPlan.cpp` | `worker_site_source_ambiguous` 处补报警 |
| 上述 6 个 pass 的 `runOnOperation` 末尾 | `if (consumeDiagnosticFailure()) signalPassFailure();` |
| 上述 5 个 `CMakeLists.txt` | 链 `NCNNDiagnosticsSupport` |
| `lib/Support/CMakeLists.txt` | 新增 `NCNNDiagnosticsSupport` 目标 |
| `tools/ncnn-compile.cpp` | `--warnings-as-errors`（默认关）→ `setenv` 下沉 |

**验收**

| 判据 | 结果 |
|---|---|
| yolov5x_seg 可见预算拒绝 warning 且不失败 | ✓ 实测多个模型（含 resnet18 / squeezenet / pp_ocrv6）出现 `packing_rejected_budget`，编译 exit 0 |
| `--warnings-as-errors` 打开后失败 | ✓ lit `warning-grading.mlir` 的 ERR 通道（`not ncnn-mlir-opt` + `error: ncnn: ...`） |
| 默认路径产物不变 | ✓ T-M4 的逐字节对照同时覆盖（诊断只走 stderr） |
| 单模型 stderr 增量 | ≤ 6 行（按 reason 去重） |
| **不进 identity 串** | ✓ `build_codegen_identity()` 未纳入该选项 |

**诊断输出形态**（实测）：

```
ncnn-layer:38:0: warning: ncnn: packing_rejected_budget: linalg.matmul
```

按 location 发而不是 `op->emitWarning()`：后者会让 SourceMgr 多打一行
`see current operation:` 并把整个算子（含 20 个属性）整段吐出来。
算子名改由消息尾承载，一行就够。

---

## T-M3 · CLI 选项对照表

**新增** `docs/cli-options.md`（53 个选项，10 个分组：输入输出 / 精度累加 /
目标工具链 / 并行向量 / 卷积矩阵乘 / INT8 / 融合 / 向量数学 / 调优诊断 /
工具路径覆盖）。

每行给「取值 / 默认 / 所属 pass / 备注」，并标出哪些进 identity 串。
与散文参考 `ncnn-compile-command-line.md` 互链分工，末尾有「新增选项时看哪」。

**不改任何选项名**（roadmap §8：公开契约）。

---

## 门禁期间的顺带修复

| 项 | 说明 |
|---|---|
| `NCNN_FORMAT_SOURCES` 补 `lib/*.{h,hpp}` | 此前模块内头（`profile_json_writer.h`、`ImporterInternal.hpp` 等）根本不在 `format_check` 覆盖面内 |
| `strlen` 消除 | T-J1 引入后被 `profile_allowed` 审计拦下；改写成无需长度的 UTF-8 解析 |
| `idValue` 共享 | T-M4 拆分时差点破坏（每类各建一个常量会改 IR） |
