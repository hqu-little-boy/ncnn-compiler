# ncnn-mlir 代码质量审计与完整重构计划

> **版本**：v2（2026-09-26）
> **背景**：P18–P27 性能优化阶段收官（heavy p50=1.35×、整体 p50=1.95×），性能不再作为主攻方向。
> **文档性质**：审计结论（第一部分）+ **可执行重构计划**（第二部分）。所有数字来自当前工作树实测。
> **v2 说明**：初版只看"函数长度/库复用"两层。按"重新认真分析"要求深入后，核心问题在
> **自建基础设施各自为政**。总体健康度 **3.0 / 5**（初版高估为 3.5）。

---

# 第一部分　审计结论

## 0. 核心诊断

> **项目自建了一套基础设施（schema、序列化、哈希、常量查找、溢出检查），
> 但每一套都不完整、且互相不一致。巨型函数只是这个模式的表层症状。**

同一个问题在不同文件里有**不同答案**：

| 问题 | 实现 A | 实现 B | 后果 |
|---|---|---|---|
| 找常量 | `getConstantElements`（只看直接 def） | `findConstant`（穿透 cast/collapse/expand） | batchnorm 静默跳过的常量，packing 却能处理 |
| 溢出检查 | 手写 `checkedAdd`（返回 `FailureOr`） | `llvm::MulOverflow`（返回 `bool`） | **同一文件内两套语义** |
| 稳定哈希 | `fnv1a`（InstrumentNCNNProfile） | `fnv1a`（EmitModelPlan） | 写了两份，该用 `llvm::xxHash64` |
| 属性键 | `KernelContract::k*` 常量（56 键） | 裸字符串字面量（**187 处 / 67 键**） | "单一事实源"被架空 |
| JSON | `llvm::json`（C++/Python） | 手写 `fprintf`（C runtime，88 处） | 后者有硬约束，但转义不完整 |
| 错误处理 | `std::expected` / `LogicalResult` / `FailureOr` / `std::optional` | — | **5 种惯用法，0 个 `assert(`** |

**因此重构第一目标是收敛语义，其次才是拆函数。**

## 1. 量化现状

### 1.1 巨型函数

**C++ 三个量级异常**（前两个作者用 `NOLINTNEXTLINE(google-readability-function-size)` 豁免 linter）：

| 行数 | 文件 | 函数 | 位置 | 作者豁免 |
|---:|---|---|---|:---:|
| **1648** | `EmitModelPlan.cpp` | `runOnOperation` | 367–2017 | `:366` |
| **1312** | `ncnn-compile.cpp` | `main` | 2086–3397 | `:2084` |
| **1223** | `GenerateCAPI.cpp` | `runOnOperation` | 713–1936 | — |

次一级：

| 行数 | 文件 | 函数 | 位置 |
|---:|---|---|---|
| 499 | `NCNNToTosa.cpp` | `iterators` | 625–1124 |
| 486 | `StrategyNCNN.cpp` | `rewriteWinograd` | 934–1420 |
| 430 | `NCNNToTosa.cpp` | `matchAndRewrite` | 3095–3525 |
| 424 | `NCNNToTosa.cpp` | `matchAndRewrite` | 1125–1549 |
| 421 | `MatmulKernelNCNN.cpp` | `kernelize` | 1389–1810 |
| 408 | `InstrumentNCNNProfile.cpp` | `runOnOperation` | 460–868 |
| 379 | `StrategyNCNN.cpp` | `rewriteConvolution` | 1421–1800 |
| 369 | `GenerateCAPI.cpp` | `prepareABI` | 268–637 |
| 311 | `MatmulKernelNCNN.cpp` | `kernelizeInt8RowDot` | 2110–2421 |

**Python 同病且更集中** — `tools/perf_attribution_report.py`（1588 行 / 13 函数）：

| 行数 | 函数 | 位置 |
|---:|---|---|
| **856** | `build_report` | 246–1102 |
| **447** | `aggregate_v2_reports` | 1103–1550 |

两个函数占 1303/1588 行，而 P27/P28 的归因 bug（wall-partition 重写、worker join、
窗口归一）正是在此被发现和修复。

> **勘误（v1）**：初版把 `Graph/graph.cpp::read_bytes` 记为 1260 行巨型函数，
> 系**度量正则误报**（漏掉该文件函数定义形式，末项吞掉文件尾部）。实际 `graph.cpp`
> 结构良好（小函数 + 干净的 `BinCursor`，`std::expected`/`std::span`/`std::format`），
> **不需要重构**。

### 1.2 自建基础设施规模

| 子系统 | 位置 | 规模 | 状态 |
|---|---|---:|---|
| `ncnn.*` 属性 schema | `Support/KernelContract.hpp` | 56 常量 + 12 访问器 | 被裸字面量架空 |
| 裸属性字面量 | 全库 | **187 处 / 67 键** | 绕过常量 |
| 模块级 ledger | module/function `ArrayAttr` | 6 组 | 属性当数据总线 |
| JSON writer | `ProfileRuntime/profile_runtime.c` | 88 处 fprintf + 40 行转义 | 有硬约束 |
| 稳定哈希 | 2 处手写 FNV-1a | — | 该合并 |
| 常量查找 | 2 个不一致实现 | — | 该统一 |
| 溢出检查 | 手写 + `llvm::*Overflow` | 4+ 种 | 该统一 |

### 1.3 错误处理：5 种惯用法，0 个断言

| 惯用法 | 次数 | 分布 |
|---|---:|---|
| `std::optional` | 181 | 全库 |
| `std::expected` | 141 | `Importer/` + `Graph/`（C++23） |
| `LogicalResult` | 113 | MLIR pass |
| `FailureOr` | 90 | MLIR pass |
| `signalPassFailure` | 57 | pass 入口 |
| **`assert(`** | **0** | **全库零断言** |
| **`emitWarning`** | **0** | 要么静默要么报错 |
| `llvm::errs()` 裸 stderr | 29 | 与 217 处 MLIR 诊断并存 |
| `exit(` | 3 | CLI |

**`assert(` 为 0 最值得注意**：内部不变量（"累加器数量 == rowCount"、"panel 索引落在
packed 长度内"）无任何运行时检查，出错表现为**错误的机器码**而非断言失败。
MLIR/LLVM 自身是重度用 `assert` 的。

### 1.4 测试覆盖（lit 文件数）

```
23 GenerateCAPI   16 EmitModelPlan   11 MatmulKernelNCNN   8 VectorizeNCNN
 7 FuseLinalgEpilogue  6 StrategyNCNN  6 NormalizeNCNN  5 LowerVectorMathNCNN
 3 ReuseWorkspaceSlots  3 PackStaticMatmulNCNN
 2 RewriteLinalgCopies  2 FuseQuantChainNCNN  2 ForallizeDisjointTileLoops
 1 TileMatmulForall     1 FoldLinalgConstantTranspose   （+ 各 Verify*）
```

覆盖面全（每 pass ≥1），但核心 tiling 仅 1 个 lit、`PackStaticMatmulNCNN` 仅 3 个
（P20 packing 争议落点）。

## 2. 应当保持的优点（重构时别破坏）

1. **`KernelContract.hpp` 的单一来源意图正确** — 问题在执行（被裸字面量架空），
   不是设计方向。T-S2 是"把设计做实"，不是推翻。
2. **常量基本都命名**（`kMaxStrategyElements`、`kAccumulatorFloatBudget` 等）。
3. **`llvm::json::Object` 全量使用**，C++/Python 侧零手写 JSON 拼接。
4. **零 `TODO/FIXME/HACK/XXX`** — 技术债以"规模/不一致"形式存在。
5. **pass 类统一放匿名 namespace**，一 pass 一目录，结构可预测。
6. **lit 覆盖面完整**：21 个 pass 全部有测试。
7. **C runtime 数值只出整数**（`%llu`），从根上避开 NaN/Infinity 陷阱。
8. **`check_profile_runtime.py` 做 round-trip `json.loads`** — 用不了库时的正确替代。

---

# 第二部分　完整重构计划

## 3. 工作分解结构（WBS）

**总计 21 个任务、5 个阶段、约 22–30 人天。**

| ID | 任务 | 阶段 | 工时 | 依赖 | 风险 |
|---|---|---|---:|---|---|
| **T-S1** | 统一常量查找 | P1 | 1d | — | 低 |
| **T-S2** | 属性 schema 收敛（67 裸键 + 类型安全） | P1 | 2–3d | — | 中 |
| **T-S4** | 溢出检查统一 | P1 | 0.5d | — | 低 |
| **T-S3** | 模块 ledger 保守收敛 | P1 | 1.5d | T-S2 | 中 |
| **T-C1** | 拆 `EmitModelPlan::runOnOperation` | P2 | 2.5d | — | 低 |
| **T-C2** | 拆 Python `build_report` | P2 | 2d | — | 低 |
| **T-C3** | 拆 `GenerateCAPI` | P2 | 1.5d | — | 低 |
| **T-C4** | 拆 `ncnn-compile::main` | P2 | 1.5d | — | 低 |
| **T-A1** | `StringSwitch` 替换 27 处 if 链 | P3 | 0.5d | — | 低 |
| **T-A2** | APFloat→float/memcpy | P3 | 1d | — | 低 |
| **T-A3** | `matchPattern` 末端匹配 | P3 | 0.5d | T-S1 | 低 |
| **T-S5** | 稳定哈希合并（xxHash64） | P3 | 0.5d | — | 中（identity） |
| **T-C5** | tile-loop 公共层 | P4 | 2d | T-C1 | 中 |
| **T-C6** | 拆 `NCNNToTosa.cpp` | P4 | 2.5d | — | 中 |
| **T-C7** | Winograd 独立成模块 | P4 | 1d | — | 低 |
| **T-M1** | 补 4 个薄弱 pass 的 lit | P4 | 1.5d | T-C5 | 低 |
| **T-M2** | pass 设计注释 | P4 | 1d | T-C6/C7 | 低 |
| **T-M3** | CLI 选项对照表 | P5 | 0.5d | — | 低 |
| **T-M4** | 拆 `InstrumentNCNNProfile` 插桩器 | P5 | 1.5d | T-C1 | 中 |
| **T-M5** | `emitWarning` 分级 | P5 | 1d | — | 低 |
| **T-J1** | C-runtime JSON writer 抽独立 + UTF-8 | P5 | 1.5d | — | 中（latent bug） |

### 依赖图

```
P1 语义收敛
  T-S1 ──────────┬──→ T-A3
  T-S2 ──→ T-S3 │
  T-S4 ─────────┘

P2 巨型函数拆分（与 P1 可并行）
  T-C1 ──┬──→ T-C5 ──→ T-M1
         └──→ T-M4
  T-C2
  T-C3
  T-C4

P3 LLVM 库替换（与 P1/P2 可并行）
  T-A1
  T-A2
  T-S5 ⚠️ identity 变化

P4 结构拆分 + 测试
  T-C6 ──→ T-M2
  T-C7 ──→ T-M2

P5 收尾
  T-M3 / T-M5 / T-J1
```

**关键约束**：P2 与 P3 **可并行**（都改不同文件）；P1 的 T-S2 会改大量文件，
**必须在 P2/P3 之前或与之错开**，避免合并冲突。

---

## 4. 逐任务详述

### 4.1 P1：语义收敛（优先级最高）

#### T-S1 统一常量查找（1d，低风险）

**问题**：两套实现，答案不同。
- `lib/Transforms/FoldNCNNBatchNorm/FoldNCNNBatchNorm.cpp:24` `getConstantElements` — 只看直接 `arith::ConstantOp` def
- `lib/Transforms/PackStaticMatmulNCNN/PackStaticMatmulNCNN.cpp:115` `findConstant` — 穿透 `tensor.cast`/`collapse_shape`/`expand_shape`

**后果**：同一常量被 `tensor.cast` 包着时，batchnorm 折叠静默失败、packing 却能拿到。

**改动点**：
1. `include/ncnn-mlir/Support/ConstantFold.hpp` 新增：
   ```cpp
   std::optional<mlir::ElementsAttr> findConstantElements(mlir::Value value);
   ```
   完整穿透 `tensor.cast` / `tensor.collapse_shape` / `tensor.expand_shape` /
   `tensor.extract_slice` / `tensor.from_elements`
2. `lib/Support/ConstantFold.cpp` 实现
3. 两个调用点收敛到该函数；删除各自的局部实现
4. `PackStaticMatmulNCNN` 如需 `arith::ConstantOp` 本身，在 helper 上加第二个返回值或
   `findConstantOp(Value)`，避免两套并存

**验收**：
- 新 lit `test/Transforms/FoldNCNNBatchNorm/constant-through-cast.mlir`：
  被 `tensor.cast` 包着的常量仍被折叠
- 新 lit `test/Transforms/PackStaticMatmulNCNN/constant-through-cast.mlir`：同上
- `grep -rn "getConstantElements\|findConstant" lib` 只剩一处定义
- 完整 CTest + `format_check` + `tidy` 全绿

**风险**：batchnorm 开始折叠以前跳过的常量 → **可能改变产物数值路径**。
必须跑数值 golden，且对比 plan（`ncnn.*` 属性变化属预期，记为 identity 变化）。

---

#### T-S2 属性 schema 收敛（2–3d，中风险）

**问题**：`KernelContract.hpp` 56 常量 vs 全库 **187 处裸 `"ncnn.*"` 字面量 / 67 键**；
`setString(Operation*, StringRef, StringRef)` 无类型。

**改动点**：
1. **盘点 67 个裸键**（命令见 §6），按组补进 `KernelContract.hpp`：
   - `kShape*`：`ncnn.shape_constraints`、`ncnn.shape_program`、`ncnn.shape_carrier`、
     `ncnn.shape_source_input`、`ncnn.shape_program_version`、`ncnn.data_dependent_dim_mask`
   - `kWorkspace*`：`ncnn.workspace_slot_{bytes,alignment,owner,lifetime_begin,lifetime_end,thread_visibility}`
   - `kSource*`：`ncnn.source_layer`、`ncnn.name`
   - `kRank*`：`ncnn.rank_variant`、`ncnn.dynamic_rank`
   - `kEntry*`：`ncnn.entry_point`、`ncnn.precision`、`ncnn.packed_conv_depthwise`
   - 其余按现有 `kFusion*` / `kCopy*` / `kAttention*` 分组
2. **收紧 setter API**：
   - 删除公开的 `setString(Operation*, StringRef, StringRef)` / `setInteger(..., StringRef, ...)`
   - 只留 `setString(Operation*, StringRef kKey, StringRef)` 这类**键为具名常量**的重载，
     或改为 `annotate*` 聚合接口（已有 12 个）
   - 把"键 ↔ 类型"绑定到常量声明上（如 `constexpr StringRef kTileM = "ncnn.tile_m";` 旁
     注释 + 单测校验）
3. **白名单校验测试**：`test/Native/check_attribute_whitelist.py`
   - 遍历一个真实模型的 `.plan.json` / dump IR，断言所有 `ncnn.*` 键都在白名单
   - 负向用例：人为写入非法键应被检测

**验收**：
- `grep -rE '"ncnn\.' lib tools` **归零**（只允许白名单测试里出现）
- 全库 `setString`/`setInteger` 调用点全部用具名常量
- lit + 单测通过；plan 输出字段名不变（**不触发 identity 变化**）

**风险**：改漏一处就编译失败（这是**好事**，编译期查出错用）。
工作量主要在盘点 67 键的语义归属。

---

#### T-S4 溢出检查统一（0.5d，低风险）

**问题**：`NCNNOps.cpp:24` 手写 `checkedAdd`（返回 `FailureOr<int64_t>`）与
`NCNNOps.cpp:688` 的 `llvm::MulOverflow`（返回 `bool`）**同文件并存**；
`StrategyNCNN.cpp` 另有 `checkedProduct`。

**改动点**：
1. 新建 `include/ncnn-mlir/Support/CheckedMath.hpp`：
   ```cpp
   mlir::FailureOr<int64_t> checkedAdd(int64_t a, int64_t b);
   mlir::FailureOr<int64_t> checkedMul(int64_t a, int64_t b);
   mlir::FailureOr<int64_t> checkedSub(int64_t a, int64_t b);
   mlir::FailureOr<int64_t> checkedProduct(llvm::ArrayRef<int64_t> dims);
   ```
   内部调 `llvm::AddOverflow` / `MulOverflow` / `SubOverflow`
   （`llvm/Support/CheckedArithmetic.h` 或 `llvm/Support/Arithmetic.h`）
2. 删除 `NCNNOps.cpp` / `StrategyNCNN.cpp` 的三处手写实现
3. 直接用 `llvm::*Overflow` 的 4 处（`NCNNToFunc.cpp`、`NCNNOps.cpp`、
   `NCNNToTosa.cpp`、`ShapeProgram.hpp`）保留，或统一改为薄包装——
   **推荐保留**，只要语义一致

**验收**：全库只有一个 `checkedAdd`/`checkedProduct` 定义；CTest 全绿。

---

#### T-S3 模块 ledger 保守收敛（1.5d，中风险）

**问题**：6 组结构化数据以 `ArrayAttr` 挂在 module/function 上当 pass 间 side channel，
无 schema、无生命周期管理（`RewriteLinalgCopies.cpp:597` 靠手动 `removeAttr`）。

涉及：`ncnn.copy_records`、`ncnn.fusion_records`、`ncnn.attention_segments`、
`ncnn.shape_constraints`、`ncnn.input_dim_relations`、`ncnn.rank_variant*`

**改动点**：
1. 新建 `include/ncnn-mlir/Support/ModelLedger.hpp`：
   ```cpp
   struct ModelLedger {
     static ModelLedger read(mlir::Operation* moduleOrFunc);
     void write(mlir::Operation* moduleOrFunc) const;
     void clear(mlir::Operation* moduleOrFunc) const;
     mlir::ArrayAttr copyRecords, fusionRecords, attentionSegments;
     mlir::ArrayAttr shapeConstraints, inputDimRelations;
     // ...
   };
   ```
2. 各 pass 的 `getAttrOfType<ArrayAttr>(...)` / `setAttr(...)` 收敛到该结构
3. **不做**"彻底档"（引入 `ncnn.ledger` 字典 + 改 pass 顺序契约）——见 §8

**验收**：散落的 `getAttrOfType<ArrayAttr>("ncnn.*")` 归零；所有 ledger 读写走 `ModelLedger`。

---

### 4.2 P2：巨型函数拆分（纯机械，不动语义）

> **统一纪律**：每拆一步跑完整 CTest（449 registered）+ `format_check` + `tidy`；
> **不做任何行为变更**，使结果可直接与 P27/P28 基线比对。

#### T-C1 拆 `EmitModelPlan::runOnOperation`（2.5d）

**现状**：`lib/Transforms/EmitModelPlan/EmitModelPlan.cpp:367–2017`（1648 行），
顺序构建 13 个顶层 JSON 段 + plan-hash 派生。

**改动点**：
1. 抽 13 个纯函数（每个返回 `JsonObject`/`JsonArray`）：
   ```
   collectFunctions()   collectOperations()   collectBuffers()
   collectRegions()     collectContracts()    collectConvDepthwiseOps()
   collectFusions()     collectAttentionSegments()
   collectLowPrecision() collectUnknownFields()
   collectProvenance()  collectStaticLiveness() collectSummary()
   ```
2. 抽 `computePlanHash(const PlanSections&)`，**显式列出参与哈希的字段白名单**
3. `runOnOperation` 变成：读 module → 逐段 collect → 组装 → hash → 写文件

**验收**：
- `runOnOperation` ≤ 80 行
- 新增单测：plan-hash 字段白名单（改动未列入白名单的字段不应改变 hash）
- 生成的 `.plan.json` 与重构前**逐字段一致**（用 diff 校验）
- 完整 CTest + `format_check` + `tidy`

**风险**：低。纯提取函数。唯一风险是 hash 输入顺序变化 → **必须用 diff 校验 plan 输出**。

---

#### T-C2 拆 Python `build_report`（2d）

**现状**：`tools/perf_attribution_report.py:246–1102`（856 行）+
`aggregate_v2_reports:1103–1550`（447 行）= 1303/1588 行。

**改动点**：
1. 拆 4 个模块（放 `tools/attr/`）：
   - `attr_schema.py` — `validate_identity`、`require_string/int`
   - `attr_join.py` — plan/profile identity join、worker event join
   - `attr_partition.py` — wall 分区（`worker_ops_wall_attributed` 等）、窗口归一
   - `attr_report.py` — 报告序列化、`aggregate_v2_reports`
2. `perf_attribution_report.py` 变成编排层（`main` + `build_report` ≤ 100 行）
3. 补 Python 单测 `test/Native/test_attr_partition.py`：
   - wall 分区可加性（`accounted + unaccounted = top_level`）
   - 窗口归一、`wall_projection_factor` 一致性
   - worker join 覆盖率统计

**验收**：`python3 -m pytest test/Native/test_attr_*.py` 通过；
`check_perf_attribution.py` 端到端结果与重构前一致。

**风险**：低。但这是 P27/P28 bug 高发区，**拆完立即补单测**是重点。

---

#### T-C3 拆 `GenerateCAPI`（1.5d）

**现状**：`lib/Transforms/GenerateCAPI/GenerateCAPI.cpp` 2241 行、
**两个 pass 类挤一文件**（`runOnOperation` 在 `:118` 与 `:713`），
`prepareABI` 369 行、`finalizeDynamicRankABI` 300 行。

**改动点**：
1. ABI 构造与 manifest 序列化分两个 TU：
   `GenerateCAPIABI.cpp` / `GenerateCAPIManifest.cpp`
2. 动态 rank ABI 路径独立成 `GenerateCAPI.cpp`（保留原名，只留动态 rank pass 类）
3. `prepareABI` 拆成 `validateInputs` / `buildSignatures` / `emitDeclarations`

**验收**：每 TU ≤ 700 行；生成的 `.h` 与 manifest **逐字节一致**；CTest 全绿。

---

#### T-C4 拆 `ncnn-compile::main`（1.5d）

**现状**：`tools/ncnn-compile.cpp:2086–3397`（1312 行），混着参数规范化、
target 解析、数值校验、pipeline 构造、未定义符号审计、输出目录守卫、identity 派生。

**改动点**：
1. 抽 `class CompileSession`（放 `tools/CompileSession.{hpp,cpp}`）：
   ```cpp
   class CompileSession {
     llvm::Expected<void> parseArguments(ArrayRef<const char*>);
     llvm::Expected<void> validateAndResolveTarget();
     llvm::Expected<void> buildPipeline();
     llvm::Expected<void> auditUndefinedSymbols();
     llvm::Expected<void> emitIdentity();
     llvm::Expected<void> run();
   };
   ```
2. `main` 只做：`normalize_arguments` → 构造 session → `run()` → 返回码
3. 原子性靠 RAII / 显式 rollback，不靠"在同一个函数里"

**验收**：`main` ≤ 40 行；CLI 行为完全一致（`check_ncnn_compile.py` 全绿）。

---

### 4.3 P3：LLVM 库替换（可与 P1/P2 并行）

#### T-A1 `StringSwitch` 替换（0.5d，低风险）

**改动点**：
- `lib/Transforms/StrategyNCNN/StrategyNCNN.cpp:167` `parseStrategy` 8 行 if 链
- `tools/ncnn-compile.cpp` 中 26 处 `== "xxx"` 分支

```cpp
// 前
if (strategy == "auto") return ConvStrategy::Auto;
// 后
return llvm::StringSwitch<ConvStrategy>(strategy)
  .Case("auto", ConvStrategy::Auto)
  .Case("gemm", ConvStrategy::Gemm)
  .Case("conv", ConvStrategy::Conv)
  .Case("winograd", ConvStrategy::Winograd)
  .Default(ConvStrategy::Invalid);
```

**验收**：`grep -c 'StringSwitch' lib tools` ≥ 20；CLI 行为一致。

---

#### T-A2 APFloat→float/memcpy（1d，**唯一带性能收益的项**）

**问题**：f32 数据用 `APFloat`（约 20 B/元素 vs 4 B）。

| 位置 | 现状 |
|---|---|
| `PackStaticMatmulNCNN.cpp:614` | `SmallVector<APFloat> source` + `SmallVector<APFloat> packed` **双份物化** |
| `FoldNCNNBatchNorm.cpp:44` | `std::vector<APFloat> scaled` |
| `FoldNCNNBatchNorm.cpp:142,154` | `SmallVector<APFloat> shifted` ×2 |
| `StrategyNCNN.cpp:251` | `SmallVector<APFloat> transformed` |
| `NCNNToTosa.cpp:271-272` | `getValues<APFloat>()` 迭代 |

**改动点**：
1. `PackStaticMatmulNCNN.cpp`：panel 重排是**连续 16-float 段搬运** →
   ```cpp
   ArrayRef<float> src = elements.getValues<float>();
   SmallVector<float> packed(src.size());
   for (panel = 0; panel < columns; panel += packN)
     memcpy(&packed[out], &src[k*columns + panel], width * sizeof(float));
   ```
   **去掉中间 `source` 数组**
2. 其余 4 处改 `getValues<float>()` / `ArrayRef<float>`

**收益**：yolov5x_seg 38 MB packed 常量的临时内存从 **~390 MB 降到 ~76 MB**（−80%），
编译墙钟下降。**产物性能不变。**

**验收**：plan 的 `pack_bytes` / `pack_raw_bytes` 不变（**bit-identical packed 常量**）；
数值 golden 全绿；记录编译 wall / RSS 前后对比。

---

#### T-A3 `matchPattern` 末端匹配（0.5d，依赖 T-S1）

**改动点**：`findConstantElements` 内部末端用 `matchPattern(value, m_Constant(&attr))`，
替代手写 `dyn_cast<arith::ConstantOp>`。

---

#### T-S5 稳定哈希合并（0.5d，⚠️ identity 变化）

**改动点**：
1. 新建 `include/ncnn-mlir/Support/StableHash.hpp`：
   ```cpp
   uint64_t stableHash64(llvm::StringRef data);   // 内部 llvm::xxHash64
   ```
2. `InstrumentNCNNProfile.cpp:65` 与 `EmitModelPlan.cpp:157` 的两份 `fnv1a` 删除，改调 helper

⚠️ **换算法 = plan-hash 变值 = identity 变化**，必须：
- 在提交信息与证据目录记录 identity 变化
- **换算法那一轮停止 plan-hash 跨期 join**
- 更新 `perf_attribution_report.py` 的 identity 校验白名单

**验收**：两处调用点归一；plan-hash 变值已记录；attributions 仍能 join（同轮内）。

---

### 4.4 P4：结构拆分 + 测试补强

#### T-C5 tile-loop 公共层（2d，依赖 T-C1）

**现状**：`MatmulKernelNCNN.cpp` 三套近似重复的"行块 × 列块 × K 归约"：
- `kernelize`（421 行，f32，P3）
- `kernelizeInt8RowDot`（311 行，int8，P4）
- `vectorizeRowGeneric`（261 行，逐元素，P2）

**改动点**：
1. 抽 `emitTileLoopNest(ImplicitLocOpBuilder&, const TileShape&, BodyEmitter)`：
   ```cpp
   struct TileShape { int64_t rows, columns, depth, tileRows, accColumns; };
   using BodyEmitter = llvm::function_ref<void(Value kIndex, ValueRange accs)>;
   ```
2. 三个 kernel 只提供 body（累加类型、B 读取方式不同）

**验收**：三个 kernel 生成的 IR 在结构上等价（lit 对比）；数值 golden 全绿；
`MatmulKernelNCNN.cpp` 减少约 300 行。

---

#### T-C6 拆 `NCNNToTosa.cpp`（2.5d）

**现状**：6154 行，5 个 300–500 行 rewriter 平铺。

**改动点**：按算子族拆
```
Conversion/NCNNToTosa/ConvToTosa.cpp
Conversion/NCNNToTosa/AttentionToTosa.cpp
Conversion/NCNNToTosa/QuantToTosa.cpp
Conversion/NCNNToTosa/ElementwiseToTosa.cpp
Conversion/NCNNToTosa/DetectionToTosa.cpp
Conversion/NCNNToTosa/TosaLoweringUtils.hpp
```
`NCNNToTosa.cpp` 只留 pattern 注册。

**验收**：每文件 ≤ 1200 行；Conversion lit（48 个）全绿；产物 IR 不变。

---

#### T-C7 Winograd 独立（1d）

**改动点**：
1. `StrategyNCNN.cpp` 中 `kG[8][3]`/`kBt[8][8]`/`kAt[6][8]` →
   `include/ncnn-mlir/Support/Winograd63.hpp`（纯数据）
2. `rewriteWinograd`（486 行）→ `lib/Transforms/Winograd63NCNN/Winograd63NCNN.cpp`
3. `StrategyNCNN` 只留"conv → gemm/direct/winograd"分派

**验收**：`StrategyNCNN.cpp` 减少约 600 行；`resnet18_winograd` 数值 golden 不变。

---

#### T-M1 补薄弱 pass 的 lit（1.5d，依赖 T-C5）

| pass | 现有 | 目标 | 重点用例 |
|---|---:|---:|---|
| `TileMatmulForall` | 1 | 5 | tile 因子切分、M/N tail、forall 边界、位置对应 |
| `PackStaticMatmulNCNN` | 3 | 7 | 预算拒绝、布局拒绝、小形状跳过、常量穿透 cast |
| `RewriteLinalgCopies` | 2 | 6 | 同值消除、向量化主/尾块、动态 fallback、alias 负向 |
| `FoldLinalgConstantTranspose` | 1 | 4 | 转置折叠、非方形、常量穿透、拒绝路径 |

**验收**：新增 16–18 个 lit 全绿。

---

#### T-M2 pass 设计注释（1d，依赖 T-C6/C7）

每个 pass 文件顶部补 10–20 行：**职责、不变量、与其它 pass 的顺序依赖、明确不做什么**。
参照 `StrategyNCNN.cpp` 现有的 bounds 约束注释风格。

**覆盖**：P4 拆分后的所有 `NCNNToTosa/*`、`MatmulKernelNCNN`、
`EmitModelPlan`、`GenerateCAPI`、`PackStaticMatmulNCNN`、`InstrumentNCNNProfile`。

---

### 4.5 P5：收尾

#### T-M3 CLI 选项对照表（0.5d）

不改名（公开契约），建 `docs/cli-options.md`：

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--matmul-packing` | `off`/`auto` | `auto` | PackStaticMatmul | 96 MiB 预算 |
| `--packed-conv-depthwise` | flag | off | StrategyNCNN | P21 opt-in |
| `--selective-fusion-cast-chain` | flag | on | FuseLinalgEpilogue | |
| ... | | | | |

---

#### T-M4 拆 `InstrumentNCNNProfile` 插桩器（1.5d，依赖 T-C1）

**现状**：`runOnOperation` 408 行，P27/P28 为 worker 站点、采样窗口、fusion-site 插桩各插一段。

**改动点**：按事件类别拆独立插桩器
```
InstrumentAllocationSites.cpp
InstrumentCopySites.cpp
InstrumentOperationSites.cpp
InstrumentWorkerSites.cpp
InstrumentFusionSites.cpp   // 已有，见 P22 §9.7
```

---

#### T-M5 `emitWarning` 分级（1d）

**现状**：`emitWarning` 全库 0 次——该降级的情况（fallback 原因、未支持形状、
预算拒绝）现在只能静默或报错。

**改动点**：
1. 为下列情况补 `emitWarning`：`packing_rejected_budget`、
   `packing_skipped_small_shape`、`fallback_reason`、`worker_site_source_ambiguous`
2. 加 `--warnings-as-errors` 开关（默认关），供 CI 收紧

**验收**：编译 yolov5x_seg 时能看到预算拒绝的 warning 且不失败。

---

#### T-J1 C-runtime JSON writer 抽独立 + UTF-8 修复（1.5d，中风险）

**现状**：`lib/ProfileRuntime/profile_runtime.c`（2397 行）含
88 处 `fprintf(file, ...)` JSON 片段 + 40 行 `write_json_string` 转义。

**不能换 `llvm::json`**（硬约束）：
1. 以 `.c` 源文件 install（`tools/CMakeLists.txt:36`），由 `ncnn-compile` 在
   **每个模型编译期**随产物编译进生成的 `.so`
2. 必须纯 C，受未定义符号白名单 `profile_allowed` 约束
3. 引 LLVM Support 会破坏 P26 归档的正面性质（".so 只依赖 libomp+libc+libm"）

**改动点**：
1. 抽 `lib/ProfileRuntime/profile_json_writer.{c,h}`：
   ```c
   void pj_object_start(FILE*); void pj_object_end(FILE*);
   void pj_key(FILE*, const char*); void pj_string(FILE*, const char*);
   void pj_u64(FILE*, uint64_t); void pj_i64(FILE*, int64_t);
   void pj_bool(FILE*, int); void pj_null(FILE*);
   ```
2. **修 UTF-8 转义缺口**（全审计唯一 latent bug）：
   `write_json_string` 目前对 ≥0x80 字节直接 `fputc` 透传，**不校验 UTF-8**。
   层名来自 ncnn `.param` 文本，正常 ASCII 但无强制——非法字节会产出非法 JSON。
   **改法**：校验 UTF-8 序列；非法字节转 `�` 或 `\uXXXX`
3. **补转义边界测试**：`check_profile_runtime.py` 新增用例
   （layer 名含 `"`、`\`、控制字符、非 ASCII、截断 UTF-8）
4. 补注释说明"为什么不用 `strtoul`/`qsort`/`llvm::json`"

**保留的设计**：数值字段全用 `%llu` 出整数、**完全不输出浮点** ——
避开手写 JSON writer 最经典的 NaN/Infinity 陷阱。**不要"顺手加浮点输出"。**

**验收**：
- 新增 5 个转义边界 round-trip 用例全绿
- 生成的 profile 仍能被 `json.loads` 解析
- **未定义符号白名单不变**（`profile_allowed` diff 为空）
- 生成 `.so` 的依赖仍是 libomp + libc + libm（P26 审计脚本复跑）

---

## 5. 阶段计划与时间线

| 阶段 | 任务 | 工时 | 累计 | 关键交付 |
|---|---|---:|---:|---|
| **P1 语义收敛** | T-S1, T-S2, T-S4, T-S3 | 5–6d | 5–6d | schema 单一事实源；常量查找/溢出检查唯一实现 |
| **P2 巨型函数** | T-C1, T-C2, T-C3, T-C4 | 7.5d | 12–13d | `runOnOperation` ≤80 行；Python `build_report` ≤100 行 |
| **P3 库替换** | T-A1, T-A2, T-A3, T-S5 | 2.5d | 15–16d | 27 处 if 链消失；编译内存 −80%（大模型） |
| **P4 结构+测试** | T-C5, T-C6, T-C7, T-M1, T-M2 | 8d | 23–24d | tile-loop 公共层；NCNNToTosa 拆 5 文件；+18 lit |
| **P5 收尾** | T-M3, T-M4, T-M5, T-J1 | 4.5d | 27–29d | JSON writer 独立 + UTF-8 修复；warning 分级 |

**并行建议**：P1 与 P2/P3 错开（T-S2 改大量文件，冲突面大）；
P2 与 P3 可并行（改不同文件）；P4 依赖 P2 的 T-C1/T-C5。

---

## 6. 统一验证流程

**每完成一个任务**必须跑：

```bash
# 1. 格式与静态检查
cmake --build <stage> --target format_check --parallel 16
cmake --build <stage> --target tidy --parallel 16

# 2. 完整 CTest（449 registered，2 个 upstream INT8 skip 原样披露）
ctest --test-dir <stage> --output-on-failure

# 3. 三个 numerical target
cmake --build <stage> --target \
  numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel 16
```

**按任务补充验证**：

| 任务 | 额外验证 |
|---|---|
| T-S1 | 数值 golden（可能改变折叠路径）；plan diff |
| T-S2 | `grep -rE '"ncnn\.' lib tools` 归零；属性白名单测试 |
| T-C1/T-C3 | `.plan.json` / `.h` 逐字节 diff |
| T-C2 | Python 单测 + `check_perf_attribution.py` 结果对比 |
| T-A2 | `pack_bytes` 不变（bit-identical packed 常量）；编译 wall/RSS 前后对比 |
| T-S5 | identity 变化已记录；停止跨期 plan-hash join |
| T-J1 | `profile_allowed` 白名单 diff 为空；`.so` 依赖审计 |

**阶段收尾**额外跑：

```bash
# fresh stage 重建（§7 纪律）
rm -rf /tmp/ncnn-compiler-stage-refactor-P<N>
cmake -S compiler -B /tmp/ncnn-compiler-stage-refactor-P<N> \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-refactor-P<N> --parallel 16
ctest --test-dir /tmp/ncnn-compiler-stage-refactor-P<N> --output-on-failure
```

**P2 与 P5 收尾时**额外复跑性能基线（确认未漂移）：
6T、warmup=10、iterations=20、`end_to_end`、profile-off，对比
P27 `compiler/docs/performance/p27-2026-09-25/final-all-model-ncnn-comparison.ndjson`。
**预期 p50/heavy p50 无显著变化**（1.844× / 1.437× 量级）。

---

## 7. 硬性约束与纪律

1. **不改任何性能默认值**：`NCNN_MATMUL_PACKING=auto`、
   `NCNN_PACKED_CONV_DEPTHWISE=OFF`、`NCNN_NATIVE_INT8_PROFILE=OFF` 保持原状
2. **不引入新 no-go 路径**：Winograd / layout island / global arena / native INT8
   边界原样保留
3. **每步用新 `/tmp` stage 验证**，失败后修复 + 新目录重建，不用旧产物推断
4. **数值 golden 不放宽**
5. **plan/profile 字段增减 = identity 变化**，必须记录，禁止跨 identity 误 join
6. **换哈希算法 = identity 变化**（T-S5），换算法那轮停止 plan-hash 跨期 join
7. **C runtime 不引 LLVM、不扩未定义符号白名单**（`profile_allowed` 约束）
8. **加 `assert` 是唯一允许的行为变更**：内部不变量加固，不影响数值；
   断言失败信息必须可定位（带 op/type 上下文）
9. **每步 commit 可读**：`git log -p` 能看清改动；禁止"一次大提交"

---

## 8. 明确不做

- **不做"顺便优化"** — 重构期间发现的性能线索进 backlog，不混入本期
- **不引新依赖** — 第三方 JSON 库、测试框架
- **不统一改 CLI 选项名** — 公开契约，破坏兼容不划算（只建对照表）
- **不为覆盖率写测试** — 只补 T-M1 列出的四个薄弱点
- **不做 T-S3 的"彻底档"** — 引入 `ncnn.ledger` 字典 + 改 pass 顺序契约，
  风险大于收益；先做保守档
- **不重写 `graph.cpp`** — 已核实结构良好（v1 误报）
- **不给 C runtime 加浮点 JSON 输出** — 现有"只出整数"设计更安全
- **不追求全面追平 ncnn** — 性能已收官（heavy 1.35×），本次不承诺性能收益

---

## 9. 回滚策略

| 情况 | 处理 |
|---|---|
| 单任务改动引入数值回归 | `git revert` 该任务提交；该任务重新设计后再做 |
| 阶段收尾 CTest 红 | 回退该阶段所有提交，定位后分步重做；**不带病推进下一阶段** |
| 重构后性能基线漂移 >2% | **优先怀疑重构破坏语义**（而非"顺带优化生效"）；用 P27 基线逐模型定位 |
| T-S1 改变 batchnorm 折叠路径导致数值差异 | 若超出 golden 预算，收窄 helper 的穿透集合（只保留与 packing 一致的 view 集合） |
| T-S5 plan-hash 变值导致 join 失败 | 这是预期行为；确认已停止跨期 join，而非"修 join 让它通过" |
| T-J1 UTF-8 修复改变 profile 内容 | 属修 bug；但需确认 `check_perf_attribution.py` 仍能 join |

**整体放弃点**：若 P1 完成后发现语义收敛的连锁影响过大（例如 T-S2 暴露出大量
隐式依赖），**允许暂停并在 P1 交付一份"遗留问题清单"**，而不是强行推进 P2–P5。

---

## 10. 交付物清单

每个阶段归档到 `compiler/docs/refactor/p<阶段>-<date>/`：

```
README.md                 # 本阶段做了什么、验收结果、遗留问题
task-<ID>-diff.txt        # 每个任务的 diff 摘要
verification.log          # format_check + tidy + CTest 输出
plan-before-after.diff    # T-C1/T-C3 的 plan/.h 逐字节对比
identity-changes.md       # identity 变化记录（T-S1/T-S5 触发时）
perf-baseline-diff.md     # P2/P5 的性能基线对比（若跑）
```

**不把唯一证据留在 `/tmp`**（沿用 P18 §3.3 纪律）。

---

## 11. 预期收益

| 收益 | 说明 | 对应任务 |
|---|---|---|
| **同一语义只有一个答案** | batchnorm/packing 对常量判断一致；溢出检查语义统一 | T-S1, T-S4 |
| **schema 真正单一事实源** | 187 处裸字面量消失，属性键错用编译期可查 | T-S2 |
| **新能力有自然落点** | P27/P28 那种"往 1648 行函数中段插逻辑"消失 | T-C1, T-C2 |
| **bug 定位更快** | wall-partition、worker join 不再需通读千行函数 | T-C2 |
| **内部不变量可查** | `assert` 兜底，错误机器码 → 断言失败 | §7.8 |
| **编译期内存下降** | APFloat→float 约 −80% 临时内存（大模型） | T-A2 |
| **latent bug 修复** | 非法 UTF-8 层名不再产出非法 JSON | T-J1 |
| **新人可上手** | pass 有设计注释、Winograd 独立、CLI 有对照表 | T-M2, T-C7, T-M3 |

**不承诺性能收益。** P26/P27 的 heavy 1.35× / 整体 1.95× 基线应保持不变；
若重构后基线漂移，优先怀疑重构破坏语义，而非"顺带优化生效"。

---

## 12. 进度跟踪

| ID | 状态 | 开始 | 完成 | 提交 | 备注 |
|---|---|---|---|---|---|
| T-S1 | ☑ | 2026-09-26 | 2026-09-26 | `9ed5e9d` | 常量查找收敛 `findConstantElements` |
| T-S2 | ☑ | 2026-09-26 | 2026-09-26 | `9ed5e9d` | 属性 schema 收敛 |
| T-S4 | ☑ | 2026-09-26 | 2026-09-26 | `9ed5e9d` | `CheckedMath.hpp` |
| T-S3 | ☑ | 2026-09-26 | 2026-09-26 | `9ed5e9d` | `ModelLedger.hpp` 保守档 |
| T-C1 | ☑ | 2026-09-26 | 2026-09-26 | `6cf3438` | `PlanCollector`；不按「13 个独立纯函数」拆 |
| T-C2 | ☑ | 2026-09-26 | 2026-09-26 | `6cf3438` | `tools/attr/` 四模块 |
| T-C3 | ☑ | 2026-09-26 | 2026-09-26 | `6cf3438` | 4 TU + 1 头；`prepareABI` 不拆 |
| T-C4 | ☑ | 2026-09-26 | 2026-09-26 | `6cf3438` | `CompileSession` |
| T-A1 | ☑ | 2026-09-26 | 2026-09-26 | `3d1c0fc` | 13 处 `StringSwitch` |
| T-A2 | ✗ | — | — | — | **不做**（影响交叉编译，用户判定）；见 P3 backlog |
| T-A3 | ☑ | 2026-09-26 | 2026-09-26 | `3d1c0fc` | 末端 `m_Constant` |
| T-S5 | ☑ | 2026-09-26 | 2026-09-26 | `3d1c0fc` | ⚠️ identity 变化；三处 FNV-1a 合并 |
| T-C5 | ☑ | 2026-09-26 | 2026-09-26 | P4 | `emitTileLoopNest`；「−300 行」不成立 |
| T-C6 | ☑ | 2026-09-26 | 2026-09-26 | P4 | 6160 → 13 TU；6 文件装不下 ≤1200 |
| T-C7 | ☑ | 2026-09-26 | 2026-09-26 | P4 | `StrategyNCNN.cpp` −671 行 |
| T-M1 | ☑ | 2026-09-26 | 2026-09-26 | P4 | +14 lit，192→206；「常量穿透」落负向 |
| T-M2 | ☑ | 2026-09-26 | 2026-09-26 | P4 | 18 个文件四段式设计注释 |
| T-M3 | ☐ | | | | |
| T-M4 | ☐ | | | | |
| T-M5 | ☐ | | | | |
| T-J1 | ☐ | | | | ⚠️ latent bug |

> 逐阶段归档：`compiler/docs/refactor/p{1..4}-2026-09-26/`。
> P4 的「提交」列填 P4，指本次提交（哈希见 `git log`）。
