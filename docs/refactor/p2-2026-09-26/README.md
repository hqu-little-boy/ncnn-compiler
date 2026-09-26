# P2 巨型函数拆分 · 阶段归档（2026-09-26）

> **阶段**：P2（T-C1 / T-C2 / T-C3 / T-C4）
> **性质**：纯机械拆分，不改任何语义
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.2
> **前置**：P1 语义收敛已落（`9ed5e9d`）

---

## 结论

四个巨型函数全部拆开，**产物逐字节不变**，全量门禁通过，性能基线无漂移。

| 任务 | 目标 | 拆分前 | 拆分后 | 验收 |
|---|---|---:|---|---|
| **T-C1** | `EmitModelPlan::runOnOperation` | **1648 行** | `runOnOperation` **40 行**；36 个方法、最大 169 行 | ✓ |
| **T-C2** | Python `build_report` + `aggregate_v2_reports` | 855 + 446 = **1301 行** | 编排层 `build_report` **50 行**；4 个模块 | ✓ |
| **T-C3** | `GenerateCAPI` 两个 pass 挤一文件 | 2228 行单文件 | 4 个 TU + 1 头，**最大 687 行** | ✓（一处改动点偏离，见下） |
| **T-C4** | `ncnn-compile::main` | **1313 行** | `main` **4 行**；9 个阶段方法 | ✓ |

**核心护栏是逐字节对照，不是肉眼**：同一组 9 个模型（含 2 个动态形状用例，
覆盖 `finalizeDynamicRankABI`）经重构前后的 `ncnn-compile` 产出
`.plan.json` / `.h` / manifest `.json`，`diff -ru` **输出为空（exit 0）**。

---

## 交付物

```
PLAN.md                   # 执行计划 + 对 roadmap 审计数字的勘误 + 三处硬约束
task-diffs.md             # 四个任务的改动摘要、手法、逐条披露
verification.log          # format_check / tidy / numerical / CTest / 逐字节对照
plan-before-after.diff    # 逐字节对照结果（空 = 全部一致）
identity-changes.md       # identity 变化记录（结论：无）
perf-baseline-diff.md     # 与 P27 基线的性能对比 + 与 ncnn 的差距
summarize-perf.py         # 性能汇总脚本
```

---

## 与 roadmap 的偏离（两处，均已披露）

### 1. T-C1：不按「13 个独立纯函数」拆

roadmap 建议把 13 个 JSON 段抽成各自独立遍历的纯函数。**与该 pass 的设计注释直接冲突**：

> This pass intentionally centralizes the complete static-plan walk so that
> all counters and the plan hash are derived from one deterministic visit.

实测 `plan_hash_input` 在 **6 个位置**被顺序累加。若拆成独立遍历，拼接顺序与计数器
口径都会变 → `plan_hash` 变值 → identity 变化，违反「不动语义」。

**改法**：保留单次确定性遍历，抽成 `PlanCollector`，`collect*` 方法按**固定顺序**被
调用。函数名沿用 roadmap 的 `collectXxx`。`runOnOperation` ≤ 80 行的验收不变。

### 2. T-C3：`prepareABI` 不拆三段

roadmap 改动点要求 `prepareABI` 拆成 `validateInputs` / `buildSignatures` /
`emitDeclarations`。**与同一任务的验收「每 TU ≤ 700 行」冲突**：`GenerateCAPIPrepare.cpp`
现为 687 行，拆分新增约 30 行签名/编排会把它顶到约 720 行。

按**验收标准优先于改动点**取舍，`prepareABI` 保留原样（328 行）。若要补齐，需先把
`writeManifest` 改成 `capi_detail` 自由函数腾出行数预算，建议进 backlog。

### 3. T-C2：pytest 不可用

本机未安装 pytest（也没有 pip），roadmap 验收里的
`python3 -m pytest test/Native/test_attr_*.py` 无法执行。新增单测写成纯
`unittest.TestCase`（pytest 可直接收集），实际以
`python3 test/Native/test_attr_*.py` 与 ctest 注册项执行。

---

## 拆分手法（可复用）

四项任务统一用同一手法，这是它们能保持"纯机械"的原因：

> **给新类/新结构的成员起与原函数局部变量完全相同的名字**，于是抽出的方法体是
> **逐字拷贝**，只需统一缩进。唯一允许的文本改动是把局部声明改成对成员的赋值、
> 容器成员补 `.clear()`、以及少数必要的限定/返回值调整。

T-C1 实测：`runOnOperation` 的语义体几乎逐字进入 `PlanCollector`。
T-C4 实测：1311 行方法体中 **1216 行字节相同**，73 处声明改赋值、22 处
`run(` → `::run(`（成员遮蔽自由函数）、8 处补 `return 0;`，**0 处未授权改动**。

---

## 三处 roadmap 未记的硬约束（执行中发现）

1. **`Passes.h.inc` 的 `GEN_PASS_DEF_*` 段只能单 TU 包含** —— 它同时产出
   `impl::<Pass>Base` 模板**和一个非 inline 的 `create<Pass>()` 自由函数**。
   因此 pass 类各自独占一个 TU，跨 TU 共享只能走具名 namespace 的自由函数。
   这决定了 T-C3 的最终文件布局。

2. **`tools/CMakeLists.txt` 与 `lib/Transforms/GenerateCAPI/CMakeLists.txt` 都是
   显式源列表**（不是 glob），新增 `.cpp` 必须显式补；
   `test/Numerical/CMakeLists.txt` 的 fixture `DEPENDS` 也要补，否则模型不会重编译。

3. **`llvm::json::Object` 底层是 DenseMap、序列化按键名字母序** ——
   对象键的插入顺序**不是承重的**；承重的是**数组元素顺序**与
   **`plan_hash_input` 拼接顺序**。逐字节 diff 覆盖前者，新增的
   `plan-hash-whitelist` 源码级守卫覆盖后者。

---

## 新增守卫

| 测试 | 作用 |
|---|---|
| `plan-hash-whitelist`（ctest） | 源码级断言 `plan_hash_input` 的 mutation 位点集合**恰好等于**登记白名单（14 位点 / 12 片段 / 4 折叠片段）。新增 hash 片段而不更新白名单 → 测试红，强制 identity 变更成为显式决策 |
| `Transforms/EmitModelPlan/plan-hash-whitelist.mlir` | 行为级：只改输出路径不改 `plan_hash`；改白名单字段（threads）必改 `plan_hash`；`build_identity` 恒等于 `plan_hash` |
| `attr-partition` / `attr-report`（ctest） | T-C2 拆出的 wall 分区可加性、`wall_projection_factor`、worker join 覆盖率、`aggregate_v2_reports` 中位数与 `reports[0]` 别名语义 |

---

## 遗留问题

- **T-C3 的 `prepareABI` 三段拆分**：见上文偏离 2，建议进 backlog。
- **`layout_island_hash_input` 的死写**（折叠进 `plan_hash_input` 之后仍有两次 `+=`）：
  为保持纯位移**原样保留**并已在代码注释标注。清理属行为中性的收尾，不混入本期。
- **`GenerateCAPIEmitInfer.cpp` F8 shape-carrier 分支**：`shapeSources[outputIndex - 1]`
  未检查 `outputIndex > 0`（只守卫了 `paired`）。理论上 carrier 落在 output index 0
  会下溢，但 `prepareABI` 强制每个 carrier 前面有对应数据输出，看起来不可达。
  **本次未改**（不动语义），建议单独评估。
- **`ScopedDirectory` 仍可被隐式拷贝**（拷贝后两个对象都会 `remove_all`）。
  T-C4 为了不改动既有构造/析构只加了默认构造与移动赋值，未 `= delete` 拷贝。
  现有用法不拷贝，属潜在隐患。

---

## 不做（沿用 roadmap §8）

- 不做"顺便优化"——发现的性能线索进 backlog
- 不引新依赖
- 不改 CLI 选项名
- 不改任何 plan/profile 字段名与字段值
- 不改性能默认值（`NCNN_MATMUL_PACKING=auto`、`NCNN_PACKED_CONV_DEPTHWISE=OFF`、
  `NCNN_NATIVE_INT8_PROFILE=OFF`）
- 不重写 `graph.cpp`
