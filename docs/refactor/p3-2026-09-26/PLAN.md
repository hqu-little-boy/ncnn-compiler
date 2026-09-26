# P3：LLVM 库替换 — 执行计划

> **阶段**：P3 = T-A1 + T-A3 + T-S5（**T-A2 不做**）
> **日期**：2026-09-26
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.3 / §6 / §7
> **前置**：P1 语义收敛 `9ed5e9d`、P2 巨型函数拆分 `6cf3438` 已落库
> **分支**：`p8-implementation`

---

## 0. Context

P18–P28 性能阶段已收官（heavy p50=1.35×、整体 1.95×），主线转向代码质量重构。
roadmap 把重构拆成 5 阶段 21 任务；P1、P2 已提交。P3 是第三阶段：**把项目手写的
基础设施换成 LLVM 已有的库实现**。

### 本次范围变更（用户指令）

**T-A2（APFloat→float/memcpy）不做，理由是会影响交叉编译。**

剔除 T-A2 的直接后果：

- roadmap 称 T-A2 是 P3「**唯一带性能收益的项**」，剔除后本阶段
  **不承诺任何编译期内存或性能收益**，只剩语义收敛。
- roadmap 的 P3 阶段行「编译内存 −80%（大模型）」**不再兑现**，属对 roadmap 的
  范围偏离，在 `README.md` / `task-diffs.md` 披露。
- T-A2 连同其验收证据一并取消，记录到 `backlog.md`。

### 本阶段实际交付（三项）

| 项 | 性质 | 行为影响 | 交付 |
|---|---|---|---|
| **T-A1** `StringSwitch` 替换字符串分派 | 等价改写 | 零（已证） | 多路分派链收敛到 `llvm::StringSwitch` |
| **T-A3** `matchPattern` 收尾常量匹配 | 等价改写（末端 matcher 更宽） | 产品路径零（已证） | `ConstantFold.cpp` 末端用 `m_Constant` |
| **T-S5** 稳定哈希合并 `llvm::xxHash64` | **identity 变化** | `plan_hash` / `profile_id` 全量变值 | 三处手写 FNV-1a → 一个 `stableHash64` |

### 执行决策

| 决策 | 取值 | 理由 |
|---|---|---|
| T-A2 | **不做** | 用户指令：影响交叉编译 |
| T-S5 | **纳入** | 用户确认；换算法本身即 identity 变化 |
| 逐字节对照 | **做** | 用户确认；T-A1/A3 须证明产物不变，且必须在 T-S5 之前采证 |
| 提交粒度 | **每阶段一笔 `refactor(ncnn)` 提交**（代码 + 归档） | 与 P1/P2 一致 |
| 任务顺序 | **T-A1 → T-A3 → 采证 → T-S5 → 再门禁** | T-S5 放最后：零行为变化的证据须在 **identity 稳定窗口**内采集 |
| 不推送 | 只 commit，不 push | 用户约束 |
| 门禁失败 | 不提交；修复后**重新从全新 `/tmp` 目录**构建和测试 | 用户约束；policy §7 |

---

## 1. 任务分解

### T-A1 `StringSwitch` 替换字符串分派

**收编边界**（每个站点须同时满足）：
1. **单键 → 单值**映射
2. 分支数 ≥3，**或** 2 路但产出具体值/类型（非布尔成员门）
3. 字符串来源是 CLI 选项值、pass 选项值或序列化 schema token

**硬否决条件**（`StringSwitch` 表达不了，留原样并在归档点名）：
- **复合键**（`a == "x" && b == "y"` 的组合分派）
- **带状态守卫的字段解析**（`name == "min" && !sawMinimum`）
- 分派键是**枚举而非字符串**

**不收**：2 分支布尔成员门、算子名/dialect 名成员测试、路径与产物名比较、
动态拼接串比较。

**收编 13 处**（含 2 处两段式：先 `StringSwitch` 解析成枚举，再用真 `switch`
执行语句体）：

| 文件 | 站点 |
|---|---|
| `lib/Transforms/StrategyNCNN/StrategyNCNN.cpp` | `parseStrategy` |
| `lib/Support/Precision.cpp` | `parse_precision_mode`、`parse_fp16_accumulator_mode` |
| `lib/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.cpp` | `int8Kernel` / `int8Target` 成员校验 |
| `lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` | `getLowPrecisionStorageType`、`parallelPolicy` |
| `tools/ncnn-compile.cpp` | `g_int8_kernel`、`g_optimization`、`g_tuning_profile`、`g_conv_strategy`、`tuning.matmulPacking`、`g_vector_math` 校验；`g_vector_mode` / `g_vector_math` 两段式分派 |

**按硬否决条件移出**的站点见 `backlog.md` §6。

**实现纪律**：校验链一律 `.Default(false)` + 外层 `return fail(msg)`，
**绝不写 `.Default(fail(msg))`** —— `StringSwitch::Default(T)` 按值求值，
那样会在合法输入上也触发报错。所有报错串**逐字不动**。

---

### T-A3 用 `matchPattern` 收尾常量匹配

只换 `lib/Support/ConstantFold.cpp` 中 `findConstantElements` 的**末端那一跳**：
`dyn_cast<arith::ConstantOp>` → `matchPattern(value, m_Constant(&attr))`。

- **视图穿透集合不变**：仍是 `tensor.cast` / `collapse_shape` / `expand_shape`，
  明确排除 `tensor.extract_slice` / `tensor.from_elements`。
- **视图 walk 位于末端之前**，因此即使某 view op 带 `ConstantLike`，
  也仍按 view 穿透而不是被 fold。
- **已知加宽**：`m_Constant` 接受任何 `ConstantLike` op（`ncnn.const`、
  `tosa.const` 等），旧代码只接受 `arith.constant`。产品流水线中两者均不可达
  （`ConvertNCNNModelToFunc` 先把 `ncnn.const` 重写掉；`PackStaticMatmulNCNN`
  在 `VerifyNoTosaOps` 之后），逐字节对照已证产品路径零变化。加宽对独立
  `ncnn-mlir-opt` 使用成立，记入 `identity-changes.md`。

---

### T-S5 稳定哈希合并到 `llvm::xxHash64`（⚠️ identity 变化）

**roadmap 只记了两份手写 FNV-1a，实测三份**：

| 位置 | 函数 | 产出 |
|---|---|---|
| `InstrumentNCNNProfile.cpp` | `fnv1a` | profile 事件 id |
| `EmitModelPlan.cpp` | `profileId` | `*.profile_id`、`plan_hash`、`build_identity` |
| `FuseLinalgEpilogue.cpp` | `fusionProfileId` | `ncnn.fusion_site_id`（`& 0x7fffffffffffffff`） |

三处必须**锁步**改：`plan.operations[].profile_id` 与 profile 事件 `id`
哈希的是同一输入串，只改一处会让操作静默掉进 `unattributed`。

改动：新建 header-only `include/ncnn-mlir/Support/StableHash.hpp`，
`inline std::uint64_t stableHash64(llvm::StringRef)` 包 `llvm::xxHash64`；
三处手写 FNV-1a 删除。`llvm::xxHash64(StringRef)` **无 seed 参数**，
符号来自 LLVMSupport，三个调用点的 TU 均已传递链接。

**不动**：`test/Numerical/support/performance_test_support.cpp` 的
`performance_input_hash`（浮点位模式，语义不同）、
`test/Native/check_perf_attribution.py` 的 python `fnv1a64`
（合成 fixture 生成器，自产自校）。

---

## 2. 执行顺序与逐字节对照

```
0. 建 baseline：当前 HEAD 建 stage，产出 10 模型产物 → 「改前」基线
1. T-A1  —— StringSwitch
2. T-A3  —— matchPattern 末端
3. 门禁 #1：format_check + tidy + 全新 /tmp stage 构建 + 三 numerical + 全量 CTest
4. 逐字节对照：改后产物 vs 第 0 步基线   ← identity 稳定窗口
5. T-S5  —— 三处 FNV-1a → stableHash64
6. 门禁 #2：同上再跑一遍全套
7. 写 identity-changes.md
8. 归档 + git 提交（不推送）
```

**关键**：第 4 步必须在第 5 步之前完成。T-S5 之后 `plan_hash` / `profile_id` 必变，
无法再做全量 diff。

**对照面**：10 个模型 × 7 种产物
（`.plan.json` / `.h` / manifest `.json` / `.so` / `model.{ncnn,tosa,linalg}.mlir`）。
模型清单沿用 P2 的 9 模型，另加 `resnet18_winograd` 覆盖 `--conv-strategy` 分派
（T-A1 改 `parseStrategy`）。

**归一化**：IR dump 的 `loc(...)` 会嵌入编译期临时目录
`/tmp/ncnn-compile-XXXXXX`（每轮随机）。对照前把该 token 归一为
`/tmp/ncnn-compile-TMPDIR`。**`.plan.json` / `.h` / manifest / `.so` 不含该 token，
是逐字节比对，无归一化。**

---

## 3. 统一提交门禁

```bash
# 1. 全局静态检查（含非本次产生的问题，按结果修复）
cmake --build <stage> --target format_check --parallel
cmake --build <stage> --target tidy --parallel

# 2. 全新 /tmp stage，避免旧产物误导
mkdir /tmp/ncnn-compiler-stage-refactor-p3
cmake -S compiler -B /tmp/ncnn-compiler-stage-refactor-p3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-refactor-p3 --parallel

# 3. 三个 numerical target（必须显式 build，ctest 不重建夹具）
cmake --build /tmp/ncnn-compiler-stage-refactor-p3 \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel

# 4. 全量 CTest
ctest --test-dir /tmp/ncnn-compiler-stage-refactor-p3 --output-on-failure

# 5. 提交前检查
git diff --check
git status --short
git diff --cached
git log --oneline -10
```

门禁跑**两次**（T-A1/A3 之后、T-S5 之后），两次全量 CTest 都必须过。

---

## 4. 交付物

```
docs/refactor/p3-2026-09-26/
  PLAN.md                 # 本文件
  README.md               # 阶段结论
  task-diffs.md           # 三任务改动摘要（含「不做」的逐条理由）
  verification.log        # format_check / tidy / build / numerical / CTest / 逐字节对照
  identity-changes.md     # T-S5 identity 变化记录 + T-A3 加宽记录
  plan-before-after.diff  # 归一化后逐字节对照结果（空 = 全一致）
  emit-plans.sh           # 采证脚本
  backlog.md              # 遗留问题与重启前置
```

**不把唯一证据留在 `/tmp`**（roadmap §10）。
