# P3 各任务改动摘要

改动面：11 个修改文件 + 2 个新增文件
（`include/ncnn-mlir/Support/StableHash.hpp`、`docs/refactor/p3-2026-09-26/`）。
统计：+212 / −164（不含归档）。

---

## T-A1 `StringSwitch` 替换字符串分派

**新增 include**：`llvm/ADT/StringSwitch.h`（LLVM 21 系统头，不引新依赖）。
全库此前**零** `StringSwitch` 使用，落地后 `grep -c 'StringSwitch' lib tools` = **20**，
达到 roadmap 验收门槛。

### 收编的 13 处

| 文件 | 站点 | 形态 |
|---|---|---|
| `lib/Transforms/StrategyNCNN/StrategyNCNN.cpp` | `parseStrategy` | `StringSwitch<std::optional<ConvStrategy>>` + `.Default(std::nullopt)`，**保持 `std::optional` 签名** |
| `lib/Support/Precision.cpp` | `parse_precision_mode` | `StringSwitch<std::optional<PrecisionMode>>`，**保持 `std::expected` 签名** |
| `lib/Support/Precision.cpp` | `parse_fp16_accumulator_mode` | 同上 |
| `lib/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.cpp` | `int8Kernel` / `int8Target` 成员校验 | 两个 `StringSwitch<bool>`；第三个 `vnni` 配对约束保持原布尔式 |
| `lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` | `getLowPrecisionStorageType` | `StringSwitch<Type>` |
| `lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` | `parallelPolicy` | `StringSwitch<StringRef>`（原为 4 路嵌套三元） |
| `tools/ncnn-compile.cpp` | `g_int8_kernel` / `g_optimization` / `g_tuning_profile` / `g_conv_strategy` / `tuning.matmulPacking` / `g_vector_math` 校验 | `StringSwitch<bool>` + 外层 `fail()` |
| `tools/ncnn-compile.cpp` | `g_vector_mode` | **两段式**：`StringSwitch<std::optional<VectorMode>>` 解析成函数内 `enum class`，再用真 `switch` 执行语句体 |
| `tools/ncnn-compile.cpp` | `g_vector_math` 分派 | 同上（`enum class VectorMathBackend`） |

### 实现纪律

- **校验链一律 `.Default(false)` + 外层 `return fail(msg)`**。
  绝不写 `.Default(fail(msg))`：`StringSwitch::Default(T)` 按值求值，
  那样会在合法输入上也触发报错并打印错误。
- **所有报错串逐字不动**（`check_int8_target.py:111/263` 断言
  `"must be one of"` / `"--tuning-profile must be one of"`；另有 lit `INVALID:`
  检查依赖这些文案）。`git diff` 中只出现 `return fail(` 的缩进行变化，
  字面量本身未改。
- 校验/分派的**位置与先后顺序未变**（`-O` 仍在 vector-mode 之前等）。

### 按硬否决条件**不改**（`StringSwitch` 表达不了）

| 位置 | 原因 |
|---|---|
| `lib/Transforms/EmitModelPlan/EmitModelPlan.cpp` `requested_policy_status` 段 | **复合键 + 双输出**：`int8_kernel == "auto" && int8_target == …` 组合分派，每支同时写 `requested_policy_status` 与 `requested_policy_fallback_reason` |
| 同文件 `low_precision_hash_input` 三元链 | 同上复合键 |
| `lib/Support/Precision.cpp` `target_execution_profile` | 外层分派键是 `PrecisionMode` **枚举**，不是字符串 |
| `tools/ncnn-mlir-driver.cpp` input-dim-constraint 字段解析 | `name == "min" && !sawMinimum` **带状态守卫**（拒绝重复字段），StringSwitch 会掩盖该语义 |

理由：强行改写属「顺手重构」，违背 policy §4「最小正确修改」。
逐条记入 `backlog.md` §6。

### 对 roadmap 验收的偏离

roadmap §4.3 只点名 `StrategyNCNN.cpp:167`（实为 `:145-159`）与
`ncnn-compile.cpp` 的 26 处 `== "xxx"`。实测同类链另在
`Precision.cpp`、`MatmulKernelNCNN`、`NCNNToTosa`、`ncnn-mlir-driver`。
按「单键 → 单值」边界收 13 处，`grep -c` 达到 20。
**没有**为凑数改写单点布尔开关、算子名成员测试或路径比较。

---

## T-A3 用 `matchPattern` 收尾常量匹配

`lib/Support/ConstantFold.cpp` 的 `findConstantElements`：
末端 `dyn_cast<arith::ConstantOp>` → `matchPattern(value, m_Constant(&attr))`。
新增 include `mlir/IR/Matchers.h`（全库此前零 `matchPattern` / `m_Constant`）。

**只换末端那一跳**，并且把末端放在视图 walk **之后**——这样即使某 view op 带
`ConstantLike`，也仍按 view 穿透而不是被 fold。视图穿透集合不变：
`tensor.cast` / `collapse_shape` / `expand_shape`，排除 `extract_slice` /
`from_elements`。

`include/ncnn-mlir/Support/ConstantFold.hpp` 契约注释从「`arith.constant`」改写为
「ConstantLike 常量」，并写明加宽范围与两个调用点在产品流水线中为何只见到
`arith.constant`。加宽记录见 `identity-changes.md` §2。

**测试**：不新增 lit。行为由 P1 既有 lit
（`FoldNCNNBatchNorm/constant-through-cast.mlir`、
`PackStaticMatmulNCNN/constant-through-cast.mlir` 及负向用例）+ 逐字节对照覆盖。
`from_elements` 负向用例缺失，记入 `backlog.md` §7。

---

## T-S5 稳定哈希合并到 `llvm::xxHash64`（**identity 变化**）

**新增** `include/ncnn-mlir/Support/StableHash.hpp`：
header-only `inline std::uint64_t stableHash64(llvm::StringRef)` 包
`llvm::xxHash64`（LLVM 21 签名 `uint64_t xxHash64(StringRef)`，**无 seed 参数**）。
风格照 `CheckedMath.hpp` / `Int8Target.hpp`（`#pragma once`、`namespace ncnn_mlir`）。
**不动 `lib/Support/CMakeLists.txt`**：header-only，三个调用点的 TU 均已传递链接 LLVMSupport。

**三处手写 FNV-1a 锁步合并**（roadmap 只记了两处，第三处是漏列）：

| 文件 | 处理 |
|---|---|
| `InstrumentNCNNProfile.cpp` | 删除 `fnv1a`，2 个调用点改调 `ncnn_mlir::stableHash64` |
| `EmitModelPlan.cpp` | 删除 `profileId`，4 个调用点改调 helper（含 `plan_hash`）。**连带修正一处误称**：`plan_hash` 的计算原本也走 `profileId`，名字与语义不符 |
| `FuseLinalgEpilogue.cpp` | 保留 `fusionProfileId`（它还负责拼 key 与正号掩码），只把哈希内核换成 helper；`& 0x7fffffffffffffff` 保留 |

**验收**：`grep -rn "fnv\|14695981039346656037\|1099511628211" lib tools include` **归零**。

**不动**（不同语义的另外两个哈希）：
`test/Numerical/support/performance_test_support.cpp` 的 `performance_input_hash`
（浮点位模式）、`test/Native/check_perf_attribution.py` 的 python `fnv1a64`
（合成 fixture 生成器，自产自校）。

identity 变值证据与「停止跨期 join」见 `identity-changes.md` §1。

---

## 顺带修复：存量编译告警（policy §7）

门禁要求「非本次产生的问题也须修复」。下列告警在**未改动的代码**里，
按机械方式修掉，**不改行为**：

| 位置 | 告警 | 修法 |
|---|---|---|
| `PackStaticMatmulNCNN.cpp` | `global` set but not used | 加 `[[maybe_unused]]` |
| `EmitModelPlan.cpp` `assembleRoot` | `-Wpessimizing-move`：`return std::move(root)` 阻碍 NRVO | `return root;` |
| `EmitModelPlan.cpp` `collectFusionRecords` / `collectCopyRecords` | unused parameter `module` | `[[maybe_unused]]` |
| `InstrumentNCNNProfile.cpp` `isWholeBufferView` | `-Wsign-compare`（`ArrayRef::size()` vs `getRank()`）×3 | 显式 `static_cast<std::int64_t>` |
| `NCNNToTosa.cpp` attention ledger | gcc 对 `std::optional` 三元的 `-Wmaybe-uninitialized` 误报 | 改成条件赋值，并注明是误报规避 |

修完后项目代码编译告警归零（此前 P2 门禁只验「0 错误」，未验「0 告警」）。

---

## 不做

见 `backlog.md`。最重要的一条：

**T-A2 APFloat→float/memcpy 不做** —— 用户判定**会影响交叉编译**。
该项是 roadmap 声称的 P3「唯一带性能收益的项」，剔除后本阶段
**不承诺任何编译期内存或性能收益**；roadmap 的 P3 阶段行
「编译内存 −80%（大模型）」不兑现，其验收证据（`pack_bytes` 不变、
bit-identical packed 常量、wall/RSS 对比）一并不做。
