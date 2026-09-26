# P3 LLVM 库替换 · 阶段归档（2026-09-26）

## 结论

P3 三项全部落地：**T-A1** 字符串分派收敛到 `llvm::StringSwitch`（13 处）、
**T-A3** 常量匹配末端收敛到 `m_Constant`、**T-S5** 三处手写 FNV-1a 合并为
`ncnn_mlir::stableHash64`（`llvm::xxHash64`）。

**T-A2（APFloat→float/memcpy）按用户指令不做**（影响交叉编译）。这是 roadmap
声称的本阶段「唯一带性能收益的项」，剔除后 **P3 不承诺任何编译期内存或性能收益**，
只剩语义收敛。roadmap 的 P3 阶段行「编译内存 −80%（大模型）」**不兑现**。

**T-A1 / T-A3 是零行为变化的等价改写，已证**：10 个模型 × 7 种产物
（`.plan.json` / `.h` / manifest / `.so` / 三份 IR dump）逐字节对照，
归一化后 **0 行差异**（`plan-before-after.diff` 为空）。

**T-S5 是 identity 变化**：`plan_hash` / `build_identity` / `*.profile_id`
全量变值（实测 `12677497241874684592 → 1771926188015674330`），
**跨期 plan-hash join 从 P3 边界起停止**。详见 `identity-changes.md`。

门禁两次全绿（T-A1/T-A3 后一次、T-S5 后一次），每次都是
全新 `/tmp` stage 构建 + 三个 numerical target + 全量 CTest **456/456 通过、0 失败**
（2 个既有 upstream INT8 skip 与 P2 基线一致），外加全局 `format_check` 与 `tidy`。

---

## 交付物

```
docs/refactor/p3-2026-09-26/
  PLAN.md                 # 执行计划
  README.md               # 本文件
  task-diffs.md           # 三任务改动摘要
  verification.log        # 门禁与对照输出
  identity-changes.md     # T-S5 identity 变化 + T-A3 加宽记录
  plan-before-after.diff  # 归一化后逐字节对照（空 = 全一致）
  emit-plans.sh           # 采证脚本
  backlog.md              # 遗留问题与重启前置
```

代码侧新增 `include/ncnn-mlir/Support/StableHash.hpp`；改动 11 个文件，
+212 / −164（不含归档）。

---

## 与 roadmap 的偏离（七处，均已披露）

| # | roadmap 说法 | 实测 | 处置 |
|---|---|---|---|
| 1 | T-A2 是「唯一带性能收益的项」；阶段行「编译内存 −80%」 | **T-A2 不做**（用户判定影响交叉编译） | 本阶段**无性能收益承诺**；进 `backlog.md` |
| 2 | T-S5 有**两处**手写 FNV-1a | **三处**（`FuseLinalgEpilogue.cpp` 是第三份） | 三处锁步合并 |
| 3 | T-S5 要「更新 `perf_attribution_report.py` 的 identity 校验白名单」 | 该文件**无任何 whitelist** | 无需改动；勘误 |
| 4 | T-A1 验收 `grep -c 'StringSwitch' lib tools ≥ 20` | 按行计数，需收 13 处才达线 | 实测 **20**，达标；未为凑数改写单点开关 |
| 5 | T-A1 只点名 `StrategyNCNN.cpp:167` + `ncnn-compile.cpp` 26 处 | `parseStrategy` 在 `:145-159`；同类链另在 4 个文件 | 按「单键→单值」收 13 处，另 4 处按硬否决移出 |
| 6 | T-A1 示例 `.Default(ConvStrategy::Invalid)` | enum 只有 `Auto/Gemm/Conv/Winograd` | 不新增枚举，保持 `std::optional` |
| 7 | T-S1 的 helper 穿透 `extract_slice`/`from_elements` | P1 已**收窄**为 cast/collapse/expand 三项 | T-A3 不扩大穿透集合 |

另：T-A1 例子里的「两段式」改造与 roadmap 的直接 `.Case` 形式不同——
`g_vector_mode` / `g_vector_math` 的分支体带副作用（设 lane、探测库、发诊断），
必须先解析成枚举再 `switch`，`StringSwitch` 单独表达不了。

---

## 改写手法（可复用）

1. **校验链**：`StringSwitch<bool>(x).Cases(...).Default(false)` 做成员判定，
   **外层**再 `return fail(原来那句报错)`。
   **绝不写 `.Default(fail(msg))`** —— `Default(T)` 按值求值，合法输入也会报错。
2. **分派链**：先 `StringSwitch<std::optional<Enum>>` 解析，`!e` 时 `fail`；
   再对 `*e` 用真 `switch` 执行语句体。副作用留在 `switch` 里，报错串留在外面。
3. **哈希收敛**：算法只在 `Support/StableHash.hpp` 一处；调用点只负责拼 key。
   跨 pass 的「同串同哈希」不变量要写进 helper 的注释——它是 load-bearing 的，
   破坏它不会报错，只会让归因静默丢数据。
4. **逐字节对照**：IR dump 的 `loc(...)` 嵌入编译期随机临时目录
   `/tmp/ncnn-compile-XXXXXX`，直接 `diff -ru` 会出噪声。
   `.plan.json` / `.h` / manifest / `.so` **不含**该 token，可直接逐字节比对；
   IR dump 先把 token 归一再比。脚本见 `emit-plans.sh`。

---

## 三处 roadmap 未记的硬约束（执行中发现）

1. **`grep -c 'StringSwitch'` 按行计数**，一个 `StringSwitch` 表达式只出现一次词，
   所以「≥20」实际要求收十几处站点，不是「20 个 if 链」。
2. **`llvm::xxHash64(StringRef)` 无 seed 参数**（LLVM 21，
   `/usr/include/llvm-21/llvm/Support/xxhash.h`）。想做 per-process 区分得自己加盐。
3. **`m_Constant` 比 `dyn_cast<arith::ConstantOp>` 更宽**：接受任何
   `ConstantLike` op（`ncnn.const` / `tosa.const`）。产品流水线中不可达，
   但对独立 `ncnn-mlir-opt` 是真实加宽，见 `identity-changes.md` §2。

---

## 新增守卫

P3 **没有新增测试**。理由与缺口：

- T-A1 是等价改写，报错串逐字不动，由既有 CLI 护栏
  （`check_ncnn_compile.py`、`check_int8_target.py` 的 `"must be one of"` 断言）
  与 lit `INVALID:` 检查覆盖；
- T-A3 由 P1 既有 lit + 逐字节对照覆盖；
- T-S5 的护栏**全部算法无关**（相对断言 / 字符串相等 / 累加站点白名单），
  逐条核对见 `identity-changes.md` §1.4。

**已知缺口**（记入 `backlog.md` §7）：
- `findConstantElements` 的 `from_elements` 负向用例缺失（只有 `extract_slice`）；
- `m_Constant` 的加宽面无正向用例。

---

## 遗留问题

见 `backlog.md` 全文。要点：

1. **T-A2 不做**，重启须**先解决交叉编译顾虑**。
2. **第三套常量查找仍在**：`NCNNToTosa.cpp` 的 `getConstantTensorElements`
   （接受 `tosa.const`，不穿透视图）。P1 的 grep 验收用
   `getConstantElements|findConstant` 两个名字，漏掉了它。
3. **复合键分派未收**（`EmitModelPlan` 的 `requested_policy_status` 等）——
   `StringSwitch` 表达不了，硬套是顺手重构。
4. 两个测试缺口（见上）。

---

## 不做（沿用 roadmap §8 并按本期收窄）

- **不做 T-A2** —— 用户判定影响交叉编译；不重做其验收证据
- **不做 Winograd `transformWeight` 的 float 移植**（FMA 收缩会改权重常量比特）
- **不做 `NCNNToTosa` 量化路径改写**（循环体是真量化舍入算术）
- **不动 `getConstantTensorElements`**（进 backlog）
- **不动 `performance_input_hash`** / `check_perf_attribution.py` 的 python `fnv1a64`
- **不为凑 `grep -c` 改写单点布尔开关**
- **不改复合键 / 带状态守卫的分派**
- **不改任何性能默认值**
- **不引新依赖**
- **不做「顺便优化」**
- **不复跑 P27 性能基线**（P3 不属 roadmap §6 要求复跑的 P2/P5 收尾点）
- **不改 roadmap 正文**（进度只写本归档；`compiler/docs/code-quality-refactor-roadmap.md`
  的 §12 进度表保持空白，与 P1/P2 一致）
