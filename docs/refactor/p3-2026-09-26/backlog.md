# P3 遗留问题 / Backlog

本文件记录 P3 **明确不做**的条目及其重启前置条件。条目不是「忘了」，是判定后
留在这里，避免以后被当成可随手拾起的优化。

---

## 1. T-A2：APFloat → float / memcpy（**本期最大范围偏离**）

**状态**：不做。

**原因**：用户判定**会影响交叉编译**。判定即结论，本期不展开机制讨论。

**后果**：roadmap 把 T-A2 称为 P3「**唯一带性能收益的项**」，并把阶段行写成
「编译内存 −80%（大模型）」。剔除后 **P3 不承诺任何编译期内存或性能收益**，
只剩语义收敛（字符串分派、常量匹配、稳定哈希三处收敛到 LLVM 库实现）。
roadmap 的 T-A2 验收证据（`pack_bytes` 不变、bit-identical packed 常量、
编译 wall/RSS 前后对比）**一并不做**。

**原改动点**（供重启时参考）：
- `lib/Transforms/PackStaticMatmulNCNN/PackStaticMatmulNCNN.cpp` 的
  `SmallVector<APFloat> source` + `SmallVector<APFloat> packed` 双份物化
- `lib/Transforms/FoldNCNNBatchNorm/FoldNCNNBatchNorm.cpp` 的 `scaled` / `shifted`
- 其余 `getValues<APFloat>()` 迭代点

**重启前置**：**必须先解决交叉编译顾虑**。原方案依赖 `getRawData()` /
`reinterpret_cast<const float*>` 这类宿主位模式直取；在未澄清该路径对交叉编译
目标的影响之前，不得重做。重启时须重新盘点 MLIR 21 的
`getValues<T>()` 返回类型（返回迭代器区间，不是 `ArrayRef`）。

---

## 2. Winograd `transformWeight` 的 float 移植

**状态**：不做。

**原因**：该处是**真 APFloat 算术**（`multiply`/`add` + 显式 RN-even 舍入）。
改写成 float 运算后 clang 默认 `-ffp-contract=fast` 会把 `a*b+c` 收缩成
`fmuladd`，**改变 Winograd 权重常量的比特**，违反「产物逐字节不变」。

**重启前置**：给该 TU 加 `-ffp-contract=off`，并另做**比特恒等证明**。
Winograd 是 opt-in（ncnn 对主力模型不选 F(6,3)），收益面窄。

---

## 3. `NCNNToTosa` 量化路径的 `SmallVector<APInt>`

**状态**：不做。

**原因**：真正可省的是输出 `SmallVector<APInt>`（8+ B/元素 vs i8 的 1 B），
但那不是 roadmap 的 APFloat 问题；且循环体是**真量化舍入算术**
（`roundToIntegral`/`compare`/`clamp`），不能只换容器。

**重启前置**：独立评估，与 T-A2 的交叉编译顾虑分开论证。

---

## 4. `PackStaticMatmulNCNN` int8 路径的 `SmallVector<APInt>` 双份物化

**状态**：不做。

**原因**：同形问题，不属本阶段口径。

---

## 5. `NCNNToTosa.cpp` 的 `getConstantTensorElements`（第三套常量查找）

**状态**：不做。

**问题**：`lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` 里另有一个本地常量查找
`getConstantTensorElements`（额外接受 `tosa::ConstOp`，且**不**穿透视图）。
P1 的 T-S1 grep 验收用的是 `getConstantElements|findConstant` 两个名字，
**漏掉了它**，所以「全库只有一个常量查找」的断言当时就未成立。

**重启前置**：并入 `ncnn_mlir::findConstantElements`，或显式声明二者契约差异
（`tosa.const` 支持 vs 视图穿透）并写进 `ConstantFold.hpp`。

---

## 6. T-A1 中按硬否决条件移出的分派点

`llvm::StringSwitch` 只表达「单键 → 单值」。下列形态强行改写属「顺手重构」，
本期留原样：

| 位置 | 形态 | 移出原因 |
|---|---|---|
| `lib/Transforms/EmitModelPlan/EmitModelPlan.cpp` `requested_policy_status` 段 | `int8_kernel == "auto" && int8_target == …` | **复合键 + 双输出**（同时写 `requested_policy_status` 与 `requested_policy_fallback_reason`） |
| 同文件 `low_precision_hash_input` 的三元链 | 同上复合键 | 同上 |
| `lib/Support/Precision.cpp` `target_execution_profile` | 外层分派键是 `PrecisionMode` **枚举** | 不是字符串分派，`StringSwitch` 无意义 |
| `tools/ncnn-mlir-driver.cpp` input-dim-constraint 字段解析 | `name == "min" && !sawMinimum` | **带状态守卫**（拒绝重复字段），StringSwitch 会掩盖该语义 |

**若未来要收**：先把复合键显式建模（结构体/枚举），再谈分派形式。

---

## 7. 测试与文档缺口

- **`from_elements` 负向 lit 不存在**：`ConstantFold.hpp` 契约声明
  `tensor.from_elements` 不被穿透（成立，因为它是 view-walk 之外的 op），
  但现有 lit 只有 `constant-through-cast.mlir` 里的
  `does_not_cross_extract_slice`。若要锁死契约，应补
  `does_not_cross_from_elements` 用例。
- **`m_Constant` 的加宽未被测试覆盖**：见 `identity-changes.md` 的
  「T-A3 行为加宽」一节。
