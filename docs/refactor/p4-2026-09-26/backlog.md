# P4 遗留问题 / Backlog

本文件记录 P4 **明确不做**的条目及其重启前置条件。条目不是「忘了」，是判定后
留在这里，避免以后被当成可随手拾起的优化。

---

## 1. `FoldLinalgConstantTranspose` 未接统一常量查找（**T-M1 勘误落点**）

**状态**：不做（本期）。

**问题**：该 pass 用 `transpose.getInput().getDefiningOp<arith::ConstantOp>()`
直接取常量，**没有走** P1 T-S1 收敛出的 `ncnn_mlir::findConstantElements`。
因此被 `tensor.cast` / `tensor.collapse_shape` / `tensor.expand_shape` 包着的
常量不会被折叠——与 roadmap T-M1 列的「常量穿透」用例预期相反。

**本期处置**：落成负向用例
`test/Transforms/FoldLinalgConstantTranspose/constant-through-cast.mlir`，
锁定「当前不穿透」这一契约。

**重启前置**：把取常量改为 `findConstantElements(transpose.getInput())`。
注意这是**行为放宽**（开始折叠以前跳过的常量）→ 会改变产物 IR →
必须重跑数值 golden 并把上述负向用例反转为正向。**不要与「产物逐字节不变」
的重构混在同一期做。**

---

## 2. `copyConvContract` 两份实现未收敛

**状态**：不做（本期）。

**问题**：属性集不同。

| 实现 | 位置 | 属性数 | 是否含布局岛 |
|---|---|---:|---|
| 完整版 | `include/ncnn-mlir/Support/KernelContract.hpp::contract::copyConvContract` | 22 | 是 |
| 窄版 | `lib/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.cpp` 匿名 namespace | 13 | 否 |

T-C7 把完整版收进 `KernelContract.hpp`（Winograd 与 Strategy 需要共享），
窄版原样保留。

**重启前置**：判定窄版是**有意**还是**漂移**。若为漂移，合并到完整版是行为
变化（matmul 内核开始复制布局岛属性）→ 会改 plan 的 contracts 段 → identity
可能变化。若为有意，应改名（如 `copyMatmulContract`）并在两处写明属性集差异。

---

## 3. `vectorizeRowGeneric` 的 full-block / tail 幂等未抽公共层

**状态**：不做（本期）。

**问题**：roadmap T-C5 把它列为三个「行块×列块×K 归约」之一，实测它不是
（无行分块、无 K、无累加器、尾块是标量逐元素、外层是 `scf.parallel`）。
它与 `emitRowBlock` 只共享「满块循环 + 余块」的**惯用形**，但：

* 步距语义不同（`step=1` 再乘宽 vs `step=accColumns`）；
* 尾块契约不同（标量逐元素 vs 窄向量块）。

硬抽需要 `scalarTail` / `strideMode` 之类的开关，收益小于噪音。

**重启前置**：若第三个相似形态出现（例如第二个逐元素向量化器），再抽
`emitBlockedRangeWithTail`；不要为两个调用点写参数化壳。

---

## 4. T-C5 的「减约 300 行」目标

**状态**：判定不可达，见 `README.md` 偏离 2。

**重启前置**：只有把三个 `emitTileBlock` 的差异（累加器初始化 / 三种 K 形态 /
标量 vs 向量归约 / 写回）也抽象掉才能逼近，而那必须改内核形态。
**除非愿意放弃「产物 IR 不变」，否则不要再提这个数字。**

---

## 5. 全库 20 处项目源码编译警告

**状态**：**本期已修**（clang-format / clang-tidy 要求零警告，且 tidy 的
`WarningsAsErrors: '*'` 会拦下同类问题）。

但下列历史 API 用法仍建议后续收敛（已无警告，属代码卫生）：

| 类别 | 站点数 | 说明 |
|---|---:|---|
| `applyPatternsAndFoldGreedily` | 6 | MLIR 21 已改名 `applyPatternsGreedily` |
| `OpFoldResult::get<T>()` / `is<T>()` | 8 | 应改 `cast` / `isa` |
| `Manifest` 缺字段初始化 | 2 | `ncnn-compile.cpp` |

**说明**：本轮 tidy 只报出本次新代码的 9 处问题（已修）；上表是**编译器**
`-Wdeprecated-declarations` 的历史站点，与 clang-tidy 是两套检查。
按 standards §11「零编译警告」应在后续一期统一清理，**不要混进结构拆分**。

---

## 6. T-M1 的「新增 16–18 个 lit」口径

**状态**：以表格目标文件数为准（新增 14 个文件，四目录 5/7/6/4 全部达成）。

**重启前置**：若确实要 16–18，需先修正 roadmap 的基线（PackStaticMatmulNCNN
实为 4 不是 3）与表格口径，否则「为凑数写测试」违反 roadmap §8
「不为覆盖率写测试」。
