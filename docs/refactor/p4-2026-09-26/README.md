# P4 结构拆分 + 测试补强 · 阶段归档（2026-09-26）

> **阶段**：P4（T-C5 / T-C6 / T-C7 / T-M1 / T-M2）
> **性质**：结构拆分 + 测试补强，**产物逐字节不变**
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.4
> **前置**：P1 `9ed5e9d` / P2 `6cf3438` / P3 `3d1c0fc` 已落库

---

## 结论

五个任务全部落地。**产物逐字节不变**：10 个模型 × 7 种产物
（`.plan.json` / `.h` / manifest / `.so` / 三份 IR dump）重构前后对照，
归一化临时目录名后 **0 行差异**（`plan-before-after.diff`）。

| 任务 | 目标 | 拆分前 | 拆分后 | 验收 |
|---|---|---:|---:|---|
| **T-C5** | tile-loop 公共层 | 三份**逐字重复**的行块×列块外壳（108 行）+ 三份 rowIndices | 单一 `emitTileLoopNest` / `materializeIndices` / 累加器读写 | ✓（IR 等价）；**行数不降**，见偏离 2 |
| **T-C6** | 拆 `NCNNToTosa.cpp` | **6160 行**单 TU | 13 个 TU + 2 头，**最大 1145 行** | ✓ 每文件 ≤1200；Conversion lit 48/48 |
| **T-C7** | Winograd 独立 | `StrategyNCNN.cpp` **1780 行** | **1109 行**（−671）；新增 `Winograd63NCNN` 模块 | ✓ 目标「减约 600 行」达成 |
| **T-M1** | 补薄弱 pass 的 lit | 8 个文件 | **22 个**（+14） | ✓ 四目录达标 5/7/6/4 |
| **T-M2** | pass 设计注释 | 无文件级设计注释 | 18 个文件补齐四段式 | ✓ |

**lit：192 → 206（+14），206/206 全绿。**
**门禁**：`format_check` + `tidy` 全绿（tidy 报出的 9 处问题均在本次新代码，已修）；
全新 `/tmp` stage 构建 + 三个 numerical target + 全量 CTest。

---

## 交付物

```
docs/refactor/p4-2026-09-26/
  PLAN.md                 # 执行计划 + 与 roadmap 的偏离决策
  README.md               # 本文件
  task-diffs.md           # 五任务改动摘要
  verification.log        # format_check / tidy / CTest / 逐字节对照
  plan-before-after.diff  # 逐字节对照结果（归一化后为空）
  identity-changes.md     # identity 变化记录（结论：无）
  backlog.md              # 遗留问题与重启前置
```

代码侧新增 `include/ncnn-mlir/Support/Winograd63.hpp`、
`include/ncnn-mlir/Transforms/Winograd63NCNN/`、
`include/ncnn-mlir/Conversion/NCNNToTosa/{NCNNToTosaPatterns,TosaLoweringUtils}.hpp`；
新增 `lib/Transforms/Winograd63NCNN/`、`lib/Conversion/NCNNToTosa/` 11 个族 TU。

---

## 与 roadmap 的偏离（六处，均已披露）

### 1. T-C5 的「三个 kernel」点名有误——第三个不是 `vectorizeRowGeneric`

roadmap 称三套近似重复的「行块 × 列块 × K 归约」是 `kernelize`（421 行）、
`kernelizeInt8RowDot`（311 行）、`vectorizeRowGeneric`（261 行）。

**实测 `vectorizeRowGeneric` 根本不是这套嵌套**：它没有行分块、没有 K 归约、
没有累加器，尾块是**逐元素标量循环**而不是窄向量块，外层还是
`scf.parallel` 而非 `scf.for`。把它塞进 `TileShape{rows, columns, depth,...}`
需要 `depth=0`、`tileRows=1`、外加三个开关——那是强行凑形状，不是抽象。

**真正的逐字重复是三份**：`kernelize`（f32）、**`kernelizeBatchMatmul`**
（f32 batch，roadmap 未点名）、`kernelizeInt8RowDot`（int8）。三者的
`emitRowBlock` + 满/余行循环 + `fullColumnBlocks/tailColumns/fullRowExtent`
预计算是**同一段文本**（35 行 ×3），`rowIndices` 物化也是三份逐字相同。

**处置**：`emitTileLoopNest` / `materializeIndices` / `readTileAccumulators`
/ `writeTileAccumulators` 覆盖这三个内核；`vectorizeRowGeneric` 移出范围并
在 `backlog.md` 记录。

### 2. T-C5 的「减约 300 行」不成立

roadmap 的 −300 行建立在「三个 kernel 只提供 body」之上，即 helper 连
累加器初始化、K 循环、写回都接管。这要求三个 `emitTileBlock` 共享同形，实测
不成立：

| 差异点 | kernelize (f32) | kernelizeBatchMatmul | kernelizeInt8RowDot |
|---|---|---|---|
| 累加器初始化 | 向量 `transfer_read` | 向量 `transfer_read`（面板） | **标量 `memref.load`** |
| K 循环形态 | **三种**（平坦 / packed-K 分块 / im2col 窗口） | 一种平坦 | 一种（**从 kStart 起**） |
| 归约更新 | `vector.fma` | `vector.fma` | **标量 muli+addi**（为 vpmaddwd） |
| 写回 | 向量 `transfer_write` | 向量 `transfer_write`（面板） | 标量 `store` / **经 requant** |
| 额外 | packed-B 面板地址提升、im2col 融合 | `scf.forall(b)` + `[b]` 子视图 | VNNI 前导、`columnIndices`、requant epilogue |

把这些塞进一个 `BodyEmitter(kIndex, accs)` 必须改内核形态 → 违反「不改语义 /
产物 IR 不变」。

**实测**：`MatmulKernelNCNN.cpp` **2427 → 2455 行**，其中 **+28 是 T-M2 新增的
设计注释**，代码本身**净行数 0**。但重复确实消灭了：

| 重复项 | 拆分前 | 拆分后 |
|---|---:|---:|
| `emitRowBlock` + 满/余行循环 | 3 份（108 行） | 1 份 |
| `rowIndices` 物化 | 3 份 | 0（走 `materializeIndices`） |
| 向量累加器读 / 写 | 各 2 份 | 各 1 份 |
| `fullColumnBlocks > 1` 判定 | 3 处 | 1 处 |

行数持平的原因是共享 helper 的文档开销约等于三份重复的行数——**去重换来的
是「改一处生效三处」，不是行数**。这是对 roadmap 验收数字的直接否定，按
P2 先例以「验收标准优先于改动点」取舍后如实记录。

### 3. T-C6 的 6 个文件装不下，实际 13 个 TU

roadmap 给出 6 个文件（5 个族 + `TosaLoweringUtils.hpp`）。但 pattern 代码
就有约 4870 行，`4870 / 1200 ≈ 4.1` 个文件，加 helper 实现与 pass 壳
**下限就是 8 个**。且 conv 族三个大 rewriter（424+430+254 行）单文件即 1215 行，
**超出「每文件 ≤ 1200」验收**。

按「验收标准优先于改动点」，按算子族拆成 11 个族 TU + helper `.cpp` + pass 壳：

```
NCNNToTosa.cpp      204  TosaLoweringUtils.cpp   848  TosaLoweringUtils.hpp  196
ConvToTosa.cpp      959  DeconvToTosa.cpp        342  PoolingToTosa.cpp      604
AttentionToTosa.cpp 1073 SDPAToTosa.cpp          354  QuantToTosa.cpp        358
GemmToTosa.cpp      298  ElementwiseToTosa.cpp   516  DetectionToTosa.cpp     75
LayoutToTosa.cpp   1145  ReduceToTosa.cpp        364
```

最大 **1145 行**（`LayoutToTosa.cpp`），全部 ≤1200。
roadmap 的 5 个族名全部保留（`ConvToTosa` / `AttentionToTosa` / `QuantToTosa`
/ `ElementwiseToTosa` / `DetectionToTosa`），另加 6 个族以装下 roadmap 未覆盖的
算子（reshape/layout、GEMM、SDPA、Pooling、Reduce、Deconv）。

### 4. T-M1 的 PackStaticMatmulNCNN 基线数字过期

roadmap 记「现有 3」，实测 **4**（`pack.mlir` / `pack-int8.mlir` /
`pack-int8-budget.mlir` / `constant-through-cast.mlir`）。因此「1→5、3→7、
2→6、1→4」的表格口径下新增应为 14，而验收写「新增 16–18 个 lit」。

**处置**：以表格的目标文件数为准（5/7/6/4 全部达成，新增 **14** 个文件），
不为凑数拆出无判别力的用例。roadmap 的「16–18」与自身表格不一致，如实披露。

### 5. T-M1 的 FoldLinalgConstantTranspose「常量穿透」用例不存在

roadmap 把「常量穿透」列为重点用例，预期是正向（被 `tensor.cast` 包着的常量
仍折叠）。**实测 `FoldLinalgConstantTranspose` 根本不穿透视图**——它直接
`getDefiningOp<arith::ConstantOp>()`，没有接 P1 T-S1 的 `findConstantElements`。

**处置**：落成**负向用例** `constant-through-cast.mlir`，锁定「当前不穿透」这一
契约，并在 `backlog.md` 记录收敛缺口。**不在本期改 pass**：那会改变产物 IR，
与「产物逐字节不变」冲突。

### 6. `copyConvContract` 的两份实现未收敛

P1 期间发现 `MatmulKernelNCNN.cpp` 另有一份更窄的本地 `copyConvContract`
（13 个属性，不含布局岛）与 `StrategyNCNN.cpp` 的 22 属性版本并存。本次 T-C7
把后者收敛进 `KernelContract.hpp`（Winograd 与 Strategy 需要共享），但
**未动 MatmulKernelNCNN 那份**——合并属性集是行为变化。已在 `backlog.md` 点名。

---

## 值得保留的设计（重构后仍在）

1. **`KernelContract.hpp` 继续当单一事实源** —— T-C7 把 `annotateConvContract`
   / `copyConvContract` 收进它，比留在 `StrategyNCNN.cpp` 的匿名 namespace 更实。
2. **`transformWeight` 保留 APFloat 算术** —— 与 P3 backlog §2 一致，改 float 会
   因 `-ffp-contract` 改变权重常量比特。
3. **`emitTileLoopNest` 不接管 K 循环** —— 三种 K 形态是内核的真实差异，
   抽象它只会把差异藏进回调参数。
4. **pattern 按 op 名互不重叠** —— 拆 TU 后注册顺序不选择竞争 pattern，
   这是「拆完 IR 不变」的前提，已写进 `NCNNToTosaPatterns.hpp` 的文件注释。
