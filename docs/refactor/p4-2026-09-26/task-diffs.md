# P4 任务改动摘要

> 只记「改了什么、手法、验收」。与 roadmap 的偏离统一见 `README.md`。

---

## T-C5 tile-loop 公共层

**落点**：`lib/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.cpp`

**抽取的公共层**（匿名 namespace，与 pass 类同 TU）：

| 符号 | 职责 | 消灭的重复 |
|---|---|---|
| `struct TileShape` | 行/列/归约/行块/列块五元 | 三处预计算 |
| `TileBlockEmitter` | 块体回调（初始化、K 归约、写回） | — |
| `materializeIndices` | `base + i` 物化（i==0 复用 base） | 3 份 rowIndices + 1 份 columnIndices |
| `readTileAccumulators` | 向量累加器界内读入 | kernelize / batch 各 1 份 |
| `writeTileAccumulators` | 向量累加器界内写回 | kernelize / batch 各 1 份 |
| `emitTileLoopNest` | 行块 × 列块嵌套（满块循环 + 余块） | 3 份 `emitRowBlock` + 满/余行循环 |

**接线**：`kernelize` / `kernelizeBatchMatmul` / `kernelizeInt8RowDot` 三处改为
调 `emitTileLoopNest`；`zero`/`one` 由调用方创建后传入（块体也要用，避免重复
物化常量）。

**不变量刻意不接管**：K 循环（三种形态）、累加器初始化与写回（向量 vs 标量 /
面板 vs 全量 / 经 requant）留在块体回调里——这是三个内核的真实差异，见
`README.md` 偏离 2 的差异表。

**验收**：lit `MatmulKernelNCNN/*` 8/8 全绿；10 模型产物逐字节一致。
`emitRowBlock` 3→1、`fullColumnBlocks > 1` 3→1、手工 rowIndices 3→0。

---

## T-C6 拆 `NCNNToTosa.cpp`

**手法**：结构发现（38 个 pattern 类 + 49 个 helper 的精确行界）→ 按算子族
搬进独立 TU → 共享 helper 收敛到 `tosa_lowering` 命名空间 → 每族一个
`populate*Patterns` 入口 → pass 壳只留 TypeConverter + 注册 + ConversionTarget。

**新增**：

| 文件 | 内容 | 行数 |
|---|---|---:|
| `include/.../NCNNToTosaPatterns.hpp` | 12 个 `populate*Patterns` 声明 | 64 |
| `include/.../TosaLoweringUtils.hpp` | 共享 helper 声明 | 196 |
| `TosaLoweringUtils.cpp` | 共享 helper 实现 | 848 |
| `ConvToTosa.cpp` | conv2d + depthwise | 959 |
| `DeconvToTosa.cpp` | 转置卷积 | 342 |
| `PoolingToTosa.cpp` | pooling | 604 |
| `AttentionToTosa.cpp` | softmax / layernorm / embed / MHA | 1073 |
| `SDPAToTosa.cpp` | sdpa | 354 |
| `QuantToTosa.cpp` | batchnorm / quantize / dequantize / requantize / cast / zero-point | 358 |
| `GemmToTosa.cpp` | gemm + 行标度反量化 | 298 |
| `ElementwiseToTosa.cpp` | relu / sigmoid / hard-* / swish / tanh / GELU / dropout / binary / unary | 516 |
| `DetectionToTosa.cpp` | detection_output（含 `.inc`） | 75 |
| `LayoutToTosa.cpp` | split / concat / padding / interp / grid_sample / reshape / squeeze / expand / permute / slice / shuffle | 1145 |
| `ReduceToTosa.cpp` | reduction / inner_product | 364 |
| `NCNNToTosa.cpp` | pass 壳 | 204 |

**手法细节**：

* 共享 helper 进 `namespace mlir::ncnn::tosa_lowering`，各族 TU 用**逐名
  `using tosa_lowering::X;`** 导入（clang-tidy 的 `google-build-using-namespace`
  禁止 `using namespace`）。导入名按该族实际调用点生成，不是全量。
* 9 个**单族专用** helper（`getPoolPadding`、`getHWCFType`、
  `computeGemmRowScales`、`dequantizeGemmAccumulator`、`convertSignedI8ToF32`、
  `matchBroadcastRank`、`getTensorElementCount`、`getDynamicSizes`、
  `appendAttentionSegmentRecords`）留在各自族 TU 的匿名 namespace，不进共享头。
* 默认参数留头文件、从定义剥掉（C++ 不允许两处重复给默认值）。
* `DetectionOutputLowering.inc` 原样保留，由 `DetectionToTosa.cpp` 包含。

**验收**：每文件 ≤1200（最大 1145）；Conversion lit 48/48；产物逐字节一致。

---

## T-C7 Winograd 独立成模块

**新增**：

| 文件 | 内容 | 行数 |
|---|---|---:|
| `include/ncnn-mlir/Support/Winograd63.hpp` | `kG`/`kBt`/`kAt`/`kAlpha`/`kTile`/`kBatch` + `eligible` + `transformWeight` 声明 | 84 |
| `include/ncnn-mlir/Transforms/Winograd63NCNN/Winograd63NCNN.hpp` | `rewriteWinograd` 声明 | 30 |
| `lib/Transforms/Winograd63NCNN/Winograd63NCNN.cpp` | `transformWeight` + `rewriteWinograd` | 600 |
| `lib/Transforms/Winograd63NCNN/CMakeLists.txt` | 新库 `Winograd63NCNN` | 16 |

**改动**：

* `StrategyNCNN.cpp` **1780 → 1109 行**（−671）：删除 `winograd63` namespace、
  `transformWeight`、`rewriteWinograd` 与本地 `copyConvContract` /
  `annotateConvContract`；分派点改为
  `winograd63::rewriteWinograd(rewriter, convolution, kMaxStrategyElements)`。
* `annotateConvContract` / `copyConvContract` 收进
  `KernelContract.hpp::contract`（Winograd 与 Strategy 需要共享，
  且这是契约 helper 的正确归宿——属性列表紧挨常量定义）。
* 策略预算常量 `kMaxStrategyElements` **不搬**：由调用方传参
  （`maxElements`），Winograd 模块不持有策略常量。
* `winograd63::eligible` 仍在 `Winograd63.hpp`（inline），StrategyNCNN 的
  分派与 Winograd 发射共用同一判据，避免「选了却不改写」。

**保留原样**：`transformWeight` 的 APFloat + 显式 RN-even 舍入（P3 backlog §2
的理由：改 float 会因 `-ffp-contract` 改变权重常量比特）。

**验收**：`StrategyNCNN.cpp` −671 行（目标「约 600」达成）；
lit `winograd63.mlir` 全绿；`resnet18_winograd` 数值 golden 不变。

---

## T-M1 补薄弱 pass 的 lit

| pass | 前 | 后 | 新增文件 | 覆盖的新判别点 |
|---|---:|---:|---|---|
| `TileMatmulForall` | 1 | **5** | 4 | `nested-skip`（`isTopLevel` 三种外层循环）、`prime-extents`（因子选取与两维都不可切）、`tile-position`（size 向量按迭代位对应 M/N）、`transpose-b-columns`（`MatmulTransposeBOp` 钉住列维） |
| `PackStaticMatmulNCNN` | 4 | **7** | 3 | `panel-width-guard`（`pack-n!=16`、`columns%32`）、`min-shape-gates`（`min-n`/`min-k`/单行/无 tile 因子）、`int8-gates`（int8 小形状与 pack-n 口径**与 f32 不同**） |
| `RewriteLinalgCopies` | 2 | **6** | 4 | `eliminate-same-value`（同值消除）、`alias-guard`（同根/无根/同全局三种不相交失败）、`non-contiguous`（步距/窄内维）、`vectorized-ranks`（rank 1/2/3 主块+尾块） |
| `FoldLinalgConstantTranspose` | 1 | **4** | 3 | `non-square`（非方形逐元素重排）、`constant-through-cast`（**负向**：不穿透视图）、`reject-paths`（i1 不可折叠、splat 快路径、运行期值） |

**手法**：每个用例先跑真实 pass 取输出，再把 CHECK 对齐实际行为；对
`FoldLinalgConstantTranspose` 的非方形折叠用脚本算出期望常量，只锚定首个
输出行 + 类型（避免整段嵌套字面量的转义脆弱性）。

**验收**：lit 192 → **206**，206/206 全绿。

---

## T-M2 pass 设计注释

按 roadmap 的四段式（**职责 / 不变量 / 顺序依赖 / 明确不做什么**）补齐 18 个
文件的文件头注释，风格沿用 `StrategyNCNN.cpp` 的 bounds 约束注释：

* `lib/Conversion/NCNNToTosa/`：`NCNNToTosa.cpp`、`TosaLoweringUtils.cpp`、
  11 个族 TU；
* `lib/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.cpp`；
* `lib/Transforms/EmitModelPlan/EmitModelPlan.cpp`；
* `lib/Transforms/GenerateCAPI/`：4 个 TU + `GenerateCAPIInternal.hpp`；
* `lib/Transforms/PackStaticMatmulNCNN/PackStaticMatmulNCNN.cpp`；
* `lib/Transforms/InstrumentNCNNProfile/InstrumentNCNNProfile.cpp`；
* `lib/Transforms/Winograd63NCNN/Winograd63NCNN.cpp`。

注释写的是**约束**（为什么不能改、顺序为什么必须这样），不复述代码表面行为。
