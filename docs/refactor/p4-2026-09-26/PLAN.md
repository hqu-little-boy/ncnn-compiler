# P4：结构拆分 + 测试补强 — 执行计划

> **阶段**：P4 = T-C5 + T-C6 + T-C7 + T-M1 + T-M2
> **日期**：2026-09-26
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.4 / §6 / §7
> **前置**：P1 `9ed5e9d`、P2 `6cf3438`、P3 `3d1c0fc` 已落库
> **分支**：`p8-implementation`

---

## 0. Context

P18–P28 性能阶段已收官，主线是代码质量重构。roadmap 把重构拆成 5 阶段
21 任务，P1–P3 已提交。P4 是第四阶段：**结构拆分 + 测试补强**。

本阶段**不改任何语义**——这是与 P1（语义收敛）、P3（含 identity 变化的哈希
合并）的根本差别。因此护栏是**产物逐字节对照**，而不是数值预算。

### 本阶段实际交付（五项）

| 项 | 性质 | 行为影响 | 交付 |
|---|---|---|---|
| **T-C5** tile-loop 公共层 | 等价改写 | 零（已证） | 三份逐字重复的循环外壳 → 单一实现 |
| **T-C6** 拆 `NCNNToTosa.cpp` | 纯搬家 | 零（已证） | 6160 行 → 13 TU + 2 头 |
| **T-C7** Winograd 独立 | 纯搬家 | 零（已证） | `StrategyNCNN.cpp` −671 行 |
| **T-M1** 补薄弱 pass 的 lit | 只加测试 | 零 | +14 个 lit，192→206 |
| **T-M2** pass 设计注释 | 只加注释 | 零 | 18 个文件四段式设计注释 |

### 执行决策

| 决策 | 取值 | 理由 |
|---|---|---|
| 提交粒度 | **每阶段一笔 `refactor(ncnn)` 提交**（代码 + 归档） | 与 P1/P2/P3 一致 |
| 任务顺序 | T-C7 → T-C5 → T-C6 → T-M1 → T-M2 | T-C7 最自包含；T-C6 最大放中间；T-M1 依赖 T-C5 的行为锁定 |
| 逐字节对照 | **做** | 用户确认「产物 IR 不变」是 T-C6 验收；用 P3 的 `emit-plans.sh` 采 10 模型 |
| T-C5 的 −300 行 | **不强求** | 见 §2；强行凑会改内核形态 |
| T-C6 的 6 文件 | **扩到 13 TU** | 6 个装不下「每文件 ≤1200」 |
| T-M1 的「常量穿透」 | **落成负向用例** | 该行为不存在；改 pass 会破坏「产物不变」 |
| 不推送 | 只 commit，不 push | 用户约束 |
| 门禁失败 | 不提交；修复后**重新从全新 `/tmp` 目录**构建和测试 | 用户约束；policy §7 |

---

## 1. 任务分解

### T-C7 Winograd 独立成模块（先做）

1. `include/ncnn-mlir/Support/Winograd63.hpp`：`kG`/`kBt`/`kAt`/`kAlpha`/
   `kTile`/`kBatch` + `eligible`（inline）+ `transformWeight` 声明。
2. `include/ncnn-mlir/Transforms/Winograd63NCNN/Winograd63NCNN.hpp`：
   `bool rewriteWinograd(RewriterBase&, Conv2DNhwcHwcfOp, int64_t maxElements)`。
   策略预算由**调用方传参**，模块不持有策略常量。
3. `lib/Transforms/Winograd63NCNN/Winograd63NCNN.cpp`：`transformWeight` +
   `rewriteWinograd` 逐行搬运（成员函数 → 自由函数，去 2 空格缩进，
   `kMaxStrategyElements` → `maxElements`）。
4. `annotateConvContract` / `copyConvContract` 收进
   `KernelContract.hpp::contract`（两模块共享的前置）。
5. `StrategyNCNN.cpp` 删已搬走的代码，分派点改调自由函数。

### T-C5 tile-loop 公共层

1. `TileShape` + `TileBlockEmitter` + `emitTileLoopNest`：行块 × 列块外壳。
2. `materializeIndices` / `readTileAccumulators` / `writeTileAccumulators`：
   三份/两份逐字重复的积木。
3. `kernelize` / `kernelizeBatchMatmul` / `kernelizeInt8RowDot` 接线。
4. **不接管**：K 循环、累加器初始化、写回（三个内核的真实差异）。
5. **不含** `vectorizeRowGeneric`（不是这套嵌套，见 §2）。

### T-C6 拆 `NCNNToTosa.cpp`

1. 结构发现：38 个 pattern 类 + 49 个 helper 的精确行界 + 使用矩阵。
2. 共享 helper → `mlir::ncnn::tosa_lowering`；单族 helper 留本族 TU。
3. 按算子族拆 TU；每族 `populate*Patterns` 入口。
4. pass 壳只留 TypeConverter + 注册 + ConversionTarget。
5. **验收优先**：conv 族 1215 行超线 → 再拆出 `DeconvToTosa.cpp`。

### T-M1 补薄弱 pass 的 lit

四目录补到 5/7/6/4。每个用例先跑真实 pass 取输出，再对齐 CHECK。

### T-M2 pass 设计注释

四段式（职责 / 不变量 / 顺序依赖 / 明确不做什么）补 18 个文件。

---

## 2. 与 roadmap 的偏离（决策前即判定）

| # | roadmap 说法 | 判定 | 处置 |
|---|---|---|---|
| 1 | T-C5 的三个 kernel 含 `vectorizeRowGeneric` | **前提有误**（它不是行块×列块×K 嵌套） | 换成 `kernelizeBatchMatmul`；`vectorizeRowGeneric` 进 backlog |
| 2 | T-C5「减约 300 行」 | **不可达**（差异在内核形态） | 不强求；报实测并给差异表 |
| 3 | T-C6 拆 6 个文件 | **装不下 ≤1200** | 扩到 13 TU |
| 4 | T-M1 PackStaticMatmulNCNN「现有 3」 | 过期（实为 4） | 以表格目标文件数为准 |
| 5 | T-M1「常量穿透」用例 | 该行为不存在 | 落负向用例 + backlog |
| 6 | T-C7 只提 kG/kBt/kAt 与 rewriteWinograd | 实测还需共享 `annotateConvContract` / `copyConvContract` | 收进 `KernelContract.hpp` |

---

## 3. 统一验证流程

每完成一个任务跑：

```bash
cmake --build <stage> --target format_check --parallel
cmake --build <stage> --target tidy --parallel
```

阶段收尾跑（**全新 `/tmp` 目录**，见 §7.3）：

```bash
rm -rf /tmp/ncnn-compiler-stage-refactor-P4
cmake -S compiler -B /tmp/ncnn-compiler-stage-refactor-P4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-refactor-P4 --parallel
cmake --build /tmp/ncnn-compiler-stage-refactor-P4 \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel
ctest --test-dir /tmp/ncnn-compiler-stage-refactor-P4 --output-on-failure
```

**按任务补充验证**：

| 任务 | 额外验证 |
|---|---|
| T-C5 / T-C6 / T-C7 | 10 模型产物逐字节对照（P3 `emit-plans.sh`）；归一化 scratch 目录名 |
| T-C6 | Conversion lit 48 个全绿；每文件 ≤1200 行 |
| T-C7 | `StrategyNCNN.cpp` 行数下降；`winograd63.mlir` 与 `resnet18_winograd` golden 不变 |
| T-M1 | 四目录文件数达标；新增 lit 全绿 |
| T-M2 | 只改注释，`format_check`/`tidy` 通过即可 |

---

## 4. 硬性约束（沿用 roadmap §7）

1. **不改任何性能默认值**（`NCNN_MATMUL_PACKING=auto` 等保持原状）
2. **不引入新 no-go 路径**（Winograd / layout island / global arena /
   native INT8 边界原样）
3. **每步用新 `/tmp` stage 验证**，失败后修复 + 新目录重建
4. **数值 golden 不放宽**
5. **plan/profile 字段增减 = identity 变化**，必须记录
6. **C runtime 不引 LLVM、不扩未定义符号白名单**
7. **每步 commit 可读**：`git log -p` 能看清改动
8. **本阶段另加**：产物 IR 必须逐字节不变（拆分的唯一护栏）

---

## 5. 回滚策略

| 情况 | 处理 |
|---|---|
| 逐字节对照出现非路径差异 | **立即停止**，定位差异来源；不得「归一化」掉语义差异 |
| 拆分后 Conversion lit 变红 | 回退该族 TU，检查 pattern 注册是否遗漏或重复 |
| T-C5 后数值 golden 变化 | 循环外壳抽取得不等价 → 回退 T-C5，重做 |
| 阶段收尾 CTest 红 | 回退该阶段所有提交，定位后分步重做；**不带病推进 P5** |
