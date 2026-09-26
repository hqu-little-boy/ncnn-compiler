# Identity 变化记录 — P1 语义收敛

> **结论：本阶段无 identity 变化。** 所有 `.plan.json` / profile / attribution
> 文档的字段名与字段值均未改动，跨期 plan-hash join 可正常进行。

判据：identity 变化的定义是「plan/profile 字段增减」或「哈希算法更换」
（roadmap §7.5 / §7.6）。逐项核对：

| 任务 | 是否触碰 plan/profile 字段名 | 是否触碰字段值 | 是否换哈希 |
|---|---|---|---|
| T-S4 溢出检查统一 | 否 | 否 | 否 |
| T-S1 常量查找统一 | 否 | 否 | 否 |
| T-S2 属性 schema 收敛 | 否 | 否 | 否 |
| T-S3 模块 ledger 收敛 | 否 | 否 | 否 |

## 逐项说明

### T-S4

`checkedAdd` / `checkedMul` / `checkedProduct` 收敛到 `CheckedMath.hpp` 后语义
逐点对齐：

- 非负操作数前置条件保留（`ShapedType::kDynamic == -1` 必须证伪，不能乘进去）；
- `checkedProduct` 的 `limit` 预算参数保留，`kMaxStrategyElements` / `kMaxIm2colWindowElements`
  两档预算原值不动；
- 唯一的语义调整是 `NCNNOps.cpp` 的 `checkedAdd(*result, -padBefore)` 改写为
  `checkedSub(*result, padBefore)`。两者等价：旧 `checkedAdd` 接受负操作数并做
  带符号溢出检查，新 `checkedSub` 接受非负操作数、返回带符号差值。对
  `padBefore >= 0` 结果逐位一致。

数值路径未改，plan 输出不变。

### T-S1

`findConstantElements` 的穿透集合取 packing 现行为（`tensor.cast` /
`tensor.collapse_shape` / `tensor.expand_shape`），故 `PackStaticMatmulNCNN` 行为
逐位不变。

**唯一的行为变化**：`FoldNCNNBatchNorm` 现在能穿过上述视图找到常量，会折叠
以前静默跳过的 batchnorm。这是 T-S1 的设计目标（两套实现答案不一致的收敛），
且属**编译期优化选择**而非字段增减——plan 里可能出现以前没有的
`ncnn.fusion_*` / 已折叠的 conv 契约，但**字段名集合不变**。

> 注：折叠与否会改变 plan 的 `plan_hash` 内容（hash 覆盖 op 契约），因此
> **P1 之后的 plan-hash 与 P1 之前不可直接比对**。这属于"产物语义变化"而非
> "schema/identity 变化"：`.plan.json` 的字段结构完全一致，Python 侧
> `validate_identity` 仍通过。归因分析按同轮内 join 即可。

`tensor.extract_slice` 有意不穿透（取子集，不是视图），并有负向 lit 用例锁定。

### T-S2

294 处裸 `"ncnn.*"` 字面量收敛到 `KernelContract.hpp` 的 164 个具名常量。
**字面量的字符串值逐一原样保留**，因此所有属性键、op 名、JSON kind 均未变值。
`root["kind"] = contract::kModelExecutionPlanKind` 与原先
`root["kind"] = "ncnn.model_execution_plan"` 输出完全一致。

`plan_hash` 算法未动（`fnv1a` 仍在原处，合并到 `xxHash64` 是 P3 的 T-S5）。

### T-S3

`ModelLedger` 只包读写与 clear 的**命名**，不改任何属性名、不引入
`ncnn.ledger` 字典（roadmap §8 明确不做）。`clearDimensionConstraints` 的
删除集合与 `GenerateCAPI` 原先的 `removeAttr(kShapeConstraints/kInputDimRelations)`
逐项一致——**保留** `kRankVariant` / `kDynamicRank` 不删（原行为如此）。

## 需要注意的产物差异（非 identity）

`FoldNCNNBatchNorm` 折叠范围扩大后，下列产物会与 P1 前不同：

1. **`.plan.json` 的 `plan_hash`** —— 覆盖 op 契约，折叠改变了 op 集合。
2. **生成的 `.so` 数值路径** —— batchnorm 参数被折进卷积权重/bias，浮点累加
   结构改变。数值 golden 预算不变，实测见 `verification.log`。

这两项按 roadmap §9 属"T-S1 改变 batchnorm 折叠路径导致数值差异"的预期路径；
若数值 golden 超预算，收窄 helper 穿透集合即可回退（roadmap §9 第 4 行）。
