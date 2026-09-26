# P4 identity 变化记录

## 结论：**无 identity 变化**

P4 是结构拆分 + 测试补强，**没有增减任何 plan / profile 字段**，
**没有更换哈希算法**（T-S5 的 `stableHash64` 从 P3 起沿用），
**没有改变数值路径**。因此：

* `plan_hash` / `build_identity` / `*.profile_id` **不变**；
* **跨期 plan-hash join 不受影响**，P3 起的 join 契约继续有效；
* `perf_attribution_report.py` 的 identity 校验**无需改动**。

## 证据

10 个模型 × 7 种产物（`.plan.json` / `.h` / manifest `.json` / `.so` /
三份 IR dump）经重构前后的 `ncnn-compile` 产出，`diff -ru` 归一化
`/tmp/ncnn-compile-<随机目录>` 后 **0 行差异**（`plan-before-after.diff` 为空）。

归一化只替换 ncnn-compile 的编译期 scratch 目录名（每次运行随机生成，
出现在 IR dump 的 `loc()` 元数据里），**不触碰任何语义字段**。

## 未触发 identity 的高危点（逐条核对）

| 项 | 是否触发 | 说明 |
|---|---|---|
| T-C5 抽 `emitTileLoopNest` | 否 | 循环外壳逐字等价；contract 注解字符串未改 |
| T-C6 拆 NCNNToTosa | 否 | pattern 按 op 名互不重叠，注册顺序不选择竞争 pattern |
| T-C7 Winograd 独立 | 否 | 代码逐行搬运；`annotateConvContract` 收进 KernelContract 但属性集不变 |
| T-M1 新增 lit | 否 | 只加测试，不改产品代码 |
| T-M2 设计注释 | 否 | 只加注释 |

## 若后续触发 identity 须遵守

roadmap §7.5 / §7.6：plan/profile 字段增减、或更换哈希算法，都必须
(1) 在提交信息与本文件记录，(2) **换算法那一轮停止 plan-hash 跨期 join**，
(3) 更新 `perf_attribution_report.py` 的 identity 校验。
