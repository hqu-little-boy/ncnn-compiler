# P5 identity 变化记录

> **结论：P5 不触发任何 identity 变化。** 无需停止跨期 plan-hash join，
> P3 `3d1c0fc` 之后的 join 链可以继续。

---

## 判据

roadmap §7.5：「plan/profile 字段增减 = identity 变化」。
§7.6：「换哈希算法 = identity 变化」（T-S5 已在 P3 处理并记录）。

P5 的四项任务逐条对照：

| 任务 | 动了 plan/profile 字段？ | 动了哈希算法？ | identity 变化 |
|---|---|---|---|
| **T-J1** writer 抽离 + UTF-8 修 | 否（字段名/结构未动） | 否 | **无** |
| **T-M4** 拆插桩器 | 否（只搬代码） | 否（仍走 `stableHash64`） | **无** |
| **T-M5** 诊断分级 | 否（只加 stderr 输出） | 否 | **无** |
| **T-M3** 文档 | — | — | **无** |

---

## 逐项说明

### T-J1：UTF-8 修复改的是 profile **内容**，不是 **schema**

修复只在 `pj_string` 里替换非法字节为 `�` 转义：

* **合法输入**（ASCII / 转义 / 合法 UTF-8）→ 24 份 golden 里 12 份
  **逐字节完全相同**（3 schema × ASCII / 转义 / 合法 UTF-8 / 边界）；
* **非法 UTF-8 输入** → 12 份从「非法 JSON」变成「合法 JSON 含 U+FFFD」。

字段名、字段集合、数值编码（`%llu` 整数）、浮点格式（`%.9f`，三个既有
share/projection 字段）全部未动。**`plan_hash` / `build_identity` /
`profile_id` 的派生输入没有变**，所以不构成 identity 变化。

属于 roadmap §9 的「T-J1 UTF-8 修复改变 profile 内容 = 属修 bug」——
需确认归因层仍能 join：profile 的 `id` / `plan_hash` / `build_identity`
字段编码未变，`tools/attr/attr_join.py` 的身份键不受影响。

### T-M4：产物逐字节不变，identity 串亦不变

`compare-profile-instrumentation.sh` 对 squeezenet_v1_1 / resnet18 带
`--profile --emit=all` 编译，全部产物（含 `.plan.json`）归一化 scratch
路径后**逐字节一致**。`plan_hash` 派生输入逐项相同。

### T-M5：`--warnings-as-errors` **刻意不进 identity 串**

`tools/ncnn-compile.cpp` 的 `build_codegen_identity()` 是手工拼接的
选项串。`--warnings-as-errors` **不在其中**，理由见
`include/ncnn-mlir/Support/Diagnostics.hpp`：

> 这是**诊断策略**不是代码生成参数。开与不开，产物字节与 `plan_hash`
> 都一样，进 identity 串只会制造假的身份分裂。

实测：同一批模型开/关该选项，`.plan.json` 的 `plan_hash` 相同。

新增的 `emitWarning` 只写 stderr，不写任何 IR 属性——
`grep -rE '"ncnn\.' lib tools` 的白名单（P1 T-S2 的 `check_attribute_whitelist.py`）
不受影响。

### T-M3：纯文档

---

## 复核方法

```bash
# T-J1：24 份 profile 前后对照
CC=clang-21 docs/refactor/p5-2026-09-27/capture-profile-golden.sh /tmp/golden
# → 合法输入 12 份逐字节相同；非法 UTF-8 12 份只在转义处变

# T-M4：--profile 路径前后对照
docs/refactor/p5-2026-09-27/compare-profile-instrumentation.sh \
  /tmp/p5-ref /tmp/ncnn-compiler-stage-p5 /tmp/cmp
# → 2 模型全部产物归一化后逐字节一致
```
