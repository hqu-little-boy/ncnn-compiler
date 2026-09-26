# P1 语义收敛 · 执行计划

> **阶段**：P1（T-S1 / T-S2 / T-S4 / T-S3）
> **日期**：2026-09-26
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.1 / §6 / §7
> **纪律**：不改性能默认值、不引新依赖、不放宽数值 golden、plan 字段名不变（不触发 identity 变化）

---

## 0. 对 roadmap 审计数字的勘误

roadmap v2 的量化表写于工作树同一时点，但若干细节已与当前代码不符。执行前复核结果：

| roadmap 说法 | 实测 | 影响 |
|---|---|---|
| 裸 `"ncnn.*"` **187 处 / 67 键** | `lib` 184 + `include` 95 + `tools` 3 = **282 处 / 152 键** | T-S2 盘点范围更大；其中约 1/3 是 **op 名字符串值**（如 `"ncnn.convolution"`），非属性键 |
| `NCNNOps.cpp:24` 手写 `checkedAdd`（`FailureOr`） | 该文件无此实现；手写 `checkedMul`/`checkedAdd`/`checkedProduct` 在 **`StrategyNCNN.cpp:39-64`**（`bool` + out-param） | T-S4 改动点从 2 文件缩到 1 文件；接口签名以现状为准 |
| `getConstantElements` 只看直接 def | 确认属实（`FoldNCNNBatchNorm.cpp:24`） | T-S1 按原计划 |
| `findConstant` 穿透 cast/collapse/expand | 确认属实（`PackStaticMatmulNCNN.cpp:115`） | T-S1 穿透集合以 packing 现行为为准 |

**决策**：T-S2 的"归零"验收按 **属性键** 理解并全量收敛；op 名字符串值同样收敛到
`contract::kLayer*` 常量，因为 `grep -rE '"ncnn\.' lib` 的字面量验收无法区分两者。
但 **不改任何字符串值本身**（plan/JSON 字段名、op 名一律原样），避免 identity 变化。

---

## 1. 任务分解与顺序

```
T-S4（0.5d，低风险，独立）
  └─ CheckedMath.hpp：checkedAdd / checkedSub / checkedMul / checkedProduct

T-S1（1d，低风险，独立）
  └─ ConstantFold：findConstantElements / findConstantOp
     ├─ 收敛 FoldNCNNBatchNorm::getConstantElements
     ├─ 收敛 PackStaticMatmulNCNN::findConstant
     └─ 2 个新 lit（constant-through-cast）

T-S2（2–3d，中风险，大面）
  └─ KernelContract.hpp 补键 + 全库调用点改具名常量
     └─ test/Native/check_attribute_whitelist.py

T-S3（1.5d，中风险，依赖 T-S2）
  └─ ModelLedger.hpp：read / write / clear
     └─ 收敛 6 组 module/function 级 ArrayAttr
```

执行顺序 **T-S4 → T-S1 → T-S2 → T-S3**（先小后大，T-S3 建立在 T-S2 的键常量之上）。
四项全部完成后统一走一次全量门禁（§2），不逐任务重复 449-test 全跑。

---

## 2. 统一提交门禁（P1 收尾执行一次）

```bash
# 1. 全局静态检查（含非本次产生的问题，按结果修复）
cmake --build /tmp/ncnn-compiler-stage-refactor-p1 --target format_check --parallel
cmake --build /tmp/ncnn-compiler-stage-refactor-p1 --target tidy --parallel

# 2. 全新 /tmp stage，避免旧产物误导
mkdir /tmp/ncnn-compiler-stage-refactor-p1
cmake -S compiler -B /tmp/ncnn-compiler-stage-refactor-p1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-refactor-p1 --parallel

# 3. 三个 numerical target（重建 generated fixture）
cmake --build /tmp/ncnn-compiler-stage-refactor-p1 \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel

# 4. 全量 CTest
ctest --test-dir /tmp/ncnn-compiler-stage-refactor-p1 --output-on-failure

# 5. 提交前检查
git diff --check
git status --short
git diff --cached
git log --oneline -10
```

**验收失败不提交**；修复后重新从全新 `/tmp` 目录构建和测试。

---

## 3. 逐任务验收标准

| 任务 | 额外验收（除 §2 全量门禁外） |
|---|---|
| T-S4 | `grep -rn "checkedAdd\|checkedProduct"` 只剩 `CheckedMath.hpp` 一处定义 |
| T-S1 | 2 个新 lit 全绿；`grep -rn "getConstantElements\|findConstant" lib` 只剩 helper 定义；batchnorm 数值 golden 不回归 |
| T-S2 | `grep -rE '"ncnn\.' lib` 归零（`KernelContract.hpp` 自身除外）；白名单测试绿；plan 输出字段名 diff 为空 |
| T-S3 | 散落 `getAttrOfType<ArrayAttr>("ncnn.*")` 归零；所有 ledger 读写走 `ModelLedger` |

## 4. 明确不做（本阶段）

- 不做 T-S3"彻底档"（`ncnn.ledger` 字典 + 改 pass 顺序契约）
- 不改 CLI 选项名、不改性能默认值
- 不做 T-S5 稳定哈希合并（属 P3，会触发 identity 变化）
- 不做任何性能优化（性能已收官，P1 不承诺性能收益）

## 5. 风险与回滚

| 风险 | 处置 |
|---|---|
| T-S1 使 batchnorm 开始折叠以前跳过的常量 → 数值路径变化 | 属预期；跑数值 golden，超预算则收窄穿透集合（只保留与 packing 一致的 view 集合） |
| T-S2 改漏一处 → 编译失败 | 编译期查出，属好事；靠 `grep` 归零验收兜底 |
| T-S3 生命周期管理回归 | 保守档只包读写 API，不改 `RewriteLinalgCopies` 的 `removeAttr` 时序语义 |
| 全量 CTest 红 | 不带病提交；`git revert` 对应任务后分步重做 |

---

## 6. 交付物

```
docs/refactor/p1-2026-09-26/
  PLAN.md                 # 本文件
  README.md               # 验收结果、遗留问题（收尾补）
  task-*-diff.txt         # 各任务 diff 摘要
  verification.log        # format_check + tidy + CTest 输出
  identity-changes.md     # 本阶段无 identity 变化（plan 字段名未改）
```

性能对比证据另存 `docs/performance/p1-2026-09-26/`，结论写入
`docs/ncnn-mlir-performance-optimization-plan-p18.md`。
