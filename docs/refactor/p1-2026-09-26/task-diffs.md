# P1 各任务改动摘要

工作树规模：30 个已有文件改动 + 4 个新增文件，约 +634 / −465 行。
（`profile_runtime.c` 的 27 行变动是全局 clang-format 折行，无语义改动。）

---

## T-S4 溢出检查统一

**新增** `include/ncnn-mlir/Support/CheckedMath.hpp`

```cpp
mlir::FailureOr<int64_t> checkedAdd(int64_t lhs, int64_t rhs);
mlir::FailureOr<int64_t> checkedSub(int64_t lhs, int64_t rhs);
mlir::FailureOr<int64_t> checkedMul(int64_t lhs, int64_t rhs);
mlir::FailureOr<int64_t> checkedProduct(llvm::ArrayRef<int64_t> dims, int64_t limit);
```

**删除的重复实现**

| 文件 | 原实现 | 原签名 |
|---|---|---|
| `lib/Transforms/StrategyNCNN/StrategyNCNN.cpp:39-66` | `checkedMul` / `checkedAdd` / `checkedProduct` | `bool` + out-param |
| `lib/Dialect/NCNN/IR/NCNNOps.cpp:24-40` | `checkedAdd` / `checkedMultiply` | `FailureOr<int64_t>` |

**保留不动**（roadmap §3「推荐保留，只要语义一致」）：`NCNNOps.cpp` /
`NCNNToFunc.cpp` / `NCNNToTosa.cpp` / `ShapeProgram.hpp` 中直接调用
`llvm::*Overflow` 的 20 处，语义与 helper 一致。

**未收录 `checkedSub` 于 v1 草案的原因**：初判"无调用点且符号语义含糊"。
落地时发现 `NCNNOps.cpp` 有 `checkedAdd(*result, -padBefore)` 两处依赖负操作数，
故补回 `checkedSub`（非负操作数、带符号差值）并把这两处改写过来——这正是审计
说的"同文件两套语义"的实例。

**偏离 roadmap 之处**：`checkedProduct` 的 `limit` 参数设为**必填**（roadmap 草案
无默认值也未说明）。理由是预算是 pass 的策略而非算术属性，`StrategyNCNN` 有
`kMaxStrategyElements`（2^28）与 `kMaxIm2colWindowElements`（2^29）两档，
留默认值会让调用点看不出自己用了哪档。

---

## T-S1 常量查找统一

**新增** `ConstantFold::findConstantElements(mlir::Value)`（声明于
`include/ncnn-mlir/Support/ConstantFold.hpp`，实现在 `lib/Support/ConstantFold.cpp`）。

穿透集合 = **纯视图三个**：`tensor.cast` / `tensor.collapse_shape` / `tensor.expand_shape`
（= packing 原 `findConstant` 行为）。

**有意不穿透**：
- `tensor.extract_slice` —— 取子集，源常量不是切片内容，穿透会喂给 folder 错误数据；
- `tensor.from_elements` —— 是合成而非视图，结果不是现成的 `ElementsAttr`。

两者均有注释说明；`extract_slice` 有负向 lit 用例锁定。

**删除的重复实现**：`FoldNCNNBatchNorm.cpp:24 getConstantElements`（只看直接 def）、
`PackStaticMatmulNCNN.cpp:115 findConstant`（穿透三视图）。

**未按 roadmap 加 `findConstantOp`**：packing 侧只用到 `constant.getValueAttr()`，
返回 `ElementsAttr` 即可覆盖，加第二个 API 是死代码。

**保留的拒绝原因区分**：packing 的 `packing_rejected_dynamic`（找不到常量）与
`packing_rejected_layout`（常量形状/元素类型不对）是两个 plan 字段，转换时保持
分流不变。tensor 类型的 `arith.constant` 其值必为 `ElementsAttr`，故
`!findConstantElements(rhs)` ⟺ 找不到常量，两分类不变。

**新增 lit**
- `test/Transforms/FoldNCNNBatchNorm/constant-through-cast.mlir`（正向：cast 链折叠；
  负向：`extract_slice` 不折叠）
- `test/Transforms/PackStaticMatmulNCNN/constant-through-cast.mlir`（正向：cast 链
  仍打包；负向：非常量 RHS 报 `packing_rejected_dynamic`）

---

## T-S2 属性 schema 收敛

**盘点结果（比 roadmap 审计更全）**：294 处字面量 / 164 个键。
roadmap 记的 187 处 / 67 键漏了多段键 `ncnn.c_api.*`（正则未覆盖第二个 `.`）。

**`KernelContract.hpp` 新增 69 个具名常量**，分 9 组：
`kLayer*`（22 个 op 名）、`kShape*`（7）、`kCApi*`（12）、`kEntry*`/`kRank*`/`kPrecision*`（6）、
`kSource*`（2）、`kWorkspace*`（10）、`kProfile*`（4）、`kModel*`（3 个 JSON kind）、
模型级策略旗标（3）。

**改写 22 个 .cpp**，`lib/` 下裸 `"ncnn.*"` 字面量归零。

**删除的本地别名**（单一事实源被架空的直接证据）：
- `GenerateCAPI.cpp` 12 个 `k*Attr`（如 `kExportNameAttr`）
- `ReuseWorkspaceSlots.cpp` 10 个 `kSlot*` 等
- `ReuseWorkspaceSlots.cpp` 的 `setStringAttr` / `setIntegerAttr`（与
  `contract::setString` / `setInteger` 逐字重复）

**新增** `test/Native/check_attribute_whitelist.py`，注册为 ctest `attribute-whitelist`。
白名单按「文件 → 精确键集合」收紧：允许文件里冒出新键照样报错。含负向自检
（临时树里种一个裸键 + 一个未知键，两者都必须被报出）。

**残留字面量 4 处，全部登记在案**：

| 文件 | 键 | 为何不能改 |
|---|---|---|
| `lib/ProfileRuntime/profile_runtime.c` | `ncnn.model_execution_profile` ×2 | C runtime 以 `.c` 安装、每模型单独编译，无法 include C++ 头；符号集被 `profile_allowed` 白名单冻结（roadmap §7.7） |
| `tools/perf_attribution_report.py` | 3 个 JSON kind | JSON 文档判别值，Python 无法引用 C++ 常量；改值 = identity 变化 |

---

## T-S3 模块 ledger 保守收敛

**新增** `include/ncnn-mlir/Support/ModelLedger.hpp`：`read` / `write` / `clear`，
外加两个**命名的生命周期终点**：

- `clearDimensionConstraints` —— C ABI 导出后消费掉维度约束；**保留** `kRankVariant` /
  `kDynamicRank`（描述导出的函数本身）。这一区分是刻意的：原先 `GenerateCAPI`
  只删两个键、留 rank，合并成一个 clear 就是行为变更。
- `clearRecords` —— plan 生成后消费掉三组记录。

**已收敛的读写点**：`EmitModelPlan`（3 处记录读）、`RewriteLinalgCopies`
（copy ledger 替换）、`NCNNToTosa`（attention ledger 追加）、`NCNNImporter` /
`ImportActivation`（shape 契约写）、`GenerateCAPI`（`clearDimensionConstraints` +
读）、`NCNNOps` / `NCNNToFunc` / `VerifyModelShapeContracts`（读）。

**未收敛、留作 backlog**：ledger **记录内部**的字段名字面量
（`record.set("copy_kind", ...)` / `EmitModelPlan` 的 `root["kind"]` 等 JSON
字段名）。这些是 plan JSON schema 的一部分，改名会触发 identity 变化
（roadmap §7.5「plan/profile 字段增减 = identity 变化」），不属 P1 范围。

**不做**（roadmap §8 明确）：`ncnn.ledger` 字典、改 pass 顺序契约。

---

## 附带修复（非本次产生，按门禁要求一并修）

`EmitModelPlan.cpp:252` `std::find(...) == end` → `std::ranges::contains`，
clang-tidy `modernize-use-ranges` 报 `-warnings-as-errors`。
