# P2 各任务改动摘要

> 逐字节对照、验证输出见 `verification.log` 与 `plan-before-after.diff`。

---

## T-C1 拆 `EmitModelPlan::runOnOperation`

**改动文件**
- `lib/Transforms/EmitModelPlan/EmitModelPlan.cpp` 2022 → 2287 行
- 新增 `test/Transforms/EmitModelPlan/plan-hash-whitelist.mlir`
- 新增 `test/Transforms/EmitModelPlan/plan_hash_whitelist_check.py`
- 新增 `test/Native/check_plan_hash_whitelist.py`
- `test/CMakeLists.txt` 注册 `plan-hash-whitelist`

**形态**

原 `runOnOperation`（367–2017，**1648 行**）拆成匿名 namespace 里的
`class PlanCollector`：**36 个方法，最大 169 行**。

| 原行区间 | 去向 |
|---|---|
| 375–385 JSON 段 | `PlanCollector` 成员（同名） |
| 386–438 计数器 | `PlanCollector` 成员（同名） |
| 439–491 module 属性读取 | `readModuleLedgers` |
| 492–497 `add_unknown` lambda | `add_unknown` 成员函数 |
| 499–519 `recoverSource` lambda | `recoverSource` 成员函数 |
| 521–611 attention ledger | `collectAttentionSegments` |
| 616–654 low-precision / layout-island 片段 | `collectLowPrecisionLedger` |
| 656–664 tuning 片段 | `collectTuningLedger` |
| 666–706 `plan_hash_input` 初始装配 | `initPlanHash` |
| 713–766 fusion ledger | `collectFusionRecords` |
| 768–786 `add_size_fields` lambda | `add_size_fields` 成员函数 |
| 788–1583 `module.walk` | `collectModule` → `collectFunction` → … |
| 840–1007 生命周期 + 峰值 | `collectStaticLiveness` / `collectPeakWorkspace` |
| 1009–1033 function / body region | `collectFunctionEntry` |
| 1036–1581 逐 op 主体 | `collectOperation`（31 行编排）+ 13 个分节 handler |
| 1585–1653 copy ledger | `collectCopyRecords` |
| 1655–1675 packed 常量回卷 | `collectPackedConstants` |
| 1677–1845 summary | `collectSummary` |
| 1847–2001 root 组装 | `assembleRoot` + `collectTargetSection` + `collectDiagnostics` |
| 2003–2016 落盘 | `writePlan` |

逐 op 主体再拆为：`collectOperationHash` / `collectPackedConstant` /
`hasContract` / `makeContract` / `collectConvDepthwiseOps` /
`collectLayoutIslands` / `collectContracts` / `collectLowPrecision` /
`collectOperationsEntry` / `collectMemoryOperations` / `collectKernelCounters` /
`collectTranspose` / `collectRegions`。

**手法**：`PlanCollector` 的成员名与原局部名**逐一相同**，因此方法体是**逐字拷贝**
（仅统一缩进）。唯一允许的文本改动：

1. 10 个 tablegen pass option 的 `X.getValue()` 读法 → 成员直读；
2. `writePlan` 原先在 pass 方法里靠 `signalPassFailure()` 终止，改为返回
   `LogicalResult`（`path` 作为参数传入）——这是抽取所必需，行为等价；
3. `make_contract()` → `makeContract(operation)`。

**`runOnOperation` 最终 40 行**（验收目标 ≤ 80）。

**新增守卫**
- `check_plan_hash_whitelist.py`（ctest 名 `plan-hash-whitelist`）：源码级断言
  `plan_hash_input` 的 mutation 位点集合**恰好等于**登记白名单（14 个位点 /
  12 个片段 / 4 个折叠片段）。新增 hash 片段而不更新白名单 → 测试红，强制
  identity 变更成为显式决策。
- `plan-hash-whitelist.mlir`：行为级断言 —— 只改输出路径不改 `plan_hash`；
  改白名单字段（threads）必改 `plan_hash`；`build_identity` 恒等于 `plan_hash`。

**原样保留的既有死写**：`layout_island_hash_input` 在折叠进 `plan_hash_input`
之后的两次 `+=`（原 1247 / 1250）不影响 hash，为保持纯位移未清理，已在代码
注释标注。

---

## T-C2 拆 Python `perf_attribution_report`

**改动文件**
- `tools/perf_attribution_report.py` 1588 → **127 行**（`build_report` **50 行**）
- 新增 `tools/attr/__init__.py`
- 新增 `tools/attr/attr_schema.py`（122 行）
- 新增 `tools/attr/attr_join.py`（597 行）
- 新增 `tools/attr/attr_partition.py`（276 行）
- 新增 `tools/attr/attr_report.py`（714 行）
- 新增 `test/Native/test_attr_partition.py`（10 个用例）
- 新增 `test/Native/test_attr_report.py`（7 个用例）
- `test/Native/check_attribute_whitelist.py` 白名单条目随字面量迁移
- `test/CMakeLists.txt` 注册 `attr-partition` / `attr-report`

**归属**
| 模块 | 承载 |
|---|---|
| `attr_schema.py` | `AttributionError`、`require_string`、`require_nonnegative_integer`、`validate_identity`、`category_name` |
| `attr_join.py` | 读入（`read_json`/`read_profile`/`read_perf`）、索引、事件 join（G1–G12）、fusion / packed-kernel join |
| `attr_partition.py` | worker 份额与完整性（V–Y）、wall 分区（Z）、`top_costs` / `top_allocations` 排序 |
| `attr_report.py` | 结构校验、`incomplete_reasons` 链、conv/depthwise 分解、`copy_layout`、报告装配（AA）、`aggregate_v2_reports` |

**不变量**：副作用顺序逐条保留，特别是 `unknown = list(dict.fromkeys([*unknown, …]))`
的 16 处追加顺序（决定 `static.unknown_fields` 数组序）；
`aggregate_v2_reports` 对 `reports[0]` 的原地改写语义原样保留。

---

## T-C3 拆 `GenerateCAPI`

**改动文件**（`lib/Transforms/GenerateCAPI/`）

| 文件 | 行数 | 内容 |
|---|---:|---|
| `GenerateCAPIInternal.hpp` | 80 | **新增** — `mlir::ncnn::capi_detail`：`InputDimRelation`、`FinalizeContext`、`emitWrapper` / `emitInferOutputShapes` 声明 |
| `GenerateCAPIPrepare.cpp` | 687 | `GenerateCAPIPass` 整个（`runOnOperation` / `prepareDynamicRankABI` / `prepareABI` / `writeManifest`）+ `ArgumentInfo` / `abiElementType` / `isCIdentifier`；**唯一**含 `GEN_PASS_DEF_GENERATECAPIPASS` 的 TU |
| `GenerateCAPIFinalize.cpp` | 627 | `FinalizeCAPIPass` 整个（`runOnOperation` = F0–F11 + 交接块 + F26、`finalizeDynamicRankABI`）；**唯一**含 `GEN_PASS_DEF_FINALIZECAPIPASS` 的 TU |
| `GenerateCAPIEmitWrapper.cpp` | 629 | 自由函数 `capi_detail::emitWrapper`（F12–F24 wrapper 代码生成） |
| `GenerateCAPIEmitInfer.cpp` | 349 | 自由函数 `capi_detail::emitInferOutputShapes`（F25 `_infer_output_shapes`） |
| `GenerateCAPI.cpp` | — | **删除**（内容分流到上述 TU） |

**每 TU ≤ 700 行** ✓（最大 687）。

**硬约束（roadmap 未记）**：`Passes.h.inc` 的 `GEN_PASS_DEF_*` 段同时产出
`impl::<Pass>Base` 模板**和一个非 inline 的 `create<Pass>()` 自由函数**，两个 TU
同时包含即重复符号链接错误。因此 pass 类各自独占一个 TU，跨 TU 共享只能走
`capi_detail` 自由函数。见 PLAN.md §1.3。

**手法**：`FinalizeContext` 的成员名与 `runOnOperation` 的局部名逐一相同；
两个 emit TU 用 `auto& x = ctx.x;` 引用别名前奏，使搬移后的代码体保持未限定、逐字。
`checkedMultiply` 在 `runOnOperation` 与 `finalizeDynamicRankABI` 中各有一份独立
拷贝，**按"不动语义"保留两份**，未去重。

### ⚠️ 对 roadmap 改动点的偏离（未做，已披露）

roadmap T-C3 的改动点写着「`prepareABI` 拆成 `validateInputs` / `buildSignatures` /
`emitDeclarations`」。**本阶段未做**，原因是与同任务的验收标准「每 TU ≤ 700 行」
在现有布局下直接冲突：`GenerateCAPIPrepare.cpp` 现为 687 行，把 328 行的
`prepareABI` 再拆成 3 个成员函数会新增约 30 行签名/编排，把该 TU 顶到约 720 行。

按「验收标准优先于改动点」取舍，保留 `prepareABI` 原样（328 行）。若要补齐，
需要先把 `writeManifest` 从成员改成 `capi_detail` 自由函数腾出行数预算，
属独立一步，建议进 backlog 而不是在收尾阶段插入。

**验收达成情况**：每 TU ≤ 700 行 ✓；`.h` 与 manifest **逐字节一致** ✓；
GenerateCAPI 23 个 lit 全绿 ✓。

---

## T-C4 拆 `ncnn-compile::main`

**改动文件**
- 新增 `tools/CompileSession.hpp`（230 行）：7 个从 `ncnn-compile.cpp` 平移的类型
  （`Argument` / `Manifest` / `ScopedDirectory` / `ToolResult` / `OutputDirectoryState` /
  `ClangTargetArguments` / `TuningSettings`）+ `class CompileSession` 声明
  （10 个方法、72 个成员）
- 新增 `tools/CompileSession.cpp`（36 行）：仅 `CompileSession::run()` 编排（30 行）
- `tools/ncnn-compile.cpp`：`main`（1313 行）→ 9 个方法定义（方法体逐字）+ **4 行 `main`**
- `tools/CMakeLists.txt`：`add_executable(ncnn-compile ncnn-compile.cpp CompileSession.cpp)`
- `test/Numerical/CMakeLists.txt`：fixture `DEPENDS` 补 `CompileSession.cpp`

**`main` 最终 4 行**（验收目标 ≤ 40）：

```cpp
int main(int argc, char** argv) {
  ncnn_compile::CompileSession session;
  return session.run(argc, argv);
}
```

**手法**：与 T-C1 同——成员名与原局部名逐一相同，方法体逐字拷贝。行对齐 diff：
1311 行中 **1216 行字节相同**、73 处声明改赋值、22 处 `run(` → `::run(`、
8 处补 `return 0;`，**0 处未授权改动**。

**返回码契约逐位保留**（这是 T-C4 的核心风险面）：`0` 成功、`1` 走 `fail(msg)`、
其余为子进程退出码原样透传。7 个场景前后对照：

| 场景 | 重构前 | 重构后 |
|---|---:|---:|
| 缺输入文件 | 1 | 1 |
| 非法 `--emit` 阶段 | 1 | 1 |
| 非法 `--vector-width` | 1 | 1 |
| 非法 `--tuning-profile` | 1 | 1 |
| 找不到工具 | 1 | 1 |
| driver 退出 42 | 42 | 42 |
| 成功 | 0 | 0 |

stdout/stderr 在归一化随机 staging 后缀后逐字节一致；成功产物 `.so` / `.h` md5 一致。

**允许的非纯搬移改动**（均为抽取所必需，已逐条核对行为等价）：
1. `run(` → `::run(`（成员 `CompileSession::run` 遮蔽自由函数 `run`）
2. 8 个方法补 `return 0;`（原先落到 `main` 末尾的 `return 0`）
3. `ScopedDirectory` 增加**默认构造**（`remove_=false`，析构为空操作——否则失败路径上
   会多出清理告警）与**移动赋值**（卸载源对象，避免 `staging = ScopedDirectory(...)`
   的临时对象析构把 staging 目录删掉）；原有构造/析构/`release()` 逐字未动
4. `resolveTools(char** argv)` 传入原始 `argv`，使
   `getMainExecutable(argv[0], …)` 保持逐字
5. 9 处引用别名（`*_path` / `target_args` / `isa_args` / `resolved_int8_target`）
   转为值拷贝——引用成员不能在构造后重绑定；拷贝路径字符串行为等价

**刻意保留的全局副作用**：B13 对 `g_int8_kernel` / `g_int8_depthwise` /
`g_int8_cast_chain` 三个全局 `cl::opt` 的写回原样保留，未改成传参。
