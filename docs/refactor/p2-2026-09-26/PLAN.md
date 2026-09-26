# P2 巨型函数拆分 · 执行计划

> **阶段**：P2（T-C1 / T-C2 / T-C3 / T-C4）
> **日期**：2026-09-26
> **来源**：`docs/code-quality-refactor-roadmap.md` §4.2 / §6 / §7
> **纪律**：纯机械拆分、不改任何语义、不改性能默认值、不引新依赖、plan/profile 字段名与字段值逐字不变
> **前置**：P1 语义收敛已落（`9ed5e9d`），P2 不再与 P1 错峰

---

## 0. 对 roadmap 审计数字的勘误

roadmap v2 的函数长度表写于其时点。执行前逐项复核（行号按 P1 之后的工作树）：

| roadmap 说法 | 实测 | 影响 |
|---|---|---|
| `EmitModelPlan::runOnOperation` 1648 行（367–2017） | **1651 行**（367–2017） | 一致 |
| `GenerateCAPI` 2241 行、`runOnOperation` 在 `:118` 与 `:713` | **2228 行**；`GenerateCAPIPass::runOnOperation` 在 **102–152（51 行，本来就不大）**；真正巨型的是 **`FinalizeCAPIPass::runOnOperation` 700–1927（1228 行）** | T-C3 的重心从"两个 pass 平均拆"改为"主要拆 FinalizeCAPIPass"；`prepareABI` 370 行、`finalizeDynamicRankABI` 待拆 |
| `ncnn-compile::main` 1312 行（2086–3397） | **1312 行**（2085–3397，37 个语义块） | 一致 |
| Python `build_report` 856 行 + `aggregate_v2_reports` 447 行 | **855 行**（246–1100）+ **446 行**（1103–1548） | 一致 |
| T-C1 抽「13 个纯函数，每个返回 `JsonObject`/`JsonArray`」 | **与单次遍历设计冲突**，见 §1.1 | 改为「单次遍历 + 分节 handler」 |

### 1.1 T-C1 的关键设计偏离（必须说明）

`EmitModelPlan.cpp:365` 有一条明确的设计注释：

> This pass intentionally centralizes the complete static-plan walk so that
> all counters and the plan hash are derived from one deterministic visit.

实测确认：`plan_hash_input` 在 **6 个不同位置**被顺序累加——

| 位置 | 累加内容 |
|---|---|
| 667–707 | 前缀串 + `attention_hash_input` / `low_precision_hash_input` / `layout_island_hash_input` / `tuning_hash_input` 四段折叠 |
| 754 / 760 / 764 | fusion ledger 逐条（`fusion-profile-id` / `fusion-tail-profile-id` / `fusion-record`） |
| 1040–1047 | 遍历内逐 op：`id\|kind` + operand 类型 + result 类型 + `attrs=` |
| 1070–1076 | 逐 op：`source-layer` / `source-name` |
| 1349 | 逐 op：`low-precision-op` |
| 1592 / 1608 / 1646 | copy ledger 逐条 |

若按 roadmap 把 13 个 section 拆成**各自独立遍历**的纯函数，`plan_hash_input` 的拼接顺序
与计数器口径都会变 → **plan_hash 变值 → identity 变化**，直接违反 §7.5「不动语义」。

**决策**：保留单次确定性遍历，把 runOnOperation 的语义体抽成一个 `PlanCollector`
（持有全部 JSON 段 + 计数器 + hash 累加器），并在**保持调用顺序不变**的前提下抽出分节
handler。函数名沿用 roadmap 的 `collectXxx`，但它们是 collector 的方法、按原顺序被调用，
而不是各自独立遍历。这是"纯机械"在该架构下的正确形态。

`runOnOperation` ≤ 80 行的验收目标不变。

### 1.2 一处既有死存储（原样保留，不做清理）

`layout_island_hash_input` 在 641 初始化、705 被折叠进 `plan_hash_input` 之后，
1247 / 1250 仍在 `+=`。**这两次追加是死写**——该字符串此后再未被读取，因此不影响
plan_hash。为保证 diff 是纯位移、不做"顺手清理"，**原样保留这两行追加**并在代码注释
中标注。若要清理属 P5 级别的行为中性收尾，不混入本期。

### 1.3 T-C3 的硬约束：tablegen 的 DEF 块只能单 TU 包含

`include/ncnn-mlir/Passes.h.inc` 的 `GEN_PASS_DEF_<PASS>` 段同时产出
`impl::<Pass>Base` 模板**和一个非 inline 的自由函数**
`std::unique_ptr<::mlir::Pass> create<Pass>() { … }`。两个 TU 同时包含它就是
重复符号链接错误。因此：

- 每个 pass **类定义必须独占一个 `.cpp`**，且只有该 `.cpp` 能写
  `#define GEN_PASS_DEF_<PASS>` + `#include "ncnn-mlir/Passes.h.inc"`；
- 跨 TU 共享只能走**具名 namespace 里的自由函数** + 一个新的内部头文件。

roadmap 的「ABI 构造与 manifest 序列化分两个 TU」按此调整为「按 pass 类归属 +
把 `FinalizeCAPIPass::runOnOperation` 的代码生成段抽成自由函数」，详见 §2 T-C3。

### 1.4 T-C2 的一处归属订正 + pytest 不可用

- roadmap 未指定 `category_name` 归属，本计划初稿把它写进 `attr_join.py`；
  实现按 **`attr_schema.py`**（它只做事件类别 → 报告类别的映射，属 schema 侧）落位。
- **本机未安装 pytest（也没有 pip）**，roadmap 验收里的
  `python3 -m pytest test/Native/test_attr_*.py` 无法执行。新增单测写成纯
  `unittest.TestCase`（pytest 可直接收集），实际以
  `python3 test/Native/test_attr_*.py` 与 ctest 注册项执行，并在
  `verification.log` 中披露该偏差。


---

## 1. 任务分解与执行顺序

```
T-C1（EmitModelPlan）  ← plan-hash identity 风险最高，先做，产出 plan-before/after 逐字节对照
T-C2（Python 归因）    ← 与 C++ 完全不相交，可与 T-C1/T-C3/T-C4 并行
T-C3（GenerateCAPI）   ← 依赖 plan-before 基线的同一套对照方法
T-C4（ncnn-compile）   ← 依赖 .h / manifest 逐字节对照
```

四项全部完成后统一走一次全量门禁（§3），不逐任务重复 449-test 全跑。
这与 P1 的做法一致（P1 四任务单次提交 `9ed5e9d`）。

---

## 2. 逐任务方案

### T-C1 拆 `EmitModelPlan::runOnOperation`

**现状**：`lib/Transforms/EmitModelPlan/EmitModelPlan.cpp` 2022 行，`runOnOperation`
367–2017。函数前半是一次性铺开约 60 个局部（JSON 段 + 计数器 + module 属性读取 +
2 个 lambda `add_unknown` / `recoverSource`），中段是 `module.walk(func)` 单次遍历，
尾段是 summary / root 组装 + 落盘。

**方案**：新增匿名 namespace 内的 `class PlanCollector`，成员即原局部变量的逐一平移：

```cpp
class PlanCollector final {
 public:
  // —— 遍历前：module 级 ledger 与 hash 片段 ——
  void collectAttentionSegments(ModuleOp);      // 原 529–638
  void collectLowPrecisionLedger(ModuleOp);     // 原 603–653（int8/packed-conv 属性 + hash 片段）
  void collectTuningLedger();                   // 原 655–665（tuning_hash_input）
  void initPlanHash(StringRef model, ...);      // 原 667–707
  void collectFusionRecords(ModuleOp);          // 原 711–767

  // —— 单次遍历：按原顺序调用分节 handler ——
  void collectModule(ModuleOp);                 // 原 module.walk(func) 外壳
  void collectFunction(func::FuncOp);           // 原 per-function 前半
  void collectStaticLiveness(...);              // 原 835–1006（lifetime + peak）
  void collectFunctions(func::FuncOp);          // 原 1008–1026
  void collectRegions(...);                     // 原 1028–1031 + 1573–1584
  void collectOperations(Operation*, ...);      // 原 1033 起逐 op 主体
  void collectContracts(Operation*, ...);       // 原 1265–1343
  void collectConvDepthwiseOps(Operation*, ...);// 原 1157–1234
  void collectFusions(Operation*, ...);         // 原 1329–1341
  void collectLowPrecision(Operation*, ...);    // 原 1344–1351
  void collectProvenance(...);                  // 原 1363–1372
  void collectBuffers(Operation*, ...);         // 原 1380–1500（memref.alloc）

  // —— 遍历后 ——
  void collectCopyRecords(ModuleOp);            // 原 1654–1672
  void collectPackedConstants();                // 原 1674–1675
  JsonObject collectSummary();                  // 原 1677–1844
  JsonObject assembleRoot(JsonObject summary);  // 原 1846–2010
  void writePlan(ModuleOp);                     // 原 1963–2015
};
```

**必须保持不变的三件事**：

1. `plan_hash_input` 的**拼接顺序与内容**（§1.1 表格 6 处）；
2. 全部计数器的递增时机（尤其 `has_module_fusion_records` 影响 `fusions` 是否补条目、
   `compileTimeB` 影响 `packed_buffer_bytes` 是否累加）；
3. JSON 键的**写入顺序**——`llvm::json::Object` 序列化时按键插入顺序输出，
   任何键顺序变化都会让 `.plan.json` 逐字节 diff 变红，这正是我们要的护栏。

**`runOnOperation` 目标形态**（≤ 80 行）：

```cpp
void runOnOperation() final {
  ModuleOp module = getOperation();
  if (path.empty()) { module.emitError(...); signalPassFailure(); return; }
  PlanCollector collector(...全部 pass option...);
  collector.collectAttentionSegments(module);
  collector.collectLowPrecisionLedger(module);
  collector.collectTuningLedger();
  collector.initPlanHash(model, targetTriple, ...);
  collector.collectFusionRecords(module);
  collector.collectModule(module);
  collector.collectCopyRecords(module);
  collector.collectPackedConstants();
  JsonObject root = collector.assembleRoot(collector.collectSummary());
  collector.writePlan(module, root);
}
```

**验收**：
- `runOnOperation` ≤ 80 行
- 同一组模型的 `.plan.json` **逐字节一致**（`diff` 为空）——用重构前基线 stage 生成
  `plan-before/`，重构后生成 `plan-after/`，见 §4
- plan-hash 字段白名单单测（roadmap 要求：改动未列入白名单的字段不应改变 hash）
- 完整 CTest + `format_check` + `tidy`

**风险**：低（纯位移）。唯一真实风险是 hash 输入顺序变化，已由逐字节 diff 覆盖。

---

### T-C2 拆 Python `perf_attribution_report`

**现状**：`tools/perf_attribution_report.py` 1588 行 / 14 个顶层定义，
`build_report`（246–1100，855 行）与 `aggregate_v2_reports`（1103–1548，446 行）
占 1301/1588 行。无模块级全局状态；**`aggregate_v2_reports` 原地改写 `reports[0]`**
（`runtime = first["runtime"]` 是别名），必须原样保留。

**方案**：拆到 `tools/attr/`，四个模块按 roadmap 命名：

| 模块 | 承载 | 来源行号 |
|---|---|---|
| `attr_schema.py` | `AttributionError`、`require_string`、`require_nonnegative_integer`、`validate_identity` | 20–176 |
| `attr_join.py` | `read_json`/`read_profile`/`read_perf`、`operation_index`/`allocation_index`/`allocation_metadata`、`category_name`，以及 `build_report` 的事件 join 段（G1–G12、H、I） | 24–243、352–629 |
| `attr_partition.py` | wall 分区（Z 段）、窗口归一、worker 份额与完整性（V–Y 段）、`top_costs`/`top_allocations` 排序 | 836–1044 |
| `attr_report.py` | 报告序列化（AA 段）+ `aggregate_v2_reports` 全部 | 1045–1548 |

`perf_attribution_report.py` 变成编排层：`build_report` 只做"调分节函数 + 汇总 dict"，
目标 ≤ 100 行；`main` 与 CLI 契约**逐字保留**（`check_perf_attribution.py` 断言了
11 条 stderr 子串与退出码）。

**必须处理的两个耦合点**（实测发现，roadmap 未列）：

1. `test/Native/check_attribute_whitelist.py` 的 `ALLOWED_SITES` **按 repo 相对路径 key**，
   现有条目是 `"tools/perf_attribution_report.py"`，允许 3 处 `ncnn.*` 字面量
   （`ncnn.model_execution_plan` / `ncnn.model_execution_profile` /
   `ncnn.model_performance_attribution`）。字面量随代码移动到 `tools/attr/attr_report.py`
   后，**必须同步新增该路径的白名单条目**，否则 `attribute-whitelist` 测试红。
2. `tools/attr/` 作为 `sys.path[0]`（脚本所在目录 `tools/`）下的包被 `import attr.*`。
   本机 **未安装 PyPI 的 `attr` 包**，无命名冲突；仍加 `tools/attr/__init__.py` 显式成包。

**验收**：
- `python3 -m pytest test/Native/test_attr_*.py` 通过（新增单测：wall 分区可加性
  `accounted + unaccounted = top_level`、窗口归一、`wall_projection_factor` 一致性、
  worker join 覆盖率）
- `ctest -R perf-attribution` 全绿，且 `check_perf_attribution.py` 的输出与重构前**逐字节一致**
- `ctest -R attribute-whitelist` 全绿（白名单已更新）
- `build_report` ≤ 100 行

**风险**：低，但这是 P27/P28 的 bug 高发区，拆完立即补单测是重点。

---

### T-C3 拆 `GenerateCAPI`

**现状**：`lib/Transforms/GenerateCAPI/GenerateCAPI.cpp` 2228 行，两个 pass 类挤一文件。

| 成员 | 行号 | 行数 |
|---|---|---:|
| `ArgumentInfo` / `InputDimRelation` / `abiElementType` / `isCIdentifier` | 36–95 | 60 |
| `GenerateCAPIPass::runOnOperation` | 102–152 | 51 |
| `GenerateCAPIPass::prepareDynamicRankABI` | 153–254 | 102 |
| `GenerateCAPIPass::prepareABI` | 255–624 | 370 |
| `GenerateCAPIPass::writeManifest` | 625–694 | 70 |
| `FinalizeCAPIPass::runOnOperation` | 700–1927 | **1228** |
| `FinalizeCAPIPass::finalizeDynamicRankABI` | 1928–end | ~300 |

**方案**：受 §1.3 的 tablegen 约束，pass 类不能跨 TU 拆成员函数，改为
「类各自独占 TU + 代码生成段抽成自由函数」：

```
GenerateCAPIInternal.hpp     # namespace mlir::ncnn::capi_detail：
                             #   InputDimRelation / FinalizeContext / 自由函数声明
GenerateCAPIPrepare.cpp      # class GenerateCAPIPass 全部（含 prepareABI 拆成
                             #   validateInputs / buildSignatures / emitDeclarations）
                             #   + ArgumentInfo / abiElementType / isCIdentifier
GenerateCAPIEmitWrapper.cpp  # 自由函数：wrapper 代码生成（F12–F24）
GenerateCAPIEmitInfer.cpp    # 自由函数：_infer_output_shapes 代码生成（F25）
GenerateCAPIFinalize.cpp     # class FinalizeCAPIPass（runOnOperation 变编排 +
                             #   finalizeDynamicRankABI）
```

`FinalizeContext` 的成员名与 `runOnOperation` 的局部名逐一相同（与 T-C1 的
`PlanCollector` 同一手法），使抽出的方法体保持逐字。

**CMake**：`lib/Transforms/GenerateCAPI/CMakeLists.txt` 的 `add_library(GenerateCAPI …)`
是**显式源列表**，必须补新增 `.cpp`。保持**单一 library target**（`lib/Pipelines/` 与
`bin/` 都按名字链接 `GenerateCAPI`），不新增 target。

**验收**：每 TU ≤ 700 行；生成的 `.h` 与 manifest **逐字节一致**；CTest 全绿。

> **执行结果中的偏离**：roadmap 改动点里的「`prepareABI` 拆成
> `validateInputs` / `buildSignatures` / `emitDeclarations`」**未做**——它与同一任务的
> 验收标准「每 TU ≤ 700 行」冲突（拆分后 `GenerateCAPIPrepare.cpp` 约 720 行）。
> 按验收标准优先取舍，`prepareABI` 保留原样（328 行）。详见 `task-diffs.md`。

**风险**：低。两个 pass 类有共享 helper，拆分时必须把共享部分放 `GenerateCAPIInternal.hpp`
或其中一个 TU 并在头文件声明，避免 ODR / 重复定义。

---

### T-C4 拆 `ncnn-compile::main`

**现状**：`tools/ncnn-compile.cpp` 3397 行，`main` 2085–3397（1312 行），共 **37 个语义块**。
文件其余部分结构良好（`read_manifest` 282 行、`write_header` 274 行、`write_harness`
254 行等已是合理粒度），**只需拆 `main`**。

**返回码契约（必须逐位保留）**：

| 退出码 | 含义 | 出现次数 |
|---|---|---:|
| `0` | 成功 | 1 |
| `1` | `fail(msg)`：校验 / IO / 审计失败，打印 `ncnn-compile: error: …` | 61 处 |
| 其他正整数 | **子进程退出码原样透传**（driver / opt / clang / nm / readelf / harness） | 20 处 |

**方案**：新增 `tools/CompileSession.{hpp,cpp}`，`class CompileSession` 持有跨块局部变量
（实测 40 余个：`param_path` / `bin_path` / `model_name` / `output_dir` / `output_exists` /
`emitted` / 7 个 `ToolResult` 与 6 个 path 引用 / `staging` / `effective_target_triple` /
`clang_target` / `target_args` / `isa_args` / `codegen_args` / `effective_threads` /
`vector_lanes` / `vector_scalable` / `vector_active` / `tuning` / `resolved_int8_target` /
`resolved_vector_math` / `vector_math_abi` / `vector_math_lanes` / `uses_libmvec` /
`sleef_archive` / 10 个 stage 路径 / `profile_runtime_source` / 5 个产物路径 /
`emit_execution_plan` / `codegen_identity*` / 3 个 plan-revision 串 / `uses_openmp` /
`uses_sleef` / `llvm_bitcode` / `optimization` / `manifest` / `has_dynamic_output` /
3 个 sanitizer 开关 / `capture_path` / `text` / `undefined` / `error`）。

**阶段方法的返回类型是 `int` 而非 roadmap 写的 `llvm::Expected<void>`**：
`llvm::Expected` 会把"子进程退出码透传"压成统一错误，改变 CLI 行为。改用
`int`（0 = 继续，非 0 = 原样作为 main 的返回值），语义逐位不变：

```cpp
class CompileSession {
 public:
  int run(int argc, char** argv);          // 编排，目标 ≤ 40 行
 private:
  int parseArguments(int argc, char** argv);   // B1–B5
  int resolveTools();                          // B6–B7
  int resolveTarget();                         // B8–B14（triple/OMP/vector/tuning/INT8/vector-math）
  int declareArtifacts();                      // B15
  int runPipeline();                           // B16–B24
  int emitABI();                               // B25–B26
  int linkAndAudit();                          // B27–B32
  int verifyExecution();                       // B33
  int publishOutputs();                        // B34–B37
};
```

**注意两处既有全局副作用，原样保留**：B13 会**写回全局 `cl::opt`**
`g_int8_kernel` / `g_int8_depthwise` / `g_int8_cast_chain`（12 处赋值），
后续 `build_codegen_identity` 与 linalg pipeline 选项读回它们。不做"改成传参"的
顺手重构。

**CMake 耦合（实测两处，roadmap 未列）**：
1. `tools/CMakeLists.txt:20` `add_executable(ncnn-compile ncnn-compile.cpp)` 是**显式源列表**，
   新增 `CompileSession.cpp` 必须补进去；
2. `test/Numerical/CMakeLists.txt:185` 模型 fixture 的 `DEPENDS` 含
   `${CMAKE_SOURCE_DIR}/tools/ncnn-compile.cpp`，需同步补 `CompileSession.cpp`，
   否则改动不会触发模型重编译。

**验收**：`main` ≤ 40 行；`ctest -R ncnn-compile-cli`（`check_ncnn_compile.py`）全绿；
`ctest -R squeezenet-shared-library` 全绿；CLI 行为（含返回码）完全一致。

**风险**：低。37 块中有 20 处"子进程返回码透传"，逐块平移时不得改成 `return fail(...)`。

---

## 3. 统一提交门禁（P2 收尾执行一次）

```bash
# 1. 全局静态检查（含非本次产生的问题，按结果修复）
cmake --build <stage> --target format_check --parallel
cmake --build <stage> --target tidy --parallel

# 2. 全新 /tmp stage，避免旧产物误导
mkdir /tmp/ncnn-compiler-stage-refactor-p2
cmake -S compiler -B /tmp/ncnn-compiler-stage-refactor-p2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=ON \
  -DCOMPILER_INSTALL_RPATH=ON \
  -DBUILD_TESTING=ON
cmake --build /tmp/ncnn-compiler-stage-refactor-p2 --parallel

# 3. 三个 numerical target
cmake --build /tmp/ncnn-compiler-stage-refactor-p2 \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel

# 4. 全量 CTest（451 registered，2 个既有 upstream INT8 skip 原样披露）
ctest --test-dir /tmp/ncnn-compiler-stage-refactor-p2 --output-on-failure

# 5. 提交前检查
git diff --check
git status --short
git diff --cached
git log --oneline -10
```

**验收失败不提交**；修复后重新从**全新 `/tmp` 目录**构建和测试，不用旧产物推断。

---

## 4. 逐字节对照方法（P2 的核心护栏）

「纯机械、不动语义」不能靠肉眼，靠 diff。在动手前用基线 stage 生成参照物：

```bash
# 重构前（/tmp/ncnn-compiler-stage-p2-baseline）
for m in squeezenet_v1_1 resnet18 yolov5n yolov5x_seg pp_ocrv6_tiny_rec; do
  ncnn-compile --param ... --bin ... --model-name $m \
    --output-dir /tmp/p2-before/$m --emit-manifest --emit-execution-plan
done
# 重构后同一组命令 → /tmp/p2-after/$m
diff -ru /tmp/p2-before /tmp/p2-after   # 必须为空
```

覆盖三类产物：`*.plan.json`（T-C1）、`*.h` + `*.json` manifest（T-C3）、
`.so` 的 `nm`/`readelf` 结构（T-C4，含 `check_ncnn_compile.py` 的
`static_artifact_baseline.json` 已有断言）。

---

## 5. 明确不做

- **不做"顺便优化"** —— 发现的性能线索进 backlog，不混入本期
- **不清理 `layout_island_hash_input` 死写** —— 见 §1.2
- **不改 `main` 的返回码契约** —— 不用 `llvm::Expected` 统一错误
- **不改 `aggregate_v2_reports` 的 `reports[0]` 原地改写语义**
- **不改任何 plan/profile 字段名与字段值** —— 不触发 identity 变化
- **不改性能默认值** —— `NCNN_MATMUL_PACKING=auto`、`NCNN_PACKED_CONV_DEPTHWISE=OFF`、
  `NCNN_NATIVE_INT8_PROFILE=OFF` 原样
- **不引新依赖** —— `tools/attr/` 仅用标准库

---

## 6. 回滚策略

| 情况 | 处理 |
|---|---|
| `.plan.json` / `.h` / manifest 逐字节 diff 非空 | **不提交**；定位位移错位点，改正后重新走 §3 |
| 任一 CTest 红 | 修复后换全新 `/tmp` 重建，不带病推进 |
| 性能基线漂移 >2% | **优先怀疑拆分破坏语义**，而非"顺带优化生效"；用 P27 基线逐模型定位 |
| 单任务改动引入数值回归 | `git revert` 该任务提交，重新设计后再做 |

---

## 7. 交付物

归档到 `compiler/docs/refactor/p2-2026-09-26/`：

```
PLAN.md                   # 本文件
task-diffs.md             # 四个任务的改动摘要
verification.log          # format_check + tidy + CTest + 数值 target 输出
plan-before-after.diff    # .plan.json / .h / manifest 逐字节对照结果
identity-changes.md       # identity 变化记录（预期：无）
perf-baseline-diff.md     # 与 P27 基线的性能对比 + 与 ncnn 的差距
```

**不把唯一证据留在 `/tmp`**（沿用 P18 §3.3 纪律）。
