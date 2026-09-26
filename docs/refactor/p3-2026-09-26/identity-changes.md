# P3 identity 变化记录

本文件记录 P3 中**会改变产物 identity 字段取值**的改动。identity 变化的纪律是
（roadmap §7.6）：必须记录、必须停止跨期 join，**不得靠「修 join 让它通过」**。

---

## 1. T-S5：哈希算法 FNV-1a → `llvm::xxHash64`（**本阶段唯一 identity 变化**）

### 1.1 变了什么

手写 FNV-1a（offset basis `14695981039346656037`、prime `1099511628211`）
合并为 `ncnn_mlir::stableHash64`，内部调 LLVM 21 的 `llvm::xxHash64(StringRef)`
（**无 seed 参数**，符号来自 LLVMSupport）。

**三处调用点必须锁步改**（改一处就会让归因静默丢数据）：

| 文件 | 原函数 | 产出 |
|---|---|---|
| `lib/Transforms/InstrumentNCNNProfile/InstrumentNCNNProfile.cpp` | `fnv1a`（已删除） | profile JSON `events[].id` |
| `lib/Transforms/EmitModelPlan/EmitModelPlan.cpp` | `profileId`（已删除） | `operations/buffers/static_liveness[].profile_id`、`plan_hash`、`build_identity` |
| `lib/Transforms/FuseLinalgEpilogue/FuseLinalgEpilogue.cpp` | `fusionProfileId`（保留函数，仅换哈希内核） | IR `ncnn.fusion_site_id`（保留 `& 0x7fffffffffffffff` 正号掩码） |

**锁步不变量**：`plan.operations[].profile_id` 必须等于 profile 事件 `id`。
前两者哈希的是**同一输入串** `<funcName>/<opKind>#<n>`（`tools/attr/attr_join.py`
用它做 `operations.get(identifier)`）。第三处的结果写进 IR 属性后被前两者
**读取而非重算**，所以它改了之后 plan 与 profile 两侧同步变值，同轮内 join 仍成立。

### 1.2 变值了哪些字段

实测（`squeezenet_v1_1`，编译命令见 `emit-plans.sh`）：

| 字段 | 改前（T-A1/T-A3 稳定窗口） | 改后（T-S5） |
|---|---|---|
| `plan_hash` | `12677497241874684592` | `1771926188015674330` |
| `build_identity` | `12677497241874684592` | `1771926188015674330` |
| `operations[0].profile_id` | `14712359705534835900` | `10762451831299475511` |
| `operations[1].profile_id` | `14712360805046464111` | `5007957240008162491` |
| `operations[2].profile_id` | `14712361904558092322` | `6337618360253299886` |

全量变值的字段：`plan_hash`、`build_identity`、`operations[].profile_id`、
`buffers[].profile_id`、`static_liveness[].profile_id`、
`fusions[].profile_id` / `tail_profile_id`、profile JSON `events[].id`。

**不变**：字段名、`schema_version`、`build_identity == plan_hash` 的关系、
`plan_hash` 的输出格式（`std::to_string(uint64)` 十进制字符串）。

### 1.3 停止跨期 join

**从 P3 边界起，禁止用 `plan_hash` 与 P3 之前的任何 profile/perf 记录 join。**

`tools/perf_json_summary.py` 在同一 NDJSON 内遇到 `build_identity` 冲突会直接
`ValueError`——这是**预期行为**，正是「停止跨期 join」的强制点，**不改它**。
若要比较 P3 前后的性能数据，必须**重新生成两侧的 profile 与 plan**，
而不是让 join 兼容两个哈希世代。

### 1.4 逐项核对：既有护栏为何不被打破

| 护栏 | 受影响 | 原因 |
|---|---|---|
| `test/Native/check_plan_hash_whitelist.py` | 否 | 只扫 `plan_hash_input` 的累加站点与字符串片段，不断言 hash 数值 |
| `test/Transforms/EmitModelPlan/plan_hash_whitelist_check.py` | 否 | 只做相对断言（path-only 不翻转 hash / threads 改变必须翻转 / `build_identity == plan_hash`） |
| `tools/attr/attr_schema.py::validate_identity` | 否 | 只做 plan/profile/perf 三方 hash **字符串相等**比较 |
| `test/Native/check_ncnn_compile.py` | 否 | 相对关系：`build_identity == plan_hash`；codegen 变体 hash 互异；tuning profile 互异 |
| `test/Native/check_int8_target.py` | 否 | 5 个 plan hash 互异；profile 字段等于 plan 的 |
| 硬编码 hash 数值的测试 | **无** | 全 `test/` 无针对编译器产物 hash 值的字面量断言 |
| `test/Native/check_perf_attribution.py` 的 python `fnv1a64` | 否 | **合成 fixture 生成器**，自产自校，不消费编译器输出。保持 FNV-1a，是「合成 ID，不必镜像编译器算法」 |
| `test/Numerical/support/performance_test_support.cpp` 的 `performance_input_hash` | 否 | **另一个哈希**（浮点位模式 → `input_hash` 字段），语义不同，**不动** |
| `tools/perf_json_summary.py` 的 build_identity 冲突守卫 | **会触发（预期）** | 见 §1.3 |

### 1.5 对 roadmap 的勘误

roadmap §4.3 说「**两处**手写 FNV-1a 删除」，实测**三处**——
`FuseLinalgEpilogue.cpp` 的 `fusionProfileId` 是第三份同一套常量的拷贝，
roadmap 的审计漏列。本阶段三处一并合并。

roadmap §4.3 还要求「更新 `perf_attribution_report.py` 的 identity 校验白名单」——
**该文件不存在任何 whitelist**（`grep whitelist tools/` 为空）。`validate_identity`
只校验三方字符串相等，无白名单可更新。属 roadmap 勘误，无需改动。

---

## 2. T-A3：`m_Constant` 的行为加宽（**产品路径零变化，但契约确实变宽**）

`findConstantElements` 的末端从 `dyn_cast<arith::ConstantOp>` 换成
`matchPattern(value, m_Constant(&attr))`。

### 2.1 加宽了什么

`m_Constant` 匹配 `OpTrait::ConstantLike` 并走 `op->fold()`，**不限于**
`arith.constant`。安装头文件中带 `ConstantLike` 的算子至少包括
`arith.constant`、`ncnn.const`、`tosa.const`。

**旧代码**：只接受 `arith.constant`，遇到 `ncnn.const` / `tosa.const` 返回 null。
**新代码**：接受它们（fold 结果是 `ElementsAttr` 时）。

### 2.2 为什么判定为「产品路径零变化」

两个调用点在产品流水线里只会见到 `arith.constant`：

- `FoldNCNNBatchNorm` 跑在 `ConvertNCNNModelToFunc` 之后，该 pass 已把
  `ncnn.const` 重写为 `arith.constant`；
- `PackStaticMatmulNCNN` 跑在 `VerifyNoTosaOps` 之后，`tosa.const` 不可能存在。

**证据**：10 个模型 × 7 种产物（含 `.so` 与 IR dump）逐字节对照，
归一化后 **0 行差异**（`plan-before-after.diff` 为空）。

### 2.3 明确说清楚的边界

- 该加宽对**独立 `ncnn-mlir-opt` 使用**成立（手写 IR 里放 `tosa.const` 也能被查到）。
- 加宽**没有测试覆盖**：现有 lit 只锁了 `extract_slice` 不被穿透
  （`constant-through-cast.mlir::does_not_cross_extract_slice`）；
  **没有 `from_elements` 负向用例**，也没有 ConstantLike 加宽的正向用例。
  缺口记在 `backlog.md` §7。
- `ConstantFold.hpp` 的契约注释已从「`arith.constant`」改写为
  「ConstantLike 常量」，并写明加宽范围。

### 2.4 视图穿透集合未变

仍是 `tensor.cast` / `tensor.collapse_shape` / `tensor.expand_shape`，
明确排除 `tensor.extract_slice` / `tensor.from_elements`。**视图 walk 在末端之前**，
所以即使某 view op 带 `ConstantLike`，也仍按 view 穿透而不是被 fold。

---

## 3. T-A1：**无 identity 变化**

字符串分派改写是等价改写，报错串逐字不动。证据见 `plan-before-after.diff`
（与 T-A3 同一次逐字节对照覆盖）。
