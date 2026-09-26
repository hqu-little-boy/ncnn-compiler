# Identity 变化记录 — P2 巨型函数拆分

> **结论：本阶段无 identity 变化。** 所有 `.plan.json` / `.h` / manifest `.json`
> 的字段名、字段值、字段顺序均未改动，plan-hash 逐位不变，跨期 plan-hash join
> 可正常进行。

判据：identity 变化的定义是「plan/profile 字段增减」或「哈希算法更换」
（roadmap §7.5 / §7.6）。逐项核对：

| 任务 | 是否触碰 plan/profile 字段名 | 是否触碰字段值 | 是否换哈希 |
|---|---|---|---|
| T-C1 拆 `EmitModelPlan::runOnOperation` | 否 | 否 | 否（`fnv1a` 仍在原处） |
| T-C2 拆 Python `perf_attribution_report` | 否 | 否 | 否 |
| T-C3 拆 `GenerateCAPI` | 否 | 否 | 否 |
| T-C4 拆 `ncnn-compile::main` | 否 | 否 | 否 |

## 直接证据：逐字节对照

用同一组 9 个模型（含 2 个动态形状用例，覆盖 `finalizeDynamicRankABI`）分别经
重构前后的 `ncnn-compile` 产出 `.plan.json` / `.h` / manifest `.json`：

```bash
STAGE=/tmp/ncnn-compiler-stage-p2-baseline /tmp/p2-emit-plans.sh /tmp/p2-before
STAGE=<after>                        /tmp/p2-emit-plans.sh /tmp/p2-after
diff -ru /tmp/p2-before /tmp/p2-after --exclude='*.compile.log'
```

| 模型 | 覆盖点 | 结果 |
|---|---|---|
| `squeezenet_v1_1` | 小卷积网 | 逐字节一致 |
| `resnet18` | 深层卷积 | 逐字节一致 |
| `chineseocr_lite_anglenet` | 小模型 | 逐字节一致 |
| `yolov5n` | 检测，静态 3x640x640 | 逐字节一致 |
| `yolov5x_seg` | 大模型 | 逐字节一致 |
| `pp_ocrv6_tiny_rec` | OCR 识别 | 逐字节一致 |
| `pp_ocrv6_tiny_det` | OCR 检测 | 逐字节一致 |
| `pp_ocrv6_tiny_rec_dynamic` | **动态维 + dim 约束** | 逐字节一致 |
| `pp_ocrv6_tiny_det_dynamic` | **双动态维 + dim 约束** | 逐字节一致 |

`diff -ru` 输出为空（exit 0）。对照脚本 `/tmp/p2-emit-plans.sh` 固定参数与
模型清单，只换 stage 路径。

T-C2 侧另有独立的 CLI 逐字节对照：11 组输入（正常 / worker 归因 / 多次调用聚合 /
4 类错误路径）重构前后 stdout、stderr、退出码 **34 项全 IDENTICAL，0 项 DIFF**，
证据在 `/tmp/p2-t2-evidence/`。

## 逐项说明

### T-C1

`runOnOperation` 的语义体整体平移到 `PlanCollector`，**方法体逐字复用**
（成员名与原局部名一一对应，唯一的文本改动是把 10 个 tablegen pass option 的
`X.getValue()` 读法收敛为成员直接读，语义等价）。`plan_hash_input` 的拼接顺序
未动：

1. 初始装配（原 667–706）
2. fusion ledger 逐条（原 754 / 760 / 764）
3. 遍历内逐 op（原 1040–1047、1070–1076、1349）
4. copy ledger 逐条（原 1592 / 1608 / 1646）

`llvm::json::Object` 底层是 DenseMap、序列化按键名字母序，因此**对象键的插入顺序
不是承重的**；承重的是**数组元素顺序**与 **hash 拼接顺序**，两者均未变——这正是
`diff -ru` 为空所证明的。

原 `layout_island_hash_input` 在折叠进 `plan_hash_input` 之后仍有两次追加
（原 1247 / 1250），属**死写**、不影响 hash。为保持纯位移**原样保留**，
并在代码注释中标注；未做"顺手清理"。

### T-C2

`build_report` 与 `aggregate_v2_reports` 的分节被搬到 `tools/attr/`，
**副作用顺序逐条保留**——尤其是 `unknown = list(dict.fromkeys([*unknown, …]))`
的 16 处追加顺序，它决定 `static.unknown_fields` 的数组顺序。
`aggregate_v2_reports` 对 `reports[0]` 的原地改写语义（`runtime = first["runtime"]`
是别名）原样保留。

`check_attribute_whitelist.py` 的 `ALLOWED_SITES` 按 repo 相对路径 key，三处
`ncnn.*` 字面量随代码迁到 `tools/attr/attr_schema.py` / `attr_report.py`，
白名单条目同步迁移（这是测试脚手架，不是产物 schema）。

### T-C3

纯 TU 拆分。两 pass 类之间**没有任何共享 helper**，耦合是纯数据的
（GenerateCAPIPass 写 `contract::kCApi*` module 属性，FinalizeCAPIPass 读并清理），
因此拆分不触及任何序列化逻辑。`checkedMultiply` 在两处各有一份独立拷贝，
**按"不动语义"保留两份**，未去重。

### T-C4

`main` 的 37 个语义块平移到 `CompileSession` 的 9 个阶段方法，**返回码契约逐位保留**：
`0` 成功、`1` 走 `fail(msg)`、其余为子进程退出码原样透传。B13 对全局
`cl::opt`（`g_int8_kernel` / `g_int8_depthwise` / `g_int8_cast_chain`）的写回副作用
原样保留，未改成传参。

## 需要注意的产物差异（非 identity）

**本阶段无产物差异。** P1 的 T-S1（batchnorm 折叠范围扩大）已经改变了
`.plan.json` 的 `plan_hash` 与生成 `.so` 的数值路径；那是 P1 的预期路径，
不是 P2 引入的。P2 之后的 plan-hash 与 P1 之后**完全一致**。
