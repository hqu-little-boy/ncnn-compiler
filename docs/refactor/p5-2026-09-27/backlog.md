# P5 遗留问题与重启前置

> P5 是重构（roadmap 21 任务）的最后一阶段。本文件记「做到这里为止」的
> 边界，以及将来重启需要先澄清的事。

---

## 1. T-M5 的 yolov5x_seg 验收用同族大模型等价验证

roadmap §4.5 的验收原文是「编译 yolov5x_seg 时能看到预算拒绝的 warning
且不失败」。**yolov5x_seg 模型不在本机模型仓库内**（`yolov5_v7.0_models/`
只有 resnet 系列，`ncnn_modelzoo/` 是 liteocr）。

**实测替代**：resnet18 / squeezenet_v1_1 / pp_ocrv6 系列（含
`pp_formulanet_plus_s_encoder_fp16` 等大模型）均出现
`warning: ncnn: packing_rejected_budget`，且全部编译 exit 0。

**若将来拿得到 yolov5x_seg**：跑一次 `ncnn-compile --emit-execution-plan`，
确认 stderr 有 `packing_rejected_budget`、exit code 0 即可闭环，不需要改代码。

## 2. `--warnings-as-errors` 缺一条 CLI 级端到端用例

现覆盖在 pass 级（`test/Transforms/PackStaticMatmulNCNN/warning-grading.mlir`
的 ERR 通道）。CLI 级（`ncnn-compile --warnings-as-errors` 经
`setenv` 下沉到 `ncnn-mlir-opt` 子进程）**靠机制等价保证，没有独立用例**：
下沉只是一行 `setenv`，且 lit 用 `env NCNN_WARNINGS_AS_ERRORS=1` 驱动同一
开关。

**要补的话**：在 `test/Native/check_ncnn_compile.py` 加一例——编一个会触发
`packing_skipped_small_shape` 的小模型，断言默认 exit 0 / 开关后 exit ≠ 0。
需要一个会触发报警的最小 `.param` fixture。

## 3. `pj_i64` 暂无调用点

roadmap §4.5 的 API 清单里有 `pj_i64`。实测 profile 的数值字段全是非负
（`bytes_known` 为假时走 `pj_null`，不会打印 `staticByteSize` 的 `-1` 哨兵），
所以 `pj_i64` 目前是**留作签名契约**、无生产调用点。
**不要**为了「用上它」把现有 `%llu` 站点改成 `pj_i64`——负值会从
`18446744073709551615` 变成 `-1`，那是产物变化。

## 4. 三个 share/projection 字段是浮点输出（roadmap §1 表述不准）

roadmap §4.5 写「数值字段全用 `%llu` 出整数、**完全不输出浮点**」。
实测 `flush_profile_v2` 有三处 `%.9f`：

* `wall_attributed_share_of_sampled_window`
* `wall_projection_factor`
* `wall_coverage_share`

它们是**既有**字段（不是 P5 新增），值域由调用侧约束为有限小数。
P5 把它们包进 `pj_f64`，**格式逐字节保持 `%.9f`**。
roadmap §8「不给 C runtime 加浮点 JSON 输出」仍成立——没有新增浮点字段。
把这三处改掉属于行为变更，需单开一期。

## 5. `instrumentMovementSite` 的插入点已显式化

历史上 `movement` 事件继承调用方插入点（即 after-op 若同 op 刚发过
`copy` 事件、否则 before-op）。现改为**显式 before-op**。触发分歧需要同一个
op 同时是 `memref.TransposeOp` 又带 `ncnn.copy_contract`——实测不可能
（`annotateCopy` 只由 `RewriteLinalgCopies` 打在 copy 根上）。
2 模型 `--profile` 逐字节对照已证无差异，但**这是 2 个模型的证据**，
不是穷举。若将来发现 transpose 上出现 `copy_contract`，见
`InstrumentNCNNSites.hpp` 的插入点契约注释。

## 6. T-M4 的行数不降反升（同 P4 的 T-C5）

`InstrumentNCNNProfile.cpp` 883 → 171 行，但 8 TU + 1 头的文档开销使总量
持平偏增。换到的是「改一处生效一处」与「新类别 = 新文件 + 一处调用」，
**不是行数**。roadmap 没给 T-M4 的行数指标，此处只作记录。

## 7. 报警未覆盖 `ncnn.workspace_fallback_reason`

`ReuseWorkspaceSlots::markFallback` 写的是 `ncnn.workspace_fallback_reason`
（另一个键），不是 roadmap 点名的 `fallback_reason`（= `ncnn.fallback_reason`）。
每个不可复用的 allocation 都会触发，报了就是噪音，**刻意没报**。
若将来要看，plan JSON 里 `workspace_fallback_reasons` 已经逐个记账。

---

## 重启前置

| 想做的事 | 先澄清 |
|---|---|
| 给 C runtime 加浮点字段 | §4 的三处既有浮点是否保留；NaN/Infinity 陷阱必须有测试 |
| 把 `pj_i64` 用起来 | 是否允许负值出现在 profile（会改产物） |
| `instrumentMovementSite` 回到继承插入点 | 先证伪「transpose + copy_contract 不可能」 |
| 给 `workspace_fallback_reason` 报警 | 先解决噪音（例如只报 `analysis_failed`） |
| 性能回归 | P5 无代码生成语义变更；若基线漂移 >2%，优先怀疑重构破坏语义（roadmap §9） |
