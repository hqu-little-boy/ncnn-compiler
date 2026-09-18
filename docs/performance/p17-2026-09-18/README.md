# P17 全量性能证据（2026-09-18）

这些文件来自全新 `/tmp/ncnn-compiler-stage-p17-release2` Release 构建，LLVM/MLIR 21，x86-64-v3（AVX2/FMA），6 个物理大核，编译产物默认 `stable` tuning profile。

- `stable-end-to-end.ndjson`：正式端到端口径，45 条 measured 记录（44 个正式模型 + 1 个 Winograd 诊断行）；另外 2 个 upstream ncnn 已知崩溃行按设计 skip。
- `stable-prepared.ndjson`：prepared 诊断口径，45 条 measured 记录，`gate_eligible=false`，不参与正式性能门禁。
- `stable-allocation-audit.ndjson`：allocation-audit 诊断口径，45 条 measured 记录；ncnn 分配计数/字节/峰值 live bytes 仅作为诊断，不参与正式 ratio 门禁。
- `stable-end-to-end.summary.txt`：由 `tools/perf_json_summary.py` 生成的端到端逐模型表和 aggregate 摘要。
- `stable-summary.json`：本批次的机器可读 aggregate、身份和代表模型摘要。

所有 NDJSON 行都保留 `target`、`plan_revision`、`plan_hash` 和 `build_identity`，可用于防止跨产物错误 join。prepared/allocation-audit 的 ratio 只用于诊断，不能与 end-to-end ratio 混写。

复现示例：

```bash
NCNN_PERF_JSON=/tmp/stable.ndjson \
  ctest --test-dir /tmp/ncnn-compiler-stage-p17-release2 \
  -L performance --output-on-failure
python3 tools/perf_json_summary.py /tmp/stable.ndjson --gate 48.0
```
