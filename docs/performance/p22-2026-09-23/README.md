# P22 fresh validation and performance evidence (2026-09-23)

## Scope and build stages

This batch compares the P22 working tree with the clean pre-P22 source at
`a5c3c57`. The clean baseline was built in
`/tmp/ncnn-compiler-p22-validation-baseline-20260923-a`; the candidate was built
in `/tmp/ncnn-compiler-p22-validation-candidate-20260923-b`. Both are fresh
Release builds with LLVM/MLIR 21, `BUILD_TESTING=ON`,
`COMPILER_INSTALL_RPATH=ON`, and `NCNN_PACKED_CONV_DEPTHWISE=OFF`.

Both stages passed the previously recorded global `format_check`/`tidy` gates
and full CTest (449/449; two known skips). Each of the five formal paired
performance rounds also completed successfully: 10 CTest runs, each with 47
PerformanceModel tests, 45 measured model rows, and the two known skipped INT8
models (`PPLCNetDocOriInt8`, `PPLCNetTextlineOriInt8`). The benchmark settings
are six threads, warmup 10, iterations 20, end-to-end mode. Rounds were ordered
B,C / C,B / B,C / C,B / B,C. All NDJSON rows were validated for unique model
identity, measurement status, mode, and timing metadata; each baseline/candidate
pair had the same 45-model set (44 official plus `resnet18_winograd`).

## Historical paired benchmark result (pre-postfix candidate)

The 44 official models have a median compiled-backend timing change of
`-4.19%` (candidate faster on 30/44 models); the ncnn-relative ratio p50 moves
from `1.878x` to `1.773x`. For 11 heavy models (baseline ncnn time at least
100 ms), the median timing change is `-6.86%` and 10/11 are faster.

The representative model outcomes are:

| Model | Candidate vs baseline median | Faster rounds | Gate |
|---|---:|---:|---|
| YOLOv5x | -6.46% | 5/5 | Misses ≥10% |
| YOLOv5x-seg | -4.16% | 5/5 | Misses ≥10% |
| EfficientNet-B3 | -4.64% | 3/5 | Misses ≥10% |
| FormulaNet+S encoder | -17.31% | 5/5 | Meets ≥10% |
| PP-OCRv6 medium-rec | -19.44% | 5/5 | Meets ≥10% |
| PP-OCRv5 server-det-static | +65.15% | 0/5 | Stable severe regression; investigate |

For the final candidate-vs-ncnn view, the official 44 models have ratio p50
`1.74x`, p90 `3.39x`, max `5.60x`; all 45 measured models have p50 `1.74x`,
p90 `3.44x`, max `5.60x`. The 11 heavy models have p50 `1.44x`, p90
`1.73x`, max `3.45x`.

## Gate status

**No-go; do not commit or defaultize P22.** The post-fix five-round paired
benchmark in `postfix-validation/` is now authoritative for the current source.
Across the 44 official models, the median compiled-time delta is `-5.37%`
(candidate faster on 34/44); the 11 heavy models improve by `-8.00%` median and
all 11 are faster. The server-det regression is fixed in this candidate and
changes from the historical `+65.15%` to `-57.72%` (faster in 5/5 rounds).
However, YOLOv5x (`-4.79%`), YOLOv5x-seg (`-3.02%`), and EfficientNet-B3
(`-3.03%`) still miss the required `≥10%` representative-model gate. FormulaNet+S
(`-27.40%`) and PP-OCRv6 medium-rec (`-19.61%`) meet it. Candidate NDJSON still
reports runtime counters as `not_collected`; the separate instrumented profile
diagnostic below is not a substitute for paired profile-on materialization
measurements.

### Runtime profile diagnostic (one instrumented invocation)

After the formal benchmark, a gap in fusion callback placement was fixed by
instrumenting selected fusion sites immediately after fusion, before later
canonicalization can erase a one-tile loop or tail op. A newly compiled
`pp_ocrv5_server_det_static` profile artifact (plan hash
`495060934037521121`) was executed once through its generated C ABI with a
zero-filled input. The report joins all 36 selected fusion records (45 main
and tail profile IDs; 36/36 records joined, no missing IDs). It records 13
materialized writes and 13 reads: both totals are `37,421,056` bytes, with
`materialized_read_complete=true`. The runtime allocation callbacks also prove
peak-live callback-accounted bytes of `5,604,382,264` for this invocation.

This is explicit-callback coverage, not a claim that every operation in the
model is instrumented. The profile's own event state is complete and has zero
mismatches, while the attribution report remains `complete=false` with
`runtime_exclusive_time_unknown` because OpenMP worker exclusive time is not
proven. The accompanying single-invocation `perf.ndjson` is marked diagnostic
and gate-ineligible; its instrumented elapsed time is not benchmark evidence.
The previous `runtime-profile/.../candidate/` artifacts are preserved as
historical evidence from the earlier instrumentation revision. New artifacts
are under `runtime-profile/pp_ocrv5_server_det_static/candidate-fusion-site-join/`.

A follow-up source review found that a fusion-site callback executing on an
OpenMP worker must not be treated as a root operation or added independently to
top-level wall time. Fusion callbacks now use a private event category that is
serialized as `operation` for attribution compatibility; a native pthread
regression and nested-`scf.forall` IR test cover the worker case. The new fresh
post-fix stage `/tmp/ncnn-compiler-p22-validation-final-20260923-d` passed the
full build, global `format`, `format_check`, `tidy`, and CTest (`449/449`, with
the two known INT8 skips). It also completed the fresh five-round paired
benchmark and a separate profile-on runtime diagnostic.

The post-fix candidate plan has 126 GEMM convolutions and 27 depthwise SIMD
convolutions for server-det, with no direct-convolution fallback. The paired
median compiled time is `4,007.55 ms` versus `9,478.96 ms` baseline (`-57.72%`);
this is separate from the gate-ineligible one-invocation profile measurement.
The post-fix profile plan hash is `9088708588622549195` and joins all 36 selected
fusion records (all 45 main/tail IDs). It records 13 materialized writes and 13
reads, each totaling `37,421,056` bytes, with `materialized_read_complete=true`;
callback-accounted peak live bytes are `5,599,139,384`. Profile event state is
complete with zero mismatches, but the attribution report remains incomplete
because OpenMP exclusive time is unknown. The profile is not a paired baseline
comparison and does not close the ≥20% materialization-reduction gate.

The earlier 65.15% server-det regression was traced to a large 9×9 convolution
falling back to a direct implementation after the multi-consumer fusion
rejection. The current Strategy path admits the static im2col/GEMM case (the
plan now selects GEMM for all 126 convolutions); this is the source of the
server-det recovery, while its materialization and peak-memory tradeoffs remain
subject to continued review. YOLOv5x/x-seg and EfficientNet-B3 still miss their
per-model target, and candidate NDJSON runtime counters remain
`not_collected`; P22 therefore stays **no-go**. No commit or push has been made.

## Artifacts

- `round-{1..5}-{baseline,candidate}.ndjson` and matching `.ctest.log`
- `paired-rounds.ndjson`, `paired-summary.json`, `paired-summary.txt`
- `final-all-model-ncnn-comparison.json` and `.ndjson`
- `commands.txt` records the original paired measurement configuration and aggregation command.
- `postfix-validation/` contains the fresh post-fix benchmark run (10 per-stage NDJSON/CTest logs, paired summary and final candidate-vs-ncnn views), reproduction script and runtime profile sidecar. Its separate `commands.txt` records the rerun commands.
