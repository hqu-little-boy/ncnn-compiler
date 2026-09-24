# P22 staged engineering acceptance (2026-09-24)

This stage deliberately separates an engineering submission from the larger P22 performance milestone in `docs/ncnn-mlir-performance-optimization-plan-p18.md` §9.3. The user authorized relaxing *optimization* targets, not correctness. Changing this stage's bar does **not** establish the original ≥10% improvements for all representative groups, ≥90% hot-epilogue coverage, or ≥20% paired runtime-materialization reduction. Those remain unverified until separately measured and must not be reported as passed.

## Predeclared acceptance for this submission

1. A new `/tmp` Release build of the exact candidate source completes. The full CTest suite, all numerical targets and numerical CTest tests, P22 positive/negative lit cases, ABI-sensitive integration tests, global `format_check`, global `tidy`, and `git diff --check` pass. Existing expected skips must be enumerated; new skips, numerical mismatches, crashes, failed tests, or compiler diagnostics are a hard stop. Every `cmake --build` uses `--parallel`. After a source-level fix to any failed acceptance run, use a different new `/tmp` build directory and rerun the full gate.
2. Against clean pre-P22 `a5c3c57`, perform the same five-round, alternating B,C / C,B / B,C / C,B / B,C, six-thread, warmup-10, iterations-20, end-to-end benchmark. Require 45 unique measured rows per run and matching row sets and timing metadata. The *median of the 44 official per-model paired median deltas* must improve by at least 2%; at least 23/44 official models must improve, and the 11-model heavy subset's median delta must be nonpositive. Every regression is disclosed, including repeatability, absolute time, and whether a representative group missed the original target. Any model with a new crash, wrong result, unexplained skip, or a slowdown both >25% and >15 ms blocks this stage; the original stricter stable/default regression gate remains a separate qualification.
3. Profile-on diagnostics must preserve their own build/plan identities, report unknown as unknown, and pass fixture-level read/write-completeness checks. No static selected/saved bytes, unmatched diagnostic profile, or profile-on timing may be substituted for the profile-off end-to-end benchmark. Paired runtime-materialization savings are **not** claimed unless both arms have complete, equivalent instrumentation and matched invocation evidence; a missing profile comparison is reported as an open milestone, not as a zero-byte result.
4. Keep the public typed bare-pointer C ABI unchanged and P21 packed Conv/Depthwise disabled by default. List the exact compiler options and whether they are actually on by default; do not call a default-enabled code path opt-in. If the stated engineering conditions do not pass, do not submit a commit.

The original §9.3 and §3.4.1 remain the criteria for declaring the P22 performance milestone and stable/default product qualification. This narrower submission, if it passes, is explicitly **not** that declaration.

## Fresh engineering and correctness gates

- Clean pre-P22 baseline: `a5c3c57`, source `/tmp/ncnn-compiler-p22-baseline-source-20260924-b`, complete Release build `/tmp/ncnn-compiler-p22-baseline-build-20260924-c`. Its archived source uses the unchanged local ncnn submodule at `a4d2ea1d` via a symlink; `git archive` alone does not include submodule contents.
- Current P22 candidate: complete Release build `/tmp/ncnn-compiler-p22-candidate-build-20260924-b`, `COMPILER_INSTALL_RPATH=ON`, `BUILD_TESTING=ON`, `NCNN_PACKED_CONV_DEPTHWISE=OFF`. All build commands used `--parallel 8`. The candidate's `numerical_tests`, `numerical_dynamic_operator_tests`, and `numerical_dynamic_tests` targets completed. Its **entire** CTest suite passed: 449 registered tests, zero failed, 758.27 s; only `PerformanceModel.PPLCNetDocOriInt8` and `PerformanceModel.PPLCNetTextlineOriInt8` were skipped, as in the existing test configuration. Full CTest, configuration, build, numerical-target, and global style/tidy logs are preserved losslessly under `validation/*.log.gz` (original full CTest log: `/tmp/p22-candidate-20260924-b-full-ctest.log`).
- Global first-party `format_check` and `tidy` both passed on this fresh candidate; tidy visited 90 translation units. The complete lit suite passed as part of CTest, and the additional focused P22 fusion/copy/profile/parallel lit runs passed (4 + 18 cases). Runtime attribution, profile-runtime, native ABI, numerical model, and CLI tests are included in the full CTest result. These checks establish the measured test coverage, not correctness for arbitrary untested models or shapes.

## Five-round profile-off end-to-end comparison

`formal-paired/` contains all ten 45-row arm NDJSON files, all 30 serial chunk logs and rows, the 225 paired model-round records, the strict aggregate, and the final candidate-vs-ncnn JSON/NDJSON. The prescribed order was B,C / C,B / B,C / C,B / B,C. Each arm used six threads, ten warmups, twenty timed iterations and `end_to_end` mode; all ten sets contain the same 44 official models plus the `resnet18_winograd` diagnostic. Every `rest` chunk has exactly the two known skips; no run failed or introduced another skip. The aggregator checks model sets, target, plan hash, build identity, timing metadata, and finite positive compiled durations.

| Subset | Median per-model compiled-time delta vs clean baseline | Models faster | Stage bar |
| --- | ---: | ---: | --- |
| 44 official | **−5.85%** | **36/44** | ≤−2%; ≥23/44 |
| 11 heavy (baseline ncnn ≥100 ms) | **−10.08%** | **11/11** | ≤0% |
| 45 including diagnostic | −5.84% | 36/45 | diagnostic only |

Deltas above are the median across models of `(median of five candidate compiled means / median of five baseline compiled means − 1)`, not the single fastest run. The complete per-model table, including repeatability as faster rounds out of five, is in `formal-paired/paired-summary.txt` and `.json`. The final candidate-vs-ncnn **official** view has ratio p50 `1.78×`, p90 `3.33×`, max `5.14×`; the 11-heavy view has p50 `1.31×`, p90 `1.68×`, max `1.69×`. These ratios are not the candidate-vs-baseline percentage improvements.

Every model with a positive five-round median delta is listed here, including the non-official diagnostic; the absolute differences compare the per-arm five-round medians:

| Model | Delta | Candidate minus baseline | Faster rounds |
| --- | ---: | ---: | ---: |
| `chineseocr_lite_anglenet` | +64.77% | +1.64 ms | 0/5 |
| `pp_ocrv6_medium_det_int8` | +19.54% | +1.39 ms | 0/5 |
| `resnet50` | +16.09% | +7.89 ms | 0/5 |
| `resnet101` | +13.35% | +9.89 ms | 0/5 |
| `pp_ocrv6_tiny_rec` | +12.33% | +0.96 ms | 1/5 |
| `resnet18_winograd` (diagnostic) | +10.89% | +5.88 ms | 1/5 |
| `pp_ocrv5_mobile_rec_int8` | +5.32% | +1.74 ms | 3/5 |
| `yolov5n_cls` | +4.28% | +0.32 ms | 2/5 |
| `resnet18` | +1.08% | +0.29 ms | 2/5 |

No model's **five-round median** regression exceeds both 25% and 15 ms. One individual `resnet101` pair in round 5 *does*: +25.64%, +18.57 ms; the other four pairs are below those bounds, while all five pairs remain slower. This fluctuation is disclosed rather than dropped or substituted with an extra run. The predeclared per-model regression gate is evaluated on five-round medians; the stricter stable/default qualification remains open.

Representative improvements are `yolov5x` −6.20%, `yolov5x_seg` −3.74%, `efficientnet_b3` −4.27%, `pp_formulanet_plus_s_encoder` −71.81%, and `pp_ocrv6_medium_rec` −18.29%. Thus YOLOv5x/x-seg and EfficientNet-B3 **miss** the original §9.3 ≥10% representative target even though this stage's narrower aggregate bar passes. In particular, the final candidate's YOLOv5x static plan reports **zero** selected fusion sites: 89 `repeated_producer_input`, 28 `producer_not_flowing_input`, and 9 `unsupported_consumer` rejections. Its speedup is not evidence that those hot epilogues were fused; the conservative repeated-input guard avoids the earlier measured YOLO slowdown. Top-five hot-site ≥90% runtime coverage has not been established. The baseline lacks equivalent instrumentation for a paired comparison of `materialized_intermediate_bytes`; neither missing baseline callbacks nor `not_collected` profile-off rows mean zero bytes. Consequently the original ≥20% paired materialization goal is **open, not passed**.

`ncnn-compile` / pipeline defaults are actually `selective-fusion=true`, `selective-fusion-broadcast=true`, `selective-fusion-cast-chain=true`, `layout-aware-fusion=true`, and `selective-fusion-residual=true`; `--profile` and `profile-materialized-sites` are off by default, `packed-conv-depthwise=false`, and the CLI `vector-mode=off` unless a fixture explicitly requests fixed-width vectorization. The formal model fixtures requested fixed-width vectorization. These measured P22 fusion paths are **default-enabled**, not opt-in. Passing this narrower engineering stage does not establish §9.3 or the separate §3.4.1 stable/default product gate.

## Separate runtime profile diagnostic

The separate `--profile` build for `pp_ocrv5_server_det_static` has plan hash and build identity `11284184568237843404`, versus `9362369434426107444` for its fifth-round **profile-off** candidate. A single zero-filled-input, six-thread direct-ABI invocation returned status 0. `runtime-profile/pp_ocrv5_server_det_static/` archives the matching plan as lossless `*.plan.json.gz` (decompress before using it with the report tool), manifest, profile NDJSON, diagnostic perf NDJSON, attribution report, and compiler log; `profile-serverdet-once.py` reproduces the invocation after recompiling, without archiving the 84 MiB generated library. Its diagnostic timing is expressly **not** a formal end-to-end benchmark observation.

This invocation joins **36/36** selected fusion records (no missing or partial sites), observes 13 producer-output materialized writes and 13 corresponding reads, each totalling **37,421,056 bytes**, and reports `materialized_read_complete=true` and zero event mismatches. It also observes 12,800 copy events totalling **1,638,400 bytes** and callback-accounted peak live memory of **5,607,003,704 bytes**. These are measured callback totals *for this profiled invocation only*; they are neither whole-model hardware traffic nor a baseline-to-candidate reduction. The attribution report's overall `runtime.complete=false` is explicitly due to `runtime_exclusive_time_unknown` on OpenMP workers; it must not be re-labelled complete because the materialization-read and fusion-site subchecks are complete. The clean pre-P22 compiler cannot produce equivalent fusion/materialization callbacks, so the original §9.3 paired ≥20% materialized-byte goal is not demonstrated.
