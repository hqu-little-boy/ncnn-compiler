# P24 qualification and current ncnn comparison (2026-09-25)

## Decision

**P18 does not prove that allocation/fixed overhead is a top runtime hotspot. P24 production workspace implementation is therefore no-go/not started.** The new profile-on qualification also did not pass the predeclared gate: allocation duration is observed for only a small fraction of allocation calls, deallocation duration is not instrumented, and instrumentation perturbs direct-call timing substantially. These data do not prove that allocations are never important; they mean the project does not yet have complete enough evidence to justify the production allocator/workspace change.

No production compiler source was changed. No commit was made and nothing was pushed. A full current-vs-ncnn snapshot was still run as requested; it is not a P24-vs-baseline improvement claim.

## Why the P18 entry condition is unmet

- P18's official end-to-end NDJSON recorded `runtime_counters=not_collected`; the 45-model static-plan aggregation has zero nonzero workspace slots and zero proven peak workspace values. The `relu` sidecar is a profile contract fixture, not a target model. See the requested P18 plan §5.7 and `../p18-2026-09-20/README.md` / `static-plan-coverage.json`.
- P17 server-rec prepared vs end-to-end timing can address setup sensitivity, but neither row collected compiled allocation counters. P17's allocation-audit counts/bytes were ncnn-side only (`compiled_profile_not_collected`).
- P22 server-det's one profile-on diagnostic had callback-accounted peak-live memory around 5.6 GB but incomplete overall attribution (OpenMP exclusive time unknown), no equivalent paired baseline, and no allocation-time share. A peak byte count is not a timing hotspot.

## Additional P24 allocation qualification

`qualify-allocations.py` rebuilt six frozen P18 targets with profile instrumentation and ran two warmups plus five timed direct-ABI calls on all-zero diagnostic input. Profile-off generated model libraries were timed separately to quantify profiling perturbation. All six profile streams reported `complete=true`, zero event mismatches and a proven logical peak; those flags do **not** make allocation-time attribution complete.

| model | allocs / call | requested bytes / call | peak logical bytes | allocation calls with time | measured allocation time | measured time share | site rank | profile perturbation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pp_ocrv5_server_rec` | 24,865 | 220.98 MiB | 107.00 MiB | 0.776% | 0.587 ms | 0.107% | 4 | +10.4% |
| `pp_ocrv5_server_det_static` | 255,307 | 7,715.92 MiB | 5,344.76 MiB | 0.127% | 3.409 ms | 0.072% | 10 | +8.7% |
| `pp_formulanet_plus_s_encoder` | 76,276 | 538.97 MiB | 235.40 MiB | 0.174% | 0.572 ms | 0.363% | 9 | +23.3% |
| `chineseocr_lite_anglenet` | 6,626 | 3.21 MiB | 2.41 MiB | 1.509% | 0.042 ms | 0.482% | 5 | +85.7% |
| `pp_ocrv6_tiny_rec` | 8,572 | 17.72 MiB | 15.12 MiB | 1.073% | 0.040 ms | 0.282% | 31 | +79.2% |
| `squeezenet_v1_1` | 25,605 | 50.46 MiB | 30.19 MiB | 0.344% | 0.043 ms | 0.155% | 11 | +190.4% |

“Requested bytes” is cumulative logical allocator request volume per invocation, not hardware traffic; peak is logical live bytes, not RSS. The compiler pass brackets/times only a subset of `memref.alloc` sites. Deallocation callbacks report counts/bytes but no elapsed time, so deallocation duration remains **unknown** (not zero). The timed allocation subset is below 1.51% in every model; its measured share is below 0.5% of top-level time, and no top-three target satisfies the predeclared ≥5%/top-five hotspot criterion. The large profile perturbations, especially on small models, also prohibit treating profile-on timing as formal performance.

Because allocation-time coverage is incomplete and deallocation duration is unmeasured, this is **not evidence that all allocation cost is small**. It is a failed qualification gate: no production workspace or allocator change is justified by the current evidence. A future P24 attempt must first instrument all allocation and deallocation lifetimes with complete thread-aware timing/joins and acceptable profile perturbation, then rerun the gate.

## Fresh engineering validation

- Stage: `/tmp/ncnn-compiler-stage-p24-qualification-20260925-01` (new Release build, LLVM/MLIR 21.1.8, `COMPILER_INSTALL_RPATH=ON`, `BUILD_TESTING=ON`). All build commands used `--parallel 8`.
- Global `format_check` and global `tidy` passed; tidy visited 90 first-party translation units.
- Complete build and `numerical_tests`, `numerical_dynamic_operator_tests`, `numerical_dynamic_tests` passed.
- Full CTest: **449 registered, 447 passed, 0 failed, 2 skipped** in 773.19 s. The only skips were the pre-existing upstream INT8 reference crashes `PerformanceModel.PPLCNetDocOriInt8` and `PerformanceModel.PPLCNetTextlineOriInt8`; no new skip was added.
- Detailed build, numerical, CTest, format and tidy logs are archived as `*.log.gz`; `numerical-results.json` summarizes the result.

## Fresh profile-off comparison against vendored ncnn

The current `05be0ae` source was measured on the i5-12600KF host using x86-64-v3/AVX2/FMA, 6 threads, `end_to_end`, 10 warmups and 20 timed iterations. Vendored ncnn revision: `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`. This is one fresh full-table run, not a paired P24 code-change claim.

| subset | models | ratio p50 | ratio p90 | max | geometric mean | ncnn total | compiled total | excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official | 44 | 1.766x | 2.722x | 4.900x | 1.809x | 8,642.743 ms | 10,243.633 ms | 1,600.890 ms |
| heavy (`ncnn ≥100 ms`) | 11 | 1.282x | 1.650x | 1.720x | 1.299x | 7,780.935 ms | 8,702.294 ms | 921.359 ms |
| measured incl. diagnostic | 45 | 1.79x | 3.33x | 4.90x | — | — | — | — |

Two official models were faster than ncnn in this run: server-rec (0.95x) and server-det-static (0.89x). Twelve of 44 official models were ≤1.5x. Largest positive absolute excesses were YOLOv5x-seg (+419.807 ms), YOLOv5x (+292.695 ms), YOLOv5l-seg (+246.488 ms), PP-OCRv6 medium-det (+172.578 ms) and YOLOv5l (+159.578 ms). The P18 top-excess list has shifted on the current compiler revision; P24 allocation attribution is not used to explain those remaining gaps.

The final all-model comparison is retained losslessly in:

- `final-all-model-ncnn-comparison.json` (45 rows; includes the diagnostic row)
- `final-all-model-ncnn-comparison.ndjson` and `candidate-end-to-end.ndjson`
- `official-summary.txt`, `all-measured-summary.txt`, `comparison-summary.json`

The two known upstream skips have no fabricated rows/ratios. Formal profile-off comparison data is separate from profile-on diagnostic sidecars.

## Reproduction and artifacts

- `build-identity.json`, `numerical-results.json`
- `qualify-allocations.py`, `normalize-allocation-summary.py`, `allocation-qualification.json`
- `profile-on/<model>/compile.log`, `profile.ndjson`, and five timed invocation JSON sidecars
- `finalize-end-to-end.py`, `summarize-end-to-end.py`
- `formal-end-to-end.log` and compressed validation logs

Re-run the formal all-model measurement from the fresh stage with:

```bash
NCNN_PERF_THREADS=6 NCNN_PERF_WARMUP=10 NCNN_PERF_ITERS=20 \
NCNN_PERF_MODE=end_to_end NCNN_PERF_MAX_RATIO=0 \
NCNN_PERF_JSON=/path/to/candidate-end-to-end.ndjson \
ctest --test-dir /tmp/ncnn-compiler-stage-p24-qualification-20260925-01 \
  -R '^PerformanceModel\.' --output-on-failure
```

The requested plan file is `/mnt/ncnn-compiler/docs/ncnn-mlir-performance-optimization-plan-p18.md`, outside the only Git repository (`compiler/`). Per the user's choice it is updated there but cannot be part of a compiler-repository commit. The pre-existing untracked P22 evidence directories remain untouched and excluded.
