# P22 performance evidence

This directory is reserved for the P22 fresh-stage performance batch. The formal
口径 is 44 official models plus the `resnet18_winograd` diagnostic (45 measured
rows), six end-to-end threads, warmup 10, iterations 20, and five paired
A/B↔B/A rounds. `PPLCNetDocOriInt8` and `PPLCNetTextlineOriInt8` are explicit
skips and are not measured rows.

The candidate stage is `/tmp/ncnn-compiler-stage-p22-gate-20260922-a`; the clean
pre-P22 baseline is `/tmp/ncnn-compiler-stage-p22-baseline-20260922-a`. Both
fresh Release stages passed `format_check`, `tidy`, the three numerical targets,
and full CTest. Candidate CTest was `449/449` in 1584.47 s; baseline was
`449/449` in 1158.70 s. Runtime counters and materialized intermediate bytes
must remain `unknown` unless collected by the profile runner; static plan
selected counts and saved bytes are not performance evidence.

The five paired rounds completed successfully. The official 44-row final
candidate-vs-ncnn view is p50 `1.88x`, p90 `3.17x`, max `18.08x`, geometric
mean `2.18x`; its top five are `pp_ocrv5_server_rec=18.08x`,
`pp_formulanet_plus_s_encoder=8.40x`, `pp_ocrv6_small_rec_int8=5.00x`,
`pp_ocrv6_medium_rec_int8=4.93x`, and `pp_ocrv6_tiny_rec_int8=3.23x`. The
all-45 view including `resnet18_winograd` is p50 `1.89x`, p90 `3.60x`, max
`18.08x`, geometric mean `2.21x`; the 11 models with ncnn baseline at least
100 ms are p50 `1.80x`, p90 `2.09x`, max `18.08x`, geometric mean about
`2.06x`. Across the five candidate-vs-baseline rounds, the official-model
median compiled-time delta is `+1.34%`, with candidate faster on `11/44`
models; the heavy-model median delta is `+0.37%`. These results do not
satisfy the P22 defaultization milestone: the requested representative-model
`>=10%` E2E improvements were not demonstrated, and runtime
materialized-intermediate bytes remain `null`/`not_collected`. The candidate
therefore remains opt-in/no-go for default.

Expected raw artifacts are one unique NDJSON and CTest log per run:

- `round-{1..5}-{baseline,candidate}.ndjson`
- `round-{1..5}-{baseline,candidate}.ctest.log`
- `paired-rounds.ndjson`, `paired-summary.json`, `paired-summary.txt`
- `final-all-model-ncnn-comparison.json`
- `final-all-model-ncnn-comparison.ndjson`
- `build-identity.json`, `plan-index.json`, `static-plan-coverage.json`
- numerical, format, tidy, build, and full-CTest logs

The candidate remains opt-in/no-go for default until runtime profile coverage,
paired end-to-end regression gates, and the documented P22 milestone gates are
actually satisfied.
