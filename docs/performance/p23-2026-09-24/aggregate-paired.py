#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
EXPECTED_MODELS = 45
DIAGNOSTIC = "resnet18_winograd"
OFFICIAL = {
    "squeezenet_v1_1", "resnet18", "resnet34", "resnet50", "resnet101",
    "yolov5n_cls", "yolov5s_cls", "yolov5m_cls", "yolov5l_cls", "yolov5x_cls",
    "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "pp_lcnet_x1_0_doc_ori", "pp_lcnet_x1_0_textline_ori",
    "chineseocr_lite_anglenet", "yolov5n_seg", "yolov5s_seg", "yolov5m_seg",
    "yolov5l_seg", "yolov5x_seg", "pp_ocrv6_tiny_rec",
    "pp_ocrv6_tiny_rec_int8", "pp_ocrv5_mobile_rec",
    "pp_ocrv5_mobile_rec_int8", "pp_ocrv5_server_rec", "pp_ocrv6_medium_rec",
    "pp_ocrv6_medium_rec_int8", "pp_ocrv6_small_rec",
    "pp_ocrv6_small_rec_int8", "pp_ocrv6_tiny_det", "pp_ocrv6_small_det",
    "pp_ocrv6_medium_det", "pp_ocrv6_medium_det_int8",
    "pp_ocrv5_mobile_det_static", "pp_ocrv5_server_det_static",
    "pp_structrurev2_slanet_plus_cnn", "pp_formulanet_plus_s_encoder",
}


def read(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_MODELS:
        raise SystemExit(f"{path}: expected {EXPECTED_MODELS} rows, got {len(rows)}")
    result = {}
    for row in rows:
        if row.get("status") != "measured" or row.get("mode") != "end_to_end":
            raise SystemExit(f"{path}: non-measured row for {row.get('model')}")
        if (row.get("threads"), row.get("warmup"), row.get("iterations")) != (6, 10, 20):
            raise SystemExit(f"{path}: timing metadata mismatch for {row.get('model')}")
        model = row.get("model")
        if not isinstance(model, str) or model in result:
            raise SystemExit(f"{path}: duplicate or invalid model {model!r}")
        result[model] = row
    return result


def median(values):
    return statistics.median(values)


def summary(rows):
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "baseline_compiled_median_ms_median": median([r["baseline_compiled_median_ms"] for r in rows]),
        "candidate_compiled_median_ms_median": median([r["candidate_compiled_median_ms"] for r in rows]),
        "compiled_delta_median_percent_median": median([r["compiled_delta_median_percent"] for r in rows]),
        "baseline_ratio_p50": median([r["baseline_ratio_median"] for r in rows]),
        "candidate_ratio_p50": median([r["candidate_ratio_median"] for r in rows]),
        "candidate_faster_model_count": sum(
            r["candidate_compiled_median_ms"] < r["baseline_compiled_median_ms"] for r in rows
        ),
    }


runs = {}
for round_number in range(1, 6):
    for variant in ("baseline", "candidate"):
        runs[(variant, round_number)] = read(ROOT / f"round-{round_number}-{variant}.ndjson")

expected = OFFICIAL | {DIAGNOSTIC}
for (variant, round_number), rows in runs.items():
    if set(rows) != expected:
        raise SystemExit(f"{variant} round {round_number}: unexpected model set")
    first = runs[(variant, 1)]
    for model, row in rows.items():
        if row.get("target") != first[model].get("target") or row.get("build_identity") != first[model].get("build_identity") or row.get("plan_hash") != first[model].get("plan_hash"):
            raise SystemExit(f"{variant} round {round_number}: build or target changed for {model}")

records = []
for round_number in range(1, 6):
    baseline = runs[("baseline", round_number)]
    candidate = runs[("candidate", round_number)]
    if set(baseline) != set(candidate):
        raise SystemExit(f"round {round_number}: model sets differ")
    for model in sorted(baseline):
        b = baseline[model]
        c = candidate[model]
        if not b.get("target") or b.get("target") != c.get("target"):
            raise SystemExit(f"round {round_number}: target differs for {model}")
        btime = float(b["compiled_mean_ms"])
        ctime = float(c["compiled_mean_ms"])
        if not math.isfinite(btime) or not math.isfinite(ctime) or btime <= 0 or ctime <= 0:
            raise SystemExit(f"round {round_number}: invalid timing for {model}")
        records.append({
            "round": round_number,
            "model": model,
            "baseline_compiled_mean_ms": btime,
            "candidate_compiled_mean_ms": ctime,
            "baseline_ncnn_mean_ms": float(b["ncnn_mean_ms"]),
            "candidate_ncnn_mean_ms": float(c["ncnn_mean_ms"]),
            "baseline_ratio": float(b["ratio"]),
            "candidate_ratio": float(c["ratio"]),
            "compiled_delta_ms": ctime - btime,
            "compiled_delta_percent": (ctime / btime - 1.0) * 100.0 if btime else math.nan,
            "ratio_delta": float(c["ratio"]) - float(b["ratio"]),
            "baseline_plan_hash": b.get("plan_hash"),
            "candidate_plan_hash": c.get("plan_hash"),
            "target": c.get("target"),
            "threads": c.get("threads"),
        })

(ROOT / "paired-rounds.ndjson").write_text(
    "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
)

models = sorted({record["model"] for record in records})
per_model = []
for model in models:
    entries = [record for record in records if record["model"] == model]
    base_times = [record["baseline_compiled_mean_ms"] for record in entries]
    candidate_times = [record["candidate_compiled_mean_ms"] for record in entries]
    base_ratios = [record["baseline_ratio"] for record in entries]
    candidate_ratios = [record["candidate_ratio"] for record in entries]
    base_median = median(base_times)
    candidate_median = median(candidate_times)
    per_model.append({
        "model": model,
        "rounds": len(entries),
        "baseline_compiled_median_ms": base_median,
        "candidate_compiled_median_ms": candidate_median,
        "compiled_delta_median_ms": candidate_median - base_median,
        "compiled_delta_median_percent": (candidate_median / base_median - 1.0) * 100.0 if base_median else math.nan,
        "baseline_ratio_median": median(base_ratios),
        "candidate_ratio_median": median(candidate_ratios),
        "ratio_delta_median": median(candidate_ratios) - median(base_ratios),
        "candidate_faster_rounds": sum(
            record["candidate_compiled_mean_ms"] < record["baseline_compiled_mean_ms"]
            for record in entries
        ),
    })

if len(per_model) != EXPECTED_MODELS:
    raise SystemExit(f"paired data has {len(per_model)} models, expected {EXPECTED_MODELS}")
if len([row for row in per_model if row["model"] in OFFICIAL]) != 44:
    raise SystemExit("official model set does not contain exactly 44 measured models")
official = [row for row in per_model if row["model"] in OFFICIAL]
heavy = [
    row for row in per_model
    if next(record for record in records if record["model"] == row["model"])["baseline_ncnn_mean_ms"] >= 100.0
]

output = {
    "schema_version": 1,
    "threads": 6,
    "warmup": 10,
    "iterations": 20,
    "paired_rounds": 5,
    "measured_rows_per_run": EXPECTED_MODELS,
    "run_count": 10,
    "official_model_count": 44,
    "diagnostic_model": DIAGNOSTIC,
    "known_skips": ["PPLCNetDocOriInt8", "PPLCNetTextlineOriInt8"],
    "official_44": summary(official),
    "all_45": summary(per_model),
    "heavy_models_ncnn_baseline_ge_100ms": summary(heavy),
    "models": per_model,
}
(ROOT / "paired-summary.json").write_text(json.dumps(output, indent=2) + "\n")

lines = ["P23 paired 6-thread summary", "runs=10 rounds=5 rows_per_run=45 warmup=10 iterations=20"]
for key in ("official_44", "all_45", "heavy_models_ncnn_baseline_ge_100ms"):
    item = output[key]
    lines.append(
        f"{key}: models={item['count']} "
        f"baseline_ratio_p50={item.get('baseline_ratio_p50', float('nan')):.3f} "
        f"candidate_ratio_p50={item.get('candidate_ratio_p50', float('nan')):.3f} "
        f"compiled_delta_median_percent={item.get('compiled_delta_median_percent_median', float('nan')):+.2f}% "
        f"candidate_faster_models={item.get('candidate_faster_model_count', 0)}"
    )
lines.extend([
    "",
    f"{'model':38} {'base_ms':>10} {'cand_ms':>10} {'delta%':>9} {'base_r':>8} {'cand_r':>8} {'faster':>6}",
])
for item in sorted(per_model, key=lambda row: row["compiled_delta_median_percent"]):
    lines.append(
        f"{item['model']:38} {item['baseline_compiled_median_ms']:10.2f} "
        f"{item['candidate_compiled_median_ms']:10.2f} {item['compiled_delta_median_percent']:9.2f} "
        f"{item['baseline_ratio_median']:8.3f} {item['candidate_ratio_median']:8.3f} "
        f"{item['candidate_faster_rounds']:2d}/5"
    )
(ROOT / "paired-summary.txt").write_text("\n".join(lines) + "\n")

# The final candidate round is the authoritative 45-row candidate-vs-ncnn table.
final_rows = runs[("candidate", 5)]
if set(final_rows) != set(models):
    raise SystemExit("final candidate round model set differs from paired data")
final_ndjson = ROOT / "final-all-model-ncnn-comparison.ndjson"
final_ndjson.write_text(
    "".join(json.dumps(final_rows[model], separators=(",", ":")) + "\n" for model in sorted(final_rows))
)
(ROOT / "final-all-model-ncnn-comparison.json").write_text(
    json.dumps([final_rows[model] for model in sorted(final_rows)], indent=2) + "\n"
)
