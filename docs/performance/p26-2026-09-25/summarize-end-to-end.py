#!/usr/bin/env python3
"""Summarize retained P26 model-vs-ncnn end-to-end JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[dict]) -> dict:
    ratios = [float(row["ratio"]) for row in rows]
    ncnn_total = sum(float(row["ncnn_mean_ms"]) for row in rows)
    compiled_total = sum(float(row["compiled_mean_ms"]) for row in rows)
    return {
        "models": len(rows),
        "ratio_p50": percentile(ratios, 0.5),
        "ratio_p90": percentile(ratios, 0.9),
        "ratio_max": max(ratios),
        "ratio_geometric_mean": math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)),
        "ratio_le_1_0_count": sum(value <= 1.0 for value in ratios),
        "ratio_le_1_5_count": sum(value <= 1.5 for value in ratios),
        "ncnn_total_ms": ncnn_total,
        "compiled_total_ms": compiled_total,
        "total_excess_ms": compiled_total - ncnn_total,
    }


def main() -> int:
    path = Path(sys.argv[1])
    rows = json.loads(path.read_text(encoding="utf-8"))
    official = [row for row in rows if row["model"] in OFFICIAL]
    if len(official) != 44:
        raise SystemExit(f"expected 44 official models; got {len(official)}")
    heavy = [row for row in official if float(row["ncnn_mean_ms"]) >= 100.0]
    top_excess = sorted(
        official,
        key=lambda row: float(row["compiled_mean_ms"]) - float(row["ncnn_mean_ms"]),
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "measurement": {
            "mode": "end_to_end",
            "threads": 6,
            "warmup": 10,
            "iterations": 20,
            "measured_rows": 45,
            "official_rows": 44,
            "diagnostic_model": "resnet18_winograd",
            "ncnn_submodule": "a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a",
        },
        "official_44": summarize(official),
        "heavy_official_ncnn_ge_100ms": summarize(heavy),
        "ratio_buckets": {
            "<1.0": sum(float(row["ratio"]) < 1.0 for row in official),
            "1.0-1.25": sum(1.0 <= float(row["ratio"]) < 1.25 for row in official),
            "1.25-1.5": sum(1.25 <= float(row["ratio"]) < 1.5 for row in official),
            "1.5-2.0": sum(1.5 <= float(row["ratio"]) < 2.0 for row in official),
            "2.0-3.0": sum(2.0 <= float(row["ratio"]) < 3.0 for row in official),
            "3.0-5.0": sum(3.0 <= float(row["ratio"]) < 5.0 for row in official),
            ">=5.0": sum(float(row["ratio"]) >= 5.0 for row in official),
        },
        "top_excess_models": [
            {
                "model": row["model"],
                "ncnn_mean_ms": float(row["ncnn_mean_ms"]),
                "compiled_mean_ms": float(row["compiled_mean_ms"]),
                "ratio": float(row["ratio"]),
                "excess_ms": float(row["compiled_mean_ms"])
                - float(row["ncnn_mean_ms"]),
            }
            for row in top_excess[:10]
        ],
    }
    destination = path.with_name("comparison-summary.json")
    destination.write_text(json.dumps(output, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
