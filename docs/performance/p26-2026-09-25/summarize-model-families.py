#!/usr/bin/env python3
"""Summarize official P26 E2E timing by the P18 model-family partition."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


ARCHIVE = Path(__file__).resolve().parent
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


def family(name: str) -> str:
    if name.startswith("yolov5"):
        if name.endswith("_cls"):
            return "CNN classification"
        if name.endswith("_seg"):
            return "YOLO segmentation"
        return "YOLO detection"
    if name.startswith("pp_ocrv") and "_rec" in name:
        return "OCR recognition"
    if name.startswith("pp_ocrv") and "_det" in name:
        return "OCR detection"
    if name.startswith(("resnet", "efficientnet", "squeezenet", "pp_lcnet")):
        return "CNN classification"
    if name in {"chineseocr_lite_anglenet", "pp_structrurev2_slanet_plus_cnn",
                "pp_formulanet_plus_s_encoder"}:
        return "Formula/Doc"
    raise ValueError(f"unclassified official model: {name}")


def percentile(values: list[float], fraction: float) -> float:
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def stats(rows: list[dict]) -> dict:
    ratios = sorted(float(row["ratio"]) for row in rows)
    excess = sum(float(row["compiled_mean_ms"]) - float(row["ncnn_mean_ms"])
                 for row in rows)
    ncnn = sum(float(row["ncnn_mean_ms"]) for row in rows)
    compiled = sum(float(row["compiled_mean_ms"]) for row in rows)
    return {
        "models": len(rows),
        "ratio_p50": statistics.median(ratios),
        "ratio_p90": percentile(ratios, 0.9),
        "ratio_max": max(ratios),
        "ratio_geometric_mean": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
        "ncnn_total_ms": ncnn,
        "compiled_total_ms": compiled,
        "total_excess_ms": excess,
    }


def main() -> int:
    rows = json.loads((ARCHIVE / "final-all-model-ncnn-comparison.json").read_text())
    official = [row for row in rows if row["model"] in OFFICIAL]
    if len(official) != 44:
        raise SystemExit(f"expected 44 official models, got {len(official)}")
    groups: dict[str, list[dict]] = {}
    for row in official:
        groups.setdefault(family(row["model"]), []).append(row)
    int8_rows = [row for row in official if row["model"].endswith("_int8")]
    result = {
        "measurement": "profile-off end_to_end, 6 threads, 10 warmup, 20 iterations",
        "official_models": 44,
        "families": {name: stats(group) for name, group in sorted(groups.items())},
        "orthogonal_int8_subset": stats(int8_rows),
        "orthogonal_fp32_subset": stats([row for row in official if row not in int8_rows]),
    }
    (ARCHIVE / "model-family-summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
