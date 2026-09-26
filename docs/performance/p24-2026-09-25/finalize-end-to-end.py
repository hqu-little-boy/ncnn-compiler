#!/usr/bin/env python3
"""Validate one complete P24 candidate-vs-ncnn run and retain JSON rows."""

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
DIAGNOSTIC = "resnet18_winograd"


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    rows = [json.loads(line) for line in source.read_text().splitlines()
            if line.strip()]
    by_model = {}
    for row in rows:
        name = row.get("model")
        if not isinstance(name, str) or name in by_model:
            raise SystemExit(f"duplicate or invalid model row: {name!r}")
        if row.get("status") != "measured" or row.get("mode") != "end_to_end":
            raise SystemExit(f"unexpected row state for {name}")
        if (row.get("threads"), row.get("warmup"), row.get("iterations")) != (6, 10, 20):
            raise SystemExit(f"unexpected timing metadata for {name}")
        if not row.get("diagnostics", {}).get("gate_eligible", False):
            raise SystemExit(f"non-gate-eligible row in measured data: {name}")
        for field in ("ncnn_mean_ms", "compiled_mean_ms", "ratio"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SystemExit(f"invalid {field} for {name}: {value!r}")
        by_model[name] = row

    expected = OFFICIAL | {DIAGNOSTIC}
    if set(by_model) != expected:
        raise SystemExit(
            f"expected 45 measured rows (44 official + diagnostic), "
            f"got {len(by_model)}; missing={sorted(expected - set(by_model))}; "
            f"unexpected={sorted(set(by_model) - expected)}"
        )

    ordered = [by_model[name] for name in sorted(by_model)]
    destination.write_text(json.dumps(ordered, indent=2) + "\n",
                           encoding="utf-8")
    print(f"validated_rows={len(ordered)} official={len(OFFICIAL)} "
          f"diagnostic={DIAGNOSTIC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
