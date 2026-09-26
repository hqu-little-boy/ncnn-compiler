#!/usr/bin/env python3
"""Summarise the P2 performance baseline against the P27 ncnn comparison.

usage: summarize-perf.py <p2-ndjson> [<p27-ndjson>]

Prints a per-model table of compiled/ncnn median ratio and the aggregate
p50 / heavy-model p50, which is the "how far from ncnn are we" answer.
"""
from __future__ import annotations

import json
import statistics
import sys

# Models that dominate wall-clock in the cohort (P27/P28 "heavy" set).
HEAVY = {
    "yolov5x_seg",
    "yolov5x",
    "yolov5l_seg",
    "yolov5l",
    "resnet101",
    "yolov5m_seg",
    "yolov5m",
}


def load(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") not in (None, "measured"):
                continue
            rows[row["model"]] = row
    return rows


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    current = load(argv[1])
    baseline = load(argv[2]) if len(argv) > 2 else {}

    print(f"{'model':<32} {'compiled':>10} {'ncnn':>10} {'ratio':>8} "
          f"{'p27':>8} {'delta':>8}")
    print("-" * 80)
    ratios: list[float] = []
    heavy_ratios: list[float] = []
    drift: list[float] = []
    for name in sorted(current):
        row = current[name]
        ratio = row.get("ratio")
        if ratio is None:
            continue
        old = baseline.get(name, {}).get("ratio")
        delta = f"{(ratio - old) / old * 100:+.1f}%" if old else "n/a"
        if old:
            drift.append((ratio - old) / old * 100)
        print(f"{name:<32} {row['compiled_median_ms']:>10.3f} "
              f"{row['ncnn_median_ms']:>10.3f} {ratio:>8.3f} "
              f"{(f'{old:.3f}' if old else '-'):>8} {delta:>8}")
        ratios.append(ratio)
        if name in HEAVY:
            heavy_ratios.append(ratio)
    print("-" * 80)
    if ratios:
        print(f"models measured      : {len(ratios)}")
        print(f"ratio p50            : {statistics.median(ratios):.3f}")
        if heavy_ratios:
            print(f"heavy ratio p50      : {statistics.median(heavy_ratios):.3f}"
                  f"  ({len(heavy_ratios)} models)")
        print(f"ratio min / max      : {min(ratios):.3f} / {max(ratios):.3f}")
        better = sum(1 for r in ratios if r <= 1.0)
        print(f"at or faster than ncnn: {better}/{len(ratios)}")
    if drift:
        print(f"median drift vs P27  : {statistics.median(drift):+.2f}%")
        print(f"max |drift| vs P27   : {max(abs(d) for d in drift):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
