#!/usr/bin/env python3
"""performance_tests NDJSON 汇总（追平计划 P0 度量基建）。

用法：
    python3 tools/perf_json_summary.py /tmp/perf.ndjson [--baseline 旧.ndjson]
                                        [--sort ratio|time] [--gate 0]

输入为 `NCNN_PERF_JSON` 产出的逐行 JSON（每模型一行，字段见
`append_performance_json_record`）。输出按 ratio 降序的对照表与全表
p50/p90 统计；`--baseline` 给出与历史表的逐模型 ratio 差值，供里程碑
验收（docs/ncnn-performance-parity-plan.md §1 三件套之②）使用。
"""

from __future__ import annotations

import argparse
import json
import sys


def load_rows(path: str) -> dict[str, dict]:
  rows: dict[str, dict] = {}
  with open(path, encoding="utf-8") as stream:
    for line in stream:
      if not line.strip():
        continue
      record = json.loads(line)
      rows[record["model"]] = record
  return rows


def percentile(values: list[float], fraction: float) -> float:
  """线性插值百分位（与 numpy 默认一致）"""
  if not values:
    return float("nan")
  ordered = sorted(values)
  position = fraction * (len(ordered) - 1)
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("json", help="NCNN_PERF_JSON 产出的 NDJSON 文件")
  parser.add_argument("--baseline", help="对照用的历史 NDJSON（同口径采集）")
  parser.add_argument("--sort", choices=("ratio", "time"), default="ratio",
                      help="排序键：ratio（默认）或 compiled 耗时")
  arguments = parser.parse_args()

  rows = load_rows(arguments.json)
  if not rows:
    print("没有可用记录", file=sys.stderr)
    return 1

  baseline = load_rows(arguments.baseline) if arguments.baseline else {}
  ordered = sorted(
    rows.values(),
    key=lambda row: (row["compiled_mean_ms"] if arguments.sort == "time"
                     else row["ratio"]),
    reverse=True,
  )

  header = (f"{'model':38s} {'T':>2s} {'ncnn(ms)':>10s} {'cmpd(ms)':>10s} "
            f"{'ratio':>7s}")
  if baseline:
    header += f" {'base':>7s} {'Δratio':>8s}"
  print(header)
  for row in ordered:
    line = (f"{row['model']:38s} {row['threads']:2d} "
            f"{row['ncnn_mean_ms']:10.1f} {row['compiled_mean_ms']:10.1f} "
            f"{row['ratio']:7.2f}")
    if baseline:
      previous = baseline.get(row["model"])
      if previous is None:
        line += f" {'—':>7s} {'—':>8s}"
      else:
        delta = row["ratio"] - previous["ratio"]
        line += f" {previous['ratio']:7.2f} {delta:+8.2f}"
    print(line)

  ratios = [row["ratio"] for row in ordered]
  heavy = [row["ratio"] for row in ordered if row["ncnn_mean_ms"] >= 100.0]
  print(f"\nn={len(ratios)}  p50={percentile(ratios, 0.5):.2f}  "
        f"p90={percentile(ratios, 0.9):.2f}  max={max(ratios):.2f}")
  if heavy:
    print(f"重模型(ncnn≥100ms) n={len(heavy)}  "
          f"p50={percentile(heavy, 0.5):.2f}  "
          f"p90={percentile(heavy, 0.9):.2f}  max={max(heavy):.2f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
