#!/usr/bin/env python3
"""Summarize performance_tests NDJSON by model and measurement mode.

Usage:
    python3 tools/perf_json_summary.py /tmp/perf.ndjson
        [--baseline old.ndjson] [--sort ratio|time] [--gate limit]

Records without a ``mode`` field are treated as the historical ``end_to_end``
mode.  Only measured, gate-eligible end-to-end records contribute to the
performance percentiles and gate; unsupported diagnostic records remain visible
but are never converted into zero-valued measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any


RowKey = tuple[str, str]

# Stable P17/P18 official set: 45 measured rows minus the Winograd diagnostic.
OFFICIAL_MODELS = frozenset({
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
})


def _number(record: dict[str, Any], field: str, *, required: bool) -> float | None:
  value = record.get(field)
  if value is None and not required:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"record field {field!r} must be a number or null")
  converted = float(value)
  if not math.isfinite(converted) or converted < 0.0:
    raise ValueError(f"record field {field!r} must be a finite non-negative number")
  return converted


def _normalise_record(record: Any, line_number: int) -> dict[str, Any]:
  if not isinstance(record, dict):
    raise ValueError(f"line {line_number}: record must be a JSON object")
  model = record.get("model")
  if not isinstance(model, str) or not model:
    raise ValueError(f"line {line_number}: record has no non-empty model")
  mode = record.get("mode", "end_to_end")
  if not isinstance(mode, str) or not mode:
    raise ValueError(f"line {line_number}: mode must be a non-empty string")
  status = record.get("status", "measured")
  if not isinstance(status, str) or not status:
    raise ValueError(f"line {line_number}: status must be a non-empty string")
  diagnostics = record.get("diagnostics", {})
  if diagnostics is None:
    diagnostics = {}
  if not isinstance(diagnostics, dict):
    raise ValueError(f"line {line_number}: diagnostics must be an object")
  gate_eligible = diagnostics.get("gate_eligible", record.get("gate_eligible"))
  if gate_eligible is None:
    gate_eligible = status == "measured" and mode == "end_to_end"
  if not isinstance(gate_eligible, bool):
    raise ValueError(f"line {line_number}: gate_eligible must be boolean")

  normalised = dict(record)
  normalised["model"] = model
  normalised["mode"] = mode
  normalised["status"] = status
  normalised["gate_eligible"] = gate_eligible
  for field in ("target", "plan_hash", "build_identity"):
    value = record.get(field)
    if value is not None and (not isinstance(value, str) or not value):
      raise ValueError(
        f"line {line_number}: {field} must be a non-empty string or null")
  threads = record.get("threads")
  if (threads is not None and
      (isinstance(threads, bool) or not isinstance(threads, int) or
       threads < 0)):
    raise ValueError(
      f"line {line_number}: threads must be a non-negative integer or null")

  measured = status == "measured"
  _number(record, "ratio", required=measured)
  if measured:
    _number(record, "ncnn_mean_ms", required=True)
    _number(record, "compiled_mean_ms", required=True)
  return normalised


def load_rows(path: str | None) -> dict[RowKey, dict[str, Any]]:
  if path is None:
    return {}
  rows: dict[RowKey, dict[str, Any]] = {}
  try:
    with open(path, encoding="utf-8") as stream:
      for line_number, line in enumerate(stream, start=1):
        if not line.strip():
          continue
        try:
          record = json.loads(line)
          normalised = _normalise_record(record, line_number)
        except (json.JSONDecodeError, ValueError) as error:
          raise ValueError(f"{path}:{line_number}: {error}") from error
        key = (normalised["model"], normalised["mode"])
        previous = rows.get(key)
        if previous is not None:
          identity_fields = ("target", "plan_hash", "build_identity")
          previous_identity = tuple(previous.get(field) for field in identity_fields)
          current_identity = tuple(normalised.get(field) for field in identity_fields)
          if (previous_identity != current_identity and
              (any(value is not None for value in previous_identity) or
               any(value is not None for value in current_identity))):
            raise ValueError(
              f"{path}:{line_number}: conflicting build identities for "
              f"model={key[0]!r}, mode={key[1]!r}")
        rows[key] = normalised
  except OSError as error:
    raise ValueError(f"cannot read {path}: {error}") from error
  return rows


def percentile(values: list[float], fraction: float) -> float:
  """Linear interpolation percentile (matching numpy's default method)."""
  if not values:
    return float("nan")
  ordered = sorted(values)
  position = fraction * (len(ordered) - 1)
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measured_rows(rows: dict[RowKey, dict[str, Any]], mode: str) -> list[dict[str, Any]]:
  return [
    row
    for row in rows.values()
    if row["mode"] == mode
    and row["status"] == "measured"
    and row["gate_eligible"]
    and isinstance(row.get("ratio"), (int, float))
  ]


def official_rows(rows: dict[RowKey, dict[str, Any]]) -> dict[RowKey, dict[str, Any]]:
  """Return the reproducible 44-model official end-to-end subset."""
  return {
    key: row
    for key, row in rows.items()
    if row["mode"] == "end_to_end"
    and row["status"] == "measured"
    and row["gate_eligible"]
    and row["model"] in OFFICIAL_MODELS
  }


def print_mode_summary(rows: dict[RowKey, dict[str, Any]], mode: str) -> None:
  values = [float(row["ratio"]) for row in measured_rows(rows, mode)]
  diagnostics = [row for row in rows.values() if row["mode"] == mode]
  if not diagnostics:
    return
  if not values:
    print(f"mode={mode} measured=0 records={len(diagnostics)} (no measured values)")
    return
  heavy = [
    float(row["ratio"])
    for row in measured_rows(rows, mode)
    if float(row["ncnn_mean_ms"]) >= 100.0
  ]
  print(
    f"mode={mode} n={len(values)} p50={percentile(values, 0.5):.2f} "
    f"p90={percentile(values, 0.9):.2f} max={max(values):.2f} "
    f"records={len(diagnostics)}"
  )
  if heavy:
    print(
      f"  heavy(ncnn>=100ms) n={len(heavy)} "
      f"p50={percentile(heavy, 0.5):.2f} "
      f"p90={percentile(heavy, 0.9):.2f} max={max(heavy):.2f}"
    )


def main() -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("json", help="NDJSON emitted by NCNN_PERF_JSON")
  parser.add_argument("--baseline", help="historical NDJSON with matching modes")
  parser.add_argument(
    "--sort", choices=("ratio", "time"), default="ratio",
    help="sort measured end-to-end rows by ratio (default) or compiled time",
  )
  parser.add_argument(
    "--gate", type=float, default=0.0,
    help="fail when an eligible end-to-end ratio exceeds this positive limit; 0 disables",
  )
  parser.add_argument(
    "--official-only", action="store_true",
    help="use the reproducible 44-model official end-to-end subset",
  )
  arguments = parser.parse_args()
  if not math.isfinite(arguments.gate) or arguments.gate < 0.0:
    print("--gate must be a finite non-negative number", file=sys.stderr)
    return 2

  try:
    rows = load_rows(arguments.json)
    baseline = load_rows(arguments.baseline)
  except ValueError as error:
    print(str(error), file=sys.stderr)
    return 1
  if arguments.official_only:
    rows = official_rows(rows)
    baseline = official_rows(baseline)
    if len(rows) != 44:
      print(
        f"--official-only requires exactly 44 official measured models; got {len(rows)}",
        file=sys.stderr,
      )
      return 1
    if baseline and len(baseline) != 44:
      print(
        f"--official-only baseline requires exactly 44 official measured models; "
        f"got {len(baseline)}",
        file=sys.stderr,
      )
      return 1
  if not rows:
    print("没有可用记录", file=sys.stderr)
    return 1

  end_to_end = measured_rows(rows, "end_to_end")
  ordered = sorted(
    end_to_end,
    key=lambda row: (
      float(row["compiled_mean_ms"])
      if arguments.sort == "time"
      else float(row["ratio"])
    ),
    reverse=True,
  )
  if ordered:
    header = (
      f"{'model':38s} {'mode':16s} {'T':>2s} {'ncnn(ms)':>10s} "
      f"{'cmpd(ms)':>10s} {'ratio':>7s}"
    )
    if baseline:
      header += f" {'base':>7s} {'Δratio':>8s}"
    print(header)
    for row in ordered:
      key = (row["model"], row["mode"])
      line = (
        f"{row['model']:38s} {row['mode']:16s} {int(row.get('threads', 0)):2d} "
        f"{float(row['ncnn_mean_ms']):10.1f} "
        f"{float(row['compiled_mean_ms']):10.1f} {float(row['ratio']):7.2f}"
      )
      if baseline:
        previous = baseline.get(key)
        if previous is None or not isinstance(previous.get("ratio"), (int, float)):
          line += f" {'—':>7s} {'—':>8s}"
        else:
          previous_ratio = float(previous["ratio"])
          delta = float(row["ratio"]) - previous_ratio
          line += f" {previous_ratio:7.2f} {delta:+8.2f}"
      print(line)
  else:
    print("没有可用于 ratio 的 measured end_to_end 记录")

  modes = sorted({row["mode"] for row in rows.values()})
  for mode in modes:
    print_mode_summary(rows, mode)

  if arguments.gate > 0.0:
    failures = [
      row for row in end_to_end if float(row["ratio"]) > arguments.gate
    ]
    if failures:
      print(
        f"性能门禁失败: {len(failures)} 条 end_to_end ratio 超过 "
        f"{arguments.gate:.3f}",
        file=sys.stderr,
      )
      for row in failures:
        print(
          f"  {row['model']}: ratio={float(row['ratio']):.3f}",
          file=sys.stderr,
        )
      return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
