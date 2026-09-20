#!/usr/bin/env python3
"""Focused contract tests for perf_json_summary.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


SCRIPT = pathlib.Path(__file__).parents[2] / "tools" / "perf_json_summary.py"


def write_records(path: pathlib.Path, records: list[dict]) -> None:
  path.write_text("".join(json.dumps(record) + "\n" for record in records))


def run_summary(path: pathlib.Path, *extra: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    [sys.executable, str(SCRIPT), str(path), *extra],
    capture_output=True,
    text=True,
  )


def measured(model: str, mode: str, ratio: float) -> dict:
  return {
    "model": model,
    "mode": mode,
    "status": "measured",
    "threads": 1,
    "ncnn_mean_ms": 1.0,
    "compiled_mean_ms": ratio,
    "ratio": ratio,
    "diagnostics": {"gate_eligible": True},
  }


def main() -> int:
  with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)

    mixed = root / "mixed.ndjson"
    write_records(
      mixed,
      [
        # The historical shape has no mode and must remain end_to_end.
        {
          "model": "legacy",
          "threads": 1,
          "ncnn_mean_ms": 1.0,
          "compiled_mean_ms": 2.0,
          "ratio": 2.0,
        },
        measured("same", "end_to_end", 1.5),
        {
          "model": "same",
          "mode": "prepared",
          "status": "unsupported",
          "diagnostics": {"gate_eligible": False},
          "ratio": None,
          "ncnn_mean_ms": None,
          "compiled_mean_ms": None,
        },
        *[measured(model, "end_to_end", 1.0) for model in [
          "squeezenet_v1_1", "resnet18", "resnet34", "resnet50", "resnet101",
          "yolov5n_cls", "yolov5s_cls", "yolov5m_cls", "yolov5l_cls",
          "yolov5x_cls", "yolov5n", "yolov5s", "yolov5m", "yolov5l",
          "yolov5x", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
          "efficientnet_b3", "pp_lcnet_x1_0_doc_ori",
          "pp_lcnet_x1_0_textline_ori", "chineseocr_lite_anglenet",
          "yolov5n_seg", "yolov5s_seg", "yolov5m_seg", "yolov5l_seg",
          "yolov5x_seg", "pp_ocrv6_tiny_rec", "pp_ocrv6_tiny_rec_int8",
          "pp_ocrv5_mobile_rec", "pp_ocrv5_mobile_rec_int8",
          "pp_ocrv5_server_rec", "pp_ocrv6_medium_rec",
          "pp_ocrv6_medium_rec_int8", "pp_ocrv6_small_rec",
          "pp_ocrv6_small_rec_int8", "pp_ocrv6_tiny_det", "pp_ocrv6_small_det",
          "pp_ocrv6_medium_det", "pp_ocrv6_medium_det_int8",
          "pp_ocrv5_mobile_det_static", "pp_ocrv5_server_det_static",
          "pp_structrurev2_slanet_plus_cnn", "pp_formulanet_plus_s_encoder",
        ]],
      ],
    )
    result = run_summary(mixed)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    if "mode=end_to_end n=46" not in result.stdout:
      raise RuntimeError(f"legacy mode was not included: {result.stdout}")
    if "mode=prepared measured=0 records=1" not in result.stdout:
      raise RuntimeError(f"diagnostic mode was not reported: {result.stdout}")
    if "same               prepared" in result.stdout:
      raise RuntimeError("unsupported record entered the measured table")

    baseline = root / "baseline.ndjson"
    write_records(baseline, [measured("same", "end_to_end", 1.0)])
    result = run_summary(mixed, "--baseline", str(baseline), "--gate", "2")
    if result.returncode != 0 or "same" not in result.stdout:
      raise RuntimeError(f"baseline/mode alignment failed: {result.stderr}")
    result = run_summary(mixed, "--gate", "1")
    if result.returncode == 0 or "性能门禁失败" not in result.stderr:
      raise RuntimeError("positive gate did not reject an over-limit ratio")
    result = run_summary(mixed, "--gate", "0")
    if result.returncode != 0:
      raise RuntimeError("zero gate did not disable the gate")
    result = run_summary(mixed, "--official-only")
    if result.returncode != 0 or "mode=end_to_end n=44" not in result.stdout:
      raise RuntimeError(f"official-only filtering failed: {result.stderr}")
    if "resnet18_winograd" in result.stdout:
      raise RuntimeError("diagnostic Winograd row entered official output")

    winograd = root / "winograd.ndjson"
    write_records(winograd, [measured("resnet18_winograd", "end_to_end", 1.1)])
    result = run_summary(winograd, "--official-only")
    if result.returncode == 0 or "requires exactly 44" not in result.stderr:
      raise RuntimeError("official-only accepted an incomplete input")

    malformed = root / "malformed.ndjson"
    malformed.write_text(json.dumps({"model": "bad", "status": "measured", "ratio": None}) + "\n")
    result = run_summary(malformed)
    if result.returncode == 0 or "ratio" not in result.stderr:
      raise RuntimeError("malformed measured record was accepted")

    malformed_threads = root / "malformed-threads.ndjson"
    bad_threads = measured("bad_threads", "end_to_end", 1.0)
    bad_threads["threads"] = "six"
    write_records(malformed_threads, [bad_threads])
    result = run_summary(malformed_threads)
    if result.returncode == 0 or "threads" not in result.stderr:
      raise RuntimeError("malformed thread count was accepted")

    conflict = root / "conflict.ndjson"
    first = measured("same", "prepared", 1.0)
    first.update({
      "target": "x86_64-pc-linux-gnu",
      "plan_hash": "one",
      "build_identity": "one",
    })
    second = dict(first)
    second["target"] = "aarch64-unknown-linux-gnu"
    write_records(conflict, [first, second])
    result = run_summary(conflict)
    if result.returncode == 0 or "conflicting build identities" not in result.stderr:
      raise RuntimeError("conflicting performance identities were overwritten")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
