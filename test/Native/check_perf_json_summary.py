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
      ],
    )
    result = run_summary(mixed)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    if "mode=end_to_end n=2" not in result.stdout:
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

    malformed = root / "malformed.ndjson"
    malformed.write_text(json.dumps({"model": "bad", "status": "measured", "ratio": None}) + "\n")
    result = run_summary(malformed)
    if result.returncode == 0 or "ratio" not in result.stderr:
      raise RuntimeError("malformed measured record was accepted")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
