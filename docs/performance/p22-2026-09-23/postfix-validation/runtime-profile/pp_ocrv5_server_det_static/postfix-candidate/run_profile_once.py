#!/usr/bin/env python3
"""Run one diagnostic invocation of the instrumented server-det library."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path


MODEL = "pp_ocrv5_server_det_static"
INPUT_ELEMENTS = 3 * 640 * 640
OUTPUT_ELEMENTS = 640 * 640


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--library", required=True, help="generated model .so")
  parser.add_argument("--profile", required=True, help="new profile NDJSON path")
  parser.add_argument("--perf", required=True, help="new diagnostic perf NDJSON path")
  parser.add_argument(
    "--plan", default=str(Path(__file__).with_name(f"{MODEL}.plan.json")),
    help="execution plan matching the generated library")
  args = parser.parse_args()

  plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
  if plan.get("model") != MODEL:
    parser.error(f"plan model must be {MODEL}")
  target = plan["target"]
  os.environ.update({
    "NCNN_PROFILE_SCHEMA": "2",
    "NCNN_PROFILE_PATH": args.profile,
    "NCNN_PROFILE_MODE": "end_to_end",
    "NCNN_PROFILE_MODEL": plan["model"],
    "NCNN_PROFILE_PLAN_HASH": plan["plan_hash"],
    "NCNN_PROFILE_BUILD_IDENTITY": plan["build_identity"],
    "NCNN_PROFILE_PLAN_REVISION": plan["plan_revision"],
    "NCNN_PROFILE_ATTRIBUTION_REVISION": plan["attribution_revision"],
    "NCNN_PROFILE_THREADS": str(target["threads"]),
    "NCNN_PROFILE_TARGET": target["triple"],
    "OMP_NUM_THREADS": str(target["threads"]),
  })

  library = ctypes.CDLL(args.library)
  run = getattr(library, MODEL)
  float_pointer = ctypes.POINTER(ctypes.c_float)
  run.argtypes = [float_pointer, float_pointer]
  run.restype = ctypes.c_int
  input_data = (ctypes.c_float * INPUT_ELEMENTS)()
  output_data = (ctypes.c_float * OUTPUT_ELEMENTS)()
  start = time.perf_counter_ns()
  status = run(input_data, output_data)
  elapsed_ms = (time.perf_counter_ns() - start) / 1e6
  if status != 0:
    raise RuntimeError(f"generated model returned status {status}")

  perf_row = {
    "model": plan["model"],
    "plan_revision": plan["plan_revision"],
    "plan_hash": plan["plan_hash"],
    "build_identity": plan["build_identity"],
    "target": target["triple"],
    "threads": target["threads"],
    "mode": "end_to_end",
    "status": "diagnostic",
    "ratio": None,
    "ncnn_mean_ms": None,
    "compiled_mean_ms": elapsed_ms,
    "diagnostics": {
      "gate_eligible": False,
      "measurement": "single instrumented ctypes invocation",
    },
  }
  Path(args.perf).write_text(
    json.dumps(perf_row, sort_keys=True) + "\n", encoding="utf-8")
  print(f"model status={status}; instrumented elapsed_ms={elapsed_ms:.3f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
