#!/usr/bin/env python3
"""Collect diagnostic schema-2 runtime profiles from a compiled model library."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def element_count(argument: dict[str, Any]) -> int:
  if argument.get("element_type") != "f32":
    raise ValueError(f"{argument.get('name')}: only f32 is supported")
  shape = argument.get("shape")
  if not isinstance(shape, list) or not shape or any(
      isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
      for dimension in shape):
    raise ValueError(f"{argument.get('name')}: expected a static positive shape")
  return math.prod(shape)


def invoke(function: Any, input_buffers: list[Any],
           output_buffers: list[Any]) -> int:
  return int(function(*input_buffers, *output_buffers))


def time_calls(function: Any, input_buffers: list[Any],
               output_buffers: list[Any], warmups: int,
               iterations: int) -> list[int]:
  for _ in range(warmups):
    status = invoke(function, input_buffers, output_buffers)
    if status:
      raise RuntimeError(f"profiled function warmup returned status {status}")
  timings = []
  for _ in range(iterations):
    start = time.perf_counter_ns()
    status = invoke(function, input_buffers, output_buffers)
    elapsed = time.perf_counter_ns() - start
    if status:
      raise RuntimeError(f"profiled function returned status {status}")
    timings.append(elapsed)
  return timings


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, required=True)
  parser.add_argument("--profile", type=Path, required=True)
  parser.add_argument("--iterations", type=int, default=5)
  parser.add_argument("--warmups", type=int, default=2)
  args = parser.parse_args()
  if args.iterations < 1 or args.warmups < 0:
    parser.error("iterations must be positive and warmups non-negative")
  model_dir = args.model_dir
  manifest_paths = [path for path in model_dir.glob("*.json")
                    if not path.name.endswith(".plan.json")]
  manifest_path = manifest_paths[0] if len(manifest_paths) == 1 else None
  plan_path = next(model_dir.glob("*.plan.json"), None)
  if manifest_path is None or plan_path is None:
    parser.error("model directory must contain a manifest and execution plan")
  manifest = read_object(manifest_path)
  plan = read_object(plan_path)
  model = manifest.get("function")
  if not isinstance(model, str) or plan.get("model") != model:
    parser.error("manifest and execution plan model identities do not match")
  inputs = manifest.get("inputs")
  outputs = manifest.get("outputs")
  if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or not outputs:
    parser.error("manifest must contain non-empty input and output arrays")
  target = plan.get("target")
  if not isinstance(target, dict):
    parser.error("execution plan has no target identity")
  threads = target.get("threads")
  triple = target.get("triple")
  if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0 or \
      not isinstance(triple, str) or not triple:
    parser.error("execution plan target has invalid thread or triple metadata")

  args.profile.parent.mkdir(parents=True, exist_ok=True)
  if args.profile.exists():
    parser.error(f"refusing to append to existing profile: {args.profile}")
  os.environ.update({
      "NCNN_PROFILE_SCHEMA": "2",
      "NCNN_PROFILE_PATH": str(args.profile),
      "NCNN_PROFILE_MODE": "end_to_end",
      "NCNN_PROFILE_MODEL": model,
      "NCNN_PROFILE_PLAN_HASH": str(plan["plan_hash"]),
      "NCNN_PROFILE_BUILD_IDENTITY": str(plan["build_identity"]),
      "NCNN_PROFILE_PLAN_REVISION": plan["plan_revision"],
      "NCNN_PROFILE_ATTRIBUTION_REVISION": plan["attribution_revision"],
      "NCNN_PROFILE_THREADS": str(threads),
      "NCNN_PROFILE_TARGET": triple,
      "OMP_NUM_THREADS": str(threads),
      "OMP_DYNAMIC": "FALSE",
  })
  input_pointer = ctypes.POINTER(ctypes.c_float)
  input_storage = [(ctypes.c_float * element_count(item))() for item in inputs]
  output_storage = [(ctypes.c_float * element_count(item))() for item in outputs]
  pointer_type = ctypes.POINTER(ctypes.c_float)
  buffers = [ctypes.cast(buffer, pointer_type)
             for buffer in [*input_storage, *output_storage]]
  library = ctypes.CDLL(str(model_dir / f"lib{model}.so"))
  function = getattr(library, model)
  function.argtypes = [input_pointer] * len(buffers)
  function.restype = ctypes.c_int
  timings = time_calls(function, buffers[:len(inputs)], buffers[len(inputs):],
                       args.warmups, args.iterations)
  profile_rows = [json.loads(line) for line in args.profile.read_text(
      encoding="utf-8").splitlines() if line.strip()]
  if len(profile_rows) != args.warmups + args.iterations:
    raise RuntimeError(
        f"expected {args.warmups + args.iterations} profile rows, got {len(profile_rows)}")
  for index, profile in enumerate(profile_rows[-args.iterations:], start=1):
    (args.profile.parent / f"invocation-{index:02}.profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8")

  diagnostic = {
      "model": model,
      "mode": "end_to_end",
      "status": "diagnostic",
      "threads": threads,
      "target": triple,
      "plan_revision": plan["plan_revision"],
      "plan_hash": plan["plan_hash"],
      "build_identity": plan["build_identity"],
      "compiled_mean_ms": statistics.mean(timings) / 1e6,
      "ncnn_mean_ms": None,
      "ratio": None,
      "diagnostics": {
          "gate_eligible": False,
          "measurement": "profile-on ctypes invocation; not formal performance data",
          "warmups": args.warmups,
          "iterations": args.iterations,
          "input_data": "all-zero diagnostic input",
      },
  }
  perf_path = args.profile.parent / "diagnostic-perf.ndjson"
  perf_path.write_text(json.dumps(diagnostic, sort_keys=True) + "\n",
                       encoding="utf-8")
  print(json.dumps({
      "model": model,
      "profile_rows": len(profile_rows),
      "timing_median_ms": statistics.median(timings) / 1e6,
      "plan_hash": plan["plan_hash"],
      "build_identity": plan["build_identity"],
      "profile": str(args.profile),
      "diagnostic_perf": str(perf_path),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
