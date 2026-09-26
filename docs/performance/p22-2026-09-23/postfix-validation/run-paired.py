#!/usr/bin/env python3
"""Run the five-round post-fix P22 paired benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


BASELINE = Path("/tmp/ncnn-compiler-p22-validation-baseline-20260923-a")
CANDIDATE = Path("/tmp/ncnn-compiler-p22-validation-final-20260923-d")
EVIDENCE = Path(__file__).resolve().parent
ROUNDS = (
  ("baseline", "candidate"),
  ("candidate", "baseline"),
  ("baseline", "candidate"),
  ("candidate", "baseline"),
  ("baseline", "candidate"),
)
STAGES = {"baseline": BASELINE, "candidate": CANDIDATE}


def validate_rows(path: Path) -> set[str]:
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
          if line.strip()]
  models = [row["model"] for row in rows]
  if len(rows) != 45 or len(set(models)) != 45:
    raise RuntimeError(f"{path.name}: expected 45 unique model rows, found {len(rows)}")
  for row in rows:
    if row.get("status") != "measured" or row.get("mode") != "end_to_end":
      raise RuntimeError(f"{path.name}: invalid row for {row.get('model')}: {row}")
  return set(models)


def main() -> int:
  expected_models: set[str] | None = None
  environment = os.environ.copy()
  environment.update({
    "NCNN_PERF_THREADS": "6",
    "NCNN_PERF_WARMUP": "10",
    "NCNN_PERF_ITERS": "20",
    "NCNN_PERF_MAX_RATIO": "0",
    "NCNN_PERF_MODE": "end_to_end",
  })
  for index, order in enumerate(ROUNDS, start=1):
    for stage_name in order:
      stage = STAGES[stage_name]
      stem = f"round-{index}-{stage_name}"
      result = EVIDENCE / f"{stem}.ndjson"
      log = EVIDENCE / f"{stem}.ctest.log"
      environment["NCNN_PERF_JSON"] = str(result)
      command = [
        "ctest", "--test-dir", str(stage), "-R", r"^PerformanceModel\.",
        "--output-on-failure",
      ]
      print(f"Running {stem}", flush=True)
      with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, check=False)
      if completed.returncode != 0:
        print(f"{stem} failed; see {log}", file=sys.stderr)
        return completed.returncode
      models = validate_rows(result)
      if expected_models is None:
        expected_models = models
      elif models != expected_models:
        raise RuntimeError(f"{stem}: model set differs from the first run")
      print(f"Completed {stem}: {len(models)} measured models", flush=True)
  print("Completed all five alternating paired rounds.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
