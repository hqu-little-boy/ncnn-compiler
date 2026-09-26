#!/usr/bin/env python3
"""P28: validate sampled worker attribution against full instrumentation.

One instrumented build per model is executed at two duty-cycle settings, so the
only variable is the sampling gate.  A profile-off run in the same batch is the
perturbation denominator.  Profile-on times are diagnostic only and are never
used as formal performance results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


MODELS = {
  "yolov5x": {
    "group": "yolo_absolute_excess",
    "asset_group": "yolo",
    "stem": "yolov5x",
    "shape": "3x640x640",
    "test": "Yolov5xDet",
    "large": True,
  },
  "yolov5x_seg": {
    "group": "yolo_regression_control",
    "asset_group": "yolo",
    "stem": "yolov5x_seg",
    "shape": "3x640x640",
    "test": "Yolov5xSeg",
    "large": True,
  },
  "pp_ocrv6_medium_rec_int8": {
    "group": "int8_rec_high_ratio",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_medium_rec_int8",
    "shape": "3x48x320",
    "test": "PPOcrv6MediumRecInt8",
  },
  "pp_ocrv6_small_rec_int8": {
    "group": "int8_rec_high_ratio",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_small_rec_int8",
    "shape": "3x48x320",
    "test": "PPOcrv6SmallRecInt8",
  },
  "pp_ocrv6_tiny_rec_int8": {
    "group": "int8_rec_high_ratio",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_tiny_rec_int8",
    "shape": "3x48x320",
    "test": "PPOcrv6TinyRecInt8",
  },
  "pp_ocrv5_mobile_rec_int8": {
    "group": "int8_rec_reference",
    "asset_group": "ocr",
    "stem": "PP-OCRv5_mobile_rec_int8",
    "shape": "3x48x320",
    "test": "PPOcrv5MobileRecInt8",
  },
  "chineseocr_lite_anglenet": {
    "group": "small_fixed_overhead",
    "asset_group": "ocr",
    "stem": "Chineseocr_Lite_AngleNet",
    "shape": "3x32x192",
    "test": "ChineseOCRLiteAngleNet",
  },
  "pp_ocrv6_tiny_rec": {
    "group": "small_fixed_overhead_control",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_tiny_rec",
    "shape": "3x48x320",
    "test": "PPOcrv6TinyRec",
  },
}

DUTY_FULL = 1
DUTY_SAMPLED = 8

def run_ctest(stage: Path, tests: list[str], rows: Path, log_path: Path,
              profile_library_dir: Path | None = None,
              profile_path: Path | None = None,
              duty: int | None = None) -> dict:
  expression = r"^PerformanceModel\.({})$".format("|".join(tests))
  command = ["ctest", "--test-dir", str(stage), "-R", expression,
             "--output-on-failure"]
  environment = os.environ.copy()
  environment.update({
    "NCNN_PERF_THREADS": "6",
    "NCNN_PERF_WARMUP": "2",
    "NCNN_PERF_ITERS": "5",
    "NCNN_PERF_MODE": "end_to_end",
    "NCNN_PERF_MAX_RATIO": "0",
    "NCNN_PERF_JSON": str(rows),
  })
  for name in ("NCNN_PERF_PROFILED", "NCNN_PERF_LIBRARY_OVERRIDE_DIR",
               "NCNN_PROFILE_SCHEMA", "NCNN_PROFILE_PATH",
               "NCNN_PROFILE_SAMPLE_DUTY",
               "NCNN_PROFILE_INPUT_HASH", "NCNN_PERF_SKIP_SANITY"):
    environment.pop(name, None)
  if profile_library_dir is not None:
    environment.update({
      "NCNN_PERF_PROFILED": "1",
      "NCNN_PERF_LIBRARY_OVERRIDE_DIR": str(profile_library_dir),
      "NCNN_PERF_SKIP_SANITY": "1",
      "NCNN_PROFILE_SCHEMA": "3",
      "NCNN_PROFILE_MODE": "end_to_end",
      "NCNN_PROFILE_THREADS": "6",
      "NCNN_PROFILE_PATH": str(profile_path),
    })
    if duty is not None:
      environment["NCNN_PROFILE_SAMPLE_DUTY"] = str(duty)
  started = time.perf_counter()
  with log_path.open("w", encoding="utf-8") as output:
    process = subprocess.Popen(command, env=environment, stdout=output,
                               stderr=subprocess.STDOUT)
    while process.poll() is None:
      time.sleep(0.05)
  wall_seconds = time.perf_counter() - started
  if process.returncode:
    raise RuntimeError(f"ctest failed ({process.returncode}); see {log_path}")
  rows_data = [json.loads(line) for line in rows.read_text(
    encoding="utf-8").splitlines() if line.strip()]
  return {"wall_seconds": wall_seconds, "rows": rows_data}


def attribution(stage: Path, model: str, plan_dir: Path, run_dir: Path,
                out: Path) -> dict:
  profiler = Path(__file__).resolve().parents[3] / "tools" / \
    "perf_attribution_report.py"
  result = subprocess.run([
    sys.executable, str(profiler),
    f"--plan={plan_dir / (model + '.plan.json')}",
    f"--profile={run_dir / 'profile.ndjson'}",
    f"--perf={run_dir / 'diagnostic-perf.ndjson'}",
    "--mode=end_to_end",
    f"--output={out}",
  ], capture_output=True, text=True, check=False)
  (run_dir / "attribution.log").write_text(
    result.stdout + result.stderr, encoding="utf-8")
  if result.returncode:
    raise RuntimeError(f"attribution failed for {model}; see attribution.log")
  return json.loads(out.read_text(encoding="utf-8"))


def top_worker_shares(report: dict, key: str, limit: int = 10) -> list[dict]:
  rows = report["runtime"].get("worker_attribution", {}).get("operations", [])
  ranked = sorted(rows, key=lambda row: row.get(key) or 0, reverse=True)
  return [{
    "operation": row.get("operation"),
    "kind": row.get("kind"),
    "source_layer": row.get("source_layer"),
    "source_name": row.get("source_name"),
    "wall_attributed_share_of_top_level":
      row.get("wall_attributed_share_of_top_level"),
    "wall_union_share_of_top_level": row.get("wall_union_share_of_top_level"),
    "exclusive_cpu_ns": row.get("exclusive_cpu_ns"),
    "wall_attributed_ns": row.get("wall_attributed_ns"),
  } for row in ranked[:limit]]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", type=Path, required=True)
  parser.add_argument("--yolo-root", type=Path, required=True)
  parser.add_argument("--ocr-root", type=Path, required=True)
  parser.add_argument("--evidence", type=Path, required=True)
  parser.add_argument("--models", nargs="*", default=list(MODELS))
  args = parser.parse_args()
  evidence = args.evidence
  for child in ("profile-off", "profile-on", "profile-logs", "summary.json"):
    if (evidence / child).exists():
      parser.error(f"refusing to overwrite existing evidence: {evidence / child}")
  profile_off = evidence / "profile-off"
  profile_on = evidence / "profile-on"
  logs = evidence / "profile-logs"
  profile_off.mkdir(parents=True)
  profile_on.mkdir()
  logs.mkdir()
  compiler = args.stage / "tools/ncnn-compile"
  driver = args.stage / "tools/ncnn-mlir-driver"
  optimizer = args.stage / "bin/ncnn-mlir-opt"
  compile_commands: list[str] = []
  summary: dict = {"schema_version": 1, "study": "p28-worker-sampling", "models": {}}

  selected = [name for name in args.models if name in MODELS]
  all_tests = [MODELS[name]["test"] for name in selected]
  off = run_ctest(args.stage, all_tests, profile_off / "off.ndjson",
                  profile_off / "off.ctest.log")
  off_by_model = {row["model"]: row for row in off["rows"]}
  summary["profile_off_rows"] = len(off["rows"])
  summary["profile_off_wall_seconds"] = off["wall_seconds"]

  for name in selected:
    spec = MODELS[name]
    asset_root = args.yolo_root if spec["asset_group"] == "yolo" else args.ocr_root
    if spec["asset_group"] == "yolo":
      parameter = asset_root / f"{spec['stem']}.ncnn.param"
      weights = asset_root / f"{spec['stem']}.ncnn.bin"
    else:
      parameter = asset_root / f"{spec['stem']}.param"
      weights = asset_root / f"{spec['stem']}.bin"
    out = profile_on / name
    out.mkdir()
    command = [
      str(compiler),
      f"--driver={driver}",
      f"--opt={optimizer}",
      "--translate=/usr/bin/mlir-translate-21",
      "--clang=/usr/bin/clang-21",
      "--nm=/usr/bin/llvm-nm-21",
      "--readelf=/usr/bin/llvm-readelf-21",
      "--llvm-as=/usr/bin/llvm-as-21",
      f"--param={parameter}",
      f"--bin={weights}",
      f"--model-name={name}",
      f"--input-shape={spec['shape']}",
      "--tuning-profile=stable",
      "--matmul-packing=auto",
      "--int8-kernel=portable",
      "--march=x86-64-v3",
      "--target-feature=+avx2",
      "--target-feature=+fma",
      "--vector-mode=fixed-width",
      "--threads=6",
      "--profile",
      "--emit-manifest",
      "--emit-execution-plan",
      f"--output-dir={out}",
    ]
    compile_commands.append(shlex.join(command))
    budget = 900 if spec.get("large") else 300
    compile_start = time.perf_counter()
    with (logs / f"{name}-compile.log").open("w", encoding="utf-8") as log_file:
      try:
        result = subprocess.run(command, stdout=log_file,
                                stderr=subprocess.STDOUT, timeout=budget)
      except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{name}: compile exceeded {budget}s") from error
    if result.returncode:
      raise RuntimeError(f"{name}: compile failed; see profile-logs")
    compile_wall = time.perf_counter() - compile_start
    baseline_library = (args.stage / "test/Numerical/generated" / name /
                        f"lib{name}.so")
    model_summary: dict = {
      "group": spec["group"],
      "profile_compile_wall_seconds": compile_wall,
      "profile_binary_size_bytes": (out / f"lib{name}.so").stat().st_size,
      "profile_off_binary_size_bytes": baseline_library.stat().st_size,
      "profile_off_compiled_mean_ms": off_by_model[name]["compiled_mean_ms"],
      "profile_off_ncnn_mean_ms": off_by_model[name]["ncnn_mean_ms"],
      "profile_off_ratio": off_by_model[name]["ratio"],
      "runs": {},
    }

    for label, duty in (("duty1", DUTY_FULL), ("duty8", DUTY_SAMPLED)):
      run_dir = out / label
      run_dir.mkdir()
      run = run_ctest(args.stage, [spec["test"]], run_dir / "diagnostic-perf.ndjson",
                      run_dir / "run.ctest.log", out, run_dir / "profile.ndjson",
                      duty=duty)
      row = run["rows"][0]
      report = attribution(args.stage, name, out, run_dir,
                           run_dir / "attribution-report.json")
      runtime = report["runtime"]
      partition = runtime.get("wall_partition", {})
      worker = runtime.get("worker_attribution", {})
      rows_profile = [
        json.loads(line) for line in
        (run_dir / "profile.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
      ]
      worker_calls = sum(
        sum(event["calls"] for event in profile_row["events"]
            if event["category"] == "worker_operation")
        for profile_row in rows_profile)
      model_summary["runs"][label] = {
        "status": "measured",
        "duty": duty,
        "compiled_mean_ms": row["compiled_mean_ms"],
        "input_hash": row["input_hash"],
        "profile_perturbation_percent": (
          100.0 * (row["compiled_mean_ms"] /
                   off_by_model[name]["compiled_mean_ms"] - 1.0)),
        "worker_calls_sampled": worker_calls,
        "interval_overflow": runtime["summary"].get("worker_sampling", {}).get(
          "interval_overflow"),
        "worker_sampling": runtime["summary"].get("worker_sampling"),
        "runtime_complete": runtime["complete"],
        "wall_partition": partition,
        "worker_complete_for_observed_events":
          worker.get("complete_for_observed_events"),
        "top_by_wall_attributed": top_worker_shares(report,
                                                    "wall_attributed_ns"),
        "top_by_wall_union": top_worker_shares(report, "wall_union_ns"),
        "source_op_join_count": sum(
          row_item.get("source_layer") is not None
          for row_item in worker.get("operations", [])),
        "source_op_total_count": worker.get("observed_event_count"),
      }
      (run_dir / "run-summary.json").write_text(
        json.dumps(model_summary["runs"][label], indent=2, sort_keys=True) +
        "\n", encoding="utf-8")
    summary["models"][name] = model_summary
    print(f"collected {name}", flush=True)

  (evidence / "commands.txt").write_text(
    "\n".join(compile_commands) + "\n", encoding="utf-8")
  (evidence / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
