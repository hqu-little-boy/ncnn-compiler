#!/usr/bin/env python3
"""Collect separate profile-off and profile-on P27 cohort evidence."""

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
    "test": "Yolov5xDet",
    "group": "yolo_absolute_excess",
    "asset_group": "yolo",
    "stem": "yolov5x",
    "shape": "3x640x640",
  },
  "yolov5x_seg": {
    "test": "Yolov5xSeg",
    "group": "yolo_regression_control",
    "asset_group": "yolo",
    "stem": "yolov5x_seg",
    "shape": "3x640x640",
  },
  "pp_ocrv6_medium_rec_int8": {
    "test": "PPOcrv6MediumRecInt8",
    "group": "int8_rec_high_ratio",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_medium_rec_int8",
    "shape": "3x48x320",
    "int8": True,
  },
  "pp_ocrv6_small_rec_int8": {
    "test": "PPOcrv6SmallRecInt8",
    "group": "int8_rec_high_ratio",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_small_rec_int8",
    "shape": "3x48x320",
    "int8": True,
  },
  "pp_ocrv6_tiny_rec_int8": {
    "test": "PPOcrv6TinyRecInt8",
    "group": "int8_rec_high_ratio",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_tiny_rec_int8",
    "shape": "3x48x320",
    "int8": True,
  },
  "pp_ocrv5_mobile_rec_int8": {
    "test": "PPOcrv5MobileRecInt8",
    "group": "int8_rec_reference",
    "asset_group": "ocr",
    "stem": "PP-OCRv5_mobile_rec_int8",
    "shape": "3x48x320",
    "int8": True,
  },
  "chineseocr_lite_anglenet": {
    "test": "ChineseOCRLiteAngleNet",
    "group": "small_fixed_overhead",
    "asset_group": "ocr",
    "stem": "Chineseocr_Lite_AngleNet",
    "shape": "3x32x192",
  },
  "pp_ocrv6_tiny_rec": {
    "test": "PPOcrv6TinyRec",
    "group": "small_fixed_overhead_control",
    "asset_group": "ocr",
    "stem": "PP-OCRv6_tiny_rec",
    "shape": "3x48x320",
  },
}


def run(command: list[str], log_path: Path, environment: dict[str, str],
        timeout_seconds: int | None = None) -> float:
  started = time.perf_counter()
  with log_path.open("w", encoding="utf-8") as output:
    try:
      result = subprocess.run(command, env=environment, stdout=output,
                              stderr=subprocess.STDOUT, check=False,
                              timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
      raise RuntimeError(
        f"command exceeded {timeout_seconds}s; see {log_path}") from error
  if result.returncode:
    raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")
  return time.perf_counter() - started


def process_tree_rss_kb(root_pid: int) -> int | None:
  proc_root = Path("/proc")
  if not proc_root.is_dir():
    return None
  parent_by_pid: dict[int, int] = {}
  rss_by_pid: dict[int, int] = {}
  for process_dir in proc_root.iterdir():
    if not process_dir.name.isdigit():
      continue
    try:
      pid = int(process_dir.name)
      stat_fields = (process_dir / "stat").read_text().split(") ", 1)[1].split()
      parent_by_pid[pid] = int(stat_fields[1])
      for line in (process_dir / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
          rss_by_pid[pid] = int(line.split()[1])
          break
    except (OSError, ValueError, IndexError):
      continue
  active = {root_pid}
  while True:
    children = {pid for pid, parent in parent_by_pid.items()
                if parent in active and pid not in active}
    if not children:
      break
    active.update(children)
  values = [rss_by_pid[pid] for pid in active if pid in rss_by_pid]
  return sum(values) if values else None


def run_ctest(stage: Path, tests: list[str], mode: str, rows: Path,
              log_path: Path, profile_library_dir: Path | None = None,
              profile_path: Path | None = None) -> int | None:
  expression = r"^PerformanceModel\.({})$".format("|".join(tests))
  command = ["ctest", "--test-dir", str(stage), "-R", expression,
             "--output-on-failure"]
  environment = os.environ.copy()
  environment.update({
    "NCNN_PERF_THREADS": "6",
    "NCNN_PERF_WARMUP": "2",
    "NCNN_PERF_ITERS": "5",
    "NCNN_PERF_MODE": mode,
    "NCNN_PERF_MAX_RATIO": "0",
    "NCNN_PERF_JSON": str(rows),
  })
  environment.pop("NCNN_PERF_PROFILED", None)
  environment.pop("NCNN_PERF_LIBRARY_OVERRIDE_DIR", None)
  environment.pop("NCNN_PROFILE_SCHEMA", None)
  environment.pop("NCNN_PROFILE_PATH", None)
  environment.pop("NCNN_PROFILE_INPUT_HASH", None)
  if profile_library_dir is not None:
    environment.update({
      "NCNN_PERF_PROFILED": "1",
      "NCNN_PERF_LIBRARY_OVERRIDE_DIR": str(profile_library_dir),
      "NCNN_PERF_SKIP_SANITY": "1",
      "NCNN_PROFILE_SCHEMA": "3",
      "NCNN_PROFILE_MODE": mode,
      "NCNN_PROFILE_THREADS": "6",
      "NCNN_PROFILE_PATH": str(profile_path),
    })
  with log_path.open("w", encoding="utf-8") as output:
    process = subprocess.Popen(command, env=environment, stdout=output,
                               stderr=subprocess.STDOUT)
    max_rss = 0
    while process.poll() is None:
      current_rss = process_tree_rss_kb(process.pid)
      if current_rss is not None:
        max_rss = max(max_rss, current_rss)
      time.sleep(0.05)
    if process.returncode:
      raise RuntimeError(
        f"command failed ({process.returncode}); see {log_path}")
  return max_rss or None


def read_rows(path: Path, expected: int, mode: str,
              expected_status: str = "measured") -> list[dict]:
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
          if line.strip()]
  if len(rows) != expected or len({row.get("model") for row in rows}) != expected:
    raise RuntimeError(f"{path}: expected {expected} distinct records, got {len(rows)}")
  for row in rows:
    if (row.get("status") != expected_status or
        row.get("mode") != mode or row.get("threads") != 6 or
        row.get("warmup") != 2 or row.get("iterations") != 5):
      raise RuntimeError(f"{path}: invalid timing metadata for {row.get('model')}")
    if not row.get("input_hash"):
      raise RuntimeError(f"{path}: missing deterministic input hash")
    order = row.get("order", {})
    for phase, expected_count in (("warmup", 2), ("timed", 5)):
      schedule = order.get(phase)
      items = schedule.split(",") if isinstance(schedule, str) else []
      if len(items) != expected_count or any(
          items[index] == items[index - 1] for index in range(1, len(items))):
        raise RuntimeError(f"{path}: {phase} backend order is not counterbalanced")
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", type=Path, required=True)
  parser.add_argument("--compiler-root", type=Path, required=True)
  parser.add_argument("--yolo-root", type=Path, required=True)
  parser.add_argument("--ocr-root", type=Path, required=True)
  parser.add_argument("--evidence", type=Path, required=True)
  args = parser.parse_args()
  evidence = args.evidence
  for child in ("profile-off", "profile-on", "profile-logs", "commands.txt"):
    if (evidence / child).exists():
      parser.error(f"refusing to overwrite existing evidence: {evidence / child}")
  profile_off = evidence / "profile-off"
  profile_on = evidence / "profile-on"
  profile_logs = evidence / "profile-logs"
  profile_off.mkdir(parents=True)
  profile_on.mkdir()
  profile_logs.mkdir()
  compiler = args.stage / "tools/ncnn-compile"
  driver = args.stage / "tools/ncnn-mlir-driver"
  optimizer = args.stage / "bin/ncnn-mlir-opt"
  profiler = args.compiler_root / "tools/perf_attribution_report.py"
  all_tests = [entry["test"] for entry in MODELS.values()]
  compile_commands: list[str] = []
  report: dict[str, dict] = {}

  # The two off runs use the exact performance_test.cpp generator and seeds;
  # prepared only moves reference extractor/input setup outside the timed call.
  end_to_end_rows = profile_off / "end-to-end.ndjson"
  end_to_end_peak_rss_kb = run_ctest(
    args.stage, all_tests, "end_to_end", end_to_end_rows,
    profile_off / "end-to-end.ctest.log")
  end_to_end = read_rows(end_to_end_rows, len(MODELS), "end_to_end")
  prepared_rows = profile_off / "prepared.ndjson"
  prepared_peak_rss_kb = run_ctest(
    args.stage, all_tests, "prepared", prepared_rows,
    profile_off / "prepared.ctest.log")
  prepared = read_rows(prepared_rows, len(MODELS), "prepared")
  off_by_model = {row["model"]: row for row in end_to_end}
  prepared_by_model = {row["model"]: row for row in prepared}
  for model in MODELS:
    if off_by_model[model]["input_hash"] != prepared_by_model[model]["input_hash"]:
      raise RuntimeError(f"{model}: end-to-end/prepared input identities differ")

  with (evidence / "commands.txt").open("w", encoding="utf-8") as commands_file:
    for model, spec in MODELS.items():
      asset_root = args.yolo_root if spec["asset_group"] == "yolo" else args.ocr_root
      if spec["asset_group"] == "yolo":
        parameter = asset_root / f"{spec['stem']}.ncnn.param"
        weights = asset_root / f"{spec['stem']}.ncnn.bin"
      else:
        parameter = asset_root / f"{spec['stem']}.param"
        weights = asset_root / f"{spec['stem']}.bin"
      output_dir = profile_on / model
      output_dir.mkdir()
      compile_log = profile_logs / f"{model}-compile.log"
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
        f"--model-name={model}",
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
        f"--output-dir={output_dir}",
      ]
      compile_commands.append(shlex.join(command))
      compile_budget = 900 if model in {"yolov5x", "yolov5x_seg"} else 300
      compile_wall_seconds = run(
        command, compile_log, os.environ.copy(), timeout_seconds=compile_budget)
      (output_dir / "compile-time.json").write_text(
        json.dumps({"wall_seconds": compile_wall_seconds,
                    "budget_seconds": compile_budget}, indent=2) + "\n",
        encoding="utf-8")

      baseline_library = (
        args.stage / "test/Numerical/generated" / model / f"lib{model}.so")
      baseline_library_size = baseline_library.stat().st_size
      profile_path = output_dir / "profile.ndjson"
      diagnostic_perf = output_dir / "diagnostic-perf.ndjson"
      profile_peak_rss_kb = run_ctest(
        args.stage, [spec["test"]], "end_to_end", diagnostic_perf,
        output_dir / "profile-on.ctest.log", output_dir, profile_path)
      profile_rows = [
        json.loads(line) for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
      ]
      if len(profile_rows) != 7:
        raise RuntimeError(f"{model}: expected 7 profile invocations, got {len(profile_rows)}")
      perf_row = read_rows(
        diagnostic_perf, 1, "end_to_end", expected_status="diagnostic")[0]
      if perf_row.get("diagnostics", {}).get("gate_eligible") is not False or \
          perf_row.get("status") != "diagnostic":
        raise RuntimeError(f"{model}: profile-on perf was not marked diagnostic")
      if perf_row.get("input_hash") != off_by_model[model].get("input_hash"):
        raise RuntimeError(f"{model}: profile-on/off input identities differ")
      if any(row.get("input_hash") != perf_row.get("input_hash") for row in profile_rows):
        raise RuntimeError(f"{model}: profile/perf deterministic input hashes differ")
      profile_complete = all(
        row.get("complete") is True and
        row.get("summary", {}).get("event_mismatch_count") == 0
        for row in profile_rows)
      if not profile_complete:
        raise RuntimeError(f"{model}: profile runtime stream is incomplete or mismatched")

      run([
        sys.executable, str(profiler),
        f"--plan={output_dir / f'{model}.plan.json'}",
        f"--profile={profile_path}",
        f"--perf={diagnostic_perf}",
        "--mode=end_to_end",
        f"--output={output_dir / 'attribution-report.json'}",
      ], output_dir / "attribution.log", os.environ.copy())
      attribution = json.loads(
        (output_dir / "attribution-report.json").read_text(encoding="utf-8"))
      worker = attribution["runtime"].get("worker_attribution", {})
      report[model] = {
        "group": spec["group"],
        "profile_compile_wall_seconds": compile_wall_seconds,
        "profile_compile_budget_seconds": compile_budget,
        "profile_binary_size_bytes": (output_dir / f"lib{model}.so").stat().st_size,
        "profile_off_binary_size_bytes": baseline_library_size,
        "profile_binary_growth_percent": (
          100.0 * ((output_dir / f"lib{model}.so").stat().st_size /
                   baseline_library_size - 1.0)
          if baseline_library_size else None),
        "profile_rows": len(profile_rows),
        "profile_on_process_tree_peak_rss_kb": profile_peak_rss_kb,
        "input_hash": perf_row["input_hash"],
        "cpu_placement": {
          "verified": perf_row.get("diagnostics", {}).get(
            "cpu_placement_verified"),
          "status": perf_row.get("diagnostics", {}).get(
            "cpu_placement_status"),
          "cpu_list": perf_row.get("diagnostics", {}).get("cpu_list"),
          "observed_tasks": perf_row.get("diagnostics", {}).get(
            "observed_tasks"),
          "tasks_matching_mask": perf_row.get("diagnostics", {}).get(
            "tasks_matching_mask"),
        },
        "profile_on_compiled_mean_ms": perf_row["compiled_mean_ms"],
        "profile_on_cv": perf_row.get("cv"),
        "profile_off_compiled_mean_ms": off_by_model[model]["compiled_mean_ms"],
        "profile_off_cv": off_by_model[model].get("cv"),
        "end_to_end_ratio": off_by_model[model].get("ratio"),
        "prepared_ratio": prepared_by_model[model].get("ratio"),
        "profile_perturbation_percent": (
          100.0 * (perf_row["compiled_mean_ms"] /
                   off_by_model[model]["compiled_mean_ms"] - 1.0)
          if off_by_model[model]["compiled_mean_ms"] else None),
        "profile_off_cpu_placement": {
          "verified": off_by_model[model].get("diagnostics", {}).get(
            "cpu_placement_verified"),
          "status": off_by_model[model].get("diagnostics", {}).get(
            "cpu_placement_status"),
          "cpu_list": off_by_model[model].get("diagnostics", {}).get(
            "cpu_list"),
          "observed_tasks": off_by_model[model].get("diagnostics", {}).get(
            "observed_tasks"),
          "tasks_matching_mask": off_by_model[model].get(
            "diagnostics", {}).get("tasks_matching_mask"),
        },
        "prepared_ncnn_mean_ms": prepared_by_model[model]["ncnn_mean_ms"],
        "prepared_cpu_placement_verified": prepared_by_model[model].get(
          "diagnostics", {}).get("cpu_placement_verified"),
        "end_to_end_ncnn_mean_ms": off_by_model[model]["ncnn_mean_ms"],
        "end_to_end_compiled_mean_ms": off_by_model[model]["compiled_mean_ms"],
        "prepared_compiled_mean_ms": prepared_by_model[model]["compiled_mean_ms"],
        "prepared_ncnn_to_end_to_end_ncnn_ratio": (
          prepared_by_model[model]["ncnn_mean_ms"] /
          off_by_model[model]["ncnn_mean_ms"]
          if off_by_model[model]["ncnn_mean_ms"] else None),
        "runtime_complete": attribution["runtime"]["complete"],
        "runtime_unknown_wall_ns": attribution["runtime"]["unknown_time_ns"],
        "worker_attribution_complete_for_observed_events":
          worker.get("complete_for_observed_events"),
        "top_worker_operations": worker.get("top_operations_by_wall_union", [])[:10],
      }
      print(f"profiled {model}", flush=True)
    commands_file.write("\n".join(compile_commands) + "\n")

  placement_verified = all(
    row.get("diagnostics", {}).get("cpu_placement_verified") is True
    for row in [*end_to_end, *prepared]) and all(
      entry["cpu_placement"].get("verified") is True
      for entry in report.values())
  summary = {
    "schema_version": 1,
    "study": "P27 diagnostic cohorts; profile-on is not formal timing",
    "profile_off_end_to_end_rows": len(end_to_end),
    "profile_off_prepared_rows": len(prepared),
    "profile_off_end_to_end_batch_process_tree_peak_rss_kb":
      end_to_end_peak_rss_kb,
    "profile_off_prepared_batch_process_tree_peak_rss_kb":
      prepared_peak_rss_kb,
    "same_input_hash_across_boundaries": True,
    "cpu_placement_verified_all": placement_verified,
    "worker_attribution_complete_all": all(
      entry.get("worker_attribution_complete_for_observed_events") is True
      for entry in report.values()),
    "models": report,
  }
  (evidence / "cohort-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
