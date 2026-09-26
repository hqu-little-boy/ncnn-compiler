#!/usr/bin/env python3
"""Collect P24 profile-on allocation attribution and profile perturbation."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


MODELS = {
    "pp_ocrv5_server_rec": (
        "PP-OCRv5_server_rec",
        "3x48x320",
        3 * 48 * 320,
        40 * 18385,
    ),
    "pp_ocrv5_server_det_static": (
        "PP-OCRv5_server_det",
        "3x640x640",
        3 * 640 * 640,
        640 * 640,
    ),
    "pp_formulanet_plus_s_encoder": (
        "PP-FormulaNet_plus_S_encoder",
        "1x384x384",
        384 * 384,
        144 * 2048,
    ),
    "chineseocr_lite_anglenet": (
        "Chineseocr_Lite_AngleNet",
        "3x32x192",
        3 * 32 * 192,
        2,
    ),
    "pp_ocrv6_tiny_rec": (
        "PP-OCRv6_tiny_rec",
        "3x48x320",
        3 * 48 * 320,
        40 * 6906,
    ),
    "squeezenet_v1_1": (
        "squeezenet_v1.1",
        "3x227x227",
        3 * 227 * 227,
        1000,
    ),
}

WARMUPS = 2
ITERATIONS = 5


def run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def compile_profile_model(stage: Path, compiler_root: Path, model_root: Path,
                          output: Path, name: str, spec: tuple) -> dict:
    file_stem, shape, _, _ = spec
    if name == "squeezenet_v1_1":
        asset_root = compiler_root / "test/third_party/ncnn/examples"
    else:
        asset_root = model_root
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(stage / "tools/ncnn-compile"),
        f"--driver={stage / 'tools/ncnn-mlir-driver'}",
        f"--opt={stage / 'bin/ncnn-mlir-opt'}",
        "--translate=/usr/bin/mlir-translate-21",
        "--clang=/usr/bin/clang-21",
        "--nm=/usr/bin/llvm-nm-21",
        "--readelf=/usr/bin/llvm-readelf-21",
        "--llvm-as=/usr/bin/llvm-as-21",
        f"--param={asset_root / f'{file_stem}.param'}",
        f"--bin={asset_root / f'{file_stem}.bin'}",
        f"--model-name={name}",
        f"--input-shape={shape}",
        "--matmul-packing=auto",
        "--march=x86-64-v3",
        "--target-feature=+avx2",
        "--target-feature=+fma",
        "--vector-mode=fixed-width",
        "--threads=6",
        "--profile",
        "--emit-manifest",
        "--emit-execution-plan",
        f"--output-dir={output}",
    ]
    run(command, output / "compile.log")
    return json.loads((output / f"{name}.plan.json").read_text(encoding="utf-8"))


def invocation_timings(library_path: Path, name: str, input_elements: int,
                       output_elements: int, count: int) -> list[int]:
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, name)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [float_pointer, float_pointer]
    function.restype = ctypes.c_int
    input_data = (ctypes.c_float * input_elements)()
    output_data = (ctypes.c_float * output_elements)()

    for _ in range(WARMUPS):
        status = function(input_data, output_data)
        if status:
            raise RuntimeError(f"{name}: warmup returned status {status}")

    timings = []
    for _ in range(count):
        start = time.perf_counter_ns()
        status = function(input_data, output_data)
        elapsed = time.perf_counter_ns() - start
        if status:
            raise RuntimeError(f"{name}: invocation returned status {status}")
        timings.append(elapsed)
    return timings


def profile_environment(profile_path: Path, plan: dict) -> None:
    target = plan["target"]
    os.environ.update({
        "NCNN_PROFILE_SCHEMA": "2",
        "NCNN_PROFILE_PATH": str(profile_path),
        "NCNN_PROFILE_MODE": "end_to_end",
        "NCNN_PROFILE_MODEL": plan["model"],
        "NCNN_PROFILE_PLAN_HASH": str(plan["plan_hash"]),
        "NCNN_PROFILE_BUILD_IDENTITY": str(plan["build_identity"]),
        "NCNN_PROFILE_PLAN_REVISION": plan["plan_revision"],
        "NCNN_PROFILE_ATTRIBUTION_REVISION": plan["attribution_revision"],
        "NCNN_PROFILE_THREADS": str(target["threads"]),
        "NCNN_PROFILE_TARGET": target["triple"],
        "OMP_NUM_THREADS": str(target["threads"]),
        "OMP_DYNAMIC": "FALSE",
    })


def allocation_metrics(profile: dict) -> dict:
    events = profile.get("events", [])
    allocation_events = [event for event in events
                         if event.get("category") == "allocation"]
    deallocation_events = [event for event in events
                           if event.get("category") == "deallocation"]
    summary = profile.get("summary", {})
    allocation_count = summary.get("allocation_count")
    timed_allocation_calls = sum(
        event.get("calls", 0)
        for event in allocation_events
        if isinstance(event.get("exclusive_ns"), int)
        and event.get("exclusive_ns", 0) > 0
    )
    allocation_ns = sum(
        event.get("exclusive_ns", 0)
        for event in allocation_events
        if isinstance(event.get("exclusive_ns"), int)
    )
    timed_deallocation_calls = sum(
        event.get("calls", 0)
        for event in deallocation_events
        if isinstance(event.get("exclusive_ns"), int)
        and event.get("exclusive_ns", 0) > 0
    )
    deallocation_ns = (
        sum(
            event.get("exclusive_ns", 0)
            for event in deallocation_events
            if isinstance(event.get("exclusive_ns"), int)
        )
        if timed_deallocation_calls > 0
        else None
    )
    operation_sites = {}
    for event in events:
        if event.get("category") != "operation":
            continue
        duration = event.get("exclusive_ns")
        if isinstance(duration, int) and duration > 0:
            operation_sites[str(event["id"])] = duration
    other_sites = sorted(operation_sites.values(), reverse=True)
    ranked_costs = sorted([allocation_ns, *other_sites], reverse=True)
    allocation_rank = ranked_costs.index(allocation_ns) + 1
    top_level_ns = summary.get("top_level_time_ns")
    return {
        "complete": profile.get("complete"),
        "event_mismatch_count": summary.get("event_mismatch_count"),
        "allocation_count": allocation_count,
        "deallocation_count": summary.get("deallocation_count"),
        "allocation_bytes": summary.get("allocation_bytes"),
        "deallocation_bytes": summary.get("deallocation_bytes"),
        "peak_live_bytes": summary.get("peak_live_bytes"),
        "peak_live_proven": summary.get("peak_live_proven"),
        "allocation_event_site_count": len(allocation_events),
        "timed_allocation_calls": timed_allocation_calls,
        "allocation_time_coverage": (
            timed_allocation_calls / allocation_count
            if isinstance(allocation_count, int) and allocation_count > 0
            else None
        ),
        "allocation_exclusive_ns": allocation_ns,
        "deallocation_time_measured": timed_deallocation_calls > 0,
        "timed_deallocation_calls": timed_deallocation_calls,
        "deallocation_exclusive_ns": deallocation_ns,
        "allocation_time_share_of_top_level_percent": (
            allocation_ns * 100.0 / top_level_ns
            if isinstance(top_level_ns, int) and top_level_ns > 0
            else None
        ),
        "allocation_site_rank_among_operation_sites": allocation_rank,
        "top_level_time_ns": top_level_ns,
        "operation_site_count_timed": len(operation_sites),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--compiler-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "6", "OMP_DYNAMIC": "FALSE"})

    summary = {
        "schema_version": 1,
        "target": "x86-64-v3/AVX2/FMA",
        "threads": 6,
        "warmups": WARMUPS,
        "timed_invocations": ITERATIONS,
        "input_data": "all-zero diagnostic input; not formal benchmark data",
        "models": {},
    }
    for name, spec in MODELS.items():
        profile_dir = args.evidence / "profile-on" / name
        plan = compile_profile_model(
            args.stage,
            args.compiler_root,
            args.model_root,
            profile_dir,
            name,
            spec,
        )
        profile_path = profile_dir / "profile.ndjson"
        profile_environment(profile_path, plan)
        _, _, input_elements, output_elements = spec
        profile_timings = invocation_timings(
            profile_dir / f"lib{name}.so", name, input_elements, output_elements,
            ITERATIONS,
        )
        profiles = [json.loads(line) for line in profile_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(profiles) != ITERATIONS + WARMUPS:
            raise RuntimeError(
                f"{name}: expected {ITERATIONS + WARMUPS} profile rows, "
                f"got {len(profiles)}"
            )
        timed_profiles = profiles[-ITERATIONS:]
        for index, profile in enumerate(timed_profiles, start=1):
            (profile_dir / f"invocation-{index:02}.profile.json").write_text(
                json.dumps(profile, indent=2) + "\n", encoding="utf-8")

        off_dir = args.stage / "test/Numerical/generated" / name
        off_plan = json.loads((off_dir / f"{name}.plan.json").read_text(
            encoding="utf-8"))
        profile_environment(profile_path, off_plan)
        os.environ.pop("NCNN_PROFILE_PATH", None)
        os.environ.pop("NCNN_PROFILE_SCHEMA", None)
        off_timings = invocation_timings(
            off_dir / f"lib{name}.so", name, input_elements, output_elements,
            ITERATIONS,
        )

        metric_rows = [allocation_metrics(profile) for profile in timed_profiles]
        med = lambda rows, field: statistics.median(row[field] for row in rows)
        deallocation_ns_values = [
            row["deallocation_exclusive_ns"]
            for row in metric_rows
            if row["deallocation_exclusive_ns"] is not None
        ]
        summary["models"][name] = {
            "profile_on_plan_hash": plan["plan_hash"],
            "profile_on_build_identity": plan["build_identity"],
            "profile_off_plan_hash": off_plan["plan_hash"],
            "profile_off_build_identity": off_plan["build_identity"],
            "profile_on_median_ms": statistics.median(profile_timings) / 1e6,
            "profile_off_median_ms": statistics.median(off_timings) / 1e6,
            "profile_on_perturbation_percent": (
                statistics.median(profile_timings)
                / statistics.median(off_timings) - 1.0
            ) * 100.0,
            "allocation": {
                "median_allocation_count": med(metric_rows, "allocation_count"),
                "median_deallocation_count": med(metric_rows, "deallocation_count"),
                "median_allocation_time_coverage": med(
                    metric_rows, "allocation_time_coverage"),
                "median_allocation_exclusive_ns": med(
                    metric_rows, "allocation_exclusive_ns"),
                "deallocation_time_measured": all(
                    row["deallocation_time_measured"] for row in metric_rows),
                "median_deallocation_exclusive_ns": (
                    statistics.median(deallocation_ns_values)
                    if deallocation_ns_values else None
                ),
                "median_top_level_time_ns": med(metric_rows, "top_level_time_ns"),
                "median_allocation_time_share_percent": med(
                    metric_rows, "allocation_time_share_of_top_level_percent"),
                "median_allocation_site_rank": med(
                    metric_rows, "allocation_site_rank_among_operation_sites"),
                "all_profiles_complete": all(
                    row["complete"] is True for row in metric_rows),
                "all_event_mismatches_zero": all(
                    row["event_mismatch_count"] == 0 for row in metric_rows),
                "all_peaks_proven": all(
                    row["peak_live_proven"] is True for row in metric_rows),
                "per_invocation": metric_rows,
            },
        }
        print(f"profiled {name}", flush=True)

    destination = args.evidence / "allocation-qualification.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n",
                           encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
