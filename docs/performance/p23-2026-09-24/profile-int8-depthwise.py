#!/usr/bin/env python3
"""Build profile-on P23 INT8 model artifacts and audit depthwise callbacks."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess


MODELS = {
    "pp_ocrv6_tiny_rec_int8": ("PP-OCRv6_tiny_rec_int8", "3x48x320", 3 * 48 * 320, 40 * 6906),
    "pp_ocrv5_mobile_rec_int8": ("PP-OCRv5_mobile_rec_int8", "3x48x320", 3 * 48 * 320, 40 * 18385),
    "pp_ocrv6_medium_rec_int8": ("PP-OCRv6_medium_rec_int8", "3x48x320", 3 * 48 * 320, 40 * 18710),
    "pp_ocrv6_small_rec_int8": ("PP-OCRv6_small_rec_int8", "3x48x320", 3 * 48 * 320, 40 * 18710),
    "pp_ocrv6_medium_det_int8": ("PP-OCRv6_medium_det_int8", "3x32x32", 3 * 32 * 32, 32 * 32),
}


def run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def compile_model(stage: Path, model_root: Path, work: Path,
                  name: str, variant: str) -> tuple[Path, dict]:
    native = variant == "native"
    output = work / variant / name
    output.mkdir(parents=True)
    file_stem, shape, _, _ = MODELS[name]
    model_file = model_root / f"{file_stem}.param"
    weights_file = model_root / f"{file_stem}.bin"
    command = [
        str(stage / "tools/ncnn-compile"),
        f"--driver={stage / 'tools/ncnn-mlir-driver'}",
        f"--opt={stage / 'bin/ncnn-mlir-opt'}",
        "--translate=/usr/bin/mlir-translate-21",
        "--clang=/usr/bin/clang-21",
        "--nm=/usr/bin/llvm-nm-21",
        "--readelf=/usr/bin/llvm-readelf-21",
        "--llvm-as=/usr/bin/llvm-as-21",
        f"--param={model_file}",
        f"--bin={weights_file}",
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
    ]
    if native:
        command.extend([
            "--target-feature=+avxvnni",
            "--tuning-profile=native-int8",
        ])
    command.append(f"--output-dir={output}")
    run(command, output / "compile.log")
    plan = json.loads((output / f"{name}.plan.json").read_text(encoding="utf-8"))
    return output, plan


def invoke_profile(name: str, output: Path, plan: dict,
                   input_elements: int, output_elements: int) -> dict:
    profile_path = output / "profile.ndjson"
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
    })
    library = ctypes.CDLL(str(output / f"lib{name}.so"))
    function = getattr(library, name)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [float_pointer, float_pointer]
    function.restype = ctypes.c_int
    input_data = (ctypes.c_float * input_elements)()
    output_data = (ctypes.c_float * output_elements)()
    status = function(input_data, output_data)
    if status:
        raise RuntimeError(f"{name}/{output.name}: model returned {status}")
    rows = [json.loads(line) for line in profile_path.read_text().splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{name}/{output.name}: expected one profile row")
    return rows[0]


def depthwise_audit(plan: dict, profile: dict) -> dict:
    operations = {operation["id"]: operation
                  for operation in plan.get("operations", [])}
    events = {event["id"]: event for event in profile.get("events", [])}
    depthwise = plan.get("conv_depthwise_operations", [])
    sites = []
    for record in depthwise:
        operation = operations.get(record["id"])
        profile_id = operation.get("profile_id") if operation else None
        event = events.get(profile_id) if profile_id is not None else None
        sites.append({
            "id": record["id"],
            "status": record.get("status"),
            "implementation": record.get("implementation"),
            "multiplier": record.get("multiplier"),
            "kernel_static": record.get("kernel_static"),
            "fallback_reason": record.get("fallback_reason"),
            "profile_id": profile_id,
            "calls": event.get("calls") if event else None,
            "exclusive_ns": event.get("exclusive_ns") if event else None,
        })
    selected_eligible = [
        site for site in sites
        if site["status"] == "selected"
        and site["implementation"] == "depthwise_simd"
        and site["multiplier"] == 1
        and site["kernel_static"] is True
    ]
    return {
        "eligible_selected_count": len(selected_eligible),
        "eligible_selected_calls_observed": sum(
            (site["calls"] or 0) > 0 for site in selected_eligible),
        "eligible_selected_unjoined_count": sum(
            site["calls"] is None for site in selected_eligible),
        "depthwise_sites": sites,
        "baseline_exclusive_time_status": "unknown_until_stable_source_id_join",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    summary = {"schema_version": 1, "target": "avx-vnni", "models": {}}
    for name, (_, _, input_elements, output_elements) in MODELS.items():
        summary["models"][name] = {}
        for variant in ("portable", "native"):
            output, plan = compile_model(
                args.stage, args.model_root, args.work_dir, name, variant)
            profile = invoke_profile(name, output, plan,
                                     input_elements, output_elements)
            audit = depthwise_audit(plan, profile)
            (output / "runtime-coverage.json").write_text(
                json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            summary["models"][name][variant] = {
                "plan_hash": plan["plan_hash"],
                "build_identity": plan["build_identity"],
                "profile_complete": profile.get("complete"),
                "event_mismatch_count": profile.get("summary", {}).get(
                    "event_mismatch_count"),
                "runtime_event_count": len(profile.get("events", [])),
                "depthwise": audit,
            }
    summary_path = args.work_dir / "runtime-coverage-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
