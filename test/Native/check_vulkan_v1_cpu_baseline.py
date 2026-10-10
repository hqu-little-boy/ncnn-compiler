#!/usr/bin/env python3
"""Compare the refactored CPU pipeline with the captured V1 baseline."""

import argparse
import ctypes
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--translate", required=True)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--llvm-as", required=True)
    parser.add_argument("--param", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpu-baseline-", dir=args.work_dir) as temp:
        output_dir = Path(temp) / "model"
        run(
            [
                args.compiler,
                "--driver",
                args.driver,
                "--opt",
                args.opt,
                "--translate",
                args.translate,
                "--clang",
                args.clang,
                "--nm",
                args.nm,
                "--readelf",
                args.readelf,
                "--llvm-as",
                args.llvm_as,
                str(args.param),
                "--bin",
                str(args.bin),
                "--model-name",
                baseline["model"],
                "--output-dir",
                str(output_dir),
                "--emit",
                "all",
                "--emit-manifest",
                "--emit-execution-plan",
                "--threads=1",
                "--vector-width=0",
                "-O0",
            ]
        )

        model = baseline["model"]
        artifact_paths = {
            "header": output_dir / f"{model}.h",
            "manifest": output_dir / f"{model}.json",
            "execution_plan": output_dir / f"{model}.plan.json",
            "linalg_ir": output_dir / "model.linalg.mlir",
        }
        actual_hashes = {key: sha256(path) for key, path in artifact_paths.items()}
        expected_hashes = baseline["artifact_sha256"]
        if actual_hashes != expected_hashes:
            raise RuntimeError(
                "CPU pipeline artifacts differ from the captured baseline: "
                f"expected={expected_hashes}, actual={actual_hashes}"
            )

        manifest = json.loads(artifact_paths["manifest"].read_text())
        shape = baseline["input_shape"]
        expected_io = {
            "element_type": "f32",
            "dynamic_dim_mask": 0,
            "shape": shape,
        }
        for section in ("inputs", "outputs"):
            entries = manifest.get(section, [])
            if len(entries) != 1 or any(
                entries[0].get(key) != value for key, value in expected_io.items()
            ):
                raise RuntimeError(f"CPU {section} contract differs: {entries}")

        plan = json.loads(artifact_paths["execution_plan"].read_text())
        if plan.get("plan_hash") != baseline["cpu_plan_hash"]:
            raise RuntimeError(
                "CPU execution-plan identity differs from the captured baseline"
            )

        element_count = 1
        for extent in shape:
            element_count *= extent
        input_values = [(index - element_count // 2) / 10.0
                        for index in range(element_count)]
        input_bytes = struct.pack(f"<{element_count}f", *input_values)
        rounded_inputs = struct.unpack(f"<{element_count}f", input_bytes)
        expected_values = [max(value, 0.0) for value in rounded_inputs]
        expected_bytes = struct.pack(f"<{element_count}f", *expected_values)
        input_buffer = (ctypes.c_float * element_count)(*rounded_inputs)
        output_buffer = (ctypes.c_float * element_count)()
        library = ctypes.CDLL(str(output_dir / f"lib{model}.so"))
        entry = getattr(library, model)
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        entry.restype = ctypes.c_int
        status = entry(input_buffer, output_buffer)
        actual_values = list(output_buffer)
        actual_bytes = struct.pack(f"<{element_count}f", *actual_values)
        max_abs_error = max(
            abs(actual - expected)
            for actual, expected in zip(actual_values, expected_values)
        )
        input_hash = hashlib.sha256(input_bytes).hexdigest()
        output_hash = hashlib.sha256(actual_bytes).hexdigest()
        expected_hash = hashlib.sha256(expected_bytes).hexdigest()
        tolerance = float(baseline["atol"])
        if input_hash != baseline["input_sha256"]:
            raise RuntimeError("deterministic CPU input differs from baseline")
        if status != 0 or max_abs_error > tolerance:
            raise RuntimeError(
                f"CPU callable failed: status={status}, max_abs_error={max_abs_error}"
            )
        if output_hash != baseline["expected_output_sha256"] or output_hash != expected_hash:
            raise RuntimeError("CPU callable output differs from the captured golden")

        archived_dir = Path(tempfile.mkdtemp(prefix="artifacts-", dir=args.work_dir))
        shutil.copytree(output_dir, archived_dir / "model")
        for name, data in (("input.f32", input_bytes),
                           ("cpu-output.f32", actual_bytes),
                           ("scalar-golden.f32", expected_bytes)):
            (args.work_dir / name).write_bytes(data)
        artifact_files = {
            key: str((archived_dir / "model" / path.name).relative_to(args.work_dir))
            for key, path in artifact_paths.items()
        }
        library_copy = archived_dir / "model" / f"lib{model}.so"
        shutil.copyfile(args.param, args.work_dir / "model.param")
        shutil.copyfile(args.bin, args.work_dir / "model.bin")
        evidence = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_cpu_baseline_evidence",
            "baseline_revision": baseline["baseline_revision"],
            "fixture": baseline["fixture"],
            "model": model,
            "shape": shape,
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "expected_output_sha256": expected_hash,
            "max_abs_error": max_abs_error,
            "atol": tolerance,
            "status": status,
            "artifact_sha256": actual_hashes,
            "artifact_files": artifact_files,
            "cpu_library": str(library_copy.relative_to(args.work_dir)),
            "cpu_library_sha256": sha256(library_copy),
            "input_file": "input.f32",
            "output_file": "cpu-output.f32",
            "golden_file": "scalar-golden.f32",
            "source_model_files": {
                "param": "model.param", "param_sha256": sha256(args.param),
                "bin": "model.bin", "bin_sha256": sha256(args.bin),
            },
            "plan_hash": plan["plan_hash"],
            "build_identity": plan["build_identity"],
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
