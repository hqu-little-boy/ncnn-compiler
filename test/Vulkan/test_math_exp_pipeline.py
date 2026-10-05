#!/usr/bin/env python3
"""Verify math.exp is converted before the GPU Vulkan runner pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--spirv-extract", required=True)
    parser.add_argument("--spirv-val", required=True)
    parser.add_argument("--spirv-dis", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    probe = Path(args.probe)
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vulkan-math-exp-") as temporary:
        temporary_path = Path(temporary)
        bare_ir = temporary_path / "bare.mlir"
        bare = subprocess.run(
            [args.mlir_opt, str(probe), "--test-vulkan-runner-pipeline",
             "--verify-each", "-o", str(bare_ir)],
            capture_output=True,
            text=True,
        )
        if bare.returncode == 0 or "math.exp" not in (bare.stdout + bare.stderr):
            raise RuntimeError(
                "the runner-only pipeline no longer reproduces the unconverted math.exp failure"
            )

        binary_ir = evidence / "math-exp-vulkan-binary.mlir"
        run([
            args.mlir_opt,
            str(probe),
            "--convert-math-to-spirv",
            "--test-vulkan-runner-pipeline",
            "--verify-each",
            "-o",
            str(binary_ir),
        ])
        binary_text = binary_ir.read_text(encoding="utf-8")
        if "gpu.binary @v52_exp" not in binary_text or "math.exp" in binary_text:
            raise RuntimeError("math.exp did not lower to the expected GPU binary")

        shader = evidence / "math-exp.spv"
        run([
            args.spirv_extract,
            "--binary-symbol",
            "v52_exp",
            str(binary_ir),
            "-o",
            str(shader),
        ])
        run([args.spirv_val, "--target-env", "vulkan1.1", str(shader)])
        disassembly = evidence / "math-exp.spvasm"
        run([args.spirv_dis, str(shader), "-o", str(disassembly)])
        text = disassembly.read_text(encoding="utf-8")
        if 'OpExtInstImport "GLSL.std.450"' not in text or not re.search(
            r"OpExtInst\s+%\w+\s+%\w+\s+Exp\s", text
        ):
            raise RuntimeError("SPIR-V does not contain the GLSL.std.450 Exp instruction")
        if 'OpEntryPoint GLCompute' not in text or "OpExecutionMode" not in text:
            raise RuntimeError("SPIR-V is missing its compute entry point or workgroup size")

        result = {
            "probe": str(probe),
            "bare_runner_pipeline_exit_code": bare.returncode,
            "math_conversion_order": ["convert-math-to-spirv", "test-vulkan-runner-pipeline"],
            "binary_symbol": "v52_exp",
            "spirv_validation": "passed (vulkan1.1)",
            "glsl_ext_inst": "Exp",
            "binary_ir": str(binary_ir),
            "spirv": str(shader),
            "disassembly": str(disassembly),
        }
        (evidence / "math-exp-evidence.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
