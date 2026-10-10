#!/usr/bin/env python3
"""Run and archive the opt-in Vulkan V0 Add device test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_v5 import parse_vulkan_device_summary


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(name, command, log_dir, env=None, expect_success=True,
               expected_failure=None):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    record = {
        "argv": [str(item) for item in command],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (Path(log_dir) / f"{name}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"{name} failed ({result.returncode}); see {Path(log_dir) / f'{name}.json'}"
        )
    if not expect_success:
        if result.returncode == 0:
            raise RuntimeError(f"{name} unexpectedly succeeded")
        if result.returncode < 0:
            raise RuntimeError(f"{name} terminated by signal {-result.returncode}")
        if not expected_failure or expected_failure not in (
                result.stdout + result.stderr):
            raise RuntimeError(f"{name} did not report the expected failure: "
                               f"{expected_failure!r}")
    return result


def capture_device_summary(vulkaninfo, device_match, icd, output, logs, env):
    result = run_logged("vulkaninfo-summary", [vulkaninfo, "--summary"], logs,
                        env=env)
    text = result.stdout + result.stderr
    (Path(output) / "vulkaninfo-summary.txt").write_text(text, encoding="utf-8")
    return parse_vulkan_device_summary(text, device_match, icd)


def run_device(runner, shader, cpu_library, symbol, device_match, evidence,
               log_dir, env, validation, probe_bytes=None):
    command = [
        str(runner),
        "--shader", str(shader),
        "--cpu-library", str(cpu_library),
        "--cpu-symbol", symbol,
        "--device-match", device_match,
        "--evidence", str(evidence),
    ]
    if validation:
        command.append("--validation")
    if probe_bytes is not None:
        command.extend(["--probe-buffer-bytes", str(probe_bytes)])
    return run_logged(
        f"device-{Path(evidence).stem}", command, log_dir, env=env
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--spirv-val", required=True)
    parser.add_argument("--spirv-dis", required=True)
    parser.add_argument("--spirv-extract", required=True)
    parser.add_argument("--vulkaninfo", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--cpu-library", required=True)
    parser.add_argument("--cpu-symbol", default="v0_add")
    parser.add_argument("--device-match", required=True)
    parser.add_argument("--icd", help="Force a Vulkan ICD manifest, e.g. llvmpipe")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--run-negative", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    if args.icd:
        env["VK_ICD_FILENAMES"] = args.icd

    source = Path(args.kernel).resolve()
    cpu_library = Path(args.cpu_library).resolve()
    gpu_binary_ir = output_dir / "gpu-binary.mlir"
    shader = output_dir / "add.spv"
    disassembly = output_dir / "add.spvasm"
    evidence = output_dir / "device-result.json"

    run_logged("mlir-opt-version", [args.mlir_opt, "--version"], logs, env=env)
    run_logged("spirv-val-version", [args.spirv_val, "--version"], logs, env=env)
    device_info = capture_device_summary(
        args.vulkaninfo, args.device_match, args.icd, output_dir, logs, env
    )
    run_logged(
        "gpu-func-to-vulkan-binary",
        [
            args.mlir_opt,
            str(source),
            "--test-vulkan-runner-pipeline",
            "--verify-each",
            "-o", str(gpu_binary_ir),
        ],
        logs,
        env=env,
    )
    converted_ir = gpu_binary_ir.read_text(encoding="utf-8")
    if "gpu.binary @v0_add" not in converted_ir:
        raise RuntimeError("Vulkan lowering did not emit gpu.binary @v0_add")
    for residual in ("gpu.func", "memref.load", "memref.store", "unrealized_conversion_cast"):
        if residual in converted_ir:
            raise RuntimeError(f"lowered GPU binary contains residual {residual}")

    run_logged(
        "spirv-extraction",
        [args.spirv_extract, str(gpu_binary_ir), "-o", str(shader)],
        logs,
        env=env,
    )
    run_logged(
        "spirv-validation",
        [args.spirv_val, "--target-env", "vulkan1.1", str(shader)],
        logs,
        env=env,
    )
    run_logged(
        "spirv-disassembly",
        [args.spirv_dis, str(shader), "-o", str(disassembly)],
        logs,
        env=env,
    )
    disassembly_text = disassembly.read_text(encoding="utf-8")
    if "; Version: 1.0" not in disassembly_text or (
        'OpExtension "SPV_KHR_storage_buffer_storage_class"' not in disassembly_text
    ):
        raise RuntimeError("serialized shader does not match the frozen SPIR-V 1.0 target")
    if not re.search(r'OpEntryPoint\s+GLCompute\s+%\S+\s+"add"', disassembly_text):
        raise RuntimeError("serialized shader is missing the add GLCompute entry point")
    if not re.search(r"OpExecutionMode\s+%\S+\s+LocalSize\s+64\s+1\s+1", disassembly_text):
        raise RuntimeError("serialized shader has the wrong workgroup size")
    bindings = {int(value) for value in re.findall(r"OpDecorate %\S+ Binding (\d+)", disassembly_text)}
    descriptor_sets = re.findall(r"OpDecorate %\S+ DescriptorSet (\d+)", disassembly_text)
    if not {0, 1, 2}.issubset(bindings) or descriptor_sets.count("0") < 3:
        raise RuntimeError("serialized shader does not expose descriptor set 0 bindings 0-2")

    run_device(
        args.runner, shader, cpu_library, args.cpu_symbol, args.device_match,
        evidence, logs, env, args.validation
    )

    metadata = {
        "fixture_id": "v0-f32-add-257-gpu-func-v1",
        "kernel_ir": "gpu.func Add lowered by MLIR test-vulkan-runner-pipeline",
        "conversion_pipeline": "--test-vulkan-runner-pipeline",
        "target_environment": "Vulkan 1.1 / SPIR-V 1.0 + SPV_KHR_storage_buffer_storage_class",
        "cpu_reference_library_sha256": sha256(cpu_library),
        "model_param_sha256": sha256(source.parent / "add.param"),
        "gpu_func_source_sha256": sha256(source),
        "gpu_binary_ir_sha256": sha256(gpu_binary_ir),
        "spirv_sha256": sha256(shader),
        "spirv_disassembly_sha256": sha256(disassembly),
        "descriptor_bindings": sorted(bindings),
        "inputs_sha256": sha256(str(evidence) + ".inputs.bin"),
        "outputs_sha256": sha256(str(evidence) + ".outputs.bin"),
        "device_evidence": json.loads(evidence.read_text(encoding="utf-8")),
        "vulkaninfo_summary": (output_dir / "vulkaninfo-summary.txt").read_text(
            encoding="utf-8"),
        "device_info": device_info,
        "icd_override": args.icd,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if args.run_negative:
        missing_evidence = output_dir / "negative-no-device.json"
        command = [
            args.runner,
            "--shader", str(shader),
            "--cpu-library", str(cpu_library),
            "--cpu-symbol", args.cpu_symbol,
            "--device-match", "NCNN_V0_NO_SUCH_DEVICE_9D62",
            "--evidence", str(missing_evidence),
        ]
        run_logged("negative-no-device", command, logs, env=env,
                   expect_success=False,
                   expected_failure="no compute device matched "
                   "'NCNN_V0_NO_SUCH_DEVICE_9D62'")

        properties = json.loads(evidence.read_text(encoding="utf-8"))
        too_large = int(properties["max_storage_buffer_range"]) + 1
        oversize_evidence = output_dir / "negative-oversize.json"
        command = [
            args.runner,
            "--shader", str(shader),
            "--cpu-library", str(cpu_library),
            "--cpu-symbol", args.cpu_symbol,
            "--device-match", args.device_match,
            "--evidence", str(oversize_evidence),
            "--probe-buffer-bytes", str(too_large),
        ]
        if args.validation:
            command.append("--validation")
        run_logged("negative-oversize-buffer", command, logs, env=env,
                   expect_success=False,
                   expected_failure="requested storage buffer exceeds "
                   "maxStorageBufferRange")

        invalid_shader = output_dir / "invalid.spv"
        shutil.copyfile(shader, invalid_shader)
        with invalid_shader.open("r+b") as output:
            output.write(b"\x00\x00\x00\x00")
        run_logged(
            "negative-invalid-spirv",
            [args.spirv_val, "--target-env", "vulkan1.1", str(invalid_shader)],
            logs,
            env=env,
            expect_success=False,
            expected_failure="Invalid SPIR-V magic number",
        )

        recovery_evidence = output_dir / "recovery-result.json"
        run_device(
            args.runner, shader, cpu_library, args.cpu_symbol, args.device_match,
            recovery_evidence, logs, env, args.validation
        )

    print(f"V0 evidence: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"run_v0.py: {error}", file=sys.stderr)
        sys.exit(1)
