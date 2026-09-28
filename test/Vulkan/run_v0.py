#!/usr/bin/env python3
"""Run and archive the opt-in Vulkan V0 Add device test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(name, command, log_dir, env=None, expect_success=True):
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
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"{name} unexpectedly succeeded")
    return result


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
    parser.add_argument("--mlir-translate", required=True)
    parser.add_argument("--spirv-val", required=True)
    parser.add_argument("--vulkaninfo", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--gpu-probe", required=True)
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
    lowered = output_dir / "add.lowered.mlir"
    shader = output_dir / "add.spv"
    evidence = output_dir / "device-result.json"

    run_logged(
        "mlir-opt-version", [args.mlir_opt, "--version"], logs, env=env
    )
    run_logged(
        "mlir-translate-version", [args.mlir_translate, "--version"], logs, env=env
    )
    run_logged("spirv-val-version", [args.spirv_val, "--version"], logs, env=env)
    vulkan_info = run_logged(
        "vulkaninfo-summary", [args.vulkaninfo, "--summary"], logs, env=env
    )
    gpu_probe_command = [
        args.mlir_opt,
        str(args.gpu_probe),
        "--spirv-attach-target=module=v0_add ver=v1.1 caps=Shader client_api=Vulkan",
        "--convert-gpu-to-spirv",
        "--verify-each",
        "-o", str(output_dir / "gpu-func-add-probe.mlir"),
    ]
    gpu_probe = subprocess.run(
        gpu_probe_command, text=True, capture_output=True, env=env
    )
    gpu_probe_record = {
        "argv": [str(item) for item in gpu_probe_command],
        "returncode": gpu_probe.returncode,
        "stdout": gpu_probe.stdout,
        "stderr": gpu_probe.stderr,
    }
    (logs / "gpu-func-add-lowering-probe.json").write_text(
        json.dumps(gpu_probe_record, indent=2) + "\n", encoding="utf-8"
    )
    if gpu_probe.returncode != 0 and (
        "failed to legalize operation 'memref.load'" not in gpu_probe.stderr
    ):
        raise RuntimeError(
            "GPU dialect probe failed for an unexpected reason; see "
            f"{logs / 'gpu-func-add-lowering-probe.json'}"
        )
    run_logged(
        "mlir-lowering",
        [
            args.mlir_opt,
            "--no-implicit-module",
            str(source),
            "--verify-each",
            "--spirv-lower-abi-attrs",
            "-o", str(lowered),
        ],
        logs,
        env=env,
    )
    run_logged(
        "spirv-serialization",
        [
            args.mlir_translate,
            "--no-implicit-module",
            "--serialize-spirv",
            str(lowered),
            "-o", str(shader),
        ],
        logs,
        env=env,
    )
    run_logged(
        "spirv-validation",
        [args.spirv_val, "--target-env", "vulkan1.1", str(shader)],
        logs,
        env=env,
    )

    run_device(
        args.runner, shader, cpu_library, args.cpu_symbol, args.device_match,
        evidence, logs, env, args.validation
    )

    metadata = {
        "fixture_id": "v0-f32-add-257-v1",
        "kernel_ir": "SPIR-V dialect MLIR (explicit storage-buffer ABI)",
        "gpu_func_memref_lowering": {
            "status": "passed" if gpu_probe.returncode == 0 else "blocked",
            "returncode": gpu_probe.returncode,
            "diagnostic": gpu_probe.stderr,
        },
        "target_environment": "Vulkan 1.1 / SPIR-V 1.3",
        "cpu_reference_library_sha256": sha256(cpu_library),
        "model_param_sha256": sha256(source.parent / "add.param"),
        "mlir_kernel_sha256": sha256(source),
        "spirv_sha256": sha256(shader),
        "inputs_sha256": sha256(str(evidence) + ".inputs.bin"),
        "outputs_sha256": sha256(str(evidence) + ".outputs.bin"),
        "device_evidence": json.loads(evidence.read_text(encoding="utf-8")),
        "vulkaninfo_summary": vulkan_info.stdout,
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
                   expect_success=False)

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
                   expect_success=False)

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
