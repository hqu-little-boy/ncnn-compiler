#!/usr/bin/env python3
"""Prepare V2 shaders and exercise the persistent V3 Vulkan runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from run_v0 import run_logged, sha256


def prepare_shader(args, name, operation, param, cpu_library, cpu_symbol, output):
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_v2.py")),
        "--ncnn-driver",
        args.ncnn_driver,
        "--ncnn-mlir-opt",
        args.ncnn_mlir_opt,
        "--mlir-opt",
        args.mlir_opt,
        "--spirv-val",
        args.spirv_val,
        "--spirv-dis",
        args.spirv_dis,
        "--spirv-extract",
        args.spirv_extract,
        "--runner",
        args.v2_runner,
        "--operation",
        operation,
        "--elements",
        "257",
        "--binary-symbol",
        "model_vulkan",
        "--cpu-library",
        cpu_library,
        "--cpu-symbol",
        cpu_symbol,
        "--model-param",
        param,
        "--model-bin",
        args.empty_bin,
        "--device-match",
        args.device_match,
        "--output-dir",
        str(output),
        "--prepare-only",
        "--validation",
    ]
    run_logged(f"prepare-{name}-shader", command, args.logs, env=args.env)
    plan = output / "execution-plan.json"
    shader = output / "kernel.spv"
    if not plan.is_file() or not shader.is_file():
        raise RuntimeError(f"{name} shader package is incomplete")
    return plan, shader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncnn-driver", required=True)
    parser.add_argument("--ncnn-mlir-opt", required=True)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--spirv-val", required=True)
    parser.add_argument("--spirv-dis", required=True)
    parser.add_argument("--spirv-extract", required=True)
    parser.add_argument("--v2-runner", required=True)
    parser.add_argument("--runtime-runner", required=True)
    parser.add_argument("--weight-runner", required=True)
    parser.add_argument("--conv-cpu-library", required=True)
    parser.add_argument("--conv-model-param", required=True)
    parser.add_argument("--conv-model-bin", required=True)
    parser.add_argument("--cpu-library", required=True)
    parser.add_argument("--cpu-symbol", required=True)
    parser.add_argument("--add-cpu-library", required=True)
    parser.add_argument("--add-cpu-symbol", required=True)
    parser.add_argument("--relu-cpu-library", required=True)
    parser.add_argument("--relu-cpu-symbol", required=True)
    parser.add_argument("--empty-bin", required=True)
    parser.add_argument("--device-match", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--icd")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.logs = output / "logs"
    args.logs.mkdir(exist_ok=True)
    args.env = os.environ.copy()
    if args.icd:
        args.env["VK_ICD_FILENAMES"] = args.icd

    add_plan, add_shader = prepare_shader(
        args,
        "add",
        "add",
        Path(__file__).with_name("add.param"),
        args.add_cpu_library,
        args.add_cpu_symbol,
        output / "prepare-add",
    )
    relu_plan, relu_shader = prepare_shader(
        args,
        "relu",
        "relu",
        Path(__file__).with_name("relu.param"),
        args.relu_cpu_library,
        args.relu_cpu_symbol,
        output / "prepare-relu",
    )
    conv_output = output / "prepare-conv2d"
    conv_output.mkdir(exist_ok=True)
    conv_command = [
        sys.executable,
        str(Path(__file__).with_name("run_v2.py")),
        "--ncnn-driver",
        args.ncnn_driver,
        "--ncnn-mlir-opt",
        args.ncnn_mlir_opt,
        "--mlir-opt",
        args.mlir_opt,
        "--spirv-val",
        args.spirv_val,
        "--spirv-dis",
        args.spirv_dis,
        "--spirv-extract",
        args.spirv_extract,
        "--runner",
        args.v2_runner,
        "--operation",
        "conv2d",
        "--elements",
        "32",
        "--input-shape",
        "1x4x4",
        "--kernel-size",
        "3x3",
        "--stride",
        "1x1",
        "--padding",
        "1,1,1,1",
        "--output-channels",
        "2",
        "--conv-bias",
        "--binary-symbol",
        "model_vulkan",
        "--cpu-library",
        args.conv_cpu_library,
        "--cpu-symbol",
        "v2_conv3x3",
        "--model-param",
        args.conv_model_param,
        "--model-bin",
        args.conv_model_bin,
        "--device-match",
        args.device_match,
        "--output-dir",
        str(conv_output),
        "--prepare-only",
        "--validation",
    ]
    run_logged("prepare-conv2d-shader", conv_command, args.logs, env=args.env)
    conv_plan = conv_output / "execution-plan.json"
    conv_shader = conv_output / "kernel.spv"
    if not conv_plan.is_file() or not conv_shader.is_file():
        raise RuntimeError("Conv2D shader package is incomplete")

    runtime_evidence = output / "runtime-evidence.json"
    command = [
        args.runtime_runner,
        "--add-shader",
        str(add_shader),
        "--relu-shader",
        str(relu_shader),
        "--cpu-library",
        str(Path(args.cpu_library).resolve()),
        "--cpu-symbol",
        args.cpu_symbol,
        "--device-match",
        args.device_match,
        "--evidence",
        str(runtime_evidence),
        "--calls",
        "3",
        "--concurrent-calls",
        "4",
    ]
    if args.validation:
        command.append("--validation")
    run_logged("v3-persistent-runtime", command, args.logs, env=args.env)
    result = json.loads(runtime_evidence.read_text(encoding="utf-8"))
    if (
        result.get("kernel_count") != 2
        or result.get("validation_errors") != 0
        or args.device_match.lower() not in result.get("device_name", "").lower()
    ):
        raise RuntimeError("V3 runtime evidence does not satisfy its dispatch gate")

    weight_evidence = output / "weight-runtime-evidence.json"
    weight_command = [
        args.weight_runner,
        "--plan",
        str(conv_plan),
        "--shader",
        str(conv_shader),
        "--weight-file",
        str(Path(args.conv_model_bin).resolve()),
        "--cpu-library",
        str(Path(args.conv_cpu_library).resolve()),
        "--cpu-symbol",
        "v2_conv3x3",
        "--device-match",
        args.device_match,
        "--evidence",
        str(weight_evidence),
    ]
    if args.validation:
        weight_command.append("--validation")
    run_logged("v3-weight-runtime", weight_command, args.logs, env=args.env)
    weight_result = json.loads(weight_evidence.read_text(encoding="utf-8"))
    if (
        weight_result.get("weight_upload_count") != 2
        or weight_result.get("repeated_calls") != 3
        or weight_result.get("validation_errors") != 0
        or args.device_match.lower()
        not in weight_result.get("device_name", "").lower()
    ):
        raise RuntimeError("V3 persistent-weight evidence did not pass")

    metadata = {
        "fixture_id": "v3-f32-add-relu-runtime-257",
        "plan_schema": 1,
        "runtime_context_calls": result["calls"],
        "same_context_concurrent_calls": result["concurrent_calls"],
        "device_match": args.device_match,
        "validation_enabled": args.validation,
        "cpu_library_sha256": sha256(args.cpu_library),
        "conv_cpu_library_sha256": sha256(args.conv_cpu_library),
        "conv_model_param_sha256": sha256(args.conv_model_param),
        "conv_model_bin_sha256": sha256(args.conv_model_bin),
        "runtime_execution_plan_sha256": sha256(
            str(runtime_evidence) + ".execution-plan.json"
        ),
        "add_execution_plan_sha256": hashlib.sha256(add_plan.read_bytes()).hexdigest(),
        "relu_execution_plan_sha256": hashlib.sha256(relu_plan.read_bytes()).hexdigest(),
        "conv_execution_plan_sha256": hashlib.sha256(conv_plan.read_bytes()).hexdigest(),
        "add_spirv_sha256": sha256(add_shader),
        "relu_spirv_sha256": sha256(relu_shader),
        "conv_spirv_sha256": sha256(conv_shader),
        "runtime_result": result,
        "runtime_inputs_sha256": sha256(str(runtime_evidence) + ".inputs.bin"),
        "runtime_outputs_sha256": sha256(str(runtime_evidence) + ".outputs.bin"),
        "weight_runtime_result": weight_result,
        "weight_runtime_inputs_sha256": sha256(str(weight_evidence) + ".inputs.bin"),
        "weight_runtime_outputs_sha256": sha256(str(weight_evidence) + ".outputs.bin"),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"V3 runtime evidence: {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"run_v3.py: {error}", file=sys.stderr)
        sys.exit(1)
