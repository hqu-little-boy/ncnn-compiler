#!/usr/bin/env python3
"""Lower, validate, and execute a V2 single-op Vulkan fixture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys

from run_v0 import capture_device_summary, run_logged, sha256


MAX_STORAGE_BUFFER_BYTES = 128 * 1024 * 1024


def checked_resource_bytes(resource):
    shape = resource.get("shape")
    if resource.get("element_type") != "f32" or not isinstance(shape, list):
        raise RuntimeError("execution plan resources must be ranked FP32 tensors")
    if not 1 <= len(shape) <= 4:
        raise RuntimeError("execution plan resource rank is outside 1..4")
    elements = 1
    for extent in shape:
        if not isinstance(extent, int) or extent <= 0:
            raise RuntimeError("execution plan resource has a non-positive extent")
        elements *= extent
        if elements > MAX_STORAGE_BUFFER_BYTES // 4:
            raise RuntimeError("execution plan resource exceeds the storage-buffer limit")
    byte_size = elements * 4
    if resource.get("byte_size") != byte_size:
        raise RuntimeError("execution plan byte_size does not match its shape")
    return byte_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncnn-driver", required=True)
    parser.add_argument("--ncnn-mlir-opt", required=True)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--spirv-val", required=True)
    parser.add_argument("--spirv-dis", required=True)
    parser.add_argument("--spirv-extract", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--kernel")
    parser.add_argument("--operation", choices=("add", "relu", "copy", "identity", "reshape", "transpose", "conv2d"), required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--input-shape")
    parser.add_argument("--permutation")
    parser.add_argument("--kernel-size")
    parser.add_argument("--stride")
    parser.add_argument("--padding")
    parser.add_argument("--output-channels", type=int)
    parser.add_argument("--conv-bias", action="store_true")
    parser.add_argument("--binary-symbol", required=True)
    parser.add_argument("--cpu-library", required=True)
    parser.add_argument("--cpu-symbol", required=True)
    parser.add_argument("--model-param", required=True)
    parser.add_argument("--model-bin", required=True)
    parser.add_argument("--device-match", required=True)
    parser.add_argument("--vulkaninfo", default="vulkaninfo")
    parser.add_argument("--icd")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,42")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    if args.icd:
        env["VK_ICD_FILENAMES"] = args.icd
    device_info = None
    if not args.prepare_only:
        device_info = capture_device_summary(
            args.vulkaninfo, args.device_match, args.icd, output_dir, logs, env
        )

    plan = output_dir / "execution-plan.json"
    imported_ir = output_dir / "imported-model.mlir"
    device_ir = output_dir / "gpu-kernels.mlir"
    binary_ir = output_dir / "gpu-binary.mlir"
    shader = output_dir / "kernel.spv"
    disassembly = output_dir / "kernel.spvasm"
    evidence = output_dir / "device-result.json"
    shutil.copyfile(args.model_param, output_dir / "model.param")
    shutil.copyfile(args.model_bin, output_dir / "model.bin")

    run_logged("ncnn-driver-version", [args.ncnn_driver, "--version"], logs, env=env)
    run_logged("ncnn-mlir-opt-version", [args.ncnn_mlir_opt, "--version"], logs, env=env)
    run_logged("mlir-opt-version", [args.mlir_opt, "--version"], logs, env=env)
    run_logged("spirv-val-version", [args.spirv_val, "--version"], logs, env=env)
    if args.kernel:
        source = Path(args.kernel).resolve()
        lowering_command = [
            args.ncnn_mlir_opt,
            str(source),
            f"--lower-ncnn-vulkan-model=plan-path={plan}",
            "--verify-each",
            "-o",
            str(device_ir),
        ]
    else:
        source = imported_ir
        run_logged(
            "ncnn-model-import",
            [
                args.ncnn_driver,
                "--bin",
                str(Path(args.model_bin).resolve()),
                "--precision=f32",
                "--emit=mlir",
                str(Path(args.model_param).resolve()),
                "-o",
                str(imported_ir),
            ],
            logs,
            env=env,
        )
        lowering_command = [
            args.ncnn_mlir_opt,
            str(imported_ir),
            "--ncnn-vulkan-v2-prepare-pipeline",
            f"--lower-ncnn-vulkan-model=plan-path={plan}",
            "--verify-each",
            "-o",
            str(device_ir),
        ]
    run_logged("ncnn-vulkan-lowering", lowering_command, logs, env=env)
    lowered = device_ir.read_text(encoding="utf-8")
    for residual in (
        "ncnn.",
        "tosa.",
        "func.func",
        "linalg.",
        "tensor.",
        "unrealized_conversion_cast",
    ):
        if residual in lowered:
            raise RuntimeError(f"device lowering retained forbidden IR: {residual}")
    if lowered.count("gpu.module @") != 1 or lowered.count("gpu.func @") != 1:
        raise RuntimeError("V2 single-op lowering must produce exactly one GPU kernel")

    run_logged(
        "gpu-to-spirv-binary",
        [args.mlir_opt, str(device_ir), "--test-vulkan-runner-pipeline", "--verify-each", "-o", str(binary_ir)],
        logs,
        env=env,
    )
    converted = binary_ir.read_text(encoding="utf-8")
    if f"gpu.binary @{args.binary_symbol}" not in converted:
        raise RuntimeError("SPIR-V pipeline emitted an unexpected gpu.binary symbol")
    for residual in (
        "gpu.module",
        "gpu.func",
        "memref.",
        "func.func",
        "ncnn.",
        "tosa.",
        "linalg.",
        "tensor.",
        "unrealized_conversion_cast",
    ):
        if residual in converted:
            raise RuntimeError(f"serialized GPU module contains residual {residual}")

    run_logged(
        "spirv-extraction",
        [args.spirv_extract, "--binary-symbol", args.binary_symbol, str(binary_ir), "-o", str(shader)],
        logs,
        env=env,
    )
    run_logged(
        "spirv-validation",
        [args.spirv_val, "--target-env", "vulkan1.1", str(shader)],
        logs,
        env=env,
    )
    run_logged("spirv-disassembly", [args.spirv_dis, str(shader), "-o", str(disassembly)], logs, env=env)
    disassembly_text = disassembly.read_text(encoding="utf-8")
    if "; Version: 1.0" not in disassembly_text or (
        'OpExtension "SPV_KHR_storage_buffer_storage_class"' not in disassembly_text
    ):
        raise RuntimeError("shader does not match the Vulkan 1.1 / SPIR-V 1.0 target")
    if not re.search(rf'OpEntryPoint\s+GLCompute\s+%\S+\s+"{re.escape(args.operation)}"', disassembly_text):
        raise RuntimeError("SPIR-V is missing the planned compute entry point")
    if not re.search(r"OpExecutionMode\s+%\S+\s+LocalSize\s+64\s+1\s+1", disassembly_text):
        raise RuntimeError("SPIR-V has an unexpected workgroup size")
    expected_bindings = (
        4 if args.operation == "conv2d" else 3 if args.operation == "add" else 2
    )
    shader_decorations = {}
    for target, binding in re.findall(
        r"OpDecorate\s+(%\S+)\s+Binding\s+(\d+)", disassembly_text
    ):
        shader_decorations.setdefault(target, {})["binding"] = int(binding)
    for target, descriptor_set in re.findall(
        r"OpDecorate\s+(%\S+)\s+DescriptorSet\s+(\d+)", disassembly_text
    ):
        shader_decorations.setdefault(target, {})["set"] = int(descriptor_set)
    if len(shader_decorations) != expected_bindings or any(
        "binding" not in decoration or "set" not in decoration
        for decoration in shader_decorations.values()
    ):
        raise RuntimeError("SPIR-V descriptors do not match the generated plan")

    execution_plan = json.loads(plan.read_text(encoding="utf-8"))
    if execution_plan.get("schema_version") != 1 or execution_plan.get(
        "kind"
    ) != "ncnn.vulkan_execution_plan":
        raise RuntimeError("unsupported Vulkan execution-plan schema")
    resources_by_id = {}
    for resource in execution_plan["resources"]:
        resource_id = resource["id"]
        if resource_id in resources_by_id:
            raise RuntimeError("execution plan contains duplicate resource IDs")
        checked_resource_bytes(resource)
        resources_by_id[resource_id] = resource
    if execution_plan["kernels"][0]["operation"] != args.operation:
        raise RuntimeError("execution plan operation does not match the test fixture")
    if execution_plan["kernels"][0]["entry_point"] != args.operation:
        raise RuntimeError("execution plan entry point does not match SPIR-V")
    if execution_plan["kernels"][0]["local_size"] != [64, 1, 1]:
        raise RuntimeError("execution plan workgroup size does not match SPIR-V")
    kernel_plan = execution_plan["kernels"][0]
    planned_bindings = {
        (binding["set"], binding["binding"])
        for binding in kernel_plan["bindings"]
    }
    shader_bindings = {
        (decoration["set"], decoration["binding"])
        for decoration in shader_decorations.values()
    }
    if shader_bindings != planned_bindings:
        raise RuntimeError("SPIR-V descriptor set/bindings differ from the execution plan")
    bindings_by_index = sorted(kernel_plan["bindings"], key=lambda item: item["binding"])
    planned_resource_order = kernel_plan["reads"] + kernel_plan["writes"]
    if [item["resource_id"] for item in bindings_by_index] != planned_resource_order:
        raise RuntimeError("execution-plan bindings differ from test runner buffer order")
    if len(resources_by_id) != len(planned_resource_order) or len(
        kernel_plan["writes"]
    ) != 1:
        raise RuntimeError("single-op plan has an unexpected resource schedule")
    expected_read_kinds = (
        ["input", "weight", "weight"]
        if args.operation == "conv2d"
        else ["input", "input"]
        if args.operation == "add"
        else ["input"]
    )
    if len(kernel_plan["reads"]) != len(expected_read_kinds):
        raise RuntimeError("execution plan has an unexpected number of reads")
    if [resources_by_id[resource_id]["kind"] for resource_id in kernel_plan["reads"]] != expected_read_kinds:
        raise RuntimeError("execution plan read resources have unexpected roles")
    if resources_by_id[kernel_plan["writes"][0]]["kind"] != "output":
        raise RuntimeError("execution plan write resource is not a public output")
    if len(execution_plan["inputs"]) != (2 if args.operation == "add" else 1) or len(
        execution_plan["outputs"]
    ) != 1:
        raise RuntimeError("execution plan public I/O count does not match the operation")
    expected_runner_buffer_sizes = [args.elements * 4] * len(planned_resource_order)

    conv_hwcf_sha256 = None
    if args.operation == "conv2d":
        if not all((args.input_shape, args.kernel_size, args.stride,
                    args.padding, args.output_channels)):
            raise RuntimeError("Conv2D fixture is missing shape/parameter metadata")
        input_shape = [int(value) for value in args.input_shape.split("x")]
        kernel_size = [int(value) for value in args.kernel_size.split("x")]
        if len(input_shape) != 3 or any(value <= 0 for value in input_shape):
            raise RuntimeError("Conv2D input shape must be positive CHW")
        if len(kernel_size) != 2 or any(value <= 0 for value in kernel_size):
            raise RuntimeError("Conv2D kernel dimensions must be positive")
        input_channels, _, _ = input_shape
        kernel_height, kernel_width = kernel_size
        output_channels = args.output_channels
        input_elements = input_channels * input_shape[1] * input_shape[2]
        weight_count = (
            input_channels * kernel_height * kernel_width * output_channels
        )
        bias_count = output_channels if args.conv_bias else 0
        expected_runner_buffer_sizes = [
            input_elements * 4,
            weight_count * 4,
            output_channels * 4,
            args.elements * 4,
        ]
        weight_bytes = Path(args.model_bin).read_bytes()
        if len(weight_bytes) != 4 + (weight_count + bias_count) * 4:
            raise RuntimeError("FP32 Conv2D binary size does not match its parameters")
        if struct.unpack_from("<I", weight_bytes)[0] != 0:
            raise RuntimeError("Conv2D weight stream uses an unsupported storage flag")
        oihw = struct.unpack_from(f"<{weight_count}f", weight_bytes, 4)
        hwcf = [0.0] * weight_count
        for output in range(output_channels):
            for input_channel in range(input_channels):
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        source_index = (
                            ((output * input_channels + input_channel) * kernel_height + kh)
                            * kernel_width + kw
                        )
                        device_index = (
                            ((kh * kernel_width + kw) * input_channels + input_channel)
                            * output_channels + output
                        )
                        hwcf[device_index] = oihw[source_index]
        conv_hwcf_sha256 = hashlib.sha256(
            struct.pack(f"<{weight_count}f", *hwcf)
        ).hexdigest()
        bias = (
            struct.unpack_from(f"<{bias_count}f", weight_bytes, 4 + weight_count * 4)
            if bias_count
            else (0.0,) * output_channels
        )
        resources_by_name = {
            resource["name"]: resource for resource in execution_plan["resources"]
        }
        weight_resource = resources_by_name["weights"]
        if weight_resource["shape"] != [
            output_channels,
            input_channels,
            kernel_height,
            kernel_width,
        ] or weight_resource["logical_layout"] != "OIHW" or weight_resource[
            "physical_layout"
        ] != "HWCF":
            raise RuntimeError("execution plan weight layout does not describe OIHW to HWCF")
        if weight_resource["weight_sha256"] != hashlib.sha256(
            struct.pack(f"<{weight_count}f", *oihw)
        ).hexdigest():
            raise RuntimeError("execution plan weight hash differs from source OIHW data")
        if resources_by_name["bias"]["weight_sha256"] != hashlib.sha256(
            struct.pack(f"<{output_channels}f", *bias)
        ).hexdigest():
            raise RuntimeError("execution plan bias hash differs from device buffer")

    planned_buffer_sizes = [
        resources_by_id[resource_id]["byte_size"]
        for resource_id in planned_resource_order
    ]
    if planned_buffer_sizes != expected_runner_buffer_sizes:
        raise RuntimeError(
            "execution-plan buffer sizes do not match the Vulkan runner buffers"
        )
    if args.prepare_only:
        print(f"Prepared and validated V2 shader package: {output_dir}")
        return 0

    try:
        seeds = [int(value, 0) for value in args.seeds.split(",")]
    except ValueError as error:
        raise RuntimeError("seeds must be comma-separated uint32 integers") from error
    if not seeds or any(seed < 0 or seed > 0xFFFFFFFF for seed in seeds):
        raise RuntimeError("seeds must be nonempty uint32 values")
    device_evidence = {}
    for seed in seeds:
        evidence = output_dir / f"device-result-seed-{seed}.json"
        runner_command = [
            args.runner,
            "--shader",
            str(shader),
            "--cpu-library",
            str(Path(args.cpu_library).resolve()),
            "--cpu-symbol",
            args.cpu_symbol,
            "--operation",
            args.operation,
            "--elements",
            str(args.elements),
            "--seed",
            str(seed),
            "--device-match",
            args.device_match,
            "--evidence",
            str(evidence),
        ]
        if args.operation == "transpose":
            if not args.input_shape or not args.permutation:
                raise RuntimeError(
                    "transpose fixture requires input-shape and permutation"
                )
            runner_command.extend(
                ["--input-shape", args.input_shape, "--permutation", args.permutation]
            )
        elif args.operation == "conv2d":
            if not all((args.input_shape, args.kernel_size, args.stride,
                        args.padding, args.output_channels)):
                raise RuntimeError("Conv2D fixture is missing runner parameters")
            runner_command.extend(
                [
                    "--weight-file",
                    str(Path(args.model_bin).resolve()),
                    "--input-shape",
                    args.input_shape,
                    "--kernel-size",
                    args.kernel_size,
                    "--stride",
                    args.stride,
                    "--padding",
                    args.padding,
                    "--output-channels",
                    str(args.output_channels),
                ]
            )
            if args.conv_bias:
                runner_command.append("--conv-bias")
        elif (args.input_shape or args.permutation or args.kernel_size or
              args.stride or args.padding or args.output_channels or args.conv_bias):
            raise RuntimeError("shape/parameter metadata is operation-specific")
        if args.validation:
            runner_command.append("--validation")
        run_logged(
            f"vulkan-device-execution-seed-{seed}",
            runner_command,
            logs,
            env=env,
        )
        record = json.loads(evidence.read_text(encoding="utf-8"))
        record["inputs_sha256"] = sha256(str(evidence) + ".inputs.bin")
        record["outputs_sha256"] = sha256(str(evidence) + ".outputs.bin")
        device_evidence[str(seed)] = record

    metadata = {
        "fixture_id": f"v2-f32-{args.operation}-{args.elements}",
        "operation": args.operation,
        "conversion_pipeline": (
            "direct lower-ncnn-vulkan-model + MLIR test-vulkan-runner-pipeline"
            if args.kernel
            else "ncnn-mlir-driver + ncnn-vulkan-v2-prepare-pipeline + "
                 "lower-ncnn-vulkan-model + MLIR test-vulkan-runner-pipeline"
        ),
        "target_environment": "Vulkan 1.1 / SPIR-V 1.0 + SPV_KHR_storage_buffer_storage_class",
        "model_param_sha256": sha256(args.model_param),
        "model_bin_sha256": sha256(args.model_bin),
        "model_files": {"param": "model.param", "bin": "model.bin"},
        "cpu_reference_library_sha256": sha256(args.cpu_library),
        "gpu_kernel_source_sha256": sha256(source),
        "execution_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "gpu_ir_sha256": sha256(device_ir),
        "gpu_binary_ir_sha256": sha256(binary_ir),
        "spirv_sha256": sha256(shader),
        "spirv_disassembly_sha256": sha256(disassembly),
        "execution_plan": execution_plan,
        "seeds": seeds,
        "device_info": device_info,
        "vulkaninfo_summary": "vulkaninfo-summary.txt" if device_info else None,
        "device_evidence": device_evidence,
    }
    if conv_hwcf_sha256 is not None:
        metadata["device_hwcf_weights_sha256"] = conv_hwcf_sha256
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"V2 evidence: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"run_v2.py: {error}", file=sys.stderr)
        sys.exit(1)
