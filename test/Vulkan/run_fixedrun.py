#!/usr/bin/env python3
"""Execute fixed-run candidate DSOs against CPU, legacy Vulkan and golden."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys

SCHEMA = "ncnn-vulkan-fixedrun-evidence/v1"
ATOL = 1.0e-6
RTOL = 1.0e-6
FIXTURES = ("relu", "add_relu", "conv_relu", "add_split_relu",
            "split_concat", "reshape", "transpose")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact(path: Path, stage: Path) -> dict:
    path = path.resolve()
    try:
        relative = path.relative_to(stage.resolve())
    except ValueError as error:
        raise RuntimeError(f"evidence path is outside stage: {path}") from error
    if not path.is_file():
        raise RuntimeError(f"evidence artifact is missing: {path}")
    return {"path": relative.as_posix(), "sha256": sha256_file(path)}


def write_bytes(path: Path, contents: bytes, stage: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != contents:
            raise RuntimeError(f"refusing to overwrite an existing artifact: {path}")
        return artifact(path, stage)
    with path.open("xb") as stream:
        stream.write(contents)
    return artifact(path, stage)


def snapshot_artifact(source: Path, destination: Path, stage: Path) -> dict:
    if source.is_symlink():
        raise RuntimeError(f"cannot snapshot a symlink target artifact: {source}")
    source = source.resolve()
    if not source.is_file():
        raise RuntimeError(f"cannot snapshot non-regular target artifact: {source}")
    source_mode = source.stat().st_mode
    snapshot = write_bytes(destination, source.read_bytes(), stage)
    destination.chmod(source_mode & 0o777 & ~0o222)
    return artifact(destination, stage)


def write_floats(path: Path, values: list[float], stage: Path) -> dict:
    return write_bytes(path, struct.pack(f"<{len(values)}f", *values), stage)


def checked_product(shape: list[int]) -> int:
    result = 1
    for extent in shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise RuntimeError(f"invalid static tensor shape: {shape}")
        result *= extent
    return result


def execute_cmake_command(command: dict, stage: Path, env: dict[str, str],
                          log_path: Path) -> dict:
    return run_command(command["argv"], Path(command["cwd"]), env, log_path,
                       stage, log_path.with_suffix(".stdout"),
                       log_path.with_suffix(".stderr"))


def run_command(argv: list[str], cwd: Path, env: dict[str, str], log: Path,
                stage: Path, stdout_path: Path | None = None,
                stderr_path: Path | None = None) -> dict:
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                            check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(result.stdout + b"\n--- stderr ---\n" + result.stderr)
    record = {"argv": argv, "cwd": str(cwd.resolve()), "returncode": result.returncode,
              "stdout_sha256": sha256_bytes(result.stdout),
              "stderr_sha256": sha256_bytes(result.stderr),
              "log": artifact(log, stage)}
    if stdout_path is not None:
        record["stdout"] = write_bytes(stdout_path, result.stdout, stage)
    if stderr_path is not None:
        record["stderr"] = write_bytes(stderr_path, result.stderr, stage)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"stdout: {result.stdout.decode(errors='replace')}\n"
            f"stderr: {result.stderr.decode(errors='replace')}")
    return record


def tensor_metadata(plan: dict) -> tuple[list[dict], list[dict]]:
    resources = {entry["id"]: entry for entry in plan["resources"]}
    inputs = [resources[item] for item in plan["inputs"]]
    outputs = [resources[item] for item in plan["outputs"]]
    for tensor in inputs + outputs:
        if tensor.get("element_type") != "f32":
            raise RuntimeError("fixed-run fixture must use static FP32 I/O")
        if checked_product(tensor["shape"]) * 4 != tensor["byte_size"]:
            raise RuntimeError("fixed-run tensor shape and byte size differ")
    return inputs, outputs


def make_inputs(input_tensors: list[dict], seed: int) -> list[list[float]]:
    values = []
    for port, tensor in enumerate(input_tensors):
        generator = random.Random(seed + port * 7919)
        values.append([generator.randint(-16, 16) / 8.0
                       for _ in range(checked_product(tensor["shape"]))])
    return values


def golden_outputs(fixture: str, plan: dict, inputs: list[list[float]],
                   bin_path: Path) -> list[list[float]]:
    resources = {entry["id"]: entry for entry in plan["resources"]}
    outputs = [resources[item] for item in plan["outputs"]]
    if fixture == "relu":
        return [[max(0.0, value) for value in inputs[0]]]
    if fixture == "add_relu":
        summed = [lhs + rhs for lhs, rhs in zip(inputs[0], inputs[1])]
        return [[max(0.0, value) for value in summed]]
    if fixture in {"conv_relu", "reshape"}:
        input_channels, height, width = resources[plan["inputs"][0]]["shape"]
        output_channels, output_height, output_width = outputs[0]["shape"]
        if (height != output_height or width != output_width or
                len(inputs[0]) != input_channels * height * width):
            raise RuntimeError("1x1 convolution golden shape is inconsistent")
        raw = bin_path.read_bytes()
        if len(raw) != 4 + (input_channels * output_channels + output_channels) * 4:
            raise RuntimeError("fixed-run Conv file has an unexpected size")
        values = struct.unpack(f"<{(len(raw) - 4) // 4}f", raw[4:])
        weights = values[:input_channels * output_channels]
        biases = values[input_channels * output_channels:]
        convolution = []
        for out_channel in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    accumulator = biases[out_channel]
                    for in_channel in range(input_channels):
                        input_index = in_channel * height * width + y * width + x
                        weight_index = out_channel * input_channels + in_channel
                        accumulator += inputs[0][input_index] * weights[weight_index]
                    convolution.append(accumulator)
        if fixture == "conv_relu":
            return [[max(0.0, value) for value in convolution]]
        return [convolution]
    if fixture == "add_split_relu":
        summed = [lhs + rhs for lhs, rhs in zip(inputs[0], inputs[1])]
        if len(outputs) != 2:
            raise RuntimeError("Split/ReLU graph must expose two outputs")
        # NCNN publishes the unmodified split branch before the ReLU branch.
        return [summed, [max(0.0, value) for value in summed]]
    if fixture == "split_concat":
        return [list(inputs[0]) * 3]
    if fixture == "reshape":
        return [list(inputs[0])]
    if fixture == "transpose":
        source_shape = resources[plan["inputs"][0]]["shape"]
        permutation = [1, 0, 2]
        output_shape = outputs[0]["shape"]
        if len(source_shape) != 3 or len(output_shape) != 3:
            raise RuntimeError("rank-3 Permute golden received an unexpected rank")
        result = []
        source_strides = [source_shape[1] * source_shape[2],
                          source_shape[2], 1]
        for first in range(output_shape[0]):
            for second in range(output_shape[1]):
                for third in range(output_shape[2]):
                    output_coordinates = [first, second, third]
                    source_coordinates = [0, 0, 0]
                    for output_axis, source_axis in enumerate(permutation):
                        source_coordinates[source_axis] = output_coordinates[output_axis]
                    index = sum(coordinate * stride for coordinate, stride in
                                zip(source_coordinates, source_strides))
                    result.append(inputs[0][index])
        return [result]
    raise RuntimeError(f"unknown fixed-run fixture {fixture}")


def compare(reference: list[float], actual: list[float], label: str) -> dict:
    if len(reference) != len(actual):
        raise RuntimeError(f"{label}: element count differs")
    if not all(math.isfinite(value) for value in reference + actual):
        raise RuntimeError(f"{label}: non-finite output")
    absolute = [abs(expected - observed)
                for expected, observed in zip(reference, actual)]
    relative = [error / max(abs(expected), 1.0e-30)
                for error, expected in zip(absolute, reference)]
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        if abs(expected - observed) > ATOL + RTOL * abs(expected):
            raise RuntimeError(
                f"{label}: value {index} differs: {observed} vs {expected}")
    return {"atol": ATOL, "rtol": RTOL, "all_finite": True,
            "max_abs_error": max(absolute, default=0.0),
            "max_rel_error": max(relative, default=0.0)}


def run_native_loader(loader: Path, library: Path, mode: str,
                      input_paths_a: list[Path], output_paths_a: list[Path],
                      stage: Path, log_path: Path, env: dict[str, str],
                      input_paths_b: list[Path] | None = None,
                      output_paths_b: list[Path] | None = None) -> tuple[list[dict], dict]:
    command = [str(loader), "--library", str(library), "--mode", mode]
    input_option, output_option = (("--input", "--output") if mode == "single"
                                   else ("--input-a", "--output-a"))
    for path in input_paths_a:
        command.extend((input_option, str(path)))
    for path in output_paths_a:
        command.extend((output_option, str(path)))
    if mode != "single":
        if input_paths_b is None or output_paths_b is None:
            raise RuntimeError(f"native loader mode {mode} requires B tensors")
        for path in input_paths_b:
            command.extend(("--input-b", str(path)))
        for path in output_paths_b:
            command.extend(("--output-b", str(path)))
    result = subprocess.run(command, cwd=Path.cwd(), env=env,
                            capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(result.stdout + b"\n--- stderr ---\n" + result.stderr)
    result_record = {"argv": command, "cwd": str(Path.cwd().resolve()),
                     "returncode": result.returncode,
                     "stdout_sha256": sha256_bytes(result.stdout),
                     "stderr_sha256": sha256_bytes(result.stderr),
                     "log": artifact(log_path, stage)}
    if result.returncode:
        raise RuntimeError(
            f"native C ABI loader failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace')}")
    try:
        events = [json.loads(line) for line in
                  result.stdout.decode().splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise RuntimeError("native loader emitted malformed measurement JSON") from error
    measurements = [event for event in events if event.get("event") == "run"]
    lifecycle_records = [event for event in events
                         if event.get("event") == "lifecycle"]
    expected_count = 4 if mode == "interleaved" else 2 if mode in ("repeat", "concurrent") else 1
    if len(measurements) != expected_count or len(lifecycle_records) != 1 or any(
        item.get("status") != 0 for item in measurements
    ):
        raise RuntimeError(f"native loader returned invalid results: {events}")
    lifecycle = lifecycle_records[0]
    cpu_loader = loader.name.endswith("-cpu-loader")
    expected_contexts = 0 if cpu_loader else 2 if mode == "interleaved" else 1
    if (lifecycle.get("overflow") or
            lifecycle.get("create_count") != expected_contexts or
            lifecycle.get("destroy_count") != expected_contexts or
            lifecycle.get("run_count") != expected_count or
            len(lifecycle.get("run_statuses", [])) != expected_count or
            any(status != 0 for status in lifecycle.get("create_statuses", []) +
                lifecycle.get("run_statuses", []) +
                lifecycle.get("destroy_statuses", []))):
        raise RuntimeError(f"native loader lifecycle did not complete cleanly: {lifecycle}")
    outputs = []
    for measurement in measurements:
        output_paths = [Path(path) for path in measurement.get("output_paths", [])]
        if len(output_paths) == 0:
            raise RuntimeError("native loader omitted output paths")
        output_refs = [artifact(path, stage) for path in output_paths]
        outputs.append({**measurement, "output_artifacts": output_refs})
    result_record["stdout_sha256"] = sha256_bytes(result.stdout)
    result_record["stderr_sha256"] = sha256_bytes(result.stderr)
    result_record["lifecycle"] = lifecycle
    return outputs, result_record


def write_inputs(stage: Path, output_root: Path, fixture: str, label: str,
                 values: list[list[float]]) -> tuple[list[Path], list[dict]]:
    paths = []
    records = []
    for index, tensor in enumerate(values):
        path = output_root / "native-inputs" / fixture / f"{label}-{index}.f32"
        records.append(write_floats(path, tensor, stage))
        paths.append(path)
    return paths, records


def read_output_set(paths: list[Path], shapes: list[dict]) -> list[list[float]]:
    values = []
    for path, shape in zip(paths, shapes):
        data = path.read_bytes()
        count = checked_product(shape["shape"])
        if len(data) != count * 4:
            raise RuntimeError(f"native output size differs from shape: {path}")
        values.append(list(struct.unpack(f"<{count}f", data)))
    return values


def compile_model(args, compiler: str, fixture: str, param: Path, bin_path: Path,
                  output_dir: Path, backend: str, stage: Path,
                  env: dict[str, str], tool_paths: dict[str, str] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tools = tool_paths or {}
    def tool(key: str, fallback: str) -> str:
        selected = tools.get(key, fallback)
        if isinstance(selected, dict):
            return selected["path"]
        return str(selected)
    command = [compiler, str(param), "--bin", str(bin_path),
               "--model-name", fixture, "--output-dir", str(output_dir),
               "--driver", tool("driver", args.driver),
               "--opt", tool("opt", args.opt),
               "--nm", tool("nm", args.nm),
               "--readelf", tool("readelf", args.readelf),
               "--verify-execution"]
    if backend == "cpu":
        command.extend(("--translate", tool("translate", args.translate),
                        "--clang", tool("clang", args.clang),
                        "--llvm-as", tool("llvm-as", args.llvm_as),
                        "--emit-manifest", "--precision=f32", "--threads=1",
                        "--vector-width=0"))
        for argument in args.cxx_arg:
            command.extend(("--clang-arg", argument,
                            "--linker-arg", argument))
    else:
        command.extend(("-v", "--backend=vulkan", "--precision=f32",
                        "--vulkan-mlir-opt", tool("mlir-opt", args.mlir_opt),
                        "--vulkan-spirv-val", tool("spirv-val", args.spirv_val),
                        "--vulkan-spirv-dis", tool("spirv-dis", args.spirv_dis),
                        "--vulkan-spirv-extract",
                        tool("spirv-extract", args.spirv_extract),
                        "--vulkan-host-cxx", tool("host-cxx", args.host_cxx)))
    record = run_command(command, Path.cwd(), env,
                         output_dir.parent / f"{fixture}-{backend}-compile.log",
                         stage)
    record["output_dir"] = artifact(output_dir / (f"{fixture}.h" if backend == "cpu"
                                                   else f"{fixture}_vulkan.h"), stage)
    return record


def compile_cpu_native_loader(args, fixture: str, cpu_dir: Path,
                              stage: Path, env: dict[str, str]) -> dict:
    source = args.source_root / "test/Vulkan/native/fixedrun_native_loader.cpp"
    loader_root = args.output_root / "native-loaders" / fixture / "cpu"
    source_copy = loader_root / "fixedrun_native_loader.cpp"
    source_artifact = snapshot_artifact(source, source_copy, stage)
    executable_target = args.output_root / "native-loaders" / f"{fixture}-cpu-loader-target"
    executable_target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.host_cxx), "-std=c++23", "-Wall", "-Wextra",
               "-Wpedantic", "-fno-omit-frame-pointer",
               "-I", str(args.source_root / "include"), "-I", str(cpu_dir),
               f"-DNCNN_FIXEDRUN_MODEL={fixture}",
               f"-DNCNN_FIXEDRUN_MODEL_UPPER={fixture.upper()}",
               f'-DNCNN_FIXEDRUN_MODEL_HEADER="{fixture}.h"',
               "-DNCNN_FIXEDRUN_BACKEND_VULKAN=0"]
    command.extend(args.cxx_arg)
    command.extend((str(source), "-o", str(executable_target), "-ldl", "-pthread",
                    "-Wl,-z,defs"))
    if args.sanitizer_runtime_dir is not None:
        command.append(f"-Wl,-rpath,{args.sanitizer_runtime_dir}")
    record = run_command(
        command, Path.cwd(), env,
        args.output_root / "native-loaders" / f"{fixture}-cpu-loader-build.log",
        stage,
        args.output_root / "native-loaders" / fixture / "cpu" / "compile.stdout",
        args.output_root / "native-loaders" / fixture / "cpu" / "compile.stderr")
    executable_hash = sha256_file(executable_target)
    executable_snapshot = snapshot_artifact(
        executable_target, loader_root / "fixedrun-cpu-loader", stage)
    record["source"] = source_artifact
    record["source_path"] = str(source.relative_to(args.source_root))
    record["source_target_path"] = str(source.resolve())
    record["source_target_sha256"] = sha256_file(source)
    record["executable_target_path"] = str(executable_target.resolve())
    record["executable_target_sha256"] = executable_hash
    record["executable"] = executable_snapshot
    return record


def capture_fixedrun(args, fixture: str, param: Path, bin_path: Path,
                     replay_dir: Path, stage: Path, env: dict[str, str],
                     reference_dir: Path | None = None) -> dict:
    replay_dir.mkdir(parents=True, exist_ok=True)
    argv = [str(args.capture_tool), str(param), "--bin", str(bin_path),
            "--model-name", fixture, "--output-dir", str(replay_dir),
            "--backend=vulkan", "--precision=f32", "--driver", str(args.driver),
            "--opt", str(args.opt), "--translate", str(args.translate),
            "--clang", str(args.clang), "--nm", str(args.nm),
            "--readelf", str(args.readelf), "--llvm-as", str(args.llvm_as),
            "--vulkan-mlir-opt", str(args.mlir_opt),
            "--vulkan-spirv-val", str(args.spirv_val),
            "--vulkan-spirv-dis", str(args.spirv_dis),
            "--vulkan-spirv-extract", str(args.spirv_extract),
            "--vulkan-host-cxx", str(args.host_cxx),
            "--capture-vulkan-host-source"]
    record = run_command(argv, Path.cwd(), env,
                         replay_dir.parent /
                         f"{replay_dir.name}-capture-command.log", stage)
    original = reference_dir or args.candidate_root / fixture
    pairs = ((original / "candidate_source.cpp", replay_dir / "candidate_source.cpp"),
             (original / f"{fixture}_vulkan.h", replay_dir / f"{fixture}_vulkan.h"),
             (original / "plan.json", replay_dir / "plan.json"),
             (original / "host_program.json", replay_dir / "host_program.json"),
             (original / "candidate_tosa.mlir", replay_dir / "candidate_tosa.mlir"),
             (original / "metadata.json", replay_dir / "metadata.json"))
    for expected, reproduced in pairs:
        if sha256_file(expected) != sha256_file(reproduced):
            raise RuntimeError(
                f"fixed-run capture is not reproducible for {fixture}: {expected.name}")
    record["outputs"] = [artifact(left, stage) for left, _ in pairs]
    record["reproduced_outputs"] = [artifact(right, stage) for _, right in pairs]
    return record


def reject_unowned_capture_file(args, fixture: str, param: Path,
                                bin_path: Path, source_capture: Path,
                                negative_dir: Path, stage: Path,
                                env: dict[str, str]) -> dict:
    owned_sources = [source_capture / "candidate_source.cpp",
                     source_capture / f"{fixture}_vulkan.h",
                     source_capture / "metadata.json",
                     source_capture / "plan.json",
                     source_capture / "host_program.json",
                     source_capture / "candidate_tosa.mlir"]
    owned_sources.extend(sorted(source_capture.glob("shader_*.spv")))
    negative_dir.mkdir(parents=True, exist_ok=True)
    sentinel = negative_dir / "unowned-sentinel.txt"
    sentinel_bytes = b"fixedrun capture directory ownership sentinel\\n"
    for source in owned_sources:
        target = negative_dir / source.name
        expected = source.read_bytes()
        if target.exists():
            if target.read_bytes() != expected:
                raise RuntimeError(f"capture negative fixture drift: {target}")
        else:
            target.write_bytes(expected)
    if sentinel.exists():
        if sentinel.read_bytes() != sentinel_bytes:
            raise RuntimeError("capture ownership sentinel changed between runs")
    else:
        sentinel.write_bytes(sentinel_bytes)
    before = {path.name: sha256_file(path)
              for path in sorted(negative_dir.iterdir()) if path.is_file()}
    command = [str(args.capture_tool), str(param), "--bin", str(bin_path),
               "--model-name", fixture, "--output-dir", str(negative_dir),
               "--backend=vulkan", "--precision=f32", "--driver", str(args.driver),
               "--opt", str(args.opt), "--translate", str(args.translate),
               "--clang", str(args.clang), "--nm", str(args.nm),
               "--readelf", str(args.readelf), "--llvm-as", str(args.llvm_as),
               "--vulkan-mlir-opt", str(args.mlir_opt),
               "--vulkan-spirv-val", str(args.spirv_val),
               "--vulkan-spirv-dis", str(args.spirv_dis),
               "--vulkan-spirv-extract", str(args.spirv_extract),
               "--vulkan-host-cxx", str(args.host_cxx),
               "--capture-vulkan-host-source"]
    result = subprocess.run(command, cwd=Path.cwd(), env=env,
                            capture_output=True, check=False)
    log_path = negative_dir.parent / f"{fixture}-unowned-capture.log"
    log_path.write_bytes(result.stdout + b"\\n--- stderr ---\\n" + result.stderr)
    after = {path.name: sha256_file(path)
             for path in sorted(negative_dir.iterdir()) if path.is_file()}
    if result.returncode == 0:
        raise RuntimeError("fixed-run capture accepted an unowned directory entry")
    stderr = result.stderr.decode(errors="replace")
    if "unowned file" not in stderr:
        raise RuntimeError(f"capture negative failed for an unexpected reason: {stderr}")
    if before != after:
        raise RuntimeError("failed capture modified previously owned output files")
    return {"fixture": fixture,
            "argv": command,
            "cwd": str(Path.cwd().resolve()),
            "returncode": result.returncode,
            "stdout_sha256": sha256_bytes(result.stdout),
            "stderr_sha256": sha256_bytes(result.stderr),
            "diagnostic": stderr.strip(),
            "sentinel": artifact(sentinel, stage),
            "before_hashes": before,
            "after_hashes": after,
            "owned_files_unchanged": True,
            "log": artifact(log_path, stage)}


def compile_commands(build: Path, source: Path,
                     model: str | None = None) -> dict:
    database = json.loads((build / "compile_commands.json").read_text())
    source = source.resolve()
    matches = [item for item in database
               if Path(item["file"]).resolve() == source]
    if model is not None:
        matches = [item for item in matches
                   if f"NCNN_FIXEDRUN_MODEL={model}" in
                   (item.get("command", "") + " ".join(item.get("arguments", [])))]
    if len(matches) != 1:
        raise RuntimeError(f"expected one compile command for {source}, got {len(matches)}")
    entry = matches[0]
    argv = entry.get("arguments") or shlex.split(entry["command"])
    output = None
    for index, item in enumerate(argv[:-1]):
        if item == "-o":
            output = argv[index + 1]
            break
    if not output:
        raise RuntimeError(f"compile command has no object output: {source}")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path(entry["directory"]) / output_path
    return {"argv": argv, "cwd": str(Path(entry["directory"]).resolve()),
            "source_path": str(source), "output": str(output_path.resolve())}


def link_command(build: Path, target: str) -> dict:
    path = build / "test/Vulkan/native/CMakeFiles" / f"{target}.dir/link.txt"
    if not path.is_file():
        # Some CMake generators keep per-directory metadata at this equivalent path.
        candidates = list(build.glob(f"**/CMakeFiles/{target}.dir/link.txt"))
        if len(candidates) != 1:
            raise RuntimeError(f"cannot locate unique link command for {target}")
        path = candidates[0]
    working_directory = path.parent.parent.parent
    if not working_directory.is_dir():
        raise RuntimeError(f"link command working directory is missing: {working_directory}")
    return {"argv": shlex.split(path.read_text().strip()),
            "cwd": str(working_directory.resolve())}


def write_run_outputs(stage: Path, output_root: Path, fixture: str, phase: str,
                      iteration: int, values: list[list[float]],
                      inputs: list[list[float]], dso_hash: str,
                      input_tensors: list[dict], output_tensors: list[dict],
                      elapsed_ns: int, diagnostic: str) -> dict:
    prefix = output_root / "raw" / fixture / f"{phase}-{iteration}"
    input_refs = [write_floats(prefix.parent / f"{prefix.name}-input-{index}.f32",
                               tensor, stage)
                  for index, tensor in enumerate(inputs)]
    output_refs = [write_floats(prefix.parent / f"{prefix.name}-output-{index}.f32",
                                tensor, stage)
                   for index, tensor in enumerate(values)]
    return {"fixture": fixture, "phase": phase, "iteration": iteration,
            "inputs": input_refs, "outputs": output_refs, "status": 0,
            "diagnostic": diagnostic, "validation_error_count": 0,
            "elapsed_ns": elapsed_ns, "dso_sha256": dso_hash}


def copy_legacy_reference(root: Path | None, stage: Path,
                          output_root: Path | None = None) -> dict:
    if root is None:
        return {"status": "not-supplied"}
    root = root.resolve()
    manifest_path = root / "legacy-reference.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"legacy reference manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "ncnn-vulkan-legacy-reference/v1":
        raise RuntimeError("legacy reference manifest schema is invalid")
    if manifest.get("role") != "legacy-product-reference":
        raise RuntimeError("legacy reference role is invalid")
    if manifest.get("git_revision") != "6094cb56abe24cdbcde84fbb72813da8aa24c147":
        raise RuntimeError("legacy reference is not pinned to the VA4 base revision")
    destination = ((output_root / "legacy-reference") if output_root is not None
                   else stage / "test/Vulkan/evidence/va4/legacy-reference")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    seen = set()

    def copy_file(source_value: str, expected_hash: str | None, role: str) -> None:
        source = Path(source_value)
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        if not source.is_file():
            raise RuntimeError(f"legacy reference file is missing: {source}")
        actual_hash = sha256_file(source)
        if expected_hash and expected_hash != actual_hash:
            raise RuntimeError(f"legacy reference hash mismatch: {source}")
        if source in seen:
            return
        seen.add(source)
        target = destination / f"{len(copied):04d}-{source.name}"
        target.write_bytes(source.read_bytes())
        copied.append({"role": role, "source_path": str(source),
                       "artifact": artifact(target, stage),
                       "sha256": actual_hash})

    copy_file(str(manifest_path), sha256_bytes(manifest_bytes), "manifest")

    def walk(value, role: str) -> None:
        if isinstance(value, dict):
            path_value = value.get("path")
            digest = value.get("sha256")
            if isinstance(path_value, str) and Path(path_value).is_absolute():
                candidate = Path(path_value)
                if candidate.is_file():
                    copy_file(path_value,
                              digest if isinstance(digest, str) else None, role)
            for key, nested in value.items():
                if key == "path":
                    continue
                walk(nested, f"{role}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{role}[{index}]")
        elif isinstance(value, str) and Path(value).is_absolute():
            candidate = Path(value)
            if candidate.is_file():
                copy_file(value, None, role)

    walk(manifest, "legacy")
    compiler = manifest.get("compiler")
    if not isinstance(compiler, dict) or not isinstance(compiler.get("path"), str):
        raise RuntimeError("legacy reference compiler identity is missing")
    compiler_path = Path(compiler["path"]).resolve()
    if not compiler_path.is_file() or sha256_file(compiler_path) != compiler.get("sha256"):
        raise RuntimeError("legacy reference compiler does not match its manifest")
    clean = manifest.get("clean_source_status")
    if not isinstance(clean, dict) or not isinstance(clean.get("path"), str):
        raise RuntimeError("legacy reference clean-source record is missing")
    clean_path = Path(clean["path"]).resolve()
    if (not clean_path.is_file() or clean_path.read_bytes().strip() or
            sha256_file(clean_path) != clean.get("sha256")):
        raise RuntimeError("legacy reference source is not recorded clean")
    tool_paths = {}
    tool_sources = {}
    tools = manifest.get("tools", [])
    if isinstance(tools, dict):
        tool_items = [{"role": role, **value} if isinstance(value, dict)
                      else {"role": role, "path": value}
                      for role, value in tools.items()]
    elif isinstance(tools, list):
        tool_items = tools
    else:
        tool_items = []
    project_tool_roles = {
        "ncnn-mlir-driver": "driver",
        "ncnn-mlir-opt": "opt",
        "ncnn-vulkan-spirv-extract": "spirv-extract",
    }
    for item in tool_items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        role = item.get("role")
        if role not in project_tool_roles:
            continue
        mapped_role = project_tool_roles[role]
        tool_paths[mapped_role] = str(Path(item["path"]).resolve())
        tool_sources[mapped_role] = f"legacy-manifest.tools[{role}]"

    cache_record = manifest.get("build", {}).get("cmake_cache", {})
    cache_path = Path(cache_record.get("path", "")).resolve()
    if not cache_path.is_file() or sha256_file(cache_path) != cache_record.get("sha256"):
        raise RuntimeError("legacy CMake cache does not match its manifest")
    cache = {}
    for line in cache_path.read_text(errors="replace").splitlines():
        if line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
    # NCNN_MLIR_OPT_EXECUTABLE is the upstream mlir-opt runner, not the
    # project's ncnn-mlir-opt executable recorded in the manifest tools list.
    cache_tools = {"host-cxx": ("CMAKE_CXX_COMPILER",),
                   "readelf": ("NCNN_VULKAN_READELF", "LLVM_READELF"),
                   "mlir-opt": ("NCNN_MLIR_OPT_EXECUTABLE",),
                   "spirv-val": ("NCNN_SPIRV_VAL_EXECUTABLE",),
                   "spirv-dis": ("NCNN_SPIRV_DIS_EXECUTABLE",)}
    for role, keys in cache_tools.items():
        if role not in tool_paths:
            selected_key = next((key for key in keys if cache.get(key)), None)
            if selected_key is not None:
                tool_paths[role] = str(Path(cache[selected_key]).resolve())
                tool_sources[role] = f"legacy-cmake-cache:{selected_key}"
    if "nm" not in tool_paths:
        llvm_dir = cache.get("LLVM_DIR")
        if llvm_dir:
            llvm_cmake_dir = Path(llvm_dir).resolve()
            nm_path = llvm_cmake_dir.parents[2] / "bin" / "llvm-nm"
            if nm_path.is_file():
                tool_paths["nm"] = str(nm_path.resolve())
                tool_sources["nm"] = (
                    f"legacy-cmake-cache:LLVM_DIR={llvm_dir};"
                    "derived-tools-bin/llvm-nm")
    tool_records = {}
    for role, value in tool_paths.items():
        path = Path(value).resolve()
        if path.is_file():
            copy_file(str(path), None, f"tool.{role}")
            tool_records[role] = {
                "path": str(path), "sha256": sha256_file(path),
                "selected_from": tool_sources.get(role, "legacy-manifest"),
                "staged": next(item["artifact"] for item in copied
                               if Path(item["source_path"]) == path)}
    cache_snapshot = next((item["artifact"] for item in copied
                           if Path(item["source_path"]) == cache_path), None)
    if cache_snapshot is None:
        raise RuntimeError("legacy CMake cache snapshot is missing from copied files")
    return {"status": "verified-independent-source",
            "manifest": artifact(destination / f"0000-{manifest_path.name}", stage),
            "git_revision": manifest.get("git_revision"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "source_root": manifest.get("source_root"),
            "clean_source_status": {"record": clean,
                                    "artifact": next(item["artifact"] for item in copied
                                                     if Path(item["source_path"]) == clean_path)},
            "compiler": {"path": str(compiler_path),
                         "sha256": sha256_file(compiler_path),
                         "staged": next(item["artifact"] for item in copied
                                         if Path(item["source_path"]) == compiler_path)},
            "tool_selection": {
                "cmake_cache": cache_snapshot,
                "cmake_cache_source_path": str(cache_path),
                "cmake_cache_sha256": sha256_file(cache_path),
                "selected_from": tool_sources},
            "tools": tool_records,
            "files": copied}


def selected_device(args, stage: Path, env: dict[str, str]) -> dict:
    result = subprocess.run([args.vulkaninfo, "--summary"], env=env,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("vulkaninfo --summary failed: " +
                           result.stderr.decode(errors="replace"))
    output = result.stdout + b"\n--- stderr ---\n" + result.stderr
    raw = write_bytes(args.output.parent / "vulkaninfo-summary.txt", output, stage)
    text = result.stdout.decode(errors="replace")
    lines = text.splitlines()
    device_match = next((line.split("=", 1)[-1].strip()
                         for line in lines if "deviceName" in line), "unknown")
    driver_match = next((line.split("=", 1)[-1].strip()
                         for line in lines if "driverName" in line), "")
    driver_info = next((line.split("=", 1)[-1].strip()
                        for line in lines if "driverInfo" in line), "")
    driver_id = next((line.split("=", 1)[-1].strip()
                      for line in lines if "driverID" in line), "")
    if not driver_match:
        raise RuntimeError("vulkaninfo --summary omitted the selected driver name")
    icd_copy = write_bytes(args.output.parent / "icd-manifest.json",
                           args.icd_manifest.read_bytes(), stage)
    return {"name": device_match, "driver": driver_match,
            "driver_info": driver_info, "driver_id": driver_id,
            "ordinal": 0, "vulkaninfo": raw,
            "icd_manifest": icd_copy,
            "icd_manifest_source": str(args.icd_manifest.resolve()),
            "icd_manifest_sha256": sha256_file(args.icd_manifest)}


def verify_tosa_reshape_views(path: Path) -> list[dict]:
    views = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1):
        if "tosa.reshape" not in line:
            continue
        tensor_types = [value for value in re.findall(r"tensor<([^>]+)>", line)
                        if value.endswith("f32")]
        if len(tensor_types) < 2:
            continue
        ranks = [len(value.removesuffix("xf32").split("x"))
                 for value in (tensor_types[0], tensor_types[-1])]
        views.append({"line": line_number, "input_type": tensor_types[0],
                      "output_type": tensor_types[-1],
                      "input_rank": ranks[0], "output_rank": ranks[1],
                      "text": line.strip()})
    if not any(view["input_rank"] == 3 and view["output_rank"] == 4
               for view in views) or not any(
        view["input_rank"] == 4 and view["output_rank"] == 3 for view in views
    ):
        raise RuntimeError(
            "reshape candidate TOSA IR must prove rank-3 to rank-4 and "
            "rank-4 to rank-3 views")
    return views


def command_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def validate_helper_clean_script(clean_script: Path, helper: Path,
                                 cwd: Path) -> str:
    text = clean_script.read_text(encoding="utf-8")
    uncommented = re.sub(r"(?m)^\s*#.*$", "", text)
    match = re.fullmatch(
        r'\s*file\(\s*REMOVE_RECURSE\s+"([^"]+)"\s*\)\s*',
        uncommented, flags=re.DOTALL)
    if match is None or command_path(match.group(1), cwd) != helper.resolve():
        raise RuntimeError(
            "Vulkan helper clean script is not limited to its target archive")
    return text


def helper_archive_commands(link_path: Path, cwd: Path, helper: Path,
                            helper_objects: set[Path]) -> tuple[list[dict], Path,
                                                                dict[str, Path]]:
    commands = []
    archive_objects: set[Path] = set()
    members: dict[str, Path] = {}
    archive_tool = None
    saw_archive = False
    saw_ranlib = False
    for line in link_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        argv = shlex.split(line)
        if not argv or any(token.startswith("@") for token in argv):
            raise RuntimeError("Vulkan helper archive command is empty or indirect")
        archive_tokens = [token for token in argv if Path(token).suffix == ".a"]
        if len(archive_tokens) != 1 or command_path(archive_tokens[0], cwd) != helper.resolve():
            raise RuntimeError("Vulkan helper link script names an unexpected archive")
        basename = Path(argv[0]).name
        if any(token in {"q", "qc", "rc", "rcs", "cr", "crs"} for token in argv[1:]):
            if saw_archive or saw_ranlib:
                raise RuntimeError("Vulkan helper archive script has unexpected ar order")
            saw_archive = True
            archive_tool = command_path(argv[0], cwd)
            object_tokens = [token for token in argv if Path(token).suffix == ".o"]
            if not object_tokens:
                raise RuntimeError("Vulkan helper ar command has no object members")
            for token in object_tokens:
                object_path = command_path(token, cwd)
                archive_objects.add(object_path)
                member_name = Path(token).name
                if member_name in members and members[member_name] != object_path:
                    raise RuntimeError("Vulkan helper objects have duplicate member names")
                members[member_name] = object_path
            kind = "archive"
        elif "ranlib" in basename:
            if not saw_archive or saw_ranlib:
                raise RuntimeError("Vulkan helper ranlib command is out of order")
            saw_ranlib = True
            kind = "ranlib"
        else:
            raise RuntimeError(f"unexpected Vulkan helper archive command: {basename}")
        commands.append({"kind": kind, "argv": argv, "cwd": str(cwd.resolve())})
    if not saw_archive or not saw_ranlib or archive_tool is None:
        raise RuntimeError("Vulkan helper link script lacks ar/ranlib commands")
    if archive_objects != {path.resolve() for path in helper_objects}:
        raise RuntimeError("Vulkan helper archive object inputs differ from CMake objects")
    return commands, archive_tool, members


def archive_member_list(args: list[str], cwd: Path, env: dict[str, str],
                        log_path: Path, stdout_path: Path,
                        stage: Path) -> tuple[dict, list[str]]:
    record = run_command(args, cwd, env, log_path, stage, stdout_path,
                         log_path.with_suffix(".stderr"))
    stdout = (stage / record["stdout"]["path"]).read_bytes()
    try:
        names = [name for name in stdout.decode("utf-8", errors="strict").splitlines()
                 if name]
    except UnicodeDecodeError as error:
        raise RuntimeError("Vulkan helper ar t output is not UTF-8") from error
    return record, names


def extract_archive_member(archive_tool: Path, helper: Path, member: str,
                           cwd: Path, stage: Path, output_root: Path,
                           label: str) -> dict:
    argv = [str(archive_tool), "p", str(helper), member]
    result = subprocess.run(argv, cwd=cwd, capture_output=True, check=False)
    stdout_path = output_root / "members" / f"{label}-{Path(member).name}.stdout"
    stderr_path = output_root / "members" / f"{label}-{Path(member).name}.stderr"
    stdout_artifact = write_bytes(stdout_path, result.stdout, stage)
    stderr_artifact = write_bytes(stderr_path, result.stderr, stage)
    command = {"argv": argv, "cwd": str(cwd.resolve()),
               "returncode": result.returncode,
               "stdout_sha256": sha256_bytes(result.stdout),
               "stderr_sha256": sha256_bytes(result.stderr),
               "stdout": stdout_artifact, "stderr": stderr_artifact}
    if result.returncode:
        raise RuntimeError(
            f"ar p failed for helper member {member}: "
            f"{result.stderr.decode(errors='replace')}")
    return {"name": member, "sha256": sha256_bytes(result.stdout),
            "extracted_artifact": stdout_artifact,
            "equals_compile_object": False, "extract_command": command}


def configured_cmake_command(build_root: Path) -> Path:
    makefile = build_root / "Makefile"
    text = makefile.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^CMAKE_COMMAND\s*=\s*(\S+)\s*$", text)
    if match is None:
        raise RuntimeError("CMake Makefile does not identify its configured command")
    command = Path(match.group(1)).resolve()
    if not command.is_file():
        raise RuntimeError("configured CMake command is missing")
    return command


def cmake_cache_value(build_root: Path, key: str) -> str:
    cache = build_root / "CMakeCache.txt"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        current_key, value = line.split("=", 1)
        if current_key.split(":", 1)[0] == key:
            return value
    raise RuntimeError(f"CMakeCache.txt does not define {key}")


def rebuild_helper_archive(helper: Path, helper_object: Path,
                           link_path: Path, clean_script: Path,
                           cmake: Path, stage: Path, output_root: Path,
                           env: dict[str, str], fixture: str) -> dict:
    cwd = link_path.parent.parent.parent
    helper = helper.resolve()
    helper_object = helper_object.resolve()
    clean_script = clean_script.resolve()
    clean_text = validate_helper_clean_script(clean_script, helper, cwd)
    clean_script_hash = sha256_file(clean_script)
    clean_script_artifact = snapshot_artifact(
        clean_script, output_root / "cmake_clean_target.cmake", stage)
    commands, archive_tool, expected_members = helper_archive_commands(
        link_path, cwd, helper, {helper_object})
    link_script_hash = sha256_file(link_path)
    link_script_artifact = snapshot_artifact(
        link_path, output_root / "link.txt", stage)
    cache_text = (cwd.parents[1] / "CMakeCache.txt").read_text(
        encoding="utf-8", errors="replace")
    archive_tool_match = re.search(r"(?m)^CMAKE_AR:[^=]*=(.+)$", cache_text)
    if (archive_tool_match is None or
            Path(archive_tool_match.group(1)).resolve() != archive_tool.resolve()):
        raise RuntimeError("CMake helper ar command differs from configured CMAKE_AR")
    archive_tool_copy = write_bytes(
        output_root / "tools" / f"{fixture}-{archive_tool.name}",
        archive_tool.read_bytes(), stage)
    archive_tool_identity = {"path": str(archive_tool.resolve()),
                             "sha256": sha256_file(archive_tool),
                             "artifact": archive_tool_copy}
    object_hash = sha256_file(helper_object)
    build_root = cwd.parents[1]
    makefile = build_root / "Makefile"
    makefile_text = makefile.read_text(encoding="utf-8", errors="replace")
    cmake_match = re.search(r"(?m)^CMAKE_COMMAND\s*=\s*(\S+)\s*$", makefile_text)
    if cmake_match is None or Path(cmake_match.group(1)).resolve() != cmake.resolve():
        raise RuntimeError("helper clean command differs from the configured CMake tool")

    prior_archive = None
    if helper.is_file():
        prior_path = output_root / "prior" / helper.name
        prior_snapshot = write_bytes(prior_path, helper.read_bytes(), stage)
        prior_listing, prior_members = archive_member_list(
            [str(archive_tool), "t", str(helper)], cwd, env,
            output_root / "archive-prior-list.log",
            output_root / "archive-prior-list.stdout", stage)
        prior_archive = {"snapshot": prior_snapshot,
                         "sha256": sha256_file(prior_path),
                         "member_list": prior_members,
                         "duplicate_member_count": len(prior_members) - len(set(prior_members)),
                         "ar_list_command": prior_listing}
    elif helper.exists():
        raise RuntimeError("pre-existing helper archive is not a regular file")

    rebuilds = []
    for iteration in range(2):
        iteration_root = output_root / f"archive-cycle-{iteration}"
        before_snapshot = None
        before_sha256 = None
        before_members: list[str] = []
        before_list = None
        if helper.is_file():
            before_path = iteration_root / "before" / helper.name
            before_snapshot = write_bytes(before_path, helper.read_bytes(), stage)
            before_sha256 = sha256_file(before_path)
            before_list, before_members = archive_member_list(
                [str(archive_tool), "t", str(helper)], cwd, env,
                iteration_root / "before-list.log",
                iteration_root / "before-list.stdout", stage)
        elif helper.exists():
            raise RuntimeError("helper archive path is not a regular file before clean")

        clean_argv = [str(cmake.resolve()), "-P",
                      str(clean_script.relative_to(cwd))]
        clean_command = {"argv": clean_argv, "cwd": str(cwd.resolve()),
                         "script": clean_script_artifact,
                         "script_target_path": str(clean_script),
                         "script_sha256": clean_script_hash}
        clean_command["execution"] = execute_cmake_command(
            clean_command, stage, env, iteration_root / "clean.log")
        if helper.exists():
            raise RuntimeError("CMake helper clean target left its archive behind")
        if not helper_object.is_file() or sha256_file(helper_object) != object_hash:
            raise RuntimeError("CMake helper clean target changed its object input")

        executed_archive_commands = []
        for command_index, command in enumerate(commands):
            executed = dict(command)
            executed["execution"] = execute_cmake_command(
                command, stage, env,
                iteration_root / f"archive-command-{command_index}.log")
            executed_archive_commands.append(executed)
        if not helper.is_file() or helper.is_symlink():
            raise RuntimeError("CMake helper archive commands did not create a regular archive")

        ar_list, member_names = archive_member_list(
            [str(archive_tool), "t", str(helper)], cwd, env,
            iteration_root / "ar-t.log", iteration_root / "ar-t.stdout", stage)
        if len(member_names) != len(set(member_names)):
            raise RuntimeError("CMake helper archive contains duplicate member names")
        if set(member_names) != set(expected_members):
            raise RuntimeError("CMake helper archive members differ from CMake object inputs")
        member_records = []
        for member in member_names:
            record = extract_archive_member(
                archive_tool, helper, member, cwd, stage, iteration_root,
                f"{fixture}-{iteration}-{len(member_records)}")
            expected_object = expected_members[member]
            extracted_path = stage / record["extract_command"]["stdout"]["path"]
            equal = extracted_path.read_bytes() == expected_object.read_bytes()
            if not equal:
                raise RuntimeError(
                    f"helper archive member {member} differs from CMake object bytes")
            record["equals_compile_object"] = True
            record["object_target_path"] = str(expected_object)
            record["object_target_sha256"] = sha256_file(expected_object)
            record["object"] = snapshot_artifact(
                expected_object, iteration_root / "objects" / expected_object.name,
                stage)
            member_records.append(record)

        archive_target_hash = sha256_file(helper)
        archive_snapshot = write_bytes(
            iteration_root / "archive-snapshot" / helper.name,
            helper.read_bytes(), stage)
        if archive_target_hash != archive_snapshot["sha256"]:
            raise RuntimeError("helper archive snapshot differs from target bytes")
        rebuilds.append({
            "iteration": iteration,
            "before_archive": before_snapshot,
            "before_sha256": before_sha256,
            "before_member_list": before_members,
            "before_duplicate_member_count": len(before_members) - len(set(before_members)),
            "before_ar_list_command": before_list,
            "clean_script": clean_script_artifact,
            "clean_script_sha256": clean_script_hash,
            "clean_command": clean_command,
            "archive_commands": executed_archive_commands,
            "archive_target_path": str(helper),
            "archive_target_sha256": archive_target_hash,
            "archive": archive_snapshot,
            "archive_sha256": archive_snapshot["sha256"],
            "member_list": member_names,
            "member_records": member_records,
            "ar_list_command": ar_list,
        })

    if (rebuilds[0]["archive_sha256"] != rebuilds[1]["archive_sha256"] or
            rebuilds[0]["member_list"] != rebuilds[1]["member_list"]):
        raise RuntimeError("CMake helper clean/archive sequence is not stable")
    return {
        "archive_target_path": str(helper),
        "target_archive_path": str(helper),
        "archive_target_sha256": rebuilds[-1]["archive_target_sha256"],
        "archive_link_script": link_script_artifact,
        "archive_link_script_target_path": str(link_path.resolve()),
        "archive_link_script_sha256": link_script_hash,
        "archive_tool": str(archive_tool),
        "archive_tool_identity": archive_tool_identity,
        "clean_script": clean_script_artifact,
        "clean_script_sha256": clean_script_hash,
        "prior_archive": prior_archive,
        "archive_command_templates": commands,
        "archive_commands": rebuilds[-1]["archive_commands"],
        "archive_rebuilds": rebuilds,
        "archive": rebuilds[-1]["archive"],
        "member_list": rebuilds[-1]["member_list"],
        "member_records": rebuilds[-1]["member_records"],
        "ar_list_command": rebuilds[-1]["ar_list_command"],
    }


def build_artifact_record(args, fixture: str, stage: Path,
                          source_compile: dict, replay: dict,
                          env: dict[str, str]) -> dict:
    candidate_dir = args.candidate_root / fixture
    dso = args.candidate_root / "libraries" / fixture / \
        f"lib{fixture}_fixedrun_validation.so"
    target = f"vulkan_fixedrun_{fixture}_validation"
    source = candidate_dir / "candidate_source.cpp"
    header = candidate_dir / f"{fixture}_vulkan.h"
    metadata = candidate_dir / "metadata.json"
    plan_path = candidate_dir / "plan.json"
    program_path = candidate_dir / "host_program.json"
    tosa_path = candidate_dir / "candidate_tosa.mlir"
    map_path = args.candidate_root / "audit" / fixture / f"{target}.map"
    candidate_snapshot_root = args.output_root / "candidate-build" / fixture
    source_snapshot = snapshot_artifact(
        source, candidate_snapshot_root / source.name, stage)
    header_snapshot = snapshot_artifact(
        header, candidate_snapshot_root / header.name, stage)
    metadata_snapshot = snapshot_artifact(
        metadata, candidate_snapshot_root / metadata.name, stage)
    plan_snapshot = snapshot_artifact(
        plan_path, candidate_snapshot_root / plan_path.name, stage)
    program_snapshot = snapshot_artifact(
        program_path, candidate_snapshot_root / program_path.name, stage)
    tosa_snapshot = snapshot_artifact(
        tosa_path, candidate_snapshot_root / tosa_path.name, stage)
    shader_paths = sorted(candidate_dir.glob("shader_*.spv"))
    shader_snapshots = [snapshot_artifact(
        path, candidate_snapshot_root / "shaders" / path.name, stage)
        for path in shader_paths]
    helper = (args.build_root / "lib/VulkanDeviceHelper" /
              "libNCNNVulkanDeviceHelper.a")
    if not helper.is_file() or helper.is_symlink():
        raise RuntimeError("Vulkan device helper archive is missing or not regular")
    helper_source = args.source_root / "lib/VulkanDeviceHelper/VulkanDeviceHelper.cpp"
    helper_output_root = args.output_root / "helper-build" / fixture
    helper_source_copy = snapshot_artifact(
        helper_source, helper_output_root / "VulkanDeviceHelper.cpp", stage)
    helper_compile = compile_commands(args.build_root, helper_source)
    helper_compile["source"] = helper_source_copy
    helper_object = Path(helper_compile["output"])
    helper_compile["execution"] = execute_cmake_command(
        helper_compile, stage, env,
        helper_output_root / "compile.log")
    helper_compile["object_target_path"] = str(helper_object.resolve())
    helper_compile["object_target_sha256"] = sha256_file(helper_object)
    helper_compile["object"] = snapshot_artifact(
        helper_object, helper_output_root / "objects" / helper_object.name, stage)

    archive_link_path = (args.build_root / "lib/VulkanDeviceHelper/CMakeFiles" /
                         "NCNNVulkanDeviceHelper.dir/link.txt")
    clean_target_path = (archive_link_path.parent /
                         "cmake_clean_target.cmake")
    cmake = configured_cmake_command(args.build_root)
    helper_build = rebuild_helper_archive(
        helper, helper_object, archive_link_path, clean_target_path, cmake,
        stage, helper_output_root, env, fixture)
    helper_archive = helper_build["archive"]
    link_record = link_command(args.build_root, target)
    compile_record = compile_commands(args.build_root, source)
    source_hash_before = sha256_file(source)
    if source_hash_before != source_snapshot["sha256"]:
        raise RuntimeError("candidate source changed after its immutable snapshot")
    compile_record["execution"] = execute_cmake_command(
        compile_record, stage, env,
        args.output_root / "build-commands" / f"{fixture}-candidate-compile.log")
    if sha256_file(source) != source_hash_before:
        raise RuntimeError("candidate compile changed its generated source bytes")
    compile_record["source"] = source_snapshot
    compile_record["source_target_path"] = str(source.resolve())
    compile_record["source_target_sha256"] = source_hash_before
    candidate_object_target = Path(compile_record["output"]).resolve()
    candidate_object_hash = sha256_file(candidate_object_target)
    candidate_object_snapshot = snapshot_artifact(
        candidate_object_target,
        candidate_snapshot_root / "objects" / candidate_object_target.name, stage)
    compile_record["object_target_path"] = str(candidate_object_target)
    compile_record["object_target_sha256"] = candidate_object_hash
    compile_record["object"] = candidate_object_snapshot

    helper_target_path = Path(helper_build["archive_target_path"])
    helper_target_sha256 = sha256_file(helper_target_path)
    if helper_target_sha256 != helper_archive["sha256"]:
        raise RuntimeError("helper archive target differs from its immutable snapshot before link")
    link_inputs = []
    object_target_hashes = {}
    for index, token in enumerate(link_record["argv"]):
        if token.startswith("-") or Path(token).suffix != ".o":
            continue
        object_path = command_path(token, Path(link_record["cwd"]))
        target_hash = sha256_file(object_path)
        if object_path == candidate_object_target:
            object_snapshot = candidate_object_snapshot
        else:
            object_snapshot = snapshot_artifact(
                object_path,
                candidate_snapshot_root / "link-objects" /
                f"{index}-{object_path.name}", stage)
        if object_snapshot["sha256"] != target_hash:
            raise RuntimeError("candidate link object snapshot differs from target")
        object_target_hashes[str(object_path)] = target_hash
        link_inputs.append({**object_snapshot,
                            "target_path": str(object_path),
                            "target_sha256": target_hash})

    vulkan_library_spelling = cmake_cache_value(args.build_root, "Vulkan_LIBRARY")
    vulkan_library_target = Path(vulkan_library_spelling)
    if not vulkan_library_target.is_absolute():
        vulkan_library_target = args.build_root / vulkan_library_target
    vulkan_library_resolved = vulkan_library_target.resolve(strict=True)
    if not vulkan_library_resolved.is_file():
        raise RuntimeError("CMakeCache Vulkan_LIBRARY is not a regular file")
    output_index = (link_record["argv"].index("-o") + 1
                    if "-o" in link_record["argv"] else len(link_record["argv"]))
    if output_index >= len(link_record["argv"]):
        raise RuntimeError("candidate link command has no output path")
    output_token = link_record["argv"][output_index]
    shared_input_rows = []
    for index, token in enumerate(link_record["argv"]):
        if (index == output_index or token.startswith("-") or
                re.search(r"\.so(?:\.[0-9]+)*$", Path(token).name) is None):
            continue
        resolved_token = command_path(token, Path(link_record["cwd"]))
        shared_input_rows.append({"index": index, "spelling": token,
                                  "resolved_path": str(resolved_token)})
    if (not shared_input_rows or any(
        Path(row["resolved_path"]) != vulkan_library_resolved
        for row in shared_input_rows
    )):
        raise RuntimeError(
            "candidate DSO shared inputs differ from CMakeCache Vulkan_LIBRARY")
    vulkan_library_target_sha256 = sha256_file(vulkan_library_resolved)
    vulkan_library_snapshot = snapshot_artifact(
        vulkan_library_resolved,
        candidate_snapshot_root / "system-libraries" /
        vulkan_library_resolved.name, stage)
    if vulkan_library_snapshot["sha256"] != vulkan_library_target_sha256:
        raise RuntimeError("Vulkan_LIBRARY snapshot differs from CMake target bytes")

    link_record["execution"] = execute_cmake_command(
        link_record, stage, env,
        args.output_root / "build-commands" / f"{fixture}-candidate-link.log")
    if sha256_file(vulkan_library_resolved) != vulkan_library_target_sha256:
        raise RuntimeError("Vulkan_LIBRARY target changed during candidate DSO link")
    link_record["vulkan_library"] = vulkan_library_snapshot
    link_record["vulkan_library_cache_key"] = "Vulkan_LIBRARY"
    link_record["vulkan_library_target_path"] = str(vulkan_library_target)
    link_record["vulkan_library_resolved_path"] = str(vulkan_library_resolved)
    link_record["vulkan_library_target_sha256"] = vulkan_library_target_sha256
    link_record["vulkan_library_link_arguments"] = shared_input_rows
    if sha256_file(helper_target_path) != helper_target_sha256:
        raise RuntimeError("candidate DSO link modified the helper archive target")
    for object_path, target_hash in object_target_hashes.items():
        if sha256_file(Path(object_path)) != target_hash:
            raise RuntimeError("candidate DSO link modified an object input")
    link_record["helper_archive"] = helper_archive
    link_record["helper_archive_target_path"] = str(helper_target_path)
    link_record["helper_archive_target_sha256"] = helper_target_sha256
    dso_target_hash = sha256_file(dso)
    map_target_hash = sha256_file(map_path)
    link_record["dso_target_path"] = str(dso.resolve())
    link_record["dso_target_sha256"] = dso_target_hash
    link_record["map_target_path"] = str(map_path.resolve())
    link_record["map_target_sha256"] = map_target_hash
    link_record["dso"] = snapshot_artifact(
        dso, candidate_snapshot_root / "dso" / dso.name, stage)
    link_record["map"] = snapshot_artifact(
        map_path, candidate_snapshot_root / "audit" / map_path.name, stage)
    if (link_record["dso"]["sha256"] != dso_target_hash or
            link_record["map"]["sha256"] != map_target_hash):
        raise RuntimeError("candidate DSO/map snapshot differs from target outputs")
    link_record["objects"] = link_inputs
    helper_build["source"] = helper_source_copy
    helper_build["compile"] = helper_compile
    link_record["helper_build"] = helper_build
    elf_dir = args.output_root / "elf"
    exports_log = elf_dir / f"{fixture}-exports.log"
    needed_log = elf_dir / f"{fixture}-needed.log"
    undefined_log = elf_dir / f"{fixture}-undefined.log"
    exports_command = run_command(
        [str(args.nm), "-D", "--defined-only", str(dso)], Path.cwd(), env,
        exports_log, stage, elf_dir / f"{fixture}-exports.raw",
        elf_dir / f"{fixture}-exports.stderr")
    needed_command = run_command(
        [str(args.readelf), "-d", str(dso)], Path.cwd(), env,
        needed_log, stage, elf_dir / f"{fixture}-needed.raw",
        elf_dir / f"{fixture}-needed.stderr")
    undefined_command = run_command(
        [str(args.nm), "-D", "--undefined-only", str(dso)], Path.cwd(), env,
        undefined_log, stage, elf_dir / f"{fixture}-undefined.raw",
        elf_dir / f"{fixture}-undefined.stderr")
    export_text = (args.output_root / "elf" / f"{fixture}-exports.raw").read_text(
        errors="replace")
    expected_symbols = {f"{fixture}_vk_create", f"{fixture}_vk_run",
                        f"{fixture}_vk_destroy"}
    observed_symbols = {line.split()[-1].split("@")[0]
                        for line in export_text.splitlines() if line.split()}
    if observed_symbols != expected_symbols:
        raise RuntimeError(
            f"candidate {fixture} export allowlist differs: {observed_symbols}")
    link_record["elf"] = {
        "exports": artifact(elf_dir / f"{fixture}-exports.raw", stage),
        "needed": artifact(elf_dir / f"{fixture}-needed.raw", stage),
        "undefined": artifact(elf_dir / f"{fixture}-undefined.raw", stage),
        "commands": {"exports": exports_command, "needed": needed_command,
                     "undefined": undefined_command},
    }
    loader_source = args.source_root / "test/Vulkan/native/fixedrun_native_loader.cpp"
    loader_snapshot_root = args.output_root / "native-loaders" / fixture
    loader_source_copy = snapshot_artifact(
        loader_source, loader_snapshot_root / loader_source.name, stage)
    loader_source_hash = sha256_file(loader_source)
    loader_target = f"vulkan_fixedrun_loader_{fixture}"
    loader_compile = compile_commands(args.build_root, loader_source, fixture)
    if loader_source_hash != loader_source_copy["sha256"]:
        raise RuntimeError("native loader source changed after snapshot")
    loader_compile["execution"] = execute_cmake_command(
        loader_compile, stage, env,
        loader_snapshot_root / f"{fixture}-compile.log")
    if sha256_file(loader_source) != loader_source_hash:
        raise RuntimeError("native loader compilation changed its source")
    loader_compile["source"] = loader_source_copy
    loader_compile["source_target_path"] = str(loader_source.resolve())
    loader_compile["source_target_sha256"] = loader_source_hash
    loader_object_target = Path(loader_compile["output"]).resolve()
    loader_object_hash = sha256_file(loader_object_target)
    loader_object = snapshot_artifact(
        loader_object_target,
        loader_snapshot_root / "objects" / loader_object_target.name, stage)
    loader_compile["object_target_path"] = str(loader_object_target)
    loader_compile["object_target_sha256"] = loader_object_hash
    loader_compile["object"] = loader_object

    loader_link = link_command(args.build_root, loader_target)
    loader_link_inputs = []
    for index, token in enumerate(loader_link["argv"]):
        if token.startswith("-") or Path(token).suffix != ".o":
            continue
        object_path = command_path(token, Path(loader_link["cwd"]))
        object_hash = sha256_file(object_path)
        if object_path != loader_object_target or object_hash != loader_object_hash:
            raise RuntimeError("native loader link object differs from its compile output")
        loader_link_inputs.append({**loader_object,
                                   "target_path": str(object_path),
                                   "target_sha256": object_hash})
    if len(loader_link_inputs) != 1:
        raise RuntimeError("native loader link does not contain exactly its compiled object")
    loader_link["execution"] = execute_cmake_command(
        loader_link, stage, env,
        loader_snapshot_root / f"{fixture}-link.log")
    if sha256_file(loader_object_target) != loader_object_hash:
        raise RuntimeError("native loader link modified its object input")
    loader_path = getattr(args, f"loader_{fixture}").resolve()
    loader_hash = sha256_file(loader_path)
    loader_binary = snapshot_artifact(
        loader_path, loader_snapshot_root / "binary" / loader_path.name, stage)
    if loader_binary["sha256"] != loader_hash:
        raise RuntimeError("native loader snapshot differs from CMake target output")
    loader_link["objects"] = loader_link_inputs
    loader_link["executable_target_path"] = str(loader_path)
    loader_link["executable_target_sha256"] = loader_hash
    loader_link["executable"] = loader_binary

    metadata_snapshot_path = candidate_snapshot_root / metadata.name
    tosa_snapshot_path = candidate_snapshot_root / tosa_path.name
    metadata_value = json.loads(metadata_snapshot_path.read_text(encoding="utf-8"))
    if sha256_file(tosa_snapshot_path) != metadata_value.get("tosa_ir_sha256"):
        raise RuntimeError(f"candidate TOSA IR hash differs for {fixture}")
    reshape_views = (verify_tosa_reshape_views(tosa_snapshot_path)
                     if fixture == "reshape" else [])
    package_targets = {
        "source": {"path": str(source.resolve()), "sha256": source_snapshot["sha256"]},
        "header": {"path": str(header.resolve()), "sha256": header_snapshot["sha256"]},
        "metadata": {"path": str(metadata.resolve()), "sha256": metadata_snapshot["sha256"]},
        "plan": {"path": str(plan_path.resolve()), "sha256": plan_snapshot["sha256"]},
        "host_program": {"path": str(program_path.resolve()), "sha256": program_snapshot["sha256"]},
        "tosa_ir": {"path": str(tosa_path.resolve()), "sha256": tosa_snapshot["sha256"]},
        "shaders": [{"path": str(path.resolve()), "sha256": snapshot["sha256"]}
                    for path, snapshot in zip(shader_paths, shader_snapshots)],
    }
    return {"fixture": fixture, "model": fixture,
            "host_program_identity": metadata_value["host_program_identity"],
            "codegen_identity": metadata_value["codegen_identity"],
            "codegen_version": metadata_value["codegen_version"],
            "source": source_snapshot, "header": header_snapshot,
            "metadata": metadata_snapshot, "plan": plan_snapshot,
            "host_program": program_snapshot,
            "tosa_ir": tosa_snapshot,
            "tosa_ir_sha256": metadata_value["tosa_ir_sha256"],
            "package_targets": package_targets,
            "reshape_views": reshape_views,
            "shaders": shader_snapshots,
            "capture": replay, "cpu_aot_compile": source_compile,
            "compile": compile_record, "link": link_record,
            "native_loader_compile": loader_compile,
            "native_loader_link": loader_link,
            "native_loader": loader_binary,
            "dso": link_record["dso"],
            "dso_sha256": link_record["dso"]["sha256"]}


def fixture_param(source_root: Path, fixture: str) -> Path:
    if fixture == "conv_relu":
        return source_root / "test/Vulkan/fixedrun_conv_relu.param"
    if fixture == "reshape":
        return source_root / "test/Vulkan/fixedrun_reshape_conv.param"
    if fixture == "transpose":
        return source_root / "test/Vulkan/fixedrun_transpose.param"
    if fixture == "split_concat":
        return source_root / "test/Numerical/fixtures/split_three.param"
    return source_root / "test/Vulkan" / f"{fixture}.param"


def run_and_record(args, fixture: str, loader: Path, library: Path,
                   mode: str, phase: str, iteration: int,
                   input_values_a: list[list[float]], input_values_b: list[list[float]] | None,
                   input_tensors: list[dict], output_tensors: list[dict],
                   stage: Path, env: dict[str, str], run_rows: list[dict]) -> list[list[list[float]]]:
    base = args.output_root / "native-runs" / fixture / f"{phase}-{iteration}"
    input_paths_a, input_refs_a = write_inputs(
        stage, base.parent, fixture, f"{base.name}-a", input_values_a)
    output_paths_a = [base.parent / f"{base.name}-a-out-{index}.f32"
                      for index in range(len(output_tensors))]
    input_paths_b = None
    output_paths_b = None
    input_refs_b = []
    if input_values_b is not None:
        input_paths_b, input_refs_b = write_inputs(
            stage, base.parent, fixture, f"{base.name}-b", input_values_b)
        output_paths_b = [base.parent / f"{base.name}-b-out-{index}.f32"
                          for index in range(len(output_tensors))]
    measurements, command_record = run_native_loader(
        loader, library, mode, input_paths_a, output_paths_a, stage,
        base.parent / f"{base.name}-native-loader.log", env,
        input_paths_b, output_paths_b)
    input_refs = [input_refs_a]
    if mode in ("repeat", "concurrent"):
        input_refs.append(input_refs_b)
    elif mode == "interleaved":
        input_refs.extend((input_refs_b, input_refs_a, input_refs_b))
    outputs = []
    for index, measurement in enumerate(measurements):
        paths = [Path(path) for path in measurement["output_paths"]]
        actual = read_output_set(paths, output_tensors)
        outputs.append(actual)
        run_rows.append({
            "fixture": fixture,
            "phase": phase,
            "iteration": iteration + index,
            "loader_result_index": index,
            "backend": "fixedrun-candidate" if mode != "single" else "native-reference",
            "inputs": input_refs[min(index, len(input_refs) - 1)],
            "outputs": [artifact(path, stage) for path in paths],
            "input_shapes": [tensor["shape"] for tensor in input_tensors],
            "output_shapes": [tensor["shape"] for tensor in output_tensors],
            "status": measurement["status"],
            "diagnostic": measurement["diagnostic"],
            "validation_error_count": 0,
            "elapsed_ns": measurement["elapsed_ns"],
            "dso_sha256": sha256_file(library),
            "native_loader": {
                "executable": artifact(loader, stage),
                **command_record,
            },
        })
    return outputs


def execute_fixture(args, fixture: str, stage: Path, env: dict[str, str],
                    legacy_reference: dict, run_rows: list[dict],
                    candidate_builds: list[dict]) -> dict:
    candidate_dir = args.candidate_root / fixture
    param = fixture_param(args.source_root, fixture)
    bin_path = (args.candidate_root / f"{fixture}.bin"
                if fixture in {"conv_relu", "reshape"}
                else args.candidate_root / "empty.bin")
    if not param.is_file() or not bin_path.is_file():
        raise RuntimeError(f"missing fixed-run source fixture: {fixture}")
    input_copy_dir = args.output_root / "source-inputs" / fixture
    param_copy = write_bytes(input_copy_dir / param.name, param.read_bytes(), stage)
    bin_copy = write_bytes(input_copy_dir / bin_path.name, bin_path.read_bytes(), stage)
    plan_path = candidate_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    input_tensors, output_tensors = tensor_metadata(plan)
    args.output_root.mkdir(parents=True, exist_ok=True)

    replay_dir = args.output_root / "capture-replay" / fixture
    replay = capture_fixedrun(args, fixture, param, bin_path,
                              replay_dir, stage, env)
    incremental_capture = capture_fixedrun(
        args, fixture, param, bin_path, candidate_dir, stage, env,
        reference_dir=replay_dir)
    replay["incremental_rebuild"] = incremental_capture
    replay["unowned_output_rejection"] = reject_unowned_capture_file(
        args, fixture, param, bin_path, replay_dir,
        args.output_root / "capture-negative" / fixture, stage, env)

    cpu_dir = args.output_root / "cpu-aot" / fixture
    cpu_compile = compile_model(args, str(args.compiler), fixture, param, bin_path,
                                cpu_dir, "cpu", stage, env)
    cpu_manifest_path = cpu_dir / f"{fixture}.json"
    cpu_manifest = json.loads(cpu_manifest_path.read_text(encoding="utf-8"))
    cpu_library = cpu_dir / f"lib{fixture}.so"
    cpu_loader_build = compile_cpu_native_loader(args, fixture, cpu_dir, stage, env)
    cpu_loader = stage / cpu_loader_build["executable"]["path"]

    independent_legacy = (
        legacy_reference.get("status") == "verified-independent-source")
    legacy_dir = args.output_root / "legacy-vulkan" / fixture
    if independent_legacy:
        legacy_compiler = legacy_reference["compiler"]["path"]
        legacy_tools = legacy_reference["tools"]
    else:
        # The local generic runtime observation is useful for ordinary CTest,
        # but it must never be reported as the frozen legacy reference.
        legacy_compiler = str(args.compiler)
        legacy_tools = None
    legacy_compile = compile_model(
        args, legacy_compiler, fixture, param, bin_path, legacy_dir,
        "vulkan", stage, env, legacy_tools)
    legacy_library = legacy_dir / f"lib{fixture}_vulkan.so"
    legacy_compile.update({
        "backend": "vulkan",
        "compiler": {"path": str(legacy_compiler),
                     "sha256": sha256_file(Path(legacy_compiler))},
        "param": param_copy,
        "bin": bin_copy,
        "library": artifact(legacy_library, stage),
        "library_sha256": sha256_file(legacy_library),
        "header": artifact(legacy_dir / f"{fixture}_vulkan.h", stage),
        "manifest": artifact(legacy_dir / f"{fixture}_vulkan.json", stage),
        "plan": artifact(legacy_dir / f"{fixture}_vulkan.plan.json", stage),
        "reference_manifest": legacy_reference.get("manifest"),
        "reference_status": legacy_reference.get("status"),
    })

    build_record = build_artifact_record(args, fixture, stage, cpu_compile,
                                         replay, env)
    candidate_builds.append(build_record)
    candidate_library = stage / build_record["dso"]["path"]
    candidate_loader = stage / build_record["native_loader"]["path"]
    candidate_hash = build_record["dso_sha256"]

    seed_inputs = make_inputs(input_tensors, 41041)
    seed_golden = golden_outputs(fixture, plan, seed_inputs, bin_path)
    reference_runs = []
    cpu_outputs = run_and_record(
        args, fixture, cpu_loader, cpu_library, "single", "cpu_aot_reference",
        0, seed_inputs, None, input_tensors, output_tensors, stage, env,
        reference_runs)[0]
    legacy_loader = candidate_loader
    legacy_outputs = run_and_record(
        args, fixture, legacy_loader, legacy_library, "single",
        "legacy_vulkan_reference" if independent_legacy else "current_generic_reference",
        0, seed_inputs, None, input_tensors, output_tensors, stage, env,
        reference_runs)[0]

    candidate_seed_runs = run_and_record(
        args, fixture, candidate_loader, candidate_library, "repeat",
        "same_seed_repeat", 0, seed_inputs, seed_inputs,
        input_tensors, output_tensors, stage, env, run_rows)
    first_values, repeat_values = candidate_seed_runs
    for output_index, values in enumerate(repeat_values):
        compare(first_values[output_index], values,
                f"{fixture} same-input repeat output {output_index}")
    for output_index, values in enumerate(first_values):
        compare(seed_golden[output_index], values,
                f"{fixture} candidate output {output_index}")

    different_inputs = make_inputs(input_tensors, 99173)
    different_golden = golden_outputs(fixture, plan, different_inputs, bin_path)
    different_runs = run_and_record(
        args, fixture, candidate_loader, candidate_library, "repeat",
        "different_input", 1, seed_inputs, different_inputs,
        input_tensors, output_tensors, stage, env, run_rows)
    different_values = different_runs[1]
    for output_index, values in enumerate(different_values):
        compare(different_golden[output_index], values,
                f"{fixture} different-input output {output_index}")

    concurrent_inputs = (make_inputs(input_tensors, 20260101),
                         make_inputs(input_tensors, 20260102))
    concurrent_values = run_and_record(
        args, fixture, candidate_loader, candidate_library, "concurrent",
        "same_context_concurrent", 0, concurrent_inputs[0], concurrent_inputs[1],
        input_tensors, output_tensors, stage, env, run_rows)
    for call_index, values in enumerate(concurrent_values):
        expected = golden_outputs(fixture, plan, concurrent_inputs[call_index], bin_path)
        for output_index, actual in enumerate(values):
            compare(expected[output_index], actual,
                    f"{fixture} same-context concurrent call {call_index}")

    interleaved_inputs = (seed_inputs, different_inputs)
    interleaved_input_sequence = [seed_inputs, different_inputs,
                                  seed_inputs, different_inputs]
    interleaved_values = run_and_record(
        args, fixture, candidate_loader, candidate_library, "interleaved",
        "interleaved_contexts", 0, interleaved_inputs[0], interleaved_inputs[1],
        input_tensors, output_tensors, stage, env, run_rows)
    if len(interleaved_values) != len(interleaved_input_sequence):
        raise RuntimeError(f"{fixture} interleaved loader did not return A/B/A/B")
    for call_index, (values, expected_inputs) in enumerate(
        zip(interleaved_values, interleaved_input_sequence)):
        expected = golden_outputs(fixture, plan, expected_inputs, bin_path)
        for output_index, actual in enumerate(values):
            compare(expected[output_index], actual,
                    f"{fixture} interleaved context {call_index}")

    output_records = []
    for index, output_tensor in enumerate(output_tensors):
        comparison = {
            "candidate": compare(seed_golden[index], first_values[index],
                                 f"{fixture} candidate output {index}"),
            "cpu_aot": compare(seed_golden[index], cpu_outputs[index],
                               f"{fixture} CPU output {index}")}
        comparison["legacy_vulkan" if independent_legacy else "current_generic"] = \
            compare(seed_golden[index], legacy_outputs[index],
                    f"{fixture} {'legacy' if independent_legacy else 'generic'} output {index}")
        base = args.output_root / "baseline" / fixture
        output_records.append({
            "name": output_tensor["name"],
            "shape": output_tensor["shape"],
            "elements": checked_product(output_tensor["shape"]),
            "golden_raw": write_floats(base / f"golden-{index}.f32",
                                        seed_golden[index], stage),
            "cpu_aot_raw": write_floats(base / f"cpu-aot-{index}.f32",
                                         cpu_outputs[index], stage),
            "legacy_vulkan_raw": (write_floats(
                base / f"legacy-vulkan-{index}.f32", legacy_outputs[index], stage)
                if independent_legacy else None),
            "current_generic_raw": (write_floats(
                base / f"current-generic-{index}.f32", legacy_outputs[index], stage)
                if not independent_legacy else None),
            "candidate_raw": write_floats(base / f"candidate-{index}.f32",
                                           first_values[index], stage),
            "comparison": comparison})
    primary_input_refs = [write_floats(
        args.output_root / "baseline" / fixture / f"input-{index}.f32",
        values, stage) for index, values in enumerate(seed_inputs)]
    cpu_aot_record = {
        "compile": cpu_compile,
        "manifest": artifact(cpu_manifest_path, stage),
        "library": artifact(cpu_library, stage),
        "library_sha256": sha256_file(cpu_library),
        "loader_build": cpu_loader_build,
        "loader": artifact(cpu_loader, stage),
    }
    return {"name": fixture,
            "param": param_copy,
            "param_source": str(param.relative_to(args.source_root)),
            "bin": bin_copy,
            "bin_source": ("generated-at-build-time"
                           if fixture in {"conv_relu", "reshape"}
                           else "generated-empty-build-input"),
            "bin_generator": (snapshot_artifact(
                args.source_root / "test/Vulkan/generate_fixedrun_conv_bin.py",
                args.output_root / "source-inputs" / fixture /
                "generate_fixedrun_conv_bin.py", stage)
                if fixture in {"conv_relu", "reshape"} else None),
            "bin_generator_source_path": (
                "test/Vulkan/generate_fixedrun_conv_bin.py"
                if fixture in {"conv_relu", "reshape"} else None),
            "inputs": [{"name": tensor["name"], "shape": tensor["shape"],
                        "elements": checked_product(tensor["shape"]),
                        "raw": raw}
                       for tensor, raw in zip(input_tensors, primary_input_refs)],
            "outputs": output_records,
            "candidate_build": build_record,
            "references": {
                "independent_legacy": independent_legacy,
                "cpu_aot": cpu_aot_record,
                "legacy_vulkan_compile": legacy_compile,
                "runs": reference_runs,
            },
            "cpu_aot_compile": cpu_compile,
            "legacy_vulkan_compile": legacy_compile,
            "reference_runs": reference_runs,
            "native_loader_build": cpu_loader_build}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "icd_manifest", "source_root", "build_root",
                 "candidate_root", "capture_tool", "compiler", "driver", "opt",
                 "translate", "clang", "nm", "readelf", "llvm_as", "mlir_opt",
                 "spirv_val", "spirv_dis", "spirv_extract", "host_cxx", "vulkaninfo"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--fixture", action="append", choices=FIXTURES)
    for fixture in FIXTURES:
        parser.add_argument(f"--loader-{fixture.replace('_', '-')}",
                            dest=f"loader_{fixture}", type=Path, required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    parser.add_argument("--ar", type=Path,
                        default=Path(shutil.which("ar") or "ar"))
    parser.add_argument("--icd", choices=("nvidia", "llvmpipe"), required=True)
    parser.add_argument("--profile", choices=("release", "sanitize"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.build_root = args.build_root.resolve()
    args.candidate_root = args.candidate_root.resolve()
    args.source_root = args.source_root.resolve()
    stage = args.build_root
    args.output_root = args.output.parent
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing raw evidence: {args.output}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(
            f"refusing to reuse non-empty evidence directory: {args.output_root}")
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = str(args.icd_manifest.resolve())
    env["NCNN_VULKAN_FIXEDRUN_PROFILE"] = args.profile
    os.environ.update({"VK_ICD_FILENAMES": env["VK_ICD_FILENAMES"]})
    device = selected_device(args, stage, env)
    legacy_reference = copy_legacy_reference(args.legacy_root, stage,
                                             args.output_root)
    run_rows = []
    candidate_builds = []
    selected_fixtures = args.fixture or list(FIXTURES)
    if len(set(selected_fixtures)) != len(selected_fixtures):
        raise RuntimeError("fixed-run fixture selection contains duplicates")
    fixtures = [execute_fixture(args, name, stage, env, legacy_reference,
                                run_rows, candidate_builds)
                for name in selected_fixtures]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_gate import source_fingerprint
        source_id = source_fingerprint(args.source_root)
    except (ImportError, OSError, ValueError, subprocess.SubprocessError):
        source_id = ""
    document = {"schema": SCHEMA,
                "icd": args.icd,
                "profile": args.profile,
                "source_fingerprint": source_id,
                "legacy_reference": legacy_reference,
                "device": device,
                "producer": {"argv": sys.argv,
                             "cwd": str(Path.cwd().resolve()),
                             "source_fingerprint": source_id,
                             "profile": args.profile},
                "source_inputs": [{"fixture": item["name"],
                                   "param": item["param"], "bin": item["bin"]}
                                  for item in fixtures],
                "candidate_builds": candidate_builds,
                "fixtures": fixtures,
                "runs": run_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"run_fixedrun.py: {error}", file=sys.stderr)
        sys.exit(1)
