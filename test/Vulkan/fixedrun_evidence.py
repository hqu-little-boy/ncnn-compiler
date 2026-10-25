#!/usr/bin/env python3
"""Independent evidence checks for private Vulkan fixed-run candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess
import sys
import tarfile

import host_program_evidence as host_program_audit

SCHEMA = "ncnn-vulkan-fixedrun-evidence/v1"
LEGACY_SCHEMA = "ncnn-vulkan-legacy-reference/v1"
LEGACY_REVISION = "6094cb56abe24cdbcde84fbb72813da8aa24c147"
FIXTURES = frozenset({"relu", "add_relu", "conv_relu", "add_split_relu",
                      "split_concat", "reshape", "transpose"})
DEVICES = ("nvidia", "llvmpipe")
PHASES = frozenset({"same_seed_repeat", "different_input",
                    "same_context_concurrent", "interleaved_contexts"})
FIXTURE_IO = {
    "relu": ([[1, 1, 257]], [[1, 1, 257]]),
    "add_relu": ([[1, 1, 257], [1, 1, 257]], [[1, 1, 257]]),
    "conv_relu": ([[2, 1, 1]], [[2, 1, 1]]),
    "add_split_relu": ([[1, 1, 257], [1, 1, 257]],
                       [[1, 1, 257], [1, 1, 257]]),
    "split_concat": ([[3, 4, 5]], [[9, 4, 5]]),
    "reshape": ([[2, 3, 4]], [[2, 3, 4]]),
    "transpose": ([[2, 3, 4]], [[3, 2, 4]]),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_RUN = re.compile(
    r"VulkanRuntimeContext|VulkanExecutionPlan|VulkanHostProgram|"
    r"NCNNVulkanRuntime|NCNNVulkanPlanSupport|ExecutionPlan|PlanExecutor|"
    r"runtime\.run|run\s*\(\s*(?:plan|program)|"
    r"kernel[_ ]?(?:schedule|table|dispatch)|reads\s*/\s*writes|"
    r"hazard[_ ]?(?:plan|analysis|map)|(?:read|write)[_ ]?(?:set|hazard)", re.I)


class EvidenceError(ValueError):
    """Raised when evidence is incomplete or its claimed bytes disagree."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _one_option(argv: list[str], option: str, label: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise EvidenceError(f"{label} must appear exactly once with a value")
    return argv[positions[0] + 1]


def _artifact(stage: Path, record: object, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise EvidenceError(f"{label} artifact record is missing")
    relative, digest = record.get("path"), record.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or "\\" in relative:
        raise EvidenceError(f"{label} path must be stage-relative")
    path = (stage / relative).resolve()
    try:
        path.relative_to(stage.resolve())
    except ValueError as error:
        raise EvidenceError(f"{label} escapes its stage") from error
    if not path.is_file():
        raise EvidenceError(f"{label} is missing: {relative}")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise EvidenceError(f"{label} SHA-256 is malformed")
    actual = sha256_file(path)
    if actual != digest:
        raise EvidenceError(f"{label} SHA-256 differs from actual bytes")
    return path, digest


def _audit_logged_command(stage: Path, command: object, label: str) -> dict:
    if not isinstance(command, dict):
        raise EvidenceError(f"{label} execution record is missing")
    argv, cwd, returncode = command.get("argv"), command.get("cwd"), command.get("returncode")
    if (not isinstance(argv, list) or not argv or
            not all(isinstance(token, str) for token in argv) or
            not isinstance(cwd, str) or type(returncode) is not int or returncode != 0):
        raise EvidenceError(f"{label} argv/cwd/returncode is malformed or failed")
    log, log_hash = _artifact(stage, command.get("log"), f"{label} stdout/stderr log")
    payload = log.read_bytes()
    stdout, separator, stderr = payload.partition(b"\n--- stderr ---\n")
    if (not separator or
            hashlib.sha256(stdout).hexdigest() != command.get("stdout_sha256") or
            hashlib.sha256(stderr).hexdigest() != command.get("stderr_sha256")):
        raise EvidenceError(f"{label} raw log differs from captured stdout/stderr hashes")
    stdout_record = None
    if command.get("stdout") is not None:
        stdout_path, stdout_hash = _artifact(stage, command["stdout"], f"{label} stdout bytes")
        if stdout_path.read_bytes() != stdout:
            raise EvidenceError(f"{label} stdout artifact differs from execution log")
        stdout_record = {"path": stdout_path.relative_to(stage).as_posix(),
                         "sha256": stdout_hash}
    return {"argv": argv, "cwd": cwd, "returncode": returncode,
            "stdout_sha256": command["stdout_sha256"],
            "stderr_sha256": command["stderr_sha256"],
            "stdout_bytes": stdout, "stderr_bytes": stderr,
            "stdout": stdout_record,
            "log": {"path": log.relative_to(stage).as_posix(), "sha256": log_hash}}


def _command_report(summary: dict) -> dict:
    return {key: value for key, value in summary.items()
            if key not in {"stdout_bytes", "stderr_bytes"}}


def _shape_elements(shape: object, label: str) -> int:
    if (not isinstance(shape, list) or not shape or
            any(type(dim) is not int or dim <= 0 for dim in shape)):
        raise EvidenceError(f"{label} must be a positive static shape")
    total = 1
    for dim in shape:
        total *= dim
        if total > (1 << 63) - 1:
            raise EvidenceError(f"{label} element count overflows")
    return total


def _float_array(stage: Path, record: object, count: int,
                 label: str) -> tuple[list[float], dict]:
    path, digest = _artifact(stage, record, label)
    data = path.read_bytes()
    if len(data) != count * 4:
        raise EvidenceError(f"{label} byte count differs from shape")
    values = list(struct.unpack(f"<{count}f", data))
    if not all(math.isfinite(value) for value in values):
        raise EvidenceError(f"{label} contains non-finite values")
    return values, {"path": path.relative_to(stage).as_posix(), "sha256": digest}


def _compare(actual: list[float], expected: list[float], atol: object,
             rtol: object, label: str) -> dict:
    if (type(atol) not in (int, float) or type(rtol) not in (int, float) or
            not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0):
        raise EvidenceError(f"{label} tolerance is invalid")
    if len(actual) != len(expected):
        raise EvidenceError(f"{label} arrays have different sizes")
    if not all(math.isfinite(value) for value in (*actual, *expected)):
        raise EvidenceError(f"{label} contains non-finite values")
    max_abs = max_rel = 0.0
    for index, (got, want) in enumerate(zip(actual, expected)):
        difference = abs(got - want)
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, difference / max(abs(want), 1.0e-30))
        if difference > atol + rtol * abs(want):
            raise EvidenceError(f"{label} exceeds tolerance at element {index}")
    return {"max_abs_error": max_abs, "max_rel_error": max_rel,
            "elements": len(actual)}


def _scalar_golden(name: str, inputs: list[list[float]], bin_bytes: bytes) -> list[list[float]]:
    if name == "relu":
        return [[max(0.0, value) for value in inputs[0]]]
    if name == "add_relu":
        return [[max(0.0, x + y) for x, y in zip(inputs[0], inputs[1])]]
    if name == "add_split_relu":
        summed = [x + y for x, y in zip(inputs[0], inputs[1])]
        return [summed, [max(0.0, value) for value in summed]]
    if name in {"conv_relu", "reshape"}:
        if len(bin_bytes) != 4 + 6 * 4 or struct.unpack_from("<I", bin_bytes)[0] != 0:
            raise EvidenceError(f"{name} scalar golden 1x1 Conv weight payload is malformed")
        channels, height, width = FIXTURE_IO[name][0][0]
        output_channels, out_height, out_width = FIXTURE_IO[name][1][0]
        if channels != 2 or output_channels != 2 or (height, width) != (out_height, out_width):
            raise EvidenceError(f"{name} scalar golden has unsupported Conv layout")
        values = struct.unpack_from("<6f", bin_bytes, 4)
        weights, biases = values[:4], values[4:]
        output = []
        for out_channel in range(output_channels):
            for position in range(height * width):
                accumulator = biases[out_channel]
                for in_channel in range(channels):
                    input_value = inputs[0][in_channel * height * width + position]
                    product = input_value * weights[out_channel * channels + in_channel]
                    accumulator = struct.unpack(
                        "<f", struct.pack("<f", accumulator + product))[0]
                output.append(max(0.0, accumulator) if name == "conv_relu" else accumulator)
        return [output]
    if name == "split_concat":
        return [list(inputs[0]) * 3]
    if name == "reshape":
        return [list(inputs[0])]
    if name == "transpose":
        source = inputs[0]
        result = []
        for axis0 in range(3):
            for axis1 in range(2):
                for axis2 in range(4):
                    result.append(source[(axis1 * 3 + axis0) * 4 + axis2])
        return [result]
    raise EvidenceError(f"no independent scalar golden for fixture {name}")


def _extract_function_body(source: str, pattern: re.Pattern,
                           label: str) -> str:
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise EvidenceError(f"generated source has no unique {label} definition")
    match = matches[0]
    opening = source.find("{", match.end() - 1)
    if opening < 0:
        raise EvidenceError(f"generated {label} definition has no body")
    depth = 0
    for index in range(opening, len(source)):
        depth += source[index] == "{"
        depth -= source[index] == "}"
        if depth == 0:
            return source[opening + 1:index]
    raise EvidenceError(f"generated {label} body has unbalanced braces")


def _run_body(source: str, model: str) -> str:
    del model
    return _extract_function_body(
        source, re.compile(r"\bncnn_vk_status\s+run\s*\("),
        "fixed-run context method")


def _abi_run_body(source: str, model: str) -> str:
    return _extract_function_body(
        source, re.compile(
            rf'extern\s+"C"\s+ncnn_vk_status\s+{re.escape(model)}_vk_run\s*\('),
        "model-specific C ABI run")


def _audit_typed_operations(program: dict, body: str) -> int:
    operations = program.get("run_operations")
    buffers, pipelines = program.get("buffers"), program.get("pipelines")
    if not isinstance(operations, list) or not operations or not isinstance(buffers, list) or not isinstance(pipelines, list):
        raise EvidenceError("typed host program lacks declarations or run operations")
    buffer_slots = {item.get("id"): slot for slot, item in enumerate(buffers) if isinstance(item, dict)}
    pipeline_slots = {item.get("id"): slot for slot, item in enumerate(pipelines) if isinstance(item, dict)}
    stages = {"host": "VK_PIPELINE_STAGE_HOST_BIT", "transfer": "VK_PIPELINE_STAGE_TRANSFER_BIT",
              "compute": "VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"}
    accesses = {"host-write": "VK_ACCESS_HOST_WRITE_BIT", "transfer-read": "VK_ACCESS_TRANSFER_READ_BIT",
                "transfer-write": "VK_ACCESS_TRANSFER_WRITE_BIT", "shader-read": "VK_ACCESS_SHADER_READ_BIT",
                "shader-write": "VK_ACCESS_SHADER_WRITE_BIT",
                "shader-read-write": "VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT",
                "host-read": "VK_ACCESS_HOST_READ_BIT"}
    cursor = dispatches = 0
    previous_control = None
    for op in operations:
        if not isinstance(op, dict):
            raise EvidenceError("typed run operation is malformed")
        kind, marker = op.get("op"), None
        if kind == "upload":
            slot = buffer_slots.get(op.get("staging_buffer"))
            source_index = _exact_int(op.get("source_id"), "input upload source id")
            if op.get("source") != "input" or slot is None:
                raise EvidenceError("run upload is not a valid public input upload")
            marker = (f"std::memcpy(mappedBuffer({slot}, UINT64_C({op.get('byte_size')})).data(), "
                      f"inputs[{source_index}].data()")
        elif kind == "mapped-sync":
            slot, action = buffer_slots.get(op.get("buffer")), op.get("action")
            if slot is None or action not in {"flush", "invalidate"}:
                raise EvidenceError("run mapped synchronization is invalid")
            marker = f"require(buffers_[{slot}]->{action}("
        elif kind == "barrier":
            source, destination = stages.get(op.get("source_stage")), stages.get(op.get("destination_stage"))
            items = op.get("buffers")
            if source is None or destination is None or not isinstance(items, list) or not items:
                raise EvidenceError("typed barrier stage/buffer list is invalid")
            marker = f"vkCmdPipelineBarrier(commands_->get_command_buffer(), {source}, {destination}"
            for barrier in items:
                if not isinstance(barrier, dict):
                    raise EvidenceError("typed buffer barrier is malformed")
                slot = buffer_slots.get(barrier.get("buffer"))
                src, dst = accesses.get(barrier.get("source_access")), accesses.get(barrier.get("destination_access"))
                if slot is None or src is None or dst is None:
                    raise EvidenceError("typed barrier has invalid slot/access")
                tokens = (f".srcAccessMask = {src}", f".dstAccessMask = {dst}",
                          f".buffer = buffers_[{slot}]->get_handle()",
                          f".offset = UINT64_C({barrier.get('offset')})",
                          f".size = UINT64_C({barrier.get('byte_size')})")
                if any(token not in body for token in tokens):
                    raise EvidenceError("generated barrier differs from typed access/range")
        elif kind == "copy":
            src_slot, dst_slot = buffer_slots.get(op.get("source")), buffer_slots.get(op.get("destination"))
            if src_slot is None or dst_slot is None:
                raise EvidenceError("typed copy references unknown buffer")
            tokens = (f".srcOffset = UINT64_C({op.get('source_offset')})",
                      f".dstOffset = UINT64_C({op.get('destination_offset')})",
                      f".size = UINT64_C({op.get('byte_size')})")
            if any(token not in body for token in tokens):
                raise EvidenceError("generated copy range differs from typed operation")
            marker = ("vkCmdCopyBuffer(commands_->get_command_buffer(), "
                      f"buffers_[{src_slot}]->get_handle(), buffers_[{dst_slot}]->get_handle()")
        elif kind == "control":
            action = op.get("action")
            controls = {"wait-fence": "commands_->wait(", "reset-fence": "commands_->reset(",
                        "reset-command-pool": "commands_->reset(", "begin-commands": "commands_->begin(",
                        "end-commands": "commands_->end(", "submit": "commands_->submit_and_wait("}
            marker = controls.get(action)
            if ((action == "reset-command-pool" and previous_control == "reset-fence") or
                    (action == "wait-fence" and previous_control == "submit")):
                marker = None
            previous_control = action
            if marker is None:
                continue
        elif kind == "bind-pipeline":
            slot = pipeline_slots.get(op.get("pipeline"))
            if slot is None:
                raise EvidenceError("pipeline bind references unknown pipeline")
            marker = ("vkCmdBindPipeline(commands_->get_command_buffer(), "
                      f"VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[{slot}]->get_pipeline()")
        elif kind == "bind-descriptors":
            slot = pipeline_slots.get(op.get("pipeline"))
            if slot is None:
                raise EvidenceError("descriptor bind references unknown pipeline")
            marker = ("vkCmdBindDescriptorSets(commands_->get_command_buffer(), "
                      f"VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[{slot}]->get_layout(), "
                      f"{op.get('first_set')}, {op.get('set_count')}")
        elif kind == "push-constants":
            slot, payload = pipeline_slots.get(op.get("pipeline")), op.get("bytes")
            if slot is None or not isinstance(payload, str) or not re.fullmatch(r"(?:[0-9a-f]{2})+", payload):
                raise EvidenceError("push constant declaration is invalid")
            byte_values = [int(payload[index:index + 2], 16) for index in range(0, len(payload), 2)]
            if any(f"UINT8_C(0x{value:02x})" not in body for value in byte_values):
                raise EvidenceError("generated push-constant bytes differ from typed operation")
            if f"{op.get('offset')}, {len(byte_values)}, push_constants.data()" not in body:
                raise EvidenceError("generated push-constant range differs from typed operation")
            marker = ("vkCmdPushConstants(commands_->get_command_buffer(), "
                      f"pipelines_[{slot}]->get_layout()")
        elif kind == "dispatch":
            slot, groups = pipeline_slots.get(op.get("pipeline")), op.get("groups")
            if slot is None or not isinstance(groups, list) or len(groups) != 3 or any(type(x) is not int or x <= 0 for x in groups):
                raise EvidenceError("typed dispatch declaration is invalid")
            marker = "vkCmdDispatch(commands_->get_command_buffer(), " + ", ".join(map(str, groups))
            dispatches += 1
        elif kind == "prepare-outputs":
            items = op.get("buffers")
            if not isinstance(items, list) or not items:
                raise EvidenceError("output preparation declaration is empty")
            marker = "std::array<std::span<std::byte>"
            for buffer_id in items:
                slot = buffer_slots.get(buffer_id)
                declaration = next((value for value in buffers
                                    if isinstance(value, dict) and value.get("id") == buffer_id), None)
                if slot is None or declaration is None or f"mappedBuffer({slot}, UINT64_C({declaration.get('byte_size')}))" not in body:
                    raise EvidenceError("generated readback mapping differs from typed output range")
        elif kind == "publish":
            output = _exact_int(op.get("output_index"), "output index")
            marker = f"std::memcpy(outputs[{output}].data(), readbacks[{output}].data(), "
            if f"UINT64_C({op.get('byte_size')})" not in body:
                raise EvidenceError("generated output size differs from typed operation")
        else:
            raise EvidenceError(f"unknown typed run operation {kind!r}")
        if marker not in body:
            raise EvidenceError(f"generated source omits typed operation {kind}")
        position = body.find(marker, cursor)
        if position < 0:
            raise EvidenceError(f"generated call order differs at operation {kind}")
        cursor = position + len(marker)
    if dispatches != len(pipelines):
        raise EvidenceError("typed dispatch count differs from pipeline declarations")
    return dispatches


def _stage_cache_values(stage: Path) -> dict[str, str]:
    cache = stage / "CMakeCache.txt"
    if not cache.is_file():
        raise EvidenceError("fresh stage CMake cache is missing")
    values = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def _configured_gate_tool(stage: Path, cache_key: str, identity_role: str) -> Path:
    cache = _stage_cache_values(stage)
    configured = cache.get(cache_key)
    if not configured:
        raise EvidenceError(f"fresh CMake cache does not identify {identity_role}")
    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = stage / configured_path
    path = configured_path.resolve()
    if not path.is_file():
        raise EvidenceError(f"configured {identity_role} is missing: {path}")
    state_path = stage / "gate-state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read configured {identity_role} gate identity: {error}") from error
    cache_record = state.get("cmake_cache") if isinstance(state, dict) else None
    cache_path = stage / "CMakeCache.txt"
    if (not isinstance(cache_record, dict) or
            cache_record.get("path") != "CMakeCache.txt" or
            cache_record.get("sha256") != sha256_file(cache_path)):
        raise EvidenceError("configured tool CMake cache differs from its gate-bound bytes")
    identities = state.get("toolchain_identities") if isinstance(state, dict) else None
    matches = [item for item in identities or []
               if isinstance(item, dict) and item.get("role") == identity_role]
    if len(matches) != 1:
        raise EvidenceError(f"gate identity does not uniquely bind {identity_role}")
    identity = matches[0]
    digest = identity.get("sha256")
    if (not isinstance(identity.get("path"), str) or
            Path(identity["path"]).resolve() != path or
            not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or
            sha256_file(path) != digest):
        raise EvidenceError(f"configured {identity_role} differs from its gate-bound binary")
    return configured_path


def _configured_sanitizer_runtime_dir(stage: Path) -> Path:
    compiler = _configured_gate_tool(stage, "CMAKE_CXX_COMPILER", "cxx-compiler")
    result = subprocess.run([str(compiler), "--print-resource-dir"], cwd=stage,
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise EvidenceError("configured Clang cannot report its resource directory")
    try:
        resource_dir_spelling = Path(result.stdout.decode("utf-8", "strict").strip())
    except UnicodeError as error:
        raise EvidenceError("configured Clang resource directory is not UTF-8") from error
    resource_dir = resource_dir_spelling.resolve()
    if not resource_dir.is_dir():
        raise EvidenceError("configured Clang resource directory is missing")
    runtime_dir = resource_dir_spelling / "lib/linux"
    runtime_dir_resolved = runtime_dir.resolve()
    if (not runtime_dir_resolved.is_dir() or
            not list(runtime_dir_resolved.glob("libclang_rt.asan-*.so"))):
        raise EvidenceError("configured Clang shared ASan runtime directory is missing")
    return runtime_dir


def _audit_candidate_link_scope(argv: list[str], configured_cxx: Path,
                                sanitizer_runtime_dir: Path | None) -> None:
    if (not argv or not isinstance(argv[0], str) or
            Path(argv[0]).resolve() != configured_cxx.resolve()):
        raise EvidenceError("candidate DSO link does not use the configured gate C++ driver")
    allowed_runtime_rpath = (
        f"-Wl,-rpath,{sanitizer_runtime_dir}"
        if sanitizer_runtime_dir is not None else None)
    forbidden_inputs = ("NCNNVulkanRuntime", "NCNNVulkanPlanSupport",
                        "VulkanRuntime", "VulkanExecutionPlan", "LLVM", "MLIR")
    for token in argv[1:]:
        if "rpath" in token.lower():
            if token == allowed_runtime_rpath:
                continue
            raise EvidenceError("candidate DSO link command contains an unconfigured RPATH")
        if any(item.lower() in token.lower() for item in forbidden_inputs):
            raise EvidenceError(
                "candidate DSO link command contains a generic executor/compiler input")


def _option_values(argv: list[str], option: str) -> list[str]:
    values = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise EvidenceError(f"{option} is missing its forwarded value")
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
        index += 1
    return values


def _audit_forwarded_link_sanitizer_args(argv: list[str], profile: str,
                                         label: str) -> None:
    clang_args = _option_values(argv, "--clang-arg")
    linker_args = _option_values(argv, "--linker-arg")
    forwarded = clang_args + linker_args
    if any(token.startswith(("-fno-sanitize=", "-fsanitize-recover=",
                             "-fsanitize-blacklist=", "-fsanitize-ignorelist=",
                             "-fsanitize-coverage-ignorelist=", "-fno-sanitize-coverage="))
           for token in forwarded):
        raise EvidenceError(f"{label} forwards sanitizer suppressions")
    if profile == "sanitize":
        sanitizer_flags = {"-fsanitize=address,undefined",
                           "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"}
        if (not sanitizer_flags <= set(clang_args) or
                "-shared-libasan" not in linker_args or
                "-static-libasan" in linker_args):
            raise EvidenceError(f"{label} does not forward instrumented compile and shared-only ASan link flags")
    elif any(token.startswith(("-fsanitize=", "-fno-sanitize", "-shared-libasan",
                               "-static-libasan")) for token in forwarded):
        raise EvidenceError(f"{label} forwards sanitizer flags in a non-SAN profile")


def _audit_shared_asan_needed(stage: Path, binary: Path, profile: str,
                              label: str) -> dict | None:
    if profile not in {"release", "sanitize"}:
        return None
    binary = binary.resolve()
    try:
        binary.relative_to(stage.resolve())
    except ValueError as error:
        raise EvidenceError(f"{label} ELF target is outside the audited stage") from error
    if not binary.is_file():
        raise EvidenceError(f"{label} ELF target is missing: {binary}")
    readelf = _configured_gate_tool(stage, "LLVM_READELF", "llvm-readelf")
    argv = [str(readelf), "-d", str(binary)]
    result = subprocess.run(argv, cwd=stage, capture_output=True, check=False)
    if result.returncode != 0:
        raise EvidenceError(f"{label} configured readelf -d failed")
    output = result.stdout.decode("utf-8", "replace")
    needed = sorted(set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", output)))
    has_shared_asan = any("libclang_rt.asan" in item for item in needed)
    if profile == "sanitize" and not has_shared_asan:
        raise EvidenceError(f"SAN {label} does not need the shared Clang ASan runtime")
    if profile == "release" and any(
        re.search(r"(?:libclang_rt\.(?:asan|ubsan)|lib(?:asan|ubsan))", item, re.I)
        for item in needed
    ):
        raise EvidenceError(f"Release {label} unexpectedly needs a sanitizer runtime")
    return {"artifact": {"path": binary.relative_to(stage).as_posix(),
                          "sha256": sha256_file(binary)},
            "readelf": {"argv": argv, "returncode": result.returncode,
                        "tool_sha256": sha256_file(readelf),
                        "stdout": output,
                        "stderr": result.stderr.decode("utf-8", "replace"),
                        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()},
            "needed_libraries": needed}


def _command_path(stage: Path, value: str, cwd: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} path is empty")
    base = Path(cwd)
    if not base.is_absolute():
        base = (stage / base).resolve()
    path = Path(value)
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(stage.resolve())
    except ValueError as error:
        raise EvidenceError(f"{label} path is outside stage") from error
    return path


def _actual_compile_command(stage: Path, source: Path,
                            model: str | None = None) -> tuple[list[str], str]:
    database = stage / "compile_commands.json"
    if not database.is_file():
        raise EvidenceError("fresh stage compile_commands.json is missing")
    try:
        entries = json.loads(database.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"fresh compile database cannot be read: {error}") from error
    if not isinstance(entries, list):
        raise EvidenceError("fresh compile database is not an array")
    matches = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            continue
        cwd = entry.get("directory")
        if not isinstance(cwd, str):
            continue
        source_path = Path(entry["file"])
        source_path = (source_path if source_path.is_absolute() else Path(cwd) / source_path).resolve()
        if source_path != source.resolve():
            continue
        argv = entry.get("arguments")
        if not isinstance(argv, list):
            command = entry.get("command")
            if not isinstance(command, str):
                continue
            try:
                argv = shlex.split(command)
            except ValueError as error:
                raise EvidenceError("fresh compile database command cannot be parsed") from error
        if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
            continue
        if model is not None and f"-DNCNN_FIXEDRUN_MODEL={model}" not in argv:
            continue
        matches.append((argv, str(Path(cwd).resolve())))
    if len(matches) != 1:
        raise EvidenceError(f"expected one actual compile database command for {source}, got {len(matches)}")
    return matches[0]


def _audit_helper_archive_listing(stage: Path, command: object, archive: Path,
                                 expected_archive: Path, ar: str,
                                 cwd: Path, label: str) -> list[str]:
    if not isinstance(command, dict):
        raise EvidenceError(f"{label} member-list command is missing")
    summary = _audit_logged_command(stage, command, label)
    argv = summary["argv"]
    if (len(argv) != 3 or Path(argv[0]).resolve() != Path(ar).resolve() or
            argv[1] != "t" or Path(argv[2]).resolve() != expected_archive.resolve() or
            Path(summary["cwd"]).resolve() != cwd.resolve()):
        raise EvidenceError(f"{label} argv/cwd does not bind the configured archive tool")
    stdout_record, _ = _artifact(stage, command.get("stdout"), f"{label} stdout bytes")
    if stdout_record.read_bytes() != summary["stdout_bytes"]:
        raise EvidenceError(f"{label} stdout artifact differs from its raw log")
    try:
        result = subprocess.run([ar, "t", str(archive)], cwd=cwd,
                                check=False, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except OSError as error:
        raise EvidenceError(f"cannot execute configured archive tool for {label}: {error}") from error
    if (type(result.returncode) is not int or result.returncode != 0 or
            result.stdout != summary["stdout_bytes"] or
            result.stderr != summary["stderr_bytes"]):
        raise EvidenceError(f"{label} does not match actual archive member-list bytes")
    try:
        return [line for line in result.stdout.decode("utf-8", errors="strict").splitlines()
                if line]
    except UnicodeError as error:
        raise EvidenceError(f"{label} member list is not UTF-8") from error


def _audit_helper_members(stage: Path, archive_snapshot: Path,
                          archive_target: Path, ar: str, cwd: Path,
                          members: list[str], records: object,
                          expected_members: dict[str, Path],
                          expected_member_targets: dict[str, Path], label: str) -> None:
    if (not isinstance(records, list) or len(records) != len(members) or
            len(members) != len(set(members)) or set(members) != set(expected_members)):
        raise EvidenceError(f"{label} names differ from the unique CMake object member set")
    for index, (name, record) in enumerate(zip(members, records)):
        if not isinstance(record, dict) or record.get("name") != name:
            raise EvidenceError(f"{label} member {index} order/name differs")
        payload_path, payload_hash = _artifact(
            stage, record.get("extracted_artifact"), f"{label} member {index} bytes")
        object_path, object_hash = _artifact(
            stage, record.get("object"), f"{label} member {index} CMake object snapshot")
        expected_object = expected_members[name]
        expected_object_hash = sha256_file(expected_object)
        target_record = record.get("object_target_path")
        if (object_hash != expected_object_hash or
                payload_path.read_bytes() != expected_object.read_bytes() or
                not isinstance(target_record, str) or
                Path(target_record).resolve() != expected_member_targets[name].resolve() or
                record.get("object_target_sha256") != payload_hash):
            raise EvidenceError(f"{label} member {name} differs from its actual CMake object bytes/target")
        extract = record.get("extract_command")
        if not isinstance(extract, dict):
            raise EvidenceError(f"{label} member {name} lacks ar-p command evidence")
        argv = extract.get("argv")
        if (not isinstance(argv, list) or len(argv) != 4 or
                Path(argv[0]).resolve() != Path(ar).resolve() or
                argv[1:] != ["p", str(archive_target.resolve()), name] or
                Path(extract.get("cwd", "")).resolve() != cwd.resolve() or
                type(extract.get("returncode")) is not int or extract["returncode"] != 0):
            raise EvidenceError(f"{label} member {name} extraction argv/cwd/returncode differs")
        stdout, stdout_hash = _artifact(stage, extract.get("stdout"),
                                        f"{label} member {name} stdout")
        stderr, stderr_hash = _artifact(stage, extract.get("stderr"),
                                        f"{label} member {name} stderr")
        try:
            result = subprocess.run([ar, "p", str(archive_snapshot), name], cwd=cwd,
                                    check=False, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except OSError as error:
            raise EvidenceError(f"cannot execute ar-p for {label}/{name}: {error}") from error
        if (type(result.returncode) is not int or result.returncode != 0 or
                result.returncode != extract["returncode"] or stdout != payload_path or
                stdout_hash != payload_hash or extract.get("stdout_sha256") != payload_hash or
                stderr_hash != hashlib.sha256(result.stderr).hexdigest() or
                extract.get("stderr_sha256") != stderr_hash or
                stdout.read_bytes() != result.stdout or stderr.read_bytes() != result.stderr or
                payload_hash != sha256_file(expected_object)):
            raise EvidenceError(f"{label} member {name} raw streams/hash differ from object bytes")
        if record.get("equals_compile_object") is not True:
            raise EvidenceError(f"{label} producer did not record member/object equality")


def _audit_helper_archive_rebuilds(stage: Path, helper: Path, ar: str,
                                  archive_cwd: Path, clean_cmake: Path,
                                  clean_script_target: Path,
                                  clean_script: Path, clean_script_hash: str,
                                  helper_object: Path, helper_object_hash: str,
                                  expected_members: dict[str, Path],
                                  archive_commands: list[dict], build: dict,
                                  archive_hash: str) -> dict:
    cycles = build.get("archive_rebuilds")
    if len(expected_members) != 1:
        raise EvidenceError("CMake helper object inventory is not unique")
    expected_object_snapshot = next(iter(expected_members.values()))
    if (build.get("archive_target_path") != str(helper.resolve()) or
            build.get("target_archive_path") != str(helper.resolve())):
        raise EvidenceError("helper archive rebuild record does not bind actual target path")
    if not isinstance(cycles, list) or len(cycles) != 2:
        raise EvidenceError("Vulkan helper archive lacks two clean/rebuild stability cycles")
    if len(expected_members) != len(set(expected_members)):
        raise EvidenceError("CMake helper object set has duplicate archive member names")
    for index, command in enumerate(archive_commands):
        if not isinstance(command, dict):
            raise EvidenceError("CMake helper archive command is malformed")
    if not clean_script.is_file():
        raise EvidenceError("helper CMake clean script is missing")
    clean_text = clean_script.read_text(encoding="utf-8", errors="strict")
    uncommented = re.sub(r"(?m)^\s*#.*$", "", clean_text)
    clean_match = re.fullmatch(r'\s*file\(\s*REMOVE_RECURSE\s+"([^"]+)"\s*\)\s*',
                               uncommented, flags=re.DOTALL)
    if (clean_match is None or
            _command_path(stage, clean_match.group(1), str(archive_cwd),
                          "cleaned helper archive") != helper.resolve()):
        raise EvidenceError("helper CMake clean script can remove files other than its archive")
    cmake_path = clean_cmake.resolve()
    cycle_hashes = []
    cycle_members = []

    def audit_before_snapshot(record: object, expected_hash: object,
                              listed_members: object, list_command: object,
                              label: str) -> list[str]:
        if record is None:
            if expected_hash is not None or listed_members != [] or list_command is not None:
                raise EvidenceError(f"{label} absent archive has nonempty before-state claims")
            return []
        snapshot_path, snapshot_hash = _artifact(stage, record, f"{label} archive snapshot")
        if snapshot_hash != expected_hash:
            raise EvidenceError(f"{label} snapshot hash differs from its before-state claim")
        observed = _audit_helper_archive_listing(
            stage, list_command, snapshot_path, helper, ar, archive_cwd,
            f"{label} prior archive list")
        if observed != listed_members:
            raise EvidenceError(f"{label} before-member list differs from snapshot bytes")
        return observed

    prior = build.get("prior_archive")
    if prior is None:
        prior_members = []
    else:
        if not isinstance(prior, dict):
            raise EvidenceError("prior helper archive record is malformed")
        prior_members = audit_before_snapshot(
            prior.get("snapshot"), prior.get("sha256"), prior.get("member_list"),
            prior.get("ar_list_command"), "pre-clean")
        _exact_int(prior.get("duplicate_member_count"), "pre-clean duplicate helper members")
        if prior["duplicate_member_count"] != len(prior_members) - len(set(prior_members)):
            raise EvidenceError("pre-clean duplicate member count differs from raw ar-t output")

    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict) or _exact_int(cycle.get("iteration"), "archive cycle index") != index:
            raise EvidenceError("helper archive cycle indexes are not exact/ordered")
        before_members = audit_before_snapshot(
            cycle.get("before_archive"), cycle.get("before_sha256"),
            cycle.get("before_member_list"), cycle.get("before_ar_list_command"),
            f"archive cycle {index}")
        before_duplicates = _exact_int(
            cycle.get("before_duplicate_member_count"),
            f"archive cycle {index} before duplicate count")
        if before_duplicates != len(before_members) - len(set(before_members)):
            raise EvidenceError(f"archive cycle {index} before duplicate count is false")
        if index == 0 and prior is not None and (
                cycle.get("before_sha256") != prior.get("sha256") or
                cycle.get("before_member_list") != prior.get("member_list")):
            raise EvidenceError("first archive-cycle before state differs from prior snapshot")
        if index == 1 and (
                cycle.get("before_sha256") != cycle_hashes[0] or
                cycle.get("before_member_list") != cycle_members[0]):
            raise EvidenceError("second archive cycle does not start from the first clean archive")

        if (_artifact(stage, cycle.get("clean_script"),
                      f"archive cycle {index} clean script")[0] != clean_script or
                cycle.get("clean_script_sha256") != clean_script_hash):
            raise EvidenceError(f"archive cycle {index} clean script differs from audited bytes")
        clean = cycle.get("clean_command")
        if not isinstance(clean, dict):
            raise EvidenceError(f"archive cycle {index} clean command is missing")
        clean_argv, clean_cwd = clean.get("argv"), clean.get("cwd")
        expected_clean_argv = [str(cmake_path), "-P",
                               str(clean_script_target.relative_to(archive_cwd))]
        if (clean_argv != expected_clean_argv or not isinstance(clean_cwd, str) or
                Path(clean_cwd).resolve() != archive_cwd.resolve() or
                clean.get("script_target_path") != str(clean_script_target.resolve()) or
                _artifact(stage, clean.get("script"), f"archive cycle {index} script")[0] != clean_script or
                clean.get("script_sha256") != clean_script_hash):
            raise EvidenceError(f"archive cycle {index} clean argv/script is not the actual target clean")
        clean_execution = _audit_logged_command(
            stage, clean.get("execution"), f"archive cycle {index} clean execution")
        if (clean_execution["argv"] != expected_clean_argv or
                Path(clean_execution["cwd"]).resolve() != archive_cwd.resolve()):
            raise EvidenceError(f"archive cycle {index} clean execution differs from CMake command")

        executions = cycle.get("archive_commands")
        if not isinstance(executions, list) or len(executions) != len(archive_commands):
            raise EvidenceError(f"archive cycle {index} lacks every actual ar/ranlib command")
        for command_index, (expected, actual) in enumerate(zip(archive_commands, executions)):
            if (not isinstance(actual, dict) or actual.get("kind") != expected.get("kind") or
                    actual.get("argv") != expected.get("argv") or
                    Path(actual.get("cwd", "")).resolve() != archive_cwd.resolve()):
                raise EvidenceError(f"archive cycle {index} command {command_index} differs from link.txt")
            execution = _audit_logged_command(
                stage, actual.get("execution"), f"archive cycle {index} command {command_index}")
            if (execution["argv"] != expected.get("argv") or
                    Path(execution["cwd"]).resolve() != archive_cwd.resolve()):
                raise EvidenceError(f"archive cycle {index} command execution is not link.txt argv")

        output_path, output_hash = _artifact(
            stage, cycle.get("archive"), f"archive cycle {index} immutable output snapshot")
        if (output_path == helper.resolve() or
                cycle.get("archive_target_path") != str(helper.resolve()) or
                cycle.get("archive_target_sha256") != output_hash or
                cycle.get("archive_sha256") != output_hash or
                not isinstance(cycle.get("member_list"), list) or
                len(cycle["member_list"]) != len(set(cycle["member_list"]))):
            raise EvidenceError(f"archive cycle {index} final archive snapshot/target identity is malformed")
        members = _audit_helper_archive_listing(
            stage, cycle.get("ar_list_command"), output_path, helper, ar, archive_cwd,
            f"archive cycle {index} final archive list")
        if members != cycle["member_list"] or members != sorted(expected_members):
            raise EvidenceError(f"archive cycle {index} archive members differ from CMake objects")
        _audit_helper_members(
            stage, output_path, helper, ar, archive_cwd, members,
            cycle.get("member_records"), expected_members,
            {expected_object_snapshot.name: helper_object},
            f"archive cycle {index}")
        cycle_hashes.append(output_hash)
        cycle_members.append(members)

    if (cycle_hashes[0] != cycle_hashes[1] or cycle_members[0] != cycle_members[1] or
            cycle_hashes[1] != archive_hash):
        raise EvidenceError("clean/rebuild archive snapshots and member order are not stable")
    if (build.get("archive") != cycles[-1].get("archive") or
            build.get("archive_target_path") != str(helper.resolve()) or
            build.get("target_archive_path") != str(helper.resolve()) or
            build.get("archive_target_sha256") != cycles[-1].get("archive_target_sha256") or
            build.get("member_list") != cycle_members[-1] or
            build.get("member_records") != cycles[-1].get("member_records") or
            build.get("ar_list_command") != cycles[-1].get("ar_list_command")):
        raise EvidenceError("helper top-level archive record differs from final clean/rebuild cycle")
    return {"source": str(_source_root(stage) / "lib/VulkanDeviceHelper/VulkanDeviceHelper.cpp"),
            "object": {"path": expected_object_snapshot.relative_to(stage).as_posix(),
                       "sha256": helper_object_hash},
            "object_target_path": str(helper_object.resolve()),
            "object_target_sha256": helper_object_hash,
            "archive": {"path": str(build["archive"]["path"]),
                        "sha256": archive_hash},
            "archive_target_path": str(helper.resolve()),
            "archive_target_sha256": cycles[-1]["archive_target_sha256"],
            "archive_rebuilds": cycles,
            "archive_link_script": build.get("archive_link_script"),
            "archive_tool_identity": {
                "original_path": build.get("archive_tool_identity", {}).get("path"),
                "sha256": build.get("archive_tool_identity", {}).get("sha256"),
                "artifact": build.get("archive_tool_identity", {}).get("artifact")},
            "member": helper_object.name, "member_list": cycle_members[-1],
            "prior_archive": prior,
            "archive_tool": str(Path(ar).resolve())}


def _audit_helper_archive(stage: Path, helper: Path, profile: str,
                          configured_cxx: Path, build: object) -> dict:
    if not isinstance(build, dict):
        raise EvidenceError("Vulkan helper source/archive build provenance is missing")
    source_root = _source_root(stage)
    helper_source = (source_root / "lib/VulkanDeviceHelper/VulkanDeviceHelper.cpp").resolve()
    source_copy, source_hash = _artifact(stage, build.get("source"), "Vulkan helper source snapshot")
    if source_copy.read_bytes() != helper_source.read_bytes() or source_hash != sha256_file(helper_source):
        raise EvidenceError("Vulkan helper source snapshot differs from checked-in source bytes")
    compile_argv, compile_cwd = _actual_compile_command(stage, helper_source)
    compile_record = build.get("compile")
    if (not isinstance(compile_record, dict) or compile_record.get("argv") != compile_argv or
            Path(compile_record.get("cwd", "")).resolve() != Path(compile_cwd).resolve()):
        raise EvidenceError("Vulkan helper compile record differs from actual compile database")
    compile_execution = _audit_logged_command(
        stage, compile_record.get("execution"), "Vulkan device helper compile")
    if (compile_execution["argv"] != compile_argv or
            Path(compile_execution["cwd"]).resolve() != Path(compile_cwd).resolve()):
        raise EvidenceError("Vulkan helper compile execution differs from actual CMake compile row")
    if Path(compile_argv[0]).resolve() != configured_cxx.resolve():
        raise EvidenceError("Vulkan helper archive object uses a different gate C++ compiler")
    _audit_no_forced_compiler_inputs(
        compile_argv, "Vulkan device helper compile", profile)
    if not any(token in {"-fPIC", "-fpic"} for token in compile_argv):
        raise EvidenceError("Vulkan device helper archive object is not position independent")
    if compile_argv.count("-o") != 1:
        raise EvidenceError("Vulkan helper compile has no unique object output")
    helper_object_target = _command_path(
        stage, compile_argv[compile_argv.index("-o") + 1],
        compile_cwd, "Vulkan helper object target")
    helper_object_snapshot, helper_object_hash = _artifact(
        stage, compile_record.get("object"), "Vulkan helper immutable object snapshot")
    if (compile_record.get("object_target_path") != str(helper_object_target) or
            compile_record.get("object_target_sha256") != helper_object_hash or
            Path(compile_record.get("output", "")).resolve() != helper_object_target):
        raise EvidenceError("Vulkan helper compile target differs from immutable object snapshot")
    archive_link = stage / "lib/VulkanDeviceHelper/CMakeFiles/NCNNVulkanDeviceHelper.dir/link.txt"
    if not archive_link.is_file():
        raise EvidenceError("Vulkan helper target actual archive command is missing")
    try:
        archive_lines = [shlex.split(line) for line in
                         archive_link.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise EvidenceError("Vulkan helper archive link command cannot be parsed") from error
    archive_commands = build.get("archive_commands")
    if (not archive_lines or not isinstance(archive_commands, list) or
            len(archive_lines) != len(archive_commands)):
        raise EvidenceError("Vulkan helper archive execution inventory differs from link.txt")
    archive_outputs, archive_objects, archive_tool_commands = set(), set(), []
    expected_members: dict[str, Path] = {}
    archive_cwd = archive_link.parent.parent.parent
    for index, (actual_argv, command) in enumerate(zip(archive_lines, archive_commands)):
        if (not isinstance(command, dict) or command.get("argv") != actual_argv or
                Path(command.get("cwd", "")).resolve() != archive_cwd.resolve() or
                command.get("kind") not in {"archive", "ranlib"}):
            raise EvidenceError(f"Vulkan helper archive command {index} differs from actual link.txt")
        if any(token.startswith("@") for token in actual_argv):
            raise EvidenceError("Vulkan helper archive command uses a response file")
        for token in actual_argv:
            if token.startswith("-"):
                continue
            suffix = Path(token).suffix
            if suffix == ".a":
                archive_outputs.add(_command_path(stage, token, str(archive_cwd),
                                                  "Vulkan helper archive output"))
            elif suffix == ".o":
                object_path = _command_path(stage, token, str(archive_cwd),
                                            "Vulkan helper archive object input")
                archive_objects.add(object_path)
                member_name = object_path.name
                if (object_path != helper_object_target or
                        (member_name in expected_members and
                         expected_members[member_name] != helper_object_snapshot)):
                    raise EvidenceError("CMake helper object input differs from snapshot or collides by basename")
                expected_members[member_name] = helper_object_snapshot
        if command.get("kind") == "archive":
            archive_tool_commands.append(actual_argv)
    if (archive_outputs != {helper.resolve()} or archive_objects != {helper_object_target} or
            expected_members != {helper_object_snapshot.name: helper_object_snapshot} or
            len(archive_tool_commands) != 1):
        raise EvidenceError("Vulkan helper archive commands differ from source object/archive target")
    cache = _stage_cache_values(stage)
    ar = cache.get("CMAKE_AR")
    if not isinstance(ar, str) or not Path(ar).is_file():
        raise EvidenceError("fresh CMake cache does not identify the archive tool")
    if Path(archive_tool_commands[0][0]).resolve() != Path(ar).resolve():
        raise EvidenceError("Vulkan helper archive was not created by configured CMAKE_AR")
    archive_record, archive_hash = _artifact(
        stage, build.get("archive"), "Vulkan helper immutable archive snapshot")
    if (archive_record == helper.resolve() or
            build.get("archive_target_path") != str(helper.resolve()) or
            build.get("target_archive_path") != str(helper.resolve()) or
            build.get("archive_target_sha256") != archive_hash):
        raise EvidenceError("Vulkan helper archive snapshot is not bound to actual CMake target output")
    archive_script_record, archive_script_hash = _artifact(
        stage, build.get("archive_link_script"), "Vulkan helper actual link.txt snapshot")
    if (archive_script_hash != build.get("archive_link_script_sha256") or
            build.get("archive_link_script_target_path") != str(archive_link.resolve()) or
            archive_script_record.read_bytes() != archive_link.read_bytes()):
        raise EvidenceError("Vulkan helper archive link script snapshot differs from actual link.txt")
    archive_tool_identity = build.get("archive_tool_identity")
    if (not isinstance(archive_tool_identity, dict) or
            Path(archive_tool_identity.get("path", "")).resolve() != Path(ar).resolve() or
            archive_tool_identity.get("sha256") != sha256_file(Path(ar)) or
            _artifact(stage, archive_tool_identity.get("artifact"),
                      "CMake archive tool binary")[1] != archive_tool_identity.get("sha256")):
        raise EvidenceError("Vulkan helper CMAKE_AR binary identity is not captured")
    clean_script, clean_script_hash = _artifact(
        stage, build.get("clean_script"), "Vulkan helper target clean script")
    clean_script_target = (archive_link.parent / "cmake_clean_target.cmake").resolve()
    if (clean_script_hash != build.get("clean_script_sha256") or
            not clean_script_target.is_file() or
            clean_script_target.read_bytes() != clean_script.read_bytes()):
        raise EvidenceError("Vulkan helper clean script snapshot/target bytes differ")
    configured_cmake = cache.get("CMAKE_COMMAND")
    if not isinstance(configured_cmake, str) or not Path(configured_cmake).is_file():
        raise EvidenceError("fresh CMake cache does not identify its configured CMake executable")
    return _audit_helper_archive_rebuilds(
        stage, helper, str(Path(ar).resolve()), archive_cwd,
        Path(configured_cmake), clean_script_target, clean_script,
        clean_script_hash,
        helper_object_target, helper_object_hash, expected_members,
        archive_commands, build, archive_hash)


def _require_profile_instrumentation(argv: list[str], profile: str, label: str,
                                     *, link: bool = False) -> None:
    sanitizer_flags = {"-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                       "-fno-omit-frame-pointer"}
    if any(token.startswith("@") or re.search(r"(?:^|[,=])@[^,]+", token)
           for token in argv):
        raise EvidenceError(f"{label} uses an unaudited response file")
    suppressions = ("-fno-sanitize=", "-fsanitize-recover=",
                    "-fsanitize-blacklist=", "-fsanitize-ignorelist=",
                    "-fsanitize-coverage-ignorelist=", "-fno-sanitize-coverage=")
    if any(token.startswith(suppressions) for token in argv):
        raise EvidenceError(f"{label} disables or suppresses sanitizer instrumentation")
    if profile == "sanitize":
        if not sanitizer_flags <= set(argv):
            raise EvidenceError(f"{label} lacks exact ASan+UBSan/no-recover/frame-pointer flags")
        if link and ("-shared-libasan" not in argv or "-static-libasan" in argv):
            raise EvidenceError(f"{label} lacks shared-only ASan link mode")
    elif any(token.startswith(("-fsanitize=", "-fno-sanitize", "-shared-libasan",
                              "-static-libasan")) for token in argv):
        raise EvidenceError(f"{label} unexpectedly uses sanitizer flags")


def _audit_no_forced_compiler_inputs(argv: list[str], label: str,
                                     profile: str, *, link: bool = False) -> None:
    forced_include_flags = ("-include", "-include-pch", "-imacros", "-Wp,-include",
                            "-fplugin", "-fplugin-arg-", "-load", "-plugin")
    if any(token.startswith(forced_include_flags) or
           "NCNN_VULKAN_LIFECYCLE_TEST" in token or
           token == "-Xclang" and index + 1 < len(argv) and
           argv[index + 1] in {"-load", "-plugin", "-include", "-include-pch", "-imacros"}
           for index, token in enumerate(argv)):
        raise EvidenceError(f"{label} uses a forced include or compiler plugin")
    _require_profile_instrumentation(argv, profile, label, link=link)


def _audit_native_loader_phase_implementation(source: str) -> None:
    concurrent_marker = '} else if (arguments.mode == "concurrent") {'
    interleaved_marker = "    } else {\n      invokeVulkan(run, context_a.get(), prepared_a, trace);"
    concurrent_start = source.find(concurrent_marker)
    interleaved_start = source.find(interleaved_marker,
                                    concurrent_start + len(concurrent_marker))
    if concurrent_start < 0 or interleaved_start < 0:
        raise EvidenceError("checked-in native loader lacks concurrent/interleaved phase branches")
    concurrent_body = source[concurrent_start:interleaved_start]
    if ("std::latch ready(2)" not in concurrent_body or
            "std::vector<std::jthread> workers" not in concurrent_body or
            concurrent_body.count("workers.emplace_back(") != 2 or
            "invokeVulkan(run, context_a.get(), invocation, trace)" not in concurrent_body or
            "std::ref(prepared_a)" not in concurrent_body or
            "std::ref(*prepared_b)" not in concurrent_body):
        raise EvidenceError("native same-context concurrent phase is not two real worker threads")
    interleaved_end = source.find("    }\n    const bool failed", interleaved_start)
    if interleaved_end < 0:
        raise EvidenceError("native loader interleaved phase body is incomplete")
    interleaved_body = source[interleaved_start:interleaved_end]
    calls = re.findall(
        r"invokeVulkan\(run,\s*(context_[ab]\.get\(\)),\s*\*?"
        r"(prepared_[A-Za-z0-9_]+),\s*trace\)", interleaved_body)
    if calls != [("context_a.get()", "prepared_a"),
                 ("context_b.get()", "prepared_b"),
                 ("context_a.get()", "prepared_a_repeat"),
                 ("context_b.get()", "prepared_b_repeat")]:
        raise EvidenceError("native two-context phase is not A-B-A-B interleaved")


def _audit_fixedrun_structure(source_text: str, body: str,
                              abi_body: str = "") -> None:
    if FORBIDDEN_RUN.search(source_text):
        raise EvidenceError("generated source contains a generic executor/runtime/hazard structure")
    if any(re.search(r"\b(?:for|while|do)\s*(?:\(|\{)", section)
           for section in (body, abi_body)):
        raise EvidenceError("generated fixed-run or C ABI wrapper contains runtime iteration")
    if re.search(r"constexpr\s+(?:std::)?array\s*<[^;{}]*(?:Kernel|Schedule|Dispatch)",
                  source_text, re.I):
        raise EvidenceError("generated source contains a static kernel/schedule dispatch table")
    allowed_calls = {
        "if", "catch", "report", "require", "checkValidation", "mappedBuffer",
        "memcpy", "get_command_buffer", "get_handle", "get_error", "lock",
        "wait", "reset", "begin", "end", "submit_and_wait", "flush",
        "invalidate", "get_pipeline", "get_layout", "get_descriptor_sets",
        "data", "vkCmdPipelineBarrier",
        "vkCmdCopyBuffer", "vkCmdBindPipeline", "vkCmdBindDescriptorSets",
        "vkCmdPushConstants", "vkCmdDispatch", "UINT64_C", "UINT8_C",
        "static_cast", "scoped_lock", "what",
    }
    abi_calls = allowed_calls | {"validateDiagnostic", "run", "span"}
    calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", body))
    unexpected = sorted(calls - allowed_calls)
    if unexpected:
        raise EvidenceError(f"generated fixed-run contains untyped helper calls: {unexpected}")
    wrapper_calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", abi_body))
    unexpected_wrapper = sorted(wrapper_calls - abi_calls)
    if unexpected_wrapper:
        raise EvidenceError(
            f"generated C ABI wrapper contains untyped helper calls: {unexpected_wrapper}")
    if abi_body and abi_body.count("->run(") != 1:
        raise EvidenceError("model C ABI wrapper does not invoke its one fixed context run")


def _audit_generated_static_tables(source: str, body: str, abi_body: str,
                                   program: dict, shader_records: list[dict],
                                   stage: Path) -> None:
    arrays = re.findall(
        r"(?:\[\[maybe_unused\]\]\s*)?constexpr\s+std::array\s*<\s*([^,>]+)\s*,\s*(\d+)\s*>\s*"
        r"([A-Za-z_]\w*)\s*\{\{([\s\S]*?)\}\};", source)
    allowed_names = {"kDeviceRequirements", "kInputs", "kOutputs", "push_constants"}
    pipelines = program.get("pipelines")
    weights = program.get("weights")
    if not isinstance(pipelines, list) or not isinstance(weights, list):
        raise EvidenceError("HostProgram static table declarations lack typed inputs")
    allowed_names.update(f"kShaderWords{index}" for index in range(len(pipelines)))
    allowed_names.update(f"kEntryPoint{index}" for index in range(len(pipelines)))
    allowed_names.update(f"kWeightBytes{index}" for index in range(len(weights)))
    observed = {}
    for element_type, count_text, name, initializer in arrays:
        if name not in allowed_names or name in observed:
            raise EvidenceError(
                f"generated source contains an untyped or duplicate static table {name}")
        observed[name] = (element_type.strip(), int(count_text), initializer)
    if {name for name in observed if name != "push_constants"} != allowed_names - {"push_constants"}:
        raise EvidenceError("generated source static array inventory differs from typed HostProgram")

    raw_static = re.search(
        r"\b(?:static\s+|constexpr\s+|const\s+)[\w:]+\s+[A-Za-z_]\w*\s*\[[^\]]*\]",
        source)
    if raw_static:
        raise EvidenceError("generated source contains an unreviewed static C-style array")
    auto_static = re.search(
        r"\b(?:static\s+)?constexpr\s+auto\s+[A-Za-z_]\w*\s*=\s*std::array\s*<", source)
    if auto_static:
        raise EvidenceError("generated source contains an inferred static array declaration")
    if re.search(r"\bstatic\s+(?:(?:const|constexpr)\s+)?std::array\s*<", source):
        raise EvidenceError("generated source contains a runtime static std::array table")
    body_arrays = re.findall(
        r"(?:(?:alignas\([^)]*\))\s*)?(?:(?:const|constexpr|static)\s+)*"
        r"std::array\s*<\s*([^,]+),\s*\d+\s*>\s*([A-Za-z_]\w*)", body)
    permitted_body_arrays = {
        ("VkBufferMemoryBarrier", "barriers"),
        ("std::span<std::byte>", "readbacks"),
        ("std::uint8_t", "push_constants"),
    }
    if any((element_type.strip(), name) not in permitted_body_arrays
           for element_type, name in body_arrays):
        raise EvidenceError("fixed-run body declares an untyped local array/table")
    wrapper_arrays = re.findall(
        r"(?:(?:const|constexpr|static)\s+)*std::array\s*<\s*([^,]+),\s*(\d+)\s*>\s*"
        r"([A-Za-z_]\w*)", abi_body)
    input_ports, output_ports = program.get("inputs"), program.get("outputs")
    if not isinstance(input_ports, list) or not isinstance(output_ports, list):
        raise EvidenceError("HostProgram public port arrays are malformed")
    expected_wrapper_arrays = {
        ("std::span<const float>", len(input_ports), "inputs"),
        ("std::span<float>", len(output_ports), "outputs"),
    }
    if {(kind.strip(), int(count), name) for kind, count, name in wrapper_arrays} != expected_wrapper_arrays:
        raise EvidenceError("C ABI wrapper span arrays differ from HostProgram public I/O")
    for name, element_type, count, parameter_prefix in (
            ("inputs", "const float", len(input_ports), "input"),
            ("outputs", "float", len(output_ports), "output")):
        declaration = re.search(
            rf"std::array\s*<\s*std::span<{re.escape(element_type)}>\s*,\s*{count}\s*>\s*"
            rf"{name}\s*\{{\{{([\s\S]*?)\}}\}};", abi_body)
        if declaration is None:
            raise EvidenceError(f"C ABI wrapper {name} span array initializer is missing")
        spans = re.findall(
            rf"std::span<{re.escape(element_type)}>\s*\(\s*"
            rf"({parameter_prefix}\d+)\s*,\s*({parameter_prefix}\d+_elements)\s*\)",
            declaration.group(1))
        expected_spans = [(f"{parameter_prefix}{index + 1}",
                           f"{parameter_prefix}{index + 1}_elements")
                          for index in range(count)]
        if spans != expected_spans:
            raise EvidenceError(f"C ABI wrapper {name} spans differ from declared public pointer arguments")
    outside_body = source.replace(body, "", 1).replace(abi_body, "", 1)
    initialized_arrays = re.findall(
        r"(?:(?:static|const|constexpr)\s+)*std::array\s*<\s*([^,]+),\s*(\d+)\s*>\s*"
        r"([A-Za-z_]\w*)\s*(?:\{\{|=)", outside_body)
    initialization_arrays = [row for row in initialized_arrays if row[2] not in observed]
    expected_initialization_arrays = []
    for pipeline in pipelines:
        bindings = pipeline.get("bindings")
        if not isinstance(bindings, list):
            raise EvidenceError("HostProgram pipeline binding table is malformed")
        expected_initialization_arrays.append(
            ("VulkanPipelineBinding", str(len(bindings)), "bindings"))
        push_end = max((item.get("offset", 0) + len(bytes.fromhex(item.get("bytes", "")))
                        for item in pipeline.get("push_constants", [])), default=0)
        expected_initialization_arrays.append(
            ("VulkanPushConstantRange", str(1 if push_end else 0), "push_constants"))
    create_barriers = [item for item in program.get("create_operations", [])
                       if isinstance(item, dict) and item.get("op") == "barrier"]
    expected_initialization_arrays.extend(
        ("VkBufferMemoryBarrier", str(len(item.get("buffers", []))), "barriers")
        for item in create_barriers)
    if initialization_arrays != expected_initialization_arrays:
        raise EvidenceError("generated initialization array inventory differs from typed pipelines")

    binding_arrays = re.findall(
        r"const\s+std::array\s*<\s*VulkanPipelineBinding\s*,\s*(\d+)\s*>\s*bindings\s*"
        r"\{\{([\s\S]*?)\}\};", outside_body)
    push_arrays = re.findall(
        r"const\s+std::array\s*<\s*VulkanPushConstantRange\s*,\s*(\d+)\s*>\s*push_constants\s*"
        r"\{\{([\s\S]*?)\}\};", outside_body)
    if len(binding_arrays) != len(pipelines) or len(push_arrays) != len(pipelines):
        raise EvidenceError("generated pipeline initialization arrays are incomplete")
    buffer_by_id = {item.get("id"): item for item in program.get("buffers", [])
                    if isinstance(item, dict)}
    for index, pipeline in enumerate(pipelines):
        binding_count, binding_payload = binding_arrays[index]
        expected_bindings = pipeline["bindings"]
        if int(binding_count) != len(expected_bindings):
            raise EvidenceError("generated pipeline binding count differs from HostProgram")
        actual_bindings = re.findall(
            r"VulkanPipelineBinding\s*\{\s*\.set\s*=\s*(\d+)\s*,\s*"
            r"\.binding\s*=\s*(\d+)\s*,\s*\.buffer\s*=\s*buffers_\[(\d+)\]->get_handle\(\)\s*,\s*"
            r"\.buffer_size\s*=\s*UINT64_C\((\d+)\)\s*,\s*\.offset\s*=\s*(\d+)\s*,\s*"
            r"\.range\s*=\s*UINT64_C\((\d+)\)\s*\}", binding_payload)
        expected_binding_rows = []
        for binding in expected_bindings:
            buffer_id = binding.get("buffer")
            buffer = buffer_by_id.get(buffer_id)
            if not isinstance(buffer, dict) or type(buffer.get("byte_size")) is not int:
                raise EvidenceError("HostProgram binding buffer size is malformed")
            expected_binding_rows.append((str(binding.get("set")), str(binding.get("binding")),
                                          str(buffer_id), str(buffer["byte_size"]), "0",
                                          str(buffer["byte_size"])))
        if [tuple(row) for row in actual_bindings] != expected_binding_rows:
            raise EvidenceError("generated pipeline binding payload differs from HostProgram")

        push_count, push_payload = push_arrays[index]
        push_end = max((item.get("offset", 0) + len(bytes.fromhex(item.get("bytes", "")))
                        for item in pipeline.get("push_constants", [])), default=0)
        expected_push_count = 1 if push_end else 0
        if int(push_count) != expected_push_count:
            raise EvidenceError("generated push-constant range count differs from HostProgram")
        actual_pushes = re.findall(
            r"VulkanPushConstantRange\s*\{\s*\.stage_flags\s*=\s*VK_SHADER_STAGE_COMPUTE_BIT\s*,\s*"
            r"\.offset\s*=\s*(\d+)\s*,\s*\.size\s*=\s*(\d+)\s*\}", push_payload)
        expected_pushes = [("0", str(push_end))] if push_end else []
        if [tuple(row) for row in actual_pushes] != expected_pushes:
            raise EvidenceError("generated push-constant range differs from typed HostProgram")

    barrier_arrays = re.findall(
        r"const\s+std::array\s*<\s*VkBufferMemoryBarrier\s*,\s*(\d+)\s*>\s*barriers\s*"
        r"\{\{([\s\S]*?)\}\};", outside_body)
    barrier_calls = re.findall(
        r"vkCmdPipelineBarrier\(commands_->get_command_buffer\(\),\s*(VK_PIPELINE_STAGE_[A-Z_]+),\s*"
        r"(VK_PIPELINE_STAGE_[A-Z_]+),\s*0,\s*0,\s*nullptr,\s*(\d+),\s*barriers\.data\(\),"
        r"\s*0,\s*nullptr\)", outside_body)
    if len(barrier_arrays) != len(create_barriers) or len(barrier_calls) != len(create_barriers):
        raise EvidenceError("generated create-operation barrier arrays/calls differ from HostProgram")
    buffer_slots = {item.get("id"): slot for slot, item in enumerate(program.get("buffers", []))
                    if isinstance(item, dict)}
    stage_tokens = {"host": "VK_PIPELINE_STAGE_HOST_BIT",
                    "transfer": "VK_PIPELINE_STAGE_TRANSFER_BIT",
                    "compute": "VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"}
    access_tokens = {"host-write": "VK_ACCESS_HOST_WRITE_BIT",
                     "transfer-read": "VK_ACCESS_TRANSFER_READ_BIT",
                     "transfer-write": "VK_ACCESS_TRANSFER_WRITE_BIT",
                     "shader-read": "VK_ACCESS_SHADER_READ_BIT",
                     "shader-write": "VK_ACCESS_SHADER_WRITE_BIT",
                     "shader-read-write": "VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT",
                     "host-read": "VK_ACCESS_HOST_READ_BIT"}
    for index, (operation, declaration, call) in enumerate(
            zip(create_barriers, barrier_arrays, barrier_calls)):
        count_text, payload = declaration
        entries = operation.get("buffers")
        if not isinstance(entries, list) or int(count_text) != len(entries):
            raise EvidenceError(f"create barrier {index} buffer count differs from HostProgram")
        rows = re.findall(
            r"VkBufferMemoryBarrier\s*\{[^{}]*\.srcAccessMask\s*=\s*([^,]+),\s*"
            r"\.dstAccessMask\s*=\s*([^,]+),[^{}]*\.buffer\s*=\s*buffers_\[(\d+)\]->get_handle\(\),\s*"
            r"\.offset\s*=\s*UINT64_C\((\d+)\),\s*\.size\s*=\s*UINT64_C\((\d+)\)",
            payload)
        expected_rows = []
        for entry in entries:
            slot = buffer_slots.get(entry.get("buffer"))
            source_access = access_tokens.get(entry.get("source_access"))
            destination_access = access_tokens.get(entry.get("destination_access"))
            if slot is None or source_access is None or destination_access is None:
                raise EvidenceError(f"create barrier {index} references an unknown buffer/access")
            expected_rows.append((source_access, destination_access, str(slot),
                                  str(entry.get("offset")), str(entry.get("byte_size"))))
        if [tuple(row) for row in rows] != expected_rows:
            raise EvidenceError(f"create barrier {index} source payload differs from typed buffers")
        actual_source, actual_destination, actual_count = call
        if (actual_source != stage_tokens.get(operation.get("source_stage")) or
                actual_destination != stage_tokens.get(operation.get("destination_stage")) or
                actual_count != str(len(entries))):
            raise EvidenceError(f"create barrier {index} source/destination stages differ from HostProgram")

    def check_array(name: str, element_type: str, expected: list[int]) -> None:
        actual_type, count, initializer = observed[name]
        if actual_type != element_type or count != len(expected):
            raise EvidenceError(f"generated static table {name} differs from its typed inventory")
        values = [int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]+)", initializer)]
        if name.startswith("kEntryPoint") and re.search(r"'\\0'", initializer):
            values.append(0)
        if values != expected:
            raise EvidenceError(f"generated static table {name} payload differs from typed identity")

    for index, (pipeline, shader_record) in enumerate(zip(pipelines, shader_records)):
        shader_path, _ = _artifact(stage, shader_record, f"static shader table {index}")
        shader = shader_path.read_bytes()
        if len(shader) % 4:
            raise EvidenceError("shader payload is not a whole number of SPIR-V words")
        words = list(struct.unpack(f"<{len(shader) // 4}I", shader))
        check_array(f"kShaderWords{index}", "std::uint32_t", words)
        entry_point = pipeline.get("interface", {}).get("entry_point")
        if not isinstance(entry_point, str):
            raise EvidenceError("HostProgram static entry-point table identity is malformed")
        check_array(f"kEntryPoint{index}", "char",
                    list(entry_point.encode("utf-8")) + [0])

    for index, weight in enumerate(weights):
        source_type, count, initializer = observed[f"kWeightBytes{index}"]
        values = [int(value, 16) for value in
                  re.findall(r"std::byte\s*\{\s*0x([0-9a-fA-F]+)\s*\}", initializer)]
        if (source_type != "std::byte" or count != len(values) or
                hashlib.sha256(bytes(values)).hexdigest() != weight.get("physical_sha256") or
                len(values) != weight.get("byte_size")):
            raise EvidenceError(f"generated static weight table {index} differs from HostProgram bytes")

    for name, entries in (("kInputs", program.get("inputs")),
                          ("kOutputs", program.get("outputs"))):
        if not isinstance(entries, list):
            raise EvidenceError(f"generated {name} port inventory is malformed")
        element_type, count, initializer = observed[name]
        values = re.findall(
            r"Port\s*\{\s*\.device_buffer\s*=\s*(\d+)\s*,\s*"
            r"\.staging_buffer\s*=\s*(\d+)\s*,\s*\.elements\s*=\s*UINT64_C\((\d+)\)\s*\}",
            initializer)
        expected = [(str(item.get("device_buffer")), str(item.get("staging_buffer")),
                     str(item.get("elements"))) for item in entries if isinstance(item, dict)]
        if (element_type != "Port" or count != len(entries) or
                [tuple(row) for row in values] != expected or len(expected) != len(entries)):
            raise EvidenceError(f"generated static port table {name} differs from HostProgram")

    requirement = observed["kDeviceRequirements"]
    pipelines = program["pipelines"]
    buffers = program.get("buffers", [])
    device_sizes = [item.get("byte_size") for item in buffers
                    if isinstance(item, dict) and item.get("kind") == "device"]
    storage_range = max(device_sizes, default=0)
    expected_rows = []
    for pipeline in pipelines:
        interface = pipeline.get("interface", {})
        local_size = interface.get("local_size", [])
        push_end = max((item.get("offset", 0) + len(bytes.fromhex(item.get("bytes", "")))
                        for item in pipeline.get("push_constants", [])), default=0)
        sets = max((item.get("set", -1) + 1 for item in pipeline.get("bindings", [])),
                   default=0)
        expected_rows.append((local_size[0], pipeline.get("dispatch", [0])[0],
                              storage_range, push_end, sets,
                              len(pipeline.get("bindings", []))))
    if (requirement[0] != "VulkanDeviceLimitRequirements" or
            requirement[1] != len(expected_rows)):
        raise EvidenceError("generated device-limit table differs from HostProgram pipeline count")
    rows = re.findall(
        r"VulkanDeviceLimitRequirements\s*\{\s*\.local_size_x\s*=\s*(\d+)\s*,\s*"
        r"\.dispatch_count_x\s*=\s*(\d+)\s*,\s*"
        r"\.storage_buffer_range\s*=\s*UINT64_C\((\d+)\)\s*,\s*"
        r"\.push_constant_bytes\s*=\s*(\d+)\s*,\s*"
        r"\.descriptor_set_count\s*=\s*(\d+)\s*,\s*"
        r"\.storage_buffer_descriptor_count\s*=\s*(\d+)\s*\}",
        requirement[2])
    if [tuple(map(int, row)) for row in rows] != expected_rows:
        raise EvidenceError("generated device-limit table values differ from typed pipelines")


def _tosa_tensor_shape(value: str) -> tuple[int, ...]:
    fields = value.split("x")
    if len(fields) < 2 or not fields[-1]:
        raise EvidenceError("reshape TOSA tensor type is malformed")
    try:
        dimensions = tuple(int(item) for item in fields[:-1])
    except ValueError as error:
        raise EvidenceError("reshape TOSA tensor type is not static") from error
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise EvidenceError("reshape TOSA tensor dimensions must be positive/static")
    return dimensions


def _audit_reshape_tosa(path: Path, metadata: dict, candidate: dict) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="strict")
    digest = sha256_file(path)
    if (candidate.get("tosa_ir_sha256") != digest or
            metadata.get("tosa_ir_sha256") != digest or
            metadata.get("tosa_ir") != "candidate_tosa.mlir"):
        raise EvidenceError("reshape TOSA bytes are not bound to capture metadata")
    signatures = []
    pattern = re.compile(
        r"\btosa\.reshape\b[^\n]*?:\s*\(tensor<([^>]+)>,\s*"
        r"!tosa\.shape<(\d+)>\)\s*->\s*tensor<([^>]+)>")
    for line_number, line in enumerate(text.splitlines(), 1):
        if "tosa.reshape" not in line:
            continue
        match = pattern.search(line)
        if match is None:
            raise EvidenceError("reshape TOSA signature is not a typed MLIR 21 reshape")
        source_shape = _tosa_tensor_shape(match.group(1))
        result_shape = _tosa_tensor_shape(match.group(3))
        source_elements = math.prod(source_shape)
        result_elements = math.prod(result_shape)
        if (source_elements != result_elements or source_elements != 24 or
                int(match.group(2)) != len(result_shape)):
            raise EvidenceError("reshape TOSA changes element count or differs from fixture")
        signatures.append({"line": line_number, "text": line.strip(),
                           "input_type": match.group(1),
                           "output_type": match.group(3),
                           "input_rank": len(source_shape),
                           "output_rank": len(result_shape)})
    rank_pairs = {(row["input_rank"], row["output_rank"])
                  for row in signatures}
    if ((3, 4) not in rank_pairs or (4, 3) not in rank_pairs or len(signatures) < 2):
        raise EvidenceError("reshape TOSA lacks the captured rank-3/rank-4 round-trip operations")
    declared = candidate.get("reshape_views")
    if declared != signatures:
        raise EvidenceError("producer reshape view summary differs from actual TOSA IR bytes")
    return signatures


def _audit_elf_tool_invocation(stage: Path, command: object, output_record: object,
                               expected_binary: Path, expected_tool: str,
                               expected_options: list[str], label: str) -> tuple[Path, str, dict]:
    if not isinstance(command, dict):
        raise EvidenceError(f"{label} actual tool invocation is missing")
    argv, cwd = command.get("argv"), command.get("cwd")
    if (not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv) or
            not isinstance(cwd, str) or
            Path(cwd).resolve() != (stage / "test/Vulkan").resolve() or
            type(command.get("returncode")) is not int or command["returncode"] != 0 or
            Path(argv[0]).resolve() != Path(expected_tool).resolve() or
            argv[1:-1] != expected_options or Path(argv[-1]).resolve() != expected_binary.resolve()):
        raise EvidenceError(f"{label} command does not audit actual candidate DSO bytes")
    output_path, output_hash = _artifact(stage, output_record, f"{label} raw output")
    command_output, command_hash = _artifact(
        stage, command.get("stdout"), f"{label} captured stdout artifact")
    if command_output != output_path or command_hash != output_hash:
        raise EvidenceError(f"{label} output artifact differs from its executed command")
    log_path, log_hash = _artifact(stage, command.get("log"), f"{label} stdout/stderr log")
    log_bytes = log_path.read_bytes()
    stdout, separator, stderr = log_bytes.partition(b"\n--- stderr ---\n")
    if (not separator or stdout != output_path.read_bytes() or
            hashlib.sha256(stdout).hexdigest() != command.get("stdout_sha256") or
            hashlib.sha256(stderr).hexdigest() != command.get("stderr_sha256")):
        raise EvidenceError(f"{label} actual output differs from captured command streams")
    return output_path, output_hash, {
        "argv": argv, "cwd": cwd, "returncode": command["returncode"],
        "stdout": {"path": command_output.relative_to(stage).as_posix(),
                   "sha256": command_hash},
        "stdout_sha256": command["stdout_sha256"],
        "stderr_sha256": command["stderr_sha256"],
        "log": {"path": log_path.relative_to(stage).as_posix(),
                "sha256": log_hash},
    }


def _audit_host_program_against_plan(stage: Path, plan: dict, program: dict,
                                     shader_records: list[dict],
                                     model: str, plan_hash: str) -> None:
    pipelines = program.get("pipelines")
    if not isinstance(pipelines, list) or len(pipelines) != len(shader_records):
        raise EvidenceError("typed HostProgram pipelines differ from shader inventory")
    manifest_shaders = []
    shader_bytes = {}
    for index, (pipeline, record) in enumerate(zip(pipelines, shader_records)):
        if not isinstance(pipeline, dict):
            raise EvidenceError("typed HostProgram pipeline is malformed")
        shader_path, shader_hash = _artifact(
            stage, record, f"HostProgram shader payload {index}")
        kernel_id = pipeline.get("kernel_id")
        interface = pipeline.get("interface")
        if type(kernel_id) is not int or not isinstance(interface, dict):
            raise EvidenceError("typed HostProgram pipeline identity/interface is malformed")
        manifest_shaders.append({
            "kernel_id": kernel_id,
            "binary_symbol": pipeline.get("binary_symbol"),
            "entry_point": interface.get("entry_point"),
            "sha256": shader_hash,
            "size_bytes": shader_path.stat().st_size,
        })
        shader_bytes[kernel_id] = shader_path.read_bytes()
    weights = program.get("weights")
    if not isinstance(weights, list):
        raise EvidenceError("typed HostProgram weight inventory is malformed")
    manifest_weights = []
    for weight in weights:
        if not isinstance(weight, dict):
            raise EvidenceError("typed HostProgram weight entry is malformed")
        manifest_weights.append({
            "resource_id": weight.get("resource_id"),
            "sha256": weight.get("physical_sha256"),
            "size_bytes": weight.get("byte_size"),
        })
    manifest = {
        "schema_version": 1,
        "kind": "ncnn.vulkan_model_manifest",
        "model": model,
        "abi_version": 1,
        "host_program": program,
        "host_program_identity": program.get("identity"),
        "host_program_usage": "compiler-only-diagnostic",
        "host_implementation": "model-fixed-run",
        "host_codegen_version": 2,
        "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
        "host_codegen_identity": program.get("identity"),
        "execution_plan_usage": "build-evidence-only",
        "plan_sha256": plan_hash,
        "shaders": manifest_shaders,
        "weights": manifest_weights,
    }
    try:
        host_program_audit.validate_host_program(
            manifest, plan, shader_bytes=shader_bytes)
    except (host_program_audit.EvidenceError, TypeError, ValueError, KeyError) as error:
        raise EvidenceError(
            f"typed HostProgram differs from plan/read-write hazard semantics: {error}") from error


def _is_shared_object_input(value: str) -> bool:
    return re.search(r"\.so(?:\.[0-9]+)*$", Path(value).name) is not None


def _candidate_link_inputs(stage: Path, argv: list[str], cwd: str,
                           output_path: Path,
                           vulkan_library: Path) -> tuple[set[Path], set[Path], set[Path]]:
    if argv.count("-o") != 1 or argv.index("-o") + 1 >= len(argv):
        raise EvidenceError("candidate link must have one exact -o output")
    out_index = argv.index("-o") + 1
    if _command_path(stage, argv[out_index], cwd, "candidate DSO output") != output_path.resolve():
        raise EvidenceError("candidate link -o differs from its target binary")
    objects, archives, shared = set(), set(), set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-o":
            index += 2
            continue
        index += 1
        if token.startswith("-"):
            continue
        if Path(token).suffix == ".o":
            objects.add(_command_path(stage, token, cwd, "candidate linker object"))
        elif Path(token).suffix == ".a":
            archives.add(_command_path(stage, token, cwd, "candidate linker archive"))
        elif _is_shared_object_input(token):
            path = Path(token)
            path = (Path(cwd) / path).resolve() if not path.is_absolute() else path.resolve()
            if path != vulkan_library.resolve():
                raise EvidenceError(f"candidate link names a non-Vulkan/private shared object: {path}")
            shared.add(path)
    if shared != {vulkan_library.resolve()}:
        raise EvidenceError("candidate link does not uniquely bind the configured Vulkan loader DSO")
    return objects, archives, shared


def _audit_vulkan_library(stage: Path, link: dict, argv: list[str],
                          cwd: str, cache_value: str) -> dict:
    target = Path(cache_value)
    if not target.is_absolute():
        target = stage / target
    expected = target.resolve()
    snapshot, digest = _artifact(stage, link.get("vulkan_library"),
                                  "linked system Vulkan loader snapshot")
    if (snapshot == expected or
            link.get("vulkan_library_cache_key") != "Vulkan_LIBRARY" or
            link.get("vulkan_library_target_path") != str(target) or
            link.get("vulkan_library_resolved_path") != str(expected) or
            link.get("vulkan_library_target_sha256") != digest):
        raise EvidenceError("system Vulkan loader snapshot does not bind CMakeCache input")
    output = argv.index("-o") + 1 if argv.count("-o") == 1 else -1
    observed = []
    for index, token in enumerate(argv):
        if index == output or token.startswith("-") or not _is_shared_object_input(token):
            continue
        path = Path(token)
        resolved = ((Path(cwd) / path) if not path.is_absolute() else path).resolve()
        if resolved != expected:
            raise EvidenceError("candidate link names an unapproved shared library")
        observed.append({"index": index, "spelling": token,
                         "resolved_path": str(resolved)})
    if not observed or link.get("vulkan_library_link_arguments") != observed:
        raise EvidenceError("system Vulkan loader argument identities differ from actual link argv")
    return {"artifact": {"path": snapshot.relative_to(stage).as_posix(),
                          "sha256": digest},
            "target_path": str(target), "resolved_path": str(expected),
            "link_arguments": observed}


def _audit_candidate_dynamic_dependencies(needed: str) -> None:
    libraries = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", needed)
    allowed = re.compile(
        r"^lib(?:vulkan|stdc\+\+|gcc_s|c|m|dl|pthread|rt|resolv|util|atomic)\.so(?:\.[0-9]+)*$|"
        r"^libclang_rt\.(?:asan|ubsan)[A-Za-z0-9_.-]*\.so(?:\.[0-9]+)*$", re.I)
    unexpected = sorted(item for item in libraries if not allowed.fullmatch(item))
    if unexpected:
        raise EvidenceError(
            f"candidate ELF depends on non-system/private/compiler DSOs: {unexpected}")


def _native_target_link_file(stage: Path, target: str) -> Path:
    # Product-only registration tests create nested CMake projects with the same
    # target names. Only this stage's declared native target is the model input;
    # a recursive lookup conflates separate build/provenance namespaces.
    path = stage / "test/Vulkan/native/CMakeFiles" / f"{target}.dir/link.txt"
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"native target lacks its stage-local link command: {target}")
    return path


def _audit_candidate(stage: Path, candidate: object, fixture: dict,
                     profile: str, source_param: Path) -> dict:
    if not isinstance(candidate, dict):
        raise EvidenceError("candidate build record is missing")
    source, source_hash = _artifact(stage, candidate.get("source"), "generated source")
    header, header_hash = _artifact(stage, candidate.get("header"), "generated header")
    metadata_path, metadata_hash = _artifact(stage, candidate.get("metadata"), "candidate metadata")
    plan_path, plan_hash = _artifact(stage, candidate.get("plan"), "diagnostic plan")
    program_path, program_hash = _artifact(stage, candidate.get("host_program"), "typed host program")
    metadata, plan, program = (json.loads(metadata_path.read_text(encoding="utf-8")),
                               json.loads(plan_path.read_text(encoding="utf-8")),
                               json.loads(program_path.read_text(encoding="utf-8")))
    if not all(isinstance(value, dict) for value in (metadata, plan, program)):
        raise EvidenceError("candidate metadata/plan/program is not a JSON object")
    fixture_name = fixture.get("name")
    if candidate.get("fixture") != fixture_name or candidate.get("model") != fixture_name:
        raise EvidenceError("candidate build identity differs from fixture key")
    tosa_path, tosa_hash = _artifact(stage, candidate.get("tosa_ir"),
                                     "captured TOSA diagnostic IR")
    if metadata.get("tosa_ir_sha256") != tosa_hash or candidate.get("tosa_ir_sha256") != tosa_hash:
        raise EvidenceError("captured TOSA IR hash differs from candidate metadata")
    if fixture_name == "reshape":
        reshape_views = _audit_reshape_tosa(tosa_path, metadata, candidate)
    else:
        reshape_views = candidate.get("reshape_views")
        if reshape_views != []:
            raise EvidenceError("non-reshape fixture reports unexpected reshape view metadata")
    model, host_id = metadata.get("model"), program.get("identity")
    version = "ncnn-vulkan-fixed-run-codegen/v1"
    if (not isinstance(model, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", model) or
            not isinstance(host_id, str) or not SHA256_RE.fullmatch(host_id) or
            metadata.get("host_program_identity") != host_id or metadata.get("codegen_version") != version):
        raise EvidenceError("candidate host/codegen identity is invalid")
    encoded = bytearray()
    for tag, value in (("domain", "ncnn-vulkan-host-source-identity/v1"),
                       ("execution-mode", "fixed-run"), ("codegen-version", version),
                       ("host-program", host_id)):
        for field in (tag.encode("ascii"), value.encode("ascii")):
            encoded.extend(struct.pack("<Q", len(field)))
            encoded.extend(field)
    codegen_id = hashlib.sha256(encoded).hexdigest()
    if metadata.get("codegen_identity") != codegen_id:
        raise EvidenceError("codegen identity does not match framed host-program fields")
    source_text = source.read_text(encoding="utf-8")
    preamble = (f"// Fixed-run codegen: {version}\n// Host program: {host_id}\n"
                f"// Codegen identity: {codegen_id}\n")
    if not source_text.startswith(preamble):
        raise EvidenceError("generated source does not bind the fixed-run identity")
    body = _run_body(source_text, model)
    abi_body = _abi_run_body(source_text, model)
    _audit_fixedrun_structure(source_text, body, abi_body)
    dispatches = _audit_typed_operations(program, body)
    if not dispatches:
        raise EvidenceError("generated model run has no fixed dispatch")
    shader_records = candidate.get("shaders")
    if not isinstance(shader_records, list) or len(shader_records) != dispatches:
        raise EvidenceError("candidate shader count differs from plan dispatch count")
    shaders = []
    for index, record in enumerate(shader_records):
        path, digest = _artifact(stage, record, f"candidate shader {index}")
        shaders.append({"path": path.relative_to(stage).as_posix(), "sha256": digest})
    _audit_host_program_against_plan(
        stage, plan, program, shader_records, model, plan_hash)
    _audit_generated_static_tables(
        source_text, body, abi_body, program, shader_records, stage)

    capture = candidate.get("capture")
    if not isinstance(capture, dict) or type(capture.get("returncode")) is not int or capture["returncode"] != 0:
        raise EvidenceError("package capture command did not succeed")
    capture_argv, capture_cwd = capture.get("argv"), capture.get("cwd")
    if not isinstance(capture_argv, list) or not all(isinstance(token, str) for token in capture_argv):
        raise EvidenceError("package capture argv is malformed")
    capture_model = _one_option(capture_argv, "--model-name", "package capture model")
    if (not capture_argv or
            not all(isinstance(token, str) for token in capture_argv) or
            not isinstance(capture_cwd, str) or "--capture-vulkan-host-source" not in capture_argv or
            type(capture.get("returncode")) is not int or capture["returncode"] != 0 or
            Path(capture_cwd).resolve() != (stage / "test/Vulkan").resolve()):
        raise EvidenceError("package capture argv/cwd/returncode is malformed")
    capture_execution = _audit_logged_command(
        stage, capture, "fixed-run source capture")
    cache = _stage_cache_values(stage)
    source_root = _source_root(stage)
    expected_capture_tool = stage / "tools/ncnn-vulkan-fixedrun-capture"
    if Path(capture_argv[0]).resolve() != expected_capture_tool.resolve():
        raise EvidenceError("package capture did not use the fresh compiler capture adapter")
    if (capture_model != fixture.get("name") or
            capture_argv.count("--backend=vulkan") != 1 or
            capture_argv.count("--precision=f32") != 1):
        raise EvidenceError("package capture does not bind the fixture model/backend/precision")
    def require_capture_tool(option: str, expected: str | Path) -> None:
        if Path(_one_option(capture_argv, option, f"package capture {option}")).resolve() != Path(expected).resolve():
            raise EvidenceError(f"package capture {option} differs from the fresh configured tool")
    for option, expected in (("--driver", stage / "tools/ncnn-mlir-driver"),
                             ("--opt", stage / "bin/ncnn-mlir-opt"),
                             ("--translate", cache.get("MLIR_TRANSLATE", "")),
                             ("--clang", cache.get("CLANG", "")),
                             ("--nm", cache.get("LLVM_NM", "")),
                             ("--readelf", cache.get("LLVM_READELF", "")),
                             ("--llvm-as", cache.get("LLVM_AS", "")),
                             ("--vulkan-mlir-opt", cache.get("NCNN_MLIR_OPT_EXECUTABLE", "")),
                             ("--vulkan-spirv-val", cache.get("NCNN_SPIRV_VAL_EXECUTABLE", "")),
                             ("--vulkan-spirv-dis", cache.get("NCNN_SPIRV_DIS_EXECUTABLE", "")),
                             ("--vulkan-spirv-extract", stage / "tools/ncnn-vulkan-spirv-extract"),
                             ("--vulkan-host-cxx", cache.get("CMAKE_CXX_COMPILER", ""))):
        require_capture_tool(option, expected)
    bin_path, _ = _artifact(stage, fixture.get("bin"), "capture fixture bin")
    if (len(capture_argv) < 2 or Path(capture_argv[1]).resolve() != source_param.resolve() or
            sum(Path(token).resolve() == source_param.resolve()
                for token in capture_argv[1:] if not token.startswith("-")) != 1):
        raise EvidenceError("package capture command does not uniquely name source-root fixture .param")
    capture_bin = Path(_one_option(capture_argv, "--bin", "package capture .bin"))
    if (not capture_bin.is_file() or
            sha256_file(capture_bin) != sha256_file(bin_path)):
        raise EvidenceError("package capture command .bin bytes differ from hashed fixture input")
    originals = [candidate.get("source"), candidate.get("header"), candidate.get("plan"),
                 candidate.get("host_program"), candidate.get("tosa_ir"),
                 candidate.get("metadata")]
    capture_outputs, replay_outputs = capture.get("outputs"), capture.get("reproduced_outputs")
    if (not isinstance(capture_outputs, list) or not isinstance(replay_outputs, list) or
            len(capture_outputs) != len(originals) or len(replay_outputs) != len(originals)):
        raise EvidenceError("capture/replay output inventories are incomplete")
    capture_output_dir = Path(_one_option(
        capture_argv, "--output-dir", "package capture replay directory"))
    replay_source_path, _ = _artifact(stage, replay_outputs[0], "capture replay source")
    replay_dir = replay_source_path.parent
    if capture_output_dir.resolve() != replay_dir:
        raise EvidenceError("package capture output directory differs from reproduced files")
    original_hashes = []
    for index, (original, captured, replayed) in enumerate(zip(originals, capture_outputs, replay_outputs)):
        _, original_hash = _artifact(stage, original, f"candidate original output {index}")
        _, captured_hash = _artifact(stage, captured, f"captured package output {index}")
        _, replay_hash = _artifact(stage, replayed, f"replayed package output {index}")
        if original_hash != captured_hash or original_hash != replay_hash:
            raise EvidenceError("captured/reproduced package differs from compiled candidate")
        original_hashes.append(original_hash)
    incremental = capture.get("incremental_rebuild")
    incremental_summary = _audit_logged_command(
        stage, incremental, "fixed-run incremental rebuild capture")
    if (incremental_summary["argv"][0] != capture_argv[0] or
            Path(incremental_summary["cwd"]).resolve() != Path(capture_cwd).resolve()):
        raise EvidenceError("incremental capture invocation differs from checked-in producer context")
    incremental_originals = incremental.get("outputs") if isinstance(incremental, dict) else None
    incremental_replays = incremental.get("reproduced_outputs") if isinstance(incremental, dict) else None
    if (not isinstance(incremental_originals, list) or
            not isinstance(incremental_replays, list) or
            len(incremental_originals) != len(original_hashes) or
            len(incremental_replays) != len(original_hashes)):
        raise EvidenceError("incremental capture output inventories are incomplete")
    for index, (original, replayed, expected_hash) in enumerate(
            zip(incremental_originals, incremental_replays, original_hashes)):
        if (_artifact(stage, original, f"incremental original output {index}")[1] != expected_hash or
                _artifact(stage, replayed, f"incremental replay output {index}")[1] != expected_hash):
            raise EvidenceError("incremental capture differs from byte-identical candidate outputs")
    capture_log, capture_log_hash = _artifact(stage, capture.get("log"), "package capture log")

    compile_record = candidate.get("compile")
    if not isinstance(compile_record, dict):
        raise EvidenceError("candidate compile command is missing")
    compile_argv, compile_cwd = compile_record.get("argv"), compile_record.get("cwd")
    if (not isinstance(compile_argv, list) or not compile_argv or
            not all(isinstance(x, str) for x in compile_argv) or not isinstance(compile_cwd, str)):
        raise EvidenceError("candidate compile argv/cwd is malformed")
    object_snapshot_path, object_hash = _artifact(
        stage, compile_record.get("object"), "compiled candidate object snapshot")
    compile_object_target = compile_record.get("object_target_path")
    compile_source_path, compile_source_hash = _artifact(
        stage, compile_record.get("source"), "candidate compile source snapshot")
    if compile_source_path != source or compile_source_hash != source_hash:
        raise EvidenceError("candidate compile source snapshot differs from generated source bytes")
    actual_candidate_source = (stage / "test/Vulkan/fixedrun-candidates" / model /
                               "candidate_source.cpp").resolve()
    if (compile_record.get("source_target_path") != str(actual_candidate_source) or
            compile_record.get("source_target_sha256") != source_hash):
        raise EvidenceError("candidate source target path/hash differs from immutable source snapshot")
    actual_compile_argv, actual_compile_cwd = _actual_compile_command(
        stage, actual_candidate_source)
    if actual_compile_argv != compile_argv or actual_compile_cwd != str(Path(compile_cwd).resolve()):
        raise EvidenceError("candidate compile provenance differs from fresh CMake compile database")
    compile_execution = _audit_logged_command(
        stage, compile_record.get("execution"), "fixed-run candidate source compile")
    if (compile_execution["argv"] != compile_argv or
            Path(compile_execution["cwd"]).resolve() != Path(compile_cwd).resolve()):
        raise EvidenceError("candidate compile execution differs from actual compile database command")
    cache = _stage_cache_values(stage)
    configured_cxx = cache.get("CMAKE_CXX_COMPILER")
    if (not configured_cxx or Path(compile_argv[0]).resolve() != Path(configured_cxx).resolve()):
        raise EvidenceError("candidate compile does not use the fresh gate C++ compiler")
    compile_sources = []
    for token in compile_argv:
        if Path(token).suffix in {".cpp", ".cc", ".cxx"}:
            try:
                compile_sources.append(_command_path(stage, token, compile_cwd, "compile source"))
            except EvidenceError:
                continue
    if (compile_sources != [actual_candidate_source] or "-c" not in compile_argv or
            compile_argv.count("-o") != 1):
        raise EvidenceError("actual compile command does not compile exactly the generated source")
    _require_profile_instrumentation(compile_argv, profile,
                                     "fixed-run candidate source compile", link=False)
    if not any(token in {"-fPIC", "-fpic"} for token in compile_argv):
        raise EvidenceError("candidate shared-library object is not position independent")
    link_record = candidate.get("link")
    link_argv = link_record.get("argv", []) if isinstance(link_record, dict) else []
    _audit_no_forced_compiler_inputs(
        compile_argv, "fixed-run candidate compile", profile, link=False)
    includes = []
    for index, token in enumerate(compile_argv):
        if token in {"-I", "-isystem", "-iquote"} and index + 1 < len(compile_argv):
            includes.append(compile_argv[index + 1])
        elif token.startswith("-I"):
            includes.append(token[2:])
    if any("lifecycle_test_hooks" in item or "test/Vulkan/native" in item.replace("\\", "/")
           for item in includes):
        raise EvidenceError("candidate compile imports fake/test-only include roots")
    output_index = compile_argv.index("-o") + 1
    compile_output_target_path = (_command_path(
        stage, compile_argv[output_index], compile_cwd, "compile output")
        if output_index < len(compile_argv) else None)
    if (compile_output_target_path is None or
            not isinstance(compile_object_target, str) or
            compile_output_target_path != Path(compile_object_target).resolve() or
            compile_record.get("object_target_sha256") != object_hash or
            Path(compile_record.get("output", "")).resolve() != compile_output_target_path):
        raise EvidenceError("compile -o/object snapshot/target hash provenance differs")

    if not isinstance(link_record, dict):
        raise EvidenceError("candidate DSO link evidence is missing")
    link_argv, link_cwd = link_record.get("argv"), link_record.get("cwd")
    if (not isinstance(link_argv, list) or not link_argv or
            not all(isinstance(x, str) for x in link_argv) or not isinstance(link_cwd, str)):
        raise EvidenceError("candidate link argv/cwd is malformed")
    _require_profile_instrumentation(
        link_argv, profile, "fixed-run candidate DSO link", link=True)
    target = f"vulkan_fixedrun_{model}_validation"
    actual_link_file = _native_target_link_file(stage, target)
    try:
        actual_link_argv = shlex.split(actual_link_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise EvidenceError("candidate actual CMake link command cannot be parsed") from error
    actual_link_cwd = actual_link_file.parent.parent.parent
    if (actual_link_argv != link_argv or
            Path(link_cwd).resolve() != actual_link_cwd.resolve()):
        raise EvidenceError("candidate linker argv/cwd differs from actual CMake link.txt")
    link_execution = _audit_logged_command(
        stage, link_record.get("execution"), "fixed-run candidate DSO link")
    if (link_execution["argv"] != link_argv or
            Path(link_execution["cwd"]).resolve() != Path(link_cwd).resolve()):
        raise EvidenceError("candidate DSO execution differs from actual CMake link command")
    dso_snapshot_path, dso_hash = _artifact(
        stage, link_record.get("dso"), "candidate DSO snapshot")
    link_map_snapshot_path, map_hash = _artifact(
        stage, link_record.get("map"), "candidate linker map snapshot")
    helper_archive_snapshot, helper_hash = _artifact(
        stage, link_record.get("helper_archive"), "immutable helper archive snapshot")
    expected_dso_target = (stage / "test/Vulkan/fixedrun-candidates/libraries" / model /
                           f"lib{model}_fixedrun_validation.so").resolve()
    expected_map_target = (stage / "test/Vulkan/fixedrun-candidates/audit" / model /
                           f"vulkan_fixedrun_{model}_validation.map").resolve()
    expected_helper = (stage / "lib/VulkanDeviceHelper/libNCNNVulkanDeviceHelper.a").resolve()
    helper_target_path = link_record.get("helper_archive_target_path")
    dso_target_path = link_record.get("dso_target_path")
    map_target_path = link_record.get("map_target_path")
    if (dso_snapshot_path == expected_dso_target or
            dso_target_path is None or Path(dso_target_path).resolve() != expected_dso_target or
            link_record.get("dso_target_sha256") != dso_hash or
            link_map_snapshot_path == expected_map_target or
            map_target_path is None or Path(map_target_path).resolve() != expected_map_target or
            link_record.get("map_target_sha256") != map_hash or
            not isinstance(helper_target_path, str) or
            Path(helper_target_path).resolve() != expected_helper or
            link_record.get("helper_archive_target_sha256") != helper_hash or
            helper_archive_snapshot == expected_helper):
        raise EvidenceError("candidate DSO/map/helper snapshots differ from CMake target identities")
    helper_build = _audit_helper_archive(
        stage, expected_helper, profile, Path(configured_cxx), link_record.get("helper_build"))
    if helper_build["archive"]["sha256"] != helper_hash or \
            helper_build["archive"]["path"] != helper_archive_snapshot.relative_to(stage).as_posix():
        raise EvidenceError("candidate helper archive snapshot differs from clean/rebuild evidence")
    if _artifact(stage, candidate.get("dso"), "candidate DSO duplicate")[1] != dso_hash or candidate.get("dso_sha256") != dso_hash:
        raise EvidenceError("candidate DSO duplicate identity differs from link output")
    objects = link_record.get("objects")
    if not isinstance(objects, list) or not objects:
        raise EvidenceError("candidate link object list is missing")
    object_snapshot_paths, object_target_paths = set(), set()
    object_hash_by_target = {}
    for index, record in enumerate(objects):
        snapshot_path, snapshot_hash = _artifact(
            stage, record, f"link object snapshot {index}")
        target_path = record.get("target_path") if isinstance(record, dict) else None
        target_hash = record.get("target_sha256") if isinstance(record, dict) else None
        if (not isinstance(target_path, str) or not isinstance(target_hash, str) or
                not SHA256_RE.fullmatch(target_hash) or target_hash != snapshot_hash):
            raise EvidenceError(f"link object {index} target identity differs from immutable object bytes")
        object_target = Path(target_path).resolve()
        try:
            object_target.relative_to(stage.resolve())
        except ValueError as error:
            raise EvidenceError(f"link object {index} target is outside the stage") from error
        object_snapshot_paths.add(snapshot_path)
        object_target_paths.add(object_target)
        object_hash_by_target[object_target] = snapshot_hash
    if (len(object_snapshot_paths) != len(objects) or
            len(object_target_paths) != len(objects) or
            Path(compile_object_target).resolve() not in object_target_paths or
            object_hash_by_target.get(Path(compile_object_target).resolve()) != object_hash):
        raise EvidenceError("candidate link object snapshots/targets are duplicate or omit the compiled object")
    vulkan_library_value = cache.get("Vulkan_LIBRARY")
    if not isinstance(vulkan_library_value, str) or not vulkan_library_value:
        raise EvidenceError("fresh CMake cache does not identify the linked Vulkan loader DSO")
    expected_vulkan_library = Path(vulkan_library_value).resolve()
    vulkan_library = _audit_vulkan_library(
        stage, link_record, link_argv, link_cwd, vulkan_library_value)
    seen_objects, seen_archives, seen_shared = _candidate_link_inputs(
        stage, link_argv, link_cwd, expected_dso_target, expected_vulkan_library)
    if (seen_objects != object_target_paths or seen_archives != {expected_helper} or
            seen_shared != {expected_vulkan_library}):
        raise EvidenceError(
            "candidate actual linker inputs differ from generated objects, helper archive and Vulkan SDK DSO")
    allowed_named_libraries = {"-lvulkan", "-lvulkan-1", "-ldl", "-lpthread",
                               "-lm", "-lstdc++", "-lgcc_s", "-lc"}
    named_libraries = {token for token in link_argv if token.startswith("-l")}
    forwarded_library_flags = [token for token in link_argv
                               if token.startswith("-Wl,") and
                               re.search(r"(?:^|,)-l(?:[^,]+)?", token)]
    forwarded_library_flags.extend(
        f"-Xlinker {link_argv[index + 1]}"
        for index, token in enumerate(link_argv[:-1])
        if token == "-Xlinker" and link_argv[index + 1].startswith("-l"))
    if named_libraries - allowed_named_libraries or forwarded_library_flags:
        raise EvidenceError(
            "candidate actual link command names unreviewed libraries: "
            f"{sorted(named_libraries - allowed_named_libraries)} {forwarded_library_flags}")
    configured_link_driver = (
        _configured_gate_tool(stage, "CMAKE_CXX_COMPILER", "cxx-compiler")
        if profile in {"release", "sanitize"} else Path(configured_cxx).resolve())
    sanitizer_runtime_dir = (_configured_sanitizer_runtime_dir(stage)
                             if profile == "sanitize" else None)
    _audit_candidate_link_scope(
        link_argv, configured_link_driver, sanitizer_runtime_dir)
    if "-z,defs" not in ",".join(link_argv) and "-z defs" not in " ".join(link_argv):
        raise EvidenceError("candidate DSO is not linked with undefined-symbol rejection")
    if link_argv.count("-o") != 1:
        raise EvidenceError("candidate link command does not have one unique output argument")
    out_index = link_argv.index("-o") + 1
    if out_index >= len(link_argv) or _command_path(stage, link_argv[out_index], link_cwd,
                                                    "DSO output") != expected_dso_target:
        raise EvidenceError("candidate link -o differs from immutable DSO target identity")
    map_values = [re.split(r"(?:-Map=|--Map=)", token, maxsplit=1)[-1]
                  for token in link_argv if "-Map=" in token or "--Map=" in token]
    if len(map_values) != 1 or _command_path(
            stage, map_values[0], link_cwd, "map target output") != expected_map_target:
        raise EvidenceError("actual link command does not uniquely bind its map target")
    map_text = link_map_snapshot_path.read_text(encoding="utf-8", errors="replace")
    if (expected_helper.name not in map_text or
            Path(compile_object_target).name not in map_text or
            helper_build["member"] not in map_text):
        raise EvidenceError("candidate linker map omits generated object/helper archive member")
    elf = link_record.get("elf")
    if not isinstance(elf, dict):
        raise EvidenceError("actual candidate ELF audit is missing")
    exports_path, exports_hash, exports_command_summary = _audit_elf_tool_invocation(
        stage, elf.get("commands", {}).get("exports") if isinstance(elf.get("commands"), dict) else None,
        elf.get("exports"), expected_dso_target, cache.get("LLVM_NM", ""),
        ["-D", "--defined-only"], "candidate ELF export audit")
    needed_path, needed_hash, needed_command_summary = _audit_elf_tool_invocation(
        stage, elf.get("commands", {}).get("needed") if isinstance(elf.get("commands"), dict) else None,
        elf.get("needed"), expected_dso_target, cache.get("LLVM_READELF", ""),
        ["-d"], "candidate ELF needed audit")
    undefined_path, undefined_hash, undefined_command_summary = _audit_elf_tool_invocation(
        stage, elf.get("commands", {}).get("undefined") if isinstance(elf.get("commands"), dict) else None,
        elf.get("undefined"), expected_dso_target, cache.get("LLVM_NM", ""),
        ["-D", "--undefined-only"], "candidate ELF undefined audit")
    exports = exports_path.read_text(encoding="utf-8", errors="replace")
    needed = needed_path.read_text(encoding="utf-8", errors="replace")
    undefined = undefined_path.read_text(encoding="utf-8", errors="replace")
    expected_symbols = {f"{model}_vk_create", f"{model}_vk_run", f"{model}_vk_destroy"}
    exported_symbols = {line.split()[-1].split("@")[0]
                        for line in exports.splitlines() if line.split()}
    if exported_symbols != expected_symbols:
        raise EvidenceError("candidate DSO export set is not exactly its three model C ABI symbols")
    checked_needed = re.sub(r"libclang_rt\.asan[^\s]*", "", needed, flags=re.I) if profile == "sanitize" else needed
    if re.search(r"lib(?:LLVM|MLIR|clang|NCNNVulkanRuntime|NCNNVulkanPlanSupport)",
                 checked_needed, re.I):
        raise EvidenceError("candidate ELF depends on compiler or generic-runtime DSO")
    _audit_candidate_dynamic_dependencies(needed)
    if profile == "sanitize" and not re.search(r"libclang_rt\.asan", needed, re.I):
        raise EvidenceError("SAN candidate DSO lacks shared ASan runtime dependency")
    if re.search(r"VulkanRuntimeContext|VulkanExecutionPlan|NCNNVulkanRuntime|"
                 r"NCNNVulkanPlanSupport|deserializeExecutionPlan|execution_plan|"
                 r"lib(?:LLVM|MLIR)[^\\s]*", undefined, re.I):
        raise EvidenceError("candidate ELF has unresolved generic executor/compiler symbols")
    capture_summary = {"argv": capture_argv, "cwd": capture_cwd,
                       "returncode": capture["returncode"],
                       "execution": _command_report(capture_execution),
                       "incremental_rebuild": _command_report(incremental_summary),
                       "outputs": capture_outputs,
                       "reproduced_outputs": replay_outputs,
                       "log": {"path": capture_log.relative_to(stage).as_posix(),
                               "sha256": sha256_file(capture_log)}}
    native_loader, native_loader_hash = _artifact(
        stage, candidate.get("native_loader"), "model-specific C++ loader")
    loader_compile = candidate.get("native_loader_compile")
    loader_link = candidate.get("native_loader_link")
    if not isinstance(loader_compile, dict) or not isinstance(loader_link, dict):
        raise EvidenceError("native C ABI loader compile/link provenance is missing")
    loader_compile_argv, loader_compile_cwd = loader_compile.get("argv"), loader_compile.get("cwd")
    loader_source, loader_source_hash = _artifact(
        stage, loader_compile.get("source"), "native C ABI loader source copy")
    loader_object, loader_object_hash = _artifact(
        stage, loader_compile.get("object"), "native C ABI loader object")
    if (not isinstance(loader_compile_argv, list) or not isinstance(loader_compile_cwd, str) or
            not all(isinstance(x, str) for x in loader_compile_argv)):
        raise EvidenceError("native loader compile command is malformed")
    loader_source_root = (_source_root(stage) /
                          "test/Vulkan/native/fixedrun_native_loader.cpp").resolve()
    loader_source_target = loader_compile.get("source_target_path")
    if (not isinstance(loader_source_target, str) or
            Path(loader_source_target).resolve() != loader_source_root or
            loader_compile.get("source_target_sha256") != loader_source_hash):
        raise EvidenceError("native loader source target differs from its immutable source snapshot")
    _audit_native_loader_phase_implementation(
        loader_source.read_text(encoding="utf-8", errors="strict"))
    actual_loader_compile, actual_loader_compile_cwd = _actual_compile_command(
        stage, Path(loader_source_target), model=model)
    if (actual_loader_compile != loader_compile_argv or
            actual_loader_compile_cwd != str(Path(loader_compile_cwd).resolve())):
        raise EvidenceError("native loader compile differs from its actual CMake compile database row")
    if (profile in {"release", "sanitize"} and
            Path(loader_compile_argv[0]).resolve() != Path(configured_cxx).resolve()):
        raise EvidenceError("native loader compile uses a different gate C++ compiler")
    loader_compile_execution = _audit_logged_command(
        stage, loader_compile.get("execution"), "native candidate loader compile")
    if (loader_compile_execution["argv"] != loader_compile_argv or
            Path(loader_compile_execution["cwd"]).resolve() != Path(loader_compile_cwd).resolve()):
        raise EvidenceError("native loader compile execution differs from compile database")
    if ("-DNCNN_FIXEDRUN_BACKEND_VULKAN=1" not in loader_compile_argv or
            f"-DNCNN_FIXEDRUN_MODEL={model}" not in loader_compile_argv or
            any("NCNN_VULKAN_LIFECYCLE_TEST" in token for token in loader_compile_argv)):
        raise EvidenceError("candidate loader is not compiled for this Vulkan model/real ABI")
    _audit_no_forced_compiler_inputs(
        loader_compile_argv, "native candidate loader compile", profile,
        link=False)
    loader_compile_output_index = (loader_compile_argv.index("-o") + 1
                                   if loader_compile_argv.count("-o") == 1 else len(loader_compile_argv))
    loader_object_target_path = loader_compile.get("object_target_path")
    if (loader_compile_output_index >= len(loader_compile_argv) or
            not isinstance(loader_object_target_path, str) or
            _command_path(stage, loader_compile_argv[loader_compile_output_index],
                          loader_compile_cwd, "native loader object") !=
            Path(loader_object_target_path).resolve() or
            loader_compile.get("object_target_sha256") != loader_object_hash):
        raise EvidenceError("native loader compile -o differs from immutable object/target identity")
    loader_link_argv, loader_link_cwd = loader_link.get("argv"), loader_link.get("cwd")
    if (not isinstance(loader_link_argv, list) or not isinstance(loader_link_cwd, str) or
            not all(isinstance(x, str) for x in loader_link_argv)):
        raise EvidenceError("native loader link command is malformed")
    loader_target = f"vulkan_fixedrun_loader_{model}"
    loader_link_file = _native_target_link_file(stage, loader_target)
    try:
        actual_loader_link_argv = shlex.split(loader_link_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise EvidenceError("native loader CMake link command cannot be parsed") from error
    actual_loader_link_cwd = loader_link_file.parent.parent.parent
    if (actual_loader_link_argv != loader_link_argv or
            Path(loader_link_cwd).resolve() != actual_loader_link_cwd.resolve()):
        raise EvidenceError("native loader link provenance differs from actual CMake link.txt")
    configured_link_driver = (
        _configured_gate_tool(stage, "CMAKE_CXX_COMPILER", "cxx-compiler")
        if profile in {"release", "sanitize"} else Path(configured_cxx).resolve())
    if Path(loader_link_argv[0]).resolve() != configured_link_driver.resolve():
        raise EvidenceError("native loader link uses a different gate C++ compiler")
    loader_link_execution = _audit_logged_command(
        stage, loader_link.get("execution"), "native candidate loader link")
    if (loader_link_execution["argv"] != loader_link_argv or
            Path(loader_link_execution["cwd"]).resolve() != Path(loader_link_cwd).resolve()):
        raise EvidenceError("native loader link execution differs from actual CMake link.txt")
    if loader_link_argv.count("-o") != 1:
        raise EvidenceError("native loader link command has no unique output")
    loader_out_index = loader_link_argv.index("-o") + 1
    expected_loader_target = (stage / "test/Vulkan/native" / loader_target).resolve()
    loader_executable_target = loader_link.get("executable_target_path")
    if (loader_out_index >= len(loader_link_argv) or
            not isinstance(loader_executable_target, str) or
            Path(loader_executable_target).resolve() != expected_loader_target or
            _command_path(stage, loader_link_argv[loader_out_index], loader_link_cwd,
                          "native loader output") != expected_loader_target or
            loader_link.get("executable_target_sha256") != native_loader_hash):
        raise EvidenceError("native loader link -o differs from immutable target identity")
    _audit_no_forced_compiler_inputs(
        loader_link_argv, "native candidate loader link", profile, link=True)
    loader_link_objects = set()
    loader_link_object_records = loader_link.get("objects")
    if not isinstance(loader_link_object_records, list):
        raise EvidenceError("native loader link object snapshot inventory is missing")
    for index, record in enumerate(loader_link_object_records):
        snapshot, object_digest = _artifact(stage, record, f"native loader link object {index}")
        target_path = record.get("target_path")
        if (not isinstance(target_path, str) or record.get("target_sha256") != object_digest or
                snapshot != loader_object):
            raise EvidenceError("native loader linked object snapshot differs from actual compile object")
        loader_link_objects.add(Path(target_path).resolve())
    actual_loader_object_inputs = {
        _command_path(stage, token, loader_link_cwd, "native loader object input")
        for token in loader_link_argv
        if not token.startswith("-") and Path(token).suffix == ".o"}
    if (loader_link_objects != {Path(loader_object_target_path).resolve()} or
            actual_loader_object_inputs != loader_link_objects):
        raise EvidenceError("native loader link object input differs from immutable compiled object")
    loader_link_executable, loader_link_hash = _artifact(
        stage, loader_link.get("executable"), "C++ candidate loader executable snapshot")
    if (loader_link_executable != native_loader or loader_link_hash != native_loader_hash or
            loader_link_hash != loader_link.get("executable_target_sha256")):
        raise EvidenceError("candidate loader snapshot differs from its linked target identity")
    loader_dynamic_dependencies = _audit_shared_asan_needed(
        stage, native_loader, profile, f"{model} native Vulkan loader")
    return {"model": model,
            "source": {"path": source.relative_to(stage).as_posix(), "sha256": source_hash},
            "header": {"path": header.relative_to(stage).as_posix(), "sha256": header_hash},
            "metadata": {"path": metadata_path.relative_to(stage).as_posix(), "sha256": metadata_hash},
            "plan": {"path": plan_path.relative_to(stage).as_posix(), "sha256": plan_hash},
            "host_program": {"path": program_path.relative_to(stage).as_posix(), "sha256": program_hash},
            "tosa_ir": {"path": tosa_path.relative_to(stage).as_posix(), "sha256": tosa_hash},
            "reshape_views": reshape_views,
            "capture": capture_summary, "shaders": shaders,
            "compile": {"argv": compile_argv, "cwd": compile_cwd,
                        "execution": _command_report(compile_execution),
                        "object": {"path": object_snapshot_path.relative_to(stage).as_posix(),
                                   "sha256": object_hash},
                        "object_target_path": str(compile_output_target_path),
                        "object_target_sha256": compile_record["object_target_sha256"]},
            "link": {"argv": link_argv, "cwd": link_cwd,
                     "execution": _command_report(link_execution),
                     "dso": {"path": dso_snapshot_path.relative_to(stage).as_posix(),
                             "sha256": dso_hash},
                     "dso_target_path": str(expected_dso_target),
                     "dso_target_sha256": dso_hash,
                     "map": {"path": link_map_snapshot_path.relative_to(stage).as_posix(),
                             "sha256": map_hash},
                     "map_target_path": str(expected_map_target),
                     "map_target_sha256": map_hash,
                     "helper_archive": {"path": helper_archive_snapshot.relative_to(stage).as_posix(),
                                        "sha256": helper_hash},
                     "helper_archive_target_path": str(expected_helper),
                     "helper_archive_target_sha256": link_record["helper_archive_target_sha256"],
                     "helper_build": helper_build,
                     "vulkan_library": vulkan_library,
                     "exports": {"path": exports_path.relative_to(stage).as_posix(), "sha256": exports_hash},
                     "needed": {"path": needed_path.relative_to(stage).as_posix(), "sha256": needed_hash},
                     "undefined": {"path": undefined_path.relative_to(stage).as_posix(), "sha256": undefined_hash},
                     "elf_commands": {"exports": exports_command_summary,
                                      "needed": needed_command_summary,
                                      "undefined": undefined_command_summary}},
            "native_loader": {"path": native_loader.relative_to(stage).as_posix(),
                              "sha256": native_loader_hash,
                              "source": {"path": loader_source.relative_to(stage).as_posix(),
                                         "sha256": loader_source_hash},
                              "object": {"path": loader_object.relative_to(stage).as_posix(),
                                         "sha256": loader_object_hash},
                              "compile_argv": loader_compile_argv,
                              "compile_cwd": loader_compile_cwd,
                              "compile_execution": _command_report(loader_compile_execution),
                              "link_argv": loader_link_argv,
                              "link_cwd": loader_link_cwd,
                              "link_execution": _command_report(loader_link_execution),
                              "dynamic_dependencies": loader_dynamic_dependencies}}


def _legacy_link_input_paths(argv: list[str], cwd: Path) -> tuple[set[Path], set[Path]]:
    expanded = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("@"):
            raise EvidenceError("legacy linker response files are not auditable")
        if token.startswith("-Wl,"):
            forwarded = token[4:].split(",")
            if any(item.startswith("@") for item in forwarded):
                raise EvidenceError("legacy linker forwarded response files are not auditable")
            expanded.extend(forwarded)
        elif token == "-Xlinker":
            if index + 1 >= len(argv) or argv[index + 1].startswith("@"):
                raise EvidenceError("legacy -Xlinker argument is missing or response-file based")
            expanded.append(argv[index + 1])
            index += 1
        else:
            expanded.append(token)
        index += 1
    objects, archives = set(), set()
    for token in expanded:
        if token.startswith("-") or Path(token).suffix not in {".o", ".a"}:
            continue
        path = Path(token)
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        if path.suffix == ".o":
            objects.add(path)
        else:
            archives.add(path)
    return objects, archives


def _audit_legacy_reference(stage: Path, reference: object,
                            require_independent: bool) -> dict:
    if not isinstance(reference, dict):
        raise EvidenceError("legacy reference record is missing")
    status = reference.get("status")
    if status == "not-supplied":
        if require_independent:
            raise EvidenceError("VA4 qualification requires independently rebuilt old generic Vulkan")
        return {"status": status, "qualified": False}
    if status != "verified-independent-source":
        raise EvidenceError("legacy reference status is not independently source-bound")
    manifest_path, manifest_hash = _artifact(stage, reference.get("manifest"),
                                              "legacy reference manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = reference.get("source_fingerprint")
    if (not isinstance(manifest, dict) or manifest.get("schema") != LEGACY_SCHEMA or
            manifest.get("role") != "legacy-product-reference" or
            manifest.get("git_revision") != LEGACY_REVISION or
            reference.get("git_revision") != LEGACY_REVISION or
            manifest.get("source_fingerprint") != fingerprint or
            not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint)):
        raise EvidenceError("legacy root manifest is not pinned to the pre-VA4 clean source")
    files = reference.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError("legacy source/build artifacts were not copied into the stage")
    by_source, copied_by_source, roles = {}, {}, set()
    for index, item in enumerate(files):
        if (not isinstance(item, dict) or not isinstance(item.get("source_path"), str) or
                not isinstance(item.get("role"), str)):
            raise EvidenceError("legacy copied file row is malformed")
        path, digest = _artifact(stage, item.get("artifact"), f"legacy reference file {index}")
        if digest != item.get("sha256"):
            raise EvidenceError("legacy copied file hash differs from its raw record")
        by_source[item["source_path"]] = (path, digest)
        copied_by_source[item["source_path"]] = digest
        roles.add(item["role"].lower().replace("_", " "))
    def find_copied(record: object, label: str) -> tuple[Path, str]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise EvidenceError(f"legacy {label} source record is missing")
        copied = by_source.get(record["path"])
        if copied is None or copied[1] != record.get("sha256"):
            raise EvidenceError(f"legacy {label} source file is not hash-bound in the stage")
        return copied
    source_archive_record = manifest.get("source_archive")
    source_inventory_record = manifest.get("source_inventory")
    inventory_path, _ = find_copied(source_inventory_record, "source inventory")
    archive_path, _ = find_copied(source_archive_record, "source archive")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (not isinstance(inventory, dict) or
            inventory.get("schema") != "ncnn-legacy-source-inventory/v1" or
            inventory.get("git_revision") != LEGACY_REVISION or
            inventory.get("source_fingerprint") != fingerprint or
            not isinstance(inventory.get("files"), list)):
        raise EvidenceError("legacy source inventory identity is invalid")
    digest = hashlib.sha256()
    archive_files = set()
    gitlink_roots = {
        PurePosixPath(item["path"])
        for item in inventory["files"]
        if isinstance(item, dict) and item.get("kind") == "gitlink" and
        isinstance(item.get("path"), str)
    }
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()
                       if member.isfile() or member.issym()}
            for item in sorted(inventory["files"], key=lambda row: row.get("path", "")):
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise EvidenceError("legacy source inventory entry is malformed")
                relative = item["path"]
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise EvidenceError("legacy source inventory path is unsafe")
                if item.get("kind") == "file":
                    member = members.get(relative)
                    if member is None:
                        raise EvidenceError(f"legacy source archive lacks {relative}")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise EvidenceError(f"legacy source archive member cannot be read: {relative}")
                    data = stream.read()
                    file_hash = hashlib.sha256(data).hexdigest()
                    git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                    if file_hash != item.get("sha256") or git_blob != item.get("git_blob"):
                        raise EvidenceError(f"legacy source bytes differ from inventory: {relative}")
                    content = b"file:" + file_hash.encode("ascii")
                    archive_files.add(relative)
                elif item.get("kind") == "gitlink":
                    oid = item.get("oid")
                    if not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}", oid):
                        raise EvidenceError(f"legacy gitlink is malformed: {relative}")
                    content = b"gitlink:" + oid.encode("ascii")
                elif item.get("kind") == "symlink":
                    target = item.get("target")
                    member = members.get(relative)
                    if (not isinstance(target, str) or member is None or
                            not member.issym() or member.linkname != target):
                        raise EvidenceError(f"legacy source symlink differs from inventory: {relative}")
                    content = b"symlink:" + os.fsencode(target)
                    archive_files.add(relative)
                else:
                    raise EvidenceError(f"unsupported legacy inventory kind at {relative}")
                relative_path = PurePosixPath(relative)
                # The gate source fingerprint hashes a gitlink by OID; nested
                # source files remain archive-verified but are not re-hashed as
                # entries in the parent repository's fingerprint.
                nested_gitlink_content = any(
                    relative_path != gitlink and relative_path.is_relative_to(gitlink)
                    for gitlink in gitlink_roots)
                if not nested_gitlink_content:
                    digest.update(os.fsencode(relative) + b"\0" + content + b"\0")
            if set(members) != archive_files:
                raise EvidenceError("legacy source archive has extra or missing files")
    except (OSError, tarfile.TarError) as error:
        raise EvidenceError(f"legacy source archive verification failed: {error}") from error
    if digest.hexdigest() != fingerprint:
        raise EvidenceError("legacy source fingerprint differs from archived source inventory")
    clean_record = manifest.get("clean_source_status")
    clean_path, _ = find_copied(clean_record, "clean-source status")
    clean_status = reference.get("clean_source_status")
    if (not isinstance(clean_status, dict) or clean_status.get("record") != clean_record or
            _artifact(stage, clean_status.get("artifact"), "copied clean-source status")[0] != clean_path or
            clean_path.read_bytes() != b""):
        raise EvidenceError("legacy source status is not clean")
    required_roles = ("source archive", "source inventory", "clean source status",
                      "cmake cache", "compile commands", "configure", "build")
    missing = [token for token in required_roles if not any(token in role for role in roles)]
    if missing:
        raise EvidenceError(f"legacy source/build provenance lacks {missing}")
    build = manifest.get("build")
    exit_codes = build.get("exit_codes") if isinstance(build, dict) else None
    if (not isinstance(exit_codes, dict) or set(exit_codes) != {"configure", "build"} or
            any(type(code) is not int or code != 0 for code in exit_codes.values())):
        raise EvidenceError("legacy source was not configured and built successfully")
    configure, build_argv = build.get("configure_argv"), build.get("build_argv")
    if (not isinstance(configure, list) or not isinstance(build_argv, list) or
            not all(isinstance(token, str) for token in configure + build_argv) or
            "-DCMAKE_BUILD_TYPE=Release" not in configure or "-DBUILD_TESTING=OFF" not in configure or
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--trace" not in configure or
            not any(token == "--parallel" or token.startswith("--parallel=") for token in build_argv)):
        raise EvidenceError("legacy reference commands are not a fresh parallel Release build")
    legacy_cache_path = None
    legacy_cache_hash = None
    for key, record in (("cmake cache", build.get("cmake_cache")),
                        ("compile commands", build.get("compile_commands"))):
        copied_path, copied_hash = find_copied(record, key)
        if key == "cmake cache":
            legacy_cache_path, legacy_cache_hash = copied_path, copied_hash
    compiler_record = manifest.get("compiler")
    compiler_path, compiler_hash = find_copied(compiler_record, "compiler binary")
    compiler_raw = reference.get("compiler")
    if (not isinstance(compiler_raw, dict) or
            _artifact(stage, compiler_raw.get("staged"), "staged legacy compiler")[0] != compiler_path or
            compiler_raw.get("sha256") != compiler_hash):
        raise EvidenceError("legacy compiler binary copy differs from manifest")
    tools = reference.get("tools")
    required_legacy_tools = {"driver", "opt", "nm", "readelf", "mlir-opt",
                             "spirv-val", "spirv-dis", "spirv-extract", "host-cxx"}
    if not isinstance(tools, dict) or not required_legacy_tools <= set(tools):
        raise EvidenceError("legacy compiler/upstream Vulkan tool inventory is incomplete")
    tool_paths = {}
    for role, tool in tools.items():
        if not isinstance(tool, dict) or not isinstance(tool.get("path"), str):
            raise EvidenceError(f"legacy tool record is malformed: {role}")
        tool_path = Path(tool["path"]).resolve()
        _, tool_hash = _artifact(stage, tool.get("staged"), f"legacy tool {role}")
        if (tool_hash != tool.get("sha256") or
                tool_hash != copied_by_source.get(str(tool_path))):
            raise EvidenceError(f"legacy tool differs from copied source identity: {role}")
        tool_paths[role] = tool_path
    if legacy_cache_path is None:
        raise EvidenceError("legacy CMake cache copy is missing")
    legacy_cache = {}
    for line in legacy_cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        legacy_cache[key.split(":", 1)[0]] = value
    expected_cache = {
        "BUILD_TESTING": "OFF",
        "NCNN_ENABLE_VULKAN_PRODUCT": "ON",
        "NCNN_ENABLE_VULKAN_V0_TESTS": "OFF",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_HOME_DIRECTORY": manifest.get("source_root"),
        "CMAKE_EXE_LINKER_FLAGS": "-Wl,--trace",
    }
    if any(legacy_cache.get(key) != value for key, value in expected_cache.items()):
        raise EvidenceError("legacy CMake cache differs from the pinned product-only Release source")
    if legacy_cache.get("COMPILER_SANITIZERS", ""):
        raise EvidenceError("legacy Release build unexpectedly enables sanitizer instrumentation")
    llvm_dir = legacy_cache.get("LLVM_DIR")
    if not isinstance(llvm_dir, str) or not llvm_dir:
        raise EvidenceError("legacy CMake cache does not pin LLVM_DIR for the invoked nm tool")
    expected_nm = (Path(llvm_dir).resolve().parents[2] / "bin" / "llvm-nm").resolve()
    selection = reference.get("tool_selection")
    expected_selection = (
        f"legacy-cmake-cache:LLVM_DIR={llvm_dir};derived-tools-bin/llvm-nm")
    selection_sources = selection.get("selected_from") if isinstance(selection, dict) else None
    selection_cache = selection.get("cmake_cache") if isinstance(selection, dict) else None
    cache_source_path = build.get("cmake_cache", {}).get("path") if isinstance(build.get("cmake_cache"), dict) else None
    if (not isinstance(selection, dict) or not isinstance(selection_sources, dict) or
            selection_sources.get("nm") != expected_selection or
            not isinstance(tools.get("nm"), dict) or
            tools["nm"].get("selected_from") != expected_selection or
            _artifact(stage, selection_cache, "legacy nm tool-selection CMake cache")[0] != legacy_cache_path or
            selection.get("cmake_cache_sha256") != legacy_cache_hash or
            selection.get("cmake_cache_source_path") != cache_source_path):
        raise EvidenceError("legacy nm selection is not bound to its copied original LLVM_DIR cache")
    cache_tool_paths = {
        "host-cxx": "CMAKE_CXX_COMPILER",
        "readelf": "NCNN_VULKAN_READELF",
        "mlir-opt": "NCNN_MLIR_OPT_EXECUTABLE",
        "spirv-val": "NCNN_SPIRV_VAL_EXECUTABLE",
        "spirv-dis": "NCNN_SPIRV_DIS_EXECUTABLE",
    }
    if any(not isinstance(legacy_cache.get(key), str) or not legacy_cache[key]
           for key in cache_tool_paths.values()):
        raise EvidenceError("legacy CMake cache omits an invoked Vulkan tool identity")
    expected_tools = {role: Path(legacy_cache[key]).resolve()
                      for role, key in cache_tool_paths.items()}
    expected_tools["nm"] = expected_nm
    manifest_tool_rows = manifest.get("tools")
    if not isinstance(manifest_tool_rows, list):
        raise EvidenceError("legacy manifest project-tool inventory is malformed")
    project_tool_paths = {item.get("role"): Path(item["path"]).resolve()
                          for item in manifest_tool_rows if isinstance(item, dict) and
                          isinstance(item.get("role"), str) and
                          isinstance(item.get("path"), str)}
    for role, manifest_role in (("driver", "ncnn-mlir-driver"),
                                ("opt", "ncnn-mlir-opt"),
                                ("spirv-extract", "ncnn-vulkan-spirv-extract")):
        expected_path = project_tool_paths.get(manifest_role)
        if expected_path is None:
            raise EvidenceError(f"legacy manifest does not pin project tool role {manifest_role}")
        expected_tools[role] = expected_path
    if any(tool_paths.get(role) != path for role, path in expected_tools.items()):
        raise EvidenceError("legacy invoked tool paths differ from source/build-pinned identities")
    links, objects, archives = (build.get("link_commands"), build.get("objects"),
                                build.get("static_inputs"))
    if not isinstance(links, list) or not isinstance(objects, list) or not isinstance(archives, list):
        raise EvidenceError("legacy compiler link/object provenance is incomplete")
    compile_commands_path, _ = find_copied(build.get("compile_commands"),
                                           "compile commands")
    try:
        compile_database = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"legacy compile database cannot be parsed: {error}") from error
    if not isinstance(compile_database, list) or len(compile_database) != len(objects):
        raise EvidenceError("legacy compile database/object inventory counts differ")
    object_records = {}
    for index, row in enumerate(objects):
        if not isinstance(row, dict) or not isinstance(row.get("source"), str):
            raise EvidenceError(f"legacy object record {index} is malformed")
        object_record = row.get("object")
        object_copy, object_hash = find_copied(object_record, f"compiled object {index}")
        if not isinstance(object_record, dict) or not isinstance(object_record.get("path"), str):
            raise EvidenceError(f"legacy object record {index} has no original output path")
        object_path = Path(object_record["path"]).resolve()
        source_path = str(Path(row["source"]).resolve())
        if (object_path in object_records or
                copied_by_source.get(str(object_path)) != object_hash or
                sha256_file(object_copy) != object_hash):
            raise EvidenceError("legacy compiled object records are duplicate or not hash-bound")
        object_records[object_path] = (source_path, object_hash)
    archive_records = {}
    for index, archive_record in enumerate(archives):
        archive_copy, archive_hash = find_copied(archive_record, f"static archive {index}")
        if not isinstance(archive_record, dict) or not isinstance(archive_record.get("path"), str):
            raise EvidenceError(f"legacy static archive record {index} is malformed")
        archive_path = Path(archive_record["path"]).resolve()
        if (archive_path in archive_records or copied_by_source.get(str(archive_path)) != archive_hash or
                sha256_file(archive_copy) != archive_hash):
            raise EvidenceError("legacy static archive inventory has duplicate/unverified bytes")
        archive_records[archive_path] = archive_hash
    if build.get("static_input_coverage") != "explicit-link-argv+linker-trace/v1":
        raise EvidenceError("legacy static-input inventory lacks complete link-argv/linker-trace coverage")
    trace_record = build.get("link_trace")
    logs = build.get("logs")
    if not isinstance(logs, dict) or trace_record != logs.get("build"):
        raise EvidenceError("legacy linker trace is not the actual full-build log record")
    trace_path, _ = find_copied(trace_record, "full-build linker trace")
    link_commands_by_target = {}
    link_directories = set()
    command_lines = set()
    explicit_archives, explicit_objects = set(), set()
    for index, link_row in enumerate(links):
        if (not isinstance(link_row, dict) or not isinstance(link_row.get("target"), str) or
                not isinstance(link_row.get("command"), dict)):
            raise EvidenceError(f"legacy link command {index} is malformed")
        command_record = link_row["command"]
        command_copy, _ = find_copied(command_record, f"legacy link command {index}")
        if not isinstance(command_record.get("path"), str):
            raise EvidenceError(f"legacy link command {index} has no original path")
        original_link = Path(command_record["path"]).resolve()
        actual_cwd = original_link.parent.parent.parent.resolve()
        if (not isinstance(link_row.get("cwd"), str) or
                Path(link_row["cwd"]).resolve() != actual_cwd or
                command_copy.read_text(encoding="utf-8").strip() == ""):
            raise EvidenceError(f"legacy link command {index} CWD/content is invalid")
        try:
            link_text = command_copy.read_text(encoding="utf-8")
            link_argv = shlex.split(link_text)
        except (OSError, ValueError) as error:
            raise EvidenceError(f"legacy link command {index} cannot be parsed") from error
        link_objects, link_archives = _legacy_link_input_paths(link_argv, actual_cwd)
        if any(path not in object_records for path in link_objects):
            raise EvidenceError(f"legacy link command {index} names an unverified object")
        if any(path not in archive_records for path in link_archives):
            raise EvidenceError(f"legacy link command {index} names an unverified static archive")
        explicit_objects.update(link_objects)
        explicit_archives.update(link_archives)
        link_directories.add(actual_cwd)
        command_lines.update(line.strip() for line in link_text.splitlines() if line.strip())
        if link_row["target"] in link_commands_by_target:
            raise EvidenceError("legacy link command target is duplicated")
        link_commands_by_target[link_row["target"]] = (link_argv, actual_cwd)
    build_dir = Path(build["compile_commands"]["path"]).resolve().parent
    traced_archives = set()
    trace_count = 0
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    try:
        trace_lines = trace_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvidenceError(f"legacy linker trace log cannot be read: {error}") from error
    for line in trace_lines:
        line = ansi.sub("", line).strip()
        if line.startswith(("[", "make[", "gmake[")) or line in command_lines:
            continue
        match = re.fullmatch(r"(.+\.a)(?:\([^)]*\))?", line)
        if match is None:
            continue
        archive_name = Path(match.group(1))
        candidates = ([archive_name] if archive_name.is_absolute() else
                      [cwd / archive_name for cwd in link_directories] +
                      [build_dir / archive_name])
        resolved = {path.resolve() for path in candidates if path.is_file()}
        if len(resolved) != 1:
            raise EvidenceError(f"legacy linker-trace archive is missing or ambiguous: {line}")
        traced_archives.update(resolved)
        trace_count += 1
    if trace_count == 0:
        raise EvidenceError("legacy full-build log contains no actual linker archive trace")
    required_archives = explicit_archives | traced_archives
    if not required_archives <= set(archive_records):
        raise EvidenceError("legacy static archive inventory omits explicit or linker-traced inputs")
    build_archive_extras = {path for path in archive_records
                            if path.is_relative_to(build_dir)}
    if set(archive_records) != required_archives | build_archive_extras:
        raise EvidenceError("legacy static archive inventory has unexplained external archives")
    observed_compile_objects = set()
    for index, row in enumerate(compile_database):
        if not isinstance(row, dict) or not isinstance(row.get("directory"), str) or not isinstance(row.get("file"), str):
            raise EvidenceError(f"legacy compile database entry {index} is malformed")
        compile_argv = row.get("arguments")
        if compile_argv is None:
            if not isinstance(row.get("command"), str):
                raise EvidenceError(f"legacy compile database entry {index} has no command")
            try:
                compile_argv = shlex.split(row["command"])
            except ValueError as error:
                raise EvidenceError("legacy compile database command is malformed") from error
        if (not isinstance(compile_argv, list) or not compile_argv or
                not all(isinstance(token, str) for token in compile_argv) or
                any(token.startswith("@") for token in compile_argv) or
                any(token.startswith(("-fsanitize", "-fno-sanitize", "-shared-libasan"))
                    for token in compile_argv) or
                compile_argv.count("-o") != 1 or
                compile_argv.index("-o") + 1 >= len(compile_argv) or
                "-c" not in compile_argv):
            raise EvidenceError(f"legacy compile database entry {index} is not a direct Release object compile")
        compile_cwd = Path(row["directory"]).resolve()
        output_path = Path(compile_argv[compile_argv.index("-o") + 1])
        output_path = (compile_cwd / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
        source_path = Path(row["file"])
        source_path = (compile_cwd / source_path).resolve() if not source_path.is_absolute() else source_path.resolve()
        record = object_records.get(output_path)
        if record is None or record[0] != str(source_path):
            raise EvidenceError("legacy compile database source/-o differs from actual object inventory")
        actual_object_copy, actual_source_hash = find_copied(
            {"path": str(output_path), "sha256": record[1]}, f"compiled object {index}")
        if actual_source_hash != record[1] or sha256_file(actual_object_copy) != record[1]:
            raise EvidenceError("legacy compile object hash differs from staged output bytes")
        observed_compile_objects.add(output_path)
    if observed_compile_objects != set(object_records):
        raise EvidenceError("legacy compile database does not cover every actual object record")
    compiler_link = link_commands_by_target.get("ncnn-compile")
    if not isinstance(compiler_link, tuple):
        raise EvidenceError("legacy source does not record a unique ncnn-compile linker input")
    argv, cwd = compiler_link
    if argv.count("-o") != 1 or argv.index("-o") + 1 >= len(argv):
        raise EvidenceError("legacy compiler link command has no unique output")
    output = Path(argv[argv.index("-o") + 1])
    if not output.is_absolute():
        output = (cwd / output).resolve()
    if output != Path(compiler_record["path"]).resolve():
        raise EvidenceError("legacy compiler link -o differs from recorded compiler binary")
    linked_objects, linked_archives = _legacy_link_input_paths(argv, cwd)
    if (not any(path.name == "ncnn-compile.cpp.o" for path in linked_objects) or
            not any(path.name == "CompileSession.cpp.o" for path in linked_objects)):
        raise EvidenceError("legacy compiler link omits CompileSession/main objects")
    tool_summaries = {}
    for role, tool in tools.items():
        staged_path, staged_hash = _artifact(stage, tool["staged"],
                                              f"legacy tool summary {role}")
        tool_summaries[role] = {
            "tool_path": tool_paths[role].as_posix(),
            "sha256": staged_hash,
            "staged": {"path": staged_path.relative_to(stage).as_posix(),
                       "sha256": staged_hash},
        }
    return {"status": status, "qualified": True,
            "git_revision": LEGACY_REVISION, "source_fingerprint": fingerprint,
            "manifest": {"path": manifest_path.relative_to(stage).as_posix(),
                         "sha256": manifest_hash},
            "compiler": {"path": compiler_path.relative_to(stage).as_posix(),
                         "sha256": compiler_hash},
            "compiler_original_path": compiler_record["path"],
            "tools": tool_summaries}


def _audit_reference_runs(stage: Path, source_root: Path, fixture: dict,
                          output_refs: list[dict], legacy_reference: dict,
                          allow_missing_legacy: bool) -> dict:
    references = fixture.get("references")
    if not isinstance(references, dict):
        raise EvidenceError(f"{fixture.get('name')} reference build/run inventory is missing")
    name = fixture.get("name")
    independent_legacy = legacy_reference.get("qualified") is True
    if references.get("independent_legacy") is not independent_legacy:
        raise EvidenceError(f"{name} reference role differs from verified legacy source")
    reference_runs = references.get("runs")
    if (not isinstance(reference_runs, list) or len(reference_runs) != 2 or
            fixture.get("reference_runs") != reference_runs):
        raise EvidenceError(f"{name} CPU/legacy raw run rows are missing or inconsistent")
    expected_phases = {"cpu_aot_reference",
                       "legacy_vulkan_reference" if independent_legacy else
                       "current_generic_reference"}
    if {row.get("phase") for row in reference_runs if isinstance(row, dict)} != expected_phases:
        raise EvidenceError(f"{name} CPU/legacy reference run roles differ")

    cpu = references.get("cpu_aot")
    if not isinstance(cpu, dict):
        raise EvidenceError(f"{name} CPU AOT compile/run identity is missing")
    cpu_library, cpu_hash = _artifact(stage, cpu.get("library"), f"{name} CPU AOT DSO")
    if cpu_hash != cpu.get("library_sha256"):
        raise EvidenceError(f"{name} CPU AOT DSO hash differs from its build record")
    _artifact(stage, cpu.get("manifest"), f"{name} CPU AOT manifest")
    cpu_loader, cpu_loader_hash = _artifact(stage, cpu.get("loader"),
                                            f"{name} CPU native loader")
    profile = _profile_from_stage(stage)
    cpu_library_dynamic_dependencies = _audit_shared_asan_needed(
        stage, cpu_library, profile, f"{name} CPU AOT DSO")
    cpu_loader_build = cpu.get("loader_build")
    if not isinstance(cpu_loader_build, dict):
        raise EvidenceError(f"{name} CPU native loader compile provenance is missing")
    built_loader, built_loader_hash = _artifact(
        stage, cpu_loader_build.get("executable"), f"{name} compiled CPU native loader")
    if (built_loader != cpu_loader or built_loader_hash != cpu_loader_hash or
            fixture.get("native_loader_build") != cpu_loader_build):
        raise EvidenceError(f"{name} CPU loader binary/build provenance differs")
    loader_source, loader_source_hash = _artifact(
        stage, cpu_loader_build.get("source"), f"{name} CPU loader source snapshot")
    expected_loader_source = (source_root / "test/Vulkan/native/fixedrun_native_loader.cpp").resolve()
    if (cpu_loader_build.get("source_target_path") != str(expected_loader_source) or
            cpu_loader_build.get("source_target_sha256") != loader_source_hash):
        raise EvidenceError(f"{name} CPU loader target path/hash differs from immutable source snapshot")
    loader_compile_argv = cpu_loader_build.get("argv")
    if not isinstance(loader_compile_argv, list) or not all(
            isinstance(token, str) for token in loader_compile_argv):
        raise EvidenceError(f"{name} CPU loader compile argv is malformed")
    profile = _profile_from_stage(stage)
    if profile in {"release", "sanitize"}:
        configured_cxx = _configured_gate_tool(
            stage, "CMAKE_CXX_COMPILER", "cxx-compiler")
        if Path(loader_compile_argv[0]).resolve() != configured_cxx.resolve():
            raise EvidenceError(f"{name} CPU native loader uses a different gate C++ compiler")
    _audit_no_forced_compiler_inputs(
        loader_compile_argv, f"{name} CPU native loader compile/link", profile,
        link=True)
    loader_compile_cwd = cpu_loader_build.get("cwd")
    if not isinstance(loader_compile_cwd, str):
        raise EvidenceError(f"{name} CPU loader compile CWD is malformed")
    loader_execution = _audit_logged_command(
        stage, cpu_loader_build, f"{name} CPU native loader compile")
    if (loader_execution["argv"] != loader_compile_argv or
            Path(loader_execution["cwd"]).resolve() != Path(loader_compile_cwd).resolve()):
        raise EvidenceError(f"{name} CPU loader execution differs from its source command")
    output_index = (loader_compile_argv.index("-o") + 1
                    if loader_compile_argv.count("-o") == 1 else len(loader_compile_argv))
    loader_source_tokens = [token for token in loader_compile_argv
                            if Path(token).suffix in {".cpp", ".cc", ".cxx"}]
    target_path = cpu_loader_build.get("executable_target_path")
    target_hash = cpu_loader_build.get("executable_target_sha256")
    target_output = (_command_path(stage, loader_compile_argv[output_index], loader_compile_cwd,
                                   f"{name} CPU loader output")
                     if output_index < len(loader_compile_argv) else None)
    if ("-DNCNN_FIXEDRUN_BACKEND_VULKAN=0" not in loader_compile_argv or
            f"-DNCNN_FIXEDRUN_MODEL={name}" not in loader_compile_argv or
            "-Wl,-z,defs" not in loader_compile_argv or
            Path(loader_compile_cwd).resolve() != (stage / "test/Vulkan").resolve() or
            len(loader_source_tokens) != 1 or
            Path(loader_source_tokens[0]).resolve() != expected_loader_source or
            target_output is None or not isinstance(target_path, str) or
            target_output != Path(target_path).resolve() or
            not isinstance(target_hash, str) or not SHA256_RE.fullmatch(target_hash) or
            target_hash != _artifact(stage, cpu_loader_build.get("executable"),
                                     f"{name} CPU loader output snapshot")[1]):
        raise EvidenceError(f"{name} CPU loader build is not its strict source-bound CPU-only target")
    if (cpu_loader_build.get("source_target_path") != str(expected_loader_source) or
            cpu_loader_build.get("source_target_sha256") != sha256_file(expected_loader_source)):
        raise EvidenceError(f"{name} CPU loader compiler source path/hash differs from checked-in source")
    cpu_loader_dynamic_dependencies = _audit_shared_asan_needed(
        stage, cpu_loader, profile, f"{name} CPU native loader")

    cpu_compile = cpu.get("compile")
    if not isinstance(cpu_compile, dict):
        raise EvidenceError(f"{name} CPU AOT compile provenance is missing")
    cache = _stage_cache_values(stage)
    current_tools = {"driver": str(stage / "tools/ncnn-mlir-driver"),
                     "opt": str(stage / "bin/ncnn-mlir-opt"),
                     "translate": cache.get("MLIR_TRANSLATE", ""),
                     "clang": cache.get("CLANG", ""),
                     "nm": cache.get("LLVM_NM", ""),
                     "readelf": cache.get("LLVM_READELF", ""),
                     "llvm-as": cache.get("LLVM_AS", "")}
    _audit_reference_compile(stage, cpu_compile, source_root, fixture, "cpu",
                             stage / "tools/ncnn-compile", current_tools)
    legacy_compile = references.get("legacy_vulkan_compile")
    if legacy_compile != fixture.get("legacy_vulkan_compile"):
        raise EvidenceError(f"{name} old Vulkan compile provenance differs across schema locations")
    if independent_legacy:
        legacy_compiler = legacy_reference.get("compiler_original_path")
        _audit_reference_compile(stage, legacy_compile, source_root, fixture,
                                 "vulkan", Path(legacy_compiler),
                                 legacy_reference.get("tools"))
        legacy_library, legacy_hash = _artifact(
            stage, legacy_compile.get("library"), f"{name} frozen old Vulkan DSO")
        if legacy_hash != legacy_compile.get("library_sha256"):
            raise EvidenceError(f"{name} old Vulkan DSO hash differs")
        legacy_info = legacy_compile.get("compiler")
        if (not isinstance(legacy_info, dict) or
                legacy_info.get("path") != legacy_compiler or
                legacy_info.get("sha256") != legacy_reference.get("compiler", {}).get("sha256")):
            raise EvidenceError(f"{name} old Vulkan DSO was not built by the pinned compiler")
    else:
        if not allow_missing_legacy or legacy_compile.get("reference_status") != "not-supplied":
            raise EvidenceError(f"{name} unqualified old Vulkan observation has invalid role")
        legacy_library, legacy_hash = _artifact(
            stage, legacy_compile.get("library"), f"{name} current generic observation DSO")
        if legacy_compile.get("compiler", {}).get("path") != str((stage / "tools/ncnn-compile").resolve()):
            raise EvidenceError(f"{name} generic observation is not from current gate compiler")

    fixture_inputs = fixture.get("inputs")
    fixture_outputs = fixture.get("outputs")
    if not isinstance(fixture_inputs, list) or not isinstance(fixture_outputs, list):
        raise EvidenceError(f"{name} fixture public I/O evidence is malformed")
    expected_input_refs = []
    for index, item in enumerate(fixture_inputs):
        path, digest = _artifact(stage, item.get("raw"), f"{name} reference input {index}")
        if item.get("shape") != FIXTURE_IO[name][0][index]:
            raise EvidenceError(f"{name} reference input shape differs")
        expected_input_refs.append((path, digest))
    expected_output_refs = {}
    for phase in expected_phases:
        role = {"cpu_aot_reference": "cpu_aot_raw",
                "legacy_vulkan_reference": "legacy_vulkan_raw",
                "current_generic_reference": "current_generic_raw"}[phase]
        if phase == "current_generic_reference" and fixture_outputs[0].get(role) is None:
            raise EvidenceError(f"{name} current generic output bytes are missing")
        values = []
        for index, item in enumerate(fixture_outputs):
            record = item.get(role)
            if record is None:
                raise EvidenceError(f"{name} {phase} output {index} is missing")
            path, digest = _artifact(stage, record, f"{name} {phase} baseline {index}")
            values.append((path, digest))
        expected_output_refs[phase] = values

    for run in reference_runs:
        phase = run["phase"]
        expected_library = cpu_library if phase == "cpu_aot_reference" else legacy_library
        expected_library_hash = cpu_hash if phase == "cpu_aot_reference" else legacy_hash
        expected_loader = cpu_loader if phase == "cpu_aot_reference" else None
        if phase != "cpu_aot_reference":
            candidate = fixture.get("candidate_build")
            if not isinstance(candidate, dict):
                raise EvidenceError(f"{name} candidate loader identity is missing")
            expected_loader, _ = _artifact(stage, candidate.get("native_loader"),
                                            f"{name} Vulkan loader")
        if (run.get("backend") != ("native-reference") or
                run.get("dso_sha256") != expected_library_hash or
                type(run.get("status")) is not int or run.get("status") != 0 or
                run.get("diagnostic") not in (None, "") or
                _exact_int(run.get("validation_error_count"), "reference validation errors") != 0):
            raise EvidenceError(f"{name} {phase} run role/status/DSO identity is invalid")
        _exact_int(run.get("elapsed_ns"), "reference elapsed nanoseconds", 1)
        shapes = run.get("input_shapes")
        out_shapes = run.get("output_shapes")
        if shapes != FIXTURE_IO[name][0] or out_shapes != FIXTURE_IO[name][1]:
            raise EvidenceError(f"{name} {phase} run shapes differ from fixture")
        run_inputs = run.get("inputs")
        run_outputs = run.get("outputs")
        if not isinstance(run_inputs, list) or not isinstance(run_outputs, list) or len(run_inputs) != len(expected_input_refs) or len(run_outputs) != len(expected_output_refs[phase]):
            raise EvidenceError(f"{name} {phase} run tensor inventory is incomplete")
        for index, (record, shape, expected_ref) in enumerate(zip(
                run_inputs, shapes, expected_input_refs)):
            values, ref = _float_array(stage, record, _shape_elements(shape, "reference input shape"),
                                       f"{name} {phase} run input {index}")
            if ref["sha256"] != expected_ref[1]:
                raise EvidenceError(f"{name} {phase} input bytes differ from independent fixture input")
            del values
        for index, (record, shape, expected_ref) in enumerate(zip(
                run_outputs, FIXTURE_IO[name][1], expected_output_refs[phase])):
            values, ref = _float_array(stage, record, _shape_elements(shape, "reference output shape"),
                                       f"{name} {phase} run output {index}")
            expected_values, _ = _float_array(
                stage, fixture_outputs[index].get("golden_raw"), len(values),
                f"{name} {phase} independent scalar golden {index}")
            _compare(values, expected_values, 1e-6, 1e-6,
                     f"{name} {phase} output versus independent golden {index}")
            if ref["sha256"] != expected_ref[1]:
                raise EvidenceError(f"{name} {phase} output differs from its fixture baseline raw bytes")
            del values, expected_values
        _audit_reference_loader(stage, run, phase, expected_loader,
                                expected_library, expected_library_hash)
    return {"cpu_aot": "verified", "legacy_vulkan":
            "verified" if independent_legacy else "not-supplied",
            "current_generic": "not-a-legacy-reference" if not independent_legacy else None,
            "cpu_aot_runtime_linkage": {
                "library": cpu_library_dynamic_dependencies,
                "loader": cpu_loader_dynamic_dependencies}}


def _profile_from_stage(stage: Path) -> str:
    state_path = stage / "gate-state.json"
    if not state_path.is_file():
        return "development"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    profile = state.get("profile") if isinstance(state, dict) else None
    if profile not in {"release", "sanitize", "offline", "software", "hardware"}:
        raise EvidenceError("gate profile is invalid for fixed-run audit")
    return profile


def _audit_reference_compile(stage: Path, record: object, source_root: Path,
                             fixture: dict, backend: str,
                             expected_compiler: Path,
                             expected_tools: object) -> None:
    if (not isinstance(record, dict) or type(record.get("returncode")) is not int or
            record.get("returncode") != 0):
        raise EvidenceError(f"{fixture.get('name')} {backend} compile did not succeed")
    argv, cwd = record.get("argv"), record.get("cwd")
    if (not isinstance(argv, list) or not argv or
            not all(isinstance(token, str) for token in argv) or
            not isinstance(cwd, str) or
            Path(cwd).resolve() != (stage / "test/Vulkan").resolve() or
            Path(argv[0]).resolve() != expected_compiler.resolve()):
        raise EvidenceError(f"{fixture.get('name')} {backend} compile argv/cwd is invalid")
    command_streams = _audit_logged_command(
        stage, record, f"{fixture.get('name')} {backend} model compile")
    name = fixture["name"]
    param_source = source_root / fixture["param_source"]
    param_copy, _ = _artifact(stage, fixture.get("param"), f"{name} compile .param")
    bin_copy, _ = _artifact(stage, fixture.get("bin"), f"{name} compile .bin")
    if (str(param_source.resolve()) not in
            {str(Path(token).resolve()) for token in argv if not token.startswith("-")} or
            "--bin" not in argv or argv.index("--bin") + 1 >= len(argv) or
            not Path(argv[argv.index("--bin") + 1]).is_file() or
            sha256_file(Path(argv[argv.index("--bin") + 1])) != sha256_file(bin_copy) or
            "--model-name" not in argv or argv[argv.index("--model-name") + 1] != name or
            "--verify-execution" not in argv):
        raise EvidenceError(f"{name} {backend} compile does not bind source param/bin/model")
    if backend == "cpu":
        if "--backend=vulkan" in argv or "--emit-manifest" not in argv:
            raise EvidenceError(f"{name} CPU reference is not the current CPU AOT product compile")
        _require_profile_instrumentation(argv, _profile_from_stage(stage),
                                         f"{name} current CPU AOT compile/link",
                                         link=True)
        _audit_forwarded_link_sanitizer_args(
            argv, _profile_from_stage(stage), f"{name} current CPU AOT compile/link")
        option_roles = {"--driver": "driver", "--opt": "opt", "--translate": "translate",
                        "--clang": "clang", "--nm": "nm", "--readelf": "readelf",
                        "--llvm-as": "llvm-as"}
    else:
        if "--backend=vulkan" not in argv:
            raise EvidenceError(f"{name} old Vulkan reference compile is not the pinned generic backend")
        option_roles = {"--driver": "driver", "--opt": "opt", "--nm": "nm",
                        "--readelf": "readelf", "--vulkan-mlir-opt": "mlir-opt",
                        "--vulkan-spirv-val": "spirv-val", "--vulkan-spirv-dis": "spirv-dis",
                        "--vulkan-spirv-extract": "spirv-extract",
                        "--vulkan-host-cxx": "host-cxx"}
    if not isinstance(expected_tools, dict):
        raise EvidenceError(f"{name} {backend} tool inventory is missing")
    expected_tool_paths = {}
    for option, role in option_roles.items():
        tool = expected_tools.get(role)
        tool_path = (tool.get("tool_path", tool.get("path"))
                     if isinstance(tool, dict) else tool)
        if not isinstance(tool_path, str) or not tool_path:
            raise EvidenceError(f"{name} {backend} compile omits pinned {role} identity")
        if option not in argv or argv.index(option) + 1 >= len(argv) or \
                Path(argv[argv.index(option) + 1]).resolve() != Path(tool_path).resolve():
            raise EvidenceError(f"{name} {backend} compile uses a different {role} tool")
        expected_tool_paths[role] = Path(tool_path).resolve()
    if backend == "vulkan":
        if "-v" not in argv:
            raise EvidenceError(f"{name} legacy Vulkan compile omits verbose child-command capture")
        observed_tools = set()
        try:
            stderr_text = command_streams["stderr_bytes"].decode("utf-8", "strict")
        except UnicodeError as error:
            raise EvidenceError(f"{name} legacy Vulkan command log is not UTF-8") from error
        for line in stderr_text.splitlines():
            try:
                child_argv = shlex.split(line)
            except ValueError:
                continue
            if not child_argv:
                continue
            executable = Path(child_argv[0]).resolve()
            for role, tool_path in expected_tool_paths.items():
                if executable == tool_path:
                    observed_tools.add(role)
        missing_tools = set(expected_tool_paths) - observed_tools
        if missing_tools:
            raise EvidenceError(
                f"{name} legacy Vulkan log omits actual child invocations for {sorted(missing_tools)}")
    if backend == "cpu":
        _artifact(stage, record.get("output_dir"), f"{name} CPU AOT generated header")
    else:
        for key in ("library", "header", "manifest", "plan"):
            _artifact(stage, record.get(key), f"{name} {backend} compile {key}")
        if record.get("library_sha256") != _artifact(
                stage, record.get("library"), f"{name} {backend} compile DSO")[1]:
            raise EvidenceError(f"{name} {backend} compiled DSO hash is invalid")


def _audit_reference_loader(stage: Path, run: dict, phase: str,
                            expected_loader: Path, library: Path,
                            library_hash: str) -> None:
    native = run.get("native_loader")
    if not isinstance(native, dict):
        raise EvidenceError(f"{phase} native C++ loader execution record is missing")
    executable, executable_hash = _artifact(stage, native.get("executable"),
                                           f"{phase} native loader executable")
    if executable != expected_loader:
        raise EvidenceError(f"{phase} executed loader differs from its actual build target")
    argv, cwd = native.get("argv"), native.get("cwd")
    if (not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv) or
            not isinstance(cwd, str) or
            Path(cwd).resolve() != (stage / "test/Vulkan").resolve() or
            type(native.get("returncode")) is not int or
            native["returncode"] != 0 or Path(argv[0]).resolve() != executable or
            "--mode" not in argv or argv[argv.index("--mode") + 1] != "single" or
            "--library" not in argv or argv[argv.index("--library") + 1] != str(library) or
            "--input" not in argv or "--output" not in argv):
        raise EvidenceError(f"{phase} loader argv/exit does not bind a single-run reference DSO")
    log, _ = _artifact(stage, native.get("log"), f"{phase} native loader raw log")
    log_bytes = log.read_bytes()
    stdout, separator, stderr = log_bytes.partition(b"\n--- stderr ---\n")
    if (not separator or hashlib.sha256(stdout).hexdigest() != native.get("stdout_sha256") or
            hashlib.sha256(stderr).hexdigest() != native.get("stderr_sha256")):
        raise EvidenceError(f"{phase} native loader stdout/stderr hashes differ from raw log")
    try:
        events = [json.loads(line) for line in stdout.decode("utf-8", "strict").splitlines()
                  if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{phase} native loader output is malformed") from error
    measurements = [event for event in events if event.get("event") == "run"]
    lifecycle = [event for event in events if event.get("event") == "lifecycle"]
    cpu_loader = phase == "cpu_aot_reference"
    if len(lifecycle) == 1:
        for field in ("create_count", "run_count", "destroy_count"):
            _exact_int(lifecycle[0].get(field), f"{phase} lifecycle {field}")
        for field in ("create_statuses", "run_statuses", "destroy_statuses"):
            statuses = lifecycle[0].get(field)
            if not isinstance(statuses, list):
                raise EvidenceError(f"{phase} lifecycle {field} is malformed")
            for value in statuses:
                _exact_int(value, f"{phase} lifecycle {field} value")
    if (len(measurements) != 1 or len(lifecycle) != 1 or
            lifecycle[0].get("create_count") != (0 if cpu_loader else 1) or
            lifecycle[0].get("destroy_count") != (0 if cpu_loader else 1) or
            lifecycle[0].get("run_count") != 1 or
            lifecycle[0].get("create_statuses") != ([] if cpu_loader else [0]) or
            lifecycle[0].get("run_statuses") != [0] or
            lifecycle[0].get("destroy_statuses") != ([] if cpu_loader else [0])):
        raise EvidenceError(f"{phase} native loader lifecycle trace is invalid")
    measurement = measurements[0]
    if (type(measurement.get("status")) is not int or
            type(measurement.get("elapsed_ns")) is not int or
            measurement.get("status") != run.get("status") or
            measurement.get("elapsed_ns") != run.get("elapsed_ns") or
            measurement.get("diagnostic") != run.get("diagnostic") or
            measurement.get("output_paths") != [str(_artifact(stage, item, "reference output")[0])
                                                   for item in run.get("outputs", [])]):
        raise EvidenceError(f"{phase} native measurement differs from raw result row")
    expected_inputs = [str(_artifact(stage, item, "reference input")[0])
                       for item in run.get("inputs", [])]
    actual_inputs = [Path(argv[index + 1]).resolve() for index, value in enumerate(argv[:-1])
                     if value == "--input"]
    if actual_inputs != [Path(value).resolve() for value in expected_inputs]:
        raise EvidenceError(f"{phase} native loader inputs differ from raw tensors")
    expected_outputs = [str(_artifact(stage, item, "reference output")[0])
                        for item in run.get("outputs", [])]
    actual_outputs = [Path(argv[index + 1]).resolve() for index, value in enumerate(argv[:-1])
                      if value == "--output"]
    if actual_outputs != [Path(value).resolve() for value in expected_outputs]:
        raise EvidenceError(f"{phase} native loader outputs differ from raw tensors")
    dso_path = Path(argv[argv.index("--library") + 1]).resolve()
    if dso_path != library.resolve() or library_hash != run.get("dso_sha256"):
        raise EvidenceError(f"{phase} loader DSO bytes differ from compiled reference")
    if executable_hash != _artifact(stage, native.get("executable"), "reference loader")[1]:
        raise EvidenceError(f"{phase} native loader executable hash differs")


def _audit_fixture(stage: Path, source_root: Path, fixture: dict, name: str, *,
                   allow_missing_legacy: bool,
                   legacy_reference: dict) -> dict:
    if not isinstance(fixture, dict) or fixture.get("name") != name:
        raise EvidenceError(f"fixture {name} identity is invalid")
    expected_inputs, expected_outputs = FIXTURE_IO[name]
    inputs, outputs = fixture.get("inputs"), fixture.get("outputs")
    if (not isinstance(inputs, list) or not isinstance(outputs, list) or
            [row.get("shape") for row in inputs if isinstance(row, dict)] != expected_inputs or
            len([row for row in inputs if isinstance(row, dict)]) != len(inputs) or
            [row.get("shape") for row in outputs if isinstance(row, dict)] != expected_outputs or
            len([row for row in outputs if isinstance(row, dict)]) != len(outputs)):
        raise EvidenceError(f"{name} public shape/rank/port coverage differs from the required matrix")
    source_params = {
        "relu": "test/Vulkan/relu.param",
        "add_relu": "test/Vulkan/add_relu.param",
        "conv_relu": "test/Vulkan/fixedrun_conv_relu.param",
        "add_split_relu": "test/Vulkan/add_split_relu.param",
        "split_concat": "test/Numerical/fixtures/split_three.param",
        "reshape": "test/Vulkan/fixedrun_reshape_conv.param",
        "transpose": "test/Vulkan/fixedrun_transpose.param",
    }
    param_source = (source_root / source_params[name]).resolve()
    if not param_source.is_file():
        raise EvidenceError(f"source-root parameter fixture is missing: {source_params[name]}")
    param, param_hash = _artifact(stage, fixture.get("param"), f"{name} .param")
    if sha256_file(param_source) != param_hash:
        raise EvidenceError(f"{name} staged param bytes differ from source-root fixture")
    bin_path, bin_hash = _artifact(stage, fixture.get("bin"), f"{name} .bin")
    if name in {"conv_relu", "reshape"}:
        expected_bin = struct.pack("<I6f", 0, 0.5, -0.25, -0.75, 1.25, 0.125, -0.5)
        generator = (source_root / "test/Vulkan/generate_fixedrun_conv_bin.py").resolve()
        generator_copy, generator_hash = _artifact(
            stage, fixture.get("bin_generator"), f"{name} deterministic bin generator")
        if (generator_copy.read_bytes() != generator.read_bytes() or
                generator_hash != sha256_file(generator) or
                fixture.get("bin_source") != "generated-at-build-time" or
                bin_path.read_bytes() != expected_bin):
            raise EvidenceError(f"{name} bin/generator differs from deterministic 1x1 Conv weights")
    elif bin_path.read_bytes() != b"" or fixture.get("bin_generator") is not None or \
            fixture.get("bin_source") != "generated-empty-build-input":
        raise EvidenceError(f"{name} should have only generated empty model-bin bytes")
    input_shapes = [item["shape"] for item in inputs]
    output_shapes = [item["shape"] for item in outputs]
    if not isinstance(inputs, list) or len(inputs) != len(input_shapes) or not isinstance(outputs, list) or len(outputs) != len(output_shapes):
        raise EvidenceError(f"{name} public I/O raw inventory is incomplete")
    input_values, input_refs = [], []
    for index, (record, shape) in enumerate(zip(inputs, input_shapes)):
        elements = _shape_elements(shape, f"{name} input {index} shape")
        values, ref = _float_array(stage, record.get("raw"), elements, f"{name} input {index}")
        if record.get("shape") != shape or _exact_int(record.get("elements"), "input element count") != elements:
            raise EvidenceError(f"{name} input {index} shape/count differs")
        input_values.append(values)
        input_refs.append(ref)
    scalar_golden = _scalar_golden(name, input_values, bin_path.read_bytes())
    if len(scalar_golden) != len(outputs):
        raise EvidenceError(f"{name} scalar oracle output count differs")
    output_refs = []
    legacy_compile = fixture.get("legacy_vulkan_compile")
    independent_legacy = legacy_reference.get("qualified") is True
    if independent_legacy:
        if not isinstance(legacy_compile, dict) or type(legacy_compile.get("returncode")) is not int or legacy_compile["returncode"] != 0:
            raise EvidenceError(f"{name} independently rebuilt legacy compile evidence is missing")
        legacy_argv = legacy_compile.get("argv")
        legacy_command = legacy_reference.get("compiler_original_path")
        if (not isinstance(legacy_argv, list) or not legacy_argv or
                Path(legacy_argv[0]).resolve() != Path(legacy_command).resolve() or
                "--backend=vulkan" not in legacy_argv or "--verify-execution" not in legacy_argv or
                "--bin" not in legacy_argv or "--model-name" not in legacy_argv or
                name not in legacy_argv):
            raise EvidenceError(f"{name} legacy compile is not the bound old Vulkan product command")
        legacy_dso = legacy_compile.get("library")
        _, legacy_dso_hash = _artifact(stage, legacy_dso, f"{name} legacy model DSO")
        if legacy_dso_hash != legacy_compile.get("library_sha256"):
            raise EvidenceError(f"{name} legacy DSO differs from old compile record")
        compiler_info = legacy_compile.get("compiler")
        if (not isinstance(compiler_info, dict) or
                compiler_info.get("sha256") != legacy_reference.get("compiler", {}).get("sha256")):
            raise EvidenceError(f"{name} old compile uses a different legacy compiler")
        for artifact_key in ("header", "manifest", "plan"):
            _artifact(stage, legacy_compile.get(artifact_key),
                      f"{name} old Vulkan {artifact_key}")
    else:
        if fixture.get("legacy_vulkan_raw") is not None:
            raise EvidenceError(f"{name} unqualified legacy path must not masquerade as old-source output")
    for index, (record, shape, expected) in enumerate(zip(outputs, output_shapes, scalar_golden)):
        if not isinstance(record, dict) or record.get("shape") != shape:
            raise EvidenceError(f"{name} output {index} shape differs")
        count = _shape_elements(shape, f"{name} output {index} shape")
        if len(expected) != count or _exact_int(record.get("elements"), "output element count") != count:
            raise EvidenceError(f"{name} scalar oracle/output shape differs")
        golden, golden_ref = _float_array(stage, record.get("golden_raw"), count,
                                          f"{name} independent golden {index}")
        _compare(golden, expected, 1e-6, 1e-6, f"{name} golden versus scalar oracle {index}")
        role_refs, metrics = {"golden": golden_ref}, {}
        roles = ["cpu_aot_raw", "candidate_raw"] + (["legacy_vulkan_raw"] if independent_legacy else [])
        for role in roles:
            values, ref = _float_array(stage, record.get(role), count, f"{name} {role} {index}")
            role_refs[role.removesuffix("_raw")] = ref
            metrics[role.removesuffix("_raw")] = _compare(
                values, expected, 1e-6, 1e-6, f"{name} {role} versus scalar oracle {index}")
            del values
        if not independent_legacy and record.get("current_generic_raw") is not None:
            generic, generic_ref = _float_array(stage, record["current_generic_raw"], count,
                                                f"{name} unqualified current generic {index}")
            metrics["current_generic_unqualified"] = _compare(
                generic, expected, 1e-6, 1e-6,
                f"{name} current generic versus independent scalar golden {index}")
            role_refs["current_generic_unqualified"] = generic_ref
            del generic
        output_refs.append({"shape": shape, "elements": count,
                            "raw": role_refs, "recomputed": metrics})
        del golden
    reference_summary = _audit_reference_runs(
        stage, source_root, fixture, output_refs, legacy_reference, allow_missing_legacy)
    return {"name": name,
            "param": {"path": param.relative_to(stage).as_posix(), "sha256": param_hash},
            "bin": {"path": bin_path.relative_to(stage).as_posix(), "sha256": bin_hash},
            "inputs": input_refs, "outputs": output_refs,
            "legacy_vulkan_compile": legacy_compile,
            "legacy_vulkan_dso": fixture.get("legacy_vulkan_dso"),
            "references": reference_summary}


def _audit_loader_run(stage: Path, run: dict, fixture: str,
                      profile: str, candidate: dict) -> list[dict]:
    native = run.get("native_loader")
    if not isinstance(native, dict):
        raise EvidenceError("native C++ C ABI loader execution record is missing")
    executable, exe_hash = _artifact(stage, native.get("executable"),
                                     "executed native C++ loader")
    expected_loader = candidate["native_loader"]
    if (exe_hash != expected_loader["sha256"] or
            executable.relative_to(stage).as_posix() != expected_loader["path"]):
        raise EvidenceError("executed loader differs from candidate's real CMake loader target")
    argv, cwd = native.get("argv"), native.get("cwd")
    if (not isinstance(argv, list) or not argv or
            not all(isinstance(token, str) for token in argv) or not isinstance(cwd, str) or
            Path(cwd).resolve() != (stage / "test/Vulkan").resolve() or
            type(native.get("returncode")) is not int or native["returncode"] != 0):
        raise EvidenceError("native loader argv/cwd/exit code is invalid")
    if _command_path(stage, argv[0], cwd, "executed native loader") != executable:
        raise EvidenceError("native loader argv does not execute the recorded binary")
    if type(native.get("stdout_sha256")) is not str or not SHA256_RE.fullmatch(native["stdout_sha256"]):
        raise EvidenceError("native loader stdout hash is malformed")
    if type(native.get("stderr_sha256")) is not str or not SHA256_RE.fullmatch(native["stderr_sha256"]):
        raise EvidenceError("native loader stderr hash is malformed")
    log, log_hash = _artifact(stage, native.get("log"), "native loader raw log")
    log_bytes = log.read_bytes()
    output, separator, error_output = log_bytes.partition(b"\n--- stderr ---\n")
    if (not separator or hashlib.sha256(output).hexdigest() != native["stdout_sha256"] or
            hashlib.sha256(error_output).hexdigest() != native["stderr_sha256"]):
        raise EvidenceError("native loader log bytes differ from captured stdout/stderr hashes")
    events = []
    for line in output.decode("utf-8", errors="strict").splitlines():
        if line.strip():
            events.append(json.loads(line))
    measurements = [event for event in events if event.get("event") == "run"]
    lifecycle = [event for event in events if event.get("event") == "lifecycle"]
    if len(lifecycle) != 1:
        raise EvidenceError("native C++ loader did not record exactly one lifecycle summary")
    expected_contexts = 2 if run.get("phase") == "interleaved_contexts" else 1
    trace = lifecycle[0]
    for field in ("create_count", "run_count", "destroy_count"):
        _exact_int(trace.get(field), f"native lifecycle {field}")
    create_statuses, destroy_statuses = trace.get("create_statuses"), trace.get("destroy_statuses")
    expected_measurement_count = (4 if run.get("phase") == "interleaved_contexts" else 2)
    run_statuses = trace.get("run_statuses")
    for label, statuses in (("create", create_statuses), ("destroy", destroy_statuses),
                            ("run", run_statuses)):
        if isinstance(statuses, list):
            for status in statuses:
                _exact_int(status, f"native lifecycle {label} status")
    if (trace.get("create_count") != expected_contexts or
            trace.get("destroy_count") != expected_contexts or
            trace.get("run_count") != expected_measurement_count or
            not isinstance(create_statuses, list) or len(create_statuses) != expected_contexts or
            not isinstance(destroy_statuses, list) or len(destroy_statuses) != expected_contexts or
            not isinstance(run_statuses, list) or len(run_statuses) != expected_measurement_count or
            any(status != 0 for status in create_statuses + destroy_statuses + run_statuses) or
            native.get("lifecycle") != trace):
        raise EvidenceError("native C++ loader lifecycle create/run/destroy trace is invalid")
    phase_mode = {"same_seed_repeat": "repeat", "different_input": "repeat",
                  "same_context_concurrent": "concurrent",
                  "interleaved_contexts": "interleaved"}[run["phase"]]
    if "--mode" not in argv or argv[argv.index("--mode") + 1] != phase_mode:
        raise EvidenceError("native loader mode does not match concurrency/repeat phase")
    if "--library" not in argv:
        raise EvidenceError("native loader did not receive a candidate model library")
    library_index = argv.index("--library") + 1
    library, library_hash = _artifact(
        stage, {"path": Path(argv[library_index]).resolve().relative_to(stage.resolve()).as_posix(),
                "sha256": run.get("dso_sha256")}, "native loader's candidate DSO")
    if library_hash != candidate["link"]["dso"]["sha256"]:
        raise EvidenceError("native loader loaded different candidate DSO bytes")
    if run.get("backend") != "fixedrun-candidate":
        raise EvidenceError("candidate execution was not performed by the model-specific native DSO")
    expected_measurement_count = (4 if phase_mode == "interleaved" else
                                  2 if phase_mode in {"repeat", "concurrent"} else 1)
    # `run_and_record` records each measurement emitted by the C++ process. The
    # same full native log is referenced by sibling rows; rows are validated
    # individually against the exact result and output bytes below.
    if len(measurements) != expected_measurement_count:
        raise EvidenceError("native loader did not emit the expected C ABI invocation count")
    index = _exact_int(run.get("loader_result_index"), "native loader measurement index")
    if index >= len(measurements):
        raise EvidenceError("native loader measurement index is out of range")
    measurement = measurements[index]
    if (type(measurement.get("status")) is not int or
            type(measurement.get("elapsed_ns")) is not int or
            measurement.get("status") != run.get("status") or
            measurement.get("elapsed_ns") != run.get("elapsed_ns") or
            measurement.get("diagnostic") != run.get("diagnostic")):
        raise EvidenceError("native C ABI measurement differs from its raw run row")
    output_artifacts = run.get("outputs")
    measured_outputs = measurement.get("output_paths")
    if not isinstance(output_artifacts, list) or not isinstance(measured_outputs, list):
        raise EvidenceError("native loader output paths are missing")
    output_option = ("--output" if phase_mode == "single" else
                     "--output-a" if phase_mode == "repeat" and index == 0 else
                     "--output-b" if phase_mode == "repeat" else
                     "--output-a" if phase_mode == "concurrent" and index == 0 else
                     "--output-b" if phase_mode == "concurrent" else
                     "--output-a" if index % 2 == 0 else "--output-b")
    expected_outputs = [_artifact(stage, item, "run output")[0]
                        for item in output_artifacts]
    measured_outputs_for_call = []
    for arg_index, token in enumerate(argv[:-1]):
        if token == output_option:
            path = Path(argv[arg_index + 1]).resolve()
            if phase_mode == "interleaved" and index >= 2:
                path = path.with_name(path.stem + ".interleaved-1" + path.suffix)
            measured_outputs_for_call.append(path)
    if (measured_outputs_for_call != expected_outputs or
            [Path(path).resolve() for path in measured_outputs] != expected_outputs):
        raise EvidenceError("C++ loader output paths differ from raw tensor artifacts")
    input_artifacts = run.get("inputs")
    if not isinstance(input_artifacts, list):
        raise EvidenceError("native loader input tensor paths are missing")
    input_option = ("--input" if phase_mode == "single" else
                    "--input-a" if phase_mode == "repeat" and index == 0 else
                    "--input-b" if phase_mode == "repeat" else
                    "--input-a" if phase_mode == "concurrent" and index == 0 else
                    "--input-b" if phase_mode == "concurrent" else
                    "--input-a" if index % 2 == 0 else "--input-b")
    measured_inputs = []
    for arg_index, token in enumerate(argv[:-1]):
        if token == input_option:
            measured_inputs.append(Path(argv[arg_index + 1]).resolve())
    expected_inputs = [_artifact(stage, item, "run input")[0]
                       for item in input_artifacts]
    if measured_inputs != expected_inputs:
        raise EvidenceError("C++ loader input paths differ from raw tensor artifacts")
    return [{"path": executable.relative_to(stage).as_posix(), "sha256": exe_hash,
             "role": "executed native C++ C ABI loader"},
            {"path": log.relative_to(stage).as_posix(), "sha256": log_hash,
             "role": "native C++ loader stdout/stderr"},
            {"path": library.relative_to(stage).as_posix(), "sha256": library_hash,
             "role": "native C++ loader model DSO"}]


def _audit_runs(stage: Path, runs: object, candidates: dict,
                fixture_records: dict, profile: str) -> tuple[dict, list[dict]]:
    if not isinstance(runs, list) or not runs:
        raise EvidenceError("fixed-run observations are missing")
    phases = {name: set() for name in FIXTURES}
    repeat_rows = {name: [] for name in FIXTURES}
    different_hashes = {name: set() for name in FIXTURES}
    phase_indexes = {name: {phase: set() for phase in PHASES} for name in FIXTURES}
    concurrent_inputs = {name: set() for name in FIXTURES}
    interleaved_inputs = {name: {} for name in FIXTURES}
    phase_logs = {name: {phase: set() for phase in PHASES} for name in FIXTURES}
    seen_invocations = set()
    reference_rows = []
    refs = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get("fixture") not in FIXTURES:
            raise EvidenceError(f"run row {index} names an unknown fixture")
        name, phase = run["fixture"], run.get("phase")
        if phase not in PHASES:
            # CPU/legacy rows belong to each fixture's separately validated
            # reference_runs, not the model-specific phase invocation stream.
            raise EvidenceError(f"run row {index} has an unknown phase")
        phases[name].add(phase)
        loader_result_index = _exact_int(
            run.get("loader_result_index"), "native loader result index")
        invocation = (name, phase, loader_result_index)
        if invocation in seen_invocations:
            raise EvidenceError(f"{name} {phase} repeats a native invocation row")
        seen_invocations.add(invocation)
        phase_indexes[name][phase].add(loader_result_index)
        shape_inputs, shape_outputs = FIXTURE_IO[name]
        inputs, outputs = run.get("inputs"), run.get("outputs")
        if (not isinstance(inputs, list) or len(inputs) != len(shape_inputs) or
                not isinstance(outputs, list) or len(outputs) != len(shape_outputs) or
                run.get("input_shapes") != shape_inputs or run.get("output_shapes") != shape_outputs):
            raise EvidenceError(f"{name} run row {index} public tensor shapes/counts differ")
        input_values, input_hashes = [], []
        for port, (record, shape) in enumerate(zip(inputs, shape_inputs)):
            value, reference = _float_array(stage, record, _shape_elements(shape, "run input shape"),
                                            f"{name} run {index} input {port}")
            input_values.append(value)
            input_hashes.append(reference["sha256"])
            refs.append({"path": reference["path"], "sha256": reference["sha256"],
                         "role": f"VA4 {name} run input {port}"})
        bin_path, _ = _artifact(stage, fixture_records[name].get("bin"), f"{name} run weight bin")
        expected = _scalar_golden(name, input_values, bin_path.read_bytes())
        for port, (record, shape, values) in enumerate(zip(outputs, shape_outputs, expected)):
            actual, reference = _float_array(stage, record, _shape_elements(shape, "run output shape"),
                                              f"{name} run {index} output {port}")
            _compare(actual, values, 1e-6, 1e-6, f"{name} run {index} output {port}")
            refs.append({"path": reference["path"], "sha256": reference["sha256"],
                         "role": f"VA4 {name} run output {port}"})
        if (type(run.get("status")) is not int or run.get("status") != 0 or
                run.get("diagnostic") not in (None, "")):
            raise EvidenceError(f"{name} run {index} returned error or diagnostic")
        if _exact_int(run.get("validation_error_count"), "validation errors") != 0:
            raise EvidenceError(f"{name} run {index} reports validation-layer errors")
        _exact_int(run.get("elapsed_ns"), "run elapsed nanoseconds", 1)
        if run.get("dso_sha256") != candidates[name]["link"]["dso"]["sha256"]:
            raise EvidenceError(f"{name} run {index} candidate DSO hash differs")
        loader_refs = _audit_loader_run(stage, run, name, profile, candidates[name])
        refs.extend(loader_refs)
        native_log = run["native_loader"]["log"]
        phase_logs[name][phase].add((native_log["path"], native_log["sha256"]))
        if phase == "same_context_concurrent":
            concurrent_inputs[name].add(tuple(input_hashes))
        if phase == "interleaved_contexts":
            interleaved_inputs[name][loader_result_index] = tuple(input_hashes)
        if phase == "same_seed_repeat":
            repeat_rows[name].append((input_hashes, [item.get("sha256") for item in outputs]))
        if phase == "different_input":
            different_hashes[name].add(tuple(input_hashes))
    for name in FIXTURES:
        fixture = fixture_records[name]
        references = fixture.get("references")
        expected_rows = references.get("runs") if isinstance(references, dict) else None
        if not isinstance(expected_rows, list):
            raise EvidenceError(f"{name} fixture-bound CPU/legacy run inventory is missing")
        reference_rows.extend(expected_rows)
        if phases[name] != set(PHASES):
            raise EvidenceError(f"{name} lacks VA4 repeat/different/concurrent/context phases")
        required_indexes = {"same_seed_repeat": {0, 1},
                            "different_input": {0, 1},
                            "same_context_concurrent": {0, 1},
                            "interleaved_contexts": {0, 1, 2, 3}}
        for phase, expected_indexes in required_indexes.items():
            if phase_indexes[name][phase] != expected_indexes:
                raise EvidenceError(f"{name} {phase} does not cover every native invocation")
            if len(phase_logs[name][phase]) != 1:
                raise EvidenceError(f"{name} {phase} combines different native process executions")
        if len(concurrent_inputs[name]) != 2:
            raise EvidenceError(f"{name} concurrent workers do not use different input tensors")
        interleaved = interleaved_inputs[name]
        if (interleaved[0] != interleaved[2] or interleaved[1] != interleaved[3] or
                interleaved[0] == interleaved[1]):
            raise EvidenceError(f"{name} interleaved inputs do not preserve the A-B-A-B invocation binding")
        repeated = repeat_rows[name]
        if len(repeated) != 2 or repeated[0] != repeated[1]:
            raise EvidenceError(f"{name} same-seed repeat input/output bytes are not identical")
        if len(different_hashes[name]) < 2:
            raise EvidenceError(f"{name} different-input phase lacks distinct input tensors")
    return ({"invocation_count": len(runs),
             "phase_coverage": {name: sorted(values) for name, values in phases.items()},
             "reference_rows": len(reference_rows)}, refs)


def _source_root(stage: Path) -> Path:
    cache = stage / "CMakeCache.txt"
    if not cache.is_file():
        raise EvidenceError("fresh stage CMakeCache is missing")
    source = None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:") and "=" in line:
            source = line.split("=", 1)[1]
    path = Path(source or "").resolve()
    if not path.is_dir():
        raise EvidenceError("fresh gate source root is invalid")
    return path


def audit_device_raw(stage: Path, profile: str, source_fingerprint: str,
                     device: str, *, require_legacy: bool,
                     raw_path: Path | None = None) -> tuple[dict, list[dict]]:
    if device not in DEVICES:
        raise EvidenceError(f"unknown ICD {device}")
    raw_path = (Path(raw_path) if raw_path is not None else
                stage / "test/Vulkan/evidence/va4" / device / "fixedrun-raw.json")
    try:
        raw_path.resolve().relative_to(stage.resolve())
    except ValueError as error:
        raise EvidenceError("fixed-run raw JSON path escapes its stage") from error
    if not raw_path.is_file():
        raise EvidenceError(f"{device} fixed-run raw JSON is missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or raw.get("icd") != device:
        raise EvidenceError(f"{device} schema/ICD identity is invalid")
    if raw.get("profile") != profile or raw.get("source_fingerprint") != source_fingerprint:
        raise EvidenceError(f"{device} raw source/profile does not bind current gate")
    producer = raw.get("producer")
    if (not isinstance(producer, dict) or producer.get("profile") != profile or
            producer.get("source_fingerprint") != source_fingerprint or
            not isinstance(producer.get("argv"), list) or not producer.get("argv") or
            not all(isinstance(x, str) for x in producer["argv"]) or
            not isinstance(producer.get("cwd"), str)):
        raise EvidenceError(f"{device} producer argv/cwd/source identity is invalid")
    device_record = raw.get("device")
    if not isinstance(device_record, dict):
        raise EvidenceError(f"{device} physical-device identity is missing")
    name = device_record.get("name")
    expected = "llvmpipe" if device == "llvmpipe" else "nvidia"
    if not isinstance(name, str) or expected not in name.lower():
        raise EvidenceError(f"{device} selected physical device identity differs")
    driver = device_record.get("driver")
    if not isinstance(driver, str) or not driver:
        raise EvidenceError(f"{device} driver identity is missing")
    ordinal = _exact_int(device_record.get("ordinal"), f"{device} ordinal")
    vulkaninfo, info_hash = _artifact(stage, device_record.get("vulkaninfo"),
                                      f"{device} vulkaninfo --summary")
    if expected not in vulkaninfo.read_text(encoding="utf-8", errors="replace").lower():
        raise EvidenceError(f"{device} Vulkan info capture does not identify selected device")
    icd_path, icd_hash = _artifact(stage, device_record.get("icd_manifest"),
                                   f"{device} selected ICD manifest")
    if not icd_path.is_file():
        raise EvidenceError(f"{device} ICD manifest is missing")
    legacy = _audit_legacy_reference(stage, raw.get("legacy_reference"), require_legacy)
    source_root = _source_root(stage)
    fixture_list = raw.get("fixtures")
    if not isinstance(fixture_list, list) or {row.get("name") for row in fixture_list if isinstance(row, dict)} != FIXTURES:
        raise EvidenceError(f"{device} fixed-run fixture inventory differs from seven required graphs")
    fixture_rows = {row["name"]: row for row in fixture_list}
    source_rows = raw.get("source_inputs")
    if not isinstance(source_rows, list) or {row.get("fixture") for row in source_rows if isinstance(row, dict)} != FIXTURES:
        raise EvidenceError(f"{device} per-fixture source input closure is incomplete")
    source_refs = []
    for row in source_rows:
        fixture = fixture_rows[row["fixture"]]
        for field in ("param", "bin"):
            if row.get(field) != fixture.get(field):
                raise EvidenceError(f"{device}/{row['fixture']} source copy differs from fixture evidence")
            path, digest = _artifact(stage, row[field], f"{device}/{row['fixture']} {field}")
            source_refs.append({"path": path.relative_to(stage).as_posix(), "sha256": digest,
                                "role": f"VA4 source input {row['fixture']} {field}"})
    fixture_summaries = {}
    for fixture_name, fixture in fixture_rows.items():
        fixture_summaries[fixture_name] = _audit_fixture(
            stage, source_root, fixture, fixture_name,
            allow_missing_legacy=not require_legacy, legacy_reference=legacy)
    candidates_list = raw.get("candidate_builds")
    if not isinstance(candidates_list, list) or len(candidates_list) != len(FIXTURES):
        raise EvidenceError("per-fixture candidate build list is incomplete")
    records = {row.get("fixture"): row for row in candidates_list if isinstance(row, dict)}
    if set(records) != set(FIXTURES):
        raise EvidenceError("per-fixture candidate build IDs are invalid")
    candidates = {}
    dso_hashes = {}
    for fixture_name in sorted(FIXTURES):
        record = records[fixture_name]
        if record != fixture_rows[fixture_name].get("candidate_build"):
            raise EvidenceError(f"{fixture_name} fixture and candidate_builds entries differ")
        source_param = source_root / fixture_rows[fixture_name]["param_source"]
        candidate = _audit_candidate(stage, record, fixture_rows[fixture_name],
                                     profile, source_param)
        candidates[fixture_name] = candidate
        dso_hashes[fixture_name] = candidate["link"]["dso"]["sha256"]
    run_metrics, run_refs = _audit_runs(stage, raw.get("runs"), candidates,
                                         fixture_rows, profile)
    summary = {"status": "verified", "icd": device,
               "raw": {"path": raw_path.relative_to(stage).as_posix(),
                       "sha256": sha256_file(raw_path)},
               "device": {"name": name, "driver": driver, "ordinal": ordinal,
                          "vulkaninfo": {"path": vulkaninfo.relative_to(stage).as_posix(),
                                         "sha256": info_hash},
                          "icd_manifest": {"path": icd_path.relative_to(stage).as_posix(),
                                           "sha256": icd_hash}},
               "legacy_reference": legacy, "source_inputs": source_refs,
               "candidate_builds": candidates, "fixtures": fixture_summaries,
               "runs": run_metrics}
    refs_by_path = {raw_path.relative_to(stage).as_posix(): {
        "path": raw_path.relative_to(stage).as_posix(), "sha256": sha256_file(raw_path),
        "role": f"VA4 {device} raw producer JSON"}}
    for row in source_refs + run_refs:
        refs_by_path[row["path"]] = row

    def collect(value: object, role: str) -> None:
        if isinstance(value, dict):
            if (isinstance(value.get("path"), str) and
                    isinstance(value.get("sha256"), str) and
                    not Path(value["path"]).is_absolute()):
                path, digest = _artifact(stage, value, role)
                relative = path.relative_to(stage).as_posix()
                refs_by_path[relative] = {"path": relative, "sha256": digest,
                                          "role": f"VA4 {device} {role}"}
            # Absolute path/hash pairs identify executed tools, not archived
            # artifact paths. Their copied binaries are audited in the owning
            # build/reference check and collected through their artifact child.
            for key, child in value.items():
                collect(child, str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, f"{role}[{index}]")

    collect(summary, "artifact")
    return summary, list(refs_by_path.values())


def aggregate_stage_evidence(stage_root: str | Path, profile: str,
                             source_fingerprint: str, test_names: list[str],
                             skipped: list[str], failed: list[str], *,
                             require_legacy: bool = True) -> tuple[dict, list[dict]]:
    stage = Path(stage_root).resolve()
    required = {"vulkan-fixedrun-capture-publish",
                "vulkan-fixedrun-fake-unit", "vulkan-fixedrun-evidence-validation",
                "vulkan-fixedrun-independence", "vulkan-fixedrun-legacy-reference-tests",
                "vulkan-fixedrun-producer-tool-role-validation",
                "vulkan-fixedrun-nvidia",
                "vulkan-fixedrun-llvmpipe"}
    names = set(test_names)
    missing = required - names
    if missing:
        raise EvidenceError(f"CTest inventory lacks VA4 tests {sorted(missing)}")
    incomplete = sorted(required & (set(skipped) | set(failed)))
    if incomplete:
        raise EvidenceError(f"VA4 tests were skipped or failed: {incomplete}")
    devices, refs = {}, []
    for device in DEVICES:
        summary, records = audit_device_raw(stage, profile, source_fingerprint,
                                            device, require_legacy=require_legacy)
        devices[device] = summary
        refs.extend(records)
    identities = {}
    for fixture in FIXTURES:
        hashes = {summary["candidate_builds"][fixture]["link"]["dso"]["sha256"]
                  for summary in devices.values()}
        if len(hashes) != 1:
            raise EvidenceError(f"{fixture} DSO bytes differ between NVIDIA and llvmpipe")
        identities[fixture] = next(iter(hashes))
    legacy_qualified = all(
        summary.get("legacy_reference", {}).get("qualified") is True
        for summary in devices.values())
    # The archived aggregate must be byte-stable on re-audit. Wall-clock audit
    # timestamps belong to the outer report, not this hash-bound derived file.
    aggregate = {"schema": "ncnn-vulkan-fixedrun-aggregate/v1", "stage_id": "VA4",
                 "profile": profile, "source_fingerprint": source_fingerprint,
                 "role": "fixed-run-candidate", "devices": devices,
                 "legacy_reference_qualified": legacy_qualified,
                 "candidate_dso_sha256s": identities,
                 "claim_scope": "private seven-small-graph fixed-run candidate evidence; product-scope claims are audited separately from this aggregate",
                 "host_aot_release_eligible": False}
    output = stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Derived after CTest: keep it out of the test-produced raw inventory.
    # The stage adapter separately binds this file for archive completeness.
    return aggregate, refs


def _audit_independence(stage: Path, source: Path) -> None:
    candidates = sorted((stage / "test/Vulkan/fixedrun-candidates").glob("*/candidate_source.cpp"))
    if {path.parent.name for path in candidates} != FIXTURES:
        raise EvidenceError("generated fixed-run source inventory differs from seven fixtures")
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8")
        if FORBIDDEN_RUN.search(text):
            raise EvidenceError(f"{candidate.parent.name} generated code includes generic execution structures")
        if not re.search(r"_vk_run\s*\(", text):
            raise EvidenceError(f"{candidate.parent.name} generated code has no model-specific run")
    driver = source / "test/Vulkan/run_fixedrun.py"
    text = driver.read_text(encoding="utf-8")
    if re.search(r"\bctypes\b|CDLL|CFUNCTYPE", text):
        raise EvidenceError("fixed-run inference must execute through the native C++ loader, not Python ctypes")
    if not (source / "test/Vulkan/native/fixedrun_native_loader.cpp").is_file():
        raise EvidenceError("native C++ C ABI loader source is missing")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("independence",), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _audit_independence(args.stage_root.resolve(), args.source_root.resolve())
        return 0
    except (EvidenceError, OSError, TypeError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"fixedrun_evidence.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
