#!/usr/bin/env python3
"""Build, exercise, and archive native Vulkan device-helper tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


FIXTURE_ID = "va2-f32-add-257-gpu-func-v1"
HELPER_TEST_KIND = "ncnn.vulkan_device_helper_test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(
    name: str,
    command: list[str],
    log_dir: Path,
    env: dict[str, str] | None = None,
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(item) for item in command],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    record = {
        "argv": [str(item) for item in command],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if environment_overrides is not None:
        record["environment_overrides"] = environment_overrides
    (log_dir / f"{name}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}; "
            f"see {log_dir / f'{name}.json'}"
        )
    return record


def stage_relative(path: Path, stage_root: Path) -> str:
    try:
        return path.resolve().relative_to(stage_root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"raw evidence path is outside stage root: {path}") from error


def raw_entry(
    path: Path,
    stage_root: Path,
    role: str,
    *,
    origin: Path | None = None,
) -> dict[str, str | int]:
    if not path.is_file():
        raise RuntimeError(f"required raw evidence file is missing: {path}")
    entry: dict[str, str | int] = {
        "path": stage_relative(path, stage_root),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }
    source = origin if origin is not None else path
    if not source.is_file():
        raise RuntimeError(f"raw evidence origin is missing: {source}")
    if sha256(source) != entry["sha256"] or source.stat().st_size != entry["size_bytes"]:
        raise RuntimeError(f"raw evidence differs from its origin: {source}")
    entry["origin"] = str(source.resolve())
    return entry


def snapshot_raw_input(
    source: Path,
    stage_root: Path,
    evidence_dir: Path,
    role: str,
) -> dict[str, str | int]:
    source = source.resolve()
    try:
        source.relative_to(stage_root.resolve())
    except ValueError:
        destination = evidence_dir / "inputs" / role / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    else:
        destination = source
    return raw_entry(destination, stage_root, role, origin=source)


def read_cmake_cache(path: Path) -> dict[str, tuple[str, str]]:
    values: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        key_and_type, separator, value = line.partition("=")
        key, type_separator, value_type = key_and_type.partition(":")
        if separator and type_separator:
            values[key] = (value_type, value)
    return values


def cmake_arg_value(args: argparse.Namespace, name: str) -> str | None:
    prefix = f"-D{name}="
    for item in args.cmake_arg:
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def write_test_record(
    identity: str,
    stage_root: Path,
    evidence_root: Path,
    raw: list[dict[str, str | int]],
) -> Path:
    record_path = evidence_root / "tests" / f"{identity}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": HELPER_TEST_KIND,
        "identity": identity,
        "status": "passed",
        "raw": raw,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record_path


def do_unit(args: argparse.Namespace) -> None:
    stage_root = Path(args.stage_root).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    output = evidence_root / "unit"
    logs = output / "logs"
    command_log = logs / "unit-test.json"
    result_path = output / "result.json"
    log = run_logged("unit-test", [args.unit_executable], logs)
    result = {
        "schema_version": 1,
        "kind": "ncnn.vulkan_device_helper_unit_result",
        "returncode": log["returncode"],
        "status": "passed",
        "stdout": log["stdout"],
        "stderr": log["stderr"],
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_test_record(
        "vulkan-device-helper-unit",
        stage_root,
        evidence_root,
        [
            raw_entry(
                Path(args.unit_executable), stage_root, "unit-binary",
                origin=Path(args.unit_executable),
            ),
            raw_entry(command_log, stage_root, "unit-command-log"),
            raw_entry(result_path, stage_root, "result"),
        ],
    )


def do_no_device(args: argparse.Namespace) -> None:
    stage_root = Path(args.stage_root).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    output = evidence_root / "no-device"
    logs = output / "logs"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "unavailable-icd.json"
    result_path = output / "no-device-result.json"
    missing_icd = (output / "missing-test-icd.so").resolve()
    manifest.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "ICD": {"library_path": str(missing_icd), "api_version": "1.1.0"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = str(manifest)
    env["VK_DRIVER_FILES"] = str(manifest)
    no_device_command = [
        args.runner,
        "--mode",
        "no-device",
        "--result",
        str(result_path),
    ]
    icd_overrides = {
        "VK_ICD_FILENAMES": str(manifest),
        "VK_DRIVER_FILES": str(manifest),
    }
    command_log = run_logged(
        "no-device-create",
        no_device_command,
        logs,
        env,
        environment_overrides=icd_overrides,
    )
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    stdout_path.write_text(command_log["stdout"], encoding="utf-8")
    stderr_path.write_text(command_log["stderr"], encoding="utf-8")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"no-device helper result is missing or invalid: {error}") from error
    if result.get("helper_create_succeeded") is not False:
        raise RuntimeError("helper unexpectedly created a Vulkan device with the unavailable ICD")
    if type(result.get("status")) is not int or result["status"] <= 0:
        raise RuntimeError("failed helper create did not report a nonzero ABI status")
    if not isinstance(result.get("diagnostic"), str) or not result["diagnostic"]:
        raise RuntimeError("failed helper create did not preserve a diagnostic")
    if command_log["returncode"] != 0:
        raise RuntimeError("no-device probe must succeed only after observing helper create failure")
    write_test_record(
        "vulkan-device-helper-no-device",
        stage_root,
        evidence_root,
        [
            raw_entry(result_path, stage_root, "result"),
            raw_entry(
                Path(args.runner), stage_root, "runner-binary",
                origin=Path(args.runner),
            ),
            raw_entry(manifest, stage_root, "icd-manifest", origin=manifest),
            raw_entry(stdout_path, stage_root, "stdout"),
            raw_entry(stderr_path, stage_root, "stderr"),
            raw_entry(logs / "no-device-create.json", stage_root, "create-command-log"),
        ],
    )


def verify_add_spirv(disassembly: str) -> None:
    if "; Version: 1.0" not in disassembly or (
        'OpExtension "SPV_KHR_storage_buffer_storage_class"' not in disassembly
    ):
        raise RuntimeError("SPIR-V does not match the Vulkan 1.1 / SPIR-V 1.0 target")
    if not re.search(r'OpEntryPoint\s+GLCompute\s+%\S+\s+"add"', disassembly):
        raise RuntimeError("SPIR-V is missing the add compute entry point")
    if not re.search(r"OpExecutionMode\s+%\S+\s+LocalSize\s+64\s+1\s+1", disassembly):
        raise RuntimeError("SPIR-V has an unexpected workgroup size")
    bindings = {int(value) for value in re.findall(r"OpDecorate %\S+ Binding (\d+)", disassembly)}
    sets = re.findall(r"OpDecorate %\S+ DescriptorSet (\d+)", disassembly)
    if not {0, 1, 2}.issubset(bindings) or sets.count("0") < 3:
        raise RuntimeError("SPIR-V does not expose storage-buffer set 0 bindings 0-2")


def do_add(args: argparse.Namespace) -> None:
    stage_root = Path(args.stage_root).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    identity = f"vulkan-device-helper-add-{args.device}"
    output = evidence_root / "devices" / args.device
    logs = output / "logs"
    output.mkdir(parents=True, exist_ok=True)
    lowered = output / "gpu.binary.mlir"
    shader = output / "add.spv"
    disassembly = output / "add.spvasm"
    lhs = output / "lhs.f32"
    rhs = output / "rhs.f32"
    cpu_output = output / "cpu-aot-output.f32"
    gpu_output = output / "helper-gpu-output.f32"
    gpu_repeat_output = output / "helper-gpu-repeat.f32"
    golden = output / "scalar-golden.f32"
    result_path = output / "device-result.json"
    if not args.icd:
        raise RuntimeError("device helper Add requires an explicit ICD manifest")
    icd_manifest = Path(args.icd).resolve()
    if not icd_manifest.is_file():
        raise RuntimeError(f"selected ICD manifest is missing: {icd_manifest}")
    try:
        icd_configuration = json.loads(icd_manifest.read_text(encoding="utf-8"))
        icd_library = icd_configuration["ICD"]["library_path"]
        if not isinstance(icd_library, str) or not icd_library:
            raise ValueError("ICD library_path must be a nonempty string")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"selected ICD manifest is invalid: {error}") from error
    archived_icd_manifest = output / "icd-manifest.json"
    shutil.copyfile(icd_manifest, archived_icd_manifest)

    run_logged(
        "mlir-opt-version", [args.mlir_opt, "--version"], logs
    )
    run_logged(
        "spirv-val-version", [args.spirv_val, "--version"], logs
    )
    lowering = run_logged(
        "gpu-func-add-lowering",
        [
            args.mlir_opt,
            args.kernel,
            "--test-vulkan-runner-pipeline",
            "--verify-each",
            "-o",
            str(lowered),
        ],
        logs,
    )
    del lowering
    lowered_text = lowered.read_text(encoding="utf-8")
    if "gpu.binary @v0_add" not in lowered_text:
        raise RuntimeError("GPU-to-SPIR-V pipeline did not emit gpu.binary @v0_add")
    for residual in ("gpu.func", "memref.load", "memref.store", "unrealized_conversion_cast"):
        if residual in lowered_text:
            raise RuntimeError(f"lowered GPU binary retains forbidden IR: {residual}")

    run_logged(
        "spirv-extraction",
        [args.spirv_extract, str(lowered), "-o", str(shader)],
        logs,
    )
    validation_log = logs / "spirv-validation.json"
    validation_record = run_logged(
        "spirv-validation",
        [args.spirv_val, "--target-env", "vulkan1.1", str(shader)],
        logs,
    )
    validation_record["kind"] = "ncnn.spirv_tool_invocation"
    validation_log.write_text(
        json.dumps(validation_record, indent=2) + "\n", encoding="utf-8"
    )
    run_logged(
        "spirv-disassembly",
        [args.spirv_dis, str(shader), "-o", str(disassembly)],
        logs,
    )
    verify_add_spirv(disassembly.read_text(encoding="utf-8"))

    command = [
        args.runner,
        "--mode",
        "add",
        "--shader",
        str(shader),
        "--cpu-library",
        args.cpu_library,
        "--cpu-symbol",
        args.cpu_symbol,
        "--device-match",
        args.device_match,
        "--lhs",
        str(lhs),
        "--rhs",
        str(rhs),
        "--cpu-output",
        str(cpu_output),
        "--gpu-output",
        str(gpu_output),
        "--gpu-repeat-output",
        str(gpu_repeat_output),
        "--golden",
        str(golden),
        "--result",
        str(result_path),
    ]
    if args.validation:
        command.append("--validation")
    env = os.environ.copy()
    device_environment: dict[str, str] = {}
    if args.icd:
        env["VK_ICD_FILENAMES"] = args.icd
        env["VK_DRIVER_FILES"] = args.icd
        device_environment = {
            "VK_ICD_FILENAMES": str(Path(args.icd).resolve()),
            "VK_DRIVER_FILES": str(Path(args.icd).resolve()),
        }
    runner_log = run_logged(
        "helper-device-add",
        command,
        logs,
        env,
        environment_overrides=device_environment,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    device_name = result.get("device_name")
    if (result.get("device") != args.device or
            not isinstance(device_name, str) or
            args.device_match.casefold() not in device_name.casefold()):
        raise RuntimeError("helper selected device does not match the requested ICD/device")
    if (result.get("status") != "passed" or
            type(result.get("validation_errors")) is not int or
            result["validation_errors"] != 0 or
            result.get("validation_enabled") is not True or
            result.get("wait_failure_recovered") is not True or
            result.get("fixture_id") != FIXTURE_ID or
            result.get("elements") != 257 or
            result.get("local_size") != 64 or
            result.get("dispatch_groups_x") != 5):
        raise RuntimeError("helper add result failed numerical, lifecycle, or validation checks")
    if runner_log["returncode"] != 0:
        raise RuntimeError("native helper runner returned a failure status")

    raw = [
        snapshot_raw_input(Path(args.runner), stage_root, output, "runner-binary"),
        snapshot_raw_input(Path(args.mlir_opt), stage_root, output, "mlir-opt-binary"),
        snapshot_raw_input(
            Path(args.spirv_extract), stage_root, output, "spirv-extractor-binary"
        ),
        snapshot_raw_input(Path(args.spirv_val), stage_root, output, "spirv-val-binary"),
        snapshot_raw_input(Path(args.spirv_dis), stage_root, output, "spirv-dis-binary"),
        snapshot_raw_input(Path(args.kernel), stage_root, output, "source-kernel"),
        snapshot_raw_input(
            Path(args.cpu_library), stage_root, output, "cpu-aot-library"
        ),
        raw_entry(lowered, stage_root, "gpu-binary-mlir"),
        raw_entry(archived_icd_manifest, stage_root, "icd-manifest",
                  origin=icd_manifest),
        raw_entry(disassembly, stage_root, "spirv-disassembly"),
        raw_entry(logs / "mlir-opt-version.json", stage_root,
                  "mlir-opt-version-log"),
        raw_entry(logs / "spirv-val-version.json", stage_root,
                  "spirv-val-version-log"),
        raw_entry(logs / "gpu-func-add-lowering.json", stage_root,
                  "gpu-func-add-lowering-log"),
        raw_entry(logs / "spirv-extraction.json", stage_root,
                  "spirv-extraction-log"),
        raw_entry(logs / "spirv-disassembly.json", stage_root,
                  "spirv-disassembly-log"),
        raw_entry(lhs, stage_root, "lhs-f32"),
        raw_entry(rhs, stage_root, "rhs-f32"),
        raw_entry(cpu_output, stage_root, "cpu-aot-output-f32"),
        raw_entry(gpu_output, stage_root, "helper-gpu-output-f32"),
        raw_entry(gpu_repeat_output, stage_root, "helper-gpu-repeat-f32"),
        raw_entry(golden, stage_root, "scalar-golden-f32"),
        raw_entry(shader, stage_root, "spirv"),
        raw_entry(validation_log, stage_root, "spirv-validation-log"),
        raw_entry(logs / "helper-device-add.json", stage_root, "validation-log"),
        raw_entry(result_path, stage_root, "result"),
    ]
    write_test_record(identity, stage_root, evidence_root, raw)


def do_evidence_tests(args: argparse.Namespace) -> None:
    stage_root = Path(args.stage_root).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    output = evidence_root / "evidence-tests"
    logs = output / "logs"
    script_log = run_logged(
        "evidence-tests",
        [args.python, args.evidence_test_script],
        logs,
    )
    result_path = output / "result.json"
    output.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ncnn.vulkan_device_helper_evidence_tests_result",
                "status": "passed",
                "returncode": script_log["returncode"],
                "stdout": script_log["stdout"],
                "stderr": script_log["stderr"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    write_test_record(
        "vulkan-device-helper-evidence-tests",
        stage_root,
        evidence_root,
        [
            raw_entry(result_path, stage_root, "result"),
            raw_entry(logs / "evidence-tests.json", stage_root, "test-command-log"),
        ],
    )


def _clean_standalone_cmake_args(args: argparse.Namespace) -> list[str]:
    allowed_prefixes = (
        "-DCMAKE_C_COMPILER=",
        "-DCMAKE_CXX_COMPILER=",
        "-DCMAKE_BUILD_TYPE=",
        "-DCMAKE_C_FLAGS=",
        "-DCMAKE_CXX_FLAGS=",
        "-DCMAKE_EXE_LINKER_FLAGS=",
        "-DCMAKE_SHARED_LINKER_FLAGS=",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=",
    )
    return [item for item in args.cmake_arg if item.startswith(allowed_prefixes)]


def _check_product_only_inventory(inventory_path: Path) -> list[str]:
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        names = [test["name"] for test in inventory["tests"]]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid product-only CTest inventory: {error}") from error
    required = {
        "vulkan-device-helper-unit",
        "vulkan-device-helper-no-device",
        "vulkan-device-helper-independence",
        "vulkan-device-helper-evidence-tests",
        "VulkanRuntimeBudgetTest.DefaultBudgetAndStorageContractAreShared",
        "VulkanRuntimeBudgetTest.ChecksActualIoStagingBudgetBeforeAnyAllocation",
        "VulkanRuntimeBudgetTest.CountsTemporaryWeightStagingBeforeAllocation",
        "VulkanRuntimeBudgetTest.ChecksActualAllocationCountBeforeAllocation",
        "VulkanRuntimeBudgetTest.PartialBindFailureReleasesEveryOwnedObject",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"product-only CTest inventory omitted required helper tests: {missing}")
    unexpected_device_tests = sorted(
        name for name in names
        if name in {"vulkan-device-helper-add-nvidia", "vulkan-device-helper-add-llvmpipe"}
    )
    if unexpected_device_tests:
        raise RuntimeError(
            "product-only CTest inventory unexpectedly registered device tests: "
            f"{unexpected_device_tests}"
        )
    return sorted(required)


def _ensure_file_api(build_root: Path, source_root: Path,
                     cmake: str, logs: Path, label: str) -> Path | None:
    reply = build_root / ".cmake/api/v1/reply"
    if any(reply.glob("index-*.json")):
        return None
    # CMake before 3.27 cannot request a current-invocation reply from project
    # code. Preserve the existing minimum version by replaying configuration.
    query = build_root / ".cmake/api/v1/query"
    query.mkdir(parents=True, exist_ok=True)
    for name in ("codemodel-v2", "cache-v2", "cmakeFiles-v1"):
        (query / name).write_text("", encoding="utf-8")
    run_logged(label, [cmake, "-S", str(source_root), "-B", str(build_root)], logs)
    if not any(reply.glob("index-*.json")):
        raise RuntimeError(f"CMake did not generate the required File API reply: {build_root}")
    return logs / f"{label}.json"


def do_independence(args: argparse.Namespace) -> None:
    stage_root = Path(args.stage_root).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    output = evidence_root / "independence"
    logs = output / "logs"
    output.mkdir(parents=True, exist_ok=True)
    standalone_root = Path(args.standalone_build_root).resolve()
    product_root = Path(args.product_only_root).resolve()
    standalone_source = Path(args.standalone_cmake).resolve()
    source_root = Path(args.source_root).resolve()
    api_logs = []
    root_api_log = _ensure_file_api(
        stage_root, source_root, args.cmake, logs, "production-file-api-configure")
    if root_api_log is not None:
        api_logs.append(root_api_log)

    standalone_configure = [
        args.cmake,
        "-S",
        str(standalone_source),
        "-B",
        str(standalone_root),
        "-DNCNN_VULKAN_HELPER_BUILD_STANDALONE=ON",
        "-DBUILD_TESTING=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DNCNN_VULKAN_HELPER_SOURCE_DIR={source_root / 'lib/VulkanDeviceHelper'}",
        f"-DNCNN_VULKAN_HELPER_INCLUDE_DIR={source_root / 'include'}",
    ]
    standalone_configure.extend(_clean_standalone_cmake_args(args))
    if args.sanitizers:
        sanitizer_cache_value = args.sanitizers.replace(",", ";")
        standalone_configure.append(
            f"-DCOMPILER_SANITIZERS={sanitizer_cache_value}"
        )
    run_logged("standalone-configure", standalone_configure, logs)
    standalone_api_log = _ensure_file_api(
        standalone_root, standalone_source, args.cmake, logs,
        "standalone-file-api-configure")
    if standalone_api_log is not None:
        api_logs.append(standalone_api_log)
    standalone_build_log = run_logged(
        "standalone-build",
        [args.cmake, "--build", str(standalone_root), "--parallel", args.jobs],
        logs,
    )
    unit_binary = standalone_root / "vulkan_device_helper_unit_test"
    standalone_unit_log = run_logged("standalone-unit-test", [str(unit_binary)], logs)

    stage_cache_path = stage_root / "CMakeCache.txt"
    standalone_cache_path = standalone_root / "CMakeCache.txt"
    stage_cache = read_cmake_cache(stage_cache_path)
    standalone_cache = read_cmake_cache(standalone_cache_path)
    compiler_path_value = standalone_cache["CMAKE_CXX_COMPILER"][1]
    compiler_path = Path(compiler_path_value).resolve()
    stage_compiler_value = stage_cache["CMAKE_CXX_COMPILER"][1]
    stage_compiler_path = Path(stage_compiler_value).resolve()
    if compiler_path != stage_compiler_path:
        raise RuntimeError(
            f"standalone C++ compiler differs from stage: {compiler_path} != {stage_compiler_path}"
        )
    requested_compiler = cmake_arg_value(args, "CMAKE_CXX_COMPILER")
    if requested_compiler and Path(requested_compiler).resolve() != compiler_path:
        raise RuntimeError("standalone compiler differs from the parent CMake compiler argument")
    effective_flags: dict[str, str] = {}
    for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_EXE_LINKER_FLAGS",
                "CMAKE_SHARED_LINKER_FLAGS"):
        expected = cmake_arg_value(args, key)
        if expected is None:
            expected = stage_cache.get(key, (None, ""))[1]
        effective_flags[key] = expected
        if standalone_cache.get(key, (None, ""))[1] != expected:
            raise RuntimeError(f"standalone CMake cache differs from parent effective {key}")
    expected_sanitizers = sorted(filter(None, args.sanitizers.split(",")))
    cached_sanitizers = sorted(
        filter(None, standalone_cache.get("COMPILER_SANITIZERS", (None, ""))[1].split(";"))
    )
    if cached_sanitizers != expected_sanitizers:
        raise RuntimeError(
            f"standalone sanitizer identity differs: {cached_sanitizers} != {expected_sanitizers}"
        )
    compiler_version_log = run_logged(
        "standalone-cxx-version", [str(compiler_path), "--version"], logs
    )
    compiler_version = compiler_version_log["stdout"] + compiler_version_log["stderr"]
    resource_dir: Path | None = None
    resource_dir_log: Path | None = None
    asan_runtime: Path | None = None
    ldd_log_paths: list[Path] = []
    if "clang" in compiler_version.lower() or expected_sanitizers:
        resource_log = run_logged(
            "standalone-cxx-resource-dir", [str(compiler_path), "--print-resource-dir"], logs
        )
        resource_dir_log = logs / "standalone-cxx-resource-dir.json"
        resource_dir = Path(resource_log["stdout"].strip()).resolve()
        if not resource_dir.is_dir():
            raise RuntimeError(f"Clang resource directory is missing: {resource_dir}")
        if "address" in expected_sanitizers:
            runtime_dir = resource_dir / "lib/linux"
            runtimes = sorted(runtime_dir.glob("libclang_rt.asan-*.so"))
            if not runtimes:
                raise RuntimeError(f"shared Clang ASan runtime is missing under {runtime_dir}")
            asan_runtime = runtimes[0]
            for binary_name in (
                "vulkan_device_helper_runner",
                "vulkan_device_helper_unit_test",
            ):
                log_name = f"standalone-{binary_name}-ldd"
                ldd_record = run_logged(
                    log_name, ["ldd", str(standalone_root / binary_name)], logs
                )
                ldd_log_paths.append(logs / f"{log_name}.json")
                if "libclang_rt.asan" not in (
                        ldd_record["stdout"] + ldd_record["stderr"]):
                    raise RuntimeError(
                        f"SAN standalone target does not resolve shared libclang_rt.asan: {binary_name}"
                    )
    identity_path = output / "toolchain-identity.json"
    toolchain_identity = {
        "compiler_path": compiler_path_value,
        "resolved_compiler_path": str(compiler_path),
        "production_compiler_path": stage_compiler_value,
        "compiler_version": compiler_version,
        "build_type": standalone_cache.get("CMAKE_BUILD_TYPE", (None, ""))[1],
        "sanitizers": expected_sanitizers,
        "production_cache_cxx_flags": stage_cache.get("CMAKE_CXX_FLAGS", (None, ""))[1],
        "effective_cxx_flags": effective_flags["CMAKE_CXX_FLAGS"],
        "effective_exe_linker_flags": effective_flags["CMAKE_EXE_LINKER_FLAGS"],
        "effective_shared_linker_flags": effective_flags["CMAKE_SHARED_LINKER_FLAGS"],
        "cmake_cxx_flags": standalone_cache.get("CMAKE_CXX_FLAGS", (None, ""))[1],
        "cmake_exe_linker_flags": standalone_cache.get("CMAKE_EXE_LINKER_FLAGS", (None, ""))[1],
        "cmake_shared_linker_flags": standalone_cache.get("CMAKE_SHARED_LINKER_FLAGS", (None, ""))[1],
        "resource_dir": str(resource_dir) if resource_dir else None,
        "shared_asan_runtime": str(asan_runtime) if asan_runtime else None,
    }
    raw: list[dict[str, str | int]] = [
        raw_entry(logs / "standalone-configure.json", stage_root, "standalone-configure-log"),
        raw_entry(logs / "standalone-build.json", stage_root, "standalone-build-log"),
        raw_entry(logs / "standalone-unit-test.json", stage_root, "standalone-unit-result"),
        raw_entry(standalone_root / "compile_commands.json", stage_root,
                  "standalone-compile-commands"),
        raw_entry(standalone_root / "device-helper-target-link-closure.json", stage_root,
                  "standalone-target-link-closure"),
        raw_entry(standalone_root / "vulkan_device_helper_runner.map", stage_root,
                  "standalone-harness-link-map"),
        raw_entry(standalone_root / "vulkan_device_helper_unit_test.map", stage_root,
                  "standalone-unit-link-map"),
        raw_entry(standalone_root / "vulkan_device_helper_unit_test", stage_root,
                  "standalone-unit-binary"),
        raw_entry(standalone_root / "CMakeFiles/NCNNVulkanDeviceHelper.dir/link.txt",
                  stage_root, "standalone-helper-link-command"),
        raw_entry(standalone_root / "CMakeFiles/vulkan_device_helper_runner.dir/link.txt",
                  stage_root, "standalone-harness-link-command"),
        raw_entry(standalone_root / "CMakeFiles/vulkan_device_helper_unit_test.dir/link.txt",
                  stage_root, "standalone-unit-link-command"),
        raw_entry(standalone_root / "libNCNNVulkanDeviceHelper.a", stage_root,
                  "standalone-helper-archive"),
        raw_entry(stage_cache_path, stage_root, "production-cmake-cache"),
        raw_entry(standalone_cache_path, stage_root, "standalone-cmake-cache"),
        raw_entry(logs / "standalone-cxx-version.json", stage_root,
                  "standalone-compiler-version-command"),
    ]
    if resource_dir_log is not None:
        raw.append(raw_entry(resource_dir_log, stage_root,
                             "standalone-compiler-resource-dir-command"))
    raw.extend(raw_entry(item, stage_root,
                         f"standalone-sanitizer-runtime-resolution-{item.stem}")
               for item in ldd_log_paths)

    inventory_path: Path | None = None
    registered_tests: list[str] = []
    executed_tests: list[str] = []
    if not args.skip_product_only_check:
        product_configure = [
            args.cmake,
            "-S",
            str(source_root),
            "-B",
            str(product_root),
        ]
        product_configure.extend(args.cmake_arg)
        product_configure.extend(
            [
                "-DBUILD_TESTING=ON",
                "-DNCNN_ENABLE_VULKAN_PRODUCT=ON",
                "-DNCNN_ENABLE_VULKAN_V0_TESTS=OFF",
                "-DNCNN_VULKAN_DEVICE_HELPER_SKIP_PRODUCT_ONLY_CHECK=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DNCNN_VULKAN_DEVICE_HELPER_STAGE_ROOT={product_root}",
                f"-DNCNN_VULKAN_DEVICE_HELPER_EVIDENCE_ROOT={product_root / 'test/Vulkan/evidence/va2'}",
            ]
        )
        if args.sanitizers:
            sanitizer_cache_value = args.sanitizers.replace(",", ";")
            product_configure.append(
                f"-DCOMPILER_SANITIZERS={sanitizer_cache_value}"
            )
        run_logged("product-only-configure", product_configure, logs)
        product_api_log = _ensure_file_api(
            product_root, source_root, args.cmake, logs,
            "product-only-file-api-configure")
        if product_api_log is not None:
            api_logs.append(product_api_log)
        run_logged(
            "product-only-build",
            [
                args.cmake,
                "--build",
                str(product_root),
                "--target",
                "NCNNVulkanDeviceHelper",
                "vulkan_device_helper_runner",
                "vulkan_device_helper_unit_test",
                "vulkan_runtime_budget_tests",
                "--parallel",
                args.jobs,
            ],
            logs,
        )
        inventory_command = run_logged(
            "product-only-ctest-inventory",
            ["ctest", "--test-dir", str(product_root), "--show-only=json-v1"],
            logs,
        )
        inventory_path = product_root / "device-helper-ctest-inventory.json"
        inventory_path.write_text(inventory_command["stdout"], encoding="utf-8")
        registered_tests = _check_product_only_inventory(inventory_path)
        product_cache_path = product_root / "CMakeCache.txt"
        product_cache = read_cmake_cache(product_cache_path)
        product_compiler_value = product_cache["CMAKE_CXX_COMPILER"][1]
        product_compiler_path = Path(product_compiler_value).resolve()
        if product_compiler_path != stage_compiler_path:
            raise RuntimeError(
                f"product-only C++ compiler differs from stage: {product_compiler_path} != {stage_compiler_path}"
            )
        for key, expected in effective_flags.items():
            if product_cache.get(key, (None, ""))[1] != expected:
                raise RuntimeError(f"product-only CMake cache differs from effective {key}")
        product_sanitizers = sorted(
            filter(None, product_cache.get("COMPILER_SANITIZERS", (None, ""))[1].split(";"))
        )
        if product_sanitizers != expected_sanitizers:
            raise RuntimeError(
                f"product-only sanitizer identity differs: {product_sanitizers} != {expected_sanitizers}"
            )
        toolchain_identity["product_only_compiler_path"] = product_compiler_value
        toolchain_identity["product_only_resolved_compiler_path"] = str(product_compiler_path)
        toolchain_identity["product_only_build_type"] = product_cache.get(
            "CMAKE_BUILD_TYPE", (None, "")
        )[1]
        toolchain_identity["product_only_sanitizers"] = product_sanitizers
        raw.append(raw_entry(product_cache_path, stage_root, "product-only-cmake-cache"))

        executed_tests = [
            "vulkan-device-helper-unit",
            "vulkan-device-helper-no-device",
            "vulkan-device-helper-evidence-tests",
            "VulkanRuntimeBudgetTest.DefaultBudgetAndStorageContractAreShared",
            "VulkanRuntimeBudgetTest.ChecksActualIoStagingBudgetBeforeAnyAllocation",
            "VulkanRuntimeBudgetTest.CountsTemporaryWeightStagingBeforeAllocation",
            "VulkanRuntimeBudgetTest.ChecksActualAllocationCountBeforeAllocation",
            "VulkanRuntimeBudgetTest.PartialBindFailureReleasesEveryOwnedObject",
        ]
        executed_tests.sort()
        raw.extend(
            [
                raw_entry(product_root / "compile_commands.json", stage_root,
                          "product-only-compile-commands"),
                raw_entry(product_root / "test/Vulkan/device-helper-target-link-closure.json",
                          stage_root, "product-only-target-link-closure"),
                raw_entry(inventory_path, stage_root, "product-only-ctest-inventory"),
                raw_entry(logs / "product-only-configure.json", stage_root,
                          "product-only-configure-log"),
                raw_entry(logs / "product-only-build.json", stage_root,
                          "product-only-build-log"),
                raw_entry(logs / "product-only-ctest-inventory.json", stage_root,
                          "product-only-inventory-command-log"),
            ]
        )
        product_archives = list(product_root.rglob("libNCNNVulkanDeviceHelper.a"))
        if len(product_archives) != 1:
            raise RuntimeError(
                f"expected one product-only helper archive, found {product_archives}"
            )
        raw.append(raw_entry(product_archives[0], stage_root,
                             "product-only-helper-archive"))

    for api_log in api_logs:
        raw.append(raw_entry(api_log, stage_root, api_log.stem))
    identity_path.write_text(
        json.dumps(toolchain_identity, indent=2) + "\n", encoding="utf-8"
    )
    raw.append(raw_entry(identity_path, stage_root, "compiler-toolchain-identity"))
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ncnn.vulkan_device_helper_independence_result",
                "standalone_build_root": stage_relative(standalone_root, stage_root),
                "product_only_check_performed": not args.skip_product_only_check,
                "product_only_build_root": (
                    stage_relative(product_root, stage_root)
                    if not args.skip_product_only_check else None
                ),
                "standalone_build_returncode": standalone_build_log["returncode"],
                "standalone_unit_returncode": standalone_unit_log["returncode"],
                "product_only_inventory": (
                    stage_relative(inventory_path, stage_root)
                    if inventory_path else None
                ),
                "product_only_registered_tests": registered_tests,
                "product_only_executed_tests": executed_tests,
                "toolchain_identity": toolchain_identity,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "status": "passed",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    raw.append(raw_entry(result_path, stage_root, "result"))
    write_test_record(
        "vulkan-device-helper-independence", stage_root, evidence_root, raw
    )

    if not args.skip_product_only_check:
        if not args.evidence_driver:
            raise RuntimeError("--evidence-driver is required for full independence mode")
        driver_output = evidence_root / "independence-evidence.json"
        run_logged(
            "helper-evidence-audit",
            [
                args.python,
                args.evidence_driver,
                "--stage-root",
                str(stage_root),
                "--standalone-build-root",
                str(standalone_root),
                "--product-only-root",
                str(product_root),
                "--output",
                str(driver_output),
            ],
            logs,
        )
        if not driver_output.is_file():
            raise RuntimeError("helper evidence audit did not write independence-evidence.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("unit", "no-device", "add", "independence", "evidence-tests"))
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--runner")
    parser.add_argument("--unit-executable")
    parser.add_argument("--device", choices=("nvidia", "llvmpipe"))
    parser.add_argument("--device-match")
    parser.add_argument("--icd")
    parser.add_argument("--mlir-opt")
    parser.add_argument("--spirv-extract")
    parser.add_argument("--spirv-val")
    parser.add_argument("--spirv-dis")
    parser.add_argument("--kernel")
    parser.add_argument("--cpu-library")
    parser.add_argument("--cpu-symbol", default="v0_add")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--evidence-test-script")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cmake")
    parser.add_argument("--source-root")
    parser.add_argument("--standalone-cmake")
    parser.add_argument("--standalone-build-root")
    parser.add_argument("--product-only-root")
    parser.add_argument("--evidence-driver")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--sanitizers", default="")
    parser.add_argument("--jobs", default="2")
    parser.add_argument("--skip-product-only-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "unit":
            if not args.unit_executable:
                raise RuntimeError("--unit-executable is required for unit mode")
            do_unit(args)
        elif args.mode == "no-device":
            if not args.runner:
                raise RuntimeError("--runner is required for no-device mode")
            do_no_device(args)
        elif args.mode == "add":
            required = (args.runner, args.device, args.device_match, args.mlir_opt,
                        args.spirv_extract, args.spirv_val, args.spirv_dis,
                        args.kernel, args.cpu_library)
            if any(value is None for value in required):
                raise RuntimeError("add mode is missing a required argument")
            do_add(args)
        elif args.mode == "independence":
            required = (args.cmake, args.source_root, args.standalone_cmake,
                        args.standalone_build_root, args.product_only_root)
            if any(value is None for value in required):
                raise RuntimeError("independence mode is missing build paths")
            do_independence(args)
        else:
            if not args.evidence_test_script:
                raise RuntimeError("--evidence-test-script is required")
            do_evidence_tests(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"run_device_helper.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
