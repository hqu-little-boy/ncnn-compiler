#!/usr/bin/env python3
"""Capture and audit private Vulkan lifecycle-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import device_helper_evidence as device_helper_audit


SCHEMA_VERSION = 1
INDEPENDENCE_KIND = "ncnn.vulkan_lifecycle_independence"
CAPTURE_KIND = "ncnn.vulkan_lifecycle_capture"
NATIVE_RESULT_KIND = "ncnn.vulkan_lifecycle_native_result"
AGGREGATE_KIND = "ncnn.vulkan_lifecycle_evidence"
FAKE_RESULT_KIND = "ncnn.vulkan_lifecycle_fake_result"
CODEGEN_TEST_PREFIX = "VulkanLifecycleCodegenTest."
PRODUCT_ONLY_REQUIRED_TESTS = frozenset({
    "vulkan-lifecycle-fake-unit", "vulkan-lifecycle-evidence-tests",
    "vulkan-lifecycle-independence", "vulkan-lifecycle-no-device",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VALIDATION_ERROR_RE = re.compile(
    r"(?:Validation Error|Validation error|VUID-[^\s:]+.*(?:ERROR|Error)|"
    r"ERROR.*(?:Vulkan|validation))", re.IGNORECASE)
EXPECTED_EXPORTS = {
    "lifecycle_fixture_vk_create",
    "lifecycle_fixture_vk_run",
    "lifecycle_fixture_vk_destroy",
}
FORBIDDEN_INPUTS = re.compile(
    r"(?:VulkanRuntime|VulkanExecutionPlan|ExecutionPlan|PlanSupport|"
    r"VulkanProduct|NCNNVulkanProductSupport|NCNNVulkanPlanSupport|"
    r"ncnn-mlir-parser|libLLVM[A-Za-z0-9_.+-]*|libMLIR[A-Za-z0-9_.+-]*|"
    r"-l(?:LLVM|MLIR)[A-Za-z0-9_.+-]*|"
    r"(?:^|[/_-])ncnn[_-]mlir[_-](?:[^/\\s]*(?:plan|parser|runtime)[^/\\s]*)(?:[/_.-]|$)|"
    r"(?:^|[-/])(?:LLVM|MLIR)(?:IR|Support|Parser)(?:[./-]|$))"
)


class EvidenceError(RuntimeError):
    """Raised when lifecycle evidence is missing, stale, or malformed."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EvidenceError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceError(f"{label} path is unsafe")
    resolved_root = root.resolve()
    path = (resolved_root / candidate).resolve()
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise EvidenceError(f"{label} file is missing or escapes the stage: {relative}")
    return path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise EvidenceError(f"evidence file escapes stage root: {path}") from error


def _artifact(root: Path, path: Path, role: str) -> dict:
    return {"path": _relative(root, path), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size, "role": role}


def _snapshot_artifact(root: Path, source: Path, destination: Path,
                       role: str) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise EvidenceError(f"cannot snapshot missing build artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = _artifact(root, destination, role)
    copied["build_path"] = _relative(root, source)
    copied["build_sha256"] = sha256_file(source)
    if copied["sha256"] != copied["build_sha256"]:
        raise EvidenceError(f"snapshot differs from build artifact: {source}")
    return copied


def _snapshot_external(root: Path, source: Path, destination: Path,
                       role: str) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise EvidenceError(f"cannot snapshot missing external artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = _artifact(root, destination, role)
    copied["origin_path"] = str(source)
    copied["origin_sha256"] = sha256_file(source)
    if copied["sha256"] != copied["origin_sha256"]:
        raise EvidenceError(f"external artifact snapshot differs from its origin: {source}")
    return copied


def _check_artifact(root: Path, record: object, label: str) -> dict:
    if not isinstance(record, dict):
        raise EvidenceError(f"{label} artifact reference is malformed")
    path = _safe_file(root, record.get("path"), label)
    digest = record.get("sha256")
    size = record.get("size_bytes")
    role = record.get("role")
    if (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or
            digest != sha256_file(path) or type(size) is not int or
            size != path.stat().st_size or not isinstance(role, str) or not role):
        raise EvidenceError(f"{label} artifact hash, size, or role is invalid")
    build_path = record.get("build_path")
    build_digest = record.get("build_sha256")
    if build_path is not None:
        if (not isinstance(build_path, str) or Path(build_path).is_absolute() or
                ".." in Path(build_path).parts or not isinstance(build_digest, str) or
                not SHA256_RE.fullmatch(build_digest) or build_digest != digest):
            raise EvidenceError(f"{label} snapshot is not bound to its build-origin identity")
    origin_path = record.get("origin_path")
    origin_digest = record.get("origin_sha256")
    if origin_path is not None and (not isinstance(origin_path, str) or
                                    not Path(origin_path).is_absolute() or
                                    not isinstance(origin_digest, str) or
                                    not SHA256_RE.fullmatch(origin_digest) or
                                    origin_digest != digest):
        raise EvidenceError(f"{label} external snapshot is not bound to its origin")
    return {"path": _relative(root, path), "sha256": digest,
            "size_bytes": size, "role": role}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} cannot be read as JSON: {error}") from error


def _run_captured(root: Path, evidence_dir: Path, label: str,
                  argv: list[str], *, env_overrides: dict[str, str] | None = None) -> dict:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = evidence_dir / f"{label}.stdout.log"
    stderr_path = evidence_dir / f"{label}.stderr.log"
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, env=env,
                                check=False)
    return {
        "argv": [str(item) for item in argv],
        "returncode": result.returncode,
        "environment_overrides": env_overrides or {},
        "stdout": _artifact(root, stdout_path, f"{label} stdout"),
        "stderr": _artifact(root, stderr_path, f"{label} stderr"),
    }


def _native_result(root: Path, relative: str, *, mode: str,
                   device: str | None = None,
                   device_index: int | None = None,
                   validation: bool | None = None) -> tuple[dict, Path]:
    path = _safe_file(root, relative, f"{mode} native result")
    value = _load_json(path, f"{mode} native result")
    _validate_native_observations(value, mode, device, device_index)
    if validation is not None:
        if type(value.get("validation_enabled")) is not bool:
            raise EvidenceError(f"{mode} native validation_enabled must be a JSON boolean")
        if value["validation_enabled"] is not validation:
            raise EvidenceError(f"{mode} native validation layer was not enabled as required")
    return value, path


def _parse_vulkan_identity(summary: str, device: str) -> dict:
    blocks = []
    current = None
    for line in summary.splitlines():
        if re.match(r"^GPU\d+:$", line.strip()):
            if current is not None:
                blocks.append(current)
            current = {}
        elif current is not None:
            match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if match:
                current[match.group(1)] = match.group(2)
    if current is not None:
        blocks.append(current)
    if not blocks:
        raise EvidenceError("vulkaninfo did not enumerate a physical device")
    for enumeration_index, item in enumerate(blocks):
        name = item.get("deviceName")
        vendor_text = item.get("vendorID")
        api_text = item.get("apiVersion")
        device_type = item.get("deviceType")
        if not all(isinstance(value, str) and value for value in
                   (name, vendor_text, api_text, device_type, item.get("driverVersion"))):
            continue
        try:
            vendor_id = int(vendor_text, 0)
            version_parts = [int(part) for part in api_text.split(".")[:2]]
            device_id = int(item.get("deviceID", "0"), 0)
        except (ValueError, TypeError):
            continue
        if len(version_parts) != 2 or version_parts < [1, 1]:
            continue
        if device.lower() == "nvidia":
            matches = (vendor_id == 0x10DE and
                       "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" in device_type and
                       re.search(r"NVIDIA|GeForce|RTX|Quadro|Tesla", name, re.I))
        else:
            matches = (vendor_id == 0x10005 and
                       "PHYSICAL_DEVICE_TYPE_CPU" in device_type and
                       re.search(r"llvmpipe|lavapipe", name, re.I))
        if matches:
            return {"enumeration_index": enumeration_index,
                    "device_name": name, "vendor_id": vendor_id,
                    "device_id": device_id, "device_type": device_type,
                    "api_version": api_text, "driver_version": item["driverVersion"]}
    raise EvidenceError(
        f"vulkaninfo found no physical device matching requested {device}: {blocks}")


def _capture_native(args: argparse.Namespace) -> int:
    root = args.stage_root.resolve()
    _reject_sanitizer_environment(_lifecycle_build_profile(root))
    result_path = args.result.resolve()
    capture_path = args.capture.resolve()
    evidence_dir = capture_path.parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()
    if capture_path.exists():
        capture_path.unlink()
    library = args.library.resolve()
    harness = args.harness.resolve()
    icd = args.icd.resolve() if args.icd else None
    if not harness.is_file() or not library.is_file():
        raise EvidenceError("lifecycle harness or candidate DSO is missing")
    physical_device_identity = None
    invocation = {}
    if args.mode == "device":
        if icd is None or not icd.is_file() or not args.device_match or not args.vulkaninfo:
            raise EvidenceError("device capture requires ICD, vulkaninfo, and expected device identity")
        env = {"VK_ICD_FILENAMES": str(icd), "VK_DRIVER_FILES": str(icd),
               "VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation"}
        vkinfo = [str(args.vulkaninfo.resolve()), "--summary"]
        invocation["vulkaninfo"] = _run_captured(
            root, evidence_dir, "vulkaninfo", vkinfo, env_overrides=env)
        info_run = invocation["vulkaninfo"]
        info_out = _safe_file(root, info_run["stdout"]["path"], "vulkaninfo stdout")
        summary = info_out.read_text(encoding="utf-8", errors="replace")
        if info_run["returncode"] != 0:
            raise EvidenceError("vulkaninfo failed under the requested ICD")
        if "VK_LAYER_KHRONOS_validation" not in summary:
            raise EvidenceError("vulkaninfo did not report the Khronos validation layer")
        physical_device_identity = _parse_vulkan_identity(summary, args.device_match)
        info_run["physical_device_identity"] = physical_device_identity
        command = [str(harness), "--library", str(library), "--result", str(result_path),
                   "--device-match", args.device_match,
                   "--ordinal", str(physical_device_identity["enumeration_index"]),
                   "--validation"]
    elif args.mode == "no-device":
        no_device_manifest = str(root / "missing-va3-icd.json")
        env = {"VK_ICD_FILENAMES": no_device_manifest,
               "VK_DRIVER_FILES": no_device_manifest}
        command = [str(harness), "--library", str(library), "--result", str(result_path),
                   "--no-device"]
    else:
        raise EvidenceError(f"unsupported native lifecycle mode {args.mode}")
    harness_invocation = _run_captured(root, evidence_dir, "harness", command,
                                       env_overrides=env)
    harness_invocation["vulkaninfo"] = invocation.get("vulkaninfo")
    invocation = harness_invocation
    if invocation["returncode"] != 0:
        raise EvidenceError(f"lifecycle C11 harness failed with status {invocation['returncode']}")
    requested_index = (physical_device_identity["enumeration_index"]
                       if args.mode == "device" else None)
    result, result_file = _native_result(
        root, _relative(root, result_path), mode=args.mode,
        device=args.device_match if args.mode == "device" else None,
        device_index=requested_index,
        validation=True if args.mode == "device" else False)
    if result_file.resolve() != result_path:
        raise EvidenceError("lifecycle C11 harness wrote a result outside its requested path")
    validation_error_count = 0
    command_logs = [invocation]
    if isinstance(invocation.get("vulkaninfo"), dict):
        command_logs.append(invocation["vulkaninfo"])
    for command_record in command_logs:
        for stream_name in ("stdout", "stderr"):
            stream_path = _safe_file(root, command_record[stream_name]["path"],
                                     f"harness {stream_name}")
            text = stream_path.read_text(encoding="utf-8", errors="replace")
            validation_error_count += sum(
                1 for line in text.splitlines() if VALIDATION_ERROR_RE.search(line))
    if args.mode == "device" and validation_error_count != 0:
        raise EvidenceError("C11 lifecycle harness raw logs contain Vulkan validation errors")
    snapshot_dir = evidence_dir / "artifacts"
    library_snapshot = _snapshot_artifact(
        root, library, snapshot_dir / library.name, "candidate DSO exact-byte snapshot")
    harness_snapshot = _snapshot_artifact(
        root, harness, snapshot_dir / harness.name, "C11 harness exact-byte snapshot")
    icd_snapshot = None
    icd_library_record = None
    if icd:
        icd_snapshot = _snapshot_external(
            root, icd, snapshot_dir / icd.name, "requested Vulkan ICD manifest snapshot")
        manifest = _load_json(icd, "requested Vulkan ICD manifest")
        icd_library = (manifest.get("ICD", {}).get("library_path")
                       if isinstance(manifest, dict) and isinstance(manifest.get("ICD"), dict)
                       else None)
        if not isinstance(icd_library, str) or not icd_library:
            raise EvidenceError("requested ICD manifest omits library_path")
        library_path = device_helper_audit._icd_library_path(icd, icd_library).resolve()
        if not library_path.is_file():
            raise EvidenceError("requested ICD shared library is missing")
        icd_library_record = {"path": str(library_path), "sha256": sha256_file(library_path),
                              "size_bytes": library_path.stat().st_size}
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": CAPTURE_KIND,
        "role": "lifecycle-only",
        "mode": args.mode,
        "device": args.device_match,
        "requested_device_index": requested_index,
        "source_fingerprint": _gate_fingerprint(root),
        "library": library_snapshot,
        "harness": harness_snapshot,
        "icd_manifest": icd_snapshot,
        "icd_library": icd_library_record,
        "native_result": _artifact(root, result_file, "raw C11 harness result"),
        "native_observations": result,
        "validation_error_count": validation_error_count,
        "command": invocation,
    }
    _write_json(capture_path, record)
    validate_native_capture(
        root, capture_path, mode=args.mode,
        device=args.device_match if args.mode == "device" else None,
        expected_fingerprint=record["source_fingerprint"])
    print(f"captured Vulkan lifecycle {args.mode} evidence: {capture_path}")
    return 0


def _gate_fingerprint(root: Path) -> str | None:
    state_path = root / "gate-state.json"
    if not state_path.is_file():
        return None
    state = _load_json(state_path, "gate state")
    fingerprint = state.get("source_fingerprint") if isinstance(state, dict) else None
    if fingerprint is not None and (not isinstance(fingerprint, str) or
                                    not SHA256_RE.fullmatch(fingerprint)):
        raise EvidenceError("gate source fingerprint is malformed")
    return fingerprint


def _compile_database(stage: Path) -> tuple[Path, list[dict]]:
    path = stage / "compile_commands.json"
    database = _load_json(path, "CMake compile database")
    if not isinstance(database, list):
        raise EvidenceError("CMake compile database must be an array")
    return path, database


def _cmake_cache_values(root: Path) -> dict[str, str]:
    values = {}
    cache = root / "CMakeCache.txt"
    if not cache.is_file():
        return values
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def _lifecycle_build_profile(root: Path) -> str:
    gate_state_path = root / "gate-state.json"
    if not gate_state_path.is_file():
        return "unqualified"
    values = _cmake_cache_values(root)
    sanitizer_set = {item for item in values.get("COMPILER_SANITIZERS", "").split(";") if item}
    build_type = values.get("CMAKE_BUILD_TYPE", "").lower()
    if sanitizer_set:
        if sanitizer_set != {"address", "undefined"} or build_type != "debug":
            raise EvidenceError("lifecycle SAN build is not the exact ASan+UBSan Debug profile")
        profile = "sanitize"
    elif build_type == "release":
        profile = "release"
    else:
        profile = "unqualified"
    state = _load_json(gate_state_path, "gate state")
    gate_profile = state.get("profile") if isinstance(state, dict) else None
    if gate_profile in {"release", "sanitize"} and gate_profile != profile:
        raise EvidenceError("CMake cache profile differs from lifecycle gate profile")
    return profile


def _stage_source_directory(root: Path, expected_fingerprint: str | None) -> Path:
    source_directory = _cmake_cache_values(root).get("CMAKE_HOME_DIRECTORY")
    if (not isinstance(source_directory, str) or
            not Path(source_directory).is_absolute() or
            not Path(source_directory).is_dir()):
        raise EvidenceError("lifecycle CMake cache omits its actual source directory")
    source_root = Path(source_directory).resolve()
    if expected_fingerprint is not None:
        gate = _load_json(root / "gate-state.json", "lifecycle source binding")
        gate_source = gate.get("source_dir") if isinstance(gate, dict) else None
        if (not isinstance(gate_source, str) or not Path(gate_source).is_absolute() or
                Path(gate_source).resolve() != source_root or
                gate.get("source_fingerprint") != expected_fingerprint):
            raise EvidenceError("lifecycle CMake source differs from the qualified gate source")
    return source_root


def _reject_sanitizer_environment(profile: str) -> None:
    if profile != "sanitize":
        return
    inherited = [name for name in ("ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS")
                 if os.environ.get(name)]
    if inherited:
        raise EvidenceError(
            f"SAN lifecycle evidence forbids inherited sanitizer overrides: {inherited}")


def _require_profile_flags(argv: list[str], profile: str, label: str,
                           *, pic: bool = False, link: bool = False) -> None:
    sanitizer_args = [item for item in argv if item.startswith("-fsanitize=")]
    forbidden_sanitizer_overrides = (
        "-fno-sanitize=", "-fsanitize-recover=", "-fsanitize-blacklist=",
        "-fsanitize-ignorelist=", "-fsanitize-coverage-ignorelist=",
        "-fno-sanitize-coverage=",
    )
    has_suppression = any(item.startswith(forbidden_sanitizer_overrides)
                          for item in argv)
    if profile == "sanitize":
        if ("-fsanitize=address,undefined" not in sanitizer_args or
                "-fno-sanitize-recover=all" not in argv or
                "-fno-omit-frame-pointer" not in argv or
                (pic and not any(item in {"-fPIC", "-fpic"} for item in argv)) or
                (link and "-shared-libasan" not in argv) or has_suppression):
            raise EvidenceError(f"{label} command lacks required SAN/PIC flags or disables instrumentation")
    elif profile == "release" and sanitizer_args:
        raise EvidenceError(f"{label} Release command unexpectedly enables a sanitizer")


def _compile_source_record(root: Path, database: list[dict], source: Path,
                           role: str, evidence_dir: Path) -> dict:
    matches = []
    for entry in database:
        if not isinstance(entry, dict):
            continue
        directory = Path(entry.get("directory", root))
        source_path = Path(entry.get("file", ""))
        if not source_path.is_absolute():
            source_path = directory / source_path
        if source_path.resolve() == source.resolve():
            matches.append((entry, directory, source_path))
    if len(matches) != 1:
        raise EvidenceError(f"expected one actual compile command for {role}, found {len(matches)}")
    entry, directory, source_path = matches[0]
    argv = _command_arguments(entry)
    output_path = _command_output_path(argv, directory, f"{role} compile")
    if not output_path.is_file():
        raise EvidenceError(f"{role} compile object is missing")
    if FORBIDDEN_INPUTS.search(" ".join(argv)):
        raise EvidenceError(f"{role} compile command includes forbidden parser/runtime input")
    source_record = _snapshot_artifact(
        root, output_path, evidence_dir / f"{role.replace(' ', '-')}.o",
        f"{role} compiled object snapshot")
    return {"role": role, "source": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "directory": _relative(root, directory), "argv": argv,
            "object": source_record}


def _command_arguments(entry: dict) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        return arguments
    command = entry.get("command")
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError as error:
            raise EvidenceError(f"compile database command cannot be parsed: {error}") from error
    raise EvidenceError("compile database entry has no compiler command")


def _resolve_command_path(argument: str, directory: Path, label: str) -> Path:
    if not isinstance(argument, str) or not argument:
        raise EvidenceError(f"{label} path argument is empty")
    path = Path(argument)
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _command_output_path(argv: list[str], directory: Path, label: str) -> Path:
    indexes = [index for index, item in enumerate(argv) if item == "-o"]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        raise EvidenceError(f"{label} command must contain exactly one -o output")
    return _resolve_command_path(argv[indexes[0] + 1], directory, label)


def _command_object_inputs(argv: list[str], directory: Path) -> list[Path]:
    paths = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == "-o":
            skip_next = True
            continue
        if item.startswith("-") or not item.endswith(".o"):
            continue
        paths.append(_resolve_command_path(item, directory, "link object input"))
    return paths


def _command_audit_target(argv: object, expected: Path, directory: Path,
                          label: str) -> None:
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise EvidenceError(f"{label} audit argv is malformed")
    actual = _resolve_command_path(argv[-1], directory, label)
    if actual != expected.resolve():
        raise EvidenceError(f"{label} audit argv selects {actual}, expected {expected.resolve()}")


def _require_command_output(argv: list[str], directory: Path, expected: Path,
                            label: str) -> None:
    actual = _command_output_path(argv, directory, label)
    if actual != expected.resolve():
        raise EvidenceError(f"{label} -o output selects {actual}, expected {expected.resolve()}")


def _require_object_inputs(argv: list[str], directory: Path,
                           expected: list[Path], label: str) -> None:
    actual = _command_object_inputs(argv, directory)
    resolved_expected = [item.resolve() for item in expected]
    if actual != resolved_expected:
        raise EvidenceError(
            f"{label} object inputs/order differ: actual={actual}, expected={resolved_expected}")


def _require_exact_helper_archive(argv: list[str], directory: Path,
                                  expected: Path, label: str) -> None:
    archives = [item for item in argv
                if item.endswith((".a", ".lib")) and not item.startswith("-")]
    helper_archives = [item for item in archives
                       if "VulkanDeviceHelper" in Path(item).name or
                       "vulkan_device_helper" in Path(item).name.lower()]
    actual = [_resolve_command_path(item, directory, f"{label} helper archive")
              for item in helper_archives]
    if (len(archives) != 1 or len(actual) != 1 or
            actual[0] != expected.resolve()):
        raise EvidenceError(
            f"{label} does not select exactly its hash-bound helper archive: {actual}")


def _run_audit_command(root: Path, out: Path, label: str,
                       argv: list[str]) -> dict:
    result = _run_captured(root, out, label, argv)
    if result["returncode"] != 0:
        raise EvidenceError(f"{label} audit command failed ({result['returncode']})")
    return result


def _parse_linker_map_argv(argv: list[str], link_file: Path) -> Path:
    map_values = []
    has_cref = False
    index = 0

    def consume_wl(value: str) -> None:
        nonlocal has_cref
        fields = value[4:].split(",")
        cursor = 0
        while cursor < len(fields):
            field = fields[cursor]
            if field == "--cref":
                has_cref = True
            elif field.startswith("-Map="):
                map_values.append(field.split("=", 1)[1])
            elif field == "-Map":
                cursor += 1
                if cursor >= len(fields) or not fields[cursor]:
                    raise EvidenceError("-Wl,-Map is missing its output path")
                map_values.append(fields[cursor])
            elif field.startswith(("-Map", "--Map", "-map", "--map")):
                raise EvidenceError(f"unsupported linker map option: {field}")
            cursor += 1

    while index < len(argv):
        token = argv[index]
        if token.startswith("-Wl,"):
            consume_wl(token)
        elif token == "-Xlinker":
            if index + 1 >= len(argv):
                raise EvidenceError("-Xlinker is missing its forwarded argument")
            option = argv[index + 1]
            if option == "--cref":
                has_cref = True
                index += 1
            elif option.startswith("-Map="):
                map_values.append(option.split("=", 1)[1])
                index += 1
            elif option == "-Map":
                if (index + 3 >= len(argv) or argv[index + 2] != "-Xlinker" or
                        not argv[index + 3]):
                    raise EvidenceError("-Xlinker -Map is missing its forwarded output path")
                map_values.append(argv[index + 3])
                index += 3
            elif option.startswith(("-Map", "--Map", "-map", "--map")):
                raise EvidenceError(f"unsupported -Xlinker map option: {option}")
            else:
                index += 1
        elif token.startswith("-Map="):
            map_values.append(token.split("=", 1)[1])
        elif token.startswith(("-Map", "--Map", "-map", "--map")):
            raise EvidenceError(f"unsupported linker map argument: {token}")
        index += 1
    if len(map_values) != 1:
        raise EvidenceError(f"link command must contain exactly one map output, got {len(map_values)}")
    if not has_cref:
        raise EvidenceError("link map command omits --cref cross-reference output")
    requested = Path(map_values[0])
    if requested.is_absolute():
        candidates = [requested.resolve()]
    else:
        build_cwd = link_file.parent.parent.parent
        candidates = [(build_cwd / requested).resolve(),
                      link_file.parent / requested]
    existing = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate not in existing:
            existing.append(candidate)
    if len(existing) != 1:
        raise EvidenceError(f"linker map path is missing or ambiguous: {map_values[0]}")
    return existing[0]


def _link_record(root: Path, output: Path, target: str,
                 evidence_dir: Path, nm: str, readelf: str, ar: str) -> dict:
    candidates = sorted((root / "test/Vulkan/native").glob(
        f"**/CMakeFiles/{target}.dir/link.txt"))
    if len(candidates) != 1:
        raise EvidenceError(f"cannot identify unique CMake link command for {target}")
    link_path = candidates[0]
    link_text = link_path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        link_argv = shlex.split(link_text)
    except ValueError as error:
        raise EvidenceError(f"CMake link command for {target} is malformed: {error}") from error
    link_text = " ".join(link_argv)
    if FORBIDDEN_INPUTS.search(link_text):
        raise EvidenceError(f"{target} link command includes plan/parser/LLVM input")
    if not any("VulkanDeviceHelper" in token or "vulkan_device_helper" in token.lower()
               for token in link_argv):
        raise EvidenceError(f"{target} link command omits the private helper archive")
    if not any(token in {"-Wl,-z,defs", "-Wl,--no-undefined"} or
               "-Wl,-z,defs" in token or "--no-undefined" in token
               for token in link_argv):
        raise EvidenceError(f"{target} link command is not strict no-undefined")
    map_path = _parse_linker_map_argv(link_argv, link_path)
    map_text = map_path.read_text(encoding="utf-8", errors="replace")
    if FORBIDDEN_INPUTS.search(map_text):
        raise EvidenceError(f"{target} linker map contains forbidden runtime/parser inputs")
    exports = _run_audit_command(root, evidence_dir, f"{target}-exports",
                                 [nm, "-D", "--defined-only", str(output)])
    exports_out = _safe_file(root, exports["stdout"]["path"], f"{target} export audit stdout")
    defined = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_vk_(?:create|run|destroy))$",
                                    exports_out.read_text(encoding="utf-8", errors="replace"), re.M)))
    if set(defined) != EXPECTED_EXPORTS:
        raise EvidenceError(f"{target} exports do not match the three C ABI symbols: {defined}")
    dynamic = _run_audit_command(root, evidence_dir, f"{target}-dynamic",
                                 [readelf, "-d", str(output)])
    dynamic_out = _safe_file(root, dynamic["stdout"]["path"], f"{target} readelf stdout")
    dynamic_text = dynamic_out.read_text(encoding="utf-8", errors="replace")
    if re.search(r"lib(?:LLVM|MLIR|ncnn.*(?:Runtime|Plan|Parser))[^ ]*\.so", dynamic_text, re.I):
        raise EvidenceError(f"{target} DSO has a forbidden plan/parser/LLVM/runtime dependency")
    undefined = _run_audit_command(root, evidence_dir, f"{target}-undefined",
                                   [nm, "-D", "--undefined-only", str(output)])
    undefined_out = _safe_file(root, undefined["stdout"]["path"], f"{target} undefined audit stdout")
    undefined_text = undefined_out.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?:VulkanRuntimeContext|VulkanExecutionPlan|"
                 r"NCNNVulkanRuntime|NCNNVulkanPlanSupport|"
                 r"deserializeExecutionPlan|execution_plan)", undefined_text):
        raise EvidenceError(f"{target} has unresolved project runtime/parser symbols")
    helper_archives = [Path(token) for token in link_argv
                       if token.endswith((".a", ".lib")) and
                       ("VulkanDeviceHelper" in token or "vulkan_device_helper" in token.lower())]
    if len(helper_archives) != 1:
        raise EvidenceError(f"{target} link command does not identify one helper archive")
    helper = helper_archives[0]
    if not helper.is_absolute():
        helper = (link_path.parent.parent.parent / helper).resolve()
    if not helper.is_file():
        raise EvidenceError(f"{target} helper archive input is missing")
    archive_audit = _run_audit_command(root, evidence_dir, f"{target}-helper-archive",
                                       [ar, "t", str(helper)])
    archive_out = _safe_file(root, archive_audit["stdout"]["path"], f"{target} archive audit stdout")
    members = [line for line in archive_out.read_text(encoding="utf-8", errors="replace").splitlines()
               if line]
    if not members:
        raise EvidenceError(f"{target} helper archive has no members")
    return {
        "target": target,
        "dso": _artifact(root, output, f"{target} lifecycle DSO"),
        "link_file": _artifact(root, link_path, f"{target} CMake link command"),
        "link_argv": link_argv,
        "map": _artifact(root, map_path, f"{target} linker map"),
        "helper_archive": _artifact(root, helper, "private Vulkan device helper archive"),
        "helper_archive_members": members,
        "exports": defined,
        "export_audit": exports,
        "dynamic_audit": dynamic,
        "undefined_audit": undefined,
        "archive_audit": archive_audit,
    }


def _metadata_weight_bindings(source_text: str, host_program: object,
                              plan: object, manifest: object) -> list[dict]:
    if not all(isinstance(item, dict) for item in (host_program, plan, manifest)):
        raise EvidenceError("lifecycle package metadata must contain JSON objects")
    if (host_program.get("schema") != "ncnn-vulkan-host-program/v1" or
            plan.get("kind") != "ncnn.vulkan_execution_plan" or
            manifest.get("kind") != "ncnn.vulkan_model_manifest" or
            manifest.get("host_program_usage") != "compiler-only-diagnostic"):
        raise EvidenceError("lifecycle fixture package metadata schema/role is invalid")
    if manifest.get("host_program") != host_program:
        raise EvidenceError("package manifest host program differs from its serialized host program")
    if manifest.get("host_program_identity") != host_program.get("identity"):
        raise EvidenceError("package manifest/host-program identity mismatch")
    plan_resources = plan.get("resources")
    host_resources = host_program.get("resources")
    host_buffers = host_program.get("buffers")
    host_weights = host_program.get("weights")
    manifest_weights = manifest.get("weights")
    operations = host_program.get("create_operations")
    if not all(isinstance(value, list) for value in
               (plan_resources, host_resources, host_buffers, host_weights,
                manifest_weights, operations)):
        raise EvidenceError("lifecycle weight metadata arrays are malformed")
    plan_by_id = {item.get("id"): item for item in plan_resources
                  if isinstance(item, dict)}
    resources_by_id = {item.get("id"): item for item in host_resources
                       if isinstance(item, dict)}
    manifest_by_id = {item.get("resource_id"): item for item in manifest_weights
                       if isinstance(item, dict)}
    if (len(plan_by_id) != len(plan_resources) or
            len(resources_by_id) != len(host_resources) or
            len(manifest_by_id) != len(manifest_weights)):
        raise EvidenceError("lifecycle package metadata repeats resource identities")
    weight_resources = {item["id"]: item for item in plan_resources
                        if isinstance(item, dict) and item.get("kind") == "weight"}
    if (set(weight_resources) != {item.get("resource_id") for item in host_weights
                                  if isinstance(item, dict)} or
            set(weight_resources) != set(manifest_by_id)):
        raise EvidenceError("plan, host program, and manifest weight coverage differs")
    arrays = re.findall(
        r"constexpr\s+std::array<std::byte,\s*(\d+)>\s+kWeightBytes(\d+)\s*\{\{(.*?)\}\};",
        source_text, re.DOTALL)
    if len(arrays) != len(host_weights):
        raise EvidenceError("candidate source weight-array count differs from verified package weights")
    array_bytes = []
    for size_text, index_text, body in arrays:
        size = int(size_text)
        tokens = re.findall(r"std::byte\{0x([0-9a-fA-F]{2})\}", body)
        residue = re.sub(r"std::byte\{0x[0-9a-fA-F]{2}\}", "", body)
        if residue.replace(",", "").strip() or len(tokens) != size:
            raise EvidenceError("candidate weight initializer has malformed or excess byte tokens")
        array_bytes.append((int(index_text), bytes(int(token, 16) for token in tokens)))
    array_bytes.sort(key=lambda item: item[0])
    if [index for index, _ in array_bytes] != list(range(len(host_weights))):
        raise EvidenceError("candidate weight array names are not a dense ordered set")

    uploads = [item for item in operations if isinstance(item, dict) and
               item.get("op") == "upload" and item.get("source") == "weight"]
    copy_ops = [item for item in operations if isinstance(item, dict) and
                item.get("op") == "copy"]
    if len(uploads) != len(host_weights):
        raise EvidenceError("host program create operations do not upload every weight exactly once")
    bindings = []
    for index, (weight, (array_index, payload)) in enumerate(zip(host_weights, array_bytes)):
        if not isinstance(weight, dict) or array_index != index:
            raise EvidenceError("host program weights are not in candidate initializer order")
        resource_id = weight.get("resource_id")
        byte_size = weight.get("byte_size")
        physical_sha = weight.get("physical_sha256")
        logical_sha = weight.get("logical_sha256")
        if (type(resource_id) is not int or type(byte_size) is not int or byte_size <= 0 or
                not isinstance(physical_sha, str) or not SHA256_RE.fullmatch(physical_sha) or
                not isinstance(logical_sha, str) or not SHA256_RE.fullmatch(logical_sha) or
                len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != physical_sha):
            raise EvidenceError("candidate initializer bytes differ from host-program weight identity")
        plan_resource = weight_resources.get(resource_id)
        host_resource = resources_by_id.get(resource_id)
        manifest_weight = manifest_by_id.get(resource_id)
        if (not isinstance(plan_resource, dict) or not isinstance(host_resource, dict) or
                not isinstance(manifest_weight, dict) or
                type(plan_resource.get("byte_size")) is not int or
                type(host_resource.get("byte_size")) is not int or
                type(manifest_weight.get("size_bytes")) is not int or
                plan_resource["byte_size"] != byte_size or
                host_resource["byte_size"] != byte_size or
                manifest_weight["size_bytes"] != byte_size or
                plan_resource.get("weight_sha256") != logical_sha or
                host_resource.get("weight_sha256") != logical_sha or
                manifest_weight.get("sha256") != physical_sha):
            raise EvidenceError("plan/host/manifest weight metadata disagrees on byte size or identity")
        if (manifest_weight.get("resource_id") != resource_id or
                type(uploads[index].get("source_id")) is not int or
                uploads[index]["source_id"] != resource_id or
                type(uploads[index].get("byte_size")) is not int or
                uploads[index]["byte_size"] != byte_size):
            raise EvidenceError("host-program upload operation differs from verified weight metadata")
        staging_id = uploads[index].get("staging_buffer")
        staging = [item for item in host_buffers if isinstance(item, dict) and
                   item.get("id") == staging_id and item.get("kind") == "weight-staging" and
                   item.get("resource_id") == resource_id]
        if len(staging) != 1 or type(staging[0].get("byte_size")) is not int or \
                staging[0]["byte_size"] != byte_size:
            raise EvidenceError("weight staging buffer does not match the verified payload size")
        device_buffers = [item for item in host_buffers if isinstance(item, dict) and
                          item.get("kind") == "device" and item.get("resource_id") == resource_id]
        if len(device_buffers) != 1 or type(device_buffers[0].get("byte_size")) is not int or \
                device_buffers[0]["byte_size"] != byte_size:
            raise EvidenceError("weight device buffer does not match verified payload size")
        matching_copies = [item for item in copy_ops if item.get("source") == staging_id and
                           item.get("destination") == device_buffers[0].get("id")]
        if len(matching_copies) != 1:
            raise EvidenceError("host-program create operations lack one fixed copy for each weight")
        copy = matching_copies[0]
        if (type(copy.get("byte_size")) is not int or copy["byte_size"] != byte_size or
                type(copy.get("source_offset")) is not int or
                type(copy.get("destination_offset")) is not int):
            raise EvidenceError("host-program copy bytes/offsets are invalid for verified weight")
        bindings.append({"array_name": f"kWeightBytes{index}",
                         "resource_id": resource_id, "byte_size": byte_size,
                         "physical_sha256": physical_sha, "logical_sha256": logical_sha,
                         "staging_buffer": staging_id,
                         "device_buffer": device_buffers[0]["id"],
                         "source_offset": copy["source_offset"],
                         "destination_offset": copy["destination_offset"],
                         "initializer_sha256": hashlib.sha256(payload).hexdigest()})
    return bindings


def _capture_independence(args: argparse.Namespace) -> int:
    root = args.stage_root.resolve()
    evidence_path = args.output.resolve()
    out = evidence_path.parent
    build_dir = root / "test/Vulkan/native/lifecycle"
    source_root = Path(args.source_root).resolve()
    compile_path, database = _compile_database(root)
    profile = _lifecycle_build_profile(root)
    _reject_sanitizer_environment(profile)
    records = []
    snapshots = out / "artifacts"
    artifacts = [_snapshot_artifact(
        root, compile_path, snapshots / "compile_commands.json", "fresh compile database snapshot")]
    deployment_compiles = []
    for role, source, requires_pic in (
            ("Vulkan device helper", source_root / "lib/VulkanDeviceHelper/VulkanDeviceHelper.cpp", True),
            ("C11 lifecycle loader", source_root / "test/Vulkan/native/lifecycle_c_harness.c", False)):
        record = _compile_source_record(
            root, database, source, role, snapshots / "deployment")
        _require_profile_flags(record["argv"], profile, role, pic=requires_pic)
        artifacts.append(record["object"])
        deployment_compiles.append(record)
    loader_target = "vulkan_lifecycle_c_harness"
    loader_link_files = sorted((root / "test/Vulkan/native").glob(
        f"**/CMakeFiles/{loader_target}.dir/link.txt"))
    if len(loader_link_files) != 1:
        raise EvidenceError("cannot identify actual C11 lifecycle loader link command")
    loader_link_file = loader_link_files[0]
    loader_link_argv = shlex.split(loader_link_file.read_text(encoding="utf-8").strip())
    if FORBIDDEN_INPUTS.search(" ".join(loader_link_argv)):
        raise EvidenceError("C11 loader link command includes plan/parser/LLVM inputs")
    _require_profile_flags(loader_link_argv, profile, "C11 lifecycle loader link", link=True)
    loader_target_path = root / "test/Vulkan/native/vulkan_lifecycle_c_harness"
    if not loader_target_path.is_file():
        raise EvidenceError("C11 lifecycle loader executable is missing")
    loader_link_record = {
        "target": loader_target,
        "link_argv": loader_link_argv,
        "link_file": _snapshot_artifact(
            root, loader_link_file, snapshots / "deployment" / "lifecycle-loader-link.txt",
            "C11 lifecycle loader link command snapshot"),
        "executable": _snapshot_artifact(
            root, loader_target_path, snapshots / "deployment" / "vulkan_lifecycle_c_harness",
            "C11 lifecycle loader executable snapshot"),
        "dynamic_audit": _run_audit_command(
            root, out, "lifecycle-loader-dynamic",
            [args.readelf, "-d", str(loader_target_path)]),
    }
    artifacts.extend([loader_link_record["link_file"], loader_link_record["executable"],
                      loader_link_record["dynamic_audit"]["stdout"],
                      loader_link_record["dynamic_audit"]["stderr"]])
    loader_dynamic_output = _safe_file(
        root, loader_link_record["dynamic_audit"]["stdout"]["path"],
        "lifecycle loader readelf output").read_text(encoding="utf-8", errors="replace")
    if profile == "sanitize" and "libclang_rt.asan" not in loader_dynamic_output:
        raise EvidenceError("SAN C11 loader does not link the shared ASan runtime")
    if re.search(r"lib(?:LLVM|MLIR|ncnn.*(?:Runtime|Plan|Parser))[^ ]*\.so",
                 loader_dynamic_output, re.I):
        raise EvidenceError("C11 loader has a forbidden plan/parser/LLVM dependency")
    for fixture, target_prefix in (("unweighted", "vulkan_lifecycle_unweighted"),
                                   ("weighted", "vulkan_lifecycle_weighted")):
        fixture_dir = build_dir / fixture
        source = fixture_dir / "candidate_source.cpp"
        header = fixture_dir / "lifecycle_fixture_vulkan.h"
        if not source.is_file() or not header.is_file():
            raise EvidenceError(f"generated {fixture} candidate source/header is missing")
        candidates = []
        for entry in database:
            if not isinstance(entry, dict):
                continue
            directory = Path(entry.get("directory", root))
            file_path = Path(entry.get("file", ""))
            if not file_path.is_absolute():
                file_path = directory / file_path
            if file_path.resolve() == source.resolve():
                argv = _command_arguments(entry)
                command_text = " ".join(argv)
                if FORBIDDEN_INPUTS.search(command_text):
                    raise EvidenceError(f"{fixture} compile command includes forbidden compiler/runtime input")
                _require_profile_flags(argv, profile, f"{fixture} generated host", pic=True)
                candidates.append({"directory": _relative(root, directory),
                                   "file": _relative(root, file_path), "argv": argv})
        if not candidates:
            raise EvidenceError(f"{fixture} generated source has no actual compile database entry")
        outputs = set()
        for entry in candidates:
            argv = entry["argv"]
            if "-o" in argv:
                outputs.add(Path(argv[argv.index("-o") + 1]))
            for arg in argv:
                if arg.startswith("-o") and len(arg) > 2:
                    outputs.add(Path(arg[2:]))
        objects = []
        for index, object_path in enumerate(sorted(outputs, key=str)):
            if not object_path.is_absolute():
                object_path = (root / candidates[0]["directory"] / object_path).resolve()
            if object_path.is_file():
                objects.append(_snapshot_artifact(
                    root, object_path,
                    snapshots / fixture / "objects" / f"candidate-{index:02d}.o",
                    f"{fixture} generated-source object snapshot"))
        if not objects:
            raise EvidenceError(f"{fixture} compile database objects are missing")
        source_record = _snapshot_artifact(
            root, source, snapshots / fixture / "candidate_source.cpp",
            f"{fixture} generated candidate source snapshot")
        header_record = _snapshot_artifact(
            root, header, snapshots / fixture / "lifecycle_fixture_vulkan.h",
            f"{fixture} generated C ABI header snapshot")
        metadata_files = {
            "host_program": fixture_dir / "host_program.json",
            "plan": fixture_dir / "plan.json",
            "package_manifest": fixture_dir / "package_manifest.json",
            "host_program_identity": fixture_dir / "host-program-identity.txt",
            "shader": fixture_dir / "shader_0.spv",
        }
        metadata_records = {}
        for role, path in metadata_files.items():
            if not path.is_file():
                raise EvidenceError(f"generated {fixture} package metadata is missing: {role}")
            metadata_records[role] = _snapshot_artifact(
                root, path, snapshots / fixture / role / path.name,
                f"{fixture} actual {role} metadata snapshot")
        host_program = _load_json(metadata_files["host_program"],
                                  f"{fixture} generated host program")
        plan = _load_json(metadata_files["plan"], f"{fixture} generated plan")
        manifest = _load_json(metadata_files["package_manifest"],
                              f"{fixture} generated package manifest")
        identity = metadata_files["host_program_identity"].read_text(encoding="utf-8").strip()
        if not isinstance(host_program, dict) or identity != host_program.get("identity"):
            raise EvidenceError(f"{fixture} generated host-program identity file differs")
        manifest_shaders = manifest.get("shaders") if isinstance(manifest, dict) else None
        shader_digest = sha256_file(metadata_files["shader"])
        if (not isinstance(manifest_shaders, list) or len(manifest_shaders) != 1 or
                not isinstance(manifest_shaders[0], dict) or
                manifest_shaders[0].get("sha256") != shader_digest or
                type(manifest_shaders[0].get("size_bytes")) is not int or
                manifest_shaders[0]["size_bytes"] != metadata_files["shader"].stat().st_size):
            raise EvidenceError(f"{fixture} SPIR-V bytes differ from package manifest")
        source_text = source.read_text(encoding="utf-8", errors="strict")
        weight_bindings = _metadata_weight_bindings(source_text, host_program, plan, manifest)
        host_buffers = host_program.get("buffers") if isinstance(host_program, dict) else None
        context_buffer_count = (sum(1 for item in host_buffers if isinstance(item, dict) and
                                    item.get("scope") == "context")
                                if isinstance(host_buffers, list) else -1)
        if context_buffer_count <= 0:
            raise EvidenceError(f"{fixture} host program has no context-scoped buffers")
        artifacts.extend([source_record, header_record, *objects, *metadata_records.values()])
        link_records = []
        for variant in ("release", "validation"):
            target = target_prefix + ("_release" if variant == "release" else "_validation")
            output_name = target_prefix.removeprefix("vulkan_lifecycle_")
            if variant == "validation":
                output_name += "_validation"
            output = fixture_dir / f"libvulkan_lifecycle_{output_name}.so"
            if not output.is_file():
                raise EvidenceError(f"{target} candidate DSO is missing")
            link = _link_record(root, output, target, out, args.nm, args.readelf, args.ar)
            _require_profile_flags(link["link_argv"], profile, f"{target} DSO link", link=True)
            if profile == "sanitize":
                dynamic_output = _safe_file(
                    root, link["dynamic_audit"]["stdout"]["path"],
                    f"{target} sanitizer ELF audit").read_text(
                        encoding="utf-8", errors="replace")
                if "libclang_rt.asan" not in dynamic_output:
                    raise EvidenceError(f"SAN candidate DSO {target} lacks shared ASan runtime")
            target_snapshots = snapshots / fixture / target
            for key in ("dso", "link_file", "map", "helper_archive"):
                original = _safe_file(root, link[key]["path"], f"{target} {key}")
                link[key] = _snapshot_artifact(
                    root, original, target_snapshots / original.name,
                    f"{target} {key} snapshot")
                artifacts.append(link[key])
            for audit_name in ("export_audit", "dynamic_audit", "undefined_audit", "archive_audit"):
                audit_record = link[audit_name]
                artifacts.extend((audit_record["stdout"], audit_record["stderr"]))
            link_records.append(link)
        candidate_text = source.read_text(encoding="utf-8", errors="strict")
        header_text = header.read_text(encoding="utf-8", errors="strict")
        for forbidden in ("VulkanRuntime", "VulkanExecutionPlan", "ExecutionPlan",
                          "PlanSupport", "VulkanHostProgram", "VulkanModelPayload",
                          "ncnn-mlir/Support", "ncnn-mlir/Plan", "llvm/",
                          "ncnn-mlir-parser"):
            if forbidden.lower() in (candidate_text + header_text).lower():
                raise EvidenceError(f"{fixture} generated source/header mentions forbidden input {forbidden}")
        records.append({"fixture": fixture, "model": "lifecycle_fixture",
                        "role": "lifecycle-only", "compile_commands": candidates,
                        "candidate_source": source_record,
                        "generated_header": header_record,
                        "package_metadata": metadata_records,
                        "weight_bindings": weight_bindings,
                        "context_buffer_count": context_buffer_count,
                        "objects": objects, "links": link_records})
    fingerprint = _gate_fingerprint(root)
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": INDEPENDENCE_KIND,
        "role": "lifecycle-only",
        "source_fingerprint": fingerprint,
        "gate_bound": fingerprint is not None,
        "build_profile": profile,
        "deployment_compiles": deployment_compiles,
        "loader_link": loader_link_record,
        "fixtures": records,
        "raw_artifacts": artifacts,
        "claims": {"no_plan_or_parser_input": True,
                   "no_llvm_or_mlir_link_input": True,
                   "strict_no_undefined": True,
                   "export_allowlist_only": True,
                   "candidate_run": "not_run",
                   "host_aot_release_eligible": False},
    }
    _write_json(evidence_path, value)
    validate_independence_evidence(root, evidence_path, expected_fingerprint=fingerprint)
    print(f"captured Vulkan lifecycle independence evidence: {evidence_path}")
    return 0


def validate_independence_evidence(stage_root: os.PathLike[str] | str,
                                   evidence_path: os.PathLike[str] | str,
                                   *, expected_fingerprint: str | None = None) -> dict:
    root = Path(stage_root).resolve()
    path = Path(evidence_path).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise EvidenceError("lifecycle independence evidence is missing or outside stage")
    value = _load_json(path, "lifecycle independence evidence")
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or
            value.get("schema_version") != SCHEMA_VERSION or
            value.get("kind") != INDEPENDENCE_KIND or value.get("role") != "lifecycle-only"):
        raise EvidenceError("lifecycle independence schema/kind/role is invalid")
    if (expected_fingerprint is not None and
            value.get("source_fingerprint") != expected_fingerprint):
        raise EvidenceError("lifecycle independence evidence is not bound to fresh source")
    if (value.get("candidate_run") is not None or
            value.get("claims") != {"no_plan_or_parser_input": True,
                                     "no_llvm_or_mlir_link_input": True,
                                     "strict_no_undefined": True,
                                     "export_allowlist_only": True,
                                     "candidate_run": "not_run",
                                     "host_aot_release_eligible": False}):
        raise EvidenceError("lifecycle independence evidence overclaims run or dependency scope")
    refs = value.get("raw_artifacts")
    if not isinstance(refs, list) or not refs:
        raise EvidenceError("lifecycle independence evidence has no raw artifact inventory")
    seen = set()
    for item in refs:
        record = _check_artifact(root, item, "lifecycle independence raw")
        if record["path"] in seen:
            raise EvidenceError("lifecycle independence raw artifact path is duplicated")
        seen.add(record["path"])

    def require_archived_ref(record: object, label: str) -> dict:
        checked = _check_artifact(root, record, label)
        if checked["path"] not in seen:
            raise EvidenceError(f"{label} is absent from the raw archive inventory")
        if not isinstance(record, dict):
            raise EvidenceError(f"{label} evidence record is malformed")
        build_path = record.get("build_path")
        if build_path is not None:
            origin = _safe_file(root, build_path, f"{label} build origin")
            if (origin.stat().st_size != checked["size_bytes"] or
                    sha256_file(origin) != checked["sha256"]):
                raise EvidenceError(f"{label} snapshot differs from its current build-origin bytes")
        return {**record, **checked}

    profile = value.get("build_profile")
    if profile not in {"release", "sanitize", "unqualified"} or \
            profile != _lifecycle_build_profile(root):
        raise EvidenceError("lifecycle compile/link profile differs from current fresh CMake cache")
    _reject_sanitizer_environment(profile)
    source_root = _stage_source_directory(root, expected_fingerprint)
    compile_db_refs = [item for item in refs if isinstance(item, dict) and
                       item.get("role") == "fresh compile database snapshot"]
    if len(compile_db_refs) != 1:
        raise EvidenceError("lifecycle independence lacks one actual compile database snapshot")
    compile_db_path = _safe_file(root, compile_db_refs[0]["path"],
                                 "lifecycle compile database snapshot")
    compile_db = _load_json(compile_db_path, "lifecycle compile database snapshot")
    if not isinstance(compile_db, list):
        raise EvidenceError("lifecycle compile database snapshot is malformed")

    def match_compile_record(source: str, directory: str, argv: list[str], label: str) -> None:
        source_path = Path(source)
        if not source_path.is_absolute() or not source_path.is_file():
            raise EvidenceError(f"{label} source file is missing")
        matched = []
        for entry in compile_db:
            if not isinstance(entry, dict):
                continue
            entry_directory = Path(entry.get("directory", root)).resolve()
            entry_source = Path(entry.get("file", ""))
            if not entry_source.is_absolute():
                entry_source = entry_directory / entry_source
            if (entry_source.resolve() == source_path.resolve() and
                    entry_directory == (root / directory).resolve() and
                    _command_arguments(entry) == argv):
                matched.append(entry)
        if len(matched) != 1:
            raise EvidenceError(f"{label} provenance differs from the captured compile database")

    deployment_compiles = value.get("deployment_compiles")
    expected_deployment = {
        "Vulkan device helper": "lib/VulkanDeviceHelper/VulkanDeviceHelper.cpp",
        "C11 lifecycle loader": "test/Vulkan/native/lifecycle_c_harness.c",
    }
    if (not isinstance(deployment_compiles, list) or len(deployment_compiles) != 2 or
            {item.get("role") for item in deployment_compiles if isinstance(item, dict)} !=
            set(expected_deployment)):
        raise EvidenceError("lifecycle evidence omits helper/C-loader compile provenance")
    for compile_record in deployment_compiles:
        role = compile_record.get("role")
        source_path = compile_record.get("source")
        if (not isinstance(source_path, str) or not Path(source_path).is_absolute() or
                Path(source_path).resolve() !=
                (source_root / expected_deployment[role]).resolve() or
                not Path(source_path).is_file() or
                not isinstance(compile_record.get("source_sha256"), str) or
                sha256_file(source_path) != compile_record["source_sha256"] or
                not isinstance(compile_record.get("directory"), str) or
                not isinstance(compile_record.get("argv"), list)):
            raise EvidenceError(f"{role} source/compile identity is invalid")
        object_record = require_archived_ref(compile_record.get("object"), f"{role} object")
        match_compile_record(source_path, compile_record["directory"],
                             compile_record["argv"], role)
        compile_output = _command_output_path(
            compile_record["argv"], root / compile_record["directory"], f"{role} compile")
        if (not isinstance(object_record.get("build_path"), str) or
                _relative(root, compile_output) != object_record["build_path"]):
            raise EvidenceError(f"{role} compile -o output differs from its archived object")
        _require_profile_flags(compile_record["argv"], profile, role,
                               pic=role == "Vulkan device helper")
    loader_link = value.get("loader_link")
    if not isinstance(loader_link, dict) or loader_link.get("target") != "vulkan_lifecycle_c_harness":
        raise EvidenceError("lifecycle evidence omits the actual C11 loader link target")
    loader_link_file_record = require_archived_ref(loader_link.get("link_file"),
                                                   "C11 loader link file")
    loader_executable_record = require_archived_ref(loader_link.get("executable"),
                                                    "C11 loader executable")
    loader_link_argv = loader_link.get("link_argv")
    if (not isinstance(loader_link_argv, list) or
            FORBIDDEN_INPUTS.search(" ".join(map(str, loader_link_argv)))):
        raise EvidenceError("C11 loader link command is malformed or contains forbidden inputs")
    _require_profile_flags(loader_link_argv, profile, "C11 lifecycle loader link", link=True)
    loader_link_file = _safe_file(root, loader_link_file_record["path"],
                                  "C11 loader link command snapshot")
    if shlex.split(loader_link_file.read_text(encoding="utf-8").strip()) != loader_link_argv:
        raise EvidenceError("C11 loader link argv differs from captured link.txt bytes")
    loader_link_build_path = loader_link_file_record.get("build_path")
    if (loader_link_build_path !=
            "test/Vulkan/native/CMakeFiles/vulkan_lifecycle_c_harness.dir/link.txt"):
        raise EvidenceError("C11 loader link origin is not its CMake target link file")
    loader_link_cwd = (root / loader_link_build_path).parent.parent.parent
    loader_executable_build_path = loader_executable_record.get("build_path")
    if loader_executable_build_path != "test/Vulkan/native/vulkan_lifecycle_c_harness":
        raise EvidenceError("C11 loader executable is not the declared CMake target output")
    _require_command_output(loader_link_argv, loader_link_cwd,
                            root / loader_executable_build_path, "C11 loader link")
    loader_compile = next(item for item in deployment_compiles
                          if item.get("role") == "C11 lifecycle loader")
    loader_object = loader_compile.get("object")
    loader_object_build_path = loader_object.get("build_path") if isinstance(loader_object, dict) else None
    if not isinstance(loader_object_build_path, str):
        raise EvidenceError("C11 loader compile object omits its build-origin path")
    _require_object_inputs(loader_link_argv, loader_link_cwd,
                           [root / loader_object_build_path], "C11 loader link")
    dynamic = loader_link.get("dynamic_audit")
    if (not isinstance(dynamic, dict) or type(dynamic.get("returncode")) is not int or
            dynamic["returncode"] != 0 or not isinstance(dynamic.get("argv"), list)):
        raise EvidenceError("C11 loader lacks an actual ELF dependency audit")
    _command_audit_target(dynamic["argv"], root / loader_executable_build_path,
                          root, "C11 loader ELF")
    for stream in ("stdout", "stderr"):
        require_archived_ref(dynamic.get(stream), f"C11 loader readelf {stream}")
    dynamic_output = _safe_file(root, dynamic["stdout"]["path"],
                                "C11 loader readelf output").read_text(
                                    encoding="utf-8", errors="replace")
    if profile == "sanitize" and "libclang_rt.asan" not in dynamic_output:
        raise EvidenceError("SAN C11 loader ELF does not need the shared ASan runtime")
    if re.search(r"lib(?:LLVM|MLIR|ncnn.*(?:Runtime|Plan|Parser))[^ ]*\.so",
                 dynamic_output, re.I):
        raise EvidenceError("C11 loader ELF depends on a forbidden compiler/runtime DSO")

    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 2 or {item.get("fixture") for item in fixtures
                                          if isinstance(item, dict)} != {"unweighted", "weighted"}:
        raise EvidenceError("lifecycle independence must cover unweighted and weighted fixtures")
    for fixture in fixtures:
        if not isinstance(fixture, dict) or fixture.get("role") != "lifecycle-only" or fixture.get("model") != "lifecycle_fixture":
            raise EvidenceError("lifecycle fixture role/model identity is invalid")
        source_record = require_archived_ref(fixture.get("candidate_source"), "candidate source")
        header_record = require_archived_ref(fixture.get("generated_header"), "candidate header")
        fixture_origin = f"test/Vulkan/native/lifecycle/{fixture['fixture']}"
        if (source_record.get("build_path") != f"{fixture_origin}/candidate_source.cpp" or
                header_record.get("build_path") != f"{fixture_origin}/lifecycle_fixture_vulkan.h"):
            raise EvidenceError("candidate source/header is not the declared fixture output")
        metadata_records = fixture.get("package_metadata")
        if not isinstance(metadata_records, dict) or set(metadata_records) != {
                "host_program", "plan", "package_manifest", "host_program_identity", "shader"}:
            raise EvidenceError("lifecycle fixture omits captured compiler package metadata")
        metadata_paths = {key: _safe_file(root, require_archived_ref(
            item, f"{fixture['fixture']} {key} metadata")["path"],
            f"{fixture['fixture']} {key} metadata")
            for key, item in metadata_records.items()}
        host_program = _load_json(metadata_paths["host_program"], "captured host program")
        plan = _load_json(metadata_paths["plan"], "captured execution plan diagnostic")
        manifest = _load_json(metadata_paths["package_manifest"], "captured package manifest")
        host_identity = metadata_paths["host_program_identity"].read_text(
            encoding="utf-8", errors="strict").strip()
        if not isinstance(host_program, dict) or host_identity != host_program.get("identity"):
            raise EvidenceError("captured package identity text differs from host program")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("shaders"), list):
            raise EvidenceError("captured package manifest shader evidence is malformed")
        shader_record = metadata_records["shader"]
        shader_path = metadata_paths["shader"]
        if (len(manifest["shaders"]) != 1 or
                manifest["shaders"][0].get("sha256") != sha256_file(shader_path) or
                type(manifest["shaders"][0].get("size_bytes")) is not int or
                manifest["shaders"][0]["size_bytes"] != shader_path.stat().st_size):
            raise EvidenceError("captured shader bytes differ from package manifest")
        source_path = _safe_file(root, source_record["path"], "candidate source")
        expected_weight_bindings = _metadata_weight_bindings(
            source_path.read_text(encoding="utf-8", errors="strict"),
            host_program, plan, manifest)
        if fixture.get("weight_bindings") != expected_weight_bindings:
            raise EvidenceError("candidate weight byte bindings differ from captured package metadata")
        buffers = host_program.get("buffers") if isinstance(host_program, dict) else None
        expected_context_buffers = (sum(1 for item in buffers if isinstance(item, dict) and
                                        item.get("scope") == "context")
                                    if isinstance(buffers, list) else -1)
        if (expected_context_buffers <= 0 or
                type(fixture.get("context_buffer_count")) is not int or
                fixture["context_buffer_count"] != expected_context_buffers):
            raise EvidenceError("candidate fake-test expected buffer count differs from host program metadata")
        objects = fixture.get("objects")
        links = fixture.get("links")
        compile_records = fixture.get("compile_commands")
        if (not isinstance(objects, list) or not objects or not isinstance(links, list) or len(links) != 2 or
                not isinstance(compile_records, list) or not compile_records):
            raise EvidenceError("lifecycle fixture compile/object/link provenance is incomplete")
        object_origins = set()
        for obj in objects:
            require_archived_ref(obj, "candidate object")
            origin = obj.get("build_path")
            if (not isinstance(origin, str) or Path(origin).is_absolute() or
                    ".." in Path(origin).parts or origin in object_origins):
                raise EvidenceError("candidate object lacks a unique safe build origin")
            object_origins.add(origin)
        source_build_path = source_record.get("build_path")
        if not isinstance(source_build_path, str):
            raise EvidenceError("candidate source snapshot omits its build-origin path")
        for compile_record in compile_records:
            if (not isinstance(compile_record, dict) or
                    compile_record.get("file") != source_build_path or
                    not isinstance(compile_record.get("directory"), str) or
                    Path(compile_record["directory"]).is_absolute() or
                    ".." in Path(compile_record["directory"]).parts or
                    not isinstance(compile_record.get("argv"), list) or
                    FORBIDDEN_INPUTS.search(" ".join(map(str, compile_record["argv"])) )):
                raise EvidenceError("candidate compile record is malformed or has forbidden inputs")
            argv = compile_record["argv"]
            match_compile_record(str(root / source_build_path),
                                 compile_record["directory"], argv,
                                 f"{fixture} generated source")
            _require_profile_flags(argv, profile, f"{fixture} generated host", pic=True)
            object_output = _command_output_path(
                argv, root / compile_record["directory"], f"{fixture} candidate compile")
            object_build_path = _relative(root, object_output)
            if object_build_path not in object_origins:
                raise EvidenceError("candidate compile -o output is not among its hash-bound objects")
        if {link.get("target") for link in links if isinstance(link, dict)} != {
                f"vulkan_lifecycle_{fixture['fixture']}_release",
                f"vulkan_lifecycle_{fixture['fixture']}_validation"}:
            raise EvidenceError("lifecycle fixture does not bind regular and validation DSOs")
        for link in links:
            if not isinstance(link, dict):
                raise EvidenceError("lifecycle link record is malformed")
            target = link.get("target")
            if not isinstance(target, str):
                raise EvidenceError("lifecycle link target name is malformed")
            dso_record = require_archived_ref(link.get("dso"),
                                              f"{link.get('target')} DSO")
            link_file_record = require_archived_ref(link.get("link_file"),
                                                    f"{link.get('target')} link file")
            map_record = require_archived_ref(link.get("map"),
                                              f"{link.get('target')} map")
            helper_record = require_archived_ref(link.get("helper_archive"),
                                                 f"{link.get('target')} helper archive")
            link_file = _safe_file(root, link_file_record["path"],
                                   "lifecycle link command snapshot")
            if set(link.get("exports", [])) != EXPECTED_EXPORTS:
                raise EvidenceError("lifecycle DSO export set differs from the C ABI allowlist")
            if not isinstance(link.get("link_argv"), list) or FORBIDDEN_INPUTS.search(
                    " ".join(map(str, link["link_argv"]))):
                raise EvidenceError("lifecycle link provenance contains forbidden inputs")
            link_argv = link["link_argv"]
            if not any("-Wl,-z,defs" in token or "--no-undefined" in token
                       for token in link_argv):
                raise EvidenceError("lifecycle link argv omits strict no-undefined")
            link_build_path = link_file_record.get("build_path")
            dso_build_path = dso_record.get("build_path")
            helper_build_path = helper_record.get("build_path")
            map_build_path = map_record.get("build_path")
            if any(not isinstance(item, str) or Path(item).is_absolute() or
                   ".." in Path(item).parts
                   for item in (link_build_path, dso_build_path,
                                helper_build_path, map_build_path)):
                raise EvidenceError("lifecycle link artifact omits a safe build-origin path")
            target_suffix = "_validation" if target.endswith("_validation") else ""
            expected_link_build_path = (
                f"test/Vulkan/native/CMakeFiles/{target}.dir/link.txt")
            expected_dso_build_path = (
                f"test/Vulkan/native/lifecycle/{fixture['fixture']}/"
                f"libvulkan_lifecycle_{fixture['fixture']}{target_suffix}.so")
            expected_map_build_path = (
                f"test/Vulkan/native/lifecycle/{fixture['fixture']}/"
                f"vulkan_lifecycle_{fixture['fixture']}{target_suffix}.map")
            if (link_build_path != expected_link_build_path or
                    dso_build_path != expected_dso_build_path or
                    helper_build_path != "lib/VulkanDeviceHelper/libNCNNVulkanDeviceHelper.a" or
                    map_build_path != expected_map_build_path):
                raise EvidenceError("lifecycle link artifacts are not the declared CMake outputs")
            link_origin = _safe_file(root, link_build_path,
                                     f"{link.get('target')} link origin")
            link_cwd = link_origin.parent.parent.parent
            _require_command_output(link_argv, link_cwd, root / dso_build_path,
                                    f"{target} link")
            expected_objects = [item for item in object_origins
                                if f"CMakeFiles/{target}.dir/" in item]
            if len(expected_objects) != 1:
                raise EvidenceError(f"{target} has no unique fixture object snapshot")
            _require_object_inputs(link_argv, link_cwd,
                                   [root / expected_objects[0]], f"{target} link")
            _require_exact_helper_archive(link_argv, link_cwd,
                                          root / helper_build_path, f"{target} link")
            parsed_map = _parse_linker_map_argv(link_argv, link_origin)
            if _relative(root, parsed_map) != map_build_path:
                raise EvidenceError("linker map argv does not bind its snapshotted map file")
            if not isinstance(link.get("helper_archive_members"), list) or not link["helper_archive_members"]:
                raise EvidenceError("lifecycle link lacks helper archive member provenance")
            try:
                actual_link_argv = shlex.split(link_file.read_text(encoding="utf-8").strip())
            except ValueError as error:
                raise EvidenceError(f"lifecycle link command snapshot is malformed: {error}") from error
            if actual_link_argv != link["link_argv"]:
                raise EvidenceError("lifecycle link argv differs from its captured link.txt bytes")
            map_path = _safe_file(root, link["map"]["path"], "lifecycle link map snapshot")
            if FORBIDDEN_INPUTS.search(map_path.read_text(encoding="utf-8", errors="replace")):
                raise EvidenceError("lifecycle link map contains forbidden plan/parser/LLVM inputs")
            audit_outputs = {}
            for field in ("export_audit", "dynamic_audit", "undefined_audit", "archive_audit"):
                command = link.get(field)
                if (not isinstance(command, dict) or type(command.get("returncode")) is not int or
                        command["returncode"] != 0 or not isinstance(command.get("argv"), list)):
                    raise EvidenceError(f"lifecycle raw {field} command is missing or failed")
                audit_target = (root / helper_build_path
                                if field == "archive_audit" else root / dso_build_path)
                _command_audit_target(command["argv"], audit_target, root,
                                      f"{link.get('target')} {field}")
                for stream in ("stdout", "stderr"):
                    require_archived_ref(command.get(stream), f"lifecycle {field} {stream}")
                audit_outputs[field] = _safe_file(root, command["stdout"]["path"],
                                                  f"lifecycle {field} output").read_text(
                                                      encoding="utf-8", errors="replace")
            exports = sorted(set(re.findall(
                r"\b([A-Za-z_][A-Za-z0-9_]*_vk_(?:create|run|destroy))$",
                audit_outputs["export_audit"], re.M)))
            if set(exports) != EXPECTED_EXPORTS or set(exports) != set(link["exports"]):
                raise EvidenceError("captured nm output does not prove the three C ABI exports")
            if re.search(r"lib(?:LLVM|MLIR|ncnn.*(?:Runtime|Plan|Parser))[^ ]*\.so",
                         audit_outputs["dynamic_audit"], re.I):
                raise EvidenceError("captured ELF dependency output contains forbidden libraries")
            if re.search(r"(?:VulkanRuntimeContext|VulkanExecutionPlan|"
                         r"NCNNVulkanRuntime|NCNNVulkanPlanSupport|"
                         r"deserializeExecutionPlan|execution_plan)",
                         audit_outputs["undefined_audit"]):
                raise EvidenceError("captured undefined-symbol output contains a forbidden runtime")
            if [line for line in audit_outputs["archive_audit"].splitlines() if line] != link["helper_archive_members"]:
                raise EvidenceError("captured helper archive listing differs from recorded members")
    return value


def validate_native_capture(stage_root: os.PathLike[str] | str,
                            capture_path: os.PathLike[str] | str,
                            *, mode: str, device: str | None = None,
                            expected_fingerprint: str | None = None) -> dict:
    root = Path(stage_root).resolve()
    _reject_sanitizer_environment(_lifecycle_build_profile(root))
    path = Path(capture_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise EvidenceError(f"{mode} lifecycle capture is missing or outside stage")
    value = _load_json(path, f"{mode} lifecycle capture")
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or
            value.get("schema_version") != SCHEMA_VERSION or
            value.get("kind") != CAPTURE_KIND or value.get("role") != "lifecycle-only" or
            value.get("mode") != mode):
        raise EvidenceError(f"{mode} lifecycle capture schema/kind/role is invalid")
    if expected_fingerprint is not None and value.get("source_fingerprint") != expected_fingerprint:
        raise EvidenceError(f"{mode} lifecycle capture source fingerprint differs from fresh gate")
    command = value.get("command")
    if (not isinstance(command, dict) or type(command.get("returncode")) is not int or
            command.get("returncode") != 0 or not isinstance(command.get("argv"), list)):
        raise EvidenceError(f"{mode} lifecycle command record is invalid")
    if device is not None:
        selector = {"nvidia": "NVIDIA", "llvmpipe": "llvmpipe"}.get(device, device)
        if value.get("device") != selector:
            raise EvidenceError("lifecycle capture device identity differs from expected ICD")
        overrides = command.get("environment_overrides")
        if not isinstance(overrides, dict) or overrides.get("VK_ICD_FILENAMES") != overrides.get("VK_DRIVER_FILES"):
            raise EvidenceError("lifecycle command did not bind both ICD loader variables")
        icd_manifest = value.get("icd_manifest")
        if (not isinstance(icd_manifest, dict) or
                overrides.get("VK_ICD_FILENAMES") != icd_manifest.get("origin_path")):
            raise EvidenceError("lifecycle command ICD environment differs from captured manifest origin")
        if overrides.get("VK_INSTANCE_LAYERS") != "VK_LAYER_KHRONOS_validation":
            raise EvidenceError("lifecycle device command did not force Khronos validation")
        info = command.get("vulkaninfo")
        if (not isinstance(info, dict) or type(info.get("returncode")) is not int or
                info["returncode"] != 0):
            raise EvidenceError("lifecycle capture lacks successful actual vulkaninfo output")
        for stream in ("stdout", "stderr"):
            _check_artifact(root, info.get(stream), f"{device} vulkaninfo {stream}")
        info_path = _safe_file(root, info["stdout"]["path"], f"{device} vulkaninfo summary")
        summary = info_path.read_text(encoding="utf-8", errors="replace")
        if "VK_LAYER_KHRONOS_validation" not in summary:
            raise EvidenceError("captured vulkaninfo summary does not show the validation layer")
        identity = _parse_vulkan_identity(summary, selector)
        if info.get("physical_device_identity") != identity:
            raise EvidenceError("recorded physical-device identity differs from raw vulkaninfo bytes")
    else:
        overrides = command.get("environment_overrides")
        disabled_icd = overrides.get("VK_ICD_FILENAMES") if isinstance(overrides, dict) else None
        if (not isinstance(disabled_icd, str) or
                disabled_icd != overrides.get("VK_DRIVER_FILES") or
                Path(disabled_icd).exists() or
                Path(disabled_icd).name != "missing-va3-icd.json"):
            raise EvidenceError("no-device lifecycle command did not explicitly disable device discovery")
    library = _check_artifact(root, value.get("library"), f"{mode} lifecycle library")
    harness = _check_artifact(root, value.get("harness"), f"{mode} lifecycle harness")
    _check_artifact(root, value.get("native_result"), f"{mode} lifecycle native result")
    if value.get("icd_manifest") is not None:
        _check_artifact(root, value["icd_manifest"], f"{mode} lifecycle ICD manifest")
        icd_library = value.get("icd_library")
        if (not isinstance(icd_library, dict) or
                not isinstance(icd_library.get("path"), str) or
                not Path(icd_library["path"]).is_absolute() or
                not isinstance(icd_library.get("sha256"), str) or
                not SHA256_RE.fullmatch(icd_library["sha256"]) or
                type(icd_library.get("size_bytes")) is not int or
                icd_library["size_bytes"] < 0):
            raise EvidenceError(f"{mode} capture lacks actual ICD library identity")
        icd_library_path = Path(icd_library["path"])
        if (not icd_library_path.is_file() or
                icd_library_path.stat().st_size != icd_library["size_bytes"] or
                sha256_file(icd_library_path) != icd_library["sha256"]):
            raise EvidenceError(f"{mode} actual ICD shared library identity changed")
        manifest_path = _safe_file(root, value["icd_manifest"]["path"],
                                   f"{mode} captured ICD manifest")
        manifest = _load_json(manifest_path, f"{mode} captured ICD manifest")
        manifest_icd = manifest.get("ICD") if isinstance(manifest, dict) else None
        manifest_library = manifest_icd.get("library_path") if isinstance(manifest_icd, dict) else None
        origin_path = Path(value["icd_manifest"]["origin_path"])
        if not isinstance(manifest_library, str) or not manifest_library:
            raise EvidenceError("captured ICD manifest has no library path")
        expected_library = device_helper_audit._icd_library_path(
            origin_path, manifest_library).resolve()
        if expected_library.resolve() != icd_library_path.resolve():
            raise EvidenceError("captured ICD library identity does not match its manifest bytes")
    command_argv = command["argv"]
    if (not command_argv or Path(command_argv[0]).name != Path(harness["path"]).name or
            not isinstance(value["harness"].get("build_path"), str) or
            (root / value["harness"]["build_path"]).resolve() != Path(command_argv[0]).resolve()):
        raise EvidenceError(f"{mode} invocation is not bound to its captured harness binary")
    if "--library" not in command_argv or "--result" not in command_argv:
        raise EvidenceError(f"{mode} invocation omits its candidate library/result arguments")
    library_arg = command_argv[command_argv.index("--library") + 1]
    if (not isinstance(value["library"].get("build_path"), str) or
            (root / value["library"]["build_path"]).resolve() != Path(library_arg).resolve()):
        raise EvidenceError(f"{mode} invocation is not bound to its captured candidate DSO")
    result_arg = Path(command_argv[command_argv.index("--result") + 1]).resolve()
    native_result_path = _safe_file(root, value["native_result"]["path"],
                                    f"{mode} native result")
    if native_result_path.resolve() != result_arg:
        raise EvidenceError(f"{mode} invocation is not bound to its actual native result file")
    expected_flag = "--validation" if mode == "device" else "--no-device"
    if expected_flag not in command_argv:
        raise EvidenceError(f"{mode} invocation omits required harness mode flag {expected_flag}")
    if mode == "device":
        identity = command["vulkaninfo"]["physical_device_identity"]
        if "--ordinal" not in command_argv:
            raise EvidenceError("device lifecycle harness did not receive selected Vulkan ordinal")
        ordinal = command_argv[command_argv.index("--ordinal") + 1]
        native_observations = value.get("native_observations")
        if (not ordinal.isdecimal() or
                int(ordinal) != identity.get("enumeration_index") or
                value.get("requested_device_index") != int(ordinal) or
                not isinstance(native_observations, dict) or
                native_observations.get("requested_device_index") != int(ordinal)):
            raise EvidenceError("candidate create ordinal differs from the actual vulkaninfo enumeration")
        if ("--device-match" not in command_argv or
                command_argv[command_argv.index("--device-match") + 1] != selector or
                native_observations.get("expected_device_match") != selector):
            raise EvidenceError("expected-device metadata differs from actual ICD test selection")
    elif "--ordinal" in command_argv:
        raise EvidenceError("no-device harness unexpectedly received a device ordinal")
    if mode == "device" and "weighted_validation" not in Path(library_arg).name:
        raise EvidenceError("real ICD lifecycle must execute the weighted validation candidate DSO")
    if mode == "no-device" and "weighted_validation" in Path(library_arg).name:
        raise EvidenceError("no-device rejection must use the ordinary candidate DSO")
    if type(value.get("validation_error_count")) is not int:
        raise EvidenceError("lifecycle capture validation error count must be an integer")
    command_logs = [command]
    if isinstance(command.get("vulkaninfo"), dict):
        command_logs.append(command["vulkaninfo"])
    log_texts = [
        _safe_file(root, item[key]["path"], f"{mode} {key}").read_text(
            encoding="utf-8", errors="replace")
        for item in command_logs for key in ("stdout", "stderr")]
    observed_errors = sum(
        1 for text in log_texts for line in text.splitlines()
        if VALIDATION_ERROR_RE.search(line))
    if observed_errors != value["validation_error_count"] or observed_errors != 0:
        raise EvidenceError("lifecycle raw harness logs contain validation errors or mismatch count")
    selected_device_index = (
        command["vulkaninfo"]["physical_device_identity"]["enumeration_index"]
        if device is not None else None)
    _validate_native_observations(
        value.get("native_observations"), mode,
        {"nvidia": "NVIDIA", "llvmpipe": "llvmpipe"}.get(device, device),
        selected_device_index)
    for stream in ("stdout", "stderr"):
        _check_artifact(root, command.get(stream), f"{mode} harness {stream}")
    if device is not None:
        for stream in ("stdout", "stderr"):
            _check_artifact(root, command["vulkaninfo"].get(stream), f"{device} vulkaninfo {stream}")
    return value


def _validate_native_observations(value: object, mode: str,
                                  device: str | None,
                                  device_index: int | None = None) -> None:
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or
            value.get("schema_version") != SCHEMA_VERSION or
            value.get("kind") != NATIVE_RESULT_KIND or value.get("mode") != mode or
            value.get("fixture_id") != "lifecycle_fixture" or
            value.get("candidate_execution") != "not_run"):
        raise EvidenceError("native lifecycle result schema/fixture/run claim is invalid")
    expected_match = device or ""
    if value.get("expected_device_match") != expected_match:
        raise EvidenceError("native C harness expected-device metadata differs from the runner")
    if value.get("status") != "passed":
        raise EvidenceError("native lifecycle C ABI harness did not report passed")
    if type(value.get("validation_enabled")) is not bool:
        raise EvidenceError("native lifecycle validation_enabled must be a JSON boolean")
    if mode == "device" and value["validation_enabled"] is not True:
        raise EvidenceError("device lifecycle harness did not enable validation")
    if mode == "no-device" and value["validation_enabled"] is not False:
        raise EvidenceError("no-device lifecycle harness unexpectedly enabled validation")
    for field in ("create_status", "destroy_status", "destroy_null_status",
                  "options_null_status", "context_output_null_status",
                  "options_bad_size_status", "options_bad_version_status",
                  "options_bad_flags_status", "invalid_ordinal_status", "run_status",
                  "run_null_context_status", "run_null_input_status",
                  "run_null_output_status", "run_bad_input_count_status",
                  "run_bad_output_count_status", "created_contexts",
                  "requested_device_index"):
        if type(value.get(field)) is not int:
            raise EvidenceError(f"native lifecycle {field} must be a JSON integer status/count")
    for field in ("independent_contexts", "repeated_create_destroy",
                  "run_output_unchanged", "invalid_run_sentinels_unchanged",
                  "diagnostic_bounded"):
        if type(value.get(field)) is not bool:
            raise EvidenceError(f"native lifecycle {field} must be a JSON boolean")
    if any(value[field] != 1 for field in ("options_null_status",
                                            "context_output_null_status",
                                            "options_bad_size_status",
                                            "options_bad_version_status",
                                            "options_bad_flags_status")):
        raise EvidenceError("native lifecycle accepted an invalid C ABI argument")
    if value["destroy_null_status"] != 0 or not value["diagnostic_bounded"]:
        raise EvidenceError("native lifecycle null-destroy or bounded-diagnostic contract failed")
    if mode == "device":
        if (type(device_index) is not int or value["requested_device_index"] != device_index or
                value["create_status"] != 0 or value["destroy_status"] != 0 or
                value["invalid_ordinal_status"] != 2 or value["run_status"] != 5 or
                value["run_null_context_status"] != 1 or
                value["run_null_input_status"] != 1 or
                value["run_null_output_status"] != 1 or
                value["run_bad_input_count_status"] != 6 or
                value["run_bad_output_count_status"] != 6 or
                value["created_contexts"] < 3 or not value["independent_contexts"] or
                not value["repeated_create_destroy"] or
                not value["run_output_unchanged"] or
                not value["invalid_run_sentinels_unchanged"]):
            raise EvidenceError(
                f"{device or 'device'} lifecycle C ABI observations are incomplete or unsuccessful")
    elif mode == "no-device":
        if (value["create_status"] != 2 or value["requested_device_index"] != 0xFFFFFFFF or
                value["destroy_status"] != 0xFFFFFFFF or
                value["invalid_ordinal_status"] != 0xFFFFFFFF or
                value["run_status"] != 0xFFFFFFFF or
                value["run_null_context_status"] != 0xFFFFFFFF or
                value["run_null_input_status"] != 0xFFFFFFFF or
                value["run_null_output_status"] != 0xFFFFFFFF or
                value["run_bad_input_count_status"] != 0xFFFFFFFF or
                value["run_bad_output_count_status"] != 0xFFFFFFFF or
                value["created_contexts"] != 0 or value["independent_contexts"] or
                value["repeated_create_destroy"]):
            raise EvidenceError("no-device lifecycle create did not fail closed with a null context")
    else:
        raise EvidenceError(f"unsupported native lifecycle mode {mode}")


def validate_fake_result(value: object, *, weight_bindings: list[dict],
                         expected_context_buffers: int) -> dict:
    if (not isinstance(weight_bindings, list) or not weight_bindings or
            type(expected_context_buffers) is not int or expected_context_buffers <= 0 or
            any(type(item.get("byte_size")) is not int or item["byte_size"] <= 0
                for item in weight_bindings if isinstance(item, dict)) or
            any(not isinstance(item, dict) for item in weight_bindings)):
        raise EvidenceError("verified host-program weight/context metadata is incomplete")
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or
            value.get("schema_version") != SCHEMA_VERSION or
            value.get("kind") != FAKE_RESULT_KIND or
            value.get("candidate_execution") != "not_run" or
            value.get("status") != "passed"):
        raise EvidenceError("lifecycle fake result schema/kind/status is invalid")
    budget = value.get("budget_preflight")
    if (not isinstance(budget, dict) or type(budget.get("status")) is not int or
            budget["status"] != 3 or type(budget.get("allocations_after_rejection")) is not int or
            budget["allocations_after_rejection"] != 0 or
            type(budget.get("cleanup_all_objects_zero")) is not bool or
            budget["cleanup_all_objects_zero"] is not True):
        raise EvidenceError("fake budget preflight did not reject before allocation and clean up")
    failures = value.get("partial_create_failures")
    expected_failures = {
        "vkCreateBuffer": 3, "vkAllocateMemory": 3, "vkBindBufferMemory": 4,
        "vkMapMemory": 5, "vkFlushMappedMemoryRanges": 4,
        "vkCreateDescriptorPool": 3, "vkCreateShaderModule": 3,
        "vkQueueSubmit": 4, "vkWaitForFences": 4,
    }
    if not isinstance(failures, list) or len(failures) != len(expected_failures):
        raise EvidenceError("fake result omits partial-create fault observations")
    seen = set()
    for case in failures:
        if (not isinstance(case, dict) or case.get("function") not in expected_failures or
                case["function"] in seen or type(case.get("vk_result")) is not int or
                type(case.get("status")) is not int or
                case["status"] != expected_failures[case["function"]] or
                type(case.get("cleanup_all_objects_zero")) is not bool or
                case["cleanup_all_objects_zero"] is not True):
            raise EvidenceError("fake partial-create fault status/cleanup is malformed")
        seen.add(case["function"])
    if seen != set(expected_failures):
        raise EvidenceError("fake result omits a required partial-create failure point")
    operation = value.get("operation_wait_failure")
    if (not isinstance(operation, dict) or type(operation.get("status")) is not int or
            operation["status"] != 4 or
            type(operation.get("vkCmdCopyBuffer_call_count")) is not int or
            operation["vkCmdCopyBuffer_call_count"] != len(weight_bindings) or
            type(operation.get("copy_arguments_valid")) is not bool or
            operation["copy_arguments_valid"] is not True or
            type(operation.get("cleanup_all_objects_zero")) is not bool or
            operation["cleanup_all_objects_zero"] is not True):
        raise EvidenceError("fake post-submit wait failure did not preserve copy/cleanup observations")
    weighted = value.get("weighted_upload")
    if not isinstance(weighted, dict):
        raise EvidenceError("fake result omits weighted lifecycle observations")
    numeric_fields = ("observed_weight_upload_count", "expected_weight_upload_count",
                      "vkCmdCopyBuffer_call_count", "expected_vkCmdCopyBuffer_call_count",
                      "region_count", "recorded_region_count", "live_buffer_count_after_create",
                      "staging_buffers_destroyed_before_create_return", "peak_memory_bytes")
    for field in numeric_fields:
        if type(weighted.get(field)) is not int or weighted[field] < 0:
            raise EvidenceError(f"fake weighted result {field} must be a nonnegative JSON integer")
    records = weighted.get("records")
    expected_count = len(weight_bindings)
    if (not isinstance(records, list) or len(records) != expected_count or
            weighted["region_count"] != expected_count or
            weighted["recorded_region_count"] != len(records)):
        raise EvidenceError("fake helper VkBufferCopy records differ from metadata weight coverage")
    for record, expected in zip(records, weight_bindings):
        if (not isinstance(record, dict) or
                any(type(record.get(field)) is not int for field in
                    ("src_offset", "dst_offset", "size")) or
                record["src_offset"] != expected["source_offset"] or
                record["dst_offset"] != expected["destination_offset"] or
                record["size"] != expected["byte_size"]):
            raise EvidenceError(
                "fake VkBufferCopy region differs from metadata-derived weight bytes/offsets")
    if (weighted["observed_weight_upload_count"] != weighted["expected_weight_upload_count"] or
            weighted["observed_weight_upload_count"] != expected_count or
            weighted["vkCmdCopyBuffer_call_count"] != weighted["expected_vkCmdCopyBuffer_call_count"] or
            weighted["vkCmdCopyBuffer_call_count"] != expected_count or
            weighted["live_buffer_count_after_create"] != expected_context_buffers or
            weighted["staging_buffers_destroyed_before_create_return"] < expected_count or
            weighted["peak_memory_bytes"] <= 0 or
            weighted.get("device_transfer") != "not_executed_fake_command_recording" or
            any(type(weighted.get(field)) is not bool or weighted[field] is not True
                for field in ("copy_arguments_valid", "source_non_null",
                              "destination_non_null", "source_destination_distinct",
                              "cleanup_all_objects_zero"))):
        raise EvidenceError("fake weighted copy/staging/allocation observations are invalid")
    idle = value.get("wait_idle_retry")
    if (not isinstance(idle, dict) or type(idle.get("destroy_status")) is not int or
            idle["destroy_status"] != 5 or type(idle.get("wait_count")) is not int or
            idle["wait_count"] < 2 or
            type(idle.get("destruction_before_wait_resolution")) is not bool or
            idle["destruction_before_wait_resolution"] is not False or
            type(idle.get("cleanup_all_objects_zero")) is not bool or
            idle["cleanup_all_objects_zero"] is not True):
        raise EvidenceError("fake wait-idle failure did not retain ownership until safe cleanup")
    return value


def codegen_tests_from_source(source: os.PathLike[str] | str) -> frozenset[str]:
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EvidenceError(f"lifecycle codegen unit source cannot be read: {error}") from error
    declarations = re.findall(
        r"TEST\(VulkanLifecycleCodegenTest,\s*([A-Za-z0-9_]+)\)", text)
    declared = frozenset(f"VulkanLifecycleCodegenTest.{name}" for name in declarations)
    if not declared or len(declared) != len(declarations):
        raise EvidenceError("lifecycle codegen TEST declarations are empty or duplicated")
    return declared


def aggregate_stage_evidence(stage_root: os.PathLike[str] | str,
                             profile: str, source_fingerprint: str,
                             test_names: list[str], skipped: list[str],
                             failed: list[str], *, persist: bool = True) -> tuple[dict, list[dict]]:
    root = Path(stage_root).resolve()
    if profile not in {"release", "sanitize"}:
        raise EvidenceError("VA3 lifecycle evidence only supports Release and SAN stages")
    source_root = _stage_source_directory(root, source_fingerprint)
    codegen_catalog = codegen_tests_from_source(
        source_root / "test/Unit/vulkan_lifecycle_codegen_test.cpp")
    required = (set(PRODUCT_ONLY_REQUIRED_TESTS) | set(codegen_catalog) |
                {"vulkan-lifecycle-nvidia", "vulkan-lifecycle-llvmpipe"})
    names = set(test_names)
    if not required <= names:
        raise EvidenceError(f"VA3 CTest inventory is missing {sorted(required - names)}")
    unexpected_codegen = {name for name in names if name.startswith(CODEGEN_TEST_PREFIX)} - set(codegen_catalog)
    if unexpected_codegen:
        raise EvidenceError(f"VA3 CTest inventory has uncatalogued codegen tests: {sorted(unexpected_codegen)}")
    incomplete = required & (set(skipped) | set(failed))
    if incomplete:
        raise EvidenceError(f"VA3 required lifecycle tests were skipped or failed: {sorted(incomplete)}")
    codegen_tests = sorted(names & set(codegen_catalog))
    if set(codegen_tests) != set(codegen_catalog):
        raise EvidenceError("VA3 CTest inventory differs from every codegen TEST declaration")
    evidence_root = root / "test/Vulkan/evidence/va3"
    independence_path = evidence_root / "independence/independence-evidence.json"
    independence = validate_independence_evidence(
        root, independence_path, expected_fingerprint=source_fingerprint)
    no_device_path = evidence_root / "no-device/capture.json"
    no_device = validate_native_capture(root, no_device_path, mode="no-device",
                                        expected_fingerprint=source_fingerprint)
    device_records = {}
    for device in ("nvidia", "llvmpipe"):
        capture = evidence_root / device / "capture.json"
        device_records[device] = validate_native_capture(
            root, capture, mode="device", device=device,
            expected_fingerprint=source_fingerprint)
    fake_path = evidence_root / "fake/fake-result.json"
    fake_raw = _load_json(_safe_file(root, _relative(root, fake_path), "lifecycle fake result"),
                          "lifecycle fake result")
    weighted_fixture = next(item for item in independence["fixtures"]
                            if item.get("fixture") == "weighted")
    fake = validate_fake_result(
        fake_raw, weight_bindings=weighted_fixture["weight_bindings"],
        expected_context_buffers=weighted_fixture["context_buffer_count"])
    raw_refs = []
    def add_ref(path: Path, role: str):
        item = _artifact(root, path, role)
        if not any(ref["path"] == item["path"] for ref in raw_refs):
            raw_refs.append(item)
    add_ref(independence_path, "lifecycle-only generated source/link independence evidence")
    for item in independence["raw_artifacts"]:
        checked = _check_artifact(root, item, "VA3 independence raw artifact")
        add_ref(root / checked["path"], checked["role"])
    captures = {"no-device": no_device, **device_records}
    for label, capture in captures.items():
        capture_path = evidence_root / label / "capture.json"
        add_ref(capture_path, f"lifecycle-only {label} command capture")
        for key in ("library", "harness", "native_result", "icd_manifest"):
            record = capture.get(key)
            if isinstance(record, dict):
                checked = _check_artifact(root, record, f"{label} {key}")
                add_ref(root / checked["path"], checked["role"])
        command = capture["command"]
        for stream in ("stdout", "stderr"):
            checked = _check_artifact(root, command[stream], f"{label} harness {stream}")
            add_ref(root / checked["path"], checked["role"])
        if isinstance(command.get("vulkaninfo"), dict):
            for stream in ("stdout", "stderr"):
                checked = _check_artifact(root, command["vulkaninfo"][stream],
                                          f"{label} vulkaninfo {stream}")
                add_ref(root / checked["path"], checked["role"])
    add_ref(fake_path, "lifecycle-only actual fake Vulkan fault/copy observations")
    aggregate_path = evidence_root / "lifecycle-evidence.json"
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "kind": AGGREGATE_KIND,
        "stage_id": "VA3",
        "profile": profile,
        "source_fingerprint": source_fingerprint,
        "role": "lifecycle-only",
        "candidate_run": "not_run",
        "host_aot_release_eligible": False,
        "codegen_tests": codegen_tests,
        "required_tests": sorted(required),
        "tests_executed": sorted(required & names),
        "tests_skipped": sorted(required & set(skipped)),
        "tests_failed": sorted(required & set(failed)),
        "independence": {"path": _relative(root, independence_path),
                         "sha256": sha256_file(independence_path)},
        "no_device": {"path": _relative(root, no_device_path),
                       "sha256": sha256_file(no_device_path)},
        "devices": {device: {"path": _relative(root, evidence_root / device / "capture.json"),
                             "sha256": sha256_file(evidence_root / device / "capture.json"),
                             "create_status": device_records[device]["native_observations"]["create_status"],
                             "destroy_status": device_records[device]["native_observations"]["destroy_status"],
                             "validation_errors": device_records[device]["validation_error_count"],
                             "requested_device_index": device_records[device]["requested_device_index"],
                             "physical_device_identity": device_records[device]["command"]["vulkaninfo"]["physical_device_identity"],
                             "icd_library": device_records[device]["icd_library"],
                             "upload_count": "not_measured"}
                    for device in ("nvidia", "llvmpipe")},
        "fake": {"path": _relative(root, fake_path),
                 "sha256": sha256_file(fake_path),
                 "allocation_requirements_total_bytes": fake["weighted_upload"]["peak_memory_bytes"],
                 "allocation_measurement_basis": "fake Vulkan memory-requirements sum; not device-resident peak",
                 "observations": fake},
        "device_memory_telemetry": "not_measured",
        "independence_summary": {
            "fixture_count": len(independence["fixtures"]),
            "candidate_run": "not_run",
            "plan_parser_llvm_inputs": False,
            "export_count_per_dso": len(EXPECTED_EXPORTS),
            "raw_artifact_count": len(raw_refs)},
        "raw_artifacts": raw_refs,
    }
    if persist:
        # This is a post-CTest derived summary, not a test-produced raw file.
        # The stage checker binds its hash separately from the raw inventory.
        _write_json(aggregate_path, aggregate)
    return aggregate, raw_refs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("independence", "device", "no-device"), required=True)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--icd", type=Path)
    parser.add_argument("--device-match")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--vulkaninfo", type=Path)
    parser.add_argument("--nm", default=shutil.which("nm") or "nm")
    parser.add_argument("--readelf", default=shutil.which("readelf") or "readelf")
    parser.add_argument("--ar", default=shutil.which("ar") or "ar")
    args = parser.parse_args(argv)
    if args.mode == "independence" and (not args.output or not args.source_root):
        parser.error("independence mode requires --output and --source-root")
    if args.mode != "independence" and not all(
            (args.harness, args.library, args.result, args.capture)):
        parser.error("native capture mode requires --harness, --library, --result, and --capture")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.mode == "independence":
            return _capture_independence(args)
        return _capture_native(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"lifecycle_evidence.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
