#!/usr/bin/env python3
"""Fail-closed evidence capture and validation for the VA2 Vulkan helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import model_dependency_evidence as model_evidence

SCHEMA_VERSION = 1
TEST_EVIDENCE_KIND = "ncnn.vulkan_device_helper_test"
INDEPENDENCE_KIND = "ncnn.vulkan_device_helper_independence"
CLOSURE_KIND = "ncnn.vulkan_device_helper_target_link_closure"
TEST_NAMES = (
    "vulkan-device-helper-unit",
    "vulkan-device-helper-no-device",
    "vulkan-device-helper-independence",
    "vulkan-device-helper-evidence-tests",
    "vulkan-device-helper-add-nvidia",
    "vulkan-device-helper-add-llvmpipe",
)
DEVICE_TESTS = {
    "vulkan-device-helper-add-nvidia": "nvidia",
    "vulkan-device-helper-add-llvmpipe": "llvmpipe",
}
RUNTIME_BUDGET_TESTS = frozenset(
    f"VulkanRuntimeBudgetTest.{name}" for name in (
        "DefaultBudgetAndStorageContractAreShared",
        "ChecksActualIoStagingBudgetBeforeAnyAllocation",
        "CountsTemporaryWeightStagingBeforeAllocation",
        "ChecksActualAllocationCountBeforeAllocation",
        "PartialBindFailureReleasesEveryOwnedObject"))
PRODUCT_ONLY_REGISTERED_TESTS = (*TEST_NAMES[:4], *sorted(RUNTIME_BUDGET_TESTS))
PRODUCT_ONLY_EXECUTE_TESTS = (*TEST_NAMES[:2], TEST_NAMES[3],
                              *sorted(RUNTIME_BUDGET_TESTS))
HELPER_TARGET = "NCNNVulkanDeviceHelper"
RUNNER_TARGET = "vulkan_device_helper_runner"
UNIT_TARGET = "vulkan_device_helper_unit_test"
EXPECTED_ARCHIVE = "libNCNNVulkanDeviceHelper.a"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SANITIZER_RE = re.compile(
    r"ERROR: (?:AddressSanitizer|LeakSanitizer)|"
    r"SUMMARY: (?:AddressSanitizer|UndefinedBehaviorSanitizer)|"
    r"UndefinedBehaviorSanitizer:|runtime error:|Validation Error|VUID-"
)
FORBIDDEN_DEPENDENCY_RE = re.compile(
    r"\b(?:llvm|mlir|gtest|googletest)[a-z0-9_.+-]*|"
    r"lib(?:llvm|mlir)[a-z0-9_.+-]*|"
    r"(?:VulkanExecutionPlan|VulkanHostProgram|VulkanPlanSupport|"
    r"ncnn[-_]?plan[-_]?(?:parser|support)|execution[-_]plan)", re.IGNORECASE)
CLANG_RESOURCE_INCLUDE_RE = re.compile(r"/lib/clang/[0-9][^/]*/include(?:/|$)")
CLANG_SHARED_ASAN_RUNTIME_RE = re.compile(r"/lib/clang/[0-9][^/]*/lib/linux(?:/|$)")


class EvidenceError(RuntimeError):
    """Raised when VA2 evidence is missing, stale, or inconsistent."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, value: object, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError("evidence path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError(f"evidence path escapes stage: {value}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise EvidenceError(f"evidence path escapes stage: {value}")
    if must_exist and not path.is_file():
        raise EvidenceError(f"evidence file is missing: {value}")
    return path


def _copy_raw(source: Path, stage: Path, output_root: Path,
              relative: Path) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file():
        raise EvidenceError(f"raw input is missing: {source}")
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        # This is a newly collected evidence snapshot. Preserve bytes, not the
        # build input's old mtime: freshness is checked against this CTest run.
        shutil.copyfile(source, destination)
    return {"path": destination.relative_to(stage).as_posix(),
            "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {label}: {error}") from error


def _has_schema(value: object, expected: int = SCHEMA_VERSION) -> bool:
    return (isinstance(value, dict) and type(value.get("schema_version")) is int and
            value["schema_version"] == expected)


def _record_file(root: Path, record: object, label: str,
                 refs: list[dict[str, object]]) -> Path:
    if not isinstance(record, dict):
        raise EvidenceError(f"{label} record is malformed")
    relative = record.get("path")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise EvidenceError(f"{label} SHA-256 is malformed")
    path = _safe_path(root, relative)
    if sha256_file(path) != digest:
        raise EvidenceError(f"{label} hash differs from raw bytes: {relative}")
    size = record.get("size_bytes")
    if size is not None and (type(size) is not int or size != path.stat().st_size):
        raise EvidenceError(f"{label} byte size differs from raw bytes: {relative}")
    refs.append({"path": str(path.relative_to(root)), "sha256": digest,
                 "role": label})
    return path


def _is_forbidden_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if CLANG_RESOURCE_INCLUDE_RE.search(normalized):
        return False
    # Clang adds its packaged default ASan instrumentation policy to depfiles.
    # It is a compiler resource, not a project-selected suppression or LLVM API.
    if re.search(r"/lib/clang/[0-9.]+/share/asan_ignorelist\.txt$", normalized):
        return False
    runtime_match = CLANG_SHARED_ASAN_RUNTIME_RE.search(normalized)
    if runtime_match:
        suffix = normalized[runtime_match.end():]
        if (not suffix or suffix.startswith(("libclang_rt.asan", "libclang_rt.ubsan",
                                              "libclang_rt.builtins"))):
            return False
    # The project's include namespace is named ncnn-mlir; it is not an MLIR
    # dependency and must not be confused with upstream MLIR headers/targets.
    normalized = re.sub(r"ncnn[-_]mlir", "", normalized, flags=re.IGNORECASE)
    return bool(FORBIDDEN_DEPENDENCY_RE.search(normalized))


def _check_clean_dependency(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} contains a malformed path/name")
    if _is_forbidden_path(value):
        raise EvidenceError(f"forbidden LLVM/MLIR/plan/gtest dependency in {label}: {value}")


def _read_stage_identity(stage: Path) -> tuple[dict | None, str, str | None, bool]:
    state_path = stage / "gate-state.json"
    if not state_path.is_file():
        cache = _cmake_cache_values(stage / "CMakeCache.txt")
        sanitizers = set(cache.get("COMPILER_SANITIZERS", "").split(";")) - {""}
        if sanitizers and sanitizers != {"address", "undefined"}:
            raise EvidenceError("development CMake tree has an unsupported sanitizer set")
        if sanitizers == {"address", "undefined"}:
            profile = "sanitize"
        elif cache.get("CMAKE_BUILD_TYPE") == "Release":
            profile = "release"
        else:
            profile = "development"
        return None, profile, None, False
    state = _load_json(state_path, "gate-state")
    if not _has_schema(state) or state.get("kind") != "ncnn.vulkan_gate":
        raise EvidenceError("gate-state schema/kind is invalid")
    stage_id = state.get("stage_id") if "stage_id" in state else "VA0"
    if not isinstance(stage_id, str) or stage_id not in {"VA0", "VA1", "VA2"}:
        raise EvidenceError("gate-state has an invalid stage_id")
    profile = state.get("profile")
    if profile not in {"release", "sanitize", "offline", "software", "hardware"}:
        raise EvidenceError("gate-state has an invalid profile")
    if stage_id == "VA2" and profile not in {"release", "sanitize"}:
        raise EvidenceError("VA2 helper qualification requires Release or sanitize profile")
    source_fingerprint = state.get("source_fingerprint")
    if not isinstance(source_fingerprint, str) or not SHA256_RE.fullmatch(source_fingerprint):
        raise EvidenceError("gate-state source fingerprint is invalid")
    if state.get("host_execution_role") != "legacy-product" or state.get("host_aot_candidate") is not False:
        raise EvidenceError("helper evidence must preserve legacy-product / non-candidate host role")
    return state, profile, source_fingerprint, True


def _test_record_path(stage: Path, identity: str) -> Path:
    return stage / "test/Vulkan/evidence/va2/tests" / f"{identity}.json"


def _read_f32(path: Path, count: int = 257) -> list[float]:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise EvidenceError(f"expected {count} FP32 values in {path}, found {len(data)} bytes")
    values = list(struct.unpack(f"<{count}f", data))
    if any(not math.isfinite(value) for value in values):
        raise EvidenceError(f"non-finite FP32 value in {path}")
    return values


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _compare_f32(actual: list[float], expected: list[float], label: str,
                  atol: float = 1.0e-6, rtol: float = 1.0e-6) -> None:
    if len(actual) != len(expected):
        raise EvidenceError(f"{label} element count differs")
    for index, (value, reference) in enumerate(zip(actual, expected)):
        if abs(value - reference) > atol + rtol * abs(reference):
            raise EvidenceError(f"{label} mismatch at element {index}: {value} vs {reference}")


def _test_raw_entries(stage: Path, identity: str,
                      refs: list[dict[str, object]]) -> tuple[dict, dict[str, Path]]:
    path = _test_record_path(stage, identity)
    record = _load_json(path, f"{identity} evidence")
    if (not _has_schema(record) or
            record.get("kind") != TEST_EVIDENCE_KIND or record.get("identity") != identity or
            record.get("status") != "passed"):
        raise EvidenceError(f"{identity} evidence schema/identity/status is invalid")
    if "returncode" in record and type(record["returncode"]) is not int:
        raise EvidenceError(f"{identity} returncode must be an integer, not bool/number alias")
    raw = record.get("raw")
    if not isinstance(raw, list) or not raw:
        raise EvidenceError(f"{identity} has no raw evidence inventory")
    by_role: dict[str, Path] = {}
    seen_paths: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise EvidenceError(f"{identity} raw entry is malformed")
        relative, digest, role = item.get("path"), item.get("sha256"), item.get("role")
        if (not isinstance(role, str) or not role or
                not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
            raise EvidenceError(f"{identity} raw entry role/hash is malformed")
        if relative in seen_paths:
            raise EvidenceError(f"{identity} repeats raw path {relative}")
        seen_paths.add(relative)
        raw_path = _safe_path(stage, relative)
        if sha256_file(raw_path) != digest:
            raise EvidenceError(f"{identity} raw hash mismatch: {relative}")
        size = item.get("size_bytes")
        if type(size) is not int or size != raw_path.stat().st_size:
            raise EvidenceError(f"{identity} raw byte count is missing/invalid: {relative}")
        origin_value = item.get("origin")
        if not isinstance(origin_value, str) or not origin_value:
            raise EvidenceError(f"{identity} raw origin is missing: {role}")
        origin = Path(origin_value)
        if not origin.is_absolute():
            raise EvidenceError(f"{identity} raw origin is not absolute: {role}")
        origin = origin.resolve()
        external_roles = {"source-kernel", "kernel-source", "mlir-opt-binary",
                          "spirv-extractor-binary", "spirv-val-binary", "spirv-dis-binary",
                          "icd-manifest"}
        if (not origin.is_file() or
                (not origin.is_relative_to(stage.resolve()) and role not in external_roles) or
                origin.stat().st_size != size or sha256_file(origin) != digest):
            raise EvidenceError(f"{identity} raw origin bytes do not match captured record: {role}")
        if role in {"mlir-opt-binary", "spirv-val-binary", "spirv-dis-binary"}:
            tool_name = {"mlir-opt-binary": "mlir-opt",
                         "spirv-val-binary": "spirv-val",
                         "spirv-dis-binary": "spirv-dis"}[role]
            if origin != _expected_gate_tool(stage, tool_name):
                raise EvidenceError(f"{identity} {role} differs from the actual configured tool")
        elif role in {"source-kernel", "kernel-source"}:
            source_root = _cmake_source_root(stage)
            if origin != (source_root / "test/Vulkan/gpu_func_add_probe.mlir").resolve():
                raise EvidenceError(f"{identity} kernel source is not the checked-in Add fixture")
        elif role == "spirv-extractor-binary":
            _ref, target_json, _target_path = _file_api_target(stage, "vulkan_v0_spirv_extract")
            artifact_rows = target_json.get("artifacts")
            if not isinstance(artifact_rows, list) or len(artifact_rows) != 1:
                raise EvidenceError("SPIR-V extractor target artifact is missing/ambiguous")
            target_path = Path(artifact_rows[0].get("path", ""))
            if not target_path.is_absolute():
                target_path = stage / target_path
            if origin != target_path.resolve():
                raise EvidenceError(f"{identity} SPIR-V extractor is not the actual configured target")
        elif role == "icd-manifest":
            if identity not in DEVICE_TESTS:
                if not origin.is_relative_to(stage.resolve()):
                    raise EvidenceError("no-device manifest must be a stage-local unavailable ICD fixture")
            else:
                cache = _cmake_cache_values(stage / "CMakeCache.txt")
                if identity.endswith("nvidia"):
                    expected_manifest = (stage / "test/Vulkan/nvidia-egl-icd.json").resolve()
                else:
                    value = cache.get("V0_LLVMPIPE_ICD_PATH")
                    if not value:
                        raise EvidenceError("CMake cache does not identify the selected llvmpipe ICD")
                    expected_manifest = Path(value).resolve()
                if origin != expected_manifest:
                    raise EvidenceError(f"{identity} ICD manifest differs from configured device selection")
        if role in by_role:
            raise EvidenceError(f"{identity} repeats raw role {role}")
        by_role[role] = raw_path
        refs.append({"path": str(raw_path.relative_to(stage)), "sha256": digest,
                     "size_bytes": raw_path.stat().st_size,
                     "role": f"VA2 {identity} {role}"})
    _gate_state, expected_profile, expected_fingerprint, gate_bound = _read_stage_identity(stage)
    if gate_bound:
        if record.get("profile") not in (None, expected_profile):
            raise EvidenceError(f"{identity} self-reported profile differs from gate-state")
        if record.get("source_fingerprint") not in (None, expected_fingerprint):
            raise EvidenceError(f"{identity} self-reported source fingerprint differs from gate-state")
    elif record.get("profile") is not None or record.get("source_fingerprint") is not None:
        raise EvidenceError(f"{identity} claims gate identity without a gate-state")
    refs.append({"path": str(path.relative_to(stage)), "sha256": sha256_file(path),
                 "size_bytes": path.stat().st_size,
                 "role": f"VA2 {identity} raw test record"})
    return record, by_role


def _raw_metadata(record: dict, role: str) -> dict:
    matches = [item for item in record.get("raw", [])
               if isinstance(item, dict) and item.get("role") == role]
    if len(matches) != 1:
        raise EvidenceError(f"raw evidence has no unique {role} metadata")
    return matches[0]


def _origin(record: dict, role: str, copied: Path, stage: Path,
            *, allow_external: bool = False) -> Path:
    metadata = _raw_metadata(record, role)
    value = metadata.get("origin")
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{role} raw evidence omits its original path")
    origin = Path(value)
    if not origin.is_absolute():
        raise EvidenceError(f"{role} raw evidence origin is not absolute")
    origin = origin.resolve()
    if ((not allow_external and not origin.is_relative_to(stage.resolve())) or
            not origin.is_file()):
        raise EvidenceError(f"{role} raw evidence origin is outside/missing from stage")
    if (origin.stat().st_size != copied.stat().st_size or
            sha256_file(origin) != sha256_file(copied)):
        raise EvidenceError(f"{role} raw snapshot differs from its original bytes")
    return origin


def _command_log(path: Path, label: str) -> dict:
    value = _load_json(path, label)
    if (not isinstance(value, dict) or
            not isinstance(value.get("argv"), list) or not value["argv"] or
            any(not isinstance(item, str) or not item for item in value["argv"]) or
            type(value.get("returncode")) is not int or
            not isinstance(value.get("stdout"), str) or
            not isinstance(value.get("stderr"), str)):
        raise EvidenceError(f"{label} does not contain an exact command/exit/output record")
    return value


def _validate_spirv_validation(raw: dict[str, Path], label: str,
                               test_record: dict, stage: Path) -> None:
    invocation_path = raw.get("spirv-validation-log")
    shader = raw.get("spirv")
    if invocation_path is None or shader is None:
        raise EvidenceError(f"{label} lacks SPIR-V and raw spirv-val invocation")
    invocation = _command_log(invocation_path, f"{label} spirv-val invocation")
    if invocation.get("kind") != "ncnn.spirv_tool_invocation":
        raise EvidenceError(f"{label} spirv-val invocation kind is invalid")
    binary = _origin(test_record, "spirv-val-binary", raw["spirv-val-binary"], stage,
                     allow_external=True)
    if binary != _expected_gate_tool(stage, "spirv-val"):
        raise EvidenceError(f"{label} spirv-val executable differs from the configured tool")
    shader_origin = _origin(test_record, "spirv", shader, stage)
    argv = invocation["argv"]
    if (Path(argv[0]).resolve() != binary or invocation["returncode"] != 0 or
            argv[1:-1] != ["--target-env", "vulkan1.1"] or
            Path(argv[-1]).resolve() != shader_origin or
            invocation["stderr"] not in ("", None)):
        raise EvidenceError(f"{label} spirv-val argv does not validate the captured Vulkan 1.1 SPIR-V")
    version = _validate_tool_log(
        stage, raw, test_record, "spirv-val-version-log", "spirv-val-binary", "spirv-val",
        ["--version"])
    if version["returncode"] != 0:
        raise EvidenceError(f"{label} spirv-val --version failed")
    log = raw.get("validation-log")
    if log is None:
        raise EvidenceError(f"{label} has no Vulkan validation log")
    if SANITIZER_RE.search(log.read_text(encoding="utf-8", errors="replace")):
        raise EvidenceError(f"{label} validation log contains diagnostics")


def _icd_library_path(manifest: Path, library: str) -> Path:
    named = Path(library)
    if named.is_absolute():
        candidates = [named]
    elif named.name != library:
        candidates = [manifest.parent / named]
    else:
        # Bare manifest library names use the system loader search, not the
        # directory containing the JSON manifest (Mesa commonly uses sonames).
        candidates = [Path(directory) / named for directory in
                      os.environ.get("LD_LIBRARY_PATH", "").split(":") if directory]
        ldconfig = shutil.which("ldconfig")
        if ldconfig is not None:
            result = _run([ldconfig, "-p"], check=True)
            pattern = re.compile(r"^\s*" + re.escape(library) +
                                 r"\s+\([^)]*x86-64[^)]*\)\s+=>\s+(\S+)\s*$")
            candidates.extend(Path(match.group(1)) for line in
                              result.stdout.decode("utf-8", errors="strict").splitlines()
                              if (match := pattern.fullmatch(line)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise EvidenceError(f"selected Vulkan ICD library is unavailable: {library}")


def _validate_device_test(stage: Path, identity: str, raw: dict[str, Path],
                          record: dict | None = None) -> None:
    required = {"runner-binary", "icd-manifest", "source-kernel", "cpu-aot-library",
                "mlir-opt-binary", "spirv-extractor-binary", "spirv-val-binary",
                "spirv-dis-binary", "lhs-f32", "rhs-f32", "cpu-aot-output-f32",
                "helper-gpu-output-f32", "helper-gpu-repeat-f32",
                "scalar-golden-f32", "gpu-binary-mlir", "spirv-disassembly",
                "mlir-opt-version-log", "spirv-val-version-log",
                "gpu-func-add-lowering-log", "spirv-extraction-log",
                "spirv-disassembly-log", "spirv", "spirv-validation-log",
                "validation-log", "result"}
    missing = sorted(required - set(raw))
    if missing:
        raise EvidenceError(f"{identity} lacks required raw roles: {missing}")
    if not isinstance(record, dict):
        raise EvidenceError(f"{identity} has no enclosing raw record")
    runner = _origin(record, "runner-binary", raw["runner-binary"], stage)
    manifest_path = raw["icd-manifest"]
    manifest_origin = _origin(record, "icd-manifest", manifest_path, stage,
                               allow_external=True)
    result_origin = _origin(record, "result", raw["result"], stage)
    shader_origin = _origin(record, "spirv", raw["spirv"], stage)
    cpu_library = _origin(record, "cpu-aot-library", raw["cpu-aot-library"], stage)
    lhs = _read_f32(raw["lhs-f32"])
    rhs = _read_f32(raw["rhs-f32"])
    golden = [_f32(a + b) for a, b in zip(lhs, rhs)]
    cpu = _read_f32(raw["cpu-aot-output-f32"])
    gpu = _read_f32(raw["helper-gpu-output-f32"])
    repeat = _read_f32(raw["helper-gpu-repeat-f32"])
    scalar = _read_f32(raw["scalar-golden-f32"])
    _compare_f32(scalar, golden, f"{identity} scalar golden")
    _compare_f32(cpu, scalar, f"{identity} CPU AOT vs scalar golden")
    _compare_f32(gpu, scalar, f"{identity} helper GPU vs scalar golden")
    _compare_f32(gpu, cpu, f"{identity} helper GPU vs CPU AOT")
    _compare_f32(repeat, gpu, f"{identity} repeated helper GPU output")
    result = _load_json(raw["result"], f"{identity} helper execution result")
    if (not _has_schema(result) or
            result.get("kind") != "ncnn.vulkan_device_helper_add_result" or
            result.get("fixture_id") != "va2-f32-add-257-gpu-func-v1" or
            result.get("status") != "passed"):
        raise EvidenceError(f"{identity} helper result schema/fixture/status is invalid")
    if type(result.get("elements")) is not int or result["elements"] != 257:
        raise EvidenceError(f"{identity} helper result has the wrong element count")
    if type(result.get("local_size")) is not int or result["local_size"] != 64:
        raise EvidenceError(f"{identity} helper result has the wrong local size")
    if type(result.get("dispatch_groups_x")) is not int or result["dispatch_groups_x"] != 5:
        raise EvidenceError(f"{identity} helper result has the wrong tail dispatch")
    if (result.get("validation_enabled") is not True or
            result.get("wait_failure_recovered") is not True):
        raise EvidenceError(f"{identity} helper result lacks validation or wait-failure recovery")
    if type(result.get("validation_errors")) is not int or result["validation_errors"] != 0:
        raise EvidenceError(f"{identity} Vulkan validation reported errors")
    if result.get("device") not in {"nvidia", "llvmpipe"}:
        raise EvidenceError(f"{identity} helper result has an invalid device identity")
    if result.get("device") != DEVICE_TESTS[identity]:
        raise EvidenceError(f"{identity} helper result device does not match test identity")
    device_name = result.get("device_name")
    api_version = result.get("api_version")
    if not isinstance(device_name, str) or not device_name:
        raise EvidenceError(f"{identity} raw result omits the physical device name")
    if type(api_version) is not int or api_version < ((1 << 22) | (1 << 12)):
        raise EvidenceError(f"{identity} Vulkan device API version is below 1.1")
    if identity.endswith("nvidia"):
        if result["vendor_id"] != 0x10DE or not re.search(
                r"NVIDIA|GeForce|RTX|Quadro|Tesla", device_name, re.IGNORECASE):
            raise EvidenceError("NVIDIA helper test did not select an NVIDIA physical device")
    elif (result["vendor_id"] != 0x10005 or
          not re.search(r"llvmpipe|lavapipe", device_name, re.IGNORECASE)):
        raise EvidenceError("llvmpipe helper test did not select the software physical device")
    runner_log = _command_log(raw["validation-log"], f"{identity} actual runner command")
    runner_argv = runner_log["argv"]
    if (runner_log["returncode"] != 0 or Path(runner_argv[0]).resolve() != runner or
            "--mode" not in runner_argv or
            runner_argv[runner_argv.index("--mode") + 1:runner_argv.index("--mode") + 2] != ["add"]):
        raise EvidenceError(f"{identity} actual helper runner command did not execute the captured binary")
    match_option = "--device-match"
    match_index = runner_argv.index(match_option) + 1 if match_option in runner_argv else -1
    expected_match = "NVIDIA" if identity.endswith("nvidia") else "llvmpipe"
    if match_index < 0 or match_index >= len(runner_argv) or runner_argv[match_index].lower() != expected_match.lower():
        raise EvidenceError(f"{identity} runner command did not explicitly select its expected ICD device")
    overrides = runner_log.get("environment_overrides")
    expected_icd = str(manifest_origin)
    if (not isinstance(overrides, dict) or
            overrides != {"VK_ICD_FILENAMES": expected_icd,
                          "VK_DRIVER_FILES": expected_icd}):
        raise EvidenceError(f"{identity} command did not bind both Vulkan loader variables to captured ICD")
    manifest_data = _load_json(manifest_path, f"{identity} ICD manifest")
    icd = manifest_data.get("ICD") if isinstance(manifest_data, dict) else None
    library = icd.get("library_path") if isinstance(icd, dict) else None
    if not isinstance(library, str) or not library:
        raise EvidenceError(f"{identity} manifest does not identify selected ICD library")
    library_path = _icd_library_path(manifest_origin, library)
    cache = _cmake_cache_values(stage / "CMakeCache.txt")
    if identity.endswith("nvidia"):
        expected_library = cache.get("V0_NVIDIA_EGL_LIBRARY")
        if not expected_library or library_path.resolve() != Path(expected_library).resolve():
            raise EvidenceError("NVIDIA ICD manifest differs from configured EGL ICD library")
    elif not re.search(r"lvp|lavapipe", library_path.name, re.IGNORECASE):
        raise EvidenceError("llvmpipe ICD manifest points at the wrong driver library")
    def option_path(option: str) -> Path:
        if option not in runner_argv or runner_argv.index(option) + 1 >= len(runner_argv):
            raise EvidenceError(f"{identity} runner command omits {option}")
        return Path(runner_argv[runner_argv.index(option) + 1]).resolve()
    if (option_path("--result") != result_origin or
            option_path("--shader") != shader_origin or
            option_path("--cpu-library") != cpu_library):
        raise EvidenceError(f"{identity} runner argv is not bound to captured result/shader/CPU library")
    for option, role in (("--lhs", "lhs-f32"), ("--rhs", "rhs-f32"),
                         ("--cpu-output", "cpu-aot-output-f32"),
                         ("--gpu-output", "helper-gpu-output-f32"),
                         ("--gpu-repeat-output", "helper-gpu-repeat-f32"),
                         ("--golden", "scalar-golden-f32")):
        if option_path(option) != _origin(record, role, raw[role], stage):
            raise EvidenceError(f"{identity} runner argv {option} differs from captured tensor bytes")
    if "--validation" not in runner_argv:
        raise EvidenceError(f"{identity} actual helper command did not enable validation")
    if SANITIZER_RE.search(runner_log["stdout"] + runner_log["stderr"]):
        raise EvidenceError(f"{identity} helper runner emitted validation/sanitizer diagnostics")
    lowered_origin = _origin(record, "gpu-binary-mlir", raw["gpu-binary-mlir"], stage)
    disassembly_origin = _origin(record, "spirv-disassembly", raw["spirv-disassembly"], stage)
    kernel_origin = _origin(record, "source-kernel", raw["source-kernel"], stage,
                            allow_external=True)
    lower_log = _validate_tool_log(
        stage, raw, record, "gpu-func-add-lowering-log", "mlir-opt-binary", "mlir-opt",
        [str(kernel_origin), "--test-vulkan-runner-pipeline", "--verify-each",
         "-o", str(lowered_origin)])
    del lower_log
    extraction = _command_log(raw["spirv-extraction-log"], "SPIR-V extraction command")
    extractor = _origin(record, "spirv-extractor-binary", raw["spirv-extractor-binary"],
                        stage, allow_external=True)
    if (Path(extraction["argv"][0]).resolve() != extractor or
            extraction["argv"][1:] != [str(lowered_origin), "-o", str(shader_origin)] or
            extraction["returncode"] != 0):
        raise EvidenceError(f"{identity} SPIR-V extraction did not consume/produce captured files")
    disassembly = _command_log(raw["spirv-disassembly-log"], "SPIR-V disassembly command")
    dis_tool = _origin(record, "spirv-dis-binary", raw["spirv-dis-binary"], stage,
                       allow_external=True)
    if (dis_tool != _expected_gate_tool(stage, "spirv-dis") or
            Path(disassembly["argv"][0]).resolve() != dis_tool or
            disassembly["argv"][1:] != [str(shader_origin), "-o", str(disassembly_origin)] or
            disassembly["returncode"] != 0):
        raise EvidenceError(f"{identity} SPIR-V disassembly command differs from captured inputs")
    for log_role, binary_role, tool_role in (
            ("mlir-opt-version-log", "mlir-opt-binary", "mlir-opt"),
            ("spirv-val-version-log", "spirv-val-binary", "spirv-val")):
        _validate_tool_log(stage, raw, record, log_role, binary_role, tool_role,
                           ["--version"])
    metrics = {
        "max_cpu_golden_abs": max(abs(a - b) for a, b in zip(cpu, scalar)),
        "max_gpu_golden_abs": max(abs(a - b) for a, b in zip(gpu, scalar)),
        "max_repeat_abs": max(abs(a - b) for a, b in zip(repeat, gpu)),
    }
    for key, computed in metrics.items():
        claim = result.get(key)
        if type(claim) not in (int, float) or not math.isfinite(claim) or claim != computed:
            raise EvidenceError(f"{identity} reported {key} differs from raw output bytes")
    api_version = result.get("api_version")
    if not isinstance(api_version, (str, int)) or isinstance(api_version, bool):
        raise EvidenceError(f"{identity} helper result has an invalid Vulkan API version")
    for key in ("driver_version", "vendor_id", "device_id"):
        if type(result.get(key)) is not int:
            raise EvidenceError(f"{identity} helper result {key} must be an exact integer")
    _validate_spirv_validation(raw, identity, record, stage)


def _validate_no_device_test(stage: Path, raw: dict[str, Path],
                             record: dict) -> None:
    required = {"result", "runner-binary", "icd-manifest", "stdout", "stderr",
                "create-command-log"}
    missing = sorted(required - set(raw))
    if missing:
        raise EvidenceError(f"no-device test lacks raw roles: {missing}")
    runner = _origin(record, "runner-binary", raw["runner-binary"], stage)
    manifest = raw["icd-manifest"]
    manifest_origin = _origin(record, "icd-manifest", manifest, stage)
    manifest_data = _load_json(manifest, "unavailable ICD manifest")
    if not isinstance(manifest_data, dict):
        raise EvidenceError("unavailable ICD manifest is malformed")
    result = _load_json(raw["result"], "no-device helper result")
    if (not _has_schema(result) or
            result.get("kind") != "ncnn.vulkan_device_helper_no_device_result" or
            result.get("helper_create_succeeded") is not False or
            type(result.get("status")) is not int or result["status"] == 0 or
            not isinstance(result.get("diagnostic"), str) or not result["diagnostic"].strip()):
        raise EvidenceError("no-device helper raw result does not prove a clean unavailable-device failure")
    result_origin = _origin(record, "result", raw["result"], stage)
    for key in ("skip", "skipped"):
        if key in result and type(result[key]) is not bool:
            raise EvidenceError(f"no-device result {key} is not a boolean")
        if result.get(key) is True:
            raise EvidenceError("no-device helper evidence reports a skip")
    command = _command_log(raw["create-command-log"], "no-device create command")
    argv = command["argv"]
    mode_index = argv.index("--mode") + 1 if "--mode" in argv else -1
    result_index = argv.index("--result") + 1 if "--result" in argv else -1
    if (Path(argv[0]).resolve() != runner or
            mode_index < 0 or mode_index >= len(argv) or argv[mode_index] != "no-device" or
            result_index < 0 or result_index >= len(argv) or
            Path(argv[result_index]).resolve() != result_origin or
            command["returncode"] != 0):
        raise EvidenceError("no-device command log does not invoke the captured runner/result successfully")
    overrides = command.get("environment_overrides")
    expected_icd = str(manifest_origin)
    if (not isinstance(overrides, dict) or
            overrides != {"VK_ICD_FILENAMES": expected_icd,
                          "VK_DRIVER_FILES": expected_icd}):
        raise EvidenceError("no-device command does not bind both Vulkan loader variables to the captured ICD")
    for key, role in (("stdout", "stdout"), ("stderr", "stderr")):
        sidecar = raw[role].read_text(encoding="utf-8", errors="replace")
        if sidecar != command[key]:
            raise EvidenceError(f"no-device {key} sidecar differs from actual command output")
        if SANITIZER_RE.search(sidecar):
            raise EvidenceError(f"no-device {key} contains sanitizer/validation diagnostics")
    if result["diagnostic"].strip() not in command["stdout"] + command["stderr"]:
        raise EvidenceError("no-device diagnostic is not present in actual command output")
    manifest_value = manifest_data.get("ICD")
    library = manifest_value.get("library_path") if isinstance(manifest_value, dict) else None
    if not isinstance(library, str) or not library:
        raise EvidenceError("no-device ICD manifest does not identify an unavailable ICD library")
    unavailable_library = Path(library)
    if not unavailable_library.is_absolute():
        unavailable_library = manifest_origin.parent / unavailable_library
    if unavailable_library.exists():
        raise EvidenceError("no-device ICD manifest unexpectedly resolves to an available ICD library")


def _validate_unit_test(stage: Path, raw: dict[str, Path], record: dict,
                        identity: str) -> None:
    required = {"unit-binary", "unit-command-log", "result"}
    missing = sorted(required - set(raw))
    if missing:
        raise EvidenceError(f"{identity} lacks actual unit command evidence: {missing}")
    binary = _origin(record, "unit-binary", raw["unit-binary"], stage)
    result_origin = _origin(record, "result", raw["result"], stage)
    command = _command_log(raw["unit-command-log"], f"{identity} command")
    if (command["returncode"] != 0 or command["argv"] != [str(binary)]):
        raise EvidenceError(f"{identity} command did not execute the captured unit binary")
    result = _load_json(raw["result"], f"{identity} result")
    if (not _has_schema(result) or result.get("kind") !=
            "ncnn.vulkan_device_helper_unit_result" or
            result.get("status") != "passed" or type(result.get("returncode")) is not int or
            result["returncode"] != 0 or result.get("stdout") != command["stdout"] or
            result.get("stderr") != command["stderr"]):
        raise EvidenceError(f"{identity} result is not bound to its actual command output")
    if SANITIZER_RE.search(command["stdout"] + command["stderr"]):
        raise EvidenceError(f"{identity} output contains sanitizer/validation diagnostics")
    if not result_origin.is_file():
        raise EvidenceError(f"{identity} actual result is missing")


def _validate_independence_test(stage: Path, raw: dict[str, Path],
                                record: dict) -> None:
    required = {"result", "compiler-toolchain-identity", "standalone-unit-result",
                "standalone-unit-binary", "standalone-helper-link-command", "standalone-harness-link-command",
                "standalone-unit-link-command", "standalone-harness-link-map",
                "standalone-unit-link-map", "standalone-target-link-closure",
                "standalone-compile-commands", "production-cmake-cache",
                "standalone-cmake-cache", "product-only-cmake-cache",
                "product-only-ctest-inventory", "product-only-compile-commands",
                "product-only-target-link-closure", "product-only-helper-archive"}
    missing = sorted(required - set(raw))
    if missing:
        raise EvidenceError(f"independence CTest lacks actual audit inputs: {missing}")
    result = _load_json(raw["result"], "independence helper result")
    if (not _has_schema(result) or
            result.get("kind") != "ncnn.vulkan_device_helper_independence_result" or
            result.get("status") != "passed" or type(result.get("returncode")) is not int or
            result["returncode"] != 0 or
            result.get("product_only_check_performed") is not True or
            type(result.get("standalone_build_returncode")) is not int or
            result["standalone_build_returncode"] != 0 or
            type(result.get("standalone_unit_returncode")) is not int or
            result["standalone_unit_returncode"] != 0):
        raise EvidenceError("independence result does not prove its builds/unit passed")
    unit_command = _command_log(raw["standalone-unit-result"],
                                "standalone unit actual command")
    unit_argv = unit_command["argv"]
    standalone_root_value = result.get("standalone_build_root")
    if not isinstance(standalone_root_value, str) or not standalone_root_value:
        raise EvidenceError("independence standalone build root is missing")
    standalone_root = Path(standalone_root_value)
    if not standalone_root.is_absolute():
        standalone_root = stage / standalone_root
    standalone_root = standalone_root.resolve()
    if not standalone_root.is_relative_to(stage.resolve()):
        raise EvidenceError("independence standalone build root is outside the captured stage")
    standalone_unit_binary = _origin(
        record, "standalone-unit-binary", raw["standalone-unit-binary"], stage)
    if (standalone_unit_binary != (standalone_root / UNIT_TARGET).resolve() or
            unit_command["returncode"] != 0 or unit_argv != [str(standalone_unit_binary)]):
        raise EvidenceError("standalone unit log does not invoke the captured target executable")
    for role, target_name in (("standalone-helper-link-command", HELPER_TARGET),
                              ("standalone-harness-link-command", RUNNER_TARGET),
                              ("standalone-unit-link-command", UNIT_TARGET)):
        try:
            link_argv = shlex.split(raw[role].read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise EvidenceError(f"{role} is not a readable actual link command") from error
        if not link_argv or any(token.startswith("@") for token in link_argv):
            raise EvidenceError(f"{role} is empty or uses an unaudited response file")
        if target_name == HELPER_TARGET:
            if not any(token.endswith(EXPECTED_ARCHIVE) for token in link_argv):
                raise EvidenceError("standalone helper archive command omits its output archive")
        else:
            helper_archive = standalone_root / EXPECTED_ARCHIVE
            if not any(Path(token).name == EXPECTED_ARCHIVE
                       for token in link_argv if token.endswith((".a", ".so"))):
                raise EvidenceError(f"{role} does not link the standalone helper archive")
    toolchain = _load_json(raw["compiler-toolchain-identity"],
                           "independence compiler toolchain identity")
    if not isinstance(toolchain, dict) or toolchain != result.get("toolchain_identity"):
        raise EvidenceError("independence toolchain identity differs from actual result")
    child_registered = result.get("product_only_registered_tests")
    child_executed = result.get("product_only_executed_tests")
    if (not isinstance(child_registered, list) or
            sorted(child_registered) != sorted(PRODUCT_ONLY_REGISTERED_TESTS) or
            not isinstance(child_executed, list) or
            sorted(child_executed) != sorted(PRODUCT_ONLY_EXECUTE_TESTS)):
        raise EvidenceError("independence product-only inventory/result is incomplete")
    if SANITIZER_RE.search(unit_command["stdout"] + unit_command["stderr"]):
        raise EvidenceError("standalone unit output contains sanitizer/validation diagnostics")


def _validate_generic_test_command(stage: Path, raw: dict[str, Path],
                                  record: dict, identity: str,
                                  command_role: str, result_kind: str) -> None:
    required = {"test-command-log", "result"}
    missing = sorted(required - set(raw))
    if missing:
        raise EvidenceError(f"{identity} lacks raw command evidence: {missing}")
    result_origin = _origin(record, "result", raw["result"], stage)
    command = _command_log(raw[command_role], f"{identity} command")
    if command["returncode"] != 0:
        raise EvidenceError(f"{identity} command returned nonzero")
    result = _load_json(raw["result"], f"{identity} result")
    if (not _has_schema(result) or result.get("kind") != result_kind or
            result.get("status") != "passed" or type(result.get("returncode")) is not int or
            result["returncode"] != 0 or result.get("stdout") != command["stdout"] or
            result.get("stderr") != command["stderr"]):
        raise EvidenceError(f"{identity} result is not bound to its actual command output")
    if not result_origin.is_file() or SANITIZER_RE.search(command["stdout"] + command["stderr"]):
        raise EvidenceError(f"{identity} result/output is missing or has diagnostics")


def _audit_test_records(stage: Path, refs: list[dict[str, object]],
                        identities: tuple[str, ...] = TEST_NAMES) -> dict[str, dict]:
    tests = {}
    for identity in identities:
        record, raw = _test_raw_entries(stage, identity, refs)
        if identity in DEVICE_TESTS:
            _validate_device_test(stage, identity, raw, record)
        elif identity == "vulkan-device-helper-no-device":
            _validate_no_device_test(stage, raw, record)
        elif identity == "vulkan-device-helper-unit":
            _validate_unit_test(stage, raw, record, identity)
        elif identity == "vulkan-device-helper-independence":
            _validate_independence_test(stage, raw, record)
        else:
            _validate_generic_test_command(
                stage, raw, record, identity, "test-command-log",
                "ncnn.vulkan_device_helper_evidence_tests_result")
        tests[identity] = {"record": record, "raw_roles": sorted(raw)}
    return tests


def _run(command: list[str], *, cwd: Path | None = None,
         check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
    except OSError as error:
        raise EvidenceError(f"cannot run evidence command {command[0]}: {error}") from error
    if check and result.returncode != 0:
        raise EvidenceError(f"evidence command failed ({result.returncode}): {command!r}: "
                            f"{result.stderr.decode(errors='replace')}")
    return result


def _file_api_target(build_root: Path, name: str) -> tuple[dict, dict, Path]:
    index_files = sorted((build_root / ".cmake/api/v1/reply").glob("index-*.json"))
    if not index_files:
        raise EvidenceError(f"CMake File API reply missing in {build_root}")
    index = _load_json(index_files[-1], "CMake File API index")
    reply = index.get("reply", {}) if isinstance(index, dict) else {}
    codemodel_record = reply.get("codemodel-v2") if isinstance(reply, dict) else None
    if not isinstance(codemodel_record, dict):
        raise EvidenceError("CMake File API codemodel-v2 reply is absent")
    codemodel_path = (build_root / ".cmake/api/v1/reply" /
                      str(codemodel_record.get("jsonFile", ""))).resolve()
    codemodel = _load_json(codemodel_path, "CMake codemodel")
    configurations = codemodel.get("configurations") if isinstance(codemodel, dict) else None
    if not isinstance(configurations, list) or len(configurations) != 1:
        raise EvidenceError("CMake codemodel has no unique configuration")
    matches = [item for item in configurations[0].get("targets", [])
               if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise EvidenceError(f"CMake codemodel target is missing or ambiguous: {name}")
    target_ref = matches[0]
    target_json_path = (build_root / ".cmake/api/v1/reply" /
                        str(target_ref.get("jsonFile", ""))).resolve()
    target_json = _load_json(target_json_path, f"CMake target {name}")
    if not isinstance(target_json, dict) or target_json.get("name") != name:
        raise EvidenceError(f"CMake target JSON identity mismatch: {name}")
    return target_ref, target_json, target_json_path


def _read_closure_manifest(build_root: Path, target_names: set[str]) -> dict[str, dict]:
    candidates = (build_root / "test/Vulkan/device-helper-target-link-closure.json",
                  build_root / "device-helper-target-link-closure.json")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise EvidenceError(f"CMake target link closure is missing or ambiguous in {build_root}")
    path = existing[0]
    data = _load_json(path, "CMake target link closure")
    if (not _has_schema(data) or data.get("kind") != CLOSURE_KIND):
        raise EvidenceError("CMake target link closure schema/kind is invalid")
    targets = data.get("targets")
    if not isinstance(targets, list):
        raise EvidenceError("CMake target link closure has no target list")
    by_name = {}
    for item in targets:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise EvidenceError("CMake target link closure entry is malformed")
        name = item["name"]
        if name in by_name:
            raise EvidenceError(f"CMake target link closure repeats target {name}")
        by_name[name] = item
    if set(by_name) != target_names:
        raise EvidenceError("CMake target link closure target set differs from required targets")
    return by_name


def _cmake_source_root(build_root: Path) -> Path:
    cache = build_root / "CMakeCache.txt"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            path = Path(line.partition("=")[2]).resolve()
            if path.is_dir():
                return path
    raise EvidenceError(f"CMake source root is missing from cache: {cache}")


def _target_source_paths(build_root: Path, target_json: dict,
                         declared_sources: set[Path] | None = None) -> set[Path]:
    sources = target_json.get("sources")
    if not isinstance(sources, list) or not sources:
        raise EvidenceError(f"CMake target {target_json.get('name')} has no source inventory")
    top_source = _cmake_source_root(build_root)
    target_source = Path(str(target_json.get("paths", {}).get("source", "")))
    if not target_source.is_absolute():
        target_source = top_source / target_source
    target_source = target_source.resolve()
    paths = set()
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise EvidenceError("CMake target source record is malformed")
        raw = Path(item["path"])
        candidates = ({raw.resolve()} if raw.is_absolute() else
                      {(top_source / raw).resolve(), (target_source / raw).resolve()})
        candidates = {path for path in candidates if path.is_file()}
        if declared_sources is not None:
            candidates &= declared_sources
        if len(candidates) != 1:
            raise EvidenceError(f"CMake File API source is missing or ambiguous: {item['path']}")
        paths.update(candidates)
    return paths


def _cmake_cache_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def _expected_gate_tool(stage: Path, role: str) -> Path:
    state, _profile, _fingerprint, gate_bound = _read_stage_identity(stage)
    aliases = {
        "mlir-opt": {"mlir-opt", "lit-mlir-opt"},
        "spirv-val": {"spirv-val"},
        "spirv-dis": {"spirv-dis"},
    }
    if gate_bound and state is not None:
        identities = state.get("toolchain_identities")
        matches = [item for item in identities if isinstance(item, dict) and
                   item.get("role") in aliases.get(role, set())] if isinstance(identities, list) else []
        primary = [item for item in matches if item.get("role") == role]
        if primary:
            matches = primary
        if len(matches) != 1:
            raise EvidenceError(f"gate-state lacks unique {role} identity")
        path = Path(str(matches[0].get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != matches[0].get("sha256"):
            raise EvidenceError(f"gate-state {role} tool identity is stale")
        return path
    cache = _cmake_cache_values(stage / "CMakeCache.txt")
    keys = {
        "mlir-opt": ("NCNN_MLIR_OPT_EXECUTABLE", "MLIR_OPT"),
        "spirv-val": ("NCNN_SPIRV_VAL_EXECUTABLE",),
        "spirv-dis": ("NCNN_SPIRV_DIS_EXECUTABLE",),
    }.get(role, ())
    paths = [Path(cache[key]).resolve() for key in keys if cache.get(key)]
    paths = [path for path in paths if path.is_file()]
    if role == "mlir-opt" and len(set(paths)) > 1:
        raise EvidenceError("CMake MLIR optimizer cache entries disagree")
    if not paths:
        raise EvidenceError(f"CMake cache does not identify {role}")
    return paths[0]


def _validate_tool_log(stage: Path, raw: dict[str, Path], record: dict,
                       log_role: str, binary_role: str, tool_role: str,
                       expected_args: list[str]) -> dict:
    log = _command_log(raw[log_role], log_role)
    binary = _origin(record, binary_role, raw[binary_role], stage,
                     allow_external=True)
    if binary != _expected_gate_tool(stage, tool_role):
        raise EvidenceError(f"{log_role} executable differs from actual configured tool identity")
    if (Path(log["argv"][0]).resolve() != binary or
            log["argv"][1:] != expected_args or log["returncode"] != 0):
        raise EvidenceError(f"{log_role} did not run the captured tool/arguments successfully")
    return log


def _compile_database(build_root: Path) -> tuple[Path, list[dict]]:
    path = build_root / "compile_commands.json"
    data = _load_json(path, "compile_commands.json")
    if not isinstance(data, list) or not data:
        raise EvidenceError(f"compile database is empty or malformed: {path}")
    return path, data


def _argv_for_compile(entry: dict) -> list[str]:
    argv = entry.get("arguments")
    if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv):
        return list(argv)
    command = entry.get("command")
    if not isinstance(command, str) or not command:
        raise EvidenceError("compile command has no actual argv")
    try:
        return shlex.split(command)
    except ValueError as error:
        raise EvidenceError(f"cannot parse compile argv: {error}") from error


def _option_value(argv: list[str], name: str, cwd: Path) -> Path | None:
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            value = Path(argv[index + 1])
        elif token.startswith(name) and token != name and name in {"-MF", "-o"}:
            value = Path(token[len(name):])
        else:
            continue
        return (value if value.is_absolute() else cwd / value).resolve()
    return None


def _depfile_paths(path: Path, working_directory: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    separator = re.search(r"(?<!\\):\s", text)
    if separator is None:
        raise EvidenceError(f"compiler depfile has no target separator: {path}")
    dependencies = text[separator.end():]
    tokens = re.findall(r"(?:\\.|[^\s])+", dependencies)
    result = []
    for token in tokens:
        value = re.sub(r"\\(.)", r"\1", token)
        if value:
            dependency = Path(value)
            result.append((dependency if dependency.is_absolute() else
                           working_directory / dependency).resolve())
    if not result:
        raise EvidenceError(f"compiler depfile has no dependencies: {path}")
    return result


def _include_args(argv: list[str], cwd: Path) -> list[str]:
    includes = []
    index = 0
    while index < len(argv):
        token = argv[index]
        value = None
        if token in {"-I", "-isystem", "-iquote", "-idirafter", "-include", "-imacros"}:
            if index + 1 < len(argv):
                value = argv[index + 1]
                index += 1
        elif token.startswith(("-I", "-isystem", "-iquote", "-idirafter")):
            for prefix in ("-isystem", "-iquote", "-idirafter", "-I"):
                if token.startswith(prefix) and len(token) > len(prefix):
                    value = token[len(prefix):]
                    break
        if value is not None:
            include = Path(value)
            includes.append(str((include if include.is_absolute() else cwd / include).resolve()))
        index += 1
    return includes


def _copy_build_raw(stage: Path, output_root: Path, source: Path,
                    relative: Path, label: str) -> dict[str, object]:
    item = _copy_raw(source, stage, output_root, relative)
    item["role"] = label
    item["origin"] = str(source.resolve())
    return item


def _audit_target_compile(stage: Path, output_root: Path, build_root: Path,
                          role: str, target_json: dict,
                          closure_entry: dict, refs: list[dict[str, object]]) -> dict:
    name = target_json.get("name")
    declared_sources = closure_entry.get("sources")
    if (not isinstance(declared_sources, list) or
            any(not isinstance(item, str) for item in declared_sources)):
        raise EvidenceError(f"CMake closure manifest omits {name} sources")
    declared = {Path(item).resolve() for item in declared_sources}
    source_paths = _target_source_paths(build_root, target_json, declared)
    if declared != source_paths:
        raise EvidenceError(f"CMake closure source set differs from File API for {name}")
    for include in closure_entry.get("include_directories", []):
        _check_clean_dependency(include, f"{name} include directories")
    for field in ("link_libraries", "interface_link_libraries", "dependencies"):
        values = closure_entry.get(field, [])
        if not isinstance(values, list):
            raise EvidenceError(f"CMake closure {name}.{field} is not a list")
        for value in values:
            _check_clean_dependency(value, f"{name} {field}")
    compile_db_path, compile_db = _compile_database(build_root)
    selected = []
    for entry in compile_db:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise EvidenceError("compile database contains a malformed entry")
        source = Path(entry["file"])
        cwd = Path(str(entry.get("directory", build_root))).resolve()
        if not source.is_absolute():
            source = cwd / source
        source = source.resolve()
        if source not in source_paths:
            continue
        argv = _argv_for_compile(entry)
        object_path = _option_value(argv, "-o", cwd)
        if object_path is None and isinstance(entry.get("output"), str):
            object_path = Path(entry["output"])
            if not object_path.is_absolute():
                object_path = cwd / object_path
            object_path = object_path.resolve()
        if (object_path is None or
                not any(part.endswith(f"{name}.dir") for part in object_path.parts)):
            continue
        selected.append((entry, source, argv, object_path))
    selected_sources = {source for _, source, _, _ in selected}
    if selected_sources != source_paths or len(selected) != len(source_paths):
        missing = sorted(str(item) for item in source_paths - selected_sources)
        raise EvidenceError(f"actual compile argv is missing/ambiguous for target {name}: {missing}")
    compile_snapshot = _copy_build_raw(
        stage, output_root, compile_db_path,
        Path("dependency-inputs") / role / "compile_commands.json",
        f"{role} actual compile database")
    records = []
    for index, (entry, source, argv, object_path) in enumerate(
            sorted(selected, key=lambda item: str(item[1]))):
        cwd = Path(str(entry.get("directory", ""))).resolve()
        if not cwd.is_dir():
            raise EvidenceError(f"compile command working directory is unavailable: {cwd}")
        if not argv or not Path(argv[0]).is_file():
            raise EvidenceError(f"compile command compiler is unavailable for {source}")
        for include in _include_args(argv, cwd):
            _check_clean_dependency(include, f"{name} actual compile argv")
        for token in argv[1:]:
            if token.startswith("@"):
                raise EvidenceError(f"response-file compile flags cannot be audited: {token}")
            if _is_forbidden_path(token):
                raise EvidenceError(f"forbidden compile argument for {name}: {token}")
        if not object_path.is_file():
            raise EvidenceError(f"actual object output is missing for {source}")
        depfile = _option_value(argv, "-MF", cwd)
        if depfile is None:
            candidates = [object_path.with_suffix(object_path.suffix + ".d"),
                          object_path.with_suffix(".d")]
            depfile = next((item for item in candidates if item.is_file()), None)
        if depfile is None or not depfile.is_file():
            raise EvidenceError(f"actual compiler header depfile is missing for {source}")
        dependencies = _depfile_paths(depfile, cwd)
        for dependency in dependencies:
            _check_clean_dependency(str(dependency), f"{name} actual header depfile")
            if not dependency.is_file():
                raise EvidenceError(f"compiler depfile names a missing header/source: {dependency}")
        source_relative = Path("dependency-inputs") / role / "objects" / f"{index:04d}-{object_path.name}"
        object_copy = _copy_build_raw(stage, output_root, object_path, source_relative,
                                      f"{role} actual object input")
        dep_copy = _copy_build_raw(
            stage, output_root, depfile,
            Path("dependency-inputs") / role / "depfiles" / f"{index:04d}-{depfile.name}",
            f"{role} compiler header depfile")
        records.append({"source": str(source), "source_sha256": sha256_file(source),
                        "argv": argv, "working_directory": str(cwd),
                        "compile_output": str(object_path),
                        "object": object_copy, "depfile": dep_copy,
                        "dependencies": [{"path": str(path),
                                          "sha256": sha256_file(path)}
                                         for path in dependencies],
                        "include_directories": _include_args(argv, cwd)})
    return {"target": name, "cmake_source_root": str(_cmake_source_root(build_root)),
            "cmake_build_root": str(build_root.resolve()),
            "compile_database": compile_snapshot, "compile_commands": records}


def _target_artifact(build_root: Path, target_json: dict,
                     closure_entry: dict) -> Path:
    artifacts = target_json.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1 or not isinstance(artifacts[0], dict):
        raise EvidenceError(f"CMake target {target_json.get('name')} artifact is missing/ambiguous")
    value = artifacts[0].get("path")
    if not isinstance(value, str):
        raise EvidenceError("CMake target artifact path is malformed")
    path = Path(value)
    if not path.is_absolute():
        path = build_root / path
    path = path.resolve()
    declared = closure_entry.get("artifact")
    if not isinstance(declared, str) or Path(declared).resolve() != path:
        raise EvidenceError(f"CMake manifest artifact differs from File API: {path}")
    if not path.is_file():
        raise EvidenceError(f"CMake target artifact does not exist: {path}")
    return path


def _target_link_file(build_root: Path, target_json: dict) -> Path:
    target_name = target_json.get("name")
    paths = target_json.get("paths")
    target_build_value = paths.get("build") if isinstance(paths, dict) else None
    if not isinstance(target_name, str) or not target_name or not isinstance(target_build_value, str):
        raise EvidenceError("CMake File API target has no target-scoped build path")
    target_build = Path(target_build_value)
    if not target_build.is_absolute():
        target_build = build_root / target_build
    target_build = target_build.resolve()
    if not target_build.is_relative_to(build_root.resolve()):
        raise EvidenceError(f"CMake target build path escapes its build root: {target_name}")
    exact = target_build / "CMakeFiles" / f"{target_name}.dir" / "link.txt"
    if not exact.is_file():
        raise EvidenceError(f"cannot identify target-scoped actual link command for {target_name}")
    return exact


def _link_argv(link_file: Path) -> list[str]:
    text = link_file.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise EvidenceError(f"actual target link command is empty: {link_file}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        commands = [shlex.split(line) for line in lines]
    except ValueError as error:
        raise EvidenceError(f"cannot parse actual target link command {link_file}: {error}") from error
    if not commands or any(not command or any(token.startswith("@") for token in command)
                           for command in commands):
        raise EvidenceError(f"actual target link command is unresolved/response-file based: {link_file}")
    if len(commands) > 1:
        if len(commands) != 2:
            raise EvidenceError(f"actual archive link command has unexpected extra steps: {link_file}")
        archive_command, index_command = commands
        archive_tool = Path(archive_command[0]).name
        index_tool = Path(index_command[0]).name
        if archive_tool not in {"ar", "llvm-ar"} or index_tool not in {
                "ranlib", "llvm-ranlib"}:
            raise EvidenceError(f"actual multiline link is not an archive+ranlib sequence: {link_file}")
        archive_outputs = [Path(token).name for token in archive_command[1:]
                           if token.endswith((".a", ".lib"))]
        index_inputs = [Path(token).name for token in index_command[1:]
                        if token.endswith((".a", ".lib"))]
        if len(archive_outputs) != 1 or index_inputs != archive_outputs:
            raise EvidenceError(f"archive/ranlib commands name different outputs: {link_file}")
    return commands[0]


def _archive_members(ar: str, archive: Path) -> tuple[list[str], bytes]:
    listing = _run([ar, "t", str(archive)])
    try:
        members = [line for line in listing.stdout.decode("utf-8").splitlines() if line]
    except UnicodeDecodeError as error:
        raise EvidenceError(f"archive member listing is not UTF-8: {archive}") from error
    if not members or len(members) != len(set(members)):
        raise EvidenceError(f"archive is empty or has duplicate member names: {archive}")
    return members, listing.stdout


def _archive_object_inputs(build_root: Path, archive: Path,
                           link_argv: list[str]) -> list[Path]:
    objects = []
    for token in link_argv:
        if token.endswith((".o", ".obj")):
            path = Path(token)
            if not path.is_absolute():
                path = build_root / path
            path = path.resolve()
            if not path.is_file():
                raise EvidenceError(f"archive object input is missing: {path}")
            objects.append(path)
    if not objects:
        raise EvidenceError(f"actual archive command has no object inputs: {archive}")
    return objects


def _capture_archive(stage: Path, output_root: Path, archive: Path,
                     link_file: Path, build_root: Path, role: str,
                     refs: list[dict[str, object]], ar: str) -> dict:
    link_argv = _link_argv(link_file)
    object_paths = _archive_object_inputs(build_root, archive, link_argv)
    members, listing_bytes = _archive_members(ar, archive)
    by_name = {path.name: path for path in object_paths}
    if len(by_name) != len(object_paths) or set(by_name) != set(members):
        raise EvidenceError(f"archive object inputs differ from raw ar t inventory: {archive}")
    raw_dir = output_root / "dependency-inputs" / role / "archive"
    raw_dir.mkdir(parents=True, exist_ok=True)
    listing_path = raw_dir / "ar-t.txt"
    listing_path.write_bytes(listing_bytes)
    listing_record = {"path": str(listing_path.relative_to(stage)),
                      "sha256": sha256_file(listing_path),
                      "size_bytes": listing_path.stat().st_size}
    archive_copy_path = raw_dir / archive.name
    shutil.copyfile(archive, archive_copy_path)
    archive_record = {"path": str(archive_copy_path.relative_to(stage)),
                      "sha256": sha256_file(archive_copy_path),
                      "size_bytes": archive_copy_path.stat().st_size,
                      "origin": str(archive)}
    member_records = []
    for index, member in enumerate(members):
        source_object = by_name[member]
        process = _run([ar, "p", str(archive), member])
        if process.stdout != source_object.read_bytes():
            raise EvidenceError(f"archive member bytes differ from actual object input: {archive}({member})")
        copied_object = output_root / "dependency-inputs" / role / "objects" / f"archive-{index:04d}-{member}"
        copied_object.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_object, copied_object)
        raw_member = raw_dir / f"member-{index:04d}.o"
        raw_member.write_bytes(process.stdout)
        member_records.append({
            "member": member, "sha256": hashlib.sha256(process.stdout).hexdigest(),
            "size_bytes": len(process.stdout),
            "object": {"path": str(copied_object.relative_to(stage)),
                       "sha256": sha256_file(copied_object),
                       "size_bytes": copied_object.stat().st_size,
                       "origin": str(source_object)},
            "raw_member": {"path": str(raw_member.relative_to(stage)),
                           "sha256": sha256_file(raw_member),
                           "size_bytes": raw_member.stat().st_size},
        })
    link_record = _copy_build_raw(stage, output_root, link_file,
                                  Path("dependency-inputs") / role / "link.txt",
                                  f"{role} actual CMake link command")
    for item, label in ((archive_record, f"{role} actual static archive"),
                        (listing_record, f"{role} raw ar t listing"),
                        (link_record, f"{role} actual CMake link command")):
        refs.append({"path": item["path"], "sha256": item["sha256"], "role": label})
    return {"archive": archive_record, "ar_t": listing_record,
            "members": member_records, "link_command": link_record,
            "link_argv": link_argv, "working_directory": str(build_root)}


def _resolve_map_archives(argv: list[str], cwd: Path, map_text: str) -> dict[Path, list[str]]:
    references = model_evidence._archive_matches(map_text)
    if not references:
        raise EvidenceError("actual helper runner map has no selected archive members")
    search_dirs = [cwd]
    for index, token in enumerate(argv):
        if token == "-L" and index + 1 < len(argv):
            value = Path(argv[index + 1])
            search_dirs.append((value if value.is_absolute() else cwd / value).resolve())
        elif token.startswith("-L") and len(token) > 2:
            value = Path(token[2:])
            search_dirs.append((value if value.is_absolute() else cwd / value).resolve())
    resolved: dict[Path, list[str]] = {}
    for archive_text, member in references:
        archive = Path(archive_text)
        if not archive.is_absolute():
            archive = cwd / archive
        archive = archive.resolve()
        if not archive.is_file():
            matches = sorted({directory / Path(archive_text).name for directory in search_dirs
                              if (directory / Path(archive_text).name).is_file()})
            if len(matches) != 1:
                raise EvidenceError(f"cannot resolve actual map archive {archive_text}")
            archive = matches[0].resolve()
        resolved.setdefault(archive, []).append(member)
    return resolved


def _capture_runner_map(stage: Path, output_root: Path, build_root: Path,
                        runner_json: dict, helper_archive: Path,
                        refs: list[dict[str, object]], ar: str,
                        role: str) -> dict:
    runner = _target_artifact(build_root, runner_json,
                              {"artifact": str(Path(runner_json["artifacts"][0]["path"]) if
                                               Path(runner_json["artifacts"][0]["path"]).is_absolute() else
                                               (build_root / runner_json["artifacts"][0]["path"]).resolve())})
    link_file = _target_link_file(build_root, runner_json)
    argv = _link_argv(link_file)
    for token in argv[1:]:
        if token.startswith("@"):
            raise EvidenceError("standalone runner link uses an unexpanded response file")
        _check_clean_dependency(token, f"{role} actual link argv")
    cwd = build_root
    map_paths = []
    for token in argv:
        match = re.search(r"-Map=([^,]+)", token)
        if match:
            value = Path(match.group(1))
            map_paths.append((value if value.is_absolute() else cwd / value).resolve())
    if len(map_paths) != 1 or not map_paths[0].is_file():
        raise EvidenceError(f"{role} link has no unique actual linker map")
    map_path = map_paths[0]
    map_text = map_path.read_text(encoding="utf-8", errors="replace")
    if "Cross Reference Table" not in map_text or "Archive member included" not in map_text:
        raise EvidenceError("standalone helper link map lacks --cref archive provenance")
    archives = _resolve_map_archives(argv, cwd, map_text)
    for archive_path in archives:
        _check_clean_dependency(str(archive_path), "standalone runner linker-map archive")
    if not any(path == helper_archive.resolve() for path in archives):
        raise EvidenceError("standalone runner actual link map omits the helper archive")
    helper_refs = archives[helper_archive.resolve()]
    if not helper_refs:
        raise EvidenceError("standalone runner map selected no helper archive members")
    all_archives = []
    for archive, selected in sorted(archives.items(), key=lambda item: str(item[0])):
        members, listing = _archive_members(ar, archive)
        selected_hashes = []
        for member in sorted(set(selected)):
            if members.count(member) != 1:
                raise EvidenceError(f"runner link map selects absent/duplicate archive member: {archive}({member})")
            digest, size = model_evidence._hash_archive_member(ar, archive, member)
            selected_hashes.append({"member": member, "sha256": digest,
                                    "size_bytes": size})
        archive_dir = output_root / "dependency-inputs" / role / "link-archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_copy = archive_dir / f"{len(all_archives):04d}-{archive.name}"
        shutil.copyfile(archive, archive_copy)
        listing_path = archive_dir / f"{len(all_archives):04d}-ar-t.txt"
        listing_path.write_bytes(listing)
        all_archives.append({"origin": str(archive),
                             "archive": {"path": str(archive_copy.relative_to(stage)),
                                         "sha256": sha256_file(archive_copy),
                                         "size_bytes": archive_copy.stat().st_size},
                             "ar_t": {"path": str(listing_path.relative_to(stage)),
                                      "sha256": sha256_file(listing_path),
                                      "size_bytes": listing_path.stat().st_size},
                             "members": members,
                             "selected_members_from_map": sorted(set(selected)),
                             "selected_member_hashes": selected_hashes})
    map_record = _copy_build_raw(stage, output_root, map_path,
                                 Path("dependency-inputs") / role / "link.map",
                                 f"{role} actual linker map")
    link_record = _copy_build_raw(stage, output_root, link_file,
                                  Path("dependency-inputs") / role / "link.txt",
                                  f"{role} actual link command")
    runner_record = _copy_build_raw(stage, output_root, runner,
                                    Path("dependency-inputs") / role / runner.name,
                                    f"{role} executable")
    for item, label in ((map_record, f"{role} actual linker map"),
                        (link_record, f"{role} actual link command"),
                        (runner_record, f"{role} executable")):
        refs.append({"path": item["path"], "sha256": item["sha256"], "role": label})
    return {"runner": runner_record, "link_command": link_record,
            "link_map": map_record, "map_argv": argv,
            "working_directory": str(cwd), "archive_inputs": all_archives}


def _verified_compiler_cache(cache: dict[str, str], label: str) -> tuple[dict[str, str], bytes]:
    """Check the real compiler; ID/version are not normally CMake cache keys."""
    configured = cache.get("CMAKE_CXX_COMPILER")
    if not isinstance(configured, str) or not configured:
        raise EvidenceError(f"{label} CMake cache has no C++ compiler path")
    compiler = Path(configured).resolve()
    if not compiler.is_file():
        raise EvidenceError(f"{label} C++ compiler does not exist: {compiler}")
    result = _run([str(compiler), "--version"], check=True)
    raw = result.stdout + result.stderr
    text = raw.decode("utf-8", errors="replace")
    clang = re.search(r"\bclang version\s+(\d+\.\d+(?:\.\d+)?)", text)
    if clang:
        compiler_id, version = "Clang", clang.group(1)
    else:
        lines = text.splitlines()
        first = lines[0] if lines else ""
        gnu = re.search(r"\b(\d+\.\d+\.\d+)\b", first)
        if not gnu or "Free Software Foundation" not in text:
            raise EvidenceError(f"{label} C++ compiler version is not recognized GCC/Clang output")
        compiler_id, version = "GNU", gnu.group(1)
    verified = dict(cache)
    for key, actual in (("CMAKE_CXX_COMPILER_ID", compiler_id),
                        ("CMAKE_CXX_COMPILER_VERSION", version)):
        if key in cache and cache[key] != actual:
            raise EvidenceError(f"{label} cached compiler identity contradicts --version: {key}")
        verified[key] = actual
    return verified, raw


def _capture_toolchain_identity(stage: Path, output_root: Path,
                                standalone_root: Path, profile: str,
                                state: dict, targets: dict,
                                refs: list[dict[str, object]]) -> dict:
    production_cache = stage / "CMakeCache.txt"
    standalone_cache = standalone_root / "CMakeCache.txt"
    if not production_cache.is_file() or not standalone_cache.is_file():
        raise EvidenceError("production/standalone CMake cache is missing")
    production, _ = _verified_compiler_cache(
        _cmake_cache_values(production_cache), "production")
    standalone, _ = _verified_compiler_cache(
        _cmake_cache_values(standalone_cache), "standalone")
    compiler_keys = ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID",
                     "CMAKE_CXX_COMPILER_VERSION")
    for key in compiler_keys:
        if not production.get(key) or not standalone.get(key):
            raise EvidenceError(f"C++ compiler identity is missing from CMake cache: {key}")
    production_compiler = Path(production["CMAKE_CXX_COMPILER"]).resolve()
    standalone_compiler = Path(standalone["CMAKE_CXX_COMPILER"]).resolve()
    if production_compiler != standalone_compiler:
        raise EvidenceError("standalone C++ compiler path differs from production compiler")
    if any(production[key] != standalone[key] for key in compiler_keys[1:]):
        raise EvidenceError("standalone C++ compiler ID/version differs from production")
    if not production_compiler.is_file():
        raise EvidenceError(f"C++ compiler does not exist: {production_compiler}")
    compiler_hash = sha256_file(production_compiler)
    identity = None
    if state is not None:
        tool_ids = state.get("toolchain_identities")
        cxx_identities = [item for item in tool_ids if isinstance(item, dict) and
                           item.get("role") == "cxx-compiler"] if isinstance(tool_ids, list) else []
        if len(cxx_identities) != 1:
            raise EvidenceError("gate-state lacks a unique actual C++ compiler identity")
        identity = cxx_identities[0]
        if (Path(str(identity.get("path", ""))).resolve() != production_compiler or
                identity.get("sha256") != compiler_hash):
            raise EvidenceError("standalone/production compiler differs from gate tool identity")
    version_result = _run([str(production_compiler), "--version"], check=True)
    version_text = (version_result.stdout + version_result.stderr).decode(
        "utf-8", errors="replace")
    if (not version_text.strip() or
            (identity is not None and
             (not isinstance(identity.get("version"), str) or
              version_text.strip() != identity["version"].strip()))):
        raise EvidenceError("actual C++ compiler version differs from gate tool identity")
    version_path = output_root / "dependency-inputs/toolchain/cxx-compiler-version.txt"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_bytes(version_result.stdout + version_result.stderr)
    version_record = {"path": str(version_path.relative_to(stage)),
                      "sha256": sha256_file(version_path),
                      "size_bytes": version_path.stat().st_size}
    production_cache_copy = _copy_build_raw(
        stage, output_root, production_cache,
        Path("dependency-inputs/toolchain/production-CMakeCache.txt"),
        "production CMakeCache compiler/sanitizer identity")
    standalone_cache_copy = _copy_build_raw(
        stage, output_root, standalone_cache,
        Path("dependency-inputs/toolchain/standalone-CMakeCache.txt"),
        "standalone CMakeCache compiler/sanitizer identity")
    refs.extend((production_cache_copy, standalone_cache_copy,
                 {**version_record, "role": "actual gate C++ compiler --version output",
                  "origin": str(production_compiler)}))
    resource_record = None
    resource_dir = None
    if profile == "sanitize":
        sanitizers = set(production.get("COMPILER_SANITIZERS", "").split(";"))
        if sanitizers != {"address", "undefined"}:
            raise EvidenceError("SAN CMake configuration is not exact address+undefined")
        if production["CMAKE_CXX_COMPILER_ID"] != "Clang" or not production["CMAKE_CXX_COMPILER_VERSION"].startswith("21."):
            raise EvidenceError("SAN helper requires the matching Clang 21 C++ compiler")
        resource_result = _run([str(production_compiler), "--print-resource-dir"], check=True)
        resource_text = (resource_result.stdout + resource_result.stderr).decode(
            "utf-8", errors="replace").strip()
        if not resource_text:
            raise EvidenceError("Clang compiler returned no resource directory")
        resource_dir = str(Path(resource_text).resolve())
        runtime_dir = Path(resource_dir) / "lib/linux"
        runtimes = sorted(runtime_dir.glob("libclang_rt.asan-*.so"))
        if not runtime_dir.is_dir() or not runtimes:
            raise EvidenceError("matching shared Clang ASan runtime is missing")
        stand_sanitizers = set(standalone.get("COMPILER_SANITIZERS", "").split(";"))
        if stand_sanitizers != {"address", "undefined"}:
            raise EvidenceError("standalone SAN CMake cache is not exact address+undefined")
        stand_cxx_flags = standalone.get("CMAKE_CXX_FLAGS", "")
        stand_link_flags = standalone.get("CMAKE_EXE_LINKER_FLAGS", "")
        for flag in ("-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                     "-fno-omit-frame-pointer"):
            if flag not in stand_cxx_flags:
                raise EvidenceError(f"standalone CMake CXX flags omit required SAN flag {flag}")
        for flag in ("-fsanitize=address,undefined", "-shared-libasan",
                     f"-Wl,-rpath,{runtime_dir}"):
            if flag not in stand_link_flags:
                raise EvidenceError(f"standalone CMake link flags omit required SAN flag {flag}")
        resource_path = output_root / "dependency-inputs/toolchain/cxx-resource-dir.txt"
        resource_path.write_bytes(resource_result.stdout + resource_result.stderr)
        resource_record = {"path": str(resource_path.relative_to(stage)),
                           "sha256": sha256_file(resource_path),
                           "size_bytes": resource_path.stat().st_size}
        refs.append({**resource_record, "role": "actual Clang C++ compiler resource-dir output",
                     "origin": str(production_compiler)})
        readelf = shutil.which("readelf")
        if readelf is None:
            raise EvidenceError("readelf is required to audit shared ASan runtime linkage")
        runtime_records = []
        for role in ("standalone_runner", "standalone_unit"):
            binary = _safe_path(stage, targets[role]["artifact"]["path"])
            result = _run([readelf, "-d", str(binary)], check=True)
            text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
            needed = sorted(set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", text)))
            if not any("libclang_rt.asan" in item for item in needed):
                raise EvidenceError(f"{role} does not dynamically link shared ASan runtime")
            if any(re.match(r"lib(?:LLVM|MLIR).*\.so", item, re.IGNORECASE)
                   for item in needed):
                raise EvidenceError(f"{role} links a core LLVM/MLIR DSO")
            raw_path = output_root / f"dependency-inputs/toolchain/{role}-readelf-dynamic.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(result.stdout + result.stderr)
            raw = {"path": str(raw_path.relative_to(stage)),
                   "sha256": sha256_file(raw_path), "size_bytes": raw_path.stat().st_size}
            refs.append({**raw, "role": f"{role} raw readelf dynamic table",
                         "origin": str(binary)})
            runtime_records.append({"target": role, "needed": needed,
                                    "readelf": raw,
                                    "command": [readelf, "-d", str(binary)]})
    else:
        runtime_records = []
    return {"profile": profile,
            "compiler": {"path": str(production_compiler), "sha256": compiler_hash,
                         "compiler_id": production["CMAKE_CXX_COMPILER_ID"],
                         "compiler_version": production["CMAKE_CXX_COMPILER_VERSION"],
                         "version_output": {**version_record,
                                            "role": "actual gate C++ compiler --version output",
                                            "origin": str(production_compiler)}},
            "production_cache": production_cache_copy,
            "standalone_cache": standalone_cache_copy,
            "resource_dir": resource_dir, "resource_dir_output": resource_record,
            "runtime_linkage": runtime_records}


def _target_record(stage: Path, output_root: Path, build_root: Path,
                   role: str, target_name: str, closure: dict,
                   refs: list[dict[str, object]], ar: str,
                   *, capture_archive: bool) -> tuple[dict, Path]:
    target_ref, target_json, target_json_path = _file_api_target(build_root, target_name)
    expected_type = "STATIC_LIBRARY" if target_name == HELPER_TARGET else "EXECUTABLE"
    if target_json.get("type") != expected_type:
        raise EvidenceError(f"CMake target {target_name} type is not {expected_type}")
    target_data = _audit_target_compile(stage, output_root, build_root,
                                        role, target_json, closure, refs)
    target_data["file_api_target"] = _copy_build_raw(
        stage, output_root, target_json_path,
        Path("dependency-inputs") / role / "file-api-target.json",
        f"{role} CMake File API target metadata")
    target_data["file_api_target_id"] = target_ref.get("id")
    artifact = _target_artifact(build_root, target_json, closure)
    target_data["artifact"] = _copy_build_raw(
        stage, output_root, artifact,
        Path("dependency-inputs") / role / artifact.name,
        f"{role} actual CMake target artifact")
    refs.append({"path": target_data["file_api_target"]["path"],
                 "sha256": target_data["file_api_target"]["sha256"],
                 "role": target_data["file_api_target"]["role"]})
    refs.append({"path": target_data["artifact"]["path"],
                 "sha256": target_data["artifact"]["sha256"],
                 "role": target_data["artifact"]["role"]})
    if not capture_archive:
        return target_data, artifact
    if artifact.name != EXPECTED_ARCHIVE:
        raise EvidenceError(f"{target_name} artifact name is unexpected: {artifact.name}")
    link_file = _target_link_file(build_root, target_json)
    target_build_value = target_json.get("paths", {}).get("build")
    if not isinstance(target_build_value, str) or not target_build_value:
        raise EvidenceError(f"{target_name} File API target has no build working directory")
    archive_workdir = Path(target_build_value)
    if not archive_workdir.is_absolute():
        archive_workdir = build_root / archive_workdir
    archive_workdir = archive_workdir.resolve()
    if not archive_workdir.is_relative_to(build_root.resolve()):
        raise EvidenceError(f"{target_name} archive working directory escapes its build root")
    archive_record = _capture_archive(stage, output_root, artifact, link_file,
                                     archive_workdir, role, refs, ar)
    target_data["archive_provenance"] = archive_record
    return target_data, artifact


def _capture_root_test_binary(stage: Path, output_root: Path, target_name: str,
                              role: str, refs: list[dict[str, object]]) -> dict:
    target_ref, target_json, target_json_path = _file_api_target(stage, target_name)
    if target_json.get("type") != "EXECUTABLE":
        raise EvidenceError(f"root test target is not an executable: {target_name}")
    artifact_rows = target_json.get("artifacts")
    if not isinstance(artifact_rows, list) or len(artifact_rows) != 1:
        raise EvidenceError(f"root test target artifact is missing/ambiguous: {target_name}")
    artifact_value = artifact_rows[0].get("path")
    if not isinstance(artifact_value, str):
        raise EvidenceError(f"root test target artifact path is malformed: {target_name}")
    artifact_path = Path(artifact_value)
    if not artifact_path.is_absolute():
        artifact_path = stage / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file() or not artifact_path.is_relative_to(stage):
        raise EvidenceError(f"root test executable is missing/outside stage: {artifact_path}")
    target_record = _copy_build_raw(
        stage, output_root, target_json_path,
        Path("dependency-inputs/root-test-binaries") / f"{role}-file-api-target.json",
        f"{role} root CMake File API target")
    binary_record = _copy_build_raw(
        stage, output_root, artifact_path,
        Path("dependency-inputs/root-test-binaries") / artifact_path.name,
        f"{role} root CMake test executable")
    link_file = _target_link_file(stage, target_json)
    link_record = _copy_build_raw(
        stage, output_root, link_file,
        Path("dependency-inputs/root-test-binaries") / f"{role}-link.txt",
        f"{role} root actual link command")
    refs.extend((target_record, binary_record, link_record))
    return {"target": target_name, "target_id": target_ref.get("id"),
            "file_api_target": target_record, "artifact": binary_record,
            "link_command": link_record}


def _check_product_only(stage: Path, product_root: Path,
                        output_root: Path, refs: list[dict[str, object]]) -> dict:
    if not product_root.is_dir():
        raise EvidenceError(f"fresh product-only build is missing: {product_root}")
    cache = product_root / "CMakeCache.txt"
    text = cache.read_text(encoding="utf-8", errors="replace")
    cache_values = {}
    for line in text.splitlines():
        if line.startswith("NCNN_ENABLE_VULKAN_PRODUCT:") or line.startswith(
                "NCNN_ENABLE_VULKAN_V0_TESTS:") or line.startswith("BUILD_TESTING:"):
            key, _, value = line.partition("=")
            cache_values[key.split(":", 1)[0]] = value
    if cache_values.get("NCNN_ENABLE_VULKAN_PRODUCT") != "ON" or cache_values.get("BUILD_TESTING") != "ON":
        raise EvidenceError("product-only child configuration did not enable product and testing")
    if cache_values.get("NCNN_ENABLE_VULKAN_V0_TESTS") != "OFF":
        raise EvidenceError("product-only child configuration did not disable device tests")
    inventory_argv = ["ctest", "--test-dir", str(product_root), "--show-only=json-v1"]
    inventory_run = _run(inventory_argv)
    inventory_command_path = output_root / "product-only-ctest-inventory-command.json"
    inventory_command_path.write_text(json.dumps(
        {"argv": inventory_argv, "returncode": inventory_run.returncode,
         "stdout": inventory_run.stdout.decode("utf-8", errors="strict"),
         "stderr": inventory_run.stderr.decode("utf-8", errors="strict")},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        inventory = json.loads(inventory_run.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"product-only CTest inventory is malformed: {error}") from error
    tests = inventory.get("tests") if isinstance(inventory, dict) else None
    if (not isinstance(tests, list) or
            any(not isinstance(item, dict) or type(item.get("name")) is not str or
                not item["name"] for item in tests)):
        raise EvidenceError("product-only CTest inventory has a malformed test list")
    names = [item["name"] for item in tests]
    if len(names) != len(set(names)):
        raise EvidenceError("product-only CTest inventory repeats a test name")
    missing = sorted(set(PRODUCT_ONLY_REGISTERED_TESTS) - set(names))
    forbidden = sorted(set(DEVICE_TESTS) & set(names))
    if missing or forbidden:
        raise EvidenceError(f"product-only helper inventory mismatch: missing={missing}, forbidden={forbidden}")
    inventory_copy = output_root / "dependency-inputs/product-only/ctest-inventory.json"
    inventory_copy.parent.mkdir(parents=True, exist_ok=True)
    inventory_copy.write_bytes(inventory_run.stdout)
    junit_path = output_root / "product-only-ctest.xml"
    command = ["ctest", "--test-dir", str(product_root), "--output-on-failure",
               "--output-junit", str(junit_path), "-R",
               "^(" + "|".join(re.escape(name) for name in PRODUCT_ONLY_EXECUTE_TESTS) + ")$"]
    execution = _run(command, check=False)
    ctest_command_path = output_root / "product-only-ctest-command.json"
    ctest_command_path.write_text(json.dumps(
        {"argv": command, "returncode": execution.returncode,
         "stdout": execution.stdout.decode("utf-8", errors="replace"),
         "stderr": execution.stderr.decode("utf-8", errors="replace")},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stdout_path = output_root / "product-only-ctest.stdout.log"
    stderr_path = output_root / "product-only-ctest.stderr.log"
    stdout_path.write_bytes(execution.stdout)
    stderr_path.write_bytes(execution.stderr)
    if type(execution.returncode) is not int or execution.returncode != 0:
        raise EvidenceError(f"product-only helper tests failed ({execution.returncode})")
    junit_tree = ET.parse(junit_path)
    cases = list(junit_tree.getroot().iter("testcase"))
    junit_names = [case.get("name") for case in cases]
    if (set(junit_names) != set(PRODUCT_ONLY_EXECUTE_TESTS) or
            len(junit_names) != len(PRODUCT_ONLY_EXECUTE_TESTS) or
            any(case.find("skipped") is not None or case.find("failure") is not None or
                case.find("error") is not None for case in cases)):
        raise EvidenceError("product-only helper JUnit identity/result inventory is invalid")
    lasttest_source = product_root / "Testing/Temporary/LastTest.log"
    if not lasttest_source.is_file():
        raise EvidenceError("product-only LastTest.log is missing")
    lasttest_copy = output_root / "product-only-LastTest.log"
    shutil.copyfile(lasttest_source, lasttest_copy)
    lasttest_text = lasttest_copy.read_text(encoding="utf-8", errors="replace")
    lasttest_names = re.findall(r"^\d+/\d+ Testing: (.+)$", lasttest_text, re.MULTILINE)
    if (set(lasttest_names) != set(PRODUCT_ONLY_EXECUTE_TESTS) or
            len(lasttest_names) != len(PRODUCT_ONLY_EXECUTE_TESTS) or
            "Test Failed." in lasttest_text or "Test Not Run." in lasttest_text or
            "Test Skipped." in lasttest_text):
        raise EvidenceError("product-only LastTest results do not prove the exact recursion-safe test set passed")
    child_test_ids = (TEST_NAMES[0], TEST_NAMES[1], TEST_NAMES[3])
    child_refs: list[dict[str, object]] = []
    child_tests = _audit_test_records(product_root, child_refs, child_test_ids)
    for item in child_refs:
        child_path = product_root / str(item["path"])
        refs.append({"path": str(child_path.relative_to(stage)),
                     "sha256": item["sha256"],
                     "role": f"product-only actual {item['role']}"})
    copied = []
    for source, relative, label in (
            (cache, Path("dependency-inputs/product-only/CMakeCache.txt"), "product-only CMake cache"),
            (junit_path, Path("dependency-inputs/product-only/ctest.xml"), "product-only helper JUnit"),
            (stdout_path, Path("dependency-inputs/product-only/ctest.stdout.log"), "product-only CTest stdout"),
            (stderr_path, Path("dependency-inputs/product-only/ctest.stderr.log"), "product-only CTest stderr"),
            (lasttest_copy, Path("dependency-inputs/product-only/LastTest.log"), "product-only LastTest.log"),
            (inventory_command_path,
             Path("dependency-inputs/product-only/ctest-inventory-command.json"),
             "product-only CTest inventory command"),
            (ctest_command_path, Path("dependency-inputs/product-only/ctest-command.json"),
             "product-only CTest execution command")):
        copied.append(_copy_build_raw(stage, output_root, source, relative, label))
    for item in copied:
        refs.append({"path": item["path"], "sha256": item["sha256"], "role": item["role"]})
    inventory_record = {"path": str(inventory_copy.relative_to(stage)),
                        "sha256": sha256_file(inventory_copy),
                        "size_bytes": inventory_copy.stat().st_size}
    refs.append({"path": inventory_record["path"], "sha256": inventory_record["sha256"],
                 "role": "product-only actual CTest inventory"})
    return {"build_root": str(product_root), "profile_config": cache_values,
            "registered_helper_tests": sorted(set(PRODUCT_ONLY_REGISTERED_TESTS) & set(names)),
            "forbidden_device_tests": forbidden,
            "inventory": inventory_record,
            "inventory_command": next(item for item in copied
                                      if item["role"] == "product-only CTest inventory command"),
            "ctest_command": next(item for item in copied
                                  if item["role"] == "product-only CTest execution command"),
            "cache": next(item for item in copied if item["role"] == "product-only CMake cache"),
            "junit": next(item for item in copied if item["role"] == "product-only helper JUnit"),
            "lasttest": next(item for item in copied if item["role"] == "product-only LastTest.log"),
            "child_helper_test_records": {
                identity: {"record": child_tests[identity]["record"],
                           "raw_roles": child_tests[identity]["raw_roles"]}
                for identity in child_test_ids},
            "executed_test_names": sorted(junit_names), "returncode": execution.returncode}


def _canonical_raw_inventory(stage: Path, refs: list[dict[str, object]]) -> list[dict]:
    inventory: dict[str, dict] = {}
    for ref in refs:
        path = _safe_path(stage, ref.get("path"))
        relative = str(path.relative_to(stage))
        digest, size = sha256_file(path), path.stat().st_size
        if ref.get("sha256") != digest:
            raise EvidenceError(f"captured raw input changed before inventory publication: {relative}")
        if "size_bytes" in ref and (type(ref["size_bytes"]) is not int or
                                    ref["size_bytes"] != size):
            raise EvidenceError(f"captured raw input size contradicts actual bytes: {relative}")
        entry = {**ref, "path": relative, "size_bytes": size}
        previous = inventory.get(relative)
        if previous is not None and previous["sha256"] != digest:
            raise EvidenceError(f"captured raw input has inconsistent identities: {relative}")
        inventory.setdefault(relative, entry)
    return [inventory[path] for path in sorted(inventory)]


def capture_independence_evidence(stage_root: os.PathLike[str] | str,
                                  standalone_build_root: os.PathLike[str] | str,
                                  product_only_root: os.PathLike[str] | str,
                                  output: os.PathLike[str] | str) -> dict:
    stage = Path(stage_root).resolve()
    standalone = Path(standalone_build_root).resolve()
    product = Path(product_only_root).resolve()
    output_path = Path(output).resolve()
    try:
        output_path.relative_to(stage)
        standalone.relative_to(stage)
        product.relative_to(stage)
    except ValueError as error:
        raise EvidenceError("helper evidence/build roots must remain inside the stage") from error
    state, profile, source_fingerprint, gate_state_bound = _read_stage_identity(stage)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, object]] = []
    main_manifest_path = stage / "test/Vulkan/device-helper-target-link-closure.json"
    standalone_manifest_path = standalone / "device-helper-target-link-closure.json"
    main_closure = _read_closure_manifest(stage, {HELPER_TARGET})
    standalone_closure = _read_closure_manifest(
        standalone, {HELPER_TARGET, RUNNER_TARGET, UNIT_TARGET})
    closure_manifest_refs = {}
    for key, item, label in (
            ("production", main_manifest_path, "production CMake target link closure"),
            ("standalone", standalone_manifest_path, "standalone CMake target link closure")):
        copied_manifest = _copy_build_raw(
            stage, output_path.parent, item,
            Path("dependency-inputs") / f"{key}-target-link-closure.json", label)
        refs.append(copied_manifest)
        closure_manifest_refs[key] = copied_manifest
    ar = shutil.which("ar")
    if ar is None:
        raise EvidenceError("ar is required to audit helper archive object inputs")
    main_record, main_archive = _target_record(
        stage, output_path.parent, stage, "production-helper", HELPER_TARGET,
        main_closure[HELPER_TARGET], refs, ar, capture_archive=True)
    root_test_binaries = {
        "runner": _capture_root_test_binary(
            stage, output_path.parent, RUNNER_TARGET, "root-runner", refs),
        "unit": _capture_root_test_binary(
            stage, output_path.parent, UNIT_TARGET, "root-unit", refs),
    }
    standalone_helper, standalone_archive = _target_record(
        stage, output_path.parent, standalone, "standalone-helper", HELPER_TARGET,
        standalone_closure[HELPER_TARGET], refs, ar, capture_archive=True)
    runner_record, _runner_artifact = _target_record(
        stage, output_path.parent, standalone, "standalone-runner", RUNNER_TARGET,
        standalone_closure[RUNNER_TARGET], refs, ar, capture_archive=False)
    unit_record, _unit_artifact = _target_record(
        stage, output_path.parent, standalone, "standalone-unit", UNIT_TARGET,
        standalone_closure[UNIT_TARGET], refs, ar, capture_archive=False)
    runner_ref, runner_json, _runner_json_path = _file_api_target(standalone, RUNNER_TARGET)
    unit_ref, unit_json, _unit_json_path = _file_api_target(standalone, UNIT_TARGET)
    runner_record["file_api_target_id"] = runner_ref.get("id")
    unit_record["file_api_target_id"] = unit_ref.get("id")
    runner_link = _capture_runner_map(
        stage, output_path.parent, standalone, runner_json, standalone_archive,
        refs, ar, "standalone-runner")
    unit_link = _capture_runner_map(
        stage, output_path.parent, standalone, unit_json, standalone_archive,
        refs, ar, "standalone-unit")
    for link_role, link_record in (("runner", runner_link), ("unit", unit_link)):
        helper_link_members = set()
        for archive in link_record["archive_inputs"]:
            if Path(archive["origin"]).resolve() == standalone_archive.resolve():
                helper_link_members.update(archive["selected_members_from_map"])
        if not helper_link_members:
            raise EvidenceError(f"standalone {link_role} map omits helper archive members")
    toolchain = _capture_toolchain_identity(
        stage, output_path.parent, standalone, profile, state,
        {"production_helper": main_record, "standalone_helper": standalone_helper,
         "standalone_runner": runner_record, "standalone_unit": unit_record}, refs)
    product_only = _check_product_only(stage, product, output_path.parent, refs)
    evidence = {
        "schema_version": SCHEMA_VERSION, "kind": INDEPENDENCE_KIND,
        "profile": profile, "source_fingerprint": source_fingerprint,
        "stage_id": (state.get("stage_id", "VA0") if gate_state_bound and state else None),
        "gate_state_bound": gate_state_bound,
        "status": "verified" if gate_state_bound else "unqualified-development",
        "artifact_scope_only": True,
        "qualified_for_va2": False,
        "host_execution_role": (state.get("host_execution_role") if state else None),
        "host_aot_candidate": (state.get("host_aot_candidate") if state else None),
        "build_roots": {"production": str(stage), "standalone": str(standalone),
                        "product_only": str(product)},
        "closure_manifests": closure_manifest_refs,
        "targets": {"production_helper": main_record,
                    "standalone_helper": standalone_helper,
                    "standalone_runner": runner_record,
                    "standalone_unit": unit_record},
        "production_archive": {"origin": str(main_archive),
                               "sha256": sha256_file(main_archive)},
        "standalone_archive": {"origin": str(standalone_archive),
                              "sha256": sha256_file(standalone_archive)},
        "runner_link": runner_link,
        "unit_link": unit_link,
        "toolchain": toolchain,
        "product_only": product_only,
        "raw_artifacts": _canonical_raw_inventory(stage, refs),
        "claim": "helper-target-and-standalone-link-inputs-audited; shader-tool-LLVM-closure-out-of-scope",
        "dt_needed_nm_are_supplementary_only": True,
    }
    _write_json(output_path, evidence)
    validate_independence_evidence(output_path, stage, require_test_matrix=False)
    return evidence


def _verify_raw_entry(stage: Path, item: object, label: str) -> Path:
    if not isinstance(item, dict):
        raise EvidenceError(f"{label} raw record is malformed")
    path = _safe_path(stage, item.get("path"))
    digest = item.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or sha256_file(path) != digest:
        raise EvidenceError(f"{label} raw bytes are missing or tampered")
    if type(item.get("size_bytes")) is not int or item["size_bytes"] != path.stat().st_size:
        raise EvidenceError(f"{label} raw size is missing or invalid")
    return path


def _target_source_paths_from_json(target_json: object, source_root: Path,
                                   expected_sources: set[Path]) -> set[Path]:
    if not isinstance(target_json, dict) or not isinstance(target_json.get("sources"), list):
        raise EvidenceError("captured CMake File API target has no source inventory")
    target_source = Path(str(target_json.get("paths", {}).get("source", "")))
    if not target_source.is_absolute():
        target_source = source_root / target_source
    target_source = target_source.resolve()
    paths = set()
    for item in target_json["sources"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise EvidenceError("captured CMake File API source record is malformed")
        raw = Path(item["path"])
        candidates = ({raw.resolve()} if raw.is_absolute() else
                      {(source_root / raw).resolve(), (target_source / raw).resolve()})
        candidates = {path for path in candidates if path in expected_sources and path.is_file()}
        if len(candidates) != 1:
            raise EvidenceError(f"captured CMake File API source is missing or ambiguous: {raw}")
        paths.update(candidates)
    return paths


def _validate_recorded_compile(stage: Path, target: dict, label: str) -> None:
    commands = target.get("compile_commands")
    if not isinstance(commands, list) or not commands:
        raise EvidenceError(f"{label} actual compile argv inventory is empty")
    database_path = _verify_raw_entry(stage, target.get("compile_database"),
                                      f"{label} compile database")
    database = _load_json(database_path, f"{label} raw compile database")
    if not isinstance(database, list):
        raise EvidenceError(f"{label} raw compile database is malformed")
    actual_by_source = {}
    for entry in database:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise EvidenceError(f"{label} raw compile database contains an invalid row")
        source = Path(entry["file"])
        cwd = Path(str(entry.get("directory", ""))).resolve()
        if not source.is_absolute():
            source = cwd / source
        source = source.resolve()
        actual_by_source.setdefault(source, []).append((entry, cwd))
    sources = set()
    for record in commands:
        if not isinstance(record, dict) or not isinstance(record.get("argv"), list):
            raise EvidenceError(f"{label} compile argv record is malformed")
        source = Path(str(record.get("source", ""))).resolve()
        if not source.is_file() or sha256_file(source) != record.get("source_sha256"):
            raise EvidenceError(f"{label} compiled source is stale or changed: {source}")
        if source in sources:
            raise EvidenceError(f"{label} compile argv repeats source: {source}")
        sources.add(source)
        candidates = actual_by_source.get(source, [])
        matched = []
        for entry, cwd in candidates:
            actual_argv = _argv_for_compile(entry)
            output = _option_value(actual_argv, "-o", cwd)
            if output is None and isinstance(entry.get("output"), str):
                output = Path(entry["output"])
                if not output.is_absolute():
                    output = cwd / output
                output = output.resolve()
            if (record.get("working_directory") == str(cwd) and
                    record.get("compile_output") == str(output) and
                    actual_argv == record["argv"]):
                matched.append((entry, cwd))
        if len(matched) != 1:
            raise EvidenceError(f"{label} actual compiler argv/output differs from captured database: {source}")
        entry, cwd = matched[0]
        argv = record["argv"]
        if not argv or not Path(argv[0]).is_file():
            raise EvidenceError(f"{label} captured compiler executable is unavailable")
        for token in argv[1:]:
            if token.startswith("@") or _is_forbidden_path(token):
                raise EvidenceError(f"{label} compile argv contains unresolved/forbidden input: {token}")
        object_copy = _verify_raw_entry(stage, record.get("object"), f"{label} object")
        object_record = record.get("object")
        compile_output = Path(str(record.get("compile_output", ""))).resolve()
        object_origin = Path(str(object_record.get("origin", ""))).resolve()
        actual_output = _option_value(argv, "-o", cwd)
        if actual_output is None and isinstance(entry.get("output"), str):
            actual_output = Path(entry["output"])
            if not actual_output.is_absolute():
                actual_output = cwd / actual_output
            actual_output = actual_output.resolve()
        if (compile_output != object_origin or actual_output != compile_output or
                not compile_output.is_file() or
                sha256_file(compile_output) != sha256_file(object_copy) or
                compile_output.stat().st_size != object_copy.stat().st_size):
            raise EvidenceError(f"{label} object snapshot is not bound to actual compiler output")
        depfile = _verify_raw_entry(stage, record.get("depfile"), f"{label} depfile")
        dependencies = _depfile_paths(depfile, cwd)
        recorded = record.get("dependencies")
        if not isinstance(recorded, list):
            raise EvidenceError(f"{label} depfile dependency hashes are missing")
        claimed = {}
        for item in recorded:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise EvidenceError(f"{label} dependency hash entry is malformed")
            dependency = Path(item["path"]).resolve()
            if not dependency.is_file() or not isinstance(item.get("sha256"), str):
                raise EvidenceError(f"{label} depfile header is missing or unhashed: {dependency}")
            claimed[str(dependency)] = item["sha256"]
            if sha256_file(dependency) != item["sha256"]:
                raise EvidenceError(f"{label} depfile header changed since capture: {dependency}")
        expected_paths = {str(path.resolve()) for path in dependencies}
        if expected_paths != set(claimed):
            raise EvidenceError(f"{label} raw depfile dependency set differs from evidence")
        for dependency in dependencies:
            _check_clean_dependency(str(dependency), f"{label} depfile")
        actual_includes = _include_args(argv, cwd)
        recorded_includes = record.get("include_directories")
        if recorded_includes != actual_includes:
            raise EvidenceError(f"{label} include directories differ from actual compile argv")
        for include in actual_includes:
            _check_clean_dependency(include, f"{label} compile argv include")
    target_json_path = _verify_raw_entry(stage, target.get("file_api_target"),
                                         f"{label} File API target")
    target_json = _load_json(target_json_path, f"{label} raw File API target")
    source_root = Path(str(target.get("cmake_source_root", ""))).resolve()
    if not source_root.is_dir():
        raise EvidenceError(f"{label} CMake source root is invalid")
    target_sources = _target_source_paths_from_json(target_json, source_root, sources)
    if target_sources != sources:
        raise EvidenceError(f"{label} compile source set differs from actual CMake File API target")


def _validate_closure_target(stage: Path, closure_record: dict,
                             target_record: dict, name: str,
                             expected_type: str) -> dict:
    if closure_record.get("name") != name or closure_record.get("type") != expected_type:
        raise EvidenceError(f"CMake link closure target identity/type mismatch: {name}")
    for field in ("sources", "include_directories", "link_libraries",
                  "interface_link_libraries", "dependencies"):
        values = closure_record.get(field)
        if not isinstance(values, list):
            raise EvidenceError(f"CMake link closure {name}.{field} is missing")
        for value in values:
            _check_clean_dependency(value, f"{name} {field}")
    file_api_path = _verify_raw_entry(stage, target_record.get("file_api_target"),
                                      f"{name} File API target")
    target_json = _load_json(file_api_path, f"{name} captured File API target")
    if target_json.get("name") != name or target_json.get("type") != expected_type:
        raise EvidenceError(f"{name} File API target identity/type mismatch")
    declared = {Path(value).resolve() for value in closure_record["sources"]}
    source_root = Path(str(target_record.get("cmake_source_root", ""))).resolve()
    file_sources = _target_source_paths_from_json(target_json, source_root, declared)
    if file_sources != declared:
        raise EvidenceError(f"{name} CMake link closure sources differ from File API")
    artifact_record = target_record.get("artifact")
    artifact_path = _verify_raw_entry(stage, artifact_record, f"{name} target artifact")
    declared_artifact = closure_record.get("artifact")
    if (not isinstance(declared_artifact, str) or
            Path(declared_artifact).resolve() != Path(str(artifact_record.get("origin", ""))).resolve()):
        raise EvidenceError(f"{name} manifest artifact differs from captured target output")
    artifacts = target_json.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise EvidenceError(f"{name} File API target artifact list is malformed")
    api_artifact = Path(str(artifacts[0].get("path", "")))
    if not api_artifact.is_absolute():
        build_root = Path(str(target_record.get("cmake_build_root", ""))).resolve()
        api_artifact = build_root / api_artifact
    if api_artifact.resolve() != Path(str(artifact_record["origin"])).resolve():
        raise EvidenceError(f"{name} target output differs from File API artifact")
    _validate_recorded_compile(stage, target_record, name)
    return target_json


def _validate_archive_record(stage: Path, target: dict, label: str) -> dict:
    archive = target.get("archive_provenance")
    if not isinstance(archive, dict):
        raise EvidenceError(f"{label} archive provenance is missing")
    archive_path = _verify_raw_entry(stage, archive.get("archive"), f"{label} archive")
    artifact_path = _verify_raw_entry(stage, target.get("artifact"), f"{label} target artifact")
    if sha256_file(archive_path) != sha256_file(artifact_path):
        raise EvidenceError(f"{label} archive bytes differ from CMake target artifact")
    listing_path = _verify_raw_entry(stage, archive.get("ar_t"), f"{label} raw ar t")
    members = archive.get("members")
    if not isinstance(members, list) or not members:
        raise EvidenceError(f"{label} archive object member evidence is empty")
    listed = listing_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if listed != [item.get("member") for item in members]:
        raise EvidenceError(f"{label} archive member list differs from raw ar t")
    ar = shutil.which("ar")
    if ar is None:
        raise EvidenceError("ar is required to reread captured helper archive bytes")
    observed, _ = _archive_members(ar, archive_path)
    if observed != listed:
        raise EvidenceError(f"{label} captured archive differs from its raw ar t listing")
    link_path = _verify_raw_entry(stage, archive.get("link_command"),
                                  f"{label} actual archive command")
    argv = _link_argv(link_path)
    if argv != archive.get("link_argv"):
        raise EvidenceError(f"{label} archive command argv differs from captured link.txt")
    workdir = Path(str(archive.get("working_directory", ""))).resolve()
    if not workdir.is_dir():
        raise EvidenceError(f"{label} archive command working directory is unavailable")
    object_inputs = _archive_object_inputs(workdir, archive_path, argv)
    object_hashes = {}
    for path in object_inputs:
        object_hashes.setdefault(path.name, []).append((path, sha256_file(path)))
    for item in members:
        if not isinstance(item, dict) or not isinstance(item.get("member"), str):
            raise EvidenceError(f"{label} archive member record is malformed")
        member = item["member"]
        object_copy = _verify_raw_entry(stage, item.get("object"), f"{label} object input")
        raw_member = _verify_raw_entry(stage, item.get("raw_member"), f"{label} raw archive member")
        process = _run([ar, "p", str(archive_path), member])
        digest = hashlib.sha256(process.stdout).hexdigest()
        size = len(process.stdout)
        if (process.stdout != object_copy.read_bytes() or
                process.stdout != raw_member.read_bytes() or
                item.get("sha256") != digest or type(item.get("size_bytes")) is not int or
                item.get("size_bytes") != size or
                object_copy.stat().st_size != size or raw_member.stat().st_size != size):
            raise EvidenceError(f"{label} raw archive member hash/bytes differ: {member}")
        originals = object_hashes.get(member, [])
        object_origin = Path(str(item.get("object", {}).get("origin", ""))).resolve()
        if (len(originals) != 1 or originals[0][0] != object_origin or
                originals[0][1] != digest):
            raise EvidenceError(f"{label} archived object differs from actual link input: {member}")
    if set(object_hashes) != set(listed):
        raise EvidenceError(f"{label} archive object input set differs from archive members")
    return archive


def _validate_runner_link(stage: Path, runner_link: dict,
                          runner_target: dict, helper_target: dict) -> None:
    if not isinstance(runner_link, dict):
        raise EvidenceError("VA2 standalone runner link evidence is missing")
    runner_path = _verify_raw_entry(stage, runner_link.get("runner"),
                                    "standalone runner executable")
    link_path = _verify_raw_entry(stage, runner_link.get("link_command"),
                                  "standalone runner actual link command")
    map_path = _verify_raw_entry(stage, runner_link.get("link_map"),
                                 "standalone actual linker map")
    map_text = map_path.read_text(encoding="utf-8", errors="replace")
    if "Cross Reference Table" not in map_text or "Archive member included" not in map_text:
        raise EvidenceError("VA2 actual linker map has no archive provenance")
    argv = _link_argv(link_path)
    if argv != runner_link.get("map_argv"):
        raise EvidenceError("VA2 runner map argv differs from actual CMake link.txt")
    workdir = Path(str(runner_link.get("working_directory", ""))).resolve()
    if not workdir.is_dir():
        raise EvidenceError("VA2 runner link working directory is invalid")
    map_origins = [Path(match.group(1)) for token in argv
                   for match in [re.search(r"-Map=([^,]+)", token)] if match]
    if len(map_origins) != 1:
        raise EvidenceError("VA2 actual runner link command has no unique linker map option")
    actual_map = map_origins[0]
    if not actual_map.is_absolute():
        actual_map = workdir / actual_map
    if Path(str(runner_link["link_map"].get("origin", ""))).resolve() != actual_map.resolve():
        raise EvidenceError("VA2 captured map path differs from actual linker command")
    target_artifact = _verify_raw_entry(stage, runner_target.get("artifact"),
                                        "standalone runner CMake target artifact")
    if runner_path.read_bytes() != target_artifact.read_bytes():
        raise EvidenceError("VA2 runner executable copies disagree")
    archive_rows = runner_link.get("archive_inputs")
    if not isinstance(archive_rows, list) or not archive_rows:
        raise EvidenceError("VA2 runner actual archive inputs are missing")
    resolved_map_archives = _resolve_map_archives(argv, workdir, map_text)
    expected_refs = {(str(archive_path), member)
                     for archive_path, members in resolved_map_archives.items()
                     for member in members}
    recorded_refs = set()
    ar = shutil.which("ar")
    if ar is None:
        raise EvidenceError("ar is required to reread actual runner link archives")
    helper_names = []
    helper_provenance = helper_target.get("archive_provenance", {})
    helper_origin = Path(str(helper_provenance.get("archive", {}).get("origin", ""))).resolve()
    for archive in archive_rows:
        archive_copy = _verify_raw_entry(stage, archive.get("archive"), "runner actual archive")
        listing = _verify_raw_entry(stage, archive.get("ar_t"), "runner archive raw ar t")
        actual_members, _listing = _archive_members(ar, archive_copy)
        if actual_members != archive.get("members") or listing.read_text(
                encoding="utf-8", errors="replace").splitlines() != actual_members:
            raise EvidenceError("runner archive copy differs from raw ar t member inventory")
        origin = Path(str(archive.get("origin", ""))).resolve()
        selected = archive.get("selected_members_from_map")
        hashes = archive.get("selected_member_hashes")
        if not isinstance(selected, list) or not isinstance(hashes, list) or {
                item.get("member") for item in hashes if isinstance(item, dict)} != set(selected):
            raise EvidenceError("runner archive selected-member hash inventory is incomplete")
        for member in selected:
            if not isinstance(member, str) or actual_members.count(member) != 1:
                raise EvidenceError(f"runner map selected absent member {member}")
            digest, size = model_evidence._hash_archive_member(ar, archive_copy, member)
            claim = next(item for item in hashes if item.get("member") == member)
            if (claim.get("sha256") != digest or type(claim.get("size_bytes")) is not int or
                    claim["size_bytes"] != size):
                raise EvidenceError(f"runner selected archive member hash differs: {origin}({member})")
            recorded_refs.add((str(origin), member))
        if origin == helper_origin:
            helper_names.append(set(selected))
    if recorded_refs != expected_refs:
        raise EvidenceError("runner actual linker map member set differs from archived inputs")
    if len(helper_names) != 1 or not helper_names[0]:
        raise EvidenceError("runner actual linker map does not select helper archive members")


def _assert_sanitizer_flags(argv: list[str], profile: str, label: str,
                            *, link: bool = False,
                            resource_dir: str | None = None) -> None:
    forbidden_prefixes = ("-fno-sanitize", "-fsanitize-recover")
    for token in argv:
        if token.startswith(forbidden_prefixes) and token != "-fno-sanitize-recover=all":
            raise EvidenceError(f"{label} disables/recovers sanitizer checks: {token}")
        if token.startswith(("-fsanitize-ignorelist", "-fsanitize-blacklist")):
            raise EvidenceError(f"{label} uses a custom sanitizer suppression policy")
        if token == "-fomit-frame-pointer":
            raise EvidenceError(f"{label} disables frame pointers")
        if token.startswith("@"):
            raise EvidenceError(f"{label} uses an unaudited response file")
    sanitizer_tokens = [token for token in argv if token.startswith("-fsanitize")]
    if profile in {"release", "offline", "software", "hardware", "development"}:
        if sanitizer_tokens or "-shared-libasan" in argv:
            raise EvidenceError(f"non-SAN {label} unexpectedly enables sanitizer flags")
        return
    if profile != "sanitize":
        raise EvidenceError(f"unsupported helper sanitizer profile: {profile}")
    if "-fsanitize=address,undefined" not in argv:
        raise EvidenceError(f"SAN {label} lacks exact address+undefined instrumentation")
    if "-fno-sanitize-recover=all" not in argv:
        raise EvidenceError(f"SAN {label} weakens sanitizer recovery policy")
    if "-fno-omit-frame-pointer" not in argv:
        raise EvidenceError(f"SAN {label} omits frame pointers")
    if link:
        if "-shared-libasan" not in argv:
            raise EvidenceError(f"SAN {label} does not link the shared ASan runtime")
        if not isinstance(resource_dir, str) or not resource_dir:
            raise EvidenceError("SAN Clang runtime resource directory is missing")
        expected_rpath = str(Path(resource_dir) / "lib/linux")
        if not any(expected_rpath in token and "rpath" in token for token in argv):
            raise EvidenceError(f"SAN {label} does not embed the matching Clang runtime RPATH")


def _validate_toolchain_evidence(stage: Path, profile: str, toolchain: object,
                                 targets: dict, runner_link: dict,
                                 unit_link: dict, independence_test_record: dict,
                                 product_only: dict) -> None:
    if not isinstance(toolchain, dict) or toolchain.get("profile") != profile:
        raise EvidenceError("VA2 actual helper toolchain identity is missing/stale")
    compiler = toolchain.get("compiler")
    if not isinstance(compiler, dict):
        raise EvidenceError("VA2 C++ compiler identity is malformed")
    compiler_path = Path(str(compiler.get("path", ""))).resolve()
    if not compiler_path.is_file() or sha256_file(compiler_path) != compiler.get("sha256"):
        raise EvidenceError("VA2 C++ compiler binary identity is missing/tampered")
    production_cache_path = _verify_raw_entry(
        stage, toolchain.get("production_cache"), "production CMake cache")
    standalone_cache_path = _verify_raw_entry(
        stage, toolchain.get("standalone_cache"), "standalone CMake cache")
    production, _ = _verified_compiler_cache(
        _cmake_cache_values(production_cache_path), "production")
    standalone, _ = _verified_compiler_cache(
        _cmake_cache_values(standalone_cache_path), "standalone")
    for key in ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID",
                "CMAKE_CXX_COMPILER_VERSION"):
        if (not production.get(key) or not standalone.get(key) or
                production[key] != standalone[key]):
            raise EvidenceError(f"production and standalone CMake compiler identities differ: {key}")
    if Path(production["CMAKE_CXX_COMPILER"]).resolve() != compiler_path:
        raise EvidenceError("CMake C++ compiler differs from captured compiler identity")
    if (production["CMAKE_CXX_COMPILER_ID"] != compiler.get("compiler_id") or
            production["CMAKE_CXX_COMPILER_VERSION"] != compiler.get("compiler_version")):
        raise EvidenceError("CMake C++ compiler ID/version differs from captured identity")
    version_record = compiler.get("version_output")
    version_path = _verify_raw_entry(stage, version_record, "actual compiler version output")
    version_run = _run([str(compiler_path), "--version"], check=True)
    actual_version = version_run.stdout + version_run.stderr
    if actual_version != version_path.read_bytes():
        raise EvidenceError("raw compiler version output differs from current tool")
    resource_dir = toolchain.get("resource_dir")
    if profile == "sanitize":
        if (compiler.get("compiler_id") != "Clang" or
                not str(compiler.get("compiler_version", "")).startswith("21.") or
                set(production.get("COMPILER_SANITIZERS", "").split(";")) != {
                    "address", "undefined"} or
                set(standalone.get("COMPILER_SANITIZERS", "").split(";")) != {
                    "address", "undefined"}):
            raise EvidenceError("SAN CMake identity is not matching Clang 21 address+undefined")
        for flag in ("-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                     "-fno-omit-frame-pointer"):
            if flag not in standalone.get("CMAKE_CXX_FLAGS", ""):
                raise EvidenceError(f"standalone SAN CMake CXX flags omit {flag}")
        runtime_dir = Path(resource_dir) / "lib/linux"
        for flag in ("-fsanitize=address,undefined", "-shared-libasan",
                     f"-Wl,-rpath,{runtime_dir}"):
            if flag not in standalone.get("CMAKE_EXE_LINKER_FLAGS", ""):
                raise EvidenceError(f"standalone SAN CMake linker flags omit {flag}")
        resource_record = toolchain.get("resource_dir_output")
        resource_path = _verify_raw_entry(stage, resource_record,
                                          "actual Clang resource-dir output")
        resource_run = _run([str(compiler_path), "--print-resource-dir"], check=True)
        raw_resource = resource_run.stdout + resource_run.stderr
        if raw_resource != resource_path.read_bytes():
            raise EvidenceError("Clang resource-dir raw output differs from actual compiler")
        actual_resource = raw_resource.decode("utf-8", errors="replace").strip()
        if not actual_resource or str(Path(actual_resource).resolve()) != str(Path(resource_dir).resolve()):
            raise EvidenceError("captured Clang resource directory is stale or mismatched")
        runtime_dir = Path(resource_dir) / "lib/linux"
        if not list(runtime_dir.glob("libclang_rt.asan-*.so")):
            raise EvidenceError("matching Clang shared ASan runtime disappeared")
    else:
        if resource_dir is not None or toolchain.get("resource_dir_output") is not None:
            raise EvidenceError("Release helper evidence unexpectedly claims a sanitizer runtime")
        if (production.get("COMPILER_SANITIZERS", "") or
                standalone.get("COMPILER_SANITIZERS", "") or
                "-fsanitize" in production.get("CMAKE_CXX_FLAGS", "") or
                "-fsanitize" in standalone.get("CMAKE_CXX_FLAGS", "")):
            raise EvidenceError("Release CMake cache retains sanitizer compile settings")
    for target_name in ("production_helper", "standalone_helper",
                        "standalone_runner", "standalone_unit"):
        target = targets[target_name]
        commands = target.get("compile_commands")
        if not isinstance(commands, list) or not commands:
            raise EvidenceError(f"{target_name} has no actual compiler argv")
        for item in commands:
            argv = item.get("argv") if isinstance(item, dict) else None
            if not isinstance(argv, list) or not argv:
                raise EvidenceError(f"{target_name} compiler argv is malformed")
            compiler_arg = Path(str(argv[0])).resolve()
            if compiler_arg != compiler_path:
                raise EvidenceError(f"{target_name} compile uses a different compiler: {argv[0]}")
            _assert_sanitizer_flags(argv, profile, f"{target_name} compile argv")
    _assert_sanitizer_flags(runner_link.get("map_argv", []), profile,
                            "standalone runner link argv", link=True,
                            resource_dir=resource_dir)
    _assert_sanitizer_flags(unit_link.get("map_argv", []), profile,
                            "standalone unit link argv", link=True,
                            resource_dir=resource_dir)
    linkage = toolchain.get("runtime_linkage")
    if profile == "sanitize":
        if not isinstance(linkage, list) or {item.get("target") for item in linkage
                                             if isinstance(item, dict)} != {
                "standalone_runner", "standalone_unit"}:
            raise EvidenceError("SAN standalone executable readelf evidence is incomplete")
        for item in linkage:
            raw = _verify_raw_entry(stage, item.get("readelf"),
                                    f"{item.get('target')} raw dynamic table")
            target = targets.get(item.get("target"))
            if not isinstance(target, dict):
                raise EvidenceError("SAN readelf evidence names an unknown executable target")
            binary = _verify_raw_entry(stage, target.get("artifact"),
                                       f"{item.get('target')} executable")
            command = item.get("command")
            readelf = shutil.which("readelf")
            if (readelf is None or not isinstance(command, list) or
                    command != [readelf, "-d", str(binary)]):
                raise EvidenceError("SAN readelf command identity is invalid")
            current = _run(command, check=True)
            if current.stdout + current.stderr != raw.read_bytes():
                raise EvidenceError("SAN readelf raw dynamic table differs from actual executable")
            text = raw.read_text(encoding="utf-8", errors="replace")
            actual_needed = sorted(set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", text)))
            if (actual_needed != item.get("needed") or
                    not any("libclang_rt.asan" in value for value in actual_needed) or
                    any(re.match(r"lib(?:LLVM|MLIR).*\.so", value, re.IGNORECASE)
                        for value in actual_needed)):
                raise EvidenceError(f"{item.get('target')} dynamic table contradicts SAN runtime identity")
    elif linkage != []:
        raise EvidenceError("Release helper evidence unexpectedly includes sanitizer readelf records")

    if not isinstance(independence_test_record, dict):
        raise EvidenceError("independence CTest record is missing")
    identity_entries = [entry for entry in independence_test_record.get("raw", [])
                        if isinstance(entry, dict) and
                        entry.get("role") == "compiler-toolchain-identity"]
    if len(identity_entries) != 1:
        raise EvidenceError("independence CTest raw compiler toolchain identity is missing")
    identity_path = _verify_raw_entry(stage, identity_entries[0],
                                      "independence compiler toolchain identity")
    identity = _load_json(identity_path, "independence compiler toolchain identity")
    if not isinstance(identity, dict):
        raise EvidenceError("independence compiler toolchain identity is malformed")
    if (Path(str(identity.get("compiler_path", ""))).resolve() != compiler_path or
            identity.get("compiler_version") != version_path.read_text(encoding="utf-8") or
            identity.get("build_type") != standalone.get("CMAKE_BUILD_TYPE") or
            identity.get("cmake_cxx_flags") != standalone.get("CMAKE_CXX_FLAGS", "") or
            identity.get("cmake_exe_linker_flags") != standalone.get("CMAKE_EXE_LINKER_FLAGS", "") or
            identity.get("cmake_shared_linker_flags") != standalone.get("CMAKE_SHARED_LINKER_FLAGS", "")):
        raise EvidenceError("independence raw compiler identity differs from actual standalone CMake cache/tool")
    expected_sanitizers = sorted(filter(None,
        standalone.get("COMPILER_SANITIZERS", "").split(";")))
    if identity.get("sanitizers") != expected_sanitizers:
        raise EvidenceError("independence raw sanitizer list differs from actual standalone CMake cache")
    if profile == "sanitize":
        expected_runtime = sorted((Path(resource_dir) / "lib/linux").glob(
            "libclang_rt.asan-*.so"))[0]
        if (identity.get("resource_dir") != resource_dir or
                Path(str(identity.get("shared_asan_runtime", ""))).resolve() != expected_runtime.resolve()):
            raise EvidenceError("independence Clang resource/shared-ASan identity is stale")
    elif identity.get("shared_asan_runtime") is not None:
        raise EvidenceError("Release independence record claims a shared ASan runtime")

    if not isinstance(product_only, dict):
        raise EvidenceError("product-only CMake identity is missing")
    product_cache_path = _verify_raw_entry(stage, product_only.get("cache"),
                                           "product-only CMake cache")
    product_cache, _ = _verified_compiler_cache(
        _cmake_cache_values(product_cache_path), "product-only")
    for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_EXE_LINKER_FLAGS",
                "CMAKE_SHARED_LINKER_FLAGS", "CMAKE_CXX_COMPILER_ID",
                "CMAKE_CXX_COMPILER_VERSION", "COMPILER_SANITIZERS"):
        if product_cache.get(key, "") != standalone.get(key, ""):
            raise EvidenceError(f"product-only CMake cache differs from standalone {key}")
    product_compiler_path = Path(product_cache.get("CMAKE_CXX_COMPILER", "")).resolve()
    if (product_compiler_path != compiler_path or
            identity.get("product_only_resolved_compiler_path") != str(product_compiler_path) or
            identity.get("product_only_build_type") != product_cache.get("CMAKE_BUILD_TYPE") or
            identity.get("product_only_sanitizers") != sorted(filter(
                None, product_cache.get("COMPILER_SANITIZERS", "").split(";")))):
        raise EvidenceError("product-only compiler/toolchain identity differs from actual CMake cache")


def validate_independence_evidence(evidence: os.PathLike[str] | str,
                                   stage_root: os.PathLike[str] | str | None = None,
                                   *, require_test_matrix: bool = True) -> dict:
    evidence_path = Path(evidence).resolve()
    stage = Path(stage_root).resolve() if stage_root else evidence_path.parents[4]
    data = _load_json(evidence_path, "VA2 independent helper evidence")
    state, profile, source_fingerprint, gate_state_bound = _read_stage_identity(stage)
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int or
            data.get("schema_version") != SCHEMA_VERSION or
            data.get("kind") != INDEPENDENCE_KIND):
        raise EvidenceError("independence evidence schema/kind is invalid")
    if data.get("gate_state_bound") is not gate_state_bound or data.get("profile") != profile:
        raise EvidenceError("independence evidence gate binding/profile is stale")
    if data.get("artifact_scope_only") is not True or data.get("qualified_for_va2") is not False:
        raise EvidenceError("independence artifact evidence cannot claim full VA2 qualification")
    if gate_state_bound:
        expected_stage_id = state.get("stage_id", "VA0")
        if (data.get("stage_id") != expected_stage_id or
                data.get("source_fingerprint") != source_fingerprint or
                data.get("host_execution_role") != "legacy-product" or
                data.get("host_aot_candidate") is not False or
                data.get("status") != "verified"):
            raise EvidenceError("independence evidence is stale or claims an invalid gate/host role")
    elif (data.get("stage_id") is not None or data.get("source_fingerprint") is not None or
          data.get("host_execution_role") is not None or
          data.get("host_aot_candidate") is not None or
          data.get("status") != "unqualified-development"):
        raise EvidenceError("unbound development evidence makes a gate/qualification claim")
    roots = data.get("build_roots")
    if not isinstance(roots, dict) or roots.get("production") != str(stage):
        raise EvidenceError("VA2 independence evidence build roots are malformed/stale")
    standalone = (stage / "test/Vulkan/device-helper-standalone").resolve()
    product = (stage / "test/Vulkan/device-helper-product-only").resolve()
    if roots.get("standalone") != str(standalone) or roots.get("product_only") != str(product):
        raise EvidenceError("VA2 standalone/product-only build roots are stale")
    refs = data.get("raw_artifacts")
    if not isinstance(refs, list) or not refs:
        raise EvidenceError("VA2 independence raw artifacts are missing")
    seen_refs = set()
    for item in refs:
        path = _verify_raw_entry(stage, item, "VA2 dependency/input")
        relative = str(path.relative_to(stage))
        if relative in seen_refs:
            raise EvidenceError(f"VA2 independence raw artifact path is duplicated: {relative}")
        seen_refs.add(relative)
    targets = data.get("targets")
    if not isinstance(targets, dict) or set(targets) != {
            "production_helper", "standalone_helper", "standalone_runner", "standalone_unit"}:
        raise EvidenceError("VA2 actual target evidence inventory is incomplete")
    for key, target in targets.items():
        if not isinstance(target, dict):
            raise EvidenceError(f"VA2 actual target evidence is malformed: {key}")
        _validate_recorded_compile(stage, target, key)
        _verify_raw_entry(stage, target.get("artifact"), f"{key} actual target artifact")
    manifests = data.get("closure_manifests")
    if not isinstance(manifests, dict) or set(manifests) != {"production", "standalone"}:
        raise EvidenceError("VA2 CMake closure manifest inventory is incomplete")
    closure_data = {}
    for key, path_key in (("production", "production"), ("standalone", "standalone")):
        manifest_path = _verify_raw_entry(stage, manifests[path_key],
                                          f"{key} CMake target closure")
        manifest = _load_json(manifest_path, f"{key} CMake target link closure")
        if (not _has_schema(manifest) or manifest.get("kind") != CLOSURE_KIND):
            raise EvidenceError(f"{key} CMake closure manifest schema/kind is invalid")
        manifest_targets = manifest.get("targets")
        if not isinstance(manifest_targets, list):
            raise EvidenceError(f"{key} CMake target closure has no targets")
        names = [item.get("name") for item in manifest_targets if isinstance(item, dict)]
        expected_names = ([HELPER_TARGET] if key == "production" else
                          [HELPER_TARGET, RUNNER_TARGET, UNIT_TARGET])
        if len(names) != len(set(names)) or set(names) != set(expected_names):
            raise EvidenceError(f"{key} CMake closure target inventory differs")
        closure_data[key] = {item["name"]: item for item in manifest_targets}
    production_root = stage
    standalone_root = standalone
    target_specs = (
        ("production_helper", "production", production_root, HELPER_TARGET, "STATIC_LIBRARY"),
        ("standalone_helper", "standalone", standalone_root, HELPER_TARGET, "STATIC_LIBRARY"),
        ("standalone_runner", "standalone", standalone_root, RUNNER_TARGET, "EXECUTABLE"),
        ("standalone_unit", "standalone", standalone_root, UNIT_TARGET, "EXECUTABLE"),
    )
    for key, manifest_key, build_root, name, target_type in target_specs:
        _validate_closure_target(stage, closure_data[manifest_key][name],
                                 targets[key], name, target_type)
        closure_target = closure_data[manifest_key][name]
        artifact_origin = Path(str(targets[key]["artifact"].get("origin", ""))).resolve()
        if artifact_origin.parent != build_root and not artifact_origin.is_relative_to(build_root):
            raise EvidenceError(f"{key} artifact is outside its expected CMake build root")
        if key in {"production_helper", "standalone_helper"}:
            _validate_archive_record(stage, targets[key], key)
    if (Path(str(targets["production_helper"]["artifact"].get("origin", ""))).name != EXPECTED_ARCHIVE or
            Path(str(targets["standalone_helper"]["artifact"].get("origin", ""))).name != EXPECTED_ARCHIVE):
        raise EvidenceError("VA2 production/standalone helper archive names are invalid")
    for key, target in (("production", targets["production_helper"]),
                        ("standalone", targets["standalone_helper"])):
        summary = data.get(f"{key}_archive")
        actual_hash = target.get("archive_provenance", {}).get("archive", {}).get("sha256")
        if (not isinstance(summary, dict) or summary.get("origin") !=
                target.get("artifact", {}).get("origin") or summary.get("sha256") != actual_hash):
            raise EvidenceError(f"{key} helper archive summary differs from captured archive bytes")
    _validate_runner_link(stage, data.get("runner_link"), targets["standalone_runner"],
                          targets["standalone_helper"])
    _validate_runner_link(stage, data.get("unit_link"), targets["standalone_unit"],
                          targets["standalone_helper"])
    product_only = data.get("product_only")
    if (not isinstance(product_only, dict) or
            product_only.get("registered_helper_tests") != sorted(PRODUCT_ONLY_REGISTERED_TESTS) or
            product_only.get("executed_test_names") != sorted(PRODUCT_ONLY_EXECUTE_TESTS) or
            type(product_only.get("returncode")) is not int or product_only["returncode"] != 0):
        raise EvidenceError("VA2 product-only inventory/execution evidence is incomplete")
    product_cache = _verify_raw_entry(stage, product_only.get("cache"),
                                      "product-only CMake cache")
    cache_text = product_cache.read_text(encoding="utf-8", errors="replace")
    if ("NCNN_ENABLE_VULKAN_PRODUCT:BOOL=ON" not in cache_text or
            "NCNN_ENABLE_VULKAN_V0_TESTS:BOOL=OFF" not in cache_text or
            "BUILD_TESTING:BOOL=ON" not in cache_text):
        raise EvidenceError("captured product-only CMake options do not prove product-only test registration")
    inventory_command_path = _verify_raw_entry(
        stage, product_only.get("inventory_command"), "product-only inventory command")
    inventory_command = _command_log(inventory_command_path, "product-only inventory invocation")
    inventory_path = _verify_raw_entry(stage, product_only.get("inventory"),
                                       "product-only actual inventory")
    expected_inventory_argv = ["ctest", "--test-dir", str(product), "--show-only=json-v1"]
    if (inventory_command["argv"] != expected_inventory_argv or
            inventory_command["returncode"] != 0 or inventory_command["stderr"] != "" or
            inventory_command["stdout"].encode("utf-8") != inventory_path.read_bytes()):
        raise EvidenceError("product-only actual inventory command/output binding is invalid")
    junit_record = product_only.get("junit")
    junit_path = _verify_raw_entry(stage, junit_record, "product-only actual JUnit")
    junit_origin = Path(str(junit_record.get("origin", ""))).resolve()
    expected_ctest_argv = ["ctest", "--test-dir", str(product), "--output-on-failure",
                           "--output-junit", str(junit_origin), "-R",
                           "^(" + "|".join(re.escape(name) for name in PRODUCT_ONLY_EXECUTE_TESTS) + ")$"]
    ctest_command_path = _verify_raw_entry(
        stage, product_only.get("ctest_command"), "product-only execution command")
    ctest_command = _command_log(ctest_command_path, "product-only CTest invocation")
    stdout_record = next((item for item in data.get("raw_artifacts", [])
                          if isinstance(item, dict) and item.get("role") ==
                          "product-only CTest stdout"), None)
    stderr_record = next((item for item in data.get("raw_artifacts", [])
                          if isinstance(item, dict) and item.get("role") ==
                          "product-only CTest stderr"), None)
    stdout_path = _verify_raw_entry(stage, stdout_record, "product-only CTest stdout")
    stderr_path = _verify_raw_entry(stage, stderr_record, "product-only CTest stderr")
    if (ctest_command["argv"] != expected_ctest_argv or
            type(ctest_command["returncode"]) is not int or
            ctest_command["returncode"] != product_only.get("returncode") or
            ctest_command["stdout"].encode("utf-8") != stdout_path.read_bytes() or
            ctest_command["stderr"].encode("utf-8") != stderr_path.read_bytes()):
        raise EvidenceError("product-only actual CTest command/result/output binding is invalid")
    for field in ("inventory", "junit", "lasttest"):
        path = _verify_raw_entry(stage, product_only.get(field), f"product-only {field}")
        if field == "inventory":
            inventory = _load_json(path, "product-only raw CTest inventory")
            rows = inventory.get("tests") if isinstance(inventory, dict) else None
            if (not isinstance(rows, list) or
                    any(not isinstance(entry, dict) or type(entry.get("name")) is not str or
                        not entry["name"] for entry in rows)):
                raise EvidenceError("product-only raw CTest inventory contains malformed rows")
            names = {entry["name"] for entry in rows}
            if not set(PRODUCT_ONLY_REGISTERED_TESTS).issubset(names) or set(DEVICE_TESTS) & names:
                raise EvidenceError("product-only raw CTest inventory is missing or device-polluted")
        elif field == "junit":
            cases = list(ET.parse(path).getroot().iter("testcase"))
            if ({case.get("name") for case in cases} != set(PRODUCT_ONLY_EXECUTE_TESTS) or
                    len(cases) != len(PRODUCT_ONLY_EXECUTE_TESTS) or
                    any(case.find("skipped") is not None or case.find("failure") is not None or
                        case.find("error") is not None for case in cases)):
                raise EvidenceError("product-only raw JUnit is missing, skipped, or failed")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            names = re.findall(r"^\d+/\d+ Testing: (.+)$", text, re.MULTILINE)
            if (set(names) != set(PRODUCT_ONLY_EXECUTE_TESTS) or
                    len(names) != len(PRODUCT_ONLY_EXECUTE_TESTS) or
                    any(token in text for token in ("Test Failed.", "Test Not Run.", "Test Skipped."))):
                raise EvidenceError("product-only raw LastTest results are incomplete")
    child_ids = (TEST_NAMES[0], TEST_NAMES[1], TEST_NAMES[3])
    child_refs: list[dict[str, object]] = []
    child_tests = _audit_test_records(product, child_refs, child_ids)
    child_claims = product_only.get("child_helper_test_records")
    if not isinstance(child_claims, dict) or set(child_claims) != set(child_ids):
        raise EvidenceError("product-only child helper raw test record inventory is incomplete")
    for identity in child_ids:
        claim = child_claims[identity]
        if (not isinstance(claim, dict) or
                claim.get("record") != child_tests[identity]["record"] or
                claim.get("raw_roles") != child_tests[identity]["raw_roles"]):
            raise EvidenceError(f"product-only child test record differs from its raw command: {identity}")
        for item in child_refs:
            child_path = product / str(item["path"])
            if str(child_path.relative_to(stage)) not in seen_refs:
                raise EvidenceError("product-only helper raw test bytes are absent from outer archive refs")
    test_refs: list[dict[str, object]] = []
    requested_tests = TEST_NAMES if require_test_matrix else (
        "vulkan-device-helper-independence",)
    test_records = _audit_test_records(stage, test_refs, requested_tests)
    independence_record = test_records["vulkan-device-helper-independence"]["record"]
    independence_results = [entry for entry in independence_record.get("raw", [])
                            if isinstance(entry, dict) and entry.get("role") == "result"]
    if len(independence_results) != 1:
        raise EvidenceError("independence test record has no unique actual result output")
    independence_result = _load_json(
        _safe_path(stage, independence_results[0].get("path")),
        "actual independence CTest result")
    child_inventory_value = independence_result.get("product_only_inventory")
    if not isinstance(child_inventory_value, str) or not child_inventory_value:
        raise EvidenceError("independence result omits product-only actual inventory path")
    child_inventory_path = _safe_path(stage, child_inventory_value)
    audited_inventory_path = _verify_raw_entry(
        stage, product_only.get("inventory"), "product-only audited inventory")
    if (independence_result.get("product_only_check_performed") is not True or
            independence_result.get("product_only_registered_tests") !=
            product_only.get("registered_helper_tests") or
            independence_result.get("product_only_executed_tests") !=
            product_only.get("executed_test_names") or
            child_inventory_path.read_bytes() != audited_inventory_path.read_bytes()):
        raise EvidenceError("independence result self-report differs from actual product-only CTest evidence")
    if require_test_matrix and (
            not gate_state_bound or state is None or
            state.get("stage_id", "VA0") != "VA2" or
            profile not in {"release", "sanitize"}):
        raise EvidenceError("full VA2 test matrix requires a bound VA2 Release/SAN gate")
    _validate_toolchain_evidence(
        stage, profile, data.get("toolchain"), targets,
        data.get("runner_link"), data.get("unit_link"),
        test_records["vulkan-device-helper-independence"]["record"], product_only)
    return data


def _capture_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--standalone-build-root", required=True, type=Path)
    parser.add_argument("--product-only-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evidence = capture_independence_evidence(
        args.stage_root, args.standalone_build_root, args.product_only_root,
        args.output)
    print(f"VA2 helper independence evidence: {args.output} "
          f"({len(evidence['raw_artifacts'])} raw artifacts)")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _capture_cli(argv)
    except (EvidenceError, OSError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError, ET.ParseError) as error:
        print(f"device_helper_evidence.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
