#!/usr/bin/env python3
"""Fail-closed audit for fresh Vulkan gate stages and their evidence bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import host_program_evidence as host_program_audit
import model_dependency_evidence as dependency_evidence
import device_helper_evidence as device_helper_audit
import lifecycle_evidence as lifecycle_audit
import fixedrun_evidence as fixedrun_evidence
import run_va5_deployment as va5_deployment


SCHEMA_VERSION = 1
GATE_KIND = "ncnn.vulkan_gate"
REPORT_KIND = "ncnn.vulkan_release_evidence"
PROFILES = ("release", "offline", "software", "sanitize", "hardware")
STEP_NAMES = ("configure", "format_check", "build", "numerical", "tidy", "ctest")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SANITIZER_RE = re.compile(
    r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:", re.I
)

# Existing CMake-discovered performance tests. Kept explicit so a newly added
# test cannot silently become part of the sanitizer skip allowance.
SANITIZER_PERFORMANCE_SKIPS = frozenset((
    "PerformanceModel.SqueezeNetV11", "PerformanceModel.ResNet18",
    "PerformanceModel.ResNet18Winograd", "PerformanceModel.ResNet34",
    "PerformanceModel.ResNet50", "PerformanceModel.ResNet101",
    "PerformanceModel.Yolov5nCls", "PerformanceModel.Yolov5sCls",
    "PerformanceModel.Yolov5mCls", "PerformanceModel.Yolov5lCls",
    "PerformanceModel.Yolov5xCls", "PerformanceModel.Yolov5nDet",
    "PerformanceModel.Yolov5sDet", "PerformanceModel.Yolov5mDet",
    "PerformanceModel.Yolov5lDet", "PerformanceModel.Yolov5xDet",
    "PerformanceModel.EfficientNetB0", "PerformanceModel.EfficientNetB1",
    "PerformanceModel.EfficientNetB2", "PerformanceModel.EfficientNetB3",
    "PerformanceModel.PPLCNetDocOri", "PerformanceModel.PPLCNetTextlineOri",
    "PerformanceModel.ChineseOCRLiteAngleNet", "PerformanceModel.Yolov5nSeg",
    "PerformanceModel.Yolov5sSeg", "PerformanceModel.Yolov5mSeg",
    "PerformanceModel.Yolov5lSeg", "PerformanceModel.Yolov5xSeg",
    "PerformanceModel.PPLCNetDocOriInt8", "PerformanceModel.PPLCNetTextlineOriInt8",
    "PerformanceModel.PPOcrv6TinyRec", "PerformanceModel.PPOcrv6TinyRecInt8",
    "PerformanceModel.PPOcrv5MobileRec", "PerformanceModel.PPOcrv5MobileRecInt8",
    "PerformanceModel.PPOcrv5ServerRec", "PerformanceModel.PPOcrv6MediumRec",
    "PerformanceModel.PPOcrv6MediumRecInt8", "PerformanceModel.PPOcrv6SmallRec",
    "PerformanceModel.PPOcrv6SmallRecInt8", "PerformanceModel.PPOcrv6TinyDet",
    "PerformanceModel.PPOcrv6SmallDet", "PerformanceModel.PPOcrv6MediumDet",
    "PerformanceModel.PPOcrv6MediumDetInt8", "PerformanceModel.PPOcrv5MobileDetStatic",
    "PerformanceModel.PPOcrv5ServerDetStatic",
    "PerformanceModel.PPStructrureV2SlanetPlusCnn",
    "PerformanceModel.PPFormulaNetPlusSEncoder",
))
RELEASE_PERFORMANCE_SKIPS = frozenset((
    "PerformanceModel.PPLCNetDocOriInt8", "PerformanceModel.PPLCNetTextlineOriInt8"
))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def safe_stage_file(stage: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    candidate = (stage / relative).resolve()
    try:
        candidate.relative_to(stage.resolve())
    except ValueError:
        return None
    return candidate


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fingerprint_function():
    path = Path(__file__).with_name("run_gate.py")
    if not path.is_file():
        raise RuntimeError(f"gate fingerprint implementation is missing: {path}")
    spec = importlib.util.spec_from_file_location("ncnn_vulkan_run_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import run_gate.py fingerprint implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "source_fingerprint", None)
    if not callable(function):
        raise RuntimeError("run_gate.py has no source_fingerprint(Path) function")
    return function


def _finding(findings: list[dict], code: str, message: str, **details) -> None:
    item = {"code": code, "message": message}
    if details:
        item["details"] = details
    findings.append(item)


def _verify_hash_record(stage: Path, record: object, label: str,
                        findings: list[dict]) -> Path | None:
    if not isinstance(record, dict):
        _finding(findings, "missing-artifact-record", f"{label} record is missing")
        return None
    path = safe_stage_file(stage, record.get("path"))
    digest = record.get("sha256")
    if path is None:
        _finding(findings, "unsafe-artifact-path", f"{label} path is not stage-relative")
        return None
    if not path.is_file():
        _finding(findings, "missing-artifact", f"{label} file is missing", path=record.get("path"))
        return None
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        _finding(findings, "invalid-artifact-hash", f"{label} SHA-256 is invalid")
    elif sha256_file(path) != digest:
        _finding(findings, "tampered-artifact", f"{label} SHA-256 does not match", path=record.get("path"))
    return path


def _parse_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        values[key_and_type.split(":", 1)[0]] = value
    return values


def _verify_config(profile: str, state: dict, cache_path: Path,
                   findings: list[dict]) -> None:
    argv = state.get("configuration")
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        _finding(findings, "configuration-missing", "complete configure argv is required")
        return
    source_index = argv.index("-S") + 1 if "-S" in argv and argv.index("-S") + 1 < len(argv) else -1
    source_args = [arg[2:] for arg in argv if arg.startswith("-S") and len(arg) > 2]
    if not ((source_index >= 0 and "compiler" in argv[source_index])
            or any("compiler" in arg for arg in source_args)):
        _finding(findings, "configuration-source", "configure argv does not identify compiler source")
    build_index = argv.index("-B") + 1 if "-B" in argv and argv.index("-B") + 1 < len(argv) else -1
    build_args = [arg[2:] for arg in argv if arg.startswith("-B") and len(arg) > 2]
    configured_build = ([argv[build_index]] if build_index >= 0 else []) + build_args
    if not any(Path(arg).resolve() == Path(state["build_dir"]).resolve() for arg in configured_build):
        _finding(findings, "configuration-build-dir", "configure argv build directory differs from stage")
    if not any(arg == "-DBUILD_TESTING=ON" for arg in argv):
        _finding(findings, "configuration-testing", "BUILD_TESTING=ON is absent")
    if not any(arg == "-DCOMPILER_INSTALL_RPATH=ON" for arg in argv):
        _finding(findings, "configuration-rpath", "COMPILER_INSTALL_RPATH=ON is absent")
    cache = _parse_cmake_cache(cache_path)
    expected_v0 = "OFF" if profile == "offline" else "ON"
    checks = {
        "NCNN_ENABLE_VULKAN_PRODUCT": "ON",
        "NCNN_ENABLE_VULKAN_V0_TESTS": expected_v0,
        "BUILD_TESTING": "ON",
        "COMPILER_INSTALL_RPATH": "ON",
    }
    for key, expected in checks.items():
        if cache.get(key) != expected:
            _finding(findings, "cache-configuration", f"{key} must be {expected}", actual=cache.get(key))
    build_type = "Debug" if profile == "sanitize" else "Release"
    if cache.get("CMAKE_BUILD_TYPE") != build_type:
        _finding(findings, "cache-build-type", f"{profile} requires CMAKE_BUILD_TYPE={build_type}")
    devices = cache.get("NCNN_VULKAN_TEST_DEVICES", "")
    expected_devices = {"offline": "nvidia;llvmpipe", "software": "llvmpipe",
                        "release": "nvidia;llvmpipe", "sanitize": "nvidia;llvmpipe",
                        "hardware": "nvidia"}[profile]
    if devices != expected_devices:
        _finding(findings, "cache-device-matrix", "configured Vulkan test devices differ from profile",
                 expected=expected_devices, actual=devices)
    if profile == "sanitize":
        flags = " ".join(cache.get(key, "") for key in ("COMPILER_SANITIZERS", "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS",
                                                          "CMAKE_C_FLAGS_DEBUG", "CMAKE_CXX_FLAGS_DEBUG"))
        if "address" not in flags or "undefined" not in flags:
            _finding(findings, "sanitizer-config", "sanitize profile lacks address+undefined instrumentation")


SANITIZER_BUDGET_LIMITS = {
    "pp_ocrv5_server_det_static": 450,
    "pp_ocrv5_server_rec": 450,
}


def _audit_sanitizer_budget_evidence(stage: Path, state: dict,
                                     cache_path: Path | None,
                                     findings: list[dict]) -> None:
    profile = state.get("profile")
    proofs = state.get("sanitizer_budget_evidence")
    if not isinstance(proofs, list):
        _finding(findings, "budget-evidence-state", "sanitizer_budget_evidence must be a list")
        return
    config = state.get("configuration", [])
    overrides = {}
    for argument in config:
        if not isinstance(argument, str) or not argument.startswith("-D") or "=" not in argument:
            continue
        key, value = argument[2:].split("=", 1)
        if key.startswith("ncnn_model_compile_budget_"):
            fixture = key.removeprefix("ncnn_model_compile_budget_")
            if fixture in overrides:
                _finding(findings, "budget-evidence-duplicate-override", "SAN compile budget override is duplicated", fixture=fixture)
            overrides[fixture] = value
    if profile != "sanitize":
        if proofs or overrides:
            _finding(findings, "budget-evidence-profile", "compile budget overrides/evidence are SAN-only")
        return
    accepted = {}
    for proof in proofs:
        if not isinstance(proof, dict):
            _finding(findings, "budget-evidence-record", "sanitizer budget proof record is malformed")
            continue
        fixture = proof.get("fixture")
        if fixture not in SANITIZER_BUDGET_LIMITS or fixture in accepted:
            _finding(findings, "budget-evidence-fixture", "sanitizer budget proof fixture is outside the two-item allowlist",
                     fixture=fixture)
            continue
        cap = SANITIZER_BUDGET_LIMITS[fixture]
        if proof.get("requested_budget_seconds") != cap:
            _finding(findings, "budget-evidence-cap", "sanitizer budget exception is not exactly 450 seconds",
                     fixture=fixture, actual=proof.get("requested_budget_seconds"))
        record_path = _verify_hash_record(stage, proof.get("record"),
                                          f"SAN budget evidence {fixture}", findings)
        log_path = _verify_hash_record(stage, proof.get("log"),
                                       f"SAN raw timed_compile log {fixture}", findings)
        elapsed = proof.get("elapsed_seconds")
        if not isinstance(elapsed, int) or elapsed <= 300:
            _finding(findings, "budget-evidence-elapsed", "exception lacks a measured compile failure over the 300 second initial limit",
                     fixture=fixture, elapsed=elapsed)
        if record_path:
            try:
                record = load_json(record_path)
                if (record.get("schema_version") != 1 or
                        record.get("kind") != "ncnn.sanitizer_budget_failure" or
                        record.get("source_fingerprint") != state.get("source_fingerprint") or
                        record.get("profile") != "sanitize" or
                        record.get("fixture") != fixture or
                        record.get("previous_budget_seconds") != 300 or
                        record.get("requested_budget_seconds") != cap or
                        record.get("elapsed_seconds") != elapsed):
                    _finding(findings, "budget-evidence-record-mismatch", f"SAN budget record is not tied to this source/gate/fixture: {fixture}")
                if log_path:
                    if record_path.parent != log_path.parent or Path(str(record.get("log_file", ""))).name != log_path.name:
                        _finding(findings, "budget-evidence-log-path", f"SAN proof log is not the copied sibling timed_compile.cmake file: {fixture}")
                    if sha256_file(log_path) != record.get("log_sha256"):
                        _finding(findings, "budget-evidence-log-mismatch", f"SAN budget record raw log hash differs: {fixture}")
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    # CMake wraps FATAL_ERROR messages at a fixed column with
                    # a two-space continuation, which can split "超过硬预算
                    # 300s" across lines in the raw build log.  Accept both
                    # display forms; the elapsed seconds must still match.
                    pattern = rf"fixture {re.escape(fixture)} 编译耗时 (\d+)s 超过硬预算\s+300s"
                    match = re.search(pattern, text)
                    if match is None or int(match.group(1)) != elapsed:
                        _finding(findings, "budget-evidence-failure-cause", f"SAN raw log does not prove the original 300-second budget failure: {fixture}")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                _finding(findings, "budget-evidence-invalid", f"SAN budget proof is invalid for {fixture}: {error}")
        accepted[fixture] = proof
    if set(overrides) != set(accepted):
        _finding(findings, "budget-evidence-override-mismatch",
                 "sanitizer CMake budget overrides and copied proof fixtures differ",
                 overrides=sorted(overrides), proofs=sorted(accepted))
    if cache_path is not None and cache_path.is_file():
        cache = _parse_cmake_cache(cache_path)
        cached_budgets = {key.removeprefix("ncnn_model_compile_budget_"): value
                          for key, value in cache.items()
                          if key.startswith("ncnn_model_compile_budget_")}
        for fixture, value in cached_budgets.items():
            if fixture not in overrides or value != overrides[fixture]:
                _finding(findings, "budget-evidence-cache-mismatch",
                         "CMakeCache contains a compile budget not accounted for by state/configuration",
                         fixture=fixture, cache=value, argv=overrides.get(fixture))
    for fixture, text_budget in overrides.items():
        if fixture not in SANITIZER_BUDGET_LIMITS or text_budget != str(SANITIZER_BUDGET_LIMITS.get(fixture)):
            _finding(findings, "budget-evidence-unauthorized-override",
                     "only the two exact 450 second PP sanitizer exceptions are authorized",
                     fixture=fixture, budget=text_budget)


def _audit_sanitizer_toolchain(stage: Path, state: dict, cache_path: Path | None,
                               refs: list[dict], findings: list[dict]) -> list[dict]:
    profile = state.get("profile")
    evidence = state.get("sanitizer_toolchain_evidence")
    if profile != "sanitize":
        if evidence:
            _finding(findings, "sanitizer-toolchain-profile", "instrumented toolchain evidence is SAN-only")
        return []
    if not isinstance(evidence, dict):
        _finding(findings, "sanitizer-toolchain-missing", "SAN gate lacks sanitizer_toolchain_evidence")
        return []
    identity_copy = _verify_hash_record(stage, evidence.get("identity"),
                                       "copied sanitizer toolchain identity", findings)
    identity = {}
    if identity_copy:
        try:
            identity = load_json(identity_copy)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "sanitizer-toolchain-identity-invalid", str(error))
    identity_schema = identity.get("schema") if isinstance(identity, dict) else None
    if (not isinstance(identity, dict) or identity_schema not in {
            "ncnn-san-toolchain-identity/v1", "ncnn-san-toolchain-identity/v2"} or
            identity.get("status") != "verified"):
        _finding(findings, "sanitizer-toolchain-identity-status", "copied toolchain identity is not verified")
        identity = identity if isinstance(identity, dict) else {}
    is_v2 = identity_schema == "ncnn-san-toolchain-identity/v2"
    llvm = identity.get("llvm", {})
    inventory_result = None
    source_to_bundle = {}
    if is_v2:
        from sanitizer_toolchain_inventory import InventoryError, validate_inventory
        inventory_record = llvm.get("inventory", {})
        inputs = inventory_record.get("inputs", {}) if isinstance(inventory_record, dict) else {}
        required_inputs = ("compile_commands", "cmake_cache", "llvm_config", "mlir_config",
                           "cmake_exports", "install_manifest", "source_archive", "applied_patches")
        if not isinstance(inputs, dict) or any(key not in inputs for key in required_inputs):
            _finding(findings, "sanitizer-inventory-inputs", "v2 SAN toolchain inventory inputs are incomplete")
        else:
            bundle_records = evidence.get("files", [])
            if isinstance(bundle_records, list):
                for item in bundle_records:
                    if not isinstance(item, dict) or not isinstance(item.get("source"), str):
                        continue
                    target = safe_stage_file(stage, item.get("path"))
                    source = Path(item["source"]).resolve()
                    expected = item.get("source_sha256")
                    if (target is not None and target.is_file() and
                            isinstance(expected, str) and SHA256_RE.fullmatch(expected) and
                            sha256_file(target) == expected):
                        source_to_bundle[source] = target
            install_prefix_for_resolver = Path(str(llvm.get("install_prefix", ""))).resolve()
            staged_install_prefix = stage / "toolchain-evidence/llvm-install"
            def inventory_resolver(source_path):
                source_path = Path(source_path).resolve()
                try:
                    return staged_install_prefix / source_path.relative_to(install_prefix_for_resolver)
                except ValueError:
                    target = source_to_bundle.get(source_path)
                    if target is None:
                        raise InventoryError(f"v2 bundle omits source-bound input: {source_path}")
                    return target
            def staged_input(value, label):
                source = Path(value).resolve()
                target = source_to_bundle.get(source)
                if target is None:
                    raise InventoryError(f"v2 bundle omits {label}: {source}")
                return target
            source_archive_input = inputs.get("source_archive")
            try:
                if not isinstance(source_archive_input, dict):
                    raise InventoryError("source_archive input is malformed")
                inventory_result = validate_inventory(
                    staged_input(inputs["compile_commands"], "compile_commands.json"),
                    staged_input(inputs["cmake_cache"], "CMakeCache.txt"),
                    staged_input(inputs["llvm_config"], "LLVMConfig.cmake"),
                    staged_input(inputs["mlir_config"], "MLIRConfig.cmake"),
                    [staged_input(path, "CMake export") for path in inputs["cmake_exports"]],
                    staged_input(inputs["install_manifest"], "install manifest"),
                    source_archive_path=staged_input(source_archive_input.get("path", ""), "LLVM source archive"),
                    source_archive_sha256=source_archive_input.get("sha256", ""),
                    resolver=inventory_resolver,
                    applied_patch_records=inputs["applied_patches"],
                    expected_identity_sha256=inventory_record.get("identity_sha256"),
                )
            except (InventoryError, OSError, TypeError, ValueError, KeyError) as error:
                _finding(findings, "sanitizer-inventory-invalid", f"v2 SAN inventory/bundle validation failed: {error}")
            if inventory_result is not None:
                actual_inventory_bundle = {}
                for item in bundle_records if isinstance(bundle_records, list) else ():
                    if not isinstance(item, dict) or not isinstance(item.get("source"), str):
                        continue
                    target = safe_stage_file(stage, item.get("path"))
                    if target is not None:
                        actual_inventory_bundle[Path(item["source"]).resolve()] = (
                            target.resolve(), item.get("source_sha256"))
                for required_file in inventory_result["bundle_files"]:
                    source_path = Path(required_file["source"]).resolve()
                    actual = actual_inventory_bundle.get(source_path)
                    expected_target = Path(required_file["path"]).resolve()
                    if actual != (expected_target, required_file["source_sha256"]):
                        _finding(findings, "sanitizer-inventory-bundle-missing",
                                 "v2 content-bound archive/response input is absent or misbound in final bundle",
                                 source=str(source_path), expected_path=str(expected_target))
    def staged_evidence_file(path):
        source = Path(str(path)).resolve()
        if not is_v2:
            return source
        return source_to_bundle.get(
            source, stage / "toolchain-evidence/__missing_evidence__" / source.name)

    configuration = llvm.get("configuration", {})
    if (evidence.get("llvm_version") != "21.1.8" or llvm.get("version") != "21.1.8" or
            llvm.get("shared_mode") != "static" or
            configuration.get("BUILD_SHARED_LIBS") is not False or
            configuration.get("LLVM_BUILD_LLVM_DYLIB") is not False or
            configuration.get("LLVM_BUILD_LLVM_C_DYLIB") is not False or
            configuration.get("LLVM_LINK_LLVM_DYLIB") is not False or
            configuration.get("MLIR_BUILD_MLIR_C_DYLIB") is not False or
            configuration.get("sanitizers") != ["address", "undefined"] or
            configuration.get("no_sanitize_recover") is not True or
            "-fPIC" not in str(configuration.get("flags", ""))):
        _finding(findings, "sanitizer-toolchain-config", "copied LLVM/MLIR identity violates static LLVM 21.1.8 PIC ASan+UBSan contract")
    archives = llvm.get("static_archive_artifacts", [])
    archive_count = evidence.get("static_archive_count")
    inventory_record = llvm.get("inventory", {}) if is_v2 else {}
    expected_archive_count = (
        len(inventory_result["inventory"]["static_archives"])
        if is_v2 and inventory_result is not None else
        len(archives) if is_v2 and isinstance(archives, list) else 486
    )
    if is_v2 and (not isinstance(inventory_record, dict) or
                  not SHA256_RE.fullmatch(str(inventory_record.get("identity_sha256", "")))):
        _finding(findings, "sanitizer-inventory-identity", "v2 SAN toolchain lacks canonical inventory identity")
    archive_paths = []
    if (archive_count != expected_archive_count or
            llvm.get("static_archive_count") != expected_archive_count or
            not isinstance(archives, list) or len(archives) != expected_archive_count):
        message = ("SAN toolchain archive count differs from the content-bound inventory"
                   if is_v2 else "SAN toolchain must identify all 486 static LLVM/MLIR archives")
        _finding(findings, "sanitizer-archive-count", message)
    else:
        for record in archives:
            if (not isinstance(record, dict) or not isinstance(record.get("path"), str) or
                    not SHA256_RE.fullmatch(str(record.get("sha256", "")))):
                _finding(findings, "sanitizer-archive-identity", "static archive identity record is malformed")
                continue
            archive_path = Path(record["path"])
            archive_paths.append(str(archive_path.resolve()))
            audited_archive = staged_evidence_file(archive_path)
            if not audited_archive.is_file():
                _finding(findings, "sanitizer-archive-missing", "staged static LLVM/MLIR archive is missing",
                         path=record["path"])
            elif sha256_file(audited_archive) != record["sha256"]:
                _finding(findings, "sanitizer-archive-hash", "staged static LLVM/MLIR archive content changed after preflight",
                         path=record["path"])
        if len(set(archive_paths)) != len(archive_paths):
            _finding(findings, "sanitizer-archive-duplicate", "static archive manifest repeats an archive path")
    archive_list = llvm.get("static_archive_list", {})
    archive_list_source = Path(str(archive_list.get("path", ""))) if isinstance(archive_list, dict) else Path()
    archive_list_path = staged_evidence_file(archive_list_source)
    archive_list_hash = archive_list.get("sha256") if isinstance(archive_list, dict) else None
    listed_archives = []
    if (not archive_list_path.is_file() or not SHA256_RE.fullmatch(str(archive_list_hash or "")) or
            (archive_list_path.is_file() and sha256_file(archive_list_path) != archive_list_hash)):
        _finding(findings, "sanitizer-archive-list", "SAN archive manifest path/hash is missing or changed")
    else:
        try:
            listed_archives = [str(Path(line.strip()).resolve()) for line in
                               archive_list_path.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
            if (len(listed_archives) != expected_archive_count or
                    len(set(listed_archives)) != expected_archive_count):
                message = ("SAN static archive list must contain the exact content-bound archive set"
                           if is_v2 else "SAN static archive list must contain 486 unique paths")
                _finding(findings, "sanitizer-archive-list-count", message)
            if archive_paths and sorted(listed_archives) != sorted(archive_paths):
                _finding(findings, "sanitizer-archive-list-mismatch",
                         "SAN archive list path multiset differs from static archive identities")
        except (OSError, UnicodeError, ValueError) as error:
            _finding(findings, "sanitizer-archive-list-invalid", f"SAN archive list is invalid: {error}")
    source_archive = identity.get("source_archive_identity", {})
    source_archive_path = Path(str(source_archive.get("path", ""))) if isinstance(source_archive, dict) else Path()
    source_archive_hash = source_archive.get("required_sha256") if isinstance(source_archive, dict) else None
    if is_v2:
        staged_source_archive = source_to_bundle.get(source_archive_path.resolve())
        if (not SHA256_RE.fullmatch(str(source_archive_hash or "")) or
                staged_source_archive is None or not staged_source_archive.is_file() or
                sha256_file(staged_source_archive) != source_archive_hash):
            _finding(findings, "sanitizer-source-archive",
                     "SAN staged LLVM source archive path/hash is missing or changed")
    elif (not source_archive_path.is_file() or not SHA256_RE.fullmatch(str(source_archive_hash or "")) or
          sha256_file(source_archive_path) != source_archive_hash):
        _finding(findings, "sanitizer-source-archive", "SAN LLVM source archive path/hash is missing or changed")
    if is_v2:
        inventory_inputs = inventory_record.get("inputs", {}) if isinstance(inventory_record, dict) else {}
        source_input = inventory_inputs.get("source_archive", {}) if isinstance(inventory_inputs, dict) else {}
        if (not isinstance(source_input, dict) or
                Path(str(source_input.get("path", ""))).resolve() != source_archive_path.resolve() or
                source_input.get("sha256") != source_archive_hash or
                evidence.get("inventory_identity_sha256") != inventory_record.get("identity_sha256")):
            _finding(findings, "sanitizer-inventory-source-binding",
                     "v2 source archive or returned inventory identity differs from verified identity")
    tool_artifacts = identity.get("tool_artifacts", {})
    if not isinstance(tool_artifacts, dict) or not tool_artifacts:
        _finding(findings, "sanitizer-tool-artifacts", "SAN verified tool binary identities are missing")
    else:
        for role, tool_record in tool_artifacts.items():
            tool_source = Path(str(tool_record.get("path", ""))) if isinstance(tool_record, dict) else Path()
            tool_path = staged_evidence_file(tool_source)
            tool_hash = tool_record.get("sha256") if isinstance(tool_record, dict) else None
            if (not tool_path.is_file() or not SHA256_RE.fullmatch(str(tool_hash or "")) or
                    (tool_path.is_file() and sha256_file(tool_path) != tool_hash)):
                _finding(findings, "sanitizer-tool-artifact", f"SAN tool artifact path/hash is invalid: {role}")
    for smoke_key in ("gpu_runner_pipeline_fixture", "gpu_runner_pipeline_output"):
        smoke_record = identity.get("smokes", {}).get(smoke_key, {})
        smoke_source = Path(str(smoke_record.get("path", ""))) if isinstance(smoke_record, dict) else Path()
        smoke_path = staged_evidence_file(smoke_source)
        if not smoke_path.is_file() or not isinstance(smoke_record, dict) or not SHA256_RE.fullmatch(str(smoke_record.get("sha256", ""))):
            _finding(findings, "sanitizer-gpu-smoke", f"SAN GPU runner smoke identity is missing: {smoke_key}")
        elif sha256_file(smoke_path) != smoke_record.get("sha256"):
            _finding(findings, "sanitizer-gpu-smoke-hash", f"SAN GPU runner smoke identity changed: {smoke_key}")
    libomp = identity.get("libomp", {})
    libomp_shared = libomp.get("shared_library", {})
    libomp_hash = libomp_shared.get("sha256")
    libomp_source_path = Path(str(libomp_shared.get("path", "")))
    libomp_path = staged_evidence_file(libomp_source_path)
    if (evidence.get("libomp_sha256") != libomp_hash or not SHA256_RE.fullmatch(str(libomp_hash or ""))
            or not libomp_path.is_file() or (libomp_path.is_file() and sha256_file(libomp_path) != libomp_hash)):
        _finding(findings, "sanitizer-libomp-identity", "staged libomp SHA differs from verified identity")
    if libomp.get("static_library") is not None or libomp.get("libgomp_aliases") != []:
        _finding(findings, "sanitizer-libomp-alias", "SAN libomp identity includes a static/foreign OpenMP alias")
    smoke = libomp.get("lsan_smoke", {})
    smoke_binary = staged_evidence_file(smoke.get("binary", ""))
    smoke_log = staged_evidence_file(smoke.get("log", ""))
    if (smoke.get("leak_detection") is not True or smoke.get("suppressions") is not False or
            not smoke_binary.is_file() or not smoke_log.is_file()):
        _finding(findings, "sanitizer-lsan-smoke", "SAN libomp smoke did not prove unsuppressed leak detection")
    elif SANITIZER_RE.search(smoke_log.read_text(encoding="utf-8", errors="replace")):
        _finding(findings, "sanitizer-lsan-diagnostic", "SAN libomp leak smoke raw log contains sanitizer/runtime diagnostics")
    smokes = identity.get("smokes", {})
    verification_text = evidence.get("verification_status")
    verification_source = staged_evidence_file(smokes.get("verification_status", ""))
    if (not isinstance(verification_text, str) or "verification=passed" not in verification_text or
            "exit=0" not in verification_text or not verification_source.is_file() or
            verification_source.read_text(encoding="utf-8", errors="replace") != verification_text):
        _finding(findings, "sanitizer-verification-status", "SAN verifier status is not a recorded successful smoke")
    if cache_path and cache_path.is_file():
        cache = _parse_cmake_cache(cache_path)
        prefix = Path(llvm.get("install_prefix", "")).resolve()
        for key in ("LLVM_DIR", "MLIR_DIR"):
            value = cache.get(key)
            if not value or prefix not in Path(value).resolve().parents:
                _finding(findings, "sanitizer-cmake-prefix", f"SAN {key} does not use the copied toolchain install")
        openmp_dir = cache.get("NCNN_SANITIZER_OPENMP_LIBRARY_DIR")
        if not openmp_dir or Path(openmp_dir).resolve() / "libomp.so" != libomp_source_path.resolve():
            _finding(findings, "sanitizer-libomp-cache", "SAN CMake cache does not select the verified isolated libomp")
        tool_paths = llvm.get("tools", {})
        compiler_identity = llvm.get("compiler", {})
        compiler_cxx_identity = llvm.get("compiler_cxx", {})
        verified_tools = {"CMAKE_C_COMPILER": compiler_identity.get("path"),
                          "CMAKE_CXX_COMPILER": compiler_cxx_identity.get("path"),
                          "MLIR_OPT": tool_paths.get("mlir_opt"),
                          "MLIR_TRANSLATE": tool_paths.get("mlir_translate"),
                          "FILECHECK": tool_paths.get("FileCheck")}
        for cache_key, verified_path in verified_tools.items():
            if cache.get(cache_key) and (not verified_path or
                    Path(cache[cache_key]).resolve() != Path(verified_path).resolve()):
                _finding(findings, "sanitizer-tool-path-mismatch", f"SAN {cache_key} differs from verified tool identity")
    core_audit = staged_evidence_file(llvm.get("core_shared_dso_audit", ""))
    if not core_audit.is_file() or core_audit.stat().st_size != 0:
        _finding(findings, "sanitizer-core-dso-audit", "SAN core LLVM/MLIR shared-object audit is missing or nonempty")
    compile_audit_path = staged_evidence_file(llvm.get("compile_pic_sanitizer_audit", ""))
    if not compile_audit_path.is_file():
        _finding(findings, "sanitizer-pic-audit", "SAN instrumentation/PIC source audit is missing")
    else:
        compile_audit = compile_audit_path.read_text(encoding="utf-8", errors="replace")
        count_records = re.findall(
            r"^compile_commands=(\d+); sanitizer_instrumented_sources=(\d+); pic_sources=(\d+)$",
            compile_audit, re.MULTILINE)
        expected_compile_record = (
            (
                str(inventory_result["inventory"]["compile_commands"]["total"]),
                str(inventory_result["inventory"]["compile_commands"]["c_cpp_count"]),
                str(inventory_result["inventory"]["compile_commands"]["total"]),
            ) if inventory_result is not None else None
        ) if is_v2 else ("4316", "4312", "4316")
        if expected_compile_record is not None and count_records != [expected_compile_record]:
            message = ("SAN instrumentation/PIC counts differ from full compile_commands inventory"
                       if is_v2 else "SAN instrumentation/PIC source counts are not the exact verified record")
            _finding(findings, "sanitizer-pic-audit", message)
    asan_runtime = llvm.get("shared_asan_runtime_identity", {})
    asan_source_path = Path(asan_runtime.get("path", ""))
    asan_path = staged_evidence_file(asan_source_path)
    if not asan_path.is_file() or not SHA256_RE.fullmatch(str(asan_runtime.get("sha256", ""))) or (asan_path.is_file() and sha256_file(asan_path) != asan_runtime.get("sha256")):
        _finding(findings, "sanitizer-runtime-identity", "shared ASan runtime identity is missing or changed")
    if is_v2:
        log_paths = identity.get("log_files", {})
        def staged_log_text(name):
            source = log_paths.get(name) if isinstance(log_paths, dict) else None
            log_path = staged_evidence_file(source or "")
            if not source or not log_path.is_file():
                _finding(findings, "sanitizer-toolchain-log-missing",
                         "required identity-bound sanitizer runtime log is absent from the stage",
                         log=name)
                return None
            return log_path.read_text(encoding="utf-8", errors="replace")

        for name in ("mlir-opt.dynamic.log", "mlir-translate.dynamic.log",
                     "FileCheck.dynamic.log"):
            dynamic_text = staged_log_text(name)
            if dynamic_text is not None and (
                    "NEEDED" not in dynamic_text or "libclang_rt.asan" not in dynamic_text or
                    re.search(r"NEEDED.*lib(?:LLVM|MLIR|LTO)", dynamic_text, re.IGNORECASE)):
                _finding(findings, "sanitizer-tool-dynamic-audit",
                         "staged LLVM tool DT_NEEDED log lacks shared ASan or contains a core DSO",
                         log=name)

        core_tool_text = staged_log_text("core-tool-dynamic-audit.log")
        if core_tool_text is not None:
            try:
                core_tool_audit = json.loads(core_tool_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                core_tool_audit = None
            if (not isinstance(core_tool_audit, dict) or
                    core_tool_audit.get("passed") is not True or
                    core_tool_audit.get("failures") != [] or
                    core_tool_audit.get("llvm_config_version") != "21.1.8"):
                _finding(findings, "sanitizer-tool-dynamic-audit",
                         "staged LLVM tool dynamic/version audit does not record a clean pass")

        libomp_dynamic_text = staged_log_text("libomp-dynamic-and-symbol-audit.log")
        if libomp_dynamic_text is not None and (
                "READ_ELF_DYNAMIC:" not in libomp_dynamic_text or
                "NEEDED" not in libomp_dynamic_text or
                "libclang_rt.asan" not in libomp_dynamic_text or
                re.search(r"NEEDED.*lib(?:LLVM|MLIR|LTO)|libgomp", libomp_dynamic_text, re.IGNORECASE)):
            _finding(findings, "sanitizer-libomp-dynamic-audit",
                     "staged libomp dynamic/symbol audit is incomplete or has a forbidden dependency")
        ldd_text = staged_log_text("libomp-smoke-ldd.log")
        if ldd_text is not None and (
                "libclang_rt.asan" not in ldd_text or
                str(libomp_source_path) not in ldd_text or
                "libomp.so =>" not in ldd_text or "libgomp" in ldd_text):
            _finding(findings, "sanitizer-libomp-ldd-audit",
                     "staged libomp ldd evidence does not resolve isolated libomp/shared ASan")
        link_text = staged_log_text("libomp-lsan-link-audit.log")
        if link_text is not None and (
                str(libomp_source_path) not in link_text or
                "libclang_rt.asan" not in link_text or "libgomp" in link_text):
            _finding(findings, "sanitizer-libomp-link-audit",
                     "staged libomp LSan link audit resolves an unexpected runtime")
    for symbol_field in ("mlir_ir_asan_symbol_audit", "mlir_ir_ubsan_symbol_audit"):
        symbol_file = staged_evidence_file(llvm.get(symbol_field, ""))
        if not symbol_file.is_file() or symbol_file.stat().st_size == 0:
            _finding(findings, "sanitizer-symbol-audit", f"SAN static library symbol audit is missing: {symbol_field}")
    pic_probe = staged_evidence_file(Path(identity.get("root", "")) / "smoke/libMLIRIR-pic.so")
    if not pic_probe.is_file():
        _finding(findings, "sanitizer-pic-link-smoke", "SAN static archive PIC link smoke output is missing")
    relocation = staged_evidence_file(llvm.get("mlir_ir_pic_relocation_audit", ""))
    if not relocation.is_file() or "TEXTREL" in relocation.read_text(encoding="utf-8", errors="replace"):
        _finding(findings, "sanitizer-pic-relocation", "SAN static archive has no clean relocation audit")
    bound_source_hashes = {}

    def bind_source_hash(source, expected):
        if isinstance(source, (str, Path)) and isinstance(expected, str) and SHA256_RE.fullmatch(expected):
            bound_source_hashes[Path(source).resolve()] = expected

    for name, source in identity.get("log_files", {}).items():
        bind_source_hash(source, identity.get("log_sha256", {}).get(name))
    for mapping_key in ("status_files", "configuration_files"):
        records = identity.get(mapping_key, {})
        if isinstance(records, dict):
            for source, expected in records.items():
                bind_source_hash(source, expected)
    bind_source_hash(source_archive_path, source_archive_hash)
    if isinstance(tool_artifacts, dict):
        for tool_record in tool_artifacts.values():
            if isinstance(tool_record, dict):
                bind_source_hash(tool_record.get("path"), tool_record.get("sha256"))
    bind_source_hash(archive_list_source, archive_list_hash)
    bind_source_hash(libomp_source_path, libomp_hash)
    bind_source_hash(asan_source_path, asan_runtime.get("sha256"))
    for smoke_record in identity.get("smokes", {}).values():
        if isinstance(smoke_record, dict):
            bind_source_hash(smoke_record.get("path"), smoke_record.get("sha256"))
    for field in ("source_patch", "ubsan_fix_patch"):
        patch_record = libomp.get(field, {})
        if isinstance(patch_record, dict):
            bind_source_hash(patch_record.get("path"), patch_record.get("sha256"))
    if is_v2 and inventory_result is not None:
        normalized_inventory = inventory_result["inventory"]
        normalized_inputs = normalized_inventory["inputs"]
        for field in ("compile_commands", "cmake_cache", "llvm_config", "mlir_config", "install_manifest"):
            bind_source_hash(inventory_record["inputs"][field], normalized_inputs[field]["sha256"])
        install_prefix_for_hashes = Path(normalized_inventory["toolchain"]["cmake_install_prefix"])
        export_sources = sorted(
            (Path(path).resolve() for path in inventory_record["inputs"]["cmake_exports"]),
            key=lambda path: path.relative_to(install_prefix_for_hashes).as_posix())
        export_hashes = normalized_inputs["cmake_exports"]
        for source, hash_record in zip(export_sources, export_hashes):
            bind_source_hash(source, hash_record["sha256"])
        patch_hashes = {record["id"]: record for record in normalized_inventory["applied_patches"]}
        for record in inventory_record["inputs"]["applied_patches"]:
            normalized = patch_hashes.get(record.get("id"), {})
            for path_key, hash_key in (("patch_path", "patch_sha256"),
                                       ("application_record_path", "application_record_sha256"),
                                       ("smoke_evidence_path", "smoke_evidence_sha256")):
                if path_key in record:
                    bind_source_hash(record[path_key], normalized.get(hash_key))
        for response_record in normalized_inputs["response_files"]:
            bind_source_hash(response_record["path"], response_record["sha256"])
        for archive_record in normalized_inventory["static_archives"]:
            bind_source_hash(archive_record["source_path"], archive_record["sha256"])
    if is_v2:
        for source_path, expected_hash in bound_source_hashes.items():
            staged_path = source_to_bundle.get(source_path)
            if (staged_path is None or not staged_path.is_file() or
                    sha256_file(staged_path) != expected_hash):
                _finding(findings, "sanitizer-toolchain-bound-file-missing",
                         "v2 identity-bound input is absent or changed in the staged bundle",
                         source=str(source_path))
    copied = evidence.get("files")
    if not isinstance(copied, list) or (not is_v2 and len(copied) < 79):
        message = ("SAN toolchain bundle lacks the full verified log/status/config/patch inventory"
                   if not is_v2 else "SAN v2 toolchain bundle file inventory is malformed")
        _finding(findings, "sanitizer-toolchain-files", message)
        copied = []
    found_identity_copy = False
    refs_out = []
    for item in copied:
        if not isinstance(item, dict):
            _finding(findings, "sanitizer-toolchain-file-record", "SAN toolchain bundle file record is malformed")
            continue
        target = _verify_hash_record(stage, item, "SAN toolchain bundle file", findings)
        if target:
            refs_out.append({"path": item["path"], "sha256": item["sha256"],
                             "role": "sanitizer toolchain evidence"})
            if target == identity_copy:
                found_identity_copy = True
            source_hash = item.get("source_sha256")
            source_path = Path(item.get("source", ""))
            staged_hash = item.get("sha256")
            bound_hash = bound_source_hashes.get(source_path.resolve())
            if (not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash) or
                    (is_v2 and source_hash != staged_hash) or
                    (bound_hash is not None and source_hash != bound_hash)):
                _finding(findings, "sanitizer-toolchain-source-hash", "SAN toolchain source file hash is missing or differs from staged bytes")
            elif source_path.is_file() and sha256_file(source_path) != source_hash:
                _finding(findings, "sanitizer-toolchain-source-tampered", "SAN toolchain source changed after bundling",
                         source=str(source_path))
            elif not source_path.is_file() and not is_v2:
                _finding(findings, "sanitizer-toolchain-source-tampered", "SAN toolchain source is missing",
                         source=str(source_path))
    if not found_identity_copy:
        _finding(findings, "sanitizer-toolchain-copy", "SAN toolchain bundle file list omits copied identity JSON")
    relative_files = [str(item.get("path", "")) for item in copied if isinstance(item, dict)]
    required_bundle_areas = ("logs/log_files", "logs/status_files", "logs/configuration_files",
                             "patches", "status/verification.status")
    missing_areas = [area for area in required_bundle_areas
                     if not any(area in path for path in relative_files)]
    if missing_areas:
        _finding(findings, "sanitizer-toolchain-archive-content",
                 "SAN bundle lacks required logs/status/configuration/patch artifacts",
                 missing=missing_areas)
    return refs_out


def _step_filter_flags(argv: list[str]) -> list[str]:
    forbidden = {"-R", "-E", "-L", "-LE", "-I", "--tests-regex", "--exclude-regex",
                 "--label-regex", "--label-exclude", "--rerun-failed", "-N", "--show-only"}
    result = []
    for argument in argv:
        if (argument in forbidden or
                any(argument.startswith(flag + "=") for flag in forbidden) or
                (argument.startswith(("-R", "-E", "-L", "-LE", "-I")) and not argument.startswith("--")
                 and len(argument) > 2)):
            result.append(argument)
    return result


def _junit_results(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name:
            continue
        if name in results:
            raise ValueError(f"duplicate JUnit testcase name: {name}")
        if case.find("skipped") is not None:
            status = "skipped"
        elif case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        else:
            status = "passed"
        results[name] = status
    return results


def _lasttest_results(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    results: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"\s*(?:\d+/\d+\s+)?Testing:\s*(.+?)\s*$", line)
        if match:
            current = match.group(1).strip().strip('"')
            continue
        match = re.match(r"\s*Test\s+(Passed|Failed|Skipped)\.?\s*$", line, re.I)
        if match and current:
            results[current] = match.group(1).lower()
            current = None
    # Some CTest versions put a one-line summary in LastTest.log.
    if not results:
        for match in re.finditer(r"Test #\d+:\s*(\S+)\s+(Passed|Failed|Skipped)", text, re.I):
            results[match.group(1)] = match.group(2).lower()
    return results


def _inventory_tests(path: Path) -> dict[str, dict]:
    document = load_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("tests"), list):
        raise ValueError("CTest inventory is not native ctest json-v1")
    tests = {}
    for test in document["tests"]:
        if not isinstance(test, dict) or not isinstance(test.get("name"), str):
            raise ValueError("CTest inventory contains a malformed test record")
        name = test["name"]
        if name in tests:
            raise ValueError(f"CTest inventory repeats test name {name}")
        command = test.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError(f"CTest inventory has no executable command for {name}")
        tests[name] = test
    return tests


def _validate_skip_source(source_root: Path, findings: list[dict]) -> None:
    source = source_root / "test/Numerical/models/performance_test.cpp"
    if not source.is_file():
        _finding(findings, "skip-source-missing", "cannot verify CMake/GTest PerformanceModel definitions")
        return
    tests = set(re.findall(r"TEST\s*\(\s*PerformanceModel\s*,\s*(\w+)\s*\)",
                           source.read_text(encoding="utf-8", errors="replace")))
    actual = {"PerformanceModel." + name for name in tests}
    if actual != SANITIZER_PERFORMANCE_SKIPS:
        _finding(findings, "skip-whitelist-source-drift",
                 "PerformanceModel test definitions changed; skip whitelist requires review",
                 added=sorted(actual - SANITIZER_PERFORMANCE_SKIPS),
                 removed=sorted(SANITIZER_PERFORMANCE_SKIPS - actual))


def _validate_test_execution(stage: Path, profile: str, state: dict,
                             inventory_path: Path, junit_path: Path,
                             lasttest_path: Path, source_root: Path,
                             findings: list[dict]) -> dict:
    try:
        inventory = _inventory_tests(inventory_path)
        junit = _junit_results(junit_path)
        lasttest_text = lasttest_path.read_text(encoding="utf-8", errors="replace")
        if SANITIZER_RE.search(lasttest_text):
            _finding(findings, "lasttest-sanitizer-diagnostic", "raw CTest LastTest.log contains a sanitizer/runtime diagnostic")
        lasttest = _lasttest_results(lasttest_path)
    except (OSError, ValueError, json.JSONDecodeError, ET.ParseError) as error:
        _finding(findings, "ctest-evidence-invalid", str(error))
        return {"inventory_count": 0, "actual_count": 0, "passed": 0, "skipped": [], "failed": []}

    names = set(inventory)
    allowed_skips = SANITIZER_PERFORMANCE_SKIPS if profile == "sanitize" else RELEASE_PERFORMANCE_SKIPS
    if names != set(junit):
        _finding(findings, "ctest-inventory-junit-mismatch",
                 "CTest inventory and JUnit actual test names differ",
                 missing_junit=sorted(names - set(junit)), unexpected_junit=sorted(set(junit) - names))
    missing_lasttest = names - set(lasttest)
    allowed_missing_lasttest = {
        name for name in missing_lasttest
        if name in allowed_skips and junit.get(name) == "skipped"
    }
    unaccounted_missing_lasttest = missing_lasttest - allowed_missing_lasttest
    unexpected_lasttest = set(lasttest) - names
    if unaccounted_missing_lasttest or unexpected_lasttest:
        _finding(findings, "ctest-inventory-lasttest-mismatch",
                 "CTest inventory and LastTest.log differ beyond approved JUnit-confirmed disabled skips",
                 missing_lasttest=sorted(unaccounted_missing_lasttest),
                 accounted_missing_disabled_skips=sorted(allowed_missing_lasttest),
                 unexpected_lasttest=sorted(unexpected_lasttest))
    failed = sorted(name for name, status in junit.items() if status == "failed")
    skipped = sorted(name for name, status in junit.items() if status == "skipped")
    if failed:
        _finding(findings, "ctest-failed", "CTest contains failed tests", tests=failed)
    for name, status in junit.items():
        if name in lasttest and status != lasttest[name]:
            _finding(findings, "ctest-status-mismatch", f"JUnit and LastTest disagree for {name}",
                     junit=status, lasttest=lasttest[name])
    unexpected = sorted(set(skipped) - allowed_skips)
    missing_expected = sorted(allowed_skips - set(skipped))
    if unexpected:
        _finding(findings, "unexpected-ctest-skip", "CTest has a skip outside the fixed profile whitelist",
                 tests=unexpected)
    if missing_expected:
        _finding(findings, "expected-ctest-skip-missing",
                 "existing configured PerformanceModel skips differ from the fixed profile whitelist",
                 tests=missing_expected)
    if profile == "sanitize":
        _validate_skip_source(source_root, findings)

    step = next((item for item in state.get("steps", []) if item.get("name") == "ctest"), {})
    argv = step.get("argv", [])
    if not isinstance(argv, list) or not argv:
        _finding(findings, "ctest-command-missing", "gate-state has no CTest argv")
    else:
        filters = _step_filter_flags(argv)
        if filters:
            _finding(findings, "ctest-filtered", "full gate CTest command contains filters", flags=filters)
        expected_argv = ["ctest", "--test-dir", str(stage), "--output-on-failure",
                         "--output-junit", str(junit_path)]
        if argv != expected_argv:
            _finding(findings, "ctest-command-mismatch",
                     "full gate must run the exact unfiltered CTest command and preserve its JUnit output",
                     expected=expected_argv, actual=argv)

    # The CMake-generated command inventory is also checked for stage-local evidence
    # paths, not merely for a matching test count.
    for name, test in inventory.items():
        command_text = "\0".join(test.get("command", []))
        properties = test.get("properties", [])
        labels = set()
        if isinstance(properties, list):
            for prop in properties:
                if isinstance(prop, dict) and prop.get("name") == "LABELS":
                    value = prop.get("value", [])
                    labels.update(value if isinstance(value, list) else [value])
        is_gate_integration = name.startswith(("vulkan-v0-", "vulkan-v2-", "vulkan-v3-",
                                               "vulkan-v5-", "ncnn-compile-vulkan"))
        if is_gate_integration and "unit" not in labels and str(stage) not in command_text:
            _finding(findings, "nonfresh-test-command",
                     f"registered Vulkan gate test does not reference this fresh build: {name}")
    unit_names = {name for name in names if name in {
        "vulkan-release-evidence-validation", "vulkan-release-evidence-unit",
        "release-evidence-unit"}}
    if not unit_names:
        _finding(findings, "evidence-unit-not-registered",
                 "independent synthetic release-evidence unit test is absent from CTest inventory")
    elif any(junit.get(name) != "passed" for name in unit_names):
        _finding(findings, "evidence-unit-not-passing",
                 "independent synthetic release-evidence unit test did not pass",
                 tests=sorted(unit_names))
    return {"inventory_count": len(inventory), "actual_count": len(junit),
            "passed": sum(status == "passed" for status in junit.values()),
            "skipped": skipped, "failed": failed,
            "test_names": sorted(names)}


def audit_stage_gate(stage_dir: Path, *, source_root: Path | None = None,
                     verify_fingerprint: bool = True) -> dict:
    """Validate the gate envelope only; evidence adapters add domain checks later."""
    stage = stage_dir.resolve()
    findings: list[dict] = []
    state_path = stage / "gate-state.json"
    try:
        state = load_json(state_path)
    except (OSError, json.JSONDecodeError) as error:
        return {"stage": str(stage), "profile": None, "passed": False,
                "findings": [{"code": "gate-state-missing", "message": str(error)}]}
    if not isinstance(state, dict):
        return {"stage": str(stage), "profile": None, "passed": False,
                "findings": [{"code": "gate-state-invalid", "message": "gate-state must be a JSON object"}]}
    profile = state.get("profile")
    # Older gate-state records predate stage_id and represent the legacy VA0
    # product gate. Preserve that historical default without accepting an
    # explicitly invalid stage_id as VA0.
    stage_id = state["stage_id"] if "stage_id" in state else "VA0"
    if state.get("schema_version") != SCHEMA_VERSION or state.get("kind") != GATE_KIND:
        _finding(findings, "gate-schema", "unsupported gate-state schema/kind")
    if profile not in PROFILES:
        _finding(findings, "gate-profile", "gate profile is outside the agreed enum", profile=profile)
    if state.get("completed") is not True:
        _finding(findings, "gate-incomplete", "gate-state.completed is not true")
    stage_build = state.get("build_dir")
    if not isinstance(stage_build, str) or Path(stage_build).resolve() != stage:
        _finding(findings, "stale-stage", "gate build_dir does not identify this fresh stage",
                 build_dir=stage_build)
    if not isinstance(state.get("git_revision"), str) or not state["git_revision"]:
        _finding(findings, "git-revision-missing", "baseline git_revision is missing")
    if (not isinstance(stage_id, str) or stage_id not in {"VA0", "VA1", "VA2", "VA3", "VA4", "VA5", "VA6"} or
            state.get("host_execution_role") != "legacy-product" or
            state.get("host_aot_candidate") is not False):
        _finding(findings, "gate-execution-role",
                 "qualification stages retain the legacy product role and cannot claim host AOT release eligibility")
    if isinstance(stage_id, str) and stage_id in {"VA2", "VA3", "VA4", "VA5", "VA6"} and profile not in {"release", "sanitize"}:
        _finding(findings, f"{stage_id.lower()}-profile-invalid",
                 f"{stage_id} qualification accepts only Release and matching static SAN stages",
                 profile=profile)
    if stage_id == "VA3" and state.get("candidate_role") != "lifecycle-only":
        _finding(findings, "va3-candidate-role",
                 "VA3 candidate evidence must be marked lifecycle-only")
    if stage_id == "VA4" and state.get("candidate_role") != "fixed-run":
        _finding(findings, "va4-candidate-role",
                 "VA4 evidence must identify the private fixed-run candidate")
    if (isinstance(stage_id, str) and stage_id not in {"VA3", "VA4"} and
            state.get("candidate_role", "none") != "none"):
        _finding(findings, "gate-candidate-role",
                 "only VA3 lifecycle and VA4 fixed-run stages may claim private candidate roles")
    worktree = state.get("worktree")
    if not isinstance(worktree, dict):
        _finding(findings, "source-worktree-missing", "gate must preserve worktree status and tracked diff")
    else:
        status_path = _verify_hash_record(stage, worktree.get("status"), "source worktree status", findings)
        _verify_hash_record(stage, worktree.get("tracked_diff"), "source tracked diff", findings)
        if status_path and (type(worktree.get("dirty")) is not bool or
                            worktree["dirty"] != bool(status_path.read_bytes())):
            _finding(findings, "source-worktree-state", "dirty status differs from the preserved worktree bytes")
    fingerprint = state.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        _finding(findings, "source-fingerprint-invalid", "source_fingerprint is not a SHA-256 digest")
    if state.get("source_fingerprint_algorithm") != "sha256-git-files-v1":
        _finding(findings, "source-fingerprint-algorithm", "source fingerprint algorithm is not sha256-git-files-v1")
    created = parse_time(state.get("created_at"))
    finished = parse_time(state.get("finished_at"))
    if created is None or finished is None or finished < created:
        _finding(findings, "gate-time-invalid", "gate created_at/finished_at timestamps are invalid")
    elif finished > datetime.now(timezone.utc).replace(microsecond=0) + __import__("datetime").timedelta(minutes=5):
        _finding(findings, "gate-time-future", "gate finished_at is in the future")

    steps = state.get("steps")
    if not isinstance(steps, list) or [item.get("name") for item in steps if isinstance(item, dict)] != list(STEP_NAMES):
        _finding(findings, "gate-steps-incomplete", "gate must contain exactly the six ordered STEP_NAMES")
        steps = steps if isinstance(steps, list) else []
    step_times = []
    logs = []
    for expected, item in zip(STEP_NAMES, steps):
        if not isinstance(item, dict):
            _finding(findings, "gate-step-invalid", f"{expected} step record is malformed")
            continue
        if item.get("name") != expected:
            continue
        if item.get("exit_code") != 0:
            _finding(findings, "gate-step-failed", f"gate step {expected} did not exit zero",
                     exit_code=item.get("exit_code"))
        started, ended = parse_time(item.get("started_at")), parse_time(item.get("finished_at"))
        if started is None or ended is None or ended < started:
            _finding(findings, "gate-step-time", f"gate step {expected} has invalid timestamps")
        else:
            step_times.append((expected, started, ended))
        path = safe_stage_file(stage, item.get("log"))
        if path is None or not path.is_file():
            _finding(findings, "gate-step-log-missing", f"gate step {expected} log is missing or unsafe")
            continue
        digest = item.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or sha256_file(path) != digest:
            _finding(findings, "gate-step-log-hash", f"gate step {expected} log hash is invalid")
        logs.append((expected, path))
        log_text = path.read_text(encoding="utf-8", errors="replace")
        if SANITIZER_RE.search(log_text):
            _finding(findings, "raw-sanitizer-diagnostic",
                     f"gate {expected} log contains an ASan/LSan/UBSan/runtime diagnostic",
                     path=str(path.relative_to(stage)))
        if expected == "ctest" and _step_filter_flags(item.get("argv", [])):
            _finding(findings, "ctest-filtered", "CTest argv contains a filter")
    for previous, current in zip(step_times, step_times[1:]):
        if current[1] < previous[2]:
            _finding(findings, "gate-step-order", "gate step timestamps overlap or run out of order",
                     previous=previous[0], current=current[0])

    cache_path = _verify_hash_record(stage, state.get("cmake_cache"), "CMakeCache", findings)
    if cache_path:
        _verify_config(profile if profile in PROFILES else "release", state, cache_path, findings)
    _audit_sanitizer_budget_evidence(stage, state, cache_path, findings)
    toolchain_refs = _audit_sanitizer_toolchain(stage, state, cache_path, [], findings)
    identities = state.get("toolchain_identities")
    if not isinstance(identities, list) or not identities:
        _finding(findings, "tool-identities-missing", "toolchain_identities must enumerate gate tools")
    else:
        roles = set()
        for identity in identities:
            if not isinstance(identity, dict) or not all(identity.get(key) for key in ("role", "path", "sha256")):
                _finding(findings, "tool-identity-invalid", "toolchain identity lacks role/path/sha256")
                continue
            roles.add(identity["role"])
            digest = identity["sha256"]
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                _finding(findings, "tool-identity-hash", f"invalid tool identity SHA for {identity.get('role')}")
                continue
            tool_path = Path(identity["path"])
            if not tool_path.is_file():
                _finding(findings, "tool-identity-path", f"tool identity path is unavailable: {identity['path']}")
            elif sha256_file(tool_path) != digest:
                _finding(findings, "tool-identity-tampered", f"tool identity hash changed: {identity['role']}")
        if profile == "sanitize" and not {"lit", "libomp", "sanitizer-toolchain"}.issubset(roles):
            _finding(findings, "sanitizer-tool-identities", "SAN toolchain must identify lit, isolated libomp, and instrumented toolchain")

    built_tool_refs = []
    built_tools = state.get("built_tool_identities")
    expected_built_tools = {"tools/ncnn-compile", "tools/ncnn-mlir-driver",
                            "bin/ncnn-mlir-opt", "tools/ncnn-vulkan-spirv-extract"}
    if not isinstance(built_tools, list) or len(built_tools) != 4:
        _finding(findings, "built-tool-identities", "gate must identify all four freshly built compiler/Vulkan tools")
    else:
        observed_paths = []
        for record in built_tools:
            if not isinstance(record, dict):
                _finding(findings, "built-tool-record", "built tool identity is malformed")
                continue
            tool_path = _verify_hash_record(stage, record, "fresh build tool", findings)
            relative = record.get("path")
            observed_paths.append(relative)
            if relative != "tools/ncnn-vulkan-spirv-extract" and not isinstance(record.get("version"), str):
                _finding(findings, "built-tool-version", f"fresh build tool version is missing: {relative}")
            if tool_path:
                built_tool_refs.append({"path": str(tool_path.relative_to(stage)),
                                        "sha256": record["sha256"],
                                        "role": "freshly built gate executable"})
        if set(observed_paths) != expected_built_tools or len(set(observed_paths)) != 4:
            _finding(findings, "built-tool-coverage", "fresh tool identity path set differs from the required tools",
                     expected=sorted(expected_built_tools), actual=observed_paths)

    inventory_path = _verify_hash_record(stage, state.get("inventory"), "CTest inventory", findings)
    junit_path = _verify_hash_record(stage, state.get("junit"), "CTest JUnit", findings)
    lasttest_path = _verify_hash_record(stage, state.get("lasttest"), "CTest LastTest.log", findings)
    ctest_result = {"inventory_count": 0, "actual_count": 0, "passed": 0, "skipped": [], "failed": []}
    if inventory_path and junit_path and lasttest_path and profile in PROFILES:
        ctest_result = _validate_test_execution(stage, profile, state, inventory_path,
                                               junit_path, lasttest_path,
                                               source_root or Path(state.get("source_dir", "")), findings)
        _check_required_tests({"profile": profile, "stage_id": stage_id,
                               "ctest": ctest_result}, findings)

    diagnostics = state.get("sanitizer_diagnostics")
    if not isinstance(diagnostics, list) or diagnostics:
        _finding(findings, "sanitizer-diagnostics", "sanitizer_diagnostics must be an empty list",
                 diagnostics=diagnostics)
    if profile == "sanitize":
        linkage = state.get("static_linkage_audit")
        if not isinstance(linkage, list) or len(linkage) != 3:
            _finding(findings, "static-linkage-audit", "SAN requires static-linkage audit for three host executables")
        else:
            for record in linkage:
                if not isinstance(record, dict):
                    _finding(findings, "static-linkage-record", "malformed SAN linkage record")
                    continue
                executable = _verify_hash_record(stage, record.get("executable"), "SAN host executable", findings)
                needed = record.get("needed")
                if not isinstance(needed, list) or any(re.match(r"lib(?:LLVM|MLIR).*\.so", str(name)) for name in needed):
                    _finding(findings, "core-dso-linked", "SAN host executable links core LLVM/MLIR DSO")
                if not isinstance(needed, list) or not any("libclang_rt.asan" in str(name) for name in needed):
                    _finding(findings, "shared-asan-missing", "SAN host executable lacks shared ASan runtime")

    source_root = source_root or Path(state.get("source_dir", "")).resolve()
    source_refs = []
    source_inputs = state.get("source_inputs")
    if not isinstance(source_inputs, list) or not source_inputs:
        _finding(findings, "source-inputs-missing", "gate-state must snapshot raw Vulkan fixture inputs")
    else:
        seen_inputs = set()
        for item in source_inputs:
            if not isinstance(item, dict):
                _finding(findings, "source-input-record", "source_inputs contains a malformed entry")
                continue
            snapshot = _verify_hash_record(stage, item, "gate source input", findings)
            if snapshot is None:
                continue
            if item.get("source"):
                source_path = source_root / item["source"]
                identity = ("source", item["source"])
                if not source_path.is_file() or sha256_file(source_path) != item.get("sha256"):
                    _finding(findings, "source-input-mismatch", "snapshotted source input differs from current source tree",
                             source=item["source"])
            elif item.get("generated"):
                identity = ("generated", item["generated"])
                generated = safe_stage_file(stage, item["generated"])
                if generated is None or not generated.is_file() or sha256_file(generated) != item.get("sha256"):
                    _finding(findings, "generated-input-mismatch", "generated fixture input changed after snapshot",
                             generated=item["generated"])
            else:
                _finding(findings, "source-input-kind", "source input must identify source or generated origin")
                continue
            if identity in seen_inputs:
                _finding(findings, "source-input-duplicate", "source_inputs repeats an origin", input=identity[1])
            seen_inputs.add(identity)
            source_refs.append({"path": item["path"], "sha256": item["sha256"],
                                "source": item.get("source"), "generated": item.get("generated")})

    if verify_fingerprint and profile in PROFILES and isinstance(fingerprint, str) and source_root.is_dir():
        try:
            actual_fingerprint = _load_fingerprint_function()(source_root)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            _finding(findings, "source-fingerprint-unverifiable", str(error))
        else:
            if actual_fingerprint != fingerprint:
                _finding(findings, "stale-source", "source tree fingerprint differs from gate-state",
                         recorded=fingerprint, actual=actual_fingerprint)
    elif verify_fingerprint and profile in PROFILES:
        _finding(findings, "source-root-missing", "cannot locate compiler source root for fingerprint verification")

    return {"stage": str(stage), "profile": profile,
            "stage_id": stage_id,
            "source_fingerprint": fingerprint,
            "git_revision": state.get("git_revision"), "completed": state.get("completed") is True,
            "ctest": ctest_result, "passed": not findings, "findings": findings,
            "state": state, "log_files": [str(path.relative_to(stage)) for _, path in logs],
            "source_inputs": source_refs,
            "sanitizer_toolchain_files": toolchain_refs,
            "built_tool_files": built_tool_refs}


def _evidence_file(stage: Path, base: Path, relative: object, label: str,
                   refs: list[dict], findings: list[dict]) -> Path | None:
    if not isinstance(relative, str) or not relative:
        _finding(findings, "evidence-path-missing", f"{label} path is missing")
        return None
    candidate = (base / relative).resolve()
    try:
        rel = candidate.relative_to(stage.resolve())
    except ValueError:
        _finding(findings, "historical-evidence-path",
                 f"{label} is not inside the current fresh stage", path=relative)
        return None
    if not candidate.is_file():
        _finding(findings, "evidence-file-missing", f"{label} file is missing", path=str(rel))
        return None
    refs.append({"path": str(rel), "sha256": sha256_file(candidate), "role": label})
    return candidate


def _check_digest_claim(claim: object, path: Path | None, label: str,
                        findings: list[dict]) -> bool:
    if path is None:
        return False
    if not isinstance(claim, str) or not SHA256_RE.fullmatch(claim):
        _finding(findings, "evidence-hash-invalid", f"{label} hash claim is missing or malformed")
        return False
    actual = sha256_file(path)
    if actual != claim:
        _finding(findings, "evidence-hash-mismatch", f"{label} bytes do not match declared SHA-256",
                 declared=claim, actual=actual)
        return False
    return True


def read_f32(path: Path, count: int | None = None) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 or (count is not None and len(data) != count * 4):
        raise ValueError(f"invalid FP32 byte count in {path}")
    values = list(struct.unpack(f"<{len(data) // 4}f", data))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite FP32 value in {path}")
    return values


def compare_f32(actual: list[float], expected: list[float], atol: float, rtol: float) -> dict:
    if len(actual) != len(expected):
        raise ValueError("FP32 output length mismatch")
    errors = [abs(a - e) for a, e in zip(actual, expected)]
    return {"passed": all(error <= atol + rtol * abs(expected_value)
                           for error, expected_value in zip(errors, expected)),
            "max_abs": max(errors, default=0.0),
            "max_rel": max((error / max(abs(value), 1.0e-30)
                            for error, value in zip(errors, expected)), default=0.0)}


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _cpp_stream_float(value: float) -> float:
    """Numeric value produced by the runner's default six-significant-digit stream."""
    return float(format(_float32(value), ".6g"))


def compare_cpp_f32(actual: list[float], expected: list[float],
                    atol: float, rtol: float) -> dict:
    """Reproduce vulkan_add_runner's float intermediates for recorded metrics."""
    if len(actual) != len(expected):
        raise ValueError("FP32 output length mismatch")
    atol32, rtol32 = _float32(atol), _float32(rtol)
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for actual_value, expected_value in zip(actual, expected):
        actual32, expected32 = _float32(actual_value), _float32(expected_value)
        absolute = _float32(abs(_float32(actual32 - expected32)))
        expected_abs = _float32(abs(expected32))
        denominator = max(expected_abs, _float32(1.0e-30))
        relative = _float32(absolute / denominator)
        threshold = _float32(atol32 + _float32(rtol32 * expected_abs))
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        passed = passed and absolute <= threshold
    return {"passed": passed, "max_abs": max_abs, "max_rel": max_rel}


def _check_metric_claim(claim: object, actual: dict, label: str,
                        findings: list[dict]) -> None:
    if not isinstance(claim, dict) or claim.get("passed") is not actual.get("passed"):
        _finding(findings, "numeric-claim-failed", f"{label} passed claim differs from C++ float recomputation")
        return
    for key in ("max_abs", "max_rel"):
        value = claim.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value)):
            _finding(findings, "numeric-claim-invalid", f"{label}.{key} is invalid")
            continue
        serialized = _cpp_stream_float(actual[key])
        if float(value) != serialized:
            _finding(findings, "numeric-claim-tampered",
                     f"{label}.{key} differs from C++ float metric serialized with six significant digits",
                     declared=value, recomputed=actual[key], serialized=serialized)


def _verify_triplet(raw_inputs: Path, raw_outputs: Path, record: dict,
                    count: int, label: str, findings: list[dict]) -> dict | None:
    tolerance = record.get("tolerance", {})
    atol = tolerance.get("atol") if isinstance(tolerance, dict) else None
    rtol = tolerance.get("rtol") if isinstance(tolerance, dict) else None
    if not isinstance(atol, (int, float)) or not isinstance(rtol, (int, float)) or atol < 0 or rtol < 0:
        _finding(findings, "numeric-budget-missing", f"{label} fixed numerical budget is missing")
        return None
    try:
        inputs = read_f32(raw_inputs)
        outputs = read_f32(raw_outputs, count * 3)
    except (OSError, ValueError, struct.error) as error:
        _finding(findings, "numeric-raw-invalid", f"{label} raw FP32 records are invalid: {error}")
        return None
    if not inputs:
        _finding(findings, "numeric-input-empty", f"{label} input record is empty")
        return None
    cpu, gpu, golden = outputs[:count], outputs[count:2 * count], outputs[2 * count:]
    pairs = {"cpu_vs_golden": (cpu, golden),
             "gpu_vs_golden": (gpu, golden),
             "gpu_vs_cpu": (gpu, cpu)}
    # Keep the independent, full-precision comparison as the acceptance gate.
    # Only validate serialized metric fields against the runner's FP32 math.
    metrics = {key: compare_f32(actual, expected, atol, rtol)
               for key, (actual, expected) in pairs.items()}
    cpp_metrics = {key: compare_cpp_f32(actual, expected, atol, rtol)
                   for key, (actual, expected) in pairs.items()}
    for key, result in metrics.items():
        if not result["passed"]:
            _finding(findings, "numeric-budget-exceeded", f"{label} {key} exceeds fixed budget",
                     max_abs=result["max_abs"], atol=atol, rtol=rtol)
        _check_metric_claim(record.get(key), cpp_metrics[key],
                            f"{label}.{key}", findings)
    return {"elements": count, "input_sha256": sha256_file(raw_inputs),
            "output_sha256": sha256_file(raw_outputs), "metrics": metrics,
            "tolerance": {"atol": atol, "rtol": rtol}}


def _vulkan_version_number(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?\s*", text)
    if not match:
        return None
    major, minor, patch, build = (int(value or 0) for value in match.groups())
    if build or len(match.groups()) == 4 and match.group(4) is not None:
        return (major << 22) | (minor << 14) | (patch << 6) | build
    return (major << 22) | (minor << 12) | patch


def _parse_physical_summary(summary: str) -> list[dict]:
    devices = []
    current = None
    for line in summary.splitlines():
        if re.fullmatch(r"\s*GPU\d+:\s*", line):
            if current:
                devices.append(current)
            current = {}
            continue
        if current is None:
            continue
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if match:
            current[match.group(1)] = match.group(2)
        elif not line.strip():
            devices.append(current)
            current = None
    if current:
        devices.append(current)
    return devices


def _int_field(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def verify_physical_identity(stage: Path, base: Path, source: dict,
                             expected_class: str, refs: list[dict],
                             findings: list[dict], label: str) -> dict | None:
    """Validate physical identity from raw vulkaninfo, never from layer/name alone."""
    if not isinstance(source, dict):
        _finding(findings, "physical-identity-missing", f"{label} device identity is missing")
        return None
    summary = source.get("vulkaninfo_summary")
    summary_path = None
    if isinstance(summary, str) and "deviceType" in summary and "GPU" in summary:
        summary_text = summary
    elif isinstance(summary, str):
        summary_path = _evidence_file(stage, base, summary, f"{label} vulkaninfo summary", refs, findings)
        summary_text = summary_path.read_text(encoding="utf-8", errors="replace") if summary_path else ""
    else:
        _finding(findings, "physical-summary-missing", f"{label} has no raw vulkaninfo summary")
        return None
    summary_hash = source.get("vulkaninfo_summary_sha256")
    if summary_hash is not None:
        if summary_path is not None:
            _check_digest_claim(summary_hash, summary_path, f"{label} vulkaninfo summary", findings)
        elif not isinstance(summary_hash, str) or sha256_bytes(summary_text.encode("utf-8")) != summary_hash:
            _finding(findings, "physical-summary-hash", f"{label} inline vulkaninfo summary SHA-256 differs")
    devices = _parse_physical_summary(summary_text)
    device_name = source.get("device_name", source.get("device", source.get("deviceName")))
    vendor = _int_field(source.get("vendor_id", source.get("vendorID")))
    device_id = _int_field(source.get("device_id", source.get("deviceID")))
    api = source.get("api_version", source.get("apiVersion"))
    driver = _int_field(source.get("driver_version", source.get("driverVersion")))
    matches = [item for item in devices if
               item.get("deviceName") == device_name and
               _int_field(item.get("vendorID")) == vendor and
               _int_field(item.get("deviceID")) == device_id]
    if len(matches) != 1:
        _finding(findings, "physical-summary-mismatch",
                 f"{label} summary does not uniquely identify its recorded physical device",
                 device=device_name, vendor_id=vendor, device_id=device_id)
        return None
    physical = matches[0]
    api_match = physical.get("apiVersion") == api
    if isinstance(api, int):
        api_match = _vulkan_version_number(physical.get("apiVersion")) == api
    if not api_match:
        _finding(findings, "physical-api-mismatch", f"{label} API version differs from raw summary")
    parsed_driver = _vulkan_version_number(physical.get("driverVersion"))
    if driver is not None and parsed_driver is not None and driver != parsed_driver:
        _finding(findings, "physical-driver-mismatch", f"{label} driver version differs from raw summary")
    source_driver = source.get("driver_version")
    if isinstance(source_driver, str) and source_driver != physical.get("driverVersion"):
        _finding(findings, "physical-driver-string-mismatch",
                 f"{label} driver version text differs from raw summary")
    driver_info = source.get("driver_info")
    if driver_info and physical.get("driverInfo") != driver_info:
        _finding(findings, "physical-driver-info-mismatch",
                 f"{label} driverInfo differs from raw summary")
    declared_type = source.get("device_type", source.get("deviceType"))
    raw_type = physical.get("deviceType")
    if not declared_type or declared_type != raw_type:
        _finding(findings, "physical-type-mismatch", f"{label} deviceType differs from raw summary",
                 declared=declared_type, raw=raw_type)
    if expected_class == "hardware":
        if vendor != 0x10DE or raw_type not in ("PHYSICAL_DEVICE_TYPE_DISCRETE_GPU", "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"):
            _finding(findings, "fake-hardware", f"{label} is not a physical NVIDIA GPU",
                     vendor_id=vendor, device_type=raw_type)
    elif expected_class == "software":
        if raw_type != "PHYSICAL_DEVICE_TYPE_CPU" or vendor != 0x10005 or "llvmpipe" not in str(physical.get("driverName", "")).lower():
            _finding(findings, "fake-software-device", f"{label} is not a Mesa llvmpipe CPU ICD",
                     vendor_id=vendor, device_type=raw_type, driver=physical.get("driverName"))
    if summary_path is None:
        refs.append({"path": "<inline-summary>", "sha256": sha256_bytes(summary_text.encode()),
                     "role": f"{label} vulkaninfo summary"})
    return {"device_name": device_name, "api_version": api, "driver_version": driver,
            "vendor_id": vendor, "device_id": device_id, "device_type": raw_type,
            "driver_name": physical.get("driverName"),
            "summary_sha256": sha256_bytes(summary_text.encode())}


def _check_inline_device_summary(stage: Path, base: Path, inline_summary: object,
                                 declared_sha256: object, label: str,
                                 refs: list[dict], findings: list[dict]) -> str | None:
    summary_path = _evidence_file(stage, base, "vulkaninfo-summary.txt",
                                  f"{label} raw vulkaninfo summary", refs, findings)
    if summary_path is None:
        return None
    summary_text = summary_path.read_text(encoding="utf-8", errors="replace")
    if inline_summary != summary_text:
        _finding(findings, "v0-summary-inline-mismatch",
                 f"{label} inline and archived vulkaninfo summaries differ")
    _check_digest_claim(declared_sha256, summary_path,
                        f"{label} declared vulkaninfo summary", findings)
    return sha256_file(summary_path)


def _audit_v0(stage: Path, expected_devices: set[str], refs: list[dict],
              findings: list[dict]) -> dict:
    root = stage / "test/Vulkan/evidence/gpu-func"
    records = {}
    metadata_files = sorted(root.glob("*/metadata.json")) if root.is_dir() else []
    if not metadata_files:
        _finding(findings, "v0-evidence-missing", "fresh V0 gpu.func metadata is absent")
        return records
    for metadata_path in metadata_files:
        device_key = metadata_path.parent.name
        if device_key not in {"nvidia", "llvmpipe"}:
            _finding(findings, "v0-device-name", "unexpected V0 evidence directory", device=device_key)
            continue
        try:
            metadata = load_json(metadata_path)
            _evidence_file(stage, metadata_path.parent, "metadata.json",
                           f"V0 {device_key} metadata", refs, findings)
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
            source_param = _evidence_file(stage, stage / "gate-source-inputs", "add.param",
                                          f"V0 {device_key} source .param", refs, findings)
            source_kernel = _evidence_file(stage, stage / "gate-source-inputs", "gpu_func_add_probe.mlir",
                                           f"V0 {device_key} gpu.func source", refs, findings)
            _check_digest_claim(metadata.get("model_param_sha256"), source_param,
                                f"V0 {device_key} model .param", findings)
            _check_digest_claim(metadata.get("gpu_func_source_sha256"), source_kernel,
                                f"V0 {device_key} gpu.func source", findings)
            cpu_library = stage / "test/Vulkan/cpu-add/libv0_add.so"
            _evidence_file(stage, cpu_library.parent, cpu_library.name,
                           f"V0 {device_key} CPU AOT library", refs, findings)
            _check_digest_claim(metadata.get("cpu_reference_library_sha256"),
                                cpu_library if cpu_library.is_file() else None,
                                f"V0 {device_key} CPU AOT library", findings)
            result_path = _evidence_file(stage, metadata_path.parent, "device-result.json",
                                         f"V0 {device_key} result", refs, findings)
            inputs_path = _evidence_file(stage, metadata_path.parent,
                                         "device-result.json.inputs.bin",
                                         f"V0 {device_key} input FP32", refs, findings)
            outputs_path = _evidence_file(stage, metadata_path.parent,
                                          "device-result.json.outputs.bin",
                                          f"V0 {device_key} output triplet", refs, findings)
            shader_path = _evidence_file(stage, metadata_path.parent, "add.spv",
                                         f"V0 {device_key} SPIR-V", refs, findings)
            binary_ir_path = _evidence_file(stage, metadata_path.parent, "gpu-binary.mlir",
                                            f"V0 {device_key} lowered binary IR", refs, findings)
            _check_digest_claim(metadata.get("gpu_binary_ir_sha256"), binary_ir_path,
                                f"V0 {device_key} lowered binary IR", findings)
            disassembly_path = _evidence_file(stage, metadata_path.parent, "add.spvasm",
                                              f"V0 {device_key} SPIR-V disassembly", refs, findings)
            _check_digest_claim(metadata.get("spirv_disassembly_sha256"), disassembly_path,
                                f"V0 {device_key} SPIR-V disassembly", findings)
            if result_path is None or inputs_path is None or outputs_path is None:
                continue
            result = load_json(result_path)
            if not isinstance(result, dict):
                raise ValueError("device result is not an object")
            _check_digest_claim(metadata.get("inputs_sha256"), inputs_path,
                                f"V0 {device_key} input", findings)
            _check_digest_claim(metadata.get("outputs_sha256"), outputs_path,
                                f"V0 {device_key} outputs", findings)
            if shader_path:
                _check_digest_claim(metadata.get("spirv_sha256"), shader_path,
                                    f"V0 {device_key} SPIR-V", findings)
            if (type(result.get("elements")) is not int or result.get("elements") != 257 or
                    type(result.get("dispatch_groups_x")) is not int or result.get("dispatch_groups_x") != 5):
                _finding(findings, "v0-fixture-contract", f"V0 {device_key} is not 257-element Add")
            if (result.get("validation_enabled") is not True or
                    type(result.get("validation_errors")) is not int or
                    result.get("validation_errors") != 0):
                _finding(findings, "validation-errors", f"V0 {device_key} lacks clean validation")
            actual_triplet = _verify_triplet(inputs_path, outputs_path, result, 257,
                                             f"V0 {device_key}", findings)
            try:
                inputs = read_f32(inputs_path, 514)
                outputs = read_f32(outputs_path, 771)
                expected = [struct.unpack("<f", struct.pack("<f", inputs[i] + inputs[257 + i]))[0]
                            for i in range(257)]
                golden = outputs[514:]
                golden_check = compare_f32(golden, expected, 1.0e-6, 1.0e-6)
                if not golden_check["passed"]:
                    _finding(findings, "golden-input-mismatch", f"V0 {device_key} golden does not match Add input bytes",
                             max_abs=golden_check["max_abs"])
            except (OSError, ValueError, struct.error) as error:
                _finding(findings, "v0-raw-input-invalid", f"V0 {device_key} Add input recomputation failed: {error}")
            physical_source = dict(result)
            device_info = metadata.get("device_info")
            if isinstance(device_info, dict):
                physical_source.update(device_info)
            summary_hash = _check_inline_device_summary(
                stage, metadata_path.parent, metadata.get("vulkaninfo_summary"),
                device_info.get("vulkaninfo_summary_sha256") if isinstance(device_info, dict) else None,
                f"V0 {device_key}", refs, findings)
            if summary_hash is not None:
                physical_source["vulkaninfo_summary_sha256"] = summary_hash
            physical_source["vulkaninfo_summary"] = metadata.get("vulkaninfo_summary")
            identity = verify_physical_identity(
                stage, metadata_path.parent, physical_source,
                "hardware" if device_key == "nvidia" else "software", refs, findings,
                f"V0 {device_key}")
            _check_logged_validation(metadata_path.parent, "spirv-validation.json",
                                     f"V0 {device_key}", shader_path,
                                     metadata.get("spirv_sha256"), refs, findings)
            if device_key == "nvidia":
                _check_v0_negative_cases(metadata_path.parent, refs, findings)
            records[device_key] = {"metadata": metadata, "result": result,
                                   "identity": identity, "numeric": actual_triplet,
                                   "path": str(metadata_path.relative_to(stage))}
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            _finding(findings, "v0-evidence-invalid", f"V0 {device_key} evidence is invalid: {error}")
    absent = sorted(expected_devices - set(records))
    if absent:
        _finding(findings, "v0-device-coverage", "fresh V0 evidence is missing required ICD(s)", devices=absent)
    if {"nvidia", "llvmpipe"}.issubset(expected_devices) and {"nvidia", "llvmpipe"}.issubset(records):
        nvidia, software = records["nvidia"], records["llvmpipe"]
        for field in ("model_param_sha256", "gpu_func_source_sha256"):
            if nvidia["metadata"].get(field) != software["metadata"].get(field):
                _finding(findings, "v0-cross-device-source-mismatch", f"V0 ICD runs differ in {field}")
        if nvidia.get("numeric", {}).get("input_sha256") != software.get("numeric", {}).get("input_sha256"):
            _finding(findings, "v0-cross-device-input-mismatch", "V0 ICD runs use different input bytes")
    return records


def _check_logged_validation(base: Path, relative: str, label: str,
                             shader_path: Path | None, expected_sha256: object,
                             refs: list[dict], findings: list[dict]) -> None:
    stage = base.parents[4]
    path = _evidence_file(stage, base, f"logs/{relative}",
                          f"{label} validation log", refs, findings)
    if path is None:
        return
    try:
        log = load_json(path)
        if (not isinstance(log, dict) or type(log.get("returncode", log.get("exit_code"))) is not int or
                log.get("returncode", log.get("exit_code")) != 0):
            _finding(findings, "spirv-validation-failed", f"{label} SPIR-V validator did not exit zero")
            return
        argv = log.get("argv")
        if not isinstance(argv, list) or not argv or "spirv-val" not in Path(str(argv[0])).name:
            _finding(findings, "spirv-validation-command", f"{label} validator log does not identify spirv-val")
        else:
            try:
                target = argv.index("--target-env")
            except ValueError:
                target = -1
            if target < 0 or target + 1 >= len(argv) or argv[target + 1] != "vulkan1.1":
                _finding(findings, "spirv-validation-target", f"{label} validator log did not target Vulkan 1.1")
            if (shader_path is None or not argv or not isinstance(argv[-1], str) or
                    Path(argv[-1]).resolve() != shader_path.resolve()):
                _finding(findings, "spirv-validation-path-mismatch",
                         f"{label} validator argv does not identify the audited SPIR-V file")
            elif sha256_file(shader_path) != expected_sha256:
                _finding(findings, "spirv-validation-shader-hash",
                         f"{label} validator SPIR-V bytes differ from the metadata hash")
        text = str(log.get("stdout", "")) + "\n" + str(log.get("stderr", ""))
        if re.search(r"VUID-[A-Za-z0-9-]+|Validation Error|AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:", text, re.I):
            _finding(findings, "spirv-validation-diagnostic", f"{label} validator raw output contains errors")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _finding(findings, "spirv-validation-invalid", f"{label} validator log is invalid: {error}")


def _v0_negative_argv_valid(stage: Path, base: Path, filename: str,
                            record: object, device_name: object,
                            max_storage_range: object) -> bool:
    if not isinstance(record, dict):
        return False
    argv = record.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return False
    runner = str(stage / "test/Vulkan/vulkan_add_runner")
    cpu_library = str(stage / "test/Vulkan/cpu-add/libv0_add.so")
    shader = str(base / "add.spv")
    if filename == "negative-no-device.json":
        return argv == [runner, "--shader", shader, "--cpu-library", cpu_library,
                        "--cpu-symbol", "v0_add", "--device-match",
                        "NCNN_V0_NO_SUCH_DEVICE_9D62", "--evidence",
                        str(base / "negative-no-device.json")]
    if filename == "negative-oversize-buffer.json":
        try:
            requested = int(max_storage_range) + 1
        except (TypeError, ValueError):
            return False
        device_match = "NVIDIA" if base.name == "nvidia" else "llvmpipe"
        required = [runner, "--shader", shader, "--cpu-library", cpu_library,
                    "--cpu-symbol", "v0_add", "--device-match", device_match,
                    "--evidence", str(base / "negative-oversize.json"),
                    "--probe-buffer-bytes", str(requested)]
        return argv == required or argv == required + ["--validation"]
    if filename == "negative-invalid-spirv.json":
        return (len(argv) == 4 and Path(argv[0]).name == "spirv-val" and
                argv[1:3] == ["--target-env", "vulkan1.1"] and
                Path(argv[3]).resolve() == (base / "invalid.spv").resolve())
    return False


def _check_v0_negative_cases(base: Path, refs: list[dict], findings: list[dict]) -> None:
    # Retain exact failure causes and runner inputs; recovery proves negatives did not poison the next call.
    stage = base.parents[4]
    try:
        result = load_json(base / "device-result.json")
        device_name = result.get("device_name")
        max_storage_range = result.get("max_storage_buffer_range")
    except (OSError, AttributeError, ValueError, json.JSONDecodeError):
        result, device_name, max_storage_range = {}, None, None
    expectations = {
        "negative-no-device.json": r"no compute device matched ['\"]NCNN_V0_NO_SUCH_DEVICE_9D62['\"]",
        "negative-oversize-buffer.json": r"requested storage buffer exceeds maxStorageBufferRange",
        "negative-invalid-spirv.json": r"Invalid SPIR-V magic number",
    }
    for filename, pattern in expectations.items():
        path = _evidence_file(stage, base, f"logs/{filename}",
                              f"V0 negative {filename}", refs, findings)
        if path is None:
            continue
        try:
            record = load_json(path)
            status = record.get("returncode", record.get("exit_code"))
            text = str(record.get("stdout", "")) + "\n" + str(record.get("stderr", ""))
            if type(status) is not int or status <= 0 or not re.search(pattern, text, re.I):
                _finding(findings, "v0-negative-cause", f"V0 negative {filename} lacks its fixed failure cause/status")
            if not _v0_negative_argv_valid(stage, base, filename, record,
                                           device_name, max_storage_range):
                _finding(findings, "v0-negative-command", f"V0 negative {filename} command does not identify the expected runner/shader role")
            if filename == "negative-invalid-spirv.json":
                invalid_shader = base / "invalid.spv"
                if not invalid_shader.is_file() or invalid_shader.read_bytes()[:4] != bytes(4):
                    _finding(findings, "v0-negative-shader", "V0 invalid-SPIR-V negative does not use the zeroed SPIR-V magic bytes")
        except (OSError, AttributeError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "v0-negative-invalid", f"V0 negative log {filename} invalid: {error}")
    recovery = _evidence_file(base.parents[4], base, "recovery-result.json",
                              "V0 recovery result", refs, findings)
    recovery_inputs = _evidence_file(base.parents[4], base, "recovery-result.json.inputs.bin",
                                    "V0 recovery inputs", refs, findings)
    recovery_outputs = _evidence_file(base.parents[4], base, "recovery-result.json.outputs.bin",
                                     "V0 recovery output triplet", refs, findings)
    positive_inputs = base / "device-result.json.inputs.bin"
    if recovery_inputs and positive_inputs.is_file() and sha256_file(recovery_inputs) != sha256_file(positive_inputs):
        _finding(findings, "v0-recovery-input-mismatch", "V0 recovery used different input bytes from the positive case")
    if recovery and recovery_inputs and recovery_outputs:
        try:
            result = load_json(recovery)
            _verify_triplet(recovery_inputs, recovery_outputs, result, 257,
                            "V0 post-negative recovery", findings)
            if result.get("validation_errors") != 0 or not all(
                result.get(key, {}).get("passed") is True
                for key in ("cpu_vs_golden", "gpu_vs_golden", "gpu_vs_cpu")):
                _finding(findings, "v0-recovery-failed", "V0 recovery after negative tests did not pass")
        except (OSError, AttributeError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "v0-recovery-invalid", f"V0 recovery result invalid: {error}")


def _v2_conv_spec(case_name: str) -> dict | None:
    key = case_name.removesuffix("-nvidia").removesuffix("-llvmpipe")
    return {
        "conv3x3": {"input_shape": (1, 4, 4), "out_channels": 2, "kernel": 3,
                    "stride": 1, "pad": 1, "bias": True},
        "conv3x3-no-bias": {"input_shape": (1, 4, 4), "out_channels": 2, "kernel": 3,
                             "stride": 1, "pad": 1, "bias": False},
        "conv3x3-stride2": {"input_shape": (1, 4, 4), "out_channels": 2, "kernel": 3,
                             "stride": 2, "pad": 1, "bias": True},
        "conv1x1": {"input_shape": (2, 3, 5), "out_channels": 3, "kernel": 1,
                    "stride": 2, "pad": 0, "bias": True},
    }.get(key)


def _recompute_v2_golden(operation: str, inputs: list[float], count: int,
                         record: dict, case_name: str = "",
                         model_bin: Path | None = None) -> list[float] | None:
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    if operation == "add" and len(inputs) == 2 * count:
        return [f32(inputs[i] + inputs[count + i]) for i in range(count)]
    if operation == "relu" and len(inputs) == count:
        return [max(value, 0.0) for value in inputs]
    if operation in {"copy", "identity", "reshape"} and len(inputs) == count:
        return inputs
    if operation == "transpose" and len(inputs) == count:
        shape = record.get("input_shape", [51, 5, 1])
        permutation = record.get("permutation", [1, 0, 2])
        output_shape = [shape[permutation[axis]] for axis in range(len(shape))]
        expected = []
        for output_index in range(count):
            coords = [0] * len(shape)
            quotient = output_index
            for axis in range(len(output_shape) - 1, -1, -1):
                coords[axis] = quotient % output_shape[axis]
                quotient //= output_shape[axis]
            source_coords = [0] * len(shape)
            for axis, source_axis in enumerate(permutation):
                source_coords[source_axis] = coords[axis]
            source_index = 0
            for extent, coord in zip(shape, source_coords):
                source_index = source_index * extent + coord
            expected.append(inputs[source_index])
        return expected
    if operation == "conv2d" and model_bin is not None:
        spec = _v2_conv_spec(case_name)
        if spec is None:
            return None
        channels, height, width = spec["input_shape"]
        outputs, kernel, stride, padding = (spec["out_channels"], spec["kernel"],
                                            spec["stride"], spec["pad"])
        output_height = (height + 2 * padding - kernel) // stride + 1
        output_width = (width + 2 * padding - kernel) // stride + 1
        if len(inputs) < channels * height * width or count != outputs * output_height * output_width:
            return None
        data = model_bin.read_bytes()
        weight_count = outputs * channels * kernel * kernel
        bias_count = outputs if spec["bias"] else 0
        if len(data) != 4 + (weight_count + bias_count) * 4 or int.from_bytes(data[:4], "little") != 0:
            return None
        values = list(struct.unpack(f"<{weight_count + bias_count}f", data[4:]))
        oihw, biases = values[:weight_count], values[weight_count:]
        if not biases:
            biases = [0.0] * outputs
        golden = []
        source = inputs[:channels * height * width]
        for output_channel in range(outputs):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    value = biases[output_channel]
                    for kernel_y in range(kernel):
                        for kernel_x in range(kernel):
                            input_y = output_y * stride + kernel_y - padding
                            input_x = output_x * stride + kernel_x - padding
                            if 0 <= input_y < height and 0 <= input_x < width:
                                for input_channel in range(channels):
                                    input_index = input_channel * height * width + input_y * width + input_x
                                    weight_index = (((output_channel * channels + input_channel) * kernel + kernel_y) * kernel + kernel_x)
                                    value = f32(value + f32(source[input_index] * oihw[weight_index]))
                    golden.append(value)
        return golden
    return None


def _audit_v2(stage: Path, expected_devices: set[str], refs: list[dict],
              findings: list[dict]) -> dict:
    root = stage / "test/Vulkan/evidence/v2"
    records = {}
    metadata_files = sorted(root.glob("*/metadata.json")) if root.is_dir() else []
    if not metadata_files:
        _finding(findings, "v2-evidence-missing", "fresh V2 metadata is absent")
        return records
    for metadata_path in metadata_files:
        case_dir = metadata_path.parent
        try:
            metadata = load_json(metadata_path)
            _evidence_file(stage, case_dir, "metadata.json", f"V2 {case_dir.name} metadata", refs, findings)
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
            operation = metadata.get("operation")
            device_records = metadata.get("device_evidence")
            if not isinstance(device_records, dict) or not device_records:
                _finding(findings, "v2-device-results-missing", f"V2 {case_dir.name} has no executed seeds")
                continue
            expected_class = "hardware" if "nvidia" in case_dir.name else "software" if "llvmpipe" in case_dir.name else None
            if expected_class is None:
                _finding(findings, "v2-device-directory", "V2 path does not identify configured test selector")
            case_identity = {key: metadata.get(key) for key in
                             ("fixture_id", "model_param_sha256", "model_bin_sha256",
                              "cpu_reference_library_sha256")}
            model_files = metadata.get("model_files", {})
            model_file_paths = {}
            for role, hash_key in (("param", "model_param_sha256"), ("bin", "model_bin_sha256")):
                relative = model_files.get(role) if isinstance(model_files, dict) else None
                input_file = _evidence_file(stage, case_dir, relative,
                                            f"V2 {case_dir.name} model {role}", refs, findings)
                model_file_paths[role] = input_file
                _check_digest_claim(metadata.get(hash_key), input_file,
                                    f"V2 {case_dir.name} model {role}", findings)
            case_name = case_dir.name.removesuffix("-nvidia").removesuffix("-llvmpipe")
            cpu_fixtures = {
                "add": ("cpu-add", "libv0_add.so"),
                "relu": ("cpu-relu", "libv2_relu.so"),
                "copy": ("cpu-identity", "libv2_identity.so"),
                "identity": ("cpu-identity", "libv2_identity.so"),
                "transpose": ("cpu-transpose", "libv2_transpose.so"),
                "reshape": ("cpu-reshape", "libv2_reshape.so"),
                "conv3x3": ("cpu-convolution-3x3", "libv2_conv3x3.so"),
                "conv3x3-no-bias": ("cpu-convolution-3x3-no-bias", "libv2_conv3x3_no_bias.so"),
                "conv3x3-stride2": ("cpu-convolution-3x3-stride2", "libv2_conv3x3_stride2.so"),
                "conv1x1": ("cpu-convolution-1x1", "libv2_conv1x1.so"),
            }
            cpu_spec = cpu_fixtures.get(case_name)
            if cpu_spec:
                cpu_library = stage / "test/Vulkan" / cpu_spec[0] / cpu_spec[1]
                _evidence_file(stage, cpu_library.parent, cpu_library.name,
                               f"V2 {case_dir.name} fresh CPU AOT library", refs, findings)
                _check_digest_claim(metadata.get("cpu_reference_library_sha256"),
                                    cpu_library if cpu_library.is_file() else None,
                                    f"V2 {case_dir.name} CPU AOT library", findings)
            else:
                _finding(findings, "v2-cpu-fixture", f"V2 {case_dir.name} has no known paired CPU fixture")
            for seed, claim in device_records.items():
                if not isinstance(claim, dict):
                    _finding(findings, "v2-device-record", f"V2 seed {seed} record is malformed")
                    continue
                result_path = _evidence_file(stage, case_dir, f"device-result-seed-{seed}.json",
                                             f"V2 {case_dir.name} seed {seed} result", refs, findings)
                input_path = _evidence_file(stage, case_dir,
                                            f"device-result-seed-{seed}.json.inputs.bin",
                                            f"V2 {case_dir.name} seed {seed} input", refs, findings)
                output_path = _evidence_file(stage, case_dir,
                                             f"device-result-seed-{seed}.json.outputs.bin",
                                             f"V2 {case_dir.name} seed {seed} triplet", refs, findings)
                if result_path is None or input_path is None or output_path is None:
                    continue
                result = load_json(result_path)
                count = result.get("elements")
                if type(count) is not int or count <= 0:
                    _finding(findings, "v2-element-count", f"V2 {case_dir.name} has invalid element count")
                    continue
                _check_digest_claim(claim.get("inputs_sha256"), input_path,
                                    f"V2 {case_dir.name} seed {seed} input", findings)
                _check_digest_claim(claim.get("outputs_sha256"), output_path,
                                    f"V2 {case_dir.name} seed {seed} output", findings)
                triplet = _verify_triplet(input_path, output_path, result, count,
                                          f"V2 {case_dir.name} seed {seed}", findings)
                try:
                    raw_inputs = read_f32(input_path)
                    raw_outputs = read_f32(output_path, count * 3)
                    expected = _recompute_v2_golden(operation, raw_inputs, count, result,
                                                     case_dir.name, model_file_paths.get("bin"))
                    if expected is not None:
                        golden_check = compare_f32(raw_outputs[2 * count:], expected,
                                                   result["tolerance"]["atol"],
                                                   result["tolerance"]["rtol"])
                        if not golden_check["passed"]:
                            _finding(findings, "golden-input-mismatch",
                                     f"V2 {case_dir.name} scalar golden differs from input recomputation",
                                     max_abs=golden_check["max_abs"])
                    else:
                        _finding(findings, "v2-golden-unverifiable",
                                 f"V2 {case_dir.name} operation-specific golden cannot be recomputed")
                except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
                    _finding(findings, "v2-raw-invalid", f"V2 {case_dir.name} raw records invalid: {error}")
                if (result.get("validation_enabled") is not True or
                        type(result.get("validation_errors")) is not int or
                        result.get("validation_errors") != 0):
                    _finding(findings, "validation-errors", f"V2 {case_dir.name} seed {seed} has validation errors")
                physical_source = dict(result)
                if isinstance(claim.get("device_info"), dict):
                    physical_source.update(claim["device_info"])
                if isinstance(metadata.get("device_info"), dict):
                    physical_source.update(metadata["device_info"])
                summary_name = metadata.get("vulkaninfo_summary")
                if summary_name:
                    summary_path = _evidence_file(stage, case_dir, summary_name,
                                                  f"V2 {case_dir.name} raw vulkaninfo summary", refs, findings)
                    if summary_path is not None:
                        declared_summary_hash = (metadata.get("device_info", {}).get("vulkaninfo_summary_sha256")
                                                 if isinstance(metadata.get("device_info"), dict) else None)
                        _check_digest_claim(declared_summary_hash, summary_path,
                                            f"V2 {case_dir.name} declared vulkaninfo summary", findings)
                        physical_source["vulkaninfo_summary_sha256"] = sha256_file(summary_path)
                physical_source["vulkaninfo_summary"] = summary_name
                if expected_class:
                    verify_physical_identity(stage, case_dir, physical_source,
                                             expected_class, refs, findings,
                                             f"V2 {case_dir.name} seed {seed}")
                records.setdefault(case_dir.name, {"identity": case_identity,
                                                    "operation": operation,
                                                    "seeds": {}})["seeds"][str(seed)] = {
                    "result": result, "numeric": triplet,
                    "path": str(result_path.relative_to(stage))}
            source_copy = _find_hash_file(stage, metadata.get("gpu_kernel_source_sha256"))
            if source_copy is None:
                _finding(findings, "v2-source-hash-missing",
                         f"V2 {case_dir.name} GPU kernel source IR bytes are not present in the fresh stage")
            else:
                _evidence_file(stage, source_copy.parent, source_copy.name,
                               f"V2 {case_dir.name} GPU kernel source IR", refs, findings)
            for filename, claim_key in (("execution-plan.json", "execution_plan_sha256"),
                                        ("gpu-kernels.mlir", "gpu_ir_sha256"),
                                        ("gpu-binary.mlir", "gpu_binary_ir_sha256"),
                                        ("kernel.spv", "spirv_sha256"),
                                        ("kernel.spvasm", "spirv_disassembly_sha256")):
                artifact = _evidence_file(stage, case_dir, filename,
                                          f"V2 {case_dir.name} {filename}", refs, findings)
                if artifact:
                    _check_digest_claim(metadata.get(claim_key), artifact,
                                        f"V2 {case_dir.name} {filename}", findings)
                    if filename == "execution-plan.json":
                        try:
                            plan = load_json(artifact)
                            sizes = [resource.get("byte_size") for resource in plan.get("resources", [])]
                            if any(not isinstance(size, int) or size < 0 or size > 128 * 1024 * 1024 for size in sizes):
                                _finding(findings, "v2-single-buffer-cap", f"V2 {case_dir.name} plan has a buffer over 128 MiB")
                            if sum(size for size in sizes if isinstance(size, int)) > 512 * 1024 * 1024:
                                _finding(findings, "v2-plan-resource-cap", f"V2 {case_dir.name} planned allocations exceed 512 MiB")
                        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                            _finding(findings, "v2-plan-invalid", f"V2 {case_dir.name} plan JSON invalid: {error}")
            _check_logged_validation(case_dir, "spirv-validation.json",
                                     f"V2 {case_dir.name}", case_dir / "kernel.spv",
                                     metadata.get("spirv_sha256"), refs, findings)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            _finding(findings, "v2-evidence-invalid", f"V2 {case_dir.name} evidence is invalid: {error}")
    found_classes = {"nvidia" if "nvidia" in key else "llvmpipe" for key in records}
    missing = sorted(expected_devices - found_classes)
    if missing:
        _finding(findings, "v2-device-coverage", "fresh V2 evidence lacks required device(s)", devices=missing)
    by_fixture = {}
    for record in records.values():
        identity = record["identity"]
        key = (identity.get("fixture_id"), identity.get("model_param_sha256"),
               identity.get("model_bin_sha256"))
        by_fixture.setdefault(key, []).append(record)
    for copies in by_fixture.values():
        if len(copies) > 1 and any(copy["identity"] != copies[0]["identity"] for copy in copies[1:]):
            _finding(findings, "v2-input-identity-mismatch",
                     "V2 ICD runs do not share a source fixture/CPU artifact identity")
        if {"nvidia", "llvmpipe"}.issubset(expected_devices) and len(copies) > 1:
            nvidia = next((item for item in copies if "nvidia" in item.get("path", "")), None)
            software = next((item for item in copies if "llvmpipe" in item.get("path", "")), None)
            if nvidia and software:
                left, right = nvidia["seeds"], software["seeds"]
                if set(left) != set(right) or any(left[seed].get("numeric", {}).get("input_sha256") != right[seed].get("numeric", {}).get("input_sha256") for seed in set(left) & set(right)):
                    _finding(findings, "v2-input-cross-device-mismatch",
                             "V2 seed input bytes differ between ICDs for the same fixture")
    return records


def _float_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _check_v3_output_claim(claim: dict, cpu: list[float], runtime: list[float],
                           golden: list[float], label: str, findings: list[dict]) -> None:
    for key, actual, expected in (("cpu_vs_golden", cpu, golden),
                                  ("runtime_vs_golden", runtime, golden),
                                  ("cpu_vs_runtime", cpu, runtime)):
        metric = compare_f32(actual, expected, 1.0e-6, 1.0e-6)
        pass_field = key + "_pass"
        abs_field = key + "_max_abs"
        if not metric["passed"] or claim.get(pass_field) is not True:
            _finding(findings, "v3-numeric-failed", f"{label} {key} exceeds fixed budget or claim fails",
                     max_abs=metric["max_abs"])
        declared = claim.get(abs_field)
        if not isinstance(declared, (int, float)) or abs(float(declared) - metric["max_abs"]) > max(1e-12, metric["max_abs"] * 1e-7):
            _finding(findings, "v3-numeric-claim-tampered", f"{label} {abs_field} differs from raw tensors")
    for key, values in (("cpu_sha256", cpu), ("runtime_sha256", runtime),
                        ("golden_sha256", golden)):
        digest = claim.get(key)
        if digest != sha256_bytes(_float_bytes(values)):
            _finding(findings, "v3-output-hash-mismatch", f"{label} {key} differs from output bytes")


def _read_slice(values: list[float], offset: object, count: int, label: str,
                findings: list[dict]) -> list[float] | None:
    if not isinstance(offset, int) or offset < 0 or offset + count > len(values):
        _finding(findings, "v3-offset-invalid", f"{label} FP32 offset is outside raw evidence")
        return None
    return values[offset:offset + count]


def _absolute_file_in_stage(stage: Path, value: object, label: str,
                            refs: list[dict], findings: list[dict]) -> Path | None:
    if not isinstance(value, str) or not value:
        _finding(findings, "absolute-evidence-path-missing", f"{label} path is missing")
        return None
    candidate = Path(value).resolve()
    try:
        relative = candidate.relative_to(stage.resolve())
    except ValueError:
        _finding(findings, "historical-evidence-path", f"{label} resolves outside the fresh stage", path=value)
        return None
    return _evidence_file(stage, candidate.parent, candidate.name, label, refs, findings)


def _validate_record_log(stage: Path, base: Path, relative: str, label: str,
                         refs: list[dict], findings: list[dict]) -> None:
    path = _evidence_file(stage, base, relative, label, refs, findings)
    if path is None:
        return
    try:
        record = load_json(path)
        if not isinstance(record, dict) or record.get("returncode", record.get("exit_code")) != 0:
            _finding(findings, "validator-failed", f"{label} did not exit zero")
        text = str(record.get("stdout", "")) + "\n" + str(record.get("stderr", ""))
        if re.search(r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:|VUID-[A-Za-z0-9-]+|Validation Error", text, re.I):
            _finding(findings, "validator-log-diagnostic", f"{label} contains validation/sanitizer diagnostics")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _finding(findings, "validator-log-invalid", f"{label} log invalid: {error}")


def _audit_v3(stage: Path, expected_devices: set[str], refs: list[dict],
              findings: list[dict]) -> dict:
    root = stage / "test/Vulkan/evidence/v3"
    records = {}
    metadata_files = sorted(root.glob("*/metadata.json")) if root.is_dir() else []
    if not metadata_files:
        _finding(findings, "v3-evidence-missing", "fresh V3 runtime metadata is absent")
        return records
    for metadata_path in metadata_files:
        case_dir = metadata_path.parent
        try:
            metadata = load_json(metadata_path)
            _evidence_file(stage, case_dir, "metadata.json", f"V3 {case_dir.name} metadata", refs, findings)
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
            runtime_path = _evidence_file(stage, case_dir, "runtime-evidence.json",
                                          f"V3 {case_dir.name} Add→ReLU result", refs, findings)
            weight_path = _evidence_file(stage, case_dir, "weight-runtime-evidence.json",
                                         f"V3 {case_dir.name} Conv result", refs, findings)
            inputs_path = _evidence_file(stage, case_dir, "runtime-evidence.json.inputs.bin",
                                         f"V3 {case_dir.name} dual input raw", refs, findings)
            outputs_path = _evidence_file(stage, case_dir, "runtime-evidence.json.outputs.bin",
                                          f"V3 {case_dir.name} dual output raw", refs, findings)
            weight_inputs_path = _evidence_file(stage, case_dir, "weight-runtime-evidence.json.inputs.bin",
                                                f"V3 {case_dir.name} Conv input raw", refs, findings)
            weight_outputs_path = _evidence_file(stage, case_dir, "weight-runtime-evidence.json.outputs.bin",
                                                 f"V3 {case_dir.name} Conv output raw", refs, findings)
            if not all((runtime_path, weight_path, inputs_path, outputs_path,
                        weight_inputs_path, weight_outputs_path)):
                continue
            runtime = load_json(runtime_path)
            weight = load_json(weight_path)
            if not isinstance(runtime, dict) or not isinstance(weight, dict):
                raise ValueError("V3 result JSON is malformed")
            expected_class = "hardware" if "nvidia" in case_dir.name else "software" if "llvmpipe" in case_dir.name else None
            if expected_class is None:
                _finding(findings, "v3-device-directory", "V3 evidence path does not identify configured device selector")
            physical_source = dict(runtime)
            if isinstance(metadata.get("device_info"), dict):
                physical_source.update(metadata["device_info"])
            physical_source["vulkaninfo_summary"] = metadata.get("vulkaninfo_summary")
            if expected_class:
                identity = verify_physical_identity(stage, case_dir, physical_source,
                                                    expected_class, refs, findings,
                                                    f"V3 {case_dir.name}")
            else:
                identity = None
            for result, name in ((runtime, "runtime"), (weight, "weight")):
                if result.get("validation_enabled") is not True or result.get("validation_errors") != 0:
                    _finding(findings, "validation-errors", f"V3 {case_dir.name} {name} validation failed")
            _check_digest_claim(metadata.get("runtime_inputs_sha256"), inputs_path,
                                f"V3 {case_dir.name} input bundle", findings)
            _check_digest_claim(metadata.get("runtime_outputs_sha256"), outputs_path,
                                f"V3 {case_dir.name} output bundle", findings)
            _check_digest_claim(metadata.get("weight_runtime_inputs_sha256"), weight_inputs_path,
                                f"V3 {case_dir.name} Conv input", findings)
            _check_digest_claim(metadata.get("weight_runtime_outputs_sha256"), weight_outputs_path,
                                f"V3 {case_dir.name} Conv output", findings)
            count = 257
            raw_inputs = read_f32(inputs_path)
            raw_outputs = read_f32(outputs_path)
            dual = runtime.get("dual_output", {})
            cpu_aot = _absolute_file_in_stage(stage, dual.get("cpu_aot_library"),
                                              f"V3 {case_dir.name} CPU AOT library", refs, findings)
            _check_digest_claim(dual.get("cpu_aot_library_sha256"), cpu_aot,
                                f"V3 {case_dir.name} CPU AOT library", findings)
            dual_plan = _absolute_file_in_stage(stage, dual.get("plan"),
                                                f"V3 {case_dir.name} dual-output execution plan", refs, findings)
            if dual_plan:
                try:
                    dual_plan_json = load_json(dual_plan)
                    if dual_plan_json.get("outputs") != [3, 4]:
                        _finding(findings, "v3-dual-plan", f"V3 {case_dir.name} dual plan outputs are not [3,4]")
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                    _finding(findings, "v3-dual-plan-invalid", f"V3 {case_dir.name} dual plan is invalid: {error}")
            conv_cpu = stage / "test/Vulkan/cpu-convolution-3x3/libv2_conv3x3.so"
            _evidence_file(stage, conv_cpu.parent, conv_cpu.name,
                           f"V3 {case_dir.name} Conv CPU AOT library", refs, findings)
            _check_digest_claim(metadata.get("conv_cpu_library_sha256"),
                                conv_cpu if conv_cpu.is_file() else None,
                                f"V3 {case_dir.name} Conv CPU AOT library", findings)
            artifact_claims = (
                ("runtime-evidence.json.execution-plan.json", "runtime_execution_plan_sha256"),
                ("prepare-add/execution-plan.json", "add_execution_plan_sha256"),
                ("prepare-relu/execution-plan.json", "relu_execution_plan_sha256"),
                ("prepare-conv2d/execution-plan.json", "conv_execution_plan_sha256"),
                ("prepare-add/kernel.spv", "add_spirv_sha256"),
                ("prepare-relu/kernel.spv", "relu_spirv_sha256"),
                ("prepare-conv2d/kernel.spv", "conv_spirv_sha256"),
            )
            for relative, digest_key in artifact_claims:
                artifact = _evidence_file(stage, case_dir, relative,
                                          f"V3 {case_dir.name} {relative}", refs, findings)
                _check_digest_claim(metadata.get(digest_key), artifact,
                                    f"V3 {case_dir.name} {relative}", findings)
                if artifact and relative.endswith("execution-plan.json"):
                    try:
                        plan = load_json(artifact)
                        sizes = [resource.get("byte_size") for resource in plan.get("resources", [])]
                        if any(not isinstance(size, int) or size < 0 or size > 128 * 1024 * 1024 for size in sizes):
                            _finding(findings, "v3-single-buffer-cap", f"V3 {case_dir.name} plan resource exceeds 128 MiB")
                        if sum(size for size in sizes if isinstance(size, int)) > 512 * 1024 * 1024:
                            _finding(findings, "v3-plan-resource-cap", f"V3 {case_dir.name} plan resources exceed 512 MiB")
                    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                        _finding(findings, "v3-plan-invalid", f"V3 {case_dir.name} execution plan invalid: {error}")
            for prep in ("prepare-add", "prepare-relu", "prepare-conv2d"):
                _validate_record_log(stage, case_dir, f"{prep}/logs/spirv-validation.json",
                                     f"V3 {case_dir.name} {prep} SPIR-V validation", refs, findings)
            _validate_record_log(stage, case_dir, "logs/v3-persistent-runtime.json",
                                 f"V3 {case_dir.name} runtime execution", refs, findings)
            _validate_record_log(stage, case_dir, "logs/v3-weight-runtime.json",
                                 f"V3 {case_dir.name} weighted execution", refs, findings)
            if dual.get("plan_outputs") != [3, 4]:
                _finding(findings, "v3-dual-plan", f"V3 {case_dir.name} plan does not expose sum/ReLU outputs")
            expected_roles = {"host_validation_retry", "routine", "input_flush_healthy_context",
                              "input_flush_recreated_context", "second_output_invalidate_healthy_context",
                              "second_output_invalidate_recreated_context"}
            trials = dual.get("trials", [])
            roles = {item.get("role") for item in trials if isinstance(item, dict)}
            if not expected_roles.issubset(roles) or dual.get("trial_count") != len(trials):
                _finding(findings, "v3-trial-coverage", f"V3 {case_dir.name} misses direct runtime trial roles",
                         missing=sorted(expected_roles - roles))
            lifecycle = dual.get("lifecycle", {})
            required_lifecycle = ("host_validation_retry", "flush_failure_all_sentinels_unchanged",
                                  "flush_poison_retry_rejected", "healthy_context_after_flush_failure",
                                  "flush_destroy_create_recovery", "second_invalidate_failure_all_sentinels_unchanged",
                                  "second_invalidate_poison_retry_rejected",
                                  "healthy_context_after_invalidate_failure",
                                  "invalidate_destroy_create_recovery")
            if not isinstance(lifecycle, dict) or any(lifecycle.get(key) is not True for key in required_lifecycle):
                _finding(findings, "v3-lifecycle-failed", f"V3 {case_dir.name} missing a required failure/recovery assertion")
            if dual.get("sentinel_elements_checked_total", 0) < 2 * 2 * count:
                _finding(findings, "v3-sentinel-count", f"V3 {case_dir.name} sentinel checks are insufficient")
            fail_counts = dual.get("failure_counts", {})
            if fail_counts.get("input_flush_events") != 1 or fail_counts.get("output_invalidate_events") != 2 or fail_counts.get("second_output_invalidate_injected") != 1:
                _finding(findings, "v3-injection-count", f"V3 {case_dir.name} flush/invalidate injection counts differ")
            try:
                input_values = read_f32(inputs_path)
                output_values = read_f32(outputs_path)
                for trial in trials:
                    input_offset = trial.get("input_binary_offset_floats")
                    output_offset = trial.get("output_binary_offset_floats")
                    trial_input = _read_slice(input_values, input_offset, 2 * count,
                                              f"V3 {case_dir.name} {trial.get('role')} input", findings)
                    trial_output = _read_slice(output_values, output_offset, 6 * count,
                                               f"V3 {case_dir.name} {trial.get('role')} output", findings)
                    if trial_input is None or trial_output is None:
                        continue
                    left, right = trial_input[:count], trial_input[count:]
                    sum_golden = [struct.unpack("<f", struct.pack("<f", a + b))[0]
                                  for a, b in zip(left, right)]
                    relu_golden = [max(value, 0.0) for value in sum_golden]
                    arrays = [trial_output[i * count:(i + 1) * count] for i in range(6)]
                    sum_cpu, sum_runtime, sum_recorded = arrays[:3]
                    relu_cpu, relu_runtime, relu_recorded = arrays[3:]
                    if not compare_f32(sum_recorded, sum_golden, 1e-6, 1e-6)["passed"] or not compare_f32(relu_recorded, relu_golden, 1e-6, 1e-6)["passed"]:
                        _finding(findings, "v3-golden-input-mismatch", f"V3 {case_dir.name} scalar goldens differ from raw inputs")
                    _check_v3_output_claim(trial.get("sum", {}), sum_cpu, sum_runtime, sum_recorded,
                                           f"V3 {case_dir.name} {trial.get('role')} sum", findings)
                    _check_v3_output_claim(trial.get("relu", {}), relu_cpu, relu_runtime, relu_recorded,
                                           f"V3 {case_dir.name} {trial.get('role')} relu", findings)
            except (OSError, ValueError, TypeError, struct.error) as error:
                _finding(findings, "v3-raw-invalid", f"V3 {case_dir.name} raw records invalid: {error}")
            weight_count = weight.get("output_elements", weight.get("elements", None))
            if not isinstance(weight_count, int) or weight_count <= 0:
                # The current Conv fixture has a fixed [2,4,4] output.
                weight_count = 32
            try:
                conv_inputs = read_f32(weight_inputs_path)
                conv_outputs = read_f32(weight_outputs_path, weight_count * 3)
                cpu_conv, gpu_conv, golden_conv = (conv_outputs[:weight_count],
                                                   conv_outputs[weight_count:2 * weight_count],
                                                   conv_outputs[2 * weight_count:])
                weight_metrics = {"cpu_vs_golden": compare_f32(cpu_conv, golden_conv, 1e-6, 1e-6),
                                  "gpu_vs_golden": compare_f32(gpu_conv, golden_conv, 1e-6, 1e-6),
                                  "gpu_vs_cpu": compare_f32(gpu_conv, cpu_conv, 1e-6, 1e-6)}
                for key, metric in weight_metrics.items():
                    if not metric["passed"]:
                        _finding(findings, "v3-weight-numeric", f"V3 {case_dir.name} Conv {key} exceeds fixed budget")
                for key, claimed_field in (("cpu_vs_golden", "cpu_max_abs_error"),
                                           ("gpu_vs_golden", "gpu_max_abs_error")):
                    claimed = weight.get(claimed_field)
                    if not isinstance(claimed, (int, float)) or abs(float(claimed) - weight_metrics[key]["max_abs"]) > max(1e-7, weight_metrics[key]["max_abs"] * 1e-5):
                        _finding(findings, "v3-weight-claim-tampered", f"V3 {case_dir.name} {claimed_field} differs from raw tensors")
                weight_numeric = {"input_sha256": sha256_file(weight_inputs_path),
                                  "output_sha256": sha256_file(weight_outputs_path),
                                  "input_elements": len(conv_inputs), "output_elements": weight_count,
                                  "metrics": weight_metrics}
                source_bin = stage / "test/Vulkan/convolution-3x3.bin"
                if not source_bin.is_file() or sha256_file(source_bin) != metadata.get("conv_model_bin_sha256"):
                    source_bin = stage / "gate-source-inputs/convolution-3x3.bin"
                _evidence_file(stage, source_bin.parent, source_bin.name,
                               f"V3 {case_dir.name} source Conv weights", refs, findings)
                _check_digest_claim(metadata.get("conv_model_bin_sha256"),
                                    source_bin if source_bin.is_file() else None,
                                    f"V3 {case_dir.name} Conv weight source", findings)
                if source_bin.is_file() and source_bin.stat().st_size == 4 + 20 * 4 and source_bin.read_bytes()[:4] == bytes(4) and len(conv_inputs) == 16 and weight_count == 32:
                    weight_values = list(struct.unpack("<20f", source_bin.read_bytes()[4:]))
                    hwcf = [0.0] * 18
                    for out_channel in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                hwcf[(ky * 3 + kx) * 2 + out_channel] = weight_values[out_channel * 9 + ky * 3 + kx]
                    bias = weight_values[18:]
                    golden = []
                    for out_channel in range(2):
                        for out_y in range(4):
                            for out_x in range(4):
                                value = bias[out_channel]
                                for ky in range(3):
                                    for kx in range(3):
                                        iy, ix = out_y + ky - 1, out_x + kx - 1
                                        if 0 <= iy < 4 and 0 <= ix < 4:
                                            product = struct.unpack("<f", struct.pack("<f", conv_inputs[iy * 4 + ix] * hwcf[(ky * 3 + kx) * 2 + out_channel]))[0]
                                            value = struct.unpack("<f", struct.pack("<f", value + product))[0]
                                golden.append(value)
                    if not compare_f32(golden_conv, golden, 1e-6, 1e-6)["passed"]:
                        _finding(findings, "v3-conv-golden-input-mismatch", f"V3 {case_dir.name} scalar Conv golden differs from source input/weights")
                else:
                    _finding(findings, "v3-conv-golden-unverifiable", f"V3 {case_dir.name} Conv source weights are missing or malformed")
            except (OSError, ValueError, TypeError, struct.error) as error:
                _finding(findings, "v3-weight-raw-invalid", f"V3 {case_dir.name} Conv raw records invalid: {error}")
                weight_numeric = None
            if weight.get("operation") != "conv2d" or weight.get("weight_upload_count") != 2 or weight.get("repeated_calls") != 3:
                _finding(findings, "v3-weight-lifecycle", f"V3 {case_dir.name} persistent weight contract failed")
            peak = weight.get("peak_memory_bytes")
            if not isinstance(peak, int) or not 0 < peak <= 512 * 1024 * 1024:
                _finding(findings, "v3-peak-memory", f"V3 {case_dir.name} Conv allocation peak is absent or exceeds cap")
            for label, result in (("runtime", runtime), ("weight", weight)):
                if result.get("allocation_budget_bytes", 512 * 1024 * 1024) > 512 * 1024 * 1024:
                    _finding(findings, "resource-budget-exceeded", f"V3 {case_dir.name} {label} allocation cap exceeds 512 MiB")
            records[case_dir.name] = {"identity": identity, "runtime": runtime,
                                      "weight": weight, "metadata": metadata,
                                      "numeric": weight_numeric,
                                      "path": str(metadata_path.relative_to(stage))}
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            _finding(findings, "v3-evidence-invalid", f"V3 {case_dir.name} evidence is invalid: {error}")
    found = {"nvidia" if "nvidia" in key else "llvmpipe" for key in records}
    missing = sorted(expected_devices - found)
    if missing:
        _finding(findings, "v3-device-coverage", "fresh V3 evidence lacks required device(s)", devices=missing)
    if {"nvidia", "llvmpipe"}.issubset(expected_devices) and {"nvidia", "llvmpipe"}.issubset(records):
        nvidia = records["nvidia"]["metadata"]
        software = records["llvmpipe"]["metadata"]
        for field in ("cpu_library_sha256", "conv_cpu_library_sha256", "conv_model_param_sha256",
                      "conv_model_bin_sha256", "runtime_inputs_sha256", "weight_runtime_inputs_sha256"):
            if nvidia.get(field) != software.get(field):
                _finding(findings, "v3-cross-device-input-mismatch", f"V3 ICD runs differ in {field}")
        left_trials = {item.get("role"): item for item in records["nvidia"]["runtime"].get("dual_output", {}).get("trials", [])}
        right_trials = {item.get("role"): item for item in records["llvmpipe"]["runtime"].get("dual_output", {}).get("trials", [])}
        if set(left_trials) != set(right_trials):
            _finding(findings, "v3-cross-device-trial-mismatch", "V3 runtime trial roles differ between ICDs")
        for role in set(left_trials) & set(right_trials):
            for output in ("sum", "relu"):
                if left_trials[role].get(output, {}).get("cpu_sha256") != right_trials[role].get(output, {}).get("cpu_sha256"):
                    _finding(findings, "v3-cross-device-cpu-mismatch", f"V3 {role} {output} CPU baseline differs between ICDs")
    return records


def _verify_raw_tensor_record(stage: Path, base: Path, record: object,
                             expected: list[float] | None, label: str,
                             refs: list[dict], findings: list[dict]) -> tuple[Path | None, list[float] | None]:
    if not isinstance(record, dict):
        _finding(findings, "public-io-record", f"{label} record is missing")
        return None, None
    path = _evidence_file(stage, base, record.get("file"), label, refs, findings)
    if path is None:
        return None, None
    _check_digest_claim(record.get("sha256"), path, label, findings)
    try:
        raw = read_f32(path)
        values = record.get("values")
        if not isinstance(values, list) or len(values) != len(raw):
            _finding(findings, "public-io-values", f"{label} values do not match raw tensor length")
        elif _float_bytes([float(value) for value in values]) != _float_bytes(raw):
            _finding(findings, "public-io-tampered", f"{label} JSON values differ from raw FP32 bytes")
        if expected is not None and not compare_f32(raw, expected, 0.0, 0.0)["passed"]:
            _finding(findings, "public-io-oracle", f"{label} differs from recomputed oracle")
        return path, raw
    except (OSError, ValueError, TypeError, struct.error) as error:
        _finding(findings, "public-io-invalid", f"{label} raw tensor is invalid: {error}")
        return path, None


def _audit_current_host_contract(manifest: object, label: str,
                                  findings: list[dict]) -> None:
    if (not isinstance(manifest, dict) or
            manifest.get("host_implementation") != "model-fixed-run" or
            type(manifest.get("host_codegen_version")) is not int or
            manifest["host_codegen_version"] != 2 or
            manifest.get("host_source_codegen_version") !=
            "ncnn-vulkan-fixed-run-codegen/v1" or
            manifest.get("execution_plan_usage") != "build-evidence-only"):
        _finding(findings, "host-implementation-contract",
                 f"{label} must identify the actual model-fixed-run/codegen v2 "
                 "contract with a build-evidence-only plan")


def _check_product_artifact_set(stage: Path, base: Path, fixture: dict,
                                refs: list[dict], findings: list[dict]) -> None:
    vk_dir = base / "vulkan"
    cpu_dir = base / "cpu"
    mapping = {
        "cpu_library_sha256": cpu_dir / "libpaired_relu.so",
        "cpu_header_sha256": cpu_dir / "paired_relu.h",
        "cpu_manifest_sha256": cpu_dir / "paired_relu.json",
        "vulkan_library_sha256": vk_dir / "libpaired_relu_vulkan.so",
        "vulkan_header_sha256": vk_dir / "paired_relu_vulkan.h",
        "vulkan_manifest_sha256": vk_dir / "paired_relu_vulkan.json",
        "vulkan_plan_sha256": vk_dir / "paired_relu_vulkan.plan.json",
    }
    for field, path in mapping.items():
        _evidence_file(stage, path.parent, path.name, f"product {field}", refs, findings)
        _check_digest_claim(fixture.get(field), path if path.is_file() else None,
                            f"product {field}", findings)
    manifest_path = vk_dir / "paired_relu_vulkan.json"
    if manifest_path.is_file():
        try:
            _audit_current_host_contract(load_json(manifest_path), "product ReLU", findings)
        except (ValueError, OSError) as error:
            _finding(findings, "host-implementation-contract", f"product manifest is invalid: {error}")
    snapshot = fixture.get("offline_success_artifacts_sha256")
    if not isinstance(snapshot, dict) or not snapshot:
        _finding(findings, "product-artifact-snapshot", "successful Vulkan artifact snapshot is missing")
    else:
        actual_snapshot = {path.name: sha256_file(path) for path in vk_dir.iterdir() if path.is_file()} if vk_dir.is_dir() else {}
        if actual_snapshot != snapshot:
            _finding(findings, "product-artifact-tampered", "Vulkan successful artifact set differs from its snapshot",
                     missing=sorted(set(snapshot) - set(actual_snapshot)),
                     unexpected=sorted(set(actual_snapshot) - set(snapshot)))
        for name, digest in snapshot.items():
            path = vk_dir / name
            _evidence_file(stage, vk_dir, name, f"product Vulkan artifact {name}", refs, findings)
            if not path.is_file() or sha256_file(path) != digest:
                _finding(findings, "product-artifact-tampered", f"Vulkan artifact snapshot differs for {name}")


OFFLINE_NEGATIVE_EXPECTATIONS = {
    "missing-validator": (1, r"cannot execute .*posix_spawn failed:.*(?:no such file|not found)"),
    "validator-failure": (23, r"command failed with exit code 23"),
    "link-failure": (29, r"command failed with exit code 29"),
    "audit-failure": (31, r"command failed with exit code 31"),
    "no-device-execution-verification": (1,
        r"Vulkan ABI execution verification failed|no Vulkan physical devices|no compatible Vulkan"),
}


def _offline_negative_cause_valid(name: object, status: object, diagnostic: str) -> bool:
    expected = OFFLINE_NEGATIVE_EXPECTATIONS.get(name)
    return (expected is not None and type(status) is int and status == expected[0] and
            re.search(expected[1], diagnostic, re.I) is not None)


def _offline_negative_argv_valid(stage: Path, base: Path, name: object,
                                 argv: object) -> bool:
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return False
    executable = stage / "tools/ncnn-compile"
    expected_prefix = [str(executable), str(base / "model.param"), "--bin",
                       str(base / "model.bin"), "--model-name", "paired_relu",
                       "--output-dir", str(base / "vulkan"), "--backend=vulkan",
                       "--precision=f32"]
    suffixes = {
        "missing-validator": ["--vulkan-spirv-val=" + str(base / "missing-spirv-val")],
        "validator-failure": ["--vulkan-spirv-val=" + str(base / "negative-tools/reject-spirv-val")],
        "link-failure": ["--vulkan-host-cxx=" + str(base / "negative-tools/reject-host-link")],
        "audit-failure": ["--readelf=" + str(base / "negative-tools/reject-readelf")],
        "no-device-execution-verification": ["--verify-execution"],
    }
    return name in suffixes and argv == expected_prefix + suffixes[name]


def _audit_offline_cpu_loader(stage: Path, base: Path, record: object,
                              label: str, cpu_dir_name: str,
                              input_name: str, output_name: str,
                              sanitizer_flags: set[str], refs: list[dict],
                              findings: list[dict]) -> None:
    if not isinstance(record, dict):
        _finding(findings, "offline-cpu-loader-record",
                 f"{label} native CPU loader record is missing or malformed")
        return

    loader_relative = f"cpu-loader-{cpu_dir_name}/cpu-product-loader"
    result_relative = f"cpu-loader-{cpu_dir_name}/result.json"
    cpu_directory = base / cpu_dir_name
    expected_argv = [
        str((base / loader_relative).resolve()),
        "paired_relu",
        str((cpu_directory / "libpaired_relu.so").resolve()),
        "257",
        str((base / input_name).resolve()),
        str((base / output_name).resolve()),
        str((base / result_relative).resolve()),
    ]
    argv = record.get("argv")
    if not isinstance(argv, list) or argv != expected_argv:
        _finding(findings, "offline-cpu-loader-command",
                 f"{label} invocation is not bound to its fixed CPU/input/output/result paths")
    if type(record.get("compile_returncode")) is not int or record.get("compile_returncode") != 0 or \
       type(record.get("returncode")) is not int or record.get("returncode") != 0:
        _finding(findings, "offline-cpu-loader-status",
                 f"{label} compile and run return codes must be integer zero")

    compile_argv = record.get("compile_argv")
    compile_valid = isinstance(compile_argv, list) and bool(compile_argv) and \
        all(isinstance(item, str) for item in compile_argv)
    if compile_valid:
        compile_args = compile_argv
        expected_source = (Path(__file__).resolve().parents[1] /
                           "Native/vulkan_product_loader.cpp").resolve()
        source_positions = [index for index, item in enumerate(compile_args)
                            if item.endswith("vulkan_product_loader.cpp")]
        include_paths = []
        forbidden_include_options = {"-include", "-imacros", "-include-pch"}
        index = 0
        malformed_include = False
        while index < len(compile_args):
            item = compile_args[index]
            if item in ("-I", "-isystem", "-iquote", "-idirafter"):
                if index + 1 >= len(compile_args):
                    malformed_include = True
                    break
                include_paths.append(compile_args[index + 1])
                index += 2
                continue
            if item in forbidden_include_options:
                malformed_include = True
            if item.startswith("-I") and len(item) > 2:
                include_paths.append(item[2:])
            elif any(item.startswith(option) and len(item) > len(option)
                     for option in ("-isystem", "-iquote", "-idirafter")):
                malformed_include = True
            index += 1
        output_positions = [index for index, item in enumerate(compile_args)
                            if item == "-o"]
        output_valid = (len(output_positions) == 1 and
                        output_positions[0] + 1 == len(compile_args) - 1 and
                        Path(compile_args[-1]).resolve() ==
                        (base / loader_relative).resolve())
        sanitizer_args_valid = True
        expected_argument_count = 10
        sanitizer_options = [item for item in compile_args
                             if item.startswith("-fsanitize=")]
        if {"address", "undefined"}.issubset(sanitizer_flags):
            required_sanitizer_args = {
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-fno-omit-frame-pointer",
                "-shared-libasan",
            }
            expected_argument_count += len(required_sanitizer_args)
            sanitizer_args_valid = (required_sanitizer_args.issubset(compile_args) and
                                    sanitizer_options == ["-fsanitize=address,undefined"])
        elif sanitizer_options or "-shared-libasan" in compile_args:
            sanitizer_args_valid = False
        model_definitions = [item for item in compile_args
                             if item.startswith("-DNCNN_PRODUCT_TEST_MODEL")]
        cpu_only_definitions = [item for item in compile_args
                                if item.startswith("-DNCNN_PRODUCT_TEST_CPU_ONLY")]
        # Every allowed token is checked below; extra flags or response files
        # must not override instrumentation, recovery or the CPU-only mode.
        compile_valid = (
            compile_args[0] != "" and len(compile_args) == expected_argument_count and
            compile_args.count("-std=c++23") == 1 and
            model_definitions == ["-DNCNN_PRODUCT_TEST_MODEL=paired_relu"] and
            cpu_only_definitions == ["-DNCNN_PRODUCT_TEST_CPU_ONLY=1"] and
            len(source_positions) == 1 and
            Path(compile_args[source_positions[0]]).resolve() == expected_source and
            not malformed_include and len(include_paths) == 1 and
            Path(include_paths[0]).resolve() == cpu_directory.resolve() and
            "-ldl" in compile_args and output_valid and sanitizer_args_valid and
            "-lvulkan" not in compile_args
        )
    if not compile_valid:
        _finding(findings, "offline-cpu-loader-command",
                 f"{label} compile command must use the CPU-only loader source/header, "
                 "fixed CPU include/output paths, and profile sanitizer flags")

    for field, relative, digest_field, description in (
        ("loader_file", loader_relative, "loader_sha256", "native loader binary"),
        ("result_file", result_relative, "result_sha256", "native loader result"),
    ):
        if record.get(field) != relative:
            _finding(findings, "offline-cpu-loader-artifact",
                     f"{label} {description} path is not the fixed stage-relative path")
        path = _evidence_file(stage, base, record.get(field),
                              f"{label} {description}", refs, findings)
        _check_digest_claim(record.get(digest_field), path,
                            f"{label} {description}", findings)
        if field == "result_file" and path is not None:
            try:
                result = load_json(path)
                vulkan_statuses = ("vulkan_create_status", "vulkan_run_status",
                                   "vulkan_destroy_status", "vulkan_destroy_null_status")
                if (not isinstance(result, dict) or result.get("status") != "success" or
                    type(result.get("cpu_status")) is not int or result["cpu_status"] != 0 or
                    result.get("phase") != "complete" or result.get("diagnostic") != "" or
                    any(type(result.get(key)) is not int or result[key] != -1
                        for key in vulkan_statuses)):
                    _finding(findings, "offline-cpu-loader-result",
                             f"{label} result must report successful CPU-only execution")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                _finding(findings, "offline-cpu-loader-result",
                         f"{label} result JSON is invalid: {error}")

    streams = record.get("streams")
    expected_streams = {
        "compile.stdout.log", "compile.stderr.log",
        "run.stdout.log", "run.stderr.log",
    }
    if not isinstance(streams, dict) or set(streams) != expected_streams:
        _finding(findings, "offline-cpu-loader-stream",
                 f"{label} must preserve all four raw compile/run streams")
        streams = streams if isinstance(streams, dict) else {}
    for stream_name in sorted(expected_streams):
        stream = streams.get(stream_name)
        expected_relative = f"cpu-loader-{cpu_dir_name}/{stream_name}"
        if not isinstance(stream, dict):
            _finding(findings, "offline-cpu-loader-stream",
                     f"{label} {stream_name} stream record is missing")
            continue
        if stream.get("file") != expected_relative:
            _finding(findings, "offline-cpu-loader-stream",
                     f"{label} {stream_name} path is not the fixed stage-relative path")
        path = _evidence_file(stage, base, stream.get("file"),
                              f"{label} {stream_name}", refs, findings)
        _check_digest_claim(stream.get("sha256"), path,
                            f"{label} {stream_name}", findings)
        if path is not None:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as error:
                _finding(findings, "offline-cpu-loader-stream",
                         f"{label} {stream_name} cannot be read: {error}")
                continue
            if SANITIZER_RE.search(text):
                _finding(findings, "offline-cpu-loader-sanitizer-diagnostic",
                         f"{label} {stream_name} contains a sanitizer/runtime diagnostic")


def _audit_no_device_lifecycle(stage: Path, base: Path, record: object,
                               sanitizer_flags: set[str], refs: list[dict],
                               findings: list[dict]) -> None:
    if not isinstance(record, dict):
        _finding(findings, "no-device-lifecycle-missing",
                 "generated Vulkan DSO no-device lifecycle evidence is required")
        return
    result_name = "no-device-loader-result.json"
    loader_name = "no-device-vulkan-product-loader"
    expected_argv = [str((base / loader_name).resolve()), "paired_relu",
                     str((base / "cpu/libpaired_relu.so").resolve()),
                     str((base / "vulkan/libpaired_relu_vulkan.so").resolve()),
                     "257", str((base / "no-device-input.f32").resolve()),
                     str((base / "no-device-cpu-output.f32").resolve()),
                     str((base / "no-device-vulkan-output.f32").resolve()),
                     str((base / result_name).resolve())]
    if (record.get("argv") != expected_argv or
            type(record.get("returncode")) is not int or record["returncode"] != 0 or
            type(record.get("compile_returncode")) is not int or
            record["compile_returncode"] != 0 or record.get("gpu_numeric") is not False):
        _finding(findings, "no-device-lifecycle-command",
                 "no-device loader must compile/run successfully with fixed product paths")
    source = (Path(__file__).resolve().parents[1] /
              "Native/vulkan_product_loader.cpp").resolve()
    compile_argv = record.get("compile_argv", [])
    sanitizer_args = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                      "-fno-omit-frame-pointer", "-shared-libasan"] if sanitizer_flags else []
    expected_compile_tail = ["-std=c++23", *sanitizer_args,
                             "-DNCNN_PRODUCT_TEST_MODEL=paired_relu",
                             "-DNCNN_PRODUCT_TEST_EXPECT_NO_DEVICE=1",
                             "-I", str((base / "cpu").resolve()),
                             "-I", str((base / "vulkan").resolve()),
                             "-I", str(source.parents[2] / "include"),
                             str(source), "-ldl", "-o", str((base / loader_name).resolve())]
    normalized_compile_argv = list(compile_argv) if isinstance(compile_argv, list) else []
    source_arguments = [index for index, argument in enumerate(normalized_compile_argv)
                        if isinstance(argument, str) and
                        Path(argument).name == "vulkan_product_loader.cpp"]
    if len(source_arguments) == 1:
        source_index = source_arguments[0]
        try:
            if Path(normalized_compile_argv[source_index]).resolve() == source:
                normalized_compile_argv[source_index] = str(source)
        except (OSError, ValueError):
            pass
    if (not isinstance(compile_argv, list) or len(compile_argv) < 2 or
            not compile_argv[0] or normalized_compile_argv[1:] != expected_compile_tail):
        _finding(findings, "no-device-lifecycle-instrumentation",
                 "no-device loader must use generated headers and matching profile instrumentation")
    for field, name, digest_field in (("loader_file", loader_name, "loader_sha256"),
                                      ("result_file", result_name, "result_sha256")):
        if record.get(field) != name:
            _finding(findings, "no-device-lifecycle-artifact", f"wrong {field} path")
        path = _evidence_file(stage, base, name, f"no-device {field}", refs, findings)
        _check_digest_claim(record.get(digest_field), path, f"no-device {field}", findings)
    result_path = base / result_name
    if result_path.is_file():
        result = load_json(result_path)
        if (not isinstance(result, dict) or result.get("status") != "success" or
                result.get("phase") != "expected-no-device" or
                type(result.get("cpu_status")) is not int or result["cpu_status"] != 0 or
                type(result.get("vulkan_create_status")) is not int or
                result["vulkan_create_status"] not in (2, 5) or
                type(result.get("vulkan_run_status")) is not int or result["vulkan_run_status"] != -1 or
                type(result.get("vulkan_destroy_status")) is not int or result["vulkan_destroy_status"] != -1 or
                type(result.get("vulkan_destroy_null_status")) is not int or result["vulkan_destroy_null_status"] != 0 or
                result.get("context_null") is not True or
                result.get("diagnostic_bounded") is not True or
                not isinstance(result.get("diagnostic"), str) or not result["diagnostic"]):
            _finding(findings, "no-device-lifecycle-result",
                     "no-device create must fail explicitly, clear context, bound diagnostics and allow destroy(NULL)")
    environment = record.get("icd_environment", {})
    if not isinstance(environment, dict):
        _finding(findings, "no-device-lifecycle-environment", "ICD environment is malformed")
        environment = {}
    if (environment.get("VK_ICD_FILENAMES") != str(base / "no-available-icd.json") or
            environment.get("VK_DRIVER_FILES") is not None or
            environment.get("VK_ADD_DRIVER_FILES") is not None):
        _finding(findings, "no-device-lifecycle-environment",
                 "no-device lifecycle must use only the unavailable ICD")
    streams = record.get("streams", {})
    if not isinstance(streams, dict):
        _finding(findings, "no-device-lifecycle-streams", "no-device streams are malformed")
        streams = {}
    required_streams = {"compile.stdout.log", "compile.stderr.log", "stdout.log", "stderr.log"}
    if set(streams) != required_streams:
        _finding(findings, "no-device-lifecycle-streams", "no-device raw streams are incomplete")
    for name in sorted(required_streams):
        entry = streams.get(name, {})
        path = _evidence_file(stage, base, entry.get("file"),
                              f"no-device {name}", refs, findings)
        _check_digest_claim(entry.get("sha256"), path, f"no-device {name}", findings)
        if path and SANITIZER_RE.search(path.read_text(encoding="utf-8", errors="replace")):
            _finding(findings, "no-device-lifecycle-sanitizer", "no-device loader emitted a sanitizer report")
    modules = record.get("loaded_modules", {})
    if not isinstance(modules, dict):
        _finding(findings, "no-device-lifecycle-modules", "no-device module records are malformed")
        modules = {}
    for phase in ("before-load", "after-model-load", "after-create"):
        entry = modules.get(phase, {})
        path = _evidence_file(stage, base, entry.get("file"),
                              f"no-device loaded modules {phase}", refs, findings)
        _check_digest_claim(entry.get("sha256"), path, f"no-device modules {phase}", findings)
    input_path = _evidence_file(stage, base, "no-device-input.f32", "no-device input", refs, findings)
    cpu_path = _evidence_file(stage, base, "no-device-cpu-output.f32", "no-device CPU output", refs, findings)
    if input_path and cpu_path:
        inputs = read_f32(input_path, 257)
        if input_path.read_bytes() != (base / "public-input.f32").read_bytes() or not compare_f32(
                read_f32(cpu_path, 257), [max(value, 0.0) for value in inputs], 1e-6, 1e-6)["passed"]:
            _finding(findings, "no-device-lifecycle-cpu", "no-device loader CPU/golden baseline differs")
    if (base / "no-device-vulkan-output.f32").exists():
        _finding(findings, "no-device-lifecycle-output", "failed create must not publish a GPU output")


def _stage_id(stage: Path) -> str | None:
    try:
        state = load_json(stage / "gate-state.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return state.get("stage_id") if isinstance(state, dict) else None


def _audit_native_host_program(stage: Path, base: Path, record: object,
                              refs: list[dict], findings: list[dict], *,
                              required: bool) -> dict | None:
    if record is None:
        if required:
            _finding(findings, "host-program-evidence-missing",
                     "VA1 native evidence has no saved host-program recorder trace")
        return None
    try:
        if not isinstance(record, dict):
            raise ValueError("host_program_evidence reference is malformed")
        record_path = _evidence_file(stage, base, record.get("file"),
                                     "native host-program trace", refs, findings)
        _check_digest_claim(record.get("sha256"), record_path,
                            "native host-program trace", findings)
        if record_path is None:
            return None
        trace_record = load_json(record_path)
        if (not isinstance(trace_record, dict) or
                trace_record.get("schema_version") != 1 or
                trace_record.get("kind") != "ncnn.vulkan_host_program_evidence" or
                trace_record.get("host_execution_role") != "legacy-product" or
                trace_record.get("host_aot_candidate") is not False):
            raise ValueError("native host-program trace has an invalid schema or execution role")
        for name in ("manifest", "plan"):
            artifact_record = trace_record.get(name)
            if not isinstance(artifact_record, dict):
                raise ValueError(f"native host-program trace omits {name} identity")
            artifact_path = _evidence_file(
                stage, base, artifact_record.get("file"),
                f"native host-program {name}", refs, findings)
            _check_digest_claim(artifact_record.get("sha256"), artifact_path,
                                f"native host-program {name}", findings)
            if artifact_path is None:
                raise ValueError(f"native host-program {name} bytes are missing")
            if name == "manifest":
                manifest_path = artifact_path
            else:
                plan_path = artifact_path
        manifest = load_json(manifest_path)
        plan = load_json(plan_path)
        shader_records = trace_record.get("shader_raw_files")
        weight_records = trace_record.get("weight_raw_files")
        if not isinstance(shader_records, list) or not isinstance(weight_records, list):
            raise ValueError("native host-program raw shader/weight inventory is missing")
        shader_bytes, weight_bytes = {}, {}
        for item in shader_records:
            if not isinstance(item, dict):
                raise ValueError("native host-program shader raw record is malformed")
            kernel_id = item.get("kernel_id")
            if type(kernel_id) is not int or kernel_id in shader_bytes:
                raise ValueError("native host-program shader kernel ID is missing/duplicated")
            raw_path = _evidence_file(stage, base, item.get("file"),
                                      f"native host-program kernel {kernel_id} raw shader",
                                      refs, findings)
            _check_digest_claim(item.get("sha256"), raw_path,
                                f"native host-program kernel {kernel_id} raw shader", findings)
            if raw_path is None:
                raise ValueError(f"native host-program kernel {kernel_id} raw shader is missing")
            raw = raw_path.read_bytes()
            if item.get("size_bytes") != len(raw):
                raise ValueError(f"native host-program kernel {kernel_id} shader byte count differs")
            shader_bytes[kernel_id] = raw
        for item in weight_records:
            if not isinstance(item, dict):
                raise ValueError("native host-program weight raw record is malformed")
            resource_id = item.get("resource_id")
            if type(resource_id) is not int or resource_id in weight_bytes:
                raise ValueError("native host-program weight resource ID is missing/duplicated")
            raw_path = _evidence_file(stage, base, item.get("file"),
                                      f"native host-program weight {resource_id} raw bytes",
                                      refs, findings)
            _check_digest_claim(item.get("sha256"), raw_path,
                                f"native host-program weight {resource_id} raw bytes", findings)
            if raw_path is None:
                raise ValueError(f"native host-program weight {resource_id} raw bytes are missing")
            raw = raw_path.read_bytes()
            if item.get("size_bytes") != len(raw):
                raise ValueError(f"native host-program weight {resource_id} byte count differs")
            weight_bytes[resource_id] = raw
        result = host_program_audit.validate_host_program(
            manifest, plan, shader_bytes=shader_bytes, weight_bytes=weight_bytes,
            manifest_sha256=sha256_file(manifest_path),
            plan_sha256=sha256_file(plan_path))
        if trace_record.get("host_program_identity") != result["identity_sha256"]:
            raise ValueError("native host-program trace identity differs from manifest")
        if record.get("host_program_identity") != result["identity_sha256"]:
            raise ValueError("native evidence reference has a stale host-program identity")
        if trace_record.get("legacy_symbolic_trace") != result["legacy_symbolic_trace"]:
            raise ValueError("saved symbolic recorder trace differs from independent oracle")
        operations = trace_record.get("recorded_operations")
        expected_operations = {"create": manifest["host_program"]["create_operations"],
                               "run": manifest["host_program"]["run_operations"]}
        if operations != expected_operations:
            raise ValueError("saved native operation trace differs from manifest operations")
        expected_audit = {key: value for key, value in result.items()
                          if key != "legacy_symbolic_trace"}
        if trace_record.get("audit") != expected_audit:
            raise ValueError("saved host-program audit summary is stale or tampered")
        return {"path": str(record_path.relative_to(stage)),
                "identity_sha256": result["identity_sha256"],
                "shader_count": len(shader_bytes), "weight_count": len(weight_bytes),
                "legacy_symbolic_trace": result["legacy_symbolic_trace"]}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            host_program_audit.EvidenceError) as error:
        _finding(findings, "host-program-evidence-invalid",
                 f"native host-program evidence is invalid: {error}")
        return None


def _audit_offline(stage: Path, refs: list[dict], findings: list[dict]) -> dict | None:
    base = stage / "test/Native/vulkan-product-offline"
    evidence_path = _evidence_file(stage, base, "evidence.json", "V6 offline evidence", refs, findings)
    if evidence_path is None:
        _finding(findings, "offline-evidence-missing", "fresh V6 offline scope evidence is absent")
        return None
    try:
        evidence = load_json(evidence_path)
        if not isinstance(evidence, dict) or evidence.get("schema_version") != 1 or evidence.get("kind") != "ncnn.vulkan_offline_evidence":
            raise ValueError("unsupported offline schema/kind")
        execution = evidence.get("execution", {})
        if execution.get("mode") != "offline-no-device" or execution.get("vulkan_execution") != "not_run" or execution.get("gpu_output") is not None:
            _finding(findings, "offline-gpu-claim", "offline scope must explicitly state no GPU execution/output")
        if execution.get("vk_icd_manifest_exists") is not False:
            _finding(findings, "offline-icd-not-unavailable", "offline run did not prove its requested ICD manifest is unavailable")
        unavailable_icd = execution.get("vk_icd_filenames")
        if not isinstance(unavailable_icd, str) or Path(unavailable_icd).exists():
            _finding(findings, "offline-icd-identity", "offline Vulkan environment does not point at a nonexistent ICD file")
        claims = evidence.get("claims", {})
        if claims.get("vulkan_device_execution") != "not_run" or claims.get("cpu_gpu_numerical_comparison") != "not_claimed":
            _finding(findings, "offline-claim-overreach", "offline evidence overclaims GPU numerical validation")
        fixture = evidence.get("fixture", {})
        native_host_program = _audit_native_host_program(
            stage, base, evidence.get("host_program_evidence"), refs, findings,
            required=_stage_id(stage) == "VA1")
        if (native_host_program is not None and
                fixture.get("host_program_identity") !=
                native_host_program.get("identity_sha256")):
            _finding(findings, "host-program-fixture-identity",
                     "offline fixture identity differs from native host-program audit")
        if fixture.get("model") != "paired_relu":
            _finding(findings, "offline-fixture", "offline positive fixture is not paired ReLU")
        model_param = _evidence_file(stage, base, "model.param", "offline model .param", refs, findings)
        model_bin = _evidence_file(stage, base, "model.bin", "offline model .bin", refs, findings)
        for field, path in (("param_sha256", model_param), ("bin_sha256", model_bin)):
            _check_digest_claim(fixture.get(field), path, f"offline {field}", findings)
        public_io = evidence.get("public_io", {})
        if (public_io.get("element_type") != "f32" or public_io.get("shape") != [257] or
                public_io.get("shape_chw") != [1, 1, 257] or public_io.get("host_layout") != "CHW"):
            _finding(findings, "offline-io-contract", "offline fixture must be static FP32 [257]")
        input_path, input_values = _verify_raw_tensor_record(stage, base, public_io.get("input"), None,
                                                             "offline input", refs, findings)
        if input_values is not None:
            golden_values = [max(value, 0.0) for value in input_values]
            _, cpu_values = _verify_raw_tensor_record(stage, base, public_io.get("cpu_output"),
                                                       golden_values, "offline CPU output", refs, findings)
            _, recorded_golden = _verify_raw_tensor_record(stage, base, public_io.get("scalar_golden"),
                                                             golden_values, "offline scalar golden", refs, findings)
            if cpu_values is not None and recorded_golden is not None:
                tol = public_io.get("tolerance", {})
                atol, rtol = tol.get("absolute"), tol.get("relative")
                if (atol, rtol) != (1.0e-6, 1.0e-6):
                    _finding(findings, "offline-budget", "offline scalar golden budget differs from fixed 1e-6/1e-6")
                metric = compare_f32(cpu_values, recorded_golden, float(atol or 0), float(rtol or 0))
                if not metric["passed"] or public_io.get("passed") is not True:
                    _finding(findings, "offline-cpu-numeric", "offline CPU/GOLDEN comparison failed")
                for key, actual in (("max_absolute_error", metric["max_abs"]),
                                    ("max_relative_error", metric["max_rel"])):
                    claim = public_io.get(key)
                    if not isinstance(claim, (int, float)) or abs(float(claim) - actual) > max(1e-12, actual * 1e-7):
                        _finding(findings, "offline-numeric-claim", f"offline {key} differs from raw tensors")
        if public_io.get("cpu_status") != 0 or public_io.get("cpu_recovery_status") != 0:
            _finding(findings, "offline-cpu-status", "offline initial/recovery CPU call did not succeed")
        cache_path = stage / "CMakeCache.txt"
        cache = _parse_cmake_cache(cache_path) if cache_path.is_file() else {}
        sanitizer_flags = set(filter(None, cache.get("COMPILER_SANITIZERS", "").split(";")))
        commands = evidence.get("commands", {})
        _audit_offline_cpu_loader(
            stage, base, commands.get("cpu_loader"), "initial", "cpu",
            "public-input.f32", "cpu-output.f32", sanitizer_flags, refs, findings)
        _audit_offline_cpu_loader(
            stage, base, commands.get("cpu_recovery_loader"), "recovery",
            "cpu-after-negative-cases", "public-input-after-negative-cases.f32",
            "cpu-output-after-negative-cases.f32", sanitizer_flags, refs, findings)
        _audit_no_device_lifecycle(stage, base, evidence.get("no_device_lifecycle"),
                                   sanitizer_flags, refs, findings)
        recovery_input = _evidence_file(stage, base, "public-input-after-negative-cases.f32",
                                        "offline recovered CPU input", refs, findings)
        recovery_path = _evidence_file(stage, base, "cpu-output-after-negative-cases.f32",
                                       "offline recovered CPU output", refs, findings)
        _check_digest_claim(public_io.get("cpu_recovery_output_sha256"), recovery_path,
                            "offline recovered CPU output", findings)
        if recovery_input and input_path and recovery_input.read_bytes() != input_path.read_bytes():
            _finding(findings, "offline-recovery-input-mismatch", "offline CPU recovery input differs from original input")
        if recovery_path and input_values is not None:
            try:
                recovered = read_f32(recovery_path, len(input_values))
                golden = [max(value, 0.0) for value in input_values]
                recovery_metric = compare_f32(recovered, golden, 1e-6, 1e-6)
                if not recovery_metric["passed"] or public_io.get("cpu_recovery_max_absolute_error") != recovery_metric["max_abs"]:
                    _finding(findings, "offline-recovery-numeric", "offline CPU output after negatives differs from independent golden")
            except (OSError, ValueError) as error:
                _finding(findings, "offline-recovery-invalid", f"offline recovery tensor invalid: {error}")
        cpu_recovery_library = base / "cpu-after-negative-cases/libpaired_relu.so"
        _evidence_file(stage, cpu_recovery_library.parent, cpu_recovery_library.name,
                       "offline recovered CPU library", refs, findings)
        _check_digest_claim(fixture.get("cpu_recovery_library_sha256"),
                            cpu_recovery_library if cpu_recovery_library.is_file() else None,
                            "offline recovered CPU library", findings)
        _check_product_artifact_set(stage, base, fixture, refs, findings)
        shader_hash_map = fixture.get("reconstructed_shader_sha256", {})
        expected_by_kernel = {str(key): value for key, value in shader_hash_map.items()} if isinstance(shader_hash_map, dict) else {}
        shader_files = fixture.get("shader_files", [])
        if not isinstance(shader_files, list) or len(shader_files) != len(expected_by_kernel):
            _finding(findings, "offline-shader-files", "offline reconstructed shader files do not cover the plan")
        else:
            for shader_record in shader_files:
                if not isinstance(shader_record, dict):
                    _finding(findings, "offline-shader-files", "offline shader file record is malformed")
                    continue
                shader_file = _evidence_file(stage, base, shader_record.get("file"),
                                             "offline reconstructed shader", refs, findings)
                _check_digest_claim(shader_record.get("sha256"), shader_file,
                                    "offline reconstructed shader", findings)
        _audit_recorded_shaders(stage, base, evidence.get("shader_validations"),
                                expected_by_kernel, "V6 offline", refs, findings)
        negative_expectations = OFFLINE_NEGATIVE_EXPECTATIONS
        negatives = evidence.get("negative_cases")
        actual_names = []
        if not isinstance(negatives, list):
            _finding(findings, "offline-negative-list", "offline negative_cases is missing")
            negatives = []
        for item in negatives:
            if not isinstance(item, dict):
                _finding(findings, "offline-negative-record", "offline negative record is malformed")
                continue
            name = item.get("name")
            actual_names.append(name)
            status = item.get("returncode")
            diagnostic = str(item.get("stdout", "")) + "\n" + str(item.get("stderr", ""))
            expected = negative_expectations.get(name)
            matched = _offline_negative_cause_valid(name, status, diagnostic)
            if not matched or item.get("diagnostic_match") is not True:
                _finding(findings, "offline-negative-diagnostic", f"offline negative {name} lacks its fixed failure cause/status")
            if not _offline_negative_argv_valid(stage, base, name, item.get("argv")):
                _finding(findings, "offline-negative-command", f"offline negative {name} does not identify its expected ncnn-compile invocation")
            before, after = item.get("artifacts_before_sha256"), item.get("artifacts_after_sha256")
            if item.get("artifacts_unchanged") is not True or before != after or before != fixture.get("offline_success_artifacts_sha256"):
                _finding(findings, "offline-negative-overwrite", f"offline negative {name} changed successful artifacts")
            for key in ("stdout_file", "stderr_file"):
                if item.get(key):
                    _evidence_file(stage, base, item[key], f"offline {name} {key}", refs, findings)
        expected_names = set(negative_expectations)
        if len(actual_names) != len(expected_names) or set(actual_names) != expected_names:
            _finding(findings, "offline-negative-coverage", "offline negative case set differs from the approved list",
                     missing=sorted(expected_names - set(actual_names)), extra=sorted(set(actual_names) - expected_names),
                     duplicates=sorted(name for name in set(actual_names) if actual_names.count(name) > 1))
        commands = evidence.get("commands", {})
        if type(commands.get("offline_compile_returncode")) is not int or commands.get("offline_compile_returncode") != 0:
            _finding(findings, "offline-compile", "offline Vulkan artifact compilation did not pass")
        return {"path": str(evidence_path.relative_to(stage)), "fixture": fixture,
                "public_io": public_io, "negative_cases": sorted(actual_names),
                "gpu_numeric": False}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        _finding(findings, "offline-evidence-invalid", f"offline evidence is invalid: {error}")
        return None


def _audit_recorded_shaders(stage: Path, base: Path, records: object,
                            expected_by_kernel: dict[str, str], label: str,
                            refs: list[dict], findings: list[dict], *,
                            log_key: str = "log_file",
                            log_hash_key: str = "log_sha256",
                            shader_path_key: str = "shader_file") -> None:
    if not isinstance(records, list) or len(records) != len(expected_by_kernel):
        _finding(findings, "spirv-validator-coverage",
                 f"{label} validator records do not cover each planned kernel",
                 expected=len(expected_by_kernel), actual=len(records) if isinstance(records, list) else None)
        return
    expected_ids = list(expected_by_kernel)
    observed_ids = []
    observed_hashes = []
    observed_log_paths = set()
    observed_shader_paths = set()
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            _finding(findings, "spirv-validator-record", f"{label} validator record {index} is malformed")
            continue
        kernel_id = item.get("kernel_id")
        kernel_key = str(kernel_id)
        if kernel_id is None:
            _finding(findings, "spirv-validator-kernel-id",
                     f"{label} validator record lacks an explicit kernel id", index=index)
        observed_ids.append(kernel_key)
        if kernel_key not in expected_by_kernel:
            _finding(findings, "spirv-validator-kernel-id", f"{label} validator record references an unplanned kernel id",
                     kernel_id=kernel_key)
            expected_digest = None
        else:
            expected_digest = expected_by_kernel[kernel_key]
        log_path = _evidence_file(stage, base, item.get(log_key),
                                  f"{label} kernel {kernel_key} validator JSON", refs, findings)
        if log_path is not None:
            normalized_log = log_path.resolve()
            if normalized_log in observed_log_paths:
                _finding(findings, "spirv-validator-log-reused",
                         f"{label} reuses one validator JSON file for multiple kernels",
                         path=str(item.get(log_key)))
            observed_log_paths.add(normalized_log)
        _check_digest_claim(item.get(log_hash_key), log_path,
                            f"{label} kernel {kernel_key} validator JSON", findings)
        shader_path = _evidence_file(stage, base, item.get(shader_path_key),
                                     f"{label} kernel {kernel_key} validated SPIR-V", refs, findings)
        if shader_path is not None:
            normalized_shader = shader_path.resolve()
            if normalized_shader in observed_shader_paths:
                _finding(findings, "spirv-validator-shader-reused",
                         f"{label} reuses one raw shader file for multiple kernels",
                         path=str(item.get(shader_path_key)))
            observed_shader_paths.add(normalized_shader)
        shader_digest = item.get("sha256")
        _check_digest_claim(shader_digest, shader_path,
                            f"{label} kernel {kernel_key} SPIR-V", findings)
        if expected_digest is not None and shader_digest != expected_digest:
            _finding(findings, "spirv-validator-shader-mismatch",
                     f"{label} kernel {kernel_key} validated shader differs from model manifest")
        if shader_digest is not None:
            observed_hashes.append(shader_digest)
        if shader_path and item.get("size_bytes") != shader_path.stat().st_size:
            _finding(findings, "spirv-validator-shader-size",
                     f"{label} kernel {kernel_key} SPIR-V size differs from raw file")
        if not log_path:
            continue
        try:
            invocation = load_json(log_path)
            if (not isinstance(invocation, dict) or invocation.get("schema_version") != 1 or
                    invocation.get("kind") != "ncnn.spirv_tool_invocation" or
                    type(invocation.get("returncode")) is not int or invocation.get("returncode") != 0):
                _finding(findings, "spirv-validator-invocation", f"{label} kernel {kernel_key} invocation failed/malformed")
                continue
            argv = invocation.get("argv", [])
            if not isinstance(argv, list) or not argv or "spirv-val" not in Path(str(argv[0])).name:
                _finding(findings, "spirv-validator-executable", f"{label} kernel {kernel_key} raw argv is not spirv-val")
            try:
                env_index = argv.index("--target-env")
            except (ValueError, AttributeError):
                env_index = -1
            if env_index < 0 or env_index + 1 >= len(argv) or argv[env_index + 1] != "vulkan1.1":
                _finding(findings, "spirv-validator-target", f"{label} kernel {kernel_key} did not validate Vulkan 1.1")
            raw_shader = invocation.get("shader")
            if not isinstance(raw_shader, dict):
                _finding(findings, "spirv-validator-shader-record", f"{label} kernel {kernel_key} raw invocation omitted shader bytes")
            else:
                raw_shader_path = (log_path.parent / str(raw_shader.get("path", ""))).resolve()
                if shader_path is None or raw_shader_path != shader_path.resolve():
                    _finding(findings, "spirv-validator-path-mismatch", f"{label} kernel {kernel_key} raw invocation path differs from the indexed shader file")
                _check_digest_claim(raw_shader.get("sha256"), shader_path,
                                    f"{label} kernel {kernel_key} invocation shader", findings)
                if raw_shader.get("sha256") != shader_digest or raw_shader.get("size_bytes") != item.get("size_bytes"):
                    _finding(findings, "spirv-validator-record-mismatch", f"{label} kernel {kernel_key} raw invocation and index differ")
                if isinstance(argv, list) and argv and isinstance(argv[-1], str):
                    input_path = Path(argv[-1]).resolve()
                    if input_path.is_file() and sha256_file(input_path) != raw_shader.get("sha256"):
                        _finding(findings, "spirv-validator-argv-shader-mismatch",
                                 f"{label} kernel {kernel_key} preserved argv input differs from recorded SPIR-V bytes")
            stdout, stderr = invocation.get("stdout"), invocation.get("stderr")
            if not isinstance(stdout, str) or not isinstance(stderr, str):
                _finding(findings, "spirv-validator-output-missing", f"{label} kernel {kernel_key} raw stdout/stderr is missing")
            elif re.search(r"VUID-[A-Za-z0-9-]+|Validation Error|AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:",
                           stdout + "\n" + stderr, re.I):
                _finding(findings, "spirv-validator-diagnostic", f"{label} kernel {kernel_key} validator output has an error")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "spirv-validator-invalid", f"{label} kernel {kernel_key} raw validator JSON invalid: {error}")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected_by_kernel):
        _finding(findings, "spirv-validator-kernel-coverage",
                 f"{label} validator kernel ids are not a one-to-one plan mapping",
                 expected=sorted(expected_by_kernel), observed=observed_ids)
    if sorted(observed_hashes) != sorted(expected_by_kernel.values()):
        _finding(findings, "spirv-validator-hash-multiset",
                 f"{label} validator shader hash multiset differs from planned kernels")


def _v4_failure_case(stage: Path, base: Path, record: dict) -> str | None:
    argv = record.get("argv")
    if (not isinstance(argv, list) or not all(isinstance(item, str) for item in argv) or
            not argv or Path(argv[0]).resolve() != (stage / "tools/ncnn-compile").resolve()):
        return None
    try:
        if (argv.count("--model-name") != 1 or argv.count("--output-dir") != 1 or
                argv.count("--bin") != 1):
            return None
        model_index = argv.index("--model-name")
        model_name = argv[model_index + 1]
        output_index = argv.index("--output-dir")
        output_dir = Path(argv[output_index + 1]).resolve()
    except (ValueError, IndexError):
        return None
    try:
        output_dir.relative_to(base.resolve())
    except ValueError:
        return None
    if model_name != "paired_relu":
        return {"missing_device": "missing-device-selector",
                "missing_tool": "missing-validator"}.get(model_name, model_name)
    out_name = output_dir.name
    if out_name == "user-data":
        return "output-user-data"
    if "--threads=1" in argv:
        return "cpu-vulkan-output-collision"
    if "--vulkan-spirv-val=/path/that/does/not/exist" in argv:
        return "missing-validator"
    if any(arg.startswith("--vulkan-spirv-val=") and Path(arg.split("=", 1)[1]).name == "reject-spirv-val" for arg in argv):
        return "validator-failure"
    if any(arg.startswith("--vulkan-host-cxx=") and Path(arg.split("=", 1)[1]).name == "reject-host-link" for arg in argv):
        return "link-failure"
    if any(arg.startswith("--readelf=") and Path(arg.split("=", 1)[1]).name == "reject-readelf" for arg in argv):
        return "audit-failure"
    if out_name == "vulkan" and "--verify-execution" in argv:
        return "verify-execution-no-device"
    return None


def _v4_failure_diagnostic(case: str, text: str) -> bool:
    if case == "foo":
        pattern = r"output directory belongs to CPU model 'foo_vulkan'"
    elif case == "foo_vulkan":
        pattern = r"output directory belongs to Vulkan model 'foo'"
    elif case == "unsupported":
        pattern = r"tosa\.clamp.*not supported by Vulkan graph"
    elif case == "invalid-backend":
        pattern = r"--backend must be either cpu or vulkan"
    elif case == "low-precision":
        pattern = r"--backend=vulkan currently supports only FP32"
    elif case in {"dynamic-shape", "dynamic-constraint"}:
        pattern = r"dynamic/input-shape overrides are not supported by --backend=vulkan"
    elif case in {"cpu-target", "cpu-march", "cpu-mcpu", "cpu-mtune", "cpu-target-feature", "cpu-sysroot"}:
        pattern = r"CPU target options are not supported by --backend=vulkan"
    elif case in {"clang-arg", "linker-arg"}:
        pattern = r"custom clang/linker arguments are not supported by --backend=vulkan"
    elif case in {"emit-stage", "emit-manifest", "emit-plan", "profile"}:
        pattern = r"CPU intermediate, manifest, execution-plan, and profiling options are not supported by --backend=vulkan"
    elif case == "output-user-data":
        pattern = r"output directory contains a file not owned by ncnn-compile: keep\.txt"
    elif case == "cpu-vulkan-output-collision":
        pattern = r"output directory belongs to Vulkan model 'paired_relu'"
    elif case == "missing-validator":
        pattern = r"cannot execute '/path/that/does/not/exist': .*no such file"
    elif case == "validator-failure":
        pattern = r"command failed with exit code 23"
    elif case == "link-failure":
        pattern = r"command failed with exit code 29"
    elif case == "audit-failure":
        pattern = r"command failed with exit code 31"
    elif case in {"missing-device-selector", "verify-execution-no-device"}:
        pattern = r"no Vulkan physical devices|no compatible Vulkan|Vulkan ABI execution verification failed"
    else:
        pattern = r"CPU-only tuning, precision, debug, and undefined-symbol expectation options are not supported by --backend=vulkan"
    return re.search(pattern, text, re.I | re.S) is not None


def _audit_v4_failure_records(stage: Path, base: Path, records: object,
                              fixture: dict, refs: list[dict],
                              label: str, findings: list[dict]) -> None:
    if not isinstance(records, list):
        _finding(findings, "product-negative-evidence", f"{label} expected failure diagnostics are missing")
        return
    observed = []
    verified_input_names = set()
    for item in records:
        if not isinstance(item, dict):
            _finding(findings, "product-negative-evidence", f"{label} has a malformed expected-failure record")
            continue
        case = _v4_failure_case(stage, base, item)
        if case is None:
            _finding(findings, "product-negative-command", f"{label} expected-failure command identity/role is invalid")
            continue
        observed.append(case)
        argv = item.get("argv", [])
        try:
            param = Path(argv[1]).resolve()
            bin_index = argv.index("--bin")
            model_bin = Path(argv[bin_index + 1]).resolve()
            output_index = argv.index("--output-dir")
            if case == "unsupported":
                expected_param_name, expected_bin_name = (
                    "unsupported-sigmoid.param", "unsupported-sigmoid.bin")
                expected_param_hash = sha256_bytes(V4_UNSUPPORTED_SIGMOID_PARAM)
                expected_bin_hash = sha256_bytes(b"")
            else:
                expected_param_name, expected_bin_name = "model.param", "model.bin"
                expected_param_hash = fixture.get("param_sha256")
                expected_bin_hash = fixture.get("bin_sha256")
            if (param != (base / expected_param_name).resolve() or
                    model_bin != (base / expected_bin_name).resolve()):
                _finding(findings, "product-negative-command",
                         f"{label} {case} command does not use the frozen V4 fixture input paths")
            for role, name, digest in (("parameter", expected_param_name, expected_param_hash),
                                       ("binary", expected_bin_name, expected_bin_hash)):
                if name in verified_input_names:
                    continue
                input_file = _evidence_file(stage, base, name,
                                            f"{label} {case} source model {role}", refs, findings)
                _check_digest_claim(digest, input_file,
                                    f"{label} {case} {role} fixture", findings)
                verified_input_names.add(name)
            if case in V4_FAILURE_FLAGS and V4_FAILURE_FLAGS[case] not in argv:
                _finding(findings, "product-negative-command", f"{label} {case} command omits its expected rejecting option")
            if case == "missing-validator" and "--vulkan-spirv-val=/path/that/does/not/exist" not in argv:
                _finding(findings, "product-negative-command", f"{label} missing-validator case does not use its fixed absent validator path")
            if case == "missing-device-selector" and (
                    "--verify-execution" not in argv or
                    Path(argv[output_index + 1]).name != "missing-device"):
                _finding(findings, "product-negative-command", f"{label} missing-device case lacks its named verification fixture")
            if case == "verify-execution-no-device" and "--verify-execution" not in argv:
                _finding(findings, "product-negative-command", f"{label} device verification case omits --verify-execution")
        except (ValueError, IndexError, OSError):
            _finding(findings, "product-negative-command", f"{label} {case} command has malformed model arguments")
        status = item.get("returncode")
        expected_status = {"validator-failure": 23, "link-failure": 29,
                           "audit-failure": 31}.get(case)
        diagnostic = str(item.get("stdout", "")) + "\n" + str(item.get("stderr", ""))
        if (type(status) is not int or status <= 0 or
                (expected_status is not None and status != expected_status) or
                not _v4_failure_diagnostic(case, diagnostic)):
            _finding(findings, "product-negative-diagnostic", f"{label} {case} lacks its fixed nonzero cause/status")
    if len(observed) != len(V4_FAILURE_CASES) or set(observed) != V4_FAILURE_CASES:
        _finding(findings, "product-negative-coverage",
                 f"{label} expected-failure cases are not the exact distinct approved set",
                 missing=sorted(V4_FAILURE_CASES - set(observed)),
                 extra=sorted(set(observed) - V4_FAILURE_CASES),
                 duplicates=sorted(case for case in set(observed) if observed.count(case) > 1))


SANITIZER_TEMP_BITCODE_RE = re.compile(
    rb"/(?:[^/\x00]+/)*ncnn-compile-[A-Za-z0-9_-]+/vulkan_model_wrapper\.bc")


def _normalize_sanitizer_temp_bitcode_paths(data: bytes) -> tuple[bytes, int]:
    matches = list(SANITIZER_TEMP_BITCODE_RE.finditer(data))
    normalized = SANITIZER_TEMP_BITCODE_RE.sub(
        b"/ncnn-compile-XXXXXX/vulkan_model_wrapper.bc", data)
    return normalized, len(matches)


def _audit_product_reproducibility(stage: Path, base: Path, evidence: dict,
                                   sanitizer_flags: set[str], refs: list[dict],
                                   findings: list[dict]) -> None:
    records = evidence.get("vulkan_reproducibility")
    required = {"libpaired_relu_vulkan.so", "paired_relu_vulkan.h",
                "paired_relu_vulkan.json", "paired_relu_vulkan.plan.json"}
    if not isinstance(records, dict) or set(records) != required:
        _finding(findings, "product-reproducibility-record",
                 "Vulkan reproducibility evidence must cover all published artifacts")
        return
    sanitizer = {"address", "undefined"}.issubset(sanitizer_flags)
    for name in sorted(required):
        record = records.get(name)
        if not isinstance(record, dict):
            _finding(findings, "product-reproducibility-record",
                     f"Vulkan reproducibility record is malformed: {name}")
            continue
        first_path = base / "vulkan" / name
        repeat_path = base / "vulkan-repeat" / name
        _evidence_file(stage, first_path.parent, name,
                       f"Vulkan initial reproducibility artifact {name}", refs, findings)
        _evidence_file(stage, repeat_path.parent, name,
                       f"Vulkan repeat reproducibility artifact {name}", refs, findings)
        if not first_path.is_file() or not repeat_path.is_file():
            continue
        first, repeat = first_path.read_bytes(), repeat_path.read_bytes()
        first_sha, repeat_sha = sha256_bytes(first), sha256_bytes(repeat)
        if (record.get("first_sha256") != first_sha or
                record.get("repeat_sha256") != repeat_sha):
            _finding(findings, "product-reproducibility-hash",
                     f"Vulkan reproducibility byte hash differs: {name}")
            continue
        if first == repeat:
            expected_mode, normalized, normalized_count = "byte-identical", first, 0
        elif name == "libpaired_relu_vulkan.so" and sanitizer:
            normalized_first, first_count = _normalize_sanitizer_temp_bitcode_paths(first)
            normalized_repeat, repeat_count = _normalize_sanitizer_temp_bitcode_paths(repeat)
            if first_count == 0 or first_count != repeat_count or normalized_first != normalized_repeat:
                _finding(findings, "product-reproducibility-mismatch",
                         "SAN DSO differs beyond the compiler's unique temporary bitcode path")
                continue
            expected_mode, normalized, normalized_count = (
                "sanitizer-temp-path-normalized", normalized_first, first_count)
        else:
            _finding(findings, "product-reproducibility-mismatch",
                     f"Vulkan output is not byte-reproducible: {name}")
            continue
        if (record.get("mode") != expected_mode or
                record.get("normalized_temp_paths") != normalized_count or
                record.get("canonical_sha256") != sha256_bytes(normalized)):
            _finding(findings, "product-reproducibility-claim",
                     f"Vulkan reproducibility normalization claim is invalid: {name}")


def _audit_product(stage: Path, expected_devices: set[str], refs: list[dict],
                   findings: list[dict]) -> dict:
    records = {}
    paths = sorted((stage / "test/Native").glob("vulkan-product-cli*/evidence.json"))
    if not paths:
        _finding(findings, "product-evidence-missing", "fresh V4 product CLI evidence is absent")
        return records
    found = set()
    product_cache = _parse_cmake_cache(stage / "CMakeCache.txt")
    product_sanitizers = set(filter(None,
                                    product_cache.get("COMPILER_SANITIZERS", "").split(";")))
    for evidence_path in paths:
        base = evidence_path.parent
        try:
            evidence = load_json(evidence_path)
            _evidence_file(stage, base, "evidence.json", f"V4 {base.name} evidence", refs, findings)
            if not isinstance(evidence, dict) or evidence.get("schema_version") != 1 or evidence.get("kind") != "ncnn.vulkan_product_evidence":
                raise ValueError("unsupported product evidence schema/kind")
            _audit_product_reproducibility(
                stage, base, evidence, product_sanitizers, refs, findings)
            execution = evidence.get("execution", {})
            if execution.get("mode") != "device-execution" or execution.get("vulkan_execution") != "passed":
                _finding(findings, "product-execution", f"V4 {base.name} does not record a successful device execution")
            fixture = evidence.get("fixture", {})
            native_host_program = _audit_native_host_program(
                stage, base, evidence.get("host_program_evidence"), refs, findings,
                required=_stage_id(stage) == "VA1")
            if (native_host_program is not None and
                    fixture.get("host_program_identity") !=
                    native_host_program.get("identity_sha256")):
                _finding(findings, "host-program-fixture-identity",
                         f"V4 {base.name} fixture identity differs from native host-program audit")
            io = evidence.get("public_io", {})
            if (fixture.get("model") != "paired_relu" or io.get("element_type") != "f32" or
                    io.get("shape_chw") != [1, 1, 257] or io.get("host_layout") != "CHW"):
                _finding(findings, "product-fixture", "V4 product fixture is not static FP32 ReLU")
            input_path, input_values = _verify_raw_tensor_record(stage, base, io.get("input"), None,
                                                                 f"V4 {base.name} input", refs, findings)
            if input_values is None:
                continue
            expected = [max(value, 0.0) for value in input_values]
            _, cpu_values = _verify_raw_tensor_record(stage, base, io.get("cpu_output"), expected,
                                                       f"V4 {base.name} CPU output", refs, findings)
            _, vk_values = _verify_raw_tensor_record(stage, base, io.get("vulkan_output"), expected,
                                                      f"V4 {base.name} Vulkan output", refs, findings)
            _, golden = _verify_raw_tensor_record(stage, base, io.get("scalar_golden"), expected,
                                                   f"V4 {base.name} scalar golden", refs, findings)
            if any(item is None for item in (cpu_values, vk_values, golden)):
                continue
            tolerance = io.get("tolerance", {})
            atol, rtol = tolerance.get("absolute"), tolerance.get("relative")
            if (atol, rtol) != (1.0e-6, 1.0e-6):
                _finding(findings, "product-budget", f"V4 {base.name} numerical budget differs from frozen 1e-6")
            computed = {"cpu_to_golden": compare_f32(cpu_values, golden, float(atol or 0), float(rtol or 0)),
                        "vulkan_to_golden": compare_f32(vk_values, golden, float(atol or 0), float(rtol or 0)),
                        "cpu_to_vulkan": compare_f32(cpu_values, vk_values, float(atol or 0), float(rtol or 0))}
            for key, metrics in computed.items():
                if not metrics["passed"]:
                    _finding(findings, "product-numeric", f"V4 {base.name} {key} fails the fixed budget")
                for claim_group, metric_key in (("max_absolute_error", "max_abs"),
                                                ("max_relative_error", "max_rel")):
                    claim = io.get(claim_group, {}).get(key)
                    if not isinstance(claim, (int, float)) or abs(float(claim) - metrics[metric_key]) > max(1e-12, metrics[metric_key] * 1e-7):
                        _finding(findings, "product-numeric-claim", f"V4 {base.name} {key} {claim_group} differs from raw tensors")
            if io.get("passed") is not True:
                _finding(findings, "product-numeric-claim", f"V4 {base.name} public_io is not marked passed")
            _check_product_artifact_set(stage, base, fixture, refs, findings)
            # Recompute every extracted kernel hash and require a validator record.
            validator = evidence.get("validator", {})
            shader_hashes = validator.get("shader_hashes")
            if not isinstance(shader_hashes, dict) or not shader_hashes:
                _finding(findings, "product-validator", f"V4 {base.name} validator shader hashes are absent")
                shader_hashes = {}
            else:
                for kernel_id, digest in shader_hashes.items():
                    path = base / "vulkan/shader-reconstruction" / f"kernel-{kernel_id}.spv"
                    _evidence_file(stage, path.parent, path.name, f"V4 {base.name} SPIR-V kernel {kernel_id}", refs, findings)
                    _check_digest_claim(digest, path if path.is_file() else None,
                                        f"V4 {base.name} SPIR-V kernel {kernel_id}", findings)
            if not validator.get("executable") or not validator.get("version"):
                _finding(findings, "product-validator", f"V4 {base.name} spirv-val identity/version missing")
            raw_hashes = fixture.get("reconstructed_shader_sha256", {})
            expected_by_kernel = {str(key): value for key, value in raw_hashes.items()} if isinstance(raw_hashes, dict) else {}
            if {str(key): value for key, value in shader_hashes.items()} != expected_by_kernel:
                _finding(findings, "product-shader-hash-map", f"V4 {base.name} manifest and reconstructed shader hash maps differ")
            shader_files = fixture.get("shader_files")
            if not isinstance(shader_files, list) or len(shader_files) != len(expected_by_kernel):
                _finding(findings, "product-shader-files", f"V4 {base.name} shader file list does not match reconstructed plan")
            else:
                for shader_record in shader_files:
                    if not isinstance(shader_record, dict):
                        _finding(findings, "product-shader-files", f"V4 {base.name} shader file record malformed")
                        continue
                    shader_path = _evidence_file(stage, base, shader_record.get("file"),
                                                 f"V4 {base.name} reconstructed SPIR-V", refs, findings)
                    _check_digest_claim(shader_record.get("sha256"), shader_path,
                                        f"V4 {base.name} reconstructed SPIR-V", findings)
            _audit_recorded_shaders(stage, base, evidence.get("shader_validations"),
                                    expected_by_kernel, f"V4 {base.name}", refs, findings)
            device_claim = execution.get("device_evidence", {})
            summary = execution.get("vulkaninfo_summary")
            physical_source = dict(device_claim) if isinstance(device_claim, dict) else {}
            physical_source["vulkaninfo_summary"] = summary
            physical_types = execution.get("physical_device_types", [])
            parsed_devices = _parse_physical_summary(summary) if isinstance(summary, str) else []
            if not isinstance(physical_types, list) or len(physical_types) != 1 or len(parsed_devices) != 1:
                _finding(findings, "product-physical-enumeration", f"V4 {base.name} expected exactly one physical device under explicit ICD")
            elif not physical_source.get("device_name"):
                physical = parsed_devices[0]
                physical_source.update({"device_name": physical.get("deviceName"),
                                        "api_version": physical.get("apiVersion"),
                                        "driver_version": physical.get("driverVersion"),
                                        "driver_info": physical.get("driverInfo"),
                                        "vendor_id": physical.get("vendorID"),
                                        "device_id": physical.get("deviceID"),
                                        "device_type": physical.get("deviceType")})
            if isinstance(physical_types, list) and len(physical_types) == 1:
                physical_source.setdefault("device_type", physical_types[0])
            identity = verify_physical_identity(stage, base, physical_source,
                                                "hardware" if _int_field(physical_source.get("vendor_id")) == 0x10DE else "software",
                                                refs, findings, f"V4 {base.name}")
            if identity:
                found.add("nvidia" if identity["vendor_id"] == 0x10DE else "llvmpipe")
                if len(physical_types) != 1 or physical_types[0] != identity["device_type"]:
                    _finding(findings, "product-device-type-mismatch", f"V4 {base.name} physical type declaration disagrees")
            if device_claim.get("status") != "success" or any(
                device_claim.get(key) != 0 for key in ("cpu_status", "vulkan_create_status", "vulkan_run_status", "vulkan_destroy_status")
            ):
                _finding(findings, "product-loader-status", f"V4 {base.name} CPU/Vulkan loader statuses are not all successful")
            _audit_v4_failure_records(stage, base,
                                      evidence.get("expected_failure_diagnostics"),
                                      fixture, refs, f"V4 {base.name}", findings)
            validation = evidence.get("validation", execution.get("validation", {}))
            if (not isinstance(validation, dict) or validation.get("layer") != "VK_LAYER_KHRONOS_validation"
                    or validation.get("error_count") != 0):
                _finding(findings, "product-validation-errors", f"V4 {base.name} lacks a zero-error enabled validation-layer report")
            else:
                validation_file = validation.get("log_file", validation.get("log", validation.get("file")))
                validation_log = _evidence_file(stage, base, validation_file,
                                                f"V4 {base.name} validation output", refs, findings)
                _check_digest_claim(validation.get("log_sha256", validation.get("sha256")),
                                    validation_log, f"V4 {base.name} validation output", findings)
                streams = validation.get("streams")
                expected_stream_names = ["vulkaninfo.stdout.log", "vulkaninfo.stderr.log",
                                         "vulkan-verify.stdout.log", "vulkan-verify.stderr.log",
                                         "vulkan-loader.stdout.log", "vulkan-loader.stderr.log"]
                stream_contents = []
                if not isinstance(streams, list) or [item.get("file") for item in streams if isinstance(item, dict)] != expected_stream_names:
                    _finding(findings, "product-validation-streams", f"V4 {base.name} raw validation streams are incomplete")
                    streams = streams if isinstance(streams, list) else []
                for item in streams:
                    if not isinstance(item, dict):
                        _finding(findings, "product-validation-stream", f"V4 {base.name} validation stream record malformed")
                        continue
                    stream_path = _evidence_file(stage, base, item.get("file"),
                                                 f"V4 {base.name} {item.get('file')}", refs, findings)
                    _check_digest_claim(item.get("sha256"), stream_path,
                                        f"V4 {base.name} {item.get('file')}", findings)
                    stream_contents.append((item.get("file"), stream_path.read_text(encoding="utf-8", errors="replace") if stream_path else ""))
                reconstructed = "".join(f"===== {name} =====\n{text}\n" for name, text in stream_contents)
                if validation_log and validation_log.read_text(encoding="utf-8", errors="replace") != reconstructed:
                    _finding(findings, "product-validation-log-mismatch", f"V4 {base.name} combined validation output differs from raw streams")
                error_re = re.compile(r"VUID-[A-Za-z0-9-]+|Validation Error|^\s*(?:ERROR|Error)\b",
                                      re.IGNORECASE | re.MULTILINE)
                actual_validation_errors = len(error_re.findall(reconstructed))
                if actual_validation_errors != validation.get("error_count"):
                    _finding(findings, "product-validation-count-mismatch", f"V4 {base.name} validation error count differs from archived stream bytes",
                             declared=validation.get("error_count"), actual=actual_validation_errors)
            records[base.name] = {"path": str(evidence_path.relative_to(stage)),

                                  "fixture": fixture, "identity": identity,
                                  "numeric": computed, "evidence": evidence}
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            _finding(findings, "product-evidence-invalid", f"V4 {base.name} evidence invalid: {error}")
    missing = sorted(expected_devices - found)
    if missing:
        _finding(findings, "product-device-coverage", "fresh V4 product CLI evidence lacks configured device(s)",
                 devices=missing)
    return records


_V5_MODULE = None


def _v5_golden_function():
    global _V5_MODULE
    if _V5_MODULE is None:
        path = Path(__file__).with_name("run_v5.py")
        spec = importlib.util.spec_from_file_location("ncnn_vulkan_run_v5_audit", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import run_v5 independent oracle")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _V5_MODULE = module
    return _V5_MODULE.independent_golden


def _read_raw_arrays(stage: Path, base: Path, paths: object, hashes: object,
                     label: str, refs: list[dict], findings: list[dict]) -> list[list[float]]:
    if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes):
        _finding(findings, "v5-raw-list", f"{label} path/hash arrays are missing or differ in length")
        return []
    result = []
    for index, (relative, digest) in enumerate(zip(paths, hashes)):
        path = _evidence_file(stage, base, relative, f"{label}[{index}]", refs, findings)
        if path is None:
            continue
        _check_digest_claim(digest, path, f"{label}[{index}]", findings)
        try:
            result.append(read_f32(path))
        except (OSError, ValueError) as error:
            _finding(findings, "v5-raw-invalid", f"{label}[{index}] is invalid: {error}")
    return result


def _check_v5_metrics(claims: object, computed: list[dict], label: str,
                      atol: float, rtol: float, findings: list[dict]) -> None:
    if not isinstance(claims, list) or len(claims) != len(computed):
        _finding(findings, "v5-metric-count", f"{label} metric array does not cover every output")
        return
    for index, (claim, actual) in enumerate(zip(claims, computed)):
        if not actual["passed"]:
            _finding(findings, "v5-numeric-failed", f"{label}[{index}] exceeds fixed numerical budget",
                     max_abs=actual["max_abs"], atol=atol, rtol=rtol)
        if not isinstance(claim, dict):
            _finding(findings, "v5-metric-invalid", f"{label}[{index}] metric claim is malformed")
            continue
        for key in ("max_abs", "max_rel"):
            value = claim.get(key)
            if not isinstance(value, (int, float)) or abs(float(value) - actual[key]) > max(1e-12, actual[key] * 1e-7):
                _finding(findings, "v5-metric-tampered", f"{label}[{index}].{key} differs from raw tensors")


def _find_hash_file(root: Path, digest: object) -> Path | None:
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or not root.is_dir():
        return None
    for path in root.rglob("*"):
        if path.is_file() and sha256_file(path) == digest:
            return path
    return None


def _v5_port_element_counts(ports: object, label: str, role: str,
                            findings: list[dict]) -> list[int] | None:
    if not isinstance(ports, list):
        _finding(findings, "v5-port-contract", f"{label} {role} port metadata is not a list")
        return None
    counts = []
    for index, port in enumerate(ports):
        shape = port.get("shape", port.get("shape_chw")) if isinstance(port, dict) else None
        if (not isinstance(shape, list) or not shape or
                any(type(dim) is not int or dim <= 0 for dim in shape)):
            _finding(findings, "v5-port-shape", f"{label} {role}[{index}] has invalid shape metadata")
            return None
        counts.append(math.prod(shape))
    return counts


def _v5_port_arrays_match(inputs: object, ncnn: object, cpu: object, vk: object,
                          input_counts: list[int] | None,
                          output_counts: list[int] | None,
                          label: str, findings: list[dict]) -> bool:
    arrays = (inputs, ncnn, cpu, vk)
    if (input_counts is None or output_counts is None or
            any(not isinstance(array, list) for array in arrays) or
            not inputs or not ncnn or not cpu or not vk or
            len(inputs) != len(input_counts) or
            len(ncnn) != len(output_counts) or
            len(cpu) != len(output_counts) or
            len(vk) != len(output_counts)):
        _finding(findings, "v5-trial-outputs", f"{label} arrays do not cover every declared port")
        return False
    contracts = (("input", inputs, input_counts), ("ncnn output", ncnn, output_counts),
                 ("CPU output", cpu, output_counts), ("Vulkan output", vk, output_counts))
    mismatches = [(role, index, len(values) if isinstance(values, list) else None, expected)
                  for role, port_arrays, expected_sizes in contracts
                  for index, (values, expected) in enumerate(zip(port_arrays, expected_sizes))
                  if not isinstance(values, list) or len(values) != expected]
    if mismatches:
        _finding(findings, "v5-port-element-count",
                 f"{label} FP32 element counts differ from declared port shapes",
                 mismatches=mismatches)
        return False
    return True


def _audit_v5(stage: Path, expected_devices: set[str], refs: list[dict],
              findings: list[dict]) -> dict:
    root = stage / "test/Vulkan/evidence/v5"
    records = {}
    roots = sorted(root.glob("*/evidence.json")) if root.is_dir() else []
    required_suites = {"multiop", "pooling-softmax", "squeezenet"}
    if not roots:
        _finding(findings, "v5-evidence-missing", "fresh V5 product evidence is absent")
        return records
    found: dict[str, set[str]] = {device: set() for device in expected_devices}
    fixture_identities = {}
    golden_fn = _v5_golden_function()
    for summary_path in roots:
        out_dir = summary_path.parent
        suite = ("pooling-softmax" if "pooling-softmax" in out_dir.name else
                 "squeezenet" if "squeezenet" in out_dir.name else "multiop")
        try:
            summary = load_json(summary_path)
            _evidence_file(stage, out_dir, "evidence.json", f"V5 {out_dir.name} summary", refs, findings)
            if not isinstance(summary, dict) or summary.get("schema_version") != 1:
                raise ValueError("V5 summary schema is invalid")
            summary_device = summary.get("device_info")
            physical_source = dict(summary_device) if isinstance(summary_device, dict) else {}
            physical_source["vulkaninfo_summary"] = summary.get("vulkaninfo_summary")
            # The physical row is the authority; selector folder/name is not.
            raw_vendor = _int_field(physical_source.get("vendor_id"))
            expected_class = "hardware" if raw_vendor == 0x10DE else "software"
            identity = verify_physical_identity(stage, out_dir, physical_source,
                                                expected_class, refs, findings,
                                                f"V5 {out_dir.name}")
            if identity:
                class_name = "nvidia" if identity["vendor_id"] == 0x10DE else "llvmpipe"
                if class_name in expected_devices:
                    found[class_name].add(suite)
                else:
                    _finding(findings, "v5-unrequested-device", f"V5 ran an unrequested physical device {class_name}")
            if summary.get("validation_layer") != "VK_LAYER_KHRONOS_validation":
                _finding(findings, "v5-validation-layer", f"V5 {out_dir.name} does not enable Khronos validation")
            expected_fixtures = summary.get("fixtures")
            fixture_roots = sorted((out_dir / "models").glob("*/evidence.json"))
            if not isinstance(expected_fixtures, list) or not expected_fixtures:
                _finding(findings, "v5-fixture-summary", f"V5 {out_dir.name} summary has no fixture inventory")
            summary_names = {item.get("fixture") for item in expected_fixtures if isinstance(item, dict)}
            model_names = {path.parent.name for path in fixture_roots}
            if summary_names != model_names:
                _finding(findings, "v5-summary-model-mismatch", f"V5 {out_dir.name} summary and per-model evidence differ",
                         missing=sorted(summary_names - model_names), extra=sorted(model_names - summary_names))
            if suite == "multiop" and not {"conv_relu", "double_conv", "split_concat", "add_relu",
                                            "repeated_input_alias", "copy_identity"}.issubset(model_names):
                _finding(findings, "v5.1-fixture-coverage", f"V5 {out_dir.name} misses multiop fixtures")
            if suite == "pooling-softmax" and not {"max_pool_tail", "global_avg_pool", "conv1x1_pad1",
                                                    "softmax_rank1", "softmax_rank3_axis2",
                                                    "concat_height_axis", "concat_width_axis"}.issubset(model_names):
                _finding(findings, "v5.2-fixture-coverage", f"V5 {out_dir.name} misses operator fixtures")
            if suite == "squeezenet" and model_names != {"squeezenet_v1_1"}:
                _finding(findings, "v5.3-fixture-coverage", f"V5 {out_dir.name} is not the full SqueezeNet fixture")
            for evidence_path in fixture_roots:
                model_root = evidence_path.parent
                try:
                    evidence = load_json(evidence_path)
                    _evidence_file(stage, model_root, "evidence.json",
                                   f"V5 {out_dir.name}/{model_root.name} evidence", refs, findings)
                    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
                        raise ValueError("per-model V5 evidence schema is invalid")
                    if evidence.get("validation", {}).get("error_count") != 0:
                        _finding(findings, "validation-errors", f"V5 {out_dir.name}/{model_root.name} validation error count is nonzero")
                    validation_log = _evidence_file(stage, model_root, "vulkan-validation.log",
                                                    f"V5 {out_dir.name}/{model_root.name} validation log", refs, findings)
                    if validation_log:
                        _check_digest_claim(evidence.get("validation", {}).get("log_sha256"),
                                            validation_log, f"V5 {model_root.name} validation log", findings)
                        text = validation_log.read_text(encoding="utf-8", errors="replace")
                        if re.search(r"VUID-[A-Za-z0-9-]+|Validation Error|AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:", text, re.I):
                            _finding(findings, "validation-log-error", f"V5 {model_root.name} raw validation log has errors")
                    model_identity = evidence.get("identities", {})
                    per_model_device = evidence.get("device_info")
                    if not isinstance(per_model_device, dict):
                        _finding(findings, "v5-model-device-identity", f"V5 {model_root.name} per-model device identity is missing")
                    else:
                        per_model_source = dict(per_model_device)
                        per_model_source["vulkaninfo_summary"] = summary.get("vulkaninfo_summary")
                        verify_physical_identity(stage, out_dir, per_model_source,
                                                 "hardware" if identity and identity["vendor_id"] == 0x10DE else "software",
                                                 refs, findings, f"V5 {out_dir.name}/{model_root.name}")
                    for role in ("param_sha256", "model_bin_sha256"):
                        raw_identity = model_identity.get(role)
                        matching_input = _find_hash_file(out_dir, raw_identity)
                        if matching_input is None:
                            _finding(findings, "v5-model-input-missing", f"V5 {model_root.name} {role} bytes are absent from fresh stage")
                        else:
                            _evidence_file(stage, matching_input.parent, matching_input.name,
                                           f"V5 {model_root.name} source {role}", refs, findings)
                    source_files = evidence.get("source_model_files", {})
                    for role, identity_field in (("param", "param_sha256"), ("bin", "model_bin_sha256")):
                        source_file = _evidence_file(stage, model_root, source_files.get(role),
                                                     f"V5 {model_root.name} source {role}", refs, findings)
                        _check_digest_claim(model_identity.get(identity_field), source_file,
                                            f"V5 {model_root.name} source {role}", findings)
                    oracle = evidence.get("independent_golden")
                    if model_root.name in {"conv_relu", "add_relu"}:
                        if not isinstance(oracle, dict) or oracle.get("kind") != model_root.name:
                            _finding(findings, "va0-small-graph-golden",
                                     f"{model_root.name} requires its independent multi-output scalar oracle")
                    if isinstance(oracle, dict) and oracle.get("kind") in {"conv_relu", "conv1x1"}:
                        binary = model_root / str(source_files.get("bin", ""))
                        if binary.is_file():
                            values = oracle.get("weights", []) + oracle.get("biases", [])
                            expected_bytes = struct.pack("<I", 0) + struct.pack(f"<{len(values)}f", *values)
                            if binary.read_bytes() != expected_bytes:
                                _finding(findings, "va0-golden-weight-source",
                                         "scalar oracle weights/biases differ from the actual model .bin bytes")
                    ports_in = evidence.get("inputs", [])
                    ports_out = evidence.get("outputs", [])
                    input_counts = _v5_port_element_counts(ports_in, f"V5 {model_root.name}",
                                                           "inputs", findings)
                    output_counts = _v5_port_element_counts(ports_out, f"V5 {model_root.name}",
                                                            "outputs", findings)
                    tolerance = evidence.get("tolerance", {})
                    atol, rtol = tolerance.get("atol"), tolerance.get("rtol")
                    if not isinstance(atol, (int, float)) or not isinstance(rtol, (int, float)) or (atol, rtol) != (1.0e-6, 1.0e-6 if suite != "squeezenet" else 1.0e-4):
                        # SqueezeNet keeps atol=1e-6/rtol=1e-4; micro fixtures are 1e-6/1e-6.
                        if not (suite == "squeezenet" and atol == 1.0e-6 and rtol == 1.0e-4):
                            _finding(findings, "v5-budget-invalid", f"V5 {model_root.name} tolerance differs from frozen budget")
                    tolerance = evidence.get("tolerance", {})
                    atol, rtol = float(tolerance.get("atol", 0)), float(tolerance.get("rtol", 0))
                    trial_ids = []
                    for trial in evidence.get("trials", []):
                        if not isinstance(trial, dict):
                            _finding(findings, "v5-trial-invalid", f"V5 {model_root.name} trial malformed")
                            continue
                        raw = trial.get("raw_files", {})
                        inputs = _read_raw_arrays(stage, model_root, raw.get("inputs"),
                                                  trial.get("input_sha256"),
                                                  f"V5 {model_root.name} {trial.get('name')} input", refs, findings)
                        ncnn = _read_raw_arrays(stage, model_root, raw.get("ncnn_outputs"),
                                                trial.get("ncnn_output_sha256"),
                                                f"V5 {model_root.name} {trial.get('name')} ncnn/golden", refs, findings)
                        cpu = _read_raw_arrays(stage, model_root, raw.get("cpu_outputs"),
                                               trial.get("cpu_output_sha256"),
                                               f"V5 {model_root.name} {trial.get('name')} CPU", refs, findings)
                        vk = _read_raw_arrays(stage, model_root, raw.get("vulkan_outputs"),
                                              trial.get("vulkan_output_sha256"),
                                              f"V5 {model_root.name} {trial.get('name')} Vulkan", refs, findings)
                        if not _v5_port_arrays_match(inputs, ncnn, cpu, vk,
                                                      input_counts, output_counts,
                                                      f"V5 {model_root.name} {trial.get('name')}",
                                                      findings):
                            continue
                        cpu_metrics = [compare_f32(left, right, atol, rtol) for left, right in zip(cpu, ncnn)]
                        vk_metrics = [compare_f32(left, right, atol, rtol) for left, right in zip(vk, ncnn)]
                        cross_metrics = [compare_f32(right, left, atol, rtol) for left, right in zip(cpu, vk)]
                        _check_v5_metrics(trial.get("cpu_vs_ncnn"), cpu_metrics,
                                          f"V5 {model_root.name} CPU/ncnn", atol, rtol, findings)
                        _check_v5_metrics(trial.get("vulkan_vs_ncnn"), vk_metrics,
                                          f"V5 {model_root.name} Vulkan/ncnn", atol, rtol, findings)
                        _check_v5_metrics(trial.get("cpu_vs_vulkan"), cross_metrics,
                                          f"V5 {model_root.name} CPU/Vulkan", atol, rtol, findings)
                        if evidence.get("independent_golden") is not None:
                            try:
                                fixture = {"inputs": [(port["blob"], tuple(port.get("shape", port.get("shape_chw"))))
                                                       for port in ports_in],
                                           "outputs": [(port["blob"], tuple(port.get("shape", port.get("shape_chw"))))
                                                        for port in ports_out],
                                           "oracle": evidence["independent_golden"]}
                                golden = golden_fn(fixture, inputs)
                                recorded_golden = _read_raw_arrays(
                                    stage, model_root, raw.get("golden_outputs"),
                                    trial.get("golden_output_sha256"),
                                    f"V5 {model_root.name} {trial.get('name')} scalar golden", refs, findings)
                                if (len(recorded_golden) != len(golden) or any(
                                        not compare_f32(actual, expected, atol, rtol)["passed"]
                                        for actual, expected in zip(recorded_golden, golden))):
                                    _finding(findings, "va0-golden-raw-mismatch",
                                             "recorded scalar golden bytes differ from independently recomputed inputs/weights")
                                if len(golden) != len(output_counts) or any(
                                        len(values) != expected for values, expected in zip(golden, output_counts)):
                                    raise ValueError("independent golden output count/shape differs from declared ports")
                                _check_v5_metrics(trial.get("independent_golden_vs_ncnn"),
                                                  [compare_f32(actual, expected, atol, rtol) for expected, actual in zip(golden, ncnn)],
                                                  f"V5 {model_root.name} golden/ncnn", atol, rtol, findings)
                                _check_v5_metrics(trial.get("independent_golden_vs_cpu"),
                                                  [compare_f32(actual, expected, atol, rtol) for expected, actual in zip(golden, cpu)],
                                                  f"V5 {model_root.name} golden/CPU", atol, rtol, findings)
                                _check_v5_metrics(trial.get("independent_golden_vs_vulkan"),
                                                  [compare_f32(actual, expected, atol, rtol) for expected, actual in zip(golden, vk)],
                                                  f"V5 {model_root.name} golden/Vulkan", atol, rtol, findings)
                            except (KeyError, TypeError, ValueError) as error:
                                _finding(findings, "v5-independent-golden", f"V5 {model_root.name} scalar oracle could not be recomputed: {error}")
                        trial_ids.append({"name": trial.get("name"),
                                          "input_sha256": trial.get("input_sha256"),
                                          "cpu_output_sha256": trial.get("cpu_output_sha256"),
                                          "ncnn_output_sha256": trial.get("ncnn_output_sha256"),
                                          "cpu_vs_ncnn": cpu_metrics,
                                          "vulkan_vs_ncnn": vk_metrics,
                                          "cpu_vs_vulkan": cross_metrics})
                    if not trial_ids:
                        _finding(findings, "v5-trials-missing", f"V5 {model_root.name} has no verified numeric trials")
                    _audit_v5_artifacts(stage, out_dir, model_root, evidence,
                                        refs, findings)
                    lifecycle = evidence.get("lifecycle", {})
                    planned = lifecycle.get("planned_resource_bytes")
                    budget = lifecycle.get("context_memory_budget_bytes")
                    if budget != 512 * 1024 * 1024 or not isinstance(planned, int) or planned < 0 or planned > budget:
                        _finding(findings, "v5-plan-resource-cap", f"V5 {model_root.name} planned resource total exceeds/omits runtime cap")
                    peak = lifecycle.get("peak_memory_bytes")
                    if peak is None:
                        if "not" not in str(lifecycle.get("peak_memory_evidence", "")).lower():
                            _finding(findings, "peak-memory-claim", f"V5 {model_root.name} must clearly state actual peak was not measured")
                    elif not isinstance(peak, int) or peak < 0 or peak > 512 * 1024 * 1024:
                        _finding(findings, "peak-memory-cap", f"V5 {model_root.name} measured peak exceeds 512 MiB")
                    if "planned_resource_bytes" in lifecycle and peak == lifecycle.get("planned_resource_bytes"):
                        _finding(findings, "plan-is-not-peak", f"V5 {model_root.name} treats plan bytes as peak allocation")
                    if suite == "squeezenet":
                        src_files = evidence.get("source_model_files", {})
                        for key in ("param", "bin"):
                            file_path = _evidence_file(stage, model_root, src_files.get(key),
                                                       f"V5.3 model {key}", refs, findings)
                            _check_digest_claim(src_files.get(key + "_sha256"), file_path,
                                                f"V5.3 model {key}", findings)
                    key = (suite, model_root.name)
                    identity_key = (suite, model_root.name, model_identity.get("param_sha256"),
                                    model_identity.get("model_bin_sha256"))
                    fixture_identities.setdefault(identity_key, []).append({"device": class_name if identity else None,
                                                                             "trials": trial_ids,
                                                                             "cpu_sha256": model_identity.get("cpu_library_sha256")})
                    records.setdefault(out_dir.name, {})[model_root.name] = {
                        "path": str(evidence_path.relative_to(stage)), "identity": identity,
                        "cpu_library_sha256": model_identity.get("cpu_library_sha256"),
                        "fixture": evidence.get("fixture"), "trials": trial_ids,
                        "peak_memory_bytes": peak, "planned_resource_bytes": planned}
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    _finding(findings, "v5-model-evidence-invalid", f"V5 {out_dir.name}/{model_root.name} evidence invalid: {error}")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            _finding(findings, "v5-summary-invalid", f"V5 {out_dir.name} summary invalid: {error}")
    for device in expected_devices:
        if found.get(device) != required_suites:
            _finding(findings, "v5-suite-device-coverage", f"V5 {device} missing a required suite",
                     missing=sorted(required_suites - found.get(device, set())))
    for copies in fixture_identities.values():
        device_set = {item["device"] for item in copies}
        if {"nvidia", "llvmpipe"}.issubset(expected_devices):
            if device_set != {"nvidia", "llvmpipe"}:
                _finding(findings, "v5-cross-device-coverage", "same fixture lacks both configured ICD outputs")
            elif len(copies) == 2:
                indexed = {item["device"]: item for item in copies}
                lhs = {trial.get("name"): trial for trial in indexed["nvidia"]["trials"]}
                rhs = {trial.get("name"): trial for trial in indexed["llvmpipe"]["trials"]}
                if set(lhs) != set(rhs):
                    _finding(findings, "v5-cross-device-trials", "same fixture trials differ across ICDs")
                for name in set(lhs) & set(rhs):
                    for field in ("input_sha256", "cpu_output_sha256", "ncnn_output_sha256"):
                        if lhs[name].get(field) != rhs[name].get(field):
                            _finding(findings, "v5-cross-device-input-mismatch",
                                     f"{name} {field} differs across Vulkan ICD runs")
    return records


def _audit_v5_artifacts(stage: Path, out_dir: Path, model_root: Path,
                        evidence: dict, refs: list[dict], findings: list[dict]) -> None:
    identity = evidence.get("identities", {})
    model = evidence.get("model")
    cpu_dir, vk_dir = model_root / "cpu", model_root / "vulkan"
    paths = {"cpu_library_sha256": cpu_dir / f"lib{model}.so",
             "vulkan_library_sha256": vk_dir / f"lib{model}_vulkan.so",
             "plan_sha256": vk_dir / f"{model}_vulkan.plan.json",
             "cpu_manifest_sha256": cpu_dir / f"{model}.json",
             "cpu_header_sha256": cpu_dir / f"{model}.h",
             "vulkan_manifest_sha256": vk_dir / f"{model}_vulkan.json",
             "vulkan_header_sha256": vk_dir / f"{model}_vulkan.h"}
    for field, path in paths.items():
        _evidence_file(stage, path.parent, path.name, f"V5 {model} {field}", refs, findings)
        if field.endswith("sha256"):
            _check_digest_claim(identity.get(field), path if path.is_file() else None,
                                f"V5 {model} {field}", findings)
    plan = None
    plan_hash = identity.get("plan_sha256")
    if paths["plan_sha256"].is_file():
        _check_digest_claim(plan_hash, paths["plan_sha256"], f"V5 {model} plan", findings)
        try:
            plan = load_json(paths["plan_sha256"])
            for resource in plan.get("resources", []):
                size = resource.get("byte_size")
                if type(size) is not int or size < 0 or size > 128 * 1024 * 1024:
                    _finding(findings, "v5-single-buffer-cap", f"V5 {model} has a planned buffer over 128 MiB")
        except (OSError, ValueError, AttributeError, json.JSONDecodeError) as error:
            _finding(findings, "v5-plan-invalid", f"V5 {model} plan JSON invalid: {error}")
    if not isinstance(plan, dict):
        plan = {}

    plan_kernels = plan.get("kernels", [])
    plan_by_id = {}
    if not isinstance(plan_kernels, list) or not plan_kernels:
        _finding(findings, "v5-plan-kernels", f"V5 {model} plan has no kernel inventory")
        plan_kernels = []
    for kernel in plan_kernels:
        if not isinstance(kernel, dict) or kernel.get("id") is None or not isinstance(kernel.get("entry_point"), str):
            _finding(findings, "v5-plan-kernel", f"V5 {model} plan kernel identity is malformed")
            continue
        kernel_id = str(kernel["id"])
        if kernel_id in plan_by_id:
            _finding(findings, "v5-plan-kernel-duplicate", f"V5 {model} plan repeats kernel id {kernel_id}")
        plan_by_id[kernel_id] = kernel["entry_point"]

    manifest = None
    manifest_path = paths["vulkan_manifest_sha256"]
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "v5-manifest-invalid", f"V5 {model} Vulkan manifest JSON invalid: {error}")
    manifest = manifest if isinstance(manifest, dict) else {}
    _audit_current_host_contract(manifest, f"V5 {model}", findings)
    if manifest.get("host_program") is not None:
        try:
            raw_shader_files = {}
            validation_records = identity.get("shader_validations", [])
            validation_directory = identity.get("shader_validation_directory", "spirv-validation")
            if not isinstance(validation_records, list):
                raise ValueError("V5 host-program shader raw inventory is malformed")
            for record in validation_records:
                if not isinstance(record, dict) or type(record.get("kernel_id")) is not int:
                    raise ValueError("V5 host-program shader raw record lacks kernel ID")
                relative = Path(str(validation_directory)) / str(record.get("path", ""))
                raw_path = _evidence_file(stage, model_root, relative.as_posix(),
                                          f"V5 {model} host-program raw shader",
                                          refs, findings)
                if raw_path is None:
                    raise ValueError("V5 host-program shader raw bytes are missing")
                raw_shader_files[record["kernel_id"]] = raw_path.read_bytes()
            host_result = host_program_audit.validate_host_program(
                manifest, plan, shader_bytes=raw_shader_files,
                manifest_sha256=sha256_file(manifest_path),
                plan_sha256=sha256_file(paths["plan_sha256"]))
            if (identity.get("weight_manifest_entries") != manifest.get("weights") or
                    identity.get("host_program_identity") not in
                    (None, host_result["identity_sha256"])):
                _finding(findings, "v5-host-program-weight-identity",
                         f"V5 {model} host-program identity/weights differ from actual manifest")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
                host_program_audit.EvidenceError) as error:
            _finding(findings, "v5-host-program-invalid",
                     f"V5 {model} host-program validation failed: {error}")
    elif _stage_id(stage) == "VA1":
        _finding(findings, "v5-host-program-missing",
                 f"VA1 V5 {model} manifest omits the compiler-only host program")
    if plan_hash and manifest.get("plan_sha256") != plan_hash:
        _finding(findings, "v5-plan-manifest-mismatch", f"V5 {model} Vulkan manifest does not bind the audited plan bytes")
    manifest_shaders = manifest.get("shaders", [])
    manifest_by_id = {}
    if not isinstance(manifest_shaders, list) or not manifest_shaders:
        _finding(findings, "v5-manifest-shaders", f"V5 {model} Vulkan manifest has no shader inventory")
        manifest_shaders = []
    for shader in manifest_shaders:
        if (not isinstance(shader, dict) or shader.get("kernel_id") is None or
                not SHA256_RE.fullmatch(str(shader.get("sha256", ""))) or
                not isinstance(shader.get("entry_point"), str)):
            _finding(findings, "v5-manifest-shader", f"V5 {model} manifest shader identity is malformed")
            continue
        kernel_id = str(shader["kernel_id"])
        if kernel_id in manifest_by_id:
            _finding(findings, "v5-manifest-shader-duplicate", f"V5 {model} manifest repeats kernel id {kernel_id}")
        manifest_by_id[kernel_id] = {"sha256": shader["sha256"],
                                     "entry_point": shader["entry_point"],
                                     "size_bytes": shader.get("size_bytes")}
    if set(manifest_by_id) != set(plan_by_id):
        _finding(findings, "v5-plan-manifest-kernel-map",
                 f"V5 {model} plan kernel IDs and Vulkan manifest shader IDs differ",
                 plan=sorted(plan_by_id), manifest=sorted(manifest_by_id))
    for kernel_id in set(plan_by_id) & set(manifest_by_id):
        if plan_by_id[kernel_id] != manifest_by_id[kernel_id]["entry_point"]:
            _finding(findings, "v5-plan-manifest-entry-point",
                     f"V5 {model} kernel {kernel_id} manifest entry point differs from plan")

    entries = identity.get("shader_entries")
    if not isinstance(entries, list) or not entries:
        _finding(findings, "v5-shader-identity", f"V5 {model} has no per-kernel shader identities")
        entries = []
    identity_by_id = {}
    for entry in entries:
        if (not isinstance(entry, dict) or not SHA256_RE.fullmatch(str(entry.get("sha256", ""))) or
                entry.get("kernel_id") is None or not isinstance(entry.get("entry_point"), str)):
            _finding(findings, "v5-shader-identity", f"V5 {model} shader entry malformed")
            continue
        kernel_id = str(entry["kernel_id"])
        if kernel_id in identity_by_id:
            _finding(findings, "v5-shader-identity", f"V5 {model} identity repeats kernel id {kernel_id}")
        identity_by_id[kernel_id] = {"sha256": entry["sha256"],
                                     "entry_point": entry["entry_point"]}
    if set(identity_by_id) != set(manifest_by_id):
        _finding(findings, "v5-identity-manifest-kernel-map",
                 f"V5 {model} self-asserted shader IDs differ from Vulkan manifest",
                 identity=sorted(identity_by_id), manifest=sorted(manifest_by_id))
    for kernel_id in set(identity_by_id) & set(manifest_by_id):
        if identity_by_id[kernel_id] != {key: manifest_by_id[kernel_id][key]
                                         for key in ("sha256", "entry_point")}:
            _finding(findings, "v5-identity-manifest-shader-mismatch",
                     f"V5 {model} identity shader differs from manifest kernel {kernel_id}")
    expected_by_kernel = {kernel_id: manifest_by_id[kernel_id]["sha256"]
                          for kernel_id in manifest_by_id}

    validations = identity.get("shader_validations")
    validation_directory = identity.get("shader_validation_directory", "spirv-validation")
    if not isinstance(validation_directory, str) or Path(validation_directory).is_absolute() or ".." in Path(validation_directory).parts:
        _finding(findings, "v5-validator-directory", f"V5 {model} validator directory is unsafe")
        validation_directory = "spirv-validation"
    normalized = []
    if not isinstance(validations, list):
        _finding(findings, "v5-validator-coverage", f"V5 {model} validator records are missing")
        validations = []
    for record in validations:
        if not isinstance(record, dict):
            _finding(findings, "v5-validator-record", f"V5 {model} validator record is malformed")
            continue
        if record.get("kernel_id") is None:
            _finding(findings, "v5-validator-kernel-id",
                     f"V5 {model} validator record lacks an explicit kernel id")
        normalized.append({
            "kernel_id": record.get("kernel_id"),
            "log_file": str(Path(validation_directory) / str(record.get("log", ""))),
            "log_sha256": record.get("log_sha256"),
            "shader_file": str(Path(validation_directory) / str(record.get("path", ""))),
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
        })
    _audit_recorded_shaders(stage, model_root, normalized, expected_by_kernel,
                            f"V5 {model}", refs, findings)


def _audit_cpu_baseline(stage: Path, refs: list[dict], findings: list[dict]) -> dict | None:
    base = stage / "test/Native/vulkan-v1-cpu-baseline"
    path = _evidence_file(stage, base, "evidence.json", "fresh V1 CPU baseline evidence", refs, findings)
    if path is None:
        _finding(findings, "cpu-baseline-missing", "fresh V1 CPU baseline result is absent")
        return None
    try:
        evidence = load_json(path)
        if (not isinstance(evidence, dict) or evidence.get("schema_version") != 1
                or evidence.get("kind") != "ncnn.vulkan_cpu_baseline_evidence"):
            raise ValueError("unsupported CPU baseline evidence schema/kind")
        if evidence.get("status") != 0 or evidence.get("model") != "relu_v1_baseline":
            _finding(findings, "cpu-baseline-status", "fresh CPU baseline did not compile/call the expected ReLU fixture")
        shape = evidence.get("shape")
        if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            _finding(findings, "cpu-baseline-shape", "CPU baseline shape is invalid")
            return evidence
        count = math.prod(shape)
        expected_input = struct.pack(f"<{count}f", *((index - count // 2) / 10.0 for index in range(count)))
        input_values = list(struct.unpack(f"<{count}f", expected_input))
        expected_output = struct.pack(f"<{count}f", *(max(value, 0.0) for value in input_values))
        for key, data in (("input_sha256", expected_input),
                          ("expected_output_sha256", expected_output)):
            if evidence.get(key) != sha256_bytes(data):
                _finding(findings, "cpu-baseline-golden", f"CPU baseline {key} differs from deterministic fixture")
        if evidence.get("output_sha256") != evidence.get("expected_output_sha256"):
            _finding(findings, "cpu-baseline-output", "CPU baseline output differs from independent ReLU golden")
        tolerance = evidence.get("atol")
        if not isinstance(tolerance, (int, float)) or tolerance > 1.0e-6:
            _finding(findings, "cpu-baseline-budget", "CPU baseline tolerance is missing or exceeds frozen 1e-6")
        artifact_hashes = evidence.get("artifact_sha256", {})
        artifact_files = evidence.get("artifact_files", {})
        if not isinstance(artifact_hashes, dict) or not isinstance(artifact_files, dict) or set(artifact_hashes) != set(artifact_files):
            _finding(findings, "cpu-baseline-artifact-list", "CPU baseline artifact paths/hashes do not match")
        else:
            for field, digest in artifact_hashes.items():
                artifact = _evidence_file(stage, base, artifact_files[field],
                                          f"CPU baseline {field}", refs, findings)
                _check_digest_claim(digest, artifact, f"CPU baseline {field}", findings)
        library = _evidence_file(stage, base, evidence.get("cpu_library"),
                                 "fresh CPU baseline shared library", refs, findings)
        _check_digest_claim(evidence.get("cpu_library_sha256"), library,
                            "fresh CPU baseline shared library", findings)
        source_files = evidence.get("source_model_files", {})
        for role in ("param", "bin"):
            source_file = _evidence_file(stage, base, source_files.get(role),
                                         f"CPU baseline source model {role}", refs, findings)
            _check_digest_claim(source_files.get(role + "_sha256"), source_file,
                                f"CPU baseline source model {role}", findings)
        raw_files = (("input", evidence.get("input_file"), evidence.get("input_sha256"), expected_input),
                     ("cpu output", evidence.get("output_file"), evidence.get("output_sha256"), expected_output),
                     ("scalar golden", evidence.get("golden_file"), evidence.get("expected_output_sha256"), expected_output))
        for role, filename, digest, expected_bytes in raw_files:
            tensor_path = _evidence_file(stage, base, filename, f"CPU baseline {role} FP32", refs, findings)
            _check_digest_claim(digest, tensor_path, f"CPU baseline {role}", findings)
            if tensor_path and tensor_path.read_bytes() != expected_bytes:
                _finding(findings, "cpu-baseline-raw-mismatch", f"CPU baseline {role} bytes differ from oracle")
        return evidence
    except (OSError, ValueError, TypeError, KeyError, struct.error, json.JSONDecodeError) as error:
        _finding(findings, "cpu-baseline-invalid", f"CPU baseline evidence is invalid: {error}")
        return None


V2_CASES = ("add", "relu", "copy", "identity", "transpose", "reshape",
            "conv3x3", "conv3x3-no-bias", "conv3x3-stride2", "conv1x1")
V4_FAILURE_FLAGS = {
    "invalid-backend": "--backend=unknown",
    "cpu-threads": "--threads=1", "debug-info": "--debug-info",
    "low-precision": "--precision=fp16", "fp16-accumulator": "--fp16-accumulator=f32",
    "allow-fallback": "--allow-fallback", "optimization": "-O0",
    "vector-width": "--vector-width=0", "vector-mode": "--vector-mode=off",
    "int8-kernel": "--int8-kernel=portable", "int8-depthwise": "--int8-depthwise",
    "packed-conv-depthwise": "--packed-conv-depthwise", "int8-cast-chain": "--int8-cast-chain",
    "selective-fusion": "--selective-fusion=true", "layout-aware-fusion": "--layout-aware-fusion=true",
    "fusion-broadcast": "--selective-fusion-broadcast=true",
    "fusion-cast-chain": "--selective-fusion-cast-chain=true",
    "fusion-max-chain": "--selective-fusion-max-chain=4",
    "tuning-profile": "--tuning-profile=stable", "matmul-m-rows": "--matmul-m-rows=2",
    "matmul-acc-columns": "--matmul-acc-columns=2", "matmul-packing": "--matmul-packing=off",
    "row-chunk-lanes": "--row-chunk-lanes=4", "matmul-i8-rows": "--matmul-i8-rows=1",
    "matmul-i8-acc-columns": "--matmul-i8-acc-columns=2", "conv-strategy": "--conv-strategy=conv",
    "vector-math": "--vector-math=none", "sleef-path": "--sleef-path=/tmp",
    "conv-gemm-l2": "--conv-gemm-l2-bytes=1024", "dynamic-shape": "--input-shape=1x?x?",
    "dynamic-constraint": "--input-dim-constraint=0:0:min=1,multiple=1",
    "cpu-target": "--target-triple=x86_64-pc-linux-gnu", "cpu-march": "--march=x86-64",
    "cpu-mcpu": "--mcpu=x86-64", "cpu-mtune": "--mtune=generic",
    "cpu-target-feature": "--target-feature=+sse2", "cpu-sysroot": "--sysroot=/tmp",
    "clang-arg": "--clang-arg=-ffast-math", "linker-arg": "--linker-arg=-lm",
    "emit-stage": "--emit=ncnn", "emit-manifest": "--emit-manifest",
    "emit-plan": "--emit-execution-plan", "profile": "--profile",
    "expected-undefined": "--expected-undefined=malloc",
}
V4_UNSUPPORTED_SIGMOID_PARAM = (
    b"7767517\n2 2\nInput input 0 1 data 0=4 1=4 2=1\n"
    b"Sigmoid unsupported_sigmoid 1 1 data output\n"
)
V4_FAILURE_CASES = frozenset({
    "foo", "foo_vulkan", "unsupported", "invalid-backend", *V4_FAILURE_FLAGS,
    "output-user-data", "cpu-vulkan-output-collision", "missing-validator",
    "validator-failure", "link-failure", "audit-failure",
    "missing-device-selector", "verify-execution-no-device",
})
V5_CASE_TESTS = {
    "multiop": "vulkan-v5-multiop",
    "pooling-softmax": "vulkan-v5-pooling-softmax",
    "squeezenet": "vulkan-v5-squeezenet",
}


OFFLINE_RUNTIME_TESTS = frozenset({
    "VulkanRuntimeTest.RejectsInvalidPlanBeforeCreatingDevice",
    "VulkanRuntimeTest.RejectsMissingKernelPackage",
    "VulkanRuntimeTest.RejectsZeroMemoryBudgetBeforeCreatingDevice",
    "VulkanRuntimeTest.InjectedInstanceFailureReturnsStatusAndDiagnostic",
    "VulkanRuntimeTest.StatusNamesAreStableAndTotal",
})


OFFLINE_PRODUCT_TESTS = frozenset(
    "VulkanProductTest." + name for name in (
        "PackagesReproduciblyAndEmbedsBothIdentities",
        "ManifestPinsHostExecutionContractAndIdentityDomain",
        "RejectsMalformedShaderAndUnsupportedGraph",
        "ChecksSpirvEntryPointWorkgroupAndDescriptors",
        "PackagesGeneralPlanByKernelAndResourceIdentity",
        "GeneralSingleTensorAbiKeepsPointerAndCountSignature",
        "ManifestHashesPhysicalWeightsNotLogicalLayout",
        "BuildIdentityIncludesStaticPublicPlanContract",
        "WeightSidecarIsStrictAndVerifiesEveryPayload",
        "LogicalWeightHashNormalizesBothConvLayouts",
    ))
OFFLINE_PLAN_TESTS = frozenset(
    "VulkanExecutionPlanTest." + name for name in (
        "SerializesDeterministicallyAndRoundTrips",
        "PreservesUnusedPublicInputLifetimeSentinel",
        "RejectsUnusedSentinelForAccessedOrNonInputResource",
        "RejectsDuplicateDescriptorBindings", "RejectsReadBeforeWrite",
        "RejectsInvalidDispatchAndPushConstants", "RejectsOverflowAndByteSizeMismatch",
        "RejectsInvalidHostDeviceLayoutMappings", "RejectsInvalidLifetimeAndSchemaVersion",
        "RejectsUnsupportedJsonFieldsAndHashTampering",
    ))
OFFLINE_HOST_TESTS = OFFLINE_RUNTIME_TESTS | OFFLINE_PRODUCT_TESTS | OFFLINE_PLAN_TESTS


VA1_HOST_PROGRAM_TESTS = frozenset(
    f"VulkanHostProgramTest.{name}" for name in (
        "ChainMatchesLegacyTrace", "BranchAndRepeatedOutputs",
        "WeightsAndUnusedInput", "WawWarAndReadRead",
        "RejectsReferencesLifetimesAndBindings", "RejectsDispatchAndOverflow",
        "RejectsTamperedOperations", "IdentityIsCanonicalAndSensitive"))
VA1_SHADER_INTERFACE_TESTS = frozenset(
    f"VulkanShaderInterfaceTest.{name}" for name in (
        "ReflectsTypedBindingsEntryAndPushRanges",
        "ReflectsShaderWithoutPushConstantBlock",
        "RejectsTruncatedWordsAndWrongTargetVersion",
        "RejectsMismatchedEntryAndLocalSize",
        "RejectsBindingTypeExtentAndActualAccessMismatch",
        "RejectsUnsupportedPointerFlow",
        "RequiresPlanCoverageForActualPushReads",
        "RequiresKernelAndShaderIdsToMatchPlan"))
VA1_REQUIRED_TESTS = (VA1_HOST_PROGRAM_TESTS | VA1_SHADER_INTERFACE_TESTS |
                      {"vulkan-host-program-evidence-tests"})
VA2_HELPER_TESTS = frozenset({
    "vulkan-device-helper-unit",
    "vulkan-device-helper-no-device",
    "vulkan-device-helper-independence",
    "vulkan-device-helper-evidence-tests",
    "vulkan-device-helper-add-nvidia",
    "vulkan-device-helper-add-llvmpipe",
})
VA2_RUNTIME_BUDGET_TESTS = device_helper_audit.RUNTIME_BUDGET_TESTS
VA2_REQUIRED_TESTS = VA2_HELPER_TESTS | VA2_RUNTIME_BUDGET_TESTS
VA2_STEP_TESTS = {
    "VA2.1": frozenset({"vulkan-device-helper-unit"}),
    "VA2.2": frozenset({"vulkan-device-helper-unit",
                         "vulkan-device-helper-no-device",
                         *VA2_RUNTIME_BUDGET_TESTS - {
                             "VulkanRuntimeBudgetTest.PartialBindFailureReleasesEveryOwnedObject"}}),
    "VA2.3": frozenset({"vulkan-device-helper-independence",
                         "vulkan-device-helper-evidence-tests",
                         "VulkanRuntimeBudgetTest.PartialBindFailureReleasesEveryOwnedObject"}),
    "VA2.4": frozenset({"vulkan-device-helper-add-nvidia",
                         "vulkan-device-helper-add-llvmpipe",
                         "vulkan-device-helper-no-device"}),
}
VA3_DEVICE_TESTS = frozenset({"vulkan-lifecycle-nvidia", "vulkan-lifecycle-llvmpipe"})
VA3_REQUIRED_TESTS = frozenset({
    "vulkan-lifecycle-fake-unit",
    "vulkan-lifecycle-evidence-tests",
    "vulkan-lifecycle-independence",
    "vulkan-lifecycle-no-device",
    *VA3_DEVICE_TESTS,
})
VA3_STEP_TESTS = {
    "VA3.1": frozenset({
        "VulkanLifecycleCodegenTest.ProducesSameHeaderAndDeterministicSource",
        "VulkanLifecycleCodegenTest.SparseIdsBecomeDenseOwnedSlots",
        "VulkanLifecycleCodegenTest.EmptyWeightsAndPushRangesUseStandardArrays",
        "VulkanLifecycleCodegenTest.MultiplePublicPortsRemainDistinct",
        "VulkanLifecycleCodegenTest.EntryPointBytesAreNotCppSourceOrCStringLengths",
        "VulkanLifecycleCodegenTest.RejectsUnsafeOrReservedModelIdentifiers",
        "VulkanLifecycleCodegenTest.ReservedPrefixesFailBeforeHeaderOrSourceChecks",
        "VulkanLifecycleCodegenTest.KeywordPrefixProducesLegalSuffixedIdentifiers",
        "VulkanLifecycleCodegenTest.HeaderGuardUsesAsciiCaseMapping",
        "VulkanLifecycleCodegenTest.RejectsMutatedHeaderAndHostIdentity",
        "VulkanLifecycleCodegenTest.RejectsByteAndPushRangeOverflow",
        "VulkanLifecycleCodegenTest.PreservesOriginalSingleShaderPackagingHeader",
        "VulkanLifecycleCodegenTest.RepeatedPublicResourceIdsAreRejected",
    }),
    "VA3.2": frozenset({
        "vulkan-lifecycle-independence",
        "vulkan-lifecycle-evidence-tests",
        "VulkanLifecycleCodegenTest.CandidateDoesNotChangeLegacyProductSource",
        "VulkanLifecycleCodegenTest.UsesOnlyDeviceHelperInDeploymentSource",
        "VulkanLifecycleCodegenTest.RejectsRuntimeCommandsInCreateOperations",
    }),
    "VA3.3": frozenset({
        "vulkan-lifecycle-fake-unit",
        *VA3_DEVICE_TESTS,
        "vulkan-lifecycle-evidence-tests",
        "VulkanLifecycleCodegenTest.BudgetPrecedesAllBufferAllocations",
        "VulkanLifecycleCodegenTest.MultipleWeightsHaveFixedCopiesAndRelease",
        "VulkanLifecycleCodegenTest.UsesReversePipelineAndResourceOwnership",
        "VulkanLifecycleCodegenTest.RejectsShaderAndWeightPayloadMutation",
        "VulkanLifecycleCodegenTest.EmbeddedWeightsMatchVerifiedPayloadBytes",
    }),
    "VA3.4": frozenset({
        "vulkan-lifecycle-no-device",
        "vulkan-lifecycle-fake-unit",
        "vulkan-lifecycle-independence",
        "vulkan-lifecycle-evidence-tests",
        *VA3_DEVICE_TESTS,
        "VulkanLifecycleCodegenTest.UsesReversePipelineAndResourceOwnership",
        "VulkanLifecycleCodegenTest.RejectsIllegalReferencesAndLifetimeChanges",
    }),
}
VA3_CODEGEN_TEST_PREFIXES = (lifecycle_audit.CODEGEN_TEST_PREFIX,)


def _va3_codegen_tests() -> set[str]:
    source = Path(__file__).resolve().parents[1] / "Unit/vulkan_lifecycle_codegen_test.cpp"
    return set(lifecycle_audit.codegen_tests_from_source(source))


VA4_DEVICE_TESTS = frozenset({
    "vulkan-fixedrun-nvidia", "vulkan-fixedrun-llvmpipe"})
VA4_FIXEDRUN_TESTS = frozenset({
    "vulkan-fixedrun-capture-publish",
    "vulkan-fixedrun-fake-unit",
    "vulkan-fixedrun-evidence-validation",
    "vulkan-fixedrun-independence",
    "vulkan-fixedrun-legacy-reference-tests",
    "vulkan-fixedrun-producer-tool-role-validation",
    *VA4_DEVICE_TESTS,
})


def _va4_codegen_tests() -> set[str]:
    source = Path(__file__).resolve().parents[1] / "Unit/vulkan_host_codegen_test.cpp"
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"VA4 codegen test source cannot be read: {error}") from error
    declarations = re.findall(
        r"TEST\(VulkanFixedRunCodegenTest,\s*([A-Za-z0-9_]+)\)", text)
    declared = {f"VulkanFixedRunCodegenTest.{name}" for name in declarations}
    if not declared or len(declared) != len(declarations):
        raise RuntimeError("VA4 fixed-run codegen TEST declarations are empty or duplicated")
    return declared


VA4_STEP_TESTS = {
    "VA4.1": frozenset({
        "vulkan-fixedrun-capture-publish",
        "vulkan-fixedrun-independence",
    }),
    "VA4.2": frozenset({
        "vulkan-fixedrun-fake-unit",
    }),
    "VA4.3": frozenset({
        *VA4_DEVICE_TESTS,
    }),
    "VA4.4": frozenset({
        "vulkan-fixedrun-evidence-validation",
        "vulkan-fixedrun-independence",
        "vulkan-fixedrun-legacy-reference-tests",
        "vulkan-fixedrun-producer-tool-role-validation",
    }),
}


VA5_DEVICE_DEPLOYMENT_TESTS = frozenset({
    "vulkan-va5-deployment-nvidia",
    "vulkan-va5-deployment-llvmpipe",
})

VA5_STEP_TESTS = {
    # General product/offline tests are already required for every profile;
    # the VA5 IDs carry only the deployment-specific additions.
    "VA5.1": frozenset({
        "vulkan-va5-deployment-tests",
    }),
    "VA5.2": frozenset({
        "vulkan-va5-deployment-tests",
    }),
    "VA5.3": frozenset({
        "vulkan-va5-deployment-no-device",
        *VA5_DEVICE_DEPLOYMENT_TESTS,
    }),
    "VA5.4": frozenset({
        "vulkan-va5-deployment-tests",
        "vulkan-va5-deployment-no-device",
    }),
}

VA5_REQUIRED_TESTS = frozenset().union(*VA5_STEP_TESTS.values())


VA6_DEVICE_CROSS_ICD_TESTS = frozenset({
    "vulkan-va6-cross-icd",
})

VA6_STEP_TESTS = {
    # The matrix runs every existing product support row through the public
    # CLI; the fault chain executes the CLI-generated wrapper; the cross-ICD
    # scenario proves one fixed DSO on both ICDs plus explicit no-device
    # failure.  The existing v5 device suites stay required as the numeric
    # evidence for the positive matrix rows.
    "VA6.1": frozenset({
        "vulkan-va6-product-matrix",
        "vulkan-v5-multiop-nvidia", "vulkan-v5-multiop-llvmpipe",
        "vulkan-v5-pooling-softmax-nvidia", "vulkan-v5-pooling-softmax-llvmpipe",
        "vulkan-v5-squeezenet-nvidia", "vulkan-v5-squeezenet-llvmpipe",
    }),
    "VA6.2": frozenset({
        "vulkan-va6-host-faults",
    }),
    "VA6.3": frozenset({
        *VA6_DEVICE_CROSS_ICD_TESTS,
    }),
    "VA6.4": frozenset({
        "vulkan-va6-evidence-tests",
    }),
}

VA6_REQUIRED_TESTS = frozenset().union(*VA6_STEP_TESTS.values())


VA1_STEP_TESTS = {
    "VA1.1": frozenset({
        "VulkanHostProgramTest.RejectsReferencesLifetimesAndBindings",
        "VulkanHostProgramTest.WeightsAndUnusedInput",
        *VA1_SHADER_INTERFACE_TESTS}),
    "VA1.2": frozenset({
        "VulkanHostProgramTest.ChainMatchesLegacyTrace",
        "VulkanHostProgramTest.BranchAndRepeatedOutputs",
        "VulkanHostProgramTest.WawWarAndReadRead"}),
    "VA1.3": frozenset({
        "VulkanHostProgramTest.IdentityIsCanonicalAndSensitive",
        "VulkanHostProgramTest.RejectsTamperedOperations",
        "vulkan-host-program-evidence-tests"}),
    "VA1.4": frozenset({
        "VulkanHostProgramTest.RejectsDispatchAndOverflow",
        "VulkanHostProgramTest.RejectsTamperedOperations",
        "vulkan-host-program-evidence-tests"}),
}


def _required_tests(profile: str, stage_id: str = "VA0") -> set[str]:
    tests = {"vulkan-abi-c-contract", "vulkan-v1-cpu-baseline",
             "vulkan-release-evidence-validation", "vulkan-gate-validation",
             "ncnn-compile-vulkan-offline", "vulkan-v5-comparison-validation",
             "vulkan-va0-evidence-validation", "vulkan-va0-baseline-validation",
             "vulkan-model-dependency-evidence-validation",
             "vulkan-sanitizer-toolchain-inventory-validation",
             "vulkan-sanitizer-toolchain-v2-validation",
             "ncnn-mlir-lit", *OFFLINE_HOST_TESTS}
    devices = {"release": ("nvidia", "llvmpipe"),
               "sanitize": ("nvidia", "llvmpipe"),
               "software": ("llvmpipe",), "hardware": ("nvidia",)}.get(profile, ())
    for device in devices:
        selector = "NVIDIA" if device == "nvidia" else "llvmpipe"
        tests.add(f"vulkan-v0-add-{device}")
        tests.update(f"vulkan-v2-{case}-{device}" for case in V2_CASES)
        tests.add(f"vulkan-v3-runtime-{device}")
        tests.add(f"vulkan-v5-multiop-{device}")
        tests.add(f"vulkan-v5-pooling-softmax-{device}")
        tests.add(f"vulkan-v5-squeezenet-{device}")
    if profile in {"release", "sanitize"}:
        tests.update(("ncnn-compile-vulkan-product",
                      "ncnn-compile-vulkan-product-llvmpipe"))
    elif profile in {"software", "hardware"}:
        tests.add("ncnn-compile-vulkan-product")
    if stage_id == "VA1":
        tests.update(VA1_REQUIRED_TESTS)
    elif stage_id == "VA2":
        tests.update(VA2_REQUIRED_TESTS)
    elif stage_id == "VA3":
        tests.update(VA1_REQUIRED_TESTS)
        tests.update(VA2_REQUIRED_TESTS)
        tests.update(VA3_REQUIRED_TESTS)
        tests.update(_va3_codegen_tests())
    elif stage_id == "VA4":
        tests.update(VA1_REQUIRED_TESTS)
        tests.update(VA2_REQUIRED_TESTS)
        tests.update(VA3_REQUIRED_TESTS)
        tests.update(_va3_codegen_tests())
        tests.update(VA4_FIXEDRUN_TESTS)
        tests.update(_va4_codegen_tests())
    elif stage_id == "VA5":
        tests.update(VA1_REQUIRED_TESTS)
        tests.update(VA2_REQUIRED_TESTS)
        tests.update(VA3_REQUIRED_TESTS)
        tests.update(_va3_codegen_tests())
        tests.update(VA4_FIXEDRUN_TESTS)
        tests.update(_va4_codegen_tests())
        tests.update(VA5_REQUIRED_TESTS)
        # Deployment scenarios register per configured ICD; only require the
        # ones this profile can actually run.
        tests.difference_update(VA5_DEVICE_DEPLOYMENT_TESTS)
        tests.update(f"vulkan-va5-deployment-{device}" for device in devices)
    elif stage_id == "VA6":
        tests.update(VA1_REQUIRED_TESTS)
        tests.update(VA2_REQUIRED_TESTS)
        tests.update(VA3_REQUIRED_TESTS)
        tests.update(_va3_codegen_tests())
        tests.update(VA4_FIXEDRUN_TESTS)
        tests.update(_va4_codegen_tests())
        tests.update(VA5_REQUIRED_TESTS)
        tests.difference_update(VA5_DEVICE_DEPLOYMENT_TESTS)
        tests.update(f"vulkan-va5-deployment-{device}" for device in devices)
        tests.update(VA6_REQUIRED_TESTS)
        # The cross-ICD scenario needs both configured ICDs; single-device
        # profiles keep the matrix/faults/evidence coverage without it.
        tests.difference_update(VA6_DEVICE_CROSS_ICD_TESTS)
        if set(devices) == {"nvidia", "llvmpipe"}:
            tests.update(VA6_DEVICE_CROSS_ICD_TESTS)
    return tests


def _check_required_tests(gate: dict, findings: list[dict]) -> None:
    names = set(gate.get("ctest", {}).get("test_names", []))
    stage_id = gate["stage_id"] if "stage_id" in gate else "VA0"
    if not isinstance(stage_id, str) or stage_id not in {"VA0", "VA1", "VA2", "VA3", "VA4", "VA5", "VA6"}:
        _finding(findings, "gate-execution-role",
                 "required-test scope has an invalid or missing stage_id",
                 stage_id=stage_id)
        return
    missing = sorted(_required_tests(gate.get("profile"), stage_id) - names)
    if missing:
        _finding(findings, "required-ctest-missing", "profile CTest inventory is missing required gate tests",
                 profile=gate.get("profile"), tests=missing)


def _make_id_catalog() -> dict[str, dict]:
    catalog = {}
    def add(identifier: str, cpu_fixture: str, test_patterns: list[str], family: str,
            numeric: str):
        catalog[identifier] = {"cpu_fixture": cpu_fixture, "tests": test_patterns,
                               "evidence_family": family, "gpu_numeric": numeric}
    both_v0 = ["vulkan-v0-add-nvidia", "vulkan-v0-add-llvmpipe"]
    v2_element_cases = ("add", "relu", "copy", "identity", "transpose", "reshape")
    v2_conv_cases = ("conv3x3", "conv3x3-no-bias", "conv3x3-stride2", "conv1x1")
    both_v2 = [f"vulkan-v2-{case}-{device}" for device in ("nvidia", "llvmpipe")
               for case in V2_CASES]
    both_v2_element = [f"vulkan-v2-{case}-{device}" for device in ("nvidia", "llvmpipe")
                       for case in v2_element_cases]
    both_v2_conv = [f"vulkan-v2-{case}-{device}" for device in ("nvidia", "llvmpipe")
                    for case in v2_conv_cases]
    both_v5 = [f"vulkan-v5-{suite}-{device}" for device in ("nvidia", "llvmpipe")
               for suite in ("multiop", "pooling-softmax", "squeezenet")]
    both_v3 = [f"vulkan-v3-runtime-{device}" for device in ("nvidia", "llvmpipe")]
    both_v4 = ["ncnn-compile-vulkan-product", "ncnn-compile-vulkan-product-llvmpipe"]
    add("V0.1", "relu_v1_baseline", ["vulkan-v1-cpu-baseline", *both_v0], "cpu", "not-applicable")
    add("V0.2", "v0_add", both_v0, "v0", "paired-gpu-cpu-golden")
    add("V0.3", "v0_add", both_v0, "v0", "paired-gpu-cpu-golden")
    add("V0.4", "v0_add", ["vulkan-v1-cpu-baseline", *both_v0], "v0", "paired-gpu-cpu-golden")
    for i in range(1, 5):
        add(f"V1.{i}", "relu_v1_baseline",
            ["vulkan-v1-cpu-baseline", "vulkan-abi-c-contract", "ncnn-mlir-lit", *both_v0],
            "cpu", "cpu-invariance-only-no-vulkan-output")
    add("V2.1", "v2_add_relu_identity", both_v2_element, "v2-elementwise", "paired-gpu-cpu-golden")
    add("V2.2", "v2_conv2d", both_v2_conv, "v2-conv", "paired-gpu-cpu-golden")
    add("V2.3", "v2_all_positive_fixtures", [*both_v2, "ncnn-mlir-lit"], "v2", "paired-gpu-cpu-golden")
    for i in range(1, 5):
        add(f"V3.{i}", "v3_add_relu_and_conv2d", both_v3, "v3", "paired-runtime-cpu-golden")
    for i in range(1, 5):
        add(f"V4.{i}", "paired_relu", both_v4, "v4", "paired-product-cpu-golden")
    add("V5.1", "v5_multiop_fixtures", ["vulkan-v5-multiop-nvidia", "vulkan-v5-multiop-llvmpipe"], "v5.1", "per-output-cpu-vulkan-ncnn")
    add("V5.2", "v5_pooling_softmax_fixtures", ["vulkan-v5-pooling-softmax-nvidia", "vulkan-v5-pooling-softmax-llvmpipe"], "v5.2", "per-output-cpu-vulkan-ncnn-golden")
    add("V5.3", "squeezenet_v1_1", ["vulkan-v5-squeezenet-nvidia", "vulkan-v5-squeezenet-llvmpipe"], "v5.3", "per-output-cpu-vulkan-ncnn-topk")
    add("V5.4", "squeezenet_v1_1", ["ncnn-mlir-lit", "vulkan-v5-squeezenet-nvidia", "vulkan-v5-squeezenet-llvmpipe"], "v5.4", "CPU-supported-matrix-plus-positive-model")
    offline_runtime_tests = sorted(OFFLINE_RUNTIME_TESTS)
    add("V6.1", "relu_v1_baseline_and_product_fixtures", ["ncnn-compile-vulkan-offline", *offline_runtime_tests, *both_v0, *both_v2, *both_v3, *both_v4, *both_v5], "v6.1", "three-layers-no-offline-gpu-numeric")
    add("V6.2", "all-CPU-baselines-and-fixtures", ["vulkan-v1-cpu-baseline", "ncnn-compile-vulkan-offline", *offline_runtime_tests, *both_v0, *both_v2, *both_v3, *both_v4, *both_v5], "gate", "full-gate-and-numerical")
    add("V6.3", "all-fresh-stage-evidence", ["vulkan-release-evidence-validation", "vulkan-gate-validation"], "index", "hash-verified-index")
    add("V6.4", "all-release-matrix-fixtures", ["vulkan-v1-cpu-baseline", "ncnn-compile-vulkan-offline", *offline_runtime_tests, *both_v0, *both_v2, *both_v3, *both_v4, *both_v5], "decision", "release-decision-after-index")
    va0_product_tests = ["ncnn-compile-vulkan-offline", "ncnn-compile-vulkan-product",
                         "ncnn-compile-vulkan-product-llvmpipe"]
    add("VA0.1", "relu-add-relu-conv-relu-squeezenet", [
        "vulkan-v1-cpu-baseline", *va0_product_tests, *both_v5],
        "va0", "profile-scoped-cpu-current-legacy-product-old-reference-small-golden")
    add("VA0.2", "current-model-dso-dependency-closures", [
        "vulkan-model-dependency-evidence-validation", *va0_product_tests,
        *both_v5], "va0", "observed-current-legacy-dependencies-no-target-claim")
    add("VA0.3", "generic-plan-runtime-host-identity", [
        "VulkanProductTest.ManifestPinsHostExecutionContractAndIdentityDomain",
        *va0_product_tests, *both_v5], "va0",
        "generic-plan-runtime-v1-current-contract-not-host-aot")
    add("VA0.4", "legacy-tests-to-host-aot-migration-map", [
        "vulkan-va0-baseline-validation", "vulkan-va0-evidence-validation",
        "vulkan-gate-validation", "vulkan-release-evidence-validation",
        "ncnn-compile-vulkan-offline", *sorted(OFFLINE_HOST_TESTS),
        *va0_product_tests, *both_v0, *both_v2, *both_v3, *both_v4, *both_v5],
        "va0", "pending-migration-coverage-and-no-device-lifecycle")
    for identifier, tests in VA1_STEP_TESTS.items():
        add(identifier, "typed-host-program-and-static-planner",
            sorted(tests), "va1", "planner-and-legacy-symbolic-trace-only")
    for identifier, tests in VA2_STEP_TESTS.items():
        add(identifier, "vulkan-device-helper",
            sorted(tests), "va2", "plan-free-helper-native-evidence")
    for identifier, tests in VA3_STEP_TESTS.items():
        add(identifier, "lifecycle_fixture", sorted(tests), "va3",
            "create-destroy-only-no-model-run")
    for identifier, tests in VA4_STEP_TESTS.items():
        add(identifier, "fixedrun_fixture", sorted(tests), "va4",
            "model-specific-fixed-commands-and-recomputed-numerics")
    for identifier, tests in VA5_STEP_TESTS.items():
        add(identifier, "paired_relu_and_conv_relu_published",
            sorted(tests), "va5",
            "published-deployment-link-audit-and-clean-loader-execution")
    add("VA6.1", "product_support_matrix_rows",
        sorted(VA6_STEP_TESTS["VA6.1"]), "va6",
        "negative-rows-fail-closed-positive-rows-published-and-v5-numeric")
    add("VA6.2", "va6_faults_multi_output_wrapper",
        sorted(VA6_STEP_TESTS["VA6.2"]), "va6",
        "cli-generated-wrapper-fault-chain-on-fake-vulkan")
    add("VA6.3", "paired_relu_and_conv_relu_same_dso",
        sorted(VA6_STEP_TESTS["VA6.3"]), "va6",
        "one-dso-bytes-two-icds-plus-explicit-no-device-failure")
    add("VA6.4", "va6_evidence_tooling",
        sorted(VA6_STEP_TESTS["VA6.4"]), "va6",
        "registration-and-evidence-tooling-self-check")
    return catalog


ID_CATALOG = _make_id_catalog()


def _devices_for_profile(profile: str) -> set[str]:
    return {"release": {"nvidia", "llvmpipe"},
            "sanitize": {"nvidia", "llvmpipe"},
            "software": {"llvmpipe"},
            "hardware": {"nvidia"}}.get(profile, set())


VA0_LEGACY_REFERENCE_IDENTITY = {
    "profile": "release",
    "source_fingerprint": "6d2f2b2ecacab5a17385ea9acdf67482f321bb85772d513739cd2186edab20d3",
    "git_revision": "310276ec49cccbcb309ddb0f002d98a5dab7fb9d",
    "host_aot_candidate": False,
}
VA0_LEGACY_FIXTURE_KEYS = frozenset((
    "relu-product-nvidia", "relu-product-llvmpipe",
    "add-relu-nvidia", "add-relu-llvmpipe",
    "conv-relu-nvidia", "conv-relu-llvmpipe",
    "squeezenet-nvidia", "squeezenet-llvmpipe"))
VA0_MODEL_NAMES = {"add_relu": "add-relu", "conv_relu": "conv-relu",
                   "squeezenet_v1_1": "squeezenet"}
VA0_DEVICE_NAMES = {"nvidia": "nvidia", "llvmpipe": "llvmpipe"}
VA0_MIGRATION_RESPONSIBILITY = {
    "VA1": "interface/planner",
    "VA2/3": "helper/budget/lifecycle",
    "VA4": "fixedrun/sync",
    "VA5": "link/cleanpublish",
    "VA6": "fullmatrix/faults",
    "VA7": "profiles/inventory",
}


def _va0_migration_owners(test_name: str, identifiers: list[str]) -> list[str]:
    lowered = test_name.lower()
    owners = set()
    if any(token in lowered for token in ("sync", "thread", "output", "barrier", "fixedrun")):
        owners.add("VA4")
    if any(token in lowered for token in ("plan", "schema", "abi", "manifest", "identity", "interface")):
        owners.add("VA1")
    if any(token in lowered for token in ("lifecycle", "budget", "memory", "resource", "weight", "context", "destroy", "create", "no-device")):
        owners.add("VA2/3")
    if any(token in lowered for token in ("link", "publish", "transaction", "product-cli", "artifact")):
        owners.add("VA5")
    if any(token in lowered for token in ("negative", "failure", "fault", "recovery")):
        owners.add("VA6")
    if lowered.startswith(("vulkan-v0-", "vulkan-v2-", "vulkan-v3-", "vulkan-v5-")):
        owners.add("VA6")
    if lowered.startswith("vulkan-v3-runtime"):
        owners.update(("VA2/3", "VA4"))
    if any(token in lowered for token in ("gate", "inventory", "profile", "release-evidence", "ctest", "sanitizer")):
        owners.add("VA7")
    if not owners:
        for identifier in identifiers:
            if identifier.startswith(("V0.", "V5.")):
                owners.add("VA6")
            elif identifier.startswith("V1.") or identifier.startswith("V2."):
                owners.add("VA1")
            elif identifier.startswith("V3."):
                owners.update(("VA2/3", "VA4"))
            elif identifier.startswith("V4."):
                owners.add("VA5")
            elif identifier.startswith("V6."):
                owners.add("VA7")
            elif identifier == "VA0.1":
                owners.add("VA6")
            elif identifier == "VA0.2":
                owners.add("VA5")
            elif identifier == "VA0.3":
                owners.add("VA1")
            elif identifier == "VA0.4":
                owners.add("VA7")
    if not owners:
        owners.add("VA7")
    return sorted(owners, key=lambda item: (int(item.split("/")[0][2:]), item))


def _va0_migration_coverage(gate: dict) -> dict:
    va4_tests = VA4_FIXEDRUN_TESTS | _va4_codegen_tests()
    actual_names = sorted(set(gate.get("ctest", {}).get("test_names", [])) -
                          set(VA2_REQUIRED_TESTS) - set(VA3_REQUIRED_TESTS) -
                          va4_tests - set(VA5_REQUIRED_TESTS) -
                          {name for name in gate.get("ctest", {}).get("test_names", [])
                           if name.startswith(VA3_CODEGEN_TEST_PREFIXES)})
    id_rows = {}
    related_by_test: dict[str, list[str]] = {}
    for identifier, spec in ID_CATALOG.items():
        if spec["evidence_family"] in {"va2", "va3", "va4", "va5", "va6"}:
            continue
        id_rows[identifier] = {
            "legacy_tests": sorted(set(spec["tests"])),
            "future_migration": _va0_migration_owners(
                identifier, [identifier]),
            "migration_status": "pending",
            "legacy_pass_is_migration_pass": False,
        }
        for test_name in spec["tests"]:
            related_by_test.setdefault(test_name, []).append(identifier)
    test_rows = {}
    for test_name in actual_names:
        identifiers = sorted(related_by_test.get(test_name, []))
        test_rows[test_name] = {
            "related_ids": identifiers,
            "future_migration": _va0_migration_owners(test_name, identifiers),
            "migration_status": "pending",
            "legacy_pass_is_migration_pass": False,
            "legacy_test_status": (
                "failed" if test_name in set(gate.get("ctest", {}).get("failed", [])) else
                "skipped" if test_name in set(gate.get("ctest", {}).get("skipped", [])) else
                "passed"),
        }
    profile = gate.get("profile", "")
    legacy_test_rows = [row for identifier, row in id_rows.items()
                        if ID_CATALOG[identifier]["evidence_family"] not in
                        {"va1", "va2", "va3", "va4", "va5", "va6"}]
    applicable = set().union(*(_applicable_tests(row["legacy_tests"], profile)
                               for row in legacy_test_rows)) if legacy_test_rows else set()
    missing = sorted(applicable - set(actual_names))
    return {
        "status": "pending",
        "migration_status": "pending",
        "legacy_pass_is_migration_pass": False,
        "responsibilities": VA0_MIGRATION_RESPONSIBILITY,
        "current_ids": id_rows,
        "current_test_cases": test_rows,
        "unexecuted_related_tests": missing,
        "product_only_required_tests": sorted(OFFLINE_HOST_TESTS),
    }


def _va0_file_record(stage: Path, snapshot: dict, relative: str,
                     label: str, refs: list[dict], findings: list[dict]) -> Path | None:
    path = _evidence_file(stage, stage, relative, label, refs, findings)
    if path is None:
        return None
    record = next((item for item in snapshot.get("files", [])
                   if isinstance(item, dict) and item.get("path") == relative), None)
    if record is None:
        _finding(findings, "legacy-reference-index-missing",
                 f"legacy-reference snapshot file is absent from its content index: {relative}")
        return path
    _check_digest_claim(record.get("sha256"), path, label, findings)
    return path


def _audit_legacy_reference(stage: Path, gate_state: dict | None,
                            refs: list[dict], findings: list[dict]) -> dict | None:
    record = gate_state.get("legacy_reference") if isinstance(gate_state, dict) else None
    if (not isinstance(record, dict) or record.get("status") != "captured"):
        _finding(findings, "legacy-reference-missing",
                 "VA0 requires an explicitly supplied frozen legacy Release reference stage")
        return None
    snapshot_path = _verify_hash_record(stage, record, "legacy-reference snapshot", findings)
    if snapshot_path is None:
        return None
    try:
        snapshot = load_json(snapshot_path)
        if (not isinstance(snapshot, dict) or snapshot.get("kind") !=
                "ncnn.vulkan_legacy_reference_snapshot" or snapshot.get("schema_version") != 1):
            raise ValueError("unsupported legacy-reference snapshot schema")
        identity = snapshot.get("identity")
        if identity != VA0_LEGACY_REFERENCE_IDENTITY or snapshot.get("role") != "legacy-reference" or snapshot.get("status") != "captured":
            raise ValueError("legacy-reference identity does not match the approved frozen baseline")
        files = snapshot.get("files")
        fixture_rows = snapshot.get("fixtures")
        if not isinstance(files, list) or not files or not isinstance(fixture_rows, dict):
            raise ValueError("legacy-reference file/fixture index is empty or malformed")
        source_manifest_rel = "legacy-reference/source/gate-archive-manifest.json"
        source_manifest_path = _va0_file_record(
            stage, snapshot, source_manifest_rel,
            "legacy-reference original archive hash manifest", refs, findings)
        source_manifest = load_json(source_manifest_path) if source_manifest_path else None
        if (not isinstance(source_manifest, dict) or
                source_manifest.get("kind") != "ncnn.vulkan_evidence_archive" or
                not isinstance(source_manifest.get("files"), dict) or
                snapshot.get("source_archive_manifest_sha256") != sha256_file(source_manifest_path)):
            raise ValueError("legacy-reference original archive manifest is invalid/tampered")
        if set(fixture_rows) != VA0_LEGACY_FIXTURE_KEYS:
            raise ValueError("legacy-reference fixture set is incomplete or contains extras")
        file_paths = [item.get("path") for item in files if isinstance(item, dict)]
        if len(file_paths) != len(files) or len(set(file_paths)) != len(file_paths):
            raise ValueError("legacy-reference file index is malformed or duplicated")
        file_by_path = {item["path"]: item for item in files}
        for item in files:
            if (not isinstance(item.get("source"), str) or
                    not isinstance(item.get("role"), str) or
                    not item["role"].startswith("legacy-reference ")):
                _finding(findings, "legacy-reference-file-role",
                         "legacy-reference file record lacks its historical role/source",
                         path=item.get("path"))
                continue
            path = _va0_file_record(stage, snapshot, item["path"],
                                    f"{item['role']} {item['source']}", refs, findings)
            if item["source"] != "gate-archive-manifest.json":
                archived_hash = source_manifest["files"].get(item["source"])
                if archived_hash != item.get("sha256"):
                    _finding(findings, "legacy-reference-source-manifest",
                             "legacy-reference copied bytes are not bound by the original archive manifest",
                             source=item["source"])
            if path is None:
                continue
        required_provenance = {
            "legacy-reference/source/gate-state.json",
            "legacy-reference/source/release-evidence.json",
            "legacy-reference/source/gate-va0-legacy-source.json",
            "legacy-reference/source/gate-ctest.xml",
            "legacy-reference/source/gate-ctest-lasttest.log",
            "legacy-reference/source/gate-ctest.log",
            "legacy-reference/source/test/gate-tests.json",
            source_manifest_rel,
        }
        provenance = [item for item in files if item.get("path") in required_provenance]
        if {item.get("path") for item in provenance} != required_provenance:
            _finding(findings, "legacy-reference-provenance-missing",
                     "legacy-reference omitted original gate state/report/source/JUnit/LastTest/hash manifest")
        for fixture, row in fixture_rows.items():
            if (not isinstance(row, dict) or not isinstance(row.get("path"), str) or
                    not isinstance(row.get("files"), list) or not row["files"]):
                _finding(findings, "legacy-reference-fixture-index",
                         "legacy-reference fixture row is empty or malformed", fixture=fixture)
                continue
            expected_prefix = row["path"].rstrip("/") + "/"
            listed = set(row["files"])
            indexed = {path for path in file_paths if path.startswith(expected_prefix)}
            if listed != indexed or not indexed:
                _finding(findings, "legacy-reference-fixture-index",
                         "fixture file inventory differs from the copied-file index",
                         fixture=fixture, missing=sorted(indexed - listed),
                         unindexed=sorted(listed - indexed))
            if not (stage / expected_prefix / "evidence.json").is_file():
                _finding(findings, "legacy-reference-fixture-evidence",
                         "legacy-reference fixture evidence JSON is missing", fixture=fixture)
        original_state_path = stage / "legacy-reference/source/gate-state.json"
        original_state = load_json(original_state_path) if original_state_path.is_file() else {}
        original_source_path = stage / "legacy-reference/source/gate-va0-legacy-source.json"
        original_source = load_json(original_source_path) if original_source_path.is_file() else {}
        original_report_path = stage / "legacy-reference/source/release-evidence.json"
        original_report = load_json(original_report_path) if original_report_path.is_file() else {}
        if (original_state.get("profile") != "release" or original_state.get("completed") is not True or
                original_state.get("source_fingerprint") != VA0_LEGACY_REFERENCE_IDENTITY["source_fingerprint"] or
                original_state.get("git_revision") != VA0_LEGACY_REFERENCE_IDENTITY["git_revision"]):
            _finding(findings, "legacy-reference-original-gate",
                     "copied original gate state is not the frozen completed Release source")
        original_steps = original_state.get("steps", [])
        if ([item.get("name") for item in original_steps if isinstance(item, dict)] != list(STEP_NAMES) or
                any(type(item.get("exit_code")) is not int or item["exit_code"] != 0
                    for item in original_steps if isinstance(item, dict))):
            _finding(findings, "legacy-reference-original-gate",
                     "copied original gate state has incomplete/failed steps")
        if (original_source.get("role") != "legacy-reference" or
                original_source.get("stage_id") != "VA0.1" or
                original_source.get("worktree_porcelain") != "" or
                original_source.get("git_revision") != VA0_LEGACY_REFERENCE_IDENTITY["git_revision"] or
                original_source.get("host_aot_candidate") is not False or
                original_source.get("source_fingerprint") != VA0_LEGACY_REFERENCE_IDENTITY["source_fingerprint"] or
                original_report.get("source_fingerprint") != VA0_LEGACY_REFERENCE_IDENTITY["source_fingerprint"] or
                original_report.get("blocking_findings")):
            _finding(findings, "legacy-reference-source-identity",
                     "copied original source record/report does not bind the frozen clean source")
        original_release_rows = [item for item in original_report.get("stages", [])
                                 if item.get("profile") == "release"]
        if (len(original_release_rows) != 1 or
                original_release_rows[0].get("scope_passed") is not True or
                original_release_rows[0].get("findings")):
            _finding(findings, "legacy-reference-original-report",
                     "copied old Release report does not show the original profile scope passed")
        original_inventory_path = stage / "legacy-reference/source/test/gate-tests.json"
        original_junit_path = stage / "legacy-reference/source/gate-ctest.xml"
        original_lasttest_path = stage / "legacy-reference/source/gate-ctest-lasttest.log"
        try:
            original_inventory = load_json(original_inventory_path)
            original_junit = ET.parse(original_junit_path).getroot()
            original_lasttest = original_lasttest_path.read_text(encoding="utf-8")
            inventory_names = {item.get("name") for item in original_inventory.get("tests", [])
                               if isinstance(item, dict)}
            test_cases = list(original_junit.iter("testcase"))
            junit_names = {item.get("name") for item in test_cases}
            skipped_names = {item.get("name") for item in test_cases
                             if item.find("skipped") is not None}
            lasttest_names = re.findall(r"^\d+/\d+ Testing: (.+)$", original_lasttest,
                                        re.MULTILINE)
            if (len(inventory_names) != 525 or len(test_cases) != 525 or
                    junit_names != inventory_names or len(lasttest_names) != 525 or
                    set(lasttest_names) != inventory_names or
                    skipped_names != {"PerformanceModel.PPLCNetDocOriInt8",
                                      "PerformanceModel.PPLCNetTextlineOriInt8"} or
                    any(item.find("failure") is not None or item.find("error") is not None
                        for item in test_cases) or
                    "Test Failed." in original_lasttest or "Test Not Run." in original_lasttest):
                raise ValueError("old CTest inventory/JUnit/LastTest counts or statuses differ")
            for key, relative in (("inventory", "test/gate-tests.json"),
                                  ("junit", "gate-ctest.xml"),
                                  ("lasttest", "gate-ctest-lasttest.log")):
                record = original_state.get(key)
                old_file = stage / "legacy-reference/source" / relative
                if (not isinstance(record, dict) or record.get("path") != relative or
                        sha256_file(old_file) != record.get("sha256")):
                    raise ValueError(f"old gate-state {key} hash does not bind its copied raw file")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ET.ParseError) as error:
            _finding(findings, "legacy-reference-original-ctest",
                     f"copied old CTest result evidence is invalid: {error}")
        return {"status": "verified", "identity": identity,
                "fixtures": fixture_rows,
                "raw_artifacts": [item for item in files if item.get("role", "").startswith("legacy-reference ")],
                "snapshot_path": str(snapshot_path.relative_to(stage))}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        _finding(findings, "legacy-reference-invalid", f"legacy-reference snapshot is invalid: {error}")
        return None


def _va0_normalize_host_triple(value: object) -> object:
    if not isinstance(value, str):
        return value
    fields = value.split("-")
    if (len(fields) == 4 and fields[0] == "x86_64" and
            fields[1] in {"pc", "unknown"} and fields[2:] == ["linux", "gnu"]):
        return "x86_64-linux-gnu"
    return value


def _va0_normalize_semantic_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: (_va0_normalize_host_triple(item) if key == "host_triple" else
                      _va0_normalize_semantic_json(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_va0_normalize_semantic_json(item) for item in value]
    return value


def _va0_manifest_semantics(path: Path) -> object:
    value = load_json(path)
    ignored = {"build_identity", "host_implementation", "host_codegen_version",
               "execution_plan_usage", "host_program", "host_program_identity",
               "host_program_usage", "dependencies", "plan_hash", "plan_sha256",
               "shader_sha256", "shader_sha256s"}
    if isinstance(value, dict):
        normalized = {key: item for key, item in value.items() if key not in ignored}
        shaders = normalized.get("shaders")
        if isinstance(shaders, list):
            normalized["shaders"] = [
                {key: item for key, item in shader.items() if key != "sha256"}
                if isinstance(shader, dict) else shader for shader in shaders]
        value = normalized
    return _va0_normalize_semantic_json(value)


def _va0_plan_semantics(path: Path) -> object:
    value = load_json(path)
    if isinstance(value, dict):
        value = {key: item for key, item in value.items()
                 if key not in {"build_identity", "plan_hash", "plan_sha256"}}
    return _va0_normalize_semantic_json(value)


def _va0_spirv_semantic_form(disassembler: str, path: Path) -> str:
    result = subprocess.run([disassembler, "--raw-id", "--no-header", str(path)],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"SPIR-V disassembly failed for {path}: {result.stderr}")
    return _va0_normalize_spirv_disassembly(result.stdout)


def _va0_normalize_spirv_disassembly(disassembly: str) -> str:
    pure_ops = {"OpLogicalAnd", "OpULessThan", "OpUGreaterThanEqual"}
    definitions = {}
    blocks = []
    preamble = []
    current = None
    definition_pattern = re.compile(r"^(%[\w.]+)\s*=\s*(Op\w+)\s+(.*)$")
    label_pattern = re.compile(r"^%[\w.]+\s*=\s*OpLabel\b")
    for raw_line in disassembly.splitlines():
        line = raw_line.strip()
        match = definition_pattern.match(line)
        if match and match.group(2) in pure_ops:
            operands = match.group(3).split()
            definitions[match.group(1)] = (match.group(2), operands)
        if label_pattern.match(line):
            current = {"header": line, "body": []}
            blocks.append(current)
            continue
        if current is None:
            preamble.append(line)
        else:
            current["body"].append(line)

    def expression(identifier: str, stack: tuple[str, ...] = ()) -> object:
        definition = definitions.get(identifier)
        if definition is None:
            return identifier
        if identifier in stack:
            raise ValueError("cyclic pure SPIR-V expression")
        opcode, operands = definition
        if not operands:
            raise ValueError(f"malformed SPIR-V pure operation {opcode}")
        arguments = tuple(expression(item, stack + (identifier,))
                          for item in operands[1:])
        if opcode == "OpLogicalAnd":
            arguments = tuple(sorted(arguments, key=repr))
        return opcode, operands[0], arguments

    def substitute(line: str) -> str:
        return re.sub(r"%[\w.]+", lambda match: repr(expression(match.group(0)))
                      if match.group(0) in definitions else match.group(0), line)

    normalized = list(preamble)
    for block in blocks:
        normalized.append(block["header"])
        body = block["body"]
        index = 0
        while index < len(body):
            match = definition_pattern.match(body[index])
            if match and match.group(2) in pure_ops:
                identifiers = []
                while index < len(body):
                    match = definition_pattern.match(body[index])
                    if not match or match.group(2) not in pure_ops:
                        break
                    identifiers.append(match.group(1))
                    index += 1
                normalized.extend("PURE " + repr(expression(identifier))
                                  for identifier in sorted(identifiers,
                                                           key=lambda item: repr(expression(item))))
                continue
            normalized.append(substitute(body[index]))
            index += 1
    return "\n".join(normalized) + "\n"


def _va0_compare_product_artifacts(stage: Path, old_root: Path, current_root: Path,
                                   label: str, findings: list[dict],
                                   shader_comparisons: list[dict] | None = None) -> bool:
    passed = True
    for category in ("cpu", "vulkan"):
        old_dir, current_dir = old_root / category, current_root / category
        if not old_dir.is_dir() or not current_dir.is_dir():
            _finding(findings, "va0-product-artifact-missing",
                     f"{label} {category} artifact directory is missing")
            passed = False
            continue
        old_headers = {item.name: item for item in old_dir.glob("*.h")}
        current_headers = {item.name: item for item in current_dir.glob("*.h")}
        if set(old_headers) != set(current_headers) or any(
                old_headers[name].read_bytes() != current_headers[name].read_bytes()
                for name in old_headers.keys() & current_headers.keys()):
            _finding(findings, "va0-header-semantics",
                     f"{label} {category} generated public header semantics changed")
            passed = False
        old_manifests = {item.name: item for item in old_dir.glob("*.json")
                         if not item.name.endswith(".plan.json")}
        current_manifests = {item.name: item for item in current_dir.glob("*.json")
                             if not item.name.endswith(".plan.json")}
        additive_current_cpu_manifest = (
            category == "cpu" and set(old_manifests).issubset(current_manifests))
        if set(old_manifests) != set(current_manifests) and not additive_current_cpu_manifest:
            _finding(findings, "va0-manifest-semantics",
                     f"{label} {category} manifest artifact set changed")
            passed = False
        for name in old_manifests.keys() & current_manifests.keys():
            try:
                if _va0_manifest_semantics(old_manifests[name]) != _va0_manifest_semantics(current_manifests[name]):
                    _finding(findings, "va0-manifest-semantics",
                             f"{label} {category}/{name} changed beyond the declared build/host metadata")
                    passed = False
            except (OSError, ValueError, json.JSONDecodeError) as error:
                _finding(findings, "va0-manifest-invalid", f"{label} {category}/{name}: {error}")
                passed = False
        if category == "vulkan":
            old_plans = {item.name: item for item in old_dir.glob("*.plan.json")}
            current_plans = {item.name: item for item in current_dir.glob("*.plan.json")}
            if set(old_plans) != set(current_plans) or any(
                    _va0_plan_semantics(old_plans[name]) != _va0_plan_semantics(current_plans[name])
                    for name in old_plans.keys() & current_plans.keys()):
                _finding(findings, "va0-plan-semantics",
                         f"{label} embedded execution plan changed")
                passed = False
    old_shader_dir = old_root / "spirv-validation"
    current_shader_dir = current_root / "spirv-validation"
    if not old_shader_dir.is_dir():
        old_shader_dir = old_root / "vulkan/shader-reconstruction/spirv-validation"
    if not current_shader_dir.is_dir():
        current_shader_dir = current_root / "vulkan/shader-reconstruction/spirv-validation"
    old_shaders = {item.name: item for item in old_shader_dir.glob("shader-*.spv")}
    current_shaders = {item.name: item for item in current_shader_dir.glob("shader-*.spv")}
    if not old_shaders or set(old_shaders) != set(current_shaders):
        _finding(findings, "va0-shader-semantics",
                 f"{label} validated SPIR-V kernel inventory changed or is incomplete")
        passed = False
    else:
        gate_state = load_json(stage / "gate-state.json")
        disassemblers = [item.get("path") for item in gate_state.get("toolchain_identities", [])
                         if isinstance(item, dict) and item.get("role") == "spirv-dis"]
        disassembler = disassemblers[0] if len(disassemblers) == 1 else None
        for name in sorted(old_shaders):
            old_path, current_path = old_shaders[name], current_shaders[name]
            old_sha, current_sha = sha256_file(old_path), sha256_file(current_path)
            if old_sha == current_sha:
                continue
            if not disassembler or not Path(disassembler).is_file():
                _finding(findings, "va0-shader-semantics",
                         f"{label}/{name} SPIR-V bytes changed and no audited disassembler is available")
                passed = False
                continue
            try:
                old_form = _va0_spirv_semantic_form(disassembler, old_path)
                current_form = _va0_spirv_semantic_form(disassembler, current_path)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                _finding(findings, "va0-shader-disassembly-invalid",
                         f"{label}/{name} SPIR-V semantic comparison failed: {error}")
                passed = False
                continue
            if old_form != current_form:
                _finding(findings, "va0-shader-semantics",
                         f"{label}/{name} SPIR-V changed beyond pure comparison/logical-and reordering")
                passed = False
                continue
            if shader_comparisons is not None:
                shader_comparisons.append({
                    "label": label, "shader": name,
                    "status": "semantic-equivalent-pure-expression-reordering",
                    "old_sha256": old_sha, "current_sha256": current_sha,
                    "semantic_form_sha256": sha256_bytes(old_form.encode("utf-8")),
                })
    return passed


def _va0_read_raw_roles(model_root: Path, trial: dict, role: str) -> list[list[float]]:
    raw_files = trial.get("raw_files", {})
    names = raw_files.get(role) if isinstance(raw_files, dict) else None
    if not isinstance(names, list) or not names:
        raise ValueError(f"{role} raw tensor list is missing")
    return [read_f32(model_root / name) for name in names]


def _va0_compare_small_model(stage: Path, old_root: Path, current_root: Path,
                             model_name: str, label: str,
                             refs: list[dict], findings: list[dict],
                             shader_comparisons: list[dict] | None = None) -> bool:
    try:
        old = load_json(old_root / "evidence.json")
        current = load_json(current_root / "evidence.json")
        old_ids, current_ids = old.get("identities", {}), current.get("identities", {})
        for field in ("param_sha256", "model_bin_sha256"):
            if old_ids.get(field) != current_ids.get(field):
                raise ValueError(f"{field} differs from the frozen reference")
        if model_name == "add_relu":
            source_names = ("add_relu.param", "empty.bin")
            old_source_root = old_root / "inputs"
            current_source_root = current_root.parent.parent / "fixtures"
        elif model_name == "conv_relu":
            source_names = ("conv_relu.param", "conv_weights.bin")
            old_source_root = old_root / "inputs"
            current_source_root = current_root.parent.parent / "fixtures"
        else:
            source_names = ("squeezenet_v1_1.param", "squeezenet_v1_1.bin")
            old_source_root = old_root
            current_source_root = current_root
        for name, identity_field in zip(source_names, ("param_sha256", "model_bin_sha256")):
            old_source = old_source_root / name
            current_source = current_source_root / name
            if (not old_source.is_file() or not current_source.is_file() or
                    sha256_file(old_source) != old_ids.get(identity_field) or
                    sha256_file(current_source) != current_ids.get(identity_field) or
                    old_source.read_bytes() != current_source.read_bytes()):
                raise ValueError(f"{identity_field} source bytes differ from the paired models")
        old_trials = {item.get("name"): item for item in old.get("trials", [])
                      if isinstance(item, dict)}
        current_trials = {item.get("name"): item for item in current.get("trials", [])
                          if isinstance(item, dict)}
        if not old_trials or set(old_trials) != set(current_trials):
            raise ValueError("old/current raw trial inventories differ")
        atol, rtol = ((1.0e-6, 1.0e-4) if model_name == "squeezenet_v1_1"
                      else (1.0e-6, 1.0e-6))
        expected_roles = ("inputs", "cpu_outputs", "ncnn_outputs", "vulkan_outputs")
        oracle = current.get("independent_golden")
        if model_name in {"add_relu", "conv_relu"} and not isinstance(oracle, dict):
            raise ValueError("current Add/Conv fixture lacks its independent scalar golden oracle")
        if model_name == "conv_relu":
            source_inputs = old_root / "inputs"
            old_weights = source_inputs / "conv_weights.bin"
            if not old_weights.is_file():
                raise ValueError("old Conv reference weights are missing")
            payload = old_weights.read_bytes()
            oracle_values = oracle.get("weights", []) + oracle.get("biases", [])
            expected_payload = struct.pack("<I", 0) + struct.pack(
                f"<{len(oracle_values)}f", *oracle_values)
            if payload != expected_payload:
                raise ValueError("independent Conv oracle weights differ from frozen model bin bytes")
        golden_fn = _v5_golden_function() if model_name in {"add_relu", "conv_relu"} else None
        input_specs = current.get("inputs", [])
        output_specs = current.get("outputs", [])
        golden_fixture = {
            "inputs": [(item["blob"], tuple(item.get("shape", item.get("shape_chw"))))
                       for item in input_specs],
            "outputs": [(item["blob"], tuple(item.get("shape", item.get("shape_chw"))))
                        for item in output_specs],
            "oracle": oracle,
        }
        for trial_name in sorted(old_trials):
            old_trial, current_trial = old_trials[trial_name], current_trials[trial_name]
            old_arrays = {role: _va0_read_raw_roles(old_root, old_trial, role)
                          for role in expected_roles}
            current_arrays = {role: _va0_read_raw_roles(current_root, current_trial, role)
                              for role in expected_roles}
            for index, (old_values, current_values) in enumerate(zip(
                    old_arrays["inputs"], current_arrays["inputs"])):
                old_path = old_root / old_trial["raw_files"]["inputs"][index]
                current_path = current_root / current_trial["raw_files"]["inputs"][index]
                if old_path.read_bytes() != current_path.read_bytes():
                    raise ValueError(f"{trial_name} input tensor {index} changed")
            for role in expected_roles[1:]:
                if len(old_arrays[role]) != len(current_arrays[role]):
                    raise ValueError(f"{trial_name} {role} output count changed")
                for old_values, current_values in zip(old_arrays[role], current_arrays[role]):
                    if not compare_f32(current_values, old_values, atol, rtol)["passed"]:
                        raise ValueError(f"{trial_name} {role} values changed beyond the frozen budget")
            if golden_fn is not None:
                golden = golden_fn(golden_fixture, old_arrays["inputs"])
                if len(golden) != len(output_specs):
                    raise ValueError(f"{trial_name} scalar golden output count differs from model signature")
                for role in ("cpu_outputs", "ncnn_outputs", "vulkan_outputs"):
                    if len(old_arrays[role]) != len(golden):
                        raise ValueError(f"{trial_name} {role} count differs from scalar golden")
                    for values, expected in zip(old_arrays[role], golden):
                        if not compare_f32(values, expected, atol, rtol)["passed"]:
                            raise ValueError(f"{trial_name} old {role} differs from recomputed scalar golden")
                current_golden = current_trial.get("raw_files", {}).get("golden_outputs")
                if not isinstance(current_golden, list) or len(current_golden) != len(golden):
                    raise ValueError(f"{trial_name} current raw scalar golden outputs are missing")
                for path, expected in zip(current_golden, golden):
                    if not compare_f32(read_f32(current_root / path), expected,
                                       atol, rtol)["passed"]:
                        raise ValueError(f"{trial_name} current raw scalar golden bytes differ from recomputation")
            if model_name == "squeezenet_v1_1":
                for role in ("cpu_outputs", "ncnn_outputs", "vulkan_outputs"):
                    for values in old_arrays[role] + current_arrays[role]:
                        if not values or abs(sum(values) - 1.0) > 1.0e-4:
                            raise ValueError(f"{trial_name} {role} probability sum is outside 1e-4")
                old_reference = old_arrays["ncnn_outputs"][0]
                current_reference = current_arrays["ncnn_outputs"][0]
                for role in ("cpu_outputs", "vulkan_outputs"):
                    if _top_indices(old_reference, 5) != _top_indices(old_arrays[role][0], 5):
                        raise ValueError(f"{trial_name} frozen {role} top-5 differs from ncnn")
                    if _top_indices(current_reference, 5) != _top_indices(current_arrays[role][0], 5):
                        raise ValueError(f"{trial_name} current {role} top-5 differs from ncnn")
        artifact_match = _va0_compare_product_artifacts(
            stage, old_root, current_root, label, findings, shader_comparisons)
        return artifact_match
    except (OSError, ValueError, TypeError, KeyError, struct.error, json.JSONDecodeError) as error:
        _finding(findings, "va0-legacy-numeric-baseline",
                 f"{label} old/current numeric comparison could not be verified: {error}")
        return False


def _top_indices(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def _va0_relu_manifest_io(root: Path, backend: str, *, required: bool = True) -> tuple[dict, dict] | None:
    directory = root / backend
    suffix = "_vulkan.json" if backend == "vulkan" else ".json"
    manifests = [path for path in directory.glob("*.json")
                 if path.name.endswith(suffix) and not path.name.endswith(".plan.json")]
    if not manifests and not required:
        return None
    if len(manifests) != 1:
        raise ValueError(f"{backend} must have exactly one ABI manifest")
    manifest = load_json(manifests[0])
    inputs, outputs = manifest.get("inputs"), manifest.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{backend} ABI manifest must describe one input and output")
    expected_layout = "CHW"
    for role, ports in (("input", inputs), ("output", outputs)):
        tensor = ports[0]
        if tensor.get("element_type") != "f32" or tensor.get("shape") != [1, 1, 257]:
            raise ValueError(f"{backend} {role} manifest tensor type/shape changed")
        if backend == "vulkan" and any(
                tensor.get(key) != expected_layout
                for key in ("host_layout", "logical_layout", "physical_layout")):
            raise ValueError(f"Vulkan {role} manifest layout changed")
    return inputs[0], outputs[0]


def _va0_relu_manifest_contract(root: Path, *, require_cpu_manifest: bool) -> dict:
    cpu_ports = _va0_relu_manifest_io(root, "cpu", required=require_cpu_manifest)
    vk_input, vk_output = _va0_relu_manifest_io(root, "vulkan")
    if cpu_ports is not None:
        cpu_input, cpu_output = cpu_ports
        if (cpu_input.get("shape") != vk_input.get("shape") or
                cpu_output.get("shape") != vk_output.get("shape") or
                cpu_input.get("element_type") != vk_input.get("element_type") or
                cpu_output.get("element_type") != vk_output.get("element_type")):
            raise ValueError("CPU and Vulkan manifest public tensor contracts differ")
    return {"element_type": "f32", "shape": [257],
            "input_shape": vk_input["shape"], "output_shape": vk_output["shape"],
            "shape_chw": [1, 1, 257], "host_layout": "CHW"}


def _va0_compare_relu_product(stage: Path, old_root: Path, current_root: Path,
                              label: str, findings: list[dict],
                              shader_comparisons: list[dict] | None = None) -> bool:
    try:
        old = load_json(old_root / "evidence.json")
        current = load_json(current_root / "evidence.json")
        old_fixture, current_fixture = old.get("fixture", {}), current.get("fixture", {})
        for field in ("param_sha256", "bin_sha256"):
            if old_fixture.get(field) != current_fixture.get(field):
                raise ValueError(f"ReLU model {field} differs from frozen reference")
        for name, field in (("model.param", "param_sha256"), ("model.bin", "bin_sha256")):
            old_source, current_source = old_root / name, current_root / name
            if (not old_source.is_file() or not current_source.is_file() or
                    sha256_file(old_source) != old_fixture.get(field) or
                    sha256_file(current_source) != current_fixture.get(field) or
                    old_source.read_bytes() != current_source.read_bytes()):
                raise ValueError(f"ReLU {field} source model bytes changed")
        old_io, current_io = old.get("public_io", {}), current.get("public_io", {})
        old_manifest_contract = _va0_relu_manifest_contract(old_root, require_cpu_manifest=False)
        current_manifest_contract = _va0_relu_manifest_contract(current_root, require_cpu_manifest=True)
        if old_manifest_contract != current_manifest_contract:
            raise ValueError("old/current ReLU manifest tensor layout/shape contract changed")
        for label, io in (("old", old_io), ("current", current_io)):
            if io.get("element_type") != "f32" or io.get("shape") != [257]:
                raise ValueError(f"{label} ReLU public tensor type/shape contract changed")
            for key in ("shape_chw", "host_layout"):
                if key in io and io.get(key) != old_manifest_contract[key]:
                    raise ValueError(f"{label} ReLU {key} conflicts with manifest layout/shape")
            tolerance = io.get("tolerance", {})
            if (tolerance.get("absolute"), tolerance.get("relative")) != (1.0e-6, 1.0e-6):
                raise ValueError(f"{label} ReLU fixed numeric budget changed")
        for role in ("input", "scalar_golden", "cpu_output"):
            old_record, current_record = old_io.get(role), current_io.get(role)
            if not isinstance(old_record, dict) or not isinstance(current_record, dict):
                raise ValueError(f"{role} raw tensor record is missing")
            old_path, current_path = old_root / old_record["file"], current_root / current_record["file"]
            old_values, current_values = read_f32(old_path), read_f32(current_path)
            if old_values != current_values:
                raise ValueError(f"ReLU {role} FP32 values changed")
        if current_io.get("vulkan_output"):
            old_values = read_f32(old_root / old_io["vulkan_output"]["file"])
            current_values = read_f32(current_root / current_io["vulkan_output"]["file"])
            if not compare_f32(current_values, old_values, 1.0e-6, 1.0e-6)["passed"]:
                raise ValueError("ReLU Vulkan output changed beyond fixed 1e-6 budget")
        artifact_match = _va0_compare_product_artifacts(
            stage, old_root, current_root, label, findings, shader_comparisons)
        return artifact_match
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        _finding(findings, "va0-legacy-product-baseline",
                 f"{label} old/current ReLU product comparison failed: {error}")
        return False


def _dependency_raw_artifact_refs(stage: Path, root: Path, label: str,
                                  refs: list[dict], findings: list[dict]) -> list[dict]:
    output = []
    if not root.is_dir():
        _finding(findings, "dependency-evidence-directory-missing",
                 f"{label} evidence directory is missing", path=str(root))
        return output
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        _finding(findings, "dependency-evidence-empty",
                 f"{label} evidence directory contains no raw files", path=str(root))
        return output
    for path in files:
        if path.stat().st_size == 0:
            _finding(findings, "dependency-evidence-empty-file",
                     f"{label} contains an empty evidence file",
                     path=str(path.relative_to(stage)))
            continue
        record = _evidence_file(stage, stage, str(path.relative_to(stage)),
                                f"{label} raw evidence", refs, findings)
        if record is not None:
            output.append({"path": str(path.relative_to(stage)),
                           "sha256": sha256_file(path), "role": f"{label} raw evidence"})
    return output


def _check_dependency_elf_claims(summary: object, root: Path,
                                 label: str, findings: list[dict]) -> None:
    if not isinstance(summary, dict):
        _finding(findings, "dependency-elf-summary", f"{label} ELF summary is missing")
        return
    raw_records = {}
    for key in ("readelf_dynamic", "readelf_symbols"):
        record = summary.get(key)
        if not isinstance(record, dict):
            _finding(findings, "dependency-elf-raw-missing",
                     f"{label} {key} raw output record is missing")
            continue
        try:
            raw_path = dependency_evidence._safe_evidence_path(root, record.get("path"))
            if not raw_path.is_file() or raw_path.stat().st_size == 0 or \
                    dependency_evidence.sha256(raw_path) != record.get("sha256"):
                raise ValueError("raw output missing, empty, or tampered")
            raw_records[key] = raw_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError, dependency_evidence.EvidenceError) as error:
            _finding(findings, "dependency-elf-raw-invalid",
                     f"{label} {key} raw output cannot be verified: {error}")
    if "readelf_dynamic" in raw_records:
        actual = dependency_evidence.parse_readelf_dynamic(raw_records["readelf_dynamic"])
        if (actual["needed"] != summary.get("needed") or
                actual["soname"] != summary.get("soname") or
                actual["rpath_runpath"] != summary.get("rpath_runpath")):
            _finding(findings, "dependency-elf-dynamic-claim",
                     f"{label} DT_NEEDED/SONAME/RPATH claims differ from raw readelf bytes")
    if "readelf_symbols" in raw_records:
        undefined, exports = dependency_evidence.parse_readelf_symbols(
            raw_records["readelf_symbols"])
        if (undefined != summary.get("undefined_symbols") or
                exports != summary.get("exports")):
            _finding(findings, "dependency-elf-symbol-claim",
                     f"{label} exported/undefined symbol claims differ from raw readelf bytes")


def _cpu_is_sanitizer_runtime_soname(soname: str) -> bool:
    return re.fullmatch(r"libclang_rt\.asan[^/]*\.so(?:\..*)?", soname, re.I) is not None


def _cpu_forbidden_soname(soname: str, *, sanitizer: bool) -> bool:
    if sanitizer and _cpu_is_sanitizer_runtime_soname(soname):
        return False
    forbidden_vulkan_dso = re.compile(r"libvulkan(?:[_-][^/]+)?\.so(?:\..*)?$", re.I)
    forbidden_compiler_dso = re.compile(r"lib(?:LLVM|MLIR|clang).*\.so(?:\..*)?$", re.I)
    return bool(forbidden_compiler_dso.search(soname) or
                forbidden_vulkan_dso.search(soname))


def _cpu_forbidden_runtime(item: dict, *, sanitizer: bool,
                           needed: set[str] | list[str] | tuple[str, ...] = (),
                           sanitizer_runtime_identity: dict | None = None) -> bool:
    soname = str(item.get("soname") or Path(str(item.get("path", ""))).name)
    path = str(item.get("path", ""))
    path_name = Path(path).name
    expected_path = sanitizer_runtime_identity.get("path") if isinstance(
        sanitizer_runtime_identity, dict) else None
    expected_hash = sanitizer_runtime_identity.get("sha256") if isinstance(
        sanitizer_runtime_identity, dict) else None
    try:
        path_matches_identity = (isinstance(expected_path, str) and
                                 str(Path(path).resolve()) == str(Path(expected_path).resolve()))
    except (OSError, RuntimeError):
        path_matches_identity = False
    direct_needed_names = item.get("direct_needed_names", [])
    if not isinstance(direct_needed_names, list):
        return True
    asan_direct_names = {str(name) for name in direct_needed_names
                         if _cpu_is_sanitizer_runtime_soname(str(name))}
    sanitizer_runtime = (
        sanitizer and item.get("classification") == "SAN" and
        item.get("role") == "sanitizer-runtime" and
        path_name.startswith("libclang_rt.asan") and
        path_matches_identity and
        isinstance(expected_hash, str) and SHA256_RE.fullmatch(expected_hash) is not None and
        item.get("sha256") == expected_hash and
        (item.get("soname") is None or
         (_cpu_is_sanitizer_runtime_soname(str(item.get("soname"))) and
          str(item.get("soname")) in asan_direct_names)) and
        bool(asan_direct_names) and asan_direct_names.issubset(needed))
    if asan_direct_names:
        return not sanitizer_runtime
    if sanitizer and _cpu_is_sanitizer_runtime_soname(soname):
        return True
    return (item.get("classification") == "compiler" or
            _cpu_forbidden_soname(soname, sanitizer=sanitizer))


def _audit_dependency_bundle(stage: Path, base: Path, evidence: dict,
                             expected_hashes: dict[str, object], label: str,
                             refs: list[dict], findings: list[dict], *,
                             sanitizer: bool = False,
                             sanitizer_runtime_identity: dict | None = None) -> dict:
    findings_before = len(findings)
    bundle = evidence.get("build_dependencies")
    if not isinstance(bundle, dict):
        _finding(findings, "dependency-bundle-missing",
                 f"{label} build_dependencies records are missing")
        return {"status": "missing", "label": label}
    dependencies = {}
    roots = {}
    for role, validator in (("vulkan", dependency_evidence.validate_dependency_evidence),
                            ("cpu", dependency_evidence.validate_elf_dependency_closure)):
        record = bundle.get(role)
        relative = Path(record.get("file", "")) if isinstance(record, dict) else Path()
        if (not isinstance(record, dict) or not isinstance(record.get("file"), str) or
                relative.is_absolute() or ".." in relative.parts):
            _finding(findings, "dependency-bundle-record",
                     f"{label} {role} dependency record is malformed or escapes its model root")
            continue
        path = _evidence_file(stage, base, record.get("file"),
                              f"{label} {role} dependency summary", refs, findings)
        _check_digest_claim(record.get("sha256"), path,
                            f"{label} {role} dependency summary", findings)
        if path is None:
            continue
        root = path.parent
        roots[role] = root
        try:
            value = validator(path, root)
            expected = expected_hashes.get(role)
            actual = value.get("model_library", {}).get("sha256")
            if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected) or actual != expected:
                _finding(findings, "dependency-model-dso-mismatch",
                         f"{label} {role} dependency capture does not identify the audited current model DSO",
                         expected=expected, actual=actual)
            _check_dependency_elf_claims(value.get("model_elf"), root,
                                         f"{label} {role} model", findings)
            for index, item in enumerate(value.get("runtime_closure", [])):
                _check_dependency_elf_claims(item, root,
                                             f"{label} {role} closure {index}", findings)
            dependencies[role] = value
        except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                json.JSONDecodeError, dependency_evidence.EvidenceError) as error:
            _finding(findings, "dependency-evidence-invalid",
                     f"{label} {role} dependency evidence failed its primary validator: {error}")
    vulkan = dependencies.get("vulkan", {})
    cpu = dependencies.get("cpu", {})
    if not vulkan or not cpu:
        return {"status": "invalid", "label": label}
    cpu_elf = cpu.get("model_elf", {})
    cpu_symbols = cpu_elf.get("undefined_symbols", [])
    cpu_needed = cpu_elf.get("needed", [])
    cpu_closure = cpu.get("runtime_closure", [])
    bad_cpu_closure = []
    allowed_asan_needed = set()
    for item in cpu_closure:
        soname = str(item.get("soname") or Path(str(item.get("path", ""))).name)
        forbidden_runtime = _cpu_forbidden_runtime(
            item, sanitizer=sanitizer, needed=set(cpu_needed),
            sanitizer_runtime_identity=sanitizer_runtime_identity)
        if forbidden_runtime:
            bad_cpu_closure.append(soname)
        else:
            allowed_asan_needed.update(
                str(name) for name in item.get("direct_needed_names", [])
                if _cpu_is_sanitizer_runtime_soname(str(name)))
    bad_cpu_needed = [str(name) for name in cpu_needed
                      if (_cpu_forbidden_soname(str(name), sanitizer=sanitizer) or
                          (sanitizer and _cpu_is_sanitizer_runtime_soname(str(name)) and
                           str(name) not in allowed_asan_needed))]
    bad_cpu_symbols = [name for name in cpu_symbols if
                       re.match(r"(?:vk[A-Za-z0-9_]|LLVM|mlir[A-Za-z0-9_]|clang[A-Za-z0-9_])", str(name))]
    if bad_cpu_needed or bad_cpu_closure or bad_cpu_symbols:
        _finding(findings, "cpu-dependency-import-policy",
                 f"{label} CPU model dynamically imports Vulkan or LLVM/MLIR",
                 needed=bad_cpu_needed, closure=bad_cpu_closure,
                 undefined_symbols=bad_cpu_symbols)
    llvm = vulkan.get("llvm", {})
    if (llvm.get("static_archive_status") not in (None, "not_measured") or
            not isinstance(llvm.get("legacy_blockers"), list) or
            not isinstance(llvm.get("status"), str) or
            not isinstance(llvm.get("absence_claim"), str)):
        _finding(findings, "dependency-legacy-llvm-status",
                 f"{label} legacy LLVM presence/blocker classification is malformed")
    else:
        selected_compiler_members = sorted({
            str(origin.get("member"))
            for archive in vulkan.get("archive_inputs", [])
            for origin in archive.get("selected_member_origins", [])
            if isinstance(origin, dict) and origin.get("classification") == "compiler"})
        compiler_dsos = sorted({str(item.get("path"))
                                for item in vulkan.get("runtime_closure", [])
                                if item.get("classification") == "compiler"})
        observed_llvm = bool(selected_compiler_members or compiler_dsos)
        mode = llvm.get("mode")
        expected_blockers = set()
        if selected_compiler_members:
            expected_blockers.add("static_llvm_archive_members_selected")
        if compiler_dsos:
            expected_blockers.add("shared_llvm_dso_in_runtime_closure")
        if mode == "static" and not selected_compiler_members:
            expected_blockers.add("configured_static_llvm_not_observed")
        if mode == "shared" and not compiler_dsos:
            expected_blockers.add("configured_shared_llvm_not_observed")
        expected_status = ("measured_presence" if observed_llvm else
                           "not_measured" if mode == "not_measured" else
                           "measured_no_presence_observed")
        expected_claim = ("presence_observed" if observed_llvm else
                          "not_measured" if mode == "not_measured" else
                          "none_observed_in_captured_link_and_runtime_closure")
        if (sorted(set(llvm.get("static_archive_members", []))) != selected_compiler_members or
                sorted(set(llvm.get("shared_objects", []))) != compiler_dsos or
                set(llvm.get("legacy_blockers", [])) != expected_blockers or
                llvm.get("status") != expected_status or
                llvm.get("absence_claim") != expected_claim):
            _finding(findings, "dependency-legacy-llvm-claim",
                     f"{label} configured LLVM presence/blockers differ from actual archive/map/runtime evidence")
    if cpu.get("llvm", {}).get("static_archive_status") != "not_measured":
        _finding(findings, "cpu-static-inputs-overclaimed",
                 f"{label} CPU model evidence must report static archive inputs as not measured")
    driver_result = {"status": "not_measured", "reason": "driver maps were not supplied"}
    driver = bundle.get("driver")
    if isinstance(driver, dict):
        driver_file = driver.get("file")
        if driver_file:
            driver_relative = Path(driver_file) if isinstance(driver_file, str) else Path()
            if driver_relative.is_absolute() or ".." in driver_relative.parts:
                _finding(findings, "driver-module-path-invalid",
                         f"{label} driver comparison path escapes its model evidence root")
                driver_file = None
        if driver_file:
            driver_path = _evidence_file(stage, base, driver_file,
                                         f"{label} driver module comparison", refs, findings)
            _check_digest_claim(driver.get("sha256"), driver_path,
                                f"{label} driver module comparison", findings)
            if driver_path is not None:
                try:
                    driver_result = load_json(driver_path)
                    comparison = dependency_evidence.compare_driver_module_snapshots(
                        driver_path.parent, persist=False)
                    if (driver.get("comparison_status") not in {None, comparison.get("status")} or
                            (driver.get("status") == "measured" and
                             driver.get("comparison_status") != "measured")):
                        _finding(findings, "driver-module-status-claim",
                                 f"{label} driver comparison status differs from raw three-phase maps")
                    if driver.get("status") == "measured" and (
                            comparison.get("status") != "measured" or
                            comparison.get("driver_origin_status") != "measured" or
                            not isinstance(comparison.get("after_dlopen_added"), list) or
                            not isinstance(comparison.get("after_create_added"), list) or
                            not comparison.get("driver_icd_modules_after_create")):
                        _finding(findings, "driver-module-phase-evidence",
                                 f"{label} driver maps do not distinguish after-dlopen from after-create")
                    manifest_claim = driver.get("manifest")
                    if driver.get("status") == "measured" and isinstance(manifest_claim, dict):
                        driver_modules = comparison.get("driver_icd_modules_after_create", [])
                        if not any(item.get("manifest_origin") == "explicit-driver-manifest"
                                   for item in driver_modules if isinstance(item, dict)):
                            _finding(findings, "driver-module-manifest-origin",
                                     f"{label} after-create driver module is not bound to its explicit ICD manifest")
                except (OSError, ValueError, RuntimeError, TypeError,
                        json.JSONDecodeError, dependency_evidence.EvidenceError) as error:
                    _finding(findings, "driver-module-evidence-invalid",
                             f"{label} driver module evidence is invalid: {error}")
        elif driver.get("status") not in {"not_measured", "not-applicable"}:
            _finding(findings, "driver-module-evidence-missing",
                     f"{label} driver status claims measurement without a comparison file")
        elif not (driver.get("reason") or driver.get("reference")):
            _finding(findings, "driver-module-not-measured-reason",
                     f"{label} missing driver map evidence must state why it was not measured")
        manifest = driver.get("manifest")
        if isinstance(manifest, dict):
            manifest_relative = Path(manifest.get("file", ""))
            if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
                _finding(findings, "driver-manifest-path-invalid",
                         f"{label} explicit ICD manifest path escapes its model evidence root")
                manifest_relative = Path("invalid")
            manifest_path = _evidence_file(stage, base, manifest_relative.as_posix(),
                                           f"{label} explicit ICD manifest", refs, findings)
            _check_digest_claim(manifest.get("sha256"), manifest_path,
                                f"{label} explicit ICD manifest", findings)
        driver_result = {**driver_result, "status": driver.get("status", "not_measured"),
                         "comparison_status": driver.get("comparison_status"),
                         "reason": driver.get("reason"),
                         "reference": driver.get("reference")}
    evidence_roots = set(roots.values())
    if isinstance(driver, dict) and isinstance(driver.get("file"), str):
        driver_path = safe_stage_file(stage, (base / driver["file"]).relative_to(stage))
        if driver_path:
            evidence_roots.add(driver_path.parent)
    evidence_file_refs = []
    for root in sorted(evidence_roots, key=str):
        evidence_file_refs.extend(_dependency_raw_artifact_refs(
            stage, root, f"{label} dependency", refs, findings))
    return {
        "status": "verified" if len(findings) == findings_before else "invalid",
        "label": label,
        "vulkan": {"file": bundle["vulkan"].get("file"),
                    "sha256": bundle["vulkan"].get("sha256"),
                    "llvm_status": llvm.get("status"),
                    "llvm_mode": llvm.get("mode"),
                    "legacy_blockers": llvm.get("legacy_blockers", []),
                    "static_archive_members": llvm.get("static_archive_members", []),
                    "runtime_closure_count": len(vulkan.get("runtime_closure", []))},
        "cpu": {"file": bundle["cpu"].get("file"),
                "sha256": bundle["cpu"].get("sha256"),
                "static_archive_status": cpu.get("llvm", {}).get("static_archive_status"),
                "runtime_closure_count": len(cpu_closure)},
        "driver": driver_result,
        "evidence_file_refs": evidence_file_refs,
    }


def _sanitizer_runtime_identity(stage: Path, gate: dict,
                                findings: list[dict]) -> dict | None:
    state = gate.get("state", {})
    evidence = state.get("sanitizer_toolchain_evidence", {}) if isinstance(state, dict) else {}
    identity_path = _verify_hash_record(
        stage, evidence.get("identity") if isinstance(evidence, dict) else None,
        "SAN toolchain identity for CPU runtime binding", findings)
    if identity_path is None:
        return None
    try:
        identity = load_json(identity_path)
        runtime = identity.get("llvm", {}).get("shared_asan_runtime_identity", {})
        path = runtime.get("path")
        digest = runtime.get("sha256")
        if (not isinstance(path, str) or not Path(path).is_absolute() or
                not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
            raise ValueError("shared ASan runtime identity lacks an absolute path and SHA-256")
        return {"path": str(Path(path).resolve()), "sha256": digest}
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
        _finding(findings, "sanitizer-runtime-identity-invalid",
                 f"configured shared ASan runtime identity is invalid: {error}")
        return None


def _audit_va0_dependencies(stage: Path, profile: str, families: dict,
                            refs: list[dict], findings: list[dict], *,
                            sanitizer_runtime_identity: dict | None = None) -> dict:
    candidates = []

    def candidate_from_path(label, evidence_path, *, kind, device=None):
        try:
            current = load_json(evidence_path)
        except (OSError, json.JSONDecodeError) as error:
            _finding(findings, "dependency-model-evidence-invalid",
                     f"{label} model evidence cannot be read: {error}")
            return
        if kind == "v5":
            identities = current.get("identities", {})
            hashes = {"vulkan": identities.get("vulkan_library_sha256"),
                      "cpu": identities.get("cpu_library_sha256")}
        else:
            fixture = current.get("fixture", {})
            hashes = {"vulkan": fixture.get("vulkan_library_sha256"),
                      "cpu": fixture.get("cpu_library_sha256")}
        candidates.append((label, evidence_path.parent, current, hashes, kind, device))

    if profile == "offline":
        offline_path = stage / "test/Native/vulkan-product-offline/evidence.json"
        if offline_path.is_file():
            candidate_from_path("offline/ReLU", offline_path, kind="offline")
    else:
        for name, record in families.get("v4", {}).items():
            evidence_path = stage / str(record.get("path", ""))
            if evidence_path.is_file():
                vendor = (record.get("identity", {}).get("vendor_id")
                          if isinstance(record.get("identity"), dict) else None)
                device = "nvidia" if _int_field(vendor) == 0x10DE else "llvmpipe"
                candidate_from_path(f"product/{device}/{name}", evidence_path,
                                    kind="product", device=device)
    for device_rows in families.get("v5", {}).values():
        for model_name, record in device_rows.items():
            if model_name not in VA0_MODEL_NAMES:
                continue
            evidence_path = stage / str(record.get("path", ""))
            if not evidence_path.is_file():
                continue
            vendor = (record.get("identity", {}).get("vendor_id")
                      if isinstance(record.get("identity"), dict) else None)
            current = None
            try:
                current = load_json(evidence_path)
                vendor = current.get("device_info", {}).get("vendor_id", vendor)
            except (OSError, json.JSONDecodeError) as error:
                _finding(findings, "dependency-model-evidence-invalid",
                         f"V5 {model_name} evidence cannot be read: {error}")
                continue
            device = "nvidia" if _int_field(vendor) == 0x10DE else "llvmpipe"
            candidate_from_path(f"v5/{device}/{model_name}", evidence_path,
                                kind="v5", device=device)
    if not candidates:
        _finding(findings, "dependency-evidence-missing",
                 "profile has no current generated model DSO dependency evidence")
        return {"status": "missing", "models": {}}
    models = {}
    for label, base, evidence, expected_hashes, kind, device in candidates:
        record = _audit_dependency_bundle(
            stage, base, evidence, expected_hashes, label, refs, findings,
            sanitizer=(profile == "sanitize"),
            sanitizer_runtime_identity=sanitizer_runtime_identity)
        if kind == "product" and record.get("driver", {}).get("status") != "measured":
            _finding(findings, "native-driver-phase-evidence-missing",
                     f"{label} Native run must measure distinct after-dlopen/after-create driver phases")
        if kind == "v5" and record.get("driver", {}).get("status") != "measured":
            native_matches = []
            for native_name, native in families.get("v4", {}).items():
                identity = native.get("identity", {}) if isinstance(native, dict) else {}
                native_device = "nvidia" if _int_field(identity.get("vendor_id")) == 0x10DE else "llvmpipe"
                if native_device == device and native.get("evidence", {}).get("build_dependencies", {}).get("driver", {}).get("status") == "measured":
                    native_matches.append(native)
            if len(native_matches) != 1:
                _finding(findings, "v5-driver-reference-missing",
                         f"{label} driver maps are not measured by V5 and no unique same-ICD Native proof is available")
            else:
                native = native_matches[0]
                native_path = native.get("path")
                native_driver = native.get("evidence", {}).get("build_dependencies", {}).get("driver", {})
                record["driver"]["same_icd_native_reference"] = {
                    "profile": profile, "device": device,
                    "test_evidence_path": native_path,
                    "driver_comparison_file": native_driver.get("file"),
                    "driver_comparison_sha256": native_driver.get("sha256"),
                    "v5_maps_claimed": False,
                }
        models[label] = record
    configured_dirs = []
    cache_path = stage / "CMakeCache.txt"
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("LLVM_DIR:", "MLIR_DIR:")) and "=" in line:
                key, value = line.split("=", 1)
                if value and value not in {"NOTFOUND", ""}:
                    configured_dirs.append({"key": key.split(":", 1)[0], "path": value})
    return {"status": "verified" if models and all(
                item.get("status") == "verified" for item in models.values()) else "invalid",
            "models": models,
            "configured_llvm_directories": configured_dirs,
            "cpu_static_archive_inputs": "not_measured",
            "legacy_blockers_are_observations_not_va0_no_go": True}


def _audit_va0_baseline(stage: Path, gate: dict, families: dict,
                        refs: list[dict], findings: list[dict]) -> dict:
    gate_state = gate.get("state", {})
    reference = _audit_legacy_reference(stage, gate_state, refs, findings)
    comparisons = {"relu-product": {}, "add_relu": {}, "conv_relu": {}, "squeezenet_v1_1": {}}
    shader_comparisons = []
    profile = gate.get("profile")
    if reference:
        reference_fixtures = reference["fixtures"]
        reference_root = stage / "legacy-reference/fixtures"
        devices = _devices_for_profile(profile)
        if profile == "offline":
            current_root = stage / "test/Native/vulkan-product-offline"
            if (current_root / "evidence.json").is_file():
                comparisons["relu-product"]["offline"] = _va0_compare_relu_product(
                    stage, reference_root / "relu-product-nvidia", current_root,
                    "offline product-only ReLU", findings, shader_comparisons)
            else:
                _finding(findings, "va0-current-product-missing",
                         "offline profile lacks its current generated ReLU product snapshot")
        else:
            product_records = families.get("v4", {})
            if not isinstance(product_records, dict) or not product_records:
                _finding(findings, "va0-current-product-missing",
                         "device profile lacks actual current legacy-product ReLU records")
            else:
                for record_name, product in product_records.items():
                    current_root = stage / str(product.get("path", "")).rsplit("/", 1)[0]
                    vendor_id = (product.get("identity", {}).get("vendor_id")
                                 if isinstance(product.get("identity"), dict) else None)
                    reference_device = "nvidia" if _int_field(vendor_id) == 0x10DE else "llvmpipe"
                    comparisons["relu-product"][record_name] = _va0_compare_relu_product(
                        stage, reference_root / f"relu-product-{reference_device}", current_root,
                        f"current product {record_name}", findings, shader_comparisons)
        model_paths = {
            "add_relu": {"nvidia": "test/Vulkan/evidence/v5/nvidia/models/add_relu",
                         "llvmpipe": "test/Vulkan/evidence/v5/llvmpipe/models/add_relu"},
            "conv_relu": {"nvidia": "test/Vulkan/evidence/v5/nvidia/models/conv_relu",
                          "llvmpipe": "test/Vulkan/evidence/v5/llvmpipe/models/conv_relu"},
            "squeezenet_v1_1": {
                "nvidia": "test/Vulkan/evidence/v5/squeezenet-nvidia/models/squeezenet_v1_1",
                "llvmpipe": "test/Vulkan/evidence/v5/squeezenet-llvmpipe/models/squeezenet_v1_1"},
        }
        current_records = families.get("v5", {})
        for model_name, paths in model_paths.items():
            for device in sorted(devices):
                reference_key = f"{VA0_MODEL_NAMES[model_name]}-{device}"
                old_row = reference_fixtures.get(reference_key)
                old_rel = old_row.get("path") if isinstance(old_row, dict) else None
                matching = [record for by_model in current_records.values()
                            for name, record in by_model.items()
                            if name == model_name and
                            (f"/{device}/" in str(record.get("path", "/")) or
                             str(record.get("path", "")).endswith(
                                 f"{device}/models/{model_name}/evidence.json"))]
                if not isinstance(old_rel, str) or len(matching) != 1:
                    _finding(findings, "va0-small-model-reference-missing",
                             "profile lacks exactly one current and one frozen model record",
                             model=model_name, device=device, current_records=len(matching))
                    continue
                current_rel = matching[0].get("path")
                current_root = stage / str(current_rel).rsplit("/", 1)[0]
                comparisons[model_name][device] = _va0_compare_small_model(
                    stage, stage / old_rel, current_root, model_name,
                    f"{profile}/{device}/{model_name}", refs, findings,
                    shader_comparisons)
    else:
        for identifier in ("VA0.1", "VA0.2", "VA0.3", "VA0.4"):
            pass
    manifest_paths = []
    if profile == "offline":
        manifest_paths.append(stage / "test/Native/vulkan-product-offline/vulkan/paired_relu_vulkan.json")
    else:
        for product in families.get("v4", {}).values():
            record_path = product.get("path") if isinstance(product, dict) else None
            if isinstance(record_path, str):
                base = stage / record_path.rsplit("/", 1)[0]
                manifest_paths.append(base / "vulkan/paired_relu_vulkan.json")
        for models in families.get("v5", {}).values():
            for model_name, model in models.items():
                if model_name not in VA0_MODEL_NAMES:
                    continue
                record_path = model.get("path")
                if isinstance(record_path, str):
                    model_root = stage / record_path.rsplit("/", 1)[0]
                    vulkan_dir = model_root / "vulkan"
                    manifest_paths.extend(path for path in vulkan_dir.glob("*.json")
                                          if not path.name.endswith(".plan.json"))
    host_contract_verified = bool(manifest_paths)
    for manifest_path in manifest_paths:
        try:
            manifest = load_json(manifest_path)
            valid = (isinstance(manifest, dict) and
                     manifest.get("host_implementation") == "generic-plan-runtime" and
                     type(manifest.get("host_codegen_version")) is int and
                     manifest.get("host_codegen_version") == 1 and
                     manifest.get("execution_plan_usage") == "embedded-runtime-required")
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            host_contract_verified = False
            _finding(findings, "va0-current-host-contract",
                     "current legacy-product manifest does not pin generic-plan-runtime codegen v1",
                     path=str(manifest_path.relative_to(stage)) if manifest_path.is_relative_to(stage) else str(manifest_path))
    if not host_contract_verified:
        _finding(findings, "va0-current-host-contract-missing",
                 "no verified current legacy-product manifest was found for this profile")
    sanitizer_runtime = (_sanitizer_runtime_identity(stage, gate, findings)
                         if profile == "sanitize" else None)
    dependencies = _audit_va0_dependencies(
        stage, profile, families, refs, findings,
        sanitizer_runtime_identity=sanitizer_runtime)
    migration = _va0_migration_coverage(gate)
    return {"legacy_reference": reference, "comparisons": comparisons,
            "shader_comparisons": shader_comparisons,
            "dependencies": dependencies,
            "current_host_contract_verified": host_contract_verified,
            "migration_coverage": migration,
            "host_aot_release_eligible": False,
            "host_aot_status": "not-implemented-va1-and-later",
            "scope": "legacy generic-plan-runtime only; no current host AOT/no-LLVM claim"}


def _audit_va1_stage(gate: dict, families: dict, refs: list[dict],
                    findings: list[dict]) -> dict:
    stage_id = gate.get("stage_id")
    tests = set(gate.get("ctest", {}).get("test_names", []))
    missing = sorted(VA1_REQUIRED_TESTS - tests)
    if stage_id != "VA1":
        return {"status": "not-applicable", "stage_id": stage_id,
                "host_aot_candidate": False}
    if missing:
        _finding(findings, "va1-test-inventory-missing",
                 "VA1 stage lacks required host-program/planner tests",
                 tests=missing)
    host_refs = [item for item in refs
                 if "host-program" in str(item.get("role", "")).lower() or
                 "host-program" in str(item.get("path", "")).lower()]
    native_ok = bool(host_refs) and not any(
        item.get("code", "").startswith("host-program") or
        item.get("code", "").startswith("v5-host-program")
        for item in findings)
    if not native_ok:
        _finding(findings, "va1-host-program-raw-evidence-missing",
                 "VA1 lacks validated native raw host-program evidence")
    return {"status": "verified" if not missing and native_ok else "blocked",
            "stage_id": "VA1", "required_tests": sorted(VA1_REQUIRED_TESTS),
            "tests_executed": sorted(tests & VA1_REQUIRED_TESTS),
            "tests_missing": missing, "host_program_evidence": host_refs,
            "host_execution_role": "legacy-product", "host_aot_candidate": False,
            "claim_scope": "compiler-only planner evidence; no candidate GPU execution"}


def _audit_va2_helper(stage: Path, gate: dict, refs: list[dict],
                      findings: list[dict]) -> dict:
    if gate.get("stage_id") != "VA2":
        return {"status": "not-applicable", "stage_id": gate.get("stage_id"),
                "host_aot_candidate": False}
    relative = "test/Vulkan/evidence/va2/independence-evidence.json"
    evidence_path = stage / relative
    try:
        ctest = gate.get("ctest", {})
        test_names = ctest.get("test_names", []) if isinstance(ctest, dict) else []
        skipped = ctest.get("skipped", []) if isinstance(ctest, dict) else []
        failed = ctest.get("failed", []) if isinstance(ctest, dict) else []
        if (not isinstance(test_names, list) or
                any(not isinstance(name, str) for name in test_names) or
                not isinstance(skipped, list) or not isinstance(failed, list)):
            raise device_helper_audit.EvidenceError("VA2 actual CTest execution inventory is malformed")
        missing_tests = sorted(VA2_REQUIRED_TESTS - set(test_names))
        incomplete_tests = sorted(VA2_REQUIRED_TESTS & (set(skipped) | set(failed)))
        if missing_tests or incomplete_tests:
            raise device_helper_audit.EvidenceError(
                f"VA2 required tests missing/skipped/failed: missing={missing_tests}, "
                f"incomplete={incomplete_tests}")
        evidence = device_helper_audit.validate_independence_evidence(
            evidence_path, stage)
        raw = evidence.get("raw_artifacts")
        if not isinstance(raw, list) or not raw:
            raise device_helper_audit.EvidenceError(
                "VA2 independence evidence has no archived raw artifact inventory")
        seen = set()
        stage_refs = []
        for record in raw:
            if not isinstance(record, dict):
                raise device_helper_audit.EvidenceError(
                    "VA2 independence raw artifact record is malformed")
            path = device_helper_audit._safe_path(stage, record.get("path"))
            digest = record.get("sha256")
            role = record.get("role")
            if (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or
                    device_helper_audit.sha256_file(path) != digest or
                    not isinstance(role, str) or not role):
                raise device_helper_audit.EvidenceError(
                    "VA2 independence raw artifact path/hash/role is invalid")
            item_path = str(path.relative_to(stage))
            if item_path in seen:
                raise device_helper_audit.EvidenceError(
                    f"VA2 independence raw artifact repeats path: {item_path}")
            seen.add(item_path)
            item = {"path": item_path, "sha256": digest, "role": role}
            refs.append(item)
            stage_refs.append(item)
        refs.append({"path": relative, "sha256": sha256_file(evidence_path),
                     "role": "VA2 helper independence evidence"})
        return {"status": "verified", "stage_id": "VA2",
                "required_tests": sorted(VA2_REQUIRED_TESTS),
                "tests_executed": sorted(VA2_REQUIRED_TESTS & set(test_names)),
                "independence_evidence": {
                    "path": relative, "sha256": sha256_file(evidence_path)},
                "raw_artifacts": stage_refs,
                "devices": sorted(device_helper_audit.DEVICE_TESTS.values()),
                "host_execution_role": "legacy-product",
                "host_aot_candidate": False,
                "claim_scope": "plan-free helper and native raw evidence only; no VA3 model-host generation"}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.SubprocessError, ET.ParseError) as error:
        _finding(findings, "va2-helper-evidence-invalid",
                 f"VA2 helper/native/dependency evidence is missing or invalid: {error}")
        return {"status": "blocked", "stage_id": "VA2",
                "independence_evidence": None, "raw_artifacts": [],
                "host_execution_role": "legacy-product", "host_aot_candidate": False}


def _audit_va3_lifecycle(stage: Path, gate: dict, refs: list[dict],
                        findings: list[dict]) -> dict:
    if gate.get("stage_id") != "VA3":
        return {"status": "not-applicable", "stage_id": gate.get("stage_id"),
                "role": "none", "host_aot_release_eligible": False}
    ctest = gate.get("ctest", {})
    names = ctest.get("test_names", []) if isinstance(ctest, dict) else []
    skipped = ctest.get("skipped", []) if isinstance(ctest, dict) else []
    failed = ctest.get("failed", []) if isinstance(ctest, dict) else []
    try:
        if (not isinstance(names, list) or any(not isinstance(name, str) for name in names) or
                not isinstance(skipped, list) or not isinstance(failed, list)):
            raise lifecycle_audit.EvidenceError("VA3 CTest execution inventory is malformed")
        aggregate, raw = lifecycle_audit.aggregate_stage_evidence(
            stage, gate.get("profile"), gate.get("source_fingerprint"),
            names, skipped, failed)
        if (aggregate.get("stage_id") != "VA3" or
                aggregate.get("profile") != gate.get("profile") or
                aggregate.get("source_fingerprint") != gate.get("source_fingerprint") or
                aggregate.get("role") != "lifecycle-only" or
                aggregate.get("candidate_run") != "not_run" or
                aggregate.get("host_aot_release_eligible") is not False):
            raise lifecycle_audit.EvidenceError("VA3 aggregate identity or claim scope is invalid")
        for item in raw:
            path = lifecycle_audit._safe_file(stage, item.get("path"), "VA3 raw artifact")
            digest = item.get("sha256")
            if (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or
                    sha256_file(path) != digest):
                raise lifecycle_audit.EvidenceError("VA3 raw artifact hash changed during aggregation")
            refs.append({"path": item["path"], "sha256": digest,
                         "role": item["role"]})
        aggregate_path = stage / "test/Vulkan/evidence/va3/lifecycle-evidence.json"
        result = {"status": "verified", "stage_id": "VA3", "role": "lifecycle-only",
                  "path": str(aggregate_path.relative_to(stage)),
                  "sha256": sha256_file(aggregate_path),
                  "source_fingerprint": aggregate["source_fingerprint"],
                  "required_tests": aggregate["required_tests"],
                  "codegen_tests": aggregate["codegen_tests"],
                  "devices": aggregate["devices"],
                  "fake": aggregate["fake"],
                  "raw_artifacts": raw,
                  "candidate_run": "not_run",
                  "host_aot_release_eligible": False,
                  "claim_scope": "private lifecycle-only candidate; no model run or release product"}
        refs.append({"path": result["path"], "sha256": result["sha256"],
                     "role": "VA3 post-CTest lifecycle aggregate"})
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.SubprocessError) as error:
        _finding(findings, "va3-lifecycle-evidence-invalid",
                 f"VA3 lifecycle candidate evidence is missing or invalid: {error}")
        return {"status": "blocked", "stage_id": "VA3", "role": "lifecycle-only",
                "path": None, "raw_artifacts": [],
                "candidate_run": "not_run", "host_aot_release_eligible": False}


def _audit_va4_fixedrun(stage: Path, gate: dict, refs: list[dict],
                       findings: list[dict]) -> dict:
    if gate.get("stage_id") != "VA4":
        return {"status": "not-applicable", "stage_id": gate.get("stage_id"),
                "role": "none", "host_aot_release_eligible": False}
    ctest = gate.get("ctest", {})
    names = ctest.get("test_names", []) if isinstance(ctest, dict) else []
    skipped = ctest.get("skipped", []) if isinstance(ctest, dict) else []
    failed = ctest.get("failed", []) if isinstance(ctest, dict) else []
    try:
        if (not isinstance(names, list) or any(not isinstance(name, str) for name in names) or
                not isinstance(skipped, list) or not isinstance(failed, list)):
            raise fixedrun_evidence.EvidenceError("VA4 CTest execution inventory is malformed")
        cache = _parse_cmake_cache(stage / "CMakeCache.txt")
        legacy_root = cache.get("NCNN_VULKAN_FIXEDRUN_LEGACY_ROOT", "")
        aggregate, raw = fixedrun_evidence.aggregate_stage_evidence(
            stage, gate.get("profile"), gate.get("source_fingerprint"),
            names, skipped, failed, require_legacy=bool(legacy_root))
        for item in raw:
            path = safe_stage_file(stage, item.get("path"))
            digest = item.get("sha256")
            if (path is None or not path.is_file() or not isinstance(digest, str) or
                    not SHA256_RE.fullmatch(digest) or sha256_file(path) != digest):
                raise fixedrun_evidence.EvidenceError(
                    "VA4 raw or aggregate artifact changed during aggregation")
            refs.append({"path": item["path"], "sha256": digest,
                         "role": item["role"]})
        if (aggregate.get("stage_id") != "VA4" or
                aggregate.get("profile") != gate.get("profile") or
                aggregate.get("source_fingerprint") != gate.get("source_fingerprint") or
                aggregate.get("role") != "fixed-run-candidate" or
                aggregate.get("host_aot_release_eligible") is not False):
            raise fixedrun_evidence.EvidenceError("VA4 aggregate identity or scope is invalid")
        candidate_dso_sha256s = aggregate.get("candidate_dso_sha256s")
        devices = aggregate.get("devices")
        if (not isinstance(candidate_dso_sha256s, dict) or
                set(candidate_dso_sha256s) != fixedrun_evidence.FIXTURES or
                any(not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
                    for digest in candidate_dso_sha256s.values()) or
                not isinstance(devices, dict) or set(devices) != set(fixedrun_evidence.DEVICES)):
            raise fixedrun_evidence.EvidenceError(
                "VA4 aggregate fixture DSO or device inventory is invalid")
        device_legacy_qualified = []
        for device in fixedrun_evidence.DEVICES:
            device_summary = devices[device]
            device_legacy = (device_summary.get("legacy_reference")
                             if isinstance(device_summary, dict) else None)
            if (not isinstance(device_legacy, dict) or
                    type(device_legacy.get("qualified")) is not bool):
                raise fixedrun_evidence.EvidenceError(
                    f"VA4 {device} legacy-reference qualification is malformed")
            device_legacy_qualified.append(device_legacy["qualified"])
        if (type(aggregate.get("legacy_reference_qualified")) is not bool or
                aggregate["legacy_reference_qualified"] != all(device_legacy_qualified)):
            raise fixedrun_evidence.EvidenceError(
                "VA4 aggregate legacy-reference qualification is inconsistent")
        path = stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json"
        refs.append({"path": path.relative_to(stage).as_posix(),
                     "sha256": sha256_file(path),
                     "role": "VA4 post-CTest fixed-run aggregate"})
        legacy_qualified = aggregate.get("legacy_reference_qualified") is True
        return {"status": "verified" if legacy_qualified else "unqualified",
                "stage_id": "VA4", "role": "fixed-run-candidate",
                "legacy_reference_qualified": legacy_qualified,
                "path": str(path.relative_to(stage)),
                "sha256": sha256_file(path),
                "source_fingerprint": aggregate["source_fingerprint"],
                "candidate_dso_sha256": aggregate["candidate_dso_sha256s"],
                "devices": aggregate["devices"],
                "host_aot_release_eligible": False,
                "raw_artifacts": raw,
                "claim_scope": aggregate["claim_scope"]}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.SubprocessError) as error:
        _finding(findings, "va4-fixedrun-evidence-invalid",
                 f"VA4 fixed-run evidence is missing or invalid: {error}")
        return {"status": "blocked", "stage_id": "VA4",
                "role": "fixed-run-candidate", "path": None,
                "raw_artifacts": [], "host_aot_release_eligible": False}


VA5_DEPLOYMENT_EVIDENCE_PATHS = {
    "release": ("test/Vulkan/evidence/va5/nvidia/deployment-evidence.json",
                "test/Vulkan/evidence/va5/llvmpipe/deployment-evidence.json",
                "test/Vulkan/evidence/va5/no-device/deployment-evidence.json"),
    "sanitize": ("test/Vulkan/evidence/va5/nvidia/deployment-evidence.json",
                 "test/Vulkan/evidence/va5/llvmpipe/deployment-evidence.json",
                 "test/Vulkan/evidence/va5/no-device/deployment-evidence.json"),
    "offline": ("test/Vulkan/evidence/va5/no-device/deployment-evidence.json",),
    "software": ("test/Vulkan/evidence/va5/llvmpipe/deployment-evidence.json",
                 "test/Vulkan/evidence/va5/no-device/deployment-evidence.json"),
    "hardware": ("test/Vulkan/evidence/va5/nvidia/deployment-evidence.json",
                 "test/Vulkan/evidence/va5/no-device/deployment-evidence.json"),
}


def _audit_va5_deployment_evidence_document(document: dict, expected_profile: str,
                                            scenario_profile: str,
                                            findings: list[dict],
                                            label: str) -> bool:
    """scenario_profile is "device" or "no-device"; the document's own
    profile field must match it and selects the acceptance branch."""
    if (not isinstance(document, dict) or
            document.get("schema_version") != 1 or
            document.get("kind") != va5_deployment.EVIDENCE_KIND or
            document.get("profile") != expected_profile):
        _finding(findings, "va5-deployment-evidence-invalid",
                 f"{label} deployment evidence schema/profile is invalid")
        return False
    if document.get("host_contract") != va5_deployment.HOST_CONTRACT:
        _finding(findings, "va5-deployment-evidence-invalid",
                 f"{label} deployment evidence pins an unexpected host contract")
        return False
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, dict) or \
            set(scenarios) != set(va5_deployment.FIXTURES):
        _finding(findings, "va5-deployment-evidence-invalid",
                 f"{label} deployment evidence scenario inventory is invalid")
        return False
    valid = True
    for fixture, scenario in scenarios.items():
        if not isinstance(scenario, dict):
            _finding(findings, "va5-deployment-scenario-invalid",
                     f"{label} {fixture} deployment scenario is malformed")
            valid = False
            continue
        if scenario.get("status") == "not-applicable-offline":
            if expected_profile != "no-device":
                _finding(findings, "va5-deployment-scenario-invalid",
                         f"{label} {fixture} is unexpectedly skipped on a device profile")
                valid = False
            continue
        if scenario.get("status") != "passed":
            _finding(findings, "va5-deployment-scenario-failed",
                     f"{label} {fixture} deployment scenario did not pass")
            valid = False
            continue
        deployment = scenario.get("deployment")
        deployed = (deployment.get("deployed")
                    if isinstance(deployment, dict) else None)
        if (not isinstance(deployed, list) or
                sorted(item.get("file", "") for item in deployed) !=
                sorted((f"{scenario.get('model')}_vulkan.h",
                        f"lib{scenario.get('model')}_vulkan.so"))):
            _finding(findings, "va5-deployment-directory-invalid",
                     f"{label} {fixture} deployment directory is not the minimal "
                     "header+DSO pair")
            valid = False
        loader = scenario.get("loader", {})
        environment = loader.get("environment", {})
        scrubbed = str(environment.get("scrubbed_LD_LIBRARY_PATH", ""))
        if scrubbed:
            segments = [segment for segment in scrubbed.split(":") if segment]
            leaked = [segment for segment in segments
                      if any(marker.lower() in segment.lower()
                             for marker in
                             va5_deployment.FORBIDDEN_LIBRARY_PATH_MARKERS)]
            if leaked:
                _finding(findings, "va5-deployment-environment-leaked",
                         f"{label} {fixture} loader environment kept build/LLVM "
                         "search paths", segments=leaked)
                valid = False
        if scenario_profile == "no-device":
            if (loader.get("gpu_numeric") is not False or
                    scenario.get("loader", {}).get("no_device_create_status")
                    not in (2, 5)):
                _finding(findings, "va5-deployment-no-device-invalid",
                         f"{label} {fixture} no-device create failure is not "
                         "explicitly recorded")
                valid = False
        else:
            if loader.get("gpu_numeric") is not True:
                _finding(findings, "va5-deployment-numeric-missing",
                         f"{label} {fixture} device deployment lacks numeric "
                         "execution")
                valid = False
            reproducibility = scenario.get("reproducibility", {})
            if reproducibility.get("mode") != "byte-identical" or \
                    not reproducibility.get("artifacts"):
                _finding(findings, "va5-deployment-reproducibility-invalid",
                         f"{label} {fixture} artifact reproducibility is not "
                         "byte-identical")
                valid = False
            protection = scenario.get("failed_compile_preserves_outputs", {})
            preserved = protection.get("preserved_sha256s")
            published = scenario.get("published")
            if (not isinstance(preserved, dict) or not preserved or
                    preserved != published):
                _finding(findings, "va5-deployment-protection-invalid",
                         f"{label} {fixture} failed compile did not demonstrably "
                         "preserve the published artifacts")
                valid = False
    return valid


def _audit_va5_deployment(stage: Path, gate: dict, refs: list[dict],
                          findings: list[dict]) -> dict:
    if gate.get("stage_id") != "VA5":
        return {"status": "not-applicable", "stage_id": gate.get("stage_id"),
                "role": "none", "host_aot_release_eligible": False}
    ctest = gate.get("ctest", {})
    names = ctest.get("test_names", []) if isinstance(ctest, dict) else []
    skipped = ctest.get("skipped", []) if isinstance(ctest, dict) else []
    failed = ctest.get("failed", []) if isinstance(ctest, dict) else []
    missing_required = sorted(VA5_REQUIRED_TESTS - set(names))
    if missing_required:
        _finding(findings, "va5-required-ctest-missing",
                 "VA5 stage lacks required deployment/link-audit tests",
                 tests=missing_required)
    not_passed = sorted((VA5_REQUIRED_TESTS & set(skipped)) |
                        (VA5_REQUIRED_TESTS & set(failed)))
    if not_passed:
        _finding(findings, "va5-required-ctest-not-passed",
                 "VA5 required deployment tests were skipped or failed",
                 tests=not_passed)
    profile = gate.get("profile")
    paths = VA5_DEPLOYMENT_EVIDENCE_PATHS.get(profile, ())
    if not paths:
        _finding(findings, "va5-deployment-evidence-missing",
                 f"VA5 profile {profile!r} has no deployment evidence mapping")
        return {"status": "blocked", "stage_id": "VA5", "role": "deployment",
                "host_aot_release_eligible": False}
    documents_ok = True
    for relative in paths:
        path = stage / relative
        expected_profile = "no-device" if "no-device" in relative else "device"
        if not path.is_file():
            _finding(findings, "va5-deployment-evidence-missing",
                     f"VA5 deployment evidence is missing: {relative}")
            documents_ok = False
            continue
        try:
            document = load_json(path)
            refs.append({"path": relative, "sha256": sha256_file(path),
                         "role": f"VA5 deployment evidence ({expected_profile})"})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "va5-deployment-evidence-invalid",
                     f"VA5 deployment evidence is unreadable: {relative}: {error}")
            documents_ok = False
            continue
        documents_ok = _audit_va5_deployment_evidence_document(
            document, expected_profile, expected_profile, findings,
            relative) and documents_ok
    # Dynamic/static audit replay: the product dependency evidence captured by
    # the paired-ReLU product run must record the forbidden-LLVM absence claim.
    dependency_root = stage / "test/Native/vulkan-product-cli/link-evidence"
    dependency_path = dependency_root / "model-dependency-evidence.json"
    if dependency_path.is_file():
        try:
            model_evidence = load_json(dependency_path)
            llvm = model_evidence.get("llvm", {})
            if llvm.get("mode") != "forbidden":
                _finding(findings, "va5-dependency-mode-invalid",
                         "product dependency evidence did not use the "
                         "forbidden-LLVM audit mode")
                documents_ok = False
            if llvm.get("legacy_blockers"):
                _finding(findings, "va5-dependency-llvm-present",
                         "product link/runtime closure observed compiler "
                         "artifacts", blockers=llvm.get("legacy_blockers"))
                documents_ok = False
            refs.append({"path": dependency_path.relative_to(stage).as_posix(),
                         "sha256": sha256_file(dependency_path),
                         "role": "VA5 product dependency evidence"})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "va5-dependency-evidence-invalid",
                     f"product dependency evidence is unreadable: {error}")
            documents_ok = False
    else:
        _finding(findings, "va5-dependency-evidence-missing",
                 "VA5 product dependency evidence file is missing")
        documents_ok = False
    if documents_ok and not missing_required and not not_passed:
        return {"status": "verified", "stage_id": "VA5", "role": "deployment",
                "deployment_evidence": [str(path) for path in paths],
                "host_aot_release_eligible": False}
    return {"status": "blocked", "stage_id": "VA5", "role": "deployment",
            "deployment_evidence": [str(path) for path in paths],
            "host_aot_release_eligible": False}


VA6_EVIDENCE_PATHS = {
    "release": ("test/Vulkan/evidence/va6/matrix/product-matrix-evidence.json",
                "test/Vulkan/evidence/va6/faults/host-faults-evidence.json",
                "test/Vulkan/evidence/va6/cross-icd/cross-icd-evidence.json"),
    "sanitize": ("test/Vulkan/evidence/va6/matrix/product-matrix-evidence.json",
                 "test/Vulkan/evidence/va6/faults/host-faults-evidence.json",
                 "test/Vulkan/evidence/va6/cross-icd/cross-icd-evidence.json"),
}


def _audit_va6_matrix_evidence_document(document: dict, kind: str,
                                        findings: list[dict],
                                        label: str) -> bool:
    if (not isinstance(document, dict) or
            document.get("schema_version") != 1 or
            document.get("kind") != kind):
        _finding(findings, "va6-evidence-invalid",
                 f"{label} evidence schema/kind is invalid")
        return False
    return True


def _audit_va6_matrix(stage: Path, gate: dict, refs: list[dict],
                      findings: list[dict]) -> dict:
    if gate.get("stage_id") != "VA6":
        return {"status": "not-applicable", "stage_id": gate.get("stage_id"),
                "role": "none", "host_aot_release_eligible": False}
    ctest = gate.get("ctest", {})
    names = ctest.get("test_names", []) if isinstance(ctest, dict) else []
    skipped = ctest.get("skipped", []) if isinstance(ctest, dict) else []
    failed = ctest.get("failed", []) if isinstance(ctest, dict) else []
    profile = gate.get("profile")
    devices = _devices_for_profile(profile)
    required = set(VA6_REQUIRED_TESTS)
    if set(devices) != {"nvidia", "llvmpipe"}:
        required -= VA6_DEVICE_CROSS_ICD_TESTS
    missing_required = sorted(required - set(names))
    if missing_required:
        _finding(findings, "va6-required-ctest-missing",
                 "VA6 stage lacks required matrix/faults/cross-ICD tests",
                 tests=missing_required)
    not_passed = sorted((required & set(skipped)) | (required & set(failed)))
    if not_passed:
        _finding(findings, "va6-required-ctest-not-passed",
                 "VA6 required tests were skipped or failed", tests=not_passed)
    paths = VA6_EVIDENCE_PATHS.get(profile, ())
    if not paths:
        _finding(findings, "va6-evidence-missing",
                 f"VA6 profile {profile!r} has no evidence mapping")
        return {"status": "blocked", "stage_id": "VA6", "role": "matrix",
                "host_aot_release_eligible": False}
    documents_ok = True
    matrix_document = None
    faults_document = None
    cross_icd_document = None
    for relative in paths:
        path = stage / relative
        if not path.is_file():
            _finding(findings, "va6-evidence-missing",
                     f"VA6 evidence is missing: {relative}")
            documents_ok = False
            continue
        try:
            document = load_json(path)
            refs.append({"path": relative, "sha256": sha256_file(path),
                         "role": "VA6 matrix evidence"})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "va6-evidence-invalid",
                     f"VA6 evidence is unreadable: {relative}: {error}")
            documents_ok = False
            continue
        if "product-matrix" in relative:
            if not _audit_va6_matrix_evidence_document(
                    document,
                    "ncnn.vulkan_va6_product_matrix_evidence",
                    findings, relative):
                documents_ok = False
                continue
            cases = document.get("cases")
            negatives = document.get("negative_cases")
            positives = document.get("positive_controls")
            if (not isinstance(cases, dict) or not negatives or not positives
                    or set(cases) != set(negatives) | set(positives)):
                _finding(findings, "va6-evidence-invalid",
                         f"{relative} case inventory is invalid")
                documents_ok = False
                continue
            if document.get("host_contract") != {
                    "host_implementation": "model-fixed-run",
                    "host_codegen_version": 2,
                    "host_source_codegen_version":
                        "ncnn-vulkan-fixed-run-codegen/v1",
                    "execution_plan_usage": "build-evidence-only"}:
                _finding(findings, "va6-evidence-invalid",
                         f"{relative} pins an unexpected host contract")
                documents_ok = False
                continue
            for name in negatives:
                case = cases.get(name)
                if (not isinstance(case, dict) or case.get("kind") != "negative"
                        or case.get("returncode") == 0 or
                        not case.get("published") == []):
                    _finding(findings, "va6-negative-case-invalid",
                             f"{relative} negative case {name} did not "
                             "fail closed")
                    documents_ok = False
            for name in positives:
                case = cases.get(name)
                if (not isinstance(case, dict) or case.get("kind") != "positive"
                        or case.get("returncode") != 0 or
                        len(case.get("published", [])) < 4 or
                        not case.get("host_codegen_identity")):
                    _finding(findings, "va6-positive-case-invalid",
                             f"{relative} positive case {name} did not "
                             "publish the fixed-run artifacts")
                    documents_ok = False
            matrix_document = document
        elif "host-faults" in relative:
            if not _audit_va6_matrix_evidence_document(
                    document, "ncnn.vulkan_va6_host_faults_evidence",
                    findings, relative):
                documents_ok = False
                continue
            if (document.get("model") != "va6_faults" or
                    document.get("fault_test", {}).get("returncode") != 0 or
                    "passed" not in str(
                        document.get("fault_test", {}).get("stdout", "")) or
                    not document.get("wrapper", {}).get("wrapper_sha256") or
                    not document.get("host_codegen_identity") or
                    document.get("run_dispatches") != 4 or
                    document.get("run_copies") != 3 or
                    document.get("product_dependency_audit", {})
                    .get("forbidden_absent") is not True):
                _finding(findings, "va6-faults-evidence-invalid",
                         f"{relative} fault-chain evidence is incomplete or "
                         "the fault binary did not pass")
                documents_ok = False
            faults_document = document
        elif "cross-icd" in relative:
            if not _audit_va6_matrix_evidence_document(
                    document, "ncnn.vulkan_va6_cross_icd_evidence",
                    findings, relative):
                documents_ok = False
                continue
            scenarios = document.get("scenarios")
            icds = document.get("icds")
            if (not isinstance(scenarios, dict) or
                    set(scenarios) != {"relu", "conv_relu"} or
                    not isinstance(icds, list) or len(icds) < 2 or
                    len({icd.get("name") for icd in icds}) != len(icds)):
                _finding(findings, "va6-evidence-invalid",
                         f"{relative} cross-ICD scenario/ICD inventory is "
                         "invalid")
                documents_ok = False
                continue
            for fixture, scenario in scenarios.items():
                devices_seen = set()
                device_runs = scenario.get("devices")
                if not isinstance(device_runs, dict) or len(device_runs) < 2:
                    _finding(findings, "va6-cross-icd-invalid",
                             f"{relative} {fixture} did not run on two ICDs")
                    documents_ok = False
                    continue
                for icd_name, run in device_runs.items():
                    device = run.get("device", {})
                    devices_seen.add(str(device.get("device_name")))
                    for branch in ("auto", "explicit_ordinal_0"):
                        branch_record = run.get(branch, {})
                        if (int(branch_record.get("result", {})
                                .get("run_status", -1)) != 0 or
                                branch_record.get("numeric") is None):
                            _finding(findings, "va6-cross-icd-invalid",
                                     f"{relative} {fixture}/{icd_name} "
                                     f"{branch} run did not succeed")
                            documents_ok = False
                    invalid = run.get("invalid_ordinal", {}).get("result", {})
                    if (int(invalid.get("create_status", -1)) != 2 or
                            invalid.get("context_null") is not True):
                        _finding(findings, "va6-cross-icd-invalid",
                                 f"{relative} {fixture}/{icd_name} invalid "
                                 "ordinal did not fail with UNSUPPORTED_DEVICE")
                        documents_ok = False
                if len(devices_seen) < 2:
                    _finding(findings, "va6-cross-icd-invalid",
                             f"{relative} {fixture} ran on fewer than two "
                             "distinct physical devices")
                    documents_ok = False
                no_icd = scenario.get("no_icd", {}).get("result", {})
                if (int(no_icd.get("create_status", -1)) not in (2, 5) or
                        no_icd.get("context_null") is not True or
                        int(no_icd.get("destroy_null_status", -1)) != 0):
                    _finding(findings, "va6-cross-icd-invalid",
                             f"{relative} {fixture} no-ICD create failure is "
                             "not explicit")
                    documents_ok = False
            cross_icd_document = document
    if (matrix_document is not None and faults_document is not None and
            cross_icd_document is not None):
        # The codegen identity is derived per model from its own host
        # program, so documents pin different values by construction.  The
        # checkable invariants are: every published positive maps its model
        # to one well-formed identity (one-to-one), and the faults evidence
        # pins a well-formed identity for its own fixture.
        published = {
            case.get("model"): case.get("host_codegen_identity")
            for case in matrix_document.get("cases", {}).values()
            if case.get("kind") == "positive"}
        identity_pattern = re.compile(r"^[0-9a-f]{64}$")
        if (len(set(published.values())) != len(published) or
                any(not isinstance(value, str) or
                    not identity_pattern.fullmatch(value)
                    for value in published.values()) or
                not isinstance(faults_document.get("host_codegen_identity"),
                               str) or
                not identity_pattern.fullmatch(
                    faults_document["host_codegen_identity"])):
            _finding(findings, "va6-identity-mismatch",
                     "VA6 evidence does not pin well-formed per-model "
                     "codegen identities")
            documents_ok = False
    if documents_ok and not missing_required and not not_passed:
        return {"status": "verified", "stage_id": "VA6", "role": "matrix",
                "evidence": [str(path) for path in paths],
                "host_aot_release_eligible": False}
    return {"status": "blocked", "stage_id": "VA6", "role": "matrix",
            "evidence": [str(path) for path in paths],
            "host_aot_release_eligible": False}


def audit_stage_evidence(gate: dict) -> dict:
    """Adapt this fresh stage's existing V0–V5 and offline schemas."""
    stage = Path(gate["stage"]).resolve()
    profile = gate.get("profile")
    findings = list(gate.get("findings", []))
    refs = []
    devices = _devices_for_profile(profile)
    cpu_baseline = _audit_cpu_baseline(stage, refs, findings)
    offline = _audit_offline(stage, refs, findings)
    families = {"cpu_baseline": cpu_baseline, "offline": offline}
    if profile != "offline":
        families["v0"] = _audit_v0(stage, devices, refs, findings)
        families["v2"] = _audit_v2(stage, devices, refs, findings)
        families["v3"] = _audit_v3(stage, devices, refs, findings)
        families["v4"] = _audit_product(stage, devices, refs, findings)
        families["v5"] = _audit_v5(stage, devices, refs, findings)
    va0_findings = []
    families["va0"] = _audit_va0_baseline(
        stage, gate, families, refs, va0_findings)
    va1_findings = []
    families["va1"] = _audit_va1_stage(
        gate, families, refs, va1_findings)
    findings.extend(va1_findings)
    families["va2_helper"] = _audit_va2_helper(stage, gate, refs, findings)
    families["va3_lifecycle"] = _audit_va3_lifecycle(stage, gate, refs, findings)
    families["va4_fixedrun"] = _audit_va4_fixedrun(stage, gate, refs, findings)
    va5_findings = []
    families["va5_deployment"] = _audit_va5_deployment(
        stage, gate, refs, va5_findings)
    findings.extend(va5_findings)
    va6_findings = []
    families["va6_matrix"] = _audit_va6_matrix(stage, gate, refs, va6_findings)
    findings.extend(va6_findings)
    compact = {}
    for name, data in families.items():
        if name == "cpu_baseline":
            compact[name] = ({"path": "test/Native/vulkan-v1-cpu-baseline/evidence.json",
                              "fixture": data.get("fixture"),
                              "cpu_library_sha256": data.get("cpu_library_sha256"),
                              "artifact_sha256": data.get("artifact_sha256"),
                              "input_sha256": data.get("input_sha256"),
                              "output_sha256": data.get("output_sha256")} if isinstance(data, dict) else None)
        elif name == "offline":
            compact[name] = ({"path": data.get("path"), "gpu_numeric": False,
                              "fixture": data.get("fixture", {}).get("model"),
                              "negative_cases": data.get("negative_cases", [])}
                             if isinstance(data, dict) else None)
        elif name == "v0":
            compact[name] = {key: {"path": value.get("path"),
                                   "cpu_library_sha256": value.get("metadata", {}).get("cpu_reference_library_sha256"),
                                   "input_sha256": value.get("numeric", {}).get("input_sha256"),
                                   "output_sha256": value.get("numeric", {}).get("output_sha256"),
                                   "device": value.get("identity")}
                             for key, value in data.items()} if isinstance(data, dict) else {}
        elif name == "v2":
            compact[name] = {key: {"fixture_id": value.get("identity", {}).get("fixture_id"),
                                   "cpu_library_sha256": value.get("identity", {}).get("cpu_reference_library_sha256"),
                                   "operation": value.get("operation"),
                                   "seeds": {seed: {"path": record.get("path"),
                                                     "input_sha256": record.get("numeric", {}).get("input_sha256"),
                                                     "output_sha256": record.get("numeric", {}).get("output_sha256")}
                                             for seed, record in value.get("seeds", {}).items()}}
                             for key, value in data.items()} if isinstance(data, dict) else {}
        elif name == "v3":
            compact[name] = {key: {"path": value.get("path"),
                                   "cpu_library_sha256": value.get("metadata", {}).get("cpu_library_sha256"),
                                   "conv_cpu_library_sha256": value.get("metadata", {}).get("conv_cpu_library_sha256"),
                                   "device": value.get("identity"),
                                   "peak_memory_bytes": value.get("weight", {}).get("peak_memory_bytes"),
                                   "cpu_gpu_golden": value.get("numeric", {}).get("metrics")}
                             for key, value in data.items()} if isinstance(data, dict) else {}
        elif name == "v4":
            compact[name] = {key: {"path": value.get("path"),
                                   "cpu_library_sha256": value.get("fixture", {}).get("cpu_library_sha256"),
                                   "device": value.get("identity"),
                                   "numeric": value.get("numeric")}
                             for key, value in data.items()} if isinstance(data, dict) else {}
        elif name == "v5":
            compact[name] = {device: {model: {"path": record.get("path"),
                                              "cpu_library_sha256": record.get("cpu_library_sha256"),
                                              "device": record.get("identity"),
                                              "trial_count": len(record.get("trials", [])),
                                              "peak_memory_bytes": record.get("peak_memory_bytes"),
                                              "planned_resource_bytes": record.get("planned_resource_bytes")}
                                      for model, record in models.items()}
                             for device, models in data.items()} if isinstance(data, dict) else {}
        elif name == "va0":
            reference = data.get("legacy_reference") if isinstance(data, dict) else None
            compact[name] = {
                "legacy_reference": ({"status": reference.get("status"),
                                      "identity": reference.get("identity"),
                                      "fixtures": {key: value.get("path")
                                                   for key, value in reference.get("fixtures", {}).items()},
                                      "snapshot_path": reference.get("snapshot_path")}
                                     if isinstance(reference, dict) else None),
                "comparisons": data.get("comparisons", {}) if isinstance(data, dict) else {},
                "shader_comparisons": data.get("shader_comparisons", []) if isinstance(data, dict) else [],
                "dependencies": data.get("dependencies", {}) if isinstance(data, dict) else {},
                "current_host_contract_verified": data.get("current_host_contract_verified", False) if isinstance(data, dict) else False,
                "migration_coverage": data.get("migration_coverage", {}) if isinstance(data, dict) else {},
                "findings": va0_findings,
                "host_aot_release_eligible": False,
                "host_aot_status": "not-implemented-va1-and-later",
                "scope": "legacy generic-plan-runtime only; no current host AOT/no-LLVM claim",
            }
        elif name == "va1":
            compact[name] = data
        elif name == "va2_helper":
            compact[name] = data
        elif name == "va3_lifecycle":
            compact[name] = data
        elif name == "va4_fixedrun":
            compact[name] = {
                "status": data.get("status"),
                "stage_id": data.get("stage_id"),
                "role": data.get("role"),
                "legacy_reference_qualified": data.get("legacy_reference_qualified"),
                "path": data.get("path"),
                "sha256": data.get("sha256"),
                "source_fingerprint": data.get("source_fingerprint"),
                "candidate_dso_sha256": data.get("candidate_dso_sha256"),
                "devices": data.get("devices"),
                "host_aot_release_eligible": False,
                "raw_artifacts": data.get("raw_artifacts", []),
                "claim_scope": data.get("claim_scope"),
            }
        elif name == "va5_deployment":
            compact[name] = {
                "status": data.get("status"),
                "stage_id": data.get("stage_id"),
                "role": data.get("role"),
                "deployment_evidence": data.get("deployment_evidence", []),
                "host_aot_release_eligible": False,
            }
        elif name == "va6_matrix":
            compact[name] = {
                "status": data.get("status"),
                "stage_id": data.get("stage_id"),
                "role": data.get("role"),
                "evidence": data.get("evidence", []),
                "host_aot_release_eligible": False,
            }
    gate_state = gate.get("state", {})
    for label, key in (("gate-state", None), ("CMakeCache", "cmake_cache"),
                       ("CTest inventory", "inventory"), ("CTest JUnit", "junit"),
                       ("CTest LastTest", "lasttest")):
        record = {"path": "gate-state.json"} if key is None else gate_state.get(key)
        if isinstance(record, dict):
            artifact = _evidence_file(stage, stage, record.get("path"),
                                     f"{label}", refs, findings)
            if key is not None:
                _check_digest_claim(record.get("sha256"), artifact, label, findings)
    for step in gate_state.get("steps", []):
        if isinstance(step, dict):
            log_path = _evidence_file(stage, stage, step.get("log"),
                                      f"gate {step.get('name')} log", refs, findings)
            _check_digest_claim(step.get("sha256"), log_path,
                                f"gate {step.get('name')} log", findings)
    for input_item in gate.get("source_inputs", []):
        if isinstance(input_item, dict):
            source_path = _evidence_file(stage, stage, input_item.get("path"),
                                         f"fresh source input {input_item.get('source', input_item.get('generated'))}",
                                         refs, findings)
            _check_digest_claim(input_item.get("sha256"), source_path,
                                f"fresh source input {input_item.get('source', input_item.get('generated'))}",
                                findings)
    for item in gate.get("built_tool_files", []):
        if isinstance(item, dict):
            tool_path = _evidence_file(stage, stage, item.get("path"),
                                       f"fresh built tool {item.get('path')}", refs, findings)
            _check_digest_claim(item.get("sha256"), tool_path,
                                f"fresh built tool {item.get('path')}", findings)
    for item in gate.get("sanitizer_toolchain_files", []):
        if isinstance(item, dict):
            toolchain_path = _evidence_file(stage, stage, item.get("path"),
                                            f"SAN toolchain bundle {item.get('path')}", refs, findings)
            _check_digest_claim(item.get("sha256"), toolchain_path,
                                f"SAN toolchain bundle {item.get('path')}", findings)
    ctest_step = next((item for item in gate_state.get("steps", [])
                       if isinstance(item, dict) and item.get("name") == "ctest"), {})
    ctest_start, ctest_end = parse_time(ctest_step.get("started_at")), parse_time(ctest_step.get("finished_at"))
    if ctest_start is None or ctest_end is None:
        _finding(findings, "evidence-freshness-time", "cannot validate CTest evidence creation interval")
    else:
        for ref in refs:
            relative = str(ref.get("path", ""))
            is_test_output = relative.startswith((
                "test/Vulkan/evidence/", "test/Native/vulkan-v1-cpu-baseline/",
                "test/Native/vulkan-product-offline/", "test/Native/vulkan-product-cli"))
            role = str(ref.get("role", "")).lower()
            derived_aggregates = {
                "test/Vulkan/evidence/va3/lifecycle-evidence.json":
                    "va3 post-ctest lifecycle aggregate",
                "test/Vulkan/evidence/va4/fixedrun-evidence.json":
                    "va4 post-ctest fixed-run aggregate",
            }
            if derived_aggregates.get(relative) == role:
                continue
            # Model source snapshots and copied CMake-built baseline artifacts preserve
            # source/build mtimes; their byte hashes and current source/build identities
            # are checked separately.
            source_copy = ((Path(relative).suffix in {".param", ".bin"} and
                            any(word in role for word in ("model", "source", "fixture"))) or
                           "/artifacts-" in relative)
            if not is_test_output or source_copy:
                continue
            path = safe_stage_file(stage, relative)
            if path and path.is_file():
                modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                if (modified.timestamp() < ctest_start.timestamp() - 2.0 or
                        modified.timestamp() > ctest_end.timestamp() + 2.0):
                    _finding(findings, "stale-test-evidence",
                             "stage-local test evidence was not created during this fresh CTest run",
                             path=relative, mtime=modified.isoformat(),
                             ctest_started=ctest_start.isoformat(), ctest_finished=ctest_end.isoformat())
    scope_passed = not findings
    return {"stage": gate.get("stage"), "profile": profile,
            "stage_id": gate.get("stage_id"),
            "source_fingerprint": gate.get("source_fingerprint"),
            "git_revision": gate.get("git_revision"),
            "completed": gate.get("completed"), "scope_passed": scope_passed,
            "findings": findings, "va0_findings": va0_findings,
            "families": compact,
            "raw_artifacts": refs, "ctest": gate.get("ctest"),
            "source_inputs": gate.get("source_inputs", [])}


def _family_paths(family: str, families: dict) -> list[str]:
    prefix = {"cpu": "test/Native/vulkan-v1-cpu-baseline/",
              "v0": "test/Vulkan/evidence/gpu-func/",
              "v2": "test/Vulkan/evidence/v2/",
              "v3": "test/Vulkan/evidence/v3/",
              "v4": "test/Native/vulkan-product-cli",
              "v5.1": "test/Vulkan/evidence/v5/",
              "v5.2": "test/Vulkan/evidence/v5/",
              "v5.3": "test/Vulkan/evidence/v5/",
              "v5.4": "test/Vulkan/evidence/v5/",
              "gate": "gate-", "index": "", "decision": "",
              "v6.1": "test/Native/vulkan-product-offline/"}.get(family, "")
    if family == "cpu":
        return [str(item.get("path")) for item in families.get("raw_artifacts", [])
                if "vulkan-v1-cpu-baseline" in str(item.get("path"))]
    return [str(item.get("path")) for item in families.get("raw_artifacts", [])
            if prefix and prefix in str(item.get("path"))]


def _collect_cpu_identities(value: object) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "cpu_library_sha256" or key == "conv_cpu_library_sha256":
                if isinstance(item, str) and SHA256_RE.fullmatch(item):
                    found.append(item)
            else:
                found.extend(_collect_cpu_identities(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_cpu_identities(item))
    return sorted(set(found))


def _profile_tests(stage_audits: list[dict], patterns: list[str]) -> dict[str, list[str]]:
    output = {}
    for stage in stage_audits:
        names = set(stage.get("ctest", {}).get("test_names", []))
        selected = sorted(name for name in patterns if name in names)
        if selected:
            output[stage.get("profile", "unknown")] = selected
    return output


def _applicable_tests(patterns: list[str], profile: str) -> set[str]:
    devices = _devices_for_profile(profile)
    selected = set()
    for name in patterns:
        if name.startswith(("vulkan-v0-", "vulkan-v2-", "vulkan-v3-", "vulkan-v5-")):
            if profile == "offline":
                continue
            if name.endswith("-nvidia") and "nvidia" not in devices:
                continue
            if name.endswith("-llvmpipe") and "llvmpipe" not in devices:
                continue
        if name == "ncnn-compile-vulkan-product-llvmpipe" and (
                profile not in {"release", "sanitize"} or "llvmpipe" not in devices):
            continue
        if name == "ncnn-compile-vulkan-product" and not devices:
            continue
        selected.add(name)
    return selected


def _id_ref_in_scope(identifier: str, family: str, path: str) -> bool:
    if family.startswith("v2-"):
        match = re.search(r"evidence/v2/([^/]+)/", path)
        if not match:
            return False
        case = match.group(1).removesuffix("-nvidia").removesuffix("-llvmpipe")
        if family == "v2-elementwise":
            return case in {"add", "relu", "copy", "identity", "transpose", "reshape"}
        if family == "v2-conv":
            return case in {"conv3x3", "conv3x3-no-bias", "conv3x3-stride2", "conv1x1"}
        return True
    if family.startswith("v5."):
        if "evidence/v5/" not in path:
            return False
        wanted = {"v5.1": "multiop", "v5.2": "pooling-softmax",
                  "v5.3": "squeezenet", "v5.4": "squeezenet"}.get(family)
        return wanted is not None and wanted in path
    return True


def _scoped_family_identity(family: str, data: object) -> object:
    if family.startswith("v2-") and isinstance(data, dict):
        if family == "v2-elementwise":
            allowed = {"add", "relu", "copy", "identity", "transpose", "reshape"}
        elif family == "v2-conv":
            allowed = {"conv3x3", "conv3x3-no-bias", "conv3x3-stride2", "conv1x1"}
        else:
            return data
        return {key: value for key, value in data.items()
                if key.removesuffix("-nvidia").removesuffix("-llvmpipe") in allowed}
    if family.startswith("v5.") and isinstance(data, dict):
        suite = {"v5.1": "multiop", "v5.2": "pooling-softmax",
                 "v5.3": "squeezenet", "v5.4": "squeezenet"}.get(family)
        return {device: models for device, models in data.items()
                if suite and suite in device}
    return data


def _va0_id_row(identifier: str, spec: dict, stage_audits: list[dict]) -> dict:
    profile_rows = []
    selected_refs = []
    all_executed = set()
    all_missing = set()
    for stage in stage_audits:
        profile = stage.get("profile", "unknown")
        names = set(stage.get("ctest", {}).get("test_names", []))
        applicable = _applicable_tests(spec["tests"], profile)
        executed = sorted(applicable & names)
        missing = sorted(applicable - names)
        all_executed.update(executed)
        all_missing.update(missing)
        va0 = stage.get("families", {}).get("va0", {})
        reference = va0.get("legacy_reference") if isinstance(va0, dict) else None
        baseline_valid = (isinstance(reference, dict) and
                          reference.get("status") == "verified")
        comparisons = va0.get("comparisons", {}) if isinstance(va0, dict) else {}
        if identifier == "VA0.1":
            device_rows = comparisons.get("add_relu", {})
            device_rows = [*device_rows.values(),
                           *comparisons.get("conv_relu", {}).values(),
                           *comparisons.get("squeezenet_v1_1", {}).values()]
            numeric_ok = all(value is True for value in device_rows)
            product_rows = comparisons.get("relu-product", {})
            product_ok = bool(product_rows) and all(value is True for value in product_rows.values())
            scope_evidence_ok = numeric_ok and product_ok
        elif identifier == "VA0.2":
            dependency = va0.get("dependencies") if isinstance(va0, dict) else None
            scope_evidence_ok = isinstance(dependency, dict) and dependency.get("status") == "verified"
        elif identifier == "VA0.3":
            scope_evidence_ok = bool(va0.get("current_host_contract_verified")) if isinstance(va0, dict) else False
        else:
            coverage = va0.get("migration_coverage", {}) if isinstance(va0, dict) else {}
            scope_evidence_ok = (
                coverage.get("migration_status") == "pending" and
                coverage.get("legacy_pass_is_migration_pass") is False and
                not coverage.get("unexecuted_related_tests") and
                set(OFFLINE_HOST_TESTS).issubset(names) and baseline_valid)
        raw = stage.get("raw_artifacts", [])
        def select(predicate):
            return [{key: item[key] for key in ("path", "sha256", "role") if key in item}
                    for item in raw if isinstance(item, dict) and predicate(item)]
        current_cpu = select(lambda item: "vulkan-v1-cpu-baseline" in str(item.get("path", ""))
                              or " CPU" in str(item.get("role", "")))
        current_product = select(lambda item: "product" in str(item.get("role", "")).lower()
                                 or "product-cli" in str(item.get("path", ""))
                                 or "vulkan-product-offline" in str(item.get("path", "")))
        old_reference = select(lambda item: str(item.get("role", "")).startswith("legacy-reference "))
        small_golden = select(lambda item: "golden" in str(item.get("role", "")).lower()
                              or "golden" in str(item.get("path", "")).lower())
        if identifier == "VA0.2":
            current_cpu = select(lambda item: "dependency" in str(item.get("role", "")).lower()
                                 or "dependency" in str(item.get("path", "")).lower())
        elif identifier == "VA0.3":
            current_cpu = select(lambda item: any(token in str(item.get("role", "")).lower()
                                                  for token in ("manifest", "plan", "header", "product")))
        elif identifier == "VA0.4":
            current_cpu = select(lambda item: any(token in str(item.get("role", "")).lower()
                                                  for token in ("gate-state", "ctest", "no-device", "product"))
                                 or str(item.get("path", "")).startswith(("gate-", "test/gate-tests.json")))
        refs_complete = bool(old_reference)
        if identifier == "VA0.1":
            refs_complete = refs_complete and bool(current_cpu) and bool(current_product) and bool(small_golden)
        elif identifier in {"VA0.2", "VA0.3"}:
            refs_complete = refs_complete and bool(current_cpu)
        else:
            refs_complete = refs_complete and bool(current_cpu) and bool(current_product)
        status = ("passed_for_profile_scope" if baseline_valid and
                  stage.get("scope_passed") and not stage.get("va0_findings") and
                  not missing and scope_evidence_ok and refs_complete
                  else "blocked")
        row = {"profile": profile, "tests_executed": executed,
               "tests_missing": missing, "status": status,
               "gpu_numeric_claim": "not-claimed" if profile == "offline" else spec["gpu_numeric"],
               "current_cpu_or_audit_refs": current_cpu,
               "current_legacy_product_refs": current_product,
               "legacy_reference_refs": old_reference,
               "small_graph_golden_refs": small_golden,
               "comparison": comparisons,
               "host_aot_candidate": False}
        profile_rows.append(row)
        selected_refs.extend(current_cpu + current_product + old_reference + small_golden)
    overall = ("passed_for_profile_scope" if profile_rows and
               all(row["status"] == "passed_for_profile_scope" for row in profile_rows)
               else "blocked")
    unique_refs = {}
    for item in selected_refs:
        unique_refs[(item.get("path"), item.get("sha256"), item.get("role"))] = item
    migration = {}
    for stage in stage_audits:
        va0 = stage.get("families", {}).get("va0", {})
        coverage = va0.get("migration_coverage", {}) if isinstance(va0, dict) else {}
        if coverage:
            migration[stage.get("profile", "unknown")] = {
                "migration_status": "pending",
                "legacy_pass_is_migration_pass": False,
                "future_responsibilities": coverage.get("responsibilities", {}),
                "current_ids": coverage.get("current_ids", {}),
                "current_test_cases": coverage.get("current_test_cases", {}),
                "unexecuted_related_tests": coverage.get("unexecuted_related_tests", []),
            }
    return {
        "cpu_positive": {"fixture": spec["cpu_fixture"],
                         "fresh_cpu_baseline": [
                             {"stage": stage.get("stage"),
                              "path": stage.get("families", {}).get("cpu_baseline", {}).get("path"),
                              "input_sha256": stage.get("families", {}).get("cpu_baseline", {}).get("input_sha256"),
                              "output_sha256": stage.get("families", {}).get("cpu_baseline", {}).get("output_sha256")}
                             for stage in stage_audits
                             if stage.get("families", {}).get("cpu_baseline")]},
        "related_tests": spec["tests"],
        "tests_executed": sorted(all_executed),
        "tests_missing_from_supplied_stages": sorted(set(spec["tests"]) - all_executed),
        "profiles": profile_rows,
        "migration_coverage_by_profile": migration,
        "evidence_family": "va0-legacy-baseline",
        "evidence_profiles": sorted({row["profile"] for row in profile_rows}),
        "gpu_numerical_claim": spec["gpu_numeric"],
        "raw_artifacts": list(unique_refs.values()),
        "status": overall,
        "release_status": "not_eligible",
        "release_eligibility_scope": "VA0 baseline only; host AOT eligibility requires VA1 and later",
        "host_aot_release_eligible": False,
    }


def _va1_id_row(identifier: str, spec: dict, stage_audits: list[dict]) -> dict:
    target_profiles = ("release", "sanitize")
    stages = [stage for stage in stage_audits
              if stage.get("stage_id") == "VA1" and
              stage.get("profile") in target_profiles]
    tests = set(spec["tests"])
    executed = set().union(*(set(stage.get("ctest", {}).get("test_names", []))
                             for stage in stages)) if stages else set()
    missing_tests = sorted(tests - executed)
    profile_rows = []
    selected_refs = {}
    profile_valid = {}
    for profile in target_profiles:
        matches = [stage for stage in stages if stage.get("profile") == profile]
        if len(matches) != 1:
            profile_valid[profile] = False
            profile_rows.append({"profile": profile, "status": "missing" if not matches else "duplicate",
                                 "stages": [stage.get("stage") for stage in matches]})
            continue
        stage = matches[0]
        names = set(stage.get("ctest", {}).get("test_names", []))
        va1 = stage.get("families", {}).get("va1", {})
        missing = sorted(tests - names)
        valid = (stage.get("scope_passed") is True and
                 va1.get("status") == "verified" and not missing and
                 stage.get("archive_integrity", {}).get("status") == "verified")
        profile_valid[profile] = valid
        profile_rows.append({"profile": profile,
                             "stage": stage.get("stage"),
                             "source_fingerprint": stage.get("source_fingerprint"),
                             "status": "passed" if valid else "blocked",
                             "tests_executed": sorted(names & tests),
                             "tests_missing": missing,
                             "host_program_evidence": va1.get("host_program_evidence", [])})
        for ref in stage.get("raw_artifacts", []):
            path = str(ref.get("path", ""))
            role = str(ref.get("role", ""))
            if ("host-program" in path.lower() or "host-program" in role.lower() or
                    "test/Vulkan/evidence/v5/" in path):
                key = (stage.get("stage"), path)
                selected_refs[key] = {"stage": stage.get("stage"),
                                      **{field: ref[field] for field in ("path", "sha256", "role")
                                         if field in ref}}
    fingerprints = {stage.get("source_fingerprint") for stage in stages
                    if stage.get("source_fingerprint")}
    passed = (all(profile_valid.get(profile) is True for profile in target_profiles) and
              len(fingerprints) == 1)
    return {"cpu_positive": {"fixture": spec["cpu_fixture"],
                             "fresh_cpu_baseline": []},
            "related_tests": spec["tests"],
            "tests_executed": sorted(tests & executed),
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(tests),
            "tests_missing_for_supplied_profile_scopes": missing_tests,
            "profiles": profile_rows,
            "evidence_family": "va1-compiler-only-planner",
            "evidence_profiles": list(target_profiles),
            "gpu_numerical_claim": spec["gpu_numeric"],
            "raw_artifacts": list(selected_refs.values()),
            "status": "passed" if passed else "blocked",
            "release_status": "not_eligible",
            "release_eligibility_scope": "VA1 planner only; not model-specific host AOT",
            "host_aot_release_eligible": False,
            "source_fingerprints": sorted(fingerprints)}


def _va2_id_row(identifier: str, spec: dict,
                stage_audits: list[dict]) -> dict:
    target_profiles = ("release", "sanitize")
    stages = [stage for stage in stage_audits
              if stage.get("stage_id") == "VA2" and
              stage.get("profile") in target_profiles]
    tests = set(spec["tests"])
    profile_rows = []
    selected_refs = {}
    profile_valid = {}
    for profile in target_profiles:
        matches = [stage for stage in stages if stage.get("profile") == profile]
        if len(matches) != 1:
            profile_valid[profile] = False
            profile_rows.append({"profile": profile,
                                 "status": "missing" if not matches else "duplicate",
                                 "stages": [stage.get("stage") for stage in matches]})
            continue
        stage = matches[0]
        names = set(stage.get("ctest", {}).get("test_names", []))
        family = stage.get("families", {}).get("va2_helper", {})
        missing = sorted(tests - names)
        valid = (stage.get("scope_passed") is True and
                 family.get("status") == "verified" and not missing and
                 stage.get("archive_integrity", {}).get("status") == "verified")
        profile_valid[profile] = valid
        profile_rows.append({"profile": profile, "stage": stage.get("stage"),
                             "source_fingerprint": stage.get("source_fingerprint"),
                             "status": "passed" if valid else "blocked",
                             "tests_executed": sorted(names & tests),
                             "tests_missing": missing,
                             "helper_evidence": family.get("independence_evidence")})
        for ref in stage.get("raw_artifacts", []):
            path = str(ref.get("path", ""))
            if path.startswith("test/Vulkan/evidence/va2/"):
                key = (stage.get("stage"), path)
                selected_refs[key] = {"stage": stage.get("stage"),
                                      **{field: ref[field] for field in
                                         ("path", "sha256", "role") if field in ref}}
    fingerprints = {stage.get("source_fingerprint") for stage in stages
                    if isinstance(stage.get("source_fingerprint"), str)}
    passed = (all(profile_valid.get(profile) is True for profile in target_profiles) and
              len(fingerprints) == 1)
    missing_tests = sorted(tests - set().union(*(
        set(stage.get("ctest", {}).get("test_names", [])) for stage in stages))) \
        if stages else sorted(tests)
    return {"cpu_positive": {"fixture": spec["cpu_fixture"], "fresh_cpu_baseline": []},
            "related_tests": spec["tests"],
            "tests_executed": sorted(tests - set(missing_tests)),
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(tests),
            "tests_missing_for_supplied_profile_scopes": missing_tests,
            "profiles": profile_rows, "evidence_family": "va2-device-helper",
            "evidence_profiles": list(target_profiles),
            "gpu_numerical_claim": spec["gpu_numeric"],
            "raw_artifacts": list(selected_refs.values()),
            "status": "passed" if passed else "blocked",
            "release_status": "not_eligible",
            "release_eligibility_scope": "VA2 helper only; no VA3 model-specific host AOT",
            "host_aot_release_eligible": False,
            "source_fingerprints": sorted(fingerprints)}


def _va3_id_row(identifier: str, spec: dict,
                stage_audits: list[dict]) -> dict:
    target_profiles = ("release", "sanitize")
    stages = [stage for stage in stage_audits
              if stage.get("stage_id") == "VA3" and
              stage.get("profile") in target_profiles]
    tests = set(spec["tests"])
    profile_rows = []
    selected_refs = {}
    profile_valid = {}
    executed = set()
    for profile in target_profiles:
        matches = [stage for stage in stages if stage.get("profile") == profile]
        if len(matches) != 1:
            profile_valid[profile] = False
            profile_rows.append({"profile": profile,
                                 "status": "missing" if not matches else "duplicate",
                                 "stages": [stage.get("stage") for stage in matches]})
            continue
        stage = matches[0]
        ctest = stage.get("ctest", {})
        names = set(ctest.get("test_names", []))
        skipped = set(ctest.get("skipped", []))
        failed = set(ctest.get("failed", []))
        passed_tests = names - skipped - failed
        executed.update(passed_tests)
        family = stage.get("families", {}).get("va3_lifecycle", {})
        missing = sorted(tests - names)
        not_passed = sorted(tests & (skipped | failed))
        valid = (stage.get("scope_passed") is True and
                 family.get("status") == "verified" and not missing and
                 not not_passed and
                 stage.get("archive_integrity", {}).get("status") == "verified")
        profile_valid[profile] = valid
        profile_rows.append({"profile": profile, "stage": stage.get("stage"),
                             "source_fingerprint": stage.get("source_fingerprint"),
                             "status": "passed" if valid else "blocked",
                             "tests_executed": sorted(passed_tests & tests),
                             "tests_missing": missing,
                             "tests_not_passed": not_passed,
                             "lifecycle_evidence": family.get("path"),
                             "candidate_run": "not_run"})
        for ref in stage.get("raw_artifacts", []):
            path = str(ref.get("path", ""))
            if path.startswith("test/Vulkan/evidence/va3/"):
                key = (stage.get("stage"), path)
                selected_refs[key] = {"stage": stage.get("stage"),
                                      **{field: ref[field] for field in
                                         ("path", "sha256", "role") if field in ref}}
    fingerprints = {stage.get("source_fingerprint") for stage in stages
                    if isinstance(stage.get("source_fingerprint"), str)}
    missing_tests = sorted(tests - executed)
    passed = (all(profile_valid.get(profile) is True for profile in target_profiles) and
              len(fingerprints) == 1)
    return {"cpu_positive": {"fixture": spec["cpu_fixture"],
                             "fresh_cpu_baseline": []},
            "related_tests": spec["tests"],
            "tests_executed": sorted(tests & executed),
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(tests),
            "tests_missing_for_supplied_profile_scopes": missing_tests,
            "profiles": profile_rows,
            "evidence_family": "va3-lifecycle-only-candidate",
            "evidence_profiles": list(target_profiles),
            "gpu_numerical_claim": "create-destroy-only-no-model-run",
            "raw_artifacts": list(selected_refs.values()),
            "status": "passed" if passed else "blocked",
            "release_status": "not_eligible",
            "release_eligibility_scope": "private lifecycle candidate only; no model run/release product",
            "host_aot_release_eligible": False,
            "source_fingerprints": sorted(fingerprints)}


def _va4_id_row(identifier: str, spec: dict,
                stage_audits: list[dict]) -> dict:
    target_profiles = ("release", "sanitize")
    stages = [stage for stage in stage_audits
              if stage.get("stage_id") == "VA4" and
              stage.get("profile") in target_profiles]
    tests = set(spec["tests"])
    if identifier == "VA4.1":
        tests.update(_va4_codegen_tests())
    profile_rows = []
    selected_refs = {}
    profile_valid = {}
    executed = set()
    for profile in target_profiles:
        matches = [stage for stage in stages if stage.get("profile") == profile]
        if len(matches) != 1:
            profile_valid[profile] = False
            profile_rows.append({"profile": profile,
                                 "status": "missing" if not matches else "duplicate",
                                 "stages": [stage.get("stage") for stage in matches]})
            continue
        stage = matches[0]
        ctest = stage.get("ctest", {})
        names = set(ctest.get("test_names", []))
        skipped = set(ctest.get("skipped", []))
        failed = set(ctest.get("failed", []))
        passed_tests = names - skipped - failed
        executed.update(passed_tests)
        family = stage.get("families", {}).get("va4_fixedrun", {})
        missing = sorted(tests - names)
        not_passed = sorted(tests & (skipped | failed))
        valid = (stage.get("scope_passed") is True and
                 family.get("status") == "verified" and not missing and
                 not not_passed and
                 stage.get("archive_integrity", {}).get("status") == "verified")
        profile_valid[profile] = valid
        profile_rows.append({"profile": profile, "stage": stage.get("stage"),
                             "source_fingerprint": stage.get("source_fingerprint"),
                             "status": "passed" if valid else "blocked",
                             "tests_executed": sorted(passed_tests & tests),
                             "tests_missing": missing,
                             "tests_not_passed": not_passed,
                             "fixedrun_evidence": family.get("path"),
                             "candidate_run": family.get("candidate_dso_sha256")})
        for ref in stage.get("raw_artifacts", []):
            path = str(ref.get("path", ""))
            if path.startswith("test/Vulkan/evidence/va4/"):
                key = (stage.get("stage"), path)
                selected_refs[key] = {"stage": stage.get("stage"),
                                      **{field: ref[field] for field in
                                         ("path", "sha256", "role") if field in ref}}
    fingerprints = {stage.get("source_fingerprint") for stage in stages
                    if isinstance(stage.get("source_fingerprint"), str)}
    missing_tests = sorted(tests - executed)
    passed = (all(profile_valid.get(profile) is True for profile in target_profiles) and
              len(fingerprints) == 1)
    return {"cpu_positive": {"fixture": spec["cpu_fixture"],
                             "fresh_cpu_baseline": []},
            "related_tests": sorted(tests),
            "tests_executed": sorted(tests & executed),
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(tests),
            "tests_missing_for_supplied_profile_scopes": missing_tests,
            "profiles": profile_rows,
            "evidence_family": "va4-model-specific-fixed-run-candidate",
            "evidence_profiles": list(target_profiles),
            "gpu_numerical_claim": spec["gpu_numeric"],
            "raw_artifacts": list(selected_refs.values()),
            "status": "passed" if passed else "blocked",
            "release_status": "not_eligible",
            "release_eligibility_scope": "private fixed-run small-graph prototype; VA5-VA7 remain incomplete",
            "host_aot_release_eligible": False,
            "source_fingerprints": sorted(fingerprints)}


def _va5_id_row(identifier: str, spec: dict,
                stage_audits: list[dict]) -> dict:
    target_profiles = ("release", "sanitize")
    stages = [stage for stage in stage_audits
              if stage.get("stage_id") == "VA5" and
              stage.get("profile") in target_profiles]
    tests = set(spec["tests"])
    profile_rows = []
    selected_refs = {}
    profile_valid = {}
    executed = set()
    for profile in target_profiles:
        matches = [stage for stage in stages if stage.get("profile") == profile]
        if len(matches) != 1:
            profile_valid[profile] = False
            profile_rows.append({"profile": profile,
                                 "status": "missing" if not matches else "duplicate",
                                 "stages": [stage.get("stage") for stage in matches]})
            continue
        stage = matches[0]
        ctest = stage.get("ctest", {})
        names = set(ctest.get("test_names", []))
        skipped = set(ctest.get("skipped", []))
        failed = set(ctest.get("failed", []))
        passed_tests = names - skipped - failed
        executed.update(passed_tests)
        family = stage.get("families", {}).get("va5_deployment", {})
        missing = sorted(tests - names)
        not_passed = sorted(tests & (skipped | failed))
        valid = (stage.get("scope_passed") is True and
                 family.get("status") == "verified" and not missing and
                 not not_passed and
                 stage.get("archive_integrity", {}).get("status") == "verified")
        profile_valid[profile] = valid
        profile_rows.append({"profile": profile, "stage": stage.get("stage"),
                             "source_fingerprint": stage.get("source_fingerprint"),
                             "status": "passed" if valid else "blocked",
                             "tests_executed": sorted(passed_tests & tests),
                             "tests_missing": missing,
                             "tests_not_passed": not_passed,
                             "deployment_evidence": family.get("deployment_evidence", [])})
        for ref in stage.get("raw_artifacts", []):
            path = str(ref.get("path", ""))
            if path.startswith("test/Vulkan/evidence/va5/") or \
                    path == "test/Native/vulkan-product-cli/link-evidence/model-dependency-evidence.json":
                key = (stage.get("stage"), path)
                selected_refs[key] = {"stage": stage.get("stage"),
                                      **{field: ref[field] for field in
                                         ("path", "sha256", "role") if field in ref}}
    fingerprints = {stage.get("source_fingerprint") for stage in stages
                    if isinstance(stage.get("source_fingerprint"), str)}
    missing_tests = sorted(tests - executed)
    passed = (all(profile_valid.get(profile) is True for profile in target_profiles) and
              len(fingerprints) == 1)
    return {"cpu_positive": {"fixture": spec["cpu_fixture"],
                             "fresh_cpu_baseline": []},
            "related_tests": sorted(tests),
            "tests_executed": sorted(tests & executed),
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(tests),
            "tests_missing_for_supplied_profile_scopes": missing_tests,
            "profiles": profile_rows,
            "evidence_family": "va5-published-deployment",
            "evidence_profiles": list(target_profiles),
            "gpu_numerical_claim": spec["gpu_numeric"],
            "raw_artifacts": list(selected_refs.values()),
            "status": "passed" if passed else "blocked",
            "release_status": "not_eligible",
            "release_eligibility_scope": "published product deployment; full support matrix migration (VA6) and final profiles (VA7) remain incomplete",
            "host_aot_release_eligible": False,
            "source_fingerprints": sorted(fingerprints)}


def _va6_id_row(identifier: str, spec: dict,
                stage_audits: list[dict]) -> dict:
    target_profiles = ("release", "sanitize")
    stages = [stage for stage in stage_audits
              if stage.get("stage_id") == "VA6" and
              stage.get("profile") in target_profiles]
    tests = set(spec["tests"])
    profile_rows = []
    selected_refs = {}
    profile_valid = {}
    executed = set()
    for profile in target_profiles:
        matches = [stage for stage in stages if stage.get("profile") == profile]
        if len(matches) != 1:
            profile_valid[profile] = False
            profile_rows.append({"profile": profile,
                                 "status": "missing" if not matches else "duplicate",
                                 "stages": [stage.get("stage") for stage in matches]})
            continue
        stage = matches[0]
        ctest = stage.get("ctest", {})
        names = set(ctest.get("test_names", []))
        skipped = set(ctest.get("skipped", []))
        failed = set(ctest.get("failed", []))
        passed_tests = names - skipped - failed
        executed.update(passed_tests)
        family = stage.get("families", {}).get("va6_matrix", {})
        missing = sorted(tests - names)
        not_passed = sorted(tests & (skipped | failed))
        valid = (stage.get("scope_passed") is True and
                 family.get("status") == "verified" and not missing and
                 not not_passed and
                 stage.get("archive_integrity", {}).get("status") == "verified")
        profile_valid[profile] = valid
        profile_rows.append({"profile": profile, "stage": stage.get("stage"),
                             "source_fingerprint": stage.get("source_fingerprint"),
                             "status": "passed" if valid else "blocked",
                             "tests_executed": sorted(passed_tests & tests),
                             "tests_missing": missing,
                             "tests_not_passed": not_passed,
                             "matrix_evidence": family.get("evidence", [])})
        for ref in stage.get("raw_artifacts", []):
            path = str(ref.get("path", ""))
            if path.startswith("test/Vulkan/evidence/va6/"):
                key = (stage.get("stage"), path)
                selected_refs[key] = {"stage": stage.get("stage"),
                                      **{field: ref[field] for field in
                                         ("path", "sha256", "role") if field in ref}}
    fingerprints = {stage.get("source_fingerprint") for stage in stages
                    if isinstance(stage.get("source_fingerprint"), str)}
    missing_tests = sorted(tests - executed)
    passed = (all(profile_valid.get(profile) is True for profile in target_profiles) and
              len(fingerprints) == 1)
    return {"cpu_positive": {"fixture": spec["cpu_fixture"],
                             "fresh_cpu_baseline": []},
            "related_tests": sorted(tests),
            "tests_executed": sorted(tests & executed),
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(tests),
            "tests_missing_for_supplied_profile_scopes": missing_tests,
            "profiles": profile_rows,
            "evidence_family": "va6-full-matrix",
            "evidence_profiles": list(target_profiles),
            "gpu_numerical_claim": spec["gpu_numeric"],
            "raw_artifacts": list(selected_refs.values()),
            "status": "passed" if passed else "blocked",
            "release_status": "not_eligible",
            "release_eligibility_scope": "full support matrix and robustness migration complete; final profiles (VA7) remain incomplete",
            "host_aot_release_eligible": False,
            "source_fingerprints": sorted(fingerprints)}


def _make_id_index(stage_audits: list[dict], *, release_eligible: bool,
                   all_refs: list[dict]) -> dict:
    legacy_audits = [stage for stage in stage_audits
                     if stage.get("stage_id") in (None, "VA0")]
    actual_tests = set().union(*(set(stage.get("ctest", {}).get("test_names", []))
                                 for stage in legacy_audits)) if legacy_audits else set()
    cpu_baselines = [stage for stage in legacy_audits
                     if stage.get("families", {}).get("cpu_baseline")]
    indexed = {}
    for identifier, spec in ID_CATALOG.items():
        family = spec["evidence_family"]
        related = spec["tests"]
        if family == "va1":
            indexed[identifier] = _va1_id_row(identifier, spec, stage_audits)
            continue
        if family == "va2":
            indexed[identifier] = _va2_id_row(identifier, spec, stage_audits)
            continue
        if family == "va3":
            indexed[identifier] = _va3_id_row(identifier, spec, stage_audits)
            continue
        if family == "va4":
            indexed[identifier] = _va4_id_row(identifier, spec, stage_audits)
            continue
        if family == "va5":
            indexed[identifier] = _va5_id_row(identifier, spec, stage_audits)
            continue
        if family == "va6":
            indexed[identifier] = _va6_id_row(identifier, spec, stage_audits)
            continue
        if family == "va0":
            indexed[identifier] = _va0_id_row(identifier, spec, legacy_audits)
            continue
        executed = sorted(set(related) & actual_tests)
        missing_tests = sorted(set(related) - actual_tests)
        applicable_tests = set().union(*(_applicable_tests(related, stage.get("profile", ""))
                                         for stage in legacy_audits)) if legacy_audits else set()
        missing_applicable = sorted(applicable_tests - actual_tests)
        selected_refs = []
        cpu_hashes = []
        scope_profiles = []
        failed_applicable_scope = any(
            not stage.get("scope_passed") and
            bool(_applicable_tests(related, stage.get("profile", "")) & set(related))
            for stage in legacy_audits)
        for stage in legacy_audits:
            families = stage.get("families", {})
            if family in {"v6.1", "gate", "index", "decision"}:
                include = True
            elif family.startswith("v5."):
                include = bool(families.get("v5"))
            elif family.startswith("v2-"):
                include = bool(families.get("v2"))
            elif family == "cpu":
                include = bool(families.get("cpu_baseline"))
            else:
                include = bool(families.get(family))
            if include:
                scope_profiles.append(stage.get("profile"))
                for ref in stage.get("raw_artifacts", []):
                    path = str(ref.get("path", ""))
                    keep = (family == "v6.1" and ("vulkan-product-offline" in path or
                                                    any(fragment in path for fragment in
                                                        ("gpu-func/", "evidence/v2/", "evidence/v3/",
                                                         "product-cli", "evidence/v5/")))
                            or family == "gate" and (path.startswith("gate-") or path in
                                                        {"gate-state.json", "CMakeCache.txt",
                                                         "test/gate-tests.json"})
                            or family == "index"
                            or family == "decision"
                            or family == "cpu" and "vulkan-v1-cpu-baseline" in path
                            or family == "v0" and "gpu-func/" in path
                            or family.startswith("v2") and "evidence/v2/" in path
                            or family == "v3" and "evidence/v3/" in path
                            or family == "v4" and "product-cli" in path
                            or family.startswith("v5.") and "evidence/v5/" in path)
                    if keep and _id_ref_in_scope(identifier, family, path):
                        selected_refs.append({key: ref[key] for key in ("path", "sha256", "role") if key in ref})
                identity_data = (families if family in {"v6.1", "gate", "index", "decision"}
                                 else families.get("v5") if family.startswith("v5.")
                                 else families.get("v2") if family.startswith("v2-")
                                 else families.get(family))
                identity_data = _scoped_family_identity(family, identity_data)
                cpu_hashes.extend(_collect_cpu_identities(identity_data))
        if family in {"v6.1", "gate", "index", "decision"}:
            selected_refs = list(all_refs)
        cpu_ref = {
            "fixture": spec["cpu_fixture"],
            "fresh_cpu_baseline": [
                {"stage": stage["stage"], "path": stage.get("families", {}).get("cpu_baseline", {}).get("path"),
                 "input_sha256": stage.get("families", {}).get("cpu_baseline", {}).get("input_sha256"),
                 "output_sha256": stage.get("families", {}).get("cpu_baseline", {}).get("output_sha256")}
                for stage in cpu_baselines],
            "paired_cpu_artifact_sha256": sorted(set(cpu_hashes)),
            "raw_data_refs": [ref for ref in selected_refs
                               if any(token in str(ref.get("role", "")).lower()
                                      for token in ("cpu", "input", "golden", "result", "trial"))],
        }
        if failed_applicable_scope:
            status = "blocked"
        elif release_eligible:
            status = "passed" if not missing_tests else "blocked"
        elif not cpu_baselines:
            status = "blocked"
        elif not applicable_tests:
            status = "not_applicable_to_profile"
        elif not scope_profiles:
            status = "blocked"
        elif missing_applicable:
            status = "scope_incomplete"
        else:
            status = "passed_for_profile_scope"
        if (not release_eligible and any(
                stage.get("profile") in {"release", "sanitize"} and not stage.get("scope_passed")
                for stage in legacy_audits) and status == "passed_for_profile_scope"):
            status = "blocked"
        indexed[identifier] = {
            "cpu_positive": cpu_ref,
            "related_tests": related,
            "tests_executed": executed,
            "tests_missing_from_supplied_stages": missing_tests,
            "tests_applicable_to_supplied_profiles": sorted(applicable_tests),
            "tests_missing_for_supplied_profile_scopes": missing_applicable,
            "evidence_family": family,
            "evidence_profiles": sorted(set(scope_profiles)),
            "gpu_numerical_claim": spec["gpu_numeric"],
            "raw_artifacts": selected_refs,
            "status": status,
            "release_status": "passed" if release_eligible else "not_eligible",
        }
    return indexed


def _archive_captures_stage_artifact(relative: str) -> bool:
    return (
        relative.startswith((
            "gate-", "test/Native/vulkan-product-cli",
            "test/Native/vulkan-product-offline/",
            "test/Native/vulkan-v1-cpu-baseline/",
            "test/Vulkan/evidence/", "test/Vulkan/cpu-",
            "gate-source-inputs/", "gate-budget-evidence/",
            "toolchain-evidence/", "legacy-reference/")) or
        relative in {"CMakeCache.txt", "compile_commands.json",
                     "test/gate-tests.json", "release-evidence.json"}
    )


def _discover_archive_input_files(stage: Path) -> dict[str, str]:
    """Reconstruct archive_stage's required evidence inventory from stage bytes."""
    candidates: list[Path] = []
    candidates.extend(path for path in stage.glob("gate-*.json")
                      if path.name not in {"gate-archive-manifest.json",
                                           "gate-archive-receipt.json"})
    candidates.extend(stage.glob("gate-*.log"))
    candidates.extend(stage.glob("gate-*.xml"))
    candidates.extend(stage / name for name in (
        "gate-source-status.txt", "gate-source-diff.patch", "release-evidence.json",
        "CMakeCache.txt", "compile_commands.json", "test/gate-tests.json"))
    candidates.extend(stage / name for name in (
        "test/Vulkan/evidence", "test/Native/vulkan-v1-cpu-baseline",
        "test/Native/vulkan-product-offline", "gate-source-inputs",
        "gate-budget-evidence", "toolchain-evidence", "legacy-reference"))
    candidates.extend((stage / "test/Native").glob("vulkan-product-cli*"))
    candidates.extend((stage / "test/Vulkan").glob("cpu-*"))
    files = {}
    for candidate in candidates:
        if candidate.is_dir():
            paths = (path for path in candidate.rglob("*") if path.is_file())
        elif candidate.is_file():
            paths = (candidate,)
        else:
            continue
        for path in paths:
            resolved = path.resolve()
            try:
                resolved.relative_to(stage.resolve())
            except ValueError as error:
                raise ValueError(f"archive evidence path escapes its stage: {path}") from error
            relative = path.relative_to(stage).as_posix()
            files[relative] = sha256_file(resolved)
    return files


def audit_stage_archive(stage_dir: Path, *, required: bool = False,
                        required_artifacts: list[dict] | None = None) -> dict:
    stage = Path(stage_dir).resolve()
    findings = []
    receipt_path = stage / "gate-archive-receipt.json"
    if not receipt_path.is_file():
        if required:
            _finding(findings, "archive-receipt-missing", "final matrix requires a verified archive receipt")
        return {"status": "not_provided", "findings": findings}
    try:
        receipt = load_json(receipt_path)
        if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1 or
                receipt.get("kind") != "ncnn.vulkan_evidence_archive_receipt"):
            raise ValueError("invalid evidence archive receipt schema")
        record = receipt.get("archive", {})
        name = record.get("file") if isinstance(record, dict) else None
        if (not isinstance(name, str) or not name or Path(name).name != name or
                name in {".", ".."}):
            raise ValueError("archive receipt must identify a stage-adjacent file")
        archive = stage.parent / name
        if not archive.is_file() or sha256_file(archive) != record.get("sha256"):
            raise ValueError("evidence archive bytes are missing or hash-mismatched")
        from run_gate import verify_archive
        manifest = stage / "gate-archive-manifest.json"
        if not manifest.is_file():
            raise ValueError("frozen archive manifest is missing")
        verified = verify_archive(archive, manifest_sha256=sha256_file(manifest))
        if receipt.get("verification") != verified:
            raise ValueError("archive verification receipt differs from actual tar members")
        state = load_json(stage / "gate-state.json")
        if (receipt.get("source_fingerprint") != state.get("source_fingerprint") or
                receipt.get("profile") != state.get("profile") or
                receipt.get("gate_completed") is not True or state.get("completed") is not True):
            raise ValueError("archive receipt does not bind a completed same-source profile")
        archive_manifest = load_json(manifest)
        archived_files = archive_manifest.get("files") if isinstance(archive_manifest, dict) else None
        if not isinstance(archived_files, dict):
            raise ValueError("stage archive manifest has no file inventory")
        audited_artifacts = {}
        if required or required_artifacts is not None:
            audited_artifacts.update(_discover_archive_input_files(stage))
        for artifact in required_artifacts or []:
            if not isinstance(artifact, dict):
                raise ValueError("required archived artifact record is malformed")
            relative = artifact.get("path")
            digest = artifact.get("sha256")
            if (not isinstance(relative, str) or not relative or
                    not SHA256_RE.fullmatch(str(digest or ""))):
                raise ValueError("required archived artifact path/hash is malformed")
            if not _archive_captures_stage_artifact(relative):
                raise ValueError(f"required audited artifact is outside the frozen archive scope: {relative}")
            previous = audited_artifacts.get(relative)
            if previous is not None and previous != digest:
                raise ValueError(f"audited artifact has conflicting hashes: {relative}")
            audited_artifacts[relative] = digest
        for relative, digest in audited_artifacts.items():
            actual_path = safe_stage_file(stage, relative)
            if actual_path is None or not actual_path.is_file():
                raise ValueError(f"audited stage artifact is missing: {relative}")
            if sha256_file(actual_path) != digest:
                raise ValueError(f"audited stage artifact changed after evidence audit: {relative}")
            if archived_files.get(relative) != digest:
                raise ValueError(f"archive omits or changes audited stage artifact: {relative}")
        audited_archived_count = len(audited_artifacts)
        return {"status": "verified", "receipt": {"path": receipt_path.name,
                "sha256": sha256_file(receipt_path)}, "archive": record,
                "verification": verified,
                "audited_artifact_count": audited_archived_count,
                "findings": []}
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        _finding(findings, "archive-integrity-invalid", str(error))
        return {"status": "invalid", "findings": findings}


def build_report(stage_dirs: list[Path], *, require_release: bool = False,
                 verify_fingerprint: bool = True, require_archives: bool = False,
                 require_va1_planner: bool = False,
                 require_va2_helper: bool = False,
                 require_va3_lifecycle: bool = False,
                 require_va4_fixedrun: bool = False,
                 require_va5_deployment: bool = False,
                 require_va6_matrix: bool = False) -> dict:
    gates = []
    for stage_dir in stage_dirs:
        gate = audit_stage_gate(stage_dir, verify_fingerprint=verify_fingerprint)
        stage = audit_stage_evidence(gate)
        if "stage_id" in gate:
            stage["stage_id"] = gate["stage_id"]
        elif "stage_id" not in stage:
            # Historical gate/evidence rows predate stage_id and belong to VA0.
            stage["stage_id"] = "VA0"
        stage_id = stage.get("stage_id")
        required_archive_artifacts = []
        if isinstance(stage_id, str) and stage_id in {"VA1", "VA2", "VA3", "VA4", "VA5"}:
            deduplicated = {}
            for ref in stage.get("raw_artifacts", []):
                relative = str(ref.get("path", ""))
                if _archive_captures_stage_artifact(relative):
                    deduplicated[relative] = {
                        "path": relative, "sha256": ref.get("sha256")}
            required_archive_artifacts = list(deduplicated.values())
        stage["archive_integrity"] = audit_stage_archive(
            stage_dir,
            required=(require_archives and stage_id == "VA0") or
                     (require_va1_planner and stage_id == "VA1") or
                     (require_va2_helper and stage_id == "VA2") or
                     (require_va3_lifecycle and stage_id == "VA3") or
                     (require_va4_fixedrun and stage_id == "VA4") or
                     (require_va5_deployment and stage_id == "VA5") or
                     (require_va6_matrix and stage_id == "VA6"),
            required_artifacts=required_archive_artifacts)
        gates.append(stage)
    global_findings = [{"stage": stage.get("stage"),
                        "stage_id": stage.get("stage_id"), **finding}
                       for stage in gates for finding in
                       stage.get("archive_integrity", {}).get("findings", [])]
    invalid_stage_ids = [stage for stage in gates
                         if not isinstance(stage.get("stage_id"), str) or
                         stage.get("stage_id") not in {"VA0", "VA1", "VA2", "VA3", "VA4", "VA5", "VA6"}]
    invalid_stage_id_findings = []
    for stage in invalid_stage_ids:
        finding = {"code": "stage-id-invalid",
                   "message": "stage_id must be VA0 through VA6 when explicitly supplied",
                   "stage": stage.get("stage"),
                   "stage_id": stage.get("stage_id")}
        global_findings.append(finding)
        invalid_stage_id_findings.append(finding)
    seen_profiles = {}
    for stage in gates:
        stage_id = stage.get("stage_id")
        profile = stage.get("profile")
        key = (type(stage_id).__name__, repr(stage_id),
               type(profile).__name__, repr(profile))
        if key in seen_profiles:
            _finding(global_findings, "duplicate-profile-stage",
                     f"more than one {stage_id!r} stage supplied profile {profile!r}",
                     stage_id=stage_id, profile=profile,
                     stages=[seen_profiles[key], stage.get("stage")])
        else:
            seen_profiles[key] = stage.get("stage")
    grouped_fingerprints = {}
    for stage in gates:
        stage_id = stage.get("stage_id")
        fingerprint = stage.get("source_fingerprint")
        if (isinstance(stage_id, str) and stage_id in {"VA0", "VA1", "VA2", "VA3", "VA4", "VA5", "VA6"} and
                isinstance(fingerprint, str) and SHA256_RE.fullmatch(fingerprint)):
            grouped_fingerprints.setdefault(stage_id, set()).add(fingerprint)
    for stage_id, fingerprints in grouped_fingerprints.items():
        if len(fingerprints) > 1:
            _finding(global_findings, "composite-source-mismatch",
                     f"{stage_id} stages must use the same fresh source_fingerprint",
                     stage_id=stage_id, fingerprints=sorted(fingerprints))

    all_fingerprints = (set().union(*grouped_fingerprints.values())
                        if grouped_fingerprints else set())
    va0_gates = [stage for stage in gates if stage.get("stage_id") == "VA0"]
    va1_gates = [stage for stage in gates if stage.get("stage_id") == "VA1"]
    va2_gates = [stage for stage in gates if stage.get("stage_id") == "VA2"]
    va3_gates = [stage for stage in gates if stage.get("stage_id") == "VA3"]
    va4_gates = [stage for stage in gates if stage.get("stage_id") == "VA4"]
    va5_gates = [stage for stage in gates if stage.get("stage_id") == "VA5"]
    va6_gates = [stage for stage in gates if stage.get("stage_id") == "VA6"]
    required_profiles = {"release", "sanitize", "offline", "software"}
    supplied_profiles = {stage.get("profile") for stage in va0_gates
                         if isinstance(stage.get("profile"), str)}
    missing_profiles = sorted(required_profiles - supplied_profiles)
    invalid_stages = [stage.get("stage") for stage in va0_gates
                      if not stage.get("scope_passed")]
    legacy_archive_findings = [{"stage": stage.get("stage"), **finding}
                               for stage in va0_gates for finding in
                               stage.get("archive_integrity", {}).get("findings", [])]
    legacy_fingerprint_set = grouped_fingerprints.get("VA0", set())
    legacy_scope_findings = [item for item in global_findings
                             if item.get("stage_id") in (None, "VA0")]
    if invalid_stages:
        _finding(global_findings, "stage-scope-failed",
                 "one or more VA0 legacy gate stages failed their evidence audit",
                 stages=invalid_stages, stage_id="VA0")
    release_eligible = (
        not missing_profiles and not invalid_stages and not invalid_stage_ids and
        not legacy_archive_findings and not legacy_scope_findings and
        len(legacy_fingerprint_set) == 1 and all(
            sum(stage.get("profile") == profile for stage in va0_gates) == 1
            for profile in required_profiles))
    release_blockers = []
    if missing_profiles:
        release_blockers.append({"code": "required-profiles-missing",
                                 "message": "legacy release requires VA0 release, sanitize, offline, and software stages",
                                 "profiles": missing_profiles})
    if invalid_stages:
        release_blockers.append({"code": "stage-scope-incomplete",
                                 "message": "one or more complete VA0 legacy gate scopes did not pass",
                                 "stages": invalid_stages})
    release_blockers.extend(invalid_stage_id_findings)
    release_blockers.extend(legacy_archive_findings)
    release_blockers.extend(legacy_scope_findings)
    if require_release and not release_eligible and not missing_profiles:
        _finding(global_findings, "release-eligibility-required",
                 "--require-release was requested but the complete VA0 four-profile matrix is not eligible")
        release_blockers.extend(global_findings[-1:])
    all_refs = []
    legacy_refs = []
    for stage in gates:
        for ref in stage.get("raw_artifacts", []):
            item = {"stage": stage.get("stage"), **ref}
            all_refs.append(item)
            if stage.get("stage_id") == "VA0":
                legacy_refs.append(item)
    id_index = _make_id_index(gates, release_eligible=release_eligible,
                              all_refs=legacy_refs)
    if release_eligible and any(
            row.get("status") != "passed" for identifier, row in id_index.items()
            if not identifier.startswith(("VA0.", "VA1.", "VA2.", "VA3.", "VA4."))):
        release_eligible = False
        release_blockers.append({"code": "id-index-incomplete",
                                 "message": "one or more V0.1–V6.4 audit rows lacks CPU/test/raw evidence"})
        for identifier, row in id_index.items():
            if identifier.startswith(("VA1.", "VA2.", "VA3.")):
                continue
            row["release_status"] = "not_eligible"
            if row["status"] == "passed":
                row["status"] = "blocked"

    va0_required_profiles = {"release", "sanitize", "offline", "software"}
    va0_missing_profiles = sorted(va0_required_profiles -
                                  {stage.get("profile") for stage in va0_gates})
    va0_scope_gates = va0_gates if va0_gates else va1_gates
    va0_findings = [{"stage": stage.get("stage"), "profile": stage.get("profile"), **item}
                    for stage in va0_scope_gates for item in stage.get("va0_findings", [])]
    va0_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA0.")}
    va0_incomplete_ids = (sorted(identifier for identifier, row in va0_ids.items()
                                  if row.get("status") != "passed_for_profile_scope")
                          if va0_gates else [])
    va0_scope_invalid_stages = [stage.get("stage") for stage in va0_scope_gates
                                if not stage.get("scope_passed")]
    va0_index_complete = (not va0_gates or all(
        all(profile_row.get("status") == "passed_for_profile_scope"
            for profile_row in row.get("profiles", []))
        for row in va0_ids.values()))
    va0_stage_scope_passed = (
        bool(va0_scope_gates) and not va0_findings and
        not va0_scope_invalid_stages and va0_index_complete and
        (not va0_gates or (not legacy_archive_findings and
                           not legacy_scope_findings)))
    va0_baseline_stage_scope_passed = (
        bool(va0_gates) and not any(
            stage.get("va0_findings") or not stage.get("scope_passed")
            for stage in va0_gates) and not legacy_archive_findings and
        not legacy_scope_findings and not va0_incomplete_ids and va0_index_complete)
    va0_baseline_eligible = (
        va0_baseline_stage_scope_passed and not va0_missing_profiles and
        len(legacy_fingerprint_set) == 1 and all(
            sum(stage.get("profile") == profile for stage in va0_gates) == 1
            for profile in va0_required_profiles))
    va0_blockers = list(va0_findings)
    va0_blockers.extend(legacy_archive_findings)
    if va0_missing_profiles:
        va0_blockers.append({"code": "va0-required-profiles-missing",
                             "message": "VA0 baseline eligibility requires four fresh profiles",
                             "profiles": va0_missing_profiles})
    if va0_incomplete_ids:
        va0_blockers.append({"code": "va0-id-index-incomplete",
                             "message": "VA0.1–VA0.4 lack profile-scoped legacy-reference/test/raw evidence",
                             "ids": va0_incomplete_ids})

    va1_required_profiles = {"release", "sanitize"}
    va1_missing_profiles = sorted(va1_required_profiles -
                                  {stage.get("profile") for stage in va1_gates})
    va1_fingerprints = grouped_fingerprints.get("VA1", set())
    va1_invalid_stages = [stage.get("stage") for stage in va1_gates
                          if not stage.get("scope_passed") or
                          stage.get("archive_integrity", {}).get("status") != "verified"]
    va1_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA1.")}
    va1_incomplete_ids = sorted(identifier for identifier, row in va1_ids.items()
                                 if row.get("status") != "passed")
    va1_duplicate_profiles = any(
        sum(stage.get("profile") == profile for stage in va1_gates) != 1
        for profile in va1_required_profiles)
    va1_planner_eligible = (
        not va1_missing_profiles and not va1_invalid_stages and
        not va1_duplicate_profiles and len(va1_fingerprints) == 1 and
        not va1_incomplete_ids)
    va1_blockers = []
    if va1_missing_profiles:
        va1_blockers.append({"code": "va1-required-profiles-missing",
                             "profiles": va1_missing_profiles})
    if va1_invalid_stages:
        va1_blockers.append({"code": "va1-stage-incomplete",
                             "stages": va1_invalid_stages})
    if va1_duplicate_profiles:
        va1_blockers.append({"code": "va1-profile-duplicate",
                             "profiles": sorted(profile for profile in va1_required_profiles
                                                 if sum(stage.get("profile") == profile
                                                        for stage in va1_gates) != 1)})
    if len(va1_fingerprints) != 1:
        va1_blockers.append({"code": "va1-source-fingerprint-mismatch",
                             "fingerprints": sorted(va1_fingerprints)})
    if va1_incomplete_ids:
        va1_blockers.append({"code": "va1-id-index-incomplete",
                             "ids": va1_incomplete_ids})
    if require_va1_planner and not va1_planner_eligible:
        _finding(global_findings, "va1-planner-eligibility-required",
                 "--require-va1-planner needs final Release and matching SAN planner stages")

    va2_required_profiles = {"release", "sanitize"}
    va2_missing_profiles = sorted(va2_required_profiles -
                                  {stage.get("profile") for stage in va2_gates})
    va2_fingerprints = grouped_fingerprints.get("VA2", set())
    va2_invalid_stages = [stage.get("stage") for stage in va2_gates
                          if not stage.get("scope_passed") or
                          stage.get("archive_integrity", {}).get("status") != "verified"]
    va2_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA2.")}
    va2_incomplete_ids = sorted(identifier for identifier, row in va2_ids.items()
                                 if row.get("status") != "passed")
    va2_duplicate_profiles = any(
        sum(stage.get("profile") == profile for stage in va2_gates) != 1
        for profile in va2_required_profiles)
    va2_helper_eligible = (
        not va2_missing_profiles and not va2_invalid_stages and
        not va2_duplicate_profiles and len(va2_fingerprints) == 1 and
        not va2_incomplete_ids)
    va2_helper_blockers = []
    if va2_missing_profiles:
        va2_helper_blockers.append({"code": "va2-required-profiles-missing",
                                    "profiles": va2_missing_profiles})
    if va2_invalid_stages:
        va2_helper_blockers.append({"code": "va2-stage-incomplete",
                                    "stages": va2_invalid_stages})
    if va2_duplicate_profiles:
        va2_helper_blockers.append({"code": "va2-profile-duplicate",
                                    "profiles": sorted(profile for profile in va2_required_profiles
                                                        if sum(stage.get("profile") == profile
                                                               for stage in va2_gates) != 1)})
    if len(va2_fingerprints) != 1:
        va2_helper_blockers.append({"code": "va2-source-fingerprint-mismatch",
                                    "fingerprints": sorted(va2_fingerprints)})
    if va2_incomplete_ids:
        va2_helper_blockers.append({"code": "va2-id-index-incomplete",
                                    "ids": va2_incomplete_ids})
    if require_va2_helper and not va2_helper_eligible:
        _finding(global_findings, "va2-helper-eligibility-required",
                 "--require-va2-helper needs final Release and matching static SAN VA2.1-VA2.4 evidence")

    va3_required_profiles = {"release", "sanitize"}
    va3_missing_profiles = sorted(va3_required_profiles -
                                  {stage.get("profile") for stage in va3_gates})
    va3_fingerprints = grouped_fingerprints.get("VA3", set())
    va3_invalid_stages = [stage.get("stage") for stage in va3_gates
                          if not stage.get("scope_passed") or
                          stage.get("archive_integrity", {}).get("status") != "verified"]
    va3_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA3.")}
    va3_incomplete_ids = sorted(identifier for identifier, row in va3_ids.items()
                                 if row.get("status") != "passed")
    va3_duplicate_profiles = any(
        sum(stage.get("profile") == profile for stage in va3_gates) != 1
        for profile in va3_required_profiles)
    va3_lifecycle_eligible = (
        not va3_missing_profiles and not va3_invalid_stages and
        not va3_duplicate_profiles and len(va3_fingerprints) == 1 and
        not va3_incomplete_ids)
    va3_lifecycle_blockers = []
    if va3_missing_profiles:
        va3_lifecycle_blockers.append({"code": "va3-required-profiles-missing",
                                       "profiles": va3_missing_profiles})
    if va3_invalid_stages:
        va3_lifecycle_blockers.append({"code": "va3-stage-incomplete",
                                       "stages": va3_invalid_stages})
    if va3_duplicate_profiles:
        va3_lifecycle_blockers.append({"code": "va3-profile-duplicate",
                                       "profiles": sorted(
                                           profile for profile in va3_required_profiles
                                           if sum(stage.get("profile") == profile
                                                  for stage in va3_gates) != 1)})
    if len(va3_fingerprints) != 1:
        va3_lifecycle_blockers.append({"code": "va3-source-fingerprint-mismatch",
                                       "fingerprints": sorted(va3_fingerprints)})
    if va3_incomplete_ids:
        va3_lifecycle_blockers.append({"code": "va3-id-index-incomplete",
                                       "ids": va3_incomplete_ids})
    if require_va3_lifecycle and not va3_lifecycle_eligible:
        _finding(global_findings, "va3-lifecycle-eligibility-required",
                 "--require-va3-lifecycle needs final same-source Release and static SAN VA3.1-VA3.4 evidence and verified archives")

    va4_required_profiles = {"release", "sanitize"}
    va4_missing_profiles = sorted(va4_required_profiles -
                                  {stage.get("profile") for stage in va4_gates})
    va4_fingerprints = grouped_fingerprints.get("VA4", set())
    va4_invalid_stages = [stage.get("stage") for stage in va4_gates
                          if not stage.get("scope_passed") or
                          stage.get("archive_integrity", {}).get("status") != "verified" or
                          stage.get("families", {}).get("va4_fixedrun", {}).get("status") != "verified"]
    va4_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA4.")}
    va4_incomplete_ids = sorted(identifier for identifier, row in va4_ids.items()
                                 if row.get("status") != "passed")
    va4_duplicate_profiles = any(
        sum(stage.get("profile") == profile for stage in va4_gates) != 1
        for profile in va4_required_profiles)
    va4_fixedrun_eligible = (
        not va4_missing_profiles and not va4_invalid_stages and
        not va4_duplicate_profiles and len(va4_fingerprints) == 1 and
        not va4_incomplete_ids)
    va4_fixedrun_blockers = []
    if va4_missing_profiles:
        va4_fixedrun_blockers.append({"code": "va4-required-profiles-missing",
                                      "profiles": va4_missing_profiles})
    if va4_invalid_stages:
        va4_fixedrun_blockers.append({"code": "va4-stage-incomplete",
                                      "stages": va4_invalid_stages})
    if va4_duplicate_profiles:
        va4_fixedrun_blockers.append({"code": "va4-profile-duplicate",
                                      "profiles": sorted(
                                          profile for profile in va4_required_profiles
                                          if sum(stage.get("profile") == profile
                                                 for stage in va4_gates) != 1)})
    if len(va4_fingerprints) != 1:
        va4_fixedrun_blockers.append({"code": "va4-source-fingerprint-mismatch",
                                      "fingerprints": sorted(va4_fingerprints)})
    if va4_incomplete_ids:
        va4_fixedrun_blockers.append({"code": "va4-id-index-incomplete",
                                      "ids": va4_incomplete_ids})
    if require_va4_fixedrun and not va4_fixedrun_eligible:
        _finding(global_findings, "va4-fixedrun-eligibility-required",
                 "--require-va4-fixedrun needs final same-source Release and static SAN VA4.1-VA4.4 evidence, independent legacy comparison, and verified archives")

    va5_required_profiles = {"release", "sanitize"}
    va5_missing_profiles = sorted(va5_required_profiles -
                                  {stage.get("profile") for stage in va5_gates})
    va5_fingerprints = grouped_fingerprints.get("VA5", set())
    va5_invalid_stages = [stage.get("stage") for stage in va5_gates
                          if not stage.get("scope_passed") or
                          stage.get("archive_integrity", {}).get("status") != "verified" or
                          stage.get("families", {}).get("va5_deployment", {}).get("status") != "verified"]
    va5_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA5.")}
    va5_incomplete_ids = sorted(identifier for identifier, row in va5_ids.items()
                                if row.get("status") != "passed")
    va5_duplicate_profiles = any(
        sum(stage.get("profile") == profile for stage in va5_gates) != 1
        for profile in va5_required_profiles)
    va5_deployment_eligible = (
        not va5_missing_profiles and not va5_invalid_stages and
        not va5_duplicate_profiles and len(va5_fingerprints) == 1 and
        not va5_incomplete_ids)
    va5_deployment_blockers = []
    if va5_missing_profiles:
        va5_deployment_blockers.append({"code": "va5-required-profiles-missing",
                                        "profiles": va5_missing_profiles})
    if va5_invalid_stages:
        va5_deployment_blockers.append({"code": "va5-stage-incomplete",
                                        "stages": va5_invalid_stages})
    if va5_duplicate_profiles:
        va5_deployment_blockers.append({"code": "va5-profile-duplicate",
                                        "profiles": sorted(
                                            profile for profile in va5_required_profiles
                                            if sum(stage.get("profile") == profile
                                                   for stage in va5_gates) != 1)})
    if len(va5_fingerprints) != 1:
        va5_deployment_blockers.append({"code": "va5-source-fingerprint-mismatch",
                                        "fingerprints": sorted(va5_fingerprints)})
    if va5_incomplete_ids:
        va5_deployment_blockers.append({"code": "va5-id-index-incomplete",
                                        "ids": va5_incomplete_ids})
    if require_va5_deployment and not va5_deployment_eligible:
        _finding(global_findings, "va5-deployment-eligibility-required",
                 "--require-va5-deployment needs final same-source Release and "
                 "static SAN VA5.1-VA5.4 deployment evidence and verified archives")

    va6_required_profiles = {"release", "sanitize"}
    va6_missing_profiles = sorted(va6_required_profiles -
                                  {stage.get("profile") for stage in va6_gates})
    va6_fingerprints = grouped_fingerprints.get("VA6", set())
    va6_invalid_stages = [stage.get("stage") for stage in va6_gates
                          if not stage.get("scope_passed") or
                          stage.get("archive_integrity", {}).get("status") != "verified" or
                          stage.get("families", {}).get("va6_matrix", {}).get("status") != "verified"]
    va6_ids = {identifier: row for identifier, row in id_index.items()
               if identifier.startswith("VA6.")}
    va6_incomplete_ids = sorted(identifier for identifier, row in va6_ids.items()
                                if row.get("status") != "passed")
    va6_duplicate_profiles = any(
        sum(stage.get("profile") == profile for stage in va6_gates) != 1
        for profile in va6_required_profiles)
    va6_matrix_eligible = (
        not va6_missing_profiles and not va6_invalid_stages and
        not va6_duplicate_profiles and len(va6_fingerprints) == 1 and
        not va6_incomplete_ids)
    va6_matrix_blockers = []
    if va6_missing_profiles:
        va6_matrix_blockers.append({"code": "va6-required-profiles-missing",
                                    "profiles": va6_missing_profiles})
    if va6_invalid_stages:
        va6_matrix_blockers.append({"code": "va6-stage-incomplete",
                                    "stages": va6_invalid_stages})
    if va6_duplicate_profiles:
        va6_matrix_blockers.append({"code": "va6-profile-duplicate",
                                    "profiles": sorted(
                                        profile for profile in va6_required_profiles
                                        if sum(stage.get("profile") == profile
                                               for stage in va6_gates) != 1)})
    if len(va6_fingerprints) != 1:
        va6_matrix_blockers.append({"code": "va6-source-fingerprint-mismatch",
                                    "fingerprints": sorted(va6_fingerprints)})
    if va6_incomplete_ids:
        va6_matrix_blockers.append({"code": "va6-id-index-incomplete",
                                    "ids": va6_incomplete_ids})
    if require_va6_matrix and not va6_matrix_eligible:
        _finding(global_findings, "va6-matrix-eligibility-required",
                 "--require-va6-matrix needs final same-source Release and "
                 "static SAN VA6.1-VA6.4 matrix/faults/cross-ICD evidence and "
                 "verified archives")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": (next(iter(all_fingerprints))
                               if len(all_fingerprints) == 1 else None),
        "va0_source_fingerprint": (next(iter(legacy_fingerprint_set))
                                   if len(legacy_fingerprint_set) == 1 else None),
        "va1_source_fingerprint": (next(iter(va1_fingerprints))
                                   if len(va1_fingerprints) == 1 else None),
        "va2_source_fingerprint": (next(iter(va2_fingerprints))
                                   if len(va2_fingerprints) == 1 else None),
        "va3_source_fingerprint": (next(iter(va3_fingerprints))
                                   if len(va3_fingerprints) == 1 else None),
        "va4_source_fingerprint": (next(iter(va4_fingerprints))
                                   if len(va4_fingerprints) == 1 else None),
        "va5_source_fingerprint": (next(iter(va5_fingerprints))
                                   if len(va5_fingerprints) == 1 else None),
        "va6_source_fingerprint": (next(iter(va6_fingerprints))
                                   if len(va6_fingerprints) == 1 else None),
        "source_fingerprints_by_stage_id": {
            key: sorted(value) for key, value in grouped_fingerprints.items()},
        "profiles_required_for_release": sorted(required_profiles),
        "profiles_supplied": sorted(profile for profile in supplied_profiles if profile),
        "va1_profiles_required": sorted(va1_required_profiles),
        "va1_profiles_missing": va1_missing_profiles,
        "va2_profiles_required": sorted(va2_required_profiles),
        "va2_profiles_missing": va2_missing_profiles,
        "va3_profiles_required": sorted(va3_required_profiles),
        "va3_profiles_missing": va3_missing_profiles,
        "va4_profiles_required": sorted(va4_required_profiles),
        "va4_profiles_missing": va4_missing_profiles,
        "va5_profiles_required": sorted(va5_required_profiles),
        "va5_profiles_missing": va5_missing_profiles,
        "va6_profiles_required": sorted(va6_required_profiles),
        "va6_profiles_missing": va6_missing_profiles,
        "release_eligible": release_eligible,
        "legacy_vulkan_release_eligible": release_eligible,
        "va0_baseline_eligible": va0_baseline_eligible,
        "va0_stage_scope_passed": va0_stage_scope_passed,
        "va0_profiles_required": sorted(va0_required_profiles),
        "va0_profiles_missing": va0_missing_profiles,
        "va0_blocking_findings": va0_blockers,
        "va1_planner_eligible": va1_planner_eligible,
        "va1_planner_required": require_va1_planner,
        "va1_planner_stages": [
            {"stage": stage.get("stage"), "profile": stage.get("profile"),
             "source_fingerprint": stage.get("source_fingerprint"),
             "scope_passed": stage.get("scope_passed"),
             "archive_status": stage.get("archive_integrity", {}).get("status"),
             "evidence": stage.get("families", {}).get("va1", {})}
            for stage in va1_gates],
        "va1_planner_blockers": va1_blockers,
        "va2_helper_eligible": va2_helper_eligible,
        "va2_helper_required": require_va2_helper,
        "va2_helper_stages": [
            {"stage": stage.get("stage"), "profile": stage.get("profile"),
             "source_fingerprint": stage.get("source_fingerprint"),
             "scope_passed": stage.get("scope_passed"),
             "archive_status": stage.get("archive_integrity", {}).get("status"),
             "evidence": stage.get("families", {}).get("va2_helper", {})}
            for stage in va2_gates],
        "va2_helper_blockers": va2_helper_blockers,
        "va3_lifecycle_eligible": va3_lifecycle_eligible,
        "va3_lifecycle_required": require_va3_lifecycle,
        "va3_lifecycle_stages": [
            {"stage": stage.get("stage"), "profile": stage.get("profile"),
             "source_fingerprint": stage.get("source_fingerprint"),
             "scope_passed": stage.get("scope_passed"),
             "archive_status": stage.get("archive_integrity", {}).get("status"),
             "evidence": stage.get("families", {}).get("va3_lifecycle", {})}
            for stage in va3_gates],
        "va3_lifecycle_blockers": va3_lifecycle_blockers,
        "va4_fixedrun_eligible": va4_fixedrun_eligible,
        "va4_fixedrun_required": require_va4_fixedrun,
        "va4_fixedrun_stages": [
            {"stage": stage.get("stage"), "profile": stage.get("profile"),
             "source_fingerprint": stage.get("source_fingerprint"),
             "scope_passed": stage.get("scope_passed"),
             "archive_status": stage.get("archive_integrity", {}).get("status"),
             "evidence": stage.get("families", {}).get("va4_fixedrun", {})}
            for stage in va4_gates],
        "va4_fixedrun_blockers": va4_fixedrun_blockers,
        "va5_deployment_eligible": va5_deployment_eligible,
        "va5_deployment_required": require_va5_deployment,
        "va5_deployment_stages": [
            {"stage": stage.get("stage"), "profile": stage.get("profile"),
             "source_fingerprint": stage.get("source_fingerprint"),
             "scope_passed": stage.get("scope_passed"),
             "archive_status": stage.get("archive_integrity", {}).get("status"),
             "evidence": stage.get("families", {}).get("va5_deployment", {})}
            for stage in va5_gates],
        "va5_deployment_blockers": va5_deployment_blockers,
        "va6_matrix_eligible": va6_matrix_eligible,
        "va6_matrix_required": require_va6_matrix,
        "va6_matrix_stages": [
            {"stage": stage.get("stage"), "profile": stage.get("profile"),
             "source_fingerprint": stage.get("source_fingerprint"),
             "scope_passed": stage.get("scope_passed"),
             "archive_status": stage.get("archive_integrity", {}).get("status"),
             "evidence": stage.get("families", {}).get("va6_matrix", {})}
            for stage in va6_gates],
        "va6_matrix_blockers": va6_matrix_blockers,
        "host_aot_release_eligible": False,
        "baseline_scope": "VA0 legacy baseline, VA1 planner, VA2 helper, VA3 private lifecycle candidate, VA4 fixed-run prototype, VA5 published deployment, and VA6 full support matrix/robustness migration; VA7 final profiles remain open",
        "host_aot_blockers": ([] if va1_planner_eligible else
                               ["VA1 planner evidence incomplete"]) +
                              ([] if va2_helper_eligible else
                               ["VA2 helper qualification incomplete"]) +
                              ([] if va3_lifecycle_eligible else
                               ["VA3 lifecycle candidate qualification incomplete"]) +
                              (["VA3 lifecycle-only candidate has no model run"] if not va4_fixedrun_eligible else
                               ["VA4 fixed-run is a private small-graph prototype, not a switched product"]) +
                              ([] if va4_fixedrun_eligible else
                               ["VA4 fixed-run/synchronization evidence incomplete"]) +
                              ([] if va5_deployment_eligible else
                               ["VA5 link/clean-publish evidence incomplete"]) +
                              ["VA6 full matrix/faults not implemented",
                               "VA7 profiles/inventory not implemented"],
        "release_decision": {
            "decision": "release-eligible-opt-in" if release_eligible else "no-go-or-incomplete",
            "host_aot_release_eligible": False,
            "scope_id": "legacy-vulkan-functional-gate",
            "host_aot_evidence_stage": "VA0 (baseline/coverage only)",
            "default_backend": "CPU",
            "scope": "legacy static FP32 support matrix and actually tested physical devices/ICDs/drivers only",
            "blockers": release_blockers,
            "peak_memory_note": "plan bytes are not peak allocation; unmeasured peak is reported as unmeasured",
        },
        "stages": gates,
        "id_index": id_index,
        "raw_artifacts": all_refs,
        "blocking_findings": global_findings + [finding for stage in gates for finding in stage.get("findings", [])],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True, type=Path,
                        help="fresh build stage directory; may be specified repeatedly")
    parser.add_argument("--output", required=True, type=Path,
                        help="JSON release-evidence report path")
    parser.add_argument("--require-release", action="store_true",
                        help="return failure unless the complete legacy Vulkan four-profile matrix is eligible")
    parser.add_argument("--require-va0-baseline", action="store_true",
                        help="return failure unless the four-profile VA0 reference/index is complete")
    parser.add_argument("--require-va0-stage", action="store_true",
                        help="return failure unless each supplied stage passes its VA0 reference scope")
    parser.add_argument("--require-va1-planner", action="store_true",
                        help="require final Release and matching SAN compiler-only VA1.1-VA1.4 planner evidence")
    parser.add_argument("--require-va2-helper", action="store_true",
                        help="require final Release and matching static SAN VA2.1-VA2.4 helper evidence")
    parser.add_argument("--require-va3-lifecycle", action="store_true",
                        help="require final same-source Release and static SAN private VA3.1-VA3.4 lifecycle evidence and verified archives")
    parser.add_argument("--require-va4-fixedrun", action="store_true",
                        help="require final same-source Release and static SAN VA4.1-VA4.4 fixed-run evidence and verified archives")
    parser.add_argument("--require-va5-deployment", action="store_true",
                        help="require final same-source VA5 release and SAN stages")
    parser.add_argument("--require-va6-matrix", action="store_true",
                        help="require final same-source Release and static SAN VA6.1-VA6.4 matrix/faults/cross-ICD evidence and verified archives")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.stage, require_release=args.require_release,
            require_archives=args.require_va0_baseline,
            require_va1_planner=args.require_va1_planner,
            require_va2_helper=args.require_va2_helper,
            require_va3_lifecycle=args.require_va3_lifecycle,
            require_va4_fixedrun=args.require_va4_fixedrun,
            require_va5_deployment=args.require_va5_deployment,
            require_va6_matrix=args.require_va6_matrix)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report = {"schema_version": SCHEMA_VERSION, "kind": REPORT_KIND,
                  "generated_at": datetime.now(timezone.utc).isoformat(),
                  "release_eligible": False, "va0_baseline_eligible": False,
                  "va1_planner_eligible": False,
                  "va1_planner_required": args.require_va1_planner,
                  "va1_planner_blockers": [{"code": "audit-failed-closed",
                                            "message": str(error)}],
                  "va2_helper_eligible": False,
                  "va2_helper_required": args.require_va2_helper,
                  "va2_helper_blockers": [{"code": "audit-failed-closed",
                                           "message": str(error)}],
                  "va3_lifecycle_eligible": False,
                  "va3_lifecycle_required": args.require_va3_lifecycle,
                  "va3_lifecycle_blockers": [{"code": "audit-failed-closed",
                                               "message": str(error)}],
                  "va4_fixedrun_eligible": False,
                  "va4_fixedrun_required": args.require_va4_fixedrun,
                  "va4_fixedrun_blockers": [{"code": "audit-failed-closed",
                                              "message": str(error)}],
                  "host_aot_release_eligible": False,
                  "va0_blocking_findings": [{"code": "audit-failed-closed", "message": str(error)}],
                  "blocking_findings": [{"code": "audit-failed-closed", "message": str(error)}],
                  "id_index": {identifier: {"status": "blocked"} for identifier in ID_CATALOG}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"release evidence report: {args.output}")
    errors = report.get("blocking_findings", [])
    if errors:
        print(f"blocking findings: {len(errors)}", file=sys.stderr)
    if args.require_va0_baseline and not report.get("va0_baseline_eligible"):
        print("four-profile VA0 baseline is incomplete; see va0_blocking_findings", file=sys.stderr)
    if args.require_va0_stage and not report.get("va0_stage_scope_passed"):
        print("supplied VA0 stage is incomplete; see va0_blocking_findings", file=sys.stderr)
    if args.require_va1_planner and not report.get("va1_planner_eligible"):
        print("VA1 planner scope is incomplete; see va1_planner_blockers", file=sys.stderr)
    if args.require_va2_helper and not report.get("va2_helper_eligible"):
        print("VA2 helper scope is incomplete; see va2_helper_blockers", file=sys.stderr)
    if args.require_va3_lifecycle and not report.get("va3_lifecycle_eligible"):
        print("VA3 lifecycle scope is incomplete; see va3_lifecycle_blockers", file=sys.stderr)
    if args.require_va4_fixedrun and not report.get("va4_fixedrun_eligible"):
        print("VA4 fixed-run scope is incomplete; see va4_fixedrun_blockers", file=sys.stderr)
    return 1 if (errors or (args.require_release and not report.get("release_eligible")) or
                 (args.require_va0_baseline and not report.get("va0_baseline_eligible")) or
                 (args.require_va0_stage and not report.get("va0_stage_scope_passed")) or
                 (args.require_va1_planner and not report.get("va1_planner_eligible")) or
                 (args.require_va2_helper and not report.get("va2_helper_eligible")) or
                 (args.require_va3_lifecycle and not report.get("va3_lifecycle_eligible")) or
                 (args.require_va4_fixedrun and not report.get("va4_fixedrun_eligible"))) else 0


if __name__ == "__main__":
    raise SystemExit(main())
