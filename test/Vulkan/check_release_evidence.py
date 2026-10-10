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
        proof_record = proof.get("record")
        copied_record_elapsed = proof_record.get("elapsed_seconds") if isinstance(proof_record, dict) else None
        if not isinstance(elapsed, int) or elapsed <= 300 or elapsed != copied_record_elapsed:
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
                    pattern = rf"fixture {re.escape(fixture)} 编译耗时 (\d+)s 超过硬预算 300s"
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
    if not isinstance(identity, dict) or identity.get("schema") != "ncnn-san-toolchain-identity/v1" or identity.get("status") != "verified":
        _finding(findings, "sanitizer-toolchain-identity-status", "copied toolchain identity is not verified")
        identity = identity if isinstance(identity, dict) else {}
    llvm = identity.get("llvm", {})
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
    archive_paths = []
    if archive_count != 486 or llvm.get("static_archive_count") != 486 or not isinstance(archives, list) or len(archives) != 486:
        _finding(findings, "sanitizer-archive-count", "SAN toolchain must identify all 486 static LLVM/MLIR archives")
    else:
        for record in archives:
            if (not isinstance(record, dict) or not isinstance(record.get("path"), str) or
                    not SHA256_RE.fullmatch(str(record.get("sha256", "")))):
                _finding(findings, "sanitizer-archive-identity", "static archive identity record is malformed")
                continue
            archive_path = Path(record["path"])
            archive_paths.append(str(archive_path.resolve()))
            if not archive_path.is_file():
                _finding(findings, "sanitizer-archive-missing", "identified static LLVM/MLIR archive is missing",
                         path=record["path"])
            elif sha256_file(archive_path) != record["sha256"]:
                _finding(findings, "sanitizer-archive-hash", "static LLVM/MLIR archive content changed after preflight",
                         path=record["path"])
        if len(set(archive_paths)) != len(archive_paths):
            _finding(findings, "sanitizer-archive-duplicate", "static archive manifest repeats an archive path")
    archive_list = llvm.get("static_archive_list", {})
    archive_list_path = Path(str(archive_list.get("path", ""))) if isinstance(archive_list, dict) else Path()
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
            if len(listed_archives) != 486 or len(set(listed_archives)) != 486:
                _finding(findings, "sanitizer-archive-list-count",
                         "SAN static archive list must contain 486 unique paths")
            if archive_paths and sorted(listed_archives) != sorted(archive_paths):
                _finding(findings, "sanitizer-archive-list-mismatch",
                         "SAN archive list path multiset differs from static archive identities")
        except (OSError, UnicodeError, ValueError) as error:
            _finding(findings, "sanitizer-archive-list-invalid", f"SAN archive list is invalid: {error}")
    source_archive = identity.get("source_archive_identity", {})
    source_archive_path = Path(str(source_archive.get("path", ""))) if isinstance(source_archive, dict) else Path()
    source_archive_hash = source_archive.get("required_sha256") if isinstance(source_archive, dict) else None
    if (not source_archive_path.is_file() or not SHA256_RE.fullmatch(str(source_archive_hash or "")) or
            (source_archive_path.is_file() and sha256_file(source_archive_path) != source_archive_hash)):
        _finding(findings, "sanitizer-source-archive", "SAN LLVM source archive path/hash is missing or changed")
    tool_artifacts = identity.get("tool_artifacts", {})
    if not isinstance(tool_artifacts, dict) or not tool_artifacts:
        _finding(findings, "sanitizer-tool-artifacts", "SAN verified tool binary identities are missing")
    else:
        for role, tool_record in tool_artifacts.items():
            tool_path = Path(str(tool_record.get("path", ""))) if isinstance(tool_record, dict) else Path()
            tool_hash = tool_record.get("sha256") if isinstance(tool_record, dict) else None
            if (not tool_path.is_file() or not SHA256_RE.fullmatch(str(tool_hash or "")) or
                    (tool_path.is_file() and sha256_file(tool_path) != tool_hash)):
                _finding(findings, "sanitizer-tool-artifact", f"SAN tool artifact path/hash is invalid: {role}")
    for smoke_key in ("gpu_runner_pipeline_fixture", "gpu_runner_pipeline_output"):
        smoke_record = identity.get("smokes", {}).get(smoke_key, {})
        smoke_path = Path(str(smoke_record.get("path", ""))) if isinstance(smoke_record, dict) else Path()
        if not smoke_path.is_file() or not isinstance(smoke_record, dict) or not SHA256_RE.fullmatch(str(smoke_record.get("sha256", ""))):
            _finding(findings, "sanitizer-gpu-smoke", f"SAN GPU runner smoke identity is missing: {smoke_key}")
        elif sha256_file(smoke_path) != smoke_record.get("sha256"):
            _finding(findings, "sanitizer-gpu-smoke-hash", f"SAN GPU runner smoke identity changed: {smoke_key}")
    libomp = identity.get("libomp", {})
    libomp_shared = libomp.get("shared_library", {})
    libomp_hash = libomp_shared.get("sha256")
    libomp_path = Path(str(libomp_shared.get("path", "")))
    if (evidence.get("libomp_sha256") != libomp_hash or not SHA256_RE.fullmatch(str(libomp_hash or ""))
            or not libomp_path.is_file() or (libomp_path.is_file() and sha256_file(libomp_path) != libomp_hash)):
        _finding(findings, "sanitizer-libomp-identity", "bundled libomp SHA differs from verified identity")
    if libomp.get("static_library") is not None or libomp.get("libgomp_aliases") != []:
        _finding(findings, "sanitizer-libomp-alias", "SAN libomp identity includes a static/foreign OpenMP alias")
    smoke = libomp.get("lsan_smoke", {})
    smoke_binary = Path(str(smoke.get("binary", "")))
    smoke_log = Path(str(smoke.get("log", "")))
    if (smoke.get("leak_detection") is not True or smoke.get("suppressions") is not False or
            not smoke_binary.is_file() or not smoke_log.is_file()):
        _finding(findings, "sanitizer-lsan-smoke", "SAN libomp smoke did not prove unsuppressed leak detection")
    elif SANITIZER_RE.search(smoke_log.read_text(encoding="utf-8", errors="replace")):
        _finding(findings, "sanitizer-lsan-diagnostic", "SAN libomp leak smoke raw log contains sanitizer/runtime diagnostics")
    smokes = identity.get("smokes", {})
    verification_text = evidence.get("verification_status")
    verification_source = Path(str(smokes.get("verification_status", "")))
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
        if not openmp_dir or Path(openmp_dir).resolve() / "libomp.so" != libomp_path.resolve():
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
    core_audit = Path(llvm.get("core_shared_dso_audit", ""))
    if not core_audit.is_file() or core_audit.stat().st_size != 0:
        _finding(findings, "sanitizer-core-dso-audit", "SAN core LLVM/MLIR shared-object audit is missing or nonempty")
    compile_audit_path = Path(llvm.get("compile_pic_sanitizer_audit", ""))
    if not compile_audit_path.is_file():
        _finding(findings, "sanitizer-pic-audit", "SAN instrumentation/PIC source audit is missing")
    else:
        compile_audit = compile_audit_path.read_text(encoding="utf-8", errors="replace")
        count_records = re.findall(
            r"^compile_commands=(\d+); sanitizer_instrumented_sources=(\d+); pic_sources=(\d+)$",
            compile_audit, re.MULTILINE)
        if count_records != [("4316", "4312", "4316")]:
            _finding(findings, "sanitizer-pic-audit", "SAN instrumentation/PIC source counts are not the exact verified record")
    asan_runtime = llvm.get("shared_asan_runtime_identity", {})
    asan_path = Path(asan_runtime.get("path", ""))
    if not asan_path.is_file() or not SHA256_RE.fullmatch(str(asan_runtime.get("sha256", ""))) or (asan_path.is_file() and sha256_file(asan_path) != asan_runtime.get("sha256")):
        _finding(findings, "sanitizer-runtime-identity", "shared ASan runtime identity is missing or changed")
    for symbol_field in ("mlir_ir_asan_symbol_audit", "mlir_ir_ubsan_symbol_audit"):
        symbol_file = Path(llvm.get(symbol_field, ""))
        if not symbol_file.is_file() or symbol_file.stat().st_size == 0:
            _finding(findings, "sanitizer-symbol-audit", f"SAN static library symbol audit is missing: {symbol_field}")
    pic_probe = Path(identity.get("root", "")) / "smoke/libMLIRIR-pic.so"
    if not pic_probe.is_file():
        _finding(findings, "sanitizer-pic-link-smoke", "SAN static archive PIC link smoke output is missing")
    relocation = Path(llvm.get("mlir_ir_pic_relocation_audit", ""))
    if not relocation.is_file() or "TEXTREL" in relocation.read_text(encoding="utf-8", errors="replace"):
        _finding(findings, "sanitizer-pic-relocation", "SAN static archive has no clean relocation audit")
    copied = evidence.get("files")
    if not isinstance(copied, list) or len(copied) < 79:
        _finding(findings, "sanitizer-toolchain-files", "SAN toolchain bundle lacks the full verified log/status/config/patch inventory")
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
            if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
                _finding(findings, "sanitizer-toolchain-source-hash", "SAN toolchain source file hash is missing")
            elif not source_path.is_file() or sha256_file(source_path) != source_hash:
                _finding(findings, "sanitizer-toolchain-source-tampered", "SAN toolchain source is missing or changed after bundling",
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
        _check_required_tests({"profile": profile, "ctest": ctest_result}, findings)

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

    return {"stage": str(stage), "profile": profile, "source_fingerprint": fingerprint,
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


def _check_product_artifact_set(stage: Path, base: Path, fixture: dict,
                                refs: list[dict], findings: list[dict]) -> None:
    vk_dir = base / "vulkan"
    cpu_dir = base / "cpu"
    mapping = {
        "cpu_library_sha256": cpu_dir / "libpaired_relu.so",
        "cpu_header_sha256": cpu_dir / "paired_relu.h",
        "vulkan_library_sha256": vk_dir / "libpaired_relu_vulkan.so",
        "vulkan_header_sha256": vk_dir / "paired_relu_vulkan.h",
        "vulkan_manifest_sha256": vk_dir / "paired_relu_vulkan.json",
        "vulkan_plan_sha256": vk_dir / "paired_relu_vulkan.plan.json",
    }
    for field, path in mapping.items():
        _evidence_file(stage, path.parent, path.name, f"product {field}", refs, findings)
        _check_digest_claim(fixture.get(field), path if path.is_file() else None,
                            f"product {field}", findings)
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
            compile_args.count("-std=c++20") == 1 and
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
        if fixture.get("model") != "paired_relu":
            _finding(findings, "offline-fixture", "offline positive fixture is not paired ReLU")
        model_param = _evidence_file(stage, base, "model.param", "offline model .param", refs, findings)
        model_bin = _evidence_file(stage, base, "model.bin", "offline model .bin", refs, findings)
        for field, path in (("param_sha256", model_param), ("bin_sha256", model_bin)):
            _check_digest_claim(fixture.get(field), path, f"offline {field}", findings)
        public_io = evidence.get("public_io", {})
        if public_io.get("element_type") != "f32" or public_io.get("shape") != [257]:
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


def _audit_product(stage: Path, expected_devices: set[str], refs: list[dict],
                   findings: list[dict]) -> dict:
    records = {}
    paths = sorted((stage / "test/Native").glob("vulkan-product-cli*/evidence.json"))
    if not paths:
        _finding(findings, "product-evidence-missing", "fresh V4 product CLI evidence is absent")
        return records
    found = set()
    for evidence_path in paths:
        base = evidence_path.parent
        try:
            evidence = load_json(evidence_path)
            _evidence_file(stage, base, "evidence.json", f"V4 {base.name} evidence", refs, findings)
            if not isinstance(evidence, dict) or evidence.get("schema_version") != 1 or evidence.get("kind") != "ncnn.vulkan_product_evidence":
                raise ValueError("unsupported product evidence schema/kind")
            execution = evidence.get("execution", {})
            if execution.get("mode") != "device-execution" or execution.get("vulkan_execution") != "passed":
                _finding(findings, "product-execution", f"V4 {base.name} does not record a successful device execution")
            fixture = evidence.get("fixture", {})
            io = evidence.get("public_io", {})
            if fixture.get("model") != "paired_relu" or io.get("element_type") != "f32":
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
             "cpu_manifest": cpu_dir / f"{model}.json",
             "vulkan_manifest": vk_dir / f"{model}_vulkan.json",
             "vulkan_header": vk_dir / f"{model}_vulkan.h"}
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
    manifest_path = paths["vulkan_manifest"]
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _finding(findings, "v5-manifest-invalid", f"V5 {model} Vulkan manifest JSON invalid: {error}")
    manifest = manifest if isinstance(manifest, dict) else {}
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


def _required_tests(profile: str) -> set[str]:
    tests = {"vulkan-abi-c-contract", "vulkan-v1-cpu-baseline",
             "vulkan-release-evidence-validation", "vulkan-gate-validation",
             "ncnn-compile-vulkan-offline", "vulkan-v5-comparison-validation",
             "ncnn-mlir-lit", *OFFLINE_RUNTIME_TESTS}
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
    return tests


def _check_required_tests(gate: dict, findings: list[dict]) -> None:
    names = set(gate.get("ctest", {}).get("test_names", []))
    missing = sorted(_required_tests(gate.get("profile")) - names)
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
    return catalog


ID_CATALOG = _make_id_catalog()


def _devices_for_profile(profile: str) -> set[str]:
    return {"release": {"nvidia", "llvmpipe"},
            "sanitize": {"nvidia", "llvmpipe"},
            "software": {"llvmpipe"},
            "hardware": {"nvidia"}}.get(profile, set())


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
            "source_fingerprint": gate.get("source_fingerprint"),
            "git_revision": gate.get("git_revision"),
            "completed": gate.get("completed"), "scope_passed": scope_passed,
            "findings": findings, "families": compact,
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


def _make_id_index(stage_audits: list[dict], *, release_eligible: bool,
                   all_refs: list[dict]) -> dict:
    actual_tests = set().union(*(set(stage.get("ctest", {}).get("test_names", []))
                                 for stage in stage_audits)) if stage_audits else set()
    cpu_baselines = [stage for stage in stage_audits
                     if stage.get("families", {}).get("cpu_baseline")]
    indexed = {}
    for identifier, spec in ID_CATALOG.items():
        family = spec["evidence_family"]
        related = spec["tests"]
        executed = sorted(set(related) & actual_tests)
        missing_tests = sorted(set(related) - actual_tests)
        applicable_tests = set().union(*(_applicable_tests(related, stage.get("profile", ""))
                                         for stage in stage_audits)) if stage_audits else set()
        missing_applicable = sorted(applicable_tests - actual_tests)
        selected_refs = []
        cpu_hashes = []
        scope_profiles = []
        failed_applicable_scope = any(
            not stage.get("scope_passed") and
            bool(_applicable_tests(related, stage.get("profile", "")) & set(related))
            for stage in stage_audits)
        for stage in stage_audits:
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
                for stage in stage_audits) and status == "passed_for_profile_scope"):
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


def build_report(stage_dirs: list[Path], *, require_release: bool = False,
                 verify_fingerprint: bool = True) -> dict:
    gates = []
    for stage_dir in stage_dirs:
        gate = audit_stage_gate(stage_dir, verify_fingerprint=verify_fingerprint)
        gates.append(audit_stage_evidence(gate))
    global_findings = []
    seen_profiles = {}
    for stage in gates:
        profile = stage.get("profile")
        if profile in seen_profiles:
            _finding(global_findings, "duplicate-profile-stage",
                     f"more than one stage supplied profile {profile}",
                     stages=[seen_profiles[profile], stage.get("stage")])
        else:
            seen_profiles[profile] = stage.get("stage")
    fingerprints = {stage.get("source_fingerprint") for stage in gates if stage.get("source_fingerprint")}
    if len(fingerprints) > 1:
        _finding(global_findings, "composite-source-mismatch",
                 "all combined stages must have exactly the same fresh source_fingerprint",
                 fingerprints=sorted(fingerprints))
    required_profiles = {"release", "sanitize", "offline", "software"}
    supplied_profiles = {stage.get("profile") for stage in gates}
    missing_profiles = sorted(required_profiles - supplied_profiles)
    invalid_stages = [stage.get("stage") for stage in gates if not stage.get("scope_passed")]
    if invalid_stages:
        _finding(global_findings, "stage-scope-failed", "one or more gate stages failed their evidence audit",
                 stages=invalid_stages)
    release_eligible = (not missing_profiles and not invalid_stages and not global_findings
                        and len(fingerprints) == 1 and all(
                            sum(stage.get("profile") == profile for stage in gates) == 1
                            for profile in required_profiles))
    release_blockers = []
    if missing_profiles:
        release_blockers.append({"code": "required-profiles-missing",
                                 "message": "release requires release, sanitize, offline, and software stages",
                                 "profiles": missing_profiles})
    if invalid_stages:
        release_blockers.append({"code": "stage-scope-incomplete",
                                 "message": "one or more complete fresh gate scopes did not pass",
                                 "stages": invalid_stages})
    if global_findings:
        release_blockers.extend(global_findings)
    if require_release and not release_eligible and not missing_profiles:
        _finding(global_findings, "release-eligibility-required",
                 "--require-release was requested but the complete four-profile matrix is not eligible")
        release_blockers.extend(global_findings[-1:])
    all_refs = []
    for stage in gates:
        for ref in stage.get("raw_artifacts", []):
            all_refs.append({"stage": stage.get("stage"), **ref})
    id_index = _make_id_index(gates, release_eligible=release_eligible, all_refs=all_refs)
    if release_eligible and any(row.get("status") != "passed" for row in id_index.values()):
        release_eligible = False
        release_blockers.append({"code": "id-index-incomplete",
                                 "message": "one or more V0.1–V6.4 audit rows lacks CPU/test/raw evidence"})
        for row in id_index.values():
            row["release_status"] = "not_eligible"
            if row["status"] == "passed":
                row["status"] = "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": next(iter(fingerprints)) if len(fingerprints) == 1 else None,
        "profiles_required_for_release": sorted(required_profiles),
        "profiles_supplied": sorted(profile for profile in supplied_profiles if profile),
        "release_eligible": release_eligible,
        "release_decision": {
            "decision": "release-eligible-opt-in" if release_eligible else "no-go-or-incomplete",
            "default_backend": "CPU",
            "scope": "existing static FP32 support matrix and actually tested physical devices/ICDs/drivers only",
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
                        help="return failure unless the complete four-profile matrix is eligible")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.stage, require_release=args.require_release)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report = {"schema_version": SCHEMA_VERSION, "kind": REPORT_KIND,
                  "generated_at": datetime.now(timezone.utc).isoformat(),
                  "release_eligible": False,
                  "blocking_findings": [{"code": "audit-failed-closed", "message": str(error)}],
                  "id_index": {identifier: {"status": "blocked"} for identifier in ID_CATALOG}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"release evidence report: {args.output}")
    errors = report.get("blocking_findings", [])
    if errors:
        print(f"blocking findings: {len(errors)}", file=sys.stderr)
    return 1 if errors or (args.require_release and not report.get("release_eligible")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
