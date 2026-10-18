#!/usr/bin/env python3
"""Synthetic fail-closed tests for VA2 helper evidence auditing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import device_helper_evidence as audit
import run_device_helper as producer


class DeviceHelperEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="va2-helper-evidence-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.evidence = self.stage / "test/Vulkan/evidence/va2"
        self.evidence.mkdir(parents=True)
        self.source_root = self.root / "source"
        (self.source_root / "test/Vulkan").mkdir(parents=True)
        (self.source_root / "test/Vulkan/gpu_func_add_probe.mlir").write_text(
            "// synthetic Add source fixture\n", encoding="utf-8")
        self.tool_paths = {}
        for role in ("mlir-opt", "spirv-val", "spirv-dis"):
            path = self.write(f"tools/{role}", f"#!/bin/sh\nprintf '%s\\n' '{role} test'\n")
            path.chmod(0o755)
            self.tool_paths[role] = path
        self.extractor = self.write("tools/vulkan_v0_spirv_extract", b"test-extractor")
        self.write("CMakeCache.txt",
                   f"CMAKE_HOME_DIRECTORY:INTERNAL={self.source_root}\n"
                   f"NCNN_MLIR_OPT_EXECUTABLE:FILEPATH={self.tool_paths['mlir-opt']}\n"
                   f"MLIR_OPT:FILEPATH={self.tool_paths['mlir-opt']}\n"
                   f"NCNN_SPIRV_VAL_EXECUTABLE:FILEPATH={self.tool_paths['spirv-val']}\n"
                   f"NCNN_SPIRV_DIS_EXECUTABLE:FILEPATH={self.tool_paths['spirv-dis']}\n"
                   f"V0_LLVMPIPE_ICD_PATH:FILEPATH={self.stage / 'test/Vulkan/evidence/va2/devices/llvmpipe/icd.json'}\n"
                   f"V0_NVIDIA_EGL_LIBRARY:FILEPATH={self.stage / 'test/Vulkan/evidence/va2/devices/nvidia/libEGL_nvidia.so'}\n")
        self.compile_fixture_index = 0
        self.product_fixture_index = 0
        self.write_stage_state()

    def write(self, relative: str, data: bytes | str) -> Path:
        path = self.stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        return path

    def write_json(self, relative: str, value: object) -> Path:
        return self.write(relative, json.dumps(value, sort_keys=True) + "\n")

    def write_stage_state(self, **overrides) -> dict:
        state = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_gate",
            "stage_id": "VA2",
            "profile": "release",
            "source_fingerprint": "a" * 64,
            "host_execution_role": "legacy-product",
            "host_aot_candidate": False,
            "toolchain_identities": [
                {"role": role, "path": str(path.resolve()),
                 "sha256": audit.sha256_file(path),
                 "version": subprocess.run([str(path), "--version"],
                                            check=True, capture_output=True,
                                            text=True).stdout.strip()}
                for role, path in (("mlir-opt", self.tool_paths["mlir-opt"]),
                                   ("spirv-val", self.tool_paths["spirv-val"]),
                                   ("spirv-dis", self.tool_paths["spirv-dis"]))],
        }
        state.update(overrides)
        self.write_json("gate-state.json", state)
        return state

    def raw_entry(self, path: Path, role: str) -> dict:
        origin = path.resolve()
        if not origin.is_relative_to(self.stage):
            snapshot = self.write(
                f"test/Vulkan/evidence/va2/raw-inputs/{role}-{origin.name}",
                origin.read_bytes())
            path = snapshot
        return {"path": path.relative_to(self.stage).as_posix(),
                "sha256": audit.sha256_file(path), "size_bytes": path.stat().st_size,
                "origin": str(origin), "role": role}

    def record(self, identity: str, raw: list[dict], **fields) -> Path:
        value = {"schema_version": 1, "kind": audit.TEST_EVIDENCE_KIND,
                 "identity": identity, "status": "passed", "raw": raw}
        value.update(fields)
        return self.write_json(
            f"test/Vulkan/evidence/va2/tests/{identity}.json", value)

    def test_stage_identity_requires_va2_and_legacy_host_role(self) -> None:
        _, profile, fingerprint, gate_bound = audit._read_stage_identity(self.stage)
        self.assertTrue(gate_bound)
        self.assertEqual("release", profile)
        self.assertEqual("a" * 64, fingerprint)
        for field, value in (("stage_id", None), ("stage_id", ""),
                             ("stage_id", "VA9"), ("host_aot_candidate", True),
                             ("source_fingerprint", "z" * 64),
                             ("profile", "offline")):
            with self.subTest(field=field, value=value):
                self.write_stage_state(**{field: value})
                with self.assertRaises(audit.EvidenceError):
                    audit._read_stage_identity(self.stage)

    def test_va3_technical_identity_does_not_claim_va2_matrix_qualification(self) -> None:
        for profile in ("release", "sanitize"):
            with self.subTest(profile=profile):
                self.write_stage_state(stage_id="VA3", profile=profile,
                                       candidate_role="lifecycle-only")
                state, actual_profile, fingerprint, bound = audit._read_stage_identity(self.stage)
                self.assertTrue(bound)
                self.assertEqual("VA3", state["stage_id"])
                self.assertEqual(profile, actual_profile)
                self.assertEqual("a" * 64, fingerprint)
                with self.assertRaisesRegex(audit.EvidenceError, "bound VA2"):
                    audit._require_va2_matrix_stage(state, actual_profile, bound)

        for role in (None, "", "model-run", True):
            with self.subTest(role=role):
                self.write_stage_state(stage_id="VA3", candidate_role=role)
                with self.assertRaisesRegex(audit.EvidenceError, "lifecycle-only"):
                    audit._read_stage_identity(self.stage)
        self.write_stage_state(stage_id="VA3", profile="offline",
                               candidate_role="lifecycle-only")
        with self.assertRaisesRegex(audit.EvidenceError, "Release or sanitize"):
            audit._read_stage_identity(self.stage)

        self.write_stage_state(stage_id="VA2")
        state, profile, _, bound = audit._read_stage_identity(self.stage)
        audit._require_va2_matrix_stage(state, profile, bound)
        with self.assertRaises(audit.EvidenceError):
            audit._require_va2_matrix_stage(state, profile, False)

    def test_unbound_development_identity_is_unqualified_not_fabricated_va2(self) -> None:
        (self.stage / "gate-state.json").unlink()
        self.write("CMakeCache.txt",
                   "CMAKE_BUILD_TYPE:STRING=\nCOMPILER_SANITIZERS:STRING=\n")
        state, profile, fingerprint, bound = audit._read_stage_identity(self.stage)
        self.assertIsNone(state)
        self.assertEqual("development", profile)
        self.assertIsNone(fingerprint)
        self.assertFalse(bound)

        self.write("CMakeCache.txt",
                   "CMAKE_BUILD_TYPE:STRING=Debug\n"
                   "COMPILER_SANITIZERS:STRING=address;undefined\n")
        state, profile, fingerprint, bound = audit._read_stage_identity(self.stage)
        self.assertIsNone(state)
        self.assertEqual("sanitize", profile)
        self.assertIsNone(fingerprint)
        self.assertFalse(bound)

    def test_new_raw_snapshot_preserves_bytes_not_old_build_mtime(self) -> None:
        source = self.write("build-input.bin", b"build product bytes")
        os.utime(source, (1, 1))
        record = audit._copy_raw(source, self.stage, self.evidence,
                                 Path("snapshots/build-input.bin"))
        snapshot = self.stage / record["path"]
        self.assertEqual(source.read_bytes(), snapshot.read_bytes())
        self.assertEqual(audit.sha256_file(source), record["sha256"])
        self.assertGreater(snapshot.stat().st_mtime_ns, source.stat().st_mtime_ns)
        self.assertEqual(1, source.stat().st_mtime)

    def test_icd_bare_library_uses_exact_system_loader_cache_entry(self) -> None:
        driver = self.write("system-libraries/libvulkan_lvp.so", b"driver fixture")
        manifest = self.stage / "manifests/icd.json"
        cache = (f"\tlibvulkan_lvp.so (libc6,x86-64) => {driver}\n"
                 "\tlibvulkan_lvp.so (libc6,AArch64) => /unavailable/other.so\n")
        result = subprocess.CompletedProcess([], 0, cache.encode(), b"")
        with patch.dict("os.environ", {"LD_LIBRARY_PATH": ""}):
            with patch.object(audit.shutil, "which", return_value="/sbin/ldconfig"):
                with patch.object(audit, "_run", return_value=result):
                    self.assertEqual(driver, audit._icd_library_path(
                        manifest, "libvulkan_lvp.so"))
                    with self.assertRaises(audit.EvidenceError):
                        audit._icd_library_path(manifest, "different-driver.so")
        relative = self.write("manifests/drivers/libfixture.so", b"relative driver")
        self.assertEqual(relative, audit._icd_library_path(manifest, "drivers/libfixture.so"))
        self.assertEqual(driver, audit._icd_library_path(manifest, str(driver)))

    def test_gate_tool_primary_role_is_not_ambiguous_with_lit_alias(self) -> None:
        state = self.write_stage_state()
        primary = next(item for item in state["toolchain_identities"]
                       if item["role"] == "mlir-opt")
        state["toolchain_identities"].append({**primary, "role": "lit-mlir-opt"})
        self.write_json("gate-state.json", state)
        self.assertEqual(Path(primary["path"]).resolve(),
                         audit._expected_gate_tool(self.stage, "mlir-opt"))
        state["toolchain_identities"].append(dict(primary))
        self.write_json("gate-state.json", state)
        with self.assertRaises(audit.EvidenceError):
            audit._expected_gate_tool(self.stage, "mlir-opt")

    def test_missing_first_config_file_api_is_refreshed_without_skipping_audit(self) -> None:
        build = self.stage / "first-config"
        logs = self.stage / "logs"
        logs.mkdir()
        calls = []

        def configure(label, argv, log_root):
            calls.append(argv)
            self.assertEqual(["cmake", "-S", str(self.source_root), "-B", str(build)], argv)
            query = build / ".cmake/api/v1/query"
            self.assertTrue((query / "codemodel-v2").is_file())
            self.assertTrue((query / "cache-v2").is_file())
            reply = build / ".cmake/api/v1/reply"
            reply.mkdir()
            (reply / "index-fixture.json").write_text("{}\n", encoding="utf-8")
            (log_root / f"{label}.json").write_text("{}\n", encoding="utf-8")

        with patch.object(producer, "run_logged", side_effect=configure):
            result = producer._ensure_file_api(build, self.source_root, "cmake", logs, "api")
            self.assertEqual(logs / "api.json", result)
            self.assertIsNone(producer._ensure_file_api(
                build, self.source_root, "cmake", logs, "api"))
        self.assertEqual(1, len(calls))

    def test_product_only_summary_is_required_namespace_not_all_tests(self) -> None:
        expected = sorted(audit.PRODUCT_ONLY_REGISTERED_TESTS)
        inventory = self.write_json("product-inventory.json", {
            "tests": [{"name": name} for name in [*expected, "unrelated-compiler-test"]]})
        self.assertEqual(expected, producer._check_product_only_inventory(inventory))
        for names in (expected[:-1], [*expected, sorted(audit.DEVICE_TESTS)[0]]):
            with self.subTest(names=names):
                inventory = self.write_json("product-inventory.json", {
                    "tests": [{"name": name} for name in names]})
                with self.assertRaises(RuntimeError):
                    producer._check_product_only_inventory(inventory)

    def test_raw_inventory_records_actual_sizes_and_unique_paths(self) -> None:
        path = self.write("raw/artifact.bin", b"actual compiled bytes")
        entry = {"path": "raw/artifact.bin", "sha256": audit.sha256_file(path),
                 "role": "target artifact"}
        inventory = audit._canonical_raw_inventory(
            self.stage, [entry, {**entry, "role": "actual link input"}])
        self.assertEqual(1, len(inventory))
        self.assertEqual(len(path.read_bytes()), inventory[0]["size_bytes"])
        for extra in ({"sha256": "0" * 64}, {"size_bytes": True},
                      {"size_bytes": 1}):
            with self.subTest(extra=extra):
                with self.assertRaises(audit.EvidenceError):
                    audit._canonical_raw_inventory(self.stage, [{**entry, **extra}])

    def test_compiler_identity_uses_actual_version_when_cache_omits_id(self) -> None:
        compiler = self.write("tools/cxx", b"compiler fixture")
        cache = {"CMAKE_CXX_COMPILER": str(compiler)}
        cases = ((b"Debian clang version 21.1.8 (build)\nTarget: x86_64-linux-gnu\n",
                  "Clang", "21.1.8"),
                 (b"c++ (Debian 14.2.0-19) 14.2.0\n"
                  b"Copyright (C) 2024 Free Software Foundation, Inc.\n",
                  "GNU", "14.2.0"))
        for output, identity, version in cases:
            with self.subTest(compiler_id=identity):
                result = subprocess.CompletedProcess([], 0, output, b"")
                with patch.object(audit, "_run", return_value=result):
                    actual, raw = audit._verified_compiler_cache(cache, "fixture")
                self.assertEqual(identity, actual["CMAKE_CXX_COMPILER_ID"])
                self.assertEqual(version, actual["CMAKE_CXX_COMPILER_VERSION"])
                self.assertEqual(output, raw)
        self.assertNotIn("CMAKE_CXX_COMPILER_ID", cache)
        self.assertNotIn("CMAKE_CXX_COMPILER_VERSION", cache)

    def test_compiler_identity_rejects_cache_contradiction_or_unknown_tool(self) -> None:
        compiler = self.write("tools/cxx", b"compiler fixture")
        cache = {"CMAKE_CXX_COMPILER": str(compiler)}
        result = subprocess.CompletedProcess(
            [], 0, b"Debian clang version 21.1.8 (build)\n", b"")
        for extra in ({"CMAKE_CXX_COMPILER_ID": "GNU"},
                      {"CMAKE_CXX_COMPILER_VERSION": "20.1.0"}):
            with self.subTest(extra=extra):
                with patch.object(audit, "_run", return_value=result):
                    with self.assertRaises(audit.EvidenceError):
                        audit._verified_compiler_cache({**cache, **extra}, "fixture")
        unknown = subprocess.CompletedProcess([], 0, b"not a compiler 21.1.8\n", b"")
        with patch.object(audit, "_run", return_value=unknown):
            with self.assertRaises(audit.EvidenceError):
                audit._verified_compiler_cache(cache, "fixture")

    def test_forbidden_dependency_classifier_distinguishes_project_namespace(self) -> None:
        for value in ("/opt/llvm-21/include/llvm/IR/Module.h",
                      "/opt/MLIR/include/mlir/IR/MLIRContext.h",
                      "/usr/lib/libLLVM-21.so", "GTest::gtest_main",
                      "VulkanExecutionPlan.cpp", "libNCNNVulkanPlanSupport.a"):
            with self.subTest(value=value):
                self.assertTrue(audit._is_forbidden_path(value))
        self.assertFalse(audit._is_forbidden_path(
            "/work/compiler/include/ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp"))
        self.assertFalse(audit._is_forbidden_path(
            "/usr/lib/llvm-21/lib/clang/21/include/stddef.h"))
        self.assertFalse(audit._is_forbidden_path(
            "/usr/lib/llvm-21/lib/clang/21/lib/linux/libclang_rt.asan.so"))
        self.assertFalse(audit._is_forbidden_path(
            "/usr/lib/llvm-21/lib/clang/21/share/asan_ignorelist.txt"))
        self.assertTrue(audit._is_forbidden_path(
            "/usr/lib/llvm-21/lib/clang/21/share/project_ignorelist.txt"))
        self.assertTrue(audit._is_forbidden_path(
            "/usr/lib/llvm-21/lib/clang/21/lib/linux/libLLVM.so"))
        self.assertFalse(audit._is_forbidden_path(
            "/usr/lib/llvm-21/lib/clang/21/lib/linux/libclang_rt.builtins-x86_64.a"))

    def test_test_record_requires_exact_identity_status_and_raw_hashes(self) -> None:
        identity = "vulkan-device-helper-unit"
        raw_path = self.write("test/Vulkan/evidence/va2/unit/result.json", "{}\n")
        record = self.record(identity, [self.raw_entry(raw_path, "result")])
        refs = []
        parsed, roles = audit._test_raw_entries(self.stage, identity, refs)
        self.assertEqual("passed", parsed["status"])
        self.assertEqual({"result"}, set(roles))
        self.assertGreaterEqual(len(refs), 2)

        value = json.loads(record.read_text())
        value["identity"] = "vulkan-device-helper-no-device"
        record.write_text(json.dumps(value))
        with self.assertRaises(audit.EvidenceError):
            audit._test_raw_entries(self.stage, identity, [])

    def test_test_record_rejects_skips_false_status_bool_returncode_and_stale_source(self) -> None:
        identity = "vulkan-device-helper-unit"
        raw_path = self.write("test/Vulkan/evidence/va2/unit/result.json", "{}\n")
        raw = [self.raw_entry(raw_path, "result")]
        cases = (
            {"status": "skipped"}, {"status": True}, {"returncode": True},
            {"schema_version": True}, {"schema_version": 1.0},
            {"schema_version": "1"},
            {"profile": "sanitize"}, {"source_fingerprint": "b" * 64},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.record(identity, raw, **overrides)
                with self.assertRaises(audit.EvidenceError):
                    audit._test_raw_entries(self.stage, identity, [])

    def test_test_record_rejects_missing_tampered_and_escaping_raw(self) -> None:
        identity = "vulkan-device-helper-unit"
        record = self.record(identity, [])
        with self.assertRaisesRegex(audit.EvidenceError, "raw evidence"):
            audit._test_raw_entries(self.stage, identity, [])
        raw_path = self.write("test/Vulkan/evidence/va2/unit/result.json", "original\n")
        record = self.record(identity, [self.raw_entry(raw_path, "result")])
        raw_path.write_text("changed\n")
        with self.assertRaisesRegex(audit.EvidenceError, "hash mismatch"):
            audit._test_raw_entries(self.stage, identity, [])
        self.record(identity, [{"path": "../outside", "sha256": "0" * 64,
                                "role": "result"}])
        with self.assertRaises(audit.EvidenceError):
            audit._test_raw_entries(self.stage, identity, [])
        self.assertTrue(record.is_file())

    def test_unit_record_requires_actual_binary_command_and_bound_result(self) -> None:
        root = "test/Vulkan/evidence/va2/unit"
        binary = self.write(root + "/vulkan_device_helper_unit_test", b"unit binary")
        command = self.write_json(root + "/unit-command.json", {
            "argv": [str(binary)], "returncode": 0, "stdout": "passed\\n", "stderr": ""})
        result = self.write_json(root + "/result.json", {
            "schema_version": 1, "kind": "ncnn.vulkan_device_helper_unit_result",
            "status": "passed", "returncode": 0, "stdout": "passed\\n", "stderr": ""})
        identity = "vulkan-device-helper-unit"
        self.record(identity, [self.raw_entry(binary, "unit-binary"),
                               self.raw_entry(command, "unit-command-log"),
                               self.raw_entry(result, "result")])
        parsed, raw = audit._test_raw_entries(self.stage, identity, [])
        audit._validate_unit_test(self.stage, raw, parsed, identity)
        command_data = json.loads(command.read_text())
        command_data["argv"] = ["/tmp/other-unit"]
        command.write_text(json.dumps(command_data))
        with self.assertRaisesRegex(audit.EvidenceError, "hash mismatch"):
            audit._test_raw_entries(self.stage, identity, [])

    def _make_add_record(self, device: str = "nvidia") -> dict[str, Path]:
        identity = f"vulkan-device-helper-add-{device}"
        base = f"test/Vulkan/evidence/va2/devices/{device}"
        lhs = [audit._f32(float(index % 13 - 5) / 8) for index in range(257)]
        rhs = [audit._f32(float(index % 7 - 2) / 4) for index in range(257)]
        golden = [audit._f32(left + right) for left, right in zip(lhs, rhs)]
        paths: dict[str, Path] = {}
        tensor_roles = (("lhs-f32", lhs), ("rhs-f32", rhs),
                        ("cpu-aot-output-f32", golden),
                        ("helper-gpu-output-f32", golden),
                        ("helper-gpu-repeat-f32", golden),
                        ("scalar-golden-f32", golden))
        for role, values in tensor_roles:
            paths[role] = self.write(f"{base}/{role}.bin", struct.pack("<257f", *values))
        paths["runner-binary"] = self.write(f"{base}/vulkan_device_helper_runner", b"runner")
        paths["cpu-aot-library"] = self.write(f"{base}/libv0_add.so", b"cpu library")
        paths["gpu-binary-mlir"] = self.write(f"{base}/gpu.binary.mlir", "gpu.binary @v0_add\n")
        paths["source-kernel"] = self.source_root / "test/Vulkan/gpu_func_add_probe.mlir"
        paths["spirv"] = self.write(f"{base}/add.spv", b"\x03\x02\x23\x07" * 5)
        paths["spirv-disassembly"] = self.write(
            f"{base}/add.spvasm", 'OpEntryPoint GLCompute %1 "add"\n')
        icd_library_name = "libEGL_nvidia.so" if device == "nvidia" else "vulkan_lvp.so"
        icd_library = self.write(f"{base}/{icd_library_name}", b"icd")
        manifest_relative = ("test/Vulkan/nvidia-egl-icd.json" if device == "nvidia" else
                             f"{base}/icd.json")
        paths["icd-manifest"] = self.write_json(
            manifest_relative, {"file_format_version": "1.0.0",
                               "ICD": {"library_path": str(icd_library),
                                       "api_version": "1.1.0"}})
        paths["mlir-opt-binary"] = self.tool_paths["mlir-opt"]
        paths["spirv-val-binary"] = self.tool_paths["spirv-val"]
        paths["spirv-dis-binary"] = self.tool_paths["spirv-dis"]
        paths["spirv-extractor-binary"] = self.extractor

        def write_command(role: str, argv: list[str], stdout: str = "") -> Path:
            return self.write_json(f"{base}/{role}.json",
                                   {"argv": argv, "returncode": 0,
                                    "stdout": stdout, "stderr": ""})

        paths["mlir-opt-version-log"] = write_command(
            "mlir-opt-version", [str(self.tool_paths["mlir-opt"]), "--version"],
            "mlir-opt test\n")
        paths["spirv-val-version-log"] = write_command(
            "spirv-val-version", [str(self.tool_paths["spirv-val"]), "--version"],
            "spirv-val test\n")
        paths["gpu-func-add-lowering-log"] = write_command(
            "gpu-func-add-lowering", [str(self.tool_paths["mlir-opt"]),
                                      str(paths["source-kernel"]),
                                      "--test-vulkan-runner-pipeline", "--verify-each",
                                      "-o", str(paths["gpu-binary-mlir"])])
        paths["spirv-extraction-log"] = write_command(
            "spirv-extraction", [str(self.extractor), str(paths["gpu-binary-mlir"]),
                                 "-o", str(paths["spirv"])])
        paths["spirv-disassembly-log"] = write_command(
            "spirv-disassembly", [str(self.tool_paths["spirv-dis"]),
                                  str(paths["spirv"]), "-o",
                                  str(paths["spirv-disassembly"])])
        paths["spirv-validation-log"] = self.write_json(
            f"{base}/spirv-validation.json",
            {"kind": "ncnn.spirv_tool_invocation",
             "argv": [str(self.tool_paths["spirv-val"]), "--target-env",
                      "vulkan1.1", str(paths["spirv"])],
             "returncode": 0, "stdout": "", "stderr": ""})
        vendor_id = 0x10DE if device == "nvidia" else 0x10005
        device_name = ("NVIDIA GeForce RTX 4060 Ti" if device == "nvidia" else
                       "llvmpipe (LLVM 21.1.8, 256 bits)")
        paths["result"] = self.write_json(
            f"{base}/result.json",
            {"schema_version": 1, "kind": "ncnn.vulkan_device_helper_add_result",
             "fixture_id": "va2-f32-add-257-gpu-func-v1", "status": "passed",
             "device": device, "device_name": device_name, "api_version": 4198400,
             "driver_version": 1, "vendor_id": vendor_id, "device_id": 1,
             "elements": 257, "local_size": 64, "dispatch_groups_x": 5,
             "validation_enabled": True, "validation_errors": 0,
             "wait_failure_recovered": True, "max_cpu_golden_abs": 0.0,
             "max_gpu_golden_abs": 0.0, "max_repeat_abs": 0.0})
        runner_argv = [str(paths["runner-binary"]), "--mode", "add", "--shader",
                       str(paths["spirv"]), "--cpu-library", str(paths["cpu-aot-library"]),
                       "--cpu-symbol", "v0_add", "--device-match",
                       "NVIDIA" if device == "nvidia" else "llvmpipe", "--lhs",
                       str(paths["lhs-f32"]), "--rhs", str(paths["rhs-f32"]),
                       "--cpu-output", str(paths["cpu-aot-output-f32"]), "--gpu-output",
                       str(paths["helper-gpu-output-f32"]), "--gpu-repeat-output",
                       str(paths["helper-gpu-repeat-f32"]), "--golden",
                       str(paths["scalar-golden-f32"]), "--result", str(paths["result"]),
                       "--validation"]
        icd_path = str(paths["icd-manifest"].resolve())
        paths["validation-log"] = self.write_json(
            f"{base}/runner.json", {"argv": runner_argv, "returncode": 0,
                                     "stdout": "device test passed\n", "stderr": "",
                                     "environment_overrides": {
                                         "VK_ICD_FILENAMES": icd_path,
                                         "VK_DRIVER_FILES": icd_path}})
        raw = [self.raw_entry(file, role) for role, file in paths.items()]
        self.record(identity, raw)
        return paths

    def _audit_add_record(self, device: str) -> dict[str, Path]:
        identity = f"vulkan-device-helper-add-{device}"
        paths = self._make_add_record(device)
        target_json = {"artifacts": [{"path": str(self.extractor)}]}
        with patch.object(audit, "_file_api_target",
                          return_value=({}, target_json, self.extractor)):
            record, raw = audit._test_raw_entries(self.stage, identity, [])
            audit._validate_device_test(self.stage, identity, raw, record)
        return paths

    def test_add_evidence_recomputes_257_element_fp32_results_and_device_identity(self) -> None:
        for device in ("nvidia", "llvmpipe"):
            with self.subTest(device=device):
                self._audit_add_record(device)

    def _validate_add_fixture(self, record: dict, raw: dict[str, Path],
                               identity: str = "vulkan-device-helper-add-nvidia") -> None:
        target_json = {"artifacts": [{"path": str(self.extractor)}]}
        with patch.object(audit, "_file_api_target",
                          return_value=({}, target_json, self.extractor)):
            audit._validate_device_test(self.stage, identity, raw, record)

    def test_add_evidence_rejects_output_mismatch_bool_alias_missing_raw_and_bad_spirv(self) -> None:
        self._make_add_record()
        identity = "vulkan-device-helper-add-nvidia"
        target_json = {"artifacts": [{"path": str(self.extractor)}]}
        with patch.object(audit, "_file_api_target",
                          return_value=({}, target_json, self.extractor)):
            record, raw = audit._test_raw_entries(self.stage, identity, [])
        gpu_path = raw["helper-gpu-output-f32"]
        gpu_path.write_bytes(struct.pack("<257f", *([7.0] * 257)))
        with self.assertRaisesRegex(audit.EvidenceError, "mismatch"):
            self._validate_add_fixture(record, raw)

        gpu_path.write_bytes(raw["cpu-aot-output-f32"].read_bytes())
        result_path = raw["result"]
        result = json.loads(result_path.read_text())
        result["validation_errors"] = False
        result_path.write_text(json.dumps(result))
        with self.assertRaises(audit.EvidenceError):
            self._validate_add_fixture(record, raw)

        result["validation_errors"] = 0
        result["wait_failure_recovered"] = False
        result_path.write_text(json.dumps(result))
        with self.assertRaises(audit.EvidenceError):
            self._validate_add_fixture(record, raw)

        result["wait_failure_recovered"] = True
        result_path.write_text(json.dumps(result))
        repeat_path = raw.pop("helper-gpu-repeat-f32")
        with self.assertRaisesRegex(audit.EvidenceError, "required raw roles"):
            self._validate_add_fixture(record, raw)
        raw["helper-gpu-repeat-f32"] = repeat_path
        invalid_invocation = json.loads(raw["spirv-validation-log"].read_text())
        invalid_invocation["argv"][-1] = "/tmp/unrelated.spv"
        raw["spirv-validation-log"].write_text(json.dumps(invalid_invocation))
        with self.assertRaisesRegex(audit.EvidenceError, "argv"):
            self._validate_add_fixture(record, raw)

    def test_no_device_evidence_requires_actual_command_and_unavailable_icd(self) -> None:
        root = "test/Vulkan/evidence/va2/no-device"
        output = self.stage / root
        output.mkdir(parents=True, exist_ok=True)
        runner = self.write(root + "/vulkan_device_helper_runner", b"runner")
        missing_icd = output / "missing-test-icd.so"
        manifest = self.write_json(root + "/unavailable-icd.json", {
            "file_format_version": "1.0.0",
            "ICD": {"library_path": str(missing_icd), "api_version": "1.1.0"},
        })
        diagnostic = "No Vulkan physical devices were found"
        result = self.write_json(root + "/result.json", {
            "schema_version": 1, "kind": "ncnn.vulkan_device_helper_no_device_result",
            "helper_create_succeeded": False, "status": 2, "diagnostic": diagnostic,
        })
        stdout = self.write(root + "/stdout.txt", diagnostic + "\n")
        stderr = self.write(root + "/stderr.txt", "")
        command_log = self.write_json(root + "/create-command.json", {
            "argv": [str(runner), "--mode", "no-device", "--result", str(result)],
            "returncode": 0, "stdout": diagnostic + "\n", "stderr": "",
            "environment_overrides": {"VK_ICD_FILENAMES": str(manifest.resolve()),
                                      "VK_DRIVER_FILES": str(manifest.resolve())},
        })
        no_device_identity = "vulkan-device-helper-no-device"
        self.record(no_device_identity, [self.raw_entry(path, role) for path, role in (
            (runner, "runner-binary"), (manifest, "icd-manifest"),
            (result, "result"), (stdout, "stdout"), (stderr, "stderr"),
            (command_log, "create-command-log"))])
        record, raw = audit._test_raw_entries(self.stage, no_device_identity, [])
        audit._validate_no_device_test(self.stage, raw, record)
        invalid = json.loads(result.read_text())
        invalid["status"] = True
        result.write_text(json.dumps(invalid))
        with self.assertRaises(audit.EvidenceError):
            audit._validate_no_device_test(self.stage, raw, record)
        invalid["status"] = 2
        invalid["skipped"] = True
        result.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(audit.EvidenceError, "skip"):
            audit._validate_no_device_test(self.stage, raw, record)
        invalid.pop("skipped")
        result.write_text(json.dumps(invalid))
        missing_icd.write_bytes(b"not an ICD")
        with self.assertRaisesRegex(audit.EvidenceError, "available ICD library"):
            audit._validate_no_device_test(self.stage, raw, record)
        missing_icd.unlink()
        command = json.loads(command_log.read_text())
        command["environment_overrides"]["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
        command_log.write_text(json.dumps(command))
        with self.assertRaisesRegex(audit.EvidenceError, "both Vulkan loader variables"):
            audit._validate_no_device_test(self.stage, raw, record)

    def _write_product_root(self, names: list[str]) -> Path:
        self.product_fixture_index += 1
        root = self.stage / f"test/Vulkan/device-helper-product-only-{self.product_fixture_index}"
        root.mkdir(parents=True)
        (root / "CMakeCache.txt").write_text(
            "NCNN_ENABLE_VULKAN_PRODUCT:BOOL=ON\n"
            "NCNN_ENABLE_VULKAN_V0_TESTS:BOOL=OFF\n"
            "BUILD_TESTING:BOOL=ON\n", encoding="utf-8")
        self.product_names = names
        return root

    def _fake_product_ctest(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        if "--show-only=json-v1" in command:
            payload = json.dumps({"tests": [{"name": name} for name in self.product_names]})
            return subprocess.CompletedProcess(command, 0, payload.encode(), b"")
        junit_path = Path(command[command.index("--output-junit") + 1])
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        suites = ET.Element("testsuites")
        suite = ET.SubElement(suites, "testsuite", tests=str(len(audit.PRODUCT_ONLY_EXECUTE_TESTS)),
                              failures="0", errors="0", skipped="0")
        for index, name in enumerate(audit.PRODUCT_ONLY_EXECUTE_TESTS):
            case = ET.SubElement(suite, "testcase", name=name, classname="CTest")
            if getattr(self, "product_skip", False) and index == 0:
                ET.SubElement(case, "skipped", message="unexpected skip")
        junit_path.write_bytes(ET.tostring(suites, encoding="utf-8"))
        product_root = Path(command[command.index("--test-dir") + 1])
        lasttest = product_root / "Testing/Temporary/LastTest.log"
        lasttest.parent.mkdir(parents=True, exist_ok=True)
        lasttest.write_text("".join(
            f"{index}/{len(audit.PRODUCT_ONLY_EXECUTE_TESTS)} Testing: {name}\nOutput:\nTest Passed.\n\n"
            for index, name in enumerate(audit.PRODUCT_ONLY_EXECUTE_TESTS, 1)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"passed\n", b"")

    def test_sanitizer_argv_requires_instrumentation_and_rejects_disablers(self) -> None:
        release_compile = ["/usr/bin/c++", "-I", "/project/include", "-c", "source.cpp"]
        audit._assert_sanitizer_flags(release_compile, "release", "release compile")
        with self.assertRaisesRegex(audit.EvidenceError, "non-SAN"):
            audit._assert_sanitizer_flags(
                [*release_compile, "-fsanitize=address"], "release", "release compile")

        sanitized = ["/usr/lib/llvm-21/bin/clang++", "-fsanitize=address,undefined",
                     "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-c", "source.cpp"]
        audit._assert_sanitizer_flags(sanitized, "sanitize", "SAN compile")
        for bad in ("-fno-sanitize=address", "-fsanitize-recover=undefined",
                    "-fomit-frame-pointer", "@unverified.rsp",
                    "-fsanitize-ignorelist=project.txt", "-fsanitize-blacklist=project.txt"):
            with self.subTest(bad=bad):
                with self.assertRaises(audit.EvidenceError):
                    audit._assert_sanitizer_flags([*sanitized, bad], "sanitize", "SAN compile")
        link = ["/usr/lib/llvm-21/bin/clang++", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", "-fno-omit-frame-pointer",
                "-shared-libasan", "-Wl,-rpath,/usr/lib/llvm-21/lib/clang/21/lib/linux",
                "-o", "runner"]
        audit._assert_sanitizer_flags(link, "sanitize", "SAN link", link=True,
                                      resource_dir="/usr/lib/llvm-21/lib/clang/21")
        with self.assertRaisesRegex(audit.EvidenceError, "shared ASan"):
            audit._assert_sanitizer_flags(
                [token for token in link if token != "-shared-libasan"],
                "sanitize", "SAN link", link=True,
                resource_dir="/usr/lib/llvm-21/lib/clang/21")

    def test_product_only_inventory_and_execution_require_exact_helper_set(self) -> None:
        def fake_child_tests(product_root, refs, identities):
            return {identity: {"record": {}, "raw_roles": []}
                    for identity in identities}

        root = self._write_product_root(list(audit.PRODUCT_ONLY_REGISTERED_TESTS))
        refs = []
        with patch.object(audit, "_run", side_effect=self._fake_product_ctest), \
             patch.object(audit, "_audit_test_records", side_effect=fake_child_tests):
            report = audit._check_product_only(
                self.stage, root, self.evidence, refs)
        self.assertEqual(sorted(audit.PRODUCT_ONLY_REGISTERED_TESTS),
                         report["registered_helper_tests"])
        self.assertEqual(sorted(audit.PRODUCT_ONLY_EXECUTE_TESTS),
                         report["executed_test_names"])

        root = self._write_product_root(list(audit.PRODUCT_ONLY_REGISTERED_TESTS[:-1]))
        with patch.object(audit, "_run", side_effect=self._fake_product_ctest):
            with self.assertRaisesRegex(audit.EvidenceError, "inventory mismatch"):
                audit._check_product_only(self.stage, root, self.evidence, [])

        contaminated = list(audit.PRODUCT_ONLY_REGISTERED_TESTS) + [
            "vulkan-device-helper-add-nvidia"]
        root = self._write_product_root(contaminated)
        with patch.object(audit, "_run", side_effect=self._fake_product_ctest):
            with self.assertRaisesRegex(audit.EvidenceError, "inventory mismatch"):
                audit._check_product_only(self.stage, root, self.evidence, [])

        root = self._write_product_root(list(audit.PRODUCT_ONLY_REGISTERED_TESTS))
        self.product_skip = True
        with patch.object(audit, "_run", side_effect=self._fake_product_ctest):
            with self.assertRaisesRegex(audit.EvidenceError, "JUnit"):
                audit._check_product_only(self.stage, root, self.evidence, [])
        self.product_skip = False

    def test_closure_manifest_rejects_missing_extra_and_malformed_targets(self) -> None:
        path = self.write_json("test/Vulkan/device-helper-target-link-closure.json", {
            "schema_version": 1, "kind": audit.CLOSURE_KIND,
            "targets": [{"name": audit.HELPER_TARGET}],
        })
        self.assertIn(audit.HELPER_TARGET,
                      audit._read_closure_manifest(self.stage, {audit.HELPER_TARGET}))
        with self.assertRaisesRegex(audit.EvidenceError, "target set"):
            audit._read_closure_manifest(self.stage, {audit.HELPER_TARGET, "polluted-target"})
        self.write_json(path.relative_to(self.stage).as_posix(), {
            "schema_version": 1, "kind": audit.CLOSURE_KIND,
            "targets": [{"name": audit.HELPER_TARGET}, {"name": "LLVMSupport"}],
        })
        with self.assertRaisesRegex(audit.EvidenceError, "target set"):
            audit._read_closure_manifest(self.stage, {audit.HELPER_TARGET})
        for schema in (2, True, 1.0, "1"):
            self.write_json(path.relative_to(self.stage).as_posix(), {
                "schema_version": schema, "kind": audit.CLOSURE_KIND, "targets": []})
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(audit.EvidenceError, "schema"):
                    audit._read_closure_manifest(self.stage, set())

    def _compile_fixture(self, *, include: str | None = None,
                         dep_header: str = "safe.hpp") -> tuple[Path, dict, dict]:
        self.compile_fixture_index += 1
        source_root = self.root / f"compile-source-{self.compile_fixture_index}"
        build_root = self.root / f"compile-build-{self.compile_fixture_index}"
        source_root.mkdir()
        build_root.mkdir()
        include_dir = (source_root / "llvm-21/include" if include is not None else
                       source_root / "include")
        include_dir.mkdir(parents=True)
        source = source_root / "helper.cpp"
        source.write_text(f'#include "{dep_header}"\nint helper() {{ return value(); }}\n')
        header = include_dir / dep_header
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("inline int value() { return 7; }\n")
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler is required for depfile audit tests")
        object_dir = build_root / "CMakeFiles/NCNNVulkanDeviceHelper.dir"
        object_dir.mkdir(parents=True)
        object_path = object_dir / "helper.cpp.o"
        depfile = object_dir / "helper.cpp.o.d"
        include_path = include_dir
        command = [compiler, "-I", str(include_path), "-MMD", "-MF", str(depfile),
                   "-c", str(source), "-o", str(object_path)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        (build_root / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root}\n", encoding="utf-8")
        compile_db = [{"directory": str(build_root), "file": str(source),
                       "arguments": command, "output": str(object_path)}]
        (build_root / "compile_commands.json").write_text(json.dumps(compile_db))
        target_json = {"name": audit.HELPER_TARGET, "type": "STATIC_LIBRARY",
                       "paths": {"source": str(source_root), "build": str(build_root)},
                       "sources": [{"path": str(source)}]}
        declared_include = str(include_path.resolve())
        closure = {"name": audit.HELPER_TARGET, "type": "STATIC_LIBRARY",
                   "sources": [str(source)], "include_directories": [declared_include],
                   "link_libraries": [], "interface_link_libraries": [],
                   "dependencies": [], "artifact": str(build_root / "libNCNNVulkanDeviceHelper.a")}
        return build_root, target_json, closure

    def test_actual_compile_argv_and_depfile_accept_clean_helper_and_reject_pollution(self) -> None:
        build_root, target_json, closure = self._compile_fixture()
        raw = audit._audit_target_compile(self.stage, self.evidence, build_root,
                                          "production-helper", target_json, closure, [])
        target_file = self.write_json(
            "test/Vulkan/evidence/va2/production-helper-target.json", target_json)
        raw["file_api_target"] = self.raw_entry(target_file, "File API target")
        raw["cmake_source_root"] = str(build_root)
        self.assertEqual(1, len(raw["compile_commands"]))
        self.assertTrue((self.evidence / "dependency-inputs/production-helper/compile_commands.json").is_file())
        audit._validate_recorded_compile(self.stage, raw, "production helper")
        compile_record = raw["compile_commands"][0]
        object_copy = self.stage / compile_record["object"]["path"]
        object_copy.write_bytes(b"not-the-compiler-output")
        compile_record["object"]["sha256"] = audit.sha256_file(object_copy)
        compile_record["object"]["size_bytes"] = object_copy.stat().st_size
        with self.assertRaisesRegex(audit.EvidenceError, "not bound to actual compiler output"):
            audit._validate_recorded_compile(self.stage, raw, "production helper")
        actual_object = Path(compile_record["compile_output"])
        object_copy.write_bytes(actual_object.read_bytes())
        compile_record["object"]["sha256"] = audit.sha256_file(object_copy)
        compile_record["object"]["size_bytes"] = object_copy.stat().st_size
        substituted_origin = self.root / "substituted-object.o"
        shutil.copyfile(actual_object, substituted_origin)
        compile_record["object"]["origin"] = str(substituted_origin)
        with self.assertRaisesRegex(audit.EvidenceError, "not bound to actual compiler output"):
            audit._validate_recorded_compile(self.stage, raw, "production helper")

        build_root, target_json, closure = self._compile_fixture(
            include="/opt/llvm-21/include")
        with self.assertRaisesRegex(audit.EvidenceError, "forbidden"):
            audit._audit_target_compile(self.stage, self.evidence, build_root,
                                        "production-helper", target_json, closure, [])

        build_root, target_json, closure = self._compile_fixture(dep_header="llvm/IR/Module.h")
        with self.assertRaisesRegex(audit.EvidenceError, "forbidden"):
            audit._audit_target_compile(self.stage, self.evidence, build_root,
                                        "production-helper", target_json, closure, [])

        build_root, target_json, closure = self._compile_fixture()
        closure["interface_link_libraries"] = ["MLIRIR"]
        with self.assertRaisesRegex(audit.EvidenceError, "forbidden"):
            audit._audit_target_compile(self.stage, self.evidence, build_root,
                                        "production-helper", target_json, closure, [])

    def test_archive_capture_hashes_object_inputs_and_rereads_raw_members(self) -> None:
        compiler = shutil.which("c++") or shutil.which("g++")
        ar = shutil.which("ar")
        if not compiler or not ar:
            self.skipTest("C++ compiler and ar are required for archive provenance tests")
        build_root = self.root / "archive-build"
        (build_root / "CMakeFiles/NCNNVulkanDeviceHelper.dir").mkdir(parents=True)
        source = build_root / "helper.cpp"
        source.write_text("int helper() { return 7; }\n")
        obj = build_root / "CMakeFiles/NCNNVulkanDeviceHelper.dir/helper.cpp.o"
        subprocess.run([compiler, "-c", str(source), "-o", str(obj)],
                       check=True, capture_output=True, text=True)
        archive = build_root / audit.EXPECTED_ARCHIVE
        subprocess.run([ar, "rcs", str(archive), str(obj)],
                       check=True, capture_output=True, text=True)
        link_file = build_root / "CMakeFiles/NCNNVulkanDeviceHelper.dir/link.txt"
        link_file.parent.mkdir(parents=True, exist_ok=True)
        link_file.write_text(f"{ar} qc {archive} {obj}\n")
        refs = []
        provenance = audit._capture_archive(
            self.stage, self.evidence, archive, link_file, build_root,
            "standalone-helper", refs, ar)
        artifact = audit._copy_build_raw(
            self.stage, self.evidence, archive,
            Path("dependency-inputs/standalone-helper/artifact.a"), "helper target artifact")
        target = {"artifact": artifact, "archive_provenance": provenance}
        self.assertEqual(1, len(provenance["members"]))
        audit._validate_archive_record(self.stage, target, "standalone helper")

        raw_member = self.stage / provenance["members"][0]["raw_member"]["path"]
        raw_member.write_bytes(b"tampered")
        with self.assertRaisesRegex(audit.EvidenceError, "raw archive member"):
            audit._validate_archive_record(self.stage, target, "standalone helper")

    def test_standalone_root_staleness_and_missing_raw_map_fail_closed(self) -> None:
        evidence_path = self.write_json(
            "test/Vulkan/evidence/va2/independence-evidence.json",
            {"schema_version": 1, "kind": audit.INDEPENDENCE_KIND})
        self.write_stage_state(stage_id="VA1")
        with self.assertRaisesRegex(audit.EvidenceError, "gate binding/profile"):
            audit.validate_independence_evidence(evidence_path, self.stage)
        self.write_stage_state(stage_id="VA2", profile="release")
        with self.assertRaises(audit.EvidenceError):
            audit.validate_independence_evidence(evidence_path, self.stage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
