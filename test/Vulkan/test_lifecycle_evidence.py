#!/usr/bin/env python3
"""Synthetic fail-closed regressions for VA3 evidence parsing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import lifecycle_evidence as evidence


class LifecycleEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def native(self, *, mode="device"):
        success = mode == "device"
        return {
            "schema_version": 1,
            "kind": evidence.NATIVE_RESULT_KIND,
            "fixture_id": "lifecycle_fixture",
            "expected_device_match": "NVIDIA" if success else "",
            "mode": mode,
            "status": "passed",
            "validation_enabled": success,
            "create_status": 0 if success else 2,
            "destroy_status": 0 if success else 0xFFFFFFFF,
            "destroy_null_status": 0,
            "options_null_status": 1,
            "context_output_null_status": 1,
            "options_bad_size_status": 1,
            "options_bad_version_status": 1,
            "options_bad_flags_status": 1,
            "invalid_ordinal_status": 2 if success else 0xFFFFFFFF,
            "requested_device_index": 0 if success else 0xFFFFFFFF,
            "run_status": 5 if success else 0xFFFFFFFF,
            "run_null_context_status": 1 if success else 0xFFFFFFFF,
            "run_null_input_status": 1 if success else 0xFFFFFFFF,
            "run_null_output_status": 1 if success else 0xFFFFFFFF,
            "run_bad_input_count_status": 6 if success else 0xFFFFFFFF,
            "run_bad_output_count_status": 6 if success else 0xFFFFFFFF,
            "created_contexts": 3 if success else 0,
            "independent_contexts": success,
            "repeated_create_destroy": success,
            "run_output_unchanged": success,
            "invalid_run_sentinels_unchanged": success,
            "diagnostic_bounded": True,
            "candidate_execution": "not_run",
            "upload_count": "not_measured",
        }

    @staticmethod
    def weight_bindings():
        return [{"resource_id": 103, "byte_size": 16,
                 "source_offset": 0, "destination_offset": 0,
                 "physical_sha256": "a" * 64,
                 "initializer_sha256": "a" * 64}]

    def validate_fake(self, value):
        return evidence.validate_fake_result(
            value, weight_bindings=self.weight_bindings(), expected_context_buffers=5)

    def fake(self):
        failure_statuses = {
            "vkCreateBuffer": 3, "vkAllocateMemory": 3,
            "vkBindBufferMemory": 4, "vkMapMemory": 5,
            "vkFlushMappedMemoryRanges": 4,
            "vkCreateDescriptorPool": 3, "vkCreateShaderModule": 3,
            "vkQueueSubmit": 4, "vkWaitForFences": 4,
        }
        return {
            "schema_version": 1,
            "kind": evidence.FAKE_RESULT_KIND,
            "status": "passed",
            "candidate_execution": "not_run",
            "budget_preflight": {"status": 3, "allocations_after_rejection": 0,
                                 "cleanup_all_objects_zero": True},
            "partial_create_failures": [
                {"function": name, "vk_result": -1, "status": status,
                 "cleanup_all_objects_zero": True}
                for name, status in failure_statuses.items()],
            "operation_wait_failure": {"status": 4,
                                        "vkCmdCopyBuffer_call_count": 1,
                                        "copy_arguments_valid": True,
                                        "cleanup_all_objects_zero": True},
            "weighted_upload": {
                "observed_weight_upload_count": 1,
                "expected_weight_upload_count": 1,
                "vkCmdCopyBuffer_call_count": 1,
                "expected_vkCmdCopyBuffer_call_count": 1,
                "copy_arguments_valid": True,
                "source_non_null": True,
                "destination_non_null": True,
                "source_destination_distinct": True,
                "region_count": 1,
                "recorded_region_count": 1,
                "records": [{"src_offset": 0, "dst_offset": 0, "size": 16}],
                "device_transfer": "not_executed_fake_command_recording",
                "live_buffer_count_after_create": 5,
                "staging_buffers_destroyed_before_create_return": 1,
                "peak_memory_bytes": 4096,
                "cleanup_all_objects_zero": True,
            },
            "wait_idle_retry": {"destroy_status": 5, "wait_count": 2,
                                "destruction_before_wait_resolution": False,
                                "cleanup_all_objects_zero": True},
        }

    def test_device_native_result_accepts_only_explicit_not_run_lifecycle(self):
        evidence._validate_native_observations(self.native(), "device", "NVIDIA", 0)
        changed = self.native()
        changed["candidate_execution"] = "passed"
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(changed, "device", "NVIDIA", 0)

    def test_native_schema_rejects_boolean_numeric_and_missing_fields(self):
        changed = self.native()
        changed["create_status"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(changed, "device", "NVIDIA", 0)
        changed = self.native()
        changed["created_contexts"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(changed, "device", "NVIDIA", 0)
        changed = self.native()
        changed.pop("options_bad_flags_status")
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(changed, "device", "NVIDIA", 0)

    def test_validation_claim_requires_json_boolean_true(self):
        valid = self.native()
        evidence._validate_native_observations(valid, "device", "NVIDIA", 0)
        valid["validation_enabled"] = 1
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(valid, "device", "NVIDIA", 0)
        valid = self.native()
        valid["validation_enabled"] = False
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(valid, "device", "NVIDIA", 0)

    def test_no_device_must_reject_create_without_claiming_execution(self):
        evidence._validate_native_observations(self.native(mode="no-device"), "no-device", None)
        invalid = self.native(mode="no-device")
        invalid["create_status"] = 0
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(invalid, "no-device", None)
        invalid = self.native(mode="no-device")
        invalid["destroy_status"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(invalid, "no-device", None)

    def test_weight_copy_size_comes_from_verified_weight_not_public_io(self):
        payload = bytes((0x00, 0x00, 0x80, 0x3F)) * 4
        digest = hashlib.sha256(payload).hexdigest()
        resources = [
            {"id": 7, "kind": "input", "byte_size": 68, "weight_sha256": ""},
            {"id": 41, "kind": "output", "byte_size": 68, "weight_sha256": ""},
            {"id": 103, "kind": "weight", "byte_size": 16, "weight_sha256": digest},
        ]
        buffers = [
            {"id": 0, "kind": "device", "resource_id": 7, "byte_size": 68, "scope": "context"},
            {"id": 1, "kind": "device", "resource_id": 41, "byte_size": 68, "scope": "context"},
            {"id": 2, "kind": "device", "resource_id": 103, "byte_size": 16, "scope": "context"},
            {"id": 3, "kind": "input-staging", "resource_id": 7, "byte_size": 68, "scope": "context"},
            {"id": 4, "kind": "output-staging", "resource_id": 41, "byte_size": 68, "scope": "context"},
            {"id": 5, "kind": "weight-staging", "resource_id": 103, "byte_size": 16, "scope": "create-upload"},
        ]
        host = {
            "schema": "ncnn-vulkan-host-program/v1", "identity": "host-id",
            "resources": resources, "buffers": buffers,
            "weights": [{"resource_id": 103, "byte_size": 16,
                         "logical_sha256": digest, "physical_sha256": digest}],
            "create_operations": [
                {"op": "upload", "source": "weight", "source_id": 103,
                 "staging_buffer": 5, "byte_size": 16},
                {"op": "copy", "source": 5, "destination": 2,
                 "source_offset": 0, "destination_offset": 0, "byte_size": 16},
            ],
        }
        plan = {"kind": "ncnn.vulkan_execution_plan", "resources": [
            {"id": item["id"], "kind": item["kind"],
             "byte_size": item["byte_size"], "weight_sha256": item["weight_sha256"]}
            for item in resources]}
        manifest = {"kind": "ncnn.vulkan_model_manifest",
                    "host_program_usage": "compiler-only-diagnostic",
                    "host_program": host, "host_program_identity": "host-id",
                    "weights": [{"resource_id": 103, "sha256": digest, "size_bytes": 16}]}
        source = ("constexpr std::array<std::byte, 16> kWeightBytes0{{\n" +
                  ",\n".join(f"std::byte{{0x{byte:02x}}}" for byte in payload) +
                  "\n}};\n")
        binding = evidence._metadata_weight_bindings(source, host, plan, manifest)
        self.assertEqual(16, binding[0]["byte_size"])
        invalid_source = source.replace("std::array<std::byte, 16>",
                                        "std::array<std::byte, 68>")
        with self.assertRaises(evidence.EvidenceError):
            evidence._metadata_weight_bindings(invalid_source, host, plan, manifest)

    def test_fake_fault_observations_are_typed_and_balanced(self):
        self.validate_fake(self.fake())
        invalid = self.fake()
        invalid["weighted_upload"]["observed_weight_upload_count"] = True
        with self.assertRaises(evidence.EvidenceError):
            self.validate_fake(invalid)
        invalid = self.fake()
        invalid["weighted_upload"]["staging_buffers_destroyed_before_create_return"] = 0
        with self.assertRaises(evidence.EvidenceError):
            self.validate_fake(invalid)
        invalid = self.fake()
        invalid["partial_create_failures"].pop()
        with self.assertRaises(evidence.EvidenceError):
            self.validate_fake(invalid)
        invalid = self.fake()
        invalid["weighted_upload"]["records"][0]["size"] = 68
        with self.assertRaises(evidence.EvidenceError):
            self.validate_fake(invalid)
        invalid = self.fake()
        invalid["weighted_upload"]["copy_arguments_valid"] = 1
        with self.assertRaises(evidence.EvidenceError):
            self.validate_fake(invalid)

    def test_artifact_hash_and_size_are_recomputed(self):
        path = self.root / "artifact.bin"
        path.write_bytes(b"candidate")
        item = {"path": "artifact.bin", "sha256": hashlib.sha256(b"candidate").hexdigest(),
                "size_bytes": len(b"candidate"), "role": "generated source"}
        self.assertEqual(item, evidence._check_artifact(self.root, item, "fixture"))
        path.write_bytes(b"changed")
        with self.assertRaises(evidence.EvidenceError):
            evidence._check_artifact(self.root, item, "fixture")
        item["size_bytes"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence._check_artifact(self.root, item, "fixture")

    def test_raw_artifact_paths_cannot_escape_stage(self):
        with self.assertRaises(evidence.EvidenceError):
            evidence._safe_file(self.root, "../outside", "fixture")

    def test_sanitize_compile_and_link_flags_cannot_disable_instrumentation(self):
        compile_flags = ["clang++", "-fsanitize=address,undefined",
                         "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-fPIC"]
        link_flags = [*compile_flags, "-shared-libasan"]
        evidence._require_profile_flags(compile_flags, "sanitize", "host", pic=True)
        evidence._require_profile_flags(link_flags, "sanitize", "candidate link",
                                        pic=True, link=True)
        for invalid in ("-fno-sanitize=address", "-fno-sanitize=undefined",
                        "-fno-sanitize=all", "-fsanitize-recover=undefined",
                        "-fsanitize-blacklist=ignored.txt",
                        "-fsanitize-ignorelist=ignored.txt",
                        "-fsanitize-coverage-ignorelist=ignored.txt",
                        "-fno-sanitize-coverage=trace-pc"):
            with self.subTest(invalid=invalid), self.assertRaises(evidence.EvidenceError):
                evidence._require_profile_flags(
                    [*compile_flags, invalid], "sanitize", "host", pic=True)
        with self.assertRaises(evidence.EvidenceError):
            evidence._require_profile_flags(compile_flags, "sanitize", "candidate link",
                                            pic=True, link=True)

    def test_sanitize_evidence_rejects_inherited_suppression_options(self):
        for name in ("ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS"):
            with self.subTest(name=name), patch.dict("os.environ", {name: "suppressions=1"}):
                with self.assertRaises(evidence.EvidenceError):
                    evidence._reject_sanitizer_environment("sanitize")
        evidence._reject_sanitizer_environment("release")

    def test_cmake_without_gate_state_remains_unqualified(self):
        cache = self.root / "CMakeCache.txt"
        cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\nCOMPILER_SANITIZERS:STRING=\n")
        self.assertEqual("unqualified", evidence._lifecycle_build_profile(self.root))
        cache.write_text("CMAKE_BUILD_TYPE:STRING=Debug\n"
                         "COMPILER_SANITIZERS:STRING=address;undefined\n")
        self.assertEqual("unqualified", evidence._lifecycle_build_profile(self.root))
        (self.root / "gate-state.json").write_text(json.dumps({"profile": "sanitize"}))
        self.assertEqual("sanitize", evidence._lifecycle_build_profile(self.root))

    def test_command_provenance_binds_output_object_archive_and_elf_target(self):
        build = self.root / "build"
        cwd = build / "test/Vulkan/native"
        cwd.mkdir(parents=True)
        real_output = cwd / "candidate.so"
        decoy_output = cwd / "decoy.so"
        expected_object = cwd / "candidate.cpp.o"
        other_object = cwd / "other.cpp.o"
        archive_dir = build / "lib/VulkanDeviceHelper"
        archive_dir.mkdir(parents=True)
        expected_archive = archive_dir / "libNCNNVulkanDeviceHelper.a"
        decoy_archive_dir = build / "decoy"
        decoy_archive_dir.mkdir()
        decoy_archive = decoy_archive_dir / expected_archive.name
        for path in (real_output, decoy_output, expected_object, other_object,
                     expected_archive, decoy_archive):
            path.write_bytes(path.name.encode())
        link = ["clang++", "-o", "candidate.so", "candidate.cpp.o",
                str(expected_archive), "-lvulkan"]
        evidence._require_command_output(link, cwd, real_output, "candidate link")
        evidence._require_object_inputs(link, cwd, [expected_object], "candidate link")
        evidence._require_exact_helper_archive(link, cwd, expected_archive, "candidate link")
        evidence._command_audit_target(["readelf", "-d", str(real_output)],
                                       real_output, cwd, "candidate ELF")
        with self.assertRaises(evidence.EvidenceError):
            evidence._require_command_output(
                ["clang++", "-o", "decoy.so", "candidate.cpp.o", str(expected_archive)],
                cwd, real_output, "wrong output")
        with self.assertRaises(evidence.EvidenceError):
            evidence._require_exact_helper_archive(
                ["clang++", "-o", "candidate.so", "candidate.cpp.o",
                 str(decoy_archive)], cwd, expected_archive, "same-basename archive")
        with self.assertRaises(evidence.EvidenceError):
            evidence._command_audit_target(["readelf", "-d", str(decoy_output)],
                                           real_output, cwd, "wrong ELF audit")
        with self.assertRaises(evidence.EvidenceError):
            evidence._require_object_inputs(link, cwd, [other_object], "wrong object")

    def test_qualified_source_root_must_match_cmake_and_gate(self):
        source = self.root / "source"
        other = self.root / "other"
        source.mkdir()
        other.mkdir()
        raw = self.root / "raw.bin"
        raw.write_bytes(b"raw")
        record = evidence._artifact(self.root, raw, "raw")
        value = {
            "schema_version": 1, "kind": evidence.INDEPENDENCE_KIND,
            "role": "lifecycle-only", "source_fingerprint": "a" * 64,
            "build_profile": "release", "raw_artifacts": [record],
            "claims": {"no_plan_or_parser_input": True,
                       "no_llvm_or_mlir_link_input": True,
                       "strict_no_undefined": True, "export_allowlist_only": True,
                       "candidate_run": "not_run", "host_aot_release_eligible": False},
        }
        report = self.root / "independence.json"
        report.write_text(json.dumps(value))
        cache = self.root / "CMakeCache.txt"
        cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n"
                         "CMAKE_BUILD_TYPE:STRING=Release\n")
        state = self.root / "gate-state.json"
        state.write_text(json.dumps({"profile": "release", "source_dir": str(other),
                                     "source_fingerprint": "a" * 64}))
        with self.assertRaisesRegex(evidence.EvidenceError, "qualified gate source"):
            evidence.validate_independence_evidence(
                self.root, report, expected_fingerprint="a" * 64)
        state.write_text(json.dumps({"profile": "release", "source_dir": str(source),
                                     "source_fingerprint": "a" * 64}))
        # The source guard succeeds; deliberately incomplete compile evidence
        # must still be rejected at the next independent validation boundary.
        with self.assertRaisesRegex(evidence.EvidenceError, "compile database snapshot"):
            evidence.validate_independence_evidence(
                self.root, report, expected_fingerprint="a" * 64)
        cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=relative/source\n"
                         "CMAKE_BUILD_TYPE:STRING=Release\n")
        with self.assertRaisesRegex(evidence.EvidenceError, "actual source directory"):
            evidence.validate_independence_evidence(
                self.root, report, expected_fingerprint="a" * 64)

    def test_helper_namespace_is_not_mistaken_for_forbidden_mlir_link_input(self):
        self.assertIsNone(evidence.FORBIDDEN_INPUTS.search(
            "ncnn_mlir::vulkan::VulkanDeviceContext::_destroy"))
        for malicious in ("/tmp/libMLIRIR.a", "-lLLVMCore",
                          "/tmp/libNCNNVulkanPlanSupport.a", "VulkanExecutionPlan.cpp"):
            with self.subTest(malicious=malicious):
                self.assertIsNotNone(evidence.FORBIDDEN_INPUTS.search(malicious))

    def test_linker_map_parser_preserves_map_and_cross_reference_forms(self):
        link_file = self.root / "test/Vulkan/native/CMakeFiles/target.dir/link.txt"
        link_file.parent.mkdir(parents=True)
        link_file.write_text("link command\n")
        map_file = self.root / "test/Vulkan/native/lifecycle/target.map"
        map_file.parent.mkdir(parents=True)
        map_file.write_text("real linker map\n")
        forms = (
            ["clang++", f"-Wl,-Map={map_file},--cref"],
            ["clang++", f"-Wl,-Map,{map_file},--cref"],
            ["clang++", "-Xlinker", "-Map", "-Xlinker", str(map_file),
             "-Xlinker", "--cref"],
            ["clang++", "-Xlinker", f"-Map={map_file}", "-Xlinker", "--cref"],
        )
        for argv in forms:
            with self.subTest(argv=argv):
                self.assertEqual(map_file.resolve(),
                                 evidence._parse_linker_map_argv(argv, link_file))

    def test_linker_map_parser_rejects_missing_multiple_unknown_and_absent_maps(self):
        link_file = self.root / "target.dir/link.txt"
        link_file.parent.mkdir()
        link_file.write_text("link command\n")
        map_a = self.root / "a.map"
        map_b = self.root / "b.map"
        map_a.write_text("a\n")
        map_b.write_text("b\n")
        for argv in (
                ["clang++", "-Wl,--cref"],
                ["clang++", f"-Wl,-Map={map_a},-Map={map_b},--cref"],
                ["clang++", f"-Wl,-MapFile={map_a},--cref"],
                ["clang++", "-Xlinker", "-Map"],
                ["clang++", f"-Wl,-Map={self.root / 'missing.map'},--cref"],
        ):
            with self.subTest(argv=argv), self.assertRaises(evidence.EvidenceError):
                evidence._parse_linker_map_argv(argv, link_file)

    def test_aggregate_requires_both_real_device_tests_and_dynamic_codegen_inventory(self):
        source = self.root / "separate-source"
        unit = source / "test/Unit/vulkan_lifecycle_codegen_test.cpp"
        unit.parent.mkdir(parents=True)
        unit.write_text("TEST(VulkanLifecycleCodegenTest, EmitsCreateDestroy) {}\n"
                        "TEST(VulkanLifecycleCodegenTest, Weights) {}\n")
        self.assertFalse((self.root / "test/Unit/vulkan_lifecycle_codegen_test.cpp").exists())
        (self.root / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\nCMAKE_BUILD_TYPE:STRING=Release\n")
        (self.root / "gate-state.json").write_text(json.dumps({
            "source_dir": str(source), "source_fingerprint": "a" * 64,
            "profile": "release"}))
        tests = ["vulkan-lifecycle-fake-unit", "vulkan-lifecycle-evidence-tests",
                 "vulkan-lifecycle-independence", "vulkan-lifecycle-no-device",
                 "vulkan-lifecycle-nvidia", "vulkan-lifecycle-llvmpipe"]
        with self.assertRaisesRegex(evidence.EvidenceError, "inventory is missing"):
            evidence.aggregate_stage_evidence(self.root, "release", "a" * 64,
                                              tests, [], [])
        tests.append("VulkanLifecycleCodegenTest.EmitsCreateDestroy")
        with self.assertRaisesRegex(evidence.EvidenceError, "Weights"):
            evidence.aggregate_stage_evidence(self.root, "release", "a" * 64,
                                              tests, [], [])
        tests.append("VulkanLifecycleCodegenTest.Weights")
        with self.assertRaisesRegex(evidence.EvidenceError, "skipped or failed"):
            evidence.aggregate_stage_evidence(self.root, "release", "a" * 64,
                                              tests, ["vulkan-lifecycle-nvidia"], [])
        with self.assertRaisesRegex(evidence.EvidenceError, "uncatalogued"):
            evidence.aggregate_stage_evidence(self.root, "release", "a" * 64,
                tests + ["VulkanLifecycleCodegenTest.Unknown"], [], [])
        with patch.object(evidence, "validate_independence_evidence",
                          side_effect=evidence.EvidenceError("reached actual independence")) as validate:
            with self.assertRaisesRegex(evidence.EvidenceError, "reached actual independence"):
                evidence.aggregate_stage_evidence(self.root, "release", "a" * 64,
                                                  tests, [], [])
            validate.assert_called_once_with(
                self.root, self.root / "test/Vulkan/evidence/va3/independence/independence-evidence.json",
                expected_fingerprint="a" * 64)

    def test_post_ctest_aggregate_is_not_added_to_its_raw_inventory(self):
        source = self.root / "source"
        unit = source / "test/Unit/vulkan_lifecycle_codegen_test.cpp"
        unit.parent.mkdir(parents=True)
        unit.write_text("TEST(VulkanLifecycleCodegenTest, ActualCase) {}\n")
        (self.root / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\nCMAKE_BUILD_TYPE:STRING=Release\n")
        (self.root / "gate-state.json").write_text(json.dumps({
            "source_dir": str(source), "source_fingerprint": "a" * 64,
            "profile": "release"}))
        base = self.root / "test/Vulkan/evidence/va3"
        for label in ("independence", "no-device", "nvidia", "llvmpipe", "fake"):
            (base / label).mkdir(parents=True)
        (base / "independence/independence-evidence.json").write_text("{}\n")
        for label in ("no-device", "nvidia", "llvmpipe"):
            (base / label / "capture.json").write_text("{}\n")
        (base / "fake/fake-result.json").write_text(json.dumps(self.fake()))
        log = base / "test-produced.log"
        log.write_text("actual test log\n")
        raw = evidence._artifact(self.root, log, "test-produced command log")
        capture = {
            "command": {"stdout": raw, "stderr": raw,
                        "vulkaninfo": {"stdout": raw, "stderr": raw,
                                       "physical_device_identity": {"device_name": "fixture"}}},
            "native_observations": {"create_status": 0, "destroy_status": 0},
            "validation_error_count": 0, "requested_device_index": 0,
            "icd_library": {"sha256": "a" * 64},
        }
        independent = {"fixtures": [{"fixture": "weighted",
            "weight_bindings": self.weight_bindings(), "context_buffer_count": 5}],
            "raw_artifacts": []}
        names = sorted(set(evidence.PRODUCT_ONLY_REQUIRED_TESTS) |
                       {"VulkanLifecycleCodegenTest.ActualCase",
                        "vulkan-lifecycle-nvidia", "vulkan-lifecycle-llvmpipe"})
        with patch.object(evidence, "validate_independence_evidence", return_value=independent), \
                patch.object(evidence, "validate_native_capture", return_value=capture):
            aggregate, refs = evidence.aggregate_stage_evidence(
                self.root, "release", "a" * 64, names, [], [])
        output = base / "lifecycle-evidence.json"
        self.assertTrue(output.is_file())
        stored = json.loads(output.read_text())
        self.assertEqual(stored, aggregate)
        self.assertEqual(stored["raw_artifacts"], refs)
        self.assertNotIn(str(output.relative_to(self.root)),
                         {item["path"] for item in refs})

    def test_fake_or_device_run_cannot_supply_candidate_execution(self):
        bad = self.native()
        bad["candidate_execution"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_native_observations(bad, "device", "NVIDIA", 0)
        bad = self.fake()
        bad["weighted_upload"]["peak_memory_bytes"] = False
        with self.assertRaises(evidence.EvidenceError):
            self.validate_fake(bad)

    def test_vulkan_summary_requires_real_expected_device(self):
        nvidia = "\n".join(("GPU0:", "apiVersion = 1.4.351",
                             "driverVersion = 615.78.08", "vendorID = 0x10de",
                             "deviceID = 0x2803",
                             "deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                             "deviceName = NVIDIA GeForce RTX 4060 Ti"))
        self.assertEqual(0x10DE, evidence._parse_vulkan_identity(nvidia, "NVIDIA")["vendor_id"])
        with self.assertRaises(evidence.EvidenceError):
            evidence._parse_vulkan_identity(nvidia.replace("0x10de", "0x10005"), "NVIDIA")

    def test_inventory_query_preserves_active_parent_ctest_logs(self):
        build = self.root / "inventory-build"
        nested = build / "nested"
        nested.mkdir(parents=True)
        (build / "CTestTestfile.cmake").write_text(
            'include("discovered.cmake")\nsubdirs("nested")\n')
        (build / "discovered.cmake").write_text(
            f'add_test(lifecycle_top "{sys.executable}" "-c" "raise RuntimeError(\\"not executed\\")")\n')
        (nested / "CTestTestfile.cmake").write_text(
            f'add_test(lifecycle_nested "{sys.executable}" "-c" "raise RuntimeError(\\"not executed\\")")\n')
        temporary = build / "Testing/Temporary"
        temporary.mkdir(parents=True)
        logs = {temporary / name: f"active parent {name}\\n".encode()
                for name in ("LastTest.log", "LastTest.log.tmp", "LastTestsFailed.log")}
        for path, contents in logs.items():
            path.write_bytes(contents)
        stamps = {path: path.stat().st_mtime_ns for path in logs}
        inventory = query_ctest_inventory(build)
        self.assertEqual({"lifecycle_top", "lifecycle_nested"},
                         {item["name"] for item in inventory["tests"]})
        for path, contents in logs.items():
            self.assertEqual(contents, path.read_bytes())
            self.assertEqual(stamps[path], path.stat().st_mtime_ns)

    def test_codegen_source_catalog_is_exact_and_not_count_based(self):
        source = Path(__file__).resolve().parents[1] / "Unit/vulkan_lifecycle_codegen_test.cpp"
        codegen = evidence.codegen_tests_from_source(source)
        inventory = set(codegen) | set(evidence.PRODUCT_ONLY_REQUIRED_TESTS)
        validate_product_only_test_names(inventory, source)
        missing_one = inventory - {next(iter(codegen))}
        with self.assertRaises(evidence.EvidenceError):
            validate_product_only_test_names(missing_one, source)
        extra = set(inventory) | {"VulkanLifecycleCodegenTest.Unregistered"}
        with self.assertRaises(evidence.EvidenceError):
            validate_product_only_test_names(extra, source)


def validate_product_only_test_names(names: set[str], source: Path) -> None:
    codegen = evidence.codegen_tests_from_source(source)
    required = set(evidence.PRODUCT_ONLY_REQUIRED_TESTS) | set(codegen)
    missing = sorted(required - names)
    unexpected = sorted(name for name in names
                        if name.startswith(evidence.CODEGEN_TEST_PREFIX) and
                        name not in codegen)
    if missing or unexpected:
        raise evidence.EvidenceError(
            f"product-only CTest inventory differs from unit source: "
            f"missing={missing}, unexpected={unexpected}")


def query_ctest_inventory(build_dir: Path) -> dict:
    # Even --show-only truncates Testing/Temporary/LastTest.log. Never query the
    # active parent CTest tree: use a read-only metadata snapshot in a separate
    # binary directory, retaining the actual generated registration/includes.
    if not (build_dir / "CTestTestfile.cmake").is_file():
        raise evidence.EvidenceError("product-only CTest registration is missing")
    with tempfile.TemporaryDirectory(prefix="vulkan-lifecycle-inventory-") as temporary:
        query_root = Path(temporary)
        for metadata in build_dir.rglob("*.cmake"):
            if metadata.is_file():
                target = query_root / metadata.relative_to(build_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(metadata, target)
        result = subprocess.run(["ctest", "--test-dir", str(query_root),
                                 "--show-only=json-v1"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, text=True)
    if result.returncode != 0:
        raise evidence.EvidenceError(
            f"cannot query product-only CTest inventory: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise evidence.EvidenceError(f"product-only CTest inventory is malformed: {error}") from error
    if not isinstance(value, dict):
        raise evidence.EvidenceError("product-only CTest inventory is not an object")
    return value


def verify_product_only_inventory(build_dir: Path) -> None:
    value = query_ctest_inventory(build_dir)
    tests = value.get("tests") if isinstance(value, dict) else None
    if not isinstance(tests, list):
        raise evidence.EvidenceError("product-only CTest inventory has no tests array")
    names = [item.get("name") for item in tests if isinstance(item, dict)]
    if len(names) != len(tests) or len(names) != len(set(names)):
        raise evidence.EvidenceError("product-only CTest inventory has malformed/duplicate test names")
    source = Path(__file__).resolve().parents[1] / "Unit/vulkan_lifecycle_codegen_test.cpp"
    validate_product_only_test_names(set(names), source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path,
                        help="product-only build whose CTest inventory is independently checked")
    args = parser.parse_args()
    if args.build_dir:
        try:
            verify_product_only_inventory(args.build_dir.resolve())
        except (OSError, ValueError, RuntimeError) as error:
            print(f"test_lifecycle_evidence.py: {error}", file=sys.stderr)
            return 1
    unittest.main(verbosity=2, argv=[sys.argv[0]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
