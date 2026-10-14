#!/usr/bin/env python3
"""Fail-closed tests for generated-product no-device lifecycle evidence."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import check_release_evidence as audit

_NATIVE_CHECKER_PATH = Path(__file__).resolve().parents[1] / "Native/check_ncnn_compile_vulkan.py"
_NATIVE_SPEC = importlib.util.spec_from_file_location("native_vulkan_checker", _NATIVE_CHECKER_PATH)
if _NATIVE_SPEC is None or _NATIVE_SPEC.loader is None:
    raise RuntimeError("cannot load Native Vulkan checker")
native_vulkan_checker = importlib.util.module_from_spec(_NATIVE_SPEC)
_NATIVE_SPEC.loader.exec_module(native_vulkan_checker)


def no_device_record(stage, *, sanitizer=False):
    base = stage / "test/Native/vulkan-product-offline"
    base.mkdir(parents=True, exist_ok=True)

    def artifact(name, content):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"file": name, "sha256": audit.sha256_file(path)}

    loader = "no-device-vulkan-product-loader"
    result_name = "no-device-loader-result.json"
    source = (Path(audit.__file__).resolve().parents[1] /
              "Native/vulkan_product_loader.cpp").resolve()
    flags = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
             "-fno-omit-frame-pointer", "-shared-libasan"] if sanitizer else []
    compile_argv = ["/usr/bin/c++", "-std=c++23", *flags,
                    "-DNCNN_PRODUCT_TEST_MODEL=paired_relu",
                    "-DNCNN_PRODUCT_TEST_EXPECT_NO_DEVICE=1",
                    "-I", str(base / "cpu"), "-I", str(base / "vulkan"),
                    "-I", str(source.parents[2] / "include"), str(source),
                    "-ldl", "-o", str(base / loader)]
    result = {"status": "success", "phase": "expected-no-device", "cpu_status": 0,
              "vulkan_create_status": 2, "vulkan_run_status": -1,
              "vulkan_destroy_status": -1, "vulkan_destroy_null_status": 0,
              "context_null": True, "diagnostic_bounded": True,
              "diagnostic": "vkCreateInstance failed with VkResult -9"}
    loader_record = artifact(loader, b"synthetic native loader")
    result_record = artifact(result_name, json.dumps(result).encode())
    inputs = [float(index % 7 - 3) for index in range(257)]
    artifact("public-input.f32", struct.pack("<257f", *inputs))
    artifact("no-device-input.f32", struct.pack("<257f", *inputs))
    artifact("no-device-cpu-output.f32", struct.pack("<257f", *[max(x, 0.0) for x in inputs]))
    streams = {name: artifact("no-device-vulkan-loader." + name, b"")
               for name in ("compile.stdout.log", "compile.stderr.log", "stdout.log", "stderr.log")}
    modules = {phase: artifact(result_name + "." + phase + ".maps", b"synthetic module map\n")
               for phase in ("before-load", "after-model-load", "after-create")}
    return {
        "compile_argv": compile_argv, "compile_returncode": 0,
        "argv": [str(base / loader), "paired_relu", str(base / "cpu/libpaired_relu.so"),
                 str(base / "vulkan/libpaired_relu_vulkan.so"), "257",
                 str(base / "no-device-input.f32"), str(base / "no-device-cpu-output.f32"),
                 str(base / "no-device-vulkan-output.f32"), str(base / result_name)],
        "returncode": 0, "loader_file": loader,
        "loader_sha256": loader_record["sha256"], "result_file": result_name,
        "result_sha256": result_record["sha256"], "streams": streams,
        "loaded_modules": modules, "gpu_numeric": False,
        "icd_environment": {"VK_ICD_FILENAMES": str(base / "no-available-icd.json"),
                            "VK_DRIVER_FILES": None, "VK_ADD_DRIVER_FILES": None},
    }


class NoDeviceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.stage = Path(self.temporary.name)
        self.base = self.stage / "test/Native/vulkan-product-offline"
        self.record = no_device_record(self.stage)

    def tearDown(self):
        self.temporary.cleanup()

    def codes(self, record=None, *, sanitizer=False):
        findings = []
        audit._audit_no_device_lifecycle(
            self.stage, self.base, self.record if record is None else record,
            {"address", "undefined"} if sanitizer else set(), [], findings)
        return {item["code"] for item in findings}

    def change_result(self, **changes):
        path = self.base / self.record["result_file"]
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value))
        self.record["result_sha256"] = audit.sha256_file(path)

    def test_release_and_matching_san_records_pass(self):
        self.assertEqual(set(), self.codes())
        self.record = no_device_record(self.stage, sanitizer=True)
        self.assertEqual(set(), self.codes(sanitizer=True))

    def test_missing_lifecycle_is_not_an_offline_pass(self):
        findings = []
        audit._audit_no_device_lifecycle(self.stage, self.base, None, set(), [], findings)
        self.assertEqual({"no-device-lifecycle-missing"}, {row["code"] for row in findings})

    def test_context_diagnostic_status_and_destroy_are_checked(self):
        for changes in ({"context_null": False}, {"diagnostic_bounded": False},
                        {"diagnostic": ""}, {"vulkan_create_status": 0},
                        {"vulkan_destroy_null_status": 5}, {"vulkan_run_status": 0}):
            with self.subTest(changes=changes):
                self.record = no_device_record(self.stage)
                self.change_result(**changes)
                self.assertIn("no-device-lifecycle-result", self.codes())

    def test_sanitizer_cannot_be_disabled_or_overridden(self):
        for option in ("-fno-sanitize=address", "-fsanitize-recover=all", "@extra.rsp"):
            with self.subTest(option=option):
                self.record = no_device_record(self.stage, sanitizer=True)
                self.record["compile_argv"].insert(-2, option)
                self.assertIn("no-device-lifecycle-instrumentation", self.codes(sanitizer=True))

    def test_cpu_only_loader_cannot_supply_vulkan_lifecycle(self):
        self.record["compile_argv"][3] = "-DNCNN_PRODUCT_TEST_CPU_ONLY=1"
        self.assertIn("no-device-lifecycle-instrumentation", self.codes())

    def test_equivalent_relative_source_spelling_is_accepted_but_wrong_source_is_rejected(self):
        compile_argv = self.record["compile_argv"]
        source_index = next(index for index, item in enumerate(compile_argv)
                            if Path(item).name == "vulkan_product_loader.cpp")
        source = Path(compile_argv[source_index]).resolve()
        alias = source.parents[2] / "test/Vulkan/../Native/vulkan_product_loader.cpp"
        compile_argv[source_index] = str(alias)
        self.assertNotIn("no-device-lifecycle-instrumentation", self.codes())
        compile_argv[source_index] = str(source.with_name("different_loader.cpp"))
        self.assertIn("no-device-lifecycle-instrumentation", self.codes())

    def test_module_and_raw_stream_hashes_are_required(self):
        (self.base / self.record["loaded_modules"]["after-create"]["file"]).unlink()
        self.assertTrue(self.codes())
        self.record = no_device_record(self.stage)
        (self.base / self.record["streams"]["stderr.log"]["file"]).write_text("runtime error: synthetic UB")
        self.assertIn("no-device-lifecycle-sanitizer", self.codes())

    def test_successful_gpu_claim_or_published_output_is_rejected(self):
        self.record["gpu_numeric"] = True
        self.assertIn("no-device-lifecycle-command", self.codes())
        self.record["gpu_numeric"] = False
        (self.base / "no-device-vulkan-output.f32").write_bytes(b"partial output")
        self.assertIn("no-device-lifecycle-output", self.codes())

    def test_override_icd_is_rejected(self):
        self.record["icd_environment"]["VK_DRIVER_FILES"] = "working-icd.json"
        self.assertIn("no-device-lifecycle-environment", self.codes())

    def test_only_identity_bound_sanitizer_runtime_is_allowed_in_cpu_closure(self):
        configured_path = self.base / "configured" / "libclang_rt.asan-x86_64.so"
        configured_path.parent.mkdir(parents=True)
        configured_path.write_bytes(b"configured ASan runtime")
        runtime_hash = audit.sha256_file(configured_path)
        runtime = {"path": str(configured_path), "sha256": runtime_hash,
                   "soname": "libclang_rt.asan-x86_64.so", "classification": "SAN",
                   "role": "sanitizer-runtime",
                   "direct_needed_names": ["libclang_rt.asan-x86_64.so"]}
        identity = {"path": str(configured_path), "sha256": runtime_hash}
        asan_needed = {"libclang_rt.asan-x86_64.so"}
        self.assertFalse(audit._cpu_forbidden_runtime(
            runtime, sanitizer=True, needed=asan_needed,
            sanitizer_runtime_identity=identity))
        self.assertTrue(audit._cpu_forbidden_runtime(
            runtime, sanitizer=True, needed=asan_needed))
        self.assertTrue(audit._cpu_forbidden_runtime(runtime, sanitizer=True))
        self.assertTrue(audit._cpu_forbidden_runtime(runtime, sanitizer=False,
                                                     needed=asan_needed,
                                                     sanitizer_runtime_identity=identity))
        spoof_path = self.base / "spoof" / configured_path.name
        spoof_path.parent.mkdir(parents=True)
        spoof_path.write_bytes(b"unconfigured spoof runtime")
        spoof = {**runtime, "path": str(spoof_path),
                 "sha256": audit.sha256_file(spoof_path)}
        self.assertTrue(audit._cpu_forbidden_runtime(
            spoof, sanitizer=True, needed=asan_needed,
            sanitizer_runtime_identity=identity))
        shadow_path = self.base / "shadow" / "libfake.so"
        shadow_path.parent.mkdir(parents=True)
        shadow_path.write_bytes(b"shadow DSO with ASan dependency name")
        shadow = {"path": str(shadow_path), "sha256": audit.sha256_file(shadow_path),
                  "soname": "libfake.so", "classification": "model",
                  "role": "private-transitive", "direct_to_model": True,
                  "direct_needed_names": ["libclang_rt.asan-x86_64.so"]}
        self.assertTrue(audit._cpu_forbidden_runtime(
            shadow, sanitizer=True, needed=asan_needed,
            sanitizer_runtime_identity=identity))
        mismatched_llvm = {**runtime, "soname": "libLLVM.so.21"}
        self.assertTrue(audit._cpu_forbidden_runtime(
            mismatched_llvm, sanitizer=True, needed={"libLLVM.so.21"},
            sanitizer_runtime_identity=identity))
        mismatched_vulkan = {**runtime, "soname": "libvulkan.so.1"}
        self.assertTrue(audit._cpu_forbidden_runtime(
            mismatched_vulkan, sanitizer=True, needed={"libvulkan.so.1"},
            sanitizer_runtime_identity=identity))
        mislabeled = {**runtime, "classification": "compiler"}
        self.assertTrue(audit._cpu_forbidden_runtime(
            mislabeled, sanitizer=True, sanitizer_runtime_identity=identity))
        llvm = {"path": "/usr/lib/libLLVM.so.21", "soname": "libLLVM.so.21",
                "classification": "compiler", "role": "shared-library"}
        self.assertTrue(audit._cpu_forbidden_runtime(llvm, sanitizer=True))
        self.assertTrue(audit._cpu_forbidden_soname("libLLVM.so.21", sanitizer=True))
        self.assertTrue(audit._cpu_forbidden_soname("libvulkan.so.1", sanitizer=False))
        self.assertFalse(audit._cpu_forbidden_soname(
            "libclang_rt.asan-x86_64.so", sanitizer=True))
        self.assertTrue(audit._cpu_forbidden_soname(
            "libclang.so.21", sanitizer=True))

    def test_sanitizer_runtime_identity_comes_from_hash_bound_toolchain_record(self):
        identity_path = self.base / "toolchain-evidence/toolchain-identity.json"
        identity_path.parent.mkdir(parents=True)
        configured_runtime = {"path": "/configured/clang/libclang_rt.asan-x86_64.so",
                             "sha256": "a" * 64}
        identity_path.write_text(json.dumps({
            "schema": "ncnn-san-toolchain-identity/v2", "status": "verified",
            "llvm": {"shared_asan_runtime_identity": configured_runtime}}))
        record = {"path": identity_path.relative_to(self.stage).as_posix(),
                  "sha256": audit.sha256_file(identity_path)}
        gate = {"state": {"sanitizer_toolchain_evidence": {"identity": record}}}
        findings = []
        self.assertEqual(configured_runtime, audit._sanitizer_runtime_identity(
            self.stage, gate, findings))
        self.assertEqual([], findings)
        identity_path.write_text("tampered")
        findings = []
        self.assertIsNone(audit._sanitizer_runtime_identity(self.stage, gate, findings))
        self.assertTrue(findings)


class SanitizerReproducibilityTests(unittest.TestCase):
    def test_only_compiler_temporary_bitcode_token_is_normalized(self):
        first = b"head:/tmp/ncnn-compile-abc123/vulkan_model_wrapper.bc:tail"
        repeat = b"head:/tmp/ncnn-compile-987xyz/vulkan_model_wrapper.bc:tail"
        identical, identical_metadata = native_vulkan_checker.reproducible_artifact_bytes(
            "libpaired_relu_vulkan.so", first, first, sanitizer=True)
        self.assertEqual(first, identical)
        self.assertEqual("byte-identical", identical_metadata["mode"])
        normalized, metadata = native_vulkan_checker.reproducible_artifact_bytes(
            "libpaired_relu_vulkan.so", first, repeat, sanitizer=True)
        self.assertEqual("sanitizer-temp-path-normalized", metadata["mode"])
        self.assertEqual(1, metadata["normalized_temp_paths"])
        self.assertEqual("head:/ncnn-compile-XXXXXX/vulkan_model_wrapper.bc:tail",
                         normalized.decode())

    def test_unrelated_sanitizer_binary_or_release_difference_is_rejected(self):
        first = b"head:/tmp/ncnn-compile-abc123/vulkan_model_wrapper.bc:tail"
        different_code = b"head:/tmp/ncnn-compile-987xyz/vulkan_model_wrapper.bc:tail!"
        self.assertIsNone(native_vulkan_checker.reproducible_artifact_bytes(
            "libpaired_relu_vulkan.so", first, different_code, sanitizer=True)[0])
        self.assertIsNone(native_vulkan_checker.reproducible_artifact_bytes(
            "libpaired_relu_vulkan.so", first, first + b"!", sanitizer=False)[0])
        self.assertIsNone(native_vulkan_checker.reproducible_artifact_bytes(
            "paired_relu_vulkan.json", first, different_code, sanitizer=True)[0])

    def test_checker_rehashes_sanitizer_normalized_and_byte_identical_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            base = stage / "test/Native/product"
            first_dir, repeat_dir = base / "vulkan", base / "vulkan-repeat"
            first_dir.mkdir(parents=True)
            repeat_dir.mkdir()
            paths = {
                "libpaired_relu_vulkan.so": (
                    b"prefix /tmp/ncnn-compile-abc123/vulkan_model_wrapper.bc suffix",
                    b"prefix /tmp/ncnn-compile-987xyz/vulkan_model_wrapper.bc suffix"),
                "paired_relu_vulkan.h": (b"header", b"header"),
                "paired_relu_vulkan.json": (b"manifest", b"manifest"),
                "paired_relu_vulkan.plan.json": (b"plan", b"plan"),
            }
            records = {}
            for name, (first, repeat) in paths.items():
                (first_dir / name).write_bytes(first)
                (repeat_dir / name).write_bytes(repeat)
                normalized, detail = native_vulkan_checker.reproducible_artifact_bytes(
                    name, first, repeat, sanitizer=True)
                records[name] = {
                    **detail,
                    "first_sha256": audit.sha256_bytes(first),
                    "repeat_sha256": audit.sha256_bytes(repeat),
                    "canonical_sha256": audit.sha256_bytes(normalized),
                }
            findings = []
            audit._audit_product_reproducibility(
                stage, base, {"vulkan_reproducibility": records},
                {"address", "undefined"}, [], findings)
            self.assertEqual([], findings)
            (repeat_dir / "libpaired_relu_vulkan.so").write_bytes(
                paths["libpaired_relu_vulkan.so"][1] + b"!")
            findings = []
            audit._audit_product_reproducibility(
                stage, base, {"vulkan_reproducibility": records},
                {"address", "undefined"}, [], findings)
            self.assertIn("product-reproducibility-hash",
                          {finding["code"] for finding in findings})


if __name__ == "__main__":
    unittest.main()
