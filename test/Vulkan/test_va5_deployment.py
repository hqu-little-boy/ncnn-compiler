#!/usr/bin/env python3
"""Self-tests for the VA5 clean-deployment evidence helpers.

These tests exercise the pure validation logic of run_va5_deployment.py with
synthetic data: manifest contract checks, deployment minimality, environment
scrubbing, and the evidence schema.  They deliberately avoid compiling any
model; the device/no-device integration runs through CTest registration.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parent / "run_va5_deployment.py"
spec = importlib.util.spec_from_file_location("ncnn_va5_deployment", MODULE_PATH)
va5 = importlib.util.module_from_spec(spec)
sys.modules.setdefault("ncnn_va5_deployment", va5)
spec.loader.exec_module(va5)


class FakeArgs:
    work_dir = Path("/tmp/va5-fake")
    sanitizer_runtime_dir = None
    icd = None
    device = False


class ManifestContractTests(unittest.TestCase):
    def test_accepts_va5_contract_manifest(self):
        manifest = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_model_manifest",
            **va5.HOST_CONTRACT,
            "host_codegen_identity": "a" * 64,
            "dependencies": ["libvulkan.so.1", "libc.so.6"],
            "sanitizer_dependencies": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("m_vulkan.h", "libm_vulkan.so",
                         "m_vulkan.json", "m_vulkan.plan.json"):
                (output / name).write_bytes(b"x")
            (output / "m_vulkan.json").write_text(json.dumps(manifest))
            args = FakeArgs()
            args.work_dir = output
            identity = va5.check_published_artifacts(args, "m", output)
            self.assertIn("m_vulkan.h", identity["sha256s"])

    def test_rejects_generic_plan_runtime_contract(self):
        manifest = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_model_manifest",
            "host_implementation": "generic-plan-runtime",
            "host_codegen_version": 1,
            "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
            "execution_plan_usage": "build-evidence-only",
            "host_codegen_identity": "a" * 64,
            "dependencies": ["libvulkan.so.1"],
            "sanitizer_dependencies": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("m_vulkan.h", "libm_vulkan.so",
                         "m_vulkan.json", "m_vulkan.plan.json"):
                (output / name).write_bytes(b"x")
            (output / "m_vulkan.json").write_text(json.dumps(manifest))
            args = FakeArgs()
            args.work_dir = output
            with self.assertRaises(va5.DeploymentError):
                va5.check_published_artifacts(args, "m", output)

    def test_rejects_llvm_dependency_in_manifest(self):
        manifest = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_model_manifest",
            **va5.HOST_CONTRACT,
            "host_codegen_identity": "a" * 64,
            "dependencies": ["libvulkan.so.1", "libLLVM.so.21.1"],
            "sanitizer_dependencies": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("m_vulkan.h", "libm_vulkan.so",
                         "m_vulkan.json", "m_vulkan.plan.json"):
                (output / name).write_bytes(b"x")
            (output / "m_vulkan.json").write_text(json.dumps(manifest))
            args = FakeArgs()
            args.work_dir = output
            with self.assertRaises(va5.DeploymentError):
                va5.check_published_artifacts(args, "m", output)

    def test_rejects_missing_sanitizer_classification(self):
        manifest = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_model_manifest",
            **va5.HOST_CONTRACT,
            "host_codegen_identity": "a" * 64,
            "dependencies": ["libvulkan.so.1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("m_vulkan.h", "libm_vulkan.so",
                         "m_vulkan.json", "m_vulkan.plan.json"):
                (output / name).write_bytes(b"x")
            (output / "m_vulkan.json").write_text(json.dumps(manifest))
            args = FakeArgs()
            args.work_dir = output
            with self.assertRaises(va5.DeploymentError):
                va5.check_published_artifacts(args, "m", output)


class DeploymentDirectoryTests(unittest.TestCase):
    def test_minimal_deployment_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            published = work / "published"
            published.mkdir()
            (published / "m_vulkan.h").write_bytes(b"h")
            (published / "libm_vulkan.so").write_bytes(b"s")
            args = FakeArgs()
            args.work_dir = work
            deploy = va5.build_deployment_directory(args, "m", published)
            self.assertEqual(
                sorted(item["file"] for item in deploy["deployed"]),
                ["libm_vulkan.so", "m_vulkan.h"])

    def test_extra_file_breaks_minimality(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            published = work / "published"
            published.mkdir()
            (published / "m_vulkan.h").write_bytes(b"h")
            (published / "libm_vulkan.so").write_bytes(b"s")
            args = FakeArgs()
            args.work_dir = work
            deploy = va5.build_deployment_directory(args, "m", published)
            (Path(deploy["directory"]) / "stray.json").write_bytes(b"x")
            with self.assertRaises(va5.DeploymentError):
                va5.verify_deployment_directory(deploy)

    def test_tampered_deployed_file_breaks_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            published = work / "published"
            published.mkdir()
            (published / "m_vulkan.h").write_bytes(b"h")
            (published / "libm_vulkan.so").write_bytes(b"s")
            args = FakeArgs()
            args.work_dir = work
            deploy = va5.build_deployment_directory(args, "m", published)
            (Path(deploy["directory"]) / "libm_vulkan.so").write_bytes(b"t")
            with self.assertRaises(va5.DeploymentError):
                va5.verify_deployment_directory(deploy)


class EnvironmentScrubTests(unittest.TestCase):
    def test_inherited_search_paths_are_dropped_and_recorded(self):
        calls = {}

        def fake_environ_copy():
            calls["copied"] = True
            return {
                "LD_LIBRARY_PATH":
                    "/opt/llvm/lib:/build/dir:/project/lib:/usr/lib/custom",
                "VK_DRIVER_FILES": "/should/be/removed.json",
            }

        import os
        original = os.environ.copy
        os.environ.copy = fake_environ_copy
        try:
            args = FakeArgs()
            args.icd = "/fake/icd.json"
            args.device = True
            env = va5.scrubbed_environment(va5, args,
                                           allow_sanitizer_runtime=False)
        finally:
            os.environ.copy = original
        # Allowlist behavior: nothing inherited survives, and the scrubbed
        # segments are recorded for the evidence file.
        self.assertNotIn("LD_LIBRARY_PATH", env)
        self.assertEqual(
            env["_NCNN_VA5_SCRUBBED_LD_LIBRARY_PATH"],
            "/opt/llvm/lib:/build/dir:/project/lib:/usr/lib/custom")
        self.assertNotIn("VK_DRIVER_FILES", env)
        self.assertEqual(env["VK_ICD_FILENAMES"], "/fake/icd.json")

    def test_sanitizer_runtime_dir_is_kept_when_allowed(self):
        import os
        original = os.environ.copy
        os.environ.copy = lambda: {"LD_LIBRARY_PATH": "/opt/llvm/lib"}
        try:
            args = FakeArgs()
            args.icd = None
            args.device = False
            args.sanitizer_runtime_dir = "/opt/asan-runtime"
            with tempfile.TemporaryDirectory() as directory:
                args.work_dir = Path(directory)
                env = va5.scrubbed_environment(va5, args,
                                               allow_sanitizer_runtime=True)
                self.assertEqual(env.get("LD_LIBRARY_PATH"), "/opt/asan-runtime")
        finally:
            os.environ.copy = original

    def test_no_device_profile_points_at_missing_icd(self):
        import os
        args = FakeArgs()
        args.icd = None
        args.device = False
        with tempfile.TemporaryDirectory() as directory:
            args.work_dir = Path(directory)
            env = va5.scrubbed_environment(va5, args,
                                           allow_sanitizer_runtime=False)
            self.assertEqual(env["VK_ICD_FILENAMES"],
                             str(Path(directory) / "no-available-icd.json"))
            self.assertFalse(Path(env["VK_ICD_FILENAMES"]).exists())
        self.assertNotIn("VK_INSTANCE_LAYERS", env)


class EvidenceSchemaTests(unittest.TestCase):
    def test_host_contract_is_pinned(self):
        self.assertEqual(va5.HOST_CONTRACT, {
            "host_implementation": "model-fixed-run",
            "host_codegen_version": 2,
            "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
            "execution_plan_usage": "build-evidence-only",
        })

    def test_forbidden_library_path_markers_cover_build_trees(self):
        joined = ";".join(va5.FORBIDDEN_LIBRARY_PATH_MARKERS).lower()
        for candidate in ("/llvm", "libllvm", "ncnn-compiler-stage-",
                          "ncnn-compiler-va"):
            self.assertIn(candidate, joined)


if __name__ == "__main__":
    unittest.main()
