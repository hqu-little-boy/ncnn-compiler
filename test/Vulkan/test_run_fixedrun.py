#!/usr/bin/env python3
"""Regression tests for fixed-run producer tool-role isolation."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fixedrun

BASE_REVISION = "6094cb56abe24cdbcde84fbb72813da8aa24c147"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixedRunArchiveTest(unittest.TestCase):
    def test_target_clean_prevents_duplicate_members_and_is_stable(self) -> None:
        cmake_value = shutil.which("cmake")
        ar_value = shutil.which("ar")
        cxx_value = shutil.which("c++")
        self.assertIsNotNone(cmake_value)
        self.assertIsNotNone(ar_value)
        self.assertIsNotNone(cxx_value)
        cmake = Path(cmake_value).resolve()
        ar = Path(ar_value).resolve()
        cxx = Path(cxx_value).resolve()

        with tempfile.TemporaryDirectory(prefix="ncnn-fixedrun-archive-") as name:
            root = Path(name)
            helper_dir = root / "lib" / "VulkanDeviceHelper"
            target_dir = helper_dir / "CMakeFiles" / "NCNNVulkanDeviceHelper.dir"
            object_dir = target_dir
            target_dir.mkdir(parents=True)
            helper = helper_dir / "libNCNNVulkanDeviceHelper.a"
            object_path = object_dir / "VulkanDeviceHelper.cpp.o"
            source = root / "VulkanDeviceHelper.cpp"
            source.write_text(
                "int ncnn_fixedrun_archive_probe() { return 7; }\n",
                encoding="utf-8")
            subprocess.run([str(cxx), "-c", str(source), "-o", str(object_path)],
                           cwd=root, check=True, capture_output=True)

            for _ in range(45):
                subprocess.run([str(ar), "q", str(helper), str(object_path)],
                               cwd=helper_dir, check=True, capture_output=True)
            duplicate_members = subprocess.run(
                [str(ar), "t", str(helper)], cwd=helper_dir, check=True,
                capture_output=True, text=True).stdout.splitlines()
            self.assertEqual(len(duplicate_members), 45)
            self.assertEqual(len(set(duplicate_members)), 1)

            clean_script = target_dir / "cmake_clean_target.cmake"
            clean_script.write_text(
                'file(REMOVE_RECURSE\n  "libNCNNVulkanDeviceHelper.a"\n)\n',
                encoding="utf-8")
            link_path = target_dir / "link.txt"
            link_path.write_text(
                f"{ar} qc {helper.name} {object_path.relative_to(helper_dir)}\n"
                f"{shutil.which('ranlib')} {helper.name}\n",
                encoding="utf-8")
            (root / "Makefile").write_text(
                f"CMAKE_COMMAND = {cmake}\n", encoding="utf-8")
            (root / "CMakeCache.txt").write_text(
                f"CMAKE_AR:FILEPATH={ar}\n", encoding="utf-8")

            evidence = run_fixedrun.rebuild_helper_archive(
                helper, object_path, link_path, clean_script, cmake, root,
                root / "test" / "Vulkan" / "helper-evidence", dict(os.environ),
                "archive_probe")
            self.assertEqual(evidence["prior_archive"]["duplicate_member_count"], 44)
            self.assertEqual(len(evidence["archive_rebuilds"]), 2)
            self.assertEqual(evidence["member_list"], [object_path.name])
            self.assertTrue(all(row["equals_compile_object"]
                                for row in evidence["member_records"]))
            self.assertEqual(evidence["archive_rebuilds"][0]["archive_sha256"],
                             evidence["archive_rebuilds"][1]["archive_sha256"])
            snapshots = [row["archive"] for row in evidence["archive_rebuilds"]]
            self.assertEqual(len({item["path"] for item in snapshots}), 2)
            self.assertEqual(evidence["archive"], snapshots[-1])
            self.assertEqual(evidence["archive_target_path"], str(helper.resolve()))
            self.assertEqual(evidence["archive_target_sha256"], sha256(helper))
            self.assertEqual(
                evidence["archive_rebuilds"][0]["clean_script_sha256"],
                sha256(clean_script))
            self.assertEqual(
                Path(evidence["archive_rebuilds"][0]["clean_command"]["execution"]["argv"][0]),
                cmake)
            helper.write_bytes(b"later target mutation")
            for snapshot in snapshots:
                snapshot_path = root / snapshot["path"]
                self.assertEqual(sha256(snapshot_path), snapshot["sha256"])


class FixedRunToolRoleTest(unittest.TestCase):
    def test_project_opt_does_not_override_upstream_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncnn-fixedrun-tools-") as name:
            root = Path(name)
            stage = root / "stage"
            old_root = root / "legacy"
            old_root.mkdir()
            project_driver = old_root / "ncnn-mlir-driver"
            project_opt = old_root / "ncnn-mlir-opt"
            extractor = old_root / "ncnn-vulkan-spirv-extract"
            upstream_opt = old_root / "upstream-mlir-opt"
            compiler = old_root / "ncnn-compile"
            cache = old_root / "CMakeCache.txt"
            clean_status = old_root / "source-status.txt"
            files = {
                project_driver: b"project-driver",
                project_opt: b"project-opt",
                extractor: b"spirv-extractor",
                upstream_opt: b"upstream-mlir-opt",
                compiler: b"legacy-compiler",
                clean_status: b"",
            }
            for path, contents in files.items():
                path.write_bytes(contents)
            cache.write_text(
                "NCNN_MLIR_OPT_EXECUTABLE:FILEPATH=" + str(upstream_opt) + "\n",
                encoding="utf-8")
            tools = [
                {"role": "ncnn-mlir-driver", "path": str(project_driver),
                 "sha256": sha256(project_driver)},
                {"role": "ncnn-mlir-opt", "path": str(project_opt),
                 "sha256": sha256(project_opt)},
                {"role": "ncnn-vulkan-spirv-extract", "path": str(extractor),
                 "sha256": sha256(extractor)},
            ]
            manifest = {
                "schema": "ncnn-vulkan-legacy-reference/v1",
                "role": "legacy-product-reference",
                "git_revision": BASE_REVISION,
                "source_root": str(old_root / "source"),
                "source_fingerprint": "fixture-source-fingerprint",
                "compiler": {"path": str(compiler), "sha256": sha256(compiler)},
                "tools": tools,
                "build": {"cmake_cache": {
                    "path": str(cache), "sha256": sha256(cache)}},
                "clean_source_status": {
                    "path": str(clean_status), "sha256": sha256(clean_status)},
            }
            manifest_path = old_root / "legacy-reference.json"
            original_manifest = json.dumps(manifest, sort_keys=True).encode() + b"\n"
            manifest_path.write_bytes(original_manifest)
            manifest_hash = sha256(manifest_path)
            with mock.patch.dict(os.environ, {
                "NCNN_MLIR_OPT": str(project_opt),
                "MLIR_OPT": str(project_opt),
                "NCNN_VULKAN_FIXEDRUN_PROFILE": "sanitize",
            }):
                record = run_fixedrun.copy_legacy_reference(old_root, stage)
            self.assertEqual(record["status"], "verified-independent-source")
            self.assertEqual(record["tools"]["opt"]["path"], str(project_opt))
            self.assertEqual(record["tools"]["mlir-opt"]["path"], str(upstream_opt))
            self.assertNotEqual(record["tools"]["opt"]["path"],
                                record["tools"]["mlir-opt"]["path"])
            self.assertEqual(sha256(manifest_path), manifest_hash)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertTrue(record["tools"]["mlir-opt"]["staged"]["path"].endswith(
                "upstream-mlir-opt"))


if __name__ == "__main__":
    unittest.main()
