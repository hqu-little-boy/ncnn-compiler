#!/usr/bin/env python3
"""Regression tests for fresh-gate identity, logging and negative-test checks."""

import argparse
import hashlib
import importlib.util
import io
import json
import struct
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import gate_budget_evidence
import record_spirv_validation
import run_gate
import run_v0
import sanitizer_toolchain


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def repository(self):
        source = self.root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        (source / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        (source / "model.cpp").write_text("original\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        return source

    def test_fingerprint_detects_dirty_bytes_not_just_git_status(self):
        source = self.repository()
        original = run_gate.source_fingerprint(source)
        (source / "model.cpp").write_text("changed1\n", encoding="utf-8")
        changed = run_gate.source_fingerprint(source)
        (source / "model.cpp").write_text("changed2\n", encoding="utf-8")
        self.assertNotEqual(original, changed)
        self.assertNotEqual(changed, run_gate.source_fingerprint(source))

    def test_fingerprint_is_independent_of_staging(self):
        source = self.repository()
        (source / "new.py").write_text("new\n", encoding="utf-8")
        before = run_gate.source_fingerprint(source)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        self.assertEqual(before, run_gate.source_fingerprint(source))

    def test_ignored_builds_do_not_change_source_identity(self):
        source = self.repository()
        before = run_gate.source_fingerprint(source)
        (source / "ignored").mkdir()
        (source / "ignored/artifact.so").write_bytes(b"generated")
        self.assertEqual(before, run_gate.source_fingerprint(source))

    def test_deletion_and_symlink_change_identity(self):
        source = self.repository()
        before = run_gate.source_fingerprint(source)
        (source / "model.cpp").unlink()
        deleted = run_gate.source_fingerprint(source)
        self.assertNotEqual(before, deleted)
        (source / "model.cpp").symlink_to("missing-target")
        self.assertNotEqual(deleted, run_gate.source_fingerprint(source))

    def test_fingerprint_accepts_uninitialized_nested_gitlink_by_oid(self):
        child = self.root / "child"
        child.mkdir()
        subprocess.run(["git", "init", "-q", str(child)], check=True)
        subprocess.run(["git", "-C", str(child), "config", "user.name", "Unit"], check=True)
        subprocess.run(["git", "-C", str(child), "config", "user.email", "unit@example.invalid"], check=True)
        (child / "tracked.txt").write_text("submodule content\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(child), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(child), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(child), "rev-parse", "HEAD"], text=True).strip()
        parent = self.root / "parent"
        parent.mkdir()
        subprocess.run(["git", "init", "-q", str(parent)], check=True)
        (parent / "nested").mkdir()
        subprocess.run(["git", "-C", str(parent), "update-index", "--add",
                        "--cacheinfo", f"160000,{commit},nested"], check=True)

        uninitialized = run_gate.source_fingerprint(parent)
        second_read = run_gate.source_fingerprint(parent)
        self.assertEqual(uninitialized, second_read)
        subprocess.run(["git", "clone", "-q", str(child), str(parent / "nested")],
                       check=True)
        self.assertEqual(uninitialized, run_gate.source_fingerprint(parent))
        (parent / "nested/tracked.txt").write_text("dirty submodule\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "gitlink worktree is dirty"):
            run_gate.source_fingerprint(parent)

    def test_offline_tool_inventory_does_not_require_vulkaninfo(self):
        import shutil

        executable = {
            "CMAKE_MAKE_PROGRAM": shutil.which("make"),
            "CMAKE_C_COMPILER": shutil.which("clang-21"),
            "CMAKE_CXX_COMPILER": shutil.which("clang++-21"),
            "CLANG": shutil.which("clang-21"),
            "LLVM_NM": shutil.which("llvm-nm-21"),
            "LLVM_READELF": shutil.which("llvm-readelf-21"),
            "LLVM_AS": shutil.which("llvm-as-21"),
            "LLVM_OBJDUMP": shutil.which("llvm-objdump-21"),
            "LLVM_SIZE": shutil.which("llvm-size-21"),
            "LLVM_MCA": shutil.which("llvm-mca-21"),
            "NCNN_SPIRV_VAL_EXECUTABLE": shutil.which("spirv-val"),
            "NCNN_SPIRV_DIS_EXECUTABLE": shutil.which("spirv-dis"),
            "NCNN_MLIR_OPT_EXECUTABLE": shutil.which("mlir-opt-21"),
            "MLIR_TRANSLATE": shutil.which("mlir-translate-21"),
            "MLIR_OPT": shutil.which("mlir-opt-21"),
            "FILECHECK": shutil.which("FileCheck-21"),
            "Python3_EXECUTABLE": sys.executable,
            "NCNN_LIT_PY": "/usr/lib/llvm-21/build/utils/lit/lit.py",
            "NCNN_ENABLE_VULKAN_V0_TESTS": "OFF",
        }
        self.assertTrue(all(executable.values()))
        identities = run_gate.tool_identities(
            argparse.Namespace(openmp_dir=None, toolchain_identity=None), executable)
        self.assertNotIn("vulkaninfo", {item["role"] for item in identities})
        matching_lit = executable.pop("NCNN_LIT_PY")
        identities = run_gate.tool_identities(
            argparse.Namespace(openmp_dir=None, toolchain_identity=None), executable)
        lit_identity = next(item for item in identities if item["role"] == "lit")
        self.assertEqual(Path(lit_identity["path"]).resolve(), Path(matching_lit).resolve())
        executable["NCNN_LIT_PY"] = matching_lit
        executable["NCNN_ENABLE_VULKAN_V0_TESTS"] = "ON"
        executable["VULKANINFO"] = shutil.which("vulkaninfo")
        identities = run_gate.tool_identities(
            argparse.Namespace(openmp_dir=None, toolchain_identity=None), executable)
        self.assertIn("vulkaninfo", {item["role"] for item in identities})
        executable.pop("Python3_EXECUTABLE")
        identities = run_gate.tool_identities(
            argparse.Namespace(openmp_dir=None, toolchain_identity=None), executable)
        python_identity = next(item for item in identities if item["role"] == "python")
        self.assertEqual(Path(python_identity["path"]).resolve(),
                         Path(sys.executable).resolve())

    def test_stage_id_defaults_to_va0_and_va3_is_explicit(self):
        default = run_gate.parse_args(["--profile", "release"])
        planner = run_gate.parse_args(["--profile", "release", "--stage-id", "VA1"])
        lifecycle = run_gate.parse_args(["--profile", "release", "--stage-id", "VA3"])
        fixed_run = run_gate.parse_args(["--profile", "release", "--stage-id", "VA4"])
        self.assertEqual("VA0", default.stage_id)
        self.assertEqual("VA1", planner.stage_id)
        self.assertEqual("VA3", lifecycle.stage_id)
        self.assertEqual("VA4", fixed_run.stage_id)

    def test_va4_allows_unqualified_technical_run_and_binds_optional_legacy_root(self):
        args = run_gate.parse_args(["--profile", "release", "--stage-id", "VA4"])
        command = run_gate.configuration(args, self.root)
        self.assertFalse(any(item.startswith(
            "-DNCNN_VULKAN_FIXEDRUN_LEGACY_ROOT=") for item in command))
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "legacy-reference.json").write_text("{}", encoding="utf-8")
        args = run_gate.parse_args([
            "--profile", "release", "--stage-id", "VA4",
            "--fixedrun-legacy-root", str(legacy)])
        command = run_gate.configuration(args, self.root)
        self.assertIn(f"-DNCNN_VULKAN_FIXEDRUN_LEGACY_ROOT={legacy.resolve()}",
                      command)

    def test_profiles_request_exact_devices(self):
        for profile, devices in (("release", "nvidia;llvmpipe"),
                                 ("software", "llvmpipe"),
                                 ("hardware", "nvidia")):
            args = run_gate.parse_args(["--profile", profile])
            command = run_gate.configuration(args, self.root)
            self.assertIn(f"-DNCNN_VULKAN_TEST_DEVICES={devices}", command)
        args = run_gate.parse_args(["--profile", "offline"])
        self.assertIn("-DNCNN_ENABLE_VULKAN_V0_TESTS=OFF",
                      run_gate.configuration(args, self.root))

    def test_cannot_disable_required_gate_options(self):
        for override in ("-DBUILD_TESTING=OFF",
                         "-DNCNN_ENABLE_VULKAN_PRODUCT:BOOL=OFF",
                         "-DCMAKE_BUILD_TYPE=Debug",
                         "-DCOMPILER_ENABLE_FORMAT_TARGETS=OFF"):
            args = run_gate.parse_args(["--profile", "release",
                                        f"--cmake-arg={override}"])
            with self.assertRaises(ValueError):
                run_gate.configuration(args, self.root)

    def test_release_cannot_expand_fixture_compile_budgets(self):
        args = run_gate.parse_args([
            "--profile", "release",
            "--cmake-arg=-Dncnn_model_compile_budget_resnet101=900"])
        with self.assertRaises(ValueError):
            run_gate.configuration(args, self.root)

    def test_sanitize_toolchain_identity_fails_closed_before_hashing(self):
        identity = self.root / "toolchain-identity.json"
        identity.write_text(json.dumps({"schema": "unknown", "status": "verified"}),
                            encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            sanitizer_toolchain.audit_and_bundle(
                self.root / "stage", identity, self.root / "install",
                self.root / "lit.py", self.root / "openmp", self.root / "clang")

    def test_constructor_failure_preserves_gate_state_and_archive(self):
        source = self.root / "not-a-git-checkout"
        source.mkdir()
        for stage_id in ("VA0", "VA1", "VA3", "VA4"):
            args = run_gate.parse_args([
                "--profile", "release", "--stage-id", stage_id,
                "--source-dir", str(source), "--build-root", str(self.root),
                "--archive",
            ])
            with self.assertRaisesRegex(RuntimeError, "gate setup failed; preserved stage"):
                run_gate.Gate(args)
        for stage_id in ("va0", "va1"):
            stages = [path for path in self.root.glob(
                f"ncnn-compiler-stage-{stage_id}-release-*") if path.is_dir()]
            self.assertEqual(len(stages), 1)
            state = json.loads((stages[0] / "gate-state.json").read_text())
            self.assertFalse(state["completed"])
            self.assertEqual(stage_id.upper(), state["stage_id"])
            self.assertEqual("legacy-product", state["host_execution_role"])
            self.assertFalse(state["host_aot_candidate"])
            expected_role = {"VA3": "lifecycle-only", "VA4": "fixed-run"}.get(
                stage_id, "none")
            self.assertEqual(expected_role, state["candidate_role"])
            self.assertIn("gate setup failed", state["failure"])
            archive = stages[0].with_suffix(".evidence.tar.gz")
            self.assertTrue(archive.is_file())

    def test_sanitize_requires_real_prerequisites(self):
        args = run_gate.parse_args(["--profile", "sanitize"])
        with self.assertRaisesRegex(ValueError, "sanitize requires"):
            run_gate.configuration(args, self.root)

    def test_budget_exception_requires_same_source_hard_failure(self):
        fixture = "pp_ocrv5_server_rec"
        record_path = self.root / "budget.json"
        log_path = self.root / "build.log"
        source_hash = "a" * 64
        log_path.write_text(
            f"fixture {fixture} 编译耗时 302s 超过硬预算 300s——编译爆炸回归\n",
            encoding="utf-8")
        record = {
            "schema_version": 1,
            "kind": "ncnn.sanitizer_budget_failure",
            "source_fingerprint": source_hash,
            "profile": "sanitize",
            "fixture": fixture,
            "previous_budget_seconds": 300,
            "requested_budget_seconds": 450,
            "elapsed_seconds": 302,
            "log_file": log_path.name,
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        accepted = gate_budget_evidence.verify_budget_evidence(
            record_path, "ncnn_model_compile_budget_pp_ocrv5_server_rec",
            450, source_hash)
        self.assertEqual(accepted["record"]["elapsed_seconds"], 302)
        for key, seconds, fingerprint in (
                ("ncnn_model_compile_budget_resnet101", 900, source_hash),
                ("ncnn_model_compile_budget_pp_ocrv5_server_rec", 600,
                 source_hash),
                ("ncnn_model_compile_budget_pp_ocrv5_server_rec", 450,
                 "b" * 64)):
            with self.assertRaises(ValueError):
                gate_budget_evidence.verify_budget_evidence(
                    record_path, key, seconds, fingerprint)
        log_path.write_text("unrelated compile error\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "raw log hash"):
            gate_budget_evidence.verify_budget_evidence(
                record_path, "ncnn_model_compile_budget_pp_ocrv5_server_rec",
                450, source_hash)

    def test_report_scan_distinguishes_flags_from_diagnostics(self):
        log = self.root / "build.log"
        log.write_text("-fsanitize=address,undefined libclang_rt.asan-x86_64.so\n"
                       "NYI: non-trivial layout map\n", encoding="utf-8")
        self.assertEqual([], run_gate.diagnostics([log]))
        log.write_text("ERROR: LeakSanitizer: detected memory leaks\n"
                       "file.cpp:9: runtime error: null pointer\n",
                       encoding="utf-8")
        self.assertEqual(2, len(run_gate.diagnostics([log])))

    def test_project_warnings_are_not_suppressed(self):
        source = self.root / "compiler"
        log = self.root / "tidy.log"
        log.write_text(f"{source}/lib/test.cpp:1:2: warning: project problem\n"
                       f"{source}/test/third_party/ncnn/x.cpp:1:2: warning: upstream\n"
                       f"{source}/test/Numerical/bench/model.mlir:1:2: warning: expected pass result\n"
                       "/usr/include/llvm.h:1:2: warning: upstream\n",
                       encoding="utf-8")
        self.assertEqual(1, len(run_gate.project_warnings(log, source)))

    def test_gate_snapshots_direct_v2_kernel_sources(self):
        source = Path(__file__).resolve().parents[2]
        stage = self.root / "stage-source-snapshot"
        stage.mkdir()
        gate = run_gate.Gate.__new__(run_gate.Gate)
        gate.source = source
        gate.stage = stage
        gate.state = {}
        gate.save = lambda: None
        gate.snapshot_inputs()
        record = next(item for item in gate.state["source_inputs"]
                      if item.get("source") ==
                      "test/Conversion/NCNNToVulkan/lower-copy.mlir")
        self.assertEqual(
            run_gate.sha256(stage / record["path"]),
            run_gate.sha256(source / record["source"]))

    def test_archive_contains_hashes_and_refuses_overwrite(self):
        stage = self.root / "stage"
        evidence = stage / "test/Vulkan/evidence/v5"
        evidence.mkdir(parents=True)
        model = evidence / "output.bin"
        model.write_bytes(b"numerical evidence")
        (stage / "gate-state.json").write_text("{}\n", encoding="utf-8")
        archive = self.root / "evidence.tar.gz"
        result = run_gate.archive_stage(stage, archive)
        self.assertEqual(run_gate.sha256(archive), result["sha256"])
        with tarfile.open(archive) as bundle:
            manifest = json.load(bundle.extractfile("gate-archive-manifest.json"))
            relative = str(model.relative_to(stage))
            self.assertEqual(run_gate.sha256(model), manifest["files"][relative])
            self.assertIn(relative, bundle.getnames())
        with self.assertRaisesRegex(ValueError, "overwrite"):
            run_gate.archive_stage(stage, archive)

    def test_archive_reopen_rejects_missing_tampered_extra_and_duplicate_members(self):
        content = b"captured tensor bytes"
        manifest = {"schema_version": 1, "kind": "ncnn.vulkan_evidence_archive",
                    "files": {"data.f32": hashlib.sha256(content).hexdigest()}}
        manifest_bytes = json.dumps(manifest).encode()
        for case in ("valid", "missing", "tampered", "extra", "duplicate"):
            with self.subTest(case=case):
                path = self.root / f"{case}.tar.gz"
                entries = [("gate-archive-manifest.json", manifest_bytes)]
                if case != "missing":
                    entries.append(("data.f32", b"altered" if case == "tampered" else content))
                if case == "extra":
                    entries.append(("unlisted.bin", b"extra"))
                if case == "duplicate":
                    entries.append(("data.f32", content))
                with tarfile.open(path, "w:gz") as bundle:
                    for name, value in entries:
                        member = tarfile.TarInfo(name)
                        member.size = len(value)
                        bundle.addfile(member, io.BytesIO(value))
                if case == "valid":
                    result = run_gate.verify_archive(path)
                    self.assertEqual("verified", result["status"])
                    self.assertEqual(1, result["file_count"])
                else:
                    with self.assertRaises(ValueError):
                        run_gate.verify_archive(path)

    def test_archive_reopen_reads_unsorted_manifest_in_tar_member_order(self):
        path = self.root / "ordered-members.tar.gz"
        payloads = {"z-last.bin": b"last", "a-first.bin": b"first"}
        manifest = {"schema_version": 1, "kind": "ncnn.vulkan_evidence_archive",
                    "files": {name: hashlib.sha256(value).hexdigest()
                              for name, value in payloads.items()}}
        entries = [("gate-archive-manifest.json", json.dumps(manifest).encode())]
        entries.extend((name, payloads[name]) for name in sorted(payloads))
        with tarfile.open(path, "w:gz") as bundle:
            for name, value in entries:
                member = tarfile.TarInfo(name)
                member.size = len(value)
                bundle.addfile(member, io.BytesIO(value))
        result = run_gate.verify_archive(path)
        self.assertEqual("verified", result["status"])
        self.assertEqual(2, result["file_count"])

    def test_archive_receipt_is_nonrecursive_and_binds_actual_tar_bytes(self):
        stage = self.root / "receipt-stage"
        stage.mkdir()
        (stage / "gate-state.json").write_text(json.dumps({
            "completed": True, "profile": "offline", "source_fingerprint": "a" * 64}))
        path = self.root / "receipt.evidence.tar.gz"
        result = run_gate.archive_stage(stage, path)
        receipt = json.loads((stage / "gate-archive-receipt.json").read_text())
        self.assertEqual(result["sha256"], receipt["archive"]["sha256"])
        self.assertEqual(result["verification"], receipt["verification"])
        self.assertEqual("a" * 64, receipt["source_fingerprint"])
        self.assertTrue(receipt["gate_completed"])
        with tarfile.open(path) as bundle:
            self.assertNotIn("gate-archive-receipt.json", bundle.getnames())
        import check_release_evidence
        checked = check_release_evidence.audit_stage_archive(stage, required=True)
        self.assertEqual("verified", checked["status"])
        path.write_bytes(path.read_bytes() + b"tamper")
        checked = check_release_evidence.audit_stage_archive(stage, required=True)
        self.assertEqual("invalid", checked["status"])
        self.assertTrue(checked["findings"])

    def test_va1_archive_must_include_audited_native_evidence_bytes(self):
        import check_release_evidence

        stage = self.root / "archive-inventory-stage"
        native = stage / "test/Native/vulkan-product-offline/host-program-evidence.json"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"audited native host-program evidence")
        unrelated = stage / "unrelated.txt"
        unrelated.write_bytes(b"self-described archive payload")
        state = {"completed": True, "profile": "release",
                 "source_fingerprint": "a" * 64}
        state_path = stage / "gate-state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        archived_files = {
            unrelated.relative_to(stage).as_posix(): run_gate.sha256(unrelated),
            state_path.relative_to(stage).as_posix(): run_gate.sha256(state_path)}
        archive_manifest = {"schema_version": 1,
                            "kind": "ncnn.vulkan_evidence_archive",
                            "files": archived_files}
        manifest_bytes = json.dumps(archive_manifest, indent=2).encode() + bytes([10])
        manifest_path = stage / "gate-archive-manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        archive_path = self.root / f"{stage.name}.evidence.tar.gz"
        with tarfile.open(archive_path, "w:gz") as bundle:
            for name, payload in (("gate-archive-manifest.json", manifest_bytes),
                                  ("gate-state.json", state_path.read_bytes()),
                                  ("unrelated.txt", unrelated.read_bytes())):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))
        verification = run_gate.verify_archive(
            archive_path, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        receipt = {"schema_version": 1,
                   "kind": "ncnn.vulkan_evidence_archive_receipt",
                   "archive": {"file": archive_path.name,
                               "sha256": run_gate.sha256(archive_path)},
                   "verification": verification,
                   "source_fingerprint": state["source_fingerprint"],
                   "profile": state["profile"], "gate_completed": True}
        (stage / "gate-archive-receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8")

        result = check_release_evidence.audit_stage_archive(stage, required=True)
        self.assertEqual("invalid", result["status"])
        self.assertIn("archive omits or changes audited stage artifact",
                      result["findings"][0]["message"])

    def test_archive_missing_receipt_and_invalid_tar_fail_closed(self):
        import check_release_evidence
        stage = self.root / "missing-archive-stage"
        stage.mkdir()
        result = check_release_evidence.audit_stage_archive(stage, required=True)
        self.assertEqual("archive-receipt-missing", result["findings"][0]["code"])
        invalid = self.root / "invalid.tar.gz"
        invalid.write_bytes(b"not a tar archive")
        with self.assertRaises(ValueError):
            run_gate.verify_archive(invalid)

    def test_archive_dereferences_only_stage_local_symlinks(self):
        stage = self.root / "stage-links"
        evidence = stage / "test/Vulkan/evidence/v5"
        evidence.mkdir(parents=True)
        target = evidence / "real.bin"
        target.write_bytes(b"shader data")
        link = evidence / "alias.bin"
        link.symlink_to(target.name)
        archive = self.root / "links.tar.gz"
        run_gate.archive_stage(stage, archive)
        with tarfile.open(archive) as bundle:
            entry = bundle.getmember(str(link.relative_to(stage)))
            self.assertTrue(entry.isfile())
            self.assertFalse(entry.issym())
            self.assertEqual(bundle.extractfile(entry).read(), b"shader data")

        unsafe_stage = self.root / "unsafe-stage"
        unsafe_evidence = unsafe_stage / "test/Vulkan/evidence/v5"
        unsafe_evidence.mkdir(parents=True)
        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside")
        (unsafe_evidence / "escape.bin").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes its stage"):
            run_gate.archive_stage(unsafe_stage, self.root / "unsafe.tar.gz")

    def test_validator_records_duplicate_shader_bytes_per_invocation(self):
        shader = self.root / "same.spv"
        shader.write_bytes(b"same shader bytes")
        output = self.root / "validation"
        output.mkdir()
        validator = self.root / "validator"
        validator.write_text("", encoding="utf-8")
        digest = hashlib.sha256(shader.read_bytes()).hexdigest()
        for _ in range(2):
            index = len(list(output.glob("validation-*.json")))
            saved = output / f"shader-{index:04d}.spv"
            log = output / f"validation-{index:04d}.json"
            saved.write_bytes(shader.read_bytes())
            log.write_text(json.dumps({
                "schema_version": 1,
                "kind": "ncnn.spirv_tool_invocation",
                "argv": [str(validator), "--target-env", "vulkan1.1", str(shader)],
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "shader": {"path": saved.name, "sha256": digest,
                           "size_bytes": saved.stat().st_size},
            }), encoding="utf-8")
        records = record_spirv_validation.validate_records(output, [digest, digest])
        self.assertEqual(len(records), 2)
        (output / "validation-0001.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "each product shader"):
            record_spirv_validation.validate_records(output, [digest, digest])

    def test_negative_test_requires_the_expected_cause(self):
        for status, output in ((0, "expected"), (1, "unrelated failure"),
                               (-11, "expected")):
            with patch("run_v0.subprocess.run", return_value=argparse.Namespace(
                    returncode=status, stdout="", stderr=output)):
                with self.assertRaises(RuntimeError):
                    run_v0.run_logged("negative", ["tool"], self.root,
                                      expect_success=False,
                                      expected_failure="expected")
        with patch("run_v0.subprocess.run", return_value=argparse.Namespace(
                returncode=1, stdout="", stderr="the expected failure")):
            run_v0.run_logged("negative", ["tool"], self.root,
                              expect_success=False, expected_failure="expected")


class OfflineCpuLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "Native/check_ncnn_compile_vulkan.py"
        spec = importlib.util.spec_from_file_location("offline_product_test", path)
        cls.product = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.product)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cpu = self.root / "cpu"
        self.cpu.mkdir()
        self.args = argparse.Namespace(
            work_dir=self.root, host_cxx="clang++-21",
            cxx_arg=["-fsanitize=address,undefined", "-shared-libasan",
                     "-fno-sanitize-recover=all"],
            loader_source=Path("vulkan_product_loader.cpp"),
        )
        self.environment = {"LD_LIBRARY_PATH": "/test/asan", "VK_ICD_FILENAMES": "/missing.json"}
        self.input = self.root / "input.f32"
        self.output = self.root / "output.f32"
        self.result = {"status": "success", "cpu_status": 0}
        self.returncode = 0
        self.stderr = ""
        self.output_bytes = struct.pack("<3f", 0.0, 0.0, 2.0)
        self.command = None
        self.compile_command = None

    def tearDown(self):
        self.temporary.cleanup()

    def compile(self, command, *, env):
        self.assertEqual(env, self.environment)
        self.compile_command = command
        Path(command[command.index("-o") + 1]).write_bytes(b"native loader")
        return subprocess.CompletedProcess(command, 0, "", "")

    def execute(self, command, *, capture_output, text, env):
        self.assertTrue(capture_output)
        self.assertTrue(text)
        self.assertEqual(env, self.environment)
        self.command = command
        Path(command[-1]).write_text(json.dumps(self.result), encoding="utf-8")
        self.output.write_bytes(self.output_bytes)
        return subprocess.CompletedProcess(command, self.returncode, "", self.stderr)

    def run_loader(self):
        with patch.object(self.product, "invoke", side_effect=self.compile), \
                patch.object(self.product.subprocess, "run", side_effect=self.execute):
            return self.product.run_cpu_loader(
                self.args, self.cpu, [-1.0, 0.0, 2.0], self.input, self.output,
                self.environment)

    def test_native_subprocess_keeps_sanitizer_flags_and_raw_identity(self):
        status, values, output, record = self.run_loader()
        self.assertEqual(status, 0)
        self.assertEqual(values, [0.0, 0.0, 2.0])
        self.assertEqual(output, self.output_bytes)
        self.assertEqual(self.input.read_bytes(), struct.pack("<3f", -1.0, 0.0, 2.0))
        for flag in self.args.cxx_arg:
            self.assertIn(flag, self.compile_command)
        self.assertIn("-DNCNN_PRODUCT_TEST_CPU_ONLY=1", self.compile_command)
        self.assertNotIn("-lvulkan", self.compile_command)
        self.assertEqual(self.command[1:4], ["paired_relu", str(self.cpu / "libpaired_relu.so"), "3"])
        self.assertEqual(len(self.command), 7)
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(record["result_sha256"], run_gate.sha256(self.root / record["result_file"]))
        self.assertEqual(len(record["streams"]), 4)
        for stream in record["streams"].values():
            self.assertEqual(stream["sha256"], run_gate.sha256(self.root / stream["file"]))

    def test_sanitizer_reports_fail_even_after_successful_exit(self):
        for diagnostic in ("ERROR: AddressSanitizer", "ERROR: LeakSanitizer",
                           "UndefinedBehaviorSanitizer", "runtime error:"):
            with self.subTest(diagnostic=diagnostic):
                directory = self.root / "cpu-loader-cpu"
                if directory.exists():
                    import shutil
                    shutil.rmtree(directory)
                self.stderr = diagnostic
                with self.assertRaisesRegex(RuntimeError, "sanitizer diagnostic"):
                    self.run_loader()
                self.assertEqual((directory / "run.stderr.log").read_text(), diagnostic)

    def test_nonzero_process_exit_fails_despite_success_json(self):
        self.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "CPU loader failed"):
            self.run_loader()

    def test_boolean_cpu_status_is_not_success(self):
        self.result["cpu_status"] = False
        with self.assertRaisesRegex(RuntimeError, "CPU loader reported failure"):
            self.run_loader()

    def test_nonzero_cpu_status_is_not_success(self):
        self.result["cpu_status"] = 3
        with self.assertRaisesRegex(RuntimeError, "CPU loader reported failure"):
            self.run_loader()

    def test_failed_result_is_not_success(self):
        self.result["status"] = "failure"
        with self.assertRaisesRegex(RuntimeError, "CPU loader reported failure"):
            self.run_loader()

    def test_short_raw_output_is_rejected(self):
        self.output_bytes = self.output_bytes[:-4]
        with self.assertRaisesRegex(RuntimeError, "CPU loader wrote 8 bytes, expected 12"):
            self.run_loader()


if __name__ == "__main__":
    unittest.main()
