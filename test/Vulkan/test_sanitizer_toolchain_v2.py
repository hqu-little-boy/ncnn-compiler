from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import check_release_evidence as release_audit
import sanitizer_toolchain
from sanitizer_toolchain_inventory import InventoryError, validate_inventory
import test_sanitizer_toolchain_inventory as inventory_fixtures


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SanitizerToolchainV2AdapterTest(unittest.TestCase):
    def setUp(self):
        self.fixture = inventory_fixtures.SanitizerToolchainInventoryTest()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.prefix = self.fixture.prefix
        self.openmp = self.root / "openmp"
        self.openmp.mkdir()
        self.libomp = self.openmp / "libomp.so"
        self.libomp.write_bytes(b"synthetic instrumented libomp")
        self.clang_dir = self.root / "toolchain-bin"
        self.clang_dir.mkdir()
        self.tools = {}
        for name, payload in (
            ("clang", b"clang"), ("clang++", b"clang++"),
            ("mlir-opt", b"mlir-opt"), ("mlir-translate", b"mlir-translate"),
            ("FileCheck", b"FileCheck"), ("lit.py", b"lit runner"),
        ):
            path = self.clang_dir / name
            path.write_bytes(payload)
            path.chmod(0o755)
            self.tools[name] = path
        self.configuration_file = self.root / "toolchain-config.json"
        self.configuration_file.write_text("{}\\n", encoding="utf-8")
        self.status = self.root / "verification.status"
        self.status.write_text("verification=passed\nexit=0\n", encoding="utf-8")
        self.lsan_binary = self.root / "libomp-lsan-smoke"
        self.lsan_binary.write_bytes(b"smoke binary")
        self.lsan_log = self.root / "libomp-lsan.log"
        self.lsan_log.write_text("leak check passed\n", encoding="utf-8")
        self.link_log = self.root / "libomp-lsan-link-audit.log"
        self.link_log.write_text(f"{self.libomp}\nlibclang_rt.asan.so\n", encoding="utf-8")
        self.dynamic_log = self.root / "core-tool-dynamic-audit.log"
        self.dynamic_log.write_text(json.dumps({
            "passed": True, "failures": [], "llvm_config_version": "21.1.8",
        }) + "\n", encoding="utf-8")
        self.tool_dynamic_logs = {}
        for name in ("mlir-opt", "mlir-translate", "FileCheck"):
            path = self.root / f"{name}.dynamic.log"
            path.write_text(
                "Dynamic section:\nNEEDED libclang_rt.asan.so\nNEEDED libc.so\n",
                encoding="utf-8")
            self.tool_dynamic_logs[name] = path
        self.libomp_dynamic_log = self.root / "libomp-dynamic-and-symbol-audit.log"
        self.libomp_dynamic_log.write_text(
            "READ_ELF_DYNAMIC:\nNEEDED libclang_rt.asan.so\nNEEDED libc.so\n",
            encoding="utf-8")
        self.libomp_ldd_log = self.root / "libomp-smoke-ldd.log"
        self.libomp_ldd_log.write_text(
            f"libclang_rt.asan.so\nlibomp.so => {self.libomp}\n", encoding="utf-8")
        self.core_audit = self.root / "core-shared-dso-audit.log"
        self.core_audit.write_text("", encoding="utf-8")
        self.compile_audit = self.root / "compile-audit.log"
        self.asan_symbols = self.root / "asan-symbols.log"
        self.asan_symbols.write_text("asan symbols\n", encoding="utf-8")
        self.ubsan_symbols = self.root / "ubsan-symbols.log"
        self.ubsan_symbols.write_text("ubsan symbols\n", encoding="utf-8")
        self.relocation = self.root / "relocations.log"
        self.relocation.write_text("relocations clean\n", encoding="utf-8")
        self.asan_runtime = self.root / "libclang_rt.asan.so"
        self.asan_runtime.write_bytes(b"shared asan runtime")
        self.toolchain_root = self.root / "toolchain-root"
        smoke_dir = self.toolchain_root / "smoke"
        smoke_dir.mkdir(parents=True)
        (smoke_dir / "libMLIRIR-pic.so").write_bytes(b"PIC smoke output")
        self.gpu_fixture = self.root / "gpu-fixture.mlir"
        self.gpu_fixture.write_text("gpu.func @v0_add\n", encoding="utf-8")
        self.gpu_output = self.root / "gpu-output.mlir"
        self.gpu_output.write_text("gpu.binary @v0_add\n", encoding="utf-8")
        self.ubsan_patch = self.root / "ubsan-fix.patch"
        self.ubsan_patch.write_text("diff --git a/ubsan.cpp b/ubsan.cpp\n", encoding="utf-8")
        self.ubsan_application = self.root / "ubsan-application.json"
        self.ubsan_application.write_text(json.dumps({
            "applied": True, "patch_id": "libomp-ubsan-shift-fix",
            "patch_sha256": digest(self.ubsan_patch),
        }, sort_keys=True) + "\n", encoding="utf-8")
        self.ubsan_smoke = self.root / "ubsan-smoke.log"
        self.ubsan_smoke.write_text("negative shift report reproduced and fixed\n", encoding="utf-8")
        self.patch_records = self.fixture.patch_records + [{
            "id": "libomp-ubsan-shift-fix",
            "patch_path": str(self.ubsan_patch), "patch_sha256": digest(self.ubsan_patch),
            "applied": True,
            "application_record_path": str(self.ubsan_application),
            "application_record_sha256": digest(self.ubsan_application),
            "smoke_evidence_path": str(self.ubsan_smoke),
            "smoke_evidence_sha256": digest(self.ubsan_smoke),
        }]
        self.inventory_inputs = {
            "compile_commands": str(self.fixture.compile_commands),
            "cmake_cache": str(self.fixture.cache),
            "llvm_config": str(self.fixture.llvm_config),
            "mlir_config": str(self.fixture.mlir_config),
            "cmake_exports": [str(self.fixture.llvm_exports), str(self.fixture.mlir_exports)],
            "install_manifest": str(self.fixture.manifest),
            "source_archive": {"path": str(self.fixture.source_archive),
                               "sha256": self.fixture.source_archive_sha256},
            "applied_patches": self.patch_records,
        }
        self.inventory = validate_inventory(
            self.fixture.compile_commands, self.fixture.cache,
            self.fixture.llvm_config, self.fixture.mlir_config,
            [self.fixture.llvm_exports, self.fixture.mlir_exports], self.fixture.manifest,
            source_archive_path=self.fixture.source_archive,
            source_archive_sha256=self.fixture.source_archive_sha256,
            applied_patch_records=self.patch_records,
        )
        compile_counts = self.inventory["inventory"]["compile_commands"]
        self.compile_audit.write_text(
            f"compile_commands={compile_counts['total']}; "
            f"sanitizer_instrumented_sources={compile_counts['c_cpp_count']}; "
            f"pic_sources={compile_counts['total']}\n", encoding="utf-8")
        self.identity_path = self.root / "toolchain-identity.json"
        archive_records = self.inventory["inventory"]["static_archives"]
        archive_list = self.root / "static-archive-list.txt"
        archive_list.write_text("".join(item["source_path"] + "\n" for item in archive_records),
                                encoding="utf-8")
        tool_artifacts = {name: {"path": str(path), "sha256": digest(path)}
                          for name, path in self.tools.items()}
        install_root = self.prefix
        self.identity = {
            "schema": "ncnn-san-toolchain-identity/v2", "status": "verified",
            "root": str(self.toolchain_root),
            "source_archive_identity": {"path": str(self.fixture.source_archive),
                                        "required_sha256": self.fixture.source_archive_sha256},
            "tool_artifacts": tool_artifacts,
            "llvm": {
                "version": "21.1.8", "shared_mode": "static",
                "install_prefix": str(install_root),
                "configuration": {
                    "BUILD_SHARED_LIBS": False, "LLVM_BUILD_LLVM_DYLIB": False,
                    "LLVM_BUILD_LLVM_C_DYLIB": False, "LLVM_LINK_LLVM_DYLIB": False,
                    "MLIR_BUILD_MLIR_C_DYLIB": False,
                    "sanitizers": ["address", "undefined"],
                    "no_sanitize_recover": True,
                    "flags": "-fPIC -fsanitize=address,undefined",
                },
                "compiler": {"path": str(self.tools["clang"])},
                "compiler_cxx": {"path": str(self.tools["clang++"])},
                "tools": {"mlir_opt": str(self.tools["mlir-opt"]),
                          "mlir_translate": str(self.tools["mlir-translate"]),
                          "FileCheck": str(self.tools["FileCheck"])},
                "lit_runner": {"path": str(self.tools["lit.py"])},
                "static_archive_count": len(archive_records),
                "static_archive_artifacts": [
                    {"path": item["source_path"], "sha256": item["sha256"]}
                    for item in archive_records],
                "static_archive_list": {"path": str(archive_list), "sha256": digest(archive_list)},
                "core_shared_dso_audit": str(self.core_audit),
                "compile_pic_sanitizer_audit": str(self.compile_audit),
                "shared_asan_runtime_identity": {"path": str(self.asan_runtime),
                                                  "sha256": digest(self.asan_runtime)},
                "mlir_ir_asan_symbol_audit": str(self.asan_symbols),
                "mlir_ir_ubsan_symbol_audit": str(self.ubsan_symbols),
                "mlir_ir_pic_relocation_audit": str(self.relocation),
                "inventory": {"identity_sha256": self.inventory["identity_sha256"],
                              "inputs": self.inventory_inputs},
            },
            "libomp": {
                "shared_library": {"path": str(self.libomp), "sha256": digest(self.libomp)},
                "static_library": None, "libgomp_aliases": [],
                "configuration": {"sanitizers": ["address", "undefined"],
                                   "no_sanitize_recover": True,
                                   "LIBOMP_OMPD_SUPPORT": False},
                "source_patch": {"path": str(self.fixture.patch),
                                 "sha256": digest(self.fixture.patch)},
                "ubsan_fix_patch": {"path": str(self.ubsan_patch),
                                     "sha256": digest(self.ubsan_patch)},
                "lsan_smoke": {"binary": str(self.lsan_binary), "log": str(self.lsan_log),
                               "leak_detection": True, "suppressions": False},
            },
            "smokes": {
                "verification_status": str(self.status),
                "gpu_runner_pipeline_fixture": {"path": str(self.gpu_fixture),
                                                "sha256": digest(self.gpu_fixture)},
                "gpu_runner_pipeline_output": {"path": str(self.gpu_output),
                                               "sha256": digest(self.gpu_output)},
            },
            "log_files": {
                "libomp-lsan-link-audit.log": str(self.link_log),
                "core-tool-dynamic-audit.log": str(self.dynamic_log),
                "libomp-dynamic-and-symbol-audit.log": str(self.libomp_dynamic_log),
                "libomp-smoke-ldd.log": str(self.libomp_ldd_log),
                **{f"{name}.dynamic.log": str(path)
                   for name, path in self.tool_dynamic_logs.items()},
            },
            "log_sha256": {
                "libomp-lsan-link-audit.log": digest(self.link_log),
                "core-tool-dynamic-audit.log": digest(self.dynamic_log),
                "libomp-dynamic-and-symbol-audit.log": digest(self.libomp_dynamic_log),
                "libomp-smoke-ldd.log": digest(self.libomp_ldd_log),
                **{f"{name}.dynamic.log": digest(path)
                   for name, path in self.tool_dynamic_logs.items()},
            },
            "status_files": {str(self.status): digest(self.status)},
            "configuration_files": {str(self.configuration_file): digest(self.configuration_file)},
        }
        self.identity_path.write_text(json.dumps(self.identity, sort_keys=True), encoding="utf-8")

    def tearDown(self):
        self.fixture.tearDown()

    def _run_adapter(self, stage):
        def fake_check_output(argv, **kwargs):
            if argv[0] == "readelf":
                return "Dynamic section:\n"
            if argv[0] == str(self.tools["mlir-opt"]):
                return "--test-vulkan-runner-pipeline\n"
            raise AssertionError(argv)
        with mock.patch.object(sanitizer_toolchain.subprocess, "check_output", side_effect=fake_check_output):
            return sanitizer_toolchain.audit_and_bundle(
                stage, self.identity_path, self.prefix, self.tools["lit.py"],
                self.openmp, self.clang_dir)

    def test_v2_adapter_bundles_actual_archives_inputs_and_responses(self):
        stage = self.root / "stage"
        stage.mkdir()
        result = self._run_adapter(stage)
        self.assertEqual(len(self.inventory["inventory"]["static_archives"]),
                         result["static_archive_count"])
        self.assertEqual(self.inventory["identity_sha256"], result["inventory_identity_sha256"])
        kinds = {item.get("kind") for item in result["files"]}
        self.assertTrue({"static-archive", "inventory-compile_commands",
                         "inventory-cmake_cache", "inventory-install_manifest",
                         "llvm-source-archive", "cmake-export"}.issubset(kinds))
        for item in result["files"]:
            target = stage / item["path"]
            self.assertEqual(item["source_sha256"], digest(target))

    def test_v2_final_checker_accepts_content_bound_staged_inventory(self):
        stage = self.root / "stage"
        stage.mkdir()
        evidence = self._run_adapter(stage)
        state = {"profile": "sanitize", "sanitizer_toolchain_evidence": evidence}
        findings = []
        release_audit._audit_sanitizer_toolchain(stage, state, None, [], findings)
        self.assertEqual([], findings)

    def test_v2_final_checker_rejects_missing_or_changed_identity_artifact_classes(self):
        cases = (
            ("tool binary", lambda item: item.get("kind") == "tool-binary:clang"),
            ("archive list", lambda item: item.get("kind") == "static-archive-list"),
            ("libomp", lambda item: item.get("kind") == "isolated-libomp"),
            ("ASan runtime", lambda item: item.get("kind") == "shared-asan-runtime"),
            ("dynamic audit log", lambda item: Path(item.get("source", "")).name == "mlir-opt.dynamic.log"),
        )
        for artifact_class, matches in cases:
            for tamper in (False, True):
                with self.subTest(artifact_class=artifact_class, tamper=tamper):
                    stage = self.root / f"stage-{artifact_class.replace(' ', '-')}-{tamper}"
                    stage.mkdir()
                    evidence = self._run_adapter(stage)
                    record = next(item for item in evidence["files"] if matches(item))
                    if tamper:
                        (stage / record["path"]).write_bytes(b"tampered staged identity artifact")
                    else:
                        evidence["files"].remove(record)
                    findings = []
                    release_audit._audit_sanitizer_toolchain(
                        stage, {"profile": "sanitize", "sanitizer_toolchain_evidence": evidence},
                        None, [], findings)
                    self.assertTrue(findings)

    def test_v2_final_checker_rejects_patch_inputs_missing_from_staged_bundle(self):
        stage = self.root / "stage"
        stage.mkdir()
        evidence = self._run_adapter(stage)
        patch_sources = set()
        for record in self.identity["llvm"]["inventory"]["inputs"]["applied_patches"]:
            for key in ("patch_path", "application_record_path", "smoke_evidence_path"):
                if key in record:
                    patch_sources.add(str(Path(record[key]).resolve()))
        evidence["files"] = [item for item in evidence["files"]
                             if item.get("source") not in patch_sources]
        state = {"profile": "sanitize", "sanitizer_toolchain_evidence": evidence}
        findings = []
        release_audit._audit_sanitizer_toolchain(stage, state, None, [], findings)
        codes = {item["code"] for item in findings}
        self.assertIn("sanitizer-inventory-invalid", codes)

    def test_v2_relocated_stage_passes_without_original_toolchain_root(self):
        with tempfile.TemporaryDirectory(prefix="san-v2-relocation-") as relocated_parent:
            stage = Path(relocated_parent) / "first-stage"
            stage.mkdir()
            evidence = self._run_adapter(stage)
            relocated = Path(relocated_parent) / "relocated-bundle"
            shutil.move(stage, relocated)
            shutil.rmtree(self.root)
            state = {"profile": "sanitize", "sanitizer_toolchain_evidence": evidence}
            findings = []
            release_audit._audit_sanitizer_toolchain(relocated, state, None, [], findings)
            self.assertEqual([], findings)

    def test_v2_final_checker_rejects_omitted_staged_archive_record(self):
        stage = self.root / "stage"
        stage.mkdir()
        evidence = self._run_adapter(stage)
        evidence["files"] = [item for item in evidence["files"]
                             if item.get("kind") != "static-archive"]
        state = {"profile": "sanitize", "sanitizer_toolchain_evidence": evidence}
        findings = []
        release_audit._audit_sanitizer_toolchain(stage, state, None, [], findings)
        self.assertIn("sanitizer-inventory-bundle-missing",
                      {item["code"] for item in findings})

    def test_v2_adapter_rejects_tampered_full_compile_flags(self):
        stage = self.root / "stage"
        stage.mkdir()
        entries = json.loads(self.fixture.compile_commands.read_text())
        entries[0]["arguments"].append("-fno-sanitize=undefined")
        self.fixture.compile_commands.write_text(json.dumps(entries), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "inventory validation failed"):
            self._run_adapter(stage)

    def test_v2_adapter_rejects_count_adjusted_identity_archive_claim(self):
        self.identity["llvm"]["static_archive_count"] -= 1
        self.identity_path.write_text(json.dumps(self.identity), encoding="utf-8")
        stage = self.root / "stage"
        stage.mkdir()
        with self.assertRaisesRegex(RuntimeError, "archive inventory is incomplete"):
            self._run_adapter(stage)

    def test_v2_adapter_rejects_deleted_count_adjusted_exported_archive(self):
        original = self.inventory["identity_sha256"]
        removed = self.fixture.llvm_analysis_archive
        removed.unlink()
        self.fixture._write_exports(include_analysis=False)
        self.fixture._write_manifest([
            self.fixture.llvm_archive, self.fixture.lto_archive, self.fixture.mlir_archive,
        ])
        self.identity["llvm"]["static_archive_artifacts"] = [
            item for item in self.identity["llvm"]["static_archive_artifacts"]
            if Path(item["path"]).resolve() != removed.resolve()]
        self.identity["llvm"]["static_archive_count"] -= 1
        archive_list = Path(self.identity["llvm"]["static_archive_list"]["path"])
        archive_list.write_text("".join(item["path"] + "\\n"
                                         for item in self.identity["llvm"]["static_archive_artifacts"]),
                                encoding="utf-8")
        self.identity["llvm"]["static_archive_list"]["sha256"] = digest(archive_list)
        self.identity_path.write_text(json.dumps(self.identity), encoding="utf-8")
        stage = self.root / "stage"
        stage.mkdir()
        with self.assertRaisesRegex(RuntimeError, "inventory validation failed"):
            self._run_adapter(stage)
        self.assertEqual(original, self.inventory["identity_sha256"])

    def test_v2_adapter_requires_each_patch_source_file(self):
        self.identity["llvm"]["inventory"]["inputs"]["applied_patches"][0]["patch_path"] = str(
            self.root / "missing-affinity.patch")
        self.identity_path.write_text(json.dumps(self.identity), encoding="utf-8")
        stage = self.root / "stage"
        stage.mkdir()
        with self.assertRaisesRegex(RuntimeError, "inventory validation failed"):
            self._run_adapter(stage)

    def test_v2_staged_archive_byte_tampering_is_rejected(self):
        stage = self.root / "stage"
        stage.mkdir()
        result = self._run_adapter(stage)
        archive_copy = next(stage / item["path"] for item in result["files"]
                            if item.get("kind") == "static-archive")
        archive_copy.write_bytes(b"tampered staged archive")
        staged_prefix = stage / "toolchain-evidence/llvm-install"

        def resolve(source):
            source = Path(source).resolve()
            try:
                return staged_prefix / source.relative_to(self.prefix)
            except ValueError:
                return source

        with self.assertRaisesRegex(InventoryError, "differs from source bytes"):
            validate_inventory(
                self.fixture.compile_commands, self.fixture.cache,
                staged_prefix / "lib/cmake/llvm/LLVMConfig.cmake",
                staged_prefix / "lib/cmake/mlir/MLIRConfig.cmake",
                [staged_prefix / "lib/cmake/llvm/LLVMExports.cmake",
                 staged_prefix / "lib/cmake/mlir/MLIRTargets.cmake"],
                self.fixture.manifest,
                source_archive_path=self.fixture.source_archive,
                source_archive_sha256=self.fixture.source_archive_sha256,
                resolver=resolve,
                applied_patch_records=self.patch_records,
                expected_identity_sha256=self.inventory["identity_sha256"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
