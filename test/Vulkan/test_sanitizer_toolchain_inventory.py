from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from sanitizer_toolchain_inventory import InventoryError, validate_inventory


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SanitizerToolchainInventoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="san-inventory-v2-")
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "install"
        self.build = self.root / "build"
        self.lib_dir = self.prefix / "lib"
        self.llvm_cmake = self.prefix / "lib/cmake/llvm"
        self.mlir_cmake = self.prefix / "lib/cmake/mlir"
        self.lib_dir.mkdir(parents=True)
        self.llvm_cmake.mkdir(parents=True)
        self.mlir_cmake.mkdir(parents=True)

        self.llvm_archive = self.lib_dir / "libLLVMCore.a"
        self.llvm_analysis_archive = self.lib_dir / "libLLVMAnalysis.a"
        self.lto_archive = self.lib_dir / "libLTO.a"
        self.mlir_archive = self.lib_dir / "libMLIRIR.a"
        self.mlir_runner_library = self.lib_dir / "libmlir_c_runner_utils.so.21.1"
        self.llvm_archive.write_bytes(b"!<arch>\nllvm-core-object")
        self.llvm_analysis_archive.write_bytes(b"!<arch>\nllvm-analysis-object")
        self.lto_archive.write_bytes(b"!<arch>\nlto-object")
        self.mlir_archive.write_bytes(b"!<arch>\nmlir-ir-object")
        self.mlir_runner_library.write_bytes(b"mlir runner runtime shared library")

        self.source_archive = self.root / "llvm-project-21.1.8.src.tar.xz"
        self.source_archive.write_bytes(b"official llvm source archive fixture")
        self.source_archive_sha256 = digest(self.source_archive)
        self.source_tree = self.root / "src/llvm-project-21.1.8.src/llvm"
        self.source_tree.mkdir(parents=True)

        self.llvm_config = self.llvm_cmake / "LLVMConfig.cmake"
        self.mlir_config = self.mlir_cmake / "MLIRConfig.cmake"
        self.llvm_config.write_text(
            'set(LLVM_PACKAGE_VERSION "21.1.8")\n', encoding="utf-8"
        )
        self.mlir_config.write_text(
            'set(MLIR_INSTALL_PREFIX "' + str(self.prefix) + '")\n'
            "set(LLVM_VERSION 21.1.8)\n"
            'find_package(LLVM ${LLVM_VERSION} EXACT REQUIRED CONFIG\n'
            '             HINTS "${MLIR_INSTALL_PREFIX}/lib/cmake/llvm")\n',
            encoding="utf-8",
        )

        self.llvm_exports = self.llvm_cmake / "LLVMExports.cmake"
        self.mlir_exports = self.mlir_cmake / "MLIRTargets.cmake"
        self._write_exports()

        self.cache = self.build / "CMakeCache.txt"
        self.build.mkdir()
        self._write_cache()
        self.manifest = self.prefix / "install_manifest.txt"
        self._write_manifest(
            [
                self.llvm_archive,
                self.llvm_analysis_archive,
                self.lto_archive,
                self.mlir_archive,
            ]
        )

        self.compile_commands = self.build / "compile_commands.json"
        self.compile_entries = self._valid_compile_entries()
        self._write_compile_commands()

        self.patch = self.root / "affinity-fix.patch"
        self.patch.write_text("diff --git a/kmp_affinity.cpp b/kmp_affinity.cpp\n", encoding="utf-8")
        self.application_record = self.root / "affinity-application.json"
        self._write_patch_application()
        self.patch_records = [
            {
                "id": "libomp-affinity-omission-fix",
                "patch_path": str(self.patch),
                "patch_sha256": digest(self.patch),
                "applied": True,
                "application_record_path": str(self.application_record),
                "application_record_sha256": digest(self.application_record),
            }
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def _write_cache(self, *, shared=False):
        self.cache.write_text(
            "\n".join(
                [
                    "CMAKE_INSTALL_PREFIX:PATH=" + str(self.prefix),
                    "LLVM_SOURCE_DIR:STATIC=" + str(self.source_tree),
                    "BUILD_SHARED_LIBS:BOOL=" + ("ON" if shared else "OFF"),
                    "LLVM_BUILD_LLVM_DYLIB:BOOL=OFF",
                    "LLVM_BUILD_LLVM_C_DYLIB:BOOL=OFF",
                    "LLVM_LINK_LLVM_DYLIB:BOOL=OFF",
                    "MLIR_BUILD_MLIR_C_DYLIB:BOOL=OFF",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_exports(
        self,
        *,
        include_llvm=True,
        include_analysis=True,
        include_lto=True,
        include_mlir=True,
        include_mlir_runner=True,
        core_dso=False,
    ):
        llvm_lines = []
        mlir_lines = []
        if include_llvm:
            location = "${_IMPORT_PREFIX}/lib/libLLVMCore.a"
            if core_dso:
                location = "${_IMPORT_PREFIX}/lib/libLLVMCore.so"
                (self.lib_dir / "libLLVMCore.so").write_bytes(b"forbidden core DSO")
            llvm_lines.extend(
                [
                    "set_target_properties(LLVMCore PROPERTIES",
                    f'  IMPORTED_LOCATION_NOCONFIG "{location}"',
                    ")",
                ]
            )
        if include_analysis:
            llvm_lines.extend(
                [
                    "set_target_properties(LLVMAnalysis PROPERTIES",
                    '  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libLLVMAnalysis.a"',
                    ")",
                ]
            )
        if include_lto:
            llvm_lines.extend(
                [
                    "set_target_properties(LTO PROPERTIES",
                    '  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libLTO.a"',
                    ")",
                ]
            )
        if include_mlir:
            mlir_lines.extend(
                [
                    "set_target_properties(MLIRIR PROPERTIES",
                    '  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libMLIRIR.a"',
                    ")",
                ]
            )
        if include_mlir_runner:
            mlir_lines.extend(
                [
                    "set_target_properties(mlir_c_runner_utils PROPERTIES",
                    '  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmlir_c_runner_utils.so.21.1"',
                    ")",
                ]
            )
        self.llvm_exports.write_text("\n".join(llvm_lines) + "\n", encoding="utf-8")
        self.mlir_exports.write_text("\n".join(mlir_lines) + "\n", encoding="utf-8")

    def _write_manifest(self, archives):
        self.manifest.write_text(
            "".join(str(path) + "\n" for path in archives), encoding="utf-8"
        )

    def _valid_compile_entries(self):
        source_dir = self.build / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        cpp = source_dir / "instrumented.cpp"
        c = source_dir / "instrumented.c"
        asm = source_dir / "startup.S"
        return [
            {
                "directory": str(source_dir),
                "file": str(cpp),
                "arguments": [
                    "/tool/bin/clang++",
                    "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all",
                    "-fPIC",
                    "-c",
                    str(cpp),
                ],
            },
            {
                "directory": str(source_dir),
                "file": str(c),
                "command": (
                    f"/tool/bin/clang -fPIC -fsanitize=address,undefined "
                    f"-fno-sanitize-recover=address,undefined -c {c}"
                ),
            },
            {
                "directory": str(source_dir),
                "file": str(asm),
                "arguments": ["/tool/bin/clang", "-fPIC", "-c", str(asm)],
            },
        ]

    def _write_compile_commands(self):
        self.compile_commands.write_text(
            json.dumps(self.compile_entries, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_patch_application(self):
        self.application_record.write_text(
            json.dumps(
                {
                    "applied": True,
                    "patch_id": "libomp-affinity-omission-fix",
                    "patch_sha256": digest(self.patch),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _validate(self, **overrides):
        arguments = {
            "compile_commands_path": self.compile_commands,
            "cmake_cache_path": self.cache,
            "llvm_config_path": self.llvm_config,
            "mlir_config_path": self.mlir_config,
            "export_files": [self.llvm_exports, self.mlir_exports],
            "install_manifest_path": self.manifest,
            "source_archive_path": self.source_archive,
            "source_archive_sha256": self.source_archive_sha256,
            "applied_patch_records": self.patch_records,
        }
        arguments.update(overrides)
        return validate_inventory(**arguments)

    def test_returns_hash_bound_inventory_and_complete_bundle_files(self):
        result = self._validate()
        inventory = result["inventory"]
        self.assertEqual(inventory["schema"], "ncnn-san-toolchain-inventory/v2")
        self.assertEqual(inventory["schema_version"], 2)
        self.assertEqual(inventory["toolchain"]["llvm_package_version"], "21.1.8")
        self.assertEqual(
            inventory["toolchain"]["mlir_requires_llvm_version"], "21.1.8"
        )
        self.assertTrue(inventory["toolchain"]["mlir_requires_exact_same_prefix_llvm"])
        self.assertEqual(
            inventory["toolchain"]["official_source_tree"],
            self.source_tree.resolve().as_posix(),
        )
        self.assertNotIn("mlir_version", inventory["toolchain"])
        self.assertNotIn("MLIR_PACKAGE_VERSION", self.mlir_config.read_text())
        self.assertEqual(inventory["compile_commands"]["total"], 3)
        self.assertEqual(inventory["compile_commands"]["c_cpp_count"], 2)
        self.assertEqual(inventory["compile_commands"]["asm_count"], 1)
        self.assertEqual(len(inventory["static_archives"]), 4)
        self.assertEqual(len(result["bundle_files"]), 4)
        targets = {
            (item["package"], item["target"])
            for item in inventory["exported_targets"]
        }
        self.assertIn(("llvm", "LTO"), targets)
        self.assertIn(("mlir", "mlir_c_runner_utils"), targets)
        self.assertEqual(
            inventory["inputs"]["source_archive"]["sha256"],
            self.source_archive_sha256,
        )
        self.assertEqual(
            inventory["inputs"]["compile_commands"]["sha256"],
            digest(self.compile_commands),
        )
        self.assertEqual(
            inventory["inputs"]["cmake_cache"]["sha256"], digest(self.cache)
        )
        self.assertEqual(
            inventory["inputs"]["llvm_config"]["sha256"], digest(self.llvm_config)
        )
        self.assertEqual(
            inventory["inputs"]["mlir_config"]["sha256"], digest(self.mlir_config)
        )
        self.assertEqual(
            inventory["inputs"]["install_manifest"]["sha256"], digest(self.manifest)
        )
        self.assertEqual(len(inventory["inputs"]["cmake_exports"]), 2)
        self.assertEqual(len(result["identity_sha256"]), 64)

    def test_config_cache_export_manifest_hash_drift_invalidates_identity(self):
        identity = self._validate()["identity_sha256"]
        files = [
            self.cache,
            self.llvm_config,
            self.mlir_config,
            self.llvm_exports,
            self.mlir_exports,
            self.manifest,
        ]
        for path in files:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + bytes([10]))
                with self.assertRaisesRegex(InventoryError, "identity differs"):
                    self._validate(expected_identity_sha256=identity)
                path.write_bytes(original)

    def test_full_compile_database_change_invalidates_expected_identity(self):
        identity = self._validate()["identity_sha256"]
        self.compile_entries[0]["arguments"].insert(1, "-O2")
        self._write_compile_commands()
        with self.assertRaisesRegex(InventoryError, "identity differs"):
            self._validate(expected_identity_sha256=identity)

    def test_rejects_post_hoc_sanitizer_disable(self):
        self.compile_entries[0]["arguments"].append("-fno-sanitize=undefined")
        self._write_compile_commands()
        with self.assertRaisesRegex(InventoryError, "disables sanitizers"):
            self._validate()

    def test_rejects_recovery_override(self):
        self.compile_entries[1]["command"] += " -fsanitize-recover=address"
        self._write_compile_commands()
        with self.assertRaisesRegex(InventoryError, "enables sanitizer recovery"):
            self._validate()

    def test_rejects_missing_response_file(self):
        self.compile_entries[0]["arguments"].insert(1, "@absent.rsp")
        self._write_compile_commands()
        with self.assertRaisesRegex(InventoryError, "response file is missing"):
            self._validate()

    def test_hash_binds_and_expands_present_response_files(self):
        response_file = self.build / "sources/compile.rsp"
        response_file.write_text(
            "-fsanitize=address,undefined -fno-sanitize-recover=all -fPIC -c",
            encoding="utf-8",
        )
        self.compile_entries[0]["arguments"] = [
            "/tool/bin/clang++",
            "@compile.rsp",
            str(self.build / "sources/instrumented.cpp"),
        ]
        self._write_compile_commands()
        inventory = self._validate()["inventory"]
        self.assertEqual(len(inventory["inputs"]["response_files"]), 1)
        self.assertEqual(
            inventory["inputs"]["response_files"][0]["sha256"], digest(response_file)
        )
        self.assertIn(
            str(response_file),
            {record["source"] for record in self._validate()["bundle_files"]
             if record["kind"] == "compiler-response-file"},
        )

    def test_expands_nested_response_files_relative_to_parent(self):
        source_dir = self.build / "sources"
        top_rsp = source_dir / "compile.rsp"
        nested_rsp = source_dir / "nested/flags.rsp"
        nested_rsp.parent.mkdir()
        top_rsp.write_text("@nested/flags.rsp", encoding="utf-8")
        nested_rsp.write_text(
            "-fsanitize=address,undefined -fno-sanitize-recover=all -fPIC -c",
            encoding="utf-8",
        )
        self.compile_entries[0]["arguments"] = [
            "/tool/bin/clang++",
            "@compile.rsp",
            str(source_dir / "instrumented.cpp"),
        ]
        self._write_compile_commands()
        result = self._validate()
        response_files = result["inventory"]["inputs"]["response_files"]
        self.assertEqual(
            {record["path"] for record in response_files},
            {str(top_rsp.resolve()), str(nested_rsp.resolve())},
        )
        self.assertEqual(
            len([record for record in result["bundle_files"]
                 if record["kind"] == "compiler-response-file"]),
            2,
        )

    def test_rejects_missing_nested_response_file(self):
        top_rsp = self.build / "sources/compile.rsp"
        top_rsp.write_text("@nested/missing.rsp", encoding="utf-8")
        self.compile_entries[0]["arguments"] = [
            "/tool/bin/clang++",
            "@compile.rsp",
            str(self.build / "sources/instrumented.cpp"),
        ]
        self._write_compile_commands()
        with self.assertRaisesRegex(InventoryError, "response file is missing"):
            self._validate()

    def test_resolver_maps_moved_response_file_and_bundles_stage_copy(self):
        response_file = self.build / "sources/compile.rsp"
        response_file.write_text(
            "-fsanitize=address,undefined -fno-sanitize-recover=all -fPIC -c",
            encoding="utf-8",
        )
        self.compile_entries[0]["arguments"] = [
            "/tool/bin/clang++",
            "@compile.rsp",
            str(self.build / "sources/instrumented.cpp"),
        ]
        self._write_compile_commands()
        identity = self._validate()["identity_sha256"]
        staged_rsp = self.root / "stage/response/compile.rsp"
        staged_rsp.parent.mkdir(parents=True)
        shutil.copyfile(response_file, staged_rsp)

        def resolve(source_path: Path) -> Path:
            if source_path == response_file.resolve():
                return staged_rsp
            return source_path

        staged = self._validate(
            resolver=resolve, expected_identity_sha256=identity
        )
        response_bundle = next(
            item for item in staged["bundle_files"]
            if item["kind"] == "compiler-response-file"
        )
        self.assertEqual(Path(response_bundle["path"]), staged_rsp.resolve())
        self.assertEqual(staged["identity_sha256"], identity)

    def test_export_target_missing_from_manifest_is_rejected(self):
        self._write_manifest([self.llvm_archive])
        with self.assertRaisesRegex(InventoryError, "archive sets differ"):
            self._validate()

    def test_manifest_member_missing_from_exports_is_rejected(self):
        self._write_exports(include_analysis=False)
        with self.assertRaisesRegex(InventoryError, "archive sets differ"):
            self._validate()

    def test_archive_missing_from_installed_prefix_is_rejected(self):
        self.llvm_analysis_archive.unlink()
        with self.assertRaisesRegex(InventoryError, "archive files differ"):
            self._validate()

    def test_synced_manifest_count_cannot_hide_missing_export_member(self):
        # Removing the same entry from the manifest and export leaves the
        # installed archive on disk, so the three-way set comparison rejects it.
        self._write_manifest(
            [self.llvm_archive, self.lto_archive, self.mlir_archive]
        )
        self._write_exports(include_analysis=False)
        with self.assertRaisesRegex(InventoryError, "archive files differ"):
            self._validate()

    def test_rejects_core_dso_and_shared_configuration(self):
        self._write_exports(core_dso=True)
        with self.assertRaisesRegex(InventoryError, "core shared library is forbidden"):
            self._validate()

        self._write_exports(core_dso=False)
        self._write_cache(shared=True)
        with self.assertRaisesRegex(InventoryError, "forbidden core DSO"):
            self._validate()

    def test_rejects_wrong_installed_version(self):
        self.llvm_config.write_text(
            'set(LLVM_PACKAGE_VERSION "21.1.7")\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(InventoryError, "exact same-prefix LLVM package version"):
            self._validate()

    def test_mlir_config_requires_exact_same_prefix_llvm(self):
        self.mlir_config.write_text(
            'set(LLVM_VERSION 21.1.8)\n'
            'find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG\n'
            '             HINTS "${MLIR_INSTALL_PREFIX}/lib/cmake/llvm")\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(InventoryError, "EXACT REQUIRED CONFIG"):
            self._validate()

        self.mlir_config.write_text(
            'set(LLVM_VERSION 21.1.8)\n'
            'find_package(LLVM ${LLVM_VERSION} EXACT REQUIRED CONFIG\n'
            '             HINTS "${MLIR_INSTALL_PREFIX}/other/cmake/llvm")\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(InventoryError, "same install prefix"):
            self._validate()

    def test_official_source_archive_hash_is_required(self):
        self.source_archive.write_bytes(b"different source archive bytes")
        with self.assertRaisesRegex(InventoryError, "source archive bytes do not match"):
            self._validate()

    def test_cmake_source_tree_must_match_the_official_archive_root(self):
        cache_text = self.cache.read_text(encoding="utf-8")
        self.cache.write_text(
            cache_text.replace(str(self.source_tree), str(self.root / "unrelated/llvm")),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(InventoryError, "does not descend from"):
            self._validate()

    def test_requires_real_affinity_patch_hash_and_application_record(self):
        with self.assertRaisesRegex(InventoryError, "lacks the real affinity fix"):
            self._validate(applied_patch_records=[])

        self.patch.write_text("changed patch bytes\n", encoding="utf-8")
        with self.assertRaisesRegex(InventoryError, "patch bytes do not match"):
            self._validate()

    def test_does_not_accept_unsubstantiated_ubsan_patch(self):
        ubsan_patch = self.root / "ubsan-fix.patch"
        ubsan_patch.write_text("diff --git a/file b/file\n", encoding="utf-8")
        ubsan_record = self.root / "ubsan-application.json"
        ubsan_record.write_text(
            json.dumps(
                {
                    "applied": True,
                    "patch_id": "libomp-ubsan-shift-fix",
                    "patch_sha256": digest(ubsan_patch),
                }
            ),
            encoding="utf-8",
        )
        records = self.patch_records + [
            {
                "id": "libomp-ubsan-shift-fix",
                "patch_path": str(ubsan_patch),
                "patch_sha256": digest(ubsan_patch),
                "applied": True,
                "application_record_path": str(ubsan_record),
                "application_record_sha256": digest(ubsan_record),
            }
        ]
        with self.assertRaisesRegex(InventoryError, "no actual smoke evidence"):
            self._validate(applied_patch_records=records)

    def test_resolver_maps_staged_archives_without_changing_source_identity(self):
        source_identity = self._validate()["identity_sha256"]
        stage_prefix = self.root / "stage/install"
        shutil.copytree(self.prefix, stage_prefix)

        def resolve(source_path: Path) -> Path:
            try:
                relative = source_path.relative_to(self.prefix)
            except ValueError:
                return source_path
            return stage_prefix / relative

        staged = self._validate(
            resolver=resolve,
            llvm_config_path=stage_prefix / "lib/cmake/llvm/LLVMConfig.cmake",
            mlir_config_path=stage_prefix / "lib/cmake/mlir/MLIRConfig.cmake",
            export_files=[
                stage_prefix / "lib/cmake/llvm/LLVMExports.cmake",
                stage_prefix / "lib/cmake/mlir/MLIRTargets.cmake",
            ],
            expected_identity_sha256=source_identity,
        )
        self.assertEqual(staged["identity_sha256"], source_identity)
        self.assertTrue(
            all(Path(record["path"]).is_relative_to(stage_prefix)
                for record in staged["bundle_files"])
        )

    def test_resolver_detects_changed_staged_archive_bytes(self):
        source_identity = self._validate()["identity_sha256"]
        stage_prefix = self.root / "stage/install"
        shutil.copytree(self.prefix, stage_prefix)
        (stage_prefix / "lib/libLLVMCore.a").write_bytes(b"altered staged archive")

        def resolve(source_path: Path) -> Path:
            try:
                relative = source_path.relative_to(self.prefix)
            except ValueError:
                return source_path
            return stage_prefix / relative

        with self.assertRaisesRegex(InventoryError, "differs from source bytes"):
            self._validate(
                resolver=resolve,
                llvm_config_path=stage_prefix / "lib/cmake/llvm/LLVMConfig.cmake",
                mlir_config_path=stage_prefix / "lib/cmake/mlir/MLIRConfig.cmake",
                export_files=[
                    stage_prefix / "lib/cmake/llvm/LLVMExports.cmake",
                    stage_prefix / "lib/cmake/mlir/MLIRTargets.cmake",
                ],
                expected_identity_sha256=source_identity,
            )


if __name__ == "__main__":
    unittest.main()
