#!/usr/bin/env python3
"""Synthetic fail-closed regression tests for the V6 evidence auditor."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import xml.etree.ElementTree as ET

import check_release_evidence as audit
from test_va0_evidence import no_device_record


class GateEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.source = self.root / "source"
        self.source.mkdir()
        self._write_source()
        self.make_stage()

    def tearDown(self):
        self.temp.cleanup()

    def _write_source(self):
        path = self.source / "test/Numerical/models/performance_test.cpp"
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [item.removeprefix("PerformanceModel.")
                 for item in sorted(audit.SANITIZER_PERFORMANCE_SKIPS)]
        path.write_text("\n".join(f"TEST(PerformanceModel, {name}) {{}}" for name in names),
                        encoding="utf-8")
        input_path = self.source / "test/Vulkan/relu.param"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"synthetic source fixture")

    def write(self, relative: str, data: bytes) -> Path:
        path = self.stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def make_toolchain_bundle(self):
        source_files = {}

        def external(name, content):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            source_files[str(path)] = path
            return path

        install = self.source / "sanitizer-install"
        llvm_dir = install / "lib/cmake/llvm"
        mlir_dir = install / "lib/cmake/mlir"
        llvm_dir.mkdir(parents=True, exist_ok=True)
        mlir_dir.mkdir(parents=True, exist_ok=True)
        openmp_dir = self.source / "isolated-openmp"
        openmp_dir.mkdir(exist_ok=True)
        compiler = external("san-tools/clang", b"clang")
        compiler_cxx = external("san-tools/clang++", b"clang++")
        mlir_opt = external("san-tools/mlir-opt", b"mlir-opt")
        mlir_translate = external("san-tools/mlir-translate", b"mlir-translate")
        filecheck = external("san-tools/FileCheck", b"FileCheck")
        for path in (compiler, compiler_cxx, mlir_opt, mlir_translate, filecheck):
            path.chmod(0o755)
        lit = external("san-tools/lit.py", b"lit runner")
        identity_path = self.source / "sanitizer-toolchain-identity.json"
        status = external("san-smoke/verification.status", b"verification=passed\nexit=0\n")
        libomp = external("isolated-openmp/libomp.so", b"instrumented libomp")
        omp_log = external("san-smoke/libomp-lsan.log", b"LSan smoke passed\n")
        omp_binary = external("san-smoke/libomp-lsan-smoke", b"smoke binary")
        omp_link_log = external("san-smoke/libomp-lsan-link-audit.log",
                                f"{libomp}\nlibclang_rt.asan.so\n".encode())
        core_audit = external("san-smoke/core-shared-dso-audit.log", b"")
        compile_audit = external("san-smoke/compile-pic-sanitizer-audit.log",
                                 b"compile_commands=4316; sanitizer_instrumented_sources=4312; pic_sources=4316\n")
        asan_symbols = external("san-smoke/mlir-ir-asan-symbol-audit.log", b"asan symbols\n")
        ubsan_symbols = external("san-smoke/mlir-ir-ubsan-symbol-audit.log", b"ubsan symbols\n")
        pic_probe = external("sanitizer-install/smoke/libMLIRIR-pic.so", b"pic link smoke")
        relocation = external("san-smoke/mlir-ir-pic-relocation-audit.log", b"relocations clean\n")
        asan_runtime = external("san-tools/libclang_rt.asan.so", b"shared asan runtime")
        source_archive = external("san-source/llvm-project.tar.xz", b"archive")
        archive_list = external("san-smoke/static-archive-list.txt", b"486 static archives\n")
        gpu_fixture = external("san-smoke/gpu_func_add_probe.mlir", b"gpu.func @v0_add\n")
        gpu_output = external("san-smoke/gpu-runner-output.mlir", b"gpu.binary @v0_add\n")
        patch_a = external("san-smoke/libomp-affinity.patch", b"patch affinity")
        patch_b = external("san-smoke/libomp-ubsan.patch", b"patch ubsan")
        config_file = external("san-smoke/toolchain-config.json", b"{}\n")

        digest = audit.sha256_file
        static_archives = [
            {"path": str(install / "lib" / f"libLLVM-test-{index}.a"),
             "sha256": f"{index:064x}"}
            for index in range(486)
        ]
        identity = {
            "schema": "ncnn-san-toolchain-identity/v1", "status": "verified",
            "root": str(install),
            "source_archive_identity": {"path": str(source_archive),
                                        "required_sha256": digest(source_archive)},
            "tool_artifacts": {"clang": {"path": str(compiler), "sha256": digest(compiler)},
                               "mlir-opt": {"path": str(mlir_opt), "sha256": digest(mlir_opt)}},
            "llvm": {
                "version": "21.1.8", "shared_mode": "static", "install_prefix": str(install),
                "configuration": {"BUILD_SHARED_LIBS": False, "LLVM_BUILD_LLVM_DYLIB": False,
                                  "LLVM_BUILD_LLVM_C_DYLIB": False, "LLVM_LINK_LLVM_DYLIB": False,
                                  "MLIR_BUILD_MLIR_C_DYLIB": False,
                                  "sanitizers": ["address", "undefined"],
                                  "no_sanitize_recover": True, "flags": "-fPIC -fsanitize=address,undefined"},
                "compiler": {"path": str(compiler)}, "compiler_cxx": {"path": str(compiler_cxx)},
                "tools": {"mlir_opt": str(mlir_opt), "mlir_translate": str(mlir_translate),
                          "FileCheck": str(filecheck)},
                "lit_runner": {"path": str(lit)},
                "static_archive_count": 486, "static_archive_artifacts": static_archives,
                "static_archive_list": {"path": str(archive_list), "sha256": digest(archive_list)},
                "core_shared_dso_audit": str(core_audit),
                "compile_pic_sanitizer_audit": str(compile_audit),
                "shared_asan_runtime_identity": {"path": str(asan_runtime), "sha256": digest(asan_runtime)},
                "mlir_ir_asan_symbol_audit": str(asan_symbols),
                "mlir_ir_ubsan_symbol_audit": str(ubsan_symbols),
                "mlir_ir_pic_relocation_audit": str(relocation),
            },
            "libomp": {
                "shared_library": {"path": str(libomp), "sha256": digest(libomp)},
                "static_library": None, "libgomp_aliases": [],
                "configuration": {"sanitizers": ["address", "undefined"],
                                   "no_sanitize_recover": True, "LIBOMP_OMPD_SUPPORT": False},
                "source_patch": {"path": str(patch_a), "sha256": digest(patch_a)},
                "ubsan_fix_patch": {"path": str(patch_b), "sha256": digest(patch_b)},
                "lsan_smoke": {"binary": str(omp_binary), "log": str(omp_log),
                               "leak_detection": True, "suppressions": False},
            },
            "smokes": {"verification_status": str(status),
                       "gpu_runner_pipeline_fixture": {"path": str(gpu_fixture), "sha256": digest(gpu_fixture)},
                       "gpu_runner_pipeline_output": {"path": str(gpu_output), "sha256": digest(gpu_output)}},
            "log_files": {"libomp-lsan-link-audit.log": str(omp_link_log)},
            "log_sha256": {"libomp-lsan-link-audit.log": digest(omp_link_log)},
            "status_files": {str(status): digest(status)},
            "configuration_files": {str(config_file): digest(config_file)},
        }
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        identity_hash = digest(identity_path)
        files = []

        def copy_bundle(source, relative):
            target = self.write(relative, source.read_bytes())
            files.append({"source": str(source), "source_sha256": digest(source),
                          "path": str(target.relative_to(self.stage)),
                          "sha256": digest(target)})

        copy_bundle(identity_path, "toolchain-evidence/toolchain-identity.json")
        copy_bundle(status, "toolchain-evidence/status/verification.status")
        copy_bundle(omp_link_log, "toolchain-evidence/logs/log_files/link.log")
        copy_bundle(status, "toolchain-evidence/logs/status_files/verification.status")
        copy_bundle(config_file, "toolchain-evidence/logs/configuration_files/config.json")
        copy_bundle(patch_a, "toolchain-evidence/patches/libomp-affinity.patch")
        copy_bundle(patch_b, "toolchain-evidence/patches/libomp-ubsan.patch")
        index = 0
        while len(files) < 79:
            source = external(f"san-smoke/extra-{index}.log", f"toolchain evidence {index}\n".encode())
            copy_bundle(source, f"toolchain-evidence/logs/log_files/extra-{index}.log")
            index += 1
        identity_copy = self.stage / "toolchain-evidence/toolchain-identity.json"
        return {"identity": {"path": str(identity_copy.relative_to(self.stage)),
                             "sha256": identity_hash},
                "files": files, "static_archive_count": 486,
                "llvm_version": "21.1.8", "libomp_sha256": digest(libomp),
                "verification_status": status.read_text(encoding="utf-8")}

    def make_toolchain_bundle(self):
        external = self.source / "sanitizer-fixtures"
        def put(name, data):
            path = external / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path

        install = external / "sanitizer-install"
        (install / "lib/cmake/llvm").mkdir(parents=True, exist_ok=True)
        (install / "lib/cmake/mlir").mkdir(parents=True, exist_ok=True)
        def sha(path):
            return audit.sha256_file(path)

        tool_paths = {name: put(f"tools/{name}", name.encode())
                      for name in ("clang", "clang++", "mlir-opt", "mlir-translate", "FileCheck", "lit.py")}
        omp = put("openmp/libomp.so", b"isolated libomp")
        status = put("status/verification.status", b"verification=passed\nexit=0\n")
        link_log = put("logs/libomp-lsan-link-audit.log", f"{omp}\nlibclang_rt.asan.so\n".encode())
        lsan_log = put("logs/libomp-lsan.log", b"leak check passed\n")
        lsan_binary = put("logs/libomp-lsan-smoke", b"smoke")
        core_audit = put("logs/core-dso-audit.log", b"")
        compile_audit = put("logs/compile-audit.log",
                            b"compile_commands=4316; sanitizer_instrumented_sources=4312; pic_sources=4316\n")
        asan_symbols = put("logs/asan-symbols.log", b"symbols")
        ubsan_symbols = put("logs/ubsan-symbols.log", b"symbols")
        pic_probe = put("sanitizer-install/smoke/libMLIRIR-pic.so", b"pic")
        relocation = put("logs/relocations.log", b"clean")
        asan_runtime = put("tools/libclang_rt.asan.so", b"shared asan")
        source_archive = put("archives/llvm-project.tar.xz", b"archive")
        archive_list = put("archives/static-archives.txt", b"486 archives")
        gpu_fixture = put("smoke/gpu_func_add_probe.mlir", b"gpu.func @v0_add")
        gpu_output = put("smoke/gpu-runner-output.mlir", b"gpu.binary @v0_add")
        patch_a = put("patches/affinity.patch", b"patch")
        patch_b = put("patches/ubsan.patch", b"patch")
        config_file = put("config/toolchain.json", b"{}")
        digest = lambda path: sha(path)
        archive_paths = []
        archives = []
        archive_root = install / "lib"
        archive_root.mkdir(parents=True, exist_ok=True)
        for index in range(486):
            archive_path = archive_root / f"libLLVM-static-{index}.a"
            archive_path.write_bytes(f"synthetic archive {index}\n".encode())
            archive_paths.append(str(archive_path))
            archives.append({"path": str(archive_path), "sha256": sha(archive_path)})
        archive_list.write_text("".join(path + "\n" for path in archive_paths), encoding="utf-8")
        identity = {
            "schema": "ncnn-san-toolchain-identity/v1", "status": "verified",
            "root": str(install),
            "source_archive_identity": {"path": str(source_archive), "required_sha256": digest(source_archive)},
            "tool_artifacts": {"clang": {"path": str(tool_paths["clang"]), "sha256": digest(tool_paths["clang"])},
                               "mlir-opt": {"path": str(tool_paths["mlir-opt"]), "sha256": digest(tool_paths["mlir-opt"])}},
            "llvm": {
                "version": "21.1.8", "shared_mode": "static", "install_prefix": str(install),
                "configuration": {"BUILD_SHARED_LIBS": False, "LLVM_BUILD_LLVM_DYLIB": False,
                    "LLVM_BUILD_LLVM_C_DYLIB": False, "LLVM_LINK_LLVM_DYLIB": False,
                    "MLIR_BUILD_MLIR_C_DYLIB": False, "sanitizers": ["address", "undefined"],
                    "no_sanitize_recover": True, "flags": "-fPIC -fsanitize=address,undefined"},
                "compiler": {"path": str(tool_paths["clang"])},
                "compiler_cxx": {"path": str(tool_paths["clang++"])},
                "tools": {"mlir_opt": str(tool_paths["mlir-opt"]),
                          "mlir_translate": str(tool_paths["mlir-translate"]),
                          "FileCheck": str(tool_paths["FileCheck"])},
                "lit_runner": {"path": str(tool_paths["lit.py"])},
                "static_archive_count": 486, "static_archive_artifacts": archives,
                "static_archive_list": {"path": str(archive_list), "sha256": digest(archive_list)},
                "core_shared_dso_audit": str(core_audit),
                "compile_pic_sanitizer_audit": str(compile_audit),
                "shared_asan_runtime_identity": {"path": str(asan_runtime), "sha256": digest(asan_runtime)},
                "mlir_ir_asan_symbol_audit": str(asan_symbols),
                "mlir_ir_ubsan_symbol_audit": str(ubsan_symbols),
                "mlir_ir_pic_relocation_audit": str(relocation),
            },
            "libomp": {"shared_library": {"path": str(omp), "sha256": digest(omp)},
                "static_library": None, "libgomp_aliases": [],
                "configuration": {"sanitizers": ["address", "undefined"],
                    "no_sanitize_recover": True, "LIBOMP_OMPD_SUPPORT": False},
                "source_patch": {"path": str(patch_a), "sha256": digest(patch_a)},
                "ubsan_fix_patch": {"path": str(patch_b), "sha256": digest(patch_b)},
                "lsan_smoke": {"binary": str(lsan_binary), "log": str(lsan_log),
                    "leak_detection": True, "suppressions": False}},
            "smokes": {"verification_status": str(status),
                "gpu_runner_pipeline_fixture": {"path": str(gpu_fixture), "sha256": digest(gpu_fixture)},
                "gpu_runner_pipeline_output": {"path": str(gpu_output), "sha256": digest(gpu_output)}},
            "log_files": {"libomp-lsan-link-audit.log": str(link_log)},
            "log_sha256": {"libomp-lsan-link-audit.log": digest(link_log)},
            "status_files": {str(status): digest(status)},
            "configuration_files": {str(config_file): digest(config_file)},
        }
        identity_path = put("toolchain-identity.json", json.dumps(identity).encode())
        copied = []
        def copy_file(source, target_rel):
            target = self.write(target_rel, source.read_bytes())
            copied.append({"source": str(source), "source_sha256": sha(source),
                           "path": str(target.relative_to(self.stage)), "sha256": sha(target)})
        copy_file(identity_path, "toolchain-evidence/toolchain-identity.json")
        copy_file(status, "toolchain-evidence/status/verification.status")
        copy_file(link_log, "toolchain-evidence/logs/log_files/link.log")
        copy_file(status, "toolchain-evidence/logs/status_files/verification.status")
        copy_file(config_file, "toolchain-evidence/logs/configuration_files/config.json")
        copy_file(patch_a, "toolchain-evidence/patches/affinity.patch")
        copy_file(patch_b, "toolchain-evidence/patches/ubsan.patch")
        index = 0
        while len(copied) < 79:
            extra = put(f"logs/extra-{index}.log", f"extra log {index}".encode())
            copy_file(extra, f"toolchain-evidence/logs/log_files/extra-{index}.log")
            index += 1
        identity_copy = self.stage / "toolchain-evidence/toolchain-identity.json"
        return {"identity": {"path": str(identity_copy.relative_to(self.stage)),
                             "sha256": sha(identity_copy)},
                "files": copied, "static_archive_count": 486,
                "llvm_version": "21.1.8", "libomp_sha256": digest(omp),
                "verification_status": status.read_text(encoding="utf-8")}

    def make_stage(self, *, profile="release", skipped=None):
        skipped = set(audit.RELEASE_PERFORMANCE_SKIPS if profile != "sanitize" else
                      audit.SANITIZER_PERFORMANCE_SKIPS) if skipped is None else set(skipped)
        source_dir = str(self.source / "compiler")
        cache = (
            "CMAKE_BUILD_TYPE:STRING=" + ("Debug" if profile == "sanitize" else "Release") + "\n"
            "NCNN_ENABLE_VULKAN_PRODUCT:BOOL=ON\n"
            "NCNN_ENABLE_VULKAN_V0_TESTS:BOOL=" + ("OFF" if profile == "offline" else "ON") + "\n"
            "NCNN_VULKAN_TEST_DEVICES:STRING=" + {
                "release": "nvidia;llvmpipe", "sanitize": "nvidia;llvmpipe",
                "software": "llvmpipe", "offline": "nvidia;llvmpipe",
                "hardware": "nvidia",
            }[profile] + "\nBUILD_TESTING:BOOL=ON\nCOMPILER_INSTALL_RPATH:BOOL=ON\n"
            "COMPILER_SANITIZERS:STRING=" + ("address;undefined" if profile == "sanitize" else "") + "\n"
            "CMAKE_C_FLAGS_DEBUG:STRING=-O3 -g\nCMAKE_CXX_FLAGS_DEBUG:STRING=-O3 -g\n"
        ).encode()
        toolchain_bundle = self.make_toolchain_bundle() if profile == "sanitize" else None
        if profile == "sanitize":
            cache += (
                f"LLVM_DIR:PATH={self.source / 'sanitizer-fixtures/sanitizer-install/lib/cmake/llvm'}\n"
                f"MLIR_DIR:PATH={self.source / 'sanitizer-fixtures/sanitizer-install/lib/cmake/mlir'}\n"
                f"CMAKE_C_COMPILER:FILEPATH={self.source / 'sanitizer-fixtures/tools/clang'}\n"
                f"CMAKE_CXX_COMPILER:FILEPATH={self.source / 'sanitizer-fixtures/tools/clang++'}\n"
                f"MLIR_OPT:FILEPATH={self.source / 'sanitizer-fixtures/tools/mlir-opt'}\n"
                f"MLIR_TRANSLATE:FILEPATH={self.source / 'sanitizer-fixtures/tools/mlir-translate'}\n"
                f"FILECHECK:FILEPATH={self.source / 'sanitizer-fixtures/tools/FileCheck'}\n"
                f"NCNN_SANITIZER_OPENMP_LIBRARY_DIR:PATH={self.source / 'sanitizer-fixtures/openmp'}\n"
            ).encode()
        cache_path = self.write("CMakeCache.txt", cache)
        test_command = ["python3", str(self.source / "test.py"), str(self.stage)]
        names = sorted(audit._required_tests(profile) | set(skipped) | {"CPU.Positive"})
        inventory_tests = []
        for name in names:
            labels = ["unit"] if name in {"vulkan-release-evidence-validation",
                                         "vulkan-gate-validation",
                                         "vulkan-v5-comparison-validation"} else ["integration"]
            command = test_command if name != "CPU.Positive" else ["true"]
            inventory_tests.append({"name": name, "command": command,
                                    "properties": [{"name": "LABELS", "value": labels}]})
        inventory = {"kind": "ctestInfo", "version": {"major": 3, "minor": 25},
                     "tests": inventory_tests}
        inventory_path = self.write("test/gate-tests.json", json.dumps(inventory).encode())

        junit = ET.Element("testsuites")
        suite = ET.SubElement(junit, "testsuite", name="CTest", tests=str(len(names)),
                              failures="0", errors="0", skipped=str(len(skipped)))
        for name in names:
            case = ET.SubElement(suite, "testcase", name=name, classname="CTest")
            if name in skipped:
                ET.SubElement(case, "skipped", message="existing performance skip")
        junit_path = self.write("gate-ctest.xml", ET.tostring(junit, encoding="utf-8"))
        last_lines = []
        for index, name in enumerate(names, 1):
            status = "Skipped" if name in skipped else "Passed"
            last_lines.extend([f"{index}/{len(names)} Testing: {name}", "Output:",
                               f"Test {status}.", ""])
        lasttest_path = self.write("gate-ctest-lasttest.log", "\n".join(last_lines).encode())

        step_argv = {
            "configure": ["cmake", "-S", source_dir, "-B", str(self.stage),
                          "-DBUILD_TESTING=ON", "-DCOMPILER_INSTALL_RPATH=ON"],
            "format_check": ["cmake", "--build", str(self.stage), "--target", "format_check"],
            "build": ["cmake", "--build", str(self.stage)],
            "numerical": ["cmake", "--build", str(self.stage), "--target",
                          "numerical_tests", "numerical_dynamic_operator_tests",
                          "numerical_dynamic_tests"],
            "tidy": ["cmake", "--build", str(self.stage), "--target", "tidy"],
            "ctest": ["ctest", "--test-dir", str(self.stage), "--output-on-failure",
                      "--output-junit", str(self.stage / "gate-ctest.xml")],
        }
        steps = []
        for index, name in enumerate(audit.STEP_NAMES):
            log = self.write(f"gate-{name}.log", f"{name} passed\n".encode())
            steps.append({"name": name, "argv": step_argv[name], "started_at": f"2026-10-08T00:00:0{index}Z",
                          "finished_at": f"2026-10-08T00:00:0{index}Z", "exit_code": 0,
                          "log": str(log.relative_to(self.stage)), "sha256": audit.sha256_file(log)})
        tool = self.write("tool.bin", b"synthetic tool")
        snapshot_input = self.write("gate-source-inputs/relu.param", b"synthetic source fixture")
        built_tool_specs = (("tools/ncnn-compile", True),
                            ("tools/ncnn-mlir-driver", True),
                            ("bin/ncnn-mlir-opt", True),
                            ("tools/ncnn-vulkan-spirv-extract", False))
        built_tool_records = []
        for relative, versioned in built_tool_specs:
            binary = self.write(relative, b"synthetic instrumented executable" if profile == "sanitize" else b"synthetic executable")
            record = {"path": relative, "sha256": audit.sha256_file(binary)}
            if versioned:
                record["version"] = "synthetic --version"
            built_tool_records.append(record)
        status_path = self.write("gate-source-status.txt", b"")
        diff_path = self.write("gate-source-diff.patch", b"")
        state = {
            "stage_id": "VA0", "host_execution_role": "legacy-product", "host_aot_candidate": False,
            "worktree": {"dirty": False,
                         "status": {"path": status_path.name, "sha256": audit.sha256_file(status_path)},
                         "tracked_diff": {"path": diff_path.name, "sha256": audit.sha256_file(diff_path)}},
            "schema_version": 1, "kind": audit.GATE_KIND, "profile": profile,
            "source_dir": source_dir, "source_fingerprint_algorithm": "sha256-git-files-v1",
            "source_fingerprint": "a" * 64, "git_revision": "deadbeef",
            "build_dir": str(self.stage), "created_at": "2026-10-08T00:00:00Z",
            "finished_at": "2026-10-08T00:00:06Z", "configuration": step_argv["configure"],
            "steps": steps, "completed": True, "sanitizer_diagnostics": [],
            "sanitizer_budget_evidence": [],
            "source_inputs": [{"path": "gate-source-inputs/relu.param",
                               "source": "test/Vulkan/relu.param",
                               "sha256": audit.sha256_file(snapshot_input)}],
            "built_tool_identities": built_tool_records,
            "cmake_cache": {"path": "CMakeCache.txt", "sha256": audit.sha256_file(cache_path)},
            "toolchain_identities": [{"role": "python", "path": str(tool),
                                       "sha256": audit.sha256_file(tool), "version": "synthetic"}],
            "inventory": {"path": str(inventory_path.relative_to(self.stage)),
                           "sha256": audit.sha256_file(inventory_path)},
            "junit": {"path": str(junit_path.relative_to(self.stage)),
                      "sha256": audit.sha256_file(junit_path)},
            "lasttest": {"path": str(lasttest_path.relative_to(self.stage)),
                         "sha256": audit.sha256_file(lasttest_path)},
        }
        if profile == "sanitize":
            state["sanitizer_toolchain_evidence"] = toolchain_bundle
            state["toolchain_identities"].extend([
                {"role": role, "path": str(tool), "sha256": audit.sha256_file(tool)}
                for role in ("lit", "libomp", "sanitizer-toolchain")])
            state["static_linkage_audit"] = []
            for built in built_tool_records[:3]:
                executable = self.stage / built["path"]
                state["static_linkage_audit"].append({
                    "executable": {"path": str(executable.relative_to(self.stage)),
                                   "sha256": audit.sha256_file(executable)},
                    "needed": ["libclang_rt.asan-x86_64.so", "libc.so.6"],
                })
        (self.stage / "gate-state.json").write_text(json.dumps(state), encoding="utf-8")
        self.state = state

    def save_state(self):
        (self.stage / "gate-state.json").write_text(json.dumps(self.state), encoding="utf-8")

    def refresh_test_hashes(self):
        for field in ("inventory", "junit", "lasttest"):
            path = self.stage / self.state[field]["path"]
            self.state[field]["sha256"] = audit.sha256_file(path)
        self.save_state()

    def findings(self, **kwargs):
        return audit.audit_stage_gate(self.stage, source_root=self.source,
                                      verify_fingerprint=False, **kwargs)["findings"]

    def _ctest_findings_with_lasttest(self, omitted=()):
        inventory_path = self.stage / self.state["inventory"]["path"]
        junit_path = self.stage / self.state["junit"]["path"]
        lasttest_path = self.stage / self.state["lasttest"]["path"]
        existing = audit._lasttest_results(lasttest_path)
        selected = [(name, status) for name, status in existing.items()
                    if name not in set(omitted)]
        lasttest_path.write_text("".join(
            f"{index}/{len(selected)} Testing: {name}\nOutput:\n"
            f"Test {status.title()}.\n\n"
            for index, (name, status) in enumerate(selected, 1)))
        findings = []
        audit._validate_test_execution(self.stage, "release", self.state,
                                       inventory_path, junit_path, lasttest_path,
                                       self.source, findings)
        return findings

    def test_lasttest_can_omit_only_junit_confirmed_approved_skips(self):
        all_findings = self._ctest_findings_with_lasttest()
        self.assertNotIn("ctest-inventory-lasttest-mismatch",
                         {item["code"] for item in all_findings})
        skipped_findings = self._ctest_findings_with_lasttest(
            audit.RELEASE_PERFORMANCE_SKIPS)
        self.assertNotIn("ctest-inventory-lasttest-mismatch",
                         {item["code"] for item in skipped_findings})

    def test_lasttest_cannot_omit_a_non_skipped_test(self):
        inventory = audit._inventory_tests(
            self.stage / self.state["inventory"]["path"])
        omitted = next(name for name in inventory
                       if name not in audit.RELEASE_PERFORMANCE_SKIPS)
        findings = self._ctest_findings_with_lasttest((omitted,))
        self.assertIn("ctest-inventory-lasttest-mismatch",
                      {item["code"] for item in findings})

    def test_complete_gate_envelope_passes(self):
        self.assertEqual([], self.findings())

    def test_va1_gate_inventory_requires_new_python_and_cpp_tests(self):
        required = audit._required_tests("release", "VA1")
        self.assertTrue(audit.VA1_REQUIRED_TESTS.issubset(required))
        self.assertFalse(audit.VA1_REQUIRED_TESTS & audit._required_tests("release", "VA0"))
        gate = {"profile": "release", "stage_id": "VA1",
                "ctest": {"test_names": sorted(required - {
                    "VulkanHostProgramTest.ChainMatchesLegacyTrace"})}}
        findings = []
        audit._check_required_tests(gate, findings)
        missing = next(item for item in findings
                       if item["code"] == "required-ctest-missing")
        self.assertIn("VulkanHostProgramTest.ChainMatchesLegacyTrace",
                      missing["details"]["tests"])

    def test_va1_does_not_require_va7_profiles_or_claim_host_aot(self):
        stage_rows = []
        for profile in ("release", "sanitize"):
            stage_rows.append({
                "stage": f"synthetic-{profile}", "profile": profile,
                "stage_id": "VA1", "source_fingerprint": "a" * 64,
                "scope_passed": True, "findings": [], "va0_findings": [],
                "ctest": {"test_names": sorted(audit.VA1_REQUIRED_TESTS)},
                "families": {"va1": {"status": "verified",
                                     "host_program_evidence": [{
                                         "path": "test/Native/host-program-evidence.json",
                                         "sha256": "b" * 64,
                                         "role": "native host-program trace"}]}},
                "raw_artifacts": [{"path": "test/Native/host-program-evidence.json",
                                   "sha256": "b" * 64,
                                   "role": "native host-program trace"}],
            })
        with patch.object(audit, "audit_stage_gate", side_effect=stage_rows), \
                patch.object(audit, "audit_stage_evidence", side_effect=lambda gate: gate), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "verified", "findings": []}):
            report = audit.build_report([Path("release"), Path("sanitize")],
                                        require_va1_planner=True)
        self.assertTrue(report["va1_planner_eligible"])
        self.assertFalse(report["release_eligible"])
        self.assertFalse(report["host_aot_release_eligible"])
        self.assertEqual({"release", "sanitize"},
                         {item["profile"] for item in report["va1_planner_stages"]})
        self.assertTrue(all(report["id_index"][f"VA1.{index}"]["status"] == "passed"
                            for index in range(1, 5)))

    def test_va1_gate_rejects_candidate_role_claim(self):
        self.state["stage_id"] = "VA1"
        self.state["host_aot_candidate"] = True
        self.save_state()
        codes = {item["code"] for item in self.findings()}
        self.assertIn("gate-execution-role", codes)

    def test_va1_requires_native_host_program_raw_trace(self):
        findings = []
        result = audit._audit_native_host_program(
            self.stage, self.stage, None, [], findings, required=True)
        self.assertIsNone(result)
        self.assertIn("host-program-evidence-missing",
                      {item["code"] for item in findings})

    def test_missing_state_fails_closed(self):
        (self.stage / "gate-state.json").unlink()
        self.assertIn("gate-state-missing", {f["code"] for f in
                                             audit.audit_stage_gate(self.stage)["findings"]})

    def test_inventory_test_not_executed_is_rejected(self):
        inventory = json.loads((self.stage / self.state["inventory"]["path"]).read_text())
        missing = "vulkan-v0-add-nvidia"
        inventory["tests"] = [item for item in inventory["tests"] if item["name"] != missing]
        (self.stage / self.state["inventory"]["path"]).write_text(json.dumps(inventory))
        self.refresh_test_hashes()
        codes = {item["code"] for item in self.findings()}
        self.assertIn("ctest-inventory-junit-mismatch", codes)
        self.assertIn("required-ctest-missing", codes)

    def test_junit_failure_is_not_counted_as_pass(self):
        junit_path = self.stage / self.state["junit"]["path"]
        tree = ET.parse(junit_path)
        case = next(case for case in tree.getroot().iter("testcase")
                    if case.get("name") == "vulkan-v0-add-nvidia")
        ET.SubElement(case, "failure", message="synthetic failure")
        tree.write(junit_path, encoding="utf-8", xml_declaration=True)
        lasttest = self.stage / self.state["lasttest"]["path"]
        lasttest.write_text(lasttest.read_text().replace("Test Passed.", "Test Failed.", 1))
        self.refresh_test_hashes()
        self.assertIn("ctest-failed", {item["code"] for item in self.findings()})

    def test_release_existing_skip_cannot_disappear(self):
        junit_path = self.stage / self.state["junit"]["path"]
        tree = ET.parse(junit_path)
        suite = tree.getroot().find("testsuite")
        case = next(case for case in suite.findall("testcase")
                    if case.get("name") == "PerformanceModel.PPLCNetDocOriInt8")
        case.remove(case.find("skipped"))
        tree.write(junit_path, encoding="utf-8", xml_declaration=True)
        lasttest = self.stage / self.state["lasttest"]["path"]
        lasttest.write_text(lasttest.read_text().replace(
            "Testing: PerformanceModel.PPLCNetDocOriInt8\nOutput:\nTest Skipped.",
            "Testing: PerformanceModel.PPLCNetDocOriInt8\nOutput:\nTest Passed."))
        self.refresh_test_hashes()
        self.assertIn("expected-ctest-skip-missing",
                      {item["code"] for item in self.findings()})

    def test_source_input_bytes_must_match_record(self):
        source_path = self.source / "test/Vulkan/relu.param"
        source_path.write_bytes(b"changed source fixture")
        codes = {item["code"] for item in self.findings()}
        self.assertIn("source-input-mismatch", codes)

    def make_budget_exception(self, *, tamper=False):
        self.make_stage(profile="sanitize")
        fixture = "pp_ocrv5_server_rec"
        requested = 450
        elapsed = 337
        directory = self.stage / "gate-budget-evidence" / fixture
        directory.mkdir(parents=True)
        log_path = directory / "timed_compile.cmake"
        log_text = f"fixture {fixture} 编译耗时 {elapsed}s 超过硬预算 300s\n"
        log_path.write_text(log_text, encoding="utf-8")
        record = {"schema_version": 1, "kind": "ncnn.sanitizer_budget_failure",
                  "source_fingerprint": self.state["source_fingerprint"],
                  "profile": "sanitize", "fixture": fixture,
                  "previous_budget_seconds": 300, "requested_budget_seconds": requested,
                  "elapsed_seconds": elapsed, "log_file": log_path.name,
                  "log_sha256": audit.sha256_file(log_path)}
        record_path = directory / "budget-evidence.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        self.state["configuration"].append(
            f"-Dncnn_model_compile_budget_{fixture}={requested}")
        cache_path = self.stage / self.state["cmake_cache"]["path"]
        cache_path.write_text(cache_path.read_text() +
                              f"ncnn_model_compile_budget_{fixture}:STRING={requested}\n")
        self.state["cmake_cache"]["sha256"] = audit.sha256_file(cache_path)
        self.state["sanitizer_budget_evidence"] = [{
            "fixture": fixture,
            "record": {"path": str(record_path.relative_to(self.stage)),
                       "sha256": audit.sha256_file(record_path),
                       "elapsed_seconds": elapsed},
            "log": {"path": str(log_path.relative_to(self.stage)),
                    "sha256": audit.sha256_file(log_path)},
            "requested_budget_seconds": requested,
            "elapsed_seconds": elapsed,
        }]
        if tamper:
            log_path.write_text("unrelated failure\n", encoding="utf-8")
        findings = []
        audit._audit_sanitizer_budget_evidence(self.stage, self.state, cache_path, findings)
        return findings

    def test_authorized_sanitizer_budget_requires_raw_failure_proof(self):
        self.assertEqual([], self.make_budget_exception())

    def test_tampered_sanitizer_budget_log_is_rejected(self):
        codes = {item["code"] for item in self.make_budget_exception(tamper=True)}
        self.assertIn("budget-evidence-log-mismatch", codes)
        self.assertIn("budget-evidence-failure-cause", codes)

    def test_cmake_cache_tampering_is_rejected(self):
        (self.stage / "CMakeCache.txt").write_text("BUILD_TESTING:BOOL=OFF\n", encoding="utf-8")
        self.assertIn("tampered-artifact", {item["code"] for item in self.findings()})

    def test_tampered_log_is_rejected(self):
        (self.stage / self.state["steps"][1]["log"]).write_text("tampered", encoding="utf-8")
        self.assertIn("gate-step-log-hash", {f["code"] for f in self.findings()})

    def test_stale_build_directory_is_rejected(self):
        self.state["build_dir"] = str(self.root / "old-stage")
        (self.stage / "gate-state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.assertIn("stale-stage", {f["code"] for f in self.findings()})

    def test_bad_profile_is_rejected(self):
        self.state["profile"] = "hardware-typo"
        (self.stage / "gate-state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.assertIn("gate-profile", {f["code"] for f in self.findings()})

    def test_filtered_ctest_is_rejected(self):
        step = self.state["steps"][-1]
        step["argv"].extend(["-R", "vulkan-v5"])
        (self.stage / "gate-state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.assertIn("ctest-filtered", {f["code"] for f in self.findings()})

    def test_source_fingerprint_change_is_detected(self):
        repo = self.root / "git-source"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "source.cpp").write_text("old\n", encoding="utf-8")
        helper = audit._load_fingerprint_function()
        original = helper(repo)
        self.assertNotEqual(original, "0" * 64)
        (repo / "source.cpp").write_text("new\n", encoding="utf-8")
        self.assertNotEqual(original, helper(repo))

    def test_gate_rejects_stale_source_fingerprint(self):
        repo = self.root / "fresh-compiler-source"
        (repo / "test/Numerical/models").mkdir(parents=True)
        (repo / "test/Numerical/models/performance_test.cpp").write_text(
            (self.source / "test/Numerical/models/performance_test.cpp").read_text(), encoding="utf-8")
        (repo / "test/Vulkan").mkdir(parents=True)
        (repo / "test/Vulkan/relu.param").write_bytes(b"synthetic source fixture")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        helper = audit._load_fingerprint_function()
        self.state["source_dir"] = str(repo)
        self.state["source_fingerprint"] = helper(repo)
        configure = self.state["configuration"]
        configure[configure.index("-S") + 1] = str(repo)
        step_argv = self.state["steps"][0]["argv"]
        step_argv[step_argv.index("-S") + 1] = str(repo)
        self.save_state()
        valid = audit.audit_stage_gate(self.stage, source_root=repo)["findings"]
        self.assertNotIn("stale-source", {item["code"] for item in valid})
        (repo / "test/Vulkan/relu.param").write_bytes(b"changed source fixture")
        stale = audit.audit_stage_gate(self.stage, source_root=repo)["findings"]
        self.assertIn("stale-source", {item["code"] for item in stale})

    def test_new_skip_is_not_whitelisted(self):
        extra = "NewTest.Unexpected"
        self.state["inventory"]["sha256"] = "0" * 64
        inventory_path = self.stage / "test/gate-tests.json"
        data = json.loads(inventory_path.read_text())
        data["tests"].append({"name": extra, "command": ["true"], "properties": []})
        inventory_path.write_text(json.dumps(data), encoding="utf-8")
        self.state["inventory"]["sha256"] = audit.sha256_file(inventory_path)
        # An extra JUnit skip is intentionally represented, not hidden by counts.
        junit_path = self.stage / self.state["junit"]["path"]
        tree = ET.parse(junit_path)
        suite = tree.getroot().find("testsuite")
        case = ET.SubElement(suite, "testcase", name=extra, classname="CTest")
        ET.SubElement(case, "skipped", message="new skip")
        tree.write(junit_path, encoding="utf-8", xml_declaration=True)
        self.state["junit"]["sha256"] = audit.sha256_file(junit_path)
        last_path = self.stage / self.state["lasttest"]["path"]
        last_path.write_text(last_path.read_text() + f"\nnew Testing: {extra}\nTest Skipped.\n")
        self.state["lasttest"]["sha256"] = audit.sha256_file(last_path)
        (self.stage / "gate-state.json").write_text(json.dumps(self.state), encoding="utf-8")
        codes = {f["code"] for f in self.findings()}
        self.assertIn("unexpected-ctest-skip", codes)

    def test_path_traversal_artifact_is_rejected(self):
        self.assertIsNone(audit.safe_stage_file(self.stage, "../outside"))

    def test_sanitize_toolchain_evidence_is_required(self):
        self.make_stage(profile="sanitize")
        self.state.pop("sanitizer_toolchain_evidence")
        self.save_state()
        self.assertIn("sanitizer-toolchain-missing",
                      {item["code"] for item in self.findings()})

    def test_sanitize_toolchain_identity_copy_tamper_is_rejected(self):
        self.make_stage(profile="sanitize")
        relative = self.state["sanitizer_toolchain_evidence"]["identity"]["path"]
        (self.stage / relative).write_text("tampered identity", encoding="utf-8")
        codes = {item["code"] for item in self.findings()}
        self.assertIn("tampered-artifact", codes)

    def test_sanitize_skip_allowance_is_exact_and_source_derived(self):
        self.make_stage(profile="sanitize")
        self.assertEqual([], self.findings())
        self._write_source()
        source_file = self.source / "test/Numerical/models/performance_test.cpp"
        source_file.write_text(source_file.read_text() + "\nTEST(PerformanceModel, NewlyAdded) {}\n")
        codes = {f["code"] for f in self.findings()}
        self.assertIn("skip-whitelist-source-drift", codes)

    def test_current_fingerprint_is_recomputed_not_trusted(self):
        repo = self.root / "fresh-compiler-tree"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "test/Numerical/models").mkdir(parents=True)
        (repo / "test/Numerical/models/performance_test.cpp").write_text(
            (self.source / "test/Numerical/models/performance_test.cpp").read_text())
        (repo / "test/Vulkan").mkdir(parents=True)
        (repo / "test/Vulkan/relu.param").write_bytes(b"synthetic source fixture")
        fingerprint = audit._load_fingerprint_function()(repo)
        self.state["source_dir"] = str(repo)
        self.state["source_fingerprint"] = fingerprint
        self.state["configuration"][self.state["configuration"].index("-S") + 1] = str(repo)
        self.state["steps"][0]["argv"][self.state["steps"][0]["argv"].index("-S") + 1] = str(repo)
        self.save_state()
        self.assertNotIn("stale-source", {item["code"] for item in
                                           audit.audit_stage_gate(self.stage, source_root=repo)["findings"]})
        (repo / "test/Vulkan/relu.param").write_bytes(b"changed")
        codes = {item["code"] for item in audit.audit_stage_gate(
            self.stage, source_root=repo)["findings"]}
        self.assertIn("stale-source", codes)

    def test_source_fingerprint_set_mismatch_blocks_composition(self):
        reports = [
            {"profile": "release", "source_fingerprint": "a" * 64, "completed": True,
             "scope_passed": True, "findings": [], "ctest": {"test_names": []}, "families": {}},
            {"profile": "sanitize", "source_fingerprint": "b" * 64, "completed": True,
             "scope_passed": True, "findings": [], "ctest": {"test_names": []}, "families": {}},
        ]
        fingerprint_set = {entry["source_fingerprint"] for entry in reports}
        self.assertEqual(2, len(fingerprint_set))

    def test_numeric_triplet_detects_cpu_gpu_golden_error(self):
        inputs = self.write("numeric/input.bin", struct.pack("<4f", 1, 2, 3, 4))
        output_bytes = struct.pack("<6f", 4, 6, 4, 6, 5, 6)
        outputs = self.write("numeric/output.bin", output_bytes)
        record = {"tolerance": {"atol": 1e-6, "rtol": 1e-6},
                  "cpu_vs_golden": {"passed": True, "max_abs": 0.0, "max_rel": 0.0},
                  "gpu_vs_golden": {"passed": True, "max_abs": 0.0, "max_rel": 0.0},
                  "gpu_vs_cpu": {"passed": True, "max_abs": 0.0, "max_rel": 0.0}}
        findings = []
        audit._verify_triplet(inputs, outputs, record, 2, "synthetic mismatch", findings)
        self.assertIn("numeric-budget-exceeded", {item["code"] for item in findings})

    def test_add_golden_mismatch_is_recomputed_from_input_bytes(self):
        input_values = [1.0, 2.0, 3.0, 5.0]
        golden = [4.0, 6.0]
        recomputed = audit._recompute_v2_golden("add", input_values, 2, {})
        self.assertNotEqual(golden, recomputed)
        self.assertFalse(audit.compare_f32(golden, recomputed, 1e-6, 1e-6)["passed"])

    def test_relative_error_uses_reference_as_denominator(self):
        metric = audit.compare_f32([1.1], [1.0], 1.0, 1.0)
        self.assertAlmostEqual(0.1, metric["max_rel"], places=7)

    def test_cpp_float_metric_roundtrip_uses_six_significant_digits(self):
        golden = [0.010894477367401123, 0.8]
        gpu = [0.010894492268562317, 0.8000004768371582]
        cpu = list(golden)
        pairs = {"cpu_vs_golden": (cpu, golden),
                 "gpu_vs_golden": (gpu, golden),
                 "gpu_vs_cpu": (gpu, cpu)}
        record = {"tolerance": {"atol": 1.0e-6, "rtol": 1.0e-6}}
        for key, (actual, expected) in pairs.items():
            metric = audit.compare_cpp_f32(actual, expected, 1.0e-6, 1.0e-6)
            record[key] = {"passed": metric["passed"],
                           "max_abs": audit._cpp_stream_float(metric["max_abs"]),
                           "max_rel": audit._cpp_stream_float(metric["max_rel"])}
        self.assertEqual(1.36777e-6, record["gpu_vs_golden"]["max_rel"])
        self.assertAlmostEqual(1.3677720289706485e-6,
                               audit.compare_cpp_f32(gpu, golden, 1.0e-6, 1.0e-6)["max_rel"],
                               places=18)
        input_path = self.write("numeric/cpp-metric-input.bin", struct.pack("<f", 1.0))
        output_path = self.write("numeric/cpp-metric-output.bin",
                                 struct.pack("<6f", *(cpu + gpu + golden)))
        findings = []
        audit._verify_triplet(input_path, output_path, record, 2,
                              "reported V2 FP32 serialization", findings)
        self.assertNotIn("numeric-claim-tampered", {item["code"] for item in findings})
        self.assertNotIn("numeric-budget-exceeded", {item["code"] for item in findings})

        record["gpu_vs_golden"]["max_rel"] = 1.0e-6
        findings = []
        audit._verify_triplet(input_path, output_path, record, 2,
                              "tampered V2 FP32 serialization", findings)
        self.assertIn("numeric-claim-tampered", {item["code"] for item in findings})

    def test_fake_nvidia_name_with_cpu_device_type_is_rejected(self):
        summary = "\n".join((
            "Devices:", "========", "GPU0:",
            "apiVersion         = 1.4.351",
            "driverVersion      = 615.71.9.0",
            "vendorID           = 0x10de",
            "deviceID           = 0x2803",
            "deviceType         = PHYSICAL_DEVICE_TYPE_CPU",
            "deviceName         = NVIDIA GeForce RTX Fake",
            "driverName         = llvmpipe", ""))
        source = {"device_name": "NVIDIA GeForce RTX Fake", "api_version": "1.4.351",
                  "driver_version": 2580660800, "vendor_id": 0x10DE, "device_id": 0x2803,
                  "device_type": "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU", "vulkaninfo_summary": summary}
        findings = []
        audit.verify_physical_identity(self.stage, self.stage, source, "hardware", [], findings,
                                       "synthetic fake GPU")
        codes = {item["code"] for item in findings}
        self.assertIn("fake-hardware", codes)
        self.assertIn("physical-type-mismatch", codes)

    def test_recorded_validator_diagnostic_is_not_pass(self):
        base = self.stage / "test/Native/product"
        directory = base / "vulkan/shader-reconstruction/spirv-validation"
        directory.mkdir(parents=True)
        shader_bytes = b"synthetic spirv"
        shader_path = directory / "shader-0000.spv"
        shader_path.write_bytes(shader_bytes)
        digest = hashlib.sha256(shader_bytes).hexdigest()
        log_record = {"schema_version": 1, "kind": "ncnn.spirv_tool_invocation",
                      "argv": ["/usr/bin/spirv-val", "--target-env", "vulkan1.1",
                               str(shader_path)], "returncode": 0,
                      "stdout": "", "stderr": "VUID-VkBuffer-usage-00998 validation error",
                      "shader": {"path": shader_path.name, "sha256": digest,
                                 "size_bytes": len(shader_bytes)}}
        log_path = directory / "validation-0000.json"
        log_path.write_text(json.dumps(log_record))
        item = {"kernel_id": 0, "log_file": "vulkan/shader-reconstruction/spirv-validation/validation-0000.json",
                "log_sha256": audit.sha256_file(log_path),
                "shader_file": "vulkan/shader-reconstruction/spirv-validation/shader-0000.spv",
                "sha256": digest, "size_bytes": len(shader_bytes)}
        findings = []
        audit._audit_recorded_shaders(self.stage, base, [item], {"0": digest},
                                      "synthetic product", [], findings)
        self.assertIn("spirv-validator-diagnostic", {item["code"] for item in findings})

    def test_duplicate_spirv_bytes_keep_one_record_per_kernel(self):
        base = self.stage / "test/Native/product"
        directory = base / "vulkan/shader-reconstruction/spirv-validation"
        directory.mkdir(parents=True)
        payload = b"same valid bytes for two kernel outputs"
        digest = hashlib.sha256(payload).hexdigest()
        records = []
        for index, kernel_id in enumerate((3, 4)):
            shader_name = f"shader-{index:04d}.spv"
            log_name = f"validation-{index:04d}.json"
            shader_path = directory / shader_name
            shader_path.write_bytes(payload)
            invocation = {"schema_version": 1, "kind": "ncnn.spirv_tool_invocation",
                          "argv": ["/usr/bin/spirv-val", "--target-env", "vulkan1.1", shader_name],
                          "returncode": 0, "stdout": "", "stderr": "",
                          "shader": {"path": shader_name, "sha256": digest,
                                     "size_bytes": len(payload)}}
            log_path = directory / log_name
            log_path.write_text(json.dumps(invocation))
            records.append({"kernel_id": kernel_id,
                            "log_file": f"vulkan/shader-reconstruction/spirv-validation/{log_name}",
                            "log_sha256": audit.sha256_file(log_path),
                            "shader_file": f"vulkan/shader-reconstruction/spirv-validation/{shader_name}",
                            "sha256": digest, "size_bytes": len(payload)})
        findings = []
        audit._audit_recorded_shaders(self.stage, base, records,
                                      {"3": digest, "4": digest},
                                      "duplicate-hash product", [], findings)
        self.assertEqual([], findings)

    def test_offline_claim_cannot_be_upgraded_to_gpu_numeric(self):
        base = self.stage / "test/Native/vulkan-product-offline"
        base.mkdir(parents=True)
        (base / "evidence.json").write_text(json.dumps({
            "schema_version": 1, "kind": "ncnn.vulkan_offline_evidence",
            "execution": {"mode": "offline-no-device", "vulkan_execution": "passed",
                          "gpu_output": [1.0], "vk_icd_manifest_exists": False,
                          "vk_icd_filenames": str(base / "missing.json")},
            "claims": {"vulkan_device_execution": "passed",
                       "cpu_gpu_numerical_comparison": "passed"},
        }), encoding="utf-8")
        findings = []
        audit._audit_offline(self.stage, [], findings)
        codes = {item["code"] for item in findings}
        self.assertIn("offline-gpu-claim", codes)
        self.assertIn("offline-claim-overreach", codes)
        self.assertIn("offline-cpu-loader-record", codes)

    def test_device_independent_runtime_allowlist_matches_source(self):
        source = Path(__file__).resolve().parents[1] / "Unit/vulkan_runtime_test.cpp"
        cases = set(re.findall(r"TEST\(VulkanRuntimeTest,\s*([A-Za-z0-9_]+)\)",
                               source.read_text(encoding="utf-8")))
        self.assertEqual({f"VulkanRuntimeTest.{case}" for case in cases},
                         set(audit.OFFLINE_RUNTIME_TESTS))
        self.assertEqual(len(cases), 5)

    def test_every_profile_requires_device_independent_runtime_units(self):
        for profile in ("release", "sanitize", "offline", "software", "hardware"):
            required = audit._required_tests(profile)
            self.assertTrue(audit.OFFLINE_RUNTIME_TESTS.issubset(required))
            for missing in sorted(audit.OFFLINE_RUNTIME_TESTS):
                with self.subTest(profile=profile, missing=missing):
                    gate = {"profile": profile,
                            "ctest": {"test_names": sorted(required - {missing})}}
                    findings = []
                    audit._check_required_tests(gate, findings)
                    finding = next(item for item in findings
                                   if item["code"] == "required-ctest-missing")
                    self.assertIn(missing, finding["details"]["tests"])
        for identifier in ("V6.1", "V6.2", "V6.4"):
            self.assertTrue(audit.OFFLINE_RUNTIME_TESTS.issubset(
                audit.ID_CATALOG[identifier]["tests"]))

    def make_cpu_loader_record_fixture(self, *, sanitizer_args=(),
                                       cpu_dir_name="cpu",
                                       input_name="public-input.f32",
                                       output_name="cpu-output.f32"):
        base = self.stage / "test/Native/vulkan-product-offline"
        loader_relative = f"cpu-loader-{cpu_dir_name}/cpu-product-loader"
        result_relative = f"cpu-loader-{cpu_dir_name}/result.json"
        cpu_dir = base / cpu_dir_name
        loader_dir = base / f"cpu-loader-{cpu_dir_name}"
        cpu_dir.mkdir(parents=True, exist_ok=True)
        loader_dir.mkdir(parents=True, exist_ok=True)
        (cpu_dir / "paired_relu.h").write_text("/* CPU API fixture */\n", encoding="utf-8")
        (cpu_dir / "libpaired_relu.so").write_bytes(b"CPU library fixture")
        (base / input_name).write_bytes(b"input")
        (base / output_name).write_bytes(b"output")
        loader = base / loader_relative
        loader.write_bytes(b"native loader fixture")
        result_path = base / result_relative
        result_path.write_text(json.dumps({
            "status": "success", "phase": "complete", "cpu_status": 0,
            "vulkan_create_status": -1, "vulkan_run_status": -1,
            "vulkan_destroy_status": -1, "vulkan_destroy_null_status": -1,
            "diagnostic": "",
        }), encoding="utf-8")
        source = (Path(audit.__file__).resolve().parents[1] /
                  "Native/vulkan_product_loader.cpp").resolve()
        compile_argv = [
            "/usr/bin/c++", "-std=c++23", *sanitizer_args,
            "-DNCNN_PRODUCT_TEST_MODEL=paired_relu",
            "-DNCNN_PRODUCT_TEST_CPU_ONLY=1", "-I", str(cpu_dir),
            str(source), "-ldl", "-o", str(loader),
        ]
        command = [
            str(loader), "paired_relu", str(cpu_dir / "libpaired_relu.so"), "257",
            str(base / input_name), str(base / output_name), str(result_path),
        ]
        streams = {}
        for name in ("compile.stdout.log", "compile.stderr.log",
                     "run.stdout.log", "run.stderr.log"):
            path = loader_dir / name
            path.write_text("", encoding="utf-8")
            streams[name] = {"file": str(path.relative_to(base)),
                             "sha256": audit.sha256_file(path)}
        return {
            "compile_argv": compile_argv, "compile_returncode": 0,
            "argv": command, "returncode": 0,
            "loader_file": loader_relative, "loader_sha256": audit.sha256_file(loader),
            "result_file": result_relative, "result_sha256": audit.sha256_file(result_path),
            "streams": streams,
        }

    def audit_cpu_loader_record(self, record, *, sanitizer_flags=()):
        base = self.stage / "test/Native/vulkan-product-offline"
        findings = []
        audit._audit_offline_cpu_loader(
            self.stage, base, record, "initial", "cpu", "public-input.f32",
            "cpu-output.f32", set(sanitizer_flags), [], findings)
        return findings

    def test_offline_cpu_loader_record_accepts_cpu_only_invocation(self):
        record = self.make_cpu_loader_record_fixture()
        self.assertEqual([], self.audit_cpu_loader_record(record))

    def test_offline_cpu_loader_record_rejects_path_and_status_tampering(self):
        record = self.make_cpu_loader_record_fixture()
        bad_argv = dict(record)
        bad_argv["argv"] = list(record["argv"])
        bad_argv["argv"][4] = str(self.stage / "wrong-input.f32")
        codes = {item["code"] for item in self.audit_cpu_loader_record(bad_argv)}
        self.assertIn("offline-cpu-loader-command", codes)

        bad_compile = dict(record)
        bad_compile["compile_argv"] = list(record["compile_argv"])
        bad_compile["compile_argv"][bad_compile["compile_argv"].index(
            "-DNCNN_PRODUCT_TEST_CPU_ONLY=1")] = "-DNCNN_PRODUCT_TEST_CPU_ONLY=0"
        bad_compile["returncode"] = True
        codes = {item["code"] for item in self.audit_cpu_loader_record(bad_compile)}
        self.assertIn("offline-cpu-loader-command", codes)
        self.assertIn("offline-cpu-loader-status", codes)

    def test_offline_cpu_loader_record_rejects_sanitizer_logs_and_bad_hashes(self):
        record = self.make_cpu_loader_record_fixture()
        stream = record["streams"]["run.stderr.log"]
        log_path = self.stage / "test/Native/vulkan-product-offline" / stream["file"]
        log_path.write_text("ERROR: AddressSanitizer: heap-use-after-free\n", encoding="utf-8")
        stream["sha256"] = audit.sha256_file(log_path)
        codes = {item["code"] for item in self.audit_cpu_loader_record(record)}
        self.assertIn("offline-cpu-loader-sanitizer-diagnostic", codes)

        log_path.write_text("", encoding="utf-8")
        stream["sha256"] = audit.sha256_file(log_path)
        record["loader_sha256"] = "0" * 64
        codes = {item["code"] for item in self.audit_cpu_loader_record(record)}
        self.assertIn("evidence-hash-mismatch", codes)

    def test_offline_cpu_loader_sanitizer_flags_are_fail_closed(self):
        self.make_stage(profile="sanitize")
        sanitizer_args = (
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer", "-shared-libasan",
        )
        record = self.make_cpu_loader_record_fixture(sanitizer_args=sanitizer_args)
        self.assertEqual([], self.audit_cpu_loader_record(
            record, sanitizer_flags={"address", "undefined"}))
        record["compile_argv"].remove("-fno-sanitize-recover=all")
        codes = {item["code"] for item in self.audit_cpu_loader_record(
            record, sanitizer_flags={"address", "undefined"})}
        self.assertIn("offline-cpu-loader-command", codes)

    def test_offline_cpu_loader_rejects_overrides_and_response_files(self):
        sanitizer_args = (
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer", "-shared-libasan",
        )
        record = self.make_cpu_loader_record_fixture(sanitizer_args=sanitizer_args)
        for extra in ("-fno-sanitize=address", "-fno-sanitize=undefined",
                      "-fno-sanitize=all", "-fsanitize-recover=undefined",
                      "-fomit-frame-pointer", "@unverified-options.rsp",
                      "-UNCNN_PRODUCT_TEST_CPU_ONLY"):
            with self.subTest(extra=extra):
                changed = dict(record)
                changed["compile_argv"] = list(record["compile_argv"])
                changed["compile_argv"].insert(3, extra)
                codes = {item["code"] for item in self.audit_cpu_loader_record(
                    changed, sanitizer_flags={"address", "undefined"})}
                self.assertIn("offline-cpu-loader-command", codes)
        release = self.make_cpu_loader_record_fixture()
        release["compile_argv"].insert(2, "@unverified-options.rsp")
        codes = {item["code"] for item in self.audit_cpu_loader_record(release)}
        self.assertIn("offline-cpu-loader-command", codes)

    def test_offline_valid_evidence_with_cpu_loaders_passes(self):
        base = self.stage / "test/Native/vulkan-product-offline"
        initial = self.make_cpu_loader_record_fixture()
        recovery = self.make_cpu_loader_record_fixture(
            cpu_dir_name="cpu-after-negative-cases",
            input_name="public-input-after-negative-cases.f32",
            output_name="cpu-output-after-negative-cases.f32")

        def write(name, data):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return audit.sha256_file(path)

        def tensor(name, values):
            digest = write(name, struct.pack(f"<{len(values)}f", *values))
            return {"file": name, "sha256": digest, "values": values}

        inputs = [float(index % 7 - 3) for index in range(257)]
        outputs = [max(value, 0.0) for value in inputs]
        input_record = tensor("public-input.f32", inputs)
        output_record = tensor("cpu-output.f32", outputs)
        golden_record = tensor("scalar-golden.f32", outputs)
        tensor("public-input-after-negative-cases.f32", inputs)
        recovery_record = tensor("cpu-output-after-negative-cases.f32", outputs)
        fixture = {
            "model": "paired_relu",
            "param_sha256": write("model.param", b"synthetic ReLU fixture"),
            "bin_sha256": write("model.bin", b""),
        }
        artifacts = {
            "cpu_library_sha256": "cpu/libpaired_relu.so",
            "cpu_header_sha256": "cpu/paired_relu.h",
            "cpu_manifest_sha256": "cpu/paired_relu.json",
            "cpu_recovery_library_sha256": "cpu-after-negative-cases/libpaired_relu.so",
            "vulkan_library_sha256": "vulkan/libpaired_relu_vulkan.so",
            "vulkan_header_sha256": "vulkan/paired_relu_vulkan.h",
            "vulkan_manifest_sha256": "vulkan/paired_relu_vulkan.json",
            "vulkan_plan_sha256": "vulkan/paired_relu_vulkan.plan.json",
        }
        write("vulkan/paired_relu_vulkan.json", json.dumps({
            "host_implementation": "generic-plan-runtime", "host_codegen_version": 1,
            "execution_plan_usage": "embedded-runtime-required"}).encode())
        for field, name in artifacts.items():
            if not (base / name).exists():
                write(name, b"synthetic product artifact")
            fixture[field] = audit.sha256_file(base / name)
        snapshot = {path.name: audit.sha256_file(path)
                    for path in (base / "vulkan").iterdir() if path.is_file()}
        fixture["offline_success_artifacts_sha256"] = snapshot
        shader_name = "vulkan/shader-reconstruction/kernel-0.spv"
        shader_bytes = b"synthetic shader bytes"
        shader_digest = write(shader_name, shader_bytes)
        fixture["reconstructed_shader_sha256"] = {"0": shader_digest}
        fixture["shader_files"] = [{"file": shader_name, "sha256": shader_digest}]
        validated_shader = "vulkan/shader-reconstruction/spirv-validation/shader-0000.spv"
        write(validated_shader, shader_bytes)
        invocation = {
            "schema_version": 1, "kind": "ncnn.spirv_tool_invocation",
            "argv": ["/usr/bin/spirv-val", "--target-env", "vulkan1.1",
                     str(base / validated_shader)],
            "returncode": 0, "stdout": "", "stderr": "",
            "shader": {"path": "shader-0000.spv", "sha256": shader_digest,
                       "size_bytes": len(shader_bytes)},
        }
        log_name = "vulkan/shader-reconstruction/spirv-validation/validation-0000.json"
        log_digest = write(log_name, json.dumps(invocation).encode())
        command = [str(self.stage / "tools/ncnn-compile"), str(base / "model.param"),
                   "--bin", str(base / "model.bin"), "--model-name", "paired_relu",
                   "--output-dir", str(base / "vulkan"), "--backend=vulkan",
                   "--precision=f32"]
        negatives = []
        for name, status, diagnostic, option in (
            ("missing-validator", 1, "cannot execute tool: posix_spawn failed: no such file",
             "--vulkan-spirv-val=" + str(base / "missing-spirv-val")),
            ("validator-failure", 23, "command failed with exit code 23",
             "--vulkan-spirv-val=" + str(base / "negative-tools/reject-spirv-val")),
            ("link-failure", 29, "command failed with exit code 29",
             "--vulkan-host-cxx=" + str(base / "negative-tools/reject-host-link")),
            ("audit-failure", 31, "command failed with exit code 31",
             "--readelf=" + str(base / "negative-tools/reject-readelf")),
            ("no-device-execution-verification", 1, "no Vulkan physical devices",
             "--verify-execution"),
        ):
            negatives.append({
                "name": name, "argv": [*command, option], "returncode": status,
                "stdout": "", "stderr": diagnostic, "diagnostic_match": True,
                "artifacts_before_sha256": snapshot, "artifacts_after_sha256": snapshot,
                "artifacts_unchanged": True,
            })
        evidence = {
            "schema_version": 1, "kind": "ncnn.vulkan_offline_evidence",
            "execution": {"mode": "offline-no-device", "vulkan_execution": "not_run",
                          "gpu_output": None, "vk_icd_manifest_exists": False,
                          "vk_icd_filenames": str(base / "no-available-icd.json")},
            "claims": {"vulkan_device_execution": "not_run",
                       "cpu_gpu_numerical_comparison": "not_claimed"},
            "fixture": fixture,
            "public_io": {
                "element_type": "f32", "shape": [257], "shape_chw": [1, 1, 257],
                "host_layout": "CHW", "input": input_record,
                "cpu_output": output_record, "scalar_golden": golden_record,
                "cpu_recovery_output_sha256": recovery_record["sha256"],
                "tolerance": {"absolute": 1e-6, "relative": 1e-6},
                "max_absolute_error": 0.0, "max_relative_error": 0.0,
                "cpu_recovery_max_absolute_error": 0.0, "passed": True,
                "cpu_status": 0, "cpu_recovery_status": 0,
            },
            "shader_validations": [{"kernel_id": 0, "log_file": log_name,
                                    "log_sha256": log_digest,
                                    "shader_file": validated_shader,
                                    "sha256": shader_digest,
                                    "size_bytes": len(shader_bytes)}],
            "negative_cases": negatives,
            "no_device_lifecycle": no_device_record(self.stage),
            "commands": {"offline_compile_returncode": 0, "cpu_loader": initial,
                         "cpu_recovery_loader": recovery},
        }
        write("evidence.json", json.dumps(evidence).encode())
        findings = []
        result = audit._audit_offline(self.stage, [], findings)
        self.assertIsNotNone(result)
        self.assertEqual([], findings)
        del evidence["commands"]["cpu_recovery_loader"]
        write("evidence.json", json.dumps(evidence).encode())
        findings = []
        audit._audit_offline(self.stage, [], findings)
        self.assertIn("offline-cpu-loader-record", {item["code"] for item in findings})

    def make_v5_artifact_fixture(self):
        model = "synthetic_model"
        self._v5_fixture_counter = getattr(self, "_v5_fixture_counter", 0) + 1
        model_root = (self.stage / "test/Vulkan/evidence/v5" /
                      f"multiop-nvidia-run{self._v5_fixture_counter}" / "conv_relu")
        validation_root = model_root / "spirv-validation"
        validation_root.mkdir(parents=True)
        (model_root / "cpu").mkdir()
        (model_root / "vulkan").mkdir()
        payload = b"same shader bytes for two distinct kernel identities"
        digest = hashlib.sha256(payload).hexdigest()
        kernels = [{"id": 7, "entry_point": "kernel_7"},
                   {"id": 8, "entry_point": "kernel_8"}]
        plan_path = model_root / "vulkan" / f"{model}_vulkan.plan.json"
        plan_path.write_text(json.dumps({"kernels": kernels, "resources": []}))
        plan_hash = audit.sha256_file(plan_path)
        manifest_path = model_root / "vulkan" / f"{model}_vulkan.json"
        manifest_shaders = [{"kernel_id": kernel["id"],
                             "entry_point": kernel["entry_point"],
                             "sha256": digest, "size_bytes": len(payload)}
                            for kernel in kernels]
        manifest_path.write_text(json.dumps({"plan_sha256": plan_hash,
                                             "host_implementation": "generic-plan-runtime",
                                             "host_codegen_version": 1,
                                             "execution_plan_usage": "embedded-runtime-required",
                                             "shaders": manifest_shaders}))
        cpu_library = model_root / "cpu" / f"lib{model}.so"
        vk_library = model_root / "vulkan" / f"lib{model}_vulkan.so"
        cpu_library.write_bytes(b"cpu library")
        vk_library.write_bytes(b"vulkan library")
        cpu_manifest = model_root / "cpu" / f"{model}.json"
        cpu_header = model_root / "cpu" / f"{model}.h"
        vk_header = model_root / "vulkan" / f"{model}_vulkan.h"
        cpu_manifest.write_text("{}")
        cpu_header.write_text("CPU header")
        vk_header.write_text("header")
        validations = []
        identity_entries = []
        for index, kernel in enumerate(kernels):
            shader_file = validation_root / f"shader-{index:04d}.spv"
            shader_file.write_bytes(payload)
            log_path = validation_root / f"validation-{index:04d}.json"
            invocation = {"schema_version": 1, "kind": "ncnn.spirv_tool_invocation",
                          "argv": ["/usr/bin/spirv-val", "--target-env", "vulkan1.1",
                                   str(shader_file)],
                          "returncode": 0, "stdout": "", "stderr": "",
                          "shader": {"path": shader_file.name, "sha256": digest,
                                     "size_bytes": len(payload)}}
            log_path.write_text(json.dumps(invocation))
            validations.append({"kernel_id": kernel["id"],
                                "log": log_path.name,
                                "log_sha256": audit.sha256_file(log_path),
                                "path": shader_file.name,
                                "sha256": digest, "size_bytes": len(payload)})
            identity_entries.append({**manifest_shaders[index],
                                     "binary_symbol": f"{model}_kernel_{kernel['id']}"})
        identity = {"cpu_library_sha256": audit.sha256_file(cpu_library),
                    "cpu_header_sha256": audit.sha256_file(cpu_header),
                    "cpu_manifest_sha256": audit.sha256_file(cpu_manifest),
                    "vulkan_header_sha256": audit.sha256_file(vk_header),
                    "vulkan_manifest_sha256": audit.sha256_file(manifest_path),
                    "vulkan_library_sha256": audit.sha256_file(vk_library),
                    "plan_sha256": plan_hash,
                    "shader_entries": identity_entries,
                    "shader_validations": validations,
                    "shader_validation_directory": "spirv-validation"}
        return model_root, {"model": model, "identities": identity}

    def test_offline_negative_diagnostic_cannot_supply_its_own_regex(self):
        self.assertFalse(audit._offline_negative_cause_valid(
            "missing-validator", 1, "unrelated output matches any character"))
        self.assertFalse(audit._offline_negative_cause_valid(
            "missing-validator", True, "cannot execute '/tool': posix_spawn failed: no such file"))

    def test_v0_negative_rejects_boolean_returncode(self):
        base = self.stage / "test/Vulkan/evidence/gpu-func/nvidia"
        logs = base / "logs"
        logs.mkdir(parents=True)
        (base / "device-result.json").write_text(json.dumps({}))
        (base / "invalid.spv").write_bytes(bytes(4))
        record = {"argv": ["/usr/bin/spirv-val", "--target-env", "vulkan1.1",
                           str(base / "invalid.spv")],
                  "returncode": True, "stdout": "",
                  "stderr": "Invalid SPIR-V magic number"}
        (logs / "negative-invalid-spirv.json").write_text(json.dumps(record))
        findings = []
        audit._check_v0_negative_cases(base, [], findings)
        self.assertIn("v0-negative-cause", {item["code"] for item in findings})

        record["returncode"] = 1
        record["argv"][-1] = str(base / "unrelated.spv")
        record["stderr"] = "unrelated diagnostic"
        (logs / "negative-invalid-spirv.json").write_text(json.dumps(record))
        findings = []
        audit._check_v0_negative_cases(base, [], findings)
        codes = {item["code"] for item in findings}
        self.assertIn("v0-negative-cause", codes)
        self.assertIn("v0-negative-command", codes)

    def test_v0_oversize_negative_is_bound_to_the_runtime_limit(self):
        base = self.stage / "test/Vulkan/evidence/gpu-func/nvidia"
        runner = str(self.stage / "test/Vulkan/vulkan_add_runner")
        argv = [runner, "--shader", str(base / "add.spv"),
                "--cpu-library", str(self.stage / "test/Vulkan/cpu-add/libv0_add.so"),
                "--cpu-symbol", "v0_add", "--device-match", "NVIDIA",
                "--evidence", str(base / "negative-oversize.json"),
                "--probe-buffer-bytes", "4097", "--validation"]
        record = {"argv": argv}
        self.assertTrue(audit._v0_negative_argv_valid(
            self.stage, base, "negative-oversize-buffer.json", record,
            "NVIDIA GeForce", 4096))
        record["argv"][-2] = "4096"
        self.assertFalse(audit._v0_negative_argv_valid(
            self.stage, base, "negative-oversize-buffer.json", record,
            "NVIDIA GeForce", 4096))

    def test_inline_v0_summary_must_match_archived_bytes(self):
        base = self.stage / "test/Vulkan/evidence/gpu-func/nvidia"
        base.mkdir(parents=True)
        (base / "vulkaninfo-summary.txt").write_text("physical device summary\n")
        findings = []
        audit._check_inline_device_summary(self.stage, base, "different inline summary",
                                           audit.sha256_file(base / "vulkaninfo-summary.txt"),
                                           "V0 synthetic", [], findings)
        self.assertIn("v0-summary-inline-mismatch", {item["code"] for item in findings})

    def test_v0_v2_ir_shader_plan_claims_are_rehashed(self):
        claims = {}
        for relative in ("v0/gpu-binary.mlir", "v0/add.spvasm", "v2/gpu-kernels.mlir",
                         "v2/gpu-binary.mlir", "v2/execution-plan.json",
                         "v2/kernel.spv", "v2/kernel.spvasm"):
            path = self.write(relative, b"original evidence bytes")
            claims[relative] = audit.sha256_file(path)
            path.write_bytes(b"tampered evidence bytes")
        findings = []
        for relative, claimed in claims.items():
            audit._check_digest_claim(claimed, self.stage / relative,
                                      f"synthetic {relative}", findings)
        self.assertEqual(len(claims), len(findings))

    def test_validator_log_must_bind_the_audited_shader_operand(self):
        base = self.stage / "test/Vulkan/evidence/v2/add-nvidia"
        logs = base / "logs"
        logs.mkdir(parents=True)
        shader = base / "kernel.spv"
        shader.write_bytes(b"audited shader")
        unrelated = base / "other.spv"
        unrelated.write_bytes(b"different shader")
        log = {"returncode": 0,
               "argv": ["/usr/bin/spirv-val", "--target-env", "vulkan1.1", str(unrelated)],
               "stdout": "", "stderr": ""}
        (logs / "spirv-validation.json").write_text(json.dumps(log))
        findings = []
        audit._check_logged_validation(base, "spirv-validation.json", "synthetic V2",
                                       shader, audit.sha256_file(shader), [], findings)
        self.assertIn("spirv-validation-path-mismatch", {item["code"] for item in findings})

    def test_v4_duplicate_missing_wrong_cause_and_wrong_command_fail_closed(self):
        base = self.stage / "test/Native/vulkan-product-cli"
        base.mkdir(parents=True)
        param, model_bin = base / "model.param", base / "model.bin"
        param.write_bytes(b"model param")
        model_bin.write_bytes(b"model bin")
        fixture = {"param_sha256": audit.sha256_file(param),
                   "bin_sha256": audit.sha256_file(model_bin)}
        record = {"argv": [str(self.stage / "tools/ncnn-compile"), str(param),
                            "--bin", str(model_bin), "--model-name", "foo",
                            "--output-dir", str(base / "collision"),
                            "--backend=vulkan", "--precision=f32", "--verify-execution"],
                  "returncode": 1, "stdout": "",
                  "stderr": "output directory belongs to CPU model 'foo_vulkan'"}
        findings = []
        audit._audit_v4_failure_records(self.stage, base, [record, record],
                                        fixture, [], "V4 synthetic", findings)
        codes = {item["code"] for item in findings}
        self.assertIn("product-negative-coverage", codes)
        self.assertTrue(any(item.get("details", {}).get("duplicates") == ["foo"]
                            for item in findings if item["code"] == "product-negative-coverage"))

        wrong_cause = dict(record, stderr="unrelated failure")
        wrong_command = dict(record, argv=["/tmp/attacker-ncnn-compile", *record["argv"][1:]])
        findings = []
        audit._audit_v4_failure_records(self.stage, base, [wrong_cause, wrong_command],
                                        fixture, [], "V4 synthetic", findings)
        codes = {item["code"] for item in findings}
        self.assertIn("product-negative-diagnostic", codes)
        self.assertIn("product-negative-command", codes)

        unrelated_bin = base / "unrelated.bin"
        unrelated_bin.write_bytes(b"not the paired fixture")
        wrong_input_argv = list(record["argv"])
        wrong_input_argv[wrong_input_argv.index("--bin") + 1] = str(unrelated_bin)
        findings = []
        audit._audit_v4_failure_records(
            self.stage, base, [dict(record, argv=wrong_input_argv)],
            fixture, [], "V4 synthetic", findings)
        self.assertIn("product-negative-command",
                      {item["code"] for item in findings})

        bad_fixture = dict(fixture, bin_sha256="f" * 64)
        findings = []
        audit._audit_v4_failure_records(self.stage, base, [record],
                                        bad_fixture, [], "V4 synthetic", findings)
        self.assertIn("evidence-hash-mismatch", {item["code"] for item in findings})

        unsupported_param = base / "unsupported-sigmoid.param"
        unsupported_bin = base / "unsupported-sigmoid.bin"
        unsupported_param.write_bytes(audit.V4_UNSUPPORTED_SIGMOID_PARAM)
        unsupported_bin.write_bytes(b"")
        unsupported_argv = [str(self.stage / "tools/ncnn-compile"),
                            str(unsupported_param), "--bin", str(unsupported_bin),
                            "--model-name", "unsupported", "--output-dir",
                            str(base / "unsupported"), "--backend=vulkan",
                            "--precision=f32", "--verify-execution"]
        unsupported_record = {"argv": unsupported_argv, "returncode": 1,
                              "stdout": "", "stderr": "tosa.clamp is not supported by Vulkan graph"}
        findings = []
        audit._audit_v4_failure_records(self.stage, base, [unsupported_record],
                                        fixture, [], "V4 synthetic", findings)
        self.assertNotIn("product-negative-command",
                         {item["code"] for item in findings})
        unsupported_bin.write_bytes(b"unrelated bytes")
        findings = []
        audit._audit_v4_failure_records(self.stage, base, [unsupported_record],
                                        fixture, [], "V4 synthetic", findings)
        self.assertIn("evidence-hash-mismatch", {item["code"] for item in findings})

    def test_v5_plan_manifest_and_identity_bind_kernel_ids(self):
        model_root, evidence = self.make_v5_artifact_fixture()
        findings = []
        audit._audit_v5_artifacts(self.stage, model_root.parent.parent, model_root,
                                  evidence, [], findings)
        self.assertEqual([], findings)

        manifest_path = model_root / "vulkan/synthetic_model_vulkan.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["shaders"][1]["kernel_id"] = 99
        manifest_path.write_text(json.dumps(manifest))
        findings = []
        audit._audit_v5_artifacts(self.stage, model_root.parent.parent, model_root,
                                  evidence, [], findings)
        self.assertIn("v5-plan-manifest-kernel-map", {item["code"] for item in findings})

    def test_v5_validation_missing_id_extra_row_and_reused_shader_path_fail(self):
        model_root, evidence = self.make_v5_artifact_fixture()
        evidence["identities"]["shader_validations"][0].pop("kernel_id")
        findings = []
        audit._audit_v5_artifacts(self.stage, model_root.parent.parent, model_root,
                                  evidence, [], findings)
        self.assertIn("v5-validator-kernel-id", {item["code"] for item in findings})

        model_root, evidence = self.make_v5_artifact_fixture()
        extra = dict(evidence["identities"]["shader_validations"][0], kernel_id=99)
        evidence["identities"]["shader_validations"].append(extra)
        findings = []
        audit._audit_v5_artifacts(self.stage, model_root.parent.parent, model_root,
                                  evidence, [], findings)
        self.assertIn("spirv-validator-coverage", {item["code"] for item in findings})

        model_root, evidence = self.make_v5_artifact_fixture()
        evidence["identities"]["shader_validations"][1]["path"] = \
            evidence["identities"]["shader_validations"][0]["path"]
        findings = []
        audit._audit_v5_artifacts(self.stage, model_root.parent.parent, model_root,
                                  evidence, [], findings)
        self.assertIn("spirv-validator-shader-reused", {item["code"] for item in findings})

    def test_v5_duplicate_shader_digest_is_valid_one_to_one(self):
        model_root, evidence = self.make_v5_artifact_fixture()
        findings = []
        audit._audit_v5_artifacts(self.stage, model_root.parent.parent, model_root,
                                  evidence, [], findings)
        self.assertEqual([], findings)

    def test_v5_port_arrays_must_cover_every_declared_output(self):
        findings = []
        passed = audit._v5_port_arrays_match(
            [[1.0]], [[2.0]], [[2.0]], [[2.0]],
            [1], [1, 1], "synthetic multi-output", findings)
        self.assertFalse(passed)
        self.assertIn("v5-trial-outputs", {item["code"] for item in findings})

    def test_id_index_scopes_v2_and_v5_references_and_blocks_failed_gate(self):
        related = set()
        for identifier in ("V2.1", "V2.2", "V5.1", "V5.2"):
            related.update(audit.ID_CATALOG[identifier]["tests"])
        refs = [
            {"path": "test/Vulkan/evidence/v2/add-nvidia/metadata.json", "sha256": "a" * 64, "role": "V2 add"},
            {"path": "test/Vulkan/evidence/v2/conv3x3-nvidia/metadata.json", "sha256": "b" * 64, "role": "V2 conv"},
            {"path": "test/Vulkan/evidence/v5/v5-multiop-nvidia/evidence.json", "sha256": "c" * 64, "role": "V5 multiop"},
            {"path": "test/Vulkan/evidence/v5/v5-pooling-softmax-nvidia/evidence.json", "sha256": "d" * 64, "role": "V5 pool"},
        ]
        stage = {"stage": "synthetic-release", "profile": "release", "scope_passed": True,
                 "families": {"cpu_baseline": {"path": "cpu"},
                              "v2": {"add-nvidia": {"identity": {"cpu_library_sha256": "1" * 64}},
                                     "conv3x3-nvidia": {"identity": {"cpu_library_sha256": "2" * 64}}},
                              "v5": {"v5-multiop-nvidia": {"conv_relu": {"cpu_library_sha256": "3" * 64}},
                                     "v5-pooling-softmax-nvidia": {"max_pool_tail": {"cpu_library_sha256": "4" * 64}}}},
                 "raw_artifacts": refs, "ctest": {"test_names": sorted(related)}}
        index = audit._make_id_index([stage], release_eligible=False, all_refs=refs)
        self.assertTrue(all("conv3x3" not in ref["path"]
                            for ref in index["V2.1"]["raw_artifacts"]))
        self.assertTrue(all("multiop" in ref["path"]
                            for ref in index["V5.1"]["raw_artifacts"]))
        self.assertTrue(all("pooling-softmax" in ref["path"]
                            for ref in index["V5.2"]["raw_artifacts"]))

        failed = dict(stage, profile="software", scope_passed=False,
                      ctest={"test_names": sorted(audit._applicable_tests(
                          audit.ID_CATALOG["V2.1"]["tests"], "software"))})
        failed_index = audit._make_id_index([failed], release_eligible=False,
                                            all_refs=refs)
        self.assertEqual("blocked", failed_index["V2.1"]["status"])

    def test_software_v4_profile_requires_only_generic_test_name(self):
        names = ["ncnn-compile-vulkan-product",
                 "ncnn-compile-vulkan-product-llvmpipe"]
        self.assertEqual({names[0]}, audit._applicable_tests(names, "software"))
        self.assertEqual(set(names), audit._applicable_tests(names, "release"))

    def test_sanitizer_final_audit_rehashes_static_archive_and_source_archive(self):
        self.make_stage(profile="sanitize")
        identity_path = self.stage / self.state["sanitizer_toolchain_evidence"]["identity"]["path"]
        identity = json.loads(identity_path.read_text())
        archive = Path(identity["llvm"]["static_archive_artifacts"][0]["path"])
        archive.write_bytes(b"changed after preflight")
        source = Path(identity["source_archive_identity"]["path"])
        source.write_bytes(b"changed source archive")
        codes = {item["code"] for item in self.findings()}
        self.assertIn("sanitizer-archive-hash", codes)
        self.assertIn("sanitizer-source-archive", codes)

    def test_sanitizer_archive_list_and_exact_count_records_fail_closed(self):
        self.make_stage(profile="sanitize")
        identity_path = self.stage / self.state["sanitizer_toolchain_evidence"]["identity"]["path"]
        identity = json.loads(identity_path.read_text())
        archive_list = Path(identity["llvm"]["static_archive_list"]["path"])
        original = archive_list.read_text()
        archive_list.write_text(original.replace("libLLVM-static-0.a", "libLLVM-static-wrong.a"))
        identity["llvm"]["static_archive_list"]["sha256"] = audit.sha256_file(archive_list)
        identity_path.write_text(json.dumps(identity))
        findings = []
        audit._audit_sanitizer_toolchain(self.stage, self.state,
                                         self.stage / "CMakeCache.txt", [], findings)
        self.assertIn("sanitizer-archive-list-mismatch", {item["code"] for item in findings})

        identity = json.loads(identity_path.read_text())
        compile_audit = Path(identity["llvm"]["compile_pic_sanitizer_audit"])
        compile_audit.write_text("compile_commands=43160\nsanitzer_instrumented_sources=4312\npic_sources=4316\n")
        findings = []
        audit._audit_sanitizer_toolchain(self.stage, self.state,
                                         self.stage / "CMakeCache.txt", [], findings)
        self.assertIn("sanitizer-pic-audit", {item["code"] for item in findings})

    def test_stage_id_keeps_historical_default_and_rejects_explicit_invalid_values(self):
        self.state.pop("stage_id")
        self.save_state()
        gate = audit.audit_stage_gate(self.stage, verify_fingerprint=False)
        self.assertEqual("VA0", gate["stage_id"])
        self.assertNotIn("gate-execution-role",
                         {item["code"] for item in gate["findings"]})

        for invalid in (None, "", "VA9", []):
            with self.subTest(stage_id=invalid):
                self.state["stage_id"] = invalid
                self.save_state()
                gate = audit.audit_stage_gate(self.stage, verify_fingerprint=False)
                self.assertIn("gate-execution-role",
                              {item["code"] for item in gate["findings"]})

    def test_va2_inventory_and_ids_are_separate_from_va0_legacy_scope(self):
        required = audit._required_tests("release", "VA2")
        self.assertTrue(audit.VA2_HELPER_TESTS.issubset(required))
        self.assertTrue(audit.VA2_RUNTIME_BUDGET_TESTS.issubset(required))
        new_va2_tests = audit.VA2_HELPER_TESTS | audit.VA2_RUNTIME_BUDGET_TESTS
        self.assertFalse(new_va2_tests & audit._required_tests("release", "VA0"))
        self.assertEqual({"VA2.1", "VA2.2", "VA2.3", "VA2.4"},
                         {identifier for identifier in audit.ID_CATALOG
                          if identifier.startswith("VA2.")})

        inventory_names = (audit._required_tests("release", "VA0") |
                           audit.VA1_REQUIRED_TESTS | audit.VA2_REQUIRED_TESTS)
        migration = audit._va0_migration_coverage({
            "profile": "release", "ctest": {"test_names": sorted(inventory_names)}})
        self.assertFalse(any(identifier.startswith("VA2.")
                             for identifier in migration["current_ids"]))
        self.assertFalse(any(identifier.startswith("VA2.")
                             for row in migration["current_test_cases"].values()
                             for identifier in row["related_ids"]))
        self.assertFalse(audit.VA2_REQUIRED_TESTS & set(migration["current_test_cases"]))

    def test_va2_release_and_static_san_pair_qualifies_without_removing_va3_blocker(self):
        reports = []
        for profile in ("release", "sanitize"):
            reports.append({
                "stage": f"synthetic-va2-{profile}", "profile": profile,
                "stage_id": "VA2", "source_fingerprint": "f" * 64,
                "scope_passed": True, "findings": [], "va0_findings": [],
                "ctest": {"test_names": sorted(audit.VA2_REQUIRED_TESTS),
                          "skipped": [], "failed": []},
                "families": {"va2_helper": {"status": "verified",
                                              "raw_artifacts": []}},
                "raw_artifacts": [],
            })
        gate_states = [
            {"stage": row["stage"], "stage_id": "VA2", "profile": row["profile"],
             "source_fingerprint": row["source_fingerprint"], "ctest": row["ctest"],
             "findings": [], "passed": True}
            for row in reports]
        with patch.object(audit, "audit_stage_gate", side_effect=gate_states), \
                patch.object(audit, "audit_stage_evidence", side_effect=reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "verified", "findings": []}) as archive_audit:
            report = audit.build_report([Path("release"), Path("sanitize")],
                                        require_va2_helper=True)
        self.assertTrue(report["va2_helper_eligible"])
        self.assertTrue(report["va2_helper_required"])
        self.assertEqual([], report["va2_helper_blockers"])
        self.assertFalse(report["host_aot_release_eligible"])
        self.assertTrue(any("VA3" in blocker for blocker in report["host_aot_blockers"]))
        self.assertEqual({"VA2.1", "VA2.2", "VA2.3", "VA2.4"},
                         {identifier for identifier in report["id_index"]
                          if identifier.startswith("VA2.")})
        self.assertTrue(all(call.kwargs["required"] for call in archive_audit.call_args_list))

    def test_va2_eligibility_fails_closed_on_missing_budget_test_archive_or_source(self):
        base_names = set(audit.VA2_REQUIRED_TESTS)
        reports = []
        for profile, fingerprint, scope in (("release", "a" * 64, True),
                                              ("sanitize", "b" * 64, True)):
            names = set(base_names)
            if profile == "release":
                names.remove("VulkanRuntimeBudgetTest.CountsTemporaryWeightStagingBeforeAllocation")
            reports.append({
                "stage": f"synthetic-va2-{profile}", "profile": profile,
                "stage_id": "VA2", "source_fingerprint": fingerprint,
                "scope_passed": scope, "findings": [], "va0_findings": [],
                "ctest": {"test_names": sorted(names), "skipped": [], "failed": []},
                "families": {"va2_helper": {"status": "verified",
                                              "raw_artifacts": []}},
                "raw_artifacts": [],
            })
        gate_states = [
            {"stage": row["stage"], "stage_id": "VA2", "profile": row["profile"],
             "source_fingerprint": row["source_fingerprint"], "ctest": row["ctest"],
             "findings": [], "passed": True}
            for row in reports]
        with patch.object(audit, "audit_stage_gate", side_effect=gate_states), \
                patch.object(audit, "audit_stage_evidence", side_effect=reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "not_provided", "findings": []}):
            report = audit.build_report([Path("release"), Path("sanitize")],
                                        require_va2_helper=True)
        self.assertFalse(report["va2_helper_eligible"])
        self.assertIn("va2-source-fingerprint-mismatch",
                      {item["code"] for item in report["va2_helper_blockers"]})
        self.assertIn("va2-id-index-incomplete",
                      {item["code"] for item in report["va2_helper_blockers"]})
        self.assertIn("va2-helper-eligibility-required",
                      {item["code"] for item in report["blocking_findings"]})

    def _make_va3_reports(self, *, missing_test=None, fingerprints=("c" * 64, "c" * 64)):
        required = set(audit._required_tests("release", "VA3"))
        reports = []
        for index, profile in enumerate(("release", "sanitize")):
            names = set(required)
            if profile == "release" and missing_test:
                names.discard(missing_test)
            reports.append({
                "stage": f"synthetic-va3-{profile}", "profile": profile,
                "stage_id": "VA3", "source_fingerprint": fingerprints[index],
                "scope_passed": True, "findings": [], "va0_findings": [],
                "ctest": {"test_names": sorted(names), "skipped": [], "failed": []},
                "families": {"va3_lifecycle": {"status": "verified",
                                                  "path": "test/Vulkan/evidence/va3/lifecycle-evidence.json",
                                                  "raw_artifacts": []}},
                "raw_artifacts": [],
            })
        gates = [{"stage": row["stage"], "stage_id": "VA3", "profile": row["profile"],
                  "source_fingerprint": row["source_fingerprint"], "ctest": row["ctest"],
                  "findings": [], "passed": True}
                 for row in reports]
        return reports, gates

    def test_va3_same_source_release_and_san_lifecycle_pair_qualifies_but_not_aot(self):
        reports, gates = self._make_va3_reports()
        with patch.object(audit, "audit_stage_gate", side_effect=gates), \
                patch.object(audit, "audit_stage_evidence", side_effect=reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "verified", "findings": []}) as archive_audit:
            report = audit.build_report([Path("release"), Path("sanitize")],
                                        require_va3_lifecycle=True)
        self.assertTrue(report["va3_lifecycle_eligible"])
        self.assertTrue(report["va3_lifecycle_required"])
        self.assertEqual([], report["va3_lifecycle_blockers"])
        self.assertFalse(report["host_aot_release_eligible"])
        self.assertTrue(all(call.kwargs["required"] for call in archive_audit.call_args_list))
        self.assertEqual({"VA3.1", "VA3.2", "VA3.3", "VA3.4"},
                         {identifier for identifier in report["id_index"]
                          if identifier.startswith("VA3.")})

    def _make_va4_reports(self, *, unqualified=False, missing_test=None,
                          fingerprints=("e" * 64, "e" * 64)):
        required = set(audit._required_tests("release", "VA4"))
        reports = []
        for index, profile in enumerate(("release", "sanitize")):
            names = set(required)
            if profile == "release" and missing_test:
                names.discard(missing_test)
            family = {"status": "unqualified" if unqualified else "verified",
                      "path": "test/Vulkan/evidence/va4/fixedrun-evidence.json",
                      "candidate_dso_sha256": {fixture: str(index) * 64
                                                for fixture in audit.fixedrun_evidence.FIXTURES}}
            reports.append({
                "stage": f"synthetic-va4-{profile}", "profile": profile,
                "stage_id": "VA4", "source_fingerprint": fingerprints[index],
                "scope_passed": True, "findings": [], "va0_findings": [],
                "ctest": {"test_names": sorted(names), "skipped": [], "failed": []},
                "families": {"va4_fixedrun": family}, "raw_artifacts": [],
            })
        gates = [{"stage": row["stage"], "stage_id": "VA4", "profile": row["profile"],
                  "source_fingerprint": row["source_fingerprint"], "ctest": row["ctest"],
                  "findings": [], "passed": True}
                 for row in reports]
        return reports, gates

    def test_va4_real_stage_adapter_projects_evidence_into_composite_qualification(self):
        now = datetime.now(timezone.utc)
        fingerprint = "f" * 64
        required = sorted(audit._required_tests("release", "VA4") |
                          audit._va4_codegen_tests())
        stage_dirs, gates = [], []
        aggregates, aggregate_bytes = {}, {}
        for profile in ("release", "sanitize"):
            stage = self.root / f"adapter-{profile}"
            stage.mkdir()
            stage_dirs.append(stage)
            (stage / "CMakeCache.txt").write_text("", encoding="utf-8")
            evidence_path = stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json"
            aggregate = {
                "schema": "ncnn-vulkan-fixedrun-aggregate/v1",
                "stage_id": "VA4", "profile": profile,
                "source_fingerprint": fingerprint,
                "role": "fixed-run-candidate",
                "devices": {device: {"legacy_reference": {"qualified": True}}
                            for device in audit.fixedrun_evidence.DEVICES},
                "legacy_reference_qualified": True,
                "candidate_dso_sha256s": {
                    fixture: hashlib.sha256(fixture.encode()).hexdigest()
                    for fixture in audit.fixedrun_evidence.FIXTURES},
                "claim_scope": "private seven-small-graph fixed-run prototype; public product remains generic-plan-runtime",
                "host_aot_release_eligible": False,
            }
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
            aggregate_bytes[str(stage)] = evidence_path.read_bytes()
            aggregates[str(stage)] = (aggregate, [])
            log = stage / "ctest.log"
            log.write_text("CTest completed\n", encoding="utf-8")
            step = {"name": "ctest",
                    "started_at": (now - timedelta(seconds=30)).isoformat(),
                    "finished_at": (now + timedelta(seconds=30)).isoformat(),
                    "log": "ctest.log", "sha256": audit.sha256_file(log)}
            state = {"steps": [step]}
            (stage / "gate-state.json").write_text(json.dumps(state), encoding="utf-8")
            gates.append({"stage": str(stage), "stage_id": "VA4", "profile": profile,
                          "source_fingerprint": fingerprint, "findings": [],
                          "ctest": {"test_names": required, "skipped": [], "failed": []},
                          "state": state})

        gate_by_stage = {gate["stage"]: gate for gate in gates}
        empty = {}
        with patch.object(audit, "_audit_cpu_baseline", return_value=empty), \
                patch.object(audit, "_audit_offline", return_value=empty), \
                patch.object(audit, "_audit_v0", return_value=empty), \
                patch.object(audit, "_audit_v2", return_value=empty), \
                patch.object(audit, "_audit_v3", return_value=empty), \
                patch.object(audit, "_audit_product", return_value=empty), \
                patch.object(audit, "_audit_v5", return_value=empty), \
                patch.object(audit, "_audit_va0_baseline", return_value=empty), \
                patch.object(audit, "_audit_va1_stage", return_value=empty), \
                patch.object(audit, "_audit_va2_helper", return_value=empty), \
                patch.object(audit, "_audit_va3_lifecycle", return_value=empty), \
                patch.object(audit.fixedrun_evidence, "aggregate_stage_evidence",
                             side_effect=lambda stage, *args, **kwargs: aggregates[str(Path(stage))]) as aggregate_audit, \
                patch.object(audit, "audit_stage_gate",
                             side_effect=lambda stage, **kwargs: gate_by_stage[str(Path(stage))]), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "verified", "findings": []}) as archive_audit:
            report = audit.build_report(stage_dirs, require_va4_fixedrun=True)
            for stage in stage_dirs:
                (stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json").unlink()
            absent_report = audit.build_report(stage_dirs, require_va4_fixedrun=True)

            def omit_fixture_aggregate(stage, *args, **kwargs):
                aggregate, raw = aggregates[str(Path(stage))]
                malformed = dict(aggregate)
                malformed["candidate_dso_sha256s"] = dict(
                    aggregate["candidate_dso_sha256s"])
                malformed["candidate_dso_sha256s"].pop("transpose")
                output = Path(stage) / "test/Vulkan/evidence/va4/fixedrun-evidence.json"
                output.write_text(json.dumps(malformed, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
                return malformed, raw

            aggregate_audit.side_effect = omit_fixture_aggregate
            missing_fixture_report = audit.build_report(stage_dirs, require_va4_fixedrun=True)
            for stage in stage_dirs:
                (stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json").write_bytes(
                    aggregate_bytes[str(stage)])

            def reject_malformed_aggregate(*args, **kwargs):
                raise audit.fixedrun_evidence.EvidenceError("malformed raw VA4 evidence")

            aggregate_audit.side_effect = reject_malformed_aggregate
            malformed_report = audit.build_report(stage_dirs, require_va4_fixedrun=True)

        self.assertTrue(report["va4_fixedrun_eligible"],
                        (report["va4_fixedrun_blockers"],
                         report["va4_fixedrun_stages"],
                         [stage["findings"] for stage in report["stages"]],
                         {key: value["status"] for key, value in report["id_index"].items()
                          if key.startswith("VA4.")}))
        self.assertEqual([], report["va4_fixedrun_blockers"])
        self.assertFalse(report["host_aot_release_eligible"])
        self.assertFalse(absent_report["va4_fixedrun_eligible"])
        self.assertFalse(missing_fixture_report["va4_fixedrun_eligible"])
        self.assertFalse(malformed_report["va4_fixedrun_eligible"])
        self.assertTrue(all(call.kwargs["required"] for call in archive_audit.call_args_list))
        for row in report["va4_fixedrun_stages"]:
            self.assertEqual("verified", row["evidence"]["status"])
            self.assertTrue(row["evidence"]["legacy_reference_qualified"])
            self.assertEqual(fingerprint, row["evidence"]["source_fingerprint"])
            self.assertEqual(set(audit.fixedrun_evidence.FIXTURES),
                             set(row["evidence"]["candidate_dso_sha256"]))
            self.assertFalse(row["evidence"]["host_aot_release_eligible"])

    def test_va4_same_source_release_san_pair_qualifies_private_only(self):
        reports, gates = self._make_va4_reports()
        with patch.object(audit, "audit_stage_gate", side_effect=gates), \
                patch.object(audit, "audit_stage_evidence", side_effect=reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "verified", "findings": []}) as archive_audit:
            report = audit.build_report([Path("release"), Path("sanitize")],
                                        require_va4_fixedrun=True)
        self.assertTrue(report["va4_fixedrun_eligible"])
        self.assertTrue(report["va4_fixedrun_required"])
        self.assertEqual([], report["va4_fixedrun_blockers"])
        self.assertFalse(report["host_aot_release_eligible"])
        self.assertTrue(all(call.kwargs["required"] for call in archive_audit.call_args_list))
        self.assertEqual({"VA4.1", "VA4.2", "VA4.3", "VA4.4"},
                         {identifier for identifier in report["id_index"]
                          if identifier.startswith("VA4.")})
        self.assertTrue(all(report["id_index"][identifier]["status"] == "passed"
                            for identifier in ("VA4.1", "VA4.2", "VA4.3", "VA4.4")))
        self.assertIn("vulkan-fixedrun-capture-publish",
                      audit.VA4_STEP_TESTS["VA4.1"])

    def test_va4_fixedrun_requires_all_seven_fixtures_and_independent_legacy(self):
        required = set(audit._required_tests("release", "VA4"))
        device_test = next(iter(audit.VA4_DEVICE_TESTS))
        for reports, gates in (
                self._make_va4_reports(missing_test=device_test),
                self._make_va4_reports(
                    missing_test="vulkan-fixedrun-capture-publish"),
                self._make_va4_reports(unqualified=True),
                self._make_va4_reports(fingerprints=("a" * 64, "b" * 64))):
            with self.subTest(stage_states=[row["families"]["va4_fixedrun"]["status"]
                                            for row in reports]), \
                    patch.object(audit, "audit_stage_gate", side_effect=gates), \
                    patch.object(audit, "audit_stage_evidence", side_effect=reports), \
                    patch.object(audit, "audit_stage_archive", return_value={
                        "status": "verified", "findings": []}):
                report = audit.build_report([Path("release"), Path("sanitize")],
                                            require_va4_fixedrun=True)
            self.assertFalse(report["va4_fixedrun_eligible"])
            self.assertIn("va4-fixedrun-eligibility-required",
                          {item["code"] for item in report["blocking_findings"]})
            self.assertTrue(any(row["status"] == "blocked"
                                for key, row in report["id_index"].items()
                                if key.startswith("VA4.")))
        failed_reports, failed_gates = self._make_va4_reports()
        failed_reports[0]["scope_passed"] = False
        with patch.object(audit, "audit_stage_gate", side_effect=failed_gates), \
                patch.object(audit, "audit_stage_evidence", side_effect=failed_reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "verified", "findings": []}):
            failed_gate_report = audit.build_report(
                [Path("release"), Path("sanitize")], require_va4_fixedrun=True)
        self.assertFalse(failed_gate_report["va4_fixedrun_eligible"])

        archive_reports, archive_gates = self._make_va4_reports()
        with patch.object(audit, "audit_stage_gate", side_effect=archive_gates), \
                patch.object(audit, "audit_stage_evidence", side_effect=archive_reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "invalid", "findings": [{"code": "archive-integrity-invalid"}]}):
            bad_archive_report = audit.build_report(
                [Path("release"), Path("sanitize")], require_va4_fixedrun=True)
        self.assertFalse(bad_archive_report["va4_fixedrun_eligible"])

        self.assertIn("vulkan-fixedrun-capture-publish", required)
        self.assertIn("vulkan-fixedrun-legacy-reference-tests", required)
        legacy_coverage = audit._va0_migration_coverage({
            "profile": "release", "ctest": {"test_names": sorted(required)}})
        self.assertFalse(any(identifier.startswith("VA4.")
                             for identifier in legacy_coverage["current_ids"]))
        self.assertFalse({"vulkan-fixedrun-capture-publish",
                          "vulkan-fixedrun-fake-unit", *audit.VA4_DEVICE_TESTS} &
                         set(legacy_coverage["current_test_cases"]))

    def test_va3_id_catalog_matches_scope_and_codegen_test_coverage(self):
        codegen = audit._va3_codegen_tests()
        mapped_codegen = set().union(*(
            {name for name in tests if name.startswith("VulkanLifecycleCodegenTest.")}
            for tests in audit.VA3_STEP_TESTS.values()))
        self.assertEqual(codegen, mapped_codegen)
        self.assertEqual(
            {"vulkan-lifecycle-independence", "vulkan-lifecycle-evidence-tests"},
            {name for name in audit.VA3_STEP_TESTS["VA3.2"]
             if name.startswith("vulkan-lifecycle-")})
        self.assertNotIn("vulkan-lifecycle-fake-unit", audit.VA3_STEP_TESTS["VA3.2"])
        self.assertNotIn("vulkan-lifecycle-no-device", audit.VA3_STEP_TESTS["VA3.2"])
        self.assertTrue({"vulkan-lifecycle-fake-unit", *audit.VA3_DEVICE_TESTS,
                         "VulkanLifecycleCodegenTest.EmbeddedWeightsMatchVerifiedPayloadBytes"}
                        <= audit.VA3_STEP_TESTS["VA3.3"])
        self.assertTrue({"vulkan-lifecycle-no-device", "vulkan-lifecycle-fake-unit",
                         "vulkan-lifecycle-independence", *audit.VA3_DEVICE_TESTS}
                        <= audit.VA3_STEP_TESTS["VA3.4"])
        self.assertTrue(set(audit.VA3_REQUIRED_TESTS) | codegen <=
                        audit._required_tests("release", "VA3"))

    def test_va3_id_row_requires_each_mapped_test_to_pass(self):
        required = set(audit._required_tests("release", "VA3"))
        reports = [{
            "stage": f"va3-{profile}", "stage_id": "VA3", "profile": profile,
            "source_fingerprint": "d" * 64, "scope_passed": True,
            "ctest": {"test_names": sorted(required), "skipped": [], "failed": []},
            "families": {"va3_lifecycle": {"status": "verified",
                                             "path": "test/Vulkan/evidence/va3/lifecycle-evidence.json"}},
            "archive_integrity": {"status": "verified"}, "raw_artifacts": [],
        } for profile in ("release", "sanitize")]
        row = audit._va3_id_row("VA3.3", audit.ID_CATALOG["VA3.3"], reports)
        self.assertEqual("passed", row["status"])
        changed = [dict(stage, ctest=dict(stage["ctest"])) for stage in reports]
        skipped_test = next(iter(audit.ID_CATALOG["VA3.3"]["tests"]))
        changed[0]["ctest"]["skipped"] = [skipped_test]
        blocked = audit._va3_id_row("VA3.3", audit.ID_CATALOG["VA3.3"], changed)
        self.assertEqual("blocked", blocked["status"])
        release = next(item for item in blocked["profiles"] if item["profile"] == "release")
        self.assertEqual([skipped_test], release["tests_not_passed"])
        self.assertNotIn(skipped_test, release["tests_executed"])

    def test_va3_missing_declared_test_or_archive_blocks_required_pair(self):
        missing_case = sorted(audit._va3_codegen_tests())[0]
        reports, gates = self._make_va3_reports(missing_test=missing_case)
        with patch.object(audit, "audit_stage_gate", side_effect=gates), \
                patch.object(audit, "audit_stage_evidence", side_effect=reports), \
                patch.object(audit, "audit_stage_archive", return_value={
                    "status": "not_provided", "findings": []}):
            report = audit.build_report([Path("release"), Path("sanitize")],
                                        require_va3_lifecycle=True)
        self.assertFalse(report["va3_lifecycle_eligible"])
        self.assertIn("va3-id-index-incomplete",
                      {item["code"] for item in report["va3_lifecycle_blockers"]})
        self.assertIn("va3-lifecycle-eligibility-required",
                      {item["code"] for item in report["blocking_findings"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
