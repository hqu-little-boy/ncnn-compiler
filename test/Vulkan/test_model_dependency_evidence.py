#!/usr/bin/env python3
"""Unit tests for test-only Vulkan model dependency evidence capture."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("model_dependency_evidence.py")
spec = importlib.util.spec_from_file_location("model_dependency_evidence", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SyntheticModelLink:
    """Make a real ELF link with two archives and a private DSO chain."""

    def __init__(self, root: Path):
        self.root = root
        self.stage = root / "staging"
        self.capture = root / "capture"
        self.libs = root / "private-libs"
        self.stage.mkdir()
        self.capture.mkdir()
        self.libs.mkdir()
        self.compiler = shutil.which("c++") or shutil.which("g++")
        self.cc = shutil.which("cc") or shutil.which("gcc")
        self.ar = shutil.which("ar")
        self.readelf = shutil.which("readelf")
        self.ldd = shutil.which("ldd")
        if not all((self.compiler, self.cc, self.ar, self.readelf, self.ldd)):
            raise unittest.SkipTest("native compiler, binutils, and ldd are required")
        self._build_inputs()

    def run(self, command):
        result = subprocess.run(list(map(str, command)), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        if result.returncode:
            raise AssertionError(
                f"command failed ({result.returncode}): {command}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def compile_c(self, source: Path, output: Path, *flags):
        self.run([self.cc, "-fPIC", "-c", source, "-o", output, *flags])

    def _build_inputs(self):
        nested_c = self.root / "nested.c"
        nested_c.write_text("int nested_value(void) { return 3; }\n")
        nested_so = self.libs / "libnested.so"
        self.run([self.cc, "-fPIC", "-shared", nested_c, "-o", nested_so])

        private_c = self.root / "private.c"
        private_c.write_text(
            "extern int nested_value(void);\n"
            "int private_value(void) { return nested_value() + 4; }\n"
        )
        self.run([self.cc, "-fPIC", "-shared", private_c, "-L", self.libs,
                  "-lnested", "-o", self.libs / "libprivate.so"])

        archive_sources = [
            ("runtime.c", "ncnnVulkanRuntime.cpp.o",
             "int runtime_value(void) { return 1; }\n"),
            ("plan.c", "VulkanExecutionPlan.cpp.o",
             "int plan_value(void) { return 2; }\n"),
            ("payload.c", "NCNNVulkanPayloadHash.cpp.o",
             "int payload_value(void) { return 3; }\n"),
            ("llvm_fake.c", "llvm_fake.cpp.o",
             "int llvm_value(void) { return 5; }\n"),
        ]
        self.archives = []
        for archive_name, (source_name, object_name, text) in zip(
                ("libNCNNVulkanRuntime.a", "libNCNNVulkanPlanSupport.a",
                 "libNCNNVulkanPayload.a", "libLLVM.a"),
                archive_sources):
            source = self.root / source_name
            obj = self.root / object_name
            archive = self.root / archive_name
            source.write_text(text)
            self.compile_c(source, obj)
            self.run([self.ar, "rcs", archive, obj])
            self.archives.append(archive)

        (self.stage / "vulkan_model_wrapper.cpp").write_text(
            "extern \"C\" {\n"
            "int runtime_value(void);\n"
            "int plan_value(void);\n"
            "int payload_value(void);\n"
            "int llvm_value(void);\n"
            "int private_value(void);\n"
            "int tiny_relu_vk_create(void) {\n"
            "  return runtime_value() + plan_value() + payload_value() + llvm_value() + private_value();\n"
            "}\n"
            "}\n"
        )
        (self.stage / "tiny_relu_vulkan.h").write_text("/* generated test header */\n")
        (self.stage / "tiny_relu_vulkan.plan.json").write_text("{}\n")
        (self.stage / "tiny_relu_vulkan.json").write_text("{}\n")
        (self.stage / "model.vulkan-weights.json").write_text("[]\n")
        (self.stage / "kernel-0.spv").write_bytes(b"\x03\x02\x23\x07\x00\x00\x01\x00")
        self.version_script = self.stage / "vulkan-exports.map"
        self.version_script.write_text(
            "{ global: tiny_relu_vk_create; local: *; };\n"
        )
        self.library = self.root / "libtiny_relu_vulkan.so"
        self.launcher = module.make_host_cxx_launcher(self.compiler, self.capture)

    def link(self, *, relative_source=False):
        source = "vulkan_model_wrapper.cpp" if relative_source else str(
            self.stage / "vulkan_model_wrapper.cpp")
        command = [
            str(self.launcher), "-std=c++23", "-shared", "-fPIC", "-O2", "-g0",
        ]
        if relative_source:
            command.append("-working-directory=" + str(self.stage))
        command.extend([
            "-I", str(self.stage), source,
            *map(str, self.archives), "-L" + str(self.libs), "-lprivate",
            "-Wl,--gc-sections", "-Wl,--strip-all", "-Wl,-z,defs",
            "-Wl,--no-undefined", "-Wl,--exclude-libs,ALL",
            "-Wl,--version-script=" + str(self.version_script),
            "-Wl,--build-id=none", "-o", str(self.library),
        ])
        # Confirm probing flags are forwarded and not mistaken for a model link.
        probe = subprocess.run([str(self.launcher), "-dumpmachine"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if probe.returncode != 0 or not probe.stdout.strip():
            raise AssertionError(f"compiler probe failed: {probe.stderr}")
        self.run(command)
        records = list(self.capture.glob("model-link-*.json"))
        if len(records) != 1:
            raise AssertionError(f"expected exactly one recorded model link, got {records}")
        self.record_path = records[0]
        self.record = json.loads(self.record_path.read_text())

    def collect(self):
        old = os.environ.get("LD_LIBRARY_PATH")
        paths = str(self.libs)
        os.environ["LD_LIBRARY_PATH"] = paths if not old else paths + os.pathsep + old
        try:
            return module.collect_model_dependencies(
                self.library, self.capture,
                tool_paths={"readelf": self.readelf, "ldd": self.ldd, "ar": self.ar},
                llvm_mode="static",
            )
        finally:
            if old is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = old


class ModelDependencyEvidenceTest(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="va0-evidence-")
        self.addCleanup(temporary.cleanup)
        link = SyntheticModelLink(Path(temporary.name))
        link.link()
        return link

    def test_readelf_parsers_cover_gnu_and_llvm_human_readable_formats(self):
        llvm_dynamic = """DynamicSection [ (4 entries)
  Tag                Type         Name/Value
  0x0000000000000001 NEEDED       Shared library: [libclang_rt.asan-x86_64.so]
  0x000000000000000e SONAME       Library soname: [libfixture.so.1]
  0x000000000000001d RUNPATH      Library runpath: [/opt/fixture]
]
"""
        gnu_dynamic = """Dynamic section at offset 0x100 contains 4 entries:
 0x0000000000000001 (NEEDED)             Shared library: [libclang_rt.asan-x86_64.so]
 0x000000000000000e (SONAME)             Library soname: [libfixture.so.1]
 0x000000000000001d (RUNPATH)            Library runpath: [/opt/fixture]
"""
        expected_dynamic = {"needed": ["libclang_rt.asan-x86_64.so"],
                            "soname": "libfixture.so.1",
                            "rpath_runpath": ["/opt/fixture"]}
        self.assertEqual(expected_dynamic, module.parse_readelf_dynamic(llvm_dynamic))
        self.assertEqual(expected_dynamic, module.parse_readelf_dynamic(gnu_dynamic))
        llvm_symbols = """Symbols [
  Symbol {
    Name: puts (8)
    Binding: Global (0x1)
    Section: Undefined (0x0)
  }
  Symbol {
    Name: _Zfixture_export (9)
    Binding: Global (0x1)
    Section: .text (0x4)
  }
  Symbol {
    Name: _Zfixture_weak (10)
    Binding: Weak (0x2)
    Section: .text (0x4)
  }
]
"""
        gnu_symbols = """Symbol table '.dynsym' contains 3 entries:
  1: 0000 0 FUNC GLOBAL DEFAULT UND puts
  2: 0010 12 FUNC GLOBAL DEFAULT 4 _Zfixture_export
  3: 0020 12 FUNC WEAK DEFAULT 4 _Zfixture_weak
"""
        expected_symbols = (["puts"], ["_Zfixture_export", "_Zfixture_weak"])
        self.assertEqual(expected_symbols, tuple(map(list, module.parse_readelf_symbols(llvm_symbols))))
        self.assertEqual(expected_symbols, tuple(map(list, module.parse_readelf_symbols(gnu_symbols))))

    def test_real_link_capture_closure_and_static_llvm_without_soname(self):
        link = self.fixture()
        evidence = link.collect()
        self.assertTrue(link.record["staged_snapshot"]["files"])
        self.assertEqual([], link.record["staged_snapshot"]["missing_required"])
        self.assertGreater(link.record["staged_snapshot"]["spirv_count"], 0)
        self.assertEqual(link.record["original_argv"],
                         link.record["effective_argv"][:-2])
        self.assertIn("-Wl,--strip-all", link.record["effective_argv"])
        self.assertIn("-Wl,-z,defs", link.record["effective_argv"])
        self.assertTrue(any("-Map=" in arg and "--cref" in arg
                            for arg in link.record["effective_argv"]))
        self.assertIn("-save-temps=obj", link.record["effective_argv"])
        self.assertTrue(link.record["wrapper_objects"], "save-temps wrapper object was not captured")
        self.assertTrue(all(item["unstripped"] for item in evidence["wrapper_objects"]))
        archives = {Path(item["path"]).name: item for item in evidence["archive_inputs"]}
        llvm_archive = archives["libLLVM.a"]
        self.assertEqual("compiler", llvm_archive["classification"])
        self.assertEqual("model", archives["libNCNNVulkanRuntime.a"]["classification"])
        self.assertEqual("project-executor", archives["libNCNNVulkanRuntime.a"]["role"])
        self.assertEqual("plan-parser", archives["libNCNNVulkanPlanSupport.a"]["role"])
        self.assertEqual("payload-hash", archives["libNCNNVulkanPayload.a"]["role"])
        self.assertTrue(llvm_archive["selected_members_from_map"])
        self.assertTrue(llvm_archive["selected_member_hashes"])
        self.assertTrue(llvm_archive["ar_t"]["sha256"])
        self.assertTrue(link.record["linked_output"]["sha256"])
        # Static archives have no SONAME; don't infer no LLVM from the DSO table.
        self.assertIsNone(evidence["model_elf"]["soname"])
        self.assertIn("llvm_fake.cpp.o", evidence["llvm"]["static_archive_members"])
        self.assertEqual("measured_presence", evidence["llvm"]["status"])
        self.assertEqual("presence_observed", evidence["llvm"]["absence_claim"])
        self.assertEqual([], evidence["llvm"]["shared_objects"])
        self.assertIn("static_llvm_archive_members_selected",
                      evidence["llvm"]["legacy_blockers"])
        self.assertEqual("reported_only_legacy_blocker_not_rejected",
                         evidence["llvm"]["policy"])
        names = {Path(item["path"]).name for item in evidence["runtime_closure"]}
        self.assertIn("libprivate.so", names)
        self.assertIn("libnested.so", names)
        private = next(item for item in evidence["runtime_closure"]
                       if Path(item["path"]).name == "libprivate.so")
        self.assertEqual("model", private["classification"])
        self.assertEqual("private-transitive", private["role"])
        self.assertTrue(private["direct_to_model"])
        self.assertFalse(next(item for item in evidence["runtime_closure"]
                              if Path(item["path"]).name == "libnested.so")["direct_to_model"])
        self.assertEqual(("platform", "vulkan-loader"),
                         module._dependency_info("/usr/lib/libvulkan.so.1"))
        self.assertTrue(module.validate_dependency_evidence(
            link.capture / "model-dependency-evidence.json", link.capture))

    def test_relative_ldd_dependency_path_uses_recorded_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            working_directory = Path(temporary)
            nested = working_directory / "lib/libLLVM.so"
            nested.parent.mkdir()
            nested.write_bytes(b"relative ldd path fixture")
            output = "libLLVM.so => lib/libLLVM.so (0x1234)\n"
            expected = str(nested.resolve())
            self.assertEqual(([expected], []), module._parse_ldd(output, working_directory))
            self.assertEqual({"libLLVM.so": expected},
                             module._parse_ldd_named_paths(output, working_directory))

    def test_direct_needed_entry_cannot_be_missing_from_runtime_closure(self):
        link = self.fixture()
        evidence = link.collect()
        needed = set(evidence["model_elf"]["needed"])
        missing_name = next(iter(needed))
        evidence["runtime_closure"] = [
            item for item in evidence["runtime_closure"]
            if missing_name not in {Path(item["path"]).name, item.get("soname")}
        ]
        evidence_path = link.capture / "model-dependency-evidence.json"
        evidence_path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(module.EvidenceError, "runtime closure"):
            module.validate_dependency_evidence(evidence_path, link.capture)

    def test_empty_ld_library_path_component_maps_to_current_directory(self):
        target = str(Path.cwd() / "model.so")
        self.assertEqual([str(Path.cwd())], module._capture_runtime_search_paths(
            {"LD_LIBRARY_PATH": ""}, target))
        self.assertEqual(sorted([str(Path.cwd()), str(Path("/tmp").resolve())]),
                         module._capture_runtime_search_paths(
                             {"LD_LIBRARY_PATH": os.pathsep + "/tmp"}, target))

    def test_relative_runpath_alias_validation_uses_recorded_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cwd = root / "captured-cwd"
            aliases = cwd / "lib"
            aliases.mkdir(parents=True)
            actual = root / "libactual.so"
            actual.write_bytes(b"synthetic shared object")
            for name in ("libalias-a.so", "libalias-b.so"):
                (aliases / name).symlink_to(actual)
            model = root / "model.so"
            model.write_bytes(b"synthetic model")
            dynamic = root / "dynamic.txt"
            dynamic.write_text(
                "0x1 (NEEDED) Shared library: [libalias-a.so]\n"
                "0x1 (NEEDED) Shared library: [libalias-b.so]\n"
                "0x1 (RUNPATH) Library runpath: [lib]\n")
            ldd = root / "ldd.txt"
            ldd.write_text(f"libalias-a.so => lib/libalias-a.so (0x1)\n")
            model_elf = {"needed": ["libalias-a.so", "libalias-b.so"],
                         "rpath_runpath": ["lib"],
                         "readelf_dynamic": {"path": dynamic.name,
                                             "sha256": module.sha256(dynamic)}}
            closure = [{"path": str(actual), "soname": None,
                        "direct_needed_names": model_elf["needed"], "direct_to_model": True}]
            module._validate_runtime_closure(
                root, {"path": model.name}, model_elf, closure,
                {"command": ["ldd", "-r", str(model)], "path": ldd.name,
                 "working_directory": str(cwd), "search_paths": []})
            self.assertEqual([str(cwd)], module._dependency_search_paths(
                {"rpath_runpath": [""]}, str(model), [], working_directory=cwd))

    def test_unexpanded_lib_and_platform_tokens_are_rejected(self):
        target = str(Path.cwd() / "model.so")
        with self.assertRaisesRegex(module.EvidenceError, "unsupported"):
            module._capture_runtime_search_paths(
                {"LD_LIBRARY_PATH": "$LIB/extra"}, target)
        with self.assertRaisesRegex(module.EvidenceError, "unsupported"):
            module._dependency_search_paths(
                {"rpath_runpath": ["$ORIGIN/$PLATFORM"]}, target, [])

    def test_symlinked_direct_dependency_without_soname_keeps_ldd_alias(self):
        link = self.fixture()
        alias = link.libs / "libprivate.so"
        actual = link.libs / "libprivate-real.so"
        alias.rename(actual)
        alias.symlink_to(actual.name)
        second_alias_dir = link.libs / "alias-dir"
        second_alias_dir.mkdir()
        second_alias = second_alias_dir / "libprivate-alias.so"
        second_alias.symlink_to(actual)
        evidence = link.collect()
        private = next(item for item in evidence["runtime_closure"]
                       if item["path"] == str(actual.resolve()))
        self.assertIsNone(private["soname"])
        self.assertTrue(private["direct_to_model"])
        ldd_names = module._parse_ldd_named_paths(
            f"libprivate.so => {actual.resolve()} (0x1)\n")
        search_dirs = module._dependency_search_paths(
            {"rpath_runpath": ["$ORIGIN/alias-dir"]},
            str(link.libs / "model.so"), [])
        self.assertEqual([str(second_alias_dir.resolve())], search_dirs)
        self.assertEqual(str(actual.resolve()), module._resolved_needed_path(
            "libprivate-alias.so", evidence["runtime_closure"],
            {str(actual.resolve())}, ldd_names, search_dirs))
        self.assertEqual(str(actual.resolve()), module._resolved_needed_path(
            str(second_alias), evidence["runtime_closure"],
            {str(actual.resolve())}, ldd_names))

    def test_clang_relative_wrapper_uses_working_directory(self):
        clang = shutil.which("clang") or "/usr/lib/llvm-21/bin/clang"
        if not Path(clang).is_file():
            self.skipTest("Clang is unavailable")
        temporary = tempfile.TemporaryDirectory(prefix="va0-clang-working-dir-")
        self.addCleanup(temporary.cleanup)
        link = SyntheticModelLink(Path(temporary.name))
        link.launcher.unlink()
        link.compiler = clang
        link.launcher = module.make_host_cxx_launcher(clang, link.capture)
        link.link(relative_source=True)
        self.assertNotEqual(link.record["cwd"], link.record["logical_compile_dir"])
        self.assertEqual("staged/vulkan_model_wrapper.cpp",
                         link.record["wrapper_source"])
        self.assertIn("vulkan_model_wrapper.cpp", link.record["original_argv"])
        evidence = link.collect()
        self.assertEqual(evidence["linked_output"]["sha256"],
                         evidence["model_library"]["sha256"])

    def test_missing_map_fails_closed(self):
        link = self.fixture()
        (link.capture / link.record["map_path"]).unlink()
        with self.assertRaisesRegex(module.EvidenceError, "map"):
            link.collect()

    def test_missing_archive_member_evidence_fails_closed(self):
        link = self.fixture()
        snapshot = next(item for item in link.record["archive_snapshots"]
                        if Path(item["path"]) == link.archives[1])
        link.archives[1].unlink()
        (link.capture / snapshot["ar_t"]["path"]).unlink()
        with self.assertRaisesRegex(module.EvidenceError, "ar t"):
            link.collect()

    def test_removed_archive_uses_verified_link_time_member_listing(self):
        link = self.fixture()
        link.archives[1].unlink()
        evidence = link.collect()
        archive = next(item for item in evidence["archive_inputs"]
                       if Path(item["path"]) == link.archives[1])
        self.assertEqual("missing_at_collection_link_time_ar_t_verified",
                         archive["source_status"])
        self.assertTrue(archive["selected_member_origins"])

    def test_link_map_member_set_must_match_archive_inventory_exactly(self):
        link = self.fixture()
        map_path = link.capture / link.record["map_path"]
        map_path.write_text(map_path.read_text() + "\n/tmp/libFakeLLVM.a(fake.o)\n")
        with self.assertRaisesRegex(module.EvidenceError, "inventory mismatch"):
            link.collect()

    def test_version_script_is_not_accepted_as_actual_link_map(self):
        link = self.fixture()
        (link.capture / link.record["map_path"]).write_text(link.version_script.read_text())
        with self.assertRaisesRegex(module.EvidenceError, "--cref linker map"):
            link.collect()

    def test_unresolved_transitive_symbol_fails_closed(self):
        link = self.fixture()
        broken = link.root / "broken-private.c"
        broken.write_text(
            "extern int missing_private_symbol(void);\n"
            "int private_value(void) { return missing_private_symbol(); }\n"
        )
        link.run([link.cc, "-fPIC", "-shared", broken, "-o",
                  link.libs / "libprivate.so"])
        with self.assertRaisesRegex(module.EvidenceError, "unresolved"):
            link.collect()

    def test_versioned_undefined_symbol_uses_symbol_name_column(self):
        cc = shutil.which("cc") or shutil.which("gcc")
        readelf = shutil.which("readelf")
        if not cc or not readelf:
            self.skipTest("C compiler and readelf are required")
        temporary = tempfile.TemporaryDirectory(prefix="va0-versioned-symbol-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "provider.c").write_text("int missing(void) { return 1; }\n")
        (root / "provider.map").write_text("VER_1 { global: missing; };\n")
        subprocess.run([cc, "-shared", "-fPIC", str(root / "provider.c"),
                        "-Wl,--version-script=" + str(root / "provider.map"),
                        "-Wl,-soname,libprovider.so", "-o", str(root / "libprovider.so")],
                       check=True, capture_output=True, text=True)
        (root / "consumer.c").write_text(
            '__asm__(".symver missing_v1,missing@VER_1");\n'
            "extern int missing_v1(void);\n"
            "int call_missing(void) { return missing_v1(); }\n"
        )
        subprocess.run([cc, "-shared", "-fPIC", str(root / "consumer.c"),
                        "-L", str(root), "-lprovider", "-o", str(root / "libconsumer.so")],
                       check=True, capture_output=True, text=True)
        summary = module._elf_summary(readelf, root / "libconsumer.so",
                                      root / "raw", "versioned")
        self.assertIn("missing@VER_1", summary["undefined_symbols"])
        self.assertNotIn("(2)", summary["undefined_symbols"])

    def test_response_file_and_empty_closure_counterexamples(self):
        link = self.fixture()
        link.record["effective_argv"].append("@unexpanded-link.rsp")
        link.record_path.write_text(json.dumps(link.record))
        with self.assertRaisesRegex(module.EvidenceError, "response-file"):
            link.collect()

        # Build clean evidence, then demonstrate that a syntactically present
        # but empty closure cannot pass validation.
        link = self.fixture()
        link.collect()
        evidence_path = link.capture / "model-dependency-evidence.json"
        data = json.loads(evidence_path.read_text())
        data["runtime_closure"] = []
        evidence_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(module.EvidenceError, "empty"):
            module.validate_dependency_evidence(evidence_path, link.capture)

    def test_staged_ncnn_compile_vulkan_product_smoke(self):
        compiler = Path("/tmp/ncnn-compiler-stage-v6-release-tfwy9tnd/tools/ncnn-compile")
        param = Path("/mnt/ncnn-compiler/compiler/test/Vulkan/relu.param")
        clang = shutil.which("clang") or "/usr/lib/llvm-21/bin/clang"
        tools = {"readelf": shutil.which("readelf"), "ldd": shutil.which("ldd"),
                 "ar": shutil.which("ar")}
        if not compiler.is_file() or not param.is_file() or not Path(clang).is_file():
            self.skipTest("staged compiler, ReLU fixture, or Clang is unavailable")
        if not all(tools.values()):
            self.skipTest("readelf, ldd, and ar are required for product evidence")
        temporary = tempfile.TemporaryDirectory(prefix="va0-ncnn-product-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        weights = root / "empty.bin"
        weights.write_bytes(b"")
        output = root / "out"
        output.mkdir()
        capture = root / "evidence"
        launcher = module.make_host_cxx_launcher(clang, capture)
        command = [str(compiler), str(param), "--bin=" + str(weights),
                   "--output-dir=" + str(output), "--model-name=va0_relu",
                   "--precision=f32", "--backend=vulkan",
                   "--vulkan-host-cxx=" + str(launcher)]
        result = subprocess.run(command, cwd=root, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        self.assertEqual(0, result.returncode,
                         f"ncnn-compile failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertNotIn("--verify-device", command)
        self.assertNotIn("--verify-execution", command)
        library = output / "libva0_relu_vulkan.so"
        self.assertTrue(library.is_file())
        record_path = next(capture.glob("model-link-*.json"))
        record = json.loads(record_path.read_text())
        self.assertFalse(Path(record["output"]).exists(), "staging output should be removed after CLI exit")
        self.assertTrue((capture / record["linked_output"]["path"]).is_file())
        evidence = module.collect_model_dependencies(
            library, capture, tool_paths=tools, llvm_mode="not_measured")
        self.assertEqual(evidence["model_library"]["sha256"],
                         evidence["linked_output"]["sha256"])
        self.assertEqual("measured_presence", evidence["llvm"]["status"])
        self.assertEqual("presence_observed", evidence["llvm"]["absence_claim"])
        loader = next(item for item in evidence["runtime_closure"]
                      if Path(item["path"]).name.startswith("libvulkan.so"))
        self.assertEqual(("platform", "vulkan-loader"),
                         (loader["classification"], loader["role"]))
        self.assertTrue(module.validate_dependency_evidence(
            capture / "model-dependency-evidence.json", capture))

    def test_cpu_elf_closure_needs_no_vulkan_link_record(self):
        cc = shutil.which("cc") or shutil.which("gcc")
        tools = {"readelf": shutil.which("readelf"), "ldd": shutil.which("ldd")}
        if not cc or not all(tools.values()):
            self.skipTest("C compiler, readelf, and ldd are required")
        temporary = tempfile.TemporaryDirectory(prefix="va0-cpu-elf-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "cpu.c"
        source.write_text("extern int puts(const char *);\n"
                          "int cpu_model(void) { return puts(\"cpu\"); }\n")
        library = root / "libcpu_model.so"
        subprocess.run([cc, "-shared", "-fPIC", str(source), "-o", str(library)],
                       check=True, capture_output=True, text=True)
        evidence = module.collect_elf_dependency_closure(
            library, root / "cpu-evidence", tool_paths=tools)
        self.assertEqual("ncnn.elf_dependency_closure", evidence["kind"])
        self.assertNotIn("link_record", evidence)
        self.assertEqual("not_measured", evidence["llvm"]["static_archive_status"])
        self.assertEqual("not_measured", evidence["llvm"]["status"])
        self.assertEqual("not_measured", evidence["llvm"]["absence_claim"])
        self.assertTrue(module.validate_elf_dependency_closure(
            root / "cpu-evidence/elf-dependency-closure.json", root / "cpu-evidence"))
        self.assertFalse(list((root / "cpu-evidence").glob("model-link-*.json")))

    def test_loader_phase_comparison_is_distinct_and_not_measured_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix="va0-loader-") as temporary:
            root = Path(temporary)
            self.assertEqual("not_measured",
                             module.compare_driver_module_snapshots(root)["status"])
            base = root / "libbase.so"
            loader = root / "libvulkan.so.1"
            icd = root / "libvulkan_lvp.so"
            late = root / "liblate.so"
            driver_llvm = root / "libLLVM.so"
            for path in (base, loader, icd, late, driver_llvm):
                path.write_bytes(b"synthetic mapped file")
            manifest = [str(icd)]
            before = f"1000-2000 r-xp 00000000 08:01 123 {base}\n"
            after_dlopen = before + f"2000-3000 r-xp 00000000 08:01 124 {loader}\n"
            after_create = (after_dlopen + f"3000-4000 r-xp 00000000 08:01 125 {icd}\n"
                            f"4000-5000 r-xp 00000000 08:01 126 {late}\n"
                            f"5000-6000 r-xp 00000000 08:01 127 {driver_llvm}\n")
            for phase, maps in (("before-model", before), ("after-dlopen", after_dlopen),
                                ("after-create", after_create)):
                module.record_driver_module_snapshot(
                    root, phase, maps, driver_manifest_paths=manifest)
            result = module.compare_driver_module_snapshots(root)
            self.assertEqual("measured", result["status"])
            self.assertEqual("measured", result["driver_origin_status"])
            loader_delta = next(item for item in result["after_dlopen_added_modules"]
                                if item["path"] == str(loader))
            self.assertEqual(("platform", "vulkan-loader"),
                             (loader_delta["classification"], loader_delta["role"]))
            icd_delta = next(item for item in result["driver_icd_modules_after_create"]
                             if item["path"] == str(icd))
            self.assertEqual("explicit-manifest-icd-after-create", icd_delta["role"])
            self.assertTrue(any(item["path"] == str(late) and
                                item["role"] == "private-transitive"
                                for item in result["after_create_added_modules"]))
            driver_llvm_delta = next(item for item in
                                     result["driver_process_compiler_modules_after_create_delta"]
                                     if item["path"] == str(driver_llvm))
            self.assertEqual("driver-after-create-delta", driver_llvm_delta["origin_phase"])
            self.assertNotIn(driver_llvm_delta, result["driver_icd_modules_after_create"])
            self.assertEqual("not_measured", result["model_llvm_status"])
            self.assertTrue(result["driver_origin_does_not_exempt_model_llvm"])
            self.assertEqual("not_made", result["llvm_absence_claim"])
            comparison_path = root / "driver-module-comparison.json"
            prior = comparison_path.read_bytes()
            result_path_mtime = comparison_path.stat().st_mtime_ns
            checked = module.compare_driver_module_snapshots(root, persist=False)
            self.assertEqual(result, checked)
            self.assertEqual(prior, comparison_path.read_bytes())
            self.assertEqual(result_path_mtime, comparison_path.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
