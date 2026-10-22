#!/usr/bin/env python3
"""Unit tests for the local, fresh legacy-reference builder (no SDK build)."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import build_fixedrun_legacy_reference as legacy


class FixedRunLegacyReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def arguments(self, *extra):
        return legacy.parse_args(["--source-dir", str(self.root / "original"),
                                  "--build-root", str(self.root), "--jobs", "2",
                                  *extra])

    def test_argument_validation_requires_positive_jobs_and_paired_sdk(self):
        for arguments in (["--jobs", "0"], ["--jobs", "-1"],
                          ["--llvm-dir", "/llvm"], ["--mlir-dir", "/mlir"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    legacy.parse_args(arguments)
                self.assertEqual(2, failure.exception.code)
        args = self.arguments("--llvm-dir", "/llvm", "--mlir-dir", "/mlir")
        self.assertEqual(2, args.jobs)
        self.assertEqual(Path("/llvm"), args.llvm_dir)

    def test_build_root_cannot_modify_the_source_repository(self):
        original = self.root / "original"
        original.mkdir()
        for build_root in (original, original / "nested"):
            with self.subTest(build_root=build_root), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    self.arguments("--build-root", str(build_root))
                self.assertEqual(2, failure.exception.code)
        self.assertEqual([], list(original.iterdir()))

    def test_published_reference_cannot_be_replaced(self):
        path = self.root / "legacy-reference.json"
        legacy.write_json(path, {"status": "first"})
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            legacy.write_json(path, {"status": "changed"})
        self.assertEqual(before, path.read_bytes())
        self.assertEqual([path], list(self.root.iterdir()))

    def test_atomic_publish_does_not_overwrite_a_racing_file(self):
        path = self.root / "legacy-reference.json"

        def raced_link(source, destination):
            self.assertTrue(Path(source).is_file())
            Path(destination).write_text("foreign", encoding="utf-8")
            raise FileExistsError("concurrent creator")

        with mock.patch.object(legacy.os, "link", side_effect=raced_link):
            with self.assertRaisesRegex(ValueError, "refusing to replace"):
                legacy.write_json(path, {"status": "not published"})
        self.assertEqual("foreign", path.read_text())
        self.assertEqual([path], list(self.root.iterdir()))

    def test_progress_state_can_be_updated_explicitly(self):
        path = self.root / "state.json"
        legacy.write_json(path, {"steps": []}, replace=True)
        legacy.write_json(path, {"steps": ["configure"]}, replace=True)
        self.assertEqual({"steps": ["configure"]}, json.loads(path.read_text()))

    def test_inventory_preserves_empty_gitlinks_and_symlink_bytes(self):
        source = self.root / "source.cpp"
        source.write_bytes(b"source bytes\n")
        (self.root / "alias.hpp").symlink_to("source.cpp")
        (self.root / "empty").mkdir()
        index = (b"100644 " + b"1" * 40 + b" 0\tsource.cpp\0" +
                 b"120000 " + b"2" * 40 + b" 0\talias.hpp\0" +
                 b"160000 " + b"3" * 40 + b" 0\tempty\0")
        with mock.patch.object(legacy, "git_output", return_value=index) as git:
            records = legacy.tracked_inventory(self.root)
        git.assert_called_once_with(self.root, "ls-files", "--stage", "-z")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         records[0]["sha256"])
        self.assertEqual({"path": "alias.hpp", "kind": "symlink",
                          "target": "source.cpp"}, records[1])
        self.assertEqual({"path": "empty", "kind": "gitlink", "oid": "3" * 40},
                         records[2])

    def test_inventory_rejects_unmerged_absolute_and_parent_paths(self):
        for path, stage in (("../outside", "0"), ("/absolute", "0"),
                            ("source.cpp", "1")):
            index = f"100644 {'1' * 40} {stage}\t{path}\0".encode()
            with self.subTest(path=path, stage=stage):
                with mock.patch.object(legacy, "git_output", return_value=index):
                    with self.assertRaisesRegex(ValueError, "invalid or unmerged"):
                        legacy.tracked_inventory(self.root)

    def test_populated_gitlink_must_be_its_own_repository(self):
        child = self.root / "child"
        child.mkdir()
        (child / "file.cpp").write_bytes(b"not a repository")
        index = [("160000", "a" * 40, "child")]
        with mock.patch.object(legacy, "git_entries", return_value=index):
            with mock.patch.object(legacy, "git_output",
                                   return_value=str(self.root).encode()):
                with self.assertRaisesRegex(ValueError, "not its own repository"):
                    legacy.tracked_inventory(self.root)

    def test_source_archive_contains_only_inventory_entries(self):
        source = self.root / "source"
        source.mkdir()
        (source / "kept.cpp").write_bytes(b"tracked")
        (source / "extra.txt").write_bytes(b"untracked")
        (source / "alias").symlink_to("kept.cpp")
        records = [{"path": "kept.cpp", "kind": "file"},
                   {"path": "alias", "kind": "symlink"},
                   {"path": "empty-submodule", "kind": "gitlink"}]
        path = self.root / "source.tar.gz"
        legacy.archive_source(source, records, path)
        with tarfile.open(path, "r:gz") as archive:
            self.assertEqual(["kept.cpp", "alias"], archive.getnames())
            self.assertEqual(b"tracked", archive.extractfile("kept.cpp").read())
            self.assertTrue(archive.getmember("alias").issym())
            self.assertEqual("kept.cpp", archive.getmember("alias").linkname)

    def compile_entry(self, *extra):
        output = self.root / "compiled.o"
        output.write_bytes(b"actual object bytes")
        return {"directory": str(self.root),
                "file": str(self.root / "sanitizer-source.cpp"),
                "arguments": ["c++", *extra, "-c", "sanitizer-source.cpp",
                              "-o", "compiled.o"]}

    def test_object_provenance_requires_actual_outputs_not_claimed_hashes(self):
        entry = self.compile_entry()
        records = legacy.build_objects([entry])
        self.assertEqual(str(self.root / "compiled.o"),
                         records[0]["object"]["path"])
        self.assertEqual(hashlib.sha256(b"actual object bytes").hexdigest(),
                         records[0]["object"]["sha256"])
        (self.root / "compiled.o").unlink()
        with self.assertRaises(FileNotFoundError):
            legacy.build_objects([entry])

    def test_reference_rejects_sanitizer_flags_and_response_file_bypasses(self):
        for option in ("-fsanitize=address,undefined", "-fno-sanitize=all",
                       "-shared-libasan", "@hidden-flags.rsp"):
            with self.subTest(option=option):
                entry = self.compile_entry(option)
                with self.assertRaises(ValueError):
                    legacy.build_objects([entry])
        entry = self.compile_entry()
        entry["arguments"].append("-o")
        with self.assertRaisesRegex(ValueError, "unique object output"):
            legacy.build_objects([entry])
        for database in ([], {}, [{"file": True, "directory": "/tmp"}]):
            with self.subTest(database=database), self.assertRaises(ValueError):
                legacy.build_objects(database)

    def link_record(self, directory, arguments):
        path = directory / "CMakeFiles/tool.dir/link.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(shlex.join(arguments) + "\n", encoding="utf-8")
        return {"target": "tool", "cwd": str(directory),
                "command": legacy.file_record(path)}

    def test_static_inputs_include_sdk_relative_and_actual_implicit_archives(self):
        build = self.root / "build"
        cwd = build / "tools"
        project = build / "lib/libProject.a"
        sdk = self.root / "sdk/libCompiler.a"
        runtime = self.root / "runtime/libgcc.a"
        for path in (project, sdk, runtime):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        link = self.link_record(cwd, ["c++", "../lib/libProject.a", str(sdk),
                                      "-ldl", "-o", "tool"])
        trace = self.root / "build.log"
        trace.write_text(f"[100%] Linking CXX static library libProject.a\n"
                         f"../lib/libProject.a\n{sdk}\n{runtime}(unwind.o)\n")
        records = legacy.collect_static_inputs(build, [link], trace)
        self.assertEqual({str(project), str(sdk), str(runtime)},
                         {record["path"] for record in records})
        for record in records:
            self.assertEqual(legacy.sha256(record["path"]), record["sha256"])

    def test_static_inputs_bind_linker_wrapper_and_library_selection_without_path_guess(self):
        build = self.root / "build"
        cwd = build / "tools"
        sdk = self.root / "external SDK/libWrapped.a"
        selected = self.root / "external SDK/libSelected.a"
        sdk.parent.mkdir()
        sdk.write_bytes(b"wrapped archive")
        selected.write_bytes(b"linker selected this archive")
        link = self.link_record(cwd, ["c++", f"-Wl,--whole-archive,{sdk},--no-whole-archive",
                                      "-L" + str(sdk.parent), "-lSelected", "-o", "tool"])
        trace = self.root / "build.log"
        trace.write_text(f"{sdk}\n{selected}\n")
        records = legacy.collect_static_inputs(build, [link], trace)
        self.assertEqual({str(sdk), str(selected)}, {record["path"] for record in records})

    def test_static_link_response_files_fail_closed(self):
        for argument in ("@libraries.rsp", "-Wl,@libraries.rsp"):
            with self.subTest(argument=argument):
                build = self.root / "build"
                link = self.link_record(build / "tools", ["c++", argument, "-o", "tool"])
                trace = self.root / "build.log"
                trace.write_text("")
                with self.assertRaisesRegex(ValueError, "response files"):
                    legacy.collect_static_inputs(build, [link], trace)

    def test_static_inputs_reject_missing_explicit_archive(self):
        build = self.root / "build"
        link = self.link_record(build / "tools", ["c++", "missing.a", "-o", "tool"])
        trace = self.root / "build.log"
        trace.write_text("")
        with self.assertRaisesRegex(ValueError, "archive is missing"):
            legacy.collect_static_inputs(build, [link], trace)

    def test_static_inputs_reject_missing_or_ambiguous_linker_trace(self):
        build = self.root / "build"
        links = []
        for name in ("one", "two"):
            directory = build / name
            directory.mkdir(parents=True)
            (directory / "libSame.a").write_bytes(name.encode())
            links.append(self.link_record(directory, ["c++", "-o", name]))
        trace = self.root / "build.log"
        trace.write_text("libSame.a\n")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            legacy.collect_static_inputs(build, links, trace)
        trace.write_text("[100%] Linking CXX static library libSame.a\n")
        with self.assertRaisesRegex(ValueError, "no actual linker archive trace"):
            legacy.collect_static_inputs(build, links, trace)

    def test_cache_requires_release_product_only_and_exact_source(self):
        source = self.root / "source"
        values = {"BUILD_TESTING": "OFF", "NCNN_ENABLE_VULKAN_PRODUCT": "ON",
                  "NCNN_ENABLE_VULKAN_V0_TESTS": "OFF", "CMAKE_BUILD_TYPE": "Release",
                  "CMAKE_HOME_DIRECTORY": str(source), "COMPILER_SANITIZERS": "",
                  "CMAKE_EXE_LINKER_FLAGS": "-Wl,--trace"}
        path = self.root / "CMakeCache.txt"

        def write_cache(fields):
            path.write_text("\n".join(f"{key}:STRING={value}"
                                      for key, value in fields.items()))

        write_cache(values)
        legacy.validate_cache(path, source)
        for key, replacement in (("BUILD_TESTING", "ON"),
                                 ("NCNN_ENABLE_VULKAN_V0_TESTS", "ON"),
                                 ("CMAKE_BUILD_TYPE", "Debug"),
                                 ("CMAKE_HOME_DIRECTORY", "/another-source"),
                                 ("COMPILER_SANITIZERS", "address;undefined"),
                                 ("CMAKE_EXE_LINKER_FLAGS", "")):
            with self.subTest(key=key):
                write_cache({**values, key: replacement})
                with self.assertRaises(ValueError):
                    legacy.validate_cache(path, source)

    def test_failed_real_subprocess_is_preserved_and_never_qualified(self):
        builder = legacy.ReferenceBuilder(self.arguments())
        with self.assertRaisesRegex(RuntimeError, r"failed \(7\)"):
            builder.step("probe", [sys.executable, "-c",
                                   "print('actual failure'); raise SystemExit(7)"])
        state = json.loads((builder.root / "legacy-build-state.json").read_text())
        record = state["steps"][0]
        self.assertEqual(7, record["exit_code"])
        self.assertIn("actual failure", Path(record["log"]["path"]).read_text())
        self.assertEqual(legacy.sha256(record["log"]["path"]),
                         record["log"]["sha256"])
        self.assertFalse(state["completed"])
        self.assertFalse((builder.root / "legacy-reference.json").exists())

    def test_orchestration_is_local_pinned_fresh_and_parallel(self):
        original = self.root / "original"
        original.mkdir()
        sentinel = original / "uncommitted.txt"
        sentinel.write_bytes(b"must not change")
        roots = []
        for _ in range(2):
            builder = legacy.ReferenceBuilder(self.arguments())
            commands = []

            def record_step(name, argv):
                commands.append((name, [str(value) for value in argv]))
                return {"argv": commands[-1][1], "exit_code": 0}

            with mock.patch.object(legacy, "git_output",
                                   return_value=(legacy.LEGACY_REVISION + "\n").encode()):
                with mock.patch.object(builder, "step", side_effect=record_step), \
                        mock.patch.object(builder, "clone_submodules"), \
                        mock.patch.object(builder, "collect", return_value={"unit": "synthetic"}), \
                        redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    roots.append(builder.run())
            self.assertEqual(["clone", "checkout", "configure", "build"],
                             [name for name, _ in commands])
            self.assertIn("--local", commands[0][1])
            self.assertIn("--no-hardlinks", commands[0][1])
            self.assertIn(legacy.LEGACY_REVISION, commands[1][1])
            self.assertEqual(["--parallel", "2"], commands[-1][1][-2:])
            self.assertFalse(any(word in ("fetch", "worktree", "push", "install")
                                 for _, argv in commands for word in argv))
        self.assertNotEqual(roots[0], roots[1])
        self.assertEqual(b"must not change", sentinel.read_bytes())

    def test_failure_keeps_root_and_does_not_publish_a_manifest(self):
        builder = legacy.ReferenceBuilder(self.arguments())
        with mock.patch.object(legacy, "git_output",
                               return_value=(legacy.LEGACY_REVISION + "\n").encode()):
            with mock.patch.object(builder, "step", side_effect=RuntimeError("clone failed")):
                with self.assertRaisesRegex(RuntimeError, "preserved"), redirect_stdout(io.StringIO()):
                    builder.run()
        state = json.loads((builder.root / "legacy-build-state.json").read_text())
        self.assertIn("clone failed", state["failure"])
        self.assertFalse(state["completed"])
        self.assertTrue(builder.root.is_dir())
        self.assertFalse((builder.root / "legacy-reference.json").exists())


if __name__ == "__main__":
    unittest.main()
