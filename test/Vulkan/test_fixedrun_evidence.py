#!/usr/bin/env python3
"""Synthetic adversarial tests for VA4 fixed-run evidence validation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import fixedrun_evidence as evidence


class FixedRunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.stage = Path(self.temporary.name)
        self.cwd = self.stage / "test/Vulkan"
        self.cwd.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def put_bytes(self, relative, payload):
        path = self.stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {"path": path.relative_to(self.stage).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest()}

    def put_floats(self, relative, values):
        import struct

        return self.put_bytes(relative, struct.pack(f"<{len(values)}f", *values))

    def add_command_streams(self, command, label, stderr_text=None):
        stdout = f"{label} stdout\n".encode()
        stderr = (f"{label} stderr\n".encode() if stderr_text is None
                  else stderr_text.encode())
        command.update({
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "log": self.put_bytes(
                f"commands/{label}.log", stdout + b"\n--- stderr ---\n" + stderr),
        })
        return command

    def test_reshape_tosa_checks_mlir21_shape_operand_and_rank_round_trip(self):
        lines = [
            "%1 = tosa.reshape %0, %shape4 : (tensor<3x4x2xf32>, !tosa.shape<4>) -> tensor<1x3x4x2xf32>",
            "%2 = tosa.reshape %1, %shape3 : (tensor<1x3x4x2xf32>, !tosa.shape<3>) -> tensor<3x4x2xf32>",
        ]
        text = "\n".join(lines) + "\n"
        record = self.put_bytes("candidate/candidate_tosa.mlir", text.encode())
        metadata = {"tosa_ir": "candidate_tosa.mlir", "tosa_ir_sha256": record["sha256"]}
        expected = [{"line": 1, "text": lines[0], "input_type": "3x4x2xf32",
                     "output_type": "1x3x4x2xf32", "input_rank": 3, "output_rank": 4},
                    {"line": 2, "text": lines[1], "input_type": "1x3x4x2xf32",
                     "output_type": "3x4x2xf32", "input_rank": 4, "output_rank": 3}]
        candidate = {"tosa_ir_sha256": record["sha256"], "reshape_views": expected}
        path = self.stage / record["path"]
        self.assertEqual(expected, evidence._audit_reshape_tosa(path, metadata, candidate))
        with self.assertRaisesRegex(evidence.EvidenceError, "summary differs"):
            evidence._audit_reshape_tosa(path, metadata, {**candidate, "reshape_views": []})
        bad = text.replace("!tosa.shape<4>", "!tosa.shape<3>", 1)
        path.write_text(bad)
        digest = hashlib.sha256(bad.encode()).hexdigest()
        with self.assertRaisesRegex(evidence.EvidenceError, "element count or differs"):
            evidence._audit_reshape_tosa(path, {**metadata, "tosa_ir_sha256": digest},
                                        {**candidate, "tosa_ir_sha256": digest})

    def test_reshape_golden_is_untouched_chw_one_by_one_convolution(self):
        weights = struct.pack("<I6f", 0, 0.5, -0.25, -0.75, 1.25, 0.125, -0.5)
        input_values = [float(index) / 8 for index in range(24)]
        actual = evidence._scalar_golden("reshape", [input_values], weights)[0]
        expected = []
        for output_channel, bias in enumerate((0.125, -0.5)):
            for position in range(12):
                expected.append(
                    bias + input_values[position] * (0.5 if output_channel == 0 else -0.75) +
                    input_values[12 + position] * (-0.25 if output_channel == 0 else 1.25))
        self.assertEqual(24, len(actual))
        self.assertEqual(expected, actual)
        self.assertEqual(([[2, 3, 4]], [[2, 3, 4]]), evidence.FIXTURE_IO["reshape"])

    def test_fixture_success_path_checks_cpu_old_vulkan_and_native_raw_runs(self):
        import json
        import struct

        source_root = Path(__file__).resolve().parents[2]
        param_source = source_root / "test/Vulkan/relu.param"
        param_ref = self.put_bytes("fixture/relu.param", param_source.read_bytes())
        bin_ref = self.put_bytes("fixture/empty.bin", b"")
        input_values = [float((index % 19) - 9) / 8 for index in range(257)]
        golden = [max(0.0, value) for value in input_values]
        input_ref = self.put_floats("fixture/input.f32", input_values)
        expected_ref = self.put_floats("fixture/golden.f32", golden)
        cpu_output_ref = self.put_floats("fixture/cpu-aot.f32", golden)
        legacy_output_ref = self.put_floats("fixture/legacy-vulkan.f32", golden)
        candidate_output_ref = self.put_floats("fixture/candidate.f32", golden)
        cpu_dso = self.put_bytes("cpu/librelu.so", b"CPU AOT DSO")
        cpu_header = self.put_bytes("cpu/relu.h", b"CPU header")
        cpu_manifest = self.put_bytes("cpu/relu.json", b"{}")
        legacy_dso = self.put_bytes("legacy/librelu_vulkan.so", b"legacy Vulkan DSO")
        legacy_header = self.put_bytes("legacy/relu_vulkan.h", b"legacy header")
        legacy_manifest = self.put_bytes("legacy/relu_vulkan.json", b"{}")
        legacy_plan = self.put_bytes("legacy/relu_vulkan.plan.json", b"{}")
        cpu_loader_ref = self.put_bytes("loaders/relu-cpu-loader", b"CPU loader")
        old_loader_ref = self.put_bytes("loaders/relu-vulkan-loader", b"Vulkan loader")
        native_loader_source = source_root / "test/Vulkan/native/fixedrun_native_loader.cpp"
        loader_source_ref = self.put_bytes("loaders/fixedrun_native_loader.cpp",
                                           native_loader_source.read_bytes())
        cpu_compiler = self.stage / "tools/ncnn-compile"
        current_tools = {"driver": str(self.stage / "tools/ncnn-mlir-driver"),
                         "opt": str(self.stage / "bin/ncnn-mlir-opt"),
                         "translate": "/current/mlir-translate",
                         "clang": "/current/clang", "nm": "/current/llvm-nm",
                         "readelf": "/current/llvm-readelf", "llvm-as": "/current/llvm-as"}
        cache_bytes = ("MLIR_TRANSLATE:FILEPATH=/current/mlir-translate\n"
                       "CLANG:FILEPATH=/current/clang\nLLVM_NM:FILEPATH=/current/llvm-nm\n"
                       "LLVM_READELF:FILEPATH=/current/llvm-readelf\n"
                       "LLVM_AS:FILEPATH=/current/llvm-as\n").encode()
        self.put_bytes("CMakeCache.txt", cache_bytes)
        cpu_header_record = self.put_bytes("cpu/build/relu.h", b"compiled header")
        cpu_compile = {
            "argv": [str(cpu_compiler), str(param_source), "--bin",
                     str(self.stage / "build/empty.bin"), "--model-name", "relu",
                     "--output-dir", str(self.stage / "cpu/build"), "--driver", current_tools["driver"],
                     "--opt", current_tools["opt"], "--translate", current_tools["translate"],
                     "--clang", current_tools["clang"],
                     "--nm", current_tools["nm"], "--readelf", current_tools["readelf"],
                     "--llvm-as", current_tools["llvm-as"],
                     "--verify-execution", "--emit-manifest", "--precision=f32",
                     "--threads=1", "--vector-width=0"],
            "cwd": str(self.cwd), "returncode": 0,
            "output_dir": cpu_header_record,
        }
        self.add_command_streams(cpu_compile, "cpu-model-compile")
        build_bin = self.stage / "build/empty.bin"
        build_bin.parent.mkdir(parents=True, exist_ok=True)
        build_bin.write_bytes(b"")
        cpu_loader_argv = ["clang++", "-std=c++23", "-Wall", "-Wextra",
                           "-Wpedantic", "-fno-omit-frame-pointer",
                           "-I", str(source_root / "include"), "-I", str(self.stage / "cpu"),
                           "-DNCNN_FIXEDRUN_MODEL=relu", "-DNCNN_FIXEDRUN_MODEL_UPPER=RELU",
                           '-DNCNN_FIXEDRUN_MODEL_HEADER="relu.h"',
                           "-DNCNN_FIXEDRUN_BACKEND_VULKAN=0",
                           str(native_loader_source), "-o", str(self.stage / "loaders/relu-cpu-loader"),
                           "-ldl", "-pthread", "-Wl,-z,defs"]
        cpu_loader_build = {"argv": cpu_loader_argv, "cwd": str(self.cwd),
                            "returncode": 0,
                            "source": loader_source_ref,
                            "source_target_path": str(native_loader_source.resolve()),
                            "source_target_sha256": hashlib.sha256(
                                native_loader_source.read_bytes()).hexdigest(),
                            "executable": cpu_loader_ref,
                            "executable_target_path": str(self.stage / "loaders/relu-cpu-loader"),
                            "executable_target_sha256": cpu_loader_ref["sha256"]}
        self.add_command_streams(cpu_loader_build, "cpu-loader-compile")
        cpu = {"compile": cpu_compile, "manifest": cpu_manifest,
               "library": cpu_dso, "library_sha256": cpu_dso["sha256"],
               "loader_build": cpu_loader_build, "loader": cpu_loader_ref}
        old_compiler = "/tmp/va4-pinned-legacy/tools/ncnn-compile"
        old_tool_paths = {role: f"/tmp/va4-pinned-legacy/tools/{role}"
                          for role in ("driver", "opt", "translate", "clang", "nm",
                                       "readelf", "llvm-as", "mlir-opt", "spirv-val",
                                       "spirv-dis", "spirv-extract", "host-cxx")}
        legacy_tools = {role: {"path": path, "sha256": "b" * 64,
                               "staged": self.put_bytes(f"legacy/tools/{role}", role.encode())}
                        for role, path in old_tool_paths.items()}
        old_compile = {
            "argv": [old_compiler, "-v", str(param_source), "--bin", str(build_bin),
                     "--model-name", "relu", "--output-dir", str(self.stage / "legacy"),
                     "--driver", old_tool_paths["driver"], "--opt", old_tool_paths["opt"],
                     "--translate", old_tool_paths["translate"], "--clang", old_tool_paths["clang"],
                     "--nm", old_tool_paths["nm"], "--readelf", old_tool_paths["readelf"],
                     "--llvm-as", old_tool_paths["llvm-as"], "--verify-execution", "--backend=vulkan",
                     "--precision=f32", "--vulkan-mlir-opt", old_tool_paths["mlir-opt"],
                     "--vulkan-spirv-val", old_tool_paths["spirv-val"],
                     "--vulkan-spirv-dis", old_tool_paths["spirv-dis"],
                     "--vulkan-spirv-extract", old_tool_paths["spirv-extract"],
                     "--vulkan-host-cxx", old_tool_paths["host-cxx"]],
            "cwd": str(self.cwd), "returncode": 0, "backend": "vulkan",
            "compiler": {"path": old_compiler, "sha256": "a" * 64},
            "library": legacy_dso, "library_sha256": legacy_dso["sha256"],
            "header": legacy_header, "manifest": legacy_manifest, "plan": legacy_plan,
            "reference_status": "verified-independent-source",
        }
        child_log = "".join(
            f"{old_tool_paths[role]} --synthetic-invocation\n"
            for role in ("driver", "opt", "nm", "readelf", "mlir-opt",
                         "spirv-val", "spirv-dis", "spirv-extract", "host-cxx"))
        self.add_command_streams(old_compile, "legacy-vulkan-compile", child_log)
        legacy_reference = {"qualified": True, "compiler_original_path": old_compiler,
                            "compiler": {"sha256": "a" * 64}, "tools": legacy_tools}

        def native_reference_run(phase, loader_ref, library_ref, output_role):
            input_path = self.stage / f"runs/{phase}-input.f32"
            output_path = self.stage / f"runs/{phase}-output.f32"
            input_record = self.put_floats(input_path.relative_to(self.stage), input_values)
            output_record = self.put_floats(output_path.relative_to(self.stage), golden)
            loader_path = (self.stage / loader_ref["path"]).resolve()
            library_path = (self.stage / library_ref["path"]).resolve()
            argv = [str(loader_path), "--library", str(library_path), "--mode", "single",
                    "--input", str(input_path), "--output", str(output_path)]
            lifecycle = {"event": "lifecycle", "create_count": 0 if phase == "cpu_aot_reference" else 1,
                         "create_statuses": [] if phase == "cpu_aot_reference" else [0],
                         "run_count": 1, "run_statuses": [0],
                         "destroy_count": 0 if phase == "cpu_aot_reference" else 1,
                         "destroy_statuses": [] if phase == "cpu_aot_reference" else [0]}
            event = {"event": "run", "status": 0, "elapsed_ns": 17,
                     "diagnostic": "", "output_paths": [str(output_path)]}
            stdout = (json.dumps(event) + "\n" + json.dumps(lifecycle) + "\n").encode()
            stderr = b""
            log_record = self.put_bytes(
                f"runs/{phase}.log", stdout + b"\n--- stderr ---\n" + stderr)
            return {"fixture": "relu", "phase": phase, "iteration": 0,
                    "loader_result_index": 0, "backend": "native-reference",
                    "inputs": [input_record], "outputs": [output_record],
                    "input_shapes": [[1, 1, 257]], "output_shapes": [[1, 1, 257]],
                    "status": 0, "diagnostic": "", "validation_error_count": 0,
                    "elapsed_ns": 17, "dso_sha256": library_ref["sha256"],
                    "native_loader": {"executable": loader_ref, "argv": argv,
                                       "cwd": str(self.cwd), "returncode": 0,
                                       "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                                       "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                                       "log": log_record, "lifecycle": lifecycle}}

        cpu_run = native_reference_run("cpu_aot_reference", cpu_loader_ref,
                                       cpu_dso, "cpu_aot_raw")
        legacy_run = native_reference_run("legacy_vulkan_reference", old_loader_ref,
                                          legacy_dso, "legacy_vulkan_raw")
        references = {"independent_legacy": True, "cpu_aot": cpu,
                      "legacy_vulkan_compile": old_compile,
                      "runs": [cpu_run, legacy_run]}
        fixture = {"name": "relu", "param": param_ref,
                   "param_source": "test/Vulkan/relu.param", "bin": bin_ref,
                   "bin_source": "generated-empty-build-input", "bin_generator": None,
                   "inputs": [{"name": "input", "shape": [1, 1, 257],
                               "elements": 257, "raw": input_ref}],
                   "outputs": [{"name": "output", "shape": [1, 1, 257],
                                "elements": 257, "golden_raw": expected_ref,
                                "cpu_aot_raw": cpu_output_ref,
                                "legacy_vulkan_raw": legacy_output_ref,
                                "current_generic_raw": None,
                                "candidate_raw": candidate_output_ref}],
                   "candidate_build": {"native_loader": old_loader_ref},
                   "legacy_vulkan_compile": old_compile,
                   "references": references, "reference_runs": [cpu_run, legacy_run],
                   "native_loader_build": cpu_loader_build}
        summary = evidence._audit_fixture(
            self.stage, source_root, fixture, "relu", allow_missing_legacy=False,
            legacy_reference=legacy_reference)
        self.assertEqual("verified", summary["references"]["cpu_aot"])
        self.assertEqual("verified", summary["references"]["legacy_vulkan"])

    def test_candidate_native_runs_validate_full_repeat_concurrent_and_interleaved_traces(self):
        from unittest.mock import patch

        name = "relu"
        dso = self.put_bytes("candidate/librelu_fixedrun.so", b"candidate DSO")
        loader = self.put_bytes("candidate/relu-loader", b"native C++ loader")
        bin_ref = self.put_bytes("fixtures/empty.bin", b"")
        fixture_records = {name: {"name": name, "bin": bin_ref,
                                  "references": {"runs": []}}}
        candidate = {name: {"link": {"dso": dso}, "native_loader": loader}}
        seed_a = [float((index % 23) - 11) / 8 for index in range(257)]
        seed_b = [float((index % 17) - 8) / 4 for index in range(257)]
        all_runs = []

        def write_input(label, values):
            return self.put_floats(f"runs/{label}.input.f32", values)

        def make_invocation(phase, mode, inputs_a, inputs_b=None):
            base = self.stage / f"runs/{phase}"
            base.parent.mkdir(parents=True, exist_ok=True)
            input_paths_a = [self.stage / f"runs/{phase}-input-a.f32"]
            input_refs_a = [self.put_floats(input_paths_a[0].relative_to(self.stage), inputs_a[0])]
            output_paths_a = [self.stage / f"runs/{phase}-output-a.f32"]
            input_paths_b, output_paths_b, input_refs_b = [], [], []
            if inputs_b is not None:
                input_paths_b = [self.stage / f"runs/{phase}-input-b.f32"]
                input_refs_b = [self.put_floats(input_paths_b[0].relative_to(self.stage), inputs_b[0])]
                output_paths_b = [self.stage / f"runs/{phase}-output-b.f32"]
            output_refs_a = [self.put_floats(output_paths_a[0].relative_to(self.stage),
                                             [max(0.0, value) for value in inputs_a[0]])]
            output_refs_b = ([self.put_floats(output_paths_b[0].relative_to(self.stage),
                                              [max(0.0, value) for value in inputs_b[0]])]
                             if inputs_b is not None else [])
            invocation_inputs = [input_refs_a]
            invocation_outputs = [output_refs_a]
            argv = [str((self.stage / loader["path"]).resolve()), "--library",
                    str((self.stage / dso["path"]).resolve()), "--mode", mode]
            a_flag, b_flag = (("--input", "--output") if mode == "single" else
                              ("--input-a", "--output-a"))
            for path in input_paths_a:
                argv.extend((a_flag, str(path)))
            for path in output_paths_a:
                argv.extend((b_flag, str(path)))
            if inputs_b is not None:
                for path in input_paths_b:
                    argv.extend(("--input-b", str(path)))
                for path in output_paths_b:
                    argv.extend(("--output-b", str(path)))
                invocation_inputs.append(input_refs_b)
                invocation_outputs.append(output_refs_b)
            if mode == "interleaved":
                extra_paths = [path.with_name(path.stem + ".interleaved-1" + path.suffix)
                               for path in output_paths_a + output_paths_b]
                extra_refs = [self.put_floats(
                    path.relative_to(self.stage),
                    [max(0.0, value) for value in
                     (inputs_a[0] if index == 0 else inputs_b[0])])
                    for index, path in enumerate(extra_paths)]
                invocation_inputs.extend((input_refs_a, input_refs_b))
                invocation_outputs.extend(([extra_refs[0]], [extra_refs[1]]))
            input_sequence = []
            output_sequence = []
            if mode == "repeat" or mode == "concurrent":
                input_sequence = [input_refs_a, input_refs_b]
                output_sequence = [output_refs_a, output_refs_b]
            elif mode == "interleaved":
                input_sequence = [input_refs_a, input_refs_b, input_refs_a, input_refs_b]
                output_sequence = [output_refs_a, output_refs_b,
                                   [extra_refs[0]], [extra_refs[1]]]
            else:
                input_sequence, output_sequence = invocation_inputs, invocation_outputs
            measurement_events = []
            for index, output_set in enumerate(output_sequence):
                output_paths = [self.stage / item["path"] for item in output_set]
                measurement_events.append({
                    "event": "run", "status": 0, "elapsed_ns": 100 + index,
                    "diagnostic": "", "output_paths": [str(path) for path in output_paths]})
            context_count = 2 if mode == "interleaved" else 1
            lifecycle = {"event": "lifecycle", "create_count": context_count,
                         "create_statuses": [0] * context_count,
                         "run_count": len(measurement_events),
                         "run_statuses": [0] * len(measurement_events),
                         "destroy_count": context_count,
                         "destroy_statuses": [0] * context_count}
            stdout = ("".join(json.dumps(item) + "\n"
                               for item in measurement_events + [lifecycle])).encode()
            stderr = b""
            log_ref = self.put_bytes(
                f"runs/{phase}.native.log", stdout + b"\n--- stderr ---\n" + stderr)
            command_record = {"executable": loader, "argv": argv, "cwd": str(self.cwd),
                              "returncode": 0,
                              "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                              "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                              "log": log_ref, "lifecycle": lifecycle}
            phases = {"same_seed_repeat": "same_seed_repeat",
                      "different_input": "different_input",
                      "same_context_concurrent": "same_context_concurrent",
                      "interleaved_contexts": "interleaved_contexts"}
            run_phase = phases[phase]
            phase_runs = []
            for index, (input_set, output_set) in enumerate(zip(input_sequence, output_sequence)):
                phase_runs.append({
                    "fixture": name, "phase": run_phase,
                    "iteration": index, "loader_result_index": index,
                    "backend": "fixedrun-candidate", "inputs": input_set,
                    "outputs": output_set, "input_shapes": [[1, 1, 257]],
                    "output_shapes": [[1, 1, 257]], "status": 0,
                    "diagnostic": "", "validation_error_count": 0,
                    "elapsed_ns": 100 + index, "dso_sha256": dso["sha256"],
                    "native_loader": command_record})
            return phase_runs

        all_runs.extend(make_invocation("same_seed_repeat", "repeat", [seed_a], [seed_a]))
        all_runs.extend(make_invocation("different_input", "repeat", [seed_a], [seed_b]))
        all_runs.extend(make_invocation("same_context_concurrent", "concurrent",
                                        [seed_a], [seed_b]))
        all_runs.extend(make_invocation("interleaved_contexts", "interleaved",
                                        [seed_a], [seed_b]))
        with patch.object(evidence, "FIXTURES", frozenset({name})):
            metrics, refs = evidence._audit_runs(
                self.stage, all_runs, candidate, fixture_records, "release")
        self.assertEqual(10, metrics["invocation_count"])
        self.assertEqual(set(evidence.PHASES), set(metrics["phase_coverage"][name]))
        self.assertTrue(refs)
        with patch.object(evidence, "FIXTURES", frozenset({name})):
            with self.assertRaisesRegex(evidence.EvidenceError, "repeats a native invocation"):
                evidence._audit_runs(self.stage, all_runs + [all_runs[2]],
                                     candidate, fixture_records, "release")
            separate_process_rows = deepcopy(all_runs)
            separate_process_rows[3]["native_loader"] = dict(
                separate_process_rows[3]["native_loader"])
            original_log = separate_process_rows[3]["native_loader"]["log"]
            separate_process_rows[3]["native_loader"]["log"] = self.put_bytes(
                "runs/another-native-process.log",
                (self.stage / original_log["path"]).read_bytes())
            with self.assertRaisesRegex(evidence.EvidenceError, "different native process"):
                evidence._audit_runs(self.stage, separate_process_rows,
                                     candidate, fixture_records, "release")
            with self.assertRaisesRegex(evidence.EvidenceError, "unknown phase"):
                evidence._audit_runs(self.stage, all_runs + [{
                    "fixture": name, "phase": "cpu_aot_reference"}],
                    candidate, fixture_records, "release")

    def test_compiler_input_audit_rejects_forced_includes_plugins_and_nested_rsp(self):
        evidence._audit_no_forced_compiler_inputs(
            ["clang++", "-O2", "-c", "candidate.cpp"], "compile", "release")
        for arguments in (
                ["clang++", "-include", "injected.hpp", "candidate.cpp"],
                ["clang++", "-Wp,-include,hidden.hpp", "candidate.cpp"],
                ["clang++", "-Xclang", "-load", "plugin.so", "candidate.cpp"],
                ["clang++", "-Wl,@hidden.rsp", "candidate.cpp"],
                ["clang++", "@hidden.rsp", "candidate.cpp"],
                ["clang++", "-DNCNN_VULKAN_LIFECYCLE_TEST=1", "candidate.cpp"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(evidence.EvidenceError):
                evidence._audit_no_forced_compiler_inputs(arguments, "compile", "release")

    def test_sanitizer_instrumentation_distinguishes_compile_and_link_roles(self):
        sanitizer = ["-fsanitize=address,undefined",
                     "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
        evidence._require_profile_instrumentation(
            ["clang++", *sanitizer, "-c", "helper.cpp"],
            "sanitize", "helper object compile", link=False)
        evidence._require_profile_instrumentation(
            ["clang++", *sanitizer, "-shared-libasan", "-shared", "-o", "model.so"],
            "sanitize", "candidate DSO link", link=True)

        for missing in sanitizer:
            arguments = ["clang++", *[flag for flag in sanitizer if flag != missing],
                         "-c", "helper.cpp"]
            with self.subTest(missing=missing), self.assertRaisesRegex(
                    evidence.EvidenceError, r"lacks exact ASan\+UBSan"):
                evidence._require_profile_instrumentation(
                    arguments, "sanitize", "helper object compile", link=False)

        with self.assertRaisesRegex(evidence.EvidenceError, "shared-only ASan"):
            evidence._require_profile_instrumentation(
                ["clang++", *sanitizer, "-shared", "-o", "model.so"],
                "sanitize", "candidate DSO link", link=True)
        with self.assertRaisesRegex(evidence.EvidenceError, "shared-only ASan"):
            evidence._require_profile_instrumentation(
                ["clang++", *sanitizer, "-shared-libasan", "-static-libasan",
                 "-shared", "-o", "model.so"],
                "sanitize", "candidate DSO link", link=True)
        with self.assertRaisesRegex(evidence.EvidenceError, "shared-only ASan"):
            evidence._audit_forwarded_link_sanitizer_args(
                ["ncnn-compile", "--clang-arg", "-shared-libasan"],
                "sanitize", "CPU AOT compile/link")
        with self.assertRaisesRegex(evidence.EvidenceError, "shared-only ASan"):
            evidence._audit_forwarded_link_sanitizer_args(
                ["ncnn-compile", "--linker-arg", "-shared-libasan",
                 "--linker-arg", "-static-libasan"],
                "sanitize", "CPU AOT compile/link")
        evidence._audit_forwarded_link_sanitizer_args(
            ["ncnn-compile", "--clang-arg", "-fsanitize=address,undefined",
             "--clang-arg", "-fno-sanitize-recover=all",
             "--clang-arg", "-fno-omit-frame-pointer",
             "--linker-arg", "-shared-libasan"],
            "sanitize", "CPU AOT compile/link")

        evidence._require_profile_instrumentation(
            ["c++", "-c", "helper.cpp"], "release", "Release helper compile")
        with self.assertRaisesRegex(evidence.EvidenceError, "unexpectedly uses sanitizer"):
            evidence._require_profile_instrumentation(
                ["c++", "-fsanitize=address,undefined", "-c", "helper.cpp"],
                "release", "Release helper compile")

    def test_candidate_link_allows_only_bound_compiler_and_runtime_rpath(self):
        configured_driver = self.stage / "toolchain/llvm-21/bin/clang++"
        configured_driver.parent.mkdir(parents=True)
        configured_driver.write_bytes(b"configured clang driver")
        runtime_dir = self.stage / "toolchain/llvm-21/lib/clang/21/lib/linux"
        valid = [str(configured_driver), "-shared",
                 f"-Wl,-rpath,{runtime_dir}", "-o", "model.so"]
        evidence._audit_candidate_link_scope(valid, configured_driver, runtime_dir)

        impostor_driver = self.stage / "impostor/llvm-21/bin/clang++"
        impostor_driver.parent.mkdir(parents=True)
        impostor_driver.write_bytes(b"unconfigured clang lookalike")
        with self.assertRaisesRegex(evidence.EvidenceError, r"configured gate C\+\+ driver"):
            evidence._audit_candidate_link_scope(
                [str(impostor_driver), *valid[1:]], configured_driver, runtime_dir)

        impostor_rpath = self.stage / "other/llvm-21/lib/linux"
        bad_rpath = [valid[0], "-shared", f"-Wl,-rpath,{impostor_rpath}",
                     "-o", "model.so"]
        with self.assertRaisesRegex(evidence.EvidenceError, "unconfigured RPATH"):
            evidence._audit_candidate_link_scope(
                bad_rpath, configured_driver, runtime_dir)
        non_llvm_rpath = [valid[0], "-shared", "-Wl,-rpath,/tmp/unapproved-runtime",
                          "-o", "model.so"]
        with self.assertRaisesRegex(evidence.EvidenceError, "unconfigured RPATH"):
            evidence._audit_candidate_link_scope(
                non_llvm_rpath, configured_driver, runtime_dir)
        for bad_input in ("/tmp/libLLVMCompilerRuntime.a", "-lMLIRSupport"):
            with self.subTest(bad_input=bad_input), self.assertRaisesRegex(
                    evidence.EvidenceError, "generic executor/compiler input"):
                evidence._audit_candidate_link_scope(
                    [*valid, bad_input], configured_driver, runtime_dir)

    def test_sanitizer_runtime_rpath_is_derived_from_hash_bound_cxx(self):
        compiler = self.stage / "toolchain/llvm-21/bin/clang++"
        compiler.parent.mkdir(parents=True)
        compiler.write_bytes(b"configured clang driver")
        resource_dir = self.stage / "toolchain/llvm-21/lib/clang/21"
        runtime_dir = resource_dir / "lib/linux"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "libclang_rt.asan-test.so").write_bytes(b"shared ASan")
        digest = hashlib.sha256(compiler.read_bytes()).hexdigest()
        (self.stage / "CMakeCache.txt").write_text(
            f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n", encoding="utf-8")
        cache_hash = hashlib.sha256(
            (self.stage / "CMakeCache.txt").read_bytes()).hexdigest()
        gate_state = {
            "cmake_cache": {"path": "CMakeCache.txt", "sha256": cache_hash},
            "toolchain_identities": [{"role": "cxx-compiler",
                                      "path": str(compiler.resolve()),
                                      "sha256": digest}],
        }
        gate_state_path = self.stage / "gate-state.json"
        gate_state_path.write_text(json.dumps(gate_state), encoding="utf-8")
        result = subprocess.CompletedProcess(
            [str(compiler), "--print-resource-dir"], 0,
            f"{resource_dir}\n".encode(), b"")
        from unittest.mock import patch
        with patch.object(evidence.subprocess, "run", return_value=result) as run:
            self.assertEqual(runtime_dir, evidence._configured_sanitizer_runtime_dir(self.stage))
        run.assert_called_once_with(
            [str(compiler.resolve()), "--print-resource-dir"], cwd=self.stage,
            capture_output=True, check=False)

        impostor = self.stage / "toolchain/other/bin/clang++"
        impostor.parent.mkdir(parents=True)
        impostor.write_bytes(b"unconfigured compiler")
        (self.stage / "CMakeCache.txt").write_text(
            f"CMAKE_CXX_COMPILER:FILEPATH={impostor}\n", encoding="utf-8")
        gate_state["cmake_cache"]["sha256"] = hashlib.sha256(
            (self.stage / "CMakeCache.txt").read_bytes()).hexdigest()
        gate_state_path.write_text(json.dumps(gate_state), encoding="utf-8")
        with self.assertRaisesRegex(evidence.EvidenceError, "gate-bound binary"):
            evidence._configured_sanitizer_runtime_dir(self.stage)

    def test_dynamic_dependency_audit_requires_bound_readelf_shared_asan(self):
        readelf = self.stage / "toolchain/llvm-21/bin/llvm-readelf"
        readelf.parent.mkdir(parents=True)
        readelf_target = readelf.with_name("llvm-readobj")
        readelf_target.write_bytes(b"configured LLVM readelf")
        readelf.symlink_to(readelf_target.name)
        binary = self.stage / "native/loader"
        binary.parent.mkdir()
        binary.write_bytes(b"verified executable bytes")
        digest = hashlib.sha256(readelf_target.read_bytes()).hexdigest()
        (self.stage / "CMakeCache.txt").write_text(
            f"LLVM_READELF:FILEPATH={readelf}\n", encoding="utf-8")
        cache_hash = hashlib.sha256(
            (self.stage / "CMakeCache.txt").read_bytes()).hexdigest()
        gate_state = {
            "cmake_cache": {"path": "CMakeCache.txt", "sha256": cache_hash},
            "toolchain_identities": [{"role": "llvm-readelf",
                                      "path": str(readelf.resolve()),
                                      "sha256": digest}],
        }
        gate_state_path = self.stage / "gate-state.json"
        gate_state_path.write_text(json.dumps(gate_state), encoding="utf-8")
        valid = b"(NEEDED) Shared library: [libclang_rt.asan-x86_64.so]\n"
        completed = subprocess.CompletedProcess([], 0, valid, b"")
        from unittest.mock import patch
        with patch.object(evidence.subprocess, "run", return_value=completed) as run:
            record = evidence._audit_shared_asan_needed(
                self.stage, binary, "sanitize", "native loader")
        self.assertIn("libclang_rt.asan-x86_64.so", record["needed_libraries"])
        self.assertEqual(digest, record["readelf"]["tool_sha256"])
        run.assert_called_once_with(
            [str(readelf), "-d", str(binary.resolve())],
            cwd=self.stage, capture_output=True, check=False)

        with patch.object(evidence.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 0, b"no shared NEEDED\n", b"")):
            with self.assertRaisesRegex(evidence.EvidenceError, "shared Clang ASan"):
                evidence._audit_shared_asan_needed(
                    self.stage, binary, "sanitize", "native loader")
        with patch.object(evidence.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(evidence.EvidenceError, "Release .*sanitizer runtime"):
                evidence._audit_shared_asan_needed(
                    self.stage, binary, "release", "native loader")

        impostor_readelf = self.stage / "toolchain/other/llvm-readelf"
        impostor_readelf.parent.mkdir(parents=True)
        impostor_readelf.write_bytes(b"unconfigured readelf")
        (self.stage / "CMakeCache.txt").write_text(
            f"LLVM_READELF:FILEPATH={impostor_readelf}\n", encoding="utf-8")
        gate_state["cmake_cache"]["sha256"] = hashlib.sha256(
            (self.stage / "CMakeCache.txt").read_bytes()).hexdigest()
        gate_state_path.write_text(json.dumps(gate_state), encoding="utf-8")
        with self.assertRaisesRegex(evidence.EvidenceError, "gate-bound binary"):
            evidence._audit_shared_asan_needed(
                self.stage, binary, "sanitize", "native loader")

    def test_exact_integer_rejects_boolean_and_float_aliases(self):
        for value in (True, False, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaises(evidence.EvidenceError):
                    evidence._exact_int(value, "value")
        self.assertEqual(1, evidence._exact_int(1, "value"))

    def test_static_shapes_require_positive_exact_dimensions(self):
        self.assertEqual(24, evidence._shape_elements([2, 3, 4], "shape"))
        for shape in ([], [0, 2], [-1], [1.0], [True]):
            with self.subTest(shape=shape):
                with self.assertRaises(evidence.EvidenceError):
                    evidence._shape_elements(shape, "shape")

    def test_numeric_comparison_recomputes_per_element_tolerance(self):
        result = evidence._compare([1.0, -2.0], [1.0000001, -2.0],
                                  1.0e-6, 1.0e-6, "candidate")
        self.assertEqual(2, result["elements"])
        with self.assertRaisesRegex(evidence.EvidenceError, "exceeds tolerance"):
            evidence._compare([1.01], [1.0], 1.0e-6, 1.0e-6, "candidate")

    def test_numeric_comparison_rejects_nonfinite_or_boolean_tolerance(self):
        with self.assertRaises(evidence.EvidenceError):
            evidence._compare([float("nan")], [0.0], 1.0, 0.0, "candidate")
        with self.assertRaises(evidence.EvidenceError):
            evidence._compare([0.0], [0.0], True, 0.0, "candidate")

    def test_elf_audit_binds_exact_tool_argv_output_and_stream_log(self):
        tool = self.put_bytes("tools/llvm-nm", b"tool identity")
        binary = self.put_bytes("candidate/librelu.so", b"candidate DSO")
        output = b"00000000 T relu_vk_run\n"
        stdout = self.put_bytes("elf/exports.raw", output)
        stderr = b""
        log = self.put_bytes("elf/exports.log",
                             output + b"\n--- stderr ---\n" + stderr)
        command = {
            "argv": [str((self.stage / tool["path"]).resolve()), "-D",
                     "--defined-only", str((self.stage / binary["path"]).resolve())],
            "cwd": str(self.cwd), "returncode": 0,
            "stdout_sha256": hashlib.sha256(output).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stdout": stdout, "log": log,
        }
        accepted, digest, invocation = evidence._audit_elf_tool_invocation(
            self.stage, command, stdout, self.stage / binary["path"],
            self.stage / tool["path"], ["-D", "--defined-only"], "exports")
        self.assertEqual(self.stage / "elf/exports.raw", accepted)
        self.assertEqual(stdout["sha256"], digest)
        command["stdout"] = self.put_bytes("elf/other.raw", b"different")
        with self.assertRaises(evidence.EvidenceError):
            evidence._audit_elf_tool_invocation(
                self.stage, command, stdout, self.stage / binary["path"],
                self.stage / tool["path"], ["-D", "--defined-only"], "exports")

    def test_link_input_audit_ignores_soname_and_output_but_pins_vulkan_library(self):
        build_dir = self.stage / "test/Vulkan/native"
        build_dir.mkdir(parents=True)
        object_file = build_dir / "model.o"
        archive = self.stage / "lib/helper.a"
        object_file.parent.mkdir(parents=True, exist_ok=True)
        archive.parent.mkdir(parents=True, exist_ok=True)
        object_file.write_bytes(b"object")
        archive.write_bytes(b"archive")
        vulkan = self.stage / "sdk/libvulkan.so.1"
        vulkan.parent.mkdir(parents=True, exist_ok=True)
        vulkan.write_bytes(b"configured Vulkan loader")
        output = self.stage / "product/libmodel.so"
        output.parent.mkdir()
        command = ["c++", "-Wl,-soname,libmodel.so.1", "-o", str(output),
                   str(object_file), str(archive), str(vulkan), str(vulkan)]
        objects, archives, shared = evidence._candidate_link_inputs(
            self.stage, command, str(build_dir), output, vulkan)
        self.assertEqual({object_file.resolve()}, objects)
        self.assertEqual({archive.resolve()}, archives)
        self.assertEqual({vulkan.resolve()}, shared)

        for bad in (self.stage / "private/libevil.so",
                    self.stage / "private/libevil.so.1"):
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_bytes(b"private DSO")
            with self.subTest(private=bad.name), self.assertRaises(evidence.EvidenceError):
                evidence._candidate_link_inputs(
                    self.stage, command[:-2] + [str(bad)], str(build_dir), output, vulkan)

    def test_vulkan_library_snapshot_binds_cache_alias_and_repeated_link_arguments(self):
        system = self.stage / "system/libvulkan.so.1.0"
        system.parent.mkdir()
        system.write_bytes(b"system Vulkan loader bytes")
        alias = system.with_name("libvulkan.so")
        alias.symlink_to(system.name)
        snapshot = self.put_bytes("snapshots/libvulkan.so.1.0", system.read_bytes())
        output = self.stage / "product/libmodel.so"
        argv = ["c++", "-Wl,-soname,libmodel.so", "-o", str(output),
                str(alias), str(alias)]
        arguments = [{"index": index, "spelling": str(alias),
                      "resolved_path": str(system)} for index in (4, 5)]
        link = {"vulkan_library": snapshot,
                "vulkan_library_cache_key": "Vulkan_LIBRARY",
                "vulkan_library_target_path": str(alias),
                "vulkan_library_resolved_path": str(system),
                "vulkan_library_target_sha256": snapshot["sha256"],
                "vulkan_library_link_arguments": arguments}
        summary = evidence._audit_vulkan_library(
            self.stage, link, argv, str(self.cwd), str(alias))
        self.assertEqual(snapshot["sha256"], summary["artifact"]["sha256"])
        for key, bad in (("vulkan_library_target_path", str(system)),
                         ("vulkan_library_target_sha256", "0" * 64),
                         ("vulkan_library_link_arguments", arguments[:1]),
                         ("vulkan_library_cache_key", "another-key")):
            with self.subTest(key=key), self.assertRaises(evidence.EvidenceError):
                evidence._audit_vulkan_library(
                    self.stage, {**link, key: bad}, argv, str(self.cwd), str(alias))
        with self.assertRaises(evidence.EvidenceError):
            evidence._audit_vulkan_library(
                self.stage, {**link, "vulkan_library": {
                    "path": system.relative_to(self.stage).as_posix(),
                    "sha256": snapshot["sha256"]}}, argv, str(self.cwd), str(alias))

    def test_post_ctest_aggregate_is_bound_separately_from_test_raw_inventory(self):
        from unittest.mock import patch

        required = ["vulkan-fixedrun-capture-publish",
                    "vulkan-fixedrun-fake-unit", "vulkan-fixedrun-evidence-validation",
                    "vulkan-fixedrun-independence", "vulkan-fixedrun-legacy-reference-tests",
                    "vulkan-fixedrun-producer-tool-role-validation",
                    "vulkan-fixedrun-nvidia", "vulkan-fixedrun-llvmpipe"]
        def device_summary(stage, profile, fingerprint, device, **kwargs):
            raw = self.put_bytes(f"raw/{device}.json", device.encode())
            return {"legacy_reference": {"qualified": True},
                    "candidate_builds": {name: {"link": {"dso": {"sha256": "a" * 64}}}
                                         for name in evidence.FIXTURES}}, [raw]

        with patch.object(evidence, "audit_device_raw", side_effect=device_summary):
            aggregate, refs = evidence.aggregate_stage_evidence(
                self.stage, "release", "f" * 64, required, [], [])
        output = self.stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json"
        self.assertEqual(aggregate, json.loads(output.read_text()))
        self.assertEqual({"raw/nvidia.json", "raw/llvmpipe.json"},
                         {row["path"] for row in refs})
        self.assertNotIn(output.relative_to(self.stage).as_posix(),
                         {row["path"] for row in refs})

    def test_post_ctest_aggregate_reaudit_preserves_verified_archive_bytes(self):
        from datetime import datetime, timezone
        from unittest.mock import Mock, patch

        import check_release_evidence as release_evidence
        import run_gate

        stage = self.stage / "sealed-stage"
        stage.mkdir()
        fingerprint = "f" * 64
        (stage / "gate-state.json").write_text(json.dumps({
            "completed": True, "profile": "release",
            "source_fingerprint": fingerprint,
        }), encoding="utf-8")
        required = ["vulkan-fixedrun-capture-publish",
                    "vulkan-fixedrun-fake-unit", "vulkan-fixedrun-evidence-validation",
                    "vulkan-fixedrun-independence", "vulkan-fixedrun-legacy-reference-tests",
                    "vulkan-fixedrun-producer-tool-role-validation",
                    "vulkan-fixedrun-nvidia", "vulkan-fixedrun-llvmpipe"]
        raw_records = {}
        for device in evidence.DEVICES:
            relative = f"test/Vulkan/evidence/va4/{device}/raw.json"
            path = stage / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(device.encode())
            raw_records[device] = {"path": relative,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        def device_summary(stage, profile, fingerprint, device, **kwargs):
            return {"legacy_reference": {"qualified": True},
                    "candidate_builds": {name: {"link": {"dso": {"sha256": "a" * 64}}}
                                         for name in evidence.FIXTURES}}, [raw_records[device]]

        clock = Mock()
        clock.now.side_effect = [datetime(2026, 1, 1, tzinfo=timezone.utc),
                                 datetime(2026, 1, 2, tzinfo=timezone.utc)]
        output = stage / "test/Vulkan/evidence/va4/fixedrun-evidence.json"
        archive = self.stage / "sealed-stage.evidence.tar.gz"
        with patch.object(evidence, "audit_device_raw", side_effect=device_summary), \
                patch.object(evidence, "datetime", clock, create=True):
            first, first_refs = evidence.aggregate_stage_evidence(
                stage, "release", fingerprint, required, [], [])
            original = output.read_bytes()
            run_gate.archive_stage(stage, archive)
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual("verified", release_evidence.audit_stage_archive(
                stage, required=True)["status"])

            second, second_refs = evidence.aggregate_stage_evidence(
                stage, "release", fingerprint, required, [], [])
            self.assertEqual(first, second)
            self.assertEqual(first_refs, second_refs)
            self.assertEqual(original, output.read_bytes())
            self.assertEqual("verified", release_evidence.audit_stage_archive(
                stage, required=True)["status"])
            self.assertEqual(archive_digest, hashlib.sha256(archive.read_bytes()).hexdigest())

        changed = json.loads(output.read_text())
        changed["host_aot_release_eligible"] = True
        output.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
        rejected = release_evidence.audit_stage_archive(stage, required=True)
        self.assertEqual("invalid", rejected["status"])
        self.assertIn("archive omits or changes audited stage artifact",
                      rejected["findings"][0]["message"])

    def test_post_ctest_aggregate_requires_capture_publish_test(self):
        with self.assertRaisesRegex(
                evidence.EvidenceError, "vulkan-fixedrun-capture-publish"):
            evidence.aggregate_stage_evidence(
                self.stage, "release", "f" * 64, [], [], [])

    def test_native_link_lookup_does_not_conflate_nested_product_only_builds(self):
        for target in ("vulkan_fixedrun_relu_validation", "vulkan_fixedrun_loader_relu"):
            relative = Path("test/Vulkan/native/CMakeFiles") / f"{target}.dir/link.txt"
            primary = self.stage / relative
            child = self.stage / "test/Vulkan/device-helper-product-only" / relative
            primary.parent.mkdir(parents=True)
            child.parent.mkdir(parents=True)
            primary.write_text("c++ primary.o -o primary\n")
            child.write_text("c++ child.o -o child\n")
            self.assertEqual(primary, evidence._native_target_link_file(self.stage, target))
            primary.unlink()
            with self.assertRaisesRegex(evidence.EvidenceError, "stage-local link command"):
                evidence._native_target_link_file(self.stage, target)

    def test_candidate_dynamic_dependencies_reject_compiler_and_private_dsos(self):
        evidence._audit_candidate_dynamic_dependencies(
            " 0x00000001 (NEEDED) Shared library: [libvulkan.so.1]\n"
            " 0x00000001 (NEEDED) Shared library: [libstdc++.so.6]\n")
        for name in ("libLLVM.so.21", "libNCNNVulkanRuntime.so",
                     "libprivate-device-runtime.so"):
            with self.subTest(name=name), self.assertRaises(evidence.EvidenceError):
                evidence._audit_candidate_dynamic_dependencies(
                    f" 0x00000001 (NEEDED) Shared library: [{name}]\n")

    def test_artifact_records_are_stage_relative_hashed_and_contained(self):
        path = self.stage / "evidence/raw.bin"
        path.parent.mkdir()
        path.write_bytes(b"raw")
        digest = hashlib.sha256(b"raw").hexdigest()
        accepted, actual = evidence._artifact(
            self.stage, {"path": "evidence/raw.bin", "sha256": digest}, "raw")
        self.assertEqual(path, accepted)
        self.assertEqual(digest, actual)
        for record in (
            {"path": "../outside", "sha256": digest},
            {"path": "/etc/passwd", "sha256": digest},
            {"path": "evidence/raw.bin", "sha256": "0" * 64},
        ):
            with self.subTest(record=record):
                with self.assertRaises(evidence.EvidenceError):
                    evidence._artifact(self.stage, record, "raw")

    def test_native_loader_phase_audit_requires_two_threads_and_abab(self):
        source_path = (Path(__file__).resolve().parent /
                       "native/fixedrun_native_loader.cpp")
        source = source_path.read_text(encoding="utf-8")
        evidence._audit_native_loader_phase_implementation(source)
        attacks = (
            source.replace("workers.emplace_back(worker, 1U, std::ref(*prepared_b));", ""),
            source.replace("context_b.get(), *prepared_b_repeat", "context_a.get(), *prepared_b_repeat"),
        )
        for invalid in attacks:
            with self.subTest(invalid=invalid[-200:]), self.assertRaises(evidence.EvidenceError):
                evidence._audit_native_loader_phase_implementation(invalid)

    def test_fixedrun_static_tables_are_closed_and_bound_to_hostprogram(self):
        program = {"pipelines": [], "weights": [], "inputs": [],
                   "outputs": [], "buffers": []}
        valid = ("constexpr std::array<VulkanDeviceLimitRequirements, 0> "
                 "kDeviceRequirements{{}};\n"
                 "[[maybe_unused]] constexpr std::array<Port, 0> kInputs{{}};\n"
                 "[[maybe_unused]] constexpr std::array<Port, 0> kOutputs{{}};\n")
        abi_body = ("const std::array<std::span<const float>, 0> inputs{{}};\n"
                    "const std::array<std::span<float>, 0> outputs{{}};\n")
        evidence._audit_generated_static_tables(valid, "", abi_body, program, [], self.stage)
        attacks = (
            valid + "constexpr std::array<std::uint32_t, 1> kDispatchTable{{UINT32_C(0)}};\n",
            valid + "constexpr std::array<std::uint32_t, 1> renamed{{UINT32_C(0)}};\n",
            valid + "static const RuntimeOperation kOperations[] = {};\n",
        )
        for source in attacks:
            with self.subTest(source=source[-100:]), self.assertRaises(evidence.EvidenceError):
                evidence._audit_generated_static_tables(
                    source, "", abi_body, program, [], self.stage)

    def test_fixedrun_plan_hostprogram_audit_rejects_mutated_hazard_trace(self):
        import copy
        import test_host_program_evidence

        legacy_fixture = test_host_program_evidence.HostProgramEvidenceTests()
        manifest, plan, shader_bytes, _ = legacy_fixture.fixture()
        program = manifest["host_program"]
        records = []
        for kernel_id, payload in sorted(shader_bytes.items()):
            records.append(self.put_bytes(f"host-program/shader-{kernel_id}.spv", payload))
        evidence._audit_host_program_against_plan(
            self.stage, plan, program, records, "fixture", manifest["plan_sha256"])

        changed = copy.deepcopy(program)
        changed["run_operations"].pop(6)
        changed["identity"] = evidence.host_program_audit.canonical_identity(changed)
        with self.assertRaises(evidence.EvidenceError):
            evidence._audit_host_program_against_plan(
                self.stage, plan, changed, records, "fixture", manifest["plan_sha256"])

    def test_fixed_run_body_rejects_schedule_loops_and_runtime_executor(self):
        source = '''struct model_context {
          ncnn_vk_status run() {
            vkCmdPipelineBarrier(command, 1, 2, 0, 0, nullptr, 1, &barrier, 0, nullptr);
            vkCmdCopyBuffer(command, source, destination, 1, &copy);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdDispatch(command, 2, 1, 1);
            return NCNN_VK_STATUS_SUCCESS;
          }
        };
        extern "C" ncnn_vk_status model_vk_run(model_context* context) {
          validateDiagnostic();
          return context->run();
        }'''
        body = evidence._run_body(source, "model")
        abi_body = evidence._abi_run_body(source, "model")
        evidence._audit_fixedrun_structure(source, body, abi_body)
        self.assertNotRegex(body, evidence.FORBIDDEN_RUN)
        self.assertNotRegex(body, r"\b(?:for|while|do)\s*(?:\(|\{)")
        for invalid in (
            source.replace("return NCNN_VK_STATUS_SUCCESS;",
                           "for (auto kernel : kernels) { dispatch(kernel); }"),
            source.replace("return NCNN_VK_STATUS_SUCCESS;",
                           "runtime.run(plan);"),
        ):
            with self.subTest(invalid=invalid):
                invalid_body = evidence._run_body(invalid, "model")
                with self.assertRaises(evidence.EvidenceError):
                    evidence._audit_fixedrun_structure(
                        invalid, invalid_body, evidence._abi_run_body(invalid, "model"))

    def test_known_fixture_inventory_is_closed(self):
        self.assertEqual(
            {"relu", "add_relu", "conv_relu", "add_split_relu",
             "split_concat", "reshape", "transpose"},
            set(evidence.FIXTURES))

    def test_raw_array_requires_exact_little_endian_float_bytes(self):
        import struct

        payload = struct.pack("<2f", 1.25, -3.5)
        path = self.stage / "two-floats.bin"
        path.write_bytes(payload)
        record = {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}
        values, _ = evidence._float_array(self.stage, record, 2, "raw")
        self.assertEqual([1.25, -3.5], values)
        with self.assertRaisesRegex(evidence.EvidenceError, "byte count"):
            evidence._float_array(self.stage, record, 1, "raw")

    def test_required_raw_schema_rejects_missing_device_runs(self):
        with self.assertRaisesRegex(evidence.EvidenceError, "raw JSON is missing"):
            evidence.audit_device_raw(self.stage, "release", "a" * 64,
                                      "llvmpipe", require_legacy=True)


if __name__ == "__main__":
    unittest.main()
