#!/usr/bin/env python3
"""Fail-closed tests for compiler-only Vulkan host-program evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_program_evidence as evidence


class HostProgramEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_path = Path(__file__).resolve().parents[1] / "Native" / "check_ncnn_compile_vulkan.py"
        spec = importlib.util.spec_from_file_location("native_vulkan_product_checker", native_path)
        cls.native_checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.native_checker)

    def fixture(self):
        shader = b"shader-zero"
        weight = b"weight-data!"
        shader_hash = hashlib.sha256(shader).hexdigest()
        weight_hash = hashlib.sha256(weight).hexdigest()
        plan = {
            "kind": "ncnn.vulkan_execution_plan", "schema_version": 1,
            "model": "fixture", "host_triple": "x86_64-linux-gnu",
            "device_target": "vulkan1.1-spirv1.0", "inputs": [0], "outputs": [2],
            "resources": [
                {"id": 0, "kind": "input", "name": "input", "shape": [4],
                 "element_type": "f32", "logical_layout": "CHW",
                 "physical_layout": "CHW", "host_layout": "CHW",
                 "layout_transform": None, "weight_sha256": "", "byte_size": 16,
                 "first_kernel": 0, "last_kernel": 0},
                {"id": 1, "kind": "intermediate", "name": "temporary", "shape": [4],
                 "element_type": "f32", "logical_layout": "row_major",
                 "physical_layout": "row_major", "host_layout": "",
                 "layout_transform": None, "weight_sha256": "", "byte_size": 16,
                 "first_kernel": 0, "last_kernel": 1},
                {"id": 2, "kind": "output", "name": "output", "shape": [4],
                 "element_type": "f32", "logical_layout": "CHW",
                 "physical_layout": "CHW", "host_layout": "CHW",
                 "layout_transform": None, "weight_sha256": "", "byte_size": 16,
                 "first_kernel": 1, "last_kernel": 1},
                {"id": 3, "kind": "weight", "name": "weight", "shape": [3],
                 "element_type": "f32", "logical_layout": "row_major",
                 "physical_layout": "row_major", "host_layout": "",
                 "layout_transform": None, "weight_sha256": weight_hash, "byte_size": 12,
                 "first_kernel": 0, "last_kernel": 0},
            ],
            "kernels": [
                {"id": 0, "operation": "first", "entry_point": "kernel_0",
                 "local_size": [4, 1, 1], "reads": [0, 3], "writes": [1],
                 "bindings": [{"set": 0, "binding": 0, "resource_id": 0, "access": "read"},
                              {"set": 0, "binding": 1, "resource_id": 3, "access": "read"},
                              {"set": 0, "binding": 2, "resource_id": 1, "access": "write"}],
                 "push_constants": []},
                {"id": 1, "operation": "second", "entry_point": "kernel_1",
                 "local_size": [4, 1, 1], "reads": [1], "writes": [2],
                 "bindings": [{"set": 0, "binding": 0, "resource_id": 1, "access": "read"},
                              {"set": 0, "binding": 1, "resource_id": 2, "access": "write"}],
                 "push_constants": []},
            ],
        }
        program = {
            "schema": evidence.SCHEMA, "codegen_version": evidence.CODEGEN,
            "helper_contract": evidence.HELPER_CONTRACT,
            "synchronization_version": evidence.SYNC_VERSION,
            "canonical_encoding": evidence.CANONICAL_ENCODING,
            "abi_version": 1, "model": "fixture", "host_triple": plan["host_triple"],
            "device_target": plan["device_target"], "options": copy.deepcopy(evidence.OPTIONS),
            "resources": copy.deepcopy(plan["resources"]),
            "buffers": [
                {"id": 0, "resource_id": 0, "kind": "device", "scope": "context", "byte_size": 16},
                {"id": 1, "resource_id": 1, "kind": "device", "scope": "context", "byte_size": 16},
                {"id": 2, "resource_id": 2, "kind": "device", "scope": "context", "byte_size": 16},
                {"id": 3, "resource_id": 3, "kind": "device", "scope": "context", "byte_size": 12},
                {"id": 4, "resource_id": 0, "kind": "input-staging", "scope": "context", "byte_size": 16},
                {"id": 5, "resource_id": 2, "kind": "output-staging", "scope": "context", "byte_size": 16},
                {"id": 6, "resource_id": 3, "kind": "weight-staging", "scope": "create-upload", "byte_size": 12},
            ],
            "inputs": [{"resource_id": 0, "device_buffer": 0, "staging_buffer": 4, "elements": 4}],
            "outputs": [{"resource_id": 2, "device_buffer": 2, "staging_buffer": 5, "elements": 4}],
            "pipelines": [],
            "weights": [{"resource_id": 3, "byte_size": 12,
                         "logical_sha256": weight_hash, "physical_sha256": weight_hash}],
            "initialize": [], "destroy": [], "create_operations": [],
            "run_operations": [], "identity": "",
        }
        for kernel_id in (0, 1):
            kernel = plan["kernels"][kernel_id]
            shader_id = shader_hash if kernel_id == 0 else hashlib.sha256(b"shader-one").hexdigest()
            shader_bytes = shader if kernel_id == 0 else b"shader-one"
            pipeline_id = kernel_id
            program["pipelines"].append({
                "id": pipeline_id, "kernel_id": kernel_id,
                "binary_symbol": f"fixture_vulkan_k{kernel_id}",
                "shader_sha256": shader_id, "shader_size_bytes": len(shader_bytes),
                "interface": {"entry_point": kernel["entry_point"],
                              "local_size": kernel["local_size"],
                              "bindings": [{"set": item["set"],
                                            "binding": item["binding"],
                                            "access": item["access"],
                                            "byte_size": next(resource["byte_size"]
                                                for resource in plan["resources"]
                                                if resource["id"] == item["resource_id"])}
                                           for item in kernel["bindings"]],
                              "push_constants": []},
                "bindings": [{"set": item["set"], "binding": item["binding"],
                              "buffer": item["resource_id"], "access": item["access"]}
                             for item in kernel["bindings"]],
                "push_constants": [], "dispatch": [1, 1, 1],
            })
        buffer_map = {item["id"]: item for item in program["buffers"]}
        program["initialize"] = ([{"kind": "buffer", "id": item["id"]}
                                   for item in program["buffers"]] +
                                  [{"kind": "pipeline", "id": item["id"]}
                                   for item in program["pipelines"]])
        program["destroy"] = [item for item in reversed(program["initialize"])
                              if not (item["kind"] == "buffer" and
                                      buffer_map[item["id"]]["scope"] == "create-upload")]
        program["create_operations"] = [
            {"op": "upload", "staging_buffer": 6, "source": "weight", "source_id": 3,
             "byte_size": 12},
            {"op": "barrier", "source_stage": "host", "destination_stage": "transfer",
             "buffers": [{"buffer": 6, "source_access": "host-write",
                          "destination_access": "transfer-read", "offset": 0, "byte_size": 12}]},
            {"op": "copy", "source": 6, "destination": 3, "source_offset": 0,
             "destination_offset": 0, "byte_size": 12},
        ]
        program["run_operations"] = [
            {"op": "upload", "staging_buffer": 4, "source": "input", "source_id": 0,
             "byte_size": 16},
            {"op": "barrier", "source_stage": "host", "destination_stage": "transfer",
             "buffers": [{"buffer": 4, "source_access": "host-write",
                          "destination_access": "transfer-read", "offset": 0, "byte_size": 16}]},
            {"op": "copy", "source": 4, "destination": 0, "source_offset": 0,
             "destination_offset": 0, "byte_size": 16},
            {"op": "bind-pipeline", "pipeline": 0},
            {"op": "bind-descriptors", "pipeline": 0, "first_set": 0, "set_count": 1},
            {"op": "dispatch", "pipeline": 0, "groups": [1, 1, 1]},
            {"op": "barrier", "source_stage": "compute", "destination_stage": "compute",
             "buffers": [{"buffer": 1, "source_access": "shader-write",
                          "destination_access": "shader-read", "offset": 0, "byte_size": 16}]},
            {"op": "bind-pipeline", "pipeline": 1},
            {"op": "bind-descriptors", "pipeline": 1, "first_set": 0, "set_count": 1},
            {"op": "dispatch", "pipeline": 1, "groups": [1, 1, 1]},
            {"op": "copy", "source": 2, "destination": 5, "source_offset": 0,
             "destination_offset": 0, "byte_size": 16},
            {"op": "barrier", "source_stage": "transfer", "destination_stage": "host",
             "buffers": [{"buffer": 5, "source_access": "transfer-write",
                          "destination_access": "host-read", "offset": 0, "byte_size": 16}]},
            {"op": "publish", "staging_buffer": 5, "output_index": 0, "byte_size": 16},
        ]
        device_map = {item["resource_id"]: item["id"] for item in program["buffers"]
                      if item["kind"] == "device"}
        pipeline_map = {item["kernel_id"]: item for item in program["pipelines"]}
        program["create_operations"], program["run_operations"] = (
            evidence._legacy_operation_trace(plan, program, device_map,
                                             buffer_map, pipeline_map))
        manifest = {"model": "fixture", "abi_version": 1,
                    "schema_version": 1, "kind": "ncnn.vulkan_model_manifest",
                    "host_program_usage": "compiler-only-diagnostic",
                    "host_implementation": "generic-plan-runtime",
                    "host_codegen_version": 1,
                    "execution_plan_usage": "embedded-runtime-required",
                    "weights": [{"resource_id": 3, "sha256": weight_hash,
                                 "size_bytes": len(weight)}],
                    "shaders": [
                        {"kernel_id": 0, "entry_point": "kernel_0",
                         "binary_symbol": "fixture_vulkan_k0", "sha256": shader_hash,
                         "size_bytes": len(shader)},
                        {"kernel_id": 1, "entry_point": "kernel_1",
                         "binary_symbol": "fixture_vulkan_k1",
                         "sha256": hashlib.sha256(b"shader-one").hexdigest(),
                         "size_bytes": len(b"shader-one")},
                    ], "plan_sha256": hashlib.sha256(
                        json.dumps(plan).encode("utf-8")).hexdigest()}
        program["identity"] = evidence.canonical_identity(program)
        manifest["host_program"] = program
        manifest["host_program_identity"] = program["identity"]
        return manifest, plan, {0: shader, 1: b"shader-one"}, {3: weight}

    def test_canonical_identity_and_independent_legacy_trace_pass(self):
        manifest, plan, shaders, weights = self.fixture()
        result = evidence.validate_host_program(
            manifest, plan, shader_bytes=shaders, weight_bytes=weights,
            plan_sha256=manifest["plan_sha256"])
        self.assertEqual("verified", result["status"])
        self.assertEqual(2, result["pipeline_count"])
        trace = result["legacy_symbolic_trace"]
        self.assertEqual([1], trace["kernels"][1]["compute_barrier_buffers"])
        self.assertEqual([1, 1, 1], trace["kernels"][0]["dispatch"])
        self.assertTrue(result["shader_raw_bytes_verified"])
        self.assertTrue(result["weight_raw_bytes_verified"])

    def test_missing_program_op_identity_version_and_raw_bytes_fail_closed(self):
        manifest, plan, shaders, weights = self.fixture()
        cases = []
        missing = copy.deepcopy(manifest)
        del missing["host_program"]
        cases.append(("missing program", missing, plan, shaders, weights))
        missing_op = copy.deepcopy(manifest)
        missing_op["host_program"]["run_operations"] = [
            op for op in missing_op["host_program"]["run_operations"]
            if op["op"] != "dispatch"]
        missing_op["host_program"]["identity"] = evidence.canonical_identity(
            missing_op["host_program"])
        missing_op["host_program_identity"] = missing_op["host_program"]["identity"]
        cases.append(("missing dispatch", missing_op, plan, shaders, weights))
        no_identity = copy.deepcopy(manifest)
        del no_identity["host_program_identity"]
        cases.append(("missing identity", no_identity, plan, shaders, weights))
        unknown = copy.deepcopy(manifest)
        unknown["host_program"]["schema"] = "future/v99"
        cases.append(("unknown schema", unknown, plan, shaders, weights))
        for label, changed, actual_plan, actual_shaders, actual_weights in cases:
            with self.subTest(label=label), self.assertRaises(evidence.EvidenceError):
                evidence.validate_host_program(changed, actual_plan,
                                              shader_bytes=actual_shaders,
                                              weight_bytes=actual_weights)
        for bad_shaders, bad_weights in (({}, weights), (shaders, {})):
            with self.assertRaises(evidence.EvidenceError):
                evidence.validate_host_program(manifest, plan,
                                               shader_bytes=bad_shaders,
                                               weight_bytes=bad_weights)

    def test_native_recorder_writer_emits_loadable_json_with_real_newline(self):
        manifest, plan, shaders, weights = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vulkan_dir = root / "vulkan"
            reconstruction_dir = vulkan_dir / "shader-reconstruction"
            reconstruction_dir.mkdir(parents=True)
            (vulkan_dir / "paired_relu_vulkan.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            (vulkan_dir / "paired_relu_vulkan.plan.json").write_text(
                json.dumps(plan), encoding="utf-8")
            (reconstruction_dir / "weights.json").write_text(json.dumps({
                "weights": [{"resource_id": resource_id, "bytes": raw.hex()}
                            for resource_id, raw in weights.items()]}),
                encoding="utf-8")
            for kernel_id, raw in shaders.items():
                (reconstruction_dir / f"kernel-{kernel_id}.spv").write_bytes(raw)
            reference = self.native_checker.record_host_program_evidence(
                SimpleNamespace(work_dir=root), vulkan_dir, reconstruction_dir)
            raw_record = (root / reference["file"]).read_bytes()
            parsed = json.loads(raw_record)
        self.assertTrue(raw_record.endswith(bytes([10])))
        self.assertEqual("ncnn.vulkan_host_program_evidence", parsed["kind"])
        self.assertEqual(manifest["host_program_identity"],
                         parsed["host_program_identity"])

    def test_identity_changes_and_recomputed_mutation_does_not_hide_bad_operations(self):
        manifest, plan, shaders, weights = self.fixture()
        changed = copy.deepcopy(manifest)
        changed["host_program"]["run_operations"][-1]["output_index"] = 7
        with self.assertRaisesRegex(evidence.EvidenceError, "canonical identity"):
            evidence.validate_host_program(changed, plan, shader_bytes=shaders,
                                           weight_bytes=weights)
        changed["host_program"]["identity"] = evidence.canonical_identity(changed["host_program"])
        changed["host_program_identity"] = changed["host_program"]["identity"]
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_host_program(changed, plan, shader_bytes=shaders,
                                           weight_bytes=weights)

    def test_candidate_cannot_be_misdeclared_as_aot(self):
        manifest, plan, shaders, weights = self.fixture()
        manifest["host_implementation"] = "generic-plan-runtime"
        manifest["host_codegen_version"] = 1
        manifest["execution_plan_usage"] = "embedded-runtime-required"
        manifest["host_aot_candidate"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_host_program(manifest, plan, shader_bytes=shaders,
                                           weight_bytes=weights)

    def test_shader_hash_dispatch_hazard_resource_and_identity_mutations_fail(self):
        mutators = (
            lambda p: p["pipelines"][0].update(shader_sha256="0" * 64),
            lambda p: p["pipelines"][1].update(dispatch=[2, 1, 1]),
            lambda p: p["run_operations"].pop(6),
            lambda p: p["resources"][0].update(byte_size=8),
            lambda p: p["resources"][0].update(first_kernel=False),
            lambda p: p["pipelines"][0]["interface"]["bindings"][0].update(set=False),
            lambda p: p["pipelines"][0].update(dispatch=[True, 1, 1]),
            lambda p: p.update(abi_version=True),
            lambda p: p["run_operations"][0].update(unrecognized="field"),
            lambda p: p.update(synchronization_version="future/v2"),
        )
        for mutate in mutators:
            manifest, plan, shaders, weights = self.fixture()
            mutate(manifest["host_program"])
            manifest["host_program"]["identity"] = evidence.canonical_identity(
                manifest["host_program"])
            manifest["host_program_identity"] = manifest["host_program"]["identity"]
            with self.assertRaises(evidence.EvidenceError):
                evidence.validate_host_program(manifest, plan,
                                               shader_bytes=shaders,
                                               weight_bytes=weights)

        manifest, plan, shaders, weights = self.fixture()
        manifest["abi_version"] = True
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_host_program(manifest, plan, shader_bytes=shaders,
                                           weight_bytes=weights)

        plan_mutators = (
            lambda p: p["resources"][0].update(first_kernel=False),
            lambda p: p["kernels"][0]["bindings"][0].update(set=False),
            lambda p: p["kernels"][0]["local_size"].__setitem__(0, False),
            lambda p: p["inputs"].__setitem__(0, False),
        )
        for mutate in plan_mutators:
            manifest, plan, shaders, weights = self.fixture()
            mutate(plan)
            with self.assertRaises(evidence.EvidenceError):
                evidence.validate_host_program(manifest, plan,
                                               shader_bytes=shaders,
                                               weight_bytes=weights)


if __name__ == "__main__":
    unittest.main(verbosity=2)
