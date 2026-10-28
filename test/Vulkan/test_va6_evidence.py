#!/usr/bin/env python3
"""Self-tests for the VA6 matrix/faults/cross-ICD evidence helpers.

These tests exercise the pure validation logic of the VA6 runners with
synthetic data: the product-matrix fixture serialization and case inventory,
the frozen diagnostic substrings, the host-faults wrapper identity check, and
the cross-ICD status handling.  They never compile a model; the integration
runs through the CTest registrations.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

TEST_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


matrix = load_module("ncnn_va6_matrix", TEST_DIR / "run_va6_product_matrix.py")
cross_icd = load_module("ncnn_va6_cross_icd", TEST_DIR / "run_va6_cross_icd.py")
checker = load_module("ncnn_va6_checker", TEST_DIR / "check_release_evidence.py")

HOST_CONTRACT = {
    "host_implementation": "model-fixed-run",
    "host_codegen_version": 2,
    "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
    "execution_plan_usage": "build-evidence-only",
}


class ProductMatrixCaseTests(unittest.TestCase):
    def test_negative_and_positive_inventories_are_disjoint(self):
        cases = matrix.matrix_cases()
        names = [case["name"] for case in cases]
        self.assertEqual(len(names), len(set(names)))
        negatives = [case for case in cases if case["kind"] == "negative"]
        positives = [case for case in cases if case["kind"] == "positive"]
        self.assertTrue(negatives)
        self.assertTrue(positives)
        for case in negatives:
            self.assertTrue(case["expect_any"], f"{case['name']} lacks a "
                             "frozen diagnostic substring")

    def test_param_serialization_covers_all_inputs(self):
        param = matrix.make_param(
            [("top", (1, 2, 2)), ("bottom", (2, 2, 2))],
            [{"line": "Concat join 2 1 top bottom output 0=0",
              "tops": ["output"]}])
        lines = param.strip().splitlines()
        self.assertEqual(lines[0], "7767517")
        self.assertEqual(lines[1], "3 3")
        self.assertIn("Input input_top 0 1 top 0=2 1=2 2=1", lines[2])
        self.assertIn("Input input_bottom 0 1 bottom 0=2 1=2 2=2", lines[3])

    def test_dynamic_input_has_no_dimension_parameters(self):
        param = matrix.make_param([("data", None)], [])
        self.assertIn("Input input_data 0 1 data", param)

    def test_conv_weights_use_the_ncnn_flag_word(self):
        import struct
        payload = matrix.conv_weights(2, 1, 2, bias=True, seed=3)
        flag = struct.unpack("<I", payload[:4])[0]
        self.assertEqual(flag, 0)
        element_count = 2 * 1 * 1 * 2 + 2
        self.assertEqual(len(payload), 4 + 4 * element_count)


class CrossIcdLoaderArgTests(unittest.TestCase):
    def test_device_auto_is_the_max_ordinal(self):
        self.assertEqual(0xFFFFFFFF, 4294967295)
        self.assertNotEqual(cross_icd.MAX_ABS_ERROR, 0.0)

    def test_host_contract_pins_the_fixed_run_codegen(self):
        self.assertEqual(cross_icd.HOST_CONTRACT, HOST_CONTRACT)
        self.assertEqual(matrix.HOST_CONTRACT, HOST_CONTRACT)


class EvidenceSchemaTests(unittest.TestCase):
    def test_evidence_kinds_are_distinct_and_versioned(self):
        kinds = {
            matrix.EVIDENCE_KIND,
            "ncnn.vulkan_va6_host_faults_evidence",
            cross_icd.EVIDENCE_KIND,
        }
        self.assertEqual(len(kinds), 3)
        self.assertEqual(matrix.SCHEMA_VERSION, 1)
        self.assertEqual(cross_icd.SCHEMA_VERSION, 1)


class WrapperIdentityTests(unittest.TestCase):
    def test_wrapper_identity_is_extracted_and_checked(self):
        import hashlib
        faults = load_module("ncnn_va6_faults",
                             TEST_DIR / "run_va6_host_faults.py")
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vulkan_model_wrapper.cpp"
            identity = hashlib.sha256(b"x").hexdigest()
            wrapper.write_text(
                "// Fixed-run codegen: ncnn-vulkan-fixed-run-codegen/v1\n"
                f"// Codegen identity: {identity}\n"
                "int main() { return 0; }\n")
            record = faults.check_wrapper_identity(wrapper, identity)
            self.assertEqual(record["codegen_identity"], identity)
            with self.assertRaises(faults.HostFaultsError):
                faults.check_wrapper_identity(wrapper, "f" * 64)


class AuditWiringTests(unittest.TestCase):
    """Synthetic-gate checks for the VA6 family in the evidence checker."""

    @staticmethod
    def _write_evidence(root: Path):
        matrix_doc = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_va6_product_matrix_evidence",
            "host_contract": dict(HOST_CONTRACT),
            "negative_cases": ["broadcast_add"],
            "positive_controls": ["relu"],
            "cases": {
                "broadcast_add": {"kind": "negative", "returncode": 1,
                                  "published": []},
                "relu": {"kind": "positive", "returncode": 0,
                         "published": ["a", "b", "c", "d"],
                         "model": "va6_relu",
                         "host_codegen_identity": "e" * 64},
            },
        }
        faults_doc = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_va6_host_faults_evidence",
            "model": "va6_faults",
            "host_codegen_identity": "e" * 64,
            "run_dispatches": 4,
            "run_copies": 3,
            "wrapper": {"wrapper_sha256": "a" * 64},
            "product_dependency_audit": {"forbidden_absent": True},
            "fault_test": {"returncode": 0,
                           "stdout": "Vulkan VA6 product host fault tests "
                                     "passed\n"},
        }
        cross_doc = {
            "schema_version": 1,
            "kind": "ncnn.vulkan_va6_cross_icd_evidence",
            "icds": [{"name": "nvidia"}, {"name": "llvmpipe"}],
            "scenarios": {
                fixture: {
                    "devices": {
                        "nvidia": {"device": {"device_name": "GPU-A"},
                                   "auto": {"result": {"run_status": 0},
                                            "numeric": {"max_abs": 0.0}},
                                   "explicit_ordinal_0": {
                                       "result": {"run_status": 0},
                                       "numeric": {"max_abs": 0.0}},
                                   "invalid_ordinal": {"result": {
                                       "create_status": 2,
                                       "context_null": True}}},
                        "llvmpipe": {"device": {"device_name": "GPU-B"},
                                     "auto": {"result": {"run_status": 0},
                                              "numeric": {"max_abs": 0.0}},
                                     "explicit_ordinal_0": {
                                         "result": {"run_status": 0},
                                         "numeric": {"max_abs": 0.0}},
                                     "invalid_ordinal": {"result": {
                                         "create_status": 2,
                                         "context_null": True}}},
                    },
                    "no_icd": {"result": {"create_status": 2,
                                          "context_null": True,
                                          "destroy_null_status": 0}},
                }
                for fixture in ("relu", "conv_relu")
            },
        }
        matrix_root = root / "test/Vulkan/evidence/va6/matrix"
        faults_root = root / "test/Vulkan/evidence/va6/faults"
        cross_root = root / "test/Vulkan/evidence/va6/cross-icd"
        for directory, document in (
                (matrix_root, matrix_doc), (faults_root, faults_doc),
                (cross_root, cross_doc)):
            directory.mkdir(parents=True, exist_ok=True)
            name = ("product-matrix-evidence.json"
                    if directory is matrix_root else
                    "host-faults-evidence.json"
                    if directory is faults_root else
                    "cross-icd-evidence.json")
            (directory / name).write_text(json.dumps(document))

    @staticmethod
    def _gate(root: Path, *, stage_id="VA6", profile="release"):
        required = set()
        for name in ("VA1_REQUIRED_TESTS", "VA2_REQUIRED_TESTS",
                     "VA3_REQUIRED_TESTS", "VA5_REQUIRED_TESTS"):
            required.update(getattr(checker, name))
        names = sorted(required |
                       checker.VA6_REQUIRED_TESTS |
                       {"vulkan-va6-cross-icd"})
        return {"stage": str(root), "profile": profile, "stage_id": stage_id,
                "scope_passed": True,
                "archive_integrity": {"status": "verified"},
                "findings": [],
                "ctest": {"test_names": names, "skipped": [], "failed": []}}

    def test_valid_va6_stage_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_evidence(root)
            findings = []
            family = checker._audit_va6_matrix(
                root, self._gate(root), [], findings)
            self.assertEqual("verified", family["status"], findings)
            self.assertEqual([], findings)

    def test_missing_test_and_bad_negative_case_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_evidence(root)
            matrix_path = (root / "test/Vulkan/evidence/va6/matrix/"
                                 "product-matrix-evidence.json")
            document = json.loads(matrix_path.read_text())
            document["cases"]["broadcast_add"]["returncode"] = 0
            matrix_path.write_text(json.dumps(document))
            findings = []
            family = checker._audit_va6_matrix(
                root, self._gate(root), [], findings)
            self.assertEqual("blocked", family["status"])
            codes = {item["code"] for item in findings}
            self.assertIn("va6-negative-case-invalid", codes)

    def test_cross_icd_requirement_follows_the_configured_devices(self):
        dual = checker._required_tests("release", "VA6")
        single = checker._required_tests("software", "VA6")
        self.assertIn("vulkan-va6-cross-icd", dual)
        self.assertNotIn("vulkan-va6-cross-icd", single)
        self.assertIn("vulkan-va6-product-matrix", single)

    def test_stage_id_va6_is_accepted_by_required_test_scope(self):
        findings = []
        checker._check_required_tests(
            {"profile": "release", "stage_id": "VA6",
             "ctest": {"test_names":
                       sorted(checker._required_tests("release", "VA6"))}},
            findings)
        self.assertEqual([], findings)
        self.assertIn("vulkan-va6-product-matrix",
                      checker._required_tests("release", "VA6"))


if __name__ == "__main__":
    unittest.main()
