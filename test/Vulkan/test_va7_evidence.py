#!/usr/bin/env python3
"""Self-tests for the VA7 final product-switch evidence tooling.

These tests exercise the pure validation logic with synthetic data: the
32-ID catalog closure (VA0.1 through VA7.4), the profile-scoped required
CTest inventories, the VA7 product-entry evidence audit, and — the core
regression — a synthetic gate inventory that deliberately omits a
product-only unit test must be rejected by the audit, proving the checker
does not reproduce the old V6 registration gap.  They never compile a
model; the integration runs through the CTest registrations.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TEST_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


checker = load_module("ncnn_va7_checker", TEST_DIR / "check_release_evidence.py")
entry = load_module("ncnn_va7_entry", TEST_DIR / "run_va7_product_entry.py")

HOST_CONTRACT = dict(entry.HOST_CONTRACT)
FINGERPRINT = "a" * 64


def synthetic_entry_evidence() -> dict:
    """A well-formed product-entry evidence document."""
    vk_identity = "b" * 64
    cases = {
        "cpu_default": {
            "kind": "cpu-default", "returncode": 0,
            "published": sorted(name.format(model="entry_cpu")
                                for name in entry.CPU_ARTIFACTS),
            "artifacts": {"libentry_cpu.so": "c" * 64, "entry_cpu.h": "d" * 64},
        },
        "vulkan_backend": {
            "kind": "vulkan-fixed-run", "returncode": 0,
            "published": sorted(name.format(model="entry_vk")
                                for name in entry.VULKAN_ARTIFACTS),
            "manifest_contract": dict(HOST_CONTRACT),
            "host_codegen_identity": vk_identity,
        },
        "env_generic_runtime_noop": {
            "kind": "vulkan-fixed-run", "returncode": 0,
            "published": sorted(name.format(model="entry_vk")
                                for name in entry.VULKAN_ARTIFACTS),
            "manifest_contract": dict(HOST_CONTRACT),
            "host_codegen_identity": vk_identity,
        },
        "env_generic_runtime_cpu_default": {
            "kind": "cpu-default", "returncode": 0,
            "published": ["libentry_env_cpu.so", "entry_env_cpu.h"],
        },
    }
    for value in entry.INVALID_BACKEND_VALUES:
        cases[f"reject_backend_{value.replace('-', '_')}"] = {
            "kind": "invalid-backend", "returncode": 1, "published": [],
            "diagnostic_substring": entry.INVALID_BACKEND_DIAGNOSTIC,
        }
    for flag in entry.UNKNOWN_OPT_OUT_FLAGS:
        cases[f"reject_flag_{flag.lstrip('-').replace('-', '_')}"] = {
            "kind": "unknown-flag", "returncode": 1, "published": [],
            "diagnostic_substring": entry.UNKNOWN_FLAG_DIAGNOSTIC,
        }
    return {
        "schema_version": 1,
        "kind": "ncnn.vulkan_va7_product_entry_evidence",
        "default_backend": "cpu",
        "cases": cases,
        "structural_audit": {
            "forbidden_undefined_absent": True,
            "forbidden_needed_absent": True,
            "unclassified_needed_absent": True,
            "no_rpath_runpath": True,
        },
    }


def synthetic_va7_stage(profile: str, *, omit: frozenset[str] = frozenset(),
                        fingerprint: str = FINGERPRINT) -> dict:
    """A synthetic VA7 stage whose scope state is produced by the real
    required-inventory detector, so the composite tests exercise the same
    omission path a fresh gate audit would take."""
    names = sorted(checker._required_tests(profile, "VA7") - set(omit))
    stage = {
        "stage": f"synthetic-va7-{profile}", "profile": profile,
        "stage_id": "VA7", "source_fingerprint": fingerprint,
        "scope_passed": True, "findings": [], "va0_findings": [],
        "ctest": {"test_names": names, "skipped": [], "failed": []},
        "families": {"va7_final": {
            "status": "verified", "stage_id": "VA7", "role": "final-switch",
            "evidence": ["test/Vulkan/evidence/va7/product-entry/"
                         "product-entry-evidence.json"],
            "host_aot_release_eligible": False}},
        "raw_artifacts": [],
    }
    findings: list[dict] = []
    checker._check_required_tests(stage, findings)
    stage["findings"] = findings
    stage["scope_passed"] = not findings
    return stage


def build_synthetic_report(stages: list[dict], **kwargs) -> dict:
    with patch.object(checker, "audit_stage_gate", side_effect=list(stages)), \
            patch.object(checker, "audit_stage_evidence",
                         side_effect=lambda gate: gate), \
            patch.object(checker, "audit_stage_archive",
                         return_value={"status": "verified", "findings": []}):
        return checker.build_report(
            [Path(stage["stage"]) for stage in stages], **kwargs)


class CatalogClosureTests(unittest.TestCase):
    def test_va_catalog_covers_exactly_32_ids(self):
        va_ids = [identifier for identifier in checker.ID_CATALOG
                  if identifier.startswith("VA")]
        expected = [f"VA{stage}.{step}"
                    for stage in range(0, 8) for step in range(1, 5)]
        self.assertEqual(expected, sorted(va_ids))
        self.assertEqual(32, len(va_ids))

    def test_va7_ids_use_the_va7_family(self):
        for step in range(1, 5):
            spec = checker.ID_CATALOG[f"VA7.{step}"]
            self.assertEqual("va7", spec["evidence_family"])

    def test_va7_step_tests_carry_the_registered_ctests(self):
        self.assertEqual({"vulkan-va7-product-entry"},
                         set(checker.VA7_STEP_TESTS["VA7.1"]))
        self.assertEqual({"vulkan-va7-evidence-tests"},
                         set(checker.VA7_STEP_TESTS["VA7.3"]))
        self.assertEqual(set(), set(checker.VA7_STEP_TESTS["VA7.2"]))
        self.assertEqual(set(), set(checker.VA7_STEP_TESTS["VA7.4"]))
        self.assertEqual({"vulkan-va7-product-entry", "vulkan-va7-evidence-tests"},
                         set(checker.VA7_REQUIRED_TESTS))


class RequiredInventoryTests(unittest.TestCase):
    def test_dual_device_profile_keeps_full_cumulative_inventory(self):
        tests = checker._required_tests("release", "VA7")
        for name in ("vulkan-va7-product-entry", "vulkan-va7-evidence-tests",
                     "vulkan-va6-product-matrix", "vulkan-va6-host-faults",
                     "vulkan-va6-cross-icd", "vulkan-va5-deployment-nvidia",
                     "vulkan-va5-deployment-llvmpipe",
                     "vulkan-device-helper-add-nvidia",
                     "vulkan-lifecycle-llvmpipe", "vulkan-fixedrun-nvidia",
                     "ncnn-compile-vulkan-product",
                     "ncnn-compile-vulkan-product-llvmpipe",
                     "VulkanProductTest.ManifestPinsHostExecutionContractAndIdentityDomain"):
            self.assertIn(name, tests)

    def test_offline_profile_drops_every_device_bound_test(self):
        tests = checker._required_tests("offline", "VA7")
        self.assertIn("vulkan-va7-product-entry", tests)
        self.assertIn("vulkan-va6-product-matrix", tests)
        self.assertIn("ncnn-compile-vulkan-offline", tests)
        self.assertNotIn("vulkan-va6-cross-icd", tests)
        for name in tests:
            self.assertFalse(name.endswith(("-nvidia", "-llvmpipe")),
                             f"offline inventory keeps a device test: {name}")

    def test_software_profile_keeps_only_llvmpipe_device_tests(self):
        tests = checker._required_tests("software", "VA7")
        self.assertIn("vulkan-va7-product-entry", tests)
        self.assertIn("vulkan-device-helper-add-llvmpipe", tests)
        self.assertIn("vulkan-lifecycle-llvmpipe", tests)
        self.assertIn("vulkan-fixedrun-llvmpipe", tests)
        self.assertIn("vulkan-va5-deployment-llvmpipe", tests)
        self.assertNotIn("vulkan-va6-cross-icd", tests)
        for name in tests:
            self.assertFalse(name.endswith("-nvidia"),
                             f"software inventory keeps an NVIDIA test: {name}")


class OmittedProductUnitTestRejectedTests(unittest.TestCase):
    """The old V6 registration gap: a product-only unit test silently
    missing from the inventory must fail the audit, not pass unnoticed."""

    OMITTED = "VulkanProductTest.ManifestPinsHostExecutionContractAndIdentityDomain"

    def test_required_test_check_reports_the_omission(self):
        findings = []
        gate = {"profile": "release", "stage_id": "VA7",
                "ctest": {"test_names": sorted(
                    checker._required_tests("release", "VA7") - {self.OMITTED})}}
        checker._check_required_tests(gate, findings)
        codes = {item["code"] for item in findings}
        self.assertIn("required-ctest-missing", codes)
        missing = next(item for item in findings
                       if item["code"] == "required-ctest-missing")
        self.assertIn(self.OMITTED, missing["details"]["tests"])

    def test_complete_inventory_has_no_missing_test_finding(self):
        findings = []
        gate = {"profile": "release", "stage_id": "VA7",
                "ctest": {"test_names": sorted(
                    checker._required_tests("release", "VA7"))}}
        checker._check_required_tests(gate, findings)
        self.assertNotIn("required-ctest-missing",
                         {item["code"] for item in findings})

    def test_composite_with_omission_is_not_eligible(self):
        stages = [synthetic_va7_stage(profile, omit=frozenset({self.OMITTED}))
                  for profile in ("release", "sanitize", "offline", "software")]
        report = build_synthetic_report(stages, require_va7_final=True)
        self.assertFalse(report["va7_final_eligible"])
        self.assertFalse(report["host_aot_release_eligible"])
        codes = {item["code"] for item in report["blocking_findings"]}
        self.assertIn("va7-final-eligibility-required", codes)
        row = report["id_index"]["VA7.1"]
        self.assertNotEqual("passed", row["status"])

    def test_composite_with_complete_inventory_is_eligible(self):
        stages = [synthetic_va7_stage(profile)
                  for profile in ("release", "sanitize", "offline", "software")]
        report = build_synthetic_report(stages, require_va7_final=True)
        self.assertTrue(report["va7_final_eligible"])
        self.assertTrue(report["host_aot_release_eligible"])
        self.assertEqual("release-eligible-opt-in",
                         report["host_aot_release_decision"]["decision"])
        for step in range(1, 5):
            self.assertEqual("passed", report["id_index"][f"VA7.{step}"]["status"])
        self.assertEqual([], report["host_aot_blockers"])

    def test_composite_without_all_four_profiles_is_ineligible(self):
        stages = [synthetic_va7_stage("release"), synthetic_va7_stage("sanitize")]
        report = build_synthetic_report(stages, require_va7_final=True)
        self.assertFalse(report["va7_final_eligible"])
        self.assertFalse(report["host_aot_release_eligible"])
        blockers = report["va7_final_blockers"]
        self.assertTrue(any(item["code"] == "va7-required-profiles-missing"
                            for item in blockers))
        missing = next(item for item in blockers
                       if item["code"] == "va7-required-profiles-missing")
        self.assertEqual({"offline", "software"}, set(missing["profiles"]))

    def test_composite_with_fingerprint_mismatch_is_ineligible(self):
        stages = [synthetic_va7_stage("release"),
                  synthetic_va7_stage("sanitize"),
                  synthetic_va7_stage("offline"),
                  synthetic_va7_stage("software", fingerprint="e" * 64)]
        report = build_synthetic_report(stages, require_va7_final=True)
        self.assertFalse(report["va7_final_eligible"])
        self.assertTrue(any(
            item["code"] == "va7-source-fingerprint-mismatch"
            for item in report["va7_final_blockers"]))


class EntryEvidenceAuditTests(unittest.TestCase):
    def write_stage(self, directory: Path, evidence: dict | None,
                    profile: str = "offline") -> Path:
        stage = directory / f"stage-{profile}"
        (stage / "test/Vulkan/evidence/va7/product-entry").mkdir(parents=True)
        if evidence is not None:
            (stage / "test/Vulkan/evidence/va7/product-entry/"
             "product-entry-evidence.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        return stage

    def audit(self, stage: Path, profile: str = "offline"):
        findings = []
        gate = {"stage_id": "VA7", "profile": profile,
                "ctest": {"test_names": sorted(checker.VA7_REQUIRED_TESTS),
                          "skipped": [], "failed": []}}
        result = checker._audit_va7_final(stage, gate, [], findings)
        return result, findings

    def test_valid_evidence_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), synthetic_entry_evidence())
            result, findings = self.audit(stage)
        self.assertEqual("verified", result["status"])
        self.assertEqual([], findings)

    def test_missing_evidence_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), None)
            result, findings = self.audit(stage)
        self.assertEqual("blocked", result["status"])
        self.assertIn("va7-evidence-missing", {f["code"] for f in findings})

    def test_tampered_host_contract_is_blocked(self):
        document = synthetic_entry_evidence()
        document["cases"]["vulkan_backend"]["manifest_contract"][
            "host_implementation"] = "generic-plan-runtime"
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), document)
            result, findings = self.audit(stage)
        self.assertEqual("blocked", result["status"])
        self.assertIn("va7-vulkan-backend-invalid",
                      {f["code"] for f in findings})

    def test_accepted_opt_out_flag_is_blocked(self):
        document = synthetic_entry_evidence()
        document["cases"]["reject_flag_generic_runtime"] = {
            "kind": "unknown-flag", "returncode": 0, "published": []}
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), document)
            result, findings = self.audit(stage)
        self.assertEqual("blocked", result["status"])
        self.assertIn("va7-opt-out-not-rejected",
                      {f["code"] for f in findings})

    def test_missing_structural_audit_is_blocked(self):
        document = synthetic_entry_evidence()
        document["structural_audit"]["forbidden_undefined_absent"] = False
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), document)
            result, findings = self.audit(stage)
        self.assertEqual("blocked", result["status"])
        self.assertIn("va7-structural-audit-invalid",
                      {f["code"] for f in findings})

    def test_changed_identity_under_forged_env_is_blocked(self):
        document = synthetic_entry_evidence()
        document["cases"]["env_generic_runtime_noop"][
            "host_codegen_identity"] = "f" * 64
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), document)
            result, findings = self.audit(stage)
        self.assertEqual("blocked", result["status"])
        self.assertIn("va7-env-noop-invalid", {f["code"] for f in findings})

    def test_non_va7_stage_is_not_applicable(self):
        findings = []
        result = checker._audit_va7_final(
            Path("/unused"), {"stage_id": "VA6", "profile": "release"},
            [], findings)
        self.assertEqual("not-applicable", result["status"])
        self.assertEqual([], findings)

    def test_required_test_skip_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.write_stage(Path(tmp), synthetic_entry_evidence())
            findings = []
            gate = {"stage_id": "VA7", "profile": "offline",
                    "ctest": {"test_names": sorted(checker.VA7_REQUIRED_TESTS),
                              "skipped": ["vulkan-va7-evidence-tests"],
                              "failed": []}}
            result = checker._audit_va7_final(stage, gate, [], findings)
        self.assertEqual("blocked", result["status"])
        self.assertIn("va7-required-ctest-not-passed",
                      {f["code"] for f in findings})


class EntryRunnerSelfTests(unittest.TestCase):
    def test_runner_self_test_passes(self):
        self.assertEqual(0, entry.self_test())

    def test_evidence_kind_matches_the_audited_constant(self):
        self.assertEqual("ncnn.vulkan_va7_product_entry_evidence",
                         entry.EVIDENCE_KIND)
        self.assertEqual(
            ("test/Vulkan/evidence/va7/product-entry/"
             "product-entry-evidence.json",),
            checker.VA7_EVIDENCE_PATHS["software"])


if __name__ == "__main__":
    unittest.main()
