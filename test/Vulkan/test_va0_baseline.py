#!/usr/bin/env python3
"""Synthetic regression tests for VA0 frozen-reference capture and indexing."""

import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch
from unittest.mock import patch

import check_release_evidence as audit
import run_gate


class Va0LegacyReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "frozen-source"
        self.source.mkdir()
        self._make_source()

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, relative, payload):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _make_source(self):
        names = [f"Legacy.Case{index:03d}" for index in range(523)] + [
            "PerformanceModel.PPLCNetDocOriInt8",
            "PerformanceModel.PPLCNetTextlineOriInt8",
        ]
        inventory_path = self.source / "test/gate-tests.json"
        inventory_path.parent.mkdir(parents=True)
        inventory_path.write_text(json.dumps({"tests": [{"name": name} for name in names]}))
        suites = ET.Element("testsuites")
        suite = ET.SubElement(suites, "testsuite", name="CTest", tests="525",
                              failures="0", errors="0", skipped="2")
        for name in names:
            case = ET.SubElement(suite, "testcase", name=name, classname="CTest")
            if name.startswith("PerformanceModel."):
                ET.SubElement(case, "skipped", message="pre-existing upstream skip")
        junit_path = self.source / "gate-ctest.xml"
        junit_path.write_bytes(ET.tostring(suites, encoding="utf-8"))
        lasttest_path = self.source / "gate-ctest-lasttest.log"
        lasttest_path.write_text("".join(
            f"{index}/525 Testing: {name}\n" for index, name in enumerate(names, 1)))
        ctest_log = self.put("gate-ctest.log", b"CTest completed\n")
        status = {
            "role": "legacy-reference", "stage_id": "VA0.1",
            "git_revision": run_gate.LEGACY_REFERENCE_REVISION,
            "worktree_porcelain": "",
            "source_fingerprint": run_gate.LEGACY_REFERENCE_FINGERPRINT,
            "source_fingerprint_algorithm": "sha256-git-files-v1",
            "host_aot_candidate": False,
        }
        self.put("gate-va0-legacy-source.json", json.dumps(status).encode())
        self.put("release-evidence.json", json.dumps({
            "kind": "ncnn.vulkan_release_evidence",
            "source_fingerprint": run_gate.LEGACY_REFERENCE_FINGERPRINT,
            "blocking_findings": [],
            "stages": [{"profile": "release", "scope_passed": True, "findings": []}],
        }).encode())
        state = {
            "schema_version": 1, "kind": "ncnn.vulkan_gate", "profile": "release",
            "completed": True,
            "source_fingerprint": run_gate.LEGACY_REFERENCE_FINGERPRINT,
            "git_revision": run_gate.LEGACY_REFERENCE_REVISION,
            "steps": [{"name": name, "exit_code": 0} for name in run_gate.STEP_NAMES],
            "inventory": {"path": "test/gate-tests.json",
                          "sha256": run_gate.sha256(inventory_path)},
            "junit": {"path": "gate-ctest.xml", "sha256": run_gate.sha256(junit_path)},
            "lasttest": {"path": "gate-ctest-lasttest.log",
                         "sha256": run_gate.sha256(lasttest_path)},
        }
        self.put("gate-state.json", json.dumps(state).encode())

        for key, spec in run_gate.LEGACY_REFERENCE_FIXTURES.items():
            source_root = self.source / spec["source"]
            source_root.mkdir(parents=True, exist_ok=True)
            self.put(f"{spec['source']}/evidence.json", b'{"synthetic":true}\n')
            for name in spec.get("files", ()):
                self.put(f"{spec['source']}/{name}", (key + "/" + name).encode())
            for tree in spec.get("trees", ()):
                if tree == ".":
                    tree_root = source_root
                else:
                    tree_root = source_root / tree
                if tree == "cpu":
                    files = ("libpaired_relu.so", "paired_relu.h")
                elif tree == "vulkan":
                    files = ("libpaired_relu_vulkan.so", "paired_relu_vulkan.h",
                             "paired_relu_vulkan.json", "paired_relu_vulkan.plan.json")
                else:
                    files = ()
                for name in files:
                    target = tree_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((key + "/" + tree + "/" + name).encode())
                if tree in {"vulkan", "."}:
                    shader = tree_root / "spirv-validation/shader-0000.spv"
                    shader.parent.mkdir(parents=True, exist_ok=True)
                    shader.write_bytes(b"synthetic SPIR-V")
                    raw = tree_root / "trials/seed-0/input-0.f32"
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(struct.pack("<f", 1.0))
            for source_input in spec.get("source_inputs", ()):
                self.put(source_input, (key + "/" + Path(source_input).name).encode())
            suite_root = run_gate.LEGACY_REFERENCE_SUITE_FILES.get(key)
            if suite_root:
                self.put(f"{suite_root}/evidence.json", b'{"synthetic":true}\n')
                self.put(f"{suite_root}/vulkaninfo-summary.txt", b"synthetic device summary\n")

        files = {}
        for path in sorted(self.source.rglob("*")):
            if path.is_file() and path.name != "gate-archive-manifest.json":
                files[path.relative_to(self.source).as_posix()] = run_gate.sha256(path)
        archive_manifest = {"schema_version": 1,
                            "kind": "ncnn.vulkan_evidence_archive", "files": files}
        self.put("gate-archive-manifest.json", json.dumps(archive_manifest).encode())

    def snapshot(self, target):
        target.mkdir()
        reference = run_gate.snapshot_legacy_baseline(self.source, target)
        state = {"legacy_reference": reference}
        return state

    def test_snapshot_is_stage_local_archived_and_validated_after_source_removal(self):
        target = self.root / "fresh-stage"
        state = self.snapshot(target)
        self.assertEqual(4, state["legacy_reference"]["fixture_count"])
        snapshot = json.loads((target / state["legacy_reference"]["path"]).read_text())
        self.assertEqual(run_gate.LEGACY_REFERENCE_FIXTURES.keys(), snapshot["fixtures"].keys())
        self.assertEqual(8, state["legacy_reference"]["fixture_profile_record_count"])
        self.assertEqual("legacy-reference", snapshot["role"])
        self.assertTrue(all(path.resolve().is_relative_to(target.resolve())
                            for path in (target / "legacy-reference").rglob("*") if path.is_file()))
        self.source.rename(self.root / "removed-frozen-source")

        refs, findings = [], []
        baseline = audit._audit_legacy_reference(target, state, refs, findings)
        self.assertIsNotNone(baseline)
        self.assertEqual([], findings)
        self.assertGreater(len(refs), 80)

        archive = self.root / "fresh-stage.evidence.tar.gz"
        run_gate.archive_stage(target, archive)
        with tarfile.open(archive, "r:gz") as bundle:
            members = set(bundle.getnames())
        self.assertIn("legacy-reference/snapshot.json", members)
        self.assertIn("legacy-reference/source/gate-state.json", members)

    def test_relu_comparison_propagates_product_artifact_mismatch(self):
        roots = [self.root / "relu-old", self.root / "relu-current"]
        values = [1.0] * 257
        outputs = [0.0] * 257
        for root in roots:
            root.mkdir()
            (root / "model.param").write_bytes(b"paired relu param")
            (root / "model.bin").write_bytes(b"")
            records = {}
            for name, array in (("input.f32", values), ("golden.f32", outputs),
                                ("cpu.f32", outputs), ("vulkan.f32", outputs)):
                (root / name).write_bytes(struct.pack(f"<{len(array)}f", *array))
                role = {"input.f32": "input", "golden.f32": "scalar_golden",
                        "cpu.f32": "cpu_output", "vulkan.f32": "vulkan_output"}[name]
                records[role] = {"file": name}
            cpu = root / "cpu"
            vulkan = root / "vulkan"
            cpu.mkdir()
            vulkan.mkdir()
            cpu_manifest = {"inputs": [{"element_type": "f32", "shape": [1, 1, 257]}],
                            "outputs": [{"element_type": "f32", "shape": [1, 1, 257]}]}
            vulkan_tensor = {"element_type": "f32", "shape": [1, 1, 257],
                             "host_layout": "CHW", "logical_layout": "CHW",
                             "physical_layout": "CHW"}
            vulkan_manifest = {"inputs": [vulkan_tensor], "outputs": [vulkan_tensor]}
            if root != roots[0]:
                (cpu / "paired_relu.json").write_text(json.dumps(cpu_manifest))
            (vulkan / "paired_relu_vulkan.json").write_text(json.dumps(vulkan_manifest))
            evidence = {
                "fixture": {"param_sha256": run_gate.sha256(root / "model.param"),
                            "bin_sha256": run_gate.sha256(root / "model.bin")},
                "public_io": {"element_type": "f32", "shape": [257],
                              "tolerance": {"absolute": 1.0e-6, "relative": 1.0e-6},
                              **records},
            }
            (root / "evidence.json").write_text(json.dumps(evidence))

        findings = []
        with patch.object(audit, "_va0_compare_product_artifacts", return_value=False) as compare:
            result = audit._va0_compare_relu_product(
                self.root, roots[0], roots[1], "synthetic ReLU", findings)
        self.assertFalse(result)
        compare.assert_called_once()
        self.assertEqual([], findings)
        self.assertEqual("CHW", audit._va0_relu_manifest_contract(
            roots[0], require_cpu_manifest=False)["host_layout"])
        self.assertEqual("CHW", audit._va0_relu_manifest_contract(
            roots[1], require_cpu_manifest=True)["host_layout"])

        old_manifest_path = roots[0] / "vulkan/paired_relu_vulkan.json"
        old_manifest = json.loads(old_manifest_path.read_text())
        old_manifest["inputs"][0]["host_layout"] = "HWC"
        old_manifest_path.write_text(json.dumps(old_manifest))
        findings = []
        result = audit._va0_compare_relu_product(
            self.root, roots[0], roots[1], "synthetic ReLU", findings)
        self.assertFalse(result)
        self.assertTrue(findings)

    def test_equivalent_linux_host_triple_vendor_spelling_is_normalized(self):
        self.assertEqual("x86_64-linux-gnu",
                         audit._va0_normalize_host_triple("x86_64-pc-linux-gnu"))
        self.assertEqual("x86_64-linux-gnu",
                         audit._va0_normalize_host_triple("x86_64-linux-gnu"))
        self.assertEqual("aarch64-unknown-linux-gnu",
                         audit._va0_normalize_host_triple("aarch64-unknown-linux-gnu"))
        self.assertEqual("x86_64-unknown-linux-musl",
                         audit._va0_normalize_host_triple("x86_64-unknown-linux-musl"))

    def test_spirv_pure_guard_reordering_is_normalized_but_semantic_changes_fail(self):
        release = """%1 = OpLabel
%10 = OpULessThan %bool %value %upper
%11 = OpUGreaterThanEqual %bool %value %lower
%12 = OpLogicalAnd %bool %11 %10
%13 = OpLogicalAnd %bool %otherA %otherB
%14 = OpLogicalAnd %bool %13 %12
OpBranchConditional %14 %true %false
"""
        sanitizer = """%1 = OpLabel
%10 = OpUGreaterThanEqual %bool %value %lower
%11 = OpULessThan %bool %value %upper
%12 = OpLogicalAnd %bool %10 %11
%13 = OpLogicalAnd %bool %otherB %otherA
%14 = OpLogicalAnd %bool %12 %13
OpBranchConditional %14 %true %false
"""
        self.assertEqual(audit._va0_normalize_spirv_disassembly(release),
                         audit._va0_normalize_spirv_disassembly(sanitizer))
        changed = sanitizer.replace("OpUGreaterThanEqual", "OpUGreaterThan")
        self.assertNotEqual(audit._va0_normalize_spirv_disassembly(release),
                            audit._va0_normalize_spirv_disassembly(changed))

    def test_wrong_frozen_identity_and_missing_manifest_members_are_rejected(self):
        state_path = self.source / "gate-va0-legacy-source.json"
        state = json.loads(state_path.read_text())
        state["source_fingerprint"] = "0" * 64
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "source identity/status"):
            run_gate.snapshot_legacy_baseline(self.source, self.root / "bad-identity")

        self.source = self.root / "another-source"
        self.source.mkdir()
        self._make_source()
        archive_path = self.source / "gate-archive-manifest.json"
        archive = json.loads(archive_path.read_text())
        archive["files"].pop("test/Native/vulkan-product-cli/evidence.json")
        archive_path.write_text(json.dumps(archive))
        with self.assertRaisesRegex(ValueError, "omits required files"):
            run_gate.snapshot_legacy_baseline(self.source, self.root / "bad-manifest")

    def test_missing_or_tampered_reference_blocks_checker(self):
        findings = []
        self.assertIsNone(audit._audit_legacy_reference(
            self.root, {"legacy_reference": {"status": "not-supplied"}}, [], findings))
        self.assertIn("legacy-reference-missing", {item["code"] for item in findings})

        target = self.root / "tampered-stage"
        state = self.snapshot(target)
        snapshot = json.loads((target / state["legacy_reference"]["path"]).read_text())
        record = next(item for item in snapshot["files"]
                      if item["path"].endswith("fixtures/conv-relu-nvidia/evidence.json"))
        (target / record["path"]).write_bytes(b"tampered")
        findings = []
        baseline = audit._audit_legacy_reference(target, state, [], findings)
        self.assertIsNotNone(baseline)
        codes = {item["code"] for item in findings}
        self.assertTrue({"legacy-reference-invalid", "evidence-hash-mismatch",
                         "legacy-reference-source-manifest"} & codes)

    def test_migration_map_covers_all_ids_and_ctest_cases_as_pending(self):
        names = audit._required_tests("release")
        coverage = audit._va0_migration_coverage({
            "profile": "release", "ctest": {"test_names": sorted(names), "failed": [], "skipped": []}})
        expected_ids = set(audit.ID_CATALOG)
        self.assertEqual(expected_ids, set(coverage["current_ids"]))
        self.assertEqual(names, set(coverage["current_test_cases"]))
        owners = {owner for rows in coverage["current_ids"].values()
                  for owner in rows["future_migration"]}
        self.assertTrue(owners.issubset(set(audit.VA0_MIGRATION_RESPONSIBILITY)))
        self.assertEqual(set(audit.VA0_MIGRATION_RESPONSIBILITY),
                         {owner for item in coverage["current_test_cases"].values()
                          for owner in item["future_migration"]})
        self.assertTrue(all(item["migration_status"] == "pending" and
                            item["legacy_pass_is_migration_pass"] is False
                            for item in coverage["current_test_cases"].values()))
        self.assertFalse(coverage["unexecuted_related_tests"])
        self.assertEqual("VA1", run_gate.parse_args(["--profile", "release"]).profile and
                         audit._va0_migration_owners("VulkanExecutionPlanTest.RoundTrips", ["V2.1"])[0])

    def test_va1_test_inventory_stays_strict_and_outside_va0_coverage(self):
        va0_tests = audit._required_tests("release")
        va1_tests = audit._required_tests("release", stage_id="VA1")
        self.assertEqual(audit.VA1_REQUIRED_TESTS, va1_tests - va0_tests)

        findings = []
        audit._check_required_tests({
            "profile": "release", "stage_id": "VA1",
            "ctest": {"test_names": sorted(va0_tests)}}, findings)
        missing = {name for item in findings
                   for name in item.get("details", {}).get("tests", [])}
        self.assertEqual(audit.VA1_REQUIRED_TESTS, missing)

        coverage = audit._va0_migration_coverage({
            "profile": "release",
            "ctest": {"test_names": sorted(va0_tests), "failed": [], "skipped": []}})
        self.assertFalse(coverage["unexecuted_related_tests"])
        self.assertEqual(set(audit.ID_CATALOG), set(coverage["current_ids"]))

    def test_va0_ids_are_profile_scoped_and_not_host_aot_eligible(self):
        profile = "release"
        tests = sorted(audit._required_tests(profile))
        migration = audit._va0_migration_coverage({
            "profile": profile, "ctest": {"test_names": tests, "failed": [], "skipped": []}})
        refs = [
            {"path": "test/Native/vulkan-v1-cpu-baseline/evidence.json",
             "sha256": "a" * 64, "role": "fresh CPU baseline"},
            {"path": "test/Native/vulkan-product-cli/evidence.json",
             "sha256": "b" * 64, "role": "product evidence"},
            {"path": "test/Vulkan/evidence/v5/nvidia/models/add_relu/trials/seed-0/golden-0.f32",
             "sha256": "c" * 64, "role": "V5 scalar golden"},
            {"path": "legacy-reference/source/gate-state.json",
             "sha256": "d" * 64, "role": "legacy-reference gate provenance"},
            {"path": "link-evidence/model-dependency-evidence.json",
             "sha256": "e" * 64, "role": "dependency evidence"},
            {"path": "gate-state.json", "sha256": "f" * 64, "role": "gate-state"},
            {"path": "test/gate-tests.json", "sha256": "1" * 64, "role": "CTest inventory"},
        ]
        comparisons = {
            "relu-product": {"nvidia": True, "llvmpipe": True},
            "add_relu": {"nvidia": True, "llvmpipe": True},
            "conv_relu": {"nvidia": True, "llvmpipe": True},
            "squeezenet_v1_1": {"nvidia": True, "llvmpipe": True},
        }
        va0 = {"legacy_reference": {"status": "verified"},
               "comparisons": comparisons,
               "dependencies": {"status": "verified"},
               "current_host_contract_verified": True,
               "migration_coverage": migration}
        stage = {"stage": "release", "profile": profile, "scope_passed": True,
                 "families": {"va0": va0, "cpu_baseline": {"path": "cpu"}},
                 "ctest": {"test_names": tests, "failed": [], "skipped": []},
                 "raw_artifacts": refs}
        index = audit._make_id_index([stage], release_eligible=False, all_refs=refs)
        for identifier in ("VA0.1", "VA0.2", "VA0.3", "VA0.4"):
            row = index[identifier]
            self.assertEqual("passed_for_profile_scope", row["status"])
            self.assertEqual("not_eligible", row["release_status"])
            self.assertFalse(row["host_aot_release_eligible"])
            self.assertEqual(["release"], [item["profile"] for item in row["profiles"]])
            self.assertTrue(row["profiles"][0]["tests_executed"])
        for identifier in audit.VA1_STEP_TESTS:
            row = index[identifier]
            self.assertEqual("blocked", row["status"])
            self.assertEqual("not_eligible", row["release_status"])
            self.assertFalse(row["host_aot_release_eligible"])
            self.assertEqual([], row["tests_executed"])

    def test_missing_legacy_reference_preserves_functional_release_scope(self):
        profiles = ("release", "sanitize", "offline", "software")
        stages = [{"stage": f"/stage/{profile}", "profile": profile,
                   "source_fingerprint": "a" * 64, "scope_passed": True,
                   "findings": [], "va0_findings": [{"code": "legacy-reference-missing"}],
                   "ctest": {"test_names": []}, "raw_artifacts": [], "families": {}}
                  for profile in profiles]
        gates = [{"stage": row["stage"], "profile": row["profile"],
                  "source_fingerprint": row["source_fingerprint"]} for row in stages]
        id_index = {identifier: {"status": "blocked" if identifier.startswith("VA0.") else "passed",
                                 "release_status": "not_eligible" if identifier.startswith("VA0.") else "passed"}
                    for identifier in audit.ID_CATALOG}
        with patch.object(audit, "audit_stage_gate", side_effect=gates), \
                patch.object(audit, "audit_stage_evidence", side_effect=stages), \
                patch.object(audit, "_make_id_index", return_value=id_index):
            report = audit.build_report([Path(row["stage"]) for row in stages])
        self.assertTrue(report["release_eligible"])
        self.assertFalse(report["va0_baseline_eligible"])
        self.assertFalse(report["host_aot_release_eligible"])
        self.assertEqual([], report["blocking_findings"])
        self.assertTrue(report["va0_blocking_findings"])

    def test_explicit_invalid_stage_ids_fail_closed(self):
        for stage_id in (None, "", "VA2"):
            with self.subTest(stage_id=stage_id):
                profiles = ("release", "sanitize", "offline", "software")
                stages = [{"stage": f"/stage/{profile}", "profile": profile,
                           "stage_id": stage_id if profile == "release" else "VA0",
                           "source_fingerprint": "a" * 64, "scope_passed": True,
                           "findings": [], "va0_findings": [],
                           "ctest": {"test_names": []}, "raw_artifacts": [],
                           "families": {}}
                          for profile in profiles]
                gates = [{"stage": row["stage"], "profile": row["profile"],
                          "stage_id": row["stage_id"],
                          "source_fingerprint": row["source_fingerprint"]}
                         for row in stages]
                id_index = {
                    identifier: {"status": "passed", "release_status": "passed"}
                    for identifier in audit.ID_CATALOG}
                with patch.object(audit, "audit_stage_gate", side_effect=gates), \
                        patch.object(audit, "audit_stage_evidence", side_effect=stages), \
                        patch.object(audit, "_make_id_index", return_value=id_index):
                    report = audit.build_report(
                        [Path(row["stage"]) for row in stages])
                self.assertFalse(report["release_eligible"])
                self.assertEqual(stage_id, report["stages"][0]["stage_id"])
                self.assertIn("stage-id-invalid",
                              {item["code"] for item in report["blocking_findings"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
