#!/usr/bin/env python3
"""Unit tests for wall-partition / worker-share attribution (T-C2 split).

Pure-stdlib fixtures; no compiled artifacts are required.  Runs under both
``python3 test/Native/test_attr_partition.py`` and ``python3 -m pytest``.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
  sys.path.insert(0, str(REPO / "tools"))

from perf_attribution_report import build_report  # noqa: E402


def make_plan(operations, buffers=None):
  return {
      "schema_version": 1,
      "plan_revision": "static-v1",
      "kind": "ncnn.model_execution_plan",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": {"triple": "x86_64-pc-linux-gnu", "threads": 2},
      "operations": operations,
      "buffers": buffers if buffers is not None else [],
      "summary": {},
      "diagnostics": {"unknown_fields": []},
  }


def make_profile(events, summary, schema_version=3, complete=True):
  return {
      "schema_version": schema_version,
      "kind": "ncnn.model_execution_profile",
      "plan_revision": "static-v1",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": "x86_64-pc-linux-gnu",
      "threads": 2,
      "mode": "prepared",
      "complete": complete,
      "instrumentation": {"coverage": "coarse"},
      "summary": summary,
      "events": events,
  }


def make_perf():
  return {
      "model": "fixture",
      "mode": "prepared",
      "status": "measured",
      "target": "x86_64-pc-linux-gnu",
      "plan_revision": "static-v1",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "threads": 2,
      "ratio": 1.2,
      "ncnn_mean_ms": 1.0,
      "compiled_mean_ms": 1.2,
      "diagnostics": {"gate_eligible": False},
  }


def wall_summary(top_level):
  return {
      "top_level_time_ns": top_level,
      "top_level_time_known": True,
      "event_mismatch_count": 0,
      "peak_live_proven": False,
      "peak_live_bytes": None,
  }


def gap_free_report():
  """Sequential 400ns + worker-attributed 600ns over a 1000ns top level."""
  plan = make_plan([
      {"id": "model/seq#0", "kind": "seq", "profile_id": 101},
      {"id": "model/worker#0", "kind": "worker", "profile_id": 102},
  ])
  events = [
      {"id": 101, "category": "operation", "time_domain": "wall",
       "calls": 1, "inclusive_ns": 400, "exclusive_ns": 400},
      {"id": 102, "category": "worker_operation", "time_domain": "worker_cpu",
       "calls": 1, "inclusive_ns": 600, "exclusive_ns": 500,
       "worker_wall_union_ns": 100, "worker_wall_union_known": True,
       "wall_attributed_ns": 600, "wall_attributed_estimated_ns": 600,
       "wall_attributed_share_of_sampled_window": 1.0,
       "wall_attributed_known": True},
  ]
  profile = make_profile(events, wall_summary(1000))
  return build_report(plan, profile, make_perf(), "prepared")


def projected_report():
  """Parallel region with projection factor 2.0 plus a residual wall gap."""
  plan = make_plan([
      {"id": "model/seq#0", "kind": "seq", "profile_id": 101},
      {"id": "model/worker#0", "kind": "worker", "profile_id": 102},
      {"id": "model/parallel#0", "kind": "parallel", "profile_id": 103},
  ])
  events = [
      {"id": 101, "category": "operation", "time_domain": "wall",
       "calls": 1, "inclusive_ns": 500, "exclusive_ns": 500},
      {"id": 103, "category": "parallel", "time_domain": "wall",
       "calls": 1, "inclusive_ns": 1800, "exclusive_ns": 400,
       "worker_covered_wall_ns": 900,
       "worker_covered_wall_sampled_ns": 450,
       "sampled_window_wall_ns": 900, "region_wall_ns": 1800,
       "region_worker_spans": 2, "sample_scale": 1,
       "wall_projection_factor": 2.0, "wall_coverage_share": 0.5,
       "wall_exclusive_estimated_ns": 100,
       "wall_exclusive_estimated_known": True,
       "exclusive_semantics": "sampled_window_minus_worker_span_union",
       "bytes": None, "bytes_known": False},
      {"id": 102, "category": "worker_operation", "time_domain": "worker_cpu",
       "calls": 1, "inclusive_ns": 800, "exclusive_ns": 700,
       "worker_wall_union_ns": 300, "worker_wall_union_known": True,
       "wall_attributed_ns": 800, "wall_attributed_estimated_ns": 800,
       "wall_attributed_share_of_sampled_window": 0.5,
       "wall_attributed_known": True},
  ]
  profile = make_profile(events, wall_summary(2000))
  return build_report(plan, profile, make_perf(), "prepared")


def worker_coverage_report():
  """Two worker events; one joins the plan, one has no plan operation."""
  plan = make_plan([
      {"id": "model/known#0", "kind": "known", "profile_id": 101},
  ])
  events = [
      {"id": 101, "category": "worker_operation", "time_domain": "worker_cpu",
       "calls": 1, "inclusive_ns": 100, "exclusive_ns": 50,
       "worker_wall_union_ns": 60, "worker_wall_union_known": True,
       "wall_attributed_ns": 40, "wall_attributed_estimated_ns": 40,
       "wall_attributed_known": True},
      {"id": 999, "category": "worker_operation", "time_domain": "worker_cpu",
       "calls": 1, "inclusive_ns": 200, "exclusive_ns": 100,
       "worker_wall_union_ns": 50, "worker_wall_union_known": True},
  ]
  profile = make_profile(events, wall_summary(1000))
  return build_report(plan, profile, make_perf(), "prepared")


class WallPartitionAdditivityTest(unittest.TestCase):

  def test_gap_free_additivity(self):
    partition = gap_free_report()["runtime"]["wall_partition"]
    self.assertEqual(partition["top_level_wall_ns"], 1000)
    self.assertEqual(partition["worker_ops_wall_attributed_ns"], 600)
    self.assertEqual(partition["sequential_ops_exclusive_ns"], 400)
    self.assertEqual(partition["unaccounted_wall_ns"], 0)
    self.assertEqual(
        partition["worker_ops_wall_attributed_ns"] +
        partition["sequential_ops_exclusive_ns"] +
        partition["unaccounted_wall_ns"],
        partition["top_level_wall_ns"])

  def test_accounted_plus_unaccounted_is_top_level(self):
    for report in (gap_free_report(), projected_report()):
      partition = report["runtime"]["wall_partition"]
      self.assertEqual(
          partition["accounted_wall_ns"] + partition["unaccounted_wall_ns"],
          partition["top_level_wall_ns"])

  def test_generalized_additivity_with_gap_and_unjoined(self):
    partition = projected_report()["runtime"]["wall_partition"]
    unjoined = partition["worker_ops_wall_unjoined_ns"] or 0
    self.assertEqual(
        partition["worker_ops_wall_attributed_ns"] +
        partition["sequential_ops_exclusive_ns"] +
        unjoined +
        partition["parallel_wall_gap_wall_ns"] +
        partition["unaccounted_wall_ns"],
        partition["top_level_wall_ns"])

  def test_default_projection_factor_without_parallel_region(self):
    partition = gap_free_report()["runtime"]["wall_partition"]
    self.assertEqual(partition["wall_projection_factor"], 1.0)
    self.assertEqual(partition["parallel_regions"], [])


class WallProjectionFactorTest(unittest.TestCase):

  def test_projection_factor_matches_parallel_region(self):
    partition = projected_report()["runtime"]["wall_partition"]
    self.assertEqual(partition["wall_projection_factor"], 2.0)
    self.assertEqual(len(partition["parallel_regions"]), 1)
    self.assertEqual(partition["parallel_regions"][0]["wall_projection_factor"],
                     2.0)

  def test_sampled_window_projects_onto_region_wall(self):
    partition = projected_report()["runtime"]["wall_partition"]
    self.assertEqual(partition["parallel_region_wall_ns"], 1800)
    self.assertEqual(partition["sampled_window_wall_ns"], 900)
    self.assertAlmostEqual(
        partition["parallel_region_wall_ns"] /
        partition["sampled_window_wall_ns"],
        partition["wall_projection_factor"])
    self.assertAlmostEqual(
        partition["sampled_window_fraction_of_parallel_wall"], 0.5)

  def test_window_normalized_coverage_shares(self):
    partition = projected_report()["runtime"]["wall_partition"]
    self.assertEqual(partition["worker_covered_wall_sampled_ns"], 450)
    self.assertEqual(partition["worker_covered_wall_estimated_ns"], 900)
    self.assertAlmostEqual(partition["parallel_wall_coverage_share"], 0.5)
    self.assertAlmostEqual(partition["parallel_wall_gap_share"], 0.5)
    self.assertEqual(partition["worker_ops_wall_unjoined_ns"], 100)
    self.assertEqual(partition["accounted_wall_ns"], 1500)
    self.assertEqual(partition["unaccounted_wall_ns"], 500)


class WorkerJoinCoverageTest(unittest.TestCase):

  def test_join_coverage_counting(self):
    worker = worker_coverage_report()["runtime"]["worker_attribution"]
    self.assertEqual(worker["observed_event_count"], 2)
    self.assertEqual(worker["joined_event_count"], 1)
    self.assertEqual(worker["unattributed_event_count"], 1)
    self.assertEqual(worker["exclusive_cpu_known_count"], 1)
    self.assertEqual(worker["exclusive_cpu_unknown_count"], 1)
    self.assertEqual(worker["wall_union_unknown_count"], 0)
    self.assertEqual(worker["wall_attributed_unknown_count"], 0)
    self.assertFalse(worker["complete_for_observed_events"])
    self.assertEqual([row["id"] for row in worker["operations"]], [101])

  def test_unjoined_worker_reasons_keep_append_order(self):
    report = worker_coverage_report()
    self.assertEqual(report["static"]["unknown_fields"], [
        "runtime_worker_operation_unattributed",
        "runtime_peak_live_unknown",
        "runtime_worker_attribution_incomplete",
    ])
    self.assertEqual(report["runtime"]["incomplete_reasons"], [
        "runtime_worker_operation_unattributed",
        "runtime_worker_attribution_incomplete",
    ])
    self.assertFalse(report["runtime"]["complete"])

  def test_worker_wall_share_denominator_is_top_level(self):
    worker = gap_free_report()["runtime"]["worker_attribution"]
    row = worker["top_operations_by_wall_union"][0]
    self.assertEqual(row["wall_union_ns"], 100)
    self.assertEqual(row["wall_union_share_of_top_level"], 0.1)
    self.assertEqual(row["wall_attributed_share_of_top_level"], 0.6)


if __name__ == "__main__":
  unittest.main()
