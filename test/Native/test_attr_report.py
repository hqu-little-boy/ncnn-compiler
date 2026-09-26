#!/usr/bin/env python3
"""Unit tests for report assembly and multi-invocation aggregation (T-C2 split).

Pure-stdlib fixtures; no compiled artifacts are required.  Runs under both
``python3 test/Native/test_attr_report.py`` and ``python3 -m pytest``.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
  sys.path.insert(0, str(REPO / "tools"))

from perf_attribution_report import (  # noqa: E402
    aggregate_v2_reports, build_report)


def make_plan(operations, buffers=None, diagnostics=None):
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
      "diagnostics": (diagnostics if diagnostics is not None
                      else {"unknown_fields": []}),
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


def invocation_profile(invocation_id, top_level, exclusive):
  """A schema-2 per-invocation profile with one joined operation event."""
  return {
      "schema_version": 2,
      "kind": "ncnn.model_execution_profile",
      "plan_revision": "static-v1",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": "x86_64-pc-linux-gnu",
      "threads": 2,
      "mode": "prepared",
      "invocation_id": invocation_id,
      "complete": True,
      "instrumentation": {"coverage": "coarse"},
      "summary": {
          "top_level_time_ns": top_level,
          "top_level_time_known": True,
          "event_mismatch_count": 0,
          "peak_live_proven": False,
          "peak_live_bytes": None,
      },
      "events": [{
          "id": 101,
          "category": "operation",
          "calls": 1,
          "inclusive_ns": exclusive + 5,
          "exclusive_ns": exclusive,
      }],
  }


def build_reports(spec):
  plan = make_plan([{"id": "model/op#0", "kind": "op", "profile_id": 101}])
  perf = make_perf()
  return [
      build_report(plan, invocation_profile(*row), perf, "prepared")
      for row in spec
  ]


class AggregateMedianTest(unittest.TestCase):

  def test_odd_invocation_medians(self):
    reports = build_reports([(1, 100, 10), (2, 200, 20), (3, 300, 30)])
    result = aggregate_v2_reports(reports)
    self.assertIs(result, reports[0])
    summary = result["runtime"]["summary"]
    self.assertEqual(summary["top_level_time_ns"], 200)
    self.assertIs(summary["top_level_time_known"], True)
    self.assertEqual(
        result["runtime"]["per_invocation"]["top_level_time_ns"],
        {"known_count": 3, "median": 200, "worst": 300})
    # Count fields aggregate to null; their statistics stay per-invocation.
    self.assertIsNone(summary["event_mismatch_count"])
    self.assertEqual(
        result["runtime"]["per_invocation"]["summary"]["event_mismatch_count"],
        {"known_count": 3, "median": 0, "worst": 0})
    self.assertEqual(result["runtime"]["category_time_ns"]["kernel"], 20)
    self.assertEqual(
        result["runtime"]["per_invocation"]["category_time_ns"]["kernel"],
        {"known_count": 3, "median": 20, "worst": 30})

  def test_top_costs_use_invocation_medians(self):
    reports = build_reports([(1, 100, 10), (2, 200, 20), (3, 300, 30)])
    result = aggregate_v2_reports(reports)
    top = result["top_costs"][0]
    self.assertEqual(top["attributed_ns"], 20)
    self.assertEqual(top["inclusive_ns"], 25)
    self.assertEqual(top["exclusive_ns"], 20)
    self.assertEqual(top["time_basis"], "exclusive")
    self.assertEqual(top["time_fraction"], 0.1)

  def test_even_invocation_median_stays_float(self):
    reports = build_reports([(1, 100, 10), (2, 301, 11)])
    result = aggregate_v2_reports(reports)
    summary = result["runtime"]["summary"]
    self.assertEqual(summary["top_level_time_ns"], 200.5)
    self.assertIsInstance(summary["top_level_time_ns"], float)
    top = result["top_costs"][0]
    self.assertEqual(top["attributed_ns"], 10.5)
    self.assertEqual(top["inclusive_ns"], 15.5)
    self.assertEqual(top["time_fraction"], 10.5 / 200.5)

  def test_wall_partition_medians(self):
    reports = build_reports([(1, 100, 10), (2, 200, 20), (3, 300, 30)])
    result = aggregate_v2_reports(reports)
    partition = result["runtime"]["wall_partition"]
    self.assertEqual(partition["top_level_wall_ns"], 200)
    self.assertEqual(partition["invocation_count"], 3)
    self.assertEqual(partition["sequential_ops_exclusive_ns"], 20)
    self.assertEqual(partition["unaccounted_wall_ns"], 180)
    self.assertEqual(
        partition["sequential_ops_exclusive_ns_per_invocation"],
        {"known_count": 3, "median": 20, "worst": 30})


class AggregateMutationTest(unittest.TestCase):

  def test_returns_and_mutates_first_report(self):
    reports = build_reports([(1, 100, 10), (2, 200, 20), (3, 300, 30)])
    result = aggregate_v2_reports(reports)
    self.assertIs(result, reports[0])
    self.assertIs(result["runtime"], reports[0]["runtime"])
    self.assertEqual(result["runtime"]["invocation_count"], 3)
    self.assertIs(result["runtime"]["complete_all"], True)
    self.assertIsNone(result["invocation_id"])
    self.assertEqual(result["invocation_ids"], [1, 2, 3])
    self.assertEqual(result["aggregation"], "per-invocation-median")
    self.assertIs(result["runtime"]["complete"], True)
    # Only reports[0] is rewritten; later invocations keep their identity.
    self.assertNotIn("invocation_ids", reports[1])
    self.assertNotIn("aggregation", reports[1])
    self.assertEqual(reports[1]["invocation_id"], 2)

  def test_incomplete_reasons_union_is_sorted_and_deduplicated(self):
    reports = build_reports([(1, 100, 10), (2, 200, 20)])
    first_runtime = reports[0]["runtime"]
    first_runtime["incomplete_reasons"] = ["runtime_exclusive_time_unknown"]
    reports[1]["runtime"]["incomplete_reasons"] = [
        "runtime_profile_incomplete", "runtime_exclusive_time_unknown",
        "runtime_worker_attribution_incomplete",
    ]
    reports[1]["runtime"]["complete"] = False
    result = aggregate_v2_reports(reports)
    self.assertEqual(result["runtime"]["incomplete_reasons"], [
        "runtime_exclusive_time_unknown",
        "runtime_profile_incomplete",
        "runtime_worker_attribution_incomplete",
    ])
    self.assertIs(result["runtime"]["complete_all"], False)
    self.assertIs(result["runtime"]["complete"], False)


class UnknownFieldAppendOrderTest(unittest.TestCase):

  def test_ordered_unknown_chain_is_preserved(self):
    plan = make_plan(
        [{"id": "model/known#0", "kind": "known", "profile_id": 101}],
        diagnostics={"unknown_fields": ["plan_side_note"]})
    events = [
        {"id": 998, "category": "operation", "time_domain": "wall",
         "calls": 1, "inclusive_ns": 70, "exclusive_ns": 70},
        {"id": 101, "category": "worker_operation", "time_domain": "worker_cpu",
         "calls": 1, "inclusive_ns": 100, "exclusive_ns": 50,
         "worker_wall_union_ns": 60, "worker_wall_union_known": True,
         "wall_attributed_ns": 40, "wall_attributed_estimated_ns": 40,
         "wall_attributed_known": True},
        {"id": 999, "category": "worker_operation", "time_domain": "worker_cpu",
         "calls": 1, "inclusive_ns": 200, "exclusive_ns": 100,
         "worker_wall_union_ns": 50, "worker_wall_union_known": True},
    ]
    profile = {
        "schema_version": 3,
        "kind": "ncnn.model_execution_profile",
        "plan_revision": "static-v1",
        "model": "fixture",
        "plan_hash": "plan-123",
        "build_identity": "plan-123",
        "target": "x86_64-pc-linux-gnu",
        "threads": 2,
        "mode": "prepared",
        "complete": False,
        "instrumentation": {"coverage": "coarse"},
        "summary": {
            "top_level_time_ns": 1000,
            "top_level_time_known": True,
            "event_mismatch_count": 0,
            "peak_live_proven": False,
            "peak_live_bytes": None,
        },
        "events": events,
    }
    report = build_report(plan, profile, make_perf(), "prepared")
    self.assertEqual(report["static"]["unknown_fields"], [
        "plan_side_note",
        "runtime_profile_incomplete",
        "runtime_unattributed_events",
        "runtime_worker_operation_unattributed",
        "runtime_peak_live_unknown",
        "runtime_worker_attribution_incomplete",
    ])
    self.assertEqual(report["runtime"]["incomplete_reasons"], [
        "runtime_profile_incomplete",
        "runtime_unattributed_events",
        "runtime_worker_operation_unattributed",
        "runtime_worker_attribution_incomplete",
    ])
    self.assertIs(report["runtime"]["complete"], False)
    self.assertEqual(report["runtime"]["unattributed_event_count"], 1)
    # No non-worker event joined a plan operation, so none of the top-level
    # time is known-attributed.
    self.assertEqual(report["runtime"]["unattributed_time_ns"], 1000)


if __name__ == "__main__":
  unittest.main()
