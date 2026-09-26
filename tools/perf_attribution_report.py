#!/usr/bin/env python3
"""Join a static execution plan, a diagnostic profile, and perf NDJSON.

The join is deliberately strict: model, plan revision, target triple, thread
count, and measurement mode must agree.  Missing runtime values remain null in
the report and are never converted to zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from attr.attr_join import (collect_fusions, fusion_runtime_join,
                            join_profile_events, operation_index,
                            allocation_index, allocation_metadata,
                            packed_kernel_join, read_json, read_perf,
                            read_profile, validate_profile_events)
from attr.attr_partition import (build_wall_partition,
                                 build_worker_section,
                                 build_workspace_coverage,
                                 rank_top_allocations, rank_top_costs,
                                 window_normalization)
from attr.attr_report import (append_runtime_byte_reasons,
                              assemble_report, aggregate_v2_reports,
                              build_copy_layout,
                              collect_incomplete_reasons,
                              conv_depthwise_breakdown,
                              validate_low_precision,
                              validate_report_structure)
from attr.attr_schema import (AttributionError, require_string,
                              validate_identity)


def build_report(plan: dict[str, Any], profile: dict[str, Any],
                 perf: dict[str, Any], mode: str) -> dict[str, Any]:
  operations = operation_index(plan)
  fusion_records, fusion_profile_ids = collect_fusions(plan)
  static_allocations = allocation_index(plan)
  allocation_objects = allocation_metadata(plan)
  events, profile_summary = validate_profile_events(profile)
  join = join_profile_events(
      profile, events, operations, static_allocations, fusion_profile_ids)
  fusion_site_runtime, fusion_site_summary = fusion_runtime_join(
      fusion_records, join.operation_events)
  packed_kernel_runtime, packed_kernel_summary = packed_kernel_join(
      plan, operations, join.operation_events)
  total_ns, top = rank_top_costs(
      join.joined, profile_summary, join.measured_event_time_ns)
  summary, unknown, perf_diagnostics, instrumentation, coverage = \
      validate_report_structure(plan, perf, profile)
  unknown, incomplete_reasons, missing_allocations, allocation_coverage = \
      collect_incomplete_reasons(
          profile, profile_summary, instrumentation, fusion_site_summary,
          static_allocations, join.observed_allocations, join.unattributed,
          join.worker_unattributed, join.unknown_time_ns,
          join.allocation_bytes_unknown, unknown)
  family_breakdown = conv_depthwise_breakdown(plan)
  low_precision = validate_low_precision(plan)
  unknown, incomplete_reasons = append_runtime_byte_reasons(
      join.movement_totals, profile_summary, unknown, incomplete_reasons)
  copy_layout = build_copy_layout(join.movement_totals)
  top_allocations, workspace_total, workspace_joined = rank_top_allocations(
      allocation_objects, join.allocation_events)
  workspace_coverage, unknown, incomplete_reasons = build_workspace_coverage(
      workspace_total, workspace_joined, unknown, incomplete_reasons)
  runtime_peak_live_proven, top_level_wall_ns, sample_scale = \
      window_normalization(profile_summary, missing_allocations)
  worker_attribution, unknown, incomplete_reasons = build_worker_section(
      join.worker_joined, join.worker_operation_events,
      join.worker_unattributed, profile, profile_summary, top_level_wall_ns,
      unknown, incomplete_reasons)
  wall_partition = build_wall_partition(
      join.worker_joined, join.parallel_coverage, join.joined, sample_scale,
      top_level_wall_ns)
  return assemble_report(
      plan, profile, perf, mode, summary, unknown, perf_diagnostics, coverage,
      profile_summary, family_breakdown, low_precision, join.category_totals,
      join.unattributed, total_ns, join.known_event_time_ns,
      join.unknown_time_ns, allocation_coverage, workspace_coverage,
      copy_layout, worker_attribution, wall_partition,
      packed_kernel_runtime, packed_kernel_summary, fusion_site_runtime,
      fusion_site_summary, runtime_peak_live_proven, incomplete_reasons, top,
      top_allocations)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--perf", required=True, help="performance NDJSON")
  parser.add_argument("--plan", required=True, help="execution plan JSON")
  parser.add_argument("--profile", required=True, help="runtime profile JSON")
  parser.add_argument("--mode", default="end_to_end")
  parser.add_argument("--output", help="write report JSON instead of stdout")
  arguments = parser.parse_args()
  try:
    plan = read_json(arguments.plan)
    profile_value = read_profile(arguments.profile)
    perf = read_perf(arguments.perf, require_string(plan.get("model"), "plan.model"),
                     arguments.mode)
    if isinstance(profile_value, list):
      reports = []
      for profile in profile_value:
        if profile.get("schema_version") not in {2, 3}:
          raise AttributionError(
            "profile NDJSON rows must use schema-2/3 per-invocation records")
        validate_identity(plan, profile, perf, arguments.mode)
        reports.append(build_report(plan, profile, perf, arguments.mode))
      report = aggregate_v2_reports(reports)
    else:
      validate_identity(plan, profile_value, perf, arguments.mode)
      report = build_report(plan, profile_value, perf, arguments.mode)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
      Path(arguments.output).write_text(text, encoding="utf-8")
    else:
      sys.stdout.write(text)
  except (AttributionError, OSError) as error:
    print(str(error), file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
