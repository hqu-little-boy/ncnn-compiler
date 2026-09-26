"""Report section assembly and multi-invocation aggregation."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .attr_schema import AttributionError


def validate_report_structure(
    plan: dict[str, Any], perf: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, Any], Any]:
  summary = plan.get("summary")
  if not isinstance(summary, dict):
    raise AttributionError("plan.summary must be an object")
  plan_diagnostics = plan.get("diagnostics")
  if not isinstance(plan_diagnostics, dict):
    raise AttributionError("plan.diagnostics must be an object")
  unknown = plan_diagnostics.get("unknown_fields", [])
  if not isinstance(unknown, list):
    raise AttributionError("plan diagnostics unknown_fields must be an array")
  perf_diagnostics = perf.get("diagnostics", {})
  if not isinstance(perf_diagnostics, dict):
    raise AttributionError("perf.diagnostics must be an object")
  instrumentation = profile.get("instrumentation")
  if not isinstance(instrumentation, dict):
    raise AttributionError("profile.instrumentation must be an object")
  coverage = instrumentation.get("coverage")
  if coverage is not None and not isinstance(coverage, str):
    raise AttributionError("profile instrumentation coverage must be a string")
  for field in ("record_overflow", "allocation_table_overflow"):
    if field in instrumentation and not isinstance(instrumentation[field], bool):
      raise AttributionError(f"profile instrumentation {field} must be boolean")
  return summary, unknown, perf_diagnostics, instrumentation, coverage


def collect_incomplete_reasons(
    profile: dict[str, Any], profile_summary: dict[str, Any],
    instrumentation: dict[str, Any], fusion_site_summary: dict[str, Any],
    static_allocations: set[int], observed_allocations: set[int],
    unattributed: int, worker_unattributed: int, unknown_time_ns: int,
    allocation_bytes_unknown: bool, unknown: list[str]
) -> tuple[list[str], list[str], list[int], dict[str, Any]]:
  missing_allocations = sorted(static_allocations - observed_allocations)
  incomplete_reasons: list[str] = []
  if profile.get("schema_version") in {2, 3} and \
      profile.get("complete") is False:
    unknown = list(dict.fromkeys([*unknown, "runtime_profile_incomplete"]))
    incomplete_reasons.append("runtime_profile_incomplete")
  if unattributed:
    unknown = list(dict.fromkeys([*unknown, "runtime_unattributed_events"]))
    incomplete_reasons.append("runtime_unattributed_events")
  if worker_unattributed:
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_worker_operation_unattributed"]))
    incomplete_reasons.append("runtime_worker_operation_unattributed")
  if fusion_site_summary["missing_count"] or fusion_site_summary["partial_count"]:
    unknown = list(dict.fromkeys([*unknown, "runtime_fusion_site_profile_missing"]))
    incomplete_reasons.append("runtime_fusion_site_profile_missing")
  if unknown_time_ns:
    unknown = list(dict.fromkeys([*unknown, "runtime_exclusive_time_unknown"]))
    incomplete_reasons.append("runtime_exclusive_time_unknown")
  if profile_summary.get("materialized_read_complete") is False:
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_materialized_read_incomplete"]))
    incomplete_reasons.append("runtime_materialized_read_incomplete")
  elif profile_summary.get("materialized_write_count", 0) and \
      profile_summary.get("materialized_read_complete") is None:
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_materialized_read_completeness_unknown"]))
    incomplete_reasons.append("runtime_materialized_read_completeness_unknown")
  if "materialized_write_count" in profile_summary and \
      profile_summary.get("materialized_write_count") == 0:
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_materialized_bytes_not_observed"]))
    incomplete_reasons.append("runtime_materialized_bytes_not_observed")
  if missing_allocations:
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_profile_missing_allocation_events"]))
    incomplete_reasons.append("runtime_profile_missing_allocation_events")
  for count_field, known_field in (
      ("allocation_count", "allocation_bytes_known"),
      ("deallocation_count", "deallocation_bytes_known")):
    if (profile_summary.get(count_field, 0) and
        profile_summary.get(known_field) is False):
      allocation_bytes_unknown = True
  if allocation_bytes_unknown:
    unknown = list(dict.fromkeys([*unknown, "runtime_allocation_bytes_unknown"]))
    incomplete_reasons.append("runtime_allocation_bytes_unknown")
  if not profile_summary.get("peak_live_proven", False) or missing_allocations:
    unknown = list(dict.fromkeys([*unknown, "runtime_peak_live_unknown"]))
  if instrumentation.get("record_overflow", False) or \
      instrumentation.get("allocation_table_overflow", False):
    unknown = list(dict.fromkeys([*unknown, "runtime_profile_capacity_overflow"]))
    incomplete_reasons.append("runtime_profile_capacity_overflow")
  allocation_coverage = {
    "static_count": len(static_allocations),
    "observed_count": len(observed_allocations),
    "missing_profile_ids": missing_allocations,
    "complete": not missing_allocations,
  }
  return unknown, incomplete_reasons, missing_allocations, allocation_coverage


def conv_depthwise_breakdown(
    plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
  family_breakdown: dict[str, dict[str, Any]] = {}
  family_operations = plan.get("conv_depthwise_operations", [])
  if not isinstance(family_operations, list):
    raise AttributionError("plan.conv_depthwise_operations must be an array")
  for family in ("conv", "depthwise"):
    family_breakdown[family] = {
      "operation_count": 0,
      "implementations": {},
      "fallback_reasons": {},
      "source_layers": [],
      "runtime_time_ns": None,
    }
  for operation in family_operations:
    if not isinstance(operation, dict):
      raise AttributionError("plan.conv_depthwise_operations contains a non-object")
    family = operation.get("family")
    if family not in family_breakdown:
      raise AttributionError(f"unsupported Conv/Depthwise family: {family!r}")
    entry = family_breakdown[family]
    entry["operation_count"] += 1
    implementation = operation.get("implementation", "unknown")
    if not isinstance(implementation, str) or not implementation:
      implementation = "unknown"
    implementations = entry["implementations"]
    implementations[implementation] = implementations.get(implementation, 0) + 1
    if implementation == "fallback":
      reason = operation.get("fallback_reason", "unknown")
      if not isinstance(reason, str) or not reason:
        reason = "unknown"
      reasons = entry["fallback_reasons"]
      reasons[reason] = reasons.get(reason, 0) + 1
    source_layer = operation.get("source_layer")
    if isinstance(source_layer, int) and not isinstance(source_layer, bool):
      entry["source_layers"].append(source_layer)
  for entry in family_breakdown.values():
    entry["source_layers"] = sorted(set(entry["source_layers"]))
  return family_breakdown


def validate_low_precision(plan: dict[str, Any]) -> Any:
  low_precision = plan.get("low_precision")
  if low_precision is not None:
    if not isinstance(low_precision, dict):
      raise AttributionError("plan.low_precision must be an object")
    low_precision_operations = low_precision.get("operations")
    if not isinstance(low_precision_operations, list) or any(
        not isinstance(operation, dict) for operation in low_precision_operations):
      raise AttributionError("plan.low_precision.operations must be an array of objects")
  return low_precision


def append_runtime_byte_reasons(
    movement_totals: dict[str, dict[str, Any]],
    profile_summary: dict[str, Any], unknown: list[str],
    incomplete_reasons: list[str]
) -> tuple[list[str], list[str]]:
  if any(not value["bytes_known"] for value in movement_totals.values()
         if value["count"]):
    unknown = list(dict.fromkeys([*unknown, "runtime_movement_bytes_unknown"]))
    incomplete_reasons.append("runtime_movement_bytes_unknown")
  if any(profile_summary.get(count_field, 0) and
         profile_summary.get(known_field) is False
         for count_field, known_field in (
           ("materialized_write_count", "materialized_write_bytes_known"),
           ("materialized_read_count", "materialized_read_bytes_known"))):
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_materialized_bytes_unknown"]))
    incomplete_reasons.append("runtime_materialized_bytes_unknown")
  return unknown, incomplete_reasons


def build_copy_layout(movement_totals: dict[str, dict[str, Any]]) -> dict[str, Any]:
  copy_layout = {}
  for kind, value in movement_totals.items():
    observed = value["count"] != 0
    bytes_known = value["bytes_known"] if observed else None
    copy_layout[kind] = {
      "count": value["count"],
      "bytes": value["bytes"] if observed and value["bytes_known"] else None,
      "bytes_known": bytes_known,
      "status": "known" if observed and value["bytes_known"]
        else "unknown" if observed else "not_observed",
    }
  return copy_layout


def assemble_report(
    plan: dict[str, Any], profile: dict[str, Any], perf: dict[str, Any],
    mode: str, summary: dict[str, Any], unknown: list[str],
    perf_diagnostics: dict[str, Any], coverage: Any,
    profile_summary: dict[str, Any], family_breakdown: dict[str, Any],
    low_precision: Any, category_totals: dict[str, int], unattributed: int,
    total_ns: int, known_event_time_ns: int, unknown_time_ns: int,
    allocation_coverage: dict[str, Any], workspace_coverage: dict[str, Any],
    copy_layout: dict[str, Any], worker_attribution: dict[str, Any],
    wall_partition: dict[str, Any],
    packed_kernel_runtime: list[dict[str, Any]],
    packed_kernel_summary: dict[str, Any],
    fusion_site_runtime: list[dict[str, Any]],
    fusion_site_summary: dict[str, Any], runtime_peak_live_proven: bool,
    incomplete_reasons: list[str], top: list[dict[str, Any]],
    top_allocations: list[dict[str, Any]]) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "kind": "ncnn.model_performance_attribution",
    "model": plan["model"],
    "plan_revision": plan["plan_revision"],
    "attribution_revision": plan.get("attribution_revision",
                                      profile.get("attribution_revision")),
    "invocation_id": profile.get("invocation_id"),
    "mode": mode,
    "identity": {
      "target": plan["target"].get("triple"),
      "threads": plan["target"].get("threads"),
      "plan_hash": plan.get("plan_hash"),
      "build_identity": plan.get("build_identity"),
      "input_hash": perf.get("input_hash"),
      "codegen_identity": plan.get("codegen_identity"),
    },
    "performance": {
      "status": perf.get("status"),
      "gate_eligible": perf_diagnostics.get("gate_eligible"),
      "ratio": perf.get("ratio"),
      "ncnn_mean_ms": perf.get("ncnn_mean_ms"),
      "compiled_mean_ms": perf.get("compiled_mean_ms"),
    },
    "static": {
      "summary": summary,
      "unknown_fields": unknown,
      "conv_depthwise": family_breakdown,
      # This is a compiler-side static audit.  It carries no runtime evidence,
      # so it stays out of the runtime section.
      "low_precision": low_precision,
    },
    "runtime": {
      "summary": profile_summary,
      "category_time_ns": dict(sorted(category_totals.items())),
      "unattributed_event_count": unattributed,
      "unattributed_time_ns": max(0, total_ns - known_event_time_ns),
      "unknown_time_ns": unknown_time_ns,
      "coverage": coverage,
      "allocation_coverage": allocation_coverage,
      "workspace_coverage": workspace_coverage,
      "copy_layout": copy_layout,
      "worker_attribution": worker_attribution,
      "wall_partition": wall_partition,
      "packed_kernels": packed_kernel_runtime,
      "packed_kernel_summary": packed_kernel_summary,
      "fusion_sites": fusion_site_runtime,
      "top_fusion_sites": fusion_site_runtime[:5],
      "fusion_site_summary": fusion_site_summary,
      "peak_live_proven": runtime_peak_live_proven,
      "complete": not incomplete_reasons,
      "incomplete_reasons": incomplete_reasons,
    },
    "top_costs": top,
    "top_allocations": top_allocations[:10],
  }


def aggregate_v2_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
  if not reports:
    raise AttributionError("schema-2 profile contains no invocations")
  if any(report.get("schema_version") != 1 for report in reports):
    raise AttributionError("internal attribution reports must use schema 1")
  invocation_ids = [report.get("invocation_id") for report in reports]
  if any(not isinstance(identifier, int) or isinstance(identifier, bool)
         for identifier in invocation_ids):
    raise AttributionError("schema-2 profile invocation_id must be an integer")
  if len(set(invocation_ids)) != len(invocation_ids):
    raise AttributionError("duplicate schema-2 profile invocation_id")

  first = reports[0]
  runtime = first["runtime"]

  def numeric_values(key: str) -> list[int]:
    values = []
    for report in reports:
      value = report["runtime"].get(key)
      if isinstance(value, int) and not isinstance(value, bool):
        values.append(value)
    return values

  def median_scalar(values: list[int]) -> int | float | None:
    if not values:
      return None
    value = statistics.median(values)
    return int(value) if isinstance(value, float) and value.is_integer() else value

  def numeric_stats(values: list[int]) -> dict[str, Any]:
    return {
      "known_count": len(values),
      "median": median_scalar(values),
      "worst": max(values) if values else None,
    }

  summary_values = {
    "runtime_transpose_write_bytes": [],
    "runtime_materialized_write_bytes": [],
    "runtime_materialized_read_bytes": [],
    "expected_materialized_read_bytes": [],
    "top_level_time_ns": [],
    "peak_live_bytes": [],
  }
  for report in reports:
    summary = report["runtime"]["summary"]
    if summary.get("runtime_transpose_write_bytes_known") and \
        isinstance(summary.get("runtime_transpose_write_bytes"), int):
      summary_values["runtime_transpose_write_bytes"].append(
        summary["runtime_transpose_write_bytes"])
    for key, known_key in (
        ("runtime_materialized_write_bytes", "materialized_write_bytes_known"),
        ("runtime_materialized_read_bytes", "materialized_read_bytes_known"),
        ("expected_materialized_read_bytes",
         "expected_materialized_read_bytes_known")):
      if summary.get(known_key) is True and isinstance(summary.get(key), int):
        summary_values[key].append(summary[key])
    if summary.get("top_level_time_known") and \
        isinstance(summary.get("top_level_time_ns"), int):
      summary_values["top_level_time_ns"].append(summary["top_level_time_ns"])
    if summary.get("peak_live_proven") and \
        isinstance(summary.get("peak_live_bytes"), int):
      summary_values["peak_live_bytes"].append(summary["peak_live_bytes"])

  runtime["invocation_count"] = len(reports)
  runtime["complete_all"] = all(
    bool(report["runtime"].get("complete")) for report in reports)
  runtime["per_invocation"] = {
    key: numeric_stats(values) for key, values in summary_values.items()
  }
  runtime["per_invocation"]["category_time_ns"] = {}
  category_names = sorted({
    category
    for report in reports
    for category in report["runtime"].get("category_time_ns", {})
  })
  for category in category_names:
    values = [
      report["runtime"].get("category_time_ns", {}).get(category, 0)
      for report in reports
    ]
    runtime["per_invocation"]["category_time_ns"][category] = numeric_stats(values)
  runtime["category_time_ns"] = {
    category: runtime["per_invocation"]["category_time_ns"][category]["median"]
    for category in category_names
  }
  first_fusion_sites = runtime.get("fusion_sites", [])
  aggregated_fusion_sites = []
  for first_site in first_fusion_sites:
    site_id = first_site.get("id")
    invocation_sites = [
      next((site for site in report["runtime"].get("fusion_sites", [])
            if site.get("id") == site_id), None)
      for report in reports
    ]
    joined_invocations = sum(
      site is not None and site.get("event_join_status") == "joined"
      for site in invocation_sites)
    calls = [site["call_count"] for site in invocation_sites
             if site is not None and isinstance(site.get("call_count"), int)]
    inclusive = [site["inclusive_ns"] for site in invocation_sites
                 if site is not None and isinstance(site.get("inclusive_ns"), int)]
    exclusive = [site["exclusive_ns"] for site in invocation_sites
                 if site is not None and isinstance(site.get("exclusive_ns"), int)]
    event_join_status = (
      "joined" if joined_invocations == len(reports) else
      "partial" if joined_invocations else "missing")
    aggregated_fusion_sites.append({
      **first_site,
      "event_join_status": event_join_status,
      "joined_invocation_count": joined_invocations,
      "invocation_count": len(reports),
      "call_count": median_scalar(calls) if len(calls) == len(reports) else None,
      "inclusive_ns": median_scalar(inclusive)
      if len(inclusive) == len(reports) else None,
      "exclusive_ns": median_scalar(exclusive)
      if len(exclusive) == len(reports) else None,
      "per_invocation": {
        "call_count": numeric_stats(calls),
        "inclusive_ns": numeric_stats(inclusive),
        "exclusive_ns": numeric_stats(exclusive),
      },
    })
  aggregated_fusion_sites.sort(
    key=lambda row: (row["inclusive_ns"] or 0, row["id"] or ""),
    reverse=True)
  runtime["fusion_sites"] = aggregated_fusion_sites
  runtime["top_fusion_sites"] = aggregated_fusion_sites[:5]
  fusion_site_summary = {
    "selected_count": len(aggregated_fusion_sites),
    "joined_count": sum(
      site["event_join_status"] == "joined" for site in aggregated_fusion_sites),
    "partial_count": sum(
      site["event_join_status"] == "partial" for site in aggregated_fusion_sites),
    "missing_count": sum(
      site["event_join_status"] == "missing" for site in aggregated_fusion_sites),
    "complete": all(
      site["event_join_status"] == "joined" for site in aggregated_fusion_sites),
    "invocation_count": len(reports),
  }
  runtime["fusion_site_summary"] = fusion_site_summary
  runtime["per_invocation"]["fusion_site_summary"] = fusion_site_summary
  aggregate_summary = dict(runtime["summary"])
  summary_stats = {}
  for key in (
      "allocation_count", "deallocation_count", "copy_count",
      "materialized_write_count", "materialized_read_count",
      "transpose_count", "pack_count", "unpack_count",
      "parallel_region_count", "event_mismatch_count"):
    values = [report["runtime"]["summary"].get(key)
              for report in reports
              if isinstance(report["runtime"]["summary"].get(key), int)]
    summary_stats[key] = numeric_stats(values)
    aggregate_summary[key] = None
  for key, known_key in (
      ("allocation_bytes", "allocation_bytes_known"),
      ("deallocation_bytes", "deallocation_bytes_known"),
      ("copy_bytes", "copy_bytes_known"),
      ("runtime_materialized_write_bytes", "materialized_write_bytes_known"),
      ("runtime_materialized_read_bytes", "materialized_read_bytes_known"),
      ("expected_materialized_read_bytes",
       "expected_materialized_read_bytes_known"),
      ("runtime_transpose_write_bytes", "runtime_transpose_write_bytes_known"),
      ("pack_bytes", "pack_bytes_known"),
      ("unpack_bytes", "unpack_bytes_known"),
      ("peak_live_bytes", "peak_live_proven"),
      ("top_level_time_ns", "top_level_time_known")):
    known_rows = [report["runtime"]["summary"] for report in reports
                  if report["runtime"]["summary"].get(known_key) is True]
    values = [row.get(key) for row in known_rows
              if isinstance(row.get(key), int)]
    summary_stats[key] = {
      "known_count": len(values),
      "median": statistics.median(values) if values else None,
      "worst": max(values) if values else None,
    }
    all_known = len(values) == len(reports)
    aggregate_summary[key] = median_scalar(values) if all_known else None
    aggregate_summary[known_key] = all_known
  materialized_completion = [
    report["runtime"]["summary"].get("materialized_read_complete")
    for report in reports
  ]
  aggregate_summary["materialized_read_complete"] = (
    False if any(value is False for value in materialized_completion) else
    True if all(value is True for value in materialized_completion) else None)
  summary_stats["materialized_read_complete"] = {
    "complete_count": sum(value is True for value in materialized_completion),
    "incomplete_count": sum(value is False for value in materialized_completion),
    "unknown_count": sum(value is None for value in materialized_completion),
  }
  runtime["summary"] = aggregate_summary
  runtime["per_invocation"]["summary"] = summary_stats
  for key in ("unattributed_event_count", "unattributed_time_ns",
              "unknown_time_ns"):
    values = numeric_values(key)
    runtime["per_invocation"][key] = numeric_stats(values)
    runtime[key] = (statistics.median(values) if values else None)

  copy_layout = {}
  copy_layout_per_invocation = {}
  for kind in sorted({
      kind for report in reports for kind in report["runtime"].get(
        "copy_layout", {})}):
    rows = [report["runtime"].get("copy_layout", {}).get(kind, {})
            for report in reports]
    counts = [row.get("count") for row in rows
              if isinstance(row.get("count"), int)]
    observed = [row for row in rows if row.get("count", 0) > 0]
    known_values = [row.get("bytes") for row in observed
                    if row.get("bytes_known") is True and
                    isinstance(row.get("bytes"), int)]
    bytes_known = bool(observed) and len(known_values) == len(observed)
    copy_layout_per_invocation[kind] = {
      "count": numeric_stats(counts),
      "bytes": numeric_stats(known_values),
      "bytes_known_count": len(known_values),
    }
    copy_layout[kind] = {
      "count": statistics.median(counts) if counts else None,
      "bytes": statistics.median(known_values) if bytes_known else None,
      "bytes_known": bytes_known if observed else None,
      "status": "known" if bytes_known else
        "unknown" if observed else "not_observed",
    }
  runtime["copy_layout"] = copy_layout
  runtime["per_invocation"]["copy_layout"] = copy_layout_per_invocation

  # A missing event in any invocation is a missing join, not an observed zero.
  allocation_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
  for report in reports:
    for row in report.get("top_allocations", []):
      allocation_rows[row["id"]].append(row)
  top_allocations = []
  for identifier, rows in allocation_rows.items():
    row = dict(rows[0])
    runtime_bytes = [item["runtime_bytes"] for item in rows
                     if item.get("runtime_bytes_known") is True and
                     isinstance(item.get("runtime_bytes"), int)]
    runtime_calls = [item["runtime_calls"] for item in rows
                     if isinstance(item.get("runtime_calls"), int)]
    row["runtime_bytes"] = (
      statistics.median(runtime_bytes)
      if len(rows) == len(runtime_bytes) else None)
    row["runtime_bytes_known"] = len(rows) == len(runtime_bytes)
    row["runtime_calls"] = (
      statistics.median(runtime_calls)
      if len(rows) == len(runtime_calls) else None)
    row["missing_runtime_event"] = any(
      item.get("missing_runtime_event") is True for item in rows)
    top_allocations.append(row)
  top_allocations.sort(
    key=lambda item: (
      item.get("static_bytes") or 0,
      item["runtime_calls"] if isinstance(item.get("runtime_calls"), int)
      else -1,
      item["id"],
    ),
    reverse=True,
  )
  allocation_coverages = [report["runtime"].get(
    "allocation_coverage", {}) for report in reports]
  runtime["allocation_coverage"] = dict(allocation_coverages[0])
  runtime["allocation_coverage"]["complete"] = all(
    coverage.get("complete", False) for coverage in allocation_coverages)
  runtime["allocation_coverage"]["missing_profile_ids"] = sorted({
    identifier
    for coverage in allocation_coverages
    for identifier in coverage.get("missing_profile_ids", [])
  })
  observed_counts = [coverage.get("observed_count") for coverage in allocation_coverages
                     if isinstance(coverage.get("observed_count"), int)]
  if observed_counts:
    runtime["allocation_coverage"]["observed_count"] = min(observed_counts)
  runtime["allocation_coverage"]["all_invocations_observed_count"] = max(
    0, runtime["allocation_coverage"].get("static_count", 0) -
    len(runtime["allocation_coverage"]["missing_profile_ids"])
  )
  workspace_coverages = [report["runtime"].get(
    "workspace_coverage", {}) for report in reports]
  runtime["workspace_coverage"] = dict(workspace_coverages[0])
  runtime["workspace_coverage"]["complete"] = all(
    coverage.get("complete", False) for coverage in workspace_coverages)
  joined_counts = [coverage.get("joined_count")
                   for coverage in workspace_coverages
                   if isinstance(coverage.get("joined_count"), int)]
  if joined_counts:
    runtime["workspace_coverage"]["joined_count"] = min(joined_counts)
    runtime["workspace_coverage"]["all_invocations_joined_count"] = min(
      joined_counts)
  runtime["peak_live_proven"] = all(
    bool(report["runtime"].get("peak_live_proven")) for report in reports)
  first_total = summary_values["top_level_time_ns"]
  total_ns = statistics.median(first_total) if first_total else None

  # Re-rank operation hotspots using all invocations instead of retaining the
  # first invocation's bytes and time values.
  cost_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
  for report in reports:
    for row in report.get("top_costs", []):
      cost_rows[row["id"]].append(row)
  top_costs = []
  for identifier, rows in cost_rows.items():
    row = dict(rows[0])
    attributed = [item["attributed_ns"] for item in rows
                  if isinstance(item.get("attributed_ns"), int)]
    inclusive = [item["inclusive_ns"] for item in rows
                 if isinstance(item.get("inclusive_ns"), int)]
    exclusive = [item["exclusive_ns"] for item in rows
                 if isinstance(item.get("exclusive_ns"), int)]
    row["attributed_ns"] = statistics.median(attributed) if attributed else None
    row["inclusive_ns"] = statistics.median(inclusive) \
      if len(rows) == len(inclusive) else None
    row["exclusive_ns"] = statistics.median(exclusive) \
      if len(rows) == len(exclusive) else None
    row["time_basis"] = "exclusive" if row["exclusive_ns"] is not None \
      else "inclusive_unknown_exclusive"
    row["time_fraction"] = (
      row["attributed_ns"] / total_ns
      if total_ns and row["attributed_ns"] is not None else None)
    top_costs.append(row)
  top_costs.sort(key=lambda item: (item["attributed_ns"] or 0, item["id"]),
                 reverse=True)

  worker_reports = [report["runtime"].get("worker_attribution", {})
                    for report in reports]
  worker_rows_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
  for worker_report in worker_reports:
    for row in worker_report.get("operations", []):
      worker_rows_by_id[row["id"]].append(row)
  aggregated_worker_operations = []
  for identifier, rows in worker_rows_by_id.items():
    row = dict(rows[0])
    union_values = [item["wall_union_ns"] for item in rows
                    if item.get("wall_union_known") is True and
                    isinstance(item.get("wall_union_ns"), int)]
    cpu_values = [item["exclusive_cpu_ns"] for item in rows
                  if isinstance(item.get("exclusive_cpu_ns"), int)]
    inclusive_values = [item["inclusive_cpu_ns"] for item in rows
                        if isinstance(item.get("inclusive_cpu_ns"), int)]
    share_values = [item["wall_union_share_of_top_level"] for item in rows
                    if isinstance(item.get("wall_union_share_of_top_level"),
                                  (int, float))]
    attributed_values = [item["wall_attributed_ns"] for item in rows
                         if item.get("wall_attributed_known") is True and
                         isinstance(item.get("wall_attributed_ns"), int)]
    attributed_share_values = [
      item["wall_attributed_share_of_top_level"] for item in rows
      if isinstance(item.get("wall_attributed_share_of_top_level"),
                    (int, float))]
    estimated_calls = [item["calls_estimated"] for item in rows
                       if isinstance(item.get("calls_estimated"), int)]
    row["event_join_status"] = (
      "joined" if len(rows) == len(reports) else "partial")
    row["invocation_count"] = len(reports)
    row["joined_invocation_count"] = len(rows)
    row["wall_union_ns"] = median_scalar(union_values) \
      if len(union_values) == len(reports) else None
    row["exclusive_cpu_ns"] = median_scalar(cpu_values) \
      if len(cpu_values) == len(reports) else None
    row["inclusive_cpu_ns"] = median_scalar(inclusive_values) \
      if len(inclusive_values) == len(reports) else None
    row["wall_union_share_of_top_level"] = median_scalar(share_values) \
      if len(share_values) == len(reports) else None
    row["wall_attributed_ns"] = median_scalar(attributed_values) \
      if len(attributed_values) == len(reports) else None
    row["wall_attributed_known"] = len(attributed_values) == len(reports)
    row["wall_attributed_share_of_top_level"] = median_scalar(
      attributed_share_values) \
      if len(attributed_share_values) == len(reports) else None
    row["calls_estimated"] = median_scalar(estimated_calls) \
      if len(estimated_calls) == len(reports) else None
    row["per_invocation"] = {
      "wall_union_ns": numeric_stats(union_values),
      "exclusive_cpu_ns": numeric_stats(cpu_values),
      "inclusive_cpu_ns": numeric_stats(inclusive_values),
      "wall_union_share_of_top_level": numeric_stats(share_values),
      "wall_attributed_ns": numeric_stats(attributed_values),
      "wall_attributed_share_of_top_level": numeric_stats(
        attributed_share_values),
      "calls_estimated": numeric_stats(estimated_calls),
    }
    aggregated_worker_operations.append(row)
  aggregated_worker_operations.sort(
    key=lambda row: (row.get("wall_union_ns") or 0, row["id"]), reverse=True)
  worker_known_totals = [worker_report.get("exclusive_cpu_total_ns")
                         for worker_report in worker_reports
                         if isinstance(worker_report.get(
                           "exclusive_cpu_total_ns"), int)]
  worker_attribution_complete = bool(worker_rows_by_id) and all(
    worker_report.get("complete_for_observed_events") is True
    for worker_report in worker_reports) and all(
      len(rows) == len(reports) for rows in worker_rows_by_id.values())
  runtime["worker_attribution"] = {
    "time_domain": "worker_cpu",
    "wall_union_semantics": "per_operation_non_additive",
    "observed_event_count": len(worker_rows_by_id),
    "joined_event_count": sum(
      len(rows) == len(reports) for rows in worker_rows_by_id.values()),
    "invocation_count": len(reports),
    "exclusive_cpu_total_ns": median_scalar(worker_known_totals)
      if len(worker_known_totals) == len(reports) else None,
    "exclusive_cpu_total_per_invocation": numeric_stats(worker_known_totals),
    "complete_for_observed_events": worker_attribution_complete,
    "operations": aggregated_worker_operations,
    "top_operations_by_wall_union": aggregated_worker_operations[:20],
    "top_operations_by_wall_attributed": sorted(
      aggregated_worker_operations,
      key=lambda row: (row.get("wall_attributed_ns") or 0, row["id"]),
      reverse=True)[:20],
    "wall_attributed_unknown_count": sum(
      row.get("wall_attributed_ns") is None
      for row in aggregated_worker_operations),
  }
  wall_partitions = [report["runtime"].get("wall_partition", {})
                     for report in reports]
  partition_fields = (
    "worker_ops_wall_attributed_ns", "parallel_region_wall_ns",
    "parallel_region_gap_wall_ns", "parallel_region_estimated_gap_wall_ns",
    "parallel_region_unproven_wall_ns", "sequential_ops_exclusive_ns",
    "accounted_wall_ns", "unaccounted_wall_ns")
  aggregated_partition: dict[str, Any] = {
    "basis": "equal_split_across_concurrent_worker_spans",
    "additive": True,
    "top_level_wall_ns": median_scalar(summary_values["top_level_time_ns"])
      if len(summary_values["top_level_time_ns"]) == len(reports) else None,
    "invocation_count": len(reports),
    "parallel_regions": wall_partitions[0].get("parallel_regions", []),
  }
  for field in partition_fields:
    values = [partition.get(field) for partition in wall_partitions
              if isinstance(partition.get(field), int)]
    aggregated_partition[field] = (
      median_scalar(values) if len(values) == len(reports) else None)
    aggregated_partition[f"{field}_per_invocation"] = numeric_stats(values)
  runtime["wall_partition"] = aggregated_partition
  runtime["complete"] = runtime["complete"] and runtime["complete_all"]
  runtime["incomplete_reasons"] = sorted(set(
    reason for report in reports for reason in report["runtime"]["incomplete_reasons"]
  ))
  first["invocation_id"] = None
  first["invocation_ids"] = sorted(invocation_ids)
  first["top_costs"] = top_costs[:20]
  first["top_allocations"] = top_allocations[:10]
  first["aggregation"] = "per-invocation-median"
  return first
