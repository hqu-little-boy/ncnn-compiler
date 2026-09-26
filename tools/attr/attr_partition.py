"""Wall-domain partition, worker shares, and hotspot ranking."""

from __future__ import annotations

from typing import Any

from .attr_schema import AttributionError, require_nonnegative_integer


def rank_top_costs(
    joined: list[dict[str, Any]], profile_summary: dict[str, Any],
    measured_event_time_ns: int) -> tuple[int, list[dict[str, Any]]]:
  profile_total_value = profile_summary.get("top_level_time_ns")
  if profile_total_value is None:
    total_ns = measured_event_time_ns
  else:
    total_ns = require_nonnegative_integer(
      profile_total_value, "profile summary top_level_time_ns")
    if total_ns == 0:
      total_ns = measured_event_time_ns
  joined.sort(key=lambda row: (row["attributed_ns"], row["id"]), reverse=True)
  top = []
  for row in joined[:20]:
    item = dict(row)
    item["time_basis"] = "exclusive" if row["exclusive_ns"] is not None \
      else "inclusive_unknown_exclusive"
    item["time_fraction"] = (
      row["attributed_ns"] / total_ns if total_ns else None)
    top.append(item)
  return total_ns, top


def rank_top_allocations(
    allocation_objects: dict[int, dict[str, Any]],
    allocation_events: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
  top_allocations = []
  workspace_total = 0
  workspace_joined = 0
  for identifier, buffer in allocation_objects.items():
    if "workspace_slot" in buffer:
      workspace_total += 1
    event = allocation_events.get(identifier)
    if event is not None:
      workspace_joined += int("workspace_slot" in buffer)
    static_bytes = buffer.get("bytes")
    static_bytes_known = buffer.get("bytes_known")
    if static_bytes_known is False or isinstance(static_bytes, bool) or \
        not isinstance(static_bytes, int):
      static_bytes = None
    top_allocations.append({
      "id": identifier,
      "operation": buffer.get("id"),
      "function": buffer.get("function"),
      "static_bytes": static_bytes,
      "static_bytes_known": static_bytes is not None and static_bytes_known is not False,
      "runtime_bytes": event.get("bytes") if event else None,
      "runtime_bytes_known": event.get("bytes_known") if event else None,
      "runtime_calls": event.get("calls") if event else None,
      "liveness_status": buffer.get("liveness_status", "unknown"),
      "workspace_slot": buffer.get("workspace_slot"),
      "workspace_slot_owner": buffer.get("workspace_slot_owner"),
      "workspace_join_status": buffer.get(
        "workspace_join_status", "not_applicable"),
      "missing_runtime_event": event is None,
    })
  top_allocations.sort(
    key=lambda item: (
      item["static_bytes"] or 0,
      item["runtime_calls"] if isinstance(item["runtime_calls"], int) else -1,
      item["id"],
    ),
    reverse=True,
  )
  return top_allocations, workspace_total, workspace_joined


def build_workspace_coverage(
    workspace_total: int, workspace_joined: int, unknown: list[str],
    incomplete_reasons: list[str]
) -> tuple[dict[str, Any], list[str], list[str]]:
  workspace_coverage = {
    "static_count": workspace_total,
    "joined_count": workspace_joined,
    "complete": workspace_joined == workspace_total,
  }
  if not workspace_coverage["complete"] and workspace_total:
    unknown = list(dict.fromkeys([*unknown, "runtime_workspace_join_incomplete"]))
    incomplete_reasons.append("runtime_workspace_join_incomplete")
  return workspace_coverage, unknown, incomplete_reasons


def window_normalization(
    profile_summary: dict[str, Any], missing_allocations: list[int]
) -> tuple[bool, int | None, int]:
  runtime_peak_live_proven = bool(
    profile_summary.get("peak_live_proven", False) and not missing_allocations)
  top_level_wall_ns = profile_summary.get("top_level_time_ns")
  if top_level_wall_ns is not None:
    top_level_wall_ns = require_nonnegative_integer(
      top_level_wall_ns, "profile summary top_level_time_ns")
  sampling = profile_summary.get("worker_sampling", {})
  sample_scale = sampling.get("duty", 1)
  if isinstance(sample_scale, bool) or not isinstance(sample_scale, int) or \
      sample_scale < 1:
    raise AttributionError("worker sampling duty must be a positive integer")
  return runtime_peak_live_proven, top_level_wall_ns, sample_scale


def build_worker_section(
    worker_joined: list[dict[str, Any]],
    worker_operation_events: dict[int, dict[str, Any]],
    worker_unattributed: int, profile: dict[str, Any],
    profile_summary: dict[str, Any], top_level_wall_ns: int | None,
    unknown: list[str], incomplete_reasons: list[str]
) -> tuple[dict[str, Any], list[str], list[str]]:
  # Wall shares are window-normalized and already projected by the runtime;
  # do not scale them again.  Wall-union remains a raw sampled per-op union and
  # is never projected, because unions are not additive across workers.
  for event in worker_joined:
    event["wall_union_share_of_top_level"] = (
      event["wall_union_ns"] / top_level_wall_ns
      if event["wall_union_known"] and top_level_wall_ns else None)
  worker_joined.sort(
    key=lambda row: (row["wall_union_ns"] or 0, row["id"]), reverse=True)
  for event in worker_joined:
    projected = event.get("wall_attributed_estimated_ns")
    if event.get("wall_attributed_known") and isinstance(projected, int):
      if top_level_wall_ns is not None and projected > top_level_wall_ns:
        raise AttributionError(
          "projected worker wall attribution cannot exceed top-level wall time")
      event["wall_attributed_share_of_top_level"] = (
        projected / top_level_wall_ns if top_level_wall_ns else None)
    else:
      event["wall_attributed_share_of_top_level"] = None
  worker_joined_by_attributed = sorted(
    worker_joined,
    key=lambda row: (row.get("wall_attributed_ns") or 0, row["id"]),
    reverse=True)
  worker_exclusive_values = [
    event["exclusive_cpu_ns"] for event in worker_joined
    if isinstance(event["exclusive_cpu_ns"], int)]
  worker_exclusive_unknown_count = (
    len(worker_operation_events) - len(worker_exclusive_values))
  worker_union_unknown_count = sum(
    not event["wall_union_known"] for event in worker_joined)
  wall_attributed_unknown_count = sum(
    not event.get("wall_attributed_known") for event in worker_joined)
  wall_attribution_expected = "worker_wall_attribution" in profile_summary
  worker_attribution_complete = bool(worker_operation_events) and \
    len(worker_joined) == len(worker_operation_events) and \
    worker_exclusive_unknown_count == 0 and worker_union_unknown_count == 0 and \
    (not wall_attribution_expected or wall_attributed_unknown_count == 0) and \
    profile_summary.get("event_mismatch_count", 0) == 0 and \
    profile.get("complete") is True
  if worker_operation_events and not worker_attribution_complete:
    unknown = list(dict.fromkeys(
      [*unknown, "runtime_worker_attribution_incomplete"]))
    incomplete_reasons.append("runtime_worker_attribution_incomplete")
  worker_attribution = {
    "time_domain": "worker_cpu",
    "wall_union_semantics": "per_operation_non_additive",
    "observed_event_count": len(worker_operation_events),
    "joined_event_count": len(worker_joined),
    "unattributed_event_count": worker_unattributed,
    "exclusive_cpu_known_count": len(worker_exclusive_values),
    "exclusive_cpu_unknown_count": worker_exclusive_unknown_count,
    "exclusive_cpu_total_ns": sum(worker_exclusive_values),
    "wall_union_unknown_count": worker_union_unknown_count,
    "wall_attributed_unknown_count": sum(
      not row.get("wall_attributed_known") for row in worker_joined),
    "complete_for_observed_events": worker_attribution_complete,
    "operations": worker_joined,
    "top_operations_by_wall_union": worker_joined[:20],
    "top_operations_by_wall_attributed": worker_joined_by_attributed[:20],
  }
  return worker_attribution, unknown, incomplete_reasons


def build_wall_partition(
    worker_joined: list[dict[str, Any]],
    parallel_coverage: list[dict[str, Any]],
    joined: list[dict[str, Any]], sample_scale: int,
    top_level_wall_ns: int | None) -> dict[str, Any]:
  # Wall-domain partition of top-level time. Worker CPU time is never mixed in.
  # Two normalizations are reported: raw sampled-window measurements, and the
  # runtime's uniform projection of those measurements onto full region wall.
  attributed_wall_total = sum(
    row.get("wall_attributed_estimated_ns") or 0 for row in worker_joined
    if row.get("wall_attributed_known") and
    isinstance(row.get("wall_attributed_estimated_ns"), int))
  attributed_share_total = sum(
    row.get("wall_attributed_share_of_sampled_window") or 0
    for row in worker_joined
    if isinstance(row.get("wall_attributed_share_of_sampled_window"),
                  (int, float)))
  region_wall_total = 0
  sampled_window_total = 0
  covered_sampled_total = 0
  covered_estimated_total = 0
  exact_gap_wall_total = 0
  estimated_gap_wall_total = 0
  unproven_region_wall = 0
  region_wall_without_sample = 0
  for region in parallel_coverage:
    region_wall_total += region["wall_ns"]
    if region["wall_exclusive_estimated_known"] and isinstance(
        region.get("wall_exclusive_estimated_ns"), int):
      if region["exclusive_semantics"] == \
          "region_wall_minus_worker_span_union":
        exact_gap_wall_total += region["wall_exclusive_estimated_ns"]
      else:
        estimated_gap_wall_total += region["wall_exclusive_estimated_ns"]
      if isinstance(region.get("worker_covered_wall_ns"), int):
        covered_estimated_total += region["worker_covered_wall_ns"]
      if isinstance(region.get("worker_covered_wall_sampled_ns"), int):
        covered_sampled_total += region["worker_covered_wall_sampled_ns"]
      if isinstance(region.get("sampled_window_wall_ns"), int):
        sampled_window_total += region["sampled_window_wall_ns"]
    else:
      unproven_region_wall += region["wall_ns"]
      if not region.get("region_worker_spans"):
        region_wall_without_sample += region["wall_ns"]
  sequential_exclusive_total = sum(
    row["attributed_ns"] for row in joined
    if row["category"] != "parallel" and isinstance(row.get("attributed_ns"), int))
  projection_values = [region.get("wall_projection_factor")
                       for region in parallel_coverage
                       if isinstance(region.get("wall_projection_factor"),
                                     (int, float))]
  projection = projection_values[0] if projection_values else 1.0
  parallel_gap_total = exact_gap_wall_total + estimated_gap_wall_total
  worker_unjoined_wall = (
    covered_estimated_total - attributed_wall_total
    if covered_estimated_total >= attributed_wall_total else None)
  accounted_wall_total = (sequential_exclusive_total + attributed_wall_total +
                          (worker_unjoined_wall or 0) + parallel_gap_total)
  wall_partition = {
    "basis": "equal_split_across_concurrent_worker_spans",
    "additive": True,
    "time_domain": "wall",
    "normalization": "sampled_window_duration",
    "sample_scale": sample_scale,
    "wall_projection_factor": projection,
    "top_level_wall_ns": top_level_wall_ns,
    "parallel_region_wall_ns": region_wall_total,
    "parallel_region_exact_gap_wall_ns": exact_gap_wall_total,
    "parallel_region_estimated_gap_wall_ns": estimated_gap_wall_total,
    "parallel_wall_gap_wall_ns": parallel_gap_total,
    "parallel_region_unproven_wall_ns": unproven_region_wall,
    "parallel_region_wall_without_sampled_spans_ns": region_wall_without_sample,
    "sampled_window_wall_ns": sampled_window_total,
    "sampled_window_fraction_of_parallel_wall": (
      sampled_window_total / region_wall_total if region_wall_total else None),
    "worker_covered_wall_sampled_ns": covered_sampled_total,
    "worker_covered_wall_estimated_ns": covered_estimated_total,
    "worker_ops_wall_attributed_ns": attributed_wall_total,
    "worker_ops_wall_unjoined_ns": worker_unjoined_wall,
    "worker_ops_wall_share_of_sampled_window": attributed_share_total,
    "parallel_wall_coverage_share": (
      covered_sampled_total / sampled_window_total
      if sampled_window_total else None),
    "parallel_wall_gap_share": (
      1.0 - covered_sampled_total / sampled_window_total
      if sampled_window_total else None),
    "sequential_ops_exclusive_ns": sequential_exclusive_total,
    "sequential_ops_share_of_top_level": (
      sequential_exclusive_total / top_level_wall_ns
      if top_level_wall_ns else None),
    "accounted_wall_ns": accounted_wall_total,
    "unaccounted_wall_ns": (
      top_level_wall_ns - accounted_wall_total
      if top_level_wall_ns is not None else None),
    "parallel_regions": parallel_coverage,
  }
  return wall_partition
