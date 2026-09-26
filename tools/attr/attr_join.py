"""Input readers, plan indexes, and the runtime-event join."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from .attr_schema import (AttributionError, category_name,
                          require_nonnegative_integer)


class JoinResult(NamedTuple):
  """Accumulators produced by the single pass over profile events."""

  joined: list[dict[str, Any]]
  allocation_events: dict[int, dict[str, Any]]
  operation_events: dict[int, dict[str, Any]]
  worker_operation_events: dict[int, dict[str, Any]]
  worker_joined: list[dict[str, Any]]
  worker_unattributed: int
  parallel_coverage: list[dict[str, Any]]
  movement_totals: dict[str, dict[str, Any]]
  unattributed: int
  unknown_time_ns: int
  measured_event_time_ns: int
  known_event_time_ns: int
  category_totals: dict[str, int]
  observed_allocations: set[int]
  allocation_bytes_unknown: bool


def read_json(path: str) -> dict[str, Any]:
  try:
    with Path(path).open(encoding="utf-8") as stream:
      value = json.load(stream)
  except (OSError, json.JSONDecodeError) as error:
    raise AttributionError(f"cannot read {path}: {error}") from error
  if not isinstance(value, dict):
    raise AttributionError(f"{path}: root must be an object")
  return value


def read_profile(path: str) -> dict[str, Any] | list[dict[str, Any]]:
  try:
    text = Path(path).read_text(encoding="utf-8")
  except OSError as error:
    raise AttributionError(f"cannot read {path}: {error}") from error
  try:
    value = json.loads(text)
  except json.JSONDecodeError:
    rows = []
    try:
      rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
      raise AttributionError(f"cannot read {path}: {error}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
      raise AttributionError(f"{path}: profile NDJSON rows must be objects")
    value = rows
  if isinstance(value, dict):
    return value
  if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
    return value
  raise AttributionError(f"{path}: profile must be an object or non-empty NDJSON")


def read_perf(path: str, model: str, mode: str) -> dict[str, Any]:
  try:
    with Path(path).open(encoding="utf-8") as stream:
      rows = [json.loads(line) for line in stream if line.strip()]
  except (OSError, json.JSONDecodeError) as error:
    raise AttributionError(f"cannot read {path}: {error}") from error
  matches = [
    row for row in rows
    if isinstance(row, dict)
    and row.get("model") == model
    and row.get("mode", "end_to_end") == mode
  ]
  if not matches:
    raise AttributionError(f"no perf row for model={model!r}, mode={mode!r}")
  if len(matches) > 1:
    raise AttributionError(
      f"ambiguous perf rows for model={model!r}, mode={mode!r}")
  return matches[0]


def operation_index(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
  operations = plan.get("operations", [])
  if not isinstance(operations, list):
    raise AttributionError("plan.operations must be an array")
  result: dict[int, dict[str, Any]] = {}
  for operation in operations:
    if not isinstance(operation, dict):
      raise AttributionError("plan.operations contains a non-object")
    identifier = operation.get("profile_id")
    if (isinstance(identifier, bool) or not isinstance(identifier, int) or
        identifier < 0 or identifier > (1 << 64) - 1):
      raise AttributionError(
        "operation profile_id must be an unsigned 64-bit integer")
    if identifier in result:
      raise AttributionError(f"duplicate operation profile_id={identifier}")
    result[identifier] = operation
  return result


def allocation_index(plan: dict[str, Any]) -> set[int]:
  buffers = plan.get("buffers")
  if not isinstance(buffers, list):
    raise AttributionError("plan.buffers must be an array")
  result: set[int] = set()
  for buffer in buffers:
    if not isinstance(buffer, dict):
      raise AttributionError("plan.buffers contains a non-object")
    identifier = buffer.get("profile_id")
    if (isinstance(identifier, bool) or not isinstance(identifier, int) or
        identifier < 0 or identifier > (1 << 64) - 1):
      raise AttributionError(
        "buffer profile_id must be an unsigned 64-bit integer")
    if identifier in result:
      raise AttributionError(f"duplicate buffer profile_id={identifier}")
    result.add(identifier)
  return result


def allocation_metadata(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
  buffers = plan.get("buffers")
  if not isinstance(buffers, list):
    raise AttributionError("plan.buffers must be an array")
  result: dict[int, dict[str, Any]] = {}
  for buffer in buffers:
    if not isinstance(buffer, dict):
      raise AttributionError("plan.buffers contains a non-object")
    identifier = buffer.get("profile_id")
    if (isinstance(identifier, bool) or not isinstance(identifier, int) or
        identifier < 0 or identifier > (1 << 64) - 1):
      raise AttributionError(
        "buffer profile_id must be an unsigned 64-bit integer")
    result[identifier] = buffer
  return result


def collect_fusions(plan: dict[str, Any]) -> tuple[list[Any], set[int]]:
  fusion_records = plan.get("fusions", [])
  if not isinstance(fusion_records, list):
    raise AttributionError("plan.fusions must be an array")
  fusion_profile_ids: set[int] = set()
  for fusion in fusion_records:
    if not isinstance(fusion, dict):
      raise AttributionError("plan.fusions contains a non-object")
    for field in ("profile_id", "tail_profile_id"):
      identifier = fusion.get(field)
      if identifier is None:
        continue
      if (isinstance(identifier, bool) or not isinstance(identifier, int) or
          identifier < 0 or identifier > (1 << 64) - 1):
        raise AttributionError(f"fusion {field} must be an unsigned 64-bit integer")
      if identifier in fusion_profile_ids:
        raise AttributionError(f"duplicate fusion profile_id={identifier}")
      fusion_profile_ids.add(identifier)
  return fusion_records, fusion_profile_ids


def validate_profile_events(
    profile: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
  events = profile.get("events", [])
  if not isinstance(events, list):
    raise AttributionError("profile.events must be an array")
  profile_summary = profile.get("summary")
  if not isinstance(profile_summary, dict):
    raise AttributionError("profile.summary must be an object")
  for bytes_field, known_field in (
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
    if known_field not in profile_summary:
      continue
    known = profile_summary[known_field]
    if not isinstance(known, bool):
      raise AttributionError(f"profile summary {known_field} must be boolean")
    value = profile_summary.get(bytes_field)
    if known:
      if value is None:
        raise AttributionError(
          f"profile summary {known_field}=true requires {bytes_field}")
      require_nonnegative_integer(value, f"profile summary {bytes_field}")
    elif value is not None:
      raise AttributionError(
        f"profile summary {known_field}=false requires null {bytes_field}")
  if "materialized_read_complete" in profile_summary:
    materialized_complete = profile_summary["materialized_read_complete"]
    if materialized_complete is not None and not isinstance(
        materialized_complete, bool):
      raise AttributionError(
        "profile summary materialized_read_complete must be boolean or null")
    if materialized_complete is True and (
        profile_summary.get("materialized_read_bytes_known") is not True or
        profile_summary.get("expected_materialized_read_bytes_known") is not True):
      raise AttributionError(
        "complete materialized reads require known observed and expected bytes")
    if materialized_complete is not None and \
        profile_summary.get("materialized_read_bytes_known") is True and \
        profile_summary.get("expected_materialized_read_bytes_known") is True:
      expected = profile_summary["expected_materialized_read_bytes"]
      observed = profile_summary["runtime_materialized_read_bytes"]
      if materialized_complete != (expected == observed):
        raise AttributionError(
          "materialized read completeness disagrees with byte totals")
  if ("peak_live_proven" in profile_summary and
      not isinstance(profile_summary["peak_live_proven"], bool)):
    raise AttributionError("profile summary peak_live_proven must be boolean")
  mismatch_value = profile_summary.get("event_mismatch_count")
  if mismatch_value is not None:
    mismatch_count = require_nonnegative_integer(
      mismatch_value, "profile summary event_mismatch_count")
    if mismatch_count != 0:
      raise AttributionError(
        "profile event mismatch count is non-zero; attribution is incomplete")
  if "top_level_time_known" in profile_summary and not isinstance(
      profile_summary["top_level_time_known"], bool):
    raise AttributionError("profile summary top_level_time_known must be boolean")
  return events, profile_summary


def join_profile_events(
    profile: dict[str, Any], events: list[Any],
    operations: dict[int, dict[str, Any]], static_allocations: set[int],
    fusion_profile_ids: set[int]) -> JoinResult:
  joined: list[dict[str, Any]] = []
  allocation_events: dict[int, dict[str, Any]] = {}
  operation_events: dict[int, dict[str, Any]] = {}
  worker_operation_events: dict[int, dict[str, Any]] = {}
  worker_joined: list[dict[str, Any]] = []
  worker_unattributed = 0
  parallel_coverage: list[dict[str, Any]] = []
  movement_totals: dict[str, dict[str, Any]] = {
    kind: {"count": 0, "bytes": 0, "bytes_known": True}
    for kind in ("copy", "transpose", "pack", "unpack")
  }
  unattributed = 0
  unknown_time_ns = 0
  measured_event_time_ns = 0
  known_event_time_ns = 0
  category_totals: dict[str, int] = defaultdict(int)
  observed_allocations: set[int] = set()
  allocation_bytes_unknown = False
  seen_events: set[tuple[int, str]] = set()
  for event in events:
    if not isinstance(event, dict):
      raise AttributionError("profile.events contains a non-object")
    identifier = event.get("id")
    if (isinstance(identifier, bool) or not isinstance(identifier, int) or
        identifier < 0 or identifier > (1 << 64) - 1):
      raise AttributionError("profile event id must be an unsigned 64-bit integer")
    raw_category = event.get("category")
    if raw_category not in {"operation", "allocation", "deallocation",
                            "copy", "parallel", "transpose", "pack",
                            "unpack", "materialized_write",
                            "materialized_read", "worker_operation"}:
      raise AttributionError("profile event category is unsupported")
    bytes_value = event.get("bytes")
    if bytes_value is not None:
      bytes_value = require_nonnegative_integer(bytes_value, "event.bytes")
      if bytes_value > (1 << 64) - 1:
        raise AttributionError("event.bytes must fit in an unsigned 64-bit integer")
    bytes_known = event.get("bytes_known")
    if bytes_known is not None and not isinstance(bytes_known, bool):
      raise AttributionError("event.bytes_known must be boolean")
    if bytes_known is True and bytes_value is None:
      raise AttributionError("known event bytes cannot be null")
    if bytes_known is False and bytes_value is not None:
      raise AttributionError("unknown event bytes must be null")
    event_key = (identifier, raw_category)
    if event_key in seen_events:
      raise AttributionError("duplicate profile event id/category")
    seen_events.add(event_key)
    if raw_category == "worker_operation":
      if profile.get("schema_version") != 3 or \
          event.get("time_domain") != "worker_cpu":
        raise AttributionError(
          "worker_operation events require schema-3 worker_cpu time_domain")
      inclusive_worker_ns = require_nonnegative_integer(
        event.get("inclusive_ns"), "worker event.inclusive_ns")
      calls = require_nonnegative_integer(
        event.get("calls"), "worker event.calls")
      if calls == 0:
        raise AttributionError("worker event.calls must be positive")
      exclusive_worker_value = event.get("exclusive_ns")
      if exclusive_worker_value is None:
        exclusive_worker_ns = None
      else:
        exclusive_worker_ns = require_nonnegative_integer(
          exclusive_worker_value, "worker event.exclusive_ns")
        if exclusive_worker_ns > inclusive_worker_ns:
          raise AttributionError(
            "worker exclusive_ns cannot exceed worker inclusive_ns")
      wall_union_known = event.get("worker_wall_union_known")
      if not isinstance(wall_union_known, bool):
        raise AttributionError(
          "worker event.worker_wall_union_known must be boolean")
      wall_union_value = event.get("worker_wall_union_ns")
      if wall_union_known:
        wall_union_ns = require_nonnegative_integer(
          wall_union_value, "worker event.worker_wall_union_ns")
      elif wall_union_value is not None:
        raise AttributionError(
          "unknown worker wall union must have a null duration")
      else:
        wall_union_ns = None
      worker_operation_events[identifier] = {
        "calls": calls,
        "inclusive_cpu_ns": inclusive_worker_ns,
        "exclusive_cpu_ns": exclusive_worker_ns,
        "wall_union_ns": wall_union_ns,
        "wall_union_known": wall_union_known,
      }
      attributed_wall_value = event.get("wall_attributed_ns")
      attributed_wall_known = event.get("wall_attributed_known") is True
      if attributed_wall_value is None:
        attributed_wall_ns = None
        if attributed_wall_known:
          raise AttributionError(
            "known worker wall_attributed_ns cannot be null")
      else:
        attributed_wall_ns = require_nonnegative_integer(
          attributed_wall_value, "worker event.wall_attributed_ns")
        if not attributed_wall_known:
          raise AttributionError(
            "numeric worker wall_attributed_ns requires wall_attributed_known")
      calls_estimated_value = event.get("calls_estimated")
      calls_estimated = (
        require_nonnegative_integer(calls_estimated_value,
                                    "worker event.calls_estimated")
        if calls_estimated_value is not None else None)
      projected_wall_value = event.get("wall_attributed_estimated_ns")
      projected_wall_ns = (
        require_nonnegative_integer(projected_wall_value,
                                    "worker event.wall_attributed_estimated_ns")
        if projected_wall_value is not None else None)
      window_share = event.get("wall_attributed_share_of_sampled_window")
      if window_share is not None and (
          isinstance(window_share, bool) or
          not isinstance(window_share, (int, float)) or window_share < 0):
        raise AttributionError(
          "worker wall_attributed_share_of_sampled_window must be non-negative")
      worker_operation_events[identifier].update({
        "wall_attributed_ns": attributed_wall_ns,
        "wall_attributed_estimated_ns": projected_wall_ns,
        "wall_attributed_share_of_sampled_window": window_share,
        "wall_attributed_known": attributed_wall_known,
        "calls_estimated": calls_estimated,
      })
      operation = operations.get(identifier)
      if operation is None:
        worker_unattributed += 1
        continue
      worker_joined.append({
        "id": identifier,
        "operation": operation.get("id"),
        "kind": operation.get("kind"),
        "source_layer": operation.get("source_layer"),
        "source_name": operation.get("source_name"),
        "calls": calls,
        "calls_estimated": calls_estimated,
        "inclusive_cpu_ns": inclusive_worker_ns,
        "exclusive_cpu_ns": exclusive_worker_ns,
        "wall_union_ns": wall_union_ns,
        "wall_union_known": wall_union_known,
        "wall_attributed_ns": attributed_wall_ns,
        "wall_attributed_estimated_ns": projected_wall_ns,
        "wall_attributed_share_of_sampled_window": window_share,
        "wall_attributed_known": attributed_wall_known,
      })
      continue
    if profile.get("schema_version") == 3 and \
        event.get("time_domain") != "wall":
      raise AttributionError(
        "schema-3 non-worker events must use wall time_domain")
    category = category_name(raw_category)
    if raw_category == "allocation":
      observed_allocations.add(identifier)
    inclusive = require_nonnegative_integer(
      event.get("inclusive_ns"), "event.inclusive_ns")
    calls = require_nonnegative_integer(event.get("calls"), "event.calls")
    if raw_category in movement_totals:
      movement = movement_totals[raw_category]
      movement["count"] += calls
      if bytes_known is False or bytes_value is None:
        movement["bytes_known"] = False
      elif movement["bytes"] <= (1 << 64) - 1 - bytes_value:
        movement["bytes"] += bytes_value
      else:
        movement["bytes_known"] = False
    if raw_category in {"allocation", "deallocation"}:
      if bytes_known is not True or bytes_value is None:
        allocation_bytes_unknown = True
    if raw_category == "allocation":
      allocation_events[identifier] = {
        "calls": calls,
        "bytes": bytes_value,
        "bytes_known": bytes_known,
      }
    if raw_category in {"operation", "parallel"}:
      # Packed tiled GEMM operations are emitted as scf.forall and therefore
      # are reported by the runtime under the parallel event category.
      operation_events[identifier] = {
        "calls": calls,
        "inclusive_ns": inclusive,
        "exclusive_ns": event.get("exclusive_ns"),
      }
    if raw_category == "parallel":
      coverage_known = event.get("wall_exclusive_estimated_known")
      if coverage_known is not None and not isinstance(coverage_known, bool):
        raise AttributionError(
          "parallel wall_exclusive_estimated_known must be boolean")
      parallel_coverage.append({
        "id": identifier,
        "wall_ns": inclusive,
        "worker_covered_wall_ns": event.get("worker_covered_wall_ns"),
        "worker_covered_wall_sampled_ns":
          event.get("worker_covered_wall_sampled_ns"),
        "sampled_window_wall_ns": event.get("sampled_window_wall_ns"),
        "region_wall_ns": event.get("region_wall_ns"),
        "region_worker_spans": event.get("region_worker_spans"),
        "sample_scale": event.get("sample_scale"),
        "wall_projection_factor": event.get("wall_projection_factor"),
        "wall_coverage_share": event.get("wall_coverage_share"),
        "wall_exclusive_estimated_ns": event.get("wall_exclusive_estimated_ns"),
        "wall_exclusive_estimated_known": coverage_known is True,
        "exclusive_semantics": event.get("exclusive_semantics"),
        "exclusive_ns": event.get("exclusive_ns"),
      })
    exclusive_value = event.get("exclusive_ns")
    if exclusive_value is None:
      exclusive = None
      attributed_ns = inclusive
      unknown_time_ns += inclusive
    else:
      exclusive = require_nonnegative_integer(
        exclusive_value, "event.exclusive_ns")
      if exclusive > inclusive:
        raise AttributionError(
          "event.exclusive_ns cannot exceed event.inclusive_ns")
      # Inclusive durations overlap for nested events.  Exclusive duration is
      # the non-overlapping attribution metric whenever the runtime proves it.
      attributed_ns = exclusive
    measured_event_time_ns += attributed_ns
    operation = operations.get(identifier)
    known_non_operation = (
      (raw_category in {"allocation", "deallocation"} and
       identifier in static_allocations) or
      raw_category in {"materialized_write", "materialized_read"} or
      identifier in fusion_profile_ids)
    if operation is None and not known_non_operation:
      unattributed += 1
      continue
    known_event_time_ns += attributed_ns
    category_totals[category] += attributed_ns
    if operation is None:
      continue
    joined.append({
      "id": identifier,
      "operation": operation.get("id"),
      "kind": operation.get("kind"),
      "source_layer": operation.get("source_layer"),
      "source_name": operation.get("source_name"),
      "category": category,
      "calls": calls,
      "inclusive_ns": inclusive,
      "exclusive_ns": exclusive,
      "bytes": bytes_value,
      "bytes_known": bytes_known,
      "attributed_ns": attributed_ns,
    })
  return JoinResult(
    joined, allocation_events, operation_events, worker_operation_events,
    worker_joined, worker_unattributed, parallel_coverage, movement_totals,
    unattributed, unknown_time_ns, measured_event_time_ns,
    known_event_time_ns, category_totals, observed_allocations,
    allocation_bytes_unknown)


def fusion_runtime_join(
    fusion_records: list[Any],
    operation_events: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  fusion_site_runtime: list[dict[str, Any]] = []
  for fusion in fusion_records:
    if not isinstance(fusion, dict):
      raise AttributionError("plan.fusions contains a non-object")
    if fusion.get("fusion_status") != "selected":
      continue
    profile_ids = []
    for field in ("profile_id", "tail_profile_id"):
      identifier = fusion.get(field)
      if identifier is None:
        continue
      if (isinstance(identifier, bool) or not isinstance(identifier, int) or
          identifier < 0 or identifier > (1 << 64) - 1):
        raise AttributionError(f"fusion {field} must be an unsigned 64-bit integer")
      profile_ids.append(identifier)
    matches = [operation_events.get(identifier) for identifier in profile_ids]
    matched = [event for event in matches if event is not None]
    status = (
      "missing" if not matched else
      "joined" if len(matched) == len(profile_ids) and profile_ids else
      "partial")
    call_count = sum(event["calls"] for event in matched)
    inclusive_ns = sum(event["inclusive_ns"] for event in matched)
    exclusive_ns = (
      sum(event["exclusive_ns"] for event in matched)
      if matched and len(matched) == len(profile_ids) and
      all(event["exclusive_ns"] is not None for event in matched) else None)
    fusion_site_runtime.append({
      "id": fusion.get("id"),
      "function": fusion.get("function"),
      "fusion_kind": fusion.get("fusion_kind"),
      "profile_ids": profile_ids,
      "event_join_status": status,
      "call_count": call_count if matched else None,
      "runtime_consumed": call_count > 0 if matched else None,
      "inclusive_ns": inclusive_ns if matched else None,
      "exclusive_ns": exclusive_ns,
    })
  fusion_site_runtime.sort(
    key=lambda row: (row["inclusive_ns"] or 0, row["id"] or ""), reverse=True)
  fusion_site_summary = {
    "selected_count": len(fusion_site_runtime),
    "joined_count": sum(
      row["event_join_status"] == "joined" for row in fusion_site_runtime),
    "partial_count": sum(
      row["event_join_status"] == "partial" for row in fusion_site_runtime),
    "missing_count": sum(
      row["event_join_status"] == "missing" for row in fusion_site_runtime),
    "complete": all(
      row["event_join_status"] == "joined" for row in fusion_site_runtime),
  }
  return fusion_site_runtime, fusion_site_summary


def packed_kernel_join(
    plan: dict[str, Any], operations: dict[int, dict[str, Any]],
    operation_events: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  packed_kernel_runtime: list[dict[str, Any]] = []
  for identifier, operation in operations.items():
    contract = operation.get("kernel_contract")
    if not isinstance(contract, dict) or contract.get("packing") not in {
        "prepacked_B", "packed_A_B"}:
      continue
    event = operation_events.get(identifier)
    calls = event["calls"] if event is not None else None
    packed_kernel_runtime.append({
      "id": operation.get("id"),
      "profile_id": identifier,
      "kernel": contract.get("kernel"),
      "packing": contract.get("packing"),
      "pack_runtime": contract.get("pack_runtime"),
      "event_join_status": "joined" if event is not None else "missing",
      "runtime_consumed": calls > 0 if calls is not None else None,
      "call_count": calls,
      "inclusive_ns": event["inclusive_ns"] if event is not None else None,
      "exclusive_ns": event["exclusive_ns"] if event is not None else None,
    })
  packed_kernel_runtime.sort(key=lambda row: (row["id"], row["profile_id"]))
  packed_selected_count = len(packed_kernel_runtime)
  packed_joined_count = sum(
    row["event_join_status"] == "joined" for row in packed_kernel_runtime)
  packed_consumed_count = sum(
    row["runtime_consumed"] is True for row in packed_kernel_runtime)
  packed_call_counts = [
    row["call_count"] for row in packed_kernel_runtime
    if isinstance(row["call_count"], int)
  ]
  packed_kernel_summary = {
    "selected_count": packed_selected_count,
    "joined_count": packed_joined_count,
    "consumed_count": packed_consumed_count,
    "runtime_consumed": (
      None if packed_selected_count == 0 else
      packed_consumed_count == packed_selected_count
      if packed_joined_count == packed_selected_count else None),
    "call_count": (
      sum(packed_call_counts)
      if len(packed_call_counts) == packed_selected_count else None),
    "packed_buffer_bytes": plan.get("summary", {}).get("packed_buffer_bytes"),
  }
  return packed_kernel_runtime, packed_kernel_summary
