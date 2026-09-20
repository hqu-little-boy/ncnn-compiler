#!/usr/bin/env python3
"""Join a static execution plan, a diagnostic profile, and perf NDJSON.

The join is deliberately strict: model, plan revision, target triple, thread
count, and measurement mode must agree.  Missing runtime values remain null in
the report and are never converted to zero.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class AttributionError(ValueError):
  pass


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


def require_string(value: Any, field: str) -> str:
  if not isinstance(value, str) or not value:
    raise AttributionError(f"{field} must be a non-empty string")
  return value


def require_nonnegative_integer(value: Any, field: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise AttributionError(f"{field} must be a non-negative integer")
  return value


def validate_identity(plan: dict[str, Any], profile: dict[str, Any],
                      perf: dict[str, Any], mode: str) -> None:
  if plan.get("schema_version") != 1 or plan.get("kind") != \
      "ncnn.model_execution_plan":
    raise AttributionError("plan has unsupported schema or kind")
  if profile.get("schema_version") not in {1, 2} or profile.get("kind") != \
      "ncnn.model_execution_profile":
    raise AttributionError("profile has unsupported schema or kind")
  plan_attribution = plan.get("attribution_revision")
  profile_attribution = profile.get("attribution_revision")
  if plan_attribution is not None:
    require_string(plan_attribution, "plan.attribution_revision")
    if profile_attribution != plan_attribution:
      raise AttributionError(
        "attribution revision mismatch between plan and profile")
  elif profile_attribution is not None:
    require_string(profile_attribution, "profile.attribution_revision")
  if profile.get("schema_version") == 2:
    instrumentation = profile.get("instrumentation")
    if not isinstance(instrumentation, dict) or \
        instrumentation.get("aggregation") != "per-invocation":
      raise AttributionError("schema-2 profile must be per-invocation")
    if not isinstance(profile.get("complete"), bool):
      raise AttributionError("schema-2 profile complete must be boolean")
    require_nonnegative_integer(profile.get("invocation_id"),
                               "schema-2 profile invocation_id")
  model = require_string(plan.get("model"), "plan.model")
  if profile.get("model") != model or perf.get("model") != model:
    raise AttributionError("model identity mismatch across plan/profile/perf")
  revision = require_string(plan.get("plan_revision"), "plan.plan_revision")
  if profile.get("plan_revision") != revision:
    raise AttributionError("plan revision mismatch between plan and profile")
  perf_revision = require_string(perf.get("plan_revision"),
                                "perf.plan_revision")
  if perf_revision != revision:
    raise AttributionError("plan revision mismatch between plan and perf")
  target = plan.get("target")
  if not isinstance(target, dict):
    raise AttributionError("plan.target must be an object")
  triple = require_string(target.get("triple"), "plan.target.triple")
  plan_threads = require_nonnegative_integer(target.get("threads"),
                                             "plan.target.threads")
  plan_hash = require_string(plan.get("plan_hash"), "plan.plan_hash")
  plan_build_identity = require_string(
    plan.get("build_identity"), "plan.build_identity")
  profile_target = require_string(profile.get("target"), "profile.target")
  if profile_target != triple:
    raise AttributionError("target triple mismatch between plan and profile")
  profile_threads = require_nonnegative_integer(profile.get("threads"),
                                                "profile.threads")
  if plan_threads != 0 and profile_threads != plan_threads:
    raise AttributionError("thread count mismatch between plan and profile")
  profile_hash = require_string(profile.get("plan_hash"), "profile.plan_hash")
  profile_build_identity = require_string(
    profile.get("build_identity"), "profile.build_identity")
  if profile_hash != plan_hash:
    raise AttributionError("plan hash mismatch")
  if profile_build_identity != plan_build_identity:
    raise AttributionError("build identity mismatch between plan and profile")
  perf_target = require_string(perf.get("target"), "perf.target")
  if perf_target != triple:
    raise AttributionError("target triple mismatch between plan and perf")
  perf_threads = require_nonnegative_integer(perf.get("threads"),
                                             "perf.threads")
  if plan_threads != 0 and perf_threads != plan_threads:
    raise AttributionError("thread count mismatch between plan and perf")
  if profile_threads != perf_threads:
    raise AttributionError("thread count mismatch between profile and perf")
  perf_hash = require_string(perf.get("plan_hash"), "perf.plan_hash")
  perf_build_identity = require_string(
    perf.get("build_identity"), "perf.build_identity")
  if perf_hash != plan_hash:
    raise AttributionError("plan hash mismatch between plan and perf")
  if perf_build_identity != plan_build_identity:
    raise AttributionError("build identity mismatch between plan and perf")
  profile_mode = profile.get("mode")
  if profile_mode != mode:
    raise AttributionError("measurement mode mismatch between requested mode and profile")
  perf_mode = perf.get("mode", "end_to_end")
  if perf_mode != mode:
    raise AttributionError("measurement mode mismatch between profile and perf")


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


def category_name(category: Any) -> str:
  if not isinstance(category, str) or not category:
    return "unknown"
  if category in {"allocation", "deallocation", "copy", "parallel",
                  "transpose", "pack", "unpack"}:
    return category
  return "kernel" if category == "operation" else category


def build_report(plan: dict[str, Any], profile: dict[str, Any],
                 perf: dict[str, Any], mode: str) -> dict[str, Any]:
  operations = operation_index(plan)
  static_allocations = allocation_index(plan)
  allocation_objects = allocation_metadata(plan)
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
  joined: list[dict[str, Any]] = []
  allocation_events: dict[int, dict[str, Any]] = {}
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
                            "unpack"}:
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
    known_non_operation = raw_category in {"allocation", "deallocation"} and \
      identifier in static_allocations
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
  missing_allocations = sorted(static_allocations - observed_allocations)
  incomplete_reasons: list[str] = []
  if profile.get("schema_version") == 2 and profile.get("complete") is False:
    unknown = list(dict.fromkeys([*unknown, "runtime_profile_incomplete"]))
    incomplete_reasons.append("runtime_profile_incomplete")
  if unattributed:
    unknown = list(dict.fromkeys([*unknown, "runtime_unattributed_events"]))
    incomplete_reasons.append("runtime_unattributed_events")
  if unknown_time_ns:
    unknown = list(dict.fromkeys([*unknown, "runtime_exclusive_time_unknown"]))
    incomplete_reasons.append("runtime_exclusive_time_unknown")
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
  low_precision = plan.get("low_precision")
  if low_precision is not None:
    if not isinstance(low_precision, dict):
      raise AttributionError("plan.low_precision must be an object")
    low_precision_operations = low_precision.get("operations")
    if not isinstance(low_precision_operations, list) or any(
        not isinstance(operation, dict) for operation in low_precision_operations):
      raise AttributionError("plan.low_precision.operations must be an array of objects")
  if any(not value["bytes_known"] for value in movement_totals.values()
         if value["count"]):
    unknown = list(dict.fromkeys([*unknown, "runtime_movement_bytes_unknown"]))
    incomplete_reasons.append("runtime_movement_bytes_unknown")
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
  workspace_coverage = {
    "static_count": workspace_total,
    "joined_count": workspace_joined,
    "complete": workspace_joined == workspace_total,
  }
  if not workspace_coverage["complete"] and workspace_total:
    unknown = list(dict.fromkeys([*unknown, "runtime_workspace_join_incomplete"]))
    incomplete_reasons.append("runtime_workspace_join_incomplete")
  runtime_peak_live_proven = bool(
    profile_summary.get("peak_live_proven", False) and not missing_allocations)
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
    "top_level_time_ns": [],
    "peak_live_bytes": [],
  }
  for report in reports:
    summary = report["runtime"]["summary"]
    if summary.get("runtime_transpose_write_bytes_known") and \
        isinstance(summary.get("runtime_transpose_write_bytes"), int):
      summary_values["runtime_transpose_write_bytes"].append(
        summary["runtime_transpose_write_bytes"])
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
  aggregate_summary = dict(runtime["summary"])
  summary_stats = {}
  for key in (
      "allocation_count", "deallocation_count", "copy_count",
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
        if profile.get("schema_version") != 2:
          raise AttributionError(
            "profile NDJSON rows must use schema-2 per-invocation records")
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
