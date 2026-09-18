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
  if profile.get("schema_version") != 1 or profile.get("kind") != \
      "ncnn.model_execution_profile":
    raise AttributionError("profile has unsupported schema or kind")
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


def category_name(category: Any) -> str:
  if not isinstance(category, str) or not category:
    return "unknown"
  if category in {"allocation", "deallocation", "copy", "parallel"}:
    return category
  return "kernel" if category == "operation" else category


def build_report(plan: dict[str, Any], profile: dict[str, Any],
                 perf: dict[str, Any], mode: str) -> dict[str, Any]:
  operations = operation_index(plan)
  static_allocations = allocation_index(plan)
  events = profile.get("events", [])
  if not isinstance(events, list):
    raise AttributionError("profile.events must be an array")
  profile_summary = profile.get("summary")
  if not isinstance(profile_summary, dict):
    raise AttributionError("profile.summary must be an object")
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
  unattributed = 0
  unknown_time_ns = 0
  measured_event_time_ns = 0
  known_event_time_ns = 0
  category_totals: dict[str, int] = defaultdict(int)
  observed_allocations: set[int] = set()
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
                            "copy", "parallel"}:
      raise AttributionError("profile event category is unsupported")
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
  for row in joined[:3]:
    item = dict(row)
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
  runtime_peak_live_proven = bool(
    profile_summary.get("peak_live_proven", False) and not missing_allocations)
  return {
    "schema_version": 1,
    "kind": "ncnn.model_performance_attribution",
    "model": plan["model"],
    "plan_revision": plan["plan_revision"],
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
      "peak_live_proven": runtime_peak_live_proven,
      "complete": not incomplete_reasons,
      "incomplete_reasons": incomplete_reasons,
    },
    "top_costs": top,
  }


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
    profile = read_json(arguments.profile)
    perf = read_perf(arguments.perf, require_string(plan.get("model"), "plan.model"),
                      arguments.mode)
    validate_identity(plan, profile, perf, arguments.mode)
    report = build_report(plan, profile, perf, arguments.mode)
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
