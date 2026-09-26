"""Schema, identity, and field validation for the attribution join."""

from __future__ import annotations

from typing import Any


class AttributionError(ValueError):
  pass


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
  if profile.get("schema_version") not in {1, 2, 3} or profile.get("kind") != \
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
  if profile.get("schema_version") in {2, 3}:
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
  if profile.get("schema_version") == 3:
    profile_input_hash = require_string(
      profile.get("input_hash"), "profile.input_hash")
    perf_input_hash = require_string(perf.get("input_hash"), "perf.input_hash")
    if profile_input_hash != perf_input_hash:
      raise AttributionError("input hash mismatch between profile and perf")
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


def category_name(category: Any) -> str:
  if not isinstance(category, str) or not category:
    return "unknown"
  if category == "worker_operation":
    return "worker_cpu"
  if category in {"allocation", "deallocation", "copy", "parallel",
                  "transpose", "pack", "unpack", "materialized_write",
                  "materialized_read"}:
    return category
  return "kernel" if category == "operation" else category
