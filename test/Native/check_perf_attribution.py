#!/usr/bin/env python3
"""Contract tests for the strict plan/profile/perf attribution join."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


SCRIPT = pathlib.Path(__file__).parents[2] / "tools" / "perf_attribution_report.py"


def fnv1a64(value: str) -> int:
  result = 14695981039346656037
  for character in value.encode("utf-8"):
    result ^= character
    result = (result * 1099511628211) & ((1 << 64) - 1)
  return result


def main() -> int:
  if fnv1a64("model/linalg.matmul#0") != 16011668676398822909:
    raise RuntimeError("fixture FNV-1a stable ID is incorrect")
  with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    plan = root / "model.plan.json"
    profile = root / "model.profile.json"
    perf = root / "perf.ndjson"
    low_precision = {
      "revision": "int8-target-v1",
      "requested_policy": "auto",
      "resolved_capability": "avx-vnni",
      "depthwise_enabled": True,
      "cast_chain_enabled": False,
      "operations": [{
        "id": "model/scf.forall#0",
        "kernel": "int8_vnni_row_dot",
        "required_isa": "avx-vnni",
        "signedness": "signed8*signed8",
        "signedness_correction": "xor_0x80_subtract_128_times_rhs_sum",
        "accumulator": "modulo32",
        "static_tails": {"output_policy": "none", "reduction_elements": 3},
        "fallback_reason": None,
      }],
    }
    plan.write_text(json.dumps({
      "low_precision": low_precision,
      "schema_version": 1,
      "plan_revision": "static-v1",
      "attribution_revision": "attribution-v1",
      "kind": "ncnn.model_execution_plan",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": {"triple": "x86_64-pc-linux-gnu", "threads": 2},
      "buffers": [{"profile_id": 1234}],
      "operations": [{
        "id": "model/linalg.matmul#0",
        "kind": "linalg.matmul",
        "profile_id": 16011668676398822909,
        "source_layer": 3,
        "source_name": "projection",
      }],
      "summary": {"peak_workspace_bytes": None},
      "diagnostics": {"unknown_fields": []},
      "conv_depthwise_operations": [{
        "family": "conv",
        "implementation": "gather_free",
        "source_layer": 3,
        "fallback_reason": None,
      }],
    }))
    profile.write_text(json.dumps({
      "schema_version": 1,
      "plan_revision": "static-v1",
      "attribution_revision": "attribution-v1",
      "kind": "ncnn.model_execution_profile",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": "x86_64-pc-linux-gnu",
      "threads": 2,
      "mode": "prepared",
      "instrumentation": {"coverage": "explicit-callbacks"},
      "summary": {"peak_live_proven": True, "peak_live_bytes": 64},
      "events": [{
        "id": 16011668676398822909,
        "category": "operation",
        "calls": 4,
        "inclusive_ns": 100,
        "exclusive_ns": 80,
      }, {
        "id": 1234,
        "category": "allocation",
        "calls": 1,
        "inclusive_ns": 0,
        "exclusive_ns": 0,
      }, {
        "id": 99,
        "category": "operation",
        "calls": 1,
        "inclusive_ns": 900,
        "exclusive_ns": 900,
      }],
    }))
    perf.write_text(json.dumps({
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
    }) + "\n")
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    report = json.loads(result.stdout)
    if report["static"].get("low_precision") != low_precision:
      raise RuntimeError("static low-precision audit was not preserved")
    if "low_precision" in report["runtime"]:
      raise RuntimeError("static low-precision audit advertised runtime data")
    if report["top_costs"][0]["source_name"] != "projection":
      raise RuntimeError("provenance was not joined")
    conv_breakdown = report["static"]["conv_depthwise"]["conv"]
    if conv_breakdown["implementations"] != {"gather_free": 1}:
      raise RuntimeError("Conv implementation breakdown was not joined")
    if conv_breakdown["source_layers"] != [3]:
      raise RuntimeError("Conv source-layer breakdown was not joined")
    if report["top_costs"][0]["time_fraction"] != 80 / 980:
      raise RuntimeError("top-cost fraction omitted unattributed time")
    if report["runtime"]["unattributed_event_count"] != 1:
      raise RuntimeError("unknown event was not reported")
    if report["runtime"]["unattributed_time_ns"] != 900:
      raise RuntimeError("unknown event time was not reported")
    if report["runtime"]["category_time_ns"]["kernel"] != 80:
      raise RuntimeError("nested inclusive time was double-counted")
    if not report["runtime"]["allocation_coverage"]["complete"]:
      raise RuntimeError("complete allocation coverage was rejected")
    if report["runtime"]["complete"]:
      raise RuntimeError("unattributed event was reported as complete")
    if not report["runtime"]["peak_live_proven"]:
      raise RuntimeError("complete allocation coverage lost peak proof")
    if report["top_allocations"][0]["id"] != 1234:
      raise RuntimeError("top allocation report was not generated")
    if report["runtime"]["copy_layout"]["copy"]["status"] != "not_observed":
      raise RuntimeError("copy layout status was not explicit")

    v2_profile = root / "profile-v2.ndjson"
    v2_base = json.loads(profile.read_text())
    v2_rows = []
    for invocation_id, allocation_bytes, copy_bytes, complete in [
        (1, 64, 32, True), (2, 128, None, False)]:
      row = dict(v2_base)
      row.update({
        "schema_version": 2,
        "instrumentation": {
          "coverage": "explicit-callbacks",
          "aggregation": "per-invocation",
        },
        "invocation_id": invocation_id,
        "complete": complete,
        "summary": {
          "peak_live_proven": True,
          "peak_live_bytes": allocation_bytes,
          "top_level_time_ns": 100 * invocation_id,
          "top_level_time_known": True,
        },
        "events": [{
          "id": 16011668676398822909,
          "category": "operation",
          "calls": 1,
          "inclusive_ns": 100 * invocation_id,
          "exclusive_ns": 100 * invocation_id,
        }, {
          "id": 1234,
          "category": "allocation",
          "calls": 1,
          "inclusive_ns": 0,
          "exclusive_ns": 0,
          "bytes": allocation_bytes,
          "bytes_known": True,
        }, {
          "id": 16011668676398822909,
          "category": "copy",
          "calls": 1,
          "inclusive_ns": 0,
          "exclusive_ns": 0,
          "bytes": copy_bytes,
          "bytes_known": copy_bytes is not None,
        }],
      })
      v2_rows.append(row)
    v2_profile.write_text("".join(json.dumps(row) + "\n" for row in v2_rows))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(v2_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    v2_report = json.loads(result.stdout)
    v2_runtime = v2_report["runtime"]
    if v2_runtime["invocation_count"] != 2:
      raise RuntimeError("schema-2 invocation count was not aggregated")
    if v2_report["invocation_id"] is not None or \
        v2_report["invocation_ids"] != [1, 2]:
      raise RuntimeError("schema-2 invocation identity was not aggregated")
    if v2_runtime["copy_layout"]["copy"]["status"] != "unknown":
      raise RuntimeError("schema-2 unknown copy bytes were lost")
    if v2_runtime["copy_layout"]["copy"]["bytes"] is not None:
      raise RuntimeError("schema-2 unknown copy bytes became numeric")
    if v2_runtime["complete"]:
      raise RuntimeError("schema-2 incomplete invocation was accepted")
    if "runtime_profile_incomplete" not in v2_runtime["incomplete_reasons"]:
      raise RuntimeError("schema-2 incomplete reason was not reported")
    if v2_report["top_allocations"][0]["runtime_bytes"] != 96:
      raise RuntimeError("schema-2 allocation bytes were not aggregated")

    schema1_ndjson = root / "schema1.ndjson"
    schema1_row = json.loads(profile.read_text())
    schema1_ndjson.write_text(
      json.dumps(schema1_row) + "\n" + json.dumps(schema1_row) + "\n")
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(schema1_ndjson), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "schema-2" not in result.stderr:
      raise RuntimeError("schema-1 NDJSON was accepted as schema-2")

    runtime_default_plan = json.loads(plan.read_text())
    runtime_default_plan["target"]["threads"] = 0
    plan.write_text(json.dumps(runtime_default_plan))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError("runtime-default thread identity was rejected")
    runtime_default_plan["target"]["threads"] = 2
    plan.write_text(json.dumps(runtime_default_plan))

    missing_allocation = json.loads(profile.read_text())
    missing_allocation["events"] = [
      event for event in missing_allocation["events"]
      if event["id"] != 1234
    ]
    profile.write_text(json.dumps(missing_allocation))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    incomplete = json.loads(result.stdout)
    if incomplete["runtime"]["complete"]:
      raise RuntimeError("missing allocation coverage was accepted as complete")
    if "runtime_profile_missing_allocation_events" not in \
        incomplete["runtime"]["incomplete_reasons"]:
      raise RuntimeError("missing allocation reason was not reported")
    if incomplete["runtime"]["peak_live_proven"]:
      raise RuntimeError("missing allocation coverage retained peak proof")

    restored_profile = json.loads(profile.read_text())
    restored_profile["events"].append({
      "id": 1234,
      "category": "allocation",
      "calls": 1,
      "inclusive_ns": 0,
      "exclusive_ns": 0,
    })
    profile.write_text(json.dumps(restored_profile))

    bad_perf = json.loads(perf.read_text())
    bad_perf["target"] = "aarch64-unknown-linux-gnu"
    perf.write_text(json.dumps(bad_perf) + "\n")
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "target triple mismatch" not in result.stderr:
      raise RuntimeError("perf target mismatch was accepted")
    valid_perf = {
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
    perf.write_text(json.dumps(valid_perf) + "\n")
    bad_perf = dict(valid_perf)
    bad_perf["plan_hash"] = "other-plan"
    perf.write_text(json.dumps(bad_perf) + "\n")
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "plan hash mismatch" not in result.stderr:
      raise RuntimeError("perf plan hash mismatch was accepted")
    bad_perf = dict(valid_perf)
    bad_perf["threads"] = True
    perf.write_text(json.dumps(bad_perf) + "\n")
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "non-negative integer" not in result.stderr:
      raise RuntimeError("boolean perf thread count was accepted")
    perf.write_text(json.dumps(valid_perf) + "\n")
    bad_profile = json.loads(profile.read_text())
    del bad_profile["target"]
    profile.write_text(json.dumps(bad_profile))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "profile.target" not in result.stderr:
      raise RuntimeError("missing profile target was accepted")
    profile.write_text(json.dumps(json.loads(profile.read_text()) | {
      "target": "x86_64-pc-linux-gnu",
    }))

    mismatched = json.loads(profile.read_text())
    mismatched["threads"] = 3
    profile.write_text(json.dumps(mismatched))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "thread count mismatch" not in result.stderr:
      raise RuntimeError("identity mismatch was accepted")

    mode_mismatched = json.loads(profile.read_text())
    mode_mismatched["threads"] = 2
    mode_mismatched["mode"] = "end_to_end"
    profile.write_text(json.dumps(mode_mismatched))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "measurement mode mismatch" not in result.stderr:
      raise RuntimeError("profile mode mismatch was accepted")

    invalid = json.loads(profile.read_text())
    invalid["mode"] = "prepared"
    invalid["events"][0]["calls"] = 1.5
    profile.write_text(json.dumps(invalid))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "non-negative integer" not in result.stderr:
      raise RuntimeError("fractional event counter was accepted")

    invalid["events"][0]["calls"] = 1
    invalid["summary"]["event_mismatch_count"] = 1
    profile.write_text(json.dumps(invalid))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "attribution is incomplete" not in result.stderr:
      raise RuntimeError("event mismatch was accepted")

    invalid["summary"]["event_mismatch_count"] = 0
    invalid["summary"]["copy_bytes"] = None
    invalid["summary"]["copy_bytes_known"] = True
    profile.write_text(json.dumps(invalid))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "copy_bytes_known=true" not in result.stderr:
      raise RuntimeError("inconsistent summary bytes were accepted")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
