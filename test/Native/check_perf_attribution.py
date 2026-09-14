#!/usr/bin/env python3
"""Contract tests for the strict plan/profile/perf attribution join."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


SCRIPT = pathlib.Path(__file__).parents[2] / "tools" / "perf_attribution_report.py"


def main() -> int:
  with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    plan = root / "model.plan.json"
    profile = root / "model.profile.json"
    perf = root / "perf.ndjson"
    plan.write_text(json.dumps({
      "schema_version": 1,
      "plan_revision": "static-v1",
      "kind": "ncnn.model_execution_plan",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": {"triple": "x86_64-pc-linux-gnu", "threads": 2},
      "buffers": [{"profile_id": 1234}],
      "operations": [{
        "id": "model/linalg.matmul#0",
        "kind": "linalg.matmul",
        "profile_id": 17610563127416308805,
        "source_layer": 3,
        "source_name": "projection",
      }],
      "summary": {"peak_workspace_bytes": None},
      "diagnostics": {"unknown_fields": []},
    }))
    profile.write_text(json.dumps({
      "schema_version": 1,
      "plan_revision": "static-v1",
      "kind": "ncnn.model_execution_profile",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "target": "x86_64-pc-linux-gnu",
      "threads": 2,
      "mode": "prepared",
      "instrumentation": {"coverage": "explicit-callbacks"},
      "summary": {"peak_live_proven": True},
      "events": [{
        "id": 17610563127416308805,
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
    if report["top_costs"][0]["source_name"] != "projection":
      raise RuntimeError("provenance was not joined")
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
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
