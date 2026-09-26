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
        "kernel_contract": {
          "kernel": "f32_packed_mxn_fma",
          "packing": "prepacked_B",
          "pack_runtime": "compile_time_B",
        },
      }],
      "fusions": [{
        "id": "fusion/model/matmul_epilogue#0",
        "function": "model",
        "fusion_status": "selected",
        "fusion_kind": "matmul_epilogue",
        "profile_id": 6789,
        "tail_profile_id": 6790,
      }],
      "summary": {"peak_workspace_bytes": None, "packed_buffer_bytes": 64},
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
      "summary": {
        "peak_live_proven": True,
        "peak_live_bytes": 64,
        "materialized_write_count": 1,
        "runtime_materialized_write_bytes": 64,
        "materialized_write_bytes_known": True,
        "materialized_read_count": 1,
        "runtime_materialized_read_bytes": 64,
        "materialized_read_bytes_known": True,
        "expected_materialized_read_bytes": 64,
        "expected_materialized_read_bytes_known": True,
        "materialized_read_complete": True,
      },
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
        "id": 1234,
        "category": "materialized_write",
        "calls": 1,
        "inclusive_ns": 0,
        "exclusive_ns": 0,
        "bytes": 64,
        "bytes_known": True,
      }, {
        "id": 1234,
        "category": "materialized_read",
        "calls": 1,
        "inclusive_ns": 0,
        "exclusive_ns": 0,
        "bytes": 64,
        "bytes_known": True,
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
    packed_runtime = report["runtime"]["packed_kernel_summary"]
    if packed_runtime["runtime_consumed"] is not True or \
        packed_runtime["call_count"] != 4 or \
        packed_runtime["packed_buffer_bytes"] != 64:
      raise RuntimeError("packed kernel runtime join was not preserved")
    if report["runtime"]["packed_kernels"][0]["event_join_status"] != "joined":
      raise RuntimeError("packed kernel event was not joined")
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
    if report["runtime"]["summary"]["runtime_materialized_write_bytes"] != 64 or \
        report["runtime"]["summary"]["runtime_materialized_read_bytes"] != 64 or \
        report["runtime"]["summary"]["materialized_read_complete"] is not True:
      raise RuntimeError("runtime materialized byte metrics were not preserved")
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

    incomplete_materialized_profile = root / "incomplete-materialized.json"
    incomplete_materialized = json.loads(profile.read_text())
    incomplete_materialized["summary"].update({
      "expected_materialized_read_bytes": 128,
      "materialized_read_complete": False,
    })
    incomplete_materialized_profile.write_text(json.dumps(incomplete_materialized))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(incomplete_materialized_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    incomplete_materialized_report = json.loads(result.stdout)
    if "runtime_materialized_read_incomplete" not in \
        incomplete_materialized_report["runtime"]["incomplete_reasons"]:
      raise RuntimeError("incomplete materialized reads were not reported")

    no_materialized_profile = root / "no-materialized.json"
    no_materialized = json.loads(profile.read_text())
    no_materialized["summary"].update({
      "materialized_write_count": 0,
      "runtime_materialized_write_bytes": None,
      "materialized_write_bytes_known": False,
      "materialized_read_count": 0,
      "runtime_materialized_read_bytes": None,
      "materialized_read_bytes_known": False,
      "expected_materialized_read_bytes": None,
      "expected_materialized_read_bytes_known": False,
      "materialized_read_complete": None,
    })
    no_materialized["events"] = [
      event for event in no_materialized["events"]
      if not event["category"].startswith("materialized_")
    ]
    no_materialized_profile.write_text(json.dumps(no_materialized))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(no_materialized_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    no_materialized_report = json.loads(result.stdout)
    if no_materialized_report["runtime"]["complete"] or \
        "runtime_materialized_bytes_not_observed" not in \
        no_materialized_report["runtime"]["incomplete_reasons"]:
      raise RuntimeError("missing materialized evidence was reported as complete")

    site_profile = root / "materialized-sites.json"
    site_profile_data = json.loads(profile.read_text())
    site_profile_data["events"] = [
      {**event, "id": 987654} if event["category"].startswith("materialized_")
      else event
      for event in site_profile_data["events"]
      if event["id"] != 99
    ]
    site_profile.write_text(json.dumps(site_profile_data))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(site_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    site_report = json.loads(result.stdout)
    if site_report["runtime"]["unattributed_event_count"] != 0:
      raise RuntimeError("materialized site callbacks were treated as unknown ops")

    fusion_profile = root / "fusion-sites.json"
    fusion_profile_data = json.loads(site_profile.read_text())
    fusion_profile_data["events"].extend([{
      "id": 6789,
      "category": "operation",
      "calls": 1,
      "inclusive_ns": 30,
      "exclusive_ns": 25,
    }, {
      "id": 6790,
      "category": "operation",
      "calls": 1,
      "inclusive_ns": 5,
      "exclusive_ns": 4,
    }])
    fusion_profile.write_text(json.dumps(fusion_profile_data))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(fusion_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    fusion_report = json.loads(result.stdout)
    fusion_runtime = fusion_report["runtime"]
    if fusion_runtime["fusion_site_summary"] != {
        "selected_count": 1,
        "joined_count": 1,
        "partial_count": 0,
        "missing_count": 0,
        "complete": True,
    }:
      raise RuntimeError("fusion site runtime coverage was not joined")
    top_fusion = fusion_runtime["top_fusion_sites"][0]
    if top_fusion["profile_ids"] != [6789, 6790] or \
        top_fusion["inclusive_ns"] != 35 or \
        top_fusion["exclusive_ns"] != 29 or \
        top_fusion["event_join_status"] != "joined":
      raise RuntimeError("fusion tail runtime events were not aggregated")
    if fusion_runtime["unattributed_event_count"] != 0:
      raise RuntimeError("fusion callbacks without operation rows were unattributed")

    missing_fusion_profile = root / "missing-fusion-sites.json"
    missing_fusion_data = json.loads(fusion_profile.read_text())
    missing_fusion_data["events"] = [
      event for event in missing_fusion_data["events"] if event["id"] != 6790
    ]
    missing_fusion_profile.write_text(json.dumps(missing_fusion_data))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(missing_fusion_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    missing_fusion_report = json.loads(result.stdout)
    if missing_fusion_report["runtime"]["fusion_sites"][0]["event_join_status"] != \
        "partial" or \
        "runtime_fusion_site_profile_missing" not in \
        missing_fusion_report["runtime"]["incomplete_reasons"]:
      raise RuntimeError("missing fusion tail profile was reported as complete")

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
        }, {
          "id": 6789,
          "category": "operation",
          "calls": 1,
          "inclusive_ns": 30 * invocation_id,
          "exclusive_ns": 25 * invocation_id,
        }, {
          "id": 6790,
          "category": "operation",
          "calls": 1,
          "inclusive_ns": 5 * invocation_id,
          "exclusive_ns": 4 * invocation_id,
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
    v2_fusion_runtime = v2_runtime["fusion_site_summary"]
    if v2_fusion_runtime["selected_count"] != 1 or \
        v2_fusion_runtime["joined_count"] != 1 or \
        not v2_fusion_runtime["complete"] or \
        v2_runtime["top_fusion_sites"][0]["inclusive_ns"] != 52.5:
      raise RuntimeError("schema-2 fusion-site events were not aggregated")
    parallel_profile = root / "parallel-profile.json"
    parallel_value = json.loads(profile.read_text())
    parallel_value["events"] = [
      dict(event, category="parallel")
      if event.get("category") == "operation" and
      event.get("id") == 16011668676398822909 else event
      for event in parallel_value["events"]
    ]
    parallel_profile.write_text(json.dumps(parallel_value))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(perf), "--plan", str(plan),
      "--profile", str(parallel_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    parallel_report = json.loads(result.stdout)
    parallel_runtime = parallel_report["runtime"]["packed_kernel_summary"]
    if parallel_runtime["runtime_consumed"] is not True or \
        parallel_runtime["call_count"] != 4:
      raise RuntimeError("parallel packed kernel event was not joined")

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

    worker_plan = root / "worker.plan.json"
    worker_profile = root / "worker.profile.json"
    worker_perf = root / "worker-perf.ndjson"
    worker_perf_value = json.loads(perf.read_text().splitlines()[0])
    worker_perf_value["input_hash"] = "input-abc123"
    worker_perf.write_text(json.dumps(worker_perf_value) + "\n")
    worker_plan_value = json.loads(plan.read_text())
    worker_plan_value["attribution_revision"] = "attribution-v4"
    worker_plan.write_text(json.dumps(worker_plan_value))
    worker_profile.write_text(json.dumps({
      "schema_version": 3,
      "kind": "ncnn.model_execution_profile",
      "plan_revision": "static-v1",
      "attribution_revision": "attribution-v4",
      "model": "fixture",
      "plan_hash": "plan-123",
      "build_identity": "plan-123",
      "input_hash": "input-abc123",
      "target": "x86_64-pc-linux-gnu",
      "threads": 2,
      "mode": "prepared",
      "invocation_id": 1,
      "complete": True,
      "instrumentation": {
        "coverage": "parallel-worker-coarse-sites",
        "aggregation": "per-invocation",
        "invocation_count": 1,
        "record_overflow": False,
        "allocation_table_overflow": False,
      },
      "summary": {
        "event_mismatch_count": 0,
        "top_level_time_ns": 1000,
        "top_level_time_known": True,
        "peak_live_proven": False,
        "peak_live_bytes": None,
        "worker_sampling": {
          "duty": 1,
          "window_ns": 2000000,
          "basis": "time_window_duty_cycle",
          "interval_overflow": False,
        },
        "worker_wall_attribution": {
          "basis": "equal_split_across_concurrent_worker_spans",
          "additive": True,
          "time_domain": "wall",
        },
      },
      "events": [{
        "id": 16011668676398822909,
        "category": "parallel",
        "time_domain": "wall",
        "calls": 1,
        "inclusive_ns": 1000,
        "exclusive_ns": 300,
        "worker_covered_wall_ns": 700,
        "worker_covered_wall_sampled_ns": 700,
        "sampled_window_wall_ns": 1000,
        "region_wall_ns": 1000,
        "region_worker_spans": 4,
        "sample_scale": 1,
        "wall_projection_factor": 1.0,
        "wall_coverage_share": 0.7,
        "wall_exclusive_estimated_ns": 300,
        "wall_exclusive_estimated_known": True,
        "exclusive_semantics": "region_wall_minus_worker_span_union",
        "bytes": None,
        "bytes_known": False,
      }, {
        "id": 16011668676398822909,
        "category": "worker_operation",
        "time_domain": "worker_cpu",
        "calls": 4,
        "calls_estimated": 4,
        "inclusive_ns": 1000,
        "exclusive_ns": 800,
        "worker_wall_union_ns": 300,
        "worker_wall_union_known": True,
        "wall_attributed_ns": 700,
        "wall_attributed_estimated_ns": 700,
        "wall_attributed_share_of_sampled_window": 0.7,
        "wall_attributed_known": True,
        "bytes": None,
        "bytes_known": False,
      }],
    }))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(worker_perf), "--plan",
      str(worker_plan), "--profile", str(worker_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(result.stderr)
    worker_report = json.loads(result.stdout)
    worker_attribution = worker_report["runtime"]["worker_attribution"]
    if worker_attribution["complete_for_observed_events"] is not True:
      raise RuntimeError("complete worker-operation join was not recognized")
    if worker_attribution["exclusive_cpu_total_ns"] != 800:
      raise RuntimeError("worker CPU exclusive time was not retained separately")
    worker_cost = worker_attribution["top_operations_by_wall_union"][0]
    if worker_cost["wall_union_share_of_top_level"] != 0.3:
      raise RuntimeError("worker wall-union share used the wrong denominator")
    if worker_report["runtime"]["unknown_time_ns"] != 0:
      raise RuntimeError("worker CPU time polluted wall-exclusive uncertainty")
    partition = worker_report["runtime"]["wall_partition"]
    if partition["worker_ops_wall_attributed_ns"] != 700:
      raise RuntimeError("additive worker wall attribution was not reported")
    if partition["parallel_wall_gap_wall_ns"] != 300:
      raise RuntimeError("parallel-region wall gap was not reported")
    if partition["worker_ops_wall_unjoined_ns"] != 0:
      raise RuntimeError("joined worker wall attribution was not exhaustive")
    if abs(partition["parallel_wall_coverage_share"] - 0.7) > 1e-9 or \
        abs(partition["parallel_wall_gap_share"] - 0.3) > 1e-9:
      raise RuntimeError("wall coverage share was not window-normalized")
    if partition["sampled_window_wall_ns"] != 1000:
      raise RuntimeError("sampled-window normalization was not reported")
    if partition["accounted_wall_ns"] != 1000 or \
        partition["unaccounted_wall_ns"] != 0:
      raise RuntimeError("wall partition did not account for top-level wall")
    if worker_cost["wall_attributed_ns"] != 700 or \
        worker_cost["wall_attributed_share_of_top_level"] != 0.7:
      raise RuntimeError("worker additive wall share used the wrong denominator")
    if worker_cost["calls_estimated"] != 4:
      raise RuntimeError("sampled call estimate was not preserved")
    mismatched_worker_profile = json.loads(worker_profile.read_text())
    mismatched_worker_profile["input_hash"] = "different-input"
    worker_profile.write_text(json.dumps(mismatched_worker_profile))
    result = subprocess.run([
      sys.executable, str(SCRIPT), "--perf", str(worker_perf), "--plan",
      str(worker_plan), "--profile", str(worker_profile), "--mode", "prepared",
    ], capture_output=True, text=True)
    if result.returncode == 0 or "input hash mismatch" not in result.stderr:
      raise RuntimeError("profile/perf input mismatch was accepted")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
