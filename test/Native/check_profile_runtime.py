#!/usr/bin/env python3
"""Smoke-test the opt-in profile runtime contract."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import tempfile


HARNESS = r'''
#include <stdint.h>
extern void __ncnn_profile_event_begin(int64_t, int64_t);
extern void __ncnn_profile_event_end(int64_t);
extern void __ncnn_profile_alloc(int64_t, int64_t);
extern void __ncnn_profile_dealloc(int64_t);
extern void __ncnn_profile_copy(int64_t, int64_t);
extern void __ncnn_profile_movement(int64_t, int64_t, int64_t);
extern void __ncnn_profile_materialized(int64_t, int64_t, int64_t, int64_t);
extern void __ncnn_profile_flush(void);
int main(void) {
  __ncnn_profile_event_begin(7, 0);
  __ncnn_profile_event_begin(8, 4);
  __ncnn_profile_event_end(8);
  __ncnn_profile_event_end(7);
  __ncnn_profile_alloc(9, 64);
  __ncnn_profile_copy(10, 32);
  __ncnn_profile_materialized(14, 0, 256, 1);
  __ncnn_profile_materialized(14, 1, 256, 0);
  __ncnn_profile_movement(11, 0, 128);
  __ncnn_profile_movement(12, 1, 256);
  __ncnn_profile_movement(13, 2, 512);
  __ncnn_profile_dealloc(9);
  __ncnn_profile_flush();
  __ncnn_profile_flush();
  return 0;
}
'''

V2_HARNESS = r'''
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void __ncnn_profile_event_begin(int64_t, int64_t);
extern void __ncnn_profile_event_end(int64_t);
extern void __ncnn_profile_alloc(int64_t, int64_t);
extern void __ncnn_profile_dealloc(int64_t);
extern void __ncnn_profile_movement(int64_t, int64_t, int64_t);
extern void __ncnn_profile_materialized(int64_t, int64_t, int64_t, int64_t);
extern void __ncnn_profile_flush(void);
static void* fusion_worker(void* unused) {
  const struct timespec delay = {0, 5000000};
  (void)unused;
  __ncnn_profile_event_begin(17, 10);
  nanosleep(&delay, NULL);
  __ncnn_profile_event_end(17);
  return NULL;
}
static void invoke(int64_t bytes) {
  pthread_t worker;
  __ncnn_profile_event_begin(7, 0);
  if (pthread_create(&worker, NULL, fusion_worker, NULL) != 0) abort();
  if (pthread_join(worker, NULL) != 0) abort();
  __ncnn_profile_alloc(9, bytes);
  __ncnn_profile_materialized(15, 0, bytes, 1);
  __ncnn_profile_materialized(15, 1, bytes, 0);
  __ncnn_profile_movement(11, 0, bytes);
  __ncnn_profile_dealloc(9);
  __ncnn_profile_event_end(7);
  __ncnn_profile_flush();
}
static void invoke_missing_read(int64_t bytes) {
  __ncnn_profile_event_begin(7, 0);
  __ncnn_profile_alloc(9, bytes);
  __ncnn_profile_materialized(15, 0, bytes, 1);
  __ncnn_profile_event_end(7);
  __ncnn_profile_flush();
}
static void invoke_without_materialized(int64_t bytes) {
  __ncnn_profile_event_begin(7, 0);
  __ncnn_profile_alloc(9, bytes);
  __ncnn_profile_dealloc(9);
  __ncnn_profile_event_end(7);
  __ncnn_profile_flush();
}
int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "no-materialized") == 0) {
    invoke_without_materialized(32);
    return 0;
  }
  invoke(64);
  invoke(128);
  if (argc > 1) invoke_missing_read(64);
  return 0;
}
'''

CONCURRENT_HARNESS = r'''
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
extern void __ncnn_profile_alloc(int64_t, int64_t);
extern void __ncnn_profile_dealloc(int64_t);
extern void __ncnn_profile_flush(void);
static pthread_barrier_t barrier;
static void* worker(void* unused) {
  (void)unused;
  for (int i = 0; i < 20; ++i) {
    __ncnn_profile_alloc(9, 64);
    pthread_barrier_wait(&barrier);
    __ncnn_profile_dealloc(9);
    pthread_barrier_wait(&barrier);
  }
  return NULL;
}
int main(int argc, char** argv) {
  (void)argv;
  if (argc > 1) {
    __ncnn_profile_alloc(9, 64);
    __ncnn_profile_alloc(9, 128);
    __ncnn_profile_dealloc(9);
    __ncnn_profile_dealloc(9);
  } else {
    pthread_t threads[4];
    if (pthread_barrier_init(&barrier, NULL, 4)) return 1;
    for (int i = 0; i < 4; ++i)
      if (pthread_create(&threads[i], NULL, worker, NULL)) abort();
    for (int i = 0; i < 4; ++i)
      if (pthread_join(threads[i], NULL)) abort();
    pthread_barrier_destroy(&barrier);
    /* A later, non-overlapping instance may have a different size. */
    __ncnn_profile_alloc(9, 32);
    __ncnn_profile_dealloc(9);
  }
  __ncnn_profile_flush();
  return 0;
}
'''

WORKER_HARNESS = r'''
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
extern void __ncnn_profile_event_begin(int64_t, int64_t);
extern void __ncnn_profile_event_end(int64_t);
extern void __ncnn_profile_worker_event_begin(int64_t);
extern void __ncnn_profile_worker_event_end(int64_t);
extern void __ncnn_profile_flush(void);
static pthread_barrier_t start_barrier;
static void delay(long nanoseconds) {
  const struct timespec duration = {0, nanoseconds};
  nanosleep(&duration, NULL);
}
static void* worker(void* unused) {
  (void)unused;
  pthread_barrier_wait(&start_barrier);
  __ncnn_profile_worker_event_begin(42);
  delay(20000000);
  __ncnn_profile_worker_event_begin(43);
  delay(2000000);
  __ncnn_profile_worker_event_end(43);
  __ncnn_profile_worker_event_end(42);
  return NULL;
}
int main(void) {
  pthread_t workers[2];
  if (pthread_barrier_init(&start_barrier, NULL, 3) != 0) abort();
  __ncnn_profile_event_begin(7, 0);
  __ncnn_profile_event_begin(8, 4);
  for (int i = 0; i < 2; ++i)
    if (pthread_create(&workers[i], NULL, worker, NULL) != 0) abort();
  pthread_barrier_wait(&start_barrier);
  for (int i = 0; i < 2; ++i)
    if (pthread_join(workers[i], NULL) != 0) abort();
  pthread_barrier_destroy(&start_barrier);
  __ncnn_profile_event_end(8);
  __ncnn_profile_event_end(7);
  __ncnn_profile_flush();
  return 0;
}
'''


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cc", required=True)
  parser.add_argument("--runtime", required=True)
  args = parser.parse_args()
  with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source = root / "harness.c"
    binary = root / "harness"
    profile = root / "profile.json"
    source.write_text(HARNESS, encoding="utf-8")
    subprocess.run([
      args.cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
      '-DNCNN_PROFILE_DEFAULT_PLAN_REVISION="static-v1|int8-target-v1"',
      str(source), args.runtime, "-o", str(binary),
    ], check=True)
    environment = dict(os.environ)
    environment.update({
      "NCNN_PROFILE_PATH": str(profile),
      "NCNN_PROFILE_MODEL": "fixture",
      "NCNN_PROFILE_PLAN_HASH": "plan-123",
      "NCNN_PROFILE_BUILD_IDENTITY": "build-123",
      "NCNN_PROFILE_TARGET": "x86_64-pc-linux-gnu",
      "NCNN_PROFILE_THREADS": "2",
      "NCNN_PROFILE_MODE": "line\bfeed\f",
    })
    subprocess.run([str(binary)], check=True, env=environment)
    document = json.loads(profile.read_text(encoding="utf-8"))
    assert document["kind"] == "ncnn.model_execution_profile"
    assert document["plan_revision"] == "static-v1|int8-target-v1"
    assert document["attribution_revision"] == "attribution-v4"
    assert document["instrumentation"]["aggregation"] == "process-cumulative"
    assert document["instrumentation"]["invocation_count"] == 2
    assert document["mode"] == "line\bfeed\f"
    assert document["plan_hash"] == "plan-123"
    assert document["build_identity"] == "build-123"
    summary = document["summary"]
    assert summary["allocation_count"] == 1
    assert summary["allocation_bytes"] == 64
    assert summary["deallocation_count"] == 1
    assert summary["deallocation_bytes"] == 64
    assert summary["copy_count"] == 1
    assert summary["copy_bytes"] == 32
    assert summary["materialized_write_count"] == 1
    assert summary["runtime_materialized_write_bytes"] == 256
    assert summary["materialized_write_bytes_known"] is True
    assert summary["materialized_read_count"] == 1
    assert summary["runtime_materialized_read_bytes"] == 256
    assert summary["materialized_read_bytes_known"] is True
    assert summary["expected_materialized_read_bytes"] == 256
    assert summary["expected_materialized_read_bytes_known"] is True
    assert summary["materialized_read_complete"] is True
    assert summary["transpose_count"] == 1
    assert summary["runtime_transpose_write_bytes"] == 128
    assert summary["runtime_transpose_write_bytes_known"] is True
    assert summary["pack_count"] == 1
    assert summary["pack_bytes"] == 256
    assert summary["pack_bytes_known"] is True
    assert summary["unpack_count"] == 1
    assert summary["unpack_bytes"] == 512
    assert summary["unpack_bytes_known"] is True
    assert summary["peak_live_bytes"] == 64
    assert summary["peak_live_proven"] is True
    assert summary["top_level_time_ns"] > 0
    assert summary["top_level_time_known"] is True
    assert summary["event_mismatch_count"] == 0
    records = {(event["id"], event["category"]): event
               for event in document["events"]}
    assert records[(9, "allocation")]["calls"] == 1
    assert records[(9, "allocation")]["bytes"] == 64
    assert records[(9, "allocation")]["bytes_known"] is True
    assert records[(10, "copy")]["calls"] == 1
    assert records[(10, "copy")]["bytes"] == 32
    assert records[(14, "materialized_write")]["bytes"] == 256
    assert records[(14, "materialized_read")]["bytes"] == 256
    assert records[(11, "transpose")]["calls"] == 1
    assert records[(11, "transpose")]["bytes"] == 128
    assert records[(12, "pack")]["calls"] == 1
    assert records[(12, "pack")]["bytes"] == 256
    assert records[(13, "unpack")]["calls"] == 1
    assert records[(13, "unpack")]["bytes"] == 512
    assert records[(8, "parallel")]["exclusive_ns"] is None
    source.write_text(CONCURRENT_HARNESS, encoding="utf-8")
    subprocess.run([args.cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-pthread", str(source), args.runtime, "-o", str(binary)],
                   check=True)
    subprocess.run([str(binary)], check=True, env=environment)
    concurrent = json.loads(profile.read_text())["summary"]
    assert concurrent["allocation_count"] == 81
    assert concurrent["deallocation_count"] == 81
    assert concurrent["allocation_bytes"] == 80 * 64 + 32
    assert concurrent["deallocation_bytes"] == 80 * 64 + 32
    assert concurrent["peak_live_bytes"] == 256
    assert concurrent["peak_live_proven"] is True
    subprocess.run([str(binary), "conflicting-sizes"], check=True, env=environment)
    conflicting = json.loads(profile.read_text())["summary"]
    assert conflicting["peak_live_proven"] is False
    assert conflicting["peak_live_bytes"] is None

    v2_source = root / "v2_harness.c"
    v2_binary = root / "v2_harness"
    v2_profile = root / "profile-v2.ndjson"
    v2_source.write_text(V2_HARNESS, encoding="utf-8")
    subprocess.run([
      args.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread",
      str(v2_source), args.runtime, "-o", str(v2_binary),
    ], check=True)
    v2_environment = dict(environment)
    v2_environment.update({
      "NCNN_PROFILE_PATH": str(v2_profile),
      "NCNN_PROFILE_SCHEMA": "2",
    })
    subprocess.run([str(v2_binary)], check=True, env=v2_environment)
    rows = [json.loads(line) for line in v2_profile.read_text().splitlines()
            if line.strip()]
    assert len(rows) == 2
    assert [row["schema_version"] for row in rows] == [2, 2]
    assert [row["invocation_id"] for row in rows] == [1, 2]
    assert all(row["attribution_revision"] == "attribution-v4" for row in rows)
    assert all(row["complete"] is True for row in rows)
    for row in rows:
      events = {(event["id"], event["category"]): event
                for event in row["events"]}
      assert events[(17, "operation")]["calls"] == 1
      assert events[(17, "operation")]["inclusive_ns"] > 0
      assert (row["summary"]["top_level_time_ns"] ==
              events[(7, "operation")]["inclusive_ns"])
    assert [row["summary"]["allocation_bytes"] for row in rows] == [64, 128]
    assert [row["summary"]["runtime_transpose_write_bytes"]
            for row in rows] == [64, 128]
    assert [row["summary"]["runtime_materialized_write_bytes"]
            for row in rows] == [64, 128]
    assert [row["summary"]["runtime_materialized_read_bytes"]
            for row in rows] == [64, 128]
    assert [row["summary"]["expected_materialized_read_bytes"]
            for row in rows] == [64, 128]
    assert all(row["summary"]["materialized_read_complete"] is True
               for row in rows)
    no_materialized_profile = root / "profile-v2-no-materialized.ndjson"
    no_materialized_environment = dict(v2_environment)
    no_materialized_environment["NCNN_PROFILE_PATH"] = str(no_materialized_profile)
    subprocess.run([str(v2_binary), "no-materialized"], check=True,
                   env=no_materialized_environment)
    no_materialized_rows = [
      json.loads(line) for line in no_materialized_profile.read_text().splitlines()
    ]
    assert len(no_materialized_rows) == 1
    no_materialized_summary = no_materialized_rows[0]["summary"]
    assert no_materialized_summary["materialized_write_count"] == 0
    assert no_materialized_summary["runtime_materialized_write_bytes"] is None
    assert no_materialized_summary["materialized_write_bytes_known"] is False
    assert no_materialized_summary["materialized_read_count"] == 0
    assert no_materialized_summary["runtime_materialized_read_bytes"] is None
    assert no_materialized_summary["materialized_read_bytes_known"] is False
    assert no_materialized_summary["expected_materialized_read_bytes"] is None
    assert no_materialized_summary["expected_materialized_read_bytes_known"] is False
    assert no_materialized_summary["materialized_read_complete"] is None
    missing_profile = root / "profile-v2-missing-read.ndjson"
    missing_environment = dict(v2_environment)
    missing_environment["NCNN_PROFILE_PATH"] = str(missing_profile)
    subprocess.run([str(v2_binary), "missing-read"], check=True,
                   env=missing_environment)
    missing_rows = [json.loads(line) for line in missing_profile.read_text().splitlines()]
    assert len(missing_rows) == 3
    missing_row = missing_rows[-1]
    assert missing_row["summary"]["runtime_materialized_read_bytes"] is None
    assert missing_row["summary"]["materialized_read_bytes_known"] is False
    assert missing_row["summary"]["expected_materialized_read_bytes"] == 64
    assert missing_row["summary"]["materialized_read_complete"] is False

    worker_source = root / "worker_harness.c"
    worker_binary = root / "worker_harness"
    worker_profile = root / "profile-v3.ndjson"
    worker_source.write_text(WORKER_HARNESS, encoding="utf-8")
    subprocess.run([
      args.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread",
      str(worker_source), args.runtime, "-o", str(worker_binary),
    ], check=True)
    worker_environment = dict(environment)
    worker_environment.update({
      "NCNN_PROFILE_PATH": str(worker_profile),
      "NCNN_PROFILE_SCHEMA": "3",
      "NCNN_PROFILE_ATTRIBUTION_REVISION": "attribution-v4",
      "NCNN_PROFILE_INPUT_HASH": "input-abc123",
    })
    subprocess.run([str(worker_binary)], check=True, env=worker_environment)
    worker_rows = [
      json.loads(line) for line in worker_profile.read_text().splitlines()
      if line.strip()
    ]
    assert len(worker_rows) == 1
    worker_document = worker_rows[0]
    assert worker_document["schema_version"] == 3
    assert worker_document["complete"] is True
    assert worker_document["attribution_revision"] == "attribution-v4"
    assert worker_document["input_hash"] == "input-abc123"
    assert worker_document["summary"]["event_mismatch_count"] == 0
    worker_events = {
      (event["id"], event["category"]): event
      for event in worker_document["events"]
    }
    parallel = worker_events[(8, "parallel")]
    assert parallel["exclusive_semantics"] == \
      "region_wall_minus_worker_span_union"
    assert parallel["region_worker_spans"] > 0
    assert parallel["worker_covered_wall_ns"] > 0
    assert parallel["region_wall_ns"] == parallel["inclusive_ns"]
    assert parallel["sample_scale"] == 1
    assert 0.0 < parallel["wall_coverage_share"] <= 1.0
    assert parallel["wall_exclusive_estimated_known"] is True
    assert parallel["exclusive_ns"] == \
      parallel["inclusive_ns"] - parallel["worker_covered_wall_ns"]
    assert worker_document["summary"]["worker_sampling"]["duty"] == 1
    assert worker_document["summary"]["worker_sampling"]["interval_overflow"] \
      is False
    assert worker_document["summary"]["worker_wall_attribution"]["additive"] \
      is True
    assert worker_events[(42, "worker_operation")]["time_domain"] == "worker_cpu"
    assert worker_events[(42, "worker_operation")]["calls"] == 2
    assert worker_events[(42, "worker_operation")]["calls_estimated"] == 2
    assert worker_events[(42, "worker_operation")]["inclusive_ns"] > \
      worker_events[(42, "worker_operation")]["exclusive_ns"]
    assert worker_events[(42, "worker_operation")]["wall_attributed_known"]
    assert worker_events[(42, "worker_operation")]["wall_attributed_ns"] > 0
    assert worker_events[(42, "worker_operation")]["worker_wall_union_known"]
    assert worker_events[(42, "worker_operation")]["worker_wall_union_ns"] > 0
    assert worker_events[(42, "worker_operation")]["worker_wall_union_ns"] < \
      worker_events[(7, "operation")]["inclusive_ns"]
    assert worker_events[(43, "worker_operation")]["calls"] == 2
    attributed_total = sum(
      event.get("wall_attributed_ns") or 0
      for event in worker_document["events"]
      if event["category"] == "worker_operation")
    assert attributed_total <= parallel["worker_covered_wall_ns"]
    assert worker_document["summary"]["top_level_time_ns"] == \
      worker_events[(7, "operation")]["inclusive_ns"]

    sampled_profile = root / "profile-v3-sampled.ndjson"
    sampled_environment = dict(worker_environment)
    sampled_environment.update({
      "NCNN_PROFILE_PATH": str(sampled_profile),
      "NCNN_PROFILE_SAMPLE_DUTY": "4",
    })
    subprocess.run([str(worker_binary)], check=True, env=sampled_environment)
    sampled_rows = [
      json.loads(line) for line in sampled_profile.read_text().splitlines()
      if line.strip()
    ]
    assert len(sampled_rows) == 1
    sampled = sampled_rows[0]
    assert sampled["summary"]["worker_sampling"]["duty"] == 4
    assert sampled["summary"]["worker_sampling"]["basis"] == \
      "instance_counter_duty"
    sampled_parallel = {
      (event["id"], event["category"]): event
      for event in sampled["events"]
    }[(8, "parallel")]
    assert sampled_parallel["exclusive_semantics"] == "not_proven"
    assert sampled_parallel["wall_exclusive_estimated_known"] is True
    sampled_worker = {
      (event["id"], event["category"]): event
      for event in sampled["events"]
    }.get((42, "worker_operation"))
    if sampled_worker is not None and sampled_worker["calls"]:
      assert sampled_worker["calls_estimated"] == \
        sampled_worker["calls"] * 4
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
