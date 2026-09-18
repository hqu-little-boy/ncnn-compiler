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
extern void __ncnn_profile_flush(void);
int main(void) {
  __ncnn_profile_event_begin(7, 0);
  __ncnn_profile_event_begin(8, 4);
  __ncnn_profile_event_end(8);
  __ncnn_profile_event_end(7);
  __ncnn_profile_alloc(9, 64);
  __ncnn_profile_copy(10, 32);
  __ncnn_profile_dealloc(9);
  __ncnn_profile_flush();
  __ncnn_profile_flush();
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
    assert summary["peak_live_bytes"] == 64
    assert summary["peak_live_proven"] is True
    assert summary["top_level_time_ns"] > 0
    assert summary["top_level_time_known"] is True
    assert summary["event_mismatch_count"] == 0
    records = {(event["id"], event["category"]): event
               for event in document["events"]}
    assert records[(9, "allocation")]["calls"] == 1
    assert records[(10, "copy")]["calls"] == 1
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
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
