#!/usr/bin/env python3
"""Validate narrowly scoped sanitizer-only model compile budget exceptions."""

import hashlib
import json
from pathlib import Path
import re


LIMITS = {
    "ncnn_model_compile_budget_pp_ocrv5_server_det_static": 450,
    "ncnn_model_compile_budget_pp_ocrv5_server_rec": 450,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_budget_evidence(path, key, requested_seconds, source_fingerprint):
    if key not in LIMITS or requested_seconds != LIMITS[key]:
        raise ValueError(f"sanitizer budget exception is not authorized: {key}")
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    fixture = key.removeprefix("ncnn_model_compile_budget_")
    if (record.get("schema_version") != 1 or
            record.get("kind") != "ncnn.sanitizer_budget_failure" or
            record.get("source_fingerprint") != source_fingerprint or
            record.get("profile") != "sanitize" or
            record.get("fixture") != fixture or
            record.get("previous_budget_seconds") != 300 or
            record.get("requested_budget_seconds") != requested_seconds or
            type(record.get("elapsed_seconds")) is not int or
            record["elapsed_seconds"] <= record["previous_budget_seconds"]):
        raise ValueError("sanitizer budget evidence does not match this gate/source")
    raw_log = (path.parent / record["log_file"]).resolve()
    if raw_log.parent != path.parent or not raw_log.is_file():
        raise ValueError("sanitizer budget evidence must reference a sibling raw log")
    if sha256(raw_log) != record.get("log_sha256"):
        raise ValueError("sanitizer budget raw log hash does not match its record")
    text = raw_log.read_text(encoding="utf-8", errors="replace")
    # CMake wraps FATAL_ERROR messages at a fixed column with a two-space
    # continuation, which can split "超过硬预算 300s" across lines in the raw
    # build log.  Accept both the unwrapped and the wrapped display form; the
    # elapsed seconds must still match the record exactly.
    match = re.search(
        rf"fixture {re.escape(fixture)} 编译耗时 (\d+)s 超过硬预算\s+300s", text)
    if match is None or int(match.group(1)) != record["elapsed_seconds"]:
        raise ValueError("raw log does not prove the reported hard-budget failure")
    return {"record": record, "record_sha256": sha256(path),
            "log_sha256": sha256(raw_log), "record_path": str(path),
            "log_path": str(raw_log)}
