#!/usr/bin/env python3
"""Run one serial chunk of a five-round profile-off paired benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


CHUNKS = {
    "serverdet": (r"^PerformanceModel\.PPOcrv5ServerDetStatic$", None, 1),
    "serverrec": (r"^PerformanceModel\.PPOcrv5ServerRec$", None, 1),
    "rest": (r"^PerformanceModel\.",
             r"^PerformanceModel\.PPOcrv5Server(DetStatic|Rec)$", 43),
}


def validate_rows(path: Path, expected: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    models = [row["model"] for row in rows]
    if len(rows) != expected or len(set(models)) != expected:
        raise RuntimeError(f"{path}: expected {expected} unique models, found {len(rows)}")
    for row in rows:
        if (row.get("status") != "measured" or row.get("mode") != "end_to_end"
                or row.get("threads") != 6 or row.get("warmup") != 10
                or row.get("iterations") != 20):
            raise RuntimeError(f"{path}: invalid performance metadata for {row['model']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--chunk", choices=tuple(CHUNKS), required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    stage = args.baseline if args.variant == "baseline" else args.candidate
    stem = f"round-{args.round}-{args.variant}"
    result = args.evidence / f"{stem}-{args.chunk}.ndjson"
    log = args.evidence / f"{stem}-{args.chunk}.ctest.log"
    if result.exists() or log.exists():
        parser.error(f"refusing to overwrite existing evidence for {stem}-{args.chunk}")
    environment = os.environ.copy()
    environment.update({
        "NCNN_PERF_THREADS": "6", "NCNN_PERF_WARMUP": "10",
        "NCNN_PERF_ITERS": "20", "NCNN_PERF_MAX_RATIO": "0",
        "NCNN_PERF_MODE": "end_to_end", "NCNN_PERF_JSON": str(result),
    })
    include, exclude, count = CHUNKS[args.chunk]
    command = ["ctest", "--test-dir", str(stage), "-R", include,
               "--output-on-failure"]
    if exclude:
        command.extend(["-E", exclude])
    print(f"Running {stem}-{args.chunk}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        print(f"{stem}-{args.chunk} failed; see {log}", file=sys.stderr)
        return completed.returncode
    validate_rows(result, count)
    if args.chunk == "rest":
        rows = []
        for chunk, (_, _, expected) in CHUNKS.items():
            rows.extend(validate_rows(args.evidence / f"{stem}-{chunk}.ndjson",
                                      expected))
        combined = args.evidence / f"{stem}.ndjson"
        if combined.exists():
            parser.error(f"refusing to overwrite existing evidence: {combined}")
        if len({row["model"] for row in rows}) != 45:
            raise RuntimeError(f"{stem}: chunks do not contain 45 unique models")
        combined.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                                    for row in sorted(rows, key=lambda row: row["model"])),
                            encoding="utf-8")
        print(f"Completed {stem}: 45 unique measured models", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
