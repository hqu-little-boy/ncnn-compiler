#!/usr/bin/env python3
"""Recompute allocation timing fields from retained per-invocation profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def audit(profile: dict) -> dict:
    events = profile.get("events", [])
    summary = profile.get("summary", {})
    allocation_events = [event for event in events
                         if event.get("category") == "allocation"]
    deallocation_events = [event for event in events
                           if event.get("category") == "deallocation"]
    allocation_ns = sum(
        event["exclusive_ns"]
        for event in allocation_events
        if isinstance(event.get("exclusive_ns"), int)
    )
    timed_allocation_calls = sum(
        event.get("calls", 0)
        for event in allocation_events
        if isinstance(event.get("exclusive_ns"), int)
        and event["exclusive_ns"] > 0
    )
    operation_sites = [
        event["exclusive_ns"] for event in events
        if event.get("category") == "operation"
        and isinstance(event.get("exclusive_ns"), int)
        and event["exclusive_ns"] > 0
    ]
    ranked = sorted([allocation_ns, *operation_sites], reverse=True)
    allocation_rank = ranked.index(allocation_ns) + 1
    top_level_ns = summary.get("top_level_time_ns")
    allocation_count = summary.get("allocation_count")
    return {
        "complete": profile.get("complete"),
        "event_mismatch_count": summary.get("event_mismatch_count"),
        "allocation_count": allocation_count,
        "deallocation_count": summary.get("deallocation_count"),
        "allocation_bytes": summary.get("allocation_bytes"),
        "deallocation_bytes": summary.get("deallocation_bytes"),
        "peak_live_bytes": summary.get("peak_live_bytes"),
        "peak_live_proven": summary.get("peak_live_proven"),
        "allocation_event_site_count": len(allocation_events),
        "timed_allocation_calls": timed_allocation_calls,
        "allocation_time_coverage": (
            timed_allocation_calls / allocation_count
            if isinstance(allocation_count, int) and allocation_count > 0
            else None
        ),
        "allocation_exclusive_ns": allocation_ns,
        "deallocation_time_measured": False,
        "timed_deallocation_calls": 0,
        "deallocation_exclusive_ns": None,
        "allocation_time_share_of_top_level_percent": (
            allocation_ns * 100.0 / top_level_ns
            if isinstance(top_level_ns, int) and top_level_ns > 0
            else None
        ),
        "allocation_site_rank_among_operation_sites": allocation_rank,
        "top_level_time_ns": top_level_ns,
        "operation_site_count_timed": len(operation_sites),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.evidence / "allocation-qualification.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))

    for name, model in result["models"].items():
        profile_dir = args.evidence / "profile-on" / name
        paths = sorted(profile_dir.glob("invocation-*.profile.json"))
        if len(paths) != result["timed_invocations"]:
            raise SystemExit(
                f"{name}: expected {result['timed_invocations']} profiles, "
                f"got {len(paths)}"
            )
        rows = [audit(json.loads(path.read_text(encoding="utf-8")))
                for path in paths]
        median = lambda field: statistics.median(row[field] for row in rows)
        allocation = model["allocation"]
        allocation.update({
            "median_allocation_time_coverage": median("allocation_time_coverage"),
            "median_allocation_exclusive_ns": median("allocation_exclusive_ns"),
            "deallocation_time_measured": False,
            "median_deallocation_exclusive_ns": None,
            "median_top_level_time_ns": median("top_level_time_ns"),
            "median_allocation_time_share_percent": median(
                "allocation_time_share_of_top_level_percent"),
            "median_allocation_site_rank": median(
                "allocation_site_rank_among_operation_sites"),
            "all_profiles_complete": all(row["complete"] is True for row in rows),
            "all_event_mismatches_zero": all(
                row["event_mismatch_count"] == 0 for row in rows),
            "all_peaks_proven": all(row["peak_live_proven"] is True for row in rows),
            "per_invocation": rows,
        })
        for obsolete in (
            "median_allocation_and_deallocation_exclusive_ns",
            "median_time_share_percent",
            "median_rank",
        ):
            allocation.pop(obsolete, None)

    summary_path.write_text(json.dumps(result, indent=2) + "\n",
                            encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
