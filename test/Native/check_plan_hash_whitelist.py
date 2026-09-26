#!/usr/bin/env python3
"""Guard the plan-hash field whitelist (T-C1).

`plan_hash` (and therefore `build_identity`) is derived from a fixed set of
fields.  Anything outside that set must not flip the hash; anything inside it
must.  This check pins the whitelist by asserting that the set of
`plan_hash_input` mutation sites in EmitModelPlan.cpp is exactly the documented
set.  Adding a new hash fragment without updating WHITELIST here fails the test
on purpose: it forces the change to be a deliberate, reviewed identity decision
(roadmap 7.5).

usage: check_plan_hash_whitelist.py
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGET = REPO / "lib" / "Transforms" / "EmitModelPlan" / "EmitModelPlan.cpp"

# Key = the concatenation of every string literal on the mutation statement,
# space-separated (that is what the source parser below extracts).
# Value = which fields this fragment covers.
WHITELIST: dict[str, str] = {
    "static-v1": (
        "initial assembly: plan/contract revisions, attribution revision, "
        "model, target triple, threads, vector lanes/scalable/tail, codegen "
        "identity, fusion ledger summary, copy ledger summary, fusion "
        "rejection reasons, and the four folded fragments below"
    ),
    "|fusion-profile-id=": "per fusion-ledger record: profile_id",
    "|fusion-tail-profile-id=": "per fusion-ledger record: tail_profile_id",
    "|fusion-record= |": "per fusion-ledger record: function | kind",
    "| |": "per walked operation: id | kind",
    "|": (
        "per walked operation: one entry per operand type and one per result "
        "type (two sites share this fragment)"
    ),
    "|attrs=": "per walked operation: printed attribute dictionary",
    "|source-layer=": "recovered NCNN source layer index",
    "|source-name=": "recovered NCNN source layer name",
    "|low-precision-op=": "per INT8-audited operation: id",
    "|copy-record-malformed#": "malformed copy-ledger record: ordinal (two sites)",
    "|copy-record# |function= |contract= |kind= |bytes= unknown |lanes=": (
        "per copy-ledger record: ordinal, function, contract, kind, bytes, lanes"
    ),
}

# Named accumulators folded into the initial assembly.  Each covers its own
# field group; losing one silently shrinks the identity, so assert presence.
FOLDED: dict[str, str] = {
    "low_precision_hash_input": "low-precision revision + int8 policy/capability",
    "layout_island_hash_input": "layout-island revision + packed-conv-depthwise",
    "tuning_hash_input": "tuning profile/status/fallback + matmul tiles",
    "attention_hash_input": "attention ledger revision + per-segment fields",
}

# Sites allowed to appear more than once (same fragment, distinct code paths).
MULTI_ALLOWED = {
    "|": 2,
    "|copy-record-malformed#": 2,
}


def key_of(statement: str) -> str:
    return " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', statement))


def collect_sites(source: str) -> list[str]:
    lines = source.splitlines()
    sites: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"\s*plan_hash_input\s*(\+)?=", lines[i]):
            block = lines[i]
            while not block.rstrip().endswith(";") and i + 1 < len(lines):
                i += 1
                block += " " + lines[i].strip()
            sites.append(key_of(block))
        i += 1
    return sites


def main() -> int:
    if not TARGET.is_file():
        print(f"missing {TARGET}", file=sys.stderr)
        return 2

    source = TARGET.read_text(encoding="utf-8")
    sites = collect_sites(source)
    errors: list[str] = []

    for name, why in FOLDED.items():
        if not re.search(rf"\b{name}\b", source):
            errors.append(f"folded hash fragment {name} disappeared ({why})")

    counts: dict[str, int] = {}
    for site in sites:
        counts[site] = counts.get(site, 0) + 1

    for key, why in WHITELIST.items():
        if key == "static-v1":
            if not any(s.startswith("static-v1") for s in sites):
                errors.append(f"the initial plan_hash_input assembly is gone ({why})")
            continue
        expected = MULTI_ALLOWED.get(key, 1)
        actual = counts.get(key, 0)
        if actual != expected:
            errors.append(
                f"hash fragment {key!r} appears {actual} time(s), expected "
                f"{expected} ({why})"
            )

    for key, count in counts.items():
        if key.startswith("static-v1"):
            continue
        if key not in WHITELIST:
            errors.append(
                f"plan_hash_input is extended from a site that is not on the "
                f"whitelist: {key!r} (x{count}). If this field is meant to be "
                "covered by the plan identity, add it to WHITELIST here and "
                "record the identity change; if it is not covered, it must not "
                "be appended to plan_hash_input."
            )

    if errors:
        print("plan-hash whitelist check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"plan-hash whitelist check passed ({len(sites)} mutation sites, "
        f"{len(WHITELIST)} whitelisted fragments, {len(FOLDED)} folded fragments)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
