#!/usr/bin/env python3
"""Assert the plan-hash field whitelist holds (T-C1).

usage: plan_hash_whitelist_check.py <baseline.json> <path-only-change.json> <hash-change.json>

1. Changing only the output path must leave `plan_hash` alone -- the path is not
   on the whitelist.
2. Changing a whitelisted field (thread count) must change `plan_hash`.
3. `build_identity` must stay pinned to `plan_hash`.
"""
from __future__ import annotations

import json
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    baseline, path_only, hash_change = (load(p) for p in argv[1:4])

    if baseline["plan_hash"] != path_only["plan_hash"]:
        print(
            "plan_hash changed although only the output path differed; "
            "the path must not be part of the hash whitelist",
            file=sys.stderr,
        )
        return 1
    if baseline["plan_hash"] == hash_change["plan_hash"]:
        print(
            "plan_hash did not change although a whitelisted field (threads) "
            "changed",
            file=sys.stderr,
        )
        return 1
    if baseline["build_identity"] != baseline["plan_hash"]:
        print("build_identity must equal plan_hash", file=sys.stderr)
        return 1
    if hash_change["build_identity"] != hash_change["plan_hash"]:
        print("build_identity must equal plan_hash", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
