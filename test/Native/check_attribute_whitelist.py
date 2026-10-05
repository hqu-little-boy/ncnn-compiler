#!/usr/bin/env python3
"""Guard the ncnn.* attribute schema's single source of truth.

KernelContract.hpp and the dedicated VulkanDocumentKinds.hpp contain the
canonical "ncnn.*" key definitions. Passes, conversions, the importer and tools
must reference those named constants; otherwise the schema silently grows a
second definition with a different answer -- which is how the audit found 282
bare literals against the 56 constants that were supposed to be authoritative.

The whitelist below pins the exact residual literal set, per file and per key.
An allowed file may not grow a new key: adding one is a violation until it is
either folded into a contract header or explicitly recorded with a reason.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
SCAN_DIRS = ("lib", "include", "tools")
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".c", ".py"}

# file -> {key: reason}.  Every reason must name a real constraint.
ALLOWED_SITES: dict[str, dict[str, str]] = {
    "lib/ProfileRuntime/profile_runtime.c": {
        "ncnn.model_execution_profile": (
            "the C profile runtime is installed as a standalone .c and compiled "
            "into each generated .so, so it cannot include KernelContract.hpp, "
            "and its symbol set is frozen by the profile_allowed allow-list"
        ),
    },
    "tools/attr/attr_schema.py": {
        "ncnn.model_execution_plan": (
            "JSON document-kind discriminator read from the plan; Python cannot "
            "link the C++ contract constant"
        ),
        "ncnn.model_execution_profile": (
            "JSON document-kind discriminator read from the profile"
        ),
    },
    "tools/attr/attr_report.py": {
        "ncnn.model_performance_attribution": (
            "JSON document-kind value written into the attribution report"
        ),
    },
}

KEY_RE = re.compile(r'"(ncnn\.[A-Za-z0-9_.]+)"')
CONST_RE = re.compile(
    r"inline constexpr llvm::StringLiteral\s+(\w+)\s*=\s*\n?\s*"
    r"\"(ncnn\.[A-Za-z0-9_.]+)\";"
)


def contract_keys(root: pathlib.Path) -> dict[str, str]:
    """Collect keys from the canonical contract headers."""
    mapping: dict[str, str] = {}
    headers = (
        root / "include/ncnn-mlir/Support/KernelContract.hpp",
        root / "include/ncnn-mlir/Support/VulkanDocumentKinds.hpp",
    )
    for header in headers:
        if not header.exists():
            continue
        for name, key in CONST_RE.findall(header.read_text()):
            if key in mapping:
                raise AssertionError(
                    f"contract headers define {key} twice: {mapping[key]} and {name}"
                )
            mapping[key] = name
    return mapping


def find_bare_literals(root: pathlib.Path) -> list[tuple[str, int, str]]:
    """Every "ncnn.*" literal outside the contract headers."""
    headers = {
        root / "include/ncnn-mlir/Support/KernelContract.hpp",
        root / "include/ncnn-mlir/Support/VulkanDocumentKinds.hpp",
    }
    offences: list[tuple[str, int, str]] = []
    for directory in SCAN_DIRS:
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            if path in headers:
                continue
            rel = path.relative_to(root).as_posix()
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                for key in KEY_RE.findall(line):
                    offences.append((rel, line_number, key))
    return offences


def check(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    try:
        keys = contract_keys(root)
    except OSError as failure:
        return [f"cannot read contract headers under {root}: {failure}"]
    if not keys:
        errors.append("contract headers defined no ncnn.* keys; parser broke?")

    for rel, line_number, key in find_bare_literals(root):
        allowed = ALLOWED_SITES.get(rel, {})
        where = f"{rel}:{line_number}"
        if key in allowed:
            continue
        name = keys.get(key)
        if name is None:
            errors.append(
                f"{where}: {key!r} is not in a contract header -- add "
                f"a constant, or record the site in ALLOWED_SITES with a reason"
            )
        else:
            errors.append(
                f"{where}: bare {key!r}; use contract::{name} instead "
                f"(or record the site in ALLOWED_SITES)"
            )
    return errors


def negative_case() -> list[str]:
    """A hand-written bare literal must be caught, or this guard is decoration."""
    errors: list[str] = []
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="ncnn-whitelist-"))
    try:
        header = scratch / "include/ncnn-mlir/Support/KernelContract.hpp"
        header.parent.mkdir(parents=True)
        header.write_text(
            'inline constexpr llvm::StringLiteral kReal = "ncnn.real_key";\n'
        )
        lib = scratch / "lib"
        lib.mkdir(parents=True)
        (lib / "known_key_bare.cpp").write_text(
            'void f() { setAttr("ncnn.real_key", 1); }\n'
        )
        (lib / "unknown_key.cpp").write_text(
            'void g() { setAttr("ncnn.not_a_real_key", 1); }\n'
        )
        (lib / "clean.cpp").write_text('void h() { setAttr("nope", 1); }\n')

        found = check(scratch)
        if not any("ncnn.real_key" in e for e in found):
            errors.append("negative case failed: bare use of a known key missed")
        if not any("ncnn.not_a_real_key" in e for e in found):
            errors.append("negative case failed: unknown key missed")
        if any("nope" in e for e in found):
            errors.append("negative case failed: non-ncnn attribute flagged")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return errors


def main() -> int:
    errors = check(REPO) + negative_case()
    if errors:
        print("attribute whitelist violations:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        print(f"total: {len(errors)}", file=sys.stderr)
        return 1

    keys = contract_keys(REPO)
    allowed_literals = sum(len(v) for v in ALLOWED_SITES.values())
    print(
        f"attribute whitelist ok: {len(keys)} keys defined in the contract "
        f"headers, 0 bare literals outside them, "
        f"{allowed_literals} recorded literals in "
        f"{len(ALLOWED_SITES)} allow-listed files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
