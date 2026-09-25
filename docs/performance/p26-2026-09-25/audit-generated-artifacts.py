#!/usr/bin/env python3
"""Audit P26 generated model sidecars, code size and E2E identities."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys


ARCHIVE = Path(__file__).resolve().parent
DIAGNOSTIC = "resnet18_winograd"


def official_models() -> set[str]:
    module = ast.parse((ARCHIVE / "finalize-end-to-end.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "OFFICIAL"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("OFFICIAL model set not found in finalizer")


def main() -> int:
    stage = Path(sys.argv[1])
    generated = stage / "test/Numerical/generated"
    libraries = sorted(generated.glob("*/lib*.so"))
    missing_plan: list[str] = []
    invalid_plan: list[str] = []
    missing_manifest: list[str] = []
    invalid_manifest: list[str] = []
    for library in libraries:
        model = library.stem.removeprefix("lib")
        for path, missing, invalid in (
            (library.parent / f"{model}.plan.json", missing_plan, invalid_plan),
            (library.parent / f"{model}.json", missing_manifest, invalid_manifest),
        ):
            if not path.is_file():
                missing.append(str(path.relative_to(generated)))
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid.append(str(path.relative_to(generated)))

    sidecar_result = {
        "generated_shared_libraries": len(libraries),
        "valid_execution_plans": len(libraries) - len(missing_plan) - len(invalid_plan),
        "valid_model_manifests": len(libraries) - len(missing_manifest) - len(invalid_manifest),
        "missing_plan": missing_plan,
        "invalid_plan": invalid_plan,
        "missing_manifest": missing_manifest,
        "invalid_manifest": invalid_manifest,
    }
    (ARCHIVE / "build-sidecar-integrity.json").write_text(
        json.dumps(sidecar_result, indent=2) + "\n", encoding="utf-8")

    models = sorted(official_models() | {DIAGNOSTIC})
    code_rows = []
    missing_libraries = []
    for model in models:
        path = generated / model / f"lib{model}.so"
        if not path.is_file():
            missing_libraries.append(model)
        else:
            code_rows.append({"model": model, "shared_library_bytes": path.stat().st_size})
    code_result = {
        "scope": "44 official models plus resnet18_winograd diagnostic",
        "model_count": len(code_rows),
        "missing": missing_libraries,
        "total_shared_library_bytes": sum(row["shared_library_bytes"] for row in code_rows),
        "models": code_rows,
    }
    (ARCHIVE / "compiled-model-code-size.json").write_text(
        json.dumps(code_result, indent=2) + "\n", encoding="utf-8")

    rows = json.loads((ARCHIVE / "final-all-model-ncnn-comparison.json").read_text())
    expected = set(models)
    row_names = {row.get("model") for row in rows}
    missing_identity = [
        row.get("model") for row in rows
        if not row.get("target") or not row.get("plan_hash") or not row.get("build_identity")
    ]
    identity_result = {
        "rows": len(rows),
        "expected_rows": len(expected),
        "missing_models": sorted(expected - row_names),
        "unexpected_models": sorted(row_names - expected),
        "missing_identity_rows": missing_identity,
        "distinct_targets": sorted({row.get("target") for row in rows if row.get("target")}),
        "distinct_plan_hashes": len({row.get("plan_hash") for row in rows if row.get("plan_hash")}),
        "distinct_build_identities": len({row.get("build_identity") for row in rows if row.get("build_identity")}),
    }
    (ARCHIVE / "performance-identity-check.json").write_text(
        json.dumps(identity_result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "sidecars": sidecar_result,
        "code_size_bytes": code_result["total_shared_library_bytes"],
        "performance_identity": identity_result,
    }, indent=2))
    if any((missing_plan, invalid_plan, missing_manifest, invalid_manifest,
            missing_libraries, identity_result["missing_models"],
            identity_result["unexpected_models"], identity_result["missing_identity_rows"])):
        return 1
    if identity_result["rows"] != len(expected):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
