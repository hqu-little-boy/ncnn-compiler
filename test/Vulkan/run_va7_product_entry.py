#!/usr/bin/env python3
"""VA7.1 product entry convergence evidence.

Every case drives the public product CLI (``ncnn-compile``) only.  The
convergence contract proven here:

- The default backend is still the CPU AOT product: ``lib<model>.so`` plus
  ``<model>.h``, with no Vulkan artifact of any kind.
- ``--backend=vulkan`` publishes the model-fixed-run host-AOT artifact set
  whose manifest pins the frozen VA5+ host contract and a well-formed
  ``host_codegen_identity``.
- There is no generic-runtime user mode and no silent fallback: invalid
  backend values and undocumented opt-out flags fail closed with a frozen
  diagnostic and publish nothing, and a forged generic-runtime environment
  variable changes neither the Vulkan contract nor the CPU default.
- The published Vulkan DSO stays structurally clean: ``nm -D
  --undefined-only`` finds no compiler/plan namespace or generic executor
  symbol and ``readelf -d`` finds no LLVM/MLIR dynamic dependency, so the
  retired ``NCNNVulkanRuntime``/plan sources stay unreachable from the
  product link graph.

Numerical execution of the published models is carried by the v5 device
suites and the VA6 cross-ICD test; this runner deliberately does not run the
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

SCHEMA_VERSION = 1
EVIDENCE_KIND = "ncnn.vulkan_va7_product_entry_evidence"
# Published-product host contract pinned by the compiler since VA5.
HOST_CONTRACT = {
    "host_implementation": "model-fixed-run",
    "host_codegen_version": 2,
    "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
    "execution_plan_usage": "build-evidence-only",
}
IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VULKAN_ARTIFACTS = (
    "{model}_vulkan.h",
    "lib{model}_vulkan.so",
    "{model}_vulkan.json",
    "{model}_vulkan.plan.json",
)
CPU_ARTIFACTS = ("lib{model}.so", "{model}.h")
# Frozen CLI diagnostics for the fail-closed cases.
INVALID_BACKEND_DIAGNOSTIC = "--backend must be either cpu or vulkan"
UNKNOWN_FLAG_DIAGNOSTIC = "Unknown command line argument"
INVALID_BACKEND_VALUES = ("generic", "generic-runtime", "vulkan-runtime")
UNKNOWN_OPT_OUT_FLAGS = ("--generic-runtime", "--use-generic-runtime",
                         "--plan-runtime", "--generic-executor")
# The product structural audit re-verified here with independent tools; the
# compiler already fails closed on the same patterns at publish time.  Core
# compiler DSOs are rejected outright; sanitizer instrumentation runtimes
# (libclang_rt.*) are legal only through the manifest's separate
# `sanitizer_dependencies` classification, so every NEEDED entry must be
# covered by the manifest's `dependencies` or `sanitizer_dependencies`.
FORBIDDEN_UNDEFINED = re.compile(
    r"llvm|mlir|ExecutionPlan|HostProgram|VulkanRuntimeContext")
FORBIDDEN_NEEDED = re.compile(r"lib(?:LLVM|MLIR)[-.]")


class EntryError(RuntimeError):
    """Raised when a product-entry case violates its acceptance check."""


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_param(model_input: str = "data", channels: int = 1,
               height: int = 4, width: int = 4) -> str:
    return ("7767517\n2 2\n"
            f"Input input_{model_input} 0 1 {model_input} "
            f"0={width} 1={height} 2={channels}\n"
            f"ReLU relu 1 1 {model_input} output 0=0.000000\n")


def write_fixture(directory: Path, model: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    param = directory / f"{model}.param"
    param.write_text(make_param(), encoding="utf-8")
    binary = directory / f"{model}.bin"
    binary.write_bytes(b"")
    return param, binary


def environment(args) -> dict:
    env = dict(os.environ)
    extra = []
    if args.llvm_lib_dir is not None:
        extra.append(str(args.llvm_lib_dir))
    if args.sanitizer_runtime_dir is not None:
        extra.append(str(args.sanitizer_runtime_dir))
    if extra:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*extra, existing]) if existing \
            else os.pathsep.join(extra)
    return env


def invoke(argv: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


def clear_directory(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def published_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.is_file())


def cpu_compile(argv_prefix: list[str], env: dict, model: str,
                param: Path, binary: Path, output_dir: Path) -> dict:
    clear_directory(output_dir)
    argv = [*argv_prefix, str(param), "--bin", str(binary),
            "--model-name", model, "--output-dir", str(output_dir),
            "--precision=f32", "--threads=1", "--vector-width=0"]
    result = invoke(argv, env)
    if result.returncode != 0:
        raise EntryError(
            f"CPU default compile failed ({result.returncode}): {result.stdout}")
    expected = sorted(name.format(model=model) for name in CPU_ARTIFACTS)
    published = published_names(output_dir)
    if published != expected:
        raise EntryError(
            f"CPU default published {published}, expected exactly {expected}")
    leaked = [name for name in published if "_vulkan" in name]
    if leaked:
        raise EntryError(f"CPU default published Vulkan artifacts: {leaked}")
    return {"kind": "cpu-default", "argv": argv, "returncode": 0,
            "published": published,
            "artifacts": {name: sha256(output_dir / name)
                          for name in published},
            "stderr_tail": result.stdout[-2000:]}


def vulkan_compile(argv_prefix: list[str], env: dict, model: str,
                   param: Path, binary: Path, output_dir: Path,
                   env_overrides: dict[str, str] | None = None) -> dict:
    clear_directory(output_dir)
    run_env = dict(env)
    run_env.update(env_overrides or {})
    argv = [*argv_prefix, str(param), "--bin", str(binary),
            "--model-name", model, "--output-dir", str(output_dir),
            "--backend=vulkan", "--precision=f32"]
    result = invoke(argv, run_env)
    if result.returncode != 0:
        raise EntryError(
            f"Vulkan backend compile failed ({result.returncode}): {result.stdout}")
    expected = sorted(name.format(model=model) for name in VULKAN_ARTIFACTS)
    published = published_names(output_dir)
    if published != expected:
        raise EntryError(
            f"Vulkan backend published {published}, expected exactly {expected}")
    manifest = json.loads((output_dir / f"{model}_vulkan.json").read_text(
        encoding="utf-8"))
    for key, value in HOST_CONTRACT.items():
        if manifest.get(key) != value:
            raise EntryError(
                f"manifest {key} differs from the fixed-run contract: "
                f"{manifest.get(key)!r}")
    identity = manifest.get("host_codegen_identity")
    if not isinstance(identity, str) or not IDENTITY_PATTERN.fullmatch(identity):
        raise EntryError(
            f"manifest host_codegen_identity is not a SHA-256 digest: {identity!r}")
    return {"kind": "vulkan-fixed-run", "argv": argv, "returncode": 0,
            "published": published,
            "artifacts": {name: sha256(output_dir / name)
                          for name in published},
            "manifest_contract": dict(HOST_CONTRACT),
            "host_codegen_identity": identity,
            "stderr_tail": result.stdout[-2000:]}


def rejection_case(argv_prefix: list[str], env: dict, model: str,
                   param: Path, binary: Path, output_dir: Path,
                   case_kind: str, extra_args: list[str],
                   diagnostic: str) -> dict:
    clear_directory(output_dir)
    argv = [*argv_prefix, str(param), "--bin", str(binary),
            "--model-name", model, "--output-dir", str(output_dir),
            "--precision=f32", *extra_args]
    result = invoke(argv, env)
    if result.returncode == 0:
        raise EntryError(
            f"{case_kind} was accepted; the entry must fail closed: {argv}")
    if diagnostic not in result.stdout:
        raise EntryError(
            f"{case_kind} diagnostic {diagnostic!r} missing: {result.stdout}")
    if published_names(output_dir):
        raise EntryError(
            f"{case_kind} published artifacts: {published_names(output_dir)}")
    return {"kind": case_kind, "argv": argv, "returncode": result.returncode,
            "diagnostic_substring": diagnostic,
            "published": [], "stderr_tail": result.stdout[-2000:]}


def structural_audit(args, model: str, output_dir: Path) -> dict:
    library = output_dir / f"lib{model}_vulkan.so"
    manifest = json.loads(
        (output_dir / f"{model}_vulkan.json").read_text(encoding="utf-8"))
    undefined = subprocess.run(
        [args.nm, "-D", "--undefined-only", str(library)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if undefined.returncode != 0:
        raise EntryError(f"nm -D failed: {undefined.stdout}")
    symbols = sorted({line.rsplit(" ", 1)[-1]
                      for line in undefined.stdout.splitlines() if line.strip()})
    forbidden_symbols = sorted(
        symbol for symbol in symbols if FORBIDDEN_UNDEFINED.search(symbol))
    if forbidden_symbols:
        raise EntryError(
            f"published DSO references compiler or generic plan symbols: "
            f"{forbidden_symbols}")
    dynamic = subprocess.run(
        [args.readelf, "-d", str(library)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if dynamic.returncode != 0:
        raise EntryError(f"readelf -d failed: {dynamic.stdout}")
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic.stdout)
    forbidden_needed = sorted(
        item for item in needed if FORBIDDEN_NEEDED.search(item))
    if forbidden_needed:
        raise EntryError(
            f"published DSO carries compiler dynamic dependencies: "
            f"{forbidden_needed}")
    # libclang_rt.* sanitizer runtimes are a separate classification, never
    # a Release allowlist item: every NEEDED entry must be classified by the
    # manifest itself (platform dependencies vs sanitizer runtimes).
    platform = set(manifest.get("dependencies", []))
    sanitizer = set(manifest.get("sanitizer_dependencies", []))
    unclassified = sorted(set(needed) - platform - sanitizer)
    if unclassified:
        raise EntryError(
            f"published DSO NEEDED entries are not classified by the "
            f"manifest: {unclassified}")
    if "RPATH" in dynamic.stdout or "RUNPATH" in dynamic.stdout:
        raise EntryError("published DSO contains RPATH or RUNPATH")
    return {
        "library": f"lib{model}_vulkan.so",
        "library_sha256": sha256(library),
        "undefined_symbol_count": len(symbols),
        "forbidden_undefined_absent": True,
        "dynamic_dependencies": needed,
        "classified_dependencies": sorted(platform),
        "classified_sanitizer_dependencies": sorted(sanitizer),
        "unclassified_needed_absent": True,
        "forbidden_needed_absent": True,
        "no_rpath_runpath": True,
        "tools": {"nm": args.nm, "readelf": args.readelf},
    }


def run_cases(args) -> dict:
    env = environment(args)
    prefix = [args.compiler]
    cases: dict[str, dict] = {}

    cpu_param, cpu_bin = write_fixture(args.work_dir / "fixtures", "entry_cpu")
    cases["cpu_default"] = cpu_compile(
        prefix, env, "entry_cpu", cpu_param, cpu_bin,
        args.work_dir / "out-cpu-default")

    vk_param, vk_bin = write_fixture(args.work_dir / "fixtures", "entry_vk")
    vulkan = vulkan_compile(prefix, env, "entry_vk", vk_param, vk_bin,
                            args.work_dir / "out-vulkan")
    cases["vulkan_backend"] = vulkan

    for value in INVALID_BACKEND_VALUES:
        name = f"reject_backend_{value.replace('-', '_')}"
        param, binary = write_fixture(args.work_dir / "fixtures", name)
        cases[name] = rejection_case(
            prefix, env, name, param, binary,
            args.work_dir / f"out-{name}", "invalid-backend",
            [f"--backend={value}"], INVALID_BACKEND_DIAGNOSTIC)
    for flag in UNKNOWN_OPT_OUT_FLAGS:
        name = f"reject_flag_{flag.lstrip('-').replace('-', '_')}"
        param, binary = write_fixture(args.work_dir / "fixtures", name)
        cases[name] = rejection_case(
            prefix, env, name, param, binary,
            args.work_dir / f"out-{name}", "unknown-flag",
            [flag, "--backend=vulkan"], UNKNOWN_FLAG_DIAGNOSTIC)

    # A forged generic-runtime environment must not switch the Vulkan
    # contract and must not turn the CPU default into a runtime product.
    # The codegen identity covers the model-specific host program, so the
    # no-op probe recompiles the same model name and compares identities.
    cases["env_generic_runtime_noop"] = vulkan_compile(
        prefix, env, "entry_vk", vk_param, vk_bin,
        args.work_dir / "out-env-vulkan",
        env_overrides={"NCNN_VULKAN_GENERIC_RUNTIME": "1"})
    env_cpu_param, env_cpu_bin = write_fixture(args.work_dir / "fixtures",
                                               "entry_env_cpu")
    cases["env_generic_runtime_cpu_default"] = cpu_compile(
        prefix, {**env, "NCNN_VULKAN_GENERIC_RUNTIME": "1"}, "entry_env_cpu",
        env_cpu_param, env_cpu_bin, args.work_dir / "out-env-cpu")

    if cases["vulkan_backend"]["host_codegen_identity"] != \
            cases["env_generic_runtime_noop"]["host_codegen_identity"]:
        raise EntryError(
            "the forged generic-runtime environment changed the codegen identity")

    structural = structural_audit(args, "entry_vk",
                                  args.work_dir / "out-vulkan")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "default_backend": "cpu",
        "cases": cases,
        "structural_audit": structural,
    }


def self_test() -> int:
    """Pure checks: fixture serialization, frozen diagnostics, patterns."""
    param = make_param()
    lines = param.strip().splitlines()
    if lines[0] != "7767517" or lines[1] != "2 2":
        raise AssertionError(f"unexpected param header: {lines[:2]}")
    if f"Input input_data 0 1 data 0=4 1=4 2=1" not in param:
        raise AssertionError("input layer serialization changed")
    if "ReLU relu 1 1 data output 0=0.000000" not in param:
        raise AssertionError("relu layer serialization changed")
    if INVALID_BACKEND_DIAGNOSTIC not in (
            "--backend must be either cpu or vulkan"):
        raise AssertionError("diagnostic constant drifted")
    for value in INVALID_BACKEND_VALUES:
        if value in ("cpu", "vulkan"):
            raise AssertionError("invalid backend value collides with a real one")
    for flag in UNKNOWN_OPT_OUT_FLAGS:
        if not flag.startswith("--"):
            raise AssertionError(f"opt-out probe must be a flag: {flag}")
    if FORBIDDEN_UNDEFINED.search("ncnn_profile_reset"):
        raise AssertionError("forbidden pattern false positive")
    if not FORBIDDEN_UNDEFINED.search("VulkanRuntimeContext"):
        raise AssertionError("forbidden pattern misses the generic executor")
    for soname in ("libLLVM.so", "libLLVM-21.so", "libMLIR.so.21.1.8"):
        if not FORBIDDEN_NEEDED.search(soname):
            raise AssertionError(f"needed pattern misses {soname}")
    for soname in ("libvulkan.so.1", "libclang_rt.asan-x86_64.so",
                   "libc.so.6"):
        if FORBIDDEN_NEEDED.search(soname):
            raise AssertionError(f"needed pattern false positive on {soname}")
    if math.prod((1, 4, 4)) != 16:
        raise AssertionError("fixture element count drifted")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    evidence = run_cases(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"VA7 product entry evidence: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EntryError, OSError, ValueError, json.JSONDecodeError,
            subprocess.SubprocessError) as error:
        print(f"run_va7_product_entry.py: {error}", file=sys.stderr)
        sys.exit(1)
