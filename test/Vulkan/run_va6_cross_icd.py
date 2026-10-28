#!/usr/bin/env python3
"""VA6.3 same-DSO cross-ICD and device-late-binding evidence.

The product CLI compiles the paired ReLU and weighted Conv→ReLU fixtures
exactly once.  Only the model ``.so`` and public header are copied into an
isolated deployment directory; every later step loads that one fixed byte
sequence.  For each configured ICD the dl-only cross-ICD loader runs, under
a scrubbed environment:

- the ``vulkaninfo`` device/driver identity for that ICD;
- an ``NCNN_VK_DEVICE_AUTO`` create/run/destroy against the CPU output
  (max absolute error ≤ 1e-6);
- an explicit ordinal-0 create/run/destroy;
- an invalid ordinal, which must fail with ``UNSUPPORTED_DEVICE``, leave the
  context null, and publish nothing;
- the DSO hash is re-verified after every run.

A no-ICD environment (loader pointed at a nonexistent ICD manifest) must
fail ``create`` explicitly (status 2/5) and still report
``destroy(NULL) == SUCCESS``; the deployed DSO stays loadable and no
fallback executes the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from run_v5 import parse_vulkan_device_summary
from model_dependency_evidence import make_host_cxx_launcher

SCHEMA_VERSION = 1
EVIDENCE_KIND = "ncnn.vulkan_va6_cross_icd_evidence"
HOST_CONTRACT = {
    "host_implementation": "model-fixed-run",
    "host_codegen_version": 2,
    "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
    "execution_plan_usage": "build-evidence-only",
}
FIXTURES = ("relu", "conv_relu")
MAX_ABS_ERROR = 1.0e-6
FORBIDDEN_LIBRARY_PATH_MARKERS = (
    "/llvm",
    "libLLVM",
    "ncnn-compiler-stage-",
    "ncnn-compiler-va",
)


class CrossIcdError(RuntimeError):
    """Raised when a cross-ICD scenario fails its acceptance check."""


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(command, *, env=None, expect_success=True) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    record = {"argv": [str(item) for item in command],
              "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if expect_success and result.returncode != 0:
        raise CrossIcdError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise CrossIcdError(f"command unexpectedly succeeded: {command}")
    return record


def compiler_environment(args) -> dict:
    env = os.environ.copy()
    extra = []
    if args.llvm_lib_dir:
        extra.append(str(Path(args.llvm_lib_dir).resolve()))
    if args.sanitizer_runtime_dir:
        extra.append(str(Path(args.sanitizer_runtime_dir).resolve()))
    inherited = env.get("LD_LIBRARY_PATH", "")
    if extra:
        env["LD_LIBRARY_PATH"] = ":".join(extra + ([inherited] if inherited else []))
    return env


def scrubbed_environment(args, *, with_icd: Path | None,
                         validation: bool) -> dict:
    """Allowlist loader environment: no inherited library search paths."""
    env = os.environ.copy()
    for key in ("VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES"):
        env.pop(key, None)
    inherited = env.pop("LD_LIBRARY_PATH", "")
    scrubbed = [segment for segment in inherited.split(":") if segment]
    keep = []
    if args.sanitizer_runtime_dir:
        keep.append(str(Path(args.sanitizer_runtime_dir).resolve()))
    if keep:
        env["LD_LIBRARY_PATH"] = ":".join(keep)
    if with_icd is not None:
        env["VK_ICD_FILENAMES"] = str(with_icd)
    else:
        missing = args.work_dir / "no-available-icd.json"
        if missing.exists():
            missing.unlink()
        env["VK_ICD_FILENAMES"] = str(missing)
    if validation:
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    env["_NCNN_VA6_SCRUBBED_LD_LIBRARY_PATH"] = ":".join(scrubbed)
    return env


def compile_fixture(args, model: str, param: Path, binary: Path,
                    output_dir: Path, env: dict) -> dict:
    return invoke([
        args.compiler, str(param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(output_dir),
        "--backend=vulkan",
        "--precision=f32",
        "--vulkan-host-cxx=" + str(make_host_cxx_launcher(
            args.host_cxx, args.work_dir / f"link-evidence-{model}")),
    ], env=env)


def compile_cpu_fixture(args, model: str, param: Path, binary: Path,
                        output_dir: Path, env: dict) -> dict:
    return invoke([
        args.compiler, str(param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(output_dir),
        "--precision=f32",
        "--threads=1",
        "--vector-width=0",
    ], env=env)


def check_published(args, model: str, output_dir: Path) -> dict:
    header = output_dir / f"{model}_vulkan.h"
    library = output_dir / f"lib{model}_vulkan.so"
    manifest_path = output_dir / f"{model}_vulkan.json"
    for path in (header, library, manifest_path,
                 output_dir / f"{model}_vulkan.plan.json"):
        if not path.is_file():
            raise CrossIcdError(f"published artifact missing: {path}")
    manifest = json.loads(manifest_path.read_text())
    for key, expected in HOST_CONTRACT.items():
        if manifest.get(key) != expected:
            raise CrossIcdError(
                f"manifest {key} differs from the fixed-run contract: "
                f"{manifest.get(key)!r}")
    shape = manifest["inputs"][0]["shape"]
    return {"elements": int(__import__("math").prod(shape)),
            "library_sha256": sha256(library),
            "header_sha256": sha256(header)}


def build_deployment(args, model: str, published: Path) -> dict:
    deploy_dir = args.work_dir / f"deploy-{model}"
    shutil.rmtree(deploy_dir, ignore_errors=True)
    deploy_dir.mkdir(parents=True)
    deployed = []
    for name in (f"{model}_vulkan.h", f"lib{model}_vulkan.so"):
        shutil.copy2(published / name, deploy_dir / name)
        deployed.append({"file": name, "sha256": sha256(published / name)})
    staged = sorted(path.name for path in deploy_dir.iterdir())
    if staged != sorted(item["file"] for item in deployed):
        raise CrossIcdError(f"deployment directory is not minimal: {staged}")
    return {"directory": str(deploy_dir), "deployed": deployed}


def compile_loader(args, model: str, env: dict) -> Path:
    loader = args.work_dir / f"va6-cross-icd-loader-{model}"
    invoke([
        args.host_cxx, "-std=c++23", *args.cxx_arg,
        f"-DNCNN_VA6_CROSS_ICD_MODEL={model}",
        f'-DNCNN_VA6_MODEL_HEADER="{model}.h"',
        f'-DNCNN_VA6_VULKAN_HEADER="{model}_vulkan.h"',
        "-I", str(args.cpu_dirs[model]),
        "-I", str(args.work_dir / f"deploy-{model}"),
        "-I", str(args.vulkan_include_dir),
        str(args.loader_source),
        "-ldl", "-o", str(loader),
    ], env=env)
    return loader


def device_ordinal_from_summary(summary_text: str, device_match: str,
                                icd: str) -> int:
    device = parse_vulkan_device_summary(summary_text, device_match, icd)
    return int(str(device["index"]).removeprefix("GPU"))


def run_loader(args, model: str, loader: Path, device_index: int,
               expect_success: bool, elements: int, input_path: Path,
               env: dict, tag: str) -> dict:
    cpu_output = args.work_dir / f"{tag}-cpu-output.f32"
    vulkan_output = args.work_dir / f"{tag}-vulkan-output.f32"
    result_path = args.work_dir / f"{tag}-result.json"
    for path in (cpu_output, vulkan_output, result_path):
        if path.exists():
            path.unlink()
    command = [
        str(loader), model,
        str(args.cpu_dirs[model] / f"lib{model}.so"),
        str(Path(args.work_dir / f"deploy-{model}") /
            f"lib{model}_vulkan.so"),
        str(device_index),
        "1" if expect_success else "0",
        str(elements), str(input_path), str(cpu_output), str(vulkan_output),
        str(result_path),
    ]
    child = subprocess.run(command, capture_output=True, text=True, env=env)
    streams = {}
    for name, text in (("stdout", child.stdout), ("stderr", child.stderr)):
        stream_path = args.work_dir / f"{tag}-loader.{name}.log"
        stream_path.write_text(text, encoding="utf-8")
        streams[name] = {"file": stream_path.name, "sha256": sha256(stream_path)}
    if not result_path.is_file():
        raise CrossIcdError(
            f"loader wrote no result JSON ({child.returncode}):\n"
            f"{child.stdout}\n{child.stderr}")
    result = json.loads(result_path.read_text())
    record = {"argv": command, "returncode": child.returncode,
              "streams": streams, "result": result,
              "environment": {
                  "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", ""),
                  "scrubbed_LD_LIBRARY_PATH":
                      env.get("_NCNN_VA6_SCRUBBED_LD_LIBRARY_PATH", ""),
                  "VK_ICD_FILENAMES": env.get("VK_ICD_FILENAMES", ""),
                  "VK_INSTANCE_LAYERS": env.get("VK_INSTANCE_LAYERS", "")}}
    if child.returncode != 0:
        raise CrossIcdError(f"loader failed ({child.returncode}): {result}")
    if result.get("status") != "success":
        raise CrossIcdError(f"loader reported failure: {result}")
    return record


def verify_numeric(args, record: dict, tag: str, elements: int) -> dict:
    cpu_values = struct.unpack(
        f"<{elements}f",
        (args.work_dir / f"{tag}-cpu-output.f32").read_bytes())
    vulkan_values = struct.unpack(
        f"<{elements}f",
        (args.work_dir / f"{tag}-vulkan-output.f32").read_bytes())
    errors = [abs(a - b) for a, b in zip(cpu_values, vulkan_values)]
    if max(errors, default=0.0) > MAX_ABS_ERROR:
        raise CrossIcdError(
            f"{tag}: CPU and Vulkan outputs differ: max_abs={max(errors)}")
    return {"max_abs_cpu_vs_vulkan": max(errors, default=0.0)}


def scenario_for_icd(args, model: str, icd: dict, device: dict,
                     elements: int, env: dict) -> dict:
    loader = compile_loader(args, model, env)
    input_path = args.work_dir / f"{model}-input.f32"
    import random
    generator = random.Random(20261025)
    input_path.write_bytes(struct.pack(
        f"<{elements}f",
        *[round(generator.uniform(-3.0, 3.0), 4) for _ in range(elements)]))
    library = Path(args.work_dir / f"deploy-{model}") / f"lib{model}_vulkan.so"
    library_sha256 = sha256(library)

    auto = run_loader(args, model, loader, 0xFFFFFFFF, True, elements,
                      input_path, env, f"{model}-{icd['name']}-auto")
    auto["numeric"] = verify_numeric(
        args, auto, f"{model}-{icd['name']}-auto", elements)
    explicit = run_loader(args, model, loader, 0, True, elements,
                          input_path, env, f"{model}-{icd['name']}-ordinal0")
    explicit["numeric"] = verify_numeric(
        args, explicit, f"{model}-{icd['name']}-ordinal0", elements)
    invalid = run_loader(args, model, loader, 9999, False, elements,
                         input_path, env, f"{model}-{icd['name']}-invalid")
    invalid_result = invalid["result"]
    if int(invalid_result.get("create_status", -1)) != 2 or \
            invalid_result.get("context_null") is not True:
        raise CrossIcdError(
            f"{model}: invalid ordinal did not fail with UNSUPPORTED_DEVICE: "
            f"{invalid_result}")
    if (args.work_dir /
            f"{model}-{icd['name']}-invalid-vulkan-output.f32").exists():
        raise CrossIcdError(
            f"{model}: invalid-ordinal run published an output tensor")

    if sha256(library) != library_sha256:
        raise CrossIcdError(f"{model}: DSO bytes changed after runs")
    return {"icd": str(icd["manifest"]), "device": device,
            "loader": loader.name,
            "auto": auto, "explicit_ordinal_0": explicit,
            "invalid_ordinal": invalid,
            "library_sha256_unchanged": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--host-cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--loader-source", type=Path, required=True)
    parser.add_argument("--vulkan-include-dir", type=Path, required=True)
    parser.add_argument("--vulkaninfo", required=True)
    parser.add_argument("--icd", action="append", required=True,
                        metavar="NAME=MANIFEST:DEVICE_MATCH")
    parser.add_argument("--param", type=Path, required=True)
    parser.add_argument("--conv-param", type=Path, required=True)
    parser.add_argument("--empty-bin", type=Path, required=True)
    parser.add_argument("--conv-bin", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)

    icds = []
    for spec in args.icd:
        head, _, device_match = spec.rpartition(":")
        name, _, manifest = head.partition("=")
        icds.append({"name": name, "manifest": Path(manifest),
                     "device_match": device_match})
    if len(icds) < 2:
        raise CrossIcdError(
            "cross-ICD evidence requires at least two configured ICDs")

    env = compiler_environment(args)
    scenarios = {}
    for fixture in FIXTURES:
        model = f"va6_{fixture}"
        param = args.conv_param if fixture == "conv_relu" else args.param
        binary = args.conv_bin if fixture == "conv_relu" else args.empty_bin
        published = args.work_dir / f"published-{model}"
        cpu_dir = args.work_dir / f"cpu-{model}"
        compile_fixture(args, model, param, binary, published, env)
        compile_cpu_fixture(args, model, param, binary, cpu_dir, env)
        identity = check_published(args, model, published)
        args.cpu_dirs = getattr(args, "cpu_dirs", {})
        args.cpu_dirs[model] = cpu_dir
        deployment = build_deployment(args, model, published)
        device_runs = {}
        for icd in icds:
            loader_env = scrubbed_environment(args, with_icd=icd["manifest"],
                                              validation=True)
            vulkaninfo = invoke([args.vulkaninfo, "--summary"], env=loader_env)
            summary_text = vulkaninfo["stdout"] + vulkaninfo["stderr"]
            if "VK_LAYER_KHRONOS_validation" not in summary_text:
                raise CrossIcdError(
                    "Khronos validation layer is unavailable under ICD "
                    f"{icd['name']}")
            summary_path = (args.work_dir /
                            f"vulkaninfo-{fixture}-{icd['name']}.txt")
            summary_path.write_text(summary_text, encoding="utf-8")
            device = parse_vulkan_device_summary(
                summary_text, icd["device_match"], str(icd["manifest"]))
            device_runs[icd["name"]] = scenario_for_icd(
                args, model, icd, device, identity["elements"], loader_env)
        # No-ICD environment: create must fail explicitly, nothing publishes.
        no_icd_env = scrubbed_environment(args, with_icd=None, validation=False)
        loader = compile_loader(args, model, env)
        input_path = args.work_dir / f"{model}-no-icd-input.f32"
        input_path.write_bytes(b"\0" * (identity["elements"] * 4))
        no_icd = run_loader(args, model, loader, 0xFFFFFFFF, False,
                            identity["elements"], input_path, no_icd_env,
                            f"{model}-no-icd")
        no_icd_result = no_icd["result"]
        if int(no_icd_result.get("create_status", -1)) not in (2, 5) or \
                no_icd_result.get("context_null") is not True or \
                int(no_icd_result.get("destroy_null_status", -1)) != 0:
            raise CrossIcdError(
                f"{model}: no-ICD create failure is not explicit: "
                f"{no_icd_result}")
        scenarios[fixture] = {
            "model": model,
            "published_sha256": identity["library_sha256"],
            "deployment": deployment,
            "devices": device_runs,
            "no_icd": no_icd,
        }
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "host_contract": dict(HOST_CONTRACT),
        "icds": [{"name": icd["name"], "manifest": str(icd["manifest"]),
                  "device_match": icd["device_match"]} for icd in icds],
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VA6 cross-ICD evidence: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
