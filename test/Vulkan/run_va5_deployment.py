#!/usr/bin/env python3
"""VA5 clean-directory deployment evidence for the published Vulkan model DSO.

The product CLI compiles the fixtures through the tightened VA5 link/audit
path.  Only the model ``.so`` and its public header are copied into an
isolated deployment directory; a dl-only C ABI loader compiled against that
directory runs them under a scrubbed environment (no build directory, no LLVM
library path, no project runtime path).  Device profiles verify CPU/Vulkan
numeric agreement; the offline profile verifies loadability plus an explicit
``UNSUPPORTED_DEVICE``/``API_ERROR`` create failure.  The script also proves
same-input reproducibility and that a later failed compile never overwrites
the previously published artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

SCHEMA_VERSION = 1
EVIDENCE_KIND = "ncnn.vulkan_va5_deployment_evidence"

FIXTURES = ("relu", "conv_relu")
# Substrings that must not appear in the loader's runtime library search path.
FORBIDDEN_LIBRARY_PATH_MARKERS = (
    "/llvm",
    "libLLVM",
    "ncnn-compiler-stage-",
    "ncnn-compiler-va",
)
# Fixed-run/VA5 manifest contract pinned by the compiler.
HOST_CONTRACT = {
    "host_implementation": "model-fixed-run",
    "host_codegen_version": 2,
    "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
    "execution_plan_usage": "build-evidence-only",
}


class DeploymentError(RuntimeError):
    """Raised when a deployment scenario fails its acceptance check."""


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(command, *, env=None, expect_success=True, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, env=env,
                            cwd=cwd)
    record = {"argv": [str(item) for item in command],
              "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if expect_success and result.returncode != 0:
        raise DeploymentError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise DeploymentError(f"command unexpectedly succeeded: {command}")
    return record


def load_product_module(native_dir: Path):
    import importlib.util
    source = native_dir / "check_ncnn_compile_vulkan.py"
    spec = importlib.util.spec_from_file_location(
        "ncnn_va5_product_helpers", source)
    if spec is None or spec.loader is None:
        raise DeploymentError(f"cannot load product helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ncnn_va5_product_helpers", module)
    spec.loader.exec_module(module)
    return module


def scrubbed_environment(product, args, *, allow_sanitizer_runtime: bool) -> dict:
    """Build the loader environment by allowlist: no inherited search paths.

    The deployment loader must resolve everything through system defaults and
    the artifacts it is explicitly given.  The configured sanitizer runtime
    directory stays available (and recorded) for instrumented profiles; it is
    classified separately from the Release dependency allowlist and is never a
    compiler artifact.  Every dropped inherited segment is returned so the
    evidence can show exactly what was scrubbed.
    """
    del product  # imported for make_input/queue helpers only
    env = os.environ.copy()
    for key in ("VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES"):
        env.pop(key, None)
    inherited = env.pop("LD_LIBRARY_PATH", "")
    inherited_segments = [segment for segment in inherited.split(":") if segment]
    keep = []
    if allow_sanitizer_runtime and args.sanitizer_runtime_dir:
        keep.append(str(Path(args.sanitizer_runtime_dir).resolve()))
    if keep:
        env["LD_LIBRARY_PATH"] = ":".join(keep)
    if args.icd:
        env["VK_ICD_FILENAMES"] = str(args.icd)
    else:
        unavailable = args.work_dir / "no-available-icd.json"
        if unavailable.exists():
            unavailable.unlink()
        env["VK_ICD_FILENAMES"] = str(unavailable)
    if args.device:
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    env["_NCNN_VA5_SCRUBBED_LD_LIBRARY_PATH"] = ":".join(inherited_segments)
    return env


def compile_fixture(product, args, model, param, binary, output_dir, env) -> dict:
    # The deployment scenario does not instrument the host link; link argv,
    # map, and dependency-closure evidence for VA5.1/VA5.2 come from the
    # product tests' captured link records, which share this same compiler
    # configuration.
    del product  # imported for make_input/queue helpers only
    return invoke([
        args.compiler,
        str(param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(output_dir),
        "--backend=vulkan",
        "--precision=f32",
    ], env=env)


def cpu_compile_fixture(product, args, model, param, binary, output_dir, env):
    return invoke([
        args.compiler,
        str(param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(output_dir),
        "--precision=f32",
        "--threads=1",
        "--vector-width=0",
    ], env=env)


def check_published_artifacts(args, model, output_dir):
    header = f"{model}_vulkan.h"
    library = f"lib{model}_vulkan.so"
    manifest_name = f"{model}_vulkan.json"
    for name in (header, library, manifest_name, f"{model}_vulkan.plan.json"):
        path = output_dir / name
        if not path.is_file():
            raise DeploymentError(f"published Vulkan artifact is missing: {path}")
    manifest = json.loads((output_dir / manifest_name).read_text())
    if manifest.get("schema_version") != 1 or \
            manifest.get("kind") != "ncnn.vulkan_model_manifest":
        raise DeploymentError(f"unexpected Vulkan manifest schema: {manifest}")
    for key, expected in HOST_CONTRACT.items():
        if manifest.get(key) != expected:
            raise DeploymentError(
                f"manifest {key} differs from the VA5 contract: "
                f"actual={manifest.get(key)!r} expected={expected!r}")
    if not manifest.get("host_codegen_identity"):
        raise DeploymentError("manifest host_codegen_identity is missing")
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, list) or "libvulkan.so.1" not in dependencies:
        raise DeploymentError(
            f"manifest platform dependencies are missing the Vulkan loader: "
            f"{dependencies}")
    if any(str(name).startswith("libLLVM") for name in dependencies):
        raise DeploymentError(f"manifest records an LLVM dependency: {dependencies}")
    if not isinstance(manifest.get("sanitizer_dependencies"), list):
        raise DeploymentError(
            "manifest is missing the sanitizer dependency classification")
    return {"header": header, "library": library,
            "manifest": manifest,
            "sha256s": {name: sha256(output_dir / name)
                        for name in (header, library, manifest_name,
                                     f"{model}_vulkan.plan.json")}}


def verify_deployment_directory(deploy: dict) -> None:
    """Fail unless the deployment directory holds exactly the staged pair."""
    directory = Path(deploy["directory"])
    staged = sorted(path.name for path in directory.iterdir())
    expected = sorted(item["file"] for item in deploy["deployed"])
    if staged != expected:
        raise DeploymentError(
            f"deployment directory is not minimal: {staged} vs {expected}")
    for item in deploy["deployed"]:
        path = directory / item["file"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise DeploymentError(
                f"deployed artifact differs from its published source: "
                f"{item['file']}")


def build_deployment_directory(args, model, output_dir) -> dict:
    deploy_dir = args.work_dir / f"deploy-{model}"
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir(parents=True)
    deployed = []
    for name in (f"{model}_vulkan.h", f"lib{model}_vulkan.so"):
        source = output_dir / name
        shutil.copy2(source, deploy_dir / name)
        deployed.append({"file": name, "sha256": sha256(source)})
    deploy = {"directory": str(deploy_dir), "deployed": deployed}
    verify_deployment_directory(deploy)
    return deploy


def run_loader(product, args, model, deploy: dict, cpu_dir: Path,
               input_data, env, *, expect_no_device: bool) -> dict:
    prefix = "no-device-" if expect_no_device else ""
    loader = args.work_dir / f"{prefix}va5-deployment-loader-{model}"
    elements = len(input_data)
    input_path = args.work_dir / f"{prefix}va5-input-{model}.f32"
    cpu_output = args.work_dir / f"{prefix}va5-cpu-output-{model}.f32"
    vulkan_output = args.work_dir / f"{prefix}va5-vulkan-output-{model}.f32"
    result_path = args.work_dir / f"{prefix}va5-loader-result-{model}.json"
    input_path.write_bytes(struct.pack(f"<{elements}f", *input_data))
    compile_record = invoke([
        args.host_cxx, "-std=c++23", *args.cxx_arg,
        f"-DNCNN_PRODUCT_TEST_MODEL={model}",
        *(["-DNCNN_PRODUCT_TEST_EXPECT_NO_DEVICE=1"] if expect_no_device else []),
        "-I", str(cpu_dir),
        "-I", str(deploy["directory"]),
        "-I", str(args.vulkan_include_dir),
        str(args.loader_source),
        "-ldl", "-o", str(loader),
    ], env=env)
    command = [
        str(loader), model,
        str(cpu_dir / f"lib{model}.so"),
        str(Path(deploy["directory"]) / f"lib{model}_vulkan.so"),
        str(elements), str(input_path), str(cpu_output), str(vulkan_output),
        str(result_path),
    ]
    child = subprocess.run(command, capture_output=True, text=True, env=env)
    streams = {}
    for name, text in (("loader.stdout.log", child.stdout),
                       ("loader.stderr.log", child.stderr)):
        path = args.work_dir / f"{prefix}va5-{model}.{name}"
        path.write_text(text, encoding="utf-8")
        streams[name] = {"file": path.name, "sha256": sha256(path)}
    result = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as error:
            raise DeploymentError(
                f"deployment loader wrote invalid JSON: {error}\n{child.stderr}"
            ) from error
    record = {
        "compile_argv": compile_record["argv"],
        "argv": command,
        "returncode": child.returncode,
        "streams": streams,
        "result_sha256": sha256(result_path) if result_path.is_file() else None,
        "environment": {
            "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", ""),
            "scrubbed_LD_LIBRARY_PATH":
                env.get("_NCNN_VA5_SCRUBBED_LD_LIBRARY_PATH", ""),
            "VK_ICD_FILENAMES": env.get("VK_ICD_FILENAMES", ""),
            "VK_INSTANCE_LAYERS": env.get("VK_INSTANCE_LAYERS", ""),
        },
        "gpu_numeric": not expect_no_device,
    }
    logs = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    if child.returncode != 0:
        raise DeploymentError(
            f"deployment loader failed ({child.returncode}); result={result}\n{logs}")
    if not isinstance(result, dict) or result.get("status") != "success":
        raise DeploymentError(
            f"deployment loader reported failure: {result}\n{logs}")
    if expect_no_device:
        if (result.get("phase") != "expected-no-device" or
                result.get("vulkan_create_status") not in (2, 5) or
                result.get("context_null") is not True or
                result.get("diagnostic_bounded") is not True or
                result.get("vulkan_destroy_null_status") != 0 or
                not result.get("diagnostic") or vulkan_output.exists()):
            raise DeploymentError(
                f"invalid no-device lifecycle result: {result}\n{logs}")
        record["no_device_create_status"] = result.get("vulkan_create_status")
        return record
    if not cpu_output.is_file() or not vulkan_output.is_file():
        raise DeploymentError(
            f"deployment loader did not write both outputs; {logs}")
    cpu_values = struct.unpack(f"<{elements}f", cpu_output.read_bytes())
    vulkan_values = struct.unpack(f"<{elements}f", vulkan_output.read_bytes())
    errors = [abs(a - b) for a, b in zip(cpu_values, vulkan_values)]
    if max(errors, default=0.0) > 1.0e-6:
        raise DeploymentError(
            f"CPU and deployed Vulkan outputs differ: max_abs={max(errors)}")
    record["max_abs_cpu_vs_vulkan"] = max(errors, default=0.0)
    record["cpu_output_sha256"] = sha256(cpu_output)
    record["vulkan_output_sha256"] = sha256(vulkan_output)
    return record


def reproducibility_check(args, model, param, binary, first_dir, env) -> dict:
    repeat_dir = args.work_dir / f"repeat-{model}"
    if repeat_dir.exists():
        shutil.rmtree(repeat_dir)
    invoke([
        args.compiler,
        str(param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(repeat_dir),
        "--backend=vulkan",
        "--precision=f32",
    ], env=env)
    first_files = sorted(path.name for path in first_dir.iterdir())
    repeat_files = sorted(path.name for path in repeat_dir.iterdir())
    if first_files != repeat_files:
        raise DeploymentError(
            f"repeat compile published a different artifact set: "
            f"{repeat_files} vs {first_files}")
    artifacts = {}
    for name in first_files:
        first_bytes = (first_dir / name).read_bytes()
        repeat_bytes = (repeat_dir / name).read_bytes()
        if first_bytes != repeat_bytes:
            raise DeploymentError(f"Vulkan artifact is not reproducible: {name}")
        artifacts[name] = sha256(first_dir / name)
    return {"mode": "byte-identical", "artifacts": artifacts}


def failed_compile_preserves_outputs(args, model, param, binary, first_dir,
                                     env, first_hashes) -> dict:
    bad_param = args.work_dir / f"bad-{model}.param"
    bad_param.write_text("not-a-valid-ncnn-param\n")
    failure = invoke([
        args.compiler,
        str(bad_param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(first_dir),
        "--backend=vulkan",
        "--precision=f32",
    ], env=env, expect_success=False)
    for name, digest in first_hashes.items():
        path = first_dir / name
        if not path.is_file() or sha256(path) != digest:
            raise DeploymentError(
                f"failed compile changed the previously published artifact: {name}")
    return {"failure_returncode": failure["returncode"],
            "preserved_sha256s": dict(first_hashes)}


def fixture_elements(manifest: dict) -> int:
    """Public input element count from the published manifest shape."""
    shape = manifest["inputs"][0]["shape"]
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


def deployment_input(elements: int) -> list[float]:
    generator = random.Random(20261024)
    return [round(generator.uniform(-3.0, 3.0), 4) for _ in range(elements)]


def fixture_binary(args, fixture):
    if fixture == "conv_relu":
        return args.conv_bin
    return args.empty_bin


def fixture_param(args, fixture):
    if fixture == "conv_relu":
        return args.conv_param
    return args.param


def compiler_environment(args) -> dict:
    """Environment for the compiler processes: the compiler binary itself may
    need the shared LLVM/sanitizer runtime directories; these are inherited
    from the CMake configuration and never reach the deployment loader."""
    env = os.environ.copy()
    extra = []
    if args.llvm_lib_dir:
        extra.append(str(Path(args.llvm_lib_dir).resolve()))
    if args.sanitizer_runtime_dir:
        extra.append(str(Path(args.sanitizer_runtime_dir).resolve()))
    inherited = env.get("LD_LIBRARY_PATH", "")
    if extra:
        env["LD_LIBRARY_PATH"] = ":".join(extra + (
            [inherited] if inherited else []))
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--host-cxx", required=True)
    parser.add_argument("--loader-source", type=Path, required=True)
    parser.add_argument("--vulkan-include-dir", type=Path, required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--icd", help="ICD manifest for device profiles")
    parser.add_argument("--device", action="store_true")
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

    product = load_product_module(
        Path(__file__).resolve().parent.parent / "Native")
    compile_env = compiler_environment(args)
    scenarios = {}
    for fixture in FIXTURES:
        if fixture == "conv_relu" and not args.device:
            # No-device coverage proves loadability plus explicit create
            # failure; the ReLU fixture carries that scenario.
            scenarios[fixture] = {"status": "not-applicable-offline"}
            continue
        model = f"va5_{fixture}"
        output_dir = args.work_dir / f"published-{model}"
        cpu_dir = args.work_dir / f"cpu-{model}"
        loader_env = scrubbed_environment(
            product, args,
            allow_sanitizer_runtime=bool(args.sanitizer_runtime_dir))
        compile_fixture(product, args, model, fixture_param(args, fixture),
                        fixture_binary(args, fixture), output_dir, compile_env)
        identity = check_published_artifacts(args, model, output_dir)
        cpu_compile_fixture(product, args, model, fixture_param(args, fixture),
                            fixture_binary(args, fixture), cpu_dir, compile_env)
        deploy = build_deployment_directory(args, model, output_dir)
        generator = deployment_input(fixture_elements(identity["manifest"]))
        loader_record = run_loader(product, args, model, deploy, cpu_dir,
                                   generator, loader_env,
                                   expect_no_device=not args.device)
        scenario = {
            "status": "passed",
            "model": model,
            "published": identity["sha256s"],
            "deployment": deploy,
            "loader": loader_record,
        }
        if args.device:
            scenario["reproducibility"] = reproducibility_check(
                args, model, fixture_param(args, fixture),
                fixture_binary(args, fixture), output_dir, compile_env)
            scenario["failed_compile_preserves_outputs"] = (
                failed_compile_preserves_outputs(
                    args, model, fixture_param(args, fixture),
                    fixture_binary(args, fixture), output_dir, compile_env,
                    identity["sha256s"]))
        scenarios[fixture] = scenario
    if not args.device:
        # The offline profile still proves publication protection on ReLU.
        model = "va5_relu"
        output_dir = args.work_dir / f"published-{model}"
        identity = check_published_artifacts(args, model, output_dir)
        scenarios["relu"]["reproducibility"] = reproducibility_check(
            args, model, fixture_param(args, "relu"),
            fixture_binary(args, "relu"), output_dir, compile_env)
        scenarios["relu"]["failed_compile_preserves_outputs"] = (
            failed_compile_preserves_outputs(
                args, model, fixture_param(args, "relu"),
                fixture_binary(args, "relu"), output_dir, compile_env,
                identity["sha256s"]))
    profile = "device" if args.device else "no-device"
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "profile": profile,
        "icd": str(args.icd) if args.icd else None,
        "host_contract": dict(HOST_CONTRACT),
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VA5 deployment evidence ({profile}): {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
