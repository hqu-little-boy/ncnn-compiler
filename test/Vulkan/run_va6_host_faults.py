#!/usr/bin/env python3
"""VA6.2 product-wrapper host fault-chain evidence.

The public product CLI compiles a weighted multi-output fixture
(``va6_faults``: one (2,1,1) input, two (2,1,1) outputs, two embedded weight
resources) through the tightened VA5 link path.  A host-compiler launcher
captures the actual generated ``vulkan_model_wrapper.cpp`` from the link
staging directory.  The runner proves the published manifest pins the
fixed-run contract, that the captured wrapper's codegen identity matches the
manifest, that a repeat compile reproduces the wrapper bytes, and that the
published DSO references no compiler/plan symbols.  The captured wrapper is
then compiled against the fake Vulkan test double with the device helper and
``NCNN_VULKAN_LIFECYCLE_TEST`` enabled, and the fault-chain binary executes
the argument/flush/submit/invalidate/mapping/validation failure, poison,
weight-upload, fixed-dispatch, and same-context serialization checks from
``va6_product_fault_test.cpp``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_dependency_evidence as dependency_evidence

SCHEMA_VERSION = 1
EVIDENCE_KIND = "ncnn.vulkan_va6_host_faults_evidence"
HOST_CONTRACT = {
    "host_implementation": "model-fixed-run",
    "host_codegen_version": 2,
    "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
    "execution_plan_usage": "build-evidence-only",
}
MODEL = "va6_faults"
WRAPPER_SOURCE_NAME = "vulkan_model_wrapper.cpp"
HEADER_NAME = f"{MODEL}_vulkan.h"
# Frozen fixture counts: see test/Vulkan/native/va6_product_fault_test.cpp.
DISPATCHES_PER_RUN = 4
COPIES_PER_RUN = 3
FORBIDDEN_UNDEFINED_MARKERS = (
    "llvm", "mlir", "ExecutionPlan", "HostProgram", "VulkanRuntimeContext",
)
SANITIZER_DIAGNOSTIC = re.compile(
    r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:",
    re.IGNORECASE,
)


class HostFaultsError(RuntimeError):
    """Raised when a VA6.2 host-faults scenario fails its acceptance check."""


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
    diagnostics = [
        line for line
        in (result.stdout + "\n" + result.stderr).splitlines()
        if SANITIZER_DIAGNOSTIC.search(line)
    ]
    if diagnostics:
        raise HostFaultsError(
            f"sanitizer diagnostics from {command}:\n" +
            "\n".join(diagnostics[:20]))
    if expect_success and result.returncode != 0:
        raise HostFaultsError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise HostFaultsError(f"command unexpectedly succeeded: {command}")
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


def compile_product(args, output_dir: Path, env: dict, launcher: Path) -> dict:
    command = [
        args.compiler, str(args.param),
        "--bin", str(args.conv_bin),
        "--model-name", MODEL,
        "--output-dir", str(output_dir),
        "--backend=vulkan",
        "--precision=f32",
        "--vulkan-host-cxx=" + str(launcher),
    ]
    return invoke(command, env=env)


def check_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / f"{MODEL}_vulkan.json"
    manifest = json.loads(manifest_path.read_text())
    for key, expected in HOST_CONTRACT.items():
        if manifest.get(key) != expected:
            raise HostFaultsError(
                f"manifest {key} differs from the fixed-run contract: "
                f"actual={manifest.get(key)!r} expected={expected!r}")
    if not manifest.get("host_codegen_identity"):
        raise HostFaultsError("manifest host_codegen_identity is missing")
    if any(str(name).startswith("libLLVM") for name in
           manifest.get("dependencies", [])):
        raise HostFaultsError("manifest records an LLVM dependency")
    run_ops = manifest["host_program"]["run_operations"]
    dispatches = sum(1 for op in run_ops if op.get("op") == "dispatch")
    copies = sum(1 for op in run_ops if op.get("op") == "copy")
    if dispatches != DISPATCHES_PER_RUN or copies != COPIES_PER_RUN:
        raise HostFaultsError(
            f"fixture host program changed: dispatches={dispatches} "
            f"copies={copies} (expected {DISPATCHES_PER_RUN}/{COPIES_PER_RUN})")
    if len(manifest["host_program"]["outputs"]) != 2:
        raise HostFaultsError("fixture must expose exactly two public outputs")
    if len(manifest["host_program"]["weights"]) != 2:
        raise HostFaultsError("fixture must embed exactly two weight resources")
    return {"host_contract": dict(HOST_CONTRACT),
            "host_codegen_identity": manifest["host_codegen_identity"],
            "host_program_identity": manifest["host_program_identity"],
            "run_dispatches": dispatches, "run_copies": copies}


def wrapper_capture(capture_root: Path) -> Path:
    staged = capture_root / "staged"
    wrapper = staged / WRAPPER_SOURCE_NAME
    if not wrapper.is_file():
        raise HostFaultsError(
            f"host-link capture did not stage the wrapper source: {wrapper}")
    if not (staged / HEADER_NAME).is_file():
        raise HostFaultsError(
            f"host-link capture did not stage the model header: {HEADER_NAME}")
    return wrapper


def check_wrapper_identity(wrapper: Path, manifest_identity: str) -> dict:
    text = wrapper.read_text(encoding="utf-8")
    match = re.search(r"^// Codegen identity: ([0-9a-f]{64})$", text, re.M)
    if not match:
        raise HostFaultsError("wrapper source lacks the codegen identity header")
    if match.group(1) != manifest_identity:
        raise HostFaultsError(
            "wrapper codegen identity differs from the published manifest: "
            f"{match.group(1)} vs {manifest_identity}")
    return {"wrapper_sha256": sha256(wrapper),
            "codegen_identity": match.group(1)}


def check_undefined_symbols(args, library: Path, env: dict) -> dict:
    result = invoke([args.nm, "-D", "--undefined-only", str(library)], env=env)
    offenders = []
    for line in result["stdout"].splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[-1]
        if any(marker.lower() in symbol.lower()
               for marker in FORBIDDEN_UNDEFINED_MARKERS):
            offenders.append(symbol)
    if offenders:
        raise HostFaultsError(
            f"published DSO references compiler/plan symbols: {offenders}")
    return {"undefined_symbols": len(result["stdout"].splitlines()),
            "forbidden_absent": True}


def compile_fault_test(args, wrapper: Path, staged_dir: Path,
                       env: dict) -> tuple[Path, dict]:
    executable = args.work_dir / "va6-host-faults"
    compile_record = invoke([
        args.host_cxx, "-std=c++23", "-O1", "-Wall", "-Wextra", "-Wpedantic",
        *args.cxx_arg,
        f"-DNCNN_VA6_WRAPPER_PATH=\"{wrapper}\"",
        "-DNCNN_VULKAN_LIFECYCLE_TEST=1",
        "-I", str(args.native_dir),
        "-I", str(args.project_include),
        "-I", str(staged_dir),
        str(args.fault_test_source),
        str(args.fake_vulkan_source),
        str(args.helper_archive),
        "-lpthread", "-ldl",
        *args.wrap_options,
        "-o", str(executable),
    ], env=env)
    if compile_record["stderr"].strip():
        raise HostFaultsError(
            "fault-test compilation produced diagnostics:\n"
            + compile_record["stderr"])
    return executable, compile_record


def run_fault_test(executable: Path, env: dict) -> dict:
    result = subprocess.run([str(executable)], capture_output=True, text=True,
                            env=env)
    record = {"argv": [str(executable)], "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        raise HostFaultsError(
            f"fault-chain binary failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}")
    if "passed" not in result.stdout:
        raise HostFaultsError(
            f"fault-chain binary did not report success:\n{result.stdout}")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--host-cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--param", type=Path, required=True)
    parser.add_argument("--conv-bin", type=Path, required=True)
    parser.add_argument("--fault-test-source", type=Path, required=True)
    parser.add_argument("--fake-vulkan-source", type=Path, required=True)
    parser.add_argument("--helper-archive", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--project-include", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    env = compiler_environment(args)

    wrap_functions = (
        "vkCreateInstance", "vkEnumerateInstanceLayerProperties",
        "vkEnumerateInstanceExtensionProperties", "vkEnumeratePhysicalDevices",
        "vkGetPhysicalDeviceProperties", "vkGetPhysicalDeviceMemoryProperties",
        "vkGetPhysicalDeviceQueueFamilyProperties", "vkCreateDevice",
        "vkDestroyDevice", "vkGetDeviceQueue", "vkDeviceWaitIdle",
        "vkDestroyInstance", "vkCreateBuffer",
        "vkGetBufferMemoryRequirements", "vkAllocateMemory",
        "vkBindBufferMemory", "vkMapMemory", "vkUnmapMemory",
        "vkFlushMappedMemoryRanges", "vkInvalidateMappedMemoryRanges",
        "vkDestroyBuffer", "vkFreeMemory", "vkCreateCommandPool",
        "vkAllocateCommandBuffers", "vkCreateFence", "vkResetFences",
        "vkResetCommandPool", "vkBeginCommandBuffer", "vkEndCommandBuffer",
        "vkQueueSubmit", "vkWaitForFences", "vkDestroyFence",
        "vkDestroyCommandPool", "vkCreateDescriptorSetLayout",
        "vkCreateDescriptorPool", "vkAllocateDescriptorSets",
        "vkUpdateDescriptorSets", "vkDestroyDescriptorPool",
        "vkDestroyDescriptorSetLayout", "vkCreateShaderModule",
        "vkCreatePipelineLayout", "vkCreateComputePipelines",
        "vkDestroyPipeline", "vkDestroyPipelineLayout", "vkDestroyShaderModule",
        "vkGetInstanceProcAddr",
        "vkCmdDispatch", "vkCmdBindPipeline", "vkCmdBindDescriptorSets",
        "vkCmdPushConstants", "vkCmdCopyBuffer", "vkCmdPipelineBarrier",
    )
    args.wrap_options = [
        option for function in wrap_functions
        for option in (f"-Wl,--wrap={function}",)]

    capture_root = args.work_dir / "link-evidence"
    launcher = dependency_evidence.make_host_cxx_launcher(
        args.host_cxx, capture_root)
    published = args.work_dir / "published"
    compile_product(args, published, env, launcher)
    manifest_record = check_manifest(published)
    wrapper = wrapper_capture(capture_root)
    wrapper_record = check_wrapper_identity(wrapper,
                                            manifest_record["host_codegen_identity"])
    dependency_record = check_undefined_symbols(
        args, published / f"lib{MODEL}_vulkan.so", env)

    repeat_root = args.work_dir / "repeat-link-evidence"
    repeat_launcher = dependency_evidence.make_host_cxx_launcher(
        args.host_cxx, repeat_root)
    repeat_dir = args.work_dir / "repeat-published"
    compile_product(args, repeat_dir, env, repeat_launcher)
    repeat_wrapper = wrapper_capture(repeat_root)
    if sha256(repeat_wrapper) != wrapper_record["wrapper_sha256"]:
        raise HostFaultsError("repeat compile generated a different wrapper")

    executable, fault_compile = compile_fault_test(
        args, wrapper, capture_root / "staged", env)
    fault_record = run_fault_test(executable, env)

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "model": MODEL,
        "host_contract": manifest_record["host_contract"],
        "host_codegen_identity": manifest_record["host_codegen_identity"],
        "host_program_identity": manifest_record["host_program_identity"],
        "run_dispatches": manifest_record["run_dispatches"],
        "run_copies": manifest_record["run_copies"],
        "wrapper": wrapper_record,
        "product_dependency_audit": dependency_record,
        "published_sha256": {
            name: sha256(published / name) for name in sorted(
                entry.name for entry in published.iterdir())},
        "fault_test": {
            "source": str(args.fault_test_source),
            "compile_argv": fault_compile["argv"],
            "returncode": fault_record["returncode"],
            "stdout": fault_record["stdout"],
            "stderr": fault_record["stderr"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VA6 host faults evidence: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
