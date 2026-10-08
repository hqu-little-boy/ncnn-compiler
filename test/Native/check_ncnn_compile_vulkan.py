#!/usr/bin/env python3
"""Exercise the ncnn-compile Vulkan product backend end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import shlex
import struct
import subprocess
import sys


def invoke(command, *, env=None, expect_success=True, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd)
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"command unexpectedly succeeded: {command}")
    return result


def assert_no_published_artifacts(path):
    if path.exists():
        artifacts = list(path.iterdir())
        if artifacts:
            raise RuntimeError(f"failed compile published artifacts: {artifacts}")


def make_unsupported_sigmoid_fixture(directory):
    param = directory / "unsupported-sigmoid.param"
    weights = directory / "unsupported-sigmoid.bin"
    param.write_text(
        "7767517\n"
        "2 2\n"
        "Input input 0 1 data 0=4 1=4 2=1\n"
        "Sigmoid unsupported_sigmoid 1 1 data output\n"
    )
    weights.write_bytes(b"")
    return param, weights


def output_snapshot(path):
    return {
        entry.name: entry.read_bytes()
        for entry in sorted(path.iterdir())
        if entry.is_file()
    }


def required_files(path, expected):
    actual = {entry.name for entry in path.iterdir() if entry.is_file()}
    if actual != set(expected):
        raise RuntimeError(
            f"unexpected output files in {path}: actual={sorted(actual)} "
            f"expected={sorted(expected)}"
        )


def check_exports_and_dependencies(args, library_path, manifest):
    symbols = invoke(
        [args.nm, "-D", "--defined-only", str(library_path)]
    ).stdout
    exports = sorted(
        line.split()[-1] for line in symbols.splitlines() if line.split()
    )
    expected_exports = sorted(
        [
            "paired_relu_vk_create",
            "paired_relu_vk_run",
            "paired_relu_vk_destroy",
        ]
    )
    if exports != expected_exports:
        raise RuntimeError(
            f"unexpected Vulkan API exports: actual={exports} "
            f"expected={expected_exports}"
        )

    needed = invoke(
        [args.readelf, "--needed-libs", str(library_path)]
    ).stdout
    dependencies = sorted(
        set(re.findall(r"^\s*(lib[^\s]+\.so(?:\.\d+)*)\s*$", needed, re.MULTILINE))
    )
    if dependencies != sorted(manifest["dependencies"]):
        raise RuntimeError(
            f"manifest dependencies differ from ELF DT_NEEDED: "
            f"actual={dependencies} manifest={manifest['dependencies']}"
        )
    allowed = {
        "libvulkan.so.1",
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libc.so.6",
        *args.allowed_dependency,
    }
    llvm_prefix = "libLLVM.so." + args.llvm_major
    llvm_versioned_prefix = "libLLVM-" + args.llvm_major + ".so."

    def is_llvm(name):
        return name.startswith("libLLVM")

    def is_expected_llvm(name):
        return (
            name == llvm_prefix
            or name.startswith(llvm_prefix + ".")
            or name.startswith(llvm_versioned_prefix)
        )

    llvm_dependencies = [name for name in dependencies if is_llvm(name)]
    if args.llvm_shared:
        if not any(is_expected_llvm(name) for name in llvm_dependencies):
            raise RuntimeError(
                f"shared LLVM dependency is missing or has the wrong version: "
                f"{dependencies}"
            )
    elif llvm_dependencies:
        raise RuntimeError(
            f"static LLVM product unexpectedly needs libLLVM: {dependencies}"
        )

    if any(
        name not in allowed and not (args.llvm_shared and is_expected_llvm(name))
        for name in dependencies
    ):
        raise RuntimeError(f"unexpected Vulkan DSO dependencies: {dependencies}")
    missing_sanitizers = set(args.sanitizer_dependency) - set(dependencies)
    if missing_sanitizers:
        raise RuntimeError(
            "Vulkan DSO is missing configured sanitizer dependencies: "
            f"{sorted(missing_sanitizers)}; actual={dependencies}"
        )
    dynamic = invoke([args.readelf, "-d", str(library_path)]).stdout
    if "RPATH" in dynamic or "RUNPATH" in dynamic:
        raise RuntimeError("Vulkan DSO embeds an RPATH/RUNPATH build path")


def make_input(elements):
    generator = random.Random(42)
    values = [generator.uniform(-3.0, 3.0) for _ in range(elements)]
    values[0] = -1.0
    values[1] = 0.0
    values[2] = 1.5
    packed = struct.pack(f"<{elements}f", *values)
    return list(struct.unpack(f"<{elements}f", packed))


def check_cpu_artifact(args, directory):
    required_files(directory, {"paired_relu.h", "libpaired_relu.so"})
    needed = invoke(
        [args.readelf, "--needed-libs", str(directory / "libpaired_relu.so")]
    ).stdout
    dependencies = set(
        re.findall(r"^\s*(lib[^\s]+\.so(?:\.\d+)*)\s*$", needed, re.MULTILINE)
    )
    if any("libvulkan" in name or "libLLVM" in name for name in dependencies):
        raise RuntimeError(
            "default CPU artifact acquired a Vulkan/LLVM runtime dependency"
        )

    sanitizer_flags = [flag for flag in args.cxx_arg if "-fsanitize=" in flag]
    sanitizer_dependencies = set(args.sanitizer_dependency)
    if sanitizer_flags or sanitizer_dependencies:
        if not args.sanitizer_runtime_dir:
            raise RuntimeError(
                "sanitized CPU artifact has no configured sanitizer runtime dir"
            )
        if not sanitizer_dependencies:
            raise RuntimeError(
                "sanitized CPU artifact has no configured sanitizer SONAMEs"
            )
        missing_sanitizers = sanitizer_dependencies - dependencies
        if missing_sanitizers:
            raise RuntimeError(
                "CPU DSO is missing configured sanitizer dependencies: "
                f"{sorted(missing_sanitizers)}; actual={sorted(dependencies)}"
            )


def check_vulkan_artifact(args, directory, elements):
    header_name = "paired_relu_vulkan.h"
    library_path = directory / "libpaired_relu_vulkan.so"
    required_files(
        directory,
        {
            header_name,
            "paired_relu_vulkan.json",
            "paired_relu_vulkan.plan.json",
            library_path.name,
        },
    )
    manifest = json.loads((directory / "paired_relu_vulkan.json").read_text())
    plan_bytes = (directory / "paired_relu_vulkan.plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    if manifest.get("schema_version") != 1 or manifest.get(
        "kind"
    ) != "ncnn.vulkan_model_manifest":
        raise RuntimeError(f"unexpected Vulkan manifest schema: {manifest}")
    if manifest.get("plan_hash") != plan.get("plan_hash"):
        raise RuntimeError("Vulkan manifest plan_hash differs from its plan file")
    if manifest.get("plan_sha256") != hashlib.sha256(plan_bytes).hexdigest():
        raise RuntimeError("Vulkan manifest plan_sha256 differs from its plan file")
    if manifest.get("model_param_sha256") != hashlib.sha256(
        (args.work_dir / "model.param").read_bytes()
    ).hexdigest():
        raise RuntimeError("Vulkan manifest .param identity is incorrect")
    if manifest.get("model_bin_sha256") != hashlib.sha256(
        (args.work_dir / "model.bin").read_bytes()
    ).hexdigest():
        raise RuntimeError("Vulkan manifest .bin identity is incorrect")
    if len(manifest["inputs"]) != 1 or len(manifest["outputs"]) != 1:
        raise RuntimeError("Vulkan fixture must expose exactly one input and output")
    if any(
        port.get("element_type") != "f32"
        for port in (*manifest["inputs"], *manifest["outputs"])
    ):
        raise RuntimeError("Vulkan fixture ports must use FP32 elements")
    if manifest["inputs"][0]["shape"] != [1, 1, elements]:
        raise RuntimeError("Vulkan manifest input shape differs from the fixture")
    if manifest["outputs"][0]["shape"] != [1, 1, elements]:
        raise RuntimeError("Vulkan manifest output shape differs from the fixture")
    check_exports_and_dependencies(args, library_path, manifest)
    return manifest


def sanitizer_diagnostics(output):
    return re.search(
        r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:",
        output,
        re.IGNORECASE,
    )


def run_product_loader(args, cpu_dir, vulkan_dir, model_name, input_data):
    loader = args.work_dir / "vulkan-product-loader"
    cpu_output_path = args.work_dir / "cpu-output.f32"
    vulkan_output_path = args.work_dir / "vulkan-output.f32"
    input_path = args.work_dir / "input.f32"
    result_path = args.work_dir / "loader-result.json"
    elements = len(input_data)
    input_path.write_bytes(struct.pack(f"<{elements}f", *input_data))

    compile_command = [
        args.host_cxx,
        "-std=c++20",
        *args.cxx_arg,
        f"-DNCNN_PRODUCT_TEST_MODEL={model_name}",
        "-I",
        str(cpu_dir),
        "-I",
        str(vulkan_dir),
        "-I",
        str(args.vulkan_include_dir),
        str(args.loader_source),
        "-ldl",
        "-o",
        str(loader),
    ]
    invoke(compile_command, env=runtime_env(args))
    command = [
        str(loader),
        model_name,
        str(cpu_dir / f"lib{model_name}.so"),
        str(vulkan_dir / f"lib{model_name}_vulkan.so"),
        str(elements),
        str(input_path),
        str(cpu_output_path),
        str(vulkan_output_path),
        str(result_path),
    ]
    child = subprocess.run(
        command, capture_output=True, text=True, env=runtime_env(args)
    )
    result = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"product loader wrote invalid JSON: {error}\n"
                f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
            ) from error
    logs = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    if any("-fsanitize=" in flag for flag in args.cxx_arg):
        diagnostic = sanitizer_diagnostics(child.stdout + "\n" + child.stderr)
        if diagnostic:
            raise RuntimeError(
                "sanitizer diagnostic from product loader despite its exit status "
                f"{child.returncode}: {diagnostic.group(0)}\n{logs}\n"
                f"result: {result}"
            )
    if child.returncode != 0:
        raise RuntimeError(
            f"product loader failed ({child.returncode}); result={result}\n{logs}"
        )
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError(f"product loader reported failure: {result}\n{logs}")

    outputs = []
    for output_path, label in (
        (cpu_output_path, "CPU"),
        (vulkan_output_path, "Vulkan"),
    ):
        if not output_path.is_file():
            raise RuntimeError(
                f"product loader did not write {label} output; {result}\n{logs}"
            )
        output_bytes = output_path.read_bytes()
        if len(output_bytes) != elements * 4:
            raise RuntimeError(
                f"product loader wrote {len(output_bytes)} bytes for {label} output, "
                f"expected {elements * 4}"
            )
        outputs.append(list(struct.unpack(f"<{elements}f", output_bytes)))
    return outputs


def runtime_env(args):
    env = os.environ.copy()
    library_dirs = []
    if args.sanitizer_runtime_dir:
        library_dirs.append(str(args.sanitizer_runtime_dir))
    if args.llvm_shared:
        if args.llvm_lib_dir is None:
            raise RuntimeError("shared LLVM mode requires --llvm-lib-dir")
        library_dirs.append(str(args.llvm_lib_dir))
    inherited = env.get("LD_LIBRARY_PATH", "")
    if inherited:
        library_dirs.append(inherited)
    if library_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(library_dirs)
    return env


def reconstruct_shaders(args, directory):
    output = directory / "shader-reconstruction"
    output.mkdir()
    imported = output / "imported.mlir"
    tosa = output / "strict-tosa.mlir"
    device_ir = output / "device.mlir"
    binary_ir = output / "binary.mlir"
    plan_path = output / "plan.json"
    weights_path = output / "weights.json"
    invoke(
        [
            args.driver,
            "--bin",
            str(args.work_dir / "model.bin"),
            "--precision=f32",
            "--emit=mlir",
            str(args.work_dir / "model.param"),
            "-o",
            str(imported),
        ]
    )
    invoke(
        [
            args.opt,
            "--ncnn-to-tosa-pipeline",
            "--mlir-print-debuginfo",
            str(imported),
            "-o",
            str(tosa),
        ]
    )
    invoke(
        [
            args.opt,
            str(tosa),
            f"--lower-ncnn-vulkan-graph=plan-path={plan_path} weights-path={weights_path}",
            "--verify-each",
            "-o",
            str(device_ir),
        ]
    )
    plan = json.loads(plan_path.read_text())
    weights = json.loads(weights_path.read_text())
    if (
        weights.get("kind") != "ncnn.vulkan_weight_payloads"
        or weights.get("schema_version") != 1
    ):
        raise RuntimeError(f"unexpected Vulkan weight payload schema: {weights}")
    expected_weight_ids = {
        resource["id"] for resource in plan["resources"]
        if resource.get("kind") == "weight"
    }
    actual_weight_ids = set()
    for weight in weights.get("weights", []):
        payload = weight.get("bytes")
        if (
            not isinstance(payload, str)
            or not payload
            or len(payload) % 2 != 0
            or re.fullmatch(r"[0-9a-f]*", payload) is None
        ):
            raise RuntimeError(f"weight payload is not lower-case hex: {weight}")
        actual_weight_ids.add(weight["resource_id"])
    if actual_weight_ids != expected_weight_ids:
        raise RuntimeError(
            f"weight payload resource IDs differ from plan: actual={actual_weight_ids} "
            f"expected={expected_weight_ids}"
        )
    expected_symbols = sorted(
        f"model_vulkan_k{kernel['id']}" for kernel in plan["kernels"]
    )
    invoke(
        [
            args.mlir_opt,
            str(device_ir),
            "--test-vulkan-runner-pipeline",
            "--verify-each",
            "-o",
            str(binary_ir),
        ]
    )
    binary_text = binary_ir.read_text()
    actual_symbols = sorted(
        re.findall(r"gpu\.binary\s+@([A-Za-z_][A-Za-z0-9_.$-]*)", binary_text)
    )
    if actual_symbols != expected_symbols:
        raise RuntimeError(
            f"GPU binary symbols differ from the plan: actual={actual_symbols} "
            f"expected={expected_symbols}"
        )

    shader_hashes = {}
    for kernel in plan["kernels"]:
        symbol = f"model_vulkan_k{kernel['id']}"
        shader = output / f"kernel-{kernel['id']}.spv"
        invoke(
            [
                args.spirv_extract,
                "--binary-symbol",
                symbol,
                str(binary_ir),
                "-o",
                str(shader),
            ]
        )
        invoke([args.spirv_val, "--target-env", "vulkan1.1", str(shader)])
        shader_hashes[kernel["id"]] = hashlib.sha256(shader.read_bytes()).hexdigest()
    return shader_hashes


def compile_args(args, model_name, output_dir, param, binary, backend=True):
    command = [
        args.compiler,
        str(param),
        "--bin",
        str(binary),
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
    ]
    if backend:
        command.extend(["--backend=vulkan", "--precision=f32", "--verify-execution"])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--spirv-val", required=True)
    parser.add_argument("--spirv-dis", required=True)
    parser.add_argument("--spirv-extract", required=True)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--host-cxx", required=True)
    parser.add_argument("--vulkan-include-dir", type=Path, required=True)
    parser.add_argument("--loader-source", type=Path, required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    parser.add_argument("--llvm-shared", action="store_true")
    parser.add_argument("--allowed-dependency", action="append", default=[])
    parser.add_argument("--sanitizer-dependency", action="append", default=[])
    parser.add_argument("--llvm-major", required=True)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--param", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--unsupported-param", type=Path, required=True)
    parser.add_argument("--unsupported-bin", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    args.work_dir = args.work_dir.resolve()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    shutil.copy2(args.param, args.work_dir / "model.param")
    shutil.copy2(args.bin, args.work_dir / "model.bin")
    base_env = runtime_env(args)

    elements = 257
    input_data = make_input(elements)
    expected = [value if value > 0.0 else 0.0 for value in input_data]
    cpu_dir = args.work_dir / "cpu"
    cpu_command = [
        args.compiler,
        str(args.work_dir / "model.param"),
        "--bin",
        str(args.work_dir / "model.bin"),
        "--model-name",
        "paired_relu",
        "--output-dir",
        str(cpu_dir),
        "--precision=f32",
        "--threads=1",
        "--vector-width=0",
        "-O0",
        "--verify-execution",
    ]
    for flag in args.cxx_arg:
        cpu_command.extend([f"--clang-arg={flag}", f"--linker-arg={flag}"])
    invoke(cpu_command, env=base_env)
    check_cpu_artifact(args, cpu_dir)

    vulkan_dir = args.work_dir / "vulkan"
    invoke(
        compile_args(
            args,
            "paired_relu",
            vulkan_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        ),
        env=base_env,
    )
    manifest = check_vulkan_artifact(args, vulkan_dir, elements)
    cpu_output, vulkan_output = run_product_loader(
        args, cpu_dir, vulkan_dir, "paired_relu", input_data
    )
    for label, output in (("CPU", cpu_output), ("Vulkan", vulkan_output)):
        for index, reference in enumerate(expected):
            if abs(output[index] - reference) > 1.0e-6:
                raise RuntimeError(
                    f"{label} ReLU differs from scalar golden at element {index}"
                )
    cpu_vulkan = [abs(lhs - rhs) for lhs, rhs in zip(cpu_output, vulkan_output)]
    if any(error > 1.0e-6 for error in cpu_vulkan):
        raise RuntimeError("CPU and Vulkan product outputs differ")

    reconstructed_shader_hashes = reconstruct_shaders(args, args.work_dir)
    manifest_shader_hashes = {
        int(shader["kernel_id"]): shader["sha256"]
        for shader in manifest.get("shaders", [])
    }
    if reconstructed_shader_hashes != manifest_shader_hashes:
        raise RuntimeError(
            "manifest per-kernel shader hashes differ from regenerated SPIR-V"
        )
    if len(reconstructed_shader_hashes) == 1 and next(
        iter(reconstructed_shader_hashes.values())
    ) != manifest["shader_sha256"]:
        raise RuntimeError("manifest primary shader hash differs from regenerated SPIR-V")

    repeat_dir = args.work_dir / "vulkan-repeat"
    invoke(
        [
            args.compiler,
            str(args.work_dir / "model.param"),
            "--bin",
            str(args.work_dir / "model.bin"),
            "--model-name",
            "paired_relu",
            "--output-dir",
            str(repeat_dir),
            "--backend=vulkan",
            "--precision=f32",
        ],
        env=base_env,
    )
    required_files(vulkan_dir, {entry.name for entry in repeat_dir.iterdir()})
    for filename in sorted(entry.name for entry in repeat_dir.iterdir()):
        if (vulkan_dir / filename).read_bytes() != (repeat_dir / filename).read_bytes():
            raise RuntimeError(f"Vulkan artifact is not reproducible: {filename}")

    invoke(
        [
            args.compiler,
            str(args.work_dir / "model.param"),
            "--bin",
            str(args.work_dir / "model.bin"),
            "--model-name",
            "foo",
            "--backend=vulkan",
            "--precision=f32",
        ],
        env=base_env,
        cwd=args.work_dir,
    )
    invoke(
        [
            args.compiler,
            str(args.work_dir / "model.param"),
            "--bin",
            str(args.work_dir / "model.bin"),
            "--model-name",
            "foo_vulkan",
            "--precision=f32",
            "--threads=1",
            "--vector-width=0",
        ],
        env=base_env,
        cwd=args.work_dir,
    )
    if not (args.work_dir / "ncnn-vulkan-foo" / "libfoo_vulkan.so").is_file():
        raise RuntimeError("Vulkan default output directory is not backend-distinct")
    if not (args.work_dir / "foo_vulkan" / "libfoo_vulkan.so").is_file():
        raise RuntimeError("CPU default output directory changed unexpectedly")

    cpu_collision_dir = args.work_dir / "custom-cpu-vulkan-name-collision"
    invoke(
        [
            args.compiler,
            str(args.work_dir / "model.param"),
            "--bin",
            str(args.work_dir / "model.bin"),
            "--model-name",
            "foo_vulkan",
            "--output-dir",
            str(cpu_collision_dir),
            "--precision=f32",
            "--threads=1",
            "--vector-width=0",
        ],
        env=base_env,
    )
    cpu_collision_before = output_snapshot(cpu_collision_dir)
    result = invoke(
        compile_args(
            args,
            "foo",
            cpu_collision_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        ),
        env=base_env,
        expect_success=False,
    )
    if "belongs to CPU model 'foo_vulkan'" not in result.stderr:
        raise RuntimeError(
            f"Vulkan accepted a colliding CPU output directory: {result.stderr}"
        )
    if output_snapshot(cpu_collision_dir) != cpu_collision_before:
        raise RuntimeError("Vulkan collision modified CPU artifacts")

    vulkan_collision_dir = args.work_dir / "custom-vulkan-cpu-name-collision"
    invoke(
        compile_args(
            args,
            "foo",
            vulkan_collision_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        ),
        env=base_env,
    )
    vulkan_collision_before = output_snapshot(vulkan_collision_dir)
    result = invoke(
        [
            args.compiler,
            str(args.work_dir / "model.param"),
            "--bin",
            str(args.work_dir / "model.bin"),
            "--model-name",
            "foo_vulkan",
            "--output-dir",
            str(vulkan_collision_dir),
            "--precision=f32",
            "--threads=1",
            "--vector-width=0",
        ],
        env=base_env,
        expect_success=False,
    )
    if "belongs to Vulkan model 'foo'" not in result.stderr:
        raise RuntimeError(
            f"CPU accepted a colliding Vulkan output directory: {result.stderr}"
        )
    if output_snapshot(vulkan_collision_dir) != vulkan_collision_before:
        raise RuntimeError("CPU collision modified Vulkan artifacts")

    previous = output_snapshot(vulkan_dir)
    unsupported = args.work_dir / "unsupported"
    unsupported_param, unsupported_bin = make_unsupported_sigmoid_fixture(args.work_dir)
    result = invoke(
        compile_args(
            args,
            "unsupported",
            unsupported,
            unsupported_param,
            unsupported_bin,
        ),
        env=base_env,
        expect_success=False,
    )
    diagnostic = result.stderr.lower()
    if (
        "unsupported_sigmoid" not in diagnostic
        or not any(message in diagnostic for message in ("unsupported", "not supported"))
    ):
        raise RuntimeError(
            f"unsupported source-layer diagnostic is missing: {result.stderr}"
        )
    assert_no_published_artifacts(unsupported)

    for test_name, extra in (
        ("invalid-backend", ["--backend=unknown"]),
        ("cpu-threads", ["--backend=vulkan", "--threads=1"]),
        ("debug-info", ["--backend=vulkan", "--debug-info"]),
        ("low-precision", ["--backend=vulkan", "--precision=fp16"]),
        ("fp16-accumulator", ["--backend=vulkan", "--fp16-accumulator=f32"]),
        ("allow-fallback", ["--backend=vulkan", "--allow-fallback"]),
        ("optimization", ["--backend=vulkan", "-O0"]),
        ("vector-width", ["--backend=vulkan", "--vector-width=0"]),
        ("vector-mode", ["--backend=vulkan", "--vector-mode=off"]),
        ("int8-kernel", ["--backend=vulkan", "--int8-kernel=portable"]),
        ("int8-depthwise", ["--backend=vulkan", "--int8-depthwise"]),
        ("packed-conv-depthwise", ["--backend=vulkan", "--packed-conv-depthwise"]),
        ("int8-cast-chain", ["--backend=vulkan", "--int8-cast-chain"]),
        ("selective-fusion", ["--backend=vulkan", "--selective-fusion=true"]),
        ("layout-aware-fusion", ["--backend=vulkan", "--layout-aware-fusion=true"]),
        ("fusion-broadcast", ["--backend=vulkan", "--selective-fusion-broadcast=true"]),
        ("fusion-cast-chain", ["--backend=vulkan", "--selective-fusion-cast-chain=true"]),
        ("fusion-max-chain", ["--backend=vulkan", "--selective-fusion-max-chain=4"]),
        ("tuning-profile", ["--backend=vulkan", "--tuning-profile=stable"]),
        ("matmul-m-rows", ["--backend=vulkan", "--matmul-m-rows=2"]),
        ("matmul-acc-columns", ["--backend=vulkan", "--matmul-acc-columns=2"]),
        ("matmul-packing", ["--backend=vulkan", "--matmul-packing=off"]),
        ("row-chunk-lanes", ["--backend=vulkan", "--row-chunk-lanes=4"]),
        ("matmul-i8-rows", ["--backend=vulkan", "--matmul-i8-rows=1"]),
        ("matmul-i8-acc-columns", ["--backend=vulkan", "--matmul-i8-acc-columns=2"]),
        ("conv-strategy", ["--backend=vulkan", "--conv-strategy=conv"]),
        ("vector-math", ["--backend=vulkan", "--vector-math=none"]),
        ("sleef-path", ["--backend=vulkan", "--sleef-path=/tmp"]),
        ("conv-gemm-l2", ["--backend=vulkan", "--conv-gemm-l2-bytes=1024"]),
        ("dynamic-shape", ["--backend=vulkan", "--input-shape=1x?x?"]),
        ("dynamic-constraint", ["--backend=vulkan", "--input-dim-constraint=0:0:min=1,multiple=1"]),
        ("cpu-target", ["--backend=vulkan", "--target-triple=x86_64-pc-linux-gnu"]),
        ("cpu-march", ["--backend=vulkan", "--march=x86-64"]),
        ("cpu-mcpu", ["--backend=vulkan", "--mcpu=x86-64"]),
        ("cpu-mtune", ["--backend=vulkan", "--mtune=generic"]),
        ("cpu-target-feature", ["--backend=vulkan", "--target-feature=+sse2"]),
        ("cpu-sysroot", ["--backend=vulkan", "--sysroot=/tmp"]),
        ("clang-arg", ["--backend=vulkan", "--clang-arg=-ffast-math"]),
        ("linker-arg", ["--backend=vulkan", "--linker-arg=-lm"]),
        ("emit-stage", ["--backend=vulkan", "--emit=ncnn"]),
        ("emit-manifest", ["--backend=vulkan", "--emit-manifest"]),
        ("emit-plan", ["--backend=vulkan", "--emit-execution-plan"]),
        ("profile", ["--backend=vulkan", "--profile"]),
        ("expected-undefined", ["--backend=vulkan", "--expected-undefined=malloc"]),
    ):
        output_dir = args.work_dir / test_name
        result = invoke(
            [
                args.compiler,
                str(args.work_dir / "model.param"),
                "--bin",
                str(args.work_dir / "model.bin"),
                "--model-name",
                test_name,
                "--output-dir",
                str(output_dir),
                *extra,
            ],
            env=base_env,
            expect_success=False,
        )
        if not result.stderr:
            raise RuntimeError(f"{test_name} failure did not provide a diagnostic")
        assert_no_published_artifacts(output_dir)

    user_data_dir = args.work_dir / "user-data"
    user_data_dir.mkdir()
    user_data_file = user_data_dir / "keep.txt"
    user_data_file.write_text("user data\n")
    result = invoke(
        compile_args(
            args,
            "paired_relu",
            user_data_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        ),
        env=base_env,
        expect_success=False,
    )
    if "not owned by ncnn-compile" not in result.stderr:
        raise RuntimeError(f"unowned output was not rejected: {result.stderr}")
    if user_data_file.read_text() != "user data\n":
        raise RuntimeError("unowned output directory contents were modified")

    result = invoke(
        [
            args.compiler,
            str(args.work_dir / "model.param"),
            "--bin",
            str(args.work_dir / "model.bin"),
            "--model-name",
            "paired_relu",
            "--output-dir",
            str(vulkan_dir),
            "--precision=f32",
            "--threads=1",
        ],
        env=base_env,
        expect_success=False,
    )
    if "belongs to Vulkan model 'paired_relu'" not in result.stderr:
        raise RuntimeError(f"CPU/Vulkan output collision was not rejected: {result.stderr}")
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("CPU output collision modified existing Vulkan artifacts")

    missing_tool_output = args.work_dir / "missing-tool"
    invoke(
        compile_args(
            args,
            "missing_tool",
            missing_tool_output,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        )
        + ["--vulkan-spirv-val=/path/that/does/not/exist"],
        env=base_env,
        expect_success=False,
    )
    assert_no_published_artifacts(missing_tool_output)

    bad_validator = args.work_dir / "reject-spirv-val"
    bad_validator.write_text("#!/bin/sh\nexit 23\n")
    bad_validator.chmod(0o755)
    invoke(
        compile_args(
            args,
            "paired_relu",
            vulkan_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        )
        + ["--vulkan-spirv-val=" + str(bad_validator)],
        env=base_env,
        expect_success=False,
    )
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("SPIR-V validator failure replaced prior successful artifacts")

    fake_cxx = args.work_dir / "reject-host-link"
    fake_cxx.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-dumpmachine\" ]; then\n"
        f"  exec {shlex.quote(args.host_cxx)} -dumpmachine\n"
        "fi\n"
        "exit 29\n"
    )
    fake_cxx.chmod(0o755)
    result = invoke(
        compile_args(
            args,
            "paired_relu",
            vulkan_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        )
        + ["--vulkan-host-cxx=" + str(fake_cxx)],
        env=base_env,
        expect_success=False,
    )
    if "failed with exit code 29" not in result.stderr:
        raise RuntimeError(f"link failure did not surface clearly: {result.stderr}")
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("failed Vulkan relink replaced prior successful artifacts")

    fake_readelf = args.work_dir / "reject-readelf"
    fake_readelf.write_text("#!/bin/sh\nexit 31\n")
    fake_readelf.chmod(0o755)
    result = invoke(
        compile_args(
            args,
            "paired_relu",
            vulkan_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        )
        + ["--readelf=" + str(fake_readelf)],
        env=base_env,
        expect_success=False,
    )
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("failed Vulkan dependency audit replaced prior artifacts")

    lost_icd_env = base_env.copy()
    lost_icd_env["VK_ICD_FILENAMES"] = str(args.work_dir / "missing-icd.json")
    lost_device = args.work_dir / "missing-device"
    invoke(
        compile_args(
            args,
            "missing_device",
            lost_device,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        ),
        env=lost_icd_env,
        expect_success=False,
    )
    assert_no_published_artifacts(lost_device)
    invoke(
        compile_args(
            args,
            "paired_relu",
            vulkan_dir,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        ),
        env=lost_icd_env,
        expect_success=False,
    )
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("missing Vulkan device replaced prior successful artifacts")
    print("Vulkan product CLI, CPU pairing, reproducibility, audit, and failure tests passed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"check_ncnn_compile_vulkan.py: {error}", file=sys.stderr)
        raise SystemExit(1)
