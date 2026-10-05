#!/usr/bin/env python3
"""Run V5.1 multi-I/O Vulkan product fixtures against CPU and ncnn."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys


TOLERANCE = 1.0e-6
SANITIZER_DIAGNOSTIC = re.compile(
    r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:",
    re.IGNORECASE,
)


def invoke(command, *, env=None, cwd=None, sanitizer_enabled=False):
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd)
    if sanitizer_enabled:
        diagnostics = [
            line
            for line in (result.stdout + "\n" + result.stderr).splitlines()
            if SANITIZER_DIAGNOSTIC.search(line)
        ]
        if diagnostics:
            raise RuntimeError(
                "sanitizer diagnostics from command "
                f"{command}:\n" + "\n".join(diagnostics[:20])
            )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def write_floats(path, values):
    Path(path).write_bytes(struct.pack(f"<{len(values)}f", *values))


def read_floats(path):
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise RuntimeError(f"raw float file has invalid size: {path}")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def count(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def make_input(case, shape, seed):
    size = count(shape)
    if case == "zeros":
        return [0.0] * size
    if case == "negative":
        return [-float((index % 7) + 1) / 4.0 for index in range(size)]
    if case == "boundary":
        values = [-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0]
        return [values[index % len(values)] for index in range(size)]
    generator = random.Random(seed)
    return [generator.randint(-16, 16) / 8.0 for _ in range(size)]


def convolution_bin(weights, biases):
    return struct.pack("<I", 0) + struct.pack(
        f"<{len(weights) + len(biases)}f", *(weights + biases)
    )


def convolution_weight_oracle(
    name, outputs, inputs, weights, biases, kernel_height=1, kernel_width=1
):
    logical = struct.pack(f"<{len(weights)}f", *weights)
    physical_values = [
        weights[
            ((output * inputs + input_channel) * kernel_height + height)
            * kernel_width
            + width
        ]
        for output in range(outputs)
        for height in range(kernel_height)
        for width in range(kernel_width)
        for input_channel in range(inputs)
    ]
    physical = struct.pack(f"<{len(physical_values)}f", *physical_values)
    bias = struct.pack(f"<{len(biases)}f", *biases)
    return {
        "layer": name,
        "filter_logical_sha256": sha256_bytes(logical),
        "filter_physical_sha256": sha256_bytes(physical),
        "filter_size_bytes": len(logical),
        "bias_sha256": sha256_bytes(bias),
        "bias_size_bytes": len(bias),
    }


def layer_counts(layers):
    blobs = set()
    for layer in layers:
        blobs.update(layer["tops"])
    return len(layers), len(blobs)


def make_param(inputs, layers):
    full_layers = []
    for name, shape in inputs:
        channels, height, width = shape
        full_layers.append(
            f"Input input_{name} 0 1 {name} 0={width} 1={height} 2={channels}"
        )
    full_layers.extend(layer["line"] for layer in layers)
    _, blob_count = layer_counts(
        [{"tops": [name]} for name, _ in inputs]
        + [{"tops": layer["tops"]} for layer in layers]
    )
    return "7767517\n{} {}\n{}\n".format(
        len(full_layers), blob_count, "\n".join(full_layers)
    )


def fixture_definitions(directory):
    directory.mkdir(parents=True, exist_ok=True)
    empty_bin = directory / "empty.bin"
    empty_bin.write_bytes(b"")
    conv_filter = [1.0, 0.5]
    conv_bias = [0.0, 0.25]
    conv_bin = directory / "conv_weights.bin"
    conv_bin.write_bytes(convolution_bin(conv_filter, conv_bias))
    second_filter = [
        0.125, 0.0, -0.125, 0.25, 0.5, -0.25, 0.0, 0.125, 0.25,
        0.0, -0.125, 0.125, 0.25, -0.25, 0.375, 0.125, 0.0, -0.125,
        -0.125, 0.25, 0.0, 0.125, -0.375, 0.5, 0.25, -0.25, 0.125,
        0.375, 0.0, -0.125, -0.25, 0.125, 0.25, 0.0, -0.375, 0.5,
    ]
    second_bias = [0.125, -0.25]
    conv_second_bin = directory / "double_conv_weights.bin"
    conv_second_bin.write_bytes(
        conv_bin.read_bytes() + convolution_bin(second_filter, second_bias)
    )
    branch_bin = directory / "split_concat_weights.bin"
    branch_bin.write_bytes(
        convolution_bin([1.0, 0.0], [0.0])
        + convolution_bin([0.0, 1.0], [0.0])
    )

    conv_layers = [
        {
            "line": "Convolution conv 1 1 data conv_value "
            "0=2 1=1 2=1 3=1 4=0 5=1 6=2",
            "tops": ["conv_value"],
        },
        {
            "line": "Split conv_split 1 2 conv_value conv_for_relu conv_observed",
            "tops": ["conv_for_relu", "conv_observed"],
        },
        {"line": "ReLU relu 1 1 conv_for_relu final", "tops": ["final"]},
    ]
    double_layers = [
        {
            "line": "Convolution conv_first 1 1 data first_value "
            "0=2 1=1 2=1 3=1 4=0 5=1 6=2",
            "tops": ["first_value"],
        },
        {
            "line": "Split first_split 1 2 first_value first_for_second first_observed",
            "tops": ["first_for_second", "first_observed"],
        },
        {
            "line": "Convolution conv_second 1 1 first_for_second final "
            "0=2 1=3 11=3 2=1 3=1 4=1 5=1 6=36",
            "tops": ["final"],
        },
    ]
    split_concat_layers = [
        {
            "line": "Split split 1 3 data branch_a branch_b branch_observed",
            "tops": ["branch_a", "branch_b", "branch_observed"],
        },
        {
            "line": "Convolution branch_conv_a 1 1 branch_a branch_a_out "
            "0=1 1=1 2=1 3=1 4=0 5=1 6=2",
            "tops": ["branch_a_out"],
        },
        {
            "line": "Convolution branch_conv_b 1 1 branch_b branch_b_out "
            "0=1 1=1 2=1 3=1 4=0 5=1 6=2",
            "tops": ["branch_b_out"],
        },
        {
            "line": "Concat join 2 1 branch_a_out branch_b_out joined 0=0",
            "tops": ["joined"],
        },
    ]
    add_relu_layers = [
        {"line": "BinaryOp add 2 1 left right sum 0=0", "tops": ["sum"]},
        {
            "line": "Split sum_split 1 2 sum sum_for_relu sum_observed",
            "tops": ["sum_for_relu", "sum_observed"],
        },
        {"line": "ReLU relu 1 1 sum_for_relu final", "tops": ["final"]},
    ]
    alias_layers = [
        {"line": "BinaryOp add_same 2 1 data data doubled 0=0", "tops": ["doubled"]},
        {
            "line": "Split output_split 1 2 doubled first_copy second_copy",
            "tops": ["first_copy", "second_copy"],
        },
    ]
    copy_layers = [{"line": "Dropout identity 1 1 data output 0=1.0", "tops": ["output"]}]

    fixtures = {
        "conv_relu": {
            "inputs": [("data", (1, 2, 2))],
            "outputs": [("conv_observed", (2, 2, 2)), ("final", (2, 2, 2))],
            "layers": conv_layers,
            "bin": conv_bin,
            "weight_oracles": [
                convolution_weight_oracle("conv", 2, 1, conv_filter, conv_bias)
            ],
            "copy_exact": False,
        },
        "double_conv": {
            "inputs": [("data", (1, 2, 2))],
            "outputs": [("first_observed", (2, 2, 2)), ("final", (2, 2, 2))],
            "layers": double_layers,
            "bin": conv_second_bin,
            "weight_oracles": [
                convolution_weight_oracle("conv_first", 2, 1, conv_filter, conv_bias),
                convolution_weight_oracle(
                    "conv_second", 2, 2, second_filter, second_bias,
                    kernel_height=3, kernel_width=3
                ),
            ],
            "copy_exact": False,
        },
        "split_concat": {
            "inputs": [("data", (2, 1, 2))],
            "outputs": [("branch_observed", (2, 1, 2)), ("joined", (2, 1, 2))],
            "layers": split_concat_layers,
            "bin": branch_bin,
            "weight_oracles": [
                convolution_weight_oracle("branch_conv_a", 1, 2, [1.0, 0.0], [0.0]),
                convolution_weight_oracle("branch_conv_b", 1, 2, [0.0, 1.0], [0.0]),
            ],
            "copy_exact": False,
        },
        "add_relu": {
            "inputs": [("left", (1, 1, 4)), ("right", (1, 1, 4))],
            "outputs": [("sum_observed", (1, 1, 4)), ("final", (1, 1, 4))],
            "layers": add_relu_layers,
            "bin": empty_bin,
            "copy_exact": False,
        },
        "repeated_input_alias": {
            "inputs": [("data", (1, 1, 4))],
            "outputs": [("first_copy", (1, 1, 4)), ("second_copy", (1, 1, 4))],
            "layers": alias_layers,
            "bin": empty_bin,
            "copy_exact": False,
        },
        "copy_identity": {
            "inputs": [("data", (1, 1, 8))],
            "outputs": [("output", (1, 1, 8))],
            "layers": copy_layers,
            "bin": empty_bin,
            "copy_exact": True,
        },
    }

    for name, fixture in fixtures.items():
        fixture.setdefault("weight_oracles", [])
        fixture["name"] = name
        fixture["model"] = "v5_" + name
        fixture["param"] = directory / f"{name}.param"
        fixture["param"].write_text(
            make_param(fixture["inputs"], fixture["layers"]), encoding="ascii"
        )
        fixture["bin_hash"] = sha256_file(fixture["bin"])
        fixture["param_hash"] = sha256_file(fixture["param"])
    return fixtures


def compile_model(args, fixture, backend, output_dir, env):
    output_dir.mkdir(parents=True)
    model_name = fixture["model"]
    command = [
        args.compiler,
        str(fixture["param"]),
        "--bin",
        str(fixture["bin"]),
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--driver",
        args.driver,
        "--opt",
        args.opt,
        "--translate",
        args.translate,
        "--clang",
        args.clang,
        "--nm",
        args.nm,
        "--readelf",
        args.readelf,
        "--llvm-as",
        args.llvm_as,
        "--verify-execution",
    ]
    if backend == "cpu":
        command.extend(
            ["--emit-manifest", "--precision=f32", "--threads=1", "--vector-width=0"]
        )
        for sanitizer_arg in args.cxx_arg:
            command.extend(["--clang-arg", sanitizer_arg])
            command.extend(["--linker-arg", sanitizer_arg])
    else:
        command.extend(
            [
                "--backend=vulkan",
                "--precision=f32",
                "--vulkan-mlir-opt",
                args.mlir_opt,
                "--vulkan-spirv-val",
                args.spirv_val,
                "--vulkan-spirv-dis",
                args.spirv_dis,
                "--vulkan-spirv-extract",
                args.spirv_extract,
                "--vulkan-host-cxx",
                args.host_cxx,
            ]
        )
    invoke(
        command,
        env=env,
        sanitizer_enabled=bool(args.sanitizer_dependency),
    )
    return output_dir


def load_manifest(model_dir, model, backend):
    suffix = "_vulkan" if backend == "vulkan" else ""
    path = model_dir / f"{model}{suffix}.json"
    if not path.is_file():
        raise RuntimeError(f"generated {backend} manifest is missing: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(args, fixture, cpu_dir, vk_dir, env):
    model = fixture["model"]
    cpu_header = cpu_dir / f"{model}.h"
    cpu_library = cpu_dir / f"lib{model}.so"
    vk_header = vk_dir / f"{model}_vulkan.h"
    vk_library = vk_dir / f"lib{model}_vulkan.so"
    vk_plan_path = vk_dir / f"{model}_vulkan.plan.json"
    for path in (cpu_header, cpu_library, vk_header, vk_library, vk_plan_path):
        if not path.is_file():
            raise RuntimeError(f"generated model artifact is missing: {path}")

    cpu_manifest_path, cpu_manifest = load_manifest(cpu_dir, model, "cpu")
    vk_manifest_path, vk_manifest = load_manifest(vk_dir, model, "vulkan")
    plan_bytes = vk_plan_path.read_bytes()
    plan = json.loads(plan_bytes)

    input_count = len(fixture["inputs"])
    output_count = len(fixture["outputs"])
    for label, manifest in (("CPU", cpu_manifest), ("Vulkan", vk_manifest)):
        if len(manifest.get("inputs", [])) != input_count:
            raise RuntimeError(f"{label} manifest input arity differs for {model}")
        if len(manifest.get("outputs", [])) != output_count:
            raise RuntimeError(f"{label} manifest output arity differs for {model}")
        for index, (_, shape) in enumerate(fixture["inputs"]):
            actual = manifest["inputs"][index].get("shape")
            if actual != list(shape):
                raise RuntimeError(f"{label} input {index} shape/order differs for {model}")
        for index, (_, shape) in enumerate(fixture["outputs"]):
            actual = manifest["outputs"][index].get("shape")
            if actual != list(shape):
                raise RuntimeError(f"{label} output {index} shape/order differs for {model}")

    if vk_manifest.get("schema_version") != 1 or vk_manifest.get(
        "kind"
    ) != "ncnn.vulkan_model_manifest":
        raise RuntimeError(f"unexpected Vulkan manifest: {vk_manifest}")
    if vk_manifest.get("plan_hash") != plan.get("plan_hash"):
        raise RuntimeError("Vulkan manifest plan_hash differs from plan JSON")
    if vk_manifest.get("plan_sha256") != sha256_bytes(plan_bytes):
        raise RuntimeError("Vulkan manifest plan_sha256 differs from plan bytes")
    if vk_manifest.get("model_param_sha256") != fixture["param_hash"]:
        raise RuntimeError("Vulkan manifest model-param identity is incorrect")
    if vk_manifest.get("model_bin_sha256") != fixture["bin_hash"]:
        raise RuntimeError("Vulkan manifest model-bin identity is incorrect")
    shaders = vk_manifest.get("shaders", [])
    kernels = plan.get("kernels", [])
    if len(shaders) != len(kernels) or not shaders:
        raise RuntimeError("Vulkan manifest shader entries differ from plan kernels")
    shader_hashes = []
    for kernel, shader in zip(kernels, shaders):
        digest = shader.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("Vulkan manifest contains an invalid shader digest")
        if shader.get("kernel_id") != kernel.get("id"):
            raise RuntimeError("Vulkan manifest shader kernel order differs from plan")
        if shader.get("entry_point") != kernel.get("entry_point"):
            raise RuntimeError("Vulkan manifest shader entry point differs from plan")
        if int(shader.get("size_bytes", 0)) <= 0:
            raise RuntimeError("Vulkan manifest shader byte size is invalid")
        shader_hashes.append(digest)
    if vk_manifest.get("shader_sha256s") != shader_hashes:
        raise RuntimeError("Vulkan manifest shader_sha256s differ from shader entries")
    if vk_manifest.get("shader_sha256") != shader_hashes[0]:
        raise RuntimeError("legacy Vulkan shader_sha256 is not the first kernel hash")
    if vk_manifest.get("shader_size_bytes") != shaders[0].get("size_bytes"):
        raise RuntimeError("legacy Vulkan shader size is not the first kernel size")
    if vk_manifest.get("binary_symbol") != shaders[0].get("binary_symbol"):
        raise RuntimeError("legacy Vulkan binary symbol is not the first kernel symbol")
    if not vk_manifest.get("build_identity"):
        raise RuntimeError("Vulkan manifest build identity is missing")

    resources = plan.get("resources", [])
    resources_by_id = {resource.get("id"): resource for resource in resources}
    conv_kernels = [kernel for kernel in kernels if kernel.get("operation") == "conv2d"]
    weight_oracles = fixture.get("weight_oracles", [])
    if len(conv_kernels) != len(weight_oracles):
        raise RuntimeError("Conv2D kernels differ from fixed fixture weight oracles")

    expected_manifest_weights = {}
    for kernel, oracle in zip(conv_kernels, weight_oracles):
        reads = kernel.get("reads", [])
        if len(reads) < 3:
            raise RuntimeError("Conv2D plan kernel does not read input/filter/bias")
        filter_id, bias_id = reads[1], reads[2]
        filter_resource = resources_by_id.get(filter_id)
        bias_resource = resources_by_id.get(bias_id)
        if filter_resource is None or bias_resource is None:
            raise RuntimeError("Conv2D reads reference missing filter/bias resources")
        if filter_resource.get("weight_sha256") != oracle["filter_logical_sha256"]:
            raise RuntimeError(
                f"{oracle['layer']} logical OIHW filter identity differs from fixture"
            )
        if bias_resource.get("weight_sha256") != oracle["bias_sha256"]:
            raise RuntimeError(
                f"{oracle['layer']} row-major bias identity differs from fixture"
            )
        if filter_resource.get("byte_size") != oracle["filter_size_bytes"]:
            raise RuntimeError("Conv2D logical filter byte size differs from fixture")
        if bias_resource.get("byte_size") != oracle["bias_size_bytes"]:
            raise RuntimeError("Conv2D bias byte size differs from fixture")
        expected_manifest_weights[filter_id] = {
            "sha256": oracle["filter_physical_sha256"],
            "size_bytes": oracle["filter_size_bytes"],
            "role": "physical-ohwi-filter",
        }
        expected_manifest_weights[bias_id] = {
            "sha256": oracle["bias_sha256"],
            "size_bytes": oracle["bias_size_bytes"],
            "role": "row-major-bias",
        }

    weights = vk_manifest.get("weights", [])
    weights_by_id = {weight.get("resource_id"): weight for weight in weights}
    if len(weights_by_id) != len(weights):
        raise RuntimeError("Vulkan manifest repeats a weight resource id")
    if set(weights_by_id) != set(expected_manifest_weights):
        raise RuntimeError("Vulkan manifest weight resources differ from Conv2D operands")
    filter_resource_ids = {kernel["reads"][1] for kernel in conv_kernels}
    bias_resource_ids = {kernel["reads"][2] for kernel in conv_kernels}
    if not filter_resource_ids.issubset(weights_by_id):
        raise RuntimeError("Vulkan manifest omits a physical Conv2D filter payload")
    if not bias_resource_ids.issubset(weights_by_id):
        raise RuntimeError("Vulkan manifest omits a Conv2D bias payload")
    for resource_id, weight in weights_by_id.items():
        expected = expected_manifest_weights[resource_id]
        digest = weight.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("Vulkan manifest contains an invalid weight digest")
        if digest != expected["sha256"]:
            raise RuntimeError(
                f"Vulkan physical weight identity differs for resource {resource_id}"
            )
        if weight.get("size_bytes") != expected["size_bytes"]:
            raise RuntimeError(
                f"Vulkan physical weight size differs for resource {resource_id}"
            )
    ordered_manifest_weights = [
        weights_by_id[resource["id"]]
        for resource in resources
        if resource.get("id") in weights_by_id
    ]
    if ordered_manifest_weights != weights:
        raise RuntimeError("Vulkan manifest weights are not in plan resource order")
    weight_hashes = [weight["sha256"] for weight in weights]
    if vk_manifest.get("weight_sha256s") != weight_hashes:
        raise RuntimeError("Vulkan manifest weight_sha256s differ from weight entries")
    if fixture["bin"].stat().st_size and not filter_resource_ids:
        raise RuntimeError("weighted fixture has no Conv2D filter resources")

    symbols = invoke([args.nm, "-D", "--defined-only", str(vk_library)]).stdout
    exports = sorted(line.split()[-1] for line in symbols.splitlines() if line.split())
    expected_exports = sorted(
        [f"{model}_vk_create", f"{model}_vk_run", f"{model}_vk_destroy"]
    )
    if exports != expected_exports:
        raise RuntimeError(f"unexpected model Vulkan exports: {exports}")

    def needed_dependencies(path):
        output = invoke([args.readelf, "--needed-libs", str(path)]).stdout
        return sorted(
            set(re.findall(r"^\s*(lib[^\s]+\.so(?:\.\d+)*)\s*$", output, re.MULTILINE))
        )

    actual_dependencies = needed_dependencies(vk_library)
    manifest_dependencies = sorted(set(vk_manifest.get("dependencies", [])))
    if actual_dependencies != manifest_dependencies:
        raise RuntimeError(
            "Vulkan manifest dependencies differ from ELF DT_NEEDED: "
            f"actual={actual_dependencies} manifest={manifest_dependencies}"
        )
    base_dependencies = {
        "libvulkan.so.1",
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libc.so.6",
    }
    allowed_dependencies = base_dependencies | set(args.allowed_dependency)
    llvm_dependencies = [
        name for name in actual_dependencies
        if name.startswith("libLLVM")
    ]
    if args.shared_llvm:
        llvm_prefixes = (
            f"libLLVM.so.{args.llvm_major}",
            f"libLLVM-{args.llvm_major}.so.",
        )
        if not llvm_dependencies or any(
            not any(name == prefix or name.startswith(prefix) for prefix in llvm_prefixes)
            for name in llvm_dependencies
        ):
            raise RuntimeError(
                f"shared LLVM dependency does not match configured LLVM {args.llvm_major}: "
                f"{llvm_dependencies}"
            )
        allowed_dependencies.update(llvm_dependencies)
    elif llvm_dependencies:
        raise RuntimeError(
            f"static LLVM product unexpectedly depends on shared LLVM: {llvm_dependencies}"
        )
    unexpected_dependencies = sorted(set(actual_dependencies) - allowed_dependencies)
    if unexpected_dependencies:
        raise RuntimeError(
            f"Vulkan DSO has unconfigured dependencies: {unexpected_dependencies}"
        )
    cpu_dependencies = needed_dependencies(cpu_library)
    if any("vulkan" in name.lower() or name.startswith("libLLVM")
           for name in cpu_dependencies):
        raise RuntimeError("CPU AOT model acquired Vulkan/LLVM runtime dependencies")
    if args.sanitizer_dependency:
        expected_sanitizers = set(args.sanitizer_dependency)
        if not expected_sanitizers:
            raise RuntimeError("sanitizer build has no configured sanitizer dependencies")
        if not expected_sanitizers.issubset(set(args.allowed_dependency)):
            raise RuntimeError("sanitizer dependencies are not in the configured allowlist")
        missing_vk_sanitizers = sorted(expected_sanitizers - set(actual_dependencies))
        if missing_vk_sanitizers:
            raise RuntimeError(
                "Vulkan DSO is missing configured sanitizer dependencies: "
                f"{missing_vk_sanitizers}"
            )
        missing_cpu_sanitizers = sorted(expected_sanitizers - set(cpu_dependencies))
        if missing_cpu_sanitizers:
            raise RuntimeError(
                "CPU DSO is missing configured sanitizer dependencies: "
                f"{missing_cpu_sanitizers}"
            )

    return {
        "cpu_header": cpu_header,
        "cpu_library": cpu_library,
        "cpu_manifest_path": cpu_manifest_path,
        "cpu_manifest": cpu_manifest,
        "vk_header": vk_header,
        "vk_library": vk_library,
        "vk_manifest_path": vk_manifest_path,
        "vk_manifest": vk_manifest,
        "plan_path": vk_plan_path,
        "plan": plan,
        "plan_sha256": sha256_bytes(plan_bytes),
        "shader_sha256": shader_hashes,
        "vulkan_dependencies": actual_dependencies,
        "cpu_dependencies": cpu_dependencies,
        "shaders": shaders,
        "weights": weights,
    }


def build_typed_runner(args, fixture, artifacts, work_dir, env):
    # Keep sanitizer runtimes first in an instrumented process: Python never dlopens
    # the product DSOs; this typed executable loads them after its own ASan startup.
    model = fixture["model"]
    macro = re.sub(r"[^A-Za-z0-9]", "_", model).upper()
    vulkan_macro = macro + "_VULKAN"
    input_count = len(fixture["inputs"])
    output_count = len(fixture["outputs"])
    cpu_args = ", ".join(
        [f"inputs[{index}].data()" for index in range(input_count)]
        + [f"cpu_outputs[{index}].data()" for index in range(output_count)]
    )
    vk_args = ["context"]
    for index in range(input_count):
        vk_args.extend([f"inputs[{index}].data()", f"inputs[{index}].size()"])
    for index in range(output_count):
        vk_args.extend([f"gpu_outputs[{index}].data()", f"gpu_outputs[{index}].size()"])
    vk_args.extend(["diagnostic.data()", "diagnostic.size()"])
    source_path = work_dir / f"{model}_typed_runner.cpp"
    executable_path = work_dir / f"{model}_typed_runner"
    cpu_header = artifacts["cpu_header"].name
    vk_header = artifacts["vk_header"].name
    input_counts = ", ".join(
        f"{vulkan_macro}_INPUT{index + 1}_ELEMENTS"
        for index in range(input_count)
    )
    output_counts = ", ".join(
        f"{vulkan_macro}_OUTPUT{index + 1}_ELEMENTS"
        for index in range(output_count)
    )
    source = r'''#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "@CPU_HEADER@"
#include "@VK_HEADER@"

static_assert(@MACRO@_INPUT_COUNT == @INPUT_COUNT@);
static_assert(@MACRO@_OUTPUT_COUNT == @OUTPUT_COUNT@);
static_assert(@VULKAN_MACRO@_INPUT_COUNT == @INPUT_COUNT@);
static_assert(@VULKAN_MACRO@_OUTPUT_COUNT == @OUTPUT_COUNT@);

namespace {
using CpuApi = decltype(&@MODEL@);
using VkCreateApi = decltype(&@MODEL@_vk_create);
using VkRunApi = decltype(&@MODEL@_vk_run);
using VkDestroyApi = decltype(&@MODEL@_vk_destroy);
using ModelContext = @MODEL@_vk_context;

constexpr std::size_t kInputCount = @INPUT_COUNT@;
constexpr std::size_t kOutputCount = @OUTPUT_COUNT@;
const std::array<std::size_t, kInputCount> kInputElements{@INPUT_ELEMENTS@};
const std::array<std::size_t, kOutputCount> kOutputElements{@OUTPUT_ELEMENTS@};

struct CallSpec {
  std::string label;
  std::size_t context_index = 0;
  bool recreate_before = false;
  std::vector<std::string> input_paths;
  std::vector<std::string> cpu_output_paths;
  std::vector<std::string> gpu_output_paths;
};

struct CallResult {
  std::string label;
  std::size_t context_index = 0;
  std::string diagnostic;
};

std::string jsonEscape(const std::string& text) {
  std::string result;
  for (unsigned char character : text) {
    if (character == '\\' || character == '"') {
      result.push_back('\\');
      result.push_back(static_cast<char>(character));
    } else if (character == '\n') {
      result += "\\n";
    } else if (character == '\r') {
      result += "\\r";
    } else if (character == '\t') {
      result += "\\t";
    } else if (character >= 0x20) {
      result.push_back(static_cast<char>(character));
    }
  }
  return result;
}

bool readFloats(const std::string& path,
                std::size_t expected,
                std::vector<float>& values,
                std::string& error) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    error = "cannot open input tensor: " + path;
    return false;
  }
  const std::streamoff byte_count = stream.tellg();
  if (byte_count < 0 ||
      static_cast<std::uint64_t>(byte_count) != expected * sizeof(float)) {
    error = "input tensor has the wrong f32 size: " + path;
    return false;
  }
  values.resize(expected);
  stream.seekg(0);
  if (expected != 0 &&
      !stream.read(reinterpret_cast<char*>(values.data()), byte_count)) {
    error = "cannot read input tensor: " + path;
    return false;
  }
  return true;
}

bool writeFloats(const std::string& path,
                 const std::vector<float>& values,
                 std::string& error) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  const auto byte_count = static_cast<std::streamsize>(values.size() * sizeof(float));
  if (!stream ||
      (byte_count != 0 &&
       !stream.write(reinterpret_cast<const char*>(values.data()), byte_count))) {
    error = "cannot write output tensor: " + path;
    return false;
  }
  return true;
}

template <typename Function>
Function loadSymbol(void* library, const char* name) {
  dlerror();
  return reinterpret_cast<Function>(dlsym(library, name));
}

bool createContext(VkCreateApi create,
                   std::uint32_t device_index,
                   ModelContext*& context,
                   std::string& error) {
  ncnn_vk_options options;
  ncnn_vk_options_init(&options);
  options.device_index = device_index;
  std::array<char, 1024> diagnostic{};
  const ncnn_vk_status status =
    create(&options, &context, diagnostic.data(), diagnostic.size());
  if (status != NCNN_VK_STATUS_SUCCESS || context == nullptr) {
    error = "Vulkan create failed (" + std::to_string(status) + "): " +
            diagnostic.data();
    return false;
  }
  return true;
}

bool destroyContext(VkDestroyApi destroy,
                    ModelContext* context,
                    std::string& error) {
  if (context == nullptr) {
    return true;
  }
  const ncnn_vk_status status = destroy(context);
  if (status != NCNN_VK_STATUS_SUCCESS) {
    error = "Vulkan destroy failed (" + std::to_string(status) + ")";
    return false;
  }
  return true;
}

bool executeCall(CpuApi cpu,
                 VkRunApi run,
                 ModelContext* context,
                 const CallSpec& call,
                 CallResult& result,
                 std::string& error) {
  std::vector<std::vector<float>> inputs(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    if (!readFloats(call.input_paths[index], kInputElements[index], inputs[index], error)) {
      return false;
    }
  }
  std::vector<std::vector<float>> cpu_outputs(kOutputCount);
  std::vector<std::vector<float>> gpu_outputs(kOutputCount);
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    cpu_outputs[index].resize(kOutputElements[index]);
    gpu_outputs[index].resize(kOutputElements[index]);
  }

  const int cpu_status = cpu(@CPU_ARGS@);
  if (cpu_status != 0) {
    error = "CPU model call failed (" + std::to_string(cpu_status) + ")";
    return false;
  }
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    if (!writeFloats(call.cpu_output_paths[index], cpu_outputs[index], error)) {
      return false;
    }
  }

  std::array<char, 1024> diagnostic{};
  const ncnn_vk_status vk_status = run(@VK_ARGS@);
  result.label = call.label;
  result.context_index = call.context_index;
  result.diagnostic = diagnostic.data();
  if (vk_status != NCNN_VK_STATUS_SUCCESS) {
    error = "Vulkan run failed (" + std::to_string(vk_status) + "): " +
            diagnostic.data();
    return false;
  }
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    if (!writeFloats(call.gpu_output_paths[index], gpu_outputs[index], error)) {
      return false;
    }
  }
  return true;
}

bool parseIndex(const char* text, std::size_t& value) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  std::string cpu_path;
  std::string vk_path;
  std::vector<CallSpec> calls;
  bool recreate_next = false;
  int index = 1;
  while (index < argc) {
    const std::string argument(argv[index++]);
    if (argument == "--cpu-library" && index < argc) {
      cpu_path = argv[index++];
    } else if (argument == "--vulkan-library" && index < argc) {
      vk_path = argv[index++];
    } else if (argument == "--recreate") {
      if (recreate_next) {
        std::cerr << "duplicate --recreate marker\n";
        return 2;
      }
      recreate_next = true;
    } else if (argument == "--call") {
      const std::size_t required = 2 + kInputCount + 2 * kOutputCount;
      if (static_cast<std::size_t>(argc - index) < required) {
        std::cerr << "incomplete --call record\n";
        return 2;
      }
      CallSpec call;
      call.label = argv[index++];
      if (!parseIndex(argv[index++], call.context_index)) {
        std::cerr << "invalid context index\n";
        return 2;
      }
      call.recreate_before = recreate_next;
      recreate_next = false;
      for (std::size_t input = 0; input < kInputCount; ++input) {
        call.input_paths.emplace_back(argv[index++]);
      }
      for (std::size_t output = 0; output < kOutputCount; ++output) {
        call.cpu_output_paths.emplace_back(argv[index++]);
      }
      for (std::size_t output = 0; output < kOutputCount; ++output) {
        call.gpu_output_paths.emplace_back(argv[index++]);
      }
      calls.push_back(std::move(call));
    } else {
      std::cerr << "unexpected argument: " << argument << '\n';
      return 2;
    }
  }
  if (cpu_path.empty() || vk_path.empty() || calls.empty() || recreate_next) {
    std::cerr << "missing model library or call records\n";
    return 2;
  }

  void* cpu_library = dlopen(cpu_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!cpu_library) {
    std::cerr << "cannot load CPU library: " << dlerror() << '\n';
    return 1;
  }
  void* vk_library = dlopen(vk_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!vk_library) {
    std::cerr << "cannot load Vulkan library: " << dlerror() << '\n';
    dlclose(cpu_library);
    return 1;
  }
  const CpuApi cpu = loadSymbol<CpuApi>(cpu_library, "@MODEL@");
  const VkCreateApi create = loadSymbol<VkCreateApi>(vk_library, "@MODEL@_vk_create");
  const VkRunApi run = loadSymbol<VkRunApi>(vk_library, "@MODEL@_vk_run");
  const VkDestroyApi destroy = loadSymbol<VkDestroyApi>(vk_library, "@MODEL@_vk_destroy");
  if (!cpu || !create || !run || !destroy) {
    std::cerr << "typed model symbol is missing\n";
    dlclose(vk_library);
    dlclose(cpu_library);
    return 1;
  }

  std::vector<ModelContext*> contexts;
  std::vector<CallResult> results;
  std::size_t contexts_created = 0;
  std::size_t contexts_destroyed = 0;
  auto cleanContexts = [&]() {
    bool success = true;
    std::string error;
    for (ModelContext* context : contexts) {
      if (context && destroyContext(destroy, context, error)) {
        ++contexts_destroyed;
      } else if (context) {
        std::cerr << error << '\n';
        success = false;
      }
    }
    contexts.clear();
    return success;
  };
  auto createContexts = [&](std::size_t amount) {
    for (std::size_t slot = 0; slot < amount; ++slot) {
      ModelContext* context = nullptr;
      std::string error;
      if (!createContext(create, NCNN_VK_DEVICE_AUTO, context, error)) {
        std::cerr << error << '\n';
        return false;
      }
      contexts.push_back(context);
      ++contexts_created;
    }
    return true;
  };

  for (const CallSpec& call : calls) {
    if (call.recreate_before) {
      if (contexts.empty() || !cleanContexts() || !createContexts(1)) {
        cleanContexts();
        dlclose(vk_library);
        dlclose(cpu_library);
        return 1;
      }
    } else if (contexts.empty() && !createContexts(2)) {
      cleanContexts();
      dlclose(vk_library);
      dlclose(cpu_library);
      return 1;
    }
    if (call.context_index >= contexts.size()) {
      std::cerr << "call references missing context slot\n";
      cleanContexts();
      dlclose(vk_library);
      dlclose(cpu_library);
      return 1;
    }
    CallResult result;
    std::string error;
    if (!executeCall(cpu, run, contexts[call.context_index], call, result, error)) {
      std::cerr << error << '\n';
      cleanContexts();
      dlclose(vk_library);
      dlclose(cpu_library);
      return 1;
    }
    results.push_back(std::move(result));
  }
  if (!cleanContexts()) {
    dlclose(vk_library);
    dlclose(cpu_library);
    return 1;
  }
  dlclose(vk_library);
  dlclose(cpu_library);

  std::cout << "{\"contexts_created\":" << contexts_created
            << ",\"contexts_destroyed\":" << contexts_destroyed
            << ",\"call_count\":" << results.size() << ",\"calls\":[";
  for (std::size_t call = 0; call < results.size(); ++call) {
    if (call != 0) std::cout << ',';
    std::cout << "{\"label\":\"" << jsonEscape(results[call].label)
              << "\",\"context_index\":" << results[call].context_index
              << ",\"diagnostic\":\"" << jsonEscape(results[call].diagnostic)
              << "\"}";
  }
  std::cout << "]}\n";
  return 0;
}
'''
    replacements = {
        "@CPU_HEADER@": cpu_header,
        "@VK_HEADER@": vk_header,
        "@MACRO@": macro,
        "@VULKAN_MACRO@": vulkan_macro,
        "@MODEL@": model,
        "@INPUT_COUNT@": str(input_count),
        "@OUTPUT_COUNT@": str(output_count),
        "@INPUT_ELEMENTS@": input_counts,
        "@OUTPUT_ELEMENTS@": output_counts,
        "@CPU_ARGS@": cpu_args,
        "@VK_ARGS@": ", ".join(vk_args),
    }
    for token, value in replacements.items():
        source = source.replace(token, value)
    source_path.write_text(source, encoding="ascii")
    invoke(
        [
            args.host_cxx,
            "-std=c++20",
            "-O1",
            "-I",
            args.project_include,
            "-I",
            str(artifacts["cpu_header"].parent),
            "-I",
            str(artifacts["vk_header"].parent),
            *args.cxx_arg,
            str(source_path),
            "-ldl",
            "-o",
            str(executable_path),
        ],
        env=env,
        sanitizer_enabled=bool(args.sanitizer_dependency),
    )
    return executable_path


def compare_outputs(fixture, reference, actual, label):
    if len(reference) != len(actual):
        raise RuntimeError(f"{label} output arity differs")
    result = []
    for output_index, (expected, observed) in enumerate(zip(reference, actual)):
        if len(expected) != len(observed):
            raise RuntimeError(f"{label} output {output_index} count differs")
        if any(not math.isfinite(float(value)) for value in expected):
            raise RuntimeError(
                f"{label} reference output {output_index} contains non-finite values"
            )
        if any(not math.isfinite(float(value)) for value in observed):
            raise RuntimeError(
                f"{label} output {output_index} contains non-finite values"
            )
        errors = [abs(float(lhs) - float(rhs)) for lhs, rhs in zip(expected, observed)]
        max_abs = max(errors, default=0.0)
        max_rel = max(
            (error / max(abs(float(lhs)), 1.0e-30)
             for error, lhs in zip(errors, expected)),
            default=0.0,
        )
        if fixture["copy_exact"]:
            equal = struct.pack(f"<{len(expected)}f", *expected) == struct.pack(
                f"<{len(observed)}f", *observed
            )
            if not equal:
                raise RuntimeError(f"{label} copy output {output_index} is not exact")
        elif any(
            error > TOLERANCE + TOLERANCE * abs(float(lhs))
            for error, lhs in zip(errors, expected)
        ):
            raise RuntimeError(
                f"{label} output {output_index} exceeds tolerance: max_abs={max_abs}"
            )
        result.append({"max_abs": max_abs, "max_rel": max_rel})
    return result


def run_reference(args, fixture, input_paths, output_paths, env=None):
    command = [args.runner, "--param", str(fixture["param"]), "--bin", str(fixture["bin"])]
    for (name, shape), path in zip(fixture["inputs"], input_paths):
        command.extend(["--input", name, str(path), *(str(dim) for dim in shape)])
    for (name, shape), path in zip(fixture["outputs"], output_paths):
        command.extend(["--output", name, str(path), *(str(dim) for dim in shape)])
    invoke(
        command,
        env=env,
        sanitizer_enabled=bool(args.sanitizer_dependency),
    )
    return [read_floats(path) for path in output_paths]


def scan_validation_text(text):
    errors = [
        line
        for line in text.splitlines()
        if re.search(
            r"Validation Error|VUID-[A-Za-z0-9-]+|ERROR.*validation|"
            r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:",
            line,
            re.I,
        )
    ]
    if errors:
        raise RuntimeError("Vulkan validation or sanitizer errors:\n" + "\n".join(errors[:20]))
    return text


def scan_validation_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return scan_validation_text(text)


def check_device(args, env):
    result = invoke([args.vulkaninfo, "--summary"], env=env)
    text = result.stdout + result.stderr
    if args.device_match.lower() not in text.lower():
        raise RuntimeError(
            f"required Vulkan device {args.device_match!r} is unavailable under ICD {args.icd}"
        )
    if "VK_LAYER_KHRONOS_validation" not in text:
        raise RuntimeError("Khronos Vulkan validation layer is unavailable")
    match = next(
        (line.strip() for line in text.splitlines()
         if args.device_match.lower() in line.lower()),
        args.device_match,
    )
    return match


def run_fixture(args, fixture, device_name, env, evidence_root):
    model_root = evidence_root / fixture["name"]
    shutil.rmtree(model_root, ignore_errors=True)
    model_root.mkdir(parents=True)
    cpu_dir = compile_model(args, fixture, "cpu", model_root / "cpu", env)
    vk_dir = compile_model(args, fixture, "vulkan", model_root / "vulkan", env)
    artifacts = validate_artifacts(args, fixture, cpu_dir, vk_dir, env)
    typed_runner = build_typed_runner(args, fixture, artifacts, model_root, env)

    trial_specs = [("zeros", None), ("negative", None), ("boundary", None)] + [
        (f"seed-{seed}", seed) for seed in (0, 1, 42)
    ]
    prepared = []

    def prepare_trial(name, seed, context_index, *, recreate=False):
        trial_dir = model_root / "trials" / name
        trial_dir.mkdir(parents=True)
        input_values = [
            make_input(
                name if seed is None else "random",
                shape,
                (seed or 0) + input_index * 97,
            )
            for input_index, (_, shape) in enumerate(fixture["inputs"])
        ]
        input_paths = []
        for input_index, values in enumerate(input_values):
            path = trial_dir / f"input-{input_index}.f32"
            write_floats(path, values)
            input_paths.append(path)
        ncnn_paths = [trial_dir / f"ncnn-{index}.f32"
                      for index in range(len(fixture["outputs"]))]
        reference = run_reference(args, fixture, input_paths, ncnn_paths, env)
        cpu_paths = [trial_dir / f"cpu-{index}.f32"
                     for index in range(len(fixture["outputs"]))]
        vk_paths = [trial_dir / f"vulkan-{index}.f32"
                    for index in range(len(fixture["outputs"]))]
        prepared.append(
            {
                "name": name,
                "seed": seed,
                "context_index": context_index,
                "recreate": recreate,
                "input_values": input_values,
                "input_paths": input_paths,
                "ncnn_paths": ncnn_paths,
                "reference": reference,
                "cpu_paths": cpu_paths,
                "vk_paths": vk_paths,
                "directory": trial_dir,
            }
        )

    for trial_index, (name, seed) in enumerate(trial_specs):
        prepare_trial(name, seed, trial_index % 2)
    replay_seed = 42
    prepare_trial("destroy-recreate-seed-42", replay_seed, 0, recreate=True)

    command = [
        str(typed_runner),
        "--cpu-library",
        str(artifacts["cpu_library"]),
        "--vulkan-library",
        str(artifacts["vk_library"]),
    ]
    for trial in prepared:
        if trial["recreate"]:
            command.append("--recreate")
        command.extend(["--call", trial["name"], str(trial["context_index"])])
        command.extend(str(path) for path in trial["input_paths"])
        command.extend(str(path) for path in trial["cpu_paths"])
        command.extend(str(path) for path in trial["vk_paths"])

    runtime_log = model_root / "typed-runner.stdout.json"
    validation_log = model_root / "vulkan-validation.log"
    process = subprocess.run(command, capture_output=True, text=True, env=env)
    runtime_log.write_text(process.stdout, encoding="utf-8")
    validation_log.write_text(process.stderr, encoding="utf-8")
    validation_text = scan_validation_text(process.stdout + "\n" + process.stderr)
    if process.returncode:
        raise RuntimeError(
            f"typed Vulkan model runner failed ({process.returncode}): {command}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    runtime_lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not runtime_lines:
        raise RuntimeError("typed Vulkan runner emitted no lifecycle evidence")
    runtime_result = json.loads(runtime_lines[-1])
    if (
        runtime_result.get("contexts_created") != 3
        or runtime_result.get("contexts_destroyed") != 3
        or runtime_result.get("call_count") != len(prepared)
    ):
        raise RuntimeError(f"typed runner lifecycle evidence is incomplete: {runtime_result}")
    runtime_calls = {
        item["label"]: item for item in runtime_result.get("calls", [])
    }
    if len(runtime_calls) != len(prepared):
        raise RuntimeError("typed runner emitted duplicate or missing trial records")

    raw_trials = []
    for trial in prepared:
        name = trial["name"]
        reference = trial["reference"]
        cpu_outputs = [read_floats(path) for path in trial["cpu_paths"]]
        gpu_outputs = [read_floats(path) for path in trial["vk_paths"]]
        cpu_vs_ncnn = compare_outputs(fixture, reference, cpu_outputs, "CPU/ncnn")
        if fixture["copy_exact"] and any(
            struct.pack(f"<{len(values)}f", *values)
            != struct.pack(f"<{len(cpu_outputs[index])}f", *cpu_outputs[index])
            for index, values in enumerate(trial["input_values"])
        ):
            raise RuntimeError("CPU copy fixture differs from its raw input")
        vk_vs_ncnn = compare_outputs(fixture, reference, gpu_outputs, "Vulkan/ncnn")
        if fixture["copy_exact"] and struct.pack(
            f"<{len(trial['input_values'][0])}f", *trial["input_values"][0]
        ) != struct.pack(f"<{len(gpu_outputs[0])}f", *gpu_outputs[0]):
            raise RuntimeError("Vulkan copy fixture differs from its raw input")
        cpu_vs_vk = compare_outputs(fixture, cpu_outputs, gpu_outputs, "CPU/Vulkan")
        runtime_call = runtime_calls.get(name)
        expected_context = 0 if trial["recreate"] else trial["context_index"]
        if runtime_call is None or runtime_call.get("context_index") != expected_context:
            raise RuntimeError(f"typed runner context evidence differs for {name}")
        raw_trials.append(
            {
                "name": name,
                "seed": trial["seed"],
                "context": "recreated-A" if trial["recreate"] else (
                    "A" if expected_context == 0 else "B"
                ),
                "raw_files": {
                    "inputs": [str(path.relative_to(model_root))
                               for path in trial["input_paths"]],
                    "ncnn_outputs": [str(path.relative_to(model_root))
                                     for path in trial["ncnn_paths"]],
                    "cpu_outputs": [str(path.relative_to(model_root))
                                    for path in trial["cpu_paths"]],
                    "vulkan_outputs": [str(path.relative_to(model_root))
                                       for path in trial["vk_paths"]],
                },
                "input_sha256": [sha256_file(path) for path in trial["input_paths"]],
                "ncnn_output_sha256": [sha256_file(path) for path in trial["ncnn_paths"]],
                "cpu_output_sha256": [sha256_file(path) for path in trial["cpu_paths"]],
                "vulkan_output_sha256": [sha256_file(path) for path in trial["vk_paths"]],
                "cpu_vs_ncnn": cpu_vs_ncnn,
                "vulkan_vs_ncnn": vk_vs_ncnn,
                "cpu_vs_vulkan": cpu_vs_vk,
                "diagnostic": runtime_call.get("diagnostic", ""),
            }
        )

    expected_labels = [trial["name"] for trial in prepared]
    if list(runtime_calls) != expected_labels:
        raise RuntimeError("typed runner trial order differs from the requested interleave")
    manifest = artifacts["vk_manifest"]
    plan = artifacts["plan"]
    plan_resources = plan.get("resources", [])
    planned_resource_bytes = sum(
        int(resource.get("byte_size", 0)) for resource in plan_resources
    )
    evidence = {
        "schema_version": 1,
        "fixture": fixture["name"],
        "model": fixture["model"],
        "device": device_name,
        "validation": {
            "layer": "VK_LAYER_KHRONOS_validation",
            "forced_by_environment": env.get("VK_INSTANCE_LAYERS"),
            "error_count": 0,
            "log_sha256": sha256_file(validation_log),
            "log_bytes": len(validation_text.encode("utf-8")),
        },
        "sanitizers": {
            "cxx_args": args.cxx_arg,
            "runtime_dir": args.sanitizer_runtime_dir,
            "enabled": bool(args.sanitizer_dependency),
        },
        "inputs": [
            {"blob": name, "shape_chw": shape}
            for name, shape in fixture["inputs"]
        ],
        "outputs": [
            {"blob": name, "shape_chw": shape}
            for name, shape in fixture["outputs"]
        ],
        "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE,
                      "copy_exact": fixture["copy_exact"]},
        "lifecycle": {
            "contexts_created": runtime_result["contexts_created"],
            "contexts_destroyed": runtime_result["contexts_destroyed"],
            "two_contexts_interleaved": True,
            "destroy_recreate_exercised": True,
            "peak_memory_bytes": manifest.get("peak_memory_bytes"),
            "peak_memory_evidence": "not exposed by the public Vulkan model ABI",
            "context_memory_budget_bytes": plan.get("context_memory_budget_bytes"),
            "planned_resource_bytes": planned_resource_bytes,
            "plan_resource_count": len(plan_resources),
            "plan_kernel_count": len(plan.get("kernels", [])),
            "resource_budget": plan.get("resource_budget"),
        },
        "identities": {
            "param_sha256": fixture["param_hash"],
            "model_bin_sha256": fixture["bin_hash"],
            "cpu_library_sha256": sha256_file(artifacts["cpu_library"]),
            "vulkan_library_sha256": sha256_file(artifacts["vk_library"]),
            "vulkan_dependencies": artifacts["vulkan_dependencies"],
            "cpu_dependencies": artifacts["cpu_dependencies"],
            "plan_sha256": artifacts["plan_sha256"],
            "plan_hash": plan.get("plan_hash"),
            "shader_sha256": artifacts["shader_sha256"],
            "shader_entries": artifacts["shaders"],
            "build_identity": manifest.get("build_identity"),
            "weight_manifest_entries": artifacts["weights"],
            "logical_weight_oracles": fixture.get("weight_oracles", []),
            "model_bin_identity": fixture["bin_hash"],
        },
        "runtime_process": runtime_result,
        "trials": raw_trials,
    }
    (model_root / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "compiler", "driver", "opt", "mlir-opt", "translate", "clang", "nm", "readelf",
        "llvm-as", "spirv-val", "spirv-dis", "spirv-extract", "host-cxx",
        "llvm-major", "llvm-lib-dir", "runner", "vulkaninfo", "project-include",
        "device-match", "icd", "output-dir",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--sanitizer-runtime-dir")
    parser.add_argument("--shared-llvm", action="store_true")
    parser.add_argument("--allowed-dependency", action="append", default=[])
    parser.add_argument("--sanitizer-dependency", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = Path(args.output_dir).resolve()
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True)
    args.icd = str(Path(args.icd).resolve())
    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = args.icd
    env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    old_library_path = env.get("LD_LIBRARY_PATH", "")
    library_paths = [args.llvm_lib_dir]
    if args.sanitizer_runtime_dir:
        library_paths.append(str(Path(args.sanitizer_runtime_dir).resolve()))
    if old_library_path:
        library_paths.append(old_library_path)
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    device_name = check_device(args, env)
    fixtures = fixture_definitions(args.output_dir / "fixtures")
    results = []
    for fixture in fixtures.values():
        results.append(run_fixture(args, fixture, device_name, env, args.output_dir / "models"))
    summary = {
        "schema_version": 1,
        "device": device_name,
        "icd": args.icd,
        "validation_layer": env["VK_INSTANCE_LAYERS"],
        "fixtures": [
            {"fixture": result["fixture"],
             "plan_hash": result["identities"]["plan_hash"],
             "shader_sha256": result["identities"]["shader_sha256"],
             "trials": len(result["trials"])}
            for result in results
        ],
    }
    (args.output_dir / "evidence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"V5.1 Vulkan product evidence: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"run_v5.py: {error}", file=sys.stderr)
        raise SystemExit(1)
