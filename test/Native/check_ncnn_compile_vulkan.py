#!/usr/bin/env python3
"""Exercise the ncnn-compile Vulkan product backend end to end."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import shlex
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


def assert_no_published_library(path):
    if path.exists():
        libraries = list(path.glob("*.so"))
        if libraries:
            raise RuntimeError(f"failed compile published a library: {libraries}")


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
    }
    llvm_prefix = "libLLVM.so." + args.llvm_major
    llvm_versioned_prefix = "libLLVM-" + args.llvm_major + ".so."
    def is_allowed_llvm(name):
        return (
            name == llvm_prefix
            or name.startswith(llvm_prefix + ".")
            or name.startswith(llvm_versioned_prefix)
        )

    if not any(is_allowed_llvm(name) for name in dependencies) or any(
        name not in allowed and not is_allowed_llvm(name) for name in dependencies
    ):
        raise RuntimeError(f"unexpected Vulkan DSO dependencies: {dependencies}")
    dynamic = invoke([args.readelf, "-d", str(library_path)]).stdout
    if "RPATH" in dynamic or "RUNPATH" in dynamic:
        raise RuntimeError("Vulkan DSO embeds an RPATH/RUNPATH build path")


def make_input(elements):
    generator = random.Random(42)
    values = [generator.uniform(-3.0, 3.0) for _ in range(elements)]
    values[0] = -1.0
    values[1] = 0.0
    values[2] = 1.5
    return (ctypes.c_float * elements)(*values)


def check_cpu_artifact(args, directory, input_data, expected):
    required_files(directory, {"paired_relu.h", "libpaired_relu.so"})
    needed = invoke(
        [args.readelf, "--needed-libs", str(directory / "libpaired_relu.so")]
    ).stdout
    if "libvulkan" in needed or "libLLVM" in needed:
        raise RuntimeError("default CPU artifact acquired a Vulkan/LLVM runtime dependency")

    library = ctypes.CDLL(str(directory / "libpaired_relu.so"))
    function = library.paired_relu
    function.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    function.restype = ctypes.c_int
    output = (ctypes.c_float * len(input_data))()
    if function(input_data, output) != 0:
        raise RuntimeError("paired CPU ReLU C ABI call failed")
    for index, reference in enumerate(expected):
        if abs(output[index] - reference) > 1.0e-6:
            raise RuntimeError(f"CPU ReLU differs from scalar golden at element {index}")
    return list(output)


def check_vulkan_artifact(args, directory, input_data, expected):
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
    if manifest["inputs"][0]["shape"] != [1, 1, len(input_data)]:
        raise RuntimeError("Vulkan manifest input shape differs from the fixture")
    if manifest["outputs"][0]["shape"] != [1, 1, len(input_data)]:
        raise RuntimeError("Vulkan manifest output shape differs from the fixture")
    check_exports_and_dependencies(args, library_path, manifest)

    options_type = type(
        "NcnnVulkanOptions",
        (ctypes.Structure,),
        {"_fields_": [("struct_size", ctypes.c_uint32),
                       ("abi_version", ctypes.c_uint32),
                       ("device_index", ctypes.c_uint32),
                       ("flags", ctypes.c_uint32)]},
    )
    library = ctypes.CDLL(str(library_path))
    create = library.paired_relu_vk_create
    create.argtypes = [
        ctypes.POINTER(options_type),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_uint32
    run = library.paired_relu_vk_run
    run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    run.restype = ctypes.c_uint32
    destroy = library.paired_relu_vk_destroy
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_uint32

    options = options_type(
        ctypes.sizeof(options_type), 1, 0xFFFFFFFF, 0
    )
    context = ctypes.c_void_p()
    diagnostic = ctypes.create_string_buffer(512)
    if create(ctypes.byref(options), ctypes.byref(context), diagnostic, 512) != 0:
        raise RuntimeError(f"Vulkan create failed: {diagnostic.value.decode()}")
    output = (ctypes.c_float * len(input_data))()
    status = run(
        context,
        input_data,
        len(input_data),
        output,
        len(output),
        diagnostic,
        512,
    )
    destroy_status = destroy(context)
    if status != 0 or destroy_status != 0:
        raise RuntimeError(
            f"Vulkan run/destroy failed ({status}/{destroy_status}): "
            f"{diagnostic.value.decode()}"
        )
    for index, reference in enumerate(expected):
        if abs(output[index] - reference) > 1.0e-6:
            raise RuntimeError(f"Vulkan ReLU differs from scalar golden at element {index}")
    if destroy(None) != 0:
        raise RuntimeError("destroy(NULL) did not return SUCCESS")
    return manifest, list(output)


def reconstruct_shader(args, directory):
    output = directory / "shader-reconstruction"
    output.mkdir()
    imported = output / "imported.mlir"
    prepared = output / "prepared.mlir"
    legal = output / "legal.mlir"
    device_ir = output / "device.mlir"
    binary_ir = output / "binary.mlir"
    plan = output / "plan.json"
    shader = output / "kernel.spv"
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
            str(imported),
            "--ncnn-vulkan-v2-prepare-pipeline",
            "--verify-each",
            "-o",
            str(prepared),
        ]
    )
    invoke(
        [
            args.opt,
            str(prepared),
            "--verify-vulkan-supported-model",
            "--verify-each",
            "-o",
            str(legal),
        ]
    )
    invoke(
        [
            args.opt,
            str(legal),
            f"--lower-ncnn-vulkan-model=plan-path={plan}",
            "--verify-each",
            "-o",
            str(device_ir),
        ]
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
    invoke(
        [
            args.spirv_extract,
            "--binary-symbol",
            "model_vulkan",
            str(binary_ir),
            "-o",
            str(shader),
        ]
    )
    invoke([args.spirv_val, "--target-env", "vulkan1.1", str(shader)])
    return hashlib.sha256(shader.read_bytes()).hexdigest()


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
    parser.add_argument("--llvm-major", required=True)
    parser.add_argument("--llvm-lib-dir", type=Path, required=True)
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
    base_env = os.environ.copy()
    old_library_path = base_env.get("LD_LIBRARY_PATH", "")
    base_env["LD_LIBRARY_PATH"] = (
        str(args.llvm_lib_dir) + (":" + old_library_path if old_library_path else "")
    )

    elements = 257
    input_data = make_input(elements)
    expected = [value if value > 0.0 else 0.0 for value in input_data]
    cpu_dir = args.work_dir / "cpu"
    invoke(
        [
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
        ],
        env=base_env,
    )
    cpu_output = check_cpu_artifact(args, cpu_dir, input_data, expected)
    if cpu_output != expected:
        raise RuntimeError("CPU fixture does not match the scalar golden")

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
    manifest, vulkan_output = check_vulkan_artifact(
        args, vulkan_dir, input_data, expected
    )
    cpu_vulkan = [abs(lhs - rhs) for lhs, rhs in zip(cpu_output, vulkan_output)]
    if any(error > 1.0e-6 for error in cpu_vulkan):
        raise RuntimeError("CPU and Vulkan product outputs differ")

    reconstructed_shader_sha256 = reconstruct_shader(args, args.work_dir)
    if reconstructed_shader_sha256 != manifest["shader_sha256"]:
        raise RuntimeError("manifest shader hash differs from regenerated SPIR-V")

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
    result = invoke(
        compile_args(
            args,
            "unsupported",
            unsupported,
            args.unsupported_param,
            args.unsupported_bin,
        ),
        env=base_env,
        expect_success=False,
    )
    if "verify-vulkan-supported-model" not in result.stderr and "outside" not in result.stderr:
        raise RuntimeError(f"unsupported op diagnostic is not actionable: {result.stderr}")
    assert_no_published_library(unsupported)

    for test_name, extra in (
        ("invalid-backend", ["--backend=unknown"]),
        ("cpu-option", ["--backend=vulkan", "--threads=1"]),
        ("debug-info", ["--backend=vulkan", "--debug-info"]),
        ("low-precision", ["--backend=vulkan", "--precision=fp16"]),
        ("dynamic-shape", ["--backend=vulkan", "--input-shape=1x?x?"]),
        ("cpu-target", ["--backend=vulkan", "--target-triple=x86_64-pc-linux-gnu"]),
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
        assert_no_published_library(output_dir)

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
    assert_no_published_library(missing_tool_output)

    bad_validator = args.work_dir / "reject-spirv-val"
    bad_validator.write_text("#!/bin/sh\nexit 23\n")
    bad_validator.chmod(0o755)
    invalid_shader_output = args.work_dir / "invalid-shader"
    invoke(
        compile_args(
            args,
            "invalid_shader",
            invalid_shader_output,
            args.work_dir / "model.param",
            args.work_dir / "model.bin",
        )
        + ["--vulkan-spirv-val=" + str(bad_validator)],
        env=base_env,
        expect_success=False,
    )
    assert_no_published_library(invalid_shader_output)

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
    assert_no_published_library(lost_device)
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
