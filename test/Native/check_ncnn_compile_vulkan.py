#!/usr/bin/env python3
"""Exercise the ncnn-compile Vulkan product backend end to end."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


_VULKAN_TEST_DIR = Path(__file__).resolve().parents[1] / "Vulkan"
sys.path.insert(0, str(_VULKAN_TEST_DIR))
try:
    import model_dependency_evidence as dependency_evidence
finally:
    sys.path.remove(str(_VULKAN_TEST_DIR))


EXPECTED_FAILURE_LOGS = []
VALIDATION_ERROR_RE = re.compile(
    r"VUID-[A-Za-z0-9-]+|Validation Error|^\s*(?:ERROR|Error)\b",
    re.IGNORECASE | re.MULTILINE,
)
SANITIZER_TEMP_BITCODE_RE = re.compile(
    rb"/(?:[^/\x00]+/)*ncnn-compile-[A-Za-z0-9_-]+/vulkan_model_wrapper\.bc"
)


def reproducible_artifact_bytes(filename, first, repeat, *, sanitizer):
    if first == repeat:
        return first, {"mode": "byte-identical", "normalized_temp_paths": 0}
    if not sanitizer or filename != "libpaired_relu_vulkan.so":
        return None, {"mode": "byte-identical", "normalized_temp_paths": 0}
    first_paths = SANITIZER_TEMP_BITCODE_RE.findall(first)
    repeat_paths = SANITIZER_TEMP_BITCODE_RE.findall(repeat)
    if not first_paths or len(first_paths) != len(repeat_paths):
        return None, {"mode": "byte-identical", "normalized_temp_paths": 0}

    def normalize(data):
        return SANITIZER_TEMP_BITCODE_RE.sub(
            b"/ncnn-compile-XXXXXX/vulkan_model_wrapper.bc", data)

    normalized_first, normalized_repeat = normalize(first), normalize(repeat)
    if normalized_first != normalized_repeat:
        return None, {"mode": "byte-identical", "normalized_temp_paths": 0}
    return normalized_first, {"mode": "sanitizer-temp-path-normalized",
                              "normalized_temp_paths": len(first_paths)}


def validation_error_count(text):
    return len(VALIDATION_ERROR_RE.findall(text))


def invoke(command, *, env=None, expect_success=True, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd)
    if not expect_success:
        EXPECTED_FAILURE_LOGS.append(
            {
                "argv": [str(item) for item in command],
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"command unexpectedly succeeded: {command}")
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host_link_launcher(args):
    return dependency_evidence.make_host_cxx_launcher(
        args.host_cxx, args.work_dir / "link-evidence")


def _driver_manifest_library(args, evidence_root, env):
    if not args.icd:
        return [], None
    manifest = Path(args.icd).resolve()
    if not manifest.is_file():
        return [], None
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read selected Vulkan ICD manifest: {error}") from error
    library = document.get("ICD", {}).get("library_path")
    if not isinstance(library, str) or not library:
        return [], None
    library_path = Path(library)
    candidates = [library_path] if library_path.is_absolute() else [manifest.parent / library_path]
    search_path = env.get("LD_LIBRARY_PATH", "")
    candidates.extend(Path(directory) / library_path.name
                      for directory in search_path.split(os.pathsep) if directory)

    ldconfig_record = None
    ldconfig = shutil.which("ldconfig")
    if not library_path.is_absolute() and ldconfig:
        command = [ldconfig, "-p"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        raw = evidence_root / "ldconfig-p.txt"
        raw.write_text(result.stdout + result.stderr, encoding="utf-8")
        ldconfig_record = {"file": "link-evidence/ldconfig-p.txt",
                           "sha256": sha256(raw), "argv": command,
                           "returncode": result.returncode}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=>" not in line:
                    continue
                soname, path = line.split("=>", 1)
                if soname.strip() == library_path.name:
                    candidates.append(Path(path.strip()))
    if not library_path.is_absolute():
        candidates.extend(Path(directory) / library_path.name for directory in (
            "/lib", "/usr/lib", "/lib64", "/usr/lib64",
            f"/lib/{os.uname().machine}-linux-gnu",
            f"/usr/lib/{os.uname().machine}-linux-gnu"))
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    copy = evidence_root / "driver-manifest.json"
    shutil.copyfile(manifest, copy)
    manifest_record = {"file": "link-evidence/driver-manifest.json",
                       "sha256": sha256(copy)}
    if ldconfig_record:
        manifest_record["system_loader_cache"] = ldconfig_record
    return ([str(resolved)] if resolved else []), manifest_record


def collect_build_dependencies(args, cpu_dir, vulkan_dir, loader_record_name, env):
    link_root = args.work_dir / "link-evidence"
    cpu_root = args.work_dir / "cpu-dependencies"
    tools = {"readelf": args.readelf, "ldd": shutil.which("ldd") or "ldd",
             "ar": shutil.which("ar") or "ar"}
    sanitizers = ([args.sanitizer_runtime_dir]
                  if args.sanitizer_runtime_dir else [])
    llvm_mode = "shared" if args.llvm_shared else "static"
    vulkan_path = vulkan_dir / "libpaired_relu_vulkan.so"
    cpu_path = cpu_dir / "libpaired_relu.so"
    vulkan_evidence = dependency_evidence.collect_model_dependencies(
        vulkan_path, link_root, tool_paths=tools, sanitizer_dirs=sanitizers,
        llvm_mode=llvm_mode, runtime_env=env)
    cpu_evidence = dependency_evidence.collect_elf_dependency_closure(
        cpu_path, cpu_root,
        tool_paths={"readelf": args.readelf, "ldd": tools["ldd"]},
        sanitizer_dirs=sanitizers, llvm_mode=llvm_mode, runtime_env=env)
    loader_path = args.work_dir / loader_record_name
    if not loader_path.is_file():
        raise RuntimeError(f"native loader module-map record is missing: {loader_path}")
    loader_record = json.loads(loader_path.read_text(encoding="utf-8"))
    driver_paths, manifest_record = _driver_manifest_library(args, link_root, env)
    phase_names = {
        "before-load": "before-model",
        "after-model-load": "after-dlopen",
        "after-create": "after-create",
    }
    for source_phase, evidence_phase in phase_names.items():
        entry = loader_record.get("loaded_modules", {}).get(source_phase)
        if not isinstance(entry, dict):
            raise RuntimeError(f"native loader omitted {source_phase} module maps")
        source = Path(str(entry.get("file", "")))
        if not source.is_absolute():
            source = args.work_dir / source
        if not source.is_file() or sha256(source) != entry.get("sha256"):
            raise RuntimeError(f"native loader {source_phase} module maps changed")
        dependency_evidence.record_driver_module_snapshot(
            link_root, evidence_phase,
            maps_text=source.read_text(encoding="utf-8", errors="replace"),
            driver_manifest_paths=driver_paths)
    driver_comparison = dependency_evidence.compare_driver_module_snapshots(link_root)
    driver_path = link_root / "driver-module-comparison.json"
    driver = {"file": "link-evidence/driver-module-comparison.json",
              "sha256": sha256(driver_path),
              "status": driver_comparison["driver_origin_status"],
              "comparison_status": driver_comparison["status"]}
    if manifest_record:
        driver["manifest"] = manifest_record
    if driver["status"] != "measured":
        driver["reason"] = (
            "selected ICD library was not observed in the native after-create module-map delta"
            if driver_paths else
            "no resolvable selected ICD library path; no driver classification"
        )
    return {
        "vulkan": {"file": "link-evidence/model-dependency-evidence.json",
                    "sha256": sha256(link_root / "model-dependency-evidence.json")},
        "cpu": {"file": "cpu-dependencies/elf-dependency-closure.json",
                "sha256": sha256(cpu_root / "elf-dependency-closure.json")},
        "driver": driver,
    }, vulkan_evidence, cpu_evidence


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
    required_files(directory, {"paired_relu.h", "paired_relu.json", "libpaired_relu.so"})
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
    for key, expected in (("host_implementation", "generic-plan-runtime"),
                          ("host_codegen_version", 1),
                          ("execution_plan_usage", "embedded-runtime-required")):
        if manifest.get(key) != expected:
            raise RuntimeError(f"Vulkan manifest {key} differs from the host contract")
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


def run_product_loader(args, cpu_dir, vulkan_dir, model_name, input_data,
                       *, expect_no_device=False, env=None):
    prefix = "no-device-" if expect_no_device else ""
    loader = args.work_dir / f"{prefix}vulkan-product-loader"
    cpu_output_path = args.work_dir / f"{prefix}cpu-output.f32"
    vulkan_output_path = args.work_dir / f"{prefix}vulkan-output.f32"
    input_path = args.work_dir / f"{prefix}input.f32"
    result_path = args.work_dir / f"{prefix}loader-result.json"
    env = runtime_env(args) if env is None else env
    elements = len(input_data)
    input_path.write_bytes(struct.pack(f"<{elements}f", *input_data))

    compile_command = [
        args.host_cxx,
        "-std=c++23",
        *args.cxx_arg,
        f"-DNCNN_PRODUCT_TEST_MODEL={model_name}",
        *(["-DNCNN_PRODUCT_TEST_EXPECT_NO_DEVICE=1"] if expect_no_device else []),
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
    compile_result = invoke(compile_command, env=env)
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
    child = subprocess.run(command, capture_output=True, text=True, env=env)
    streams = {}
    for name, text in (
        ("compile.stdout.log", compile_result.stdout),
        ("compile.stderr.log", compile_result.stderr),
        ("stdout.log", child.stdout),
        ("stderr.log", child.stderr),
    ):
        path = args.work_dir / f"{prefix}vulkan-loader.{name}"
        path.write_text(text, encoding="utf-8")
        streams[name] = {"file": path.name, "sha256": sha256(path)}
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
    if expect_no_device and (
        result.get("phase") != "expected-no-device"
        or result.get("vulkan_create_status") not in (2, 5)
        or result.get("context_null") is not True
        or result.get("diagnostic_bounded") is not True
        or result.get("vulkan_destroy_null_status") != 0
        or not result.get("diagnostic")
        or vulkan_output_path.exists()
    ):
        raise RuntimeError(f"invalid no-device lifecycle result: {result}")
    modules = {}
    for phase in ("before-load", "after-model-load", "after-create"):
        path = Path(str(result_path) + f".{phase}.maps")
        if not path.is_file():
            raise RuntimeError(f"loader module evidence is absent: {path}")
        modules[phase] = {"file": path.name, "sha256": sha256(path)}
    record = {
        "compile_argv": [str(item) for item in compile_command],
        "compile_returncode": compile_result.returncode,
        "argv": [str(item) for item in command],
        "returncode": child.returncode,
        "loader_file": loader.name,
        "loader_sha256": sha256(loader),
        "result_file": result_path.name,
        "result_sha256": sha256(result_path),
        "streams": streams,
        "loaded_modules": modules,
        "icd_environment": {key: env.get(key) for key in (
            "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES")},
        "gpu_numeric": not expect_no_device,
    }
    (args.work_dir / f"{prefix}loader-invocation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if expect_no_device:
        if cpu_output_path.read_bytes() != struct.pack(
            f"<{elements}f", *[max(value, 0.0) for value in input_data]
        ):
            raise RuntimeError("CPU golden failed in no-device lifecycle loader")
        return record

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
    if args.icd:
        env["VK_ICD_FILENAMES"] = str(args.icd)
    if not args.offline:
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    inherited = env.get("LD_LIBRARY_PATH", "")
    if inherited:
        library_dirs.append(inherited)
    if library_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(library_dirs)
    return env


def load_spirv_validation_recorder():
    recorder_path = Path(__file__).resolve().parents[1] / "Vulkan" / "record_spirv_validation.py"
    spec = importlib.util.spec_from_file_location(
        "ncnn_spirv_validation_recorder", recorder_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SPIR-V validation recorder: {recorder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(recorder_path.parent))
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(recorder_path.parent))
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def reconstruct_shaders(args, directory):
    output = directory / "shader-reconstruction"
    output.mkdir()
    validation_dir = output / "spirv-validation"
    recorder = load_spirv_validation_recorder()
    spirv_val_launcher = recorder.make_launcher(args.spirv_val, validation_dir)
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
        invoke(
            [str(spirv_val_launcher), "--target-env", "vulkan1.1", str(shader)]
        )
        shader_hashes[kernel["id"]] = hashlib.sha256(shader.read_bytes()).hexdigest()
    recorded_validations = recorder.validate_records(
        validation_dir,
        [shader_hashes[kernel["id"]] for kernel in plan["kernels"]],
    )
    if len(recorded_validations) != len(plan["kernels"]):
        raise RuntimeError("recorded SPIR-V validation count differs from the kernel plan")
    shader_validations = []
    for kernel, record in zip(plan["kernels"], recorded_validations):
        kernel_id = kernel["id"]
        expected_hash = shader_hashes[kernel_id]
        if record["sha256"] != expected_hash:
            raise RuntimeError(
                f"SPIR-V validation identity differs for kernel {kernel_id}"
            )
        shader_validations.append(
            {
                "kernel_id": kernel_id,
                "log_file": (
                    "vulkan/shader-reconstruction/spirv-validation/"
                    + record["log"]
                ),
                "log_sha256": record["log_sha256"],
                "shader_file": (
                    "vulkan/shader-reconstruction/spirv-validation/"
                    + record["path"]
                ),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    if [item["kernel_id"] for item in shader_validations] != [
        kernel["id"] for kernel in plan["kernels"]
    ]:
        raise RuntimeError("recorded SPIR-V validations do not cover every product kernel")
    return shader_hashes, shader_validations


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


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def artifact_hashes(path):
    return {
        entry.name: sha256(entry)
        for entry in sorted(path.iterdir())
        if entry.is_file()
    }


def run_cpu_loader(args, cpu_dir, input_values, input_path, output_path, env):
    directory = args.work_dir / f"cpu-loader-{cpu_dir.name}"
    directory.mkdir()
    loader = directory / "cpu-product-loader"
    result_path = directory / "result.json"
    count = len(input_values)
    input_path.write_bytes(struct.pack(f"<{count}f", *input_values))
    compile_command = [
        args.host_cxx,
        "-std=c++23",
        *args.cxx_arg,
        "-DNCNN_PRODUCT_TEST_MODEL=paired_relu",
        "-DNCNN_PRODUCT_TEST_CPU_ONLY=1",
        "-I",
        str(cpu_dir),
        str(args.loader_source),
        "-ldl",
        "-o",
        str(loader),
    ]
    compile_result = invoke(compile_command, env=env)
    command = [
        str(loader),
        "paired_relu",
        str(cpu_dir / "libpaired_relu.so"),
        str(count),
        str(input_path),
        str(output_path),
        str(result_path),
    ]
    child = subprocess.run(command, capture_output=True, text=True, env=env)
    streams = {}
    for name, text in (
        ("compile.stdout.log", compile_result.stdout),
        ("compile.stderr.log", compile_result.stderr),
        ("run.stdout.log", child.stdout),
        ("run.stderr.log", child.stderr),
    ):
        path = directory / name
        path.write_text(text, encoding="utf-8")
        streams[name] = {
            "file": str(path.relative_to(args.work_dir)),
            "sha256": sha256(path),
        }
    record = {
        "compile_argv": [str(item) for item in compile_command],
        "compile_returncode": compile_result.returncode,
        "argv": [str(item) for item in command],
        "returncode": child.returncode,
        "loader_file": str(loader.relative_to(args.work_dir)),
        "loader_sha256": sha256(loader),
        "result_file": str(result_path.relative_to(args.work_dir)),
        "streams": streams,
    }
    (directory / "invocation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logs = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    diagnostic = sanitizer_diagnostics(child.stdout + "\n" + child.stderr)
    if diagnostic:
        raise RuntimeError(
            f"sanitizer diagnostic from CPU loader ({child.returncode}): "
            f"{diagnostic.group(0)}\n{logs}"
        )
    if child.returncode != 0:
        raise RuntimeError(f"CPU loader failed ({child.returncode})\n{logs}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        not isinstance(result, dict)
        or result.get("status") != "success"
        or type(result.get("cpu_status")) is not int
        or result["cpu_status"] != 0
    ):
        raise RuntimeError(f"CPU loader reported failure: {result}\n{logs}")
    output_bytes = output_path.read_bytes()
    if len(output_bytes) != count * 4:
        raise RuntimeError(
            f"CPU loader wrote {len(output_bytes)} bytes, expected {count * 4}"
        )
    record["result_sha256"] = sha256(result_path)
    (directory / "invocation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_values = list(struct.unpack(f"<{count}f", output_bytes))
    return result["cpu_status"], output_values, output_bytes, record


def run_cpu_relu(args, cpu_dir, input_values, input_path, output_path, env):
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
        "--emit-manifest",
        "-O0",
        "--verify-execution",
    ]
    for flag in args.cxx_arg:
        cpu_command.extend([f"--clang-arg={flag}", f"--linker-arg={flag}"])
    result = invoke(cpu_command, env=env)
    check_cpu_artifact(args, cpu_dir)

    status, output_values, output_bytes, loader_record = run_cpu_loader(
        args, cpu_dir, input_values, input_path, output_path, env
    )
    return result, status, output_values, output_bytes, loader_record


def offline_failure(args, name, command, env, output_dir, expected_diagnostic):
    before = artifact_hashes(output_dir)
    result = invoke(command, env=env, expect_success=False)
    diagnostic = result.stdout + "\n" + result.stderr
    if not re.search(expected_diagnostic, diagnostic, re.IGNORECASE):
        raise RuntimeError(
            f"{name} did not report expected diagnostic /{expected_diagnostic}/: "
            f"{diagnostic}"
        )
    after = artifact_hashes(output_dir)
    unchanged = before == after
    record = {
        "name": name,
        "argv": [str(item) for item in command],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "expected_diagnostic": expected_diagnostic,
        "diagnostic_match": True,
        "artifacts_before_sha256": before,
        "artifacts_after_sha256": after,
        "artifacts_unchanged": unchanged,
    }
    if not unchanged:
        raise RuntimeError(f"{name} modified successful Vulkan artifacts")
    return record


def run_offline(args):
    args.work_dir = args.work_dir.resolve()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    shutil.copy2(args.param, args.work_dir / "model.param")
    shutil.copy2(args.bin, args.work_dir / "model.bin")
    env = runtime_env(args)
    env.pop("VK_DRIVER_FILES", None)
    env.pop("VK_ADD_DRIVER_FILES", None)
    unavailable_icd = args.work_dir / "no-available-icd.json"
    env["VK_ICD_FILENAMES"] = str(unavailable_icd)
    if unavailable_icd.exists():
        unavailable_icd.unlink()

    elements = 257
    input_values = make_input(elements)
    golden_values = [value if value > 0.0 else 0.0 for value in input_values]
    expected_bytes = struct.pack(f"<{elements}f", *golden_values)
    cpu_dir = args.work_dir / "cpu"
    input_path = args.work_dir / "public-input.f32"
    cpu_output_path = args.work_dir / "cpu-output.f32"
    golden_path = args.work_dir / "scalar-golden.f32"
    golden_path.write_bytes(expected_bytes)
    cpu_compile, cpu_status, cpu_values, cpu_bytes, cpu_loader = run_cpu_relu(
        args, cpu_dir, input_values, input_path, cpu_output_path, env
    )
    input_bytes = input_path.read_bytes()
    if cpu_status != 0:
        raise RuntimeError(f"CPU ReLU returned status {cpu_status}")
    cpu_errors = [abs(actual - expected) for actual, expected in zip(cpu_values, golden_values)]
    max_abs_error = max(cpu_errors, default=0.0)
    if max_abs_error > 1.0e-6:
        raise RuntimeError(f"CPU ReLU differs from scalar golden: max_abs={max_abs_error}")

    vulkan_dir = args.work_dir / "vulkan"
    offline_compile_command = [
        args.compiler,
        str(args.work_dir / "model.param"),
        "--bin",
        str(args.work_dir / "model.bin"),
        "--model-name",
        "paired_relu",
        "--output-dir",
        str(vulkan_dir),
        "--backend=vulkan",
        "--precision=f32",
    ]
    initial_compile_command = [
        *offline_compile_command,
        "--vulkan-host-cxx=" + str(host_link_launcher(args)),
    ]
    compile_result = invoke(initial_compile_command, env=env)
    manifest = check_vulkan_artifact(args, vulkan_dir, elements)
    no_device_lifecycle = run_product_loader(
        args, cpu_dir, vulkan_dir, "paired_relu", input_values,
        expect_no_device=True, env=env)
    build_dependencies, vulkan_dependency_evidence, cpu_dependency_evidence = (
        collect_build_dependencies(
            args, cpu_dir, vulkan_dir, "no-device-loader-invocation.json", env)
    )
    shader_hashes, shader_validations = reconstruct_shaders(args, args.work_dir)
    manifest_shader_hashes = {
        int(shader["kernel_id"]): shader["sha256"]
        for shader in manifest.get("shaders", [])
    }
    if shader_hashes != manifest_shader_hashes:
        raise RuntimeError("manifest shader hashes differ from offline SPIR-V")
    if len(shader_hashes) == 1 and next(iter(shader_hashes.values())) != manifest["shader_sha256"]:
        raise RuntimeError("manifest primary shader hash differs from offline SPIR-V")
    successful_artifacts = artifact_hashes(vulkan_dir)
    negative_dir = args.work_dir / "negative-tools"
    negative_dir.mkdir()
    negatives = []

    missing_validator = [
        *offline_compile_command,
        "--vulkan-spirv-val=" + str(args.work_dir / "missing-spirv-val"),
    ]
    negatives.append(
        offline_failure(
            args,
            "missing-validator",
            missing_validator,
            env,
            vulkan_dir,
            r"cannot execute|no such file|not found",
        )
    )

    rejecting_validator = negative_dir / "reject-spirv-val"
    rejecting_validator.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    rejecting_validator.chmod(0o755)
    negatives.append(
        offline_failure(
            args,
            "validator-failure",
            [*offline_compile_command, "--vulkan-spirv-val=" + str(rejecting_validator)],
            env,
            vulkan_dir,
            r"exit code 23",
        )
    )

    rejecting_cxx = negative_dir / "reject-host-link"
    rejecting_cxx.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-dumpmachine\" ]; then\n"
        f"  exec {shlex.quote(args.host_cxx)} -dumpmachine\n"
        "fi\n"
        "exit 29\n",
        encoding="utf-8",
    )
    rejecting_cxx.chmod(0o755)
    negatives.append(
        offline_failure(
            args,
            "link-failure",
            [*offline_compile_command, "--vulkan-host-cxx=" + str(rejecting_cxx)],
            env,
            vulkan_dir,
            r"exit code 29",
        )
    )

    rejecting_readelf = negative_dir / "reject-readelf"
    rejecting_readelf.write_text("#!/bin/sh\nexit 31\n", encoding="utf-8")
    rejecting_readelf.chmod(0o755)
    negatives.append(
        offline_failure(
            args,
            "audit-failure",
            [*offline_compile_command, "--readelf=" + str(rejecting_readelf)],
            env,
            vulkan_dir,
            r"exit code 31",
        )
    )

    no_device_verify = [*offline_compile_command, "--verify-execution"]
    negatives.append(
        offline_failure(
            args,
            "no-device-execution-verification",
            no_device_verify,
            env,
            vulkan_dir,
            r"Vulkan ABI execution verification failed|no Vulkan physical devices|no compatible Vulkan",
        )
    )

    # Recompile and execute the CPU fixture after all failed product attempts.
    cpu_repeat_dir = args.work_dir / "cpu-after-negative-cases"
    repeat_input_path = args.work_dir / "public-input-after-negative-cases.f32"
    repeat_output_path = args.work_dir / "cpu-output-after-negative-cases.f32"
    _, repeat_status, repeat_values, repeat_bytes, recovery_loader = run_cpu_relu(
        args,
        cpu_repeat_dir,
        input_values,
        repeat_input_path,
        repeat_output_path,
        env,
    )
    if repeat_status != 0 or repeat_bytes != expected_bytes:
        raise RuntimeError("CPU ReLU did not recover after Vulkan failure cases")
    cpu_library_sha256 = sha256(cpu_dir / "libpaired_relu.so")
    recovery_library_sha256 = sha256(cpu_repeat_dir / "libpaired_relu.so")
    if recovery_library_sha256 != cpu_library_sha256:
        raise RuntimeError("CPU ReLU recovery changed the CPU artifact identity")
    shutil.move(
        args.work_dir / "shader-reconstruction",
        vulkan_dir / "shader-reconstruction",
    )

    evidence = {
        "schema_version": 1,
        "kind": "ncnn.vulkan_offline_evidence",
        "build_dependencies": build_dependencies,
        "execution": {
            "mode": "offline-no-device",
            "vk_icd_filenames": str(unavailable_icd),
            "vk_icd_manifest_exists": unavailable_icd.exists(),
            "vk_driver_files_override": env.get("VK_DRIVER_FILES"),
            "vk_add_driver_files_override": env.get("VK_ADD_DRIVER_FILES"),
            "vulkan_execution": "not_run",
            "gpu_output": None,
        },
        "fixture": {
            "model": "paired_relu",
            "param_sha256": sha256(args.work_dir / "model.param"),
            "bin_sha256": sha256(args.work_dir / "model.bin"),
            "cpu_library_sha256": sha256(cpu_dir / "libpaired_relu.so"),
            "cpu_header_sha256": sha256(cpu_dir / "paired_relu.h"),
            "cpu_manifest_sha256": sha256(cpu_dir / "paired_relu.json"),
            "vulkan_library_sha256": sha256(vulkan_dir / "libpaired_relu_vulkan.so"),
            "vulkan_header_sha256": sha256(vulkan_dir / "paired_relu_vulkan.h"),
            "vulkan_manifest_sha256": sha256(vulkan_dir / "paired_relu_vulkan.json"),
            "vulkan_plan_sha256": sha256(vulkan_dir / "paired_relu_vulkan.plan.json"),
            "plan_hash": manifest["plan_hash"],
            "shader_sha256": manifest["shader_sha256"],
            "reconstructed_shader_sha256": {
                str(key): value for key, value in sorted(shader_hashes.items())
            },
            "shader_files": [
                {
                    "file": f"vulkan/shader-reconstruction/kernel-{key}.spv",
                    "sha256": value,
                }
                for key, value in sorted(shader_hashes.items())
            ],
            "exports": [
                "paired_relu_vk_create",
                "paired_relu_vk_destroy",
                "paired_relu_vk_run",
            ],
            "dependencies": manifest["dependencies"],
            "offline_success_artifacts_sha256": successful_artifacts,
            "cpu_recovery_library_sha256": recovery_library_sha256,
        },
        "public_io": {
            "shape": [elements],
            "shape_chw": [1, 1, elements],
            "host_layout": "CHW",
            "element_type": "f32",
            "input": {
                "file": input_path.name,
                "sha256": sha256_bytes(input_bytes),
                "values": list(struct.unpack(f"<{elements}f", input_bytes)),
            },
            "cpu_output": {
                "file": cpu_output_path.name,
                "sha256": sha256_bytes(cpu_bytes),
                "values": cpu_values,
            },
            "cpu_recovery_output": {
                "file": repeat_output_path.name,
                "sha256": sha256_bytes(repeat_bytes),
                "values": repeat_values,
            },
            "scalar_golden": {
                "file": golden_path.name,
                "sha256": sha256_bytes(expected_bytes),
                "values": golden_values,
            },
            "tolerance": {"absolute": 1.0e-6, "relative": 1.0e-6},
            "max_absolute_error": max_abs_error,
            "max_relative_error": max(
                (
                    abs(actual - expected) / max(abs(expected), 1.0e-30)
                    for actual, expected in zip(cpu_values, golden_values)
                ),
                default=0.0,
            ),
            "passed": True,
            "cpu_status": cpu_status,
            "cpu_recovery_status": repeat_status,
            "cpu_recovery_output_sha256": sha256_bytes(repeat_bytes),
            "cpu_recovery_max_absolute_error": max(
                (abs(actual - expected) for actual, expected in zip(repeat_values, golden_values)),
                default=0.0,
            ),
        },
        "negative_cases": negatives,
        "no_device_lifecycle": no_device_lifecycle,
        "shader_validations": shader_validations,
        "commands": {
            "cpu_compile": [
                str(item) for item in cpu_compile.args
            ] if hasattr(cpu_compile, "args") else None,
            "cpu_compile_returncode": cpu_compile.returncode,
            "cpu_compile_stdout": cpu_compile.stdout,
            "cpu_compile_stderr": cpu_compile.stderr,
            "cpu_loader": cpu_loader,
            "cpu_recovery_loader": recovery_loader,
            "offline_vulkan_compile": [str(item) for item in initial_compile_command],
            "offline_compile_returncode": compile_result.returncode,
            "offline_compile_stdout": compile_result.stdout,
            "offline_compile_stderr": compile_result.stderr,
            "validator": str(args.spirv_val),
            "validator_version": invoke([args.spirv_val, "--version"]).stdout,
        },
        "claims": {
            "cpu_golden": "passed",
            "vulkan_static_product_artifacts": "validated",
            "vulkan_device_execution": "not_run",
            "cpu_gpu_numerical_comparison": "not_claimed",
        },
    }
    evidence_path = args.work_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"Offline Vulkan product evidence: {evidence_path}")
    return 0


def parse_vulkan_device_identity(args, summary):
    parser_path = Path(__file__).resolve().parents[1] / "Vulkan" / "run_v5.py"
    spec = importlib.util.spec_from_file_location(
        "ncnn_vulkan_device_parser", parser_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Vulkan identity parser: {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(parser_path.parent))
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(parser_path.parent))
        sys.dont_write_bytecode = previous_dont_write_bytecode
    parse_summary = getattr(module, "parse_vulkan_device_summary", None)
    if not callable(parse_summary):
        raise RuntimeError("run_v5.py has no parse_vulkan_device_summary helper")
    return parse_summary(summary, args.device_match, args.icd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--icd", help="Force a Vulkan ICD manifest for product execution")
    parser.add_argument("--device-match", help="Expected physical-device name under the selected ICD")
    parser.add_argument("--vulkaninfo", help="Path to vulkaninfo for explicit device evidence")
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
    EXPECTED_FAILURE_LOGS.clear()
    if args.offline:
        return run_offline(args)

    args.work_dir = args.work_dir.resolve()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    shutil.copy2(args.param, args.work_dir / "model.param")
    shutil.copy2(args.bin, args.work_dir / "model.bin")
    base_env = runtime_env(args)
    if not args.icd:
        raise RuntimeError("device product test requires an explicit --icd")
    if not args.device_match:
        raise RuntimeError("device product test requires an explicit --device-match")
    if not args.vulkaninfo:
        raise RuntimeError("device product test requires an explicit --vulkaninfo")
    vulkaninfo_command = [args.vulkaninfo, "--summary"]
    vulkan_info = subprocess.run(
        vulkaninfo_command,
        capture_output=True,
        text=True,
        env=base_env,
    )
    (args.work_dir / "vulkaninfo.stdout.log").write_text(
        vulkan_info.stdout, encoding="utf-8"
    )
    (args.work_dir / "vulkaninfo.stderr.log").write_text(
        vulkan_info.stderr, encoding="utf-8"
    )
    if vulkan_info.returncode != 0:
        raise RuntimeError(
            f"vulkaninfo --summary failed ({vulkan_info.returncode})\n"
            f"stdout:\n{vulkan_info.stdout}stderr:\n{vulkan_info.stderr}"
        )
    vulkan_info_summary = vulkan_info.stdout + vulkan_info.stderr
    if "VK_LAYER_KHRONOS_validation" not in vulkan_info_summary:
        raise RuntimeError("VK_LAYER_KHRONOS_validation is required for product execution")
    device_identity = parse_vulkan_device_identity(args, vulkan_info_summary)
    physical_device_types = [device_identity["device_type"]]

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
        "--emit-manifest",
        "-O0",
        "--verify-execution",
    ]
    for flag in args.cxx_arg:
        cpu_command.extend([f"--clang-arg={flag}", f"--linker-arg={flag}"])
    invoke(cpu_command, env=base_env)
    check_cpu_artifact(args, cpu_dir)

    vulkan_dir = args.work_dir / "vulkan"
    vulkan_compile_command = compile_args(
        args,
        "paired_relu",
        vulkan_dir,
        args.work_dir / "model.param",
        args.work_dir / "model.bin",
    )
    vulkan_compile_command.append(
        "--vulkan-host-cxx=" + str(host_link_launcher(args)))
    vulkan_compile_result = subprocess.run(
        vulkan_compile_command,
        capture_output=True,
        text=True,
        env=base_env,
    )
    (args.work_dir / "vulkan-verify.stdout.log").write_text(
        vulkan_compile_result.stdout, encoding="utf-8"
    )
    (args.work_dir / "vulkan-verify.stderr.log").write_text(
        vulkan_compile_result.stderr, encoding="utf-8"
    )
    if vulkan_compile_result.returncode != 0:
        raise RuntimeError(
            "Vulkan product compile/verify failed "
            f"({vulkan_compile_result.returncode})\n"
            f"stdout:\n{vulkan_compile_result.stdout}"
            f"stderr:\n{vulkan_compile_result.stderr}"
        )
    manifest = check_vulkan_artifact(args, vulkan_dir, elements)
    cpu_output, vulkan_output = run_product_loader(
        args, cpu_dir, vulkan_dir, "paired_relu", input_data
    )
    build_dependencies, vulkan_dependency_evidence, cpu_dependency_evidence = (
        collect_build_dependencies(
            args, cpu_dir, vulkan_dir, "loader-invocation.json", base_env)
    )
    validation_streams = [
        "vulkaninfo.stdout.log",
        "vulkaninfo.stderr.log",
        "vulkan-verify.stdout.log",
        "vulkan-verify.stderr.log",
        "vulkan-loader.stdout.log",
        "vulkan-loader.stderr.log",
    ]
    validation_text = "".join(
        f"===== {name} =====\n"
        f"{(args.work_dir / name).read_text(encoding='utf-8')}\n"
        for name in validation_streams
    )
    validation_path = args.work_dir / "validation.log"
    validation_path.write_text(validation_text, encoding="utf-8")
    validation_error_count_value = validation_error_count(validation_text)
    if validation_error_count_value != 0:
        raise RuntimeError(
            "Vulkan validation emitted error diagnostics; see "
            f"{validation_path} (count={validation_error_count_value})"
        )
    public_input_bytes = (args.work_dir / "input.f32").read_bytes()
    cpu_output_bytes = (args.work_dir / "cpu-output.f32").read_bytes()
    vulkan_output_bytes = (args.work_dir / "vulkan-output.f32").read_bytes()
    golden_bytes = struct.pack(f"<{elements}f", *expected)
    (args.work_dir / "scalar-golden.f32").write_bytes(golden_bytes)
    for label, output in (("CPU", cpu_output), ("Vulkan", vulkan_output)):
        for index, reference in enumerate(expected):
            if abs(output[index] - reference) > 1.0e-6:
                raise RuntimeError(
                    f"{label} ReLU differs from scalar golden at element {index}"
                )
    cpu_vulkan = [abs(lhs - rhs) for lhs, rhs in zip(cpu_output, vulkan_output)]
    if any(error > 1.0e-6 for error in cpu_vulkan):
        raise RuntimeError("CPU and Vulkan product outputs differ")

    reconstructed_shader_hashes, shader_validations = reconstruct_shaders(
        args, args.work_dir
    )
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
    repeat_link_launcher = dependency_evidence.make_host_cxx_launcher(
        args.host_cxx, args.work_dir / "repeat-link-evidence")
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
            "--vulkan-host-cxx=" + str(repeat_link_launcher),
        ],
        env=base_env,
    )
    required_files(vulkan_dir, {entry.name for entry in repeat_dir.iterdir()})
    vulkan_reproducibility = {}
    for filename in sorted(entry.name for entry in repeat_dir.iterdir()):
        first_bytes = (vulkan_dir / filename).read_bytes()
        repeat_bytes = (repeat_dir / filename).read_bytes()
        normalized, comparison = reproducible_artifact_bytes(
            filename, first_bytes, repeat_bytes,
            sanitizer=bool(args.sanitizer_runtime_dir))
        if normalized is None:
            raise RuntimeError(f"Vulkan artifact is not reproducible: {filename}")
        if filename in {"libpaired_relu_vulkan.so", "paired_relu_vulkan.json",
                        "paired_relu_vulkan.plan.json", "paired_relu_vulkan.h"}:
            vulkan_reproducibility[filename] = {
                **comparison,
                "first_sha256": hashlib.sha256(first_bytes).hexdigest(),
                "repeat_sha256": hashlib.sha256(repeat_bytes).hexdigest(),
                "canonical_sha256": hashlib.sha256(normalized).hexdigest(),
            }

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
    result = invoke(
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
    if not re.search(r"cannot execute|no such file|not found", result.stderr, re.IGNORECASE):
        raise RuntimeError(f"missing validator diagnostic is unclear: {result.stderr}")
    assert_no_published_artifacts(missing_tool_output)

    bad_validator = args.work_dir / "reject-spirv-val"
    bad_validator.write_text("#!/bin/sh\nexit 23\n")
    bad_validator.chmod(0o755)
    result = invoke(
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
    if "failed with exit code 23" not in result.stderr:
        raise RuntimeError(f"validator failure diagnostic is unclear: {result.stderr}")
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
    if "failed with exit code 31" not in result.stderr:
        raise RuntimeError(f"audit failure diagnostic is unclear: {result.stderr}")
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("failed Vulkan dependency audit replaced prior artifacts")

    lost_icd_env = base_env.copy()
    lost_icd_env["VK_ICD_FILENAMES"] = str(args.work_dir / "missing-icd.json")
    lost_device = args.work_dir / "missing-device"
    result = invoke(
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
    if not re.search(
        r"Vulkan ABI execution verification failed|no Vulkan physical devices|no compatible Vulkan",
        result.stderr,
        re.IGNORECASE,
    ):
        raise RuntimeError(f"missing-device diagnostic is unclear: {result.stderr}")
    assert_no_published_artifacts(lost_device)
    result = invoke(
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
    if not re.search(
        r"Vulkan ABI execution verification failed|no Vulkan physical devices|no compatible Vulkan",
        result.stderr,
        re.IGNORECASE,
    ):
        raise RuntimeError(f"existing-artifact no-device diagnostic is unclear: {result.stderr}")
    if output_snapshot(vulkan_dir) != previous:
        raise RuntimeError("missing Vulkan device replaced prior successful artifacts")
    shutil.move(
        args.work_dir / "shader-reconstruction",
        vulkan_dir / "shader-reconstruction",
    )

    max_abs_cpu_error = max(
        (abs(actual - reference) for actual, reference in zip(cpu_output, expected)),
        default=0.0,
    )
    max_abs_vulkan_error = max(
        (abs(actual - reference) for actual, reference in zip(vulkan_output, expected)),
        default=0.0,
    )
    max_abs_cpu_vulkan_error = max(
        (abs(actual - reference) for actual, reference in zip(cpu_output, vulkan_output)),
        default=0.0,
    )
    product_evidence = {
        "schema_version": 1,
        "kind": "ncnn.vulkan_product_evidence",
        "build_dependencies": build_dependencies,
        "vulkan_reproducibility": vulkan_reproducibility,
        "execution": {
            "mode": "device-execution",
            "icd_manifest": args.icd,
            "vulkan_execution": "passed",
            "vulkaninfo": args.vulkaninfo,
            "vulkaninfo_summary": vulkan_info_summary,
            "physical_device_types": physical_device_types,
            "device_evidence": {
                **json.loads((args.work_dir / "loader-result.json").read_text()),
                **device_identity,
            },
        },
        "fixture": {
            "model": "paired_relu",
            "param_sha256": sha256(args.work_dir / "model.param"),
            "bin_sha256": sha256(args.work_dir / "model.bin"),
            "cpu_library_sha256": sha256(cpu_dir / "libpaired_relu.so"),
            "cpu_header_sha256": sha256(cpu_dir / "paired_relu.h"),
            "cpu_manifest_sha256": sha256(cpu_dir / "paired_relu.json"),
            "vulkan_library_sha256": sha256(vulkan_dir / "libpaired_relu_vulkan.so"),
            "vulkan_header_sha256": sha256(vulkan_dir / "paired_relu_vulkan.h"),
            "vulkan_manifest_sha256": sha256(vulkan_dir / "paired_relu_vulkan.json"),
            "vulkan_plan_sha256": sha256(vulkan_dir / "paired_relu_vulkan.plan.json"),
            "plan_hash": manifest["plan_hash"],
            "shader_sha256": manifest["shader_sha256"],
            "reconstructed_shader_sha256": {
                str(key): value
                for key, value in sorted(reconstructed_shader_hashes.items())
            },
            "shader_files": [
                {
                    "file": f"vulkan/shader-reconstruction/kernel-{key}.spv",
                    "sha256": value,
                }
                for key, value in sorted(reconstructed_shader_hashes.items())
            ],
            "offline_success_artifacts_sha256": artifact_hashes(vulkan_dir),
            "exports": [
                "paired_relu_vk_create",
                "paired_relu_vk_destroy",
                "paired_relu_vk_run",
            ],
            "dependencies": manifest["dependencies"],
        },
        "public_io": {
            "shape": [elements],
            "shape_chw": [1, 1, elements],
            "host_layout": "CHW",
            "element_type": "f32",
            "input": {
                "file": "input.f32",
                "sha256": sha256_bytes(public_input_bytes),
                "values": list(struct.unpack(f"<{elements}f", public_input_bytes)),
            },
            "cpu_output": {
                "file": "cpu-output.f32",
                "sha256": sha256_bytes(cpu_output_bytes),
                "values": cpu_output,
            },
            "vulkan_output": {
                "file": "vulkan-output.f32",
                "sha256": sha256_bytes(vulkan_output_bytes),
                "values": vulkan_output,
            },
            "scalar_golden": {
                "file": "scalar-golden.f32",
                "sha256": sha256_bytes(golden_bytes),
                "values": expected,
            },
            "tolerance": {"absolute": 1.0e-6, "relative": 1.0e-6},
            "max_absolute_error": {
                "cpu_to_golden": max_abs_cpu_error,
                "vulkan_to_golden": max_abs_vulkan_error,
                "cpu_to_vulkan": max_abs_cpu_vulkan_error,
            },
            "max_relative_error": {
                "cpu_to_golden": max(
                    (
                        abs(actual - reference) / max(abs(reference), 1.0e-30)
                        for actual, reference in zip(cpu_output, expected)
                    ),
                    default=0.0,
                ),
                "vulkan_to_golden": max(
                    (
                        abs(actual - reference) / max(abs(reference), 1.0e-30)
                        for actual, reference in zip(vulkan_output, expected)
                    ),
                    default=0.0,
                ),
                "cpu_to_vulkan": max(
                    (
                        abs(actual - reference) / max(abs(reference), 1.0e-30)
                        for actual, reference in zip(cpu_output, vulkan_output)
                    ),
                    default=0.0,
                ),
            },
            "passed": True,
        },
        "validator": {
            "executable": args.spirv_val,
            "version": invoke([args.spirv_val, "--version"]).stdout,
            "shader_hashes": {
                str(key): value
                for key, value in sorted(reconstructed_shader_hashes.items())
            },
        },
        "shader_validations": shader_validations,
        "commands": {
            "vulkaninfo_summary": {
                "argv": [str(item) for item in vulkaninfo_command],
                "returncode": vulkan_info.returncode,
                "stdout_file": "vulkaninfo.stdout.log",
                "stderr_file": "vulkaninfo.stderr.log",
            },
            "vulkan_verify_execution": {
                "argv": [str(item) for item in vulkan_compile_command],
                "returncode": vulkan_compile_result.returncode,
                "stdout_file": "vulkan-verify.stdout.log",
                "stderr_file": "vulkan-verify.stderr.log",
            },
            "product_loader": {
                "stdout_file": "vulkan-loader.stdout.log",
                "stderr_file": "vulkan-loader.stderr.log",
                "result_file": "loader-result.json",
            },
        },
        "validation": {
            "layer": "VK_LAYER_KHRONOS_validation",
            "error_count": validation_error_count_value,
            "log_file": validation_path.name,
            "log_sha256": sha256(validation_path),
            "streams": [
                {"file": name, "sha256": sha256(args.work_dir / name)}
                for name in validation_streams
            ],
        },
        "expected_failure_diagnostics": EXPECTED_FAILURE_LOGS,
        "claims": {
            "cpu_golden": "passed",
            "vulkan_device_execution": "passed",
            "cpu_vulkan_numerical_comparison": "passed",
        },
    }
    evidence_path = args.work_dir / "evidence.json"
    evidence_path.write_text(json.dumps(product_evidence, indent=2, sort_keys=True) + "\n")
    print(f"Vulkan product evidence: {evidence_path}")
    print("Vulkan product CLI, CPU pairing, reproducibility, audit, and failure tests passed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"check_ncnn_compile_vulkan.py: {error}", file=sys.stderr)
        raise SystemExit(1)
