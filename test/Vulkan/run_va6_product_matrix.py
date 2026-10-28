#!/usr/bin/env python3
"""VA6.1 product support-matrix positive/negative migration evidence.

Every case drives the public product CLI (``ncnn-compile --backend=vulkan``),
never a private harness.  Negative fixtures enumerate the frozen support
matrix rows that must keep failing closed (broadcast Add, invalid Concat
axis/non-axis extents, out-of-subset MaxPool/AveragePool parameters, Conv2D
kernel/stride/dilation/padding bounds, dynamic input extents, size-changing
Reshape, unsupported operations, and the ResNet18 first 7x7 convolution as
the no-new-capability probe).  Each negative expects a nonzero exit, a
diagnostic, and zero published artifacts.  Each positive control is the
neighboring in-subset form compiled through the same CLI; it must publish
the fixed-run artifacts whose manifest pins the VA5+ host contract.  Numeric
execution of the positive forms is carried by the v5 device suites and the
VA6 cross-ICD test; this runner deliberately does not re-run numerics.
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

SCHEMA_VERSION = 1
EVIDENCE_KIND = "ncnn.vulkan_va6_product_matrix_evidence"
# Published-product host contract pinned by the compiler since VA5.
HOST_CONTRACT = {
    "host_implementation": "model-fixed-run",
    "host_codegen_version": 2,
    "host_source_codegen_version": "ncnn-vulkan-fixed-run-codegen/v1",
    "execution_plan_usage": "build-evidence-only",
}
PUBLISHED_ARTIFACTS = (
    "{model}_vulkan.h",
    "lib{model}_vulkan.so",
    "{model}_vulkan.json",
    "{model}_vulkan.plan.json",
)


class MatrixError(RuntimeError):
    """Raised when a product-matrix case violates its acceptance check."""


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def floats_bin(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def conv_weights(output_channels: int, kernel: int, input_channels: int,
                 *, bias: bool, seed: int = 0) -> bytes:
    """NCNN weight payload: flag word, OIHW filter, optional bias block."""
    count = output_channels * kernel * kernel * input_channels
    weights = [((index * 37 + seed * 11) % 13 - 6) / 7.0 for index in range(count)]
    values = weights
    if bias:
        values = weights + [
            ((channel + seed) % 5 - 2) / 5.0 for channel in range(output_channels)]
    return struct.pack("<I", 0) + floats_bin(values)


def make_param(inputs: list[tuple[str, tuple[int, ...]]],
               layers: list[dict]) -> str:
    """Serialize a minimal NCNN .param document (same shape as run_v5)."""
    full_layers = []
    for name, shape in inputs:
        if shape is None:
            # Dynamic extents: the Input layer carries no dimension params.
            parameters = ""
        elif len(shape) == 3:
            channels, height, width = shape
            parameters = f"0={width} 1={height} 2={channels}"
        elif len(shape) == 2:
            height, width = shape
            parameters = f"0={width} 1={height}"
        elif len(shape) == 1:
            parameters = f"0={shape[0]}"
        else:
            raise ValueError("VA6 fixtures support ranks one through three")
        full_layers.append(f"Input input_{name} 0 1 {name} {parameters}".rstrip())
    full_layers.extend(layer["line"] for layer in layers)
    blobs: set[str] = set()
    for name, _ in inputs:
        blobs.add(name)
    for layer in layers:
        blobs.update(layer["tops"])
    return "7767517\n{} {}\n{}\n".format(
        len(full_layers), len(blobs), "\n".join(full_layers))


def matrix_cases() -> list[dict]:
    """The product support-matrix migration rows.

    Each negative pins the fail-closed behavior of one matrix row; each
    positive is the neighboring accepted form that the same CLI must still
    publish.  ``expect_any`` lists acceptable diagnostic substrings; the
    empirical set is frozen after running against the real compiler.
    """
    relu_layer = {"line": "ReLU relu 1 1 data output 0=0.000000", "tops": ["output"]}
    cases: list[dict] = []

    def negative(name, inputs, layers, bin_bytes, expect_any):
        cases.append({"name": name, "kind": "negative", "inputs": inputs,
                      "layers": layers, "bin": bin_bytes, "expect_any": expect_any})

    def positive(name, inputs, layers, bin_bytes):
        cases.append({"name": name, "kind": "positive", "inputs": inputs,
                      "layers": layers, "bin": bin_bytes, "expect_any": []})

    # ── Add: broadcasting stays rejected; same-shape Add still compiles ──
    negative(
        "broadcast_add",
        [("left", (1, 1, 4)), ("right", (1, 1, 8))],
        [{"line": "BinaryOp add 2 1 left right sum 0=0", "tops": ["sum"]}],
        b"",
        ["not broadcastable"])
    positive(
        "same_shape_add",
        [("left", (1, 1, 4)), ("right", (1, 1, 4))],
        [{"line": "BinaryOp add 2 1 left right sum 0=0", "tops": ["sum"]}],
        b"")

    # ── Concat: invalid axis / mismatched non-axis extents ──
    negative(
        "concat_bad_axis",
        [("top", (1, 2, 2)), ("bottom", (1, 2, 2))],
        [{"line": "Concat join 2 1 top bottom output 0=3", "tops": ["output"]}],
        b"",
        ["concat axis is outside operand rank"])
    negative(
        "concat_non_axis_mismatch",
        [("top", (1, 2, 2)), ("bottom", (1, 3, 2))],
        [{"line": "Concat join 2 1 top bottom output 0=0", "tops": ["output"]}],
        b"",
        ["concat non-axis dimensions must match"])
    positive(
        "concat_channel",
        [("top", (1, 2, 2)), ("bottom", (2, 2, 2))],
        [{"line": "Concat join 2 1 top bottom output 0=0", "tops": ["output"]}],
        b"")

    # ── MaxPool: kernel/stride/padding subset ──
    negative(
        "maxpool_bad_kernel",
        [("data", (1, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=0 1=2 2=2", "tops": ["output"]}],
        b"",
        ["supports only a 3x3 kernel"])
    negative(
        "maxpool_bad_stride",
        [("data", (1, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=0 1=3 2=1", "tops": ["output"]}],
        b"",
        ["supports only stride 2x2"])
    negative(
        "maxpool_bad_padding",
        [("data", (1, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=0 1=3 2=2 13=1", "tops": ["output"]}],
        b"",
        ["requires zero top and left padding"])
    positive(
        "maxpool_tail",
        [("data", (1, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=0 1=3 2=2", "tops": ["output"]}],
        b"")

    # ── AveragePool: only the global spatial reduction is accepted ──
    negative(
        "regular_avgpool",
        [("data", (1, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=1 1=3 2=2", "tops": ["output"]}],
        b"",
        ["requires a full-spatial kernel and 1x1 output"])
    negative(
        "adaptive_avgpool",
        [("data", (1, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=1 2=1 4=0 7=1 8=2",
          "tops": ["output"]}],
        b"",
        ["not supported by vulkan graph"])
    positive(
        "global_avgpool",
        [("data", (2, 8, 8))],
        [{"line": "Pooling pool 1 1 data output 0=1 1=1 2=1 4=1 5=0 6=0",
          "tops": ["output"]}],
        b"")

    # ── Conv2D parameter subset; ResNet first conv as the no-new-capability
    # probe (7x7 stride-2 must keep failing closed) ──
    negative(
        "conv_kernel5",
        [("data", (1, 6, 6))],
        [{"line": "Convolution conv 1 1 data output "
                  "0=1 1=5 11=5 2=1 3=1 4=2 14=2 15=2 16=2 5=1 6=25",
          "tops": ["output"]}],
        conv_weights(1, 5, 1, bias=True, seed=1),
        ["supports only 1x1 and 3x3 kernels"])
    negative(
        "resnet_first_conv_7x7",
        [("data", (3, 224, 224))],
        [{"line": "Convolution convrelu_0 1 1 data output "
                  "0=64 1=7 11=7 2=1 3=2 13=2 4=3 14=3 15=3 16=3 5=0 6=9408",
          "tops": ["output"]}],
        conv_weights(64, 7, 3, bias=False, seed=2),
        ["supports only 1x1 and 3x3 kernels"])
    negative(
        "conv_stride3",
        [("data", (1, 6, 6))],
        [{"line": "Convolution conv 1 1 data output "
                  "0=1 1=3 11=3 3=3 13=3 5=1 6=9",
          "tops": ["output"]}],
        conv_weights(1, 3, 1, bias=True, seed=3),
        ["stride must be one or two"])
    negative(
        "conv_dilation2",
        [("data", (1, 6, 6))],
        [{"line": "Convolution conv 1 1 data output "
                  "0=1 1=3 11=3 2=2 12=2 3=1 5=1 6=9",
          "tops": ["output"]}],
        conv_weights(1, 3, 1, bias=True, seed=4),
        ["static padding/stride and dilation one"])
    negative(
        "conv_padding2",
        [("data", (1, 6, 6))],
        [{"line": "Convolution conv 1 1 data output "
                  "0=1 1=3 11=3 3=1 4=2 14=2 15=2 16=2 5=1 6=9",
          "tops": ["output"]}],
        conv_weights(1, 3, 1, bias=True, seed=5),
        ["padding is outside the supported 1x1/3x3 subset"])
    positive(
        "conv_3x3_stride2",
        [("data", (1, 4, 4))],
        [{"line": "Convolution conv 1 1 data output "
                  "0=2 1=3 2=1 3=2 4=1 5=1 6=18 11=3 13=2 14=1 15=1 16=1",
          "tops": ["output"]}],
        conv_weights(2, 3, 1, bias=True, seed=6))
    positive(
        "conv_1x1",
        [("data", (2, 3, 5))],
        [{"line": "Convolution conv 1 1 data output "
                  "0=1 1=1 2=1 3=2 4=0 5=1 6=2",
          "tops": ["output"]}],
        conv_weights(1, 1, 2, bias=True, seed=7))

    # ── Softmax: in-subset rank forms stay accepted ──
    positive(
        "softmax_rank1",
        [("data", (16, 1, 1))],
        [{"line": "Pooling global_vector 1 1 data vector 0=1 1=0 2=1 3=0 4=1",
          "tops": ["vector"]},
         {"line": "Softmax softmax 1 1 vector prob 0=0 1=1", "tops": ["prob"]}],
        b"")

    # ── Model boundary: static extents are mandatory ──
    negative(
        "dynamic_input",
        [("data", None)],
        [relu_layer],
        b"",
        ["requires static tensor dimensions"])

    # ── Reshape: linear view only ──
    negative(
        "reshape_size_change",
        [("data", (1, 1, 4))],
        [{"line": "Reshape reshape 1 1 data output 0=5 1=1 2=1",
          "tops": ["output"]}],
        b"",
        ["reshape element count does not match input"])

    # ── Unsupported operation keeps its source-layer diagnostic ──
    negative(
        "unsupported_sigmoid",
        [("data", (1, 1, 8))],
        [{"line": "Sigmoid unsupported_sigmoid 1 1 data output", "tops": ["output"]}],
        b"",
        ["only fp32 relu clamp [0,+inf]"])
    positive(
        "relu",
        [("data", (1, 1, 8))],
        [relu_layer],
        b"")
    return cases


def compiler_environment(args) -> dict:
    """The compiler process may need shared LLVM/sanitizer runtimes."""
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


def write_fixture(work_dir: Path, case: dict) -> tuple[Path, Path]:
    name = case["name"]
    fixtures = work_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    param = fixtures / f"{name}.param"
    binary = fixtures / f"{name}.bin"
    param.write_text(make_param(case["inputs"], case["layers"]), encoding="ascii")
    binary.write_bytes(case["bin"])
    return param, binary


def run_case(args, case: dict, env: dict) -> dict:
    name = case["name"]
    model = f"va6_{name}"
    output_dir = args.work_dir / "published" / name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    param, binary = write_fixture(args.work_dir / "fixtures", case)
    command = [
        args.compiler, str(param),
        "--bin", str(binary),
        "--model-name", model,
        "--output-dir", str(output_dir),
        "--backend=vulkan",
        "--precision=f32",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    record = {
        "kind": case["kind"],
        "model": model,
        "argv": [str(item) for item in command],
        "param_sha256": sha256(param),
        "bin_sha256": sha256(binary),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "published": sorted(
            entry.name for entry in output_dir.iterdir())
        if output_dir.is_dir() else [],
        "published_sha256": {
            entry.name: sha256(entry)
            for entry in sorted(output_dir.iterdir())
            if entry.is_file()} if output_dir.is_dir() else {},
    }
    if case["kind"] == "negative":
        if result.returncode == 0:
            raise MatrixError(f"{name}: negative case compiled successfully")
        if not result.stderr.strip():
            raise MatrixError(f"{name}: negative case produced no diagnostic")
        lowered = result.stderr.lower()
        if not any(token in lowered for token in case["expect_any"]):
            raise MatrixError(
                f"{name}: diagnostic does not match the frozen matrix row; "
                f"expected one of {case['expect_any']}\n{result.stderr}")
        if record["published"]:
            raise MatrixError(
                f"{name}: negative case published artifacts: {record['published']}")
    else:
        if result.returncode != 0:
            raise MatrixError(
                f"{name}: positive control failed to compile\n{result.stderr}")
        expected = [template.format(model=model) for template in PUBLISHED_ARTIFACTS]
        missing = [item for item in expected if item not in record["published"]]
        if missing:
            raise MatrixError(
                f"{name}: positive control is missing artifacts: {missing}")
        manifest_path = output_dir / f"{model}_vulkan.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise MatrixError(
                f"{name}: published manifest is unreadable: {error}") from error
        for key, expected_value in HOST_CONTRACT.items():
            if manifest.get(key) != expected_value:
                raise MatrixError(
                    f"{name}: manifest {key} differs from the fixed-run "
                    f"contract: actual={manifest.get(key)!r} "
                    f"expected={expected_value!r}")
        if not manifest.get("host_codegen_identity"):
            raise MatrixError(f"{name}: manifest host_codegen_identity is missing")
        if any(str(item).startswith("libLLVM") for item in manifest.get(
                "dependencies", [])):
            raise MatrixError(f"{name}: manifest records an LLVM dependency")
        record["host_contract"] = dict(HOST_CONTRACT)
        record["host_codegen_identity"] = manifest.get("host_codegen_identity")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--sanitizer-runtime-dir", type=Path)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    env = compiler_environment(args)
    cases = matrix_cases()
    records = {}
    for case in cases:
        records[case["name"]] = run_case(args, case, env)
    negatives = [name for name, record in records.items()
                 if record["kind"] == "negative"]
    positives = [name for name, record in records.items()
                 if record["kind"] == "positive"]
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "host_contract": dict(HOST_CONTRACT),
        "negative_cases": negatives,
        "positive_controls": positives,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VA6 product matrix evidence: {args.output} "
          f"({len(negatives)} negative / {len(positives)} positive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
