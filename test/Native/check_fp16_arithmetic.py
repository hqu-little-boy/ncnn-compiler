import argparse
import json
import pathlib
import re
import shutil
import subprocess


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def compiler_command(args, output, extra):
    return [
        args.compiler,
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
        "--param",
        args.param,
        "--bin",
        args.bin,
        "--model-name",
        "fp16_policy",
        "--output-dir",
        str(output),
        "--input-shape",
        "1x4x4",
        "--precision",
        "fp16",
        "--fp16-accumulator",
        "f16",
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "compiler",
        "driver",
        "opt",
        "translate",
        "clang",
        "nm",
        "readelf",
        "objdump",
        "size",
        "mca",
        "param",
        "bin",
        "codegen-dir",
        "work-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    work_dir = pathlib.Path(args.work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    rejected = subprocess.run(
        compiler_command(args, work_dir / "rejected", ["--target-feature=+f16c"]),
        capture_output=True,
        text=True,
    )
    if rejected.returncode == 0 or "FP16 arithmetic is not supported" not in rejected.stderr:
        raise RuntimeError(f"unsupported target was not rejected: {rejected.stderr}")

    fallback = run(
        compiler_command(
            args,
            work_dir / "fallback",
            ["--target-feature=+f16c", "--allow-fallback", "--emit=ncnn"],
        ),
        capture_output=True,
        text=True,
    )
    if "using FP32 accumulation" not in fallback.stderr:
        raise RuntimeError("explicit fallback was not reported")
    fallback_ir = (work_dir / "fallback" / "model.ncnn.mlir").read_text()
    if 'ncnn.fp16_accumulator = "f32"' not in fallback_ir:
        raise RuntimeError("fallback policy was not persisted in IR")

    codegen_dir = pathlib.Path(args.codegen_dir)
    linalg_ir = (codegen_dir / "model.linalg.mlir").read_text()
    llvm_ir = (codegen_dir / "model.ll").read_text()
    if "linalg.conv_2d_nhwc_hwcf" not in linalg_ir or "xf16" not in linalg_ir:
        raise RuntimeError("FP16 Linalg convolution was not generated")
    if "fmul half" not in llvm_ir or "fadd half" not in llvm_ir:
        raise RuntimeError("LLVM IR does not contain FP16 multiply and accumulation")

    disassembly = run(
        [args.objdump, "-d", "--no-show-raw-insn", codegen_dir / "model.o"],
        capture_output=True,
        text=True,
    ).stdout
    fp16_instructions = re.findall(r"\bv(?:mul|add|fmadd)[a-z0-9]*h\b", disassembly)
    if not fp16_instructions:
        raise RuntimeError("AVX512-FP16 arithmetic instructions were not generated")

    size_output = run(
        [args.size, "-A", codegen_dir / "model.o"], capture_output=True, text=True
    ).stdout
    sections = {}
    for line in size_output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith(".") and fields[1].isdigit():
            sections[fields[0]] = int(fields[1])
    if sections.get(".text", 0) == 0:
        raise RuntimeError("object contains no executable code")

    mca_output = run(
        [args.mca, "-mcpu=sapphirerapids", codegen_dir / "model.s"],
        capture_output=True,
        text=True,
    ).stdout
    throughput_match = re.search(r"Block RThroughput:\s+([0-9.]+)", mca_output)
    if not throughput_match:
        raise RuntimeError("llvm-mca did not report block throughput")

    report = {
        "schema_version": 1,
        "target_cpu": "sapphirerapids",
        "fp16_accumulator": "f16",
        "error_tolerance": 0.008,
        "block_rthroughput_cycles": float(throughput_match.group(1)),
        "fp16_instruction_count": len(fp16_instructions),
        "artifact_bytes": {
            path.name: path.stat().st_size
            for path in (
                codegen_dir / "libconvolution_codegen_fp16.so",
                codegen_dir / "model.o",
                codegen_dir / "model.s",
            )
        },
        "object_sections_bytes": sections,
        "peak_tensor_memory_bytes": {
            "input_f32": 1 * 4 * 4 * 4,
            "input_f16_padded": 1 * 6 * 6 * 1 * 2,
            "weights_f16": 2 * 1 * 3 * 3 * 2,
            "accumulator_f16": 1 * 4 * 4 * 2 * 2,
            "output_f32": 2 * 4 * 4 * 4,
            "total": 364,
        },
    }
    (work_dir / "fp16-performance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
