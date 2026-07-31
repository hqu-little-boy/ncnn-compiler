import argparse
import pathlib
import re
import shutil
import subprocess
import sys


def run(command):
    subprocess.run(command, check=True)


def capture(command):
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def symbols(output):
    result = set()
    for line in output.splitlines():
        match = re.search(r"\b[UTW]\s+(\S+)", line)
        if match:
            result.add(match.group(1).split("@")[0])
    return result


def find_tool(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return names[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="build/tools/ncnn-mlir-driver")
    parser.add_argument("--opt", default="build/bin/ncnn-mlir-opt")
    parser.add_argument("--translate", default=find_tool("mlir-translate-21", "mlir-translate"))
    parser.add_argument("--clang", default=find_tool("clang-21", "clang"))
    parser.add_argument("--nm", default=find_tool("llvm-nm-21", "llvm-nm"))
    parser.add_argument("--readelf", default=find_tool("llvm-readelf-21", "llvm-readelf"))
    parser.add_argument("--param", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--exports", default="test/Native/exports.map")
    parser.add_argument("--output-dir", default="build/native-model")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ncnn_ir = output_dir / "model.ncnn.mlir"
    tosa_ir = output_dir / "model.tosa.mlir"
    linalg_ir = output_dir / "model.linalg.mlir"
    memref_ir = output_dir / "model.memref.mlir"
    llvm_dialect_ir = output_dir / "model.llvm.mlir"
    llvm_ir = output_dir / "model.ll"
    object_file = output_dir / "model.o"
    library = output_dir / "libncnn_model.so"

    run([args.driver, args.param, "--bin", args.bin, "-o", ncnn_ir])
    run([args.opt, "--ncnn-to-tosa-pipeline", ncnn_ir, "-o", tosa_ir])
    run([args.opt, "--ncnn-tosa-to-linalg-pipeline", tosa_ir, "-o", linalg_ir])
    run([args.opt, "--ncnn-linalg-to-memref-pipeline", linalg_ir, "-o", memref_ir])
    run([args.opt, "--ncnn-memref-to-llvm-pipeline", memref_ir, "-o", llvm_dialect_ir])
    run([args.translate, "--mlir-to-llvmir", llvm_dialect_ir, "-o", llvm_ir])
    run([args.clang, "-x", "ir", "-fPIC", "-c", llvm_ir, "-o", object_file])
    run(
        [
            args.clang,
            "-shared",
            "-nostdlib",
            object_file,
            "-Wl,-z,defs",
            "-Wl,--no-undefined",
            "-Wl,--build-id=none",
            f"-Wl,--version-script={args.exports}",
            "-lc",
            "-lm",
            "-o",
            library,
        ]
    )

    undefined = symbols(capture([args.nm, "-D", "--undefined-only", library]))
    allowed_undefined = {"expf", "free", "malloc"}
    if undefined != allowed_undefined:
        print(
            f"unexpected undefined symbols: {sorted(undefined)}; "
            f"expected {sorted(allowed_undefined)}",
            file=sys.stderr,
        )
        return 1

    defined = symbols(capture([args.nm, "-D", "--defined-only", library]))
    if defined != {"_mlir_ciface_model"}:
        print(f"unexpected exported symbols: {sorted(defined)}", file=sys.stderr)
        return 1

    needed_output = capture([args.readelf, "--needed-libs", library])
    needed = set(re.findall(r"lib[^\s]+\.so(?:\.\d+)*", needed_output))
    if needed != {"libc.so.6", "libm.so.6"}:
        print(f"unexpected shared library dependencies: {sorted(needed)}", file=sys.stderr)
        return 1

    forbidden = ("memrefCopy", "runner_utils", "RunnerUtils", "ncnn_runtime")
    symbol_text = capture([args.nm, "-D", library])
    if any(name in symbol_text for name in forbidden):
        print("forbidden runtime symbol found in shared library", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
