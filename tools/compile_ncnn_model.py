import argparse
import json
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


def c_identifier(name):
    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    c_keywords = {
        "auto", "break", "case", "char", "const", "continue", "default",
        "do", "double", "else", "enum", "extern", "float", "for", "goto",
        "if", "inline", "int", "long", "register", "restrict", "return",
        "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
        "union", "unsigned", "void", "volatile", "while", "_Alignas",
        "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary",
        "_Noreturn", "_Static_assert", "_Thread_local",
    }
    if (
        not result
        or result[0].isdigit()
        or result.startswith("_")
        or result in c_keywords
    ):
        result = "ncnn_" + result
    return result


def element_count(argument):
    count = 1
    for dimension in argument["shape"]:
        count *= dimension
    return count


def write_header(path, manifest):
    function = manifest["function"]
    guard = f"NCNN_{function.upper()}_H"
    parameters = []
    for argument in manifest["inputs"]:
        parameters.append(f"const float *{argument['name']}")
    for argument in manifest["outputs"]:
        parameters.append(f"float *{argument['name']}")
    declaration = f"int {function}({', '.join(parameters)});"
    sizes = []
    for argument in manifest["inputs"] + manifest["outputs"]:
        macro = f"{function}_{argument['name']}_elements".upper()
        sizes.append(f"#define {macro} {element_count(argument)}")
    path.write_text(
        f"#ifndef {guard}\n#define {guard}\n\n"
        + "\n".join(sizes)
        + "\n\n"
        "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
        f"{declaration}\n\n"
        "#ifdef __cplusplus\n}\n#endif\n\n"
        f"#endif  // {guard}\n",
        encoding="ascii",
    )


def write_harness(path, header_name, manifest):
    function = manifest["function"]
    arguments = manifest["inputs"] + manifest["outputs"]
    declarations = []
    names = []
    for argument in arguments:
        declarations.append(
            f"  float *{argument['name']} = calloc({element_count(argument)}, sizeof(float));"
        )
        declarations.append(f"  if (!{argument['name']}) return 2;")
        names.append(argument["name"])
    null_checks = []
    for index in range(len(names)):
        call_arguments = names.copy()
        call_arguments[index] = "NULL"
        null_checks.append(
            f"  if ({function}({', '.join(call_arguments)}) == 0) return 3;"
        )
    frees = [f"  free({name});" for name in names]
    finite_checks = []
    for argument in manifest["outputs"]:
        count = element_count(argument)
        finite_checks.append(
            f"  for (size_t i = 0; i < {count}; ++i) "
            f"if (!isfinite({argument['name']}[i])) return 5;"
        )
    path.write_text(
        "#include <math.h>\n#include <stddef.h>\n#include <stdlib.h>\n"
        f'#include "{header_name}"\n\n'
        "int main(void) {\n"
        + "\n".join(declarations)
        + "\n"
        + f"  if ({function}({', '.join(names)}) != 0) return 4;\n"
        + "\n".join(finite_checks)
        + "\n"
        + "\n".join(null_checks)
        + "\n"
        + "\n".join(frees)
        + "\n  return 0;\n}\n",
        encoding="ascii",
    )


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
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir", default="build/native-model")
    parser.add_argument("--verify-execution", action="store_true")
    parser.add_argument("--expected-undefined")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = c_identifier(args.model_name or pathlib.Path(args.param).stem)
    ncnn_ir = output_dir / "model.ncnn.mlir"
    tosa_ir = output_dir / "model.tosa.mlir"
    linalg_ir = output_dir / "model.linalg.mlir"
    memref_ir = output_dir / "model.memref.mlir"
    capi_ir = output_dir / "model.capi.mlir"
    llvm_dialect_ir = output_dir / "model.llvm.mlir"
    llvm_ir = output_dir / "model.ll"
    object_file = output_dir / "model.o"
    manifest_path = output_dir / f"{model_name}.json"
    header_path = output_dir / f"{model_name}.h"
    exports_path = output_dir / "exports.map"
    library = output_dir / f"lib{model_name}.so"

    run([args.driver, args.param, "--bin", args.bin, "-o", ncnn_ir])
    run([args.opt, "--ncnn-to-tosa-pipeline", ncnn_ir, "-o", tosa_ir])
    run([args.opt, "--ncnn-tosa-to-linalg-pipeline", tosa_ir, "-o", linalg_ir])
    run([args.opt, "--ncnn-linalg-to-memref-pipeline", linalg_ir, "-o", memref_ir])
    capi_option = (
        f"--generate-ncnn-c-api=export-name={model_name} "
        f"manifest-path={manifest_path}"
    )
    run([args.opt, capi_option, memref_ir, "-o", capi_ir])
    run([args.opt, "--ncnn-memref-to-llvm-pipeline", capi_ir, "-o", llvm_dialect_ir])
    run([args.translate, "--mlir-to-llvmir", llvm_dialect_ir, "-o", llvm_ir])
    run([args.clang, "-x", "ir", "-fPIC", "-c", llvm_ir, "-o", object_file])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_header(header_path, manifest)
    exports_path.write_text(
        f"{{\n  global: {model_name};\n  local: *;\n}};\n", encoding="ascii"
    )

    run(
        [
            args.clang,
            "-shared",
            "-nostdlib",
            object_file,
            "-Wl,-z,defs",
            "-Wl,--no-undefined",
            "-Wl,--build-id=none",
            f"-Wl,--version-script={exports_path}",
            "-lc",
            "-lm",
            "-o",
            library,
        ]
    )

    undefined = symbols(capture([args.nm, "-D", "--undefined-only", library]))
    allowed_undefined = {"expf", "free", "malloc"}
    if not undefined.issubset(allowed_undefined):
        print(
            f"unexpected undefined symbols: {sorted(undefined - allowed_undefined)}",
            file=sys.stderr,
        )
        return 1
    if args.expected_undefined:
        expected = set(args.expected_undefined.split(","))
        if undefined != expected:
            print(
                f"undefined symbols {sorted(undefined)} do not match expected "
                f"{sorted(expected)}",
                file=sys.stderr,
            )
            return 1

    defined = symbols(capture([args.nm, "-D", "--defined-only", library]))
    if defined != {model_name}:
        print(f"unexpected exported symbols: {sorted(defined)}", file=sys.stderr)
        return 1

    needed_output = capture([args.readelf, "--needed-libs", library])
    needed = set(re.findall(r"lib[^\s]+\.so(?:\.\d+)*", needed_output))
    if not needed.issubset({"libc.so.6", "libm.so.6"}):
        print(f"unexpected shared library dependencies: {sorted(needed)}", file=sys.stderr)
        return 1

    forbidden = ("memrefCopy", "runner_utils", "RunnerUtils", "ncnn_runtime")
    symbol_text = capture([args.nm, "-D", library])
    if any(name in symbol_text for name in forbidden):
        print("forbidden runtime symbol found in shared library", file=sys.stderr)
        return 1
    if args.verify_execution:
        harness = output_dir / "harness.c"
        executable = output_dir / "harness"
        write_harness(harness, header_path.name, manifest)
        run(
            [
                args.clang,
                "-std=c23",
                harness,
                "-I",
                output_dir,
                "-L",
                output_dir,
                f"-l{model_name}",
                f"-Wl,-rpath,{output_dir.resolve()}",
                "-lm",
                "-o",
                executable,
            ]
        )
        run([executable])
    return 0


if __name__ == "__main__":
    sys.exit(main())
