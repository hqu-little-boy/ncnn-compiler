import argparse
import atexit
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

VERBOSE = False


def run(command):
    if VERBOSE:
        print(shlex.join(str(item) for item in command), file=sys.stderr)
    subprocess.run(command, check=True)


def capture(command):
    if VERBOSE:
        print(shlex.join(str(item) for item in command), file=sys.stderr)
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


def nearby_tool(relative_paths, *fallbacks):
    for relative_path in relative_paths:
        candidate = pathlib.Path(__file__).parent / relative_path
        if candidate.exists():
            return str(candidate.resolve())
    return find_tool(*fallbacks)


def normalize_passthrough_args(arguments):
    result = []
    passthrough = {"--target-feature", "--clang-arg", "--linker-arg"}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if (
            argument in passthrough
            and index + 1 < len(arguments)
            and arguments[index + 1].startswith("-")
        ):
            result.append(f"{argument}={arguments[index + 1]}")
            index += 2
            continue
        result.append(argument)
        index += 1
    return result


def is_generated_output(path, model_name):
    name = path.name
    return (
        name.startswith(".ncnn-compile-")
        or name in {f"lib{model_name}.so", f"{model_name}.h", f"{model_name}.json"}
        or name in {"libncnn_model.so", "ncnn_model.h", "ncnn_model.json"}
        or name
        in {
            "model.ncnn.mlir",
            "model.tosa.mlir",
            "model.linalg.mlir",
            "model.memref.mlir",
            "model.capi.mlir",
            "model.llvm.mlir",
            "model.ll",
            "model.o",
        }
        or name in {"exports.map", "harness", "harness.c"}
    )


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
    parser = argparse.ArgumentParser(
        prog="ncnn-compile",
        description="Compile an ncnn .param/.bin model into a C header and shared library."
    )
    parser.add_argument("input", nargs="?", help="Input .param file")
    parser.add_argument(
        "--driver",
        default=nearby_tool(
            ("ncnn-mlir-driver", "../build/tools/ncnn-mlir-driver"),
            "ncnn-mlir-driver",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--opt",
        default=nearby_tool(
            ("../bin/ncnn-mlir-opt", "../build/bin/ncnn-mlir-opt"),
            "ncnn-mlir-opt",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--translate",
        default=find_tool("mlir-translate-21", "mlir-translate"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--clang", default=find_tool("clang-21", "clang"))
    parser.add_argument(
        "--nm", default=find_tool("llvm-nm-21", "llvm-nm"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--readelf",
        default=find_tool("llvm-readelf-21", "llvm-readelf"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--param", help=argparse.SUPPRESS)
    parser.add_argument("--bin")
    parser.add_argument("--model-name")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument(
        "--emit",
        action="append",
        help="Keep intermediate MLIR stages; repeat or comma-separate values",
    )
    parser.add_argument("-O", dest="optimization", choices=("0", "1", "2", "3"), default="3")
    parser.add_argument("--target-triple")
    parser.add_argument("--march")
    parser.add_argument("--mcpu")
    parser.add_argument("--mtune")
    parser.add_argument("--target-feature", action="append", default=[])
    parser.add_argument("--sysroot")
    parser.add_argument("-g", "--debug-info", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--clang-arg", action="append", default=[])
    parser.add_argument("--linker-arg", action="append", default=[])
    parser.add_argument("--verify-execution", action="store_true")
    parser.add_argument("--expected-undefined", help=argparse.SUPPRESS)
    parser.add_argument(
        "--emit-manifest", action="store_true", help="Emit the JSON ABI manifest"
    )
    args = parser.parse_args(normalize_passthrough_args(sys.argv[1:]))

    global VERBOSE
    VERBOSE = args.verbose

    if args.input and args.param:
        parser.error("use either positional input or --param, not both")
    param_path = args.param or args.input
    if not param_path:
        parser.error("an input .param file is required")
    bin_path = args.bin
    if not bin_path:
        bin_path = str(pathlib.Path(param_path).with_suffix(".bin"))
    if not pathlib.Path(param_path).is_file():
        parser.error(f"input file does not exist: {param_path}")
    if not pathlib.Path(bin_path).is_file():
        parser.error(f"weight file does not exist: {bin_path}")

    model_name = c_identifier(args.model_name or pathlib.Path(param_path).stem)
    output_dir = pathlib.Path(args.output_dir or model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected_outputs = [
        path for path in output_dir.iterdir() if not is_generated_output(path, model_name)
    ]
    if unexpected_outputs:
        parser.error(
            "output directory contains files not owned by ncnn-compile: "
            + ", ".join(sorted(path.name for path in unexpected_outputs))
        )

    if args.target_triple:
        triple = args.target_triple.lower()
        architecture = triple.split("-", 1)[0]
        if "linux" not in triple:
            parser.error("--target-triple currently supports Linux ELF targets only")
        if "64" not in architecture and architecture not in {"s390x"}:
            parser.error("--target-triple currently supports 64-bit targets only")

    emit = []
    for value in args.emit or []:
        emit.extend(value.split(","))
    valid_emit = {"ncnn", "tosa", "linalg", "memref", "capi", "llvm", "all"}
    invalid_emit = set(emit) - valid_emit
    if invalid_emit:
        parser.error(f"invalid --emit stage(s): {', '.join(sorted(invalid_emit))}")
    if "all" in emit:
        emit = ["ncnn", "tosa", "linalg", "memref", "capi", "llvm"]
    emit = set(emit)

    stage_names = (
        "model.ncnn.mlir",
        "model.tosa.mlir",
        "model.linalg.mlir",
        "model.memref.mlir",
        "model.capi.mlir",
        "model.llvm.mlir",
    )
    staging_dir = pathlib.Path(tempfile.mkdtemp(prefix=".ncnn-compile-", dir=output_dir))
    atexit.register(shutil.rmtree, staging_dir, True)
    ncnn_ir = staging_dir / "model.ncnn.mlir"
    tosa_ir = staging_dir / "model.tosa.mlir"
    linalg_ir = staging_dir / "model.linalg.mlir"
    memref_ir = staging_dir / "model.memref.mlir"
    capi_ir = staging_dir / "model.capi.mlir"
    llvm_dialect_ir = staging_dir / "model.llvm.mlir"
    llvm_ir = staging_dir / "model.ll"
    object_file = staging_dir / "model.o"
    manifest_path = staging_dir / f"{model_name}.json"
    header_path = staging_dir / f"{model_name}.h"
    exports_path = staging_dir / "exports.map"
    library = staging_dir / f"lib{model_name}.so"

    run([args.driver, param_path, "--bin", bin_path, "-o", ncnn_ir])
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
    target_args = []
    codegen_args = []
    if args.target_triple:
        target_args.append(f"--target={args.target_triple}")
    if args.march:
        codegen_args.append(f"-march={args.march}")
    if args.mcpu:
        codegen_args.append(f"-mcpu={args.mcpu}")
    if args.mtune:
        codegen_args.append(f"-mtune={args.mtune}")
    for feature in args.target_feature:
        codegen_args.extend(["-Xclang", "-target-feature", "-Xclang", feature])
    if args.sysroot:
        target_args.append(f"--sysroot={args.sysroot}")
    optimization = f"-O{args.optimization}"
    if args.debug_info:
        codegen_args.append("-g")
    run(
        [args.clang, "-x", "ir", "-fPIC", optimization]
        + target_args
        + codegen_args
        + args.clang_arg
        + ["-c", llvm_ir, "-o", object_file]
    )

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
            optimization,
            *target_args,
            object_file,
            *args.linker_arg,
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
    allowed_undefined = {"expf", "free", "malloc", "memcpy", "memset"}
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
    if any(not name.startswith(("libc.so", "libm.so")) for name in needed):
        print(f"unexpected shared library dependencies: {sorted(needed)}", file=sys.stderr)
        return 1

    forbidden = ("memrefCopy", "runner_utils", "RunnerUtils", "ncnn_runtime")
    symbol_text = capture([args.nm, "-D", library])
    if any(name in symbol_text for name in forbidden):
        print("forbidden runtime symbol found in shared library", file=sys.stderr)
        return 1
    if args.verify_execution:
        harness = staging_dir / "harness.c"
        executable = staging_dir / "harness"
        write_harness(harness, header_path.name, manifest)
        run(
            [
                args.clang,
                "-std=c23",
                harness,
                "-I",
                staging_dir,
                "-L",
                staging_dir,
                f"-l{model_name}",
                f"-Wl,-rpath,{staging_dir.resolve()}",
                "-lm",
                "-o",
                executable,
            ]
        )
        run([executable])

    stage_paths = {
        "ncnn": ncnn_ir,
        "tosa": tosa_ir,
        "linalg": linalg_ir,
        "memref": memref_ir,
        "capi": capi_ir,
        "llvm": llvm_dialect_ir,
    }
    for path in output_dir.iterdir():
        if path.resolve() != staging_dir.resolve():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    shutil.copy2(header_path, output_dir / header_path.name)
    shutil.copy2(library, output_dir / library.name)
    public_manifest = output_dir / manifest_path.name
    if args.emit_manifest:
        shutil.copy2(manifest_path, public_manifest)
    else:
        public_manifest.unlink(missing_ok=True)
    for stage, path in stage_paths.items():
        if stage in emit:
            shutil.copy2(path, output_dir / path.name)
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(output_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        command = shlex.join(str(item) for item in error.cmd)
        print(
            f"error: command failed with exit code {error.returncode}: {command}",
            file=sys.stderr,
        )
        sys.exit(error.returncode or 1)
    except FileNotFoundError as error:
        print(f"error: file not found: {error.filename}", file=sys.stderr)
        sys.exit(1)
