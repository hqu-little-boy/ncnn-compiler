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
    shape = argument.get("maximum_shape", argument["shape"])
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"{argument['name']} has a dynamic element count")
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def c_type(argument):
    types = {
        "f16": "ncnn_float16_t", "bf16": "ncnn_bfloat16_t",
        "f32": "float", "f64": "double", "i8": "int8_t",
        "i16": "int16_t", "i32": "int32_t", "i64": "int64_t",
        "ui8": "uint8_t", "ui16": "uint16_t", "ui32": "uint32_t",
        "ui64": "uint64_t",
    }
    return types[argument["element_type"]]


def macro_name(function, argument, suffix):
    return f"{function}_{argument['name']}_{suffix}".upper()


def write_header(path, manifest):
    function = manifest["function"]
    guard = f"NCNN_{function.upper()}_H"
    parameters = []
    for argument in manifest["inputs"]:
        parameters.append(f"const {c_type(argument)} *{argument['name']}")
        if argument.get("dynamic_rank"):
            parameters.append(f"const int64_t *{argument['name']}_shape")
            parameters.append(f"uint32_t {argument['name']}_rank")
        elif argument["dynamic_dim_mask"]:
            rank = macro_name(function, argument, "rank")
            parameters.append(f"const int64_t {argument['name']}_shape[{rank}]")
    for argument in manifest["outputs"]:
        parameters.append(f"{c_type(argument)} *{argument['name']}")
    for argument in manifest["outputs"]:
        if (
            (argument.get("dynamic_rank") or argument["dynamic_dim_mask"])
            and not argument.get("shape_depends_on_data", False)
        ):
            parameters.append(f"uint64_t {argument['name']}_capacity")
    for argument in manifest["outputs"]:
        if argument.get("shape_depends_on_data", False):
            parameters.extend([
                f"int64_t *{argument['name']}_shape",
                f"uint32_t {argument['name']}_shape_capacity",
                f"uint32_t *{argument['name']}_rank",
            ])
    declaration = f"int {function}({', '.join(parameters)});"
    dynamic_outputs = [
        argument for argument in manifest["outputs"]
        if (argument.get("dynamic_rank") or argument["dynamic_dim_mask"])
        and not argument.get("shape_depends_on_data", False)
    ]
    shape_declaration = ""
    if dynamic_outputs:
        shape_parameters = []
        for argument in manifest["inputs"]:
            if argument.get("dynamic_rank"):
                shape_parameters.extend([
                    f"const int64_t *{argument['name']}_shape",
                    f"uint32_t {argument['name']}_rank",
                ])
            elif argument["dynamic_dim_mask"]:
                rank = macro_name(function, argument, "rank")
                shape_parameters.append(
                    f"const int64_t {argument['name']}_shape[{rank}]"
                )
        for argument in dynamic_outputs:
            if argument.get("dynamic_rank"):
                shape_parameters.extend([
                    f"int64_t *{argument['name']}_shape",
                    f"uint32_t {argument['name']}_shape_capacity",
                    f"uint32_t *{argument['name']}_rank",
                ])
            else:
                rank = macro_name(function, argument, "rank")
                shape_parameters.append(
                    f"int64_t {argument['name']}_shape[{rank}]"
                )
        shape_declaration = (
            f"\nint {function}_infer_output_shapes("
            f"{', '.join(shape_parameters)});"
        )
    sizes = []
    relations = manifest.get("input_dimension_relations", [])
    sizes.append(
        f"#define {function.upper()}_INPUT_DIM_RELATION_COUNT {len(relations)}"
    )
    for index, relation in enumerate(relations, 1):
        prefix = f"{function.upper()}_INPUT_DIM_RELATION{index}"
        sizes.extend([
            f"#define {prefix}_LHS_INPUT {relation['lhs_input']}",
            f"#define {prefix}_LHS_DIMENSION {relation['lhs_dimension']}",
            f"#define {prefix}_RHS_INPUT {relation['rhs_input']}",
            f"#define {prefix}_RHS_DIMENSION {relation['rhs_dimension']}",
            f"#define {prefix}_OFFSET INT64_C({relation['offset']})",
        ])
    sizes.append("")
    for argument in manifest["inputs"] + manifest["outputs"]:
        if argument.get("dynamic_rank"):
            sizes.extend([
                f"#define {macro_name(function, argument, 'rank_min')} "
                f"{argument['rank_min']}",
                f"#define {macro_name(function, argument, 'rank_max')} "
                f"{argument['rank_max']}",
            ])
            if argument in manifest["outputs"]:
                depends = int(argument.get("shape_depends_on_data", False))
                sizes.append(
                    f"#define {macro_name(function, argument, 'shape_depends_on_data')} "
                    f"{depends}"
                )
            sizes.append("")
            continue
        sizes.append(f"#define {macro_name(function, argument, 'rank')} {len(argument['shape'])}")
        for index, dimension in enumerate(argument["shape"]):
            value = "NCNN_DYNAMIC_DIM" if dimension < 0 else f"INT64_C({dimension})"
            sizes.append(f"#define {macro_name(function, argument, f'dim{index}')} {value}")
        mask = argument["dynamic_dim_mask"]
        sizes.append(
            f"#define {macro_name(function, argument, 'dynamic_dim_mask')} "
            f"UINT32_C(0x{mask:x})"
        )
        for constraint in argument.get("dimension_constraints", []):
            dimension = constraint["dimension"]
            sizes.extend([
                f"#define {macro_name(function, argument, f'dim{dimension}_minimum')} "
                f"INT64_C({constraint['minimum']})",
                f"#define {macro_name(function, argument, f'dim{dimension}_multiple_of')} "
                f"INT64_C({constraint['multiple_of']})",
            ])
        if not mask:
            sizes.append(
                f"#define {macro_name(function, argument, 'elements')} "
                f"UINT64_C({element_count(argument)})"
            )
        if argument.get("shape_depends_on_data", False):
            for index, dimension in enumerate(argument["maximum_shape"]):
                sizes.append(
                    f"#define {macro_name(function, argument, f'max_dim{index}')} "
                    f"INT64_C({dimension})"
                )
            sizes.append(
                f"#define {macro_name(function, argument, 'max_elements')} "
                f"UINT64_C({element_count(argument)})"
            )
        if argument in manifest["outputs"]:
            depends = int(argument.get("shape_depends_on_data", False))
            sizes.append(
                f"#define {macro_name(function, argument, 'shape_depends_on_data')} "
                f"{depends}"
            )
        sizes.append("")
    path.write_text(
        f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n"
        "#define NCNN_DYNAMIC_DIM INT64_C(-1)\n"
        "#define NCNN_MAX_RANK 4\n"
        "#define NCNN_STATUS_SUCCESS 0\n"
        "#define NCNN_STATUS_NULL_POINTER 1\n"
        "#define NCNN_STATUS_INVALID_SHAPE 2\n"
        "#define NCNN_STATUS_CONSTRAINT_VIOLATION 3\n"
        "#define NCNN_STATUS_SHAPE_ARITHMETIC_OVERFLOW 4\n"
        "#define NCNN_STATUS_OUTPUT_CAPACITY_INSUFFICIENT 5\n\n"
        "typedef uint16_t ncnn_float16_t;\n"
        "typedef uint16_t ncnn_bfloat16_t;\n\n"
        f"#define {function.upper()}_INPUT_COUNT {len(manifest['inputs'])}\n"
        f"#define {function.upper()}_OUTPUT_COUNT {len(manifest['outputs'])}\n\n"
        + "\n".join(sizes)
        + "\n\n"
        "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
        f"{declaration}{shape_declaration}\n\n"
        "#ifdef __cplusplus\n}\n#endif\n\n"
        f"#endif  // {guard}\n",
        encoding="ascii",
    )


def write_harness(path, header_name, manifest):
    function = manifest["function"]
    if (
        len(manifest["inputs"]) == 1
        and len(manifest["outputs"]) == 1
        and manifest["inputs"][0].get("dynamic_rank")
        and manifest["outputs"][0].get("dynamic_rank")
    ):
        input_type = c_type(manifest["inputs"][0])
        output_type = c_type(manifest["outputs"][0])
        path.write_text(
            "#include <stddef.h>\n"
            f'#include "{header_name}"\n\n'
            "int main(void) {\n"
            "  int64_t input1_shape[NCNN_MAX_RANK] = {2, 2, 2, 2};\n"
            "  int64_t output1_shape[NCNN_MAX_RANK] = {0};\n"
            "  uint32_t output1_rank = 0;\n"
            f"  {input_type} input1[16] = {{0}};\n"
            f"  {output_type} output1[16] = {{0}};\n"
            f"  for (uint32_t i = 0; i < 16; ++i) input1[i] = "
            f"({input_type})i + 1;\n"
            "  for (uint32_t rank = 1; rank <= NCNN_MAX_RANK; ++rank) {\n"
            f"    if ({function}_infer_output_shapes(input1_shape, rank, "
            "output1_shape, NCNN_MAX_RANK, &output1_rank) != 0) return 6;\n"
            "    if (output1_rank != rank) return 7;\n"
            f"    if ({function}(input1, input1_shape, rank, output1, 16) != 0) "
            "return 4;\n"
            "    size_t count = 1;\n"
            "    for (uint32_t i = 0; i < rank; ++i) count *= 2;\n"
            "    for (size_t i = 0; i < count; ++i) "
            "if (output1[i] != input1[i]) return 8;\n"
            "  }\n"
            "  return 0;\n}\n",
            encoding="ascii",
        )
        return
    arguments = manifest["inputs"] + manifest["outputs"]
    declarations = []
    names = []
    allocated_names = []
    for argument in arguments:
        declarations.append(
            f"  {c_type(argument)} *{argument['name']} = "
            f"calloc({element_count(argument)}, sizeof({c_type(argument)}));"
        )
        declarations.append(f"  if (!{argument['name']}) return 2;")
        names.append(argument["name"])
        allocated_names.append(argument["name"])
    for argument in manifest["outputs"]:
        if argument.get("shape_depends_on_data", False):
            declarations.extend([
                f"  int64_t {argument['name']}_shape["
                f"{len(argument['shape'])}] = {{0}};",
                f"  uint32_t {argument['name']}_rank = 0;",
            ])
            names.extend([
                f"{argument['name']}_shape",
                str(len(argument["shape"])),
                f"&{argument['name']}_rank",
            ])
    null_checks = []
    for index in range(len(names)):
        call_arguments = names.copy()
        call_arguments[index] = "NULL"
        null_checks.append(
            f"  if ({function}({', '.join(call_arguments)}) == 0) return 3;"
        )
    frees = [f"  free({name});" for name in allocated_names]
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
    parser.add_argument("--input-shape", action="append", default=[])
    parser.add_argument("--input-dim-constraint", action="append", default=[])
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

    driver_command = [args.driver, param_path, "--bin", bin_path]
    for input_shape in args.input_shape:
        driver_command.append(f"--input-shape={input_shape}")
    for constraint in args.input_dim_constraint:
        driver_command.append(f"--input-dim-constraint={constraint}")
    driver_command.extend(["-o", ncnn_ir])
    run(driver_command)
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
    dynamic_outputs = [
        output for output in manifest["outputs"]
        if output.get("dynamic_rank") or output["dynamic_dim_mask"]
    ]
    exported_symbols = model_name
    if dynamic_outputs:
        exported_symbols += f"; {model_name}_infer_output_shapes"
    exports_path.write_text(
        f"{{\n  global: {exported_symbols};\n  local: *;\n}};\n",
        encoding="ascii",
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
    allowed_undefined = {"expf", "free", "malloc", "memcpy", "memset", "tanhf"}
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
    expected_defined = {model_name}
    if dynamic_outputs:
        expected_defined.add(f"{model_name}_infer_output_shapes")
    if defined != expected_defined:
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
