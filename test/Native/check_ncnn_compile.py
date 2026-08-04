import argparse
import json
import pathlib
import shutil
import subprocess


def run(command, cwd=None, **kwargs):
    return subprocess.run(command, check=True, cwd=cwd, **kwargs)


def assert_files(directory, expected):
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"unexpected files in {directory}: actual={sorted(actual)} "
            f"expected={sorted(expected)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--debug-compiler", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--translate", required=True)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work_dir = pathlib.Path(args.work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    native_target = run(
        [args.clang, "-dumpmachine"], capture_output=True, text=True
    ).stdout.strip()
    if not native_target:
        raise RuntimeError("clang did not report a native target triple")

    shutil.copy2(args.param, work_dir / "model.param")
    shutil.copy2(args.bin, work_dir / "model.bin")
    default_output = work_dir / "model"
    run([args.compiler, "model.param", "-O0"], cwd=work_dir)
    assert_files(default_output, {"libmodel.so", "model.h"})

    unrelated_output = work_dir / "unrelated"
    unrelated_output.mkdir()
    unrelated = unrelated_output / "model.o"
    unrelated.write_text("user data")
    refused = subprocess.run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "unrelated",
            "--output-dir",
            unrelated_output,
        ],
        capture_output=True,
        text=True,
    )
    if refused.returncode == 0 or "not owned by ncnn-compile" not in refused.stderr:
        raise RuntimeError(f"generic user file was not refused: {refused.stderr}")
    if unrelated.read_text() != "user data":
        raise RuntimeError("refused compilation modified an unrelated file")

    all_output = work_dir / "relu-all"
    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "relu_all",
            "--output-dir",
            all_output,
            "--emit",
            "all",
            "--emit-manifest",
            "-O2",
        ]
    )
    assert_files(
        all_output,
        {
            "librelu_all.so",
            "relu_all.h",
            "relu_all.json",
            "model.ncnn.mlir",
            "model.tosa.mlir",
            "model.linalg.mlir",
            "model.memref.mlir",
            "model.capi.mlir",
            "model.llvm.mlir",
        },
    )
    manifest = json.loads((all_output / "relu_all.json").read_text())
    if manifest["function"] != "relu_all" or not manifest["inputs"]:
        raise RuntimeError(f"unexpected ABI manifest: {manifest}")

    previous_header = (all_output / "relu_all.h").read_bytes()
    previous_library = (all_output / "librelu_all.so").read_bytes()
    failed = subprocess.run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "relu_all",
            "--output-dir",
            all_output,
            "--expected-undefined",
            "deliberately_wrong",
        ],
        capture_output=True,
        text=True,
    )
    if failed.returncode == 0:
        raise RuntimeError("expected audit failure succeeded")
    if (all_output / "relu_all.h").read_bytes() != previous_header:
        raise RuntimeError("failed compilation replaced the previous header")
    if (all_output / "librelu_all.so").read_bytes() != previous_library:
        raise RuntimeError("failed compilation replaced the previous library")

    c23_output = work_dir / "c23-name"
    completed = run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "23-alignas-\N{GREEK SMALL LETTER MU}",
            "--output-dir",
            c23_output,
            "--emit-manifest",
            "--verify-execution",
            "-O0",
        ],
        capture_output=True,
        text=True,
    )
    if completed.stdout != f"{c23_output}\n":
        raise RuntimeError(f"unstable stdout output: {completed.stdout!r}")
    c23_name = "ncnn_23_alignas_"
    assert_files(c23_output, {f"lib{c23_name}.so", f"{c23_name}.h", f"{c23_name}.json"})
    c23_header = (c23_output / f"{c23_name}.h").read_text()
    if f"int {c23_name}(" not in c23_header:
        raise RuntimeError("sanitized C23 model identifier missing from header")

    stable_name_output = work_dir / "stable-name"
    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "model__v1",
            "--output-dir",
            f"{stable_name_output}/",
        ]
    )
    assert_files(stable_name_output, {"libmodel__v1.so", "model__v1.h"})

    main_output = work_dir / "main-name"
    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "main",
            "--output-dir",
            main_output,
            "--verify-execution",
        ]
    )
    assert_files(main_output, {"libncnn_main.so", "ncnn_main.h"})

    selected_output = work_dir / "relu-selected"
    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "relu_selected",
            "--output-dir",
            selected_output,
            "--emit",
            "ncnn,tosa",
            "--emit",
            "llvm",
            "-O3",
            "--target-triple",
            native_target,
            "--clang-arg",
            "-fno-math-errno",
            "--linker-arg",
            "-Wl,--hash-style=gnu",
        ]
    )
    assert_files(
        selected_output,
        {
            "librelu_selected.so",
            "relu_selected.h",
            "model.ncnn.mlir",
            "model.tosa.mlir",
            "model.llvm.mlir",
        },
    )

    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "relu_selected",
            "--output-dir",
            selected_output,
            "-O1",
        ]
    )
    assert_files(selected_output, {"librelu_selected.so", "relu_selected.h"})

    debug_output = work_dir / "debug"
    run(
        [
            "python3",
            args.debug_compiler,
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
            "debug",
            "-o",
            debug_output,
            "-O1",
        ]
    )
    assert_files(debug_output, {"libdebug.so", "debug.h"})

    install_prefix = work_dir / "install"
    run(["cmake", "--install", args.build_dir, "--prefix", install_prefix])
    installed_output = work_dir / "installed"
    run(
        [
            install_prefix / "bin" / "ncnn-compile",
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "installed",
            "-o",
            installed_output,
            "-O2",
        ]
    )
    assert_files(installed_output, {"libinstalled.so", "installed.h"})


if __name__ == "__main__":
    main()
