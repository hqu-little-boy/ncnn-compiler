import argparse
import pathlib
import shutil
import subprocess


def run(command, cwd=None):
    subprocess.run(command, check=True, cwd=cwd)


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
    parser.add_argument("--legacy-compiler", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work_dir = pathlib.Path(args.work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    shutil.copy2(args.param, work_dir / "model.param")
    shutil.copy2(args.bin, work_dir / "model.bin")
    default_output = work_dir / "model"
    run([args.compiler, "model.param", "-O0"], cwd=work_dir)
    assert_files(default_output, {"libmodel.so", "model.h"})

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
            "-O2",
        ]
    )
    assert_files(
        all_output,
        {
            "librelu_all.so",
            "relu_all.h",
            "model.ncnn.mlir",
            "model.tosa.mlir",
            "model.linalg.mlir",
            "model.memref.mlir",
            "model.capi.mlir",
            "model.llvm.mlir",
        },
    )

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
            "x86_64-unknown-linux-gnu",
            "--march",
            "x86-64",
            "--mtune",
            "generic",
            "--target-feature",
            "-avx",
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

    legacy_output = work_dir / "legacy"
    run(
        [
            "python3",
            args.legacy_compiler,
            "--param",
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "legacy",
            "-o",
            legacy_output,
            "-O1",
        ]
    )
    assert_files(legacy_output, {"liblegacy.so", "legacy.h"})

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
