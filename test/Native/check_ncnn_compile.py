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
    input1 = manifest["inputs"][0]
    if input1["element_type"] != "f32" or input1["dynamic_dim_mask"] != 0:
        raise RuntimeError(f"missing typed ABI metadata: {input1}")
    all_header = (all_output / "relu_all.h").read_text()
    required_header_text = {
        "#include <stdint.h>",
        "#define NCNN_DYNAMIC_DIM INT64_C(-1)",
        "#define RELU_ALL_INPUT1_RANK 3",
        "#define RELU_ALL_INPUT1_DYNAMIC_DIM_MASK UINT32_C(0x0)",
        "const float *input1",
        "float *output1",
    }
    missing_header_text = {
        text for text in required_header_text if text not in all_header
    }
    if missing_header_text:
        raise RuntimeError(
            f"generated header lacks typed ABI contract: {sorted(missing_header_text)}"
        )

    dynamic_param = work_dir / "dynamic_identity.param"
    dynamic_bin = work_dir / "dynamic_identity.bin"
    dynamic_param.write_text(
        "7767517\n1 1\nInput input 0 1 data\n", encoding="ascii"
    )
    dynamic_bin.write_bytes(b"")
    dynamic_output = work_dir / "dynamic-identity"
    run(
        [
            args.compiler,
            dynamic_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "dynamic_identity",
            "--output-dir",
            dynamic_output,
            "--input-shape=3x?x?",
            "--emit-manifest",
            "--verify-execution",
            "-O0",
        ]
    )
    assert_files(
        dynamic_output,
        {
            "libdynamic_identity.so",
            "dynamic_identity.h",
            "dynamic_identity.json",
        },
    )
    dynamic_manifest = json.loads(
        (dynamic_output / "dynamic_identity.json").read_text()
    )
    if dynamic_manifest["outputs"][0].get("shape_source_input") != 0:
        raise RuntimeError(f"dynamic output shape source missing: {dynamic_manifest}")
    dynamic_header = (dynamic_output / "dynamic_identity.h").read_text()
    if "dynamic_identity_infer_output_shapes" not in dynamic_header:
        raise RuntimeError("dynamic shape inference declaration missing")

    dynamic_rank_output = work_dir / "dynamic-rank-identity"
    run(
        [
            args.compiler,
            dynamic_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "dynamic_rank_identity",
            "--output-dir",
            dynamic_rank_output,
            "--input-shape=*",
            "--emit-manifest",
            "--verify-execution",
            "-O0",
        ]
    )
    assert_files(
        dynamic_rank_output,
        {
            "libdynamic_rank_identity.so",
            "dynamic_rank_identity.h",
            "dynamic_rank_identity.json",
        },
    )
    dynamic_rank_manifest = json.loads(
        (dynamic_rank_output / "dynamic_rank_identity.json").read_text()
    )
    for argument in dynamic_rank_manifest["inputs"] + dynamic_rank_manifest["outputs"]:
        if not argument.get("dynamic_rank"):
            raise RuntimeError(f"dynamic rank missing from manifest: {argument}")
        if (argument.get("rank_min"), argument.get("rank_max")) != (1, 4):
            raise RuntimeError(f"invalid rank range in manifest: {argument}")
    dynamic_rank_header = (
        dynamic_rank_output / "dynamic_rank_identity.h"
    ).read_text()
    required_dynamic_rank_abi = {
        "#define NCNN_MAX_RANK 4",
        "const int64_t *input1_shape, uint32_t input1_rank",
        "uint32_t output1_shape_capacity, uint32_t *output1_rank",
    }
    missing_dynamic_rank_abi = required_dynamic_rank_abi - {
        text for text in required_dynamic_rank_abi if text in dynamic_rank_header
    }
    if missing_dynamic_rank_abi:
        raise RuntimeError(
            f"dynamic rank header contract missing: {sorted(missing_dynamic_rank_abi)}"
        )

    dynamic_debug_output = work_dir / "dynamic-debug"
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
            dynamic_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "dynamic_debug",
            "--input-shape=3x?x?",
            "-o",
            dynamic_debug_output,
            "-O0",
        ]
    )
    assert_files(dynamic_debug_output, {"libdynamic_debug.so", "dynamic_debug.h"})
    if "dynamic_debug_infer_output_shapes" not in (
        dynamic_debug_output / "dynamic_debug.h"
    ).read_text():
        raise RuntimeError("debug compiler dynamic shape declaration missing")

    dynamic_relu_param = work_dir / "dynamic_relu.param"
    dynamic_relu_param.write_text(
        "7767517\n2 2\n"
        "Input input 0 1 data\n"
        "ReLU relu 1 1 data output\n",
        encoding="ascii",
    )
    dynamic_relu_output = work_dir / "dynamic-relu"
    run(
        [
            args.compiler,
            dynamic_relu_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "dynamic_relu",
            "--output-dir",
            dynamic_relu_output,
            "--input-shape=3x?x?",
            "--verify-execution",
            "-O0",
        ]
    )
    assert_files(
        dynamic_relu_output, {"libdynamic_relu.so", "dynamic_relu.h"}
    )

    dynamic_padding_param = work_dir / "dynamic_padding.param"
    dynamic_padding_param.write_text(
        "7767517\n2 2\n"
        "Input input 0 1 data\n"
        "Padding padding 1 1 data output "
        "0=1 1=2 2=3 3=4 4=0 5=0.0 6=0\n",
        encoding="ascii",
    )
    dynamic_padding_output = work_dir / "dynamic-padding"
    run(
        [
            args.compiler,
            dynamic_padding_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "dynamic_padding",
            "--output-dir",
            dynamic_padding_output,
            "--input-shape=3x?x?",
            "--verify-execution",
            "-O0",
        ]
    )
    assert_files(
        dynamic_padding_output,
        {"libdynamic_padding.so", "dynamic_padding.h"},
    )

    dynamic_interp_param = work_dir / "dynamic_interp.param"
    dynamic_interp_param.write_text(
        "7767517\n2 2\n"
        "Input input 0 1 data\n"
        "Interp interp 1 1 data output 0=1 1=2.0 2=3.0 3=0 4=0 6=0\n",
        encoding="ascii",
    )
    dynamic_interp_output = work_dir / "dynamic-interp"
    run(
        [
            args.compiler,
            dynamic_interp_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "dynamic_interp",
            "--output-dir",
            dynamic_interp_output,
            "--input-shape=3x?x?",
            "--verify-execution",
            "-O0",
        ]
    )
    assert_files(
        dynamic_interp_output,
        {"libdynamic_interp.so", "dynamic_interp.h"},
    )

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
