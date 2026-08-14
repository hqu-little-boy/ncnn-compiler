import argparse
import ctypes
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


def assert_static_artifact_baseline(directory, nm, target, baseline_path):
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("schema_version") != 1:
        raise RuntimeError(f"unsupported static artifact baseline: {baseline_path}")
    target_baseline = baseline.get("targets", {}).get(target)
    if target_baseline is None:
        raise RuntimeError(f"static artifact baseline has no target {target}")
    expected = target_baseline["relu_all"]

    manifest = json.loads((directory / "relu_all.json").read_text())
    if manifest != expected["manifest"]:
        raise RuntimeError(
            f"static manifest changed: actual={manifest} "
            f"expected={expected['manifest']}"
        )

    nm_output = run(
        [nm, "-D", "--defined-only", directory / "librelu_all.so"],
        capture_output=True,
        text=True,
    ).stdout
    exports = sorted(
        line.split()[-1] for line in nm_output.splitlines() if line.split()
    )
    if exports != expected["exports"]:
        raise RuntimeError(
            f"static exports changed: actual={exports} expected={expected['exports']}"
        )
    if any(symbol.endswith("_infer_output_shapes") for symbol in exports):
        raise RuntimeError("static artifact exports dynamic output shape inference")

    sizes = {
        name: (directory / name).stat().st_size for name in expected["files"]
    }
    if sizes != expected["files"]:
        raise RuntimeError(
            f"static artifact sizes changed: actual={sizes} "
            f"expected={expected['files']}"
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
    parser.add_argument("--static-baseline", type=pathlib.Path, required=True)
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
    default_compile = run(
        [args.compiler, "model.param", "-O0", "--emit=llvm", "-v"],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    assert_files(default_output, {"libmodel.so", "model.h", "model.llvm.mlir"})
    default_ir = (default_output / "model.llvm.mlir").read_text()
    if "omp.parallel" not in default_ir or "omp.wsloop" not in default_ir:
        raise RuntimeError("default compilation does not use OpenMP lowering")
    if "-mprefer-vector-width=256" not in default_compile.stderr:
        raise RuntimeError("default compilation does not request 256-bit SIMD")
    default_needed = run(
        [args.readelf, "--needed-libs", default_output / "libmodel.so"],
        capture_output=True,
        text=True,
    ).stdout
    if "libomp.so" not in default_needed:
        raise RuntimeError("default artifact does not depend on libomp")

    unrelated_output = work_dir / "unrelated"
    unrelated_output.mkdir()
    unrelated = unrelated_output / "user-data.txt"
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
            "--threads=1",
            "--vector-width=0",
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
            "model.ll",
            "model.o",
            "model.s",
        },
    )
    assert_static_artifact_baseline(
        all_output, args.nm, native_target, args.static_baseline
    )
    manifest = json.loads((all_output / "relu_all.json").read_text())
    if manifest["function"] != "relu_all" or not manifest["inputs"]:
        raise RuntimeError(f"unexpected ABI manifest: {manifest}")
    input1 = manifest["inputs"][0]
    if input1["element_type"] != "f32" or input1["dynamic_dim_mask"] != 0:
        raise RuntimeError(f"missing typed ABI metadata: {input1}")
    target = manifest.get("target")
    if target is None or target["triple"] != native_target:
        raise RuntimeError(f"missing target provenance: {target}")
    if target["execution_profile"] != "x86-64-auto":
        raise RuntimeError(f"unexpected execution profile: {target}")
    all_header = (all_output / "relu_all.h").read_text()
    required_header_text = {
        "#include <stdint.h>",
        "#define NCNN_DYNAMIC_DIM INT64_C(-1)",
        "#define RELU_ALL_INPUT_COUNT 1",
        "#define RELU_ALL_OUTPUT_COUNT 1",
        "#define RELU_ALL_INPUT1_RANK 3",
        "#define RELU_ALL_INPUT1_DIM0 INT64_C(3)",
        "#define RELU_ALL_INPUT1_DIM1 INT64_C(4)",
        "#define RELU_ALL_INPUT1_DIM2 INT64_C(5)",
        "#define RELU_ALL_INPUT1_ELEMENTS UINT64_C(60)",
        "#define RELU_ALL_INPUT1_DYNAMIC_DIM_MASK UINT32_C(0x0)",
        "#define RELU_ALL_OUTPUT1_SHAPE_DEPENDS_ON_DATA 0",
        "int relu_all(const float *input1, float *output1);",
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
    forbidden_header_text = {
        "void *",
        "void*",
        "struct ",
        "dtype",
        "element_type",
        "inputs[",
        "outputs[",
        "descriptor",
        "infer_output_shapes",
        "input1_shape",
        "input1_rank",
    }
    present_forbidden_text = {
        text for text in forbidden_header_text if text in all_header
    }
    if present_forbidden_text:
        raise RuntimeError(
            "static model header contains generic or redundant ABI state: "
            f"{sorted(present_forbidden_text)}"
        )

    sparse_param = work_dir / "sparse_input_shapes.param"
    sparse_bin = work_dir / "sparse_input_shapes.bin"
    sparse_param.write_text(
        "7767517\n"
        "3 3\n"
        "Input declared 0 1 left 0=2 1=2 2=1\n"
        "Input omitted 0 1 right\n"
        "BinaryOp add 2 1 left right output 0=0\n",
        encoding="ascii",
    )
    sparse_bin.write_bytes(b"")
    sparse_output = work_dir / "sparse-input-shapes"
    run(
        [
            args.compiler,
            sparse_param,
            "--bin",
            sparse_bin,
            "--model-name",
            "sparse_input_shapes",
            "--output-dir",
            sparse_output,
            "--input-shape=1x2x2",
            "--emit-manifest",
            "--verify-execution",
            "-O0",
        ]
    )
    sparse_manifest = json.loads(
        (sparse_output / "sparse_input_shapes.json").read_text()
    )
    sparse_shapes = [entry["shape"] for entry in sparse_manifest["inputs"]]
    if sparse_shapes != [[1, 2, 2], [1, 2, 2]]:
        raise RuntimeError(f"sparse input shapes were not inferred: {sparse_shapes}")

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
            "--input-dim-constraint=0:1:min=32,multiple=32",
            "--input-dim-constraint=0:2:min=32,multiple=32",
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
    if dynamic_manifest["outputs"][0].get("shape_program") != [[], [], []]:
        raise RuntimeError(f"dynamic output shape program missing: {dynamic_manifest}")
    expected_constraints = [
        {"dimension": 1, "minimum": 32, "multiple_of": 32},
        {"dimension": 2, "minimum": 32, "multiple_of": 32},
    ]
    if dynamic_manifest["inputs"][0].get("dimension_constraints") != expected_constraints:
        raise RuntimeError(f"dynamic input constraints missing: {dynamic_manifest}")
    dynamic_header = (dynamic_output / "dynamic_identity.h").read_text()
    required_dynamic_extent_abi = {
        "#define DYNAMIC_IDENTITY_INPUT1_RANK 3",
        "#define DYNAMIC_IDENTITY_INPUT1_DIM0 INT64_C(3)",
        "#define DYNAMIC_IDENTITY_INPUT1_DIM1 NCNN_DYNAMIC_DIM",
        "#define DYNAMIC_IDENTITY_INPUT1_DYNAMIC_DIM_MASK UINT32_C(0x6)",
        "const int64_t input1_shape[DYNAMIC_IDENTITY_INPUT1_RANK]",
        "int64_t output1_shape[DYNAMIC_IDENTITY_OUTPUT1_RANK]",
        "dynamic_identity_infer_output_shapes",
        "uint64_t output1_capacity",
        "#define NCNN_STATUS_OUTPUT_CAPACITY_INSUFFICIENT 5",
        "#define DYNAMIC_IDENTITY_INPUT1_DIM1_MINIMUM INT64_C(32)",
        "#define DYNAMIC_IDENTITY_INPUT1_DIM1_MULTIPLE_OF INT64_C(32)",
    }
    missing_dynamic_extent_abi = {
        text for text in required_dynamic_extent_abi if text not in dynamic_header
    }
    if missing_dynamic_extent_abi:
        raise RuntimeError(
            "fixed-rank dynamic header contract missing: "
            f"{sorted(missing_dynamic_extent_abi)}"
        )
    if "input1_rank" in dynamic_header or "output1_rank" in dynamic_header:
        raise RuntimeError("fixed-rank dynamic ABI contains redundant rank")

    dynamic_library = ctypes.CDLL(
        str(dynamic_output / "libdynamic_identity.so")
    )
    shape_type = ctypes.c_int64 * 3
    float_pointer = ctypes.POINTER(ctypes.c_float)
    dynamic_function = dynamic_library.dynamic_identity
    dynamic_function.argtypes = [
        float_pointer,
        ctypes.POINTER(ctypes.c_int64),
        float_pointer,
        ctypes.c_uint64,
    ]
    dynamic_function.restype = ctypes.c_int
    infer_shapes = dynamic_library.dynamic_identity_infer_output_shapes
    infer_shapes.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    infer_shapes.restype = ctypes.c_int
    valid_shape = shape_type(3, 32, 64)
    invalid_small = shape_type(3, 31, 64)
    invalid_multiple = shape_type(3, 32, 33)
    invalid_zero = shape_type(3, 0, 64)
    overflow_shape = shape_type(3, 9223372036854775776, 32)
    input_data = (ctypes.c_float * (3 * 32 * 64))()
    output_data = (ctypes.c_float * (3 * 32 * 64))()
    output_shape = shape_type()
    if dynamic_function(input_data, valid_shape, output_data, len(output_data)) != 0:
        raise RuntimeError("valid constrained dynamic shape was rejected")
    if infer_shapes(valid_shape, output_shape) != 0:
        raise RuntimeError("valid constrained shape inference was rejected")
    if dynamic_function(input_data, invalid_small, output_data, len(output_data)) != 3:
        raise RuntimeError("shape below minimum was accepted")
    if infer_shapes(invalid_multiple, output_shape) != 3:
        raise RuntimeError("non-multiple shape was accepted")
    if infer_shapes(invalid_zero, output_shape) != 2:
        raise RuntimeError("invalid zero extent returned the wrong status")
    if infer_shapes(overflow_shape, output_shape) != 4:
        raise RuntimeError("shape element-count overflow returned the wrong status")
    if infer_shapes(None, output_shape) != 1:
        raise RuntimeError("NULL shape returned the wrong status")
    if dynamic_function(input_data, valid_shape, output_data, len(output_data) - 1) != 5:
        raise RuntimeError("insufficient dynamic output capacity was accepted")

    malformed_constraint = subprocess.run(
        [
            args.driver,
            dynamic_param,
            "--bin",
            dynamic_bin,
            "--input-shape=3x?x?",
            "--input-dim-constraint=0:1:min=32",
        ],
        capture_output=True,
        text=True,
    )
    if malformed_constraint.returncode == 0 or "must be" not in malformed_constraint.stderr:
        raise RuntimeError(
            f"malformed dimension constraint was not rejected: {malformed_constraint.stderr}"
        )
    static_constraint = subprocess.run(
        [
            args.driver,
            dynamic_param,
            "--bin",
            dynamic_bin,
            "--input-shape=3x32x?",
            "--input-dim-constraint=0:1:min=32,multiple=32",
        ],
        capture_output=True,
        text=True,
    )
    if static_constraint.returncode == 0 or "dynamic dimension" not in static_constraint.stderr:
        raise RuntimeError(
            f"static dimension constraint was not rejected: {static_constraint.stderr}"
        )

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
        "#define DYNAMIC_RANK_IDENTITY_INPUT_COUNT 1",
        "#define DYNAMIC_RANK_IDENTITY_OUTPUT_COUNT 1",
        "#define DYNAMIC_RANK_IDENTITY_INPUT1_RANK_MIN 1",
        "#define DYNAMIC_RANK_IDENTITY_INPUT1_RANK_MAX 4",
        "#define DYNAMIC_RANK_IDENTITY_OUTPUT1_SHAPE_DEPENDS_ON_DATA 0",
        "const int64_t *input1_shape, uint32_t input1_rank",
        "uint64_t output1_capacity",
        "uint32_t output1_shape_capacity, uint32_t *output1_rank",
    }
    missing_dynamic_rank_abi = required_dynamic_rank_abi - {
        text for text in required_dynamic_rank_abi if text in dynamic_rank_header
    }
    if missing_dynamic_rank_abi:
        raise RuntimeError(
            f"dynamic rank header contract missing: {sorted(missing_dynamic_rank_abi)}"
        )
    for forbidden in ("void *", "void*", "struct ", "dtype", "descriptor"):
        if forbidden in dynamic_rank_header:
            raise RuntimeError(
                f"dynamic rank header contains generic descriptor state: {forbidden}"
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

    detection_param = (
        pathlib.Path(__file__).parent.parent
        / "Numerical"
        / "fixtures"
        / "detection_output.param"
    )
    detection_output = work_dir / "detection-output"
    run(
        [
            args.compiler,
            detection_param,
            "--bin",
            dynamic_bin,
            "--model-name",
            "detection_output_abi",
            "--output-dir",
            detection_output,
            "--emit-manifest",
            "-O0",
        ]
    )
    assert_files(
        detection_output,
        {
            "libdetection_output_abi.so",
            "detection_output_abi.h",
            "detection_output_abi.json",
        },
    )
    detection_header = (
        detection_output / "detection_output_abi.h"
    ).read_text()
    required_detection_abi = {
        "#define DETECTION_OUTPUT_ABI_INPUT_COUNT 3",
        "#define DETECTION_OUTPUT_ABI_OUTPUT_COUNT 1",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_RANK 2",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_DIM0 NCNN_DYNAMIC_DIM",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_DIM1 INT64_C(6)",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_MAX_DIM0 INT64_C(2)",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_MAX_DIM1 INT64_C(6)",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_MAX_ELEMENTS UINT64_C(12)",
        "#define DETECTION_OUTPUT_ABI_OUTPUT1_SHAPE_DEPENDS_ON_DATA 1",
        "float *output1, int64_t *output1_shape, "
        "uint32_t output1_shape_capacity, uint32_t *output1_rank",
    }
    missing_detection_abi = {
        text for text in required_detection_abi if text not in detection_header
    }
    if missing_detection_abi:
        raise RuntimeError(
            "data-dependent header contract missing: "
            f"{sorted(missing_detection_abi)}"
        )
    if "infer_output_shapes" in detection_header:
        raise RuntimeError(
            "data-dependent model exported a shape-only inference function"
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

    vector_output = work_dir / "relu-vector"
    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "relu_vector",
            "--output-dir",
            vector_output,
            "--threads=1",
            "--vector-width=256",
            "--emit=llvm",
            "--verify-execution",
        ]
    )
    vector_ir = (vector_output / "model.llvm.mlir").read_text()
    if "vector<8xf32>" not in vector_ir or "llvm.intr.masked" not in vector_ir:
        raise RuntimeError("explicit SIMD lowering is missing from LLVM dialect IR")

    parallel_output = work_dir / "relu-parallel"
    run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--model-name",
            "relu_parallel",
            "--output-dir",
            parallel_output,
            "--threads=4",
            "--emit=llvm",
            "--verify-execution",
        ]
    )
    parallel_ir = (parallel_output / "model.llvm.mlir").read_text()
    if "omp.parallel" not in parallel_ir or "omp.wsloop" not in parallel_ir:
        raise RuntimeError("OpenMP lowering is missing from LLVM dialect IR")
    needed = run(
        [args.readelf, "--needed-libs", parallel_output / "librelu_parallel.so"],
        capture_output=True,
        text=True,
    ).stdout
    if "libomp.so" not in needed:
        raise RuntimeError("parallel artifact does not depend on libomp")

    invalid_vector_width = subprocess.run(
        [
            args.compiler,
            args.param,
            "--bin",
            args.bin,
            "--output-dir",
            work_dir / "invalid-vector-width",
            "--vector-width=96",
        ],
        capture_output=True,
        text=True,
    )
    if (
        invalid_vector_width.returncode == 0
        or "multiple of 64" not in invalid_vector_width.stderr
    ):
        raise RuntimeError("invalid SIMD vector width was not rejected")

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
