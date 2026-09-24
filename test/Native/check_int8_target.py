"""Focused P16 target parser and production CLI contract regressions."""

import argparse
import json
import os
import pathlib
import subprocess
import tempfile


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], capture_output=True,
                          text=True, **kwargs)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "driver", "opt", "translate", "clang", "nm",
                 "readelf", "llvm-as", "param", "bin", "work-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="target-", dir=work))
    print(f"Artifacts: {root}", flush=True)

    # Compile the actual dependency-free header, not a Python reimplementation.
    source = root / "parser.cpp"
    source.write_text('''#include <iostream>
#include <iterator>
#include <string>
#include "ncnn-mlir/Support/Int8Target.hpp"
int main() {
  const std::string text{std::istreambuf_iterator<char>(std::cin), {}};
  std::cout << ncnn_mlir::int8_dot_target_name(
    ncnn_mlir::resolve_int8_dot_target(text));
}
''')
    include = pathlib.Path(__file__).resolve().parents[2] / "include"
    executable = root / "parser"
    built = run([args.clang, "-std=c++23", "-x", "c++", source,
                 "-I", include, "-lstdc++", "-o", executable])
    require(built.returncode == 0, built.stderr)
    avx = ["__x86_64__", "__AVX2__", "__AVXVNNI__"]
    avxvnniint8 = ["__x86_64__", "__AVX2__", "__AVXVNNIINT8__"]
    avx512 = ["__x86_64__", "__AVX2__", "__AVX512F__", "__AVX512BW__",
              "__AVX512VL__", "__AVX512VNNI__"]
    cases = [(avx, "avx-vnni"), (avx512, "avx512-vnni"),
             (avxvnniint8, "avx-vnni-int8"),
             (avx512 + avx, "avx-vnni"),
             (avx + avxvnniint8, "avx-vnni"),
             (avx512 + avxvnniint8, "avx512-vnni"), ([], "portable")]
    cases += [(avx[:i] + avx[i + 1:], "portable") for i in range(len(avx))]
    cases += [(avxvnniint8[:i] + avxvnniint8[i + 1:], "portable")
              for i in range(len(avxvnniint8))]
    cases += [(avx512[:i] + avx512[i + 1:], "portable")
              for i in range(len(avx512))]
    for names, expected in cases:
        text = "".join(f"#define {name} 1\n" for name in names)
        result = run([executable], input=text)
        require(result.returncode == 0 and result.stdout == expected,
                f"parser {names}: {result.stdout} != {expected}")
    for macro in ("__AVXVNNI__", "__AVXVNNIINT8__"):
        for suffix in (" 0", " 10", "_OTHER 1"):
            text = "#define __x86_64__ 1\n#define __AVX2__ 1\n#define " + macro + suffix
            require(run([executable], input=text).stdout == "portable",
                    f"parser accepted a malformed {macro} definition")
    print(f"PASS parser: {len(cases) + 6} dependency/name/value cases", flush=True)

    base = [args.compiler, args.param, "--bin=" + args.bin,
            "--model-name=target_test", "--threads=1", "--vector-math=none",
            "--emit-execution-plan", "--emit-manifest", "--emit=memref"]
    for name in ("driver", "opt", "translate", "clang", "nm", "readelf", "llvm-as"):
        base.append("--" + name + "=" + getattr(args, name.replace("-", "_")))
    base += ["--target-triple=x86_64-unknown-linux-gnu", "--march=x86-64"]
    vnni = ["--int8-kernel=vnni", "--target-feature=+avx2", "--target-feature=+avxvnni"]
    fixed = ["--vector-mode=fixed-width"]
    count = 0

    def compile_case(name, flags, error=None, capability=None, policy=None):
        nonlocal count
        output = root / name
        result = run(base + flags + ["--output-dir=" + str(output)])
        (root / (name + ".log")).write_text(result.stdout + result.stderr)
        count += 1
        if error is not None:
            require(result.returncode != 0 and error in result.stderr,
                    f"{name}: expected rejection {error!r}\n{result.stderr}")
            require(not output.exists(), f"{name}: published rejected output")
            return None
        require(result.returncode == 0, f"{name}: {result.stderr}")
        plan = json.loads((output / "target_test.plan.json").read_text())
        identity = bytes.fromhex(plan["codegen_identity"]).decode()
        require(f"|int8-target={capability}|" in identity,
                f"{name}: missing resolved capability in identity")
        require(plan["low_precision"]["capability"] == capability,
                f"{name}: inconsistent plan capability")
        require(plan["low_precision"]["requested_policy"] == policy,
                f"{name}: inconsistent requested policy")
        return plan

    compile_case("default", [], capability="portable", policy="portable")
    compile_case("portable_passthrough", ["--clang-arg=-mno-avxvnni"],
                 capability="portable", policy="portable")
    compile_case("unsupported", ["--int8-kernel=vnni"], "requires AVX2")
    compile_case("bad_policy", ["--int8-kernel=other"], "must be one of")
    compile_case("serial_off", vnni, "requires OpenMP")
    compile_case("serial_scalable", vnni + ["--vector-mode=scalable"], "requires OpenMP")
    compile_case("serial_fixed", vnni + fixed, capability="avx-vnni", policy="vnni")
    compile_case("openmp", vnni + ["--threads=2"], capability="avx-vnni", policy="vnni")
    enabled = compile_case("enable_last", vnni + ["--target-feature=-avxvnni",
                           "--target-feature=+avxvnni"] + fixed,
                           capability="avx-vnni", policy="vnni")
    compile_case("disable_last", vnni + ["--target-feature=-avxvnni"] + fixed,
                 "requires AVX2")
    compile_case("disable_dependency", vnni + ["--target-feature=-avx2"] + fixed,
                 "requires AVX2")
    auto = compile_case("auto_capable", ["--int8-kernel=auto",
                        "--target-feature=+avx2", "--target-feature=+avxvnni"],
                        capability="avx-vnni", policy="auto")
    fallback = compile_case("auto_disabled", ["--int8-kernel=auto",
                            "--target-feature=+avx2", "--target-feature=+avxvnni",
                            "--target-feature=-avxvnni"], capability="portable", policy="auto")
    require(auto["low_precision"]["requested_policy_status"] == "pending_defaultization",
            "auto must remain pending, not advertise selected VNNI")
    int8_extension_flags = ["--target-feature=+avx2",
                            "--target-feature=+avxvnniint8"]
    compile_case(
        "vnni_int8_without_backend",
        ["--int8-kernel=vnni"] + int8_extension_flags,
        "no matching vpdpbusd backend",
    )
    int8_extension_auto = compile_case(
        "auto_avx_vnni_int8",
        ["--int8-kernel=auto"] + int8_extension_flags,
        capability="avx-vnni-int8",
        policy="auto",
    )
    require(int8_extension_auto["low_precision"].get("requested_policy_status") ==
            "fallback" and
            int8_extension_auto["low_precision"].get(
                "requested_policy_fallback_reason") ==
            "unimplemented_vnni_int8_backend",
            "auto must not conflate AVX-VNNI-INT8 with the vpdpbusd backend")
    native_int8_extension = compile_case(
        "native_profile_avx_vnni_int8",
        ["--tuning-profile=native-int8", "--target-feature=+avx2",
         "--target-feature=+avxvnniint8", "--vector-mode=fixed-width"],
        capability="avx-vnni-int8",
        policy="portable",
    )
    require(native_int8_extension["tuning"].get("status") == "fallback" and
            native_int8_extension["tuning"].get("fallback_reason") ==
            "unimplemented_vnni_int8_backend",
            "native-int8 must fall back on an unimplemented capability")
    require(len({p["plan_hash"] for p in (enabled, auto, fallback,
                                          int8_extension_auto,
                                          native_int8_extension)}) == 5,
            "target/policy changes must change plan identity")
    avx512_flags = ["--int8-kernel=vnni", "--march=x86-64-v4",
                   "--target-feature=+avx512vnni"] + fixed
    compile_case("avx512", avx512_flags, capability="avx512-vnni", policy="vnni")
    for feature in ("avx512f", "avx512bw", "avx512vl", "avx512vnni"):
        compile_case("disable_" + feature, avx512_flags + ["--target-feature=-" + feature],
                     "requires AVX2")
    for index, argument in enumerate(("-mno-avxvnni", "-Xclang", "-D__AVXVNNI__=1",
                                      "-U__AVX2__", "@features.rsp", "--config=target.cfg",
                                      "--target=aarch64-linux-gnu", "-mllvm")):
        compile_case(f"opaque_{index}", vnni + ["--clang-arg=" + argument],
                     "--clang-arg cannot be combined")
    compile_case("auto_opaque", ["--int8-kernel=auto", "--clang-arg=-mno-avxvnni"],
                 "--clang-arg cannot be combined")
    compile_case("native", vnni + ["--march=native"], "native CPU selection")
    compile_case("cross_native", ["--target-triple=aarch64-linux-gnu", "--mcpu=native"],
                 "native CPU selection")
    compile_case("non_x86", ["--int8-kernel=vnni", "--target-triple=aarch64-linux-gnu"],
                 "requires an x86-64")
    compile_case("depthwise_off", ["--int8-depthwise"], "requires fixed-width")
    compile_case("depthwise_scalable", ["--int8-depthwise", "--vector-mode=scalable"],
                 "requires fixed-width")
    compile_case("depthwise_fixed", ["--int8-depthwise"] + fixed,
                 capability="portable", policy="portable")
    native_profile_flags = [
        "--tuning-profile=native-int8",
        "--target-feature=+avx2",
        "--target-feature=+avxvnni",
        "--vector-mode=fixed-width",
    ]
    native_profile = compile_case(
        "native_int8_selected",
        native_profile_flags,
        capability="avx-vnni",
        policy="vnni",
    )
    native_tuning = native_profile.get("tuning", {})
    require(native_tuning.get("profile_revision") == "native-int8-v1",
            f"native profile revision was not recorded: {native_tuning}")
    require(native_tuning.get("status") == "selected",
            f"native profile was not selected: {native_tuning}")
    require(native_profile["low_precision"].get("depthwise_enabled") is True and
            native_profile["low_precision"].get("cast_chain_enabled") is True,
            "native profile must enable the supported depthwise/cast-chain paths")
    native_identity = bytes.fromhex(native_profile["codegen_identity"]).decode()
    require("tuning-profile-revision=native-int8-v1" in native_identity,
            "native profile revision missing from codegen identity")

    native_unsupported = compile_case(
        "native_int8_unsupported",
        ["--tuning-profile=native-int8"],
        capability="portable",
        policy="portable",
    )
    require(native_unsupported["tuning"].get("status") == "fallback" and
            native_unsupported["tuning"].get("fallback_reason") ==
            "unsupported_vnni_target",
            f"unsupported native profile lacks fallback identity: "
            f"{native_unsupported['tuning']}")

    native_without_vectors = compile_case(
        "native_int8_without_vectors",
        ["--tuning-profile=native-int8", "--target-feature=+avx2",
         "--target-feature=+avxvnni"],
        capability="avx-vnni",
        policy="portable",
    )
    require(native_without_vectors["tuning"].get("status") == "fallback" and
            native_without_vectors["tuning"].get("fallback_reason") ==
            "fixed_width_vector_required",
            f"native profile did not record vector fallback: "
            f"{native_without_vectors['tuning']}")

    native_winograd = compile_case(
        "native_int8_winograd",
        native_profile_flags + ["--conv-strategy=winograd"],
        capability="avx-vnni",
        policy="vnni",
    )
    require(native_winograd["tuning"].get("status") == "selected" and
            native_winograd["tuning"].get("fallback_reason") ==
            "winograd_packing_compile_budget",
            f"Winograd packing fallback masked the native profile: "
            f"{native_winograd['tuning']}")

    native_override = compile_case(
        "native_int8_explicit_portable",
        ["--tuning-profile=native-int8", "--int8-kernel=portable",
         "--target-feature=+avx2", "--target-feature=+avxvnni"],
        capability="avx-vnni",
        policy="portable",
    )
    require(native_override["tuning"].get("status") == "explicit_override" and
            native_override["tuning"].get("fallback_reason") ==
            "explicit_int8_policy",
            f"explicit policy did not override native profile: "
            f"{native_override['tuning']}")

    compile_case("bad_profile", ["--tuning-profile=other"],
                 "--tuning-profile must be one of")

    sidecar = root / "profile-sidecar.json"
    output = root / "profile-identity"
    profile_run = run(base + ["--profile", "--verify-execution",
                             "--output-dir=" + str(output)],
                      env=dict(os.environ, NCNN_PROFILE_PATH=str(sidecar),
                               NCNN_PROFILE_MODE="abi_smoke"))
    require(profile_run.returncode == 0, profile_run.stderr)
    profile = json.loads(sidecar.read_text())
    plan = json.loads((output / "target_test.plan.json").read_text())
    for field in ("plan_revision", "plan_hash", "build_identity"):
        require(str(profile[field]) == str(plan[field]),
                f"profile {field} differs from published plan")
    print("PASS profile: production CLI sidecar matches published plan identity", flush=True)
    help_result = run([args.compiler, "--help"])
    require("portable pending performance validation" in help_result.stdout,
            "help must describe conservative auto policy")
    print(f"PASS CLI: {count} accepted/rejected cases; identity and truthful auto help", flush=True)
    print("AVX512 cases are compile-only; no target model execution requested.", flush=True)


if __name__ == "__main__":
    main()
