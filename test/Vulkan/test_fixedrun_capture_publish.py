#!/usr/bin/env python3
"""Exercise fixed-run capture publication and failure preservation offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

MODEL = "fixedrun_publish_relu"
UNOWNED_NAME = "fixedrun-capture-unowned-test.txt"
UNOWNED_BYTES = b"test-owned mutation after argument validation\n"


def snapshot(directory: Path) -> dict[str, bytes]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unexpected non-regular capture entry: {path}")
        result[path.name] = path.read_bytes()
    return result


def check_capture_inventory(directory: Path) -> dict[str, bytes]:
    files = snapshot(directory)
    required = {
        "candidate_source.cpp",
        "candidate_tosa.mlir",
        f"{MODEL}_vulkan.h",
        "host_program.json",
        "metadata.json",
        "plan.json",
    }
    if not required.issubset(files):
        raise RuntimeError(f"capture is missing expected artifacts: {required - set(files)}")
    metadata = json.loads(files["metadata.json"])
    if metadata.get("model") != MODEL:
        raise RuntimeError("capture metadata has the wrong model")
    shader_count = metadata.get("shader_count")
    if type(shader_count) is not int or shader_count <= 0:
        raise RuntimeError("capture metadata has an invalid shader count")
    shaders = {f"shader_{index}.spv" for index in range(shader_count)}
    if set(files) != required | shaders:
        raise RuntimeError("capture inventory differs from metadata shader coverage")
    return files


def make_driver_wrapper(path: Path, real_driver: Path) -> None:
    source = f'''#!{sys.executable}
import os
from pathlib import Path
import shutil
import subprocess
import sys

result = subprocess.run([{str(real_driver)!r}, *sys.argv[1:]], check=False)
if result.returncode != 0:
    sys.exit(result.returncode)
marker = Path(os.environ["NCNN_FIXEDRUN_MUTATION_MARKER"])
if marker.exists():
    sys.exit(0)
marker.write_bytes(b"mutation applied\\n")
output = Path(os.environ["NCNN_FIXEDRUN_MUTATION_OUTPUT"])
mode = os.environ["NCNN_FIXEDRUN_MUTATION_MODE"]
if mode == "unowned":
    (output / {UNOWNED_NAME!r}).write_bytes({UNOWNED_BYTES!r})
elif mode == "identity":
    preserved = Path(os.environ["NCNN_FIXEDRUN_MUTATION_PRESERVED"])
    output.rename(preserved)
    shutil.copytree(preserved, output)
else:
    raise RuntimeError(f"unknown capture mutation mode: {{mode}}")
'''
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def capture_command(args: argparse.Namespace, output: Path, driver: Path) -> list[str]:
    return [
        str(args.capture_tool),
        str(args.param),
        "--bin",
        str(args.bin),
        "--model-name",
        MODEL,
        "--output-dir",
        str(output),
        "--backend=vulkan",
        "--precision=f32",
        "--driver",
        str(driver),
        "--opt",
        str(args.opt),
        "--translate",
        str(args.translate),
        "--clang",
        str(args.clang),
        "--nm",
        str(args.nm),
        "--readelf",
        str(args.readelf),
        "--llvm-as",
        str(args.llvm_as),
        "--vulkan-mlir-opt",
        str(args.mlir_opt),
        "--vulkan-spirv-val",
        str(args.spirv_val),
        "--vulkan-spirv-dis",
        str(args.spirv_dis),
        "--vulkan-spirv-extract",
        str(args.spirv_extract),
        "--vulkan-host-cxx",
        str(args.host_cxx),
        "--capture-vulkan-host-source",
    ]


def run_capture(args: argparse.Namespace,
                output: Path,
                driver: Path,
                env: dict[str, str],
                *,
                expect_success: bool) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        capture_command(args, output, driver),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"capture failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise RuntimeError("capture unexpectedly succeeded after output mutation")
    return result


def run_mutation_case(args: argparse.Namespace,
                      work: Path,
                      output: Path,
                      wrapper: Path,
                      mode: str,
                      expected_diagnostic: str,
                      previous: dict[str, bytes]) -> None:
    marker = work / f"{mode}-mutation-applied"
    preserved = work / f"{mode}-moved-original"
    env = os.environ.copy()
    env.update(
        {
            "NCNN_FIXEDRUN_MUTATION_MODE": mode,
            "NCNN_FIXEDRUN_MUTATION_MARKER": str(marker),
            "NCNN_FIXEDRUN_MUTATION_OUTPUT": str(output),
            "NCNN_FIXEDRUN_MUTATION_PRESERVED": str(preserved),
        }
    )
    result = run_capture(
        args, output, wrapper, env, expect_success=False
    )
    if not marker.is_file():
        raise RuntimeError("driver wrapper did not mutate after initial validation")
    if expected_diagnostic not in result.stderr:
        raise RuntimeError(
            f"{mode} mutation failed for the wrong reason: {result.stderr}"
        )

    current = snapshot(output)
    if mode == "unowned":
        expected = dict(previous)
        expected[UNOWNED_NAME] = UNOWNED_BYTES
        if current != expected:
            raise RuntimeError("failed unowned-entry publication changed prior bytes")
    else:
        if not preserved.is_dir():
            raise RuntimeError("identity mutation did not retain the moved original")
        if snapshot(preserved) != previous:
            raise RuntimeError("identity mutation altered the moved original bytes")
        if current != previous:
            raise RuntimeError("identity revalidation changed prior capture bytes")
        old_identity = preserved.stat()
        new_identity = output.stat()
        if (old_identity.st_dev, old_identity.st_ino) == (
            new_identity.st_dev,
            new_identity.st_ino,
        ):
            raise RuntimeError("identity mutation did not replace the directory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capture-tool",
        "param",
        "bin",
        "driver",
        "opt",
        "translate",
        "clang",
        "nm",
        "readelf",
        "llvm-as",
        "mlir-opt",
        "spirv-val",
        "spirv-dis",
        "spirv-extract",
        "host-cxx",
        "work-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fixedrun-capture-publish-", dir=args.work_dir
    ) as temporary:
        work = Path(temporary)
        output = work / "capture"
        env = os.environ.copy()

        run_capture(args, output, args.driver, env, expect_success=True)
        first = check_capture_inventory(output)

        stale_shader = output / "shader_999999.spv"
        stale_shader.write_bytes(b"stale shader from a previous generation")
        run_capture(args, output, args.driver, env, expect_success=True)
        repeated = check_capture_inventory(output)
        if repeated != first:
            raise RuntimeError("repeated capture changed deterministic artifact bytes")
        if stale_shader.exists():
            raise RuntimeError("replacement capture retained a stale shader")
        previous = dict(repeated)

        wrapper = work / "driver-mutation-wrapper"
        make_driver_wrapper(wrapper, args.driver)
        run_mutation_case(
            args,
            work,
            output,
            wrapper,
            "unowned",
            "unowned file",
            previous,
        )

        # The next invocation must start from a valid capture-owned directory.
        (output / UNOWNED_NAME).unlink()
        if snapshot(output) != previous:
            raise RuntimeError("test cleanup changed the previous capture bytes")
        run_mutation_case(
            args,
            work,
            output,
            wrapper,
            "identity",
            "output directory changed while compilation was in progress",
            previous,
        )
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"test_fixedrun_capture_publish.py: {error}", file=sys.stderr)
        sys.exit(1)
