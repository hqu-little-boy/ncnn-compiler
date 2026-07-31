import argparse
import os
import pathlib
import subprocess
import sys
import time


def read_rss_kib(pid):
    try:
        status = pathlib.Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def has_mapped_library(pid, library):
    try:
        maps = pathlib.Path(f"/proc/{pid}/maps").read_text(encoding="ascii")
    except FileNotFoundError:
        return False
    return pathlib.Path(library).name in maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--mlir-runner", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--lowered", required=True)
    parser.add_argument("--runner-utils", required=True)
    parser.add_argument("--c-runner-utils", required=True)
    parser.add_argument("--asan-runtime")
    args = parser.parse_args()

    subprocess.run(
        [args.mlir_opt, "--test-lower-to-llvm", args.input, "-o", args.lowered],
        check=True,
    )

    environment = os.environ.copy()
    if args.asan_runtime:
        environment["LD_PRELOAD"] = args.asan_runtime
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        environment["LSAN_OPTIONS"] = "exitcode=23"

    command = [
        args.mlir_runner,
        args.lowered,
        "-e=main",
        "-entry-point-result=void",
        f"-shared-libs={args.runner_utils},{args.c_runner_utils}",
    ]
    process = subprocess.Popen(command, env=environment)
    samples = []
    sanitizer_loaded = not args.asan_runtime
    while process.poll() is None:
        if args.asan_runtime and not sanitizer_loaded:
            sanitizer_loaded = has_mapped_library(process.pid, args.asan_runtime)
        rss = read_rss_kib(process.pid)
        if rss is not None:
            samples.append(rss)
        time.sleep(0.005)
    if process.returncode != 0:
        return process.returncode
    if not sanitizer_loaded:
        print("ASan runtime was not loaded into the JIT process", file=sys.stderr)
        return 1
    if len(samples) < 8:
        print("insufficient RSS samples from repeated-call process", file=sys.stderr)
        return 1

    midpoint = len(samples) // 2
    first_peak = max(samples[:midpoint])
    second_peak = max(samples[midpoint:])
    tolerance_kib = 16 * 1024
    if second_peak > first_peak + tolerance_kib:
        print(
            f"RSS continued growing: first_peak={first_peak} KiB, "
            f"second_peak={second_peak} KiB",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
