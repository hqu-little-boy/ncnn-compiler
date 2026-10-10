#!/usr/bin/env python3
"""Transparently preserve the actual compiler's SPIR-V validator invocations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_launcher(validator, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    launcher = output / "spirv-val-recorder"
    if launcher.exists():
        raise ValueError(f"refusing to overwrite validator launcher: {launcher}")
    resolved_validator = shutil.which(str(validator))
    if resolved_validator is None:
        raise ValueError(f"validator is unavailable: {validator}")
    command = [sys.executable, str(Path(__file__).resolve()),
               "--validator", str(Path(resolved_validator).resolve()),
               "--output-dir", str(output), "--"]
    launcher.write_text(
        "#!" + sys.executable + "\n"
        "import os, sys\n"
        f"os.execv({sys.executable!r}, {command!r} + sys.argv[1:])\n",
        encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def record_call(validator, output, arguments):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        log = output / f"validation-{index:04d}.json"
        try:
            descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            index += 1
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        shader = None
        if arguments and Path(arguments[-1]).is_file():
            source = Path(arguments[-1])
            saved = output / f"shader-{index:04d}.spv"
            shutil.copyfile(source, saved)
            shader = {"path": saved.name, "sha256": sha256(saved),
                      "size_bytes": saved.stat().st_size}
        command = [str(validator), *arguments]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        record = {"schema_version": 1, "kind": "ncnn.spirv_tool_invocation",
                  "argv": command, "returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr,
                  "shader": shader}
        stream.write(json.dumps(record, indent=2) + "\n")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def validate_records(output, shader_hashes):
    output = Path(output)
    records = []
    observed = []
    for log in sorted(output.glob("validation-*.json")):
        record = json.loads(log.read_text(encoding="utf-8"))
        if (record.get("schema_version") != 1 or
                record.get("kind") != "ncnn.spirv_tool_invocation" or
                type(record.get("returncode")) is not int or
                record["returncode"] != 0):
            raise RuntimeError(f"recorded SPIR-V tool failed or is malformed: {log}")
        shader = record.get("shader")
        if shader is None:
            continue
        arguments = record.get("argv", [])
        if "--target-env" not in arguments:
            raise RuntimeError(f"validator did not declare its target environment: {log}")
        target_index = arguments.index("--target-env") + 1
        if target_index >= len(arguments) or arguments[target_index] != "vulkan1.1":
            raise RuntimeError(f"validator used an unexpected target environment: {log}")
        relative = Path(shader["path"])
        if relative.name != str(relative) or relative.suffix != ".spv":
            raise RuntimeError(f"recorded shader escapes its evidence directory: {log}")
        saved = output / relative
        digest = sha256(saved)
        if digest != shader["sha256"] or saved.stat().st_size != shader["size_bytes"]:
            raise RuntimeError(f"recorded shader identity changed: {saved}")
        observed.append(digest)
        records.append({"log": log.name, "log_sha256": sha256(log), **shader})
    if sorted(observed) != sorted(shader_hashes):
        raise RuntimeError(
            "recorded SPIR-V validations do not cover each product shader invocation")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return record_call(args.validator, args.output_dir, arguments)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"record_spirv_validation.py: {error}", file=sys.stderr)
        sys.exit(1)
