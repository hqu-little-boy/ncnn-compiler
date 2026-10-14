#!/usr/bin/env python3
"""Run a fresh, unfiltered Vulkan engineering gate and preserve its evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

from gate_budget_evidence import verify_budget_evidence
from sanitizer_toolchain import audit_and_bundle


PROFILES = ("release", "sanitize", "offline", "software", "hardware")
STEP_NAMES = ("configure", "format_check", "build", "numerical", "tidy", "ctest")
SANITIZER_BUDGET_EXCEPTION_LIMITS = {
    "ncnn_model_compile_budget_pp_ocrv5_server_det_static": 450,
    "ncnn_model_compile_budget_pp_ocrv5_server_rec": 450,
}
SANITIZER_REPORT = re.compile(
    r"ERROR: (?:AddressSanitizer|LeakSanitizer)|"
    r"SUMMARY: (?:AddressSanitizer|UndefinedBehaviorSanitizer)|"
    r"UndefinedBehaviorSanitizer:|runtime error:")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def git_output(source, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], stderr=subprocess.PIPE)


def gitlink_oid(source, relative_path):
    entries = git_output(source, "ls-files", "--stage", "-z", "--",
                         os.fsdecode(relative_path)).split(b"\0")
    for entry in entries:
        if not entry:
            continue
        metadata, separator, recorded_path = entry.partition(b"\t")
        fields = metadata.split()
        if separator and recorded_path == relative_path and len(fields) == 3 and fields[0] == b"160000":
            return fields[1].decode("ascii")
    return None


def source_fingerprint(source):
    """Hash tracked and nonignored files independent of staging or HEAD."""
    source = Path(source).resolve()
    paths = set(git_output(source, "ls-files", "-z").split(b"\0"))
    paths.update(git_output(source, "ls-files", "--others", "--exclude-standard",
                            "-z").split(b"\0"))
    paths.discard(b"")
    digest = hashlib.sha256()
    for raw in sorted(paths):
        path = source / os.fsdecode(raw)
        expected_gitlink = gitlink_oid(source, raw)
        if expected_gitlink is not None:
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                content = b"gitlink:" + expected_gitlink.encode("ascii")
            else:
                top = git_output(path, "rev-parse", "--show-toplevel").decode().strip()
                if Path(top).resolve() != path.resolve():
                    raise ValueError(f"gitlink is populated by a non-repository: {path}")
                actual_gitlink = git_output(path, "rev-parse", "HEAD").decode().strip()
                if actual_gitlink != expected_gitlink:
                    raise ValueError(f"gitlink HEAD differs from recorded object: {path}")
                if git_output(path, "status", "--porcelain", "--untracked-files=normal"):
                    raise ValueError(f"gitlink worktree is dirty: {path}")
                content = b"gitlink:" + expected_gitlink.encode("ascii")
        elif path.is_symlink():
            content = b"symlink:" + os.fsencode(os.readlink(path))
        elif path.is_file():
            content = b"file:" + sha256(path).encode("ascii")
        elif path.is_dir():
            raise ValueError(f"tracked source directory is not a gitlink or file: {path}")
        else:
            content = b"deleted"
        digest.update(raw + b"\0" + content + b"\0")
    return digest.hexdigest()


def snapshot_worktree(source, stage):
    status = git_output(source, "status", "--porcelain=v1", "--untracked-files=normal")
    diff = git_output(source, "diff", "--binary", "HEAD", "--")
    status_path = stage / "gate-source-status.txt"
    diff_path = stage / "gate-source-diff.patch"
    status_path.write_bytes(status)
    diff_path.write_bytes(diff)
    return {"status": artifact(stage, status_path),
            "tracked_diff": artifact(stage, diff_path), "dirty": bool(status)}


def cache_values(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.split(":", 1)[0]] = value
    return result


def artifact(stage, path):
    path = Path(path)
    return {"path": str(path.relative_to(stage)), "sha256": sha256(path)}


def configuration(args, stage):
    source = args.source_dir.resolve()
    for key in ("ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS",
                "VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING"):
        if os.environ.get(key):
            raise ValueError(f"gate forbids inherited sanitizer/cleanup overrides: {key}")
    command = ["cmake", "-S", str(source), "-B", str(stage),
               "-DBUILD_TESTING=ON", "-DCOMPILER_INSTALL_RPATH=ON",
               "-DCOMPILER_ENABLE_FORMAT_TARGETS=ON",
               f"-DCOMPILER_TIDY_JOBS={args.tidy_jobs}",
               "-DNCNN_ENABLE_VULKAN_PRODUCT=ON"]
    if args.profile == "offline":
        command.append("-DNCNN_ENABLE_VULKAN_V0_TESTS=OFF")
    else:
        devices = {"software": "llvmpipe", "hardware": "nvidia"}.get(
            args.profile, "nvidia;llvmpipe")
        command.extend(["-DNCNN_ENABLE_VULKAN_V0_TESTS=ON",
                        f"-DNCNN_VULKAN_TEST_DEVICES={devices}"])
    if args.profile == "sanitize":
        required = (args.sanitizer_install, args.lit, args.openmp_dir,
                    args.toolchain_identity)
        if not all(required):
            raise ValueError("sanitize requires --sanitizer-install, --lit, "
                             "--openmp-dir and --toolchain-identity")
        install = args.sanitizer_install.resolve()
        for path in (install / "lib/cmake/llvm/LLVMConfig.cmake",
                     install / "lib/cmake/mlir/MLIRConfig.cmake",
                     install / "bin/mlir-opt", install / "bin/mlir-translate",
                     install / "bin/FileCheck", args.lit,
                     args.openmp_dir / "libomp.so", args.toolchain_identity):
            if not path.is_file():
                raise ValueError(f"missing sanitizer prerequisite: {path}")
        if (args.openmp_dir / "libgomp.so").exists():
            raise ValueError("isolated OpenMP directory must not contain libgomp")
        command.extend([
            "-DCMAKE_BUILD_TYPE=Debug", "-DCMAKE_C_FLAGS_DEBUG=-O3 -g",
            "-DCMAKE_CXX_FLAGS_DEBUG=-O3 -g",
            f"-DCMAKE_C_COMPILER={args.clang_dir / 'clang'}",
            f"-DCMAKE_CXX_COMPILER={args.clang_dir / 'clang++'}",
            f"-DLLVM_DIR={install / 'lib/cmake/llvm'}",
            f"-DMLIR_DIR={install / 'lib/cmake/mlir'}",
            "-DCOMPILER_SANITIZERS=address;undefined",
            f"-DNCNN_SANITIZER_OPENMP_LIBRARY_DIR={args.openmp_dir.resolve()}",
            f"-DNCNN_LIT_PY={args.lit.resolve()}",
            f"-DFILECHECK={install / 'bin/FileCheck'}",
            f"-DNCNN_MLIR_OPT_EXECUTABLE={install / 'bin/mlir-opt'}",
            f"-DMLIR_OPT={install / 'bin/mlir-opt'}",
            f"-DMLIR_TRANSLATE={install / 'bin/mlir-translate'}",
        ])
    else:
        command.append("-DCMAKE_BUILD_TYPE=Release")
    protected = {part.split("=", 1)[0].split(":", 1)[0]
                 for part in command if part.startswith("-D")}
    budget_proofs = {}
    for item in args.budget_evidence:
        if "=" not in item:
            raise ValueError("--budget-evidence must be FIXTURE=JSON")
        fixture, path = item.split("=", 1)
        if not fixture or not path or fixture in budget_proofs:
            raise ValueError("--budget-evidence entries must be unique FIXTURE=JSON")
        budget_proofs[fixture] = Path(path)
    accepted_budget_evidence = []
    for argument in args.cmake_arg:
        if not argument.startswith("-D") or "=" not in argument:
            raise ValueError("--cmake-arg accepts only -Dkey=value arguments")
        key, value = argument.split("=", 1)
        key = key.split(":", 1)[0]
        if key in protected or key in ("-DCMAKE_C_FLAGS", "-DCMAKE_CXX_FLAGS"):
            raise ValueError(f"cannot override gate configuration {key}")
        if key.startswith("-Dncnn_model_compile_budget_"):
            if args.profile != "sanitize" or not value.isdecimal():
                raise ValueError("compile budget overrides are sanitizer-only")
            variable = key[2:]
            fixture = variable.removeprefix("ncnn_model_compile_budget_")
            proof = budget_proofs.get(fixture)
            if proof is None:
                raise ValueError("compile budget overrides require matching "
                                 "--budget-evidence FIXTURE=JSON")
            accepted_budget_evidence.append(verify_budget_evidence(
                proof, variable, int(value), source_fingerprint(source)))
        command.append(argument)
    if set(budget_proofs) != {
            record["record"]["fixture"] for record in accepted_budget_evidence}:
        raise ValueError("unused or mismatched sanitizer budget evidence was provided")
    args.accepted_budget_evidence = accepted_budget_evidence
    return command


def diagnostics(paths):
    findings = []
    for path in paths:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if SANITIZER_REPORT.search(line):
                    findings.append({"path": str(path), "line": line_number,
                                     "diagnostic": line.rstrip()})
    return findings


def project_warnings(path, source):
    findings = []
    source = str(source.resolve()) + "/"
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            source_location = re.search(
                r"^(.*\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)):\d+:\d+: (?:warning|error):",
                line)
            if (source in line and "/third_party/" not in line and
                    source_location is not None):
                findings.append({"line": line_number, "diagnostic": line.rstrip()})
    return findings


def built_tool_identities(stage):
    identities = []
    for relative, versioned in (
            ("tools/ncnn-compile", True),
            ("tools/ncnn-mlir-driver", True),
            ("bin/ncnn-mlir-opt", True),
            ("tools/ncnn-vulkan-spirv-extract", False)):
        path = stage / relative
        if not path.is_file():
            raise RuntimeError(f"fresh build did not produce expected tool: {path}")
        record = {"path": relative, "sha256": sha256(path)}
        if versioned:
            record["version"] = subprocess.check_output(
                [str(path), "--version"], text=True, stderr=subprocess.STDOUT)
        identities.append(record)
    return identities


def audit_static_linkage(stage):
    records = []
    for name in ("ncnn-compile", "ncnn-mlir-driver", "ncnn-mlir-opt"):
        path = stage / "tools" / name
        if not path.is_file():
            matches = list(stage.glob(f"**/{name}"))
            matches = [match for match in matches if match.is_file()]
            if len(matches) != 1:
                raise RuntimeError(f"cannot identify sanitizer executable {name}")
            path = matches[0]
        output = subprocess.check_output(["readelf", "-d", str(path)], text=True)
        needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", output)
        if any(re.match(r"lib(?:LLVM|MLIR).*\.so", item) for item in needed):
            raise RuntimeError(f"sanitizer executable links core LLVM/MLIR DSO: {path}")
        if not any("libclang_rt.asan" in item for item in needed):
            raise RuntimeError(f"sanitizer executable lacks shared ASan runtime: {path}")
        records.append({"executable": artifact(stage, path), "needed": needed})
    return records


def tool_identities(args, cache):
    tools = {"cmake": shutil.which("cmake"),
             "build-tool": cache.get("CMAKE_MAKE_PROGRAM"),
             "clang-format": shutil.which("clang-format-21"),
             "clang-tidy": shutil.which("clang-tidy-21"),
             "c-compiler": cache.get("CMAKE_C_COMPILER"),
             "cxx-compiler": cache.get("CMAKE_CXX_COMPILER"),
             "clang": cache.get("CLANG"),
             "llvm-nm": cache.get("LLVM_NM"),
             "llvm-readelf": cache.get("LLVM_READELF"),
             "llvm-as": cache.get("LLVM_AS"),
             "llvm-objdump": cache.get("LLVM_OBJDUMP"),
             "llvm-size": cache.get("LLVM_SIZE"),
             "llvm-mca": cache.get("LLVM_MCA"),
             "spirv-val": cache.get("NCNN_SPIRV_VAL_EXECUTABLE"),
             "spirv-dis": cache.get("NCNN_SPIRV_DIS_EXECUTABLE"),
             "mlir-opt": cache.get("NCNN_MLIR_OPT_EXECUTABLE"),
             "mlir-translate": cache.get("MLIR_TRANSLATE"),
             "lit-mlir-opt": cache.get("MLIR_OPT"),
             "filecheck": cache.get("FILECHECK"),
             "python": cache.get("Python3_EXECUTABLE") or shutil.which("python3"),
             "readelf-link-audit": shutil.which("readelf")}
    if cache.get("NCNN_ENABLE_VULKAN_V0_TESTS") == "ON":
        tools["vulkaninfo"] = cache.get("VULKANINFO")
    identities = []
    for role, value in tools.items():
        if not value or not Path(value).is_file():
            raise RuntimeError(f"missing gate tool identity: {role} ({value})")
        path = Path(value).resolve()
        version_args = ["--summary"] if role == "vulkaninfo" else ["--version"]
        version = subprocess.check_output([str(path), *version_args], text=True,
                                          stderr=subprocess.STDOUT)
        identities.append({"role": role, "path": str(path),
                           "argv": version_args, "sha256": sha256(path),
                           "version": version})
    lit = cache.get("NCNN_LIT_PY")
    if not lit:
        mlir_tool = cache.get("MLIR_OPT") or cache.get("MLIR_TRANSLATE")
        if mlir_tool:
            prefix = Path(mlir_tool).resolve().parent.parent
            candidate = prefix / "build/utils/lit/lit.py"
            if candidate.is_file():
                lit = str(candidate)
    if not lit:
        raise RuntimeError("cannot locate the matching LLVM lit.py runner")
    for role, path in (("lit", lit),
                       ("libomp", args.openmp_dir / "libomp.so"
                        if args.openmp_dir else None),
                       ("sanitizer-toolchain", args.toolchain_identity)):
        if path:
            path = Path(path).resolve()
            if not path.is_file():
                raise RuntimeError(f"missing gate dependency identity: {path}")
            identities.append({"role": role, "path": str(path),
                               "sha256": sha256(path)})
    return identities


def snapshot_budget_evidence(stage, records):
    snapshots = []
    for item in records:
        record = item["record"]
        fixture = record["fixture"]
        directory = stage / "gate-budget-evidence" / fixture
        directory.mkdir(parents=True)
        source_record = Path(item["record_path"])
        source_log = Path(item["log_path"])
        record_copy = directory / "budget-evidence.json"
        log_copy = directory / source_log.name
        shutil.copyfile(source_record, record_copy)
        shutil.copyfile(source_log, log_copy)
        snapshots.append({
            "fixture": fixture,
            "record": artifact(stage, record_copy),
            "log": artifact(stage, log_copy),
            "requested_budget_seconds": record["requested_budget_seconds"],
            "elapsed_seconds": record["elapsed_seconds"],
        })
    return snapshots


def _verify_archive_contents(destination, *, manifest_sha256=None):
    """Reopen an evidence tarball and verify its exact file inventory/bytes."""
    destination = Path(destination)
    with tarfile.open(destination, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            name = member.name
            parts = Path(name).parts
            if (not member.isfile() or not name or Path(name).is_absolute() or
                    ".." in parts or name != Path(name).as_posix() or name in members):
                raise ValueError(f"unsafe or duplicate evidence archive member: {name}")
            members[name] = member
        manifest_name = "gate-archive-manifest.json"
        if manifest_name not in members:
            raise ValueError("evidence archive has no hash manifest")
        stream = archive.extractfile(members[manifest_name])
        if stream is None:
            raise ValueError("evidence archive manifest cannot be read")
        with stream:
            manifest_bytes = stream.read()
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_sha256 is not None and actual_manifest_sha256 != manifest_sha256:
            raise ValueError("evidence archive manifest differs from the frozen stage manifest")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
                manifest.get("kind") != "ncnn.vulkan_evidence_archive" or
                not isinstance(files, dict) or not files):
            raise ValueError("evidence archive manifest is empty or malformed")
        if set(members) != set(files) | {manifest_name}:
            raise ValueError("evidence archive member inventory differs from its manifest")
        total_bytes = 0
        # The tar members are written in sorted path order. Iterate in that same
        # order: random backwards seeks in a compressed stream repeatedly
        # decompress every earlier member and can multiply work by thousands.
        for name in sorted(files):
            expected = files[name]
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"evidence archive has an invalid digest for {name}")
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError(f"evidence archive member cannot be read: {name}")
            digest = hashlib.sha256()
            size = 0
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if size != members[name].size or digest.hexdigest() != expected:
                raise ValueError(f"evidence archive member hash/size differs: {name}")
            total_bytes += size
    return {"status": "verified", "manifest_sha256": actual_manifest_sha256,
            "file_count": len(files), "uncompressed_file_bytes": total_bytes}


def verify_archive(destination, *, manifest_sha256=None):
    try:
        return _verify_archive_contents(destination, manifest_sha256=manifest_sha256)
    except (tarfile.TarError, EOFError) as error:
        raise ValueError(f"evidence archive cannot be read completely: {error}") from error


def archive_stage(stage, destination):
    """Bundle raw gate and model evidence, not unrelated numerical build products."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"refusing to overwrite evidence archive {destination}")
    candidates = [path for path in stage.glob("gate-*.json")
                  if path.name not in {"gate-archive-manifest.json", "gate-archive-receipt.json"}]
    candidates += list(stage.glob("gate-*.log"))
    candidates += list(stage.glob("gate-*.xml"))
    candidates += [stage / "gate-source-status.txt", stage / "gate-source-diff.patch"]
    candidates += [stage / "release-evidence.json", stage / "CMakeCache.txt",
                   stage / "compile_commands.json", stage / "test/gate-tests.json"]
    candidates += [stage / "test/Vulkan/evidence",
                   stage / "test/Native/vulkan-v1-cpu-baseline",
                   stage / "test/Native/vulkan-product-offline",
                   stage / "gate-source-inputs",
                   stage / "gate-budget-evidence",
                   stage / "toolchain-evidence",
                   stage / "legacy-reference"]
    candidates += list((stage / "test/Vulkan").glob("cpu-*"))
    candidates += list((stage / "test/Native").glob("vulkan-product-cli*"))
    files = {}
    for candidate in candidates:
        if candidate.is_dir():
            archive_paths = [path for path in candidate.rglob("*") if path.is_file()]
        elif candidate.is_file():
            archive_paths = [candidate]
        else:
            continue
        for path in archive_paths:
            resolved = path.resolve()
            try:
                resolved.relative_to(stage.resolve())
            except ValueError as error:
                raise ValueError(f"evidence archive path escapes its stage: {path}") from error
            files[str(path.relative_to(stage))] = sha256(resolved)
    manifest = stage / "gate-archive-manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "kind": "ncnn.vulkan_evidence_archive",
                                    "files": files}, indent=2) + "\n",
                        encoding="utf-8")
    with tarfile.open(destination, "x:gz", dereference=True) as archive:
        archive.add(manifest, arcname="gate-archive-manifest.json")
        for relative in sorted(files):
            archive.add(stage / relative, arcname=relative, recursive=False)
    verified = verify_archive(destination, manifest_sha256=sha256(manifest))
    state_path = stage / "gate-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    receipt = {"schema_version": 1, "kind": "ncnn.vulkan_evidence_archive_receipt",
               "archive": {"file": destination.name, "sha256": sha256(destination)},
               "verification": verified, "source_fingerprint": state.get("source_fingerprint"),
               "profile": state.get("profile"), "gate_completed": state.get("completed"),
               "verified_at": timestamp()}
    # Keep the archive hash outside the archive itself to avoid a recursive identity.
    receipt_path = stage / "gate-archive-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Archive bytes verified: {destination} ({receipt['archive']['sha256']})", flush=True)
    return {"path": str(destination), "sha256": receipt["archive"]["sha256"],
            "receipt": str(receipt_path), "verification": verified}


LEGACY_REFERENCE_FINGERPRINT = (
    "6d2f2b2ecacab5a17385ea9acdf67482f321bb85772d513739cd2186edab20d3")
LEGACY_REFERENCE_REVISION = "310276ec49cccbcb309ddb0f002d98a5dab7fb9d"
LEGACY_REFERENCE_METADATA = (
    "gate-state.json", "release-evidence.json", "gate-va0-legacy-source.json",
    "gate-ctest.xml", "gate-ctest-lasttest.log", "gate-ctest.log",
    "test/gate-tests.json", "gate-archive-manifest.json")
LEGACY_REFERENCE_FIXTURES = {
    "relu-product-nvidia": {
        "files": ("evidence.json", "model.param", "model.bin", "input.f32",
                  "cpu-output.f32", "vulkan-output.f32", "scalar-golden.f32",
                  "loader-result.json", "validation.log", "vulkan-verify.stdout.log",
                  "vulkan-verify.stderr.log", "vulkan-loader.stdout.log",
                  "vulkan-loader.stderr.log", "vulkaninfo.stdout.log",
                  "vulkaninfo.stderr.log"),
        "trees": ("cpu", "vulkan"),
        "source": "test/Native/vulkan-product-cli",
    },
    "relu-product-llvmpipe": {
        "files": ("evidence.json", "model.param", "model.bin", "input.f32",
                  "cpu-output.f32", "vulkan-output.f32", "scalar-golden.f32",
                  "loader-result.json", "validation.log", "vulkan-verify.stdout.log",
                  "vulkan-verify.stderr.log", "vulkan-loader.stdout.log",
                  "vulkan-loader.stderr.log", "vulkaninfo.stdout.log",
                  "vulkaninfo.stderr.log"),
        "trees": ("cpu", "vulkan"),
        "source": "test/Native/vulkan-product-cli-llvmpipe",
    },
    "add-relu-nvidia": {
        "source": "test/Vulkan/evidence/v5/nvidia/models/add_relu",
        "trees": (".",),
        "source_inputs": ("test/Vulkan/evidence/v5/nvidia/fixtures/add_relu.param",
                          "test/Vulkan/evidence/v5/nvidia/fixtures/empty.bin"),
    },
    "add-relu-llvmpipe": {
        "source": "test/Vulkan/evidence/v5/llvmpipe/models/add_relu",
        "trees": (".",),
        "source_inputs": ("test/Vulkan/evidence/v5/llvmpipe/fixtures/add_relu.param",
                          "test/Vulkan/evidence/v5/llvmpipe/fixtures/empty.bin"),
    },
    "conv-relu-nvidia": {
        "source": "test/Vulkan/evidence/v5/nvidia/models/conv_relu",
        "trees": (".",),
        "source_inputs": ("test/Vulkan/evidence/v5/nvidia/fixtures/conv_relu.param",
                          "test/Vulkan/evidence/v5/nvidia/fixtures/conv_weights.bin"),
    },
    "conv-relu-llvmpipe": {
        "source": "test/Vulkan/evidence/v5/llvmpipe/models/conv_relu",
        "trees": (".",),
        "source_inputs": ("test/Vulkan/evidence/v5/llvmpipe/fixtures/conv_relu.param",
                          "test/Vulkan/evidence/v5/llvmpipe/fixtures/conv_weights.bin"),
    },
    "squeezenet-nvidia": {
        "source": "test/Vulkan/evidence/v5/squeezenet-nvidia/models/squeezenet_v1_1",
        "trees": (".",),
    },
    "squeezenet-llvmpipe": {
        "source": "test/Vulkan/evidence/v5/squeezenet-llvmpipe/models/squeezenet_v1_1",
        "trees": (".",),
    },
}
LEGACY_REFERENCE_SUITE_FILES = {
    "add-relu-nvidia": "test/Vulkan/evidence/v5/nvidia",
    "add-relu-llvmpipe": "test/Vulkan/evidence/v5/llvmpipe",
    "conv-relu-nvidia": "test/Vulkan/evidence/v5/nvidia",
    "conv-relu-llvmpipe": "test/Vulkan/evidence/v5/llvmpipe",
    "squeezenet-nvidia": "test/Vulkan/evidence/v5/squeezenet-nvidia",
    "squeezenet-llvmpipe": "test/Vulkan/evidence/v5/squeezenet-llvmpipe",
}


def _copy_reference_file(source_stage, output_root, source_relative,
                         target_relative, source_manifest, copied, label):
    source = source_stage / source_relative
    if not source.is_file():
        raise ValueError(f"legacy reference is missing {source_relative}")
    expected = source_manifest.get(source_relative)
    actual = sha256(source)
    if expected != actual:
        raise ValueError(f"legacy reference archive manifest mismatch for {source_relative}")
    target = output_root / target_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied.append({"source": source_relative, "path": str(target.relative_to(output_root.parent)),
                   "sha256": actual, "role": label})


def _check_legacy_reference_source(source_stage):
    source_stage = Path(source_stage).resolve()
    try:
        state = json.loads((source_stage / "gate-state.json").read_text(encoding="utf-8"))
        source_status = json.loads(
            (source_stage / "gate-va0-legacy-source.json").read_text(encoding="utf-8"))
        report = json.loads((source_stage / "release-evidence.json").read_text(encoding="utf-8"))
        archive_manifest = json.loads(
            (source_stage / "gate-archive-manifest.json").read_text(encoding="utf-8"))
        inventory = json.loads((source_stage / "test/gate-tests.json").read_text(encoding="utf-8"))
        lasttest_text = (source_stage / "gate-ctest-lasttest.log").read_text(encoding="utf-8")
        junit = ET.parse(source_stage / "gate-ctest.xml").getroot()
    except (OSError, json.JSONDecodeError, ET.ParseError) as error:
        raise ValueError(f"cannot read frozen legacy gate evidence: {error}") from error
    if (state.get("kind") != "ncnn.vulkan_gate" or state.get("profile") != "release" or
            state.get("completed") is not True or
            state.get("source_fingerprint") != LEGACY_REFERENCE_FINGERPRINT or
            state.get("git_revision") != LEGACY_REFERENCE_REVISION):
        raise ValueError("legacy baseline is not the frozen completed VA0 Release source")
    steps = state.get("steps")
    if (not isinstance(steps, list) or
            [item.get("name") for item in steps if isinstance(item, dict)] !=
            ["configure", "format_check", "build", "numerical", "tidy", "ctest"] or
            any(type(item.get("exit_code")) is not int or item["exit_code"] != 0
                for item in steps if isinstance(item, dict))):
        raise ValueError("legacy baseline gate did not complete all six steps with zero failures")
    if (source_status.get("role") != "legacy-reference" or
            source_status.get("stage_id") != "VA0.1" or
            source_status.get("git_revision") != LEGACY_REFERENCE_REVISION or
            source_status.get("source_fingerprint") != LEGACY_REFERENCE_FINGERPRINT or
            source_status.get("worktree_porcelain") != "" or
            source_status.get("host_aot_candidate") is not False):
        raise ValueError("legacy baseline original source identity/status is invalid")
    if (report.get("kind") != "ncnn.vulkan_release_evidence" or
            report.get("source_fingerprint") != LEGACY_REFERENCE_FINGERPRINT or
            report.get("blocking_findings")):
        raise ValueError("legacy baseline original evidence report is invalid")
    report_stages = report.get("stages", [])
    release = [item for item in report_stages if item.get("profile") == "release"]
    if len(release) != 1 or release[0].get("scope_passed") is not True or release[0].get("findings"):
        raise ValueError("legacy baseline release report scope did not pass")
    if (archive_manifest.get("kind") != "ncnn.vulkan_evidence_archive" or
            archive_manifest.get("schema_version") != 1 or
            not isinstance(archive_manifest.get("files"), dict)):
        raise ValueError("legacy baseline archive hash manifest is invalid")
    tests = inventory.get("tests")
    cases = list(junit.iter("testcase"))
    skipped = [case for case in cases if case.find("skipped") is not None]
    failed = [case for case in cases if case.find("failure") is not None or
              case.find("error") is not None]
    inventory_names = {item.get("name") for item in tests if isinstance(item, dict)} \
        if isinstance(tests, list) else set()
    junit_names = {case.get("name") for case in cases}
    lasttest_names = re.findall(r"^\d+/\d+ Testing: (.+)$", lasttest_text, re.MULTILINE)
    if (not isinstance(tests, list) or len(tests) != 525 or len(cases) != 525 or
            len(skipped) != 2 or failed or len(inventory_names) != 525 or
            junit_names != inventory_names or len(lasttest_names) != 525 or
            set(lasttest_names) != inventory_names or "Test Failed." in lasttest_text or
            "Test Not Run." in lasttest_text or
            {case.get("name") for case in skipped} != {
                "PerformanceModel.PPLCNetDocOriInt8",
                "PerformanceModel.PPLCNetTextlineOriInt8"}):
        raise ValueError("legacy baseline JUnit/inventory/LastTest is not the recorded 525/523/2/0 run")
    for key, expected_path in (("inventory", "test/gate-tests.json"),
                               ("junit", "gate-ctest.xml"),
                               ("lasttest", "gate-ctest-lasttest.log")):
        record = state.get(key)
        if (not isinstance(record, dict) or record.get("path") != expected_path or
                record.get("sha256") != sha256(source_stage / expected_path)):
            raise ValueError(f"legacy baseline gate-state {key} hash reference is invalid")
    required = set(LEGACY_REFERENCE_METADATA) - {"gate-archive-manifest.json"}
    for fixture in LEGACY_REFERENCE_FIXTURES.values():
        fixture_root = fixture["source"]
        required.add(f"{fixture_root}/evidence.json")
        required.update(fixture.get("source_inputs", ()))
        if fixture_root.startswith("test/Vulkan/"):
            suite_root = LEGACY_REFERENCE_SUITE_FILES.get(next(
                key for key, value in LEGACY_REFERENCE_FIXTURES.items() if value is fixture))
            if suite_root:
                required.add(f"{suite_root}/evidence.json")
                required.add(f"{suite_root}/vulkaninfo-summary.txt")
    missing = sorted(path for path in required if path not in archive_manifest["files"])
    if missing:
        raise ValueError(f"legacy baseline archive hash manifest omits required files: {missing}")
    for relative in required:
        path = source_stage / relative
        if not path.is_file() or sha256(path) != archive_manifest["files"].get(relative):
            raise ValueError(f"legacy baseline artifact does not match its hash manifest: {relative}")
    return source_stage, archive_manifest["files"]


def snapshot_legacy_baseline(source_stage, destination_stage):
    """Snapshot the frozen V6 Release reference into a relocatable stage subtree."""
    destination_stage = Path(destination_stage).resolve()
    reference_root = destination_stage / "legacy-reference"
    if reference_root.exists():
        raise ValueError(f"refusing to overwrite legacy reference snapshot: {reference_root}")
    source_stage, source_manifest = _check_legacy_reference_source(source_stage)
    copied = []
    manifest_for_copy = dict(source_manifest)
    manifest_for_copy["gate-archive-manifest.json"] = sha256(
        source_stage / "gate-archive-manifest.json")
    for relative in LEGACY_REFERENCE_METADATA:
        _copy_reference_file(source_stage, reference_root, relative,
                             Path("source") / relative, manifest_for_copy,
                             copied, "legacy-reference gate provenance")
    fixtures = {}
    for fixture, spec in LEGACY_REFERENCE_FIXTURES.items():
        source_root = source_stage / spec["source"]
        if not source_root.is_dir():
            raise ValueError(f"legacy reference fixture directory is missing: {spec['source']}")
        target_root = reference_root / "fixtures" / fixture
        target_root.mkdir(parents=True)
        files = list(spec.get("files", ()))
        for name in files:
            relative = f"{spec['source']}/{name}"
            target = Path("fixtures") / fixture / name
            _copy_reference_file(source_stage, reference_root, relative, target,
                                 source_manifest, copied, f"legacy-reference {fixture}")
        if spec.get("trees") == (".",):
            for source_file in sorted(source_root.rglob("*")):
                if not source_file.is_file():
                    continue
                relative = source_file.relative_to(source_stage).as_posix()
                target = Path("fixtures") / fixture / source_file.relative_to(source_root)
                _copy_reference_file(source_stage, reference_root, relative, target,
                                     source_manifest, copied,
                                     f"legacy-reference {fixture}")
        else:
            for tree in spec.get("trees", ()):
                source_tree = source_root / tree
                if not source_tree.is_dir():
                    raise ValueError(f"legacy reference fixture tree is missing: {source_tree}")
                for source_file in sorted(source_tree.rglob("*")):
                    if source_file.is_file():
                        relative = source_file.relative_to(source_stage).as_posix()
                        target = Path("fixtures") / fixture / tree / source_file.relative_to(source_tree)
                        _copy_reference_file(source_stage, reference_root, relative, target,
                                             source_manifest, copied,
                                             f"legacy-reference {fixture}")
        for source_relative in spec.get("source_inputs", ()):
            target = (Path("fixtures") / fixture / "inputs" /
                      Path(source_relative).name)
            _copy_reference_file(source_stage, reference_root, source_relative, target,
                                 source_manifest, copied,
                                 f"legacy-reference {fixture} source model input")
        suite_root = LEGACY_REFERENCE_SUITE_FILES.get(fixture)
        if suite_root:
            for name in ("evidence.json", "vulkaninfo-summary.txt"):
                relative = f"{suite_root}/{name}"
                target = Path("fixtures") / fixture / "device" / name
                _copy_reference_file(source_stage, reference_root, relative, target,
                                     source_manifest, copied,
                                     f"legacy-reference {fixture} device")
        fixtures[fixture] = {
            "source": spec["source"],
            "path": str(target_root.relative_to(destination_stage)),
            "files": [item["path"] for item in copied
                      if item["path"].startswith(str(target_root.relative_to(destination_stage)) + "/")],
        }
    snapshot = {
        "schema_version": 1,
        "kind": "ncnn.vulkan_legacy_reference_snapshot",
        "role": "legacy-reference",
        "status": "captured",
        "identity": {"profile": "release", "source_fingerprint": LEGACY_REFERENCE_FINGERPRINT,
                     "git_revision": LEGACY_REFERENCE_REVISION,
                     "host_aot_candidate": False},
        "source_stage_name": source_stage.name,
        "source_archive_manifest_sha256": sha256(source_stage / "gate-archive-manifest.json"),
        "gate_artifacts": [item for item in copied
                           if item["role"] == "legacy-reference gate provenance"],
        "fixtures": fixtures,
        "files": copied,
    }
    snapshot_path = reference_root / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return {"status": "captured", "path": str(snapshot_path.relative_to(destination_stage)),
            "sha256": sha256(snapshot_path), "source_fingerprint": LEGACY_REFERENCE_FINGERPRINT,
            "git_revision": LEGACY_REFERENCE_REVISION, "fixture_count": 4,
            "fixture_profile_record_count": len(fixtures), "file_count": len(copied)}


class Gate:
    def __init__(self, args):
        self.args = args
        self.source = args.source_dir.resolve()
        args.build_root.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(
            prefix=f"ncnn-compiler-stage-va0-{args.profile}-", dir=args.build_root))
        self.state = {
            "schema_version": 1, "kind": "ncnn.vulkan_gate",
            "profile": args.profile, "source_dir": str(self.source),
            "source_fingerprint_algorithm": "sha256-git-files-v1",
            "source_fingerprint": None, "git_revision": None,
            "build_dir": str(self.stage), "created_at": timestamp(),
            "configuration": [], "steps": [], "completed": False,
            "sanitizer_budget_evidence": [],
            "sanitizer_diagnostics": [],
        }
        self.save()
        try:
            self.state["source_fingerprint"] = source_fingerprint(self.source)
            self.state["git_revision"] = git_output(
                self.source, "rev-parse", "HEAD").decode().strip()
            self.state["worktree"] = snapshot_worktree(self.source, self.stage)
            self.state["stage_id"] = "VA0"
            self.state["host_execution_role"] = "legacy-product"
            self.state["host_aot_candidate"] = False
            self.config = configuration(args, self.stage)
            self.state["configuration"] = self.config
            self.state["sanitizer_budget_evidence"] = snapshot_budget_evidence(
                self.stage, args.accepted_budget_evidence)
            self.save()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            self.state["failure"] = f"gate setup failed: {error}"
            self.state["finished_at"] = timestamp()
            self.save()
            archive_note = ""
            if args.archive:
                archive = self.stage.parent / f"{self.stage.name}.evidence.tar.gz"
                try:
                    archive_stage(self.stage, archive)
                    archive_note = f"; archived at {archive}"
                except (OSError, ValueError, tarfile.TarError) as archive_error:
                    archive_note = f"; archive failed: {archive_error}"
            raise RuntimeError(
                f"gate setup failed; preserved stage {self.stage}{archive_note}: {error}") from error

    def save(self):
        (self.stage / "gate-state.json").write_text(
            json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def check_source(self):
        if source_fingerprint(self.source) != self.state["source_fingerprint"]:
            raise RuntimeError("source changed during gate; start another fresh stage")

    def step(self, name, argv):
        self.check_source()
        log = self.stage / f"gate-{name}.log"
        record = {"name": name, "argv": [str(item) for item in argv],
                  "started_at": timestamp(), "exit_code": None,
                  "log": str(log.relative_to(self.stage))}
        self.state["steps"].append(record)
        self.save()
        print(f"[{self.args.profile}] {name}: {self.stage}", flush=True)
        with log.open("wb") as stream:
            result = subprocess.run(record["argv"], stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        record.update(exit_code=result.returncode, finished_at=timestamp(),
                      sha256=sha256(log))
        self.save()
        if name == "ctest":
            lasttest = self.stage / "Testing/Temporary/LastTest.log"
            if lasttest.exists():
                saved = self.stage / "gate-ctest-lasttest.log"
                shutil.copyfile(lasttest, saved)
                self.state["lasttest"] = artifact(self.stage, saved)
            junit = self.stage / "gate-ctest.xml"
            if junit.exists():
                self.state["junit"] = artifact(self.stage, junit)
            self.save()
        self.check_source()
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {log}")
        if name in ("build", "numerical", "tidy"):
            warnings = project_warnings(log, self.source)
            if warnings:
                raise RuntimeError(f"project diagnostics in {log}: {warnings[:3]}")

    def snapshot_inputs(self):
        inputs = self.stage / "gate-source-inputs"
        inputs.mkdir()
        records = []
        source_files = [
            path for path in (self.source / "test/Vulkan").iterdir()
            if path.is_file() and path.suffix in (".param", ".mlir", ".bin")
        ]
        direct_kernel_dir = self.source / "test/Conversion/NCNNToVulkan"
        if direct_kernel_dir.is_dir():
            source_files.extend(direct_kernel_dir.glob("lower-*.mlir"))
        for path in sorted(set(source_files)):
            target = inputs / path.name
            if target.exists():
                raise RuntimeError(f"duplicate source evidence input name: {path.name}")
            shutil.copyfile(path, target)
            records.append({"source": str(path.relative_to(self.source)),
                            **artifact(self.stage, target)})
        for path in sorted((self.stage / "test/Vulkan").glob("*.bin")):
            target = inputs / path.name
            if target.exists():
                raise RuntimeError(f"duplicate evidence input name: {path.name}")
            shutil.copyfile(path, target)
            records.append({"generated": str(path.relative_to(self.stage)),
                            **artifact(self.stage, target)})
        self.state["source_inputs"] = records
        self.save()

    def run(self):
        try:
            self.step("configure", self.config)
            cache = self.stage / "CMakeCache.txt"
            self.state["cmake_cache"] = artifact(self.stage, cache)
            self.state["toolchain_identities"] = tool_identities(
                self.args, cache_values(cache))
            if self.args.profile == "sanitize":
                self.state["sanitizer_toolchain_evidence"] = audit_and_bundle(
                    self.stage, self.args.toolchain_identity,
                    self.args.sanitizer_install, self.args.lit,
                    self.args.openmp_dir, self.args.clang_dir)
            self.save()
            build = ["cmake", "--build", str(self.stage)]
            parallel = ["--parallel", str(self.args.jobs)]
            self.step("format_check", build + ["--target", "format_check"] + parallel)
            self.step("build", build + parallel)
            self.state["built_tool_identities"] = built_tool_identities(self.stage)
            self.save()
            self.step("numerical", build + ["--target", "numerical_tests",
                                            "numerical_dynamic_operator_tests",
                                            "numerical_dynamic_tests"] + parallel)
            self.snapshot_inputs()
            self.step("tidy", build + ["--target", "tidy"] + parallel)
            if self.args.profile == "sanitize":
                self.state["static_linkage_audit"] = audit_static_linkage(self.stage)
                self.save()
            inventory = self.stage / "test/gate-tests.json"
            with inventory.open("wb") as stream:
                subprocess.run(["ctest", "--test-dir", str(self.stage),
                                "--show-only=json-v1"], stdout=stream, check=True)
            self.state["inventory"] = artifact(self.stage, inventory)
            self.save()
            self.step("ctest", ["ctest", "--test-dir", str(self.stage),
                                "--output-on-failure", "--output-junit",
                                str(self.stage / "gate-ctest.xml")])
            logs = [self.stage / record["log"] for record in self.state["steps"]]
            logs.append(self.stage / self.state["lasttest"]["path"])
            self.state["sanitizer_diagnostics"] = diagnostics(logs)
            if self.state["sanitizer_diagnostics"]:
                raise RuntimeError("sanitizer diagnostics in raw gate logs")
            self.check_source()
            self.state["completed"] = True
            self.state["finished_at"] = timestamp()
            if self.args.legacy_baseline_stage:
                self.state["legacy_reference"] = snapshot_legacy_baseline(
                    self.args.legacy_baseline_stage, self.stage)
            else:
                self.state["legacy_reference"] = {"status": "not-supplied"}
            self.save()
            report = self.stage / "release-evidence.json"
            audit_command = [sys.executable, str(Path(__file__).with_name(
                "check_release_evidence.py")), "--stage", str(self.stage),
                "--output", str(report)]
            if self.args.legacy_baseline_stage:
                audit_command.append("--require-va0-stage")
            subprocess.run(audit_command, check=True)
            if self.args.archive:
                archive = self.stage.parent / f"{self.stage.name}.evidence.tar.gz"
                archive_stage(self.stage, archive)
                print(f"Evidence archive: {archive}", flush=True)
            print(f"GATE PASSED: {self.stage}", flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            self.state["completed"] = False
            self.state["failure"] = str(error)
            self.state["finished_at"] = timestamp()
            logs = [self.stage / record["log"] for record in self.state["steps"]
                    if (self.stage / record["log"]).is_file()]
            if "lasttest" in self.state:
                logs.append(self.stage / self.state["lasttest"]["path"])
            self.state["sanitizer_diagnostics"] = diagnostics(logs)
            self.save()
            if self.args.archive:
                archive = self.stage.parent / f"{self.stage.name}.evidence.tar.gz"
                if not archive.exists():
                    archive_stage(self.stage, archive)
            print(f"GATE FAILED: {error}\nPreserved stage: {self.stage}", file=sys.stderr)
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tidy-jobs", type=int, default=4)
    parser.add_argument("--sanitizer-install", type=Path)
    parser.add_argument("--lit", type=Path)
    parser.add_argument("--openmp-dir", type=Path)
    parser.add_argument("--toolchain-identity", type=Path)
    parser.add_argument("--clang-dir", type=Path, default=Path("/usr/lib/llvm-21/bin"))
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--budget-evidence", action="append", default=[],
                        metavar="FIXTURE=JSON",
                        help="prior same-source hard-budget failure evidence")
    parser.add_argument("--legacy-baseline-stage", type=Path,
                        help="frozen completed VA0 Release stage used as a separate legacy reference")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs <= 0 or args.tidy_jobs <= 0:
        parser.error("parallel job counts must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    Gate(args).run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"run_gate.py: {error}", file=sys.stderr)
        sys.exit(1)
