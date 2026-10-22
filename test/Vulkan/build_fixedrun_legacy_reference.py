#!/usr/bin/env python3
"""Build the pinned generic Vulkan reference locally in a fresh directory.

No fetch, install, worktree, or source-repository modification is performed.
The reference is a comparator, not a completed engineering-gate profile.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile

from run_gate import cache_values, git_output, sha256, source_fingerprint, timestamp


LEGACY_REVISION = "6094cb56abe24cdbcde84fbb72813da8aa24c147"
TOOLS = ("ncnn-mlir-driver", "ncnn-mlir-opt", "ncnn-vulkan-spirv-extract")


def file_record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"reference artifact is not a regular file: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def write_json(path, data, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise ValueError(f"refusing to replace an existing reference: {path}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     dir=path.parent, prefix=path.name + ".",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise ValueError(f"refusing to replace an existing reference: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def git_entries(root):
    entries = []
    for item in git_output(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not item:
            continue
        metadata, raw_name = item.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        name = os.fsdecode(raw_name)
        relative = PurePosixPath(name)
        if stage != "0" or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid or unmerged reference source: {name}")
        entries.append((mode, oid, name))
    return entries


def tracked_inventory(root, prefix=""):
    """Keep empty gitlinks as OIDs, like the gate source-fingerprint contract."""
    root = Path(root).resolve()
    records = []
    for mode, oid, name in git_entries(root):
        path = root / name
        logical = prefix + name
        if mode == "160000":
            records.append({"path": logical, "kind": "gitlink", "oid": oid})
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                continue
            top = Path(git_output(path, "rev-parse", "--show-toplevel")
                       .decode().strip()).resolve()
            if top != path.resolve():
                raise ValueError(f"populated gitlink is not its own repository: {path}")
            actual = git_output(path, "rev-parse", "HEAD").decode().strip()
            if actual != oid or git_output(path, "status", "--porcelain"):
                raise ValueError(f"reference gitlink is dirty or changed: {path}")
            records.extend(tracked_inventory(path, logical + "/"))
        elif path.is_symlink():
            records.append({"path": logical, "kind": "symlink",
                            "target": os.readlink(path)})
        else:
            records.append({"path": logical, "kind": "file",
                            "sha256": sha256(path), "git_blob": oid})
    return records


def archive_source(root, records, output):
    with tarfile.open(output, "w:gz", dereference=False) as archive:
        for record in records:
            if record["kind"] != "gitlink":
                archive.add(root / record["path"], arcname=record["path"],
                            recursive=False)


def find_tool(build, name):
    candidates = [path for path in (build / "tools" / name, build / "bin" / name)
                  if path.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"missing or ambiguous reference tool {name}: {candidates}")
    return candidates[0]


def build_objects(database):
    if not isinstance(database, list) or not database:
        raise ValueError("reference compile database is empty or malformed")
    objects = []
    for record in database:
        if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) for key in ("directory", "file")):
            raise ValueError("reference compile entry has no source/directory")
        argv = record.get("arguments")
        if argv is None:
            if not isinstance(record.get("command"), str):
                raise ValueError("reference compile entry has no command")
            argv = shlex.split(record["command"])
        if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
            raise ValueError("reference compile arguments are malformed")
        if any(value.startswith("@") for value in argv):
            raise ValueError("reference compile response files are not supported")
        if argv.count("-o") != 1 or argv.index("-o") + 1 >= len(argv):
            raise ValueError("reference compile command has no unique object output")
        if any(argument.startswith(("-fsanitize", "-fno-sanitize", "-shared-libasan",
                                    "-static-libasan")) for argument in argv):
            raise ValueError("legacy comparator must be a non-instrumented Release build")
        output = Path(argv[argv.index("-o") + 1])
        if not output.is_absolute():
            output = Path(record["directory"]) / output
        objects.append({"source": record["file"], "object": file_record(output)})
    if not objects:
        raise ValueError("reference compile database is empty")
    return objects


def collect_static_inputs(build, links, trace_log):
    """Bind explicit archives and linker-resolved -l/implicit runtime archives.

    Linker tracing is captured by the real build, not reconstructed from PATH.
    CMake's link script runs in its target binary directory, which is distinct
    from the cwd of the outer cmake --build command.
    """
    archives = {path.resolve(strict=True) for path in build.glob("**/*.a")}
    directories = set()
    command_lines = set()
    for record in links:
        command = Path(record["command"]["path"])
        cwd = Path(record.get("cwd", command.parents[2]))
        directories.add(cwd)
        text = command.read_text(encoding="utf-8")
        command_lines.update(line.strip() for line in text.splitlines())
        arguments = []
        for argument in shlex.split(text):
            if argument.startswith("-Wl,"):
                arguments.extend(argument[4:].split(","))
            else:
                arguments.append(argument)
        for argument in arguments:
            if argument.startswith("@"):
                raise ValueError("reference link response files are not supported")
            if argument.startswith("-") or not argument.endswith(".a"):
                continue
            path = Path(argument)
            if not path.is_absolute():
                path = cwd / path
            if not path.is_file():
                raise ValueError(f"reference linker archive is missing: {path}")
            archives.add(path.resolve(strict=True))
    trace_count = 0
    for line in trace_log.read_text(encoding="utf-8", errors="strict").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if line.startswith(("[", "make[", "gmake[")) or line in command_lines:
            continue
        matched = re.fullmatch(r"(.+\.a)(?:\([^)]*\))?", line)
        if matched is None:
            continue
        name = Path(matched.group(1))
        candidates = [name] if name.is_absolute() else [cwd / name for cwd in directories]
        resolved = {path.resolve(strict=True) for path in candidates if path.is_file()}
        if len(resolved) != 1:
            raise ValueError(f"missing or ambiguous linker-trace archive: {line}")
        archives.update(resolved)
        trace_count += 1
    if trace_count == 0:
        raise ValueError("reference build contains no actual linker archive trace")
    return [file_record(path) for path in sorted(archives)]


def validate_cache(cache_path, source):
    values = cache_values(cache_path)
    expected = {"BUILD_TESTING": "OFF", "NCNN_ENABLE_VULKAN_PRODUCT": "ON",
                "NCNN_ENABLE_VULKAN_V0_TESTS": "OFF", "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_HOME_DIRECTORY": str(source),
                "CMAKE_EXE_LINKER_FLAGS": "-Wl,--trace"}
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("reference cache does not describe the frozen Release source")
    if values.get("COMPILER_SANITIZERS", ""):
        raise ValueError("legacy comparator is not the required Release configuration")


class ReferenceBuilder:
    def __init__(self, args):
        self.args = args
        args.build_root.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="ncnn-compiler-va4-legacy-",
                                        dir=args.build_root)).resolve()
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.state = {"schema": "ncnn-vulkan-legacy-build/v1", "completed": False,
                      "source_repository": str(args.source_dir.resolve()),
                      "git_revision": LEGACY_REVISION, "started_at": timestamp(),
                      "steps": []}
        self.save()

    def save(self):
        write_json(self.root / "legacy-build-state.json", self.state, replace=True)

    def step(self, name, argv):
        log = self.root / f"{name}.log"
        record = {"name": name, "argv": [str(value) for value in argv],
                  "cwd": str(self.root), "started_at": timestamp(),
                  "exit_code": None}
        self.state["steps"].append(record)
        self.save()
        with log.open("wb") as stream:
            result = subprocess.run(record["argv"], cwd=self.root, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        record.update(exit_code=result.returncode, finished_at=timestamp(),
                      log=file_record(log))
        self.save()
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {log}")
        return record

    def clone_submodules(self, original, checkout):
        for mode, oid, name in git_entries(checkout):
            if mode != "160000":
                continue
            local = original / name
            target = checkout / name
            if not local.exists() or (local.is_dir() and not any(local.iterdir())):
                continue
            top = Path(git_output(local, "rev-parse", "--show-toplevel")
                       .decode().strip()).resolve()
            if top != local.resolve():
                raise ValueError(f"local gitlink source is not a repository: {local}")
            number = len(self.state["steps"])
            self.step(f"gitlink-{number}-clone",
                      ["git", "clone", "--local", "--no-hardlinks", local, target])
            self.step(f"gitlink-{number}-checkout",
                      ["git", "-C", target, "checkout", "--detach", oid])
            self.clone_submodules(local, target)

    def collect(self, configure, build):
        if git_output(self.source, "rev-parse", "HEAD").decode().strip() != LEGACY_REVISION:
            raise ValueError("reference source revision changed")
        status = git_output(self.source, "status", "--porcelain",
                            "--untracked-files=normal")
        if status:
            raise ValueError("reference source must remain clean")
        status_path = self.root / "source-status.txt"
        status_path.write_bytes(status)
        fingerprint = source_fingerprint(self.source)
        records = tracked_inventory(self.source)
        inventory_path = self.root / "source-inventory.json"
        write_json(inventory_path, {"schema": "ncnn-legacy-source-inventory/v1",
                                   "git_revision": LEGACY_REVISION,
                                   "source_fingerprint": fingerprint, "files": records})
        archive_path = self.root / "source.tar.gz"
        archive_source(self.source, records, archive_path)
        cache_path = self.build / "CMakeCache.txt"
        validate_cache(cache_path, self.source)
        database_path = self.build / "compile_commands.json"
        objects = build_objects(json.loads(database_path.read_text(encoding="utf-8")))
        links = [{"target": path.parent.name.removesuffix(".dir"),
                  "cwd": str(path.parents[2]), "command": file_record(path)}
                 for path in sorted(self.build.glob("**/CMakeFiles/*.dir/link.txt"))]
        archives = collect_static_inputs(self.build, links, Path(build["log"]["path"]))
        if not links or not archives:
            raise ValueError("reference build has no linker inputs or archives")
        if source_fingerprint(self.source) != fingerprint or git_output(
                self.source, "status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("reference source changed while recording provenance")
        return {
            "schema": "ncnn-vulkan-legacy-reference/v1",
            "role": "legacy-product-reference", "git_revision": LEGACY_REVISION,
            "source_root": str(self.source), "source_fingerprint": fingerprint,
            "compiler": file_record(find_tool(self.build, "ncnn-compile")),
            "tools": [{"role": name, **file_record(find_tool(self.build, name))}
                      for name in TOOLS],
            "build": {
                "configure_argv": configure["argv"], "build_argv": build["argv"],
                "cwd": str(self.root),
                "exit_codes": {"configure": configure["exit_code"],
                               "build": build["exit_code"]},
                "logs": {"configure": configure["log"], "build": build["log"]},
                "cmake_cache": file_record(cache_path),
                "compile_commands": file_record(database_path), "objects": objects,
                "link_commands": links, "static_inputs": archives,
                "static_input_coverage": "explicit-link-argv+linker-trace/v1",
                "link_trace": build["log"]},
            "source_archive": file_record(archive_path),
            "source_inventory": file_record(inventory_path),
            "clean_source_status": file_record(status_path),
        }

    def run(self):
        print(f"LEGACY REFERENCE ROOT: {self.root}", flush=True)
        try:
            revision = git_output(self.args.source_dir, "rev-parse", "--verify",
                                  "--end-of-options", LEGACY_REVISION + "^{commit}")
            if revision.decode().strip() != LEGACY_REVISION:
                raise ValueError("pinned legacy revision is not available locally")
            self.step("clone", ["git", "clone", "--local", "--no-hardlinks",
                                self.args.source_dir.resolve(), self.source])
            self.step("checkout", ["git", "-C", self.source, "checkout", "--detach",
                                   LEGACY_REVISION])
            self.clone_submodules(self.args.source_dir.resolve(), self.source)
            configure_argv = ["cmake", "-S", str(self.source), "-B", str(self.build),
                              "-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=Release",
                              "-DBUILD_TESTING=OFF", "-DCOMPILER_INSTALL_RPATH=ON",
                              "-DCOMPILER_ENABLE_FORMAT_TARGETS=ON",
                              "-DNCNN_ENABLE_VULKAN_PRODUCT=ON",
                              "-DNCNN_ENABLE_VULKAN_V0_TESTS=OFF",
                              "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--trace"]
            if self.args.llvm_dir is not None:
                configure_argv.extend([f"-DLLVM_DIR={self.args.llvm_dir.resolve()}",
                                       f"-DMLIR_DIR={self.args.mlir_dir.resolve()}"])
            configure = self.step("configure", configure_argv)
            build = self.step("build", ["cmake", "--build", self.build, "--parallel",
                                        str(self.args.jobs)])
            manifest = self.collect(configure, build)
            write_json(self.root / "legacy-reference.json", manifest)
            self.state.update(completed=True, finished_at=timestamp(),
                              manifest=file_record(self.root / "legacy-reference.json"))
            self.save()
            print(f"LEGACY REFERENCE READY: {self.root}", flush=True)
            return self.root
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
                tarfile.TarError) as error:
            self.state.update(failure=str(error), finished_at=timestamp())
            self.save()
            raise RuntimeError(f"reference build failed; preserved {self.root}: {error}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--llvm-dir", type=Path)
    parser.add_argument("--mlir-dir", type=Path)
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if (args.llvm_dir is None) != (args.mlir_dir is None):
        parser.error("--llvm-dir and --mlir-dir must be supplied together")
    if args.build_root.resolve().is_relative_to(args.source_dir.resolve()):
        parser.error("reference build root must be outside the source tree")
    return args


def main(argv=None):
    ReferenceBuilder(parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"build_fixedrun_legacy_reference.py: {error}", file=sys.stderr)
        sys.exit(1)
