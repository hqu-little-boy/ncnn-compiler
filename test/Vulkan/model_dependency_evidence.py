#!/usr/bin/env python3
"""Test-only evidence capture for the generated ncnn Vulkan model DSO.

This module is intentionally separate from the compiler implementation.  The
host-compiler launcher only instruments the final wrapper/shared-library link;
compiler probes are exec'd unchanged.  Evidence collection is diagnostic and
never changes the production link's strip, visibility, undefined-symbol, or
RPATH policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
LINK_KIND = "ncnn.vulkan_model_link"
EVIDENCE_KIND = "ncnn.vulkan_model_dependency_evidence"
ELF_EVIDENCE_KIND = "ncnn.elf_dependency_closure"


class EvidenceError(RuntimeError):
    """Raised when dependency evidence is missing, ambiguous, or tampered."""


def sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _tool_identity(executable: str) -> dict[str, object]:
    found = shutil.which(executable) or executable
    resolved = Path(found).resolve()
    result: dict[str, object] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        result.update({"sha256": sha256(resolved), "size_bytes": resolved.stat().st_size})
    return result


def _sha_file_entry(path: Path, evidence_root: Path, relative: Path,
                    *, origin: str | None = None) -> dict[str, object]:
    destination = evidence_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    relative_path = relative.as_posix()
    return {"path": relative_path, "source": relative_path,
            "origin": origin or path.name,
            "sha256": sha256(destination), "size_bytes": destination.stat().st_size}


def _snapshot_staging(wrapper: Path, evidence_root: Path) -> dict[str, object]:
    """Copy only the generated package inputs while staging still exists."""
    stage = wrapper.parent
    requested: set[Path] = {wrapper}
    # The wrapper's filename is fixed, while model_name is not. Select only the
    # unambiguous generated Vulkan ABI/plan/manifest names in this staging dir.
    generated = {
        "header": sorted(path for path in stage.glob("*_vulkan.h") if path.is_file()),
        "plan": sorted(path for path in stage.glob("*_vulkan.plan.json") if path.is_file()),
        "manifest": sorted(path for path in stage.glob("*_vulkan.json") if path.is_file()),
    }
    for paths in generated.values():
        if len(paths) == 1:
            requested.add(paths[0])
    requested.add(stage / "model.vulkan-weights.json")
    # SPIR-V binaries are one per execution-plan kernel. Payload-like files are
    # included by name only; arbitrary staging files are deliberately excluded.
    requested.update(path for path in stage.glob("kernel-*.spv") if path.is_file())
    requested.update(path for path in stage.glob("*payload*") if path.is_file())
    requested.update(path for path in stage.glob("*.payload") if path.is_file())
    entries: list[dict[str, object]] = []
    missing: list[str] = []
    for source in sorted(requested, key=lambda item: item.name):
        if not source.is_file():
            missing.append(source.name)
            continue
        entries.append(_sha_file_entry(
            source, evidence_root, Path("staged") / source.name,
            origin=(Path("staging") / source.name).as_posix()))
    present = {Path(str(entry["source"])).name for entry in entries}
    missing_required = []
    if wrapper.name not in present:
        missing_required.append("vulkan_model_wrapper.cpp")
    missing_required.extend(key for key, paths in generated.items() if len(paths) != 1)
    shader_entries = [entry for entry in entries if str(entry["path"]).endswith(".spv")]
    return {
        "stage_directory": "staging",
        "files": entries,
        "missing_optional": sorted(set(missing)),
        "missing_required": missing_required,
        "spirv_count": len(shader_entries),
        "payload_count": sum(
            "payload" in str(entry["path"]).lower() or
            Path(str(entry["path"])).name == "model.vulkan-weights.json"
            for entry in entries),
    }


def _is_model_link(arguments: Sequence[str]) -> bool:
    return ("-shared" in arguments and any(
        Path(argument).name == "vulkan_model_wrapper.cpp" for argument in arguments
    ))


def make_host_cxx_launcher(real_cxx: os.PathLike[str] | str,
                           capture_dir: os.PathLike[str] | str) -> Path:
    """Create a private proxy usable as --vulkan-host-cxx.

    Non-model invocations, including -dumpmachine and other compiler probes,
    are handed directly to the real executable with the original argv.  Only
    the generated wrapper's -shared link receives a map and save-temps option.
    """
    executable = shutil.which(str(real_cxx))
    if executable is None:
        raise ValueError(f"host C++ compiler is unavailable: {real_cxx}")
    real_path = Path(executable).resolve()
    output = Path(capture_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    launcher = output / "host-cxx-capture"
    if launcher.exists():
        raise ValueError(f"refusing to overwrite host compiler launcher: {launcher}")
    module_path = Path(__file__).resolve()
    launcher.write_text(
        "#!" + sys.executable + "\n"
        "import sys\n"
        f"sys.path.insert(0, {str(module_path.parent)!r})\n"
        "from model_dependency_evidence import _proxy_main\n"
        f"sys.exit(_proxy_main({str(real_path)!r}, {str(output)!r}, sys.argv[1:]))\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def _link_record_paths(capture_dir: Path) -> tuple[Path, Path]:
    capture_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        record = capture_dir / f"model-link-{index:04d}.json"
        if not record.exists():
            return record, capture_dir / f"model-link-{index:04d}.map"
        index += 1


def _save_temps_candidates(cwd: Path, logical_dir: Path, wrapper: Path,
                           output: Path) -> list[Path]:
    candidates: set[Path] = set()
    for directory in {cwd, logical_dir, wrapper.parent, output.parent}:
        if not directory.is_dir():
            continue
        for pattern in ("*vulkan_model_wrapper*.o", "*vulkan_model_wrapper*.obj"):
            candidates.update(path for path in directory.glob(pattern) if path.is_file())
    return sorted(candidates)


def _working_directory(arguments: Sequence[str], cwd: Path) -> Path:
    for index, argument in enumerate(arguments):
        if argument.startswith("-working-directory="):
            value = argument.partition("=")[2]
        elif argument == "-working-directory" and index + 1 < len(arguments):
            value = arguments[index + 1]
        else:
            continue
        directory = Path(value)
        return (directory if directory.is_absolute() else cwd / directory).resolve()
    return cwd


def _proxy_main(real_cxx: str, capture_dir: str, arguments: Sequence[str]) -> int:
    real_path = Path(real_cxx).resolve()
    if not _is_model_link(arguments):
        # execv preserves probe output, exit status, signal behavior and argv.
        os.execv(str(real_path), [str(real_path), *arguments])
    capture = Path(capture_dir).resolve()
    wrapper_arg = next(arg for arg in arguments
                       if Path(arg).name == "vulkan_model_wrapper.cpp")
    cwd = Path.cwd().resolve()
    logical_dir = _working_directory(arguments, cwd)
    wrapper = Path(wrapper_arg)
    if not wrapper.is_absolute():
        wrapper = logical_dir / wrapper
    wrapper = wrapper.resolve()
    record_path, map_path = _link_record_paths(capture)
    map_arg = f"-Wl,-Map={map_path},--cref"
    effective = [*arguments, "-save-temps=obj", map_arg]
    staged = _snapshot_staging(wrapper, capture)
    output_argument = None
    try:
        output_index = effective.index("-o")
        if output_index + 1 < len(effective):
            output_argument = effective[output_index + 1]
    except ValueError:
        pass
    output_file = None
    if output_argument is not None:
        output_file = Path(output_argument)
        if not output_file.is_absolute():
            output_file = cwd / output_file
        output_file = output_file.resolve()
    identity = _tool_identity(str(real_path))
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": LINK_KIND,
        "original_argv": [str(real_path), *map(str, arguments)],
        "effective_argv": [str(real_path), *map(str, effective)],
        "cwd": str(cwd),
        "logical_compile_dir": str(logical_dir),
        "executable": identity,
        "wrapper_source": "staged/vulkan_model_wrapper.cpp",
        "output": output_file.name if output_file is not None else None,
        "linked_output": None,
        "archive_snapshots": [],
        "map_path": map_path.relative_to(capture).as_posix(),
        "staged_snapshot": staged,
        "save_temps": "-save-temps=obj",
        "diagnostics": {"stdout": "", "stderr": ""},
        "returncode": None,
        "wrapper_objects": [],
    }
    _write_json(record_path, record)
    try:
        result = subprocess.run([str(real_path), *effective], cwd=cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.buffer.flush()
        record["diagnostics"] = {
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
        }
        record["returncode"] = result.returncode
        candidates = _save_temps_candidates(
            cwd, logical_dir, wrapper, output_file if output_file else wrapper)
        saved_objects = []
        for index, source in enumerate(candidates):
            entry = _sha_file_entry(
                source, capture, Path("objects") / f"wrapper-{index:02d}.o",
                origin=(Path("compiler-temporary") / source.name).as_posix())
            saved_objects.append(entry)
        record["wrapper_objects"] = saved_objects
        if result.returncode == 0 and output_file is not None and output_file.is_file():
            record["linked_output"] = _sha_file_entry(
                output_file, capture, Path("linked-output") / output_file.name,
                origin=(Path("compiler-output") / output_file.name).as_posix())
        if result.returncode == 0 and map_path.is_file():
            record["archive_snapshots"], record["archive_capture_errors"] = \
                _snapshot_archives(effective, cwd, map_path, capture)
    except OSError as error:
        record["launcher_error"] = str(error)
        record["returncode"] = 127
        print(f"host-cxx-capture: {error}", file=sys.stderr)
    _write_json(record_path, record)
    return (int(record["returncode"]) if int(record["returncode"]) >= 0
            else 128 - int(record["returncode"]))


def _run_tool(command: Sequence[str], *, check: bool = True,
              env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(list(map(str, command)), text=True, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
    except OSError as error:
        raise EvidenceError(f"cannot run evidence tool {command[0]}: {error}") from error
    if check and result.returncode != 0:
        raise EvidenceError(
            f"evidence command failed ({result.returncode}): {list(command)!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _read_tool_paths(tool_paths: Mapping[str, os.PathLike[str] | str] | None) -> dict[str, str]:
    supplied = dict(tool_paths or {})
    result: dict[str, str] = {}
    for key, fallback in (("readelf", "readelf"), ("ldd", "ldd"), ("ar", "ar")):
        value = str(supplied.get(key, fallback))
        found = shutil.which(value)
        if found is None:
            raise EvidenceError(f"required evidence tool is unavailable: {value}")
        result[key] = str(Path(found).resolve())
    ldd_r = str(supplied.get("ldd_r", result["ldd"]))
    result["ldd_r"] = str(Path(shutil.which(ldd_r) or ldd_r).resolve())
    result["ar"] = str(Path(shutil.which(result["ar"]) or result["ar"]).resolve())
    return result


def _archive_matches(map_text: str) -> list[tuple[str, str]]:
    """Return linker-map archive path/member pairs (not version-script text)."""
    matches = []
    for line in map_text.splitlines():
        for match in re.finditer(r"([^\s()]+\.a)\(([^()]+)\)", line):
            matches.append((match.group(1), match.group(2)))
    return matches


def _archive_refs(map_text: str) -> set[str]:
    return {f"{Path(archive).name}({member})"
            for archive, member in _archive_matches(map_text)}


def _source_info(name: str, *, static_llvm: bool = False) -> tuple[str, str]:
    lowered = name.lower()
    if any(key in lowered for key in ("asan", "ubsan", "tsan", "sanitizer")):
        return "SAN", "sanitizer-object"
    if static_llvm or re.search(r"(?:^|[/_.-])(llvm|mlir|clang)(?:[/_.-]|$)", lowered):
        return "compiler", "llvm-mlir-object"
    if any(key in lowered for key in ("vulkanexecutionplan", "vulkanplansupport",
                                      "ncnnvulkanplansupport", "executionplan")):
        return "model", "plan-parser"
    if any(key in lowered for key in ("vulkanmodelpayload", "ncnnvulkanpayload",
                                      "modelpayload", "payloadhash", "payload")):
        return "model", "payload-hash"
    if "vulkandevicehelper" in lowered:
        return "model", "device-helper"
    if any(key in lowered for key in ("ncnnvulkanruntime", "vulkanruntime")):
        return "model", "project-executor"
    if "vulkan" in lowered or "ncnn" in lowered:
        return "model", "project-component"
    return "platform", "platform-object"


def _classify_source(name: str, *, static_llvm: bool = False) -> str:
    return _source_info(name, static_llvm=static_llvm)[0]


def _archive_paths(argv: Sequence[str], cwd: Path, map_text: str) -> tuple[dict[Path, str], list[str]]:
    archives: dict[Path, str] = {}
    errors: list[str] = []
    search_dirs = [cwd]
    index = 1
    while index < len(argv):
        token = str(argv[index])
        if token == "-L" and index + 1 < len(argv):
            directory = Path(str(argv[index + 1]))
            search_dirs.append(directory if directory.is_absolute() else cwd / directory)
            index += 2
            continue
        if token.startswith("-L") and len(token) > 2:
            directory = Path(token[2:])
            search_dirs.append(directory if directory.is_absolute() else cwd / directory)
        for match in re.finditer(r"(?:^|[,])-L([^,]+)", token):
            directory = Path(match.group(1))
            search_dirs.append(directory if directory.is_absolute() else cwd / directory)
        if token.startswith("@"):
            errors.append(f"response-file:{token}")
            index += 1
            continue
        archive_tokens = [token] if token.endswith(".a") else re.findall(r"[^,;\s]+\.a", token)
        for item in archive_tokens:
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = cwd / candidate
            candidate = candidate.resolve()
            if candidate.is_file():
                archives[candidate] = token
            else:
                errors.append(f"argv:{item}")
        index += 1
    for map_archive, _member in _archive_matches(map_text):
        map_path = Path(map_archive)
        candidate = map_path if map_path.is_absolute() else cwd / map_path
        if candidate.is_file():
            archives[candidate.resolve()] = f"link-map:{map_archive}"
            continue
        if any(path.name == map_path.name for path in archives):
            continue
        matches = sorted({(directory / map_path.name).resolve() for directory in search_dirs
                          if (directory / map_path.name).is_file()})
        if len(matches) == 1:
            archives[matches[0]] = f"link-map:{map_archive}"
        else:
            errors.append(f"map:{map_archive}")
    return archives, errors


def _hash_archive_member(ar: str, archive: Path, member: str) -> tuple[str, int]:
    process = subprocess.Popen([ar, "p", str(archive), member],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    stderr = process.stderr.read() if process.stderr is not None else b""
    status = process.wait()
    if status != 0:
        raise EvidenceError(f"ar p failed for {archive}({member}): "
                            f"{stderr.decode(errors='replace')}")
    return digest.hexdigest(), size


def _snapshot_archives(argv: Sequence[str], cwd: Path, map_path: Path,
                       capture: Path) -> tuple[list[dict[str, object]], list[str]]:
    try:
        map_text = map_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return [], [f"cannot read actual linker map: {error}"]
    archives, errors = _archive_paths(argv, cwd, map_text)
    ar = shutil.which("ar")
    if ar is None:
        return [], errors + ["ar is unavailable while linked inputs still exist"]
    references = _archive_refs(map_text)
    snapshots = []
    raw_dir = capture / "archives"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, (archive, original) in enumerate(sorted(archives.items(), key=lambda item: str(item[0]))):
        if not archive.is_file():
            errors.append(f"archive disappeared before capture: {archive}")
            continue
        result = subprocess.run([ar, "t", str(archive)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        if result.returncode:
            errors.append(f"ar t failed for {archive}: {result.stderr.decode(errors='replace')}")
            continue
        raw_path = raw_dir / f"archive-{index:04d}.ar-t.txt"
        raw_path.write_bytes(result.stdout)
        members = [line.strip() for line in result.stdout.decode(errors="replace").splitlines()
                   if line.strip()]
        selected = sorted(ref for ref in references if ref.startswith(archive.name + "("))
        selected_member_hashes = []
        for reference in selected:
            member = reference.split("(", 1)[1][:-1]
            if members.count(member) != 1:
                errors.append(f"selected archive member is missing or ambiguous: {archive}({member})")
                continue
            try:
                digest, size = _hash_archive_member(ar, archive, member)
            except (OSError, EvidenceError) as error:
                errors.append(str(error))
                continue
            selected_member_hashes.append({"member": member, "sha256": digest,
                                           "size_bytes": size,
                                           "origin": f"{archive}({member})"})
        snapshots.append({
            "input": original, "path": str(archive), "sha256": sha256(archive),
            "size_bytes": archive.stat().st_size, "members": members,
            "ar_t": {"path": raw_path.relative_to(capture).as_posix(),
                     "sha256": sha256(raw_path), "size_bytes": raw_path.stat().st_size,
                     "command": [str(Path(ar).resolve()), "t", str(archive)]},
            "selected_members_from_map": selected,
            "selected_member_hashes": selected_member_hashes,
        })
    if not snapshots:
        errors.append("no archive inputs were snapshotted before staging teardown")
    return snapshots, errors


def _archive_inventory(snapshots: object, map_text: str,
                       evidence_root: Path) -> list[dict[str, object]]:
    references = _archive_refs(map_text)
    if not references:
        raise EvidenceError("actual linker map contains no selected archive members")
    if not isinstance(snapshots, list) or not snapshots:
        raise EvidenceError("archive source/member snapshots are missing from actual link record")
    names = [Path(str(item.get("path", ""))).name for item in snapshots
             if isinstance(item, dict)]
    if len(names) != len(set(names)):
        raise EvidenceError("ambiguous duplicate archive basenames in link-map provenance")
    inventory = []
    assigned: set[str] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise EvidenceError("malformed archive snapshot in host link record")
        archive = Path(str(snapshot.get("path", ""))).resolve()
        selected = sorted(ref for ref in references if ref.startswith(archive.name + "("))
        captured_selected = snapshot.get("selected_members_from_map")
        if not isinstance(captured_selected, list) or sorted(captured_selected) != selected:
            raise EvidenceError(f"archive selected-member map differs from link-time capture: {archive}")
        assigned.update(selected)
        raw = snapshot.get("ar_t")
        if not isinstance(raw, dict) or not _verified_entry(evidence_root, raw):
            raise EvidenceError(f"raw ar t member listing is missing or tampered: {archive}")
        raw_path = _safe_evidence_path(evidence_root, raw.get("path"))
        members = [line.strip() for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
                   if line.strip()]
        if members != snapshot.get("members"):
            raise EvidenceError(f"captured members differ from raw ar t output: {archive}")
        member_hashes = snapshot.get("selected_member_hashes")
        if not isinstance(member_hashes, list):
            raise EvidenceError(f"selected archive member hashes are missing: {archive}")
        if {item.get("member") for item in member_hashes if isinstance(item, dict)} != {
                ref.split("(", 1)[1][:-1] for ref in selected}:
            raise EvidenceError(f"selected member hash set differs from the link map: {archive}")
        if archive.is_file():
            if sha256(archive) != snapshot.get("sha256") or archive.stat().st_size != snapshot.get("size_bytes"):
                raise EvidenceError(f"archive input changed after link-time capture: {archive}")
            source_status = "present_and_hash_verified"
        else:
            source_status = "missing_at_collection_link_time_ar_t_verified"
        selected_origins = []
        member_hash_by_name = {item["member"]: item for item in member_hashes}
        for ref in selected:
            member = ref.split("(", 1)[1][:-1]
            if member not in members:
                raise EvidenceError(f"link map selected absent archive member: {ref}")
            digest = member_hash_by_name[member]
            selected_origins.append({"archive": str(archive), "member": member,
                                     "origin": f"{archive}({member})",
                                     "sha256": digest["sha256"],
                                     "size_bytes": digest["size_bytes"]})
        static_llvm = any(key in archive.name.lower() for key in ("llvm", "mlir", "clang")) or any(
            any(key in member.lower() for key in ("llvm", "mlir", "clang")) for member in members)
        for origin in selected_origins:
            origin["classification"], origin["role"] = _source_info(
                origin["member"], static_llvm=static_llvm)
        archive_classification, archive_role = _source_info(
            archive.name, static_llvm=static_llvm)
        inventory.append({
            "input": snapshot.get("input"), "path": str(archive),
            "sha256": snapshot.get("sha256"), "size_bytes": snapshot.get("size_bytes"),
            "source_status": source_status,
            "classification": archive_classification, "role": archive_role,
            "members": [{"name": member,
                         "classification": _source_info(member, static_llvm=static_llvm)[0],
                         "role": _source_info(member, static_llvm=static_llvm)[1]}
                        for member in members],
            "ar_t": raw, "selected_members_from_map": selected,
            "selected_member_hashes": member_hashes,
            "selected_member_origins": selected_origins,
        })
    if assigned != references:
        raise EvidenceError(f"link-map archive member inventory mismatch: missing={sorted(references-assigned)}")
    return inventory

def _resolve_ldd_target(value: str, working_directory: os.PathLike[str] | str,
                        *, arrow_target: bool = False) -> str | None:
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    if arrow_target or "/" in value or value.startswith("."):
        return str((Path(working_directory) / path).resolve())
    return None


def _parse_ldd(text: str, working_directory: os.PathLike[str] | str | None = None
               ) -> tuple[list[str], list[str]]:
    cwd = Path(working_directory or Path.cwd()).resolve()
    paths: set[str] = set()
    unresolved: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "not found" in stripped or "undefined symbol:" in stripped:
            unresolved.append(stripped)
            continue
        if "=>" in stripped:
            _, target = (part.strip() for part in stripped.split("=>", 1))
            target_path = target.split()[0] if target else ""
            resolved = _resolve_ldd_target(target_path, cwd, arrow_target=True)
            if resolved:
                paths.add(resolved)
            continue
        match = re.match(r"([^\s]+)\s+\(0x[0-9a-fA-F]+\)", stripped)
        if match:
            resolved = _resolve_ldd_target(match.group(1), cwd)
            if resolved:
                paths.add(resolved)
    return sorted(paths), unresolved


def _parse_ldd_named_paths(text: str,
                           working_directory: os.PathLike[str] | str | None = None) -> dict[str, str]:
    """Map ldd dependency names, including symlink aliases, to resolved paths."""
    cwd = Path(working_directory or Path.cwd()).resolve()
    named_paths = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "not found" in stripped or "undefined symbol:" in stripped:
            continue
        if "=>" in stripped:
            name, target = (part.strip() for part in stripped.split("=>", 1))
            target_path = target.split()[0] if target else ""
            resolved = _resolve_ldd_target(target_path, cwd, arrow_target=True)
            if name and resolved:
                named_paths[name] = resolved
            continue
        match = re.match(r"([^\s]+)\s+\(0x[0-9a-fA-F]+\)", stripped)
        if match:
            resolved = _resolve_ldd_target(match.group(1), cwd)
            if resolved:
                named_paths[match.group(1)] = resolved
                named_paths.setdefault(Path(resolved).name, resolved)
    return named_paths


def _expand_loader_origin(value: str, origin: Path) -> str:
    expanded = value.replace("${ORIGIN}", str(origin)).replace("$ORIGIN", str(origin))
    if re.search(r"\$\{?(?:LIB|PLATFORM)\}?", expanded):
        raise EvidenceError("$LIB and $PLATFORM loader tokens are unsupported in captured search paths")
    return expanded


def _capture_runtime_search_paths(runtime_env: Mapping[str, str] | None,
                                  target: str) -> list[str]:
    environment = os.environ if runtime_env is None else runtime_env
    if "LD_LIBRARY_PATH" not in environment:
        return []
    origin = Path(target).resolve().parent
    value = environment["LD_LIBRARY_PATH"]
    return sorted({str(Path(_expand_loader_origin(item or str(Path.cwd()), origin)).resolve())
                   for item in value.split(os.pathsep)})


def _dependency_search_paths(model_elf: dict, target: str,
                             environment_paths: Iterable[str], *,
                             working_directory: os.PathLike[str] | str | None = None) -> list[str]:
    cwd = Path(working_directory or Path.cwd()).resolve()
    directories = {str(Path(item).resolve()) for item in environment_paths}
    origin = Path(target).resolve().parent
    for value in model_elf.get("rpath_runpath", []):
        if not isinstance(value, str):
            continue
        for item in value.split(":"):
            path = Path(_expand_loader_origin(item, origin))
            directories.add(str((path if path.is_absolute() else cwd / path).resolve()))
    return sorted(directories)


def _dependency_info(path: str, *, model_paths: Iterable[str] = (),
                     sanitizer_dirs: Iterable[os.PathLike[str] | str] = (),
                     compiler_dirs: Iterable[os.PathLike[str] | str] = ()) -> tuple[str, str]:
    resolved = str(Path(path).resolve())
    name = Path(resolved).name.lower()
    if any(resolved == str(Path(candidate).resolve()) for candidate in model_paths):
        return "model", "generated-model"
    if any(key in name for key in ("asan", "ubsan", "tsan", "sanitizer")):
        return "SAN", "sanitizer-runtime"
    if any(Path(resolved).is_relative_to(Path(directory).resolve())
           for directory in sanitizer_dirs):
        return "SAN", "configured-sanitizer-directory"
    if any(key in name for key in ("libllvm", "libmlir", "libclang")):
        return "compiler", "compiler-runtime"
    if any(Path(resolved).is_relative_to(Path(directory).resolve())
           for directory in compiler_dirs):
        return "compiler", "configured-compiler-directory"
    if re.fullmatch(r"libvulkan\.so(?:\.[0-9.]+)?", name):
        return "platform", "vulkan-loader"
    if resolved.startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/",
                            "/usr/local/lib/", "/usr/local/lib64/")):
        return "platform", "system-runtime"
    return "model", "private-transitive"


def parse_readelf_dynamic(text: str) -> dict[str, object]:
    """Parse GNU readelf and LLVM llvm-readelf human-readable dynamic tables."""
    needed = set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", text))
    needed.update(re.findall(r"\bNEEDED\s+Shared library:\s*\[([^]]+)\]", text))
    sonames = re.findall(r"\(SONAME\).*?\[([^]]+)\]", text)
    sonames.extend(re.findall(r"\bSONAME\s+(?:Library soname|Shared object name):\s*\[([^]]+)\]", text))
    rpaths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]*)\]", text)
    rpaths.extend(re.findall(r"\b(?:RPATH|RUNPATH)\s+(?:Library rpath|Library runpath):\s*\[([^]]*)\]", text))
    return {"needed": sorted(needed), "soname": sonames[0] if sonames else None,
            "rpath_runpath": rpaths}


def parse_readelf_symbols(text: str) -> tuple[list[str], list[str]]:
    """Return undefined and exported symbols from GNU or LLVM readelf output."""
    undefined: set[str] = set()
    exports: set[str] = set()
    # GNU readelf prints one whitespace-delimited row per symbol.
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0].rstrip(":").isdigit():
            name = fields[7]
            if fields[6] == "UND" and name != "0":
                undefined.add(name)
            elif fields[6] != "UND" and fields[4] in {"GLOBAL", "WEAK"}:
                exports.add(name)
    # llvm-readelf prints a structured `Symbol { ... }` block for each entry.
    for block in re.findall(r"\bSymbol\s*\{(.*?)\n\s*\}", text, re.DOTALL):
        name_match = re.search(r"^\s*Name:\s*(.*?)\s+\(\d+\)\s*$", block, re.MULTILINE)
        binding_match = re.search(r"^\s*Binding:\s*(\w+)", block, re.MULTILINE)
        section_match = re.search(r"^\s*Section:\s*(.*?)\s+\(0x[0-9A-Fa-f]+\)",
                                  block, re.MULTILINE)
        if name_match is None or binding_match is None or section_match is None:
            continue
        name = name_match.group(1).strip()
        binding = binding_match.group(1)
        section = section_match.group(1).strip()
        if section == "Undefined" and name not in {"", "(0)"}:
            undefined.add(name)
        elif binding in {"Global", "Weak"} and section != "Undefined" and name not in {"", "(0)"}:
            exports.add(name)
    return sorted(undefined), sorted(exports)


def _elf_summary(readelf: str, library: Path, evidence_root: Path,
                 prefix: str) -> dict[str, object]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    dynamic = _run_tool([readelf, "-d", str(library)])
    symbols = _run_tool([readelf, "-Ws", str(library)])
    undefined, exports = parse_readelf_symbols(symbols.stdout)
    dynamic_summary = parse_readelf_dynamic(dynamic.stdout)
    dynamic_path = evidence_root / f"{prefix}.readelf-dynamic.txt"
    symbols_path = evidence_root / f"{prefix}.readelf-symbols.txt"
    dynamic_path.write_text(dynamic.stdout + dynamic.stderr, encoding="utf-8")
    symbols_path.write_text(symbols.stdout + symbols.stderr, encoding="utf-8")
    return {
        "path": library.name,
        "sha256": sha256(library),
        "needed": dynamic_summary["needed"],
        "soname": dynamic_summary["soname"],
        "rpath_runpath": dynamic_summary["rpath_runpath"],
        "exports": exports,
        "undefined_symbols": undefined,
        "readelf_dynamic": {"path": dynamic_path.name, "sha256": sha256(dynamic_path)},
        "readelf_symbols": {"path": symbols_path.name, "sha256": sha256(symbols_path)},
        "readelf_commands": {
            "dynamic": [readelf, "-d", str(library)],
            "symbols": [readelf, "-Ws", str(library)],
        },
    }


def _link_records(evidence_root: Path) -> list[tuple[Path, dict[str, object]]]:
    results = []
    for path in sorted(evidence_root.glob("model-link-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceError(f"cannot read host-link record {path}: {error}") from error
        if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != LINK_KIND:
            continue
        results.append((path, record))
    return results


def _read_elf_tools(tool_paths: Mapping[str, os.PathLike[str] | str] | None) -> dict[str, str]:
    supplied = dict(tool_paths or {})
    tools = {}
    for key, fallback in (("readelf", "readelf"), ("ldd_r", "ldd")):
        value = str(supplied.get(key, supplied.get("ldd") if key == "ldd_r" else fallback))
        found = shutil.which(value)
        if found is None:
            raise EvidenceError(f"required ELF evidence tool is unavailable: {value}")
        tools[key] = str(Path(found).resolve())
    return tools


def collect_elf_dependency_closure(
        library: os.PathLike[str] | str,
        evidence_root: os.PathLike[str] | str,
        tool_paths: Mapping[str, os.PathLike[str] | str] | None = None,
        sanitizer_dirs: Iterable[os.PathLike[str] | str] = (),
        *, compiler_dirs: Iterable[os.PathLike[str] | str] = (),
        expected_generated_dsos: Iterable[os.PathLike[str] | str] | None = None,
        llvm_mode: str = "not_measured",
        runtime_env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Collect an ELF runtime closure without requiring a Vulkan link record.

    Use this for the CPU model counterpart. It intentionally does not fabricate
    a Vulkan host-link record or claim static-archive LLVM visibility.
    """
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = Path(library).resolve()
    if not target.is_file():
        raise EvidenceError(f"ELF library is missing: {target}")
    if llvm_mode not in {"shared", "static", "not_measured"}:
        raise ValueError("llvm_mode must be shared, static, or not_measured")
    tools = _read_elf_tools(tool_paths)
    expected = ([Path(item).resolve() for item in expected_generated_dsos]
                if expected_generated_dsos is not None else [target])
    if target not in expected:
        raise EvidenceError("ELF target is not in the explicit expected generated DSO set")
    generated = []
    for index, path in enumerate(sorted(set(expected), key=str)):
        if not path.is_file():
            raise EvidenceError(f"expected generated DSO is missing: {path}")
        header = _run_tool([tools["readelf"], "-h", str(path)])
        if not re.search(r"Type:\s*(?:DYN\b|SharedObject\b)", header.stdout):
            raise EvidenceError(f"expected generated ELF DSO is not ET_DYN: {path}")
        entry = _sha_file_entry(
            path, root, Path("elf-generated") / f"{index:02d}-{path.name}",
            origin=(Path("published-output") / path.name).as_posix())
        generated.append({**entry, "classification": "model", "role": "generated-model",
                          "elf_type": "ET_DYN"})
    target_copy = next(entry for entry in generated
                       if entry["origin"] == (Path("published-output") / target.name).as_posix())
    target_reference = target_copy["path"]
    model_paths = [str(path) for path in expected]
    model_elf = _elf_summary(tools["readelf"], target, root, "elf-model")
    ldd_working_directory = str(Path.cwd().resolve())
    ldd = _run_tool([tools["ldd_r"], "-r", str(target)], check=False,
                    env=runtime_env)
    ldd_text = ldd.stdout + ldd.stderr
    ldd_path = root / "elf-ldd-r.txt"
    ldd_path.write_text(ldd_text, encoding="utf-8")
    paths, unresolved = _parse_ldd(ldd_text, ldd_working_directory)
    ldd_named_paths = _parse_ldd_named_paths(ldd_text, ldd_working_directory)
    runtime_search_paths = _capture_runtime_search_paths(runtime_env, str(target))
    search_dirs = _dependency_search_paths(
        model_elf, str(target), runtime_search_paths,
        working_directory=ldd_working_directory)
    if ldd.returncode != 0:
        unresolved.append(f"ldd -r exited {ldd.returncode}")
    if not paths:
        unresolved.append("ldd -r resolved an empty ELF dependency closure")
    if unresolved:
        raise EvidenceError("unresolved ELF dependency closure: " + "; ".join(unresolved))
    paths = sorted(set(paths) | {str(target)})
    closure = []
    for path in paths:
        if not Path(path).is_file():
            raise EvidenceError(f"ldd resolved a missing ELF DSO: {path}")
        summary = _elf_summary(tools["readelf"], Path(path), root,
                               f"elf-closure-{len(closure):03d}")
        classification, role = _dependency_info(
            path, model_paths=model_paths, sanitizer_dirs=sanitizer_dirs,
            compiler_dirs=compiler_dirs)
        needed = set(model_elf["needed"])
        direct_needed_names = sorted(
            name for name in needed
            if _resolved_needed_path(
                name, [{"path": path, "soname": summary["soname"]}],
                {str(Path(candidate).resolve()) for candidate in paths
                 if str(Path(candidate).resolve()) != str(target)},
                ldd_named_paths, search_dirs) == str(Path(path).resolve()))
        file_reference = target_reference if str(Path(path).resolve()) == str(target) else path
        closure.append({
            "path": file_reference,
            "origin": (target_copy["origin"] if file_reference == target_reference else path),
            "classification": classification, "role": role,
            "direct_to_model": bool(direct_needed_names),
            "direct_needed_names": direct_needed_names,
            "sha256": summary["sha256"], "soname": summary["soname"],
            "needed": summary["needed"], "exports": summary["exports"],
            "undefined_symbols": summary["undefined_symbols"],
            "rpath_runpath": summary["rpath_runpath"],
            "readelf_dynamic": summary["readelf_dynamic"],
            "readelf_symbols": summary["readelf_symbols"],
            "readelf_commands": summary["readelf_commands"],
        })
    compiler_dsos = sorted({item["path"] for item in closure
                            if item["classification"] == "compiler"})
    has_compiler_dso = bool(compiler_dsos)
    llvm = {
        "mode": llvm_mode,
        "status": ("measured_presence" if has_compiler_dso else
                   "not_measured" if llvm_mode == "not_measured" else
                   "measured_no_presence_observed"),
        "shared_objects": compiler_dsos,
        "static_archive_status": "not_measured",
        "absence_claim": ("presence_observed" if has_compiler_dso else
                          "not_measured" if llvm_mode == "not_measured" else
                          "none_observed_in_runtime_closure_static_archives_not_measured"),
    }
    evidence = {
        "schema_version": SCHEMA_VERSION, "kind": ELF_EVIDENCE_KIND,
        "model_library": {"path": target_reference, "origin": target_copy["origin"],
                           "sha256": sha256(target),
                           "size_bytes": target.stat().st_size},
        "library_copy": target_copy,
        "generated_dsos": generated,
        "model_elf": model_elf,
        "ldd_r": {"path": ldd_path.name, "sha256": sha256(ldd_path),
                   "command": [tools["ldd_r"], "-r", str(target)],
                   "working_directory": ldd_working_directory,
                   "returncode": ldd.returncode, "stdout": ldd.stdout,
                   "stderr": ldd.stderr, "search_paths": runtime_search_paths},
        "runtime_closure": closure, "llvm": llvm, "tool_paths": tools,
    }
    evidence_path = root / "elf-dependency-closure.json"
    _write_json(evidence_path, evidence)
    validate_elf_dependency_closure(evidence_path, root)
    return evidence


def validate_elf_dependency_closure(
        evidence: os.PathLike[str] | str,
        evidence_root: os.PathLike[str] | str | None = None) -> dict[str, object]:
    """Validate generic ELF closure evidence; no Vulkan compiler link is required."""
    evidence_path = Path(evidence).resolve()
    root = Path(evidence_root).resolve() if evidence_root else evidence_path.parent
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read ELF dependency evidence: {error}") from error
    if data.get("schema_version") != SCHEMA_VERSION or data.get("kind") != ELF_EVIDENCE_KIND:
        raise EvidenceError("generic ELF dependency evidence schema is invalid")
    if "link_record" in data:
        raise EvidenceError("generic ELF closure must not fabricate a Vulkan link record")
    library = data.get("model_library", {})
    target = _safe_evidence_path(root, library.get("path"))
    if (not target.is_file() or sha256(target) != library.get("sha256") or
            target.stat().st_size != library.get("size_bytes")):
        raise EvidenceError("generic ELF model library is missing or tampered")
    library_copy = data.get("library_copy")
    if not _verified_entry(root, library_copy) or library_copy.get("sha256") != library.get("sha256"):
        raise EvidenceError("generic ELF library copy is missing or mismatched")
    for item in data.get("generated_dsos", []):
        if item.get("elf_type") != "ET_DYN" or not _verified_entry(root, item):
            raise EvidenceError("generic ELF generated DSO evidence is missing or tampered")
    model_elf = data.get("model_elf", {})
    for key in ("readelf_dynamic", "readelf_symbols"):
        raw = model_elf.get(key, {})
        raw_path = _safe_evidence_path(root, raw.get("path"))
        if not _verified_entry(root, {**raw, "size_bytes": raw_path.stat().st_size}):
            raise EvidenceError(f"generic ELF model readelf output is missing: {key}")
    closure = data.get("runtime_closure")
    if not isinstance(closure, list) or not closure:
        raise EvidenceError("generic ELF runtime closure is empty")
    for item in closure:
        path = _resolve_evidence_or_origin_path(root, item.get("path"))
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise EvidenceError(f"generic ELF closure DSO is missing or tampered: {path}")
        for key in ("readelf_dynamic", "readelf_symbols"):
            raw = item.get(key, {})
            raw_path = _safe_evidence_path(root, raw.get("path"))
            if not _verified_entry(root, {**raw, "size_bytes": raw_path.stat().st_size}):
                raise EvidenceError(f"generic ELF raw readelf output is missing: {key}")
    ldd = data.get("ldd_r", {})
    raw_ldd = _safe_evidence_path(root, ldd.get("path"))
    if not _verified_entry(root, {**ldd, "size_bytes": raw_ldd.stat().st_size}):
        raise EvidenceError("generic ELF raw ldd output is missing or tampered")
    _, unresolved = _parse_ldd(
        raw_ldd.read_text(encoding="utf-8", errors="replace"),
        ldd.get("working_directory"))
    if ldd.get("returncode") != 0 or unresolved:
        raise EvidenceError("generic ELF dependency closure has unresolved symbols")
    if str(target) not in {
            str(_resolve_evidence_or_origin_path(root, item["path"])) for item in closure}:
        raise EvidenceError("generic ELF closure omits its root model")
    _validate_runtime_closure(root, library, model_elf, closure, ldd)
    llvm = data.get("llvm", {})
    if llvm.get("static_archive_status") != "not_measured":
        raise EvidenceError("generic ELF API must not claim static archive coverage")
    if llvm.get("status") == "measured_presence" and llvm.get("absence_claim") != "presence_observed":
        raise EvidenceError("generic ELF LLVM presence/absence fields contradict")
    if llvm.get("status") == "not_measured" and llvm.get("absence_claim") != "not_measured":
        raise EvidenceError("generic ELF unmeasured LLVM status claims absence")
    return data


def collect_model_dependencies(library: os.PathLike[str] | str,
                               evidence_root: os.PathLike[str] | str,
                               tool_paths: Mapping[str, os.PathLike[str] | str] | None = None,
                               sanitizer_dirs: Iterable[os.PathLike[str] | str] = (),
                               *, compiler_dirs: Iterable[os.PathLike[str] | str] = (),
                               expected_generated_dsos: Iterable[os.PathLike[str] | str] | None = None,
                               llvm_mode: str = "not_measured",
                               runtime_env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Collect a model DSO's direct and ldd-resolved transitive dependencies.

    `expected_generated_dsos` is explicit to avoid filesystem-wide DSO scans.
    Static LLVM archives are represented by hash/member/map metadata, never
    duplicated into evidence. LLVM presence is reported, not rejected here.
    """
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = Path(library).resolve()
    if not target.is_file():
        raise EvidenceError(f"model library is missing: {target}")
    if llvm_mode not in {"shared", "static", "not_measured"}:
        raise ValueError("llvm_mode must be shared, static, or not_measured")
    tools = _read_tool_paths(tool_paths)
    records = _link_records(root)
    if not records:
        raise EvidenceError("missing actual model-link record from host compiler launcher")
    record_path, link_record = records[-1]
    effective = link_record.get("effective_argv")
    original = link_record.get("original_argv")
    cwd = Path(str(link_record.get("cwd", "")))
    if not isinstance(effective, list) or not isinstance(original, list) or not cwd.is_dir():
        raise EvidenceError("host compiler link record lacks argv or working directory")
    if link_record.get("returncode") != 0:
        raise EvidenceError("recorded actual model link did not complete successfully")
    linked_output = link_record.get("linked_output")
    if not _verified_entry(root, linked_output):
        raise EvidenceError("captured pre-publication model link output is missing or tampered")
    if (sha256(target) != linked_output.get("sha256") or
            target.stat().st_size != linked_output.get("size_bytes")):
        raise EvidenceError("final published model DSO differs from the actual staging link output")
    if any(str(argument).startswith("@") for argument in effective):
        raise EvidenceError("response-file link arguments make archive provenance unresolved")
    staged = link_record.get("staged_snapshot")
    if not isinstance(staged, dict) or staged.get("missing_required"):
        raise EvidenceError("generated wrapper/header/plan/manifest snapshot is incomplete")
    if not staged.get("spirv_count") or not staged.get("payload_count"):
        raise EvidenceError("generated SPIR-V or model payload snapshot is missing")
    for entry in staged.get("files", []):
        if not _verified_entry(root, entry):
            raise EvidenceError(f"staged generated input is missing or tampered: {entry}")
    map_path = _safe_evidence_path(root, link_record.get("map_path"))
    if not map_path.is_file() or map_path.stat().st_size == 0:
        raise EvidenceError("actual linker map is missing or empty")
    map_text = map_path.read_text(encoding="utf-8", errors="replace")
    if ("Cross Reference Table" not in map_text or
            "Archive member included" not in map_text or not _archive_refs(map_text)):
        raise EvidenceError("actual --cref linker map is missing archive provenance")
    capture_errors = link_record.get("archive_capture_errors", [])
    if capture_errors:
        raise EvidenceError("link-time archive capture was incomplete: " + "; ".join(capture_errors))
    archive_inputs = _archive_inventory(
        link_record.get("archive_snapshots"), map_text, root)
    map_evidence = _sha_file_entry(
        map_path, root, Path("link") / map_path.name,
        origin=(Path("linker-map") / map_path.name).as_posix())
    # Save-temps objects are captured by the proxy after the real link returns.
    objects = link_record.get("wrapper_objects")
    if not isinstance(objects, list) or not objects:
        raise EvidenceError("compiler did not produce a captured wrapper object with -save-temps=obj")
    object_evidence = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict) or not _verified_entry(root, item):
            raise EvidenceError("captured wrapper object is missing or tampered")
        object_path = _safe_evidence_path(root, item.get("path"))
        sections = _run_tool([tools["readelf"], "-S", str(object_path)])
        symbols = _run_tool([tools["readelf"], "-Ws", str(object_path)])
        if ".symtab" not in sections.stdout:
            raise EvidenceError(f"saved wrapper object is stripped: {object_path}")
        sections_file = root / f"wrapper-object-{index:02d}.sections.txt"
        symbols_file = root / f"wrapper-object-{index:02d}.symbols.txt"
        sections_file.write_text(sections.stdout + sections.stderr, encoding="utf-8")
        symbols_file.write_text(symbols.stdout + symbols.stderr, encoding="utf-8")
        object_evidence.append({
            **item,
            "classification": "model",
            "unstripped": True,
            "readelf_sections": {"path": sections_file.name,
                                 "sha256": sha256(sections_file)},
            "readelf_symbols": {"path": symbols_file.name,
                                "sha256": sha256(symbols_file)},
        })

    expected = ([Path(item).resolve() for item in expected_generated_dsos]
                if expected_generated_dsos is not None else [target])
    if target not in expected:
        raise EvidenceError("model DSO is not in the explicit expected generated DSO set")
    generated = []
    for index, path in enumerate(sorted(set(expected), key=str)):
        if not path.is_file():
            raise EvidenceError(f"expected generated DSO is missing: {path}")
        header = _run_tool([tools["readelf"], "-h", str(path)])
        if not re.search(r"Type:\s*(?:DYN\b|SharedObject\b)", header.stdout):
            raise EvidenceError(f"expected generated model DSO is not ELF ET_DYN: {path}")
        entry = _sha_file_entry(
            path, root, Path("generated") / f"{index:02d}-{path.name}",
            origin=(Path("published-output") / path.name).as_posix())
        generated.append({**entry, "classification": "model", "elf_type": "ET_DYN"})
    model_paths = [str(path) for path in expected]
    target_copy = next(entry for entry in generated
                       if Path(entry["origin"]).name == target.name and
                       Path(entry["origin"]).parent.name == "published-output")
    target_reference = target_copy["path"]
    elf = _elf_summary(tools["readelf"], target, root, "model")
    ldd_working_directory = str(Path.cwd().resolve())
    ldd = _run_tool([tools["ldd_r"], "-r", str(target)], check=False,
                    env=runtime_env)
    ldd_text = ldd.stdout + ldd.stderr
    ldd_path = root / "ldd-r.txt"
    ldd_path.write_text(ldd_text, encoding="utf-8")
    closure_paths, unresolved = _parse_ldd(ldd_text, ldd_working_directory)
    ldd_named_paths = _parse_ldd_named_paths(ldd_text, ldd_working_directory)
    runtime_search_paths = _capture_runtime_search_paths(runtime_env, str(target))
    search_dirs = _dependency_search_paths(
        elf, str(target), runtime_search_paths,
        working_directory=ldd_working_directory)
    if ldd.returncode != 0:
        unresolved.append(f"ldd -r exited {ldd.returncode}")
    if not closure_paths:
        unresolved.append("ldd -r resolved no shared-object paths (empty dependency closure)")
    if unresolved:
        raise EvidenceError("unresolved or empty runtime dependency closure: " + "; ".join(unresolved))
    closure_paths = sorted(set(closure_paths) | {str(target)})
    closure = []
    for path in closure_paths:
        if not Path(path).is_file():
            raise EvidenceError(f"ldd resolved a missing DSO path: {path}")
        summary = _elf_summary(tools["readelf"], Path(path), root,
                               f"closure-{len(closure):03d}")
        classification, role = _dependency_info(
            path, model_paths=model_paths, sanitizer_dirs=sanitizer_dirs,
            compiler_dirs=compiler_dirs)
        direct_needed = set(elf["needed"])
        direct_needed_names = sorted(
            name for name in direct_needed
            if _resolved_needed_path(
                name, [{"path": path, "soname": summary["soname"]}],
                {str(Path(candidate).resolve()) for candidate in closure_paths
                 if str(Path(candidate).resolve()) != str(target)},
                ldd_named_paths, search_dirs) == str(Path(path).resolve()))
        file_reference = target_reference if str(Path(path).resolve()) == str(target) else path
        closure.append({
            "path": file_reference,
            "origin": (target_copy["origin"] if file_reference == target_reference else path),
            "classification": classification,
            "role": role,
            "direct_to_model": bool(direct_needed_names),
            "direct_needed_names": direct_needed_names,
            "sha256": summary["sha256"],
            "soname": summary["soname"],
            "needed": summary["needed"],
            "exports": summary["exports"],
            "undefined_symbols": summary["undefined_symbols"],
            "rpath_runpath": summary["rpath_runpath"],
            "readelf_dynamic": summary["readelf_dynamic"],
            "readelf_symbols": summary["readelf_symbols"],
            "readelf_commands": summary["readelf_commands"],
        })
    if not closure:
        raise EvidenceError("runtime dependency closure is empty")

    # Report possible legacy LLVM blockers explicitly; this is not a gate.
    compiler_members = [
        origin["member"]
        for archive in archive_inputs
        for origin in archive["selected_member_origins"]
        if origin["classification"] == "compiler"
    ]
    compiler_dsos = [entry["path"] for entry in closure
                     if entry["classification"] == "compiler"]
    llvm_legacy_blockers = []
    if compiler_members:
        llvm_legacy_blockers.append("static_llvm_archive_members_selected")
    if compiler_dsos:
        llvm_legacy_blockers.append("shared_llvm_dso_in_runtime_closure")
    if llvm_mode == "static" and not compiler_members:
        llvm_legacy_blockers.append("configured_static_llvm_not_observed")
    if llvm_mode == "shared" and not compiler_dsos:
        llvm_legacy_blockers.append("configured_shared_llvm_not_observed")
    evidence: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "model_library": {"path": target_reference, "origin": target_copy["origin"],
                           "sha256": sha256(target), "size_bytes": target.stat().st_size},
        "linked_output": link_record.get("linked_output"),
        "link_record": {"path": record_path.name, "sha256": sha256(record_path),
                         "original_argv": original, "effective_argv": effective,
                         "cwd": str(cwd),
                         "logical_compile_dir": link_record.get("logical_compile_dir"),
                         "staging_output": link_record.get("output"),
                         "executable": link_record.get("executable")},
        "link_map": map_evidence,
        "archive_inputs": archive_inputs,
        "wrapper_objects": object_evidence,
        "staged_snapshot": link_record.get("staged_snapshot"),
        "generated_dsos": generated,
        "model_elf": elf,
        "ldd_r": {"path": ldd_path.name, "sha256": sha256(ldd_path),
                   "command": [tools["ldd_r"], "-r", str(target)],
                   "working_directory": ldd_working_directory,
                   "returncode": ldd.returncode, "stdout": ldd.stdout,
                   "stderr": ldd.stderr, "search_paths": runtime_search_paths},
        "runtime_closure": closure,
        "llvm": {
            "mode": llvm_mode,
            "status": ("measured_presence" if compiler_members or compiler_dsos else
                       "not_measured" if llvm_mode == "not_measured" else
                       "measured_no_presence_observed"),
            "static_archive_members": sorted(set(compiler_members)),
            "shared_objects": sorted(set(compiler_dsos)),
            "absence_claim": ("presence_observed" if compiler_members or compiler_dsos else
                              "not_measured" if llvm_mode == "not_measured" else
                              "none_observed_in_captured_link_and_runtime_closure"),
            "legacy_blockers": llvm_legacy_blockers,
            "policy": "reported_only_legacy_blocker_not_rejected",
        },
        "tool_paths": tools,
    }
    evidence_path = root / "model-dependency-evidence.json"
    _write_json(evidence_path, evidence)
    validate_dependency_evidence(evidence_path, root)
    return evidence


def _safe_evidence_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EvidenceError("evidence path is missing")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceError(f"evidence path escapes its root: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise EvidenceError(f"evidence path escapes its root: {relative}")
    return resolved


def _resolve_evidence_or_origin_path(root: Path, value: object) -> Path:
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).resolve()
    return _safe_evidence_path(root, value)


def _resolved_needed_path(name: str, closure: list[dict],
                          closure_paths: set[str],
                          named_paths: dict[str, str],
                          search_dirs: Iterable[str] = ()) -> str | None:
    resolved = named_paths.get(name)
    if resolved in closure_paths:
        return resolved
    for item in closure:
        item_path = str(item.get("path", ""))
        path = str(Path(item_path).resolve()) if Path(item_path).is_absolute() else None
        if path not in closure_paths:
            continue
        if name == Path(item_path).name or name == item.get("soname"):
            return path
        aliases = ([Path(name)] if Path(name).is_absolute() else
                   [Path(directory) / name for directory in
                    {str(Path(path).parent), *(str(Path(item).resolve()) for item in search_dirs)}])
        for alias in aliases:
            try:
                if alias.is_file() and str(alias.resolve()) == path:
                    return path
            except (OSError, RuntimeError):
                continue
    return None


def _validate_runtime_closure(root: Path, model_library: dict,
                              model_elf: dict, closure: list[dict],
                              ldd_record: dict) -> None:
    needed = model_elf.get("needed")
    if not isinstance(needed, list) or any(not isinstance(item, str) for item in needed):
        raise EvidenceError("model ELF direct DT_NEEDED inventory is malformed")
    raw_dynamic = model_elf.get("readelf_dynamic", {})
    raw_dynamic_path = _safe_evidence_path(root, raw_dynamic.get("path"))
    if (not raw_dynamic_path.is_file() or
            sha256(raw_dynamic_path) != raw_dynamic.get("sha256")):
        raise EvidenceError("model ELF raw dynamic table is missing or tampered")
    actual_needed = parse_readelf_dynamic(
        raw_dynamic_path.read_text(encoding="utf-8", errors="replace"))["needed"]
    if actual_needed != sorted(set(needed)):
        raise EvidenceError("model ELF DT_NEEDED claim differs from raw readelf output")

    closure_paths = set()
    model_path = _safe_evidence_path(root, model_library.get("path"))
    for item in closure:
        resolved = _resolve_evidence_or_origin_path(root, item.get("path"))
        if resolved != model_path:
            closure_paths.add(str(resolved))

    command = ldd_record.get("command")
    if (not isinstance(command, list) or not command or
            not isinstance(command[-1], str) or not Path(command[-1]).is_absolute()):
        raise EvidenceError("raw ldd command does not identify its absolute model target")
    target = str(Path(command[-1]).resolve())
    ldd_working_directory = ldd_record.get("working_directory")
    if (not isinstance(ldd_working_directory, str) or
            not Path(ldd_working_directory).is_absolute() or
            not Path(ldd_working_directory).is_dir()):
        raise EvidenceError("ldd working directory is missing or invalid")
    ldd_working_directory = str(Path(ldd_working_directory).resolve())
    search_paths = ldd_record.get("search_paths")
    if (not isinstance(search_paths, list) or
            any(not isinstance(item, str) or not Path(item).is_absolute()
                for item in search_paths)):
        raise EvidenceError("ldd runtime search path inventory is malformed")
    search_dirs = _dependency_search_paths(
        model_elf, target, search_paths, working_directory=ldd_working_directory)
    ldd_path = _safe_evidence_path(root, ldd_record.get("path"))
    ldd_text = ldd_path.read_text(encoding="utf-8", errors="replace")
    ldd_paths, unresolved = _parse_ldd(ldd_text, ldd_working_directory)
    if unresolved:
        raise EvidenceError("raw ldd output contains unresolved dependency records")
    observed_paths = set(ldd_paths)
    observed_paths.discard(target)
    if closure_paths != observed_paths:
        raise EvidenceError(
            "runtime closure paths differ from resolved raw ldd output: "
            f"missing={sorted(observed_paths - closure_paths)} "
            f"extra={sorted(closure_paths - observed_paths)}")
    named_paths = _parse_ldd_named_paths(ldd_text, ldd_working_directory)
    needed_paths = {name: _resolved_needed_path(
        name, closure, closure_paths, named_paths, search_dirs)
                      for name in set(needed)}
    missing_needed = sorted(name for name, path in needed_paths.items() if path is None)
    if missing_needed:
        raise EvidenceError(
            f"direct DT_NEEDED entries are absent from runtime closure: {missing_needed}")
    for item in closure:
        resolved = str(_resolve_evidence_or_origin_path(root, item.get("path")))
        expected_names = sorted(name for name, path in needed_paths.items()
                                if path == resolved)
        if (item.get("direct_needed_names") != expected_names or
                item.get("direct_to_model") is not bool(expected_names)):
            raise EvidenceError("direct-to-model closure classification differs from ELF/ldd evidence")


def _verified_entry(root: Path, entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        path = _safe_evidence_path(root, entry.get("path"))
        return (path.is_file() and sha256(path) == entry.get("sha256") and
                path.stat().st_size == entry.get("size_bytes"))
    except (OSError, EvidenceError):
        return False


def validate_dependency_evidence(evidence: os.PathLike[str] | str,
                                 evidence_root: os.PathLike[str] | str | None = None) -> dict[str, object]:
    """Validate hashes, actual-link provenance, archives, and nonempty closure."""
    evidence_path = Path(evidence).resolve()
    root = Path(evidence_root).resolve() if evidence_root else evidence_path.parent
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read dependency evidence {evidence_path}: {error}") from error
    if data.get("schema_version") != SCHEMA_VERSION or data.get("kind") != EVIDENCE_KIND:
        raise EvidenceError("dependency evidence schema/kind is invalid")
    closure = data.get("runtime_closure")
    if not isinstance(closure, list) or not closure:
        raise EvidenceError("runtime dependency closure is falsely empty")
    link_record = data.get("link_record")
    if not isinstance(link_record, dict):
        raise EvidenceError("actual host compiler link record is missing")
    link_record_file = _safe_evidence_path(root, link_record.get("path"))
    if not _verified_entry(root, {"path": link_record.get("path"),
                                  "sha256": link_record.get("sha256"),
                                  "size_bytes": link_record_file.stat().st_size}):
        raise EvidenceError("actual host compiler link record is missing or tampered")
    captured_link = json.loads(link_record_file.read_text(encoding="utf-8"))
    if captured_link.get("returncode") != 0:
        raise EvidenceError("actual host compiler link record is unsuccessful")
    linked_output = data.get("linked_output")
    if (not _verified_entry(root, linked_output) or
            linked_output != captured_link.get("linked_output")):
        raise EvidenceError("captured staging link output is missing or tampered")
    link_map = data.get("link_map")
    if not _verified_entry(root, link_map):
        raise EvidenceError("actual linker map is missing or tampered")
    map_file = _safe_evidence_path(root, link_map["path"])
    map_text = map_file.read_text(encoding="utf-8", errors="replace")
    map_refs = _archive_refs(map_text)
    if ("Cross Reference Table" not in map_text or
            "Archive member included" not in map_text or not map_refs):
        raise EvidenceError("actual --cref linker map has no archive-member provenance")
    archives = data.get("archive_inputs")
    captured_archives = captured_link.get("archive_snapshots")
    if not isinstance(archives, list) or not archives or not isinstance(captured_archives, list):
        raise EvidenceError("archive provenance is missing")
    captured_by_path = {str(Path(item.get("path", "")).resolve()): item
                        for item in captured_archives if isinstance(item, dict)}
    assigned_refs: set[str] = set()
    for archive in archives:
        source = Path(str(archive.get("path", ""))).resolve()
        captured = captured_by_path.get(str(source))
        if captured is None:
            raise EvidenceError(f"archive has no actual-link-time snapshot: {source}")
        for key in ("sha256", "size_bytes", "selected_members_from_map",
                    "selected_member_hashes", "ar_t"):
            if archive.get(key) != captured.get(key):
                raise EvidenceError(f"archive {key} differs from actual-link-time capture: {source}")
        members = archive.get("members")
        selected = archive.get("selected_members_from_map")
        raw = archive.get("ar_t")
        if not isinstance(members, list) or not members or not isinstance(selected, list):
            raise EvidenceError(f"archive member provenance is incomplete: {source}")
        if not isinstance(raw, dict) or not _verified_entry(root, raw):
            raise EvidenceError(f"raw ar t listing is missing or tampered: {source}")
        raw_file = _safe_evidence_path(root, raw.get("path"))
        raw_members = [line.strip() for line in raw_file.read_text(
            encoding="utf-8", errors="replace").splitlines() if line.strip()]
        member_names = {item.get("name") for item in members if isinstance(item, dict)}
        member_list = [item.get("name") for item in members]
        if raw_members != member_list or raw_members != captured.get("members"):
            raise EvidenceError(f"archive members differ from raw ar t listing: {source}")
        if source.is_file():
            if sha256(source) != archive.get("sha256") or source.stat().st_size != archive.get("size_bytes"):
                raise EvidenceError(f"archive source is tampered: {source}")
            if archive.get("source_status") != "present_and_hash_verified":
                raise EvidenceError(f"archive source status is inconsistent: {source}")
        elif archive.get("source_status") != "missing_at_collection_link_time_ar_t_verified":
            raise EvidenceError(f"missing archive has no accepted raw member evidence: {source}")
        selected_member_names = {
            item.split("(", 1)[1][:-1] for item in selected
            if isinstance(item, str) and "(" in item and item.endswith(")")
        }
        if not selected_member_names.issubset(member_names):
            raise EvidenceError(f"link-map selected member is absent from archive index: {source}")
        member_hashes = archive.get("selected_member_hashes")
        if not isinstance(member_hashes, list) or {
                item.get("member") for item in member_hashes if isinstance(item, dict)
        } != selected_member_names:
            raise EvidenceError(f"selected archive member hashes are incomplete: {source}")
        for item in member_hashes:
            if (not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) or
                    type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0):
                raise EvidenceError(f"invalid selected member raw hash: {source}")
        hash_by_member = {item["member"]: item for item in member_hashes}
        origins = archive.get("selected_member_origins")
        expected_origins = {
            (str(source), member, f"{source}({member})",
             hash_by_member[member]["sha256"], hash_by_member[member]["size_bytes"])
            for member in selected_member_names
        }
        if not isinstance(origins, list) or {
                (item.get("archive"), item.get("member"), item.get("origin"),
                 item.get("sha256"), item.get("size_bytes"))
                for item in origins if isinstance(item, dict)
        } != expected_origins:
            raise EvidenceError(f"selected archive member origins are incomplete: {source}")
        assigned_refs.update(selected)
    if assigned_refs != map_refs:
        raise EvidenceError(
            f"archive member provenance does not exactly match the linker map: "
            f"missing={sorted(map_refs-assigned_refs)} extra={sorted(assigned_refs-map_refs)}")
    wrapper_objects = data.get("wrapper_objects", [])
    if not wrapper_objects:
        raise EvidenceError("unstripped wrapper object evidence is missing")
    for entry in wrapper_objects:
        if not _verified_entry(root, entry) or entry.get("unstripped") is not True:
            raise EvidenceError("captured wrapper object is missing, stripped, or tampered")
        for key in ("readelf_sections", "readelf_symbols"):
            output = entry.get(key, {})
            output_path = _safe_evidence_path(root, output.get("path"))
            if not _verified_entry(root, {**output, "size_bytes": output_path.stat().st_size}):
                raise EvidenceError(f"wrapper object raw readelf output is missing: {key}")
            if key == "readelf_sections" and ".symtab" not in output_path.read_text(
                    encoding="utf-8", errors="replace"):
                raise EvidenceError("wrapper object section table has no symbol table")
    staged = data.get("staged_snapshot")
    if not isinstance(staged, dict) or staged.get("missing_required") or not staged.get("spirv_count") or not staged.get("payload_count"):
        raise EvidenceError("generated wrapper/header/plan/manifest/SPIR-V/payload snapshot is incomplete")
    for entry in staged.get("files", []):
        if not _verified_entry(root, entry):
            raise EvidenceError("staged generated input is missing or tampered")
    generated = data.get("generated_dsos")
    if not isinstance(generated, list) or not generated:
        raise EvidenceError("expected generated DSO evidence is missing")
    for item in generated:
        if item.get("elf_type") != "ET_DYN" or not _verified_entry(root, item):
            raise EvidenceError(f"generated DSO copy is missing, invalid, or tampered: {item}")
    library = data.get("model_library", {})
    if not isinstance(library, dict):
        raise EvidenceError("model ELF identity is missing")
    model_path = _safe_evidence_path(root, library.get("path"))
    if (not model_path.is_file() or sha256(model_path) != library.get("sha256") or
            model_path.stat().st_size != library.get("size_bytes")):
        raise EvidenceError("published model ELF is missing or tampered")
    linked_copy = data.get("linked_output")
    if not _verified_entry(root, linked_copy):
        raise EvidenceError("captured pre-publication link output is missing or tampered")
    if (linked_copy.get("sha256") != library.get("sha256") or
            linked_copy.get("size_bytes") != library.get("size_bytes")):
        raise EvidenceError("published model ELF differs from actual linker output bytes")
    if not any(item.get("sha256") == library.get("sha256") for item in generated):
        raise EvidenceError("generated model DSO copy does not match the linked library")
    model_elf = data.get("model_elf", {})
    for key in ("readelf_dynamic", "readelf_symbols"):
        entry = model_elf.get(key, {})
        path = _safe_evidence_path(root, entry.get("path"))
        if not _verified_entry(root, {**entry, "size_bytes": path.stat().st_size}):
            raise EvidenceError(f"raw model readelf output is missing or tampered: {key}")
    for item in closure:
        path = _resolve_evidence_or_origin_path(root, item.get("path"))
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise EvidenceError(f"runtime closure DSO is missing or tampered: {path}")
        for key in ("readelf_dynamic", "readelf_symbols"):
            entry = item.get(key, {})
            evidence_file = _safe_evidence_path(root, entry.get("path"))
            if not _verified_entry(root, {**entry, "size_bytes": evidence_file.stat().st_size}):
                raise EvidenceError(f"raw closure readelf output is missing or tampered: {key}")
    if str(model_path) not in {
            str(_resolve_evidence_or_origin_path(root, item["path"])) for item in closure}:
        raise EvidenceError("runtime dependency closure omits the model DSO")
    ldd_entry = data.get("ldd_r")
    if not isinstance(ldd_entry, dict):
        raise EvidenceError("raw ldd -r output is missing")
    ldd_file = _safe_evidence_path(root, ldd_entry.get("path"))
    if not _verified_entry(root, {**ldd_entry, "size_bytes": ldd_file.stat().st_size}):
        raise EvidenceError("raw ldd -r output is missing or tampered")
    _, unresolved = _parse_ldd(
        ldd_file.read_text(encoding="utf-8", errors="replace"),
        ldd_entry.get("working_directory"))
    if ldd_entry.get("returncode") != 0 or unresolved:
        raise EvidenceError("raw ldd -r output reports unresolved dependencies")
    _validate_runtime_closure(root, library, model_elf, closure, ldd_entry)
    if any(str(item.get("classification")) not in
           {"compiler", "model", "platform", "driver", "SAN"}
           for item in closure):
        raise EvidenceError("runtime closure has an unknown classification")
    llvm = data.get("llvm", {})
    if llvm.get("policy") != "reported_only_legacy_blocker_not_rejected" or not llvm.get("status") or not llvm.get("absence_claim"):
        raise EvidenceError("LLVM status/policy is missing; no absence claim is permitted")
    if llvm["status"] == "measured_presence" and llvm["absence_claim"] != "presence_observed":
        raise EvidenceError("LLVM presence and absence claim contradict each other")
    if llvm["status"] == "not_measured" and llvm["absence_claim"] != "not_measured":
        raise EvidenceError("unmeasured LLVM status must not claim absence")
    return data


def _mapped_paths(maps_text: str) -> list[str]:
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        path = fields[5].removesuffix(" (deleted)")
        if path.startswith("/"):
            paths.add(str(Path(path).resolve()))
    return sorted(paths)


def record_driver_module_snapshot(evidence_root: os.PathLike[str] | str,
                                  phase: str,
                                  maps_text: str | None = None,
                                  maps_path: os.PathLike[str] | str = "/proc/self/maps",
                                  driver_manifest_paths: Iterable[os.PathLike[str] | str] = ()) -> dict[str, object]:
    """Record module maps at three loader stages.

    For a child native loader, pass its captured maps text. The default reads
    `maps_path` in the caller process and does not reach into a separate PID.
    """
    allowed = {"before-model", "after-dlopen", "after-create"}
    if phase not in allowed:
        raise ValueError(f"phase must be one of {sorted(allowed)}")
    if maps_text is None:
        try:
            maps_text = Path(maps_path).read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise EvidenceError(f"cannot read loader map {maps_path}: {error}") from error
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"driver-modules-{phase}.txt"
    output.write_text(maps_text, encoding="utf-8")
    paths = _mapped_paths(maps_text)
    manifest_paths = sorted(str(Path(path).resolve()) for path in driver_manifest_paths)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ncnn.driver_module_snapshot",
        "phase": phase,
        "driver_manifest_paths": manifest_paths,
        "raw": {"path": output.name, "sha256": sha256(output)},
        "modules": [{"path": path,
                     "classification": _dependency_info(path)[0],
                     "role": _dependency_info(path)[1],
                     "explicit_manifest_origin": path in manifest_paths}
                    for path in paths],
    }
    _write_json(root / f"driver-modules-{phase}.json", record)
    return record


def compare_driver_module_snapshots(evidence_root: os.PathLike[str] | str, *,
                                    persist: bool = True) -> dict[str, object]:
    """Compare three loader phases without inferring absence from unmeasured data."""
    root = Path(evidence_root).resolve()
    phases = ("before-model", "after-dlopen", "after-create")
    snapshots: dict[str, dict[str, object]] = {}
    for phase in phases:
        path = root / f"driver-modules-{phase}.json"
        if not path.is_file():
            return {"schema_version": SCHEMA_VERSION,
                    "kind": "ncnn.driver_module_comparison",
                    "status": "not_measured", "missing_phases": [item for item in phases
                                                                      if not (root / f"driver-modules-{item}.json").is_file()],
                    "llvm_absence_claim": "not_made"}
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("phase") != phase or data.get("kind") != "ncnn.driver_module_snapshot":
            raise EvidenceError(f"malformed loader snapshot: {path}")
        raw = data.get("raw", {})
        if not _verified_entry(root, {"path": raw.get("path"), "sha256": raw.get("sha256"),
                                      "size_bytes": _safe_evidence_path(root, raw.get("path")).stat().st_size}):
            raise EvidenceError(f"loader snapshot raw map is missing or tampered: {path}")
        snapshots[phase] = data
    sets = {phase: {entry["path"] for entry in data["modules"]}
            for phase, data in snapshots.items()}
    before, loaded, created = (sets[phase] for phase in phases)
    after_dlopen_delta = loaded - before
    after_create_delta = created - loaded
    module_index = {entry["path"]: entry for phase in phases
                    for entry in snapshots[phase]["modules"]}
    manifest_sets = {tuple(data.get("driver_manifest_paths", []))
                     for data in snapshots.values()}
    if len(manifest_sets) != 1:
        raise EvidenceError("driver manifest paths differ across loader snapshots")
    manifest_paths = set(next(iter(manifest_sets)))

    def describe(paths: Iterable[str], phase_label: str,
                 *, after_create: bool = False) -> list[dict[str, object]]:
        result = []
        for path in sorted(paths):
            entry = dict(module_index[path])
            classification, role = _dependency_info(path)
            if after_create and path in manifest_paths:
                classification, role = "driver", "explicit-manifest-icd-after-create"
            entry.update({"classification": classification, "role": role,
                          "origin_phase": phase_label,
                          "manifest_origin": "explicit-driver-manifest" if
                          path in manifest_paths else "not-established"})
            result.append(entry)
        return result

    after_dlopen_modules = describe(after_dlopen_delta, "after-dlopen-delta")
    after_create_modules = describe(after_create_delta, "driver-after-create-delta",
                                    after_create=True)
    driver_modules = [entry for entry in after_create_modules
                      if entry["classification"] == "driver"]
    model_evidence_path = root / "model-dependency-evidence.json"
    model_llvm = "not_measured"
    model_llvm_claim = "not_measured"
    if model_evidence_path.is_file():
        model_evidence = json.loads(model_evidence_path.read_text(encoding="utf-8"))
        llvm = model_evidence.get("llvm", {})
        model_llvm = llvm.get("status", "malformed")
        model_llvm_claim = llvm.get("absence_claim", "malformed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ncnn.driver_module_comparison",
        "status": "measured",
        "driver_origin_status": "measured" if driver_modules else "not_measured",
        "before_model": sorted(before),
        "after_dlopen_added": sorted(after_dlopen_delta),
        "after_dlopen_added_modules": after_dlopen_modules,
        "after_create_added": sorted(after_create_delta),
        "after_create_added_modules": after_create_modules,
        "after_create_removed": sorted(loaded - created),
        "driver_icd_modules_after_create": driver_modules,
        "driver_process_compiler_modules_after_create_delta": [
            entry for entry in after_create_modules
            if _dependency_info(str(entry["path"]))[0] == "compiler"
        ],
        "model_llvm_status": model_llvm,
        "model_llvm_absence_claim": model_llvm_claim,
        "driver_origin_does_not_exempt_model_llvm": True,
        "llvm_absence_claim": "not_made",
    }
    if persist:
        _write_json(root / "driver-module-comparison.json", result)
    return result


__all__ = ["EvidenceError", "make_host_cxx_launcher", "collect_model_dependencies",
           "validate_dependency_evidence", "collect_elf_dependency_closure",
           "validate_elf_dependency_closure", "record_driver_module_snapshot",
           "compare_driver_module_snapshots"]
