"""Content-bound inventory validation for installed LLVM/MLIR sanitizer toolchains.

The module is deliberately separate from the legacy release auditor.  It checks
compiler inputs, installed CMake exports, the install manifest and actual archive
files as one hash-bound inventory.  A path resolver can map original install
paths to staged copies without changing the paths recorded by CMake exports.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
from collections.abc import Callable, Iterable, Mapping


SCHEMA_VERSION = 2
SUPPORTED_LLVM_MLIR_VERSION = "21.1.8"
_SANITIZERS_REQUIRED = frozenset({"address", "undefined"})
_C_SOURCE_SUFFIXES = {".c", ".m"}
_CPP_SOURCE_SUFFIXES = {".cc", ".cp", ".cpp", ".cxx", ".c++", ".mm", ".C"}
_ASM_SOURCE_SUFFIXES = {".s", ".S", ".asm"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class InventoryError(ValueError):
    """Raised when toolchain inventory evidence is incomplete or inconsistent."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise InventoryError(f"cannot read {description}: {path}: {error}") from error


def _parse_cmake_cache(path: Path) -> dict[str, tuple[str, str]]:
    values: dict[str, tuple[str, str]] = {}
    for line in _read_text(path, "CMakeCache.txt").splitlines():
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        match = re.fullmatch(r"([^:=]+):([^=]+)=(.*)", line)
        if match:
            values[match.group(1)] = (match.group(2), match.group(3))
    return values


def _cache_value(cache: Mapping[str, tuple[str, str]], key: str) -> str | None:
    entry = cache.get(key)
    return None if entry is None else entry[1]


def _validate_source_archive_binding(
    cache: Mapping[str, tuple[str, str]],
    archive_path: Path,
    expected_version: str,
) -> Path:
    source_dir_text = _cache_value(cache, "LLVM_SOURCE_DIR")
    if not source_dir_text:
        raise InventoryError("CMakeCache.txt lacks LLVM_SOURCE_DIR")
    archive_name = archive_path.name
    for suffix in (".tar.xz", ".tar.gz", ".tar.zst", ".tgz", ".zip"):
        if archive_name.endswith(suffix):
            archive_root = archive_name[: -len(suffix)]
            break
    else:
        raise InventoryError("official LLVM source archive has an unsupported name")
    expected_root = f"llvm-project-{expected_version}.src"
    if archive_root != expected_root:
        raise InventoryError(
            f"official source archive does not match LLVM {expected_version}: {archive_name}"
        )
    source_dir = Path(source_dir_text).resolve(strict=False)
    if source_dir.name != "llvm" or source_dir.parent.name != archive_root:
        raise InventoryError(
            "CMake LLVM_SOURCE_DIR does not descend from the hash-bound official source archive"
        )
    return source_dir


def _parse_version(path: Path, variable: str) -> str:
    text = _read_text(path, f"installed {variable} config")
    match = re.search(
        rf"set\(\s*{re.escape(variable)}\s+[\"']?([0-9]+(?:\.[0-9]+)+)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        raise InventoryError(f"{path} does not declare {variable}")
    return match.group(1)


def _parse_mlir_llvm_contract(
    path: Path, llvm_config_path: Path, expected_version: str
) -> str:
    text = _read_text(path, "installed MLIR config")
    version_match = re.search(
        r"set\(\s*LLVM_VERSION\s+[\"']?([0-9]+(?:\.[0-9]+)+)",
        text,
        re.IGNORECASE,
    )
    if version_match is None:
        raise InventoryError(f"{path} does not declare its LLVM_VERSION")
    version = version_match.group(1)
    if version != expected_version:
        raise InventoryError(
            f"MLIR requires LLVM {version}, expected {expected_version}"
        )

    package_block = None
    for match in re.finditer(r"find_package\(\s*LLVM\b(.*?)\)", text, re.DOTALL | re.IGNORECASE):
        block = match.group(1)
        if re.search(r"\$\{LLVM_VERSION\}", block, re.IGNORECASE):
            package_block = block
            break
    if package_block is None or not all(
        re.search(pattern, package_block, re.IGNORECASE)
        for pattern in (r"\bEXACT\b", r"\bREQUIRED\b", r"\bCONFIG\b", r"\bHINTS\b")
    ):
        raise InventoryError(
            "MLIRConfig.cmake must find LLVM_VERSION EXACT REQUIRED CONFIG from HINTS"
        )
    hints_match = re.search(r"\bHINTS\s+(.+)", package_block, re.DOTALL | re.IGNORECASE)
    quoted_hint = (
        None
        if hints_match is None
        else re.search(r"[\"']([^\"']+)[\"']", hints_match.group(1))
    )
    if quoted_hint is None:
        raise InventoryError(f"{path} has no single quoted LLVM CMake HINTS path")
    hint = quoted_hint.group(1)
    mlir_prefix = path.resolve().parent.parent.parent.parent
    hint = hint.replace("${MLIR_INSTALL_PREFIX}", str(mlir_prefix))
    hint_path = Path(hint)
    if "$" in hint or "@" in hint:
        raise InventoryError(f"MLIRConfig.cmake has an unresolved LLVM HINTS path: {hint}")
    if not hint_path.is_absolute():
        hint_path = mlir_prefix / hint_path
    hint_path = hint_path.resolve(strict=False)
    if hint_path != llvm_config_path.resolve().parent:
        raise InventoryError(
            "MLIRConfig.cmake does not resolve EXACT LLVM from the same install prefix"
        )
    return version


def _hash_input(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise InventoryError(f"{label} is missing: {path}")
    data = path.read_bytes()
    return {
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def _resolve_path(
    source: Path,
    resolver: Callable[[Path], Path | str] | Mapping[Path | str, Path | str] | None,
) -> Path:
    source = source.resolve(strict=False)
    if resolver is None:
        return source
    if callable(resolver):
        try:
            target = resolver(source)
        except (KeyError, OSError, ValueError) as error:
            raise InventoryError(f"resolver has no mapping for {source}") from error
    else:
        target = resolver.get(source)
        if target is None:
            target = resolver.get(str(source))
        if target is None:
            raise InventoryError(f"resolver has no mapping for {source}")
    return Path(target).resolve(strict=False)


def _expand_response_files(
    arguments: list[str],
    directory: Path,
    *,
    resolver: Callable[[Path], Path | str]
    | Mapping[Path | str, Path | str]
    | None = None,
    depth: int = 0,
    stack: tuple[Path, ...] = (),
) -> tuple[list[str], list[dict[str, object]]]:
    if depth > 16:
        raise InventoryError("compiler response-file nesting exceeds 16 levels")
    expanded: list[str] = []
    response_records: dict[str, dict[str, object]] = {}
    for argument in arguments:
        if not argument.startswith("@") or argument == "@":
            expanded.append(argument)
            continue
        source_path = Path(argument[1:])
        if not source_path.is_absolute():
            source_path = directory / source_path
        source_path = source_path.resolve(strict=False)
        if source_path in stack:
            raise InventoryError(f"compiler response-file cycle at {source_path}")
        response_path = _resolve_path(source_path, resolver)
        if not response_path.is_file():
            raise InventoryError(f"compiler response file is missing: {source_path}")
        response_bytes = response_path.read_bytes()
        if source_path.is_file() and response_path != source_path:
            if source_path.read_bytes() != response_bytes:
                raise InventoryError(
                    f"staged response file differs from source bytes: "
                    f"{source_path} -> {response_path}"
                )
        try:
            response_arguments = shlex.split(response_bytes.decode("utf-8"), posix=True)
        except (UnicodeError, ValueError) as error:
            raise InventoryError(
                f"cannot parse compiler response file {source_path}: {error}"
            ) from error
        response_records[str(source_path)] = {
            "path": str(source_path),
            "size_bytes": len(response_bytes),
            "sha256": sha256_bytes(response_bytes),
        }
        nested, nested_records = _expand_response_files(
            response_arguments,
            source_path.parent,
            resolver=resolver,
            depth=depth + 1,
            stack=stack + (source_path,),
        )
        expanded.extend(nested)
        for nested_record in nested_records:
            response_records[nested_record["path"]] = nested_record
    return expanded, [response_records[key] for key in sorted(response_records)]


def _language_for(source: Path, arguments: list[str]) -> str:
    explicit_language = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-x" and index + 1 < len(arguments):
            explicit_language = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("-x") and len(argument) > 2:
            explicit_language = argument[2:]
        index += 1
    if explicit_language is not None:
        language = explicit_language.lower()
        if language in {"assembler", "assembler-with-cpp", "asm"}:
            return "asm"
        if language in {"c", "c-header", "objective-c"}:
            return "c"
        if language in {"c++", "c++-header", "objective-c++", "cuda"}:
            return "c++"
        raise InventoryError(f"unsupported compile language {explicit_language!r}: {source}")
    suffix = source.suffix
    if suffix in _ASM_SOURCE_SUFFIXES:
        return "asm"
    if suffix in _C_SOURCE_SUFFIXES:
        return "c"
    if suffix in _CPP_SOURCE_SUFFIXES:
        return "c++"
    raise InventoryError(f"cannot classify compile_commands source language: {source}")


def _validate_compile_command(
    entry: object,
    index: int,
    resolver: Callable[[Path], Path | str]
    | Mapping[Path | str, Path | str]
    | None = None,
) -> tuple[str, dict[str, object]]:
    if not isinstance(entry, dict):
        raise InventoryError(f"compile_commands entry {index} is not an object")
    directory_text = entry.get("directory")
    source_text = entry.get("file")
    if not isinstance(directory_text, str) or not isinstance(source_text, str):
        raise InventoryError(f"compile_commands entry {index} lacks directory/file")
    directory = Path(directory_text).resolve(strict=False)
    source = Path(source_text)
    if not source.is_absolute():
        source = directory / source
    source = source.resolve(strict=False)

    arguments = entry.get("arguments")
    if arguments is not None:
        if not isinstance(arguments, list) or not all(
            isinstance(argument, str) for argument in arguments
        ):
            raise InventoryError(f"compile_commands entry {index} has invalid arguments")
        raw_arguments = list(arguments)
    else:
        command = entry.get("command")
        if not isinstance(command, str):
            raise InventoryError(f"compile_commands entry {index} lacks command/arguments")
        try:
            raw_arguments = shlex.split(command, posix=os.name != "nt")
        except ValueError as error:
            raise InventoryError(
                f"cannot parse compile_commands entry {index}: {error}"
            ) from error
    if not raw_arguments:
        raise InventoryError(f"compile_commands entry {index} has an empty command")
    expanded_arguments, response_files = _expand_response_files(
        raw_arguments, directory, resolver=resolver
    )
    language = _language_for(source, expanded_arguments)

    pic_enabled = False
    sanitizer_set: set[str] = set()
    no_recover_set: set[str] = set()
    for argument in expanded_arguments:
        if argument.startswith("-fno-sanitize="):
            raise InventoryError(
                f"compile command disables sanitizers in entry {index}: {argument}"
            )
        if argument == "-fsanitize-recover" or argument.startswith(
            "-fsanitize-recover="
        ):
            raise InventoryError(
                f"compile command enables sanitizer recovery in entry {index}: {argument}"
            )
        if argument.startswith("-fsanitize="):
            sanitizer_set.update(
                value.strip().lower()
                for value in argument.partition("=")[2].split(",")
                if value.strip()
            )
        elif argument == "-fno-sanitize-recover":
            no_recover_set.add("all")
        elif argument.startswith("-fno-sanitize-recover="):
            no_recover_set.update(
                value.strip().lower()
                for value in argument.partition("=")[2].split(",")
                if value.strip()
            )
        if argument in {"-fPIC", "-fpic"}:
            pic_enabled = True
        elif argument in {"-fno-PIC", "-fno-pic"}:
            pic_enabled = False
    if not pic_enabled:
        raise InventoryError(f"compile command is not PIC in entry {index}: {source}")
    if language != "asm":
        if not _SANITIZERS_REQUIRED.issubset(sanitizer_set):
            missing = sorted(_SANITIZERS_REQUIRED - sanitizer_set)
            raise InventoryError(
                f"compile command lacks address+undefined sanitizers in entry {index}: {missing}"
            )
        no_recover_all = "all" in no_recover_set
        if not no_recover_all and not _SANITIZERS_REQUIRED.issubset(no_recover_set):
            missing = sorted(_SANITIZERS_REQUIRED - no_recover_set)
            raise InventoryError(
                f"compile command lacks no-recover for required sanitizers in entry {index}: {missing}"
            )
    elif sanitizer_set:
        # Assembly sources must remain visible as a separate class; sanitizer
        # flags on an ASM command are accepted but are not evidence of C/C++ coverage.
        pass

    record = {
        "index": index,
        "directory": str(directory),
        "source": str(source),
        "language": language,
        "arguments": expanded_arguments,
        "pic": pic_enabled,
        "sanitizers": sorted(sanitizer_set),
        "no_sanitize_recover": sorted(no_recover_set),
        "response_files": response_files,
    }
    return language, record


def _export_source(
    export_file: Path, install_prefix: Path, resolved_prefix: Path
) -> tuple[str, Path]:
    for prefix in dict.fromkeys((resolved_prefix, install_prefix)):
        try:
            relative = export_file.resolve().relative_to(prefix.resolve())
        except ValueError:
            continue
        parts = relative.parts
        for index in range(1, len(parts) - 1):
            if parts[index - 1].startswith("lib") and parts[index] == "cmake":
                package = parts[index + 1] if index + 1 < len(parts) else ""
                if package in {"llvm", "mlir"}:
                    return package, relative
    raise InventoryError(
        f"CMake export is not from installed LLVM/MLIR config source: {export_file}"
    )


def _parse_cmake_exports(
    export_files: list[Path],
    install_prefix: Path,
    resolved_prefix: Path | None = None,
) -> tuple[list[dict[str, str]], set[Path]]:
    resolved_prefix = install_prefix if resolved_prefix is None else resolved_prefix
    target_locations: list[dict[str, str]] = []
    archive_paths: set[Path] = set()
    package_archive_counts = {"llvm": 0, "mlir": 0}
    block_pattern = re.compile(
        r"set_target_properties\(\s*([^\s)]+)\s+PROPERTIES\s+(.*?)\)", re.DOTALL
    )
    location_pattern = re.compile(
        r"\b(IMPORTED_LOCATION(?:_[A-Za-z0-9_]+)?)\s+(\"[^\"]*\"|'[^']*'|[^\s)]+)"
    )
    for export_file in sorted(export_files):
        package, relative_export = _export_source(
            export_file, install_prefix, resolved_prefix
        )
        source_export = install_prefix / relative_export
        text = _read_text(export_file, "installed CMake export")
        for block in block_pattern.finditer(text):
            target = block.group(1)
            properties = block.group(2)
            for property_name, raw_location in location_pattern.findall(properties):
                location = raw_location.strip("\"'")
                location = location.replace("${_IMPORT_PREFIX}", str(install_prefix))
                location = location.replace("${PACKAGE_PREFIX_DIR}", str(install_prefix))
                location = location.replace("${CMAKE_INSTALL_PREFIX}", str(install_prefix))
                location = location.replace(
                    "${CMAKE_CURRENT_LIST_DIR}", str(source_export.parent)
                )
                if "$" in location:
                    raise InventoryError(
                        f"cannot normalize {property_name} for target {target}: {raw_location}"
                    )
                resolved = Path(location)
                if not resolved.is_absolute():
                    resolved = install_prefix / resolved
                resolved = resolved.resolve(strict=False)
                target_locations.append(
                    {
                        "package": package,
                        "export_file": relative_export.as_posix(),
                        "target": target,
                        "property": property_name,
                        "path": str(resolved),
                    }
                )
                if resolved.suffix == ".a":
                    archive_paths.add(resolved)
                    package_archive_counts[package] += 1
    if not target_locations:
        raise InventoryError("installed LLVM/MLIR exports contain no imported locations")
    for package, count in package_archive_counts.items():
        if count == 0:
            raise InventoryError(
                f"installed {package} exports contain no static archive targets"
            )
    return target_locations, archive_paths


def _manifest_archive_paths(path: Path, install_prefix: Path) -> tuple[list[str], set[Path]]:
    lines = [line.strip() for line in _read_text(path, "install manifest").splitlines()]
    paths: list[Path] = []
    for line in lines:
        if not line:
            continue
        item = Path(line)
        if not item.is_absolute():
            raise InventoryError(f"install manifest path is not absolute: {line}")
        item = item.resolve(strict=False)
        if re.fullmatch(r"lib(?:LLVM|MLIR|LTO).*\.so(?:\..*)?", item.name):
            raise InventoryError(f"LLVM/MLIR core shared library is forbidden: {item}")
        if item.suffix == ".a":
            try:
                item.relative_to(install_prefix)
            except ValueError as error:
                raise InventoryError(
                    f"static archive is outside CMAKE_INSTALL_PREFIX: {item}"
                ) from error
            paths.append(item)
    if len(paths) != len(set(paths)):
        raise InventoryError("install manifest repeats a static archive path")
    if not paths:
        raise InventoryError("install manifest contains no static archives")
    return sorted(str(path) for path in paths), set(paths)


def _validate_patch_records(
    patch_records: Iterable[Mapping[str, object]],
    resolver: Callable[[Path], Path | str]
    | Mapping[Path | str, Path | str]
    | None = None,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    has_affinity_patch = False
    for index, record in enumerate(patch_records):
        patch_id = record.get("id")
        patch_path_text = record.get("patch_path")
        patch_hash = record.get("patch_sha256")
        applied = record.get("applied")
        application_path_text = record.get("application_record_path")
        application_hash = record.get("application_record_sha256")
        if not all(
            isinstance(value, str) and value
            for value in (patch_id, patch_path_text, patch_hash, application_path_text, application_hash)
        ):
            raise InventoryError(f"applied patch record {index} lacks required hash fields")
        if patch_id in seen_ids:
            raise InventoryError(f"duplicate applied patch record: {patch_id}")
        seen_ids.add(patch_id)
        if applied is not True:
            raise InventoryError(f"patch is not recorded as applied: {patch_id}")
        if not _SHA256_RE.fullmatch(patch_hash) or not _SHA256_RE.fullmatch(application_hash):
            raise InventoryError(f"patch record has an invalid SHA-256: {patch_id}")
        patch_source_path = Path(patch_path_text).resolve(strict=False)
        application_source_path = Path(application_path_text).resolve(strict=False)
        patch_path = _resolve_path(patch_source_path, resolver)
        application_path = _resolve_path(application_source_path, resolver)
        if not patch_path.is_file() or sha256_file(patch_path) != patch_hash:
            raise InventoryError(f"patch bytes do not match applied record: {patch_id}")
        if not application_path.is_file() or sha256_file(application_path) != application_hash:
            raise InventoryError(f"patch application record hash is invalid: {patch_id}")
        try:
            application = json.loads(application_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InventoryError(
                f"cannot parse patch application record for {patch_id}: {error}"
            ) from error
        if (
            not isinstance(application, dict)
            or application.get("applied") is not True
            or application.get("patch_id") != patch_id
            or application.get("patch_sha256") != patch_hash
        ):
            raise InventoryError(
                f"application record does not attest the applied patch: {patch_id}"
            )
        if "affinity" in patch_id.lower():
            has_affinity_patch = True
        if "ubsan" in patch_id.lower():
            smoke_path_text = record.get("smoke_evidence_path")
            smoke_hash = record.get("smoke_evidence_sha256")
            if not isinstance(smoke_path_text, str) or not isinstance(smoke_hash, str):
                raise InventoryError(
                    f"UBSan patch has no actual smoke evidence record: {patch_id}"
                )
            smoke_source_path = Path(smoke_path_text).resolve(strict=False)
            smoke_path = _resolve_path(smoke_source_path, resolver)
            if (
                not _SHA256_RE.fullmatch(smoke_hash)
                or not smoke_path.is_file()
                or sha256_file(smoke_path) != smoke_hash
            ):
                raise InventoryError(
                    f"UBSan patch smoke evidence hash is invalid: {patch_id}"
                )
        normalized.append(
            {
                "id": patch_id,
                "patch_path": str(patch_source_path),
                "patch_sha256": patch_hash,
                "application_record_path": str(application_source_path),
                "application_record_sha256": application_hash,
                "applied": True,
                **(
                    {
                        "smoke_evidence_path": str(smoke_source_path),
                        "smoke_evidence_sha256": record["smoke_evidence_sha256"],
                    }
                    if "ubsan" in patch_id.lower()
                    else {}
                ),
            }
        )
    if not has_affinity_patch:
        raise InventoryError("applied patch inventory lacks the real affinity fix record")
    return sorted(normalized, key=lambda item: str(item["id"]))


def validate_inventory(
    compile_commands_path: Path | str,
    cmake_cache_path: Path | str,
    llvm_config_path: Path | str,
    mlir_config_path: Path | str,
    export_files: Iterable[Path | str],
    install_manifest_path: Path | str,
    *,
    source_archive_path: Path | str,
    source_archive_sha256: str,
    resolver: Callable[[Path], Path | str]
    | Mapping[Path | str, Path | str]
    | None = None,
    expected_version: str = SUPPORTED_LLVM_MLIR_VERSION,
    applied_patch_records: Iterable[Mapping[str, object]] = (),
    expected_identity_sha256: str | None = None,
) -> dict[str, object]:
    """Validate and return a normalized, hash-bound installed toolchain inventory.

    ``source_archive_sha256`` is the independently recorded hash of the official
    LLVM source archive. ``resolver`` maps original absolute install paths (and
    response files) to live or staged paths. When validating a staged inventory,
    pass ``expected_identity_sha256`` from the source inventory.
    """
    compile_commands_path = Path(compile_commands_path).resolve()
    cmake_cache_path = Path(cmake_cache_path).resolve()
    llvm_config_path = Path(llvm_config_path).resolve()
    mlir_config_path = Path(mlir_config_path).resolve()
    install_manifest_path = Path(install_manifest_path).resolve()
    export_paths = sorted({Path(path).resolve() for path in export_files})
    if not export_paths:
        raise InventoryError("installed LLVM/MLIR export file list is empty")

    cache = _parse_cmake_cache(cmake_cache_path)
    prefix_text = _cache_value(cache, "CMAKE_INSTALL_PREFIX")
    if not prefix_text:
        raise InventoryError("CMakeCache.txt lacks CMAKE_INSTALL_PREFIX")
    install_prefix = Path(prefix_text).resolve(strict=False)
    resolved_prefix = _resolve_path(install_prefix, resolver)
    expected_llvm_config = _resolve_path(
        install_prefix / "lib/cmake/llvm/LLVMConfig.cmake", resolver
    )
    expected_mlir_config = _resolve_path(
        install_prefix / "lib/cmake/mlir/MLIRConfig.cmake", resolver
    )
    if llvm_config_path != expected_llvm_config:
        raise InventoryError("LLVMConfig.cmake is not from CMAKE_INSTALL_PREFIX")
    if mlir_config_path != expected_mlir_config:
        raise InventoryError("MLIRConfig.cmake is not from CMAKE_INSTALL_PREFIX")
    for key, config_path, filename in (
        ("LLVM_DIR", llvm_config_path, "LLVMConfig.cmake"),
        ("MLIR_DIR", mlir_config_path, "MLIRConfig.cmake"),
    ):
        cached_directory = _cache_value(cache, key)
        if cached_directory:
            expected_config = _resolve_path(
                Path(cached_directory) / filename, resolver
            )
            if expected_config != config_path:
                raise InventoryError(f"{filename} does not match CMakeCache {key}")

    llvm_version = _parse_version(llvm_config_path, "LLVM_PACKAGE_VERSION")
    mlir_llvm_version = _parse_mlir_llvm_contract(
        mlir_config_path, llvm_config_path, expected_version
    )
    if llvm_version != expected_version or mlir_llvm_version != llvm_version:
        raise InventoryError(
            "MLIR must require the exact same-prefix LLVM package version; "
            f"LLVM={llvm_version}, MLIR requires LLVM={mlir_llvm_version}"
        )
    source_archive_path = Path(source_archive_path).resolve()
    if not _SHA256_RE.fullmatch(source_archive_sha256):
        raise InventoryError("official LLVM source archive hash is invalid")
    if not source_archive_path.is_file() or sha256_file(source_archive_path) != source_archive_sha256:
        raise InventoryError("official LLVM source archive bytes do not match their hash")
    source_dir = _validate_source_archive_binding(
        cache, source_archive_path, expected_version
    )
    for key in (
        "BUILD_SHARED_LIBS",
        "LLVM_LINK_LLVM_DYLIB",
        "LLVM_BUILD_LLVM_DYLIB",
        "LLVM_BUILD_LLVM_C_DYLIB",
        "MLIR_BUILD_MLIR_C_DYLIB",
    ):
        value = _cache_value(cache, key)
        if value is not None and value.upper() in {"1", "ON", "YES", "TRUE"}:
            raise InventoryError(f"CMakeCache enables forbidden core DSO: {key}={value}")

    try:
        compile_bytes = compile_commands_path.read_bytes()
        compile_entries = json.loads(compile_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot parse full compile_commands.json: {error}") from error
    if not isinstance(compile_entries, list) or not compile_entries:
        raise InventoryError("compile_commands.json is not a nonempty command array")
    c_cpp_commands: list[dict[str, object]] = []
    asm_commands: list[dict[str, object]] = []
    response_inputs: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(compile_entries):
        language, record = _validate_compile_command(entry, index, resolver)
        if language == "asm":
            asm_commands.append(record)
        else:
            c_cpp_commands.append(record)
        for response_file in record["response_files"]:
            response_inputs[response_file["path"]] = response_file
    if not c_cpp_commands:
        raise InventoryError("compile_commands.json contains no C/C++ compile commands")

    if not resolved_prefix.is_dir():
        raise InventoryError(f"resolved installed toolchain prefix is missing: {resolved_prefix}")
    target_locations, export_archive_paths = _parse_cmake_exports(
        export_paths, install_prefix, resolved_prefix
    )
    for target_location in target_locations:
        if re.fullmatch(
            r"lib(?:LLVM|MLIR|LTO).*\.so(?:\..*)?",
            Path(target_location["path"]).name,
        ):
            raise InventoryError(
                "LLVM/MLIR core shared library is forbidden: "
                f"{target_location['path']} (target {target_location['target']})"
            )
    manifest_paths, manifest_archive_paths = _manifest_archive_paths(
        install_manifest_path, install_prefix
    )
    if export_archive_paths != manifest_archive_paths:
        missing_from_manifest = sorted(map(str, export_archive_paths - manifest_archive_paths))
        missing_from_exports = sorted(map(str, manifest_archive_paths - export_archive_paths))
        raise InventoryError(
            "installed static archive sets differ between CMake exports and install manifest; "
            f"missing_from_manifest={missing_from_manifest}, "
            f"missing_from_exports={missing_from_exports}"
        )

    forbidden_core_dsos = []
    for pattern in ("libLLVM*.so*", "libMLIR*.so*", "libLTO.so*"):
        forbidden_core_dsos.extend(resolved_prefix.rglob(pattern))
    if forbidden_core_dsos:
        raise InventoryError(
            f"installed LLVM/MLIR core shared libraries are forbidden: {forbidden_core_dsos[:5]}"
        )
    actual_archive_paths = {
        path.resolve()
        for path in resolved_prefix.rglob("*.a")
        if path.is_file()
    }
    resolved_export_archive_paths = {
        _resolve_path(path, resolver) for path in export_archive_paths
    }
    if actual_archive_paths != resolved_export_archive_paths:
        missing_exports = sorted(map(str, actual_archive_paths - resolved_export_archive_paths))
        missing_files = sorted(map(str, resolved_export_archive_paths - actual_archive_paths))
        raise InventoryError(
            "installed archive files differ from exported target inventory; "
            f"unexported_archives={missing_exports}, missing_archives={missing_files}"
        )

    archive_records = []
    bundle_files = []
    for source_path in sorted(export_archive_paths):
        bundled_path = _resolve_path(source_path, resolver)
        if not bundled_path.is_file():
            raise InventoryError(f"installed static archive is missing: {bundled_path}")
        source_exists = source_path.is_file()
        digest = sha256_file(bundled_path)
        if source_exists and bundled_path != source_path and sha256_file(source_path) != digest:
            raise InventoryError(
                f"staged archive differs from source bytes: {source_path} -> {bundled_path}"
            )
        try:
            prefix_relative = source_path.relative_to(install_prefix).as_posix()
        except ValueError as error:
            raise InventoryError(
                f"exported static archive is outside CMAKE_INSTALL_PREFIX: {source_path}"
            ) from error
        record = {
            "source_path": source_path.as_posix(),
            "install_relative_path": prefix_relative,
            "size_bytes": bundled_path.stat().st_size,
            "sha256": digest,
        }
        archive_records.append(record)
        bundle_files.append(
            {
                "source": source_path.as_posix(),
                "source_sha256": digest,
                "path": bundled_path.as_posix(),
                "sha256": digest,
                "size_bytes": record["size_bytes"],
                "kind": "static-archive",
            }
        )
    for source_text, response_record in sorted(response_inputs.items()):
        source_path = Path(source_text)
        bundled_path = _resolve_path(source_path, resolver)
        if not bundled_path.is_file():
            raise InventoryError(f"compiler response file is missing: {source_path}")
        digest = sha256_file(bundled_path)
        if digest != response_record["sha256"]:
            raise InventoryError(f"compiler response file hash changed: {source_path}")
        bundle_files.append(
            {
                "source": source_path.as_posix(),
                "source_sha256": digest,
                "path": bundled_path.as_posix(),
                "sha256": digest,
                "size_bytes": bundled_path.stat().st_size,
                "kind": "compiler-response-file",
            }
        )

    patch_records = _validate_patch_records(applied_patch_records, resolver)
    source_inputs = {
        "source_archive": {
            "size_bytes": source_archive_path.stat().st_size,
            "sha256": source_archive_sha256,
        },
        "compile_commands": _hash_input(compile_commands_path, "compile_commands.json"),
        "cmake_cache": _hash_input(cmake_cache_path, "CMakeCache.txt"),
        "llvm_config": _hash_input(llvm_config_path, "LLVMConfig.cmake"),
        "mlir_config": _hash_input(mlir_config_path, "MLIRConfig.cmake"),
        "cmake_exports": [
            _hash_input(path, "installed CMake export") for path in export_paths
        ],
        "install_manifest": _hash_input(install_manifest_path, "install manifest"),
        "response_files": [response_inputs[key] for key in sorted(response_inputs)],
    }
    normalized_inventory: dict[str, object] = {
        "schema": "ncnn-san-toolchain-inventory/v2",
        "schema_version": SCHEMA_VERSION,
        "toolchain": {
            "llvm_package_version": llvm_version,
            "mlir_requires_llvm_version": mlir_llvm_version,
            "mlir_requires_exact_same_prefix_llvm": True,
            "official_source_tree": source_dir.as_posix(),
            "cmake_install_prefix": install_prefix.as_posix(),
            "static_llvm_core_dso_forbidden": True,
        },
        "inputs": source_inputs,
        "compile_commands": {
            "total": len(compile_entries),
            "c_cpp_count": len(c_cpp_commands),
            "asm_count": len(asm_commands),
            "c_cpp": c_cpp_commands,
            "asm": asm_commands,
        },
        "exported_targets": target_locations,
        "static_archives": archive_records,
        "install_manifest_static_archives": manifest_paths,
        "applied_patches": patch_records,
    }
    identity_sha256 = _canonical_json_sha256(normalized_inventory)
    if expected_identity_sha256 is not None:
        if not _SHA256_RE.fullmatch(expected_identity_sha256):
            raise InventoryError("expected inventory identity is not a SHA-256 digest")
        if identity_sha256 != expected_identity_sha256:
            raise InventoryError(
                "normalized toolchain inventory identity differs from expected source identity"
            )
    return {
        "inventory": normalized_inventory,
        "identity_sha256": identity_sha256,
        "bundle_files": bundle_files,
    }
