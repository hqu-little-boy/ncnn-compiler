#!/usr/bin/env python3
"""Audit and bundle the matching instrumented LLVM/MLIR prerequisite."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(stage, path):
    path = Path(path)
    return {"path": str(path.relative_to(stage)), "sha256": sha256(path)}


def _checked(path, expected_hash, description):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"{description} hash mismatch: {path}")
    return path


def audit_and_bundle(stage, identity_path, install, lit, openmp_dir, clang_dir):
    stage = Path(stage).resolve()
    identity_path = Path(identity_path).resolve()
    if not identity_path.is_file():
        raise RuntimeError(f"sanitizer toolchain identity is missing: {identity_path}")
    identity_hash = sha256(identity_path)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity_schema = identity.get("schema")
    if (identity_schema not in {"ncnn-san-toolchain-identity/v1",
                                "ncnn-san-toolchain-identity/v2"} or
            identity.get("status") != "verified"):
        raise RuntimeError("sanitizer toolchain identity is not verified")
    is_v2 = identity_schema == "ncnn-san-toolchain-identity/v2"

    llvm = identity.get("llvm", {})
    inventory_result = None
    inventory_inputs = None
    if is_v2:
        from sanitizer_toolchain_inventory import InventoryError, validate_inventory
        inventory_record = llvm.get("inventory", {})
        if not isinstance(inventory_record, dict):
            raise RuntimeError("v2 LLVM inventory record is malformed")
        inventory_inputs = inventory_record.get("inputs", {})
        required_inputs = ("compile_commands", "cmake_cache", "llvm_config",
                           "mlir_config", "cmake_exports", "install_manifest",
                           "source_archive", "applied_patches")
        if (not isinstance(inventory_record, dict) or
                not isinstance(inventory_inputs, dict) or
                any(key not in inventory_inputs for key in required_inputs)):
            raise RuntimeError("v2 LLVM inventory inputs are incomplete")
        source_archive_record = inventory_inputs["source_archive"]
        if not isinstance(source_archive_record, dict):
            raise RuntimeError("v2 LLVM source archive inventory is malformed")
        try:
            inventory_result = validate_inventory(
                inventory_inputs["compile_commands"], inventory_inputs["cmake_cache"],
                inventory_inputs["llvm_config"], inventory_inputs["mlir_config"],
                inventory_inputs["cmake_exports"], inventory_inputs["install_manifest"],
                source_archive_path=source_archive_record.get("path", ""),
                source_archive_sha256=source_archive_record.get("sha256", ""),
                applied_patch_records=inventory_inputs["applied_patches"],
                expected_identity_sha256=inventory_record.get("identity_sha256"),
            )
        except (InventoryError, OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"v2 LLVM inventory validation failed: {error}") from error
    configuration = llvm.get("configuration", {})
    if (llvm.get("version") != "21.1.8" or
            llvm.get("shared_mode") != "static" or
            Path(llvm.get("install_prefix", "")).resolve() != Path(install).resolve() or
            configuration.get("BUILD_SHARED_LIBS") is not False or
            configuration.get("LLVM_BUILD_LLVM_DYLIB") is not False or
            configuration.get("LLVM_BUILD_LLVM_C_DYLIB") is not False or
            configuration.get("LLVM_LINK_LLVM_DYLIB") is not False or
            configuration.get("MLIR_BUILD_MLIR_C_DYLIB") is not False or
            configuration.get("sanitizers") != ["address", "undefined"] or
            configuration.get("no_sanitize_recover") is not True or
            "-fPIC" not in configuration.get("flags", "")):
        raise RuntimeError("LLVM/MLIR toolchain configuration violates the static PIC ASan+UBSan gate")
    libomp = identity.get("libomp", {})
    libomp_configuration = libomp.get("configuration", {})
    libomp_path = Path(libomp.get("shared_library", {}).get("path", "")).resolve()
    if (libomp_path != (Path(openmp_dir).resolve() / "libomp.so") or
            libomp.get("static_library") is not None or
            libomp.get("libgomp_aliases") != [] or
            libomp_configuration.get("sanitizers") != ["address", "undefined"] or
            libomp_configuration.get("no_sanitize_recover") is not True or
            libomp_configuration.get("LIBOMP_OMPD_SUPPORT") is not False):
        raise RuntimeError("isolated libomp does not match the verified ASan+UBSan profile")
    clang = llvm.get("compiler", {}).get("path")
    clangxx = llvm.get("compiler_cxx", {}).get("path")
    if (Path(clang).resolve() != (Path(clang_dir).resolve() / "clang").resolve() or
            Path(clangxx).resolve() != (Path(clang_dir).resolve() / "clang++").resolve()):
        raise RuntimeError("sanitizer compiler paths do not match the verified identity")
    if Path(lit).resolve() != Path(llvm.get("lit_runner", {}).get("path", "")).resolve():
        raise RuntimeError("lit runner does not match the verified LLVM source tree")

    verification = Path(identity.get("smokes", {}).get("verification_status", ""))
    if not verification.is_file():
        raise RuntimeError("sanitizer verifier status file is missing")
    verification_text = verification.read_text(encoding="utf-8")
    if "verification=passed" not in verification_text or "exit=0" not in verification_text:
        raise RuntimeError("sanitizer verification status does not record a successful exit")

    source_identity = identity.get("source_archive_identity", {})
    checked_source_archive = _checked(source_identity.get("path", ""),
                                      source_identity.get("required_sha256"), "LLVM source archive")
    if inventory_result is not None:
        inventory_source = inventory_inputs["source_archive"]
        if (Path(inventory_source["path"]).resolve() != checked_source_archive or
                inventory_source["sha256"] != source_identity.get("required_sha256")):
            raise RuntimeError("v2 inventory source archive differs from the verified identity archive")
    for role, artifact in identity.get("tool_artifacts", {}).items():
        _checked(artifact.get("path", ""), artifact.get("sha256"), role)
    _checked(libomp_path, libomp.get("shared_library", {}).get("sha256"), "libomp")

    archive_records = llvm.get("static_archive_artifacts", [])
    expected_archive_count = (
        len(inventory_result["inventory"]["static_archives"])
        if inventory_result is not None else 486
    )
    if (len(archive_records) != llvm.get("static_archive_count") or
            len(archive_records) != expected_archive_count):
        raise RuntimeError("verified static LLVM/MLIR archive inventory is incomplete")
    if inventory_result is not None:
        inventory_archives = {
            str(Path(record["source_path"]).resolve()): record["sha256"]
            for record in inventory_result["inventory"]["static_archives"]
        }
        identity_archives = {
            str(Path(record.get("path", "")).resolve()): record.get("sha256")
            for record in archive_records if isinstance(record, dict)
        }
        if identity_archives != inventory_archives:
            raise RuntimeError("legacy archive list differs from the v2 content-bound inventory")
    for index, record in enumerate(archive_records, 1):
        archive = _checked(record.get("path", ""), record.get("sha256"),
                           f"static LLVM/MLIR archive {index}")
        if archive.suffix != ".a":
            raise RuntimeError(f"non-static archive in toolchain inventory: {archive}")
    for name, path in identity.get("log_files", {}).items():
        _checked(path, identity.get("log_sha256", {}).get(name),
                 f"toolchain log {name}")
    for mapping_key in ("status_files", "configuration_files"):
        for path, expected_hash in identity.get(mapping_key, {}).items():
            _checked(path, expected_hash, f"toolchain {mapping_key} entry {path}")
    archive_list = llvm.get("static_archive_list", {})
    _checked(archive_list.get("path", ""), archive_list.get("sha256"),
             "static archive list")
    asan_runtime = llvm.get("shared_asan_runtime_identity", {})
    _checked(asan_runtime.get("path", ""), asan_runtime.get("sha256"),
             "compiler ASan runtime")
    _checked(verification, sha256(verification), "successful verifier status")

    install_prefix = Path(install).resolve()
    dso_audit = Path(llvm.get("core_shared_dso_audit", "")).resolve()
    if not dso_audit.is_file() or dso_audit.stat().st_size != 0:
        raise RuntimeError("core LLVM/MLIR shared-object audit is missing or non-empty")
    forbidden = list(install_prefix.rglob("libLLVM*.so*"))
    forbidden += list(install_prefix.rglob("libMLIR*.so*"))
    forbidden += list(install_prefix.rglob("libLTO.so*"))
    if forbidden:
        raise RuntimeError(f"shared core LLVM/MLIR/LTO libraries are present: {forbidden[:3]}")
    compile_audit = Path(llvm.get("compile_pic_sanitizer_audit", "")).read_text(
        encoding="utf-8")
    count_records = re.findall(
        r"^compile_commands=(\d+); sanitizer_instrumented_sources=(\d+); pic_sources=(\d+)$",
        compile_audit, re.MULTILINE)
    expected_compile_record = (
        str(inventory_result["inventory"]["compile_commands"]["total"]),
        str(inventory_result["inventory"]["compile_commands"]["c_cpp_count"]),
        str(inventory_result["inventory"]["compile_commands"]["total"]),
    ) if inventory_result is not None else ("4316", "4312", "4316")
    if count_records != [expected_compile_record]:
        raise RuntimeError("instrumentation/PIC audit coverage is incomplete or malformed")
    for field in ("mlir_ir_asan_symbol_audit", "mlir_ir_ubsan_symbol_audit"):
        symbol_log = Path(llvm[field])
        if not symbol_log.is_file() or symbol_log.stat().st_size == 0:
            raise RuntimeError(f"instrumented static archive lacks {field}")
    pic_probe = Path(identity["root"]) / "smoke/libMLIRIR-pic.so"
    if not pic_probe.is_file():
        raise RuntimeError("static archive PIC shared-link smoke artifact is missing")
    relocation_audit = Path(llvm.get("mlir_ir_pic_relocation_audit", "")).read_text(
        encoding="utf-8")
    if "TEXTREL" in relocation_audit:
        raise RuntimeError("static archive PIC link contains text relocations")

    gpu_smoke = identity.get("smokes", {}).get("gpu_runner_pipeline_fixture", {})
    fixture = _checked(gpu_smoke.get("path", ""), gpu_smoke.get("sha256"),
                       "GPU runner pipeline fixture")
    gpu_output = identity.get("smokes", {}).get("gpu_runner_pipeline_output", {})
    lowered = _checked(gpu_output.get("path", ""), gpu_output.get("sha256"),
                       "instrumented GPU runner pipeline output")
    lowered_text = lowered.read_text(encoding="utf-8")
    if "gpu.binary @v0_add" not in lowered_text or re.search(
            r"gpu\.func|memref\.load|memref\.store|unrealized_conversion_cast",
            lowered_text):
        raise RuntimeError("instrumented MLIR Vulkan runner pipeline smoke did not verify")
    mlir_opt = Path(llvm.get("tools", {}).get("mlir_opt", "")).resolve()
    mlir_translate = Path(llvm.get("tools", {}).get("mlir_translate", "")).resolve()
    filecheck = Path(llvm.get("tools", {}).get("FileCheck", "")).resolve()
    for executable in (mlir_opt, mlir_translate, filecheck):
        dynamic_log = subprocess.check_output(["readelf", "-d", str(executable)],
                                              text=True, stderr=subprocess.STDOUT)
        if re.search(r"NEEDED.*lib(?:LLVM|MLIR|LTO)", dynamic_log):
            raise RuntimeError(f"tool has a shared core dependency: {executable}")
    help_text = subprocess.check_output([str(mlir_opt), "--help"], text=True,
                                        stderr=subprocess.STDOUT)
    if "--test-vulkan-runner-pipeline" not in help_text:
        raise RuntimeError("verified mlir-opt lacks the Vulkan runner pipeline")

    omp_smoke = Path(libomp.get("lsan_smoke", {}).get("binary", ""))
    if not omp_smoke.is_file():
        raise RuntimeError("instrumented libomp LSan smoke binary is missing")
    lsan_log = Path(libomp.get("lsan_smoke", {}).get("log", ""))
    if (not lsan_log.is_file() or
            re.search(r"AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|runtime error:",
                      lsan_log.read_text(encoding="utf-8", errors="replace"))):
        raise RuntimeError("instrumented libomp LSan smoke log is missing or has diagnostics")
    link_log = identity.get("log_files", {}).get("libomp-lsan-link-audit.log")
    if not link_log:
        raise RuntimeError("isolated libomp smoke link audit is missing")
    link_text = Path(link_log).read_text(encoding="utf-8", errors="replace")
    if (str(libomp_path) not in link_text or "libclang_rt.asan" not in link_text or
            "libgomp" in link_text):
        raise RuntimeError("libomp smoke does not resolve the isolated library and shared ASan")
    if libomp.get("lsan_smoke", {}).get("leak_detection") is not True or \
            libomp.get("lsan_smoke", {}).get("suppressions") is not False:
        raise RuntimeError("libomp LSan smoke disabled leak checking or used suppressions")
    omp_prefix = Path(openmp_dir).resolve().parent
    aliases = list(omp_prefix.rglob("libgomp*"))
    if aliases:
        raise RuntimeError(f"isolated libomp prefix has libgomp aliases: {aliases}")

    bundle = stage / "toolchain-evidence"
    log_dir = bundle / "logs"
    log_dir.mkdir(parents=True)
    identity_copy = bundle / "toolchain-identity.json"
    shutil.copyfile(identity_path, identity_copy)
    copied = [{"source": str(identity_path), **_artifact(stage, identity_copy),
               "source_sha256": identity_hash}]
    verification_copy = bundle / "status" / "verification.status"
    verification_copy.parent.mkdir(parents=True)
    shutil.copyfile(verification, verification_copy)
    copied.append({"source": str(verification), **_artifact(stage, verification_copy),
                   "source_sha256": sha256(verification)})
    for mapping_key in ("log_files", "status_files", "configuration_files"):
        records = identity.get(mapping_key, {})
        hashes = identity.get("log_sha256", {}) if mapping_key == "log_files" else None
        for name, value in records.items():
            if mapping_key == "log_files":
                source = Path(value).resolve()
                expected = hashes[name]
            else:
                source, expected = Path(name).resolve(), value
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name)
            safe_name = hashlib.sha256(str(source).encode()).hexdigest()[:12] + "-" + safe_name
            target_dir = log_dir / mapping_key
            target_dir.mkdir(exist_ok=True)
            target = target_dir / safe_name
            if target.exists():
                if sha256(target) != expected:
                    raise RuntimeError(f"toolchain evidence bundle path collision: {target}")
            else:
                shutil.copyfile(source, target)
            copied.append({"source": str(source), **_artifact(stage, target),
                           "source_sha256": expected})
    for label, record in (("affinity-leak", libomp.get("source_patch", {})),
                          ("affinity-ubsan", libomp.get("ubsan_fix_patch", {}))):
        patch = _checked(record.get("path", ""), record.get("sha256"),
                         f"libomp {label} patch")
        target = bundle / "patches" / patch.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(patch, target)
        copied.append({"source": str(patch), **_artifact(stage, target),
                       "source_sha256": record["sha256"]})

    if inventory_result is not None:
        inventory = inventory_result["inventory"]
        input_records = inventory_inputs

        def bundle_inventory_file(source, relative, expected_hash, kind):
            source = Path(source).resolve()
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _checked(source, expected_hash, f"v2 inventory {kind}")
            if target.exists():
                if sha(target) != expected_hash:
                    raise RuntimeError(f"v2 inventory bundle collision: {target}")
            else:
                shutil.copyfile(source, target)
            copied.append({"source": str(source), **_artifact(stage, target),
                           "source_sha256": expected_hash, "kind": kind})
            return target

        source_archive = input_records["source_archive"]
        bundle_inventory_file(source_archive["path"],
                              Path("inventory/source") / Path(source_archive["path"]).name,
                              source_archive["sha256"], "llvm-source-archive")
        input_path_fields = ("compile_commands", "cmake_cache", "install_manifest")
        for field in input_path_fields:
            source = Path(input_records[field]).resolve()
            digest = inventory["inputs"][field]["sha256"]
            bundle_inventory_file(source, Path("inventory/inputs") / field / source.name,
                                  digest, f"inventory-{field}")

        prefix = Path(inventory["toolchain"]["cmake_install_prefix"])
        install_files = [(Path(input_records["llvm_config"]), "llvm_config"),
                         (Path(input_records["mlir_config"]), "mlir_config")]
        install_files.extend((Path(path), "cmake-export")
                             for path in input_records["cmake_exports"])
        for source, role in install_files:
            source = source.resolve()
            try:
                relative = source.relative_to(prefix)
            except ValueError as error:
                raise RuntimeError(f"v2 installed config/export escapes prefix: {source}") from error
            if role == "llvm_config":
                expected = inventory["inputs"]["llvm_config"]["sha256"]
            elif role == "mlir_config":
                expected = inventory["inputs"]["mlir_config"]["sha256"]
            else:
                export_index = [Path(path).resolve() for path in input_records["cmake_exports"]].index(source)
                expected = inventory["inputs"]["cmake_exports"][export_index]["sha256"]
            bundle_inventory_file(source, Path("llvm-install") / relative,
                                  expected, role)

        staged_archives = {}
        for record, helper_record in zip(archive_records, inventory["static_archives"]):
            source = Path(helper_record["source_path"])
            relative = Path("llvm-install") / helper_record["install_relative_path"]
            target = bundle_inventory_file(source, relative, helper_record["sha256"],
                                           "static-archive")
            staged_archives[str(source.resolve())] = target
        for response in inventory["inputs"]["response_files"]:
            source = Path(response["path"])
            bundle_inventory_file(source,
                                  Path("inventory/response-files") /
                                  (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "-" + source.name),
                                  response["sha256"], "compiler-response-file")
        patch_inputs = input_records["applied_patches"]
        for patch_record in patch_inputs:
            for path_key, hash_key, kind in (
                    ("patch_path", "patch_sha256", "applied-patch"),
                    ("application_record_path", "application_record_sha256", "patch-application-record"),
                    ("smoke_evidence_path", "smoke_evidence_sha256", "patch-smoke-evidence")):
                if path_key in patch_record:
                    source = Path(patch_record[path_key]).resolve()
                    bundle_inventory_file(source,
                        Path("inventory/patches") / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "-" + source.name),
                        patch_record[hash_key], kind)

        hash_by_source = {}
        for name, source in identity.get("log_files", {}).items():
            expected = identity.get("log_sha256", {}).get(name)
            if isinstance(expected, str):
                hash_by_source[str(Path(source).resolve())] = expected
        extra_records = [
            (llvm.get("core_shared_dso_audit"), "core-dso-audit"),
            (llvm.get("compile_pic_sanitizer_audit"), "compile-pic-sanitizer-audit"),
            (llvm.get("mlir_ir_asan_symbol_audit"), "asan-symbol-audit"),
            (llvm.get("mlir_ir_ubsan_symbol_audit"), "ubsan-symbol-audit"),
            (llvm.get("mlir_ir_pic_relocation_audit"), "pic-relocation-audit"),
            (Path(identity.get("root", "")) / "smoke/libMLIRIR-pic.so", "pic-link-smoke"),
            (libomp.get("lsan_smoke", {}).get("log"), "libomp-lsan-log"),
            (libomp.get("lsan_smoke", {}).get("binary"), "libomp-lsan-binary"),
        ]
        for smoke_key, role in (("gpu_runner_pipeline_fixture", "gpu-runner-fixture"),
                                ("gpu_runner_pipeline_output", "gpu-runner-output")):
            smoke_record = identity.get("smokes", {}).get(smoke_key, {})
            extra_records.append((smoke_record.get("path"), role))
            if isinstance(smoke_record.get("path"), str):
                hash_by_source[str(Path(smoke_record["path"]).resolve())] = smoke_record.get("sha256")
        for source_value, role in extra_records:
            if not source_value:
                continue
            source = Path(source_value).resolve()
            expected = hash_by_source.get(str(source))
            if expected is None:
                expected = sha256(source)
            bundle_inventory_file(source,
                Path("audit-records") / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "-" + source.name),
                expected, role)

        archive_list_record = llvm["static_archive_list"]
        archive_list_source = Path(archive_list_record["path"]).resolve()
        bundle_inventory_file(archive_list_source,
                              Path("inventory/inputs/static-archive-list.txt"),
                              archive_list_record["sha256"], "static-archive-list")
        for role, record in identity.get("tool_artifacts", {}).items():
            source = Path(record["path"]).resolve()
            safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", role)
            bundle_inventory_file(source,
                                  Path("tool-artifacts") / safe_role / source.name,
                                  record["sha256"], f"tool-binary:{role}")
        libomp_record = libomp["shared_library"]
        libomp_source = Path(libomp_record["path"]).resolve()
        bundle_inventory_file(libomp_source,
                              Path("runtime-libraries") / libomp_source.name,
                              libomp_record["sha256"], "isolated-libomp")
        asan_record = llvm["shared_asan_runtime_identity"]
        asan_source = Path(asan_record["path"]).resolve()
        bundle_inventory_file(asan_source,
                              Path("runtime-libraries") / asan_source.name,
                              asan_record["sha256"], "shared-asan-runtime")

    return {"identity": _artifact(stage, identity_copy),
            "files": copied,
            "static_archive_count": len(archive_records),
            "llvm_version": llvm["version"],
            "libomp_sha256": libomp["shared_library"]["sha256"],
            "verification_status": verification_text,
            **({"inventory_identity_sha256": inventory_result["identity_sha256"]}
               if inventory_result is not None else {})}
