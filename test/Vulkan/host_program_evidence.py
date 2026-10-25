#!/usr/bin/env python3
"""Validate compiler-only Vulkan host-program identities and planner traces."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any


SCHEMA = "ncnn-vulkan-host-program/v1"
CODEGEN = "ncnn-vulkan-host-codegen/v1"
HELPER_CONTRACT = "ncnn-vulkan-device-helper-contract/v1"
SYNC_VERSION = "ncnn-vulkan-host-sync/v1"
CANONICAL_ENCODING = "length-prefixed-canonical-json/v1"
IDENTITY_DOMAIN = b"ncnn-vulkan-host-program-canonical/v1"
OPTIONS = {
    "allocation": "independent-resources",
    "command_recording": "per-run",
    "mapped_sync_range": "whole-allocation",
    "output_publication": "after-all-invalidates-and-mapping-checks",
    "synchronize_same_context": True,
    "weight_upload": "once-per-context",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UINT64_MAX = (1 << 64) - 1


class EvidenceError(ValueError):
    """A malformed or inconsistent host-program evidence object."""


def _fail(message: str) -> None:
    raise EvidenceError(message)


def _integer(value: object, label: str, *, minimum: int = 0,
             maximum: int = UINT64_MAX) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        _fail(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _expect_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        _fail(f"{label} fields differ from the frozen schema")


def _u64le(value: int) -> bytes:
    if type(value) is not int or value < 0 or value > UINT64_MAX:
        _fail("canonical host-program encoding length is outside uint64")
    return struct.pack("<Q", value)


def canonical_program_bytes(program: dict[str, Any]) -> bytes:
    """Encode the program identity domain, excluding only its self identity."""
    identity_input = {key: value for key, value in program.items()
                      if key != "identity"}
    canonical = json.dumps(identity_input, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return (_u64le(len(IDENTITY_DOMAIN)) + IDENTITY_DOMAIN +
            _u64le(len(canonical)) + canonical)


def canonical_identity(program: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_program_bytes(program)).hexdigest()


def _resource_map(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    resources = _array(plan.get("resources"), "plan.resources")
    output = {}
    for resource in resources:
        resource = _object(resource, "plan resource")
        _expect_keys(resource, {
            "id", "kind", "name", "shape", "element_type", "logical_layout",
            "physical_layout", "host_layout", "layout_transform", "weight_sha256",
            "byte_size", "first_kernel", "last_kernel"}, "plan resource")
        identifier = _integer(resource.get("id"), "plan resource id", maximum=0xFFFFFFFF)
        if identifier in output:
            _fail(f"plan repeats resource id {identifier}")
        if resource.get("kind") not in {"input", "output", "intermediate", "weight"}:
            _fail(f"plan resource {identifier} has an unknown kind")
        for field in ("name", "logical_layout", "physical_layout", "host_layout"):
            if not isinstance(resource.get(field), str):
                _fail(f"plan resource {identifier} {field} must be a string")
        shape = _array(resource.get("shape"), f"resource {identifier} shape")
        for dimension in shape:
            _integer(dimension, f"resource {identifier} dimension")
        if resource.get("element_type") != "f32":
            _fail(f"plan resource {identifier} is not FP32")
        byte_size = _integer(resource.get("byte_size"),
                             f"resource {identifier} byte_size")
        expected_size = math.prod(shape) * 4
        if expected_size > UINT64_MAX or byte_size != expected_size:
            _fail(f"plan resource {identifier} byte size differs from its FP32 shape")
        transform = resource.get("layout_transform")
        if transform is not None:
            transform = _object(transform, f"resource {identifier} layout_transform")
            if set(transform) != {"from_layout", "to_layout", "operation"} or any(
                    not isinstance(transform.get(key), str) or not transform[key]
                    for key in transform):
                _fail(f"plan resource {identifier} layout transform is malformed")
        _integer(resource.get("first_kernel"), f"resource {identifier} first_kernel",
                 maximum=0xFFFFFFFF)
        _integer(resource.get("last_kernel"), f"resource {identifier} last_kernel",
                 maximum=0xFFFFFFFF)
        _sha(resource.get("weight_sha256"),
             f"resource {identifier} weight_sha256") if resource.get("kind") == "weight" else None
        output[identifier] = resource
    if not output:
        _fail("plan has no resource inventory")
    return output


def _plan_kernel_map(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    kernels = _array(plan.get("kernels"), "plan.kernels")
    output = {}
    for kernel in kernels:
        kernel = _object(kernel, "plan kernel")
        identifier = _integer(kernel.get("id"), "kernel id", maximum=0xFFFFFFFF)
        if identifier in output:
            _fail(f"plan repeats kernel id {identifier}")
        local = _array(kernel.get("local_size"), f"kernel {identifier} local_size")
        if len(local) != 3 or any(_integer(value, "local_size", minimum=1,
                                             maximum=0xFFFFFFFF) == 0 for value in local):
            _fail(f"kernel {identifier} local_size must contain three positive dimensions")
        for key in ("operation", "entry_point"):
            if not isinstance(kernel.get(key), str) or not kernel[key]:
                _fail(f"kernel {identifier} {key} must be a nonempty string")
        for key in ("reads", "writes", "bindings", "push_constants"):
            _array(kernel.get(key), f"kernel {identifier}.{key}")
        for field in ("reads", "writes"):
            for resource_id in kernel[field]:
                _integer(resource_id, f"kernel {identifier}.{field} resource",
                         maximum=0xFFFFFFFF)
        seen_bindings = set()
        for binding in kernel["bindings"]:
            binding = _object(binding, f"kernel {identifier} binding")
            _expect_keys(binding, {"set", "binding", "resource_id", "access"},
                         f"kernel {identifier} binding")
            descriptor_set = _integer(binding.get("set"), "plan descriptor set",
                                      maximum=0xFFFFFFFF)
            descriptor_binding = _integer(binding.get("binding"), "plan descriptor binding",
                                           maximum=0xFFFFFFFF)
            _integer(binding.get("resource_id"), "plan binding resource_id",
                     maximum=0xFFFFFFFF)
            if binding.get("access") not in {"read", "write"}:
                _fail(f"kernel {identifier} binding has unknown access")
            key = (descriptor_set, descriptor_binding)
            if key in seen_bindings:
                _fail(f"kernel {identifier} repeats a descriptor binding")
            seen_bindings.add(key)
        for push in kernel["push_constants"]:
            push = _object(push, f"kernel {identifier} push constant")
            _expect_keys(push, {"offset", "data_hex"},
                         f"kernel {identifier} push constant")
            _integer(push.get("offset"), "plan push-constant offset",
                     maximum=0xFFFFFFFF)
            payload = push.get("data_hex")
            if (not isinstance(payload, str) or not payload or len(payload) % 2 or
                    re.fullmatch(r"[0-9a-f]*", payload) is None):
                _fail("plan push constant data_hex must be nonempty lowercase hexadecimal")
        output[identifier] = kernel
    if not output:
        _fail("plan has no kernel inventory")
    return output


def _buffer_map(program: dict[str, Any], resources: dict[int, dict[str, Any]]) -> tuple[dict[int, dict], dict[int, int]]:
    buffers = _array(program.get("buffers"), "host_program.buffers")
    by_id = {}
    device_by_resource = {}
    for buffer in buffers:
        buffer = _object(buffer, "host buffer")
        _expect_keys(buffer, {"id", "resource_id", "kind", "scope", "byte_size"},
                     "host buffer")
        identifier = _integer(buffer.get("id"), "host buffer id", maximum=0xFFFFFFFF)
        resource_id = _integer(buffer.get("resource_id"), "host buffer resource_id",
                               maximum=0xFFFFFFFF)
        if identifier in by_id:
            _fail(f"host program repeats buffer id {identifier}")
        if resource_id not in resources:
            _fail(f"host buffer {identifier} references unknown resource {resource_id}")
        if buffer.get("kind") not in {"device", "input-staging", "output-staging", "weight-staging"}:
            _fail(f"host buffer {identifier} has unknown kind")
        if buffer.get("scope") not in {"context", "create-upload"}:
            _fail(f"host buffer {identifier} has unknown scope")
        byte_size = _integer(buffer.get("byte_size"), "host buffer byte_size")
        if byte_size != resources[resource_id]["byte_size"]:
            _fail(f"host buffer {identifier} size differs from resource {resource_id}")
        by_id[identifier] = buffer
        if buffer["kind"] == "device":
            if resource_id in device_by_resource:
                _fail(f"resource {resource_id} has multiple device buffers")
            device_by_resource[resource_id] = identifier
    if set(device_by_resource) != set(resources):
        _fail("host device-buffer inventory differs from plan resources")
    return by_id, device_by_resource


def _port_map(program: dict[str, Any], key: str, expected_ids: list[int],
              resources: dict[int, dict[str, Any]], device_by_resource: dict[int, int],
              buffers: dict[int, dict], kind: str) -> dict[int, dict]:
    ports = _array(program.get(key), f"host_program.{key}")
    output = {}
    staging_kind = f"{kind}-staging"
    for port in ports:
        port = _object(port, f"host {kind} port")
        _expect_keys(port, {"resource_id", "device_buffer", "staging_buffer", "elements"},
                     f"host {kind} port")
        resource_id = _integer(port.get("resource_id"), f"{kind} resource_id",
                               maximum=0xFFFFFFFF)
        if resource_id in output:
            _fail(f"host program repeats {kind} resource {resource_id}")
        if resource_id not in resources or resources[resource_id]["kind"] != kind:
            _fail(f"host {kind} port references a resource of another kind")
        if _integer(port.get("device_buffer"), f"{kind} device_buffer",
                    maximum=0xFFFFFFFF) != device_by_resource[resource_id]:
            _fail(f"host {kind} port device buffer does not match its resource")
        staging_id = _integer(port.get("staging_buffer"), f"{kind} staging_buffer",
                              maximum=0xFFFFFFFF)
        staging = buffers.get(staging_id)
        if (staging is None or staging.get("resource_id") != resource_id or
                staging.get("kind") != staging_kind):
            _fail(f"host {kind} port staging buffer is missing or mismatched")
        if _integer(port.get("elements"), f"{kind} elements") != math.prod(resources[resource_id]["shape"]):
            _fail(f"host {kind} element count differs from plan shape")
        output[resource_id] = port
    if list(output) != expected_ids:
        _fail(f"host {kind} port order differs from plan IDs")
    return output


def _interface_expected(kernel: dict[str, Any]) -> dict[str, Any]:
    bindings = []
    for binding in kernel["bindings"]:
        _object(binding, "plan kernel binding")
        descriptor_set = _integer(binding.get("set"), "plan descriptor set",
                                   maximum=0xFFFFFFFF)
        descriptor_binding = _integer(binding.get("binding"),
                                       "plan descriptor binding",
                                       maximum=0xFFFFFFFF)
        _integer(binding.get("resource_id"), "plan binding resource_id",
                 maximum=0xFFFFFFFF)
        if binding.get("access") not in {"read", "write"}:
            _fail("plan descriptor binding has unknown access")
        bindings.append({"set": descriptor_set, "binding": descriptor_binding,
                         "access": binding.get("access"),
                         "byte_size": None})
    push_constants = []
    for item in kernel["push_constants"]:
        _object(item, "plan push constant")
        payload = item.get("data_hex")
        if not isinstance(payload, str) or len(payload) % 2 or not re.fullmatch(r"[0-9a-f]*", payload):
            _fail("plan push constant data_hex must be lowercase hexadecimal")
        offset = _integer(item.get("offset"), "plan push-constant offset",
                          maximum=0xFFFFFFFF)
        push_constants.append({"offset": offset,
                               "byte_size": len(bytes.fromhex(payload))})
    return {"entry_point": kernel.get("entry_point"),
            "local_size": kernel["local_size"],
            "bindings": bindings, "push_constants": push_constants}


def _validate_pipelines(program: dict[str, Any], manifest: dict[str, Any],
                        plan: dict[str, Any], resources: dict[int, dict[str, Any]],
                        buffers: dict[int, dict], device_by_resource: dict[int, int],
                        shader_bytes: dict[int, bytes] | None) -> dict[int, dict]:
    kernels = _plan_kernel_map(plan)
    pipelines = _array(program.get("pipelines"), "host_program.pipelines")
    manifest_shaders = _array(manifest.get("shaders"), "manifest.shaders")
    shaders_by_id = {}
    for shader in manifest_shaders:
        _object(shader, "manifest shader")
        for key in ("kernel_id", "binary_symbol", "entry_point", "sha256", "size_bytes"):
            if key not in shader:
                _fail("manifest shader record omits a required field")
        if not isinstance(shader.get("binary_symbol"), str) or not shader["binary_symbol"]:
            _fail("manifest shader binary_symbol must be a nonempty string")
        if not isinstance(shader.get("entry_point"), str) or not shader["entry_point"]:
            _fail("manifest shader entry_point must be a nonempty string")
        _sha(shader.get("sha256"), "manifest shader sha256")
        _integer(shader.get("size_bytes"), "manifest shader size_bytes",
                 minimum=1)
        kernel_id = _integer(shader.get("kernel_id"), "manifest shader kernel_id",
                             maximum=0xFFFFFFFF)
        if kernel_id in shaders_by_id:
            _fail(f"manifest repeats shader kernel id {kernel_id}")
        shaders_by_id[kernel_id] = shader
    if set(shaders_by_id) != set(kernels):
        _fail("manifest shader inventory differs from plan kernels")
    pipeline_by_kernel = {}
    pipeline_ids = set()
    for pipeline in pipelines:
        _object(pipeline, "host pipeline")
        _expect_keys(pipeline, {
            "id", "kernel_id", "binary_symbol", "shader_sha256",
            "shader_size_bytes", "interface", "bindings", "push_constants",
            "dispatch"}, "host pipeline")
        pipeline_id = _integer(pipeline.get("id"), "pipeline id", maximum=0xFFFFFFFF)
        kernel_id = _integer(pipeline.get("kernel_id"), "pipeline kernel_id",
                             maximum=0xFFFFFFFF)
        if pipeline_id in pipeline_ids or kernel_id in pipeline_by_kernel:
            _fail("host pipeline IDs and kernel IDs must be unique")
        if pipeline_id != kernel_id:
            _fail(f"kernel {kernel_id} pipeline ID differs from frozen kernel ID")
        pipeline_ids.add(pipeline_id)
        pipeline_by_kernel[kernel_id] = pipeline
        kernel = kernels.get(kernel_id)
        shader = shaders_by_id.get(kernel_id)
        if kernel is None or shader is None:
            _fail(f"host pipeline references unknown kernel {kernel_id}")
        symbol = pipeline.get("binary_symbol")
        if not isinstance(symbol, str) or not symbol:
            _fail(f"kernel {kernel_id} binary symbol is missing")
        if symbol != shader.get("binary_symbol"):
            _fail(f"kernel {kernel_id} shader binary symbol differs from manifest")
        if pipeline.get("entry_point") is not None:
            _fail("pipeline must not contain an untyped entry_point alias")
        _sha(pipeline.get("shader_sha256"), f"kernel {kernel_id} shader_sha256")
        if pipeline["shader_sha256"] != shader.get("sha256"):
            _fail(f"kernel {kernel_id} host shader hash differs from manifest")
        shader_size = _integer(pipeline.get("shader_size_bytes"), "shader_size_bytes")
        if shader_size != shader.get("size_bytes"):
            _fail(f"kernel {kernel_id} host shader size differs from manifest")
        if shader_bytes is not None:
            raw = shader_bytes.get(kernel_id)
            if raw is None:
                _fail(f"kernel {kernel_id} raw shader bytes are missing")
            if len(raw) != shader_size or hashlib.sha256(raw).hexdigest() != pipeline["shader_sha256"]:
                _fail(f"kernel {kernel_id} raw shader bytes differ from host identity")
        expected_interface = _interface_expected(kernel)
        interface = _object(pipeline.get("interface"), f"kernel {kernel_id} interface")
        _expect_keys(interface, {"entry_point", "local_size", "bindings",
                                 "push_constants"}, "typed shader interface")
        if (not isinstance(interface.get("entry_point"), str) or
                interface.get("entry_point") != expected_interface["entry_point"]):
            _fail(f"kernel {kernel_id} entry point differs from the plan")
        actual_local_size = _array(interface.get("local_size"),
                                   "typed shader interface local_size")
        if len(actual_local_size) != 3:
            _fail(f"kernel {kernel_id} reflected local size is malformed")
        actual_local_size = [_integer(value, "reflected local-size dimension",
                                      minimum=1, maximum=0xFFFFFFFF)
                             for value in actual_local_size]
        if actual_local_size != expected_interface["local_size"]:
            _fail(f"kernel {kernel_id} local size differs from the plan")
        actual_interface_bindings = _array(interface.get("bindings"), "shader interface bindings")
        if len(actual_interface_bindings) != len(expected_interface["bindings"]):
            _fail(f"kernel {kernel_id} typed shader interface binding count differs")
        for actual, expected in zip(actual_interface_bindings, expected_interface["bindings"]):
            _object(actual, "typed shader interface binding")
            _expect_keys(actual, {"set", "binding", "access", "byte_size"},
                         "typed shader interface binding")
            actual_set = _integer(actual.get("set"), "reflected descriptor set",
                                  maximum=0xFFFFFFFF)
            actual_binding = _integer(actual.get("binding"),
                                      "reflected descriptor binding",
                                      maximum=0xFFFFFFFF)
            if actual_set != expected["set"]:
                _fail(f"kernel {kernel_id} typed shader interface set differs")
            if actual_binding != expected["binding"]:
                _fail(f"kernel {kernel_id} typed shader interface binding differs")
            if not isinstance(actual.get("access"), str) or actual.get("access") != expected["access"]:
                _fail(f"kernel {kernel_id} typed shader interface access differs")
            byte_size = _integer(actual.get("byte_size"), "shader interface binding byte_size")
            binding = next(item for item in kernel["bindings"]
                           if item.get("set") == expected["set"] and
                           item.get("binding") == expected["binding"])
            resource_id = _integer(binding.get("resource_id"), "binding resource_id",
                                   maximum=0xFFFFFFFF)
            if resource_id not in resources or byte_size != resources[resource_id]["byte_size"]:
                _fail(f"kernel {kernel_id} shader interface binding size differs from resource")
        actual_push = _array(interface.get("push_constants"), "shader interface push_constants")
        if len(actual_push) != len(expected_interface["push_constants"]):
            _fail(f"kernel {kernel_id} typed push-constant reflection count differs")
        for actual, expected in zip(actual_push, expected_interface["push_constants"]):
            _object(actual, "typed push-constant reflection")
            _expect_keys(actual, {"offset", "byte_size"},
                         "typed push-constant reflection")
            offset = _integer(actual.get("offset"), "reflected push-constant offset",
                              maximum=0xFFFFFFFF)
            byte_size = _integer(actual.get("byte_size"),
                                 "reflected push-constant byte_size")
            if offset != expected["offset"] or byte_size != expected["byte_size"]:
                _fail(f"kernel {kernel_id} typed push-constant reflection differs from plan")
        expected_bindings = []
        for item in kernel["bindings"]:
            resource_id = _integer(item.get("resource_id"), "kernel binding resource_id",
                                   maximum=0xFFFFFFFF)
            buffer_id = device_by_resource.get(resource_id)
            if buffer_id is None:
                _fail(f"kernel {kernel_id} binding references unknown resource {resource_id}")
            expected_bindings.append({"set": item.get("set"), "binding": item.get("binding"),
                                      "buffer": buffer_id, "access": item.get("access")})
        actual_bindings = _array(pipeline.get("bindings"), "pipeline bindings")
        for binding in actual_bindings:
            _object(binding, "host descriptor binding")
            _expect_keys(binding, {"set", "binding", "buffer", "access"},
                         "host descriptor binding")
            _integer(binding.get("set"), "host descriptor set",
                     maximum=0xFFFFFFFF)
            _integer(binding.get("binding"), "host descriptor binding",
                     maximum=0xFFFFFFFF)
            _integer(binding.get("buffer"), "host descriptor buffer",
                     maximum=0xFFFFFFFF)
            if binding.get("access") not in {"read", "write"}:
                _fail("host descriptor binding has unknown access")
        if actual_bindings != expected_bindings:
            _fail(f"kernel {kernel_id} host descriptor bindings differ from plan")
        push_constants = _array(pipeline.get("push_constants"), "pipeline push_constants")
        for push in push_constants:
            _object(push, "host push-constant payload")
            _expect_keys(push, {"offset", "bytes"}, "host push-constant payload")
            _integer(push.get("offset"), "host push-constant offset",
                     maximum=0xFFFFFFFF)
            payload = push.get("bytes")
            if not isinstance(payload, str) or len(payload) % 2 or re.fullmatch(r"[0-9a-f]*", payload) is None:
                _fail("host push-constant bytes must be lowercase hexadecimal")
        expected_push = [{"offset": item["offset"], "bytes": item["data_hex"]}
                         for item in kernel["push_constants"]]
        if push_constants != expected_push:
            _fail(f"kernel {kernel_id} host push-constant bytes differ from plan")
        dispatch = _array(pipeline.get("dispatch"), "pipeline dispatch")
        if len(dispatch) != 3 or any(_integer(value, "pipeline dispatch", minimum=1,
                                               maximum=0xFFFFFFFF) == 0 for value in dispatch):
            _fail(f"kernel {kernel_id} dispatch must have three positive dimensions")
        if dispatch[1:] != [1, 1]:
            _fail(f"kernel {kernel_id} dispatch Y/Z must be one")
        written = [_integer(value, "kernel write resource", maximum=0xFFFFFFFF)
                   for value in kernel["writes"]]
        if not written or any(identifier not in resources for identifier in written):
            _fail(f"kernel {kernel_id} writes an unknown or empty resource set")
        max_bytes = max(resources[identifier]["byte_size"] for identifier in written)
        local_x = kernel["local_size"][0]
        expected_x = (max_bytes + 4 * local_x - 1) // (4 * local_x)
        if dispatch != [expected_x, 1, 1]:
            _fail(f"kernel {kernel_id} dispatch differs from ceil(max written bytes/4/local_x)")
    if list(pipeline_by_kernel) != list(kernels):
        _fail("host pipeline order/inventory differs from plan kernels")
    return pipeline_by_kernel


def _validate_weights(program: dict[str, Any], manifest: dict[str, Any],
                      resources: dict[int, dict[str, Any]],
                      weight_bytes: dict[int, bytes] | None) -> None:
    expected = {identifier: resource for identifier, resource in resources.items()
                if resource["kind"] == "weight"}
    entries = _array(program.get("weights"), "host_program.weights")
    manifest_entries = _array(manifest.get("weights"), "manifest.weights")
    manifest_by_id = {}
    for item in manifest_entries:
        _object(item, "manifest weight")
        resource_id = _integer(item.get("resource_id"), "manifest weight resource_id",
                               maximum=0xFFFFFFFF)
        _sha(item.get("sha256"), "manifest weight sha256")
        _integer(item.get("size_bytes"), "manifest weight size_bytes")
        if resource_id in manifest_by_id:
            _fail(f"manifest repeats weight resource {resource_id}")
        manifest_by_id[resource_id] = item
    if set(manifest_by_id) != set(expected):
        _fail("manifest weight inventory differs from plan weight resources")
    actual = {}
    for entry in entries:
        _object(entry, "host weight")
        _expect_keys(entry, {"resource_id", "byte_size", "logical_sha256",
                             "physical_sha256"}, "host weight")
        resource_id = _integer(entry.get("resource_id"), "weight resource_id",
                               maximum=0xFFFFFFFF)
        if resource_id in actual or resource_id not in expected:
            _fail(f"host weight references duplicate/unknown resource {resource_id}")
        if _integer(entry.get("byte_size"), "host weight byte_size") != expected[resource_id]["byte_size"]:
            _fail(f"weight {resource_id} byte size differs from plan")
        logical = _sha(entry.get("logical_sha256"), "weight logical_sha256")
        physical = _sha(entry.get("physical_sha256"), "weight physical_sha256")
        if logical != expected[resource_id]["weight_sha256"]:
            _fail(f"weight {resource_id} logical hash differs from the plan resource")
        manifest_weight = manifest_by_id[resource_id]
        if (_sha(manifest_weight.get("sha256"), "manifest weight sha256") != physical or
                _integer(manifest_weight.get("size_bytes"), "manifest weight size") != entry["byte_size"]):
            _fail(f"weight {resource_id} physical identity differs from manifest")
        if weight_bytes is not None:
            raw = weight_bytes.get(resource_id)
            if raw is None:
                _fail(f"weight {resource_id} raw bytes are missing")
            if len(raw) != entry["byte_size"] or hashlib.sha256(raw).hexdigest() != physical:
                _fail(f"weight {resource_id} raw bytes differ from physical weight identity")
        actual[resource_id] = entry
    if list(actual) != list(expected):
        _fail("host weight order/inventory differs from plan weight resources")


def _validate_resources(program: dict[str, Any], plan: dict[str, Any],
                         resources: dict[int, dict[str, Any]]) -> None:
    raw = _array(program.get("resources"), "host_program.resources")
    _resource_map({"resources": raw})
    if raw != plan["resources"]:
        _fail("host program resource declarations differ from plan resources")
    inputs = [_integer(item, "plan input resource", maximum=0xFFFFFFFF)
              for item in _array(plan.get("inputs"), "plan.inputs")]
    outputs = [_integer(item, "plan output resource", maximum=0xFFFFFFFF)
               for item in _array(plan.get("outputs"), "plan.outputs")]
    if len(inputs) != len(set(inputs)) or len(outputs) != len(set(outputs)):
        _fail("plan repeats a public input or output resource")
    if any(identifier not in resources or resources[identifier]["kind"] != "input" for identifier in inputs):
        _fail("plan input inventory references a non-input resource")
    if any(identifier not in resources or resources[identifier]["kind"] != "output" for identifier in outputs):
        _fail("plan output inventory references a non-output resource")
    for identifier, resource in resources.items():
        if resource["kind"] == "input" and identifier not in inputs:
            _fail(f"plan input resource {identifier} is absent from the input inventory")
        if resource["kind"] == "output" and identifier not in outputs:
            _fail(f"plan output resource {identifier} is absent from the output inventory")


def _op_array(program: dict[str, Any], name: str) -> list[dict[str, Any]]:
    operations = _array(program.get(name), f"host_program.{name}")
    output = []
    operation_fields = {
        "barrier": {"op", "source_stage", "destination_stage", "buffers"},
        "copy": {"op", "source", "destination", "source_offset",
                 "destination_offset", "byte_size"},
        "upload": {"op", "staging_buffer", "source", "source_id", "byte_size"},
        "mapped-sync": {"op", "buffer", "action", "range"},
        "control": {"op", "action"},
        "bind-pipeline": {"op", "pipeline"},
        "bind-descriptors": {"op", "pipeline", "first_set", "set_count"},
        "push-constants": {"op", "pipeline", "offset", "bytes"},
        "dispatch": {"op", "pipeline", "groups"},
        "prepare-outputs": {"op", "buffers"},
        "publish": {"op", "staging_buffer", "output_index", "byte_size"},
        "release-buffer": {"op", "buffer"},
    }
    for operation in operations:
        operation = _object(operation, f"{name} operation")
        tag = operation.get("op")
        if tag not in operation_fields:
            _fail(f"{name} contains an unknown operation")
        _expect_keys(operation, operation_fields[tag], f"{name} {tag} operation")
        output.append(operation)
    return output


def _validate_operation_references(program: dict[str, Any], buffers: dict[int, dict],
                                   pipelines: dict[int, dict], resources: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    create_ops, run_ops = _op_array(program, "create_operations"), _op_array(program, "run_operations")
    stages = {"host", "transfer", "compute"}
    accesses = {"host-write", "transfer-read", "transfer-write", "shader-read",
                "shader-write", "shader-read-write", "host-read"}
    controls = {"wait-fence", "reset-fence", "reset-command-pool", "begin-commands",
                "end-commands", "submit"}
    for name, operations in (("create_operations", create_ops), ("run_operations", run_ops)):
        for operation in operations:
            tag = operation["op"]
            if tag == "barrier":
                if operation.get("source_stage") not in stages or operation.get("destination_stage") not in stages:
                    _fail(f"{name} barrier has an unknown stage")
                entries = _array(operation.get("buffers"), "barrier buffers")
                identifiers = []
                for entry in entries:
                    _object(entry, "barrier buffer")
                    _expect_keys(entry, {"buffer", "source_access",
                                         "destination_access", "offset", "byte_size"},
                                 "barrier buffer")
                    identifier = _integer(entry.get("buffer"), "barrier buffer", maximum=0xFFFFFFFF)
                    if identifier not in buffers:
                        _fail(f"barrier references unknown buffer {identifier}")
                    if entry.get("source_access") not in accesses or entry.get("destination_access") not in accesses:
                        _fail("barrier has an unknown access")
                    offset = _integer(entry.get("offset"), "barrier offset")
                    byte_size = _integer(entry.get("byte_size"), "barrier byte_size")
                    if not byte_size or offset + byte_size > buffers[identifier]["byte_size"]:
                        _fail(f"barrier range exceeds buffer {identifier}")
                    identifiers.append(identifier)
                if identifiers != sorted(set(identifiers)):
                    _fail("barrier buffer batches must be unique and sorted by buffer id")
            elif tag == "copy":
                source = _integer(operation.get("source"), "copy source", maximum=0xFFFFFFFF)
                destination = _integer(operation.get("destination"), "copy destination", maximum=0xFFFFFFFF)
                byte_size = _integer(operation.get("byte_size"), "copy byte_size")
                source_offset = _integer(operation.get("source_offset"), "copy source_offset")
                destination_offset = _integer(operation.get("destination_offset"), "copy destination_offset")
                if source not in buffers or destination not in buffers or not byte_size:
                    _fail("copy references missing buffers or has zero size")
                if source_offset + byte_size > buffers[source]["byte_size"] or destination_offset + byte_size > buffers[destination]["byte_size"]:
                    _fail("copy range exceeds a declared buffer")
            elif tag == "upload":
                buffer = _integer(operation.get("staging_buffer"), "upload staging_buffer", maximum=0xFFFFFFFF)
                source_id = _integer(operation.get("source_id"), "upload source_id", maximum=0xFFFFFFFF)
                if buffer not in buffers or operation.get("source") not in {"input", "weight"}:
                    _fail("upload references an invalid source or staging buffer")
                if operation.get("source") == "input":
                    ports = _array(program.get("inputs"), "host_program.inputs")
                    if source_id >= len(ports):
                        _fail("input upload references an unknown public input index")
                    port = ports[source_id]
                    if (not isinstance(port, dict) or
                            port.get("staging_buffer") != buffer or
                            resources.get(port.get("resource_id"), {}).get("kind") != "input"):
                        _fail("input upload index does not match its input staging resource")
                if operation.get("source") == "weight" and resources.get(source_id, {}).get("kind") != "weight":
                    _fail("weight upload references a non-weight resource")
                if _integer(operation.get("byte_size"), "upload byte_size") != buffers[buffer]["byte_size"]:
                    _fail("upload byte size differs from staging buffer")
            elif tag == "mapped-sync":
                identifier = _integer(operation.get("buffer"), "mapped-sync buffer", maximum=0xFFFFFFFF)
                if identifier not in buffers or operation.get("action") not in {"flush", "invalidate"} or operation.get("range") != "whole-allocation":
                    _fail("mapped-sync operation is invalid")
            elif tag == "control" and operation.get("action") not in controls:
                _fail("control operation has an unknown action")
            elif tag in {"bind-pipeline", "bind-descriptors", "push-constants", "dispatch"}:
                pipeline_id = _integer(operation.get("pipeline"), f"{tag} pipeline", maximum=0xFFFFFFFF)
                if pipeline_id not in {item["id"] for item in pipelines.values()}:
                    _fail(f"{tag} references an unknown pipeline")
                if tag == "bind-descriptors":
                    first_set = _integer(operation.get("first_set"),
                                         "bind-descriptors first_set",
                                         maximum=0xFFFFFFFF)
                    set_count = _integer(operation.get("set_count"),
                                         "bind-descriptors set_count",
                                         minimum=1, maximum=0xFFFFFFFF)
                    if first_set != 0 or set_count <= 0:
                        _fail("bind-descriptors set range is invalid")
                elif tag == "push-constants":
                    offset = _integer(operation.get("offset"), "push-constants offset", maximum=0xFFFFFFFF)
                    data = operation.get("bytes")
                    if not isinstance(data, str) or len(data) % 2 or not re.fullmatch(r"[0-9a-f]*", data):
                        _fail("push-constants bytes must be lowercase hexadecimal")
                    if not data:
                        _fail("push-constants operation cannot be empty")
                    if offset + len(bytes.fromhex(data)) > 0xFFFFFFFF:
                        _fail("push-constants range overflows uint32")
                elif tag == "dispatch":
                    groups = _array(operation.get("groups"), "dispatch groups")
                    if len(groups) != 3:
                        _fail("dispatch groups must have three dimensions")
                    for group in groups:
                        _integer(group, "dispatch group", minimum=1, maximum=0xFFFFFFFF)
            elif tag == "prepare-outputs":
                ids = [_integer(item, "prepare output buffer", maximum=0xFFFFFFFF)
                       for item in _array(operation.get("buffers"), "prepare output buffers")]
                if len(ids) != len(set(ids)) or any(identifier not in buffers for identifier in ids):
                    _fail("prepare-outputs references invalid or duplicate buffers")
            elif tag == "publish":
                buffer = _integer(operation.get("staging_buffer"), "publish staging_buffer", maximum=0xFFFFFFFF)
                _integer(operation.get("output_index"), "publish output_index", maximum=0xFFFFFFFF)
                _integer(operation.get("byte_size"), "publish byte_size")
                if buffer not in buffers:
                    _fail("publish references unknown staging buffer")
            elif tag == "release-buffer":
                identifier = _integer(operation.get("buffer"), "release buffer", maximum=0xFFFFFFFF)
                if identifier not in buffers:
                    _fail("release-buffer references unknown buffer")
    return create_ops, run_ops


def legacy_symbolic_trace(plan: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    """Independently replay the frozen plan's upload/hazard/dispatch semantics."""
    resources = _resource_map(plan)
    kernels = _plan_kernel_map(plan)
    device_buffers = {entry["resource_id"]: entry["id"]
                      for entry in _array(program.get("buffers"), "host_program.buffers")
                      if isinstance(entry, dict) and entry.get("kind") == "device"}
    inputs = [_integer(item, "plan input", maximum=0xFFFFFFFF)
              for item in _array(plan.get("inputs"), "plan.inputs")]
    outputs = [_integer(item, "plan output", maximum=0xFFFFFFFF)
               for item in _array(plan.get("outputs"), "plan.outputs")]
    weights = [identifier for identifier, resource in resources.items()
               if resource["kind"] == "weight"]
    kernel_trace = []
    shader_written: set[int] = set()
    for kernel_id in kernels:
        kernel = kernels[kernel_id]
        reads = {_integer(item, f"kernel {kernel_id} read", maximum=0xFFFFFFFF)
                 for item in kernel["reads"]}
        writes = {_integer(item, f"kernel {kernel_id} write", maximum=0xFFFFFFFF)
                  for item in kernel["writes"]}
        touched = reads | writes
        hazard_resources = sorted(shader_written & touched)
        if any(item not in resources for item in touched):
            _fail(f"kernel {kernel_id} read/write set references an unknown resource")
        write_sizes = [resources[item]["byte_size"] for item in writes]
        local_x = _integer(kernel["local_size"][0], "local_size X", minimum=1,
                           maximum=0xFFFFFFFF)
        if not write_sizes:
            _fail(f"kernel {kernel_id} has no written resource")
        groups_x = (max(write_sizes) + 4 * local_x - 1) // (4 * local_x)
        kernel_trace.append({"kernel_id": kernel_id,
                             "reads": sorted(reads), "writes": sorted(writes),
                             "compute_barrier_buffers": [device_buffers[item]
                                                          for item in hazard_resources],
                             "dispatch": [groups_x, 1, 1]})
        shader_written.update(writes)
    return {"oracle": "frozen-legacy-symbolic-recorder/v1",
            "uploads": [{"source": "input", "resource_id": item,
                         "staging_buffer": _port_staging(program, "inputs", item),
                         "device_buffer": device_buffers.get(item)} for item in inputs] +
                       [{"source": "weight", "resource_id": item,
                         "staging_buffer": _staging_for_resource(program, item, "weight-staging"),
                         "device_buffer": device_buffers.get(item)} for item in weights],
            "kernels": kernel_trace,
            "readbacks": [{"resource_id": item,
                           "device_buffer": device_buffers.get(item),
                           "staging_buffer": _port_staging(program, "outputs", item)}
                          for item in outputs]}


def _port_staging(program: dict[str, Any], name: str, resource_id: int) -> int | None:
    for port in program.get(name, []):
        if isinstance(port, dict) and port.get("resource_id") == resource_id:
            return port.get("staging_buffer")
    return None


def _staging_for_resource(program: dict[str, Any], resource_id: int,
                          kind: str) -> int | None:
    for buffer in program.get("buffers", []):
        if (isinstance(buffer, dict) and buffer.get("resource_id") == resource_id and
                buffer.get("kind") == kind):
            return buffer.get("id")
    return None


def _legacy_operation_trace(plan: dict[str, Any], program: dict[str, Any],
                            device_by_resource: dict[int, int],
                            buffers: dict[int, dict],
                            pipelines: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Build a separate symbolic recorder trace from plan read/write sets."""
    resources = _resource_map(plan)
    kernels = _plan_kernel_map(plan)

    def barrier(source_stage: str, destination_stage: str,
                entries: list[tuple[int, str, str]]) -> dict[str, Any]:
        return {"op": "barrier", "source_stage": source_stage,
                "destination_stage": destination_stage,
                "buffers": [{"buffer": identifier, "source_access": source,
                             "destination_access": destination, "offset": 0,
                             "byte_size": buffers[identifier]["byte_size"]}
                            for identifier, source, destination in sorted(entries)]}

    def control(action: str) -> dict[str, str]:
        return {"op": "control", "action": action}

    create: list[dict[str, Any]] = []
    weights = [(identifier, resource) for identifier, resource in resources.items()
               if resource["kind"] == "weight"]
    if weights:
        create.extend((control("reset-fence"), control("reset-command-pool"),
                       control("begin-commands")))
        host_entries, device_entries = [], []
        weight_staging = {}
        for resource_id, resource in weights:
            staging = _staging_for_resource(program, resource_id, "weight-staging")
            if staging is None:
                _fail(f"weight {resource_id} has no staging declaration")
            weight_staging[resource_id] = staging
            create.append({"op": "upload", "staging_buffer": staging,
                           "source": "weight", "source_id": resource_id,
                           "byte_size": resource["byte_size"]})
            create.append({"op": "mapped-sync", "buffer": staging,
                           "action": "flush", "range": "whole-allocation"})
            host_entries.append((staging, "host-write", "transfer-read"))
            device_entries.append((device_by_resource[resource_id],
                                   "transfer-write", "shader-read"))
        create.append(barrier("host", "transfer", host_entries))
        for resource_id, resource in weights:
            create.append({"op": "copy", "source": weight_staging[resource_id],
                           "destination": device_by_resource[resource_id],
                           "source_offset": 0, "destination_offset": 0,
                           "byte_size": resource["byte_size"]})
        create.append(barrier("transfer", "compute", device_entries))
        create.extend((control("end-commands"), control("submit"),
                       control("wait-fence")))
        create.extend({"op": "release-buffer", "buffer": weight_staging[resource_id]}
                      for resource_id, _ in weights)

    run: list[dict[str, Any]] = []
    input_ids = list(plan["inputs"])
    input_ports = {port["resource_id"]: port for port in program["inputs"]}
    host_entries, device_entries = [], []
    for index, resource_id in enumerate(input_ids):
        port = input_ports[resource_id]
        size = resources[resource_id]["byte_size"]
        run.extend(({"op": "upload", "staging_buffer": port["staging_buffer"],
                     "source": "input", "source_id": index, "byte_size": size},
                    {"op": "mapped-sync", "buffer": port["staging_buffer"],
                     "action": "flush", "range": "whole-allocation"}))
        host_entries.append((port["staging_buffer"], "host-write", "transfer-read"))
        device_entries.append((port["device_buffer"], "transfer-write", "shader-read"))
    run.append(control("wait-fence"))
    run.extend((control("reset-fence"), control("reset-command-pool"),
                control("begin-commands")))
    if input_ids:
        run.append(barrier("host", "transfer", host_entries))
        for resource_id in input_ids:
            port = input_ports[resource_id]
            run.append({"op": "copy", "source": port["staging_buffer"],
                        "destination": port["device_buffer"], "source_offset": 0,
                        "destination_offset": 0,
                        "byte_size": resources[resource_id]["byte_size"]})
        run.append(barrier("transfer", "compute", device_entries))
    shader_written: set[int] = set()
    for kernel_id in kernels:
        kernel = kernels[kernel_id]
        pipeline = pipelines[kernel_id]
        accessed = set(kernel["reads"]) | set(kernel["writes"])
        hazards = sorted(shader_written & accessed)
        if hazards:
            run.append(barrier("compute", "compute", [
                (device_by_resource[resource_id], "shader-write", "shader-read-write")
                for resource_id in hazards]))
        run.append({"op": "bind-pipeline", "pipeline": pipeline["id"]})
        set_count = max((binding["set"] + 1 for binding in pipeline["bindings"]),
                        default=0)
        if set_count:
            run.append({"op": "bind-descriptors", "pipeline": pipeline["id"],
                        "first_set": 0, "set_count": set_count})
        for push in pipeline["push_constants"]:
            run.append({"op": "push-constants", "pipeline": pipeline["id"],
                        "offset": push["offset"], "bytes": push["bytes"]})
        run.append({"op": "dispatch", "pipeline": pipeline["id"],
                    "groups": pipeline["dispatch"]})
        shader_written.update(kernel["writes"])
    output_ids = list(plan["outputs"])
    output_ports = {port["resource_id"]: port for port in program["outputs"]}
    if output_ids:
        run.append(barrier("compute", "transfer", [
            (output_ports[item]["device_buffer"], "shader-write", "transfer-read")
            for item in output_ids]))
        for resource_id in output_ids:
            port = output_ports[resource_id]
            run.append({"op": "copy", "source": port["device_buffer"],
                        "destination": port["staging_buffer"],
                        "source_offset": 0, "destination_offset": 0,
                        "byte_size": resources[resource_id]["byte_size"]})
        run.append(barrier("transfer", "host", [
            (output_ports[item]["staging_buffer"], "transfer-write", "host-read")
            for item in output_ids]))
    run.extend((control("end-commands"), control("submit"), control("wait-fence")))
    for resource_id in output_ids:
        run.append({"op": "mapped-sync",
                    "buffer": output_ports[resource_id]["staging_buffer"],
                    "action": "invalidate", "range": "whole-allocation"})
    run.append({"op": "prepare-outputs",
                "buffers": [output_ports[item]["staging_buffer"] for item in output_ids]})
    for output_index, resource_id in enumerate(output_ids):
        port = output_ports[resource_id]
        run.append({"op": "publish", "staging_buffer": port["staging_buffer"],
                    "output_index": output_index,
                    "byte_size": resources[resource_id]["byte_size"]})
    return create, run


def validate_host_program(manifest: dict[str, Any], plan: dict[str, Any], *,
                          shader_bytes: dict[int, bytes] | None = None,
                          weight_bytes: dict[int, bytes] | None = None,
                          manifest_sha256: str | None = None,
                          plan_sha256: str | None = None) -> dict[str, Any]:
    """Validate host program identity, plan/interface/payload links and operations."""
    manifest = _object(manifest, "Vulkan manifest")
    plan = _object(plan, "Vulkan plan")
    if (type(manifest.get("schema_version")) is not int or
            manifest.get("schema_version") != 1 or
            manifest.get("kind") != "ncnn.vulkan_model_manifest"):
        _fail("Vulkan model manifest schema/version is unknown")
    if (type(plan.get("schema_version")) is not int or
            plan.get("schema_version") != 1 or
            plan.get("kind") != "ncnn.vulkan_execution_plan"):
        _fail("Vulkan execution plan schema/version is unknown")
    for label, document, field in (
            ("manifest", manifest, "model"),
            ("plan", plan, "model"), ("plan", plan, "host_triple"),
            ("plan", plan, "device_target")):
        if not isinstance(document.get(field), str) or not document[field]:
            _fail(f"{label} {field} must be a nonempty string")
    program = _object(manifest.get("host_program"), "manifest.host_program")
    _expect_keys(program, {
        "schema", "codegen_version", "helper_contract", "synchronization_version",
        "canonical_encoding", "abi_version", "model", "host_triple",
        "device_target", "options", "resources", "buffers", "inputs", "outputs",
        "pipelines", "weights", "initialize", "destroy", "create_operations",
        "run_operations", "identity"}, "host program")
    if manifest.get("host_program_usage") != "compiler-only-diagnostic":
        _fail("manifest host_program_usage is not compiler-only-diagnostic")
    candidate_claim = manifest.get("host_aot_candidate", False)
    if (type(candidate_claim) is not bool or candidate_claim or
            manifest.get("host_implementation") != "model-fixed-run" or
            type(manifest.get("host_codegen_version")) is not int or
            manifest.get("host_codegen_version") != 2 or
            manifest.get("host_source_codegen_version") !=
            "ncnn-vulkan-fixed-run-codegen/v1" or
            manifest.get("execution_plan_usage") != "build-evidence-only"):
        _fail("manifest must identify the model-fixed-run product contract, "
              "not a generic plan runtime or an unclaimed host AOT")
    if not isinstance(manifest.get("host_codegen_identity"), str) or \
            not manifest["host_codegen_identity"]:
        _fail("manifest host_codegen_identity must be a nonempty string")
    for field, expected in (("schema", SCHEMA), ("codegen_version", CODEGEN),
                            ("helper_contract", HELPER_CONTRACT),
                            ("synchronization_version", SYNC_VERSION),
                            ("canonical_encoding", CANONICAL_ENCODING)):
        if program.get(field) != expected:
            _fail(f"host program {field} is missing or has an unknown version")
    if _integer(program.get("abi_version"), "host program abi_version", maximum=0xFFFFFFFF) != 1:
        _fail("host program ABI version is not 1")
    if type(manifest.get("abi_version")) is not int or manifest.get("abi_version") != 1:
        _fail("Vulkan model manifest ABI version is not 1")
    if (program.get("model") != manifest.get("model") or
            program.get("model") != plan.get("model")):
        _fail("host program model differs from manifest or execution plan")
    options = _object(program.get("options"), "host program options")
    _expect_keys(options, set(OPTIONS), "host program options")
    for key in ("allocation", "command_recording", "mapped_sync_range",
                "output_publication", "weight_upload"):
        if not isinstance(options.get(key), str):
            _fail(f"host program option {key} must be a string")
    if type(options.get("synchronize_same_context")) is not bool:
        _fail("host program synchronize_same_context option must be boolean")
    if options != OPTIONS:
        _fail("host program code generation options differ from the frozen contract")
    if program.get("host_triple") != plan.get("host_triple") or program.get("device_target") != plan.get("device_target"):
        _fail("host program target triple/device target differs from plan")
    identity = _sha(manifest.get("host_program_identity"), "manifest.host_program_identity")
    if program.get("identity") != identity:
        _fail("host program embedded identity differs from manifest host_program_identity")
    calculated = canonical_identity(program)
    if calculated != identity:
        _fail("host program canonical identity hash mismatch")
    if manifest_sha256 is not None:
        _sha(manifest_sha256, "manifest SHA-256")
    if plan_sha256 is not None and manifest.get("plan_sha256") != plan_sha256:
        _fail("manifest plan_sha256 differs from actual plan bytes")
    resources = _resource_map(plan)
    _validate_resources(program, plan, resources)
    buffers, device_by_resource = _buffer_map(program, resources)
    plan_inputs = [_integer(item, "plan input", maximum=0xFFFFFFFF)
                   for item in plan.get("inputs", [])]
    plan_outputs = [_integer(item, "plan output", maximum=0xFFFFFFFF)
                    for item in plan.get("outputs", [])]
    input_ports = _port_map(program, "inputs", plan_inputs, resources,
                            device_by_resource, buffers, "input")
    output_ports = _port_map(program, "outputs", plan_outputs, resources,
                             device_by_resource, buffers, "output")
    expected_buffers = [
        (resource_id, "device", "context") for resource_id in resources]
    expected_buffers.extend((resource_id, "input-staging", "context")
                            for resource_id in plan_inputs)
    expected_buffers.extend((resource_id, "output-staging", "context")
                            for resource_id in plan_outputs)
    expected_buffers.extend((resource_id, "weight-staging", "create-upload")
                            for resource_id, resource in resources.items()
                            if resource["kind"] == "weight")
    actual_buffer_sequence = [(item["resource_id"], item["kind"], item["scope"])
                              for item in program["buffers"]]
    if actual_buffer_sequence != expected_buffers or [item["id"] for item in program["buffers"]] != list(range(len(expected_buffers))):
        _fail("host buffer inventory/order differs from plan resources and public I/O")
    pipelines = _validate_pipelines(program, manifest, plan, resources, buffers,
                                    device_by_resource, shader_bytes)
    _validate_weights(program, manifest, resources, weight_bytes)
    create_ops, run_ops = _validate_operation_references(
        program, buffers, pipelines, resources)
    expected_initialize = ([{"kind": "buffer", "id": item["id"]}
                            for item in program["buffers"]] +
                           [{"kind": "pipeline", "id": item["id"]}
                            for item in program["pipelines"]])
    initialize = _array(program.get("initialize"), "host_program.initialize")
    destroy = _array(program.get("destroy"), "host_program.destroy")
    for item in (*initialize, *destroy):
        _object(item, "host lifecycle object")
        _expect_keys(item, {"kind", "id"}, "host lifecycle object")
        _integer(item.get("id"), "host lifecycle object id", maximum=0xFFFFFFFF)
        if item.get("kind") not in {"buffer", "pipeline"}:
            _fail("host lifecycle object has an unknown kind")
    if initialize != expected_initialize:
        _fail("host initialize order differs from declared buffers/pipelines")
    expected_destroy = [item for item in reversed(expected_initialize)
                        if not (item["kind"] == "buffer" and
                                buffers[item["id"]]["scope"] == "create-upload")]
    if destroy != expected_destroy:
        _fail("host destroy order is not reverse initialization without create-upload buffers")
    expected_ops_trace = legacy_symbolic_trace(plan, program)
    actual_uploads = [item for item in (*create_ops, *run_ops) if item["op"] == "upload"]
    expected_uploads = expected_ops_trace["uploads"]
    actual_upload_projection = []
    for item in actual_uploads:
        source_id = item.get("source_id")
        resource_id = source_id
        if item.get("source") == "input":
            input_order = list(plan_inputs)
            if type(source_id) is not int or source_id < 0 or source_id >= len(input_order):
                _fail("input upload source_id is not a public input index")
            resource_id = input_order[source_id]
        actual_upload_projection.append({
            "source": item.get("source"), "resource_id": resource_id,
            "staging_buffer": item.get("staging_buffer"),
            "device_buffer": device_by_resource.get(resource_id)})
    if sorted(actual_upload_projection, key=lambda item: (item["source"], item["resource_id"])) != sorted(
            expected_uploads, key=lambda item: (item["source"], item["resource_id"])):
        _fail("host upload operations differ from plan inputs/weights")
    for port in input_ports.values():
        staging_id = port["staging_buffer"]
        device_id = port["device_buffer"]
        if not any(item["op"] == "copy" and item.get("source") == staging_id and
                   item.get("destination") == device_id and
                   item.get("byte_size") == resources[port["resource_id"]]["byte_size"]
                   for item in run_ops):
            _fail(f"input resource {port['resource_id']} has no exact staging-to-device copy")
    for resource_id in resources:
        if resources[resource_id]["kind"] == "weight":
            staging_id = _staging_for_resource(program, resource_id, "weight-staging")
            if staging_id is None or not any(item["op"] == "copy" and item.get("source") == staging_id and
                                               item.get("destination") == device_by_resource[resource_id] and
                                               item.get("byte_size") == resources[resource_id]["byte_size"]
                                               for item in create_ops):
                _fail(f"weight resource {resource_id} has no exact create-upload copy")
    for port in input_ports.values():
        staging_id = port["staging_buffer"]
        if not any(item["op"] == "barrier" and item.get("source_stage") == "host" and
                   item.get("destination_stage") == "transfer" and any(
                       entry.get("buffer") == staging_id and
                       entry.get("source_access") == "host-write" and
                       entry.get("destination_access") == "transfer-read"
                       for entry in item.get("buffers", [])) for item in run_ops):
            _fail(f"input resource {port['resource_id']} lacks host-to-transfer barrier")
    for resource_id in resources:
        if resources[resource_id]["kind"] == "weight":
            staging_id = _staging_for_resource(program, resource_id, "weight-staging")
            if not any(item["op"] == "barrier" and item.get("source_stage") == "host" and
                       item.get("destination_stage") == "transfer" and any(
                           entry.get("buffer") == staging_id and
                           entry.get("source_access") == "host-write" and
                           entry.get("destination_access") == "transfer-read"
                           for entry in item.get("buffers", [])) for item in create_ops):
                _fail(f"weight resource {resource_id} lacks host-to-transfer barrier")
    # Independent compute hazard oracle: the prior shader write-set intersects
    # the current shader read/write set, with only resource-batch ordering sorted.
    compute_barriers = [item for item in run_ops if item["op"] == "barrier" and
                        item.get("source_stage") == "compute" and
                        item.get("destination_stage") == "compute"]
    kernels = _plan_kernel_map(plan)
    expected_hazards = []
    shader_written: set[int] = set()
    for kernel_id in kernels:
        kernel = kernels[kernel_id]
        current = set(kernel["reads"]) | set(kernel["writes"])
        hazard = sorted(shader_written & current)
        if hazard:
            expected_hazards.append((kernel_id, [device_by_resource[item] for item in hazard]))
        shader_written.update(kernel["writes"])
    if len(compute_barriers) != len(expected_hazards):
        _fail("compute barrier count differs from independent legacy hazard oracle")
    actual_hazards = []
    for barrier in compute_barriers:
        entries = barrier["buffers"]
        if any(entry.get("source_access") != "shader-write" or
               entry.get("destination_access") != "shader-read-write"
               for entry in entries):
            _fail("compute barrier access masks do not represent shader hazards")
        actual_hazards.append([entry["buffer"] for entry in entries])
    if actual_hazards != [buffers for _, buffers in expected_hazards]:
        _fail("compute barrier buffer batches differ from prior-write intersection oracle")
    dispatch_ops = [item for item in run_ops if item["op"] == "dispatch"]
    if len(dispatch_ops) != len(kernels):
        _fail("run dispatch count differs from plan kernel count")
    for index, kernel_id in enumerate(kernels):
        pipeline_id = pipelines[kernel_id]["id"]
        expected_groups = pipelines[kernel_id]["dispatch"]
        if dispatch_ops[index].get("pipeline") != pipeline_id or dispatch_ops[index].get("groups") != expected_groups:
            _fail(f"run dispatch order/shape differs for kernel {kernel_id}")
    for index, port in enumerate(output_ports.values()):
        staging_id = port["staging_buffer"]
        device_id = port["device_buffer"]
        byte_size = resources[port["resource_id"]]["byte_size"]
        if not any(item["op"] == "copy" and item.get("source") == device_id and
                   item.get("destination") == staging_id and item.get("byte_size") == byte_size
                   for item in run_ops):
            _fail(f"output resource {port['resource_id']} has no exact device-to-staging readback")
        if not any(item["op"] == "publish" and item.get("staging_buffer") == staging_id and
                   item.get("output_index") == index and item.get("byte_size") == byte_size
                   for item in run_ops):
            _fail(f"output resource {port['resource_id']} has no exact atomic publication operation")
        if not any(item["op"] == "barrier" and item.get("source_stage") == "transfer" and
                   item.get("destination_stage") == "host" and any(
                       entry.get("buffer") == staging_id and
                       entry.get("source_access") == "transfer-write" and
                       entry.get("destination_access") == "host-read"
                       for entry in item.get("buffers", [])) for item in run_ops):
            _fail(f"output resource {port['resource_id']} lacks transfer-to-host barrier")
    expected_create, expected_run = _legacy_operation_trace(
        plan, program, device_by_resource, buffers, pipelines)
    if create_ops != expected_create:
        _fail("create operations differ from independent legacy symbolic recorder")
    if run_ops != expected_run:
        _fail("run operations differ from independent legacy symbolic recorder")
    trace = legacy_symbolic_trace(plan, program)
    return {"status": "verified", "identity_sha256": identity,
            "canonical_bytes_sha256": hashlib.sha256(canonical_program_bytes(program)).hexdigest(),
            "resource_count": len(resources), "pipeline_count": len(pipelines),
            "weight_count": len(program["weights"]), "input_count": len(input_ports),
            "output_count": len(output_ports), "legacy_symbolic_trace": trace,
            "program_operation_counts": {
                "create": len(create_ops), "run": len(run_ops),
                "compute_barriers": len(compute_barriers),
                "dispatches": len(dispatch_ops)},
            "shader_raw_bytes_verified": shader_bytes is not None,
            "weight_raw_bytes_verified": weight_bytes is not None}
