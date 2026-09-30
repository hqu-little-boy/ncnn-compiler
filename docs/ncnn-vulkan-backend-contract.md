# Vulkan semantic, ABI, and execution-plan contracts

This document describes contracts implemented in the compiler library. It does not imply that `ncnn-compile` emits a Vulkan model library or that a Vulkan runtime is present.

## Semantic pipeline and legality

`ncnn-mlir-opt --ncnn-vulkan-semantic-pipeline` consumes the strict NCNN-to-TOSA boundary: a model function with no remaining `ncnn.*` operations (an identity graph may contain no TOSA operations). The pipeline checks that boundary, runs the common TOSA-to-Linalg lowering, rejects residual TOSA, then runs `verify-vulkan-supported-model`.

The common lowering sequence is the upstream TOSA-to-Linalg pipeline, Linalg constant-transpose folding, nested TOSA-to-tensor and TOSA-to-arith conversion, canonicalization, and CSE. It uses `preferConv2DKernelLayoutHWCF=true`; this describes an intermediate Linalg layout, not a Vulkan storage-buffer layout. The CPU `ncnn-tosa-to-linalg-pipeline` uses the same helper and retains its CPU fusion and convolution-strategy passes in their established order.

The V1 verifier accepts only static FP32 copy/identity, elementwise ReLU (`max(input, 0.0)`), and same-shape elementwise Add. The lowered elementwise forms must have identity indexing maps and parallel iteration. Layout-only `linalg.transpose`, `tensor.expand_shape`, `tensor.collapse_shape`, `tensor.reshape`, `tensor.empty`, and `linalg.copy` are accepted when their tensor types satisfy the same static constraints. `tensor.reshape` additionally requires a constant rank-one index shape whose extents exactly match the static result type and whose element count preserves the source byte size. Every data tensor must be ranked FP32 with rank 1–4, positive static extents, checked byte size, and no more than the Vulkan 1.1 minimum storage-buffer range (128 MiB); the constant index shape operand is metadata, not a data buffer. Other operations, types, ranks, parameters, dynamic/zero extents, overflow, or oversized tensors fail; there is no CPU fallback. Device-specific limits remain a runtime responsibility and are not inferred by this semantic verifier.

The verifier reports `ncnn.name` and `ncnn.source_layer` recovered from the importer’s `NameLoc` carrier (`FileLineColLoc` filename `ncnn-layer`) when available, and falls back to operation attributes. Importing `.param/.bin` is not part of this pipeline command.

## GPU operation lowering

`ncnn-mlir-opt --ncnn-vulkan-v2-prepare-pipeline` consumes an ncnn-dialect model, runs NCNN-to-TOSA and the common TOSA-to-Linalg lowering, and rejects residual TOSA. It intentionally does not run the V1-only legality verifier because V2 adds convolution forms. `ncnn-mlir-opt --ncnn-vulkan-v2-pipeline` appends `lower-ncnn-vulkan-model`; for artifact-producing tests, the preparation pipeline and lowering pass can be run separately with `plan-path` set on the pass.

The lowering accepts exactly one model function. Its pointwise/layout path handles at most one compute operation per fixture: identity, `linalg.copy`, ReLU, same-shape Add, `linalg.transpose`, or a static layout-only reshape. Same-element-count reshape views may surround that operation. The Conv2D path handles one `linalg.conv_2d_nhwc_hwcf` plus the compiler-generated static input pad/layout views, broadcast bias initializer, and output crop/transpose scaffolding. Conv2D is restricted to batch one, groups one, static FP32 CHW input/output, constant FP32 weights, dilation one, 1x1 or 3x3 kernels, per-axis stride one or two, and zero padding of at most one element per side; 1x1 requires zero padding. Bias may be absent. Unsupported graph operations, multiple convolutions, dynamic extents/weights, unsupported convolution parameters, and multiple pointwise compute operations fail closed.

Each emitted `gpu.module` uses a Vulkan 1.1 / SPIR-V 1.0 target environment, a single `gpu.func` entry point, a `[64,1,1]` workgroup, and an explicit global-ID bounds guard. Pointwise and layout kernels use flattened FP32 StorageBuffers; transpose calculates row-major remapping and reshape preserves linear element order. The naive Conv2D kernel assigns each output element to one work item, accumulates in fixed kernel/input-channel order, applies zero padding and bias, reads OIHW source weights in HWCF physical order, and writes continuous CHW output. The schema-v1 plan records the weight and bias SHA-256 identities, descriptor bindings, layouts, workgroup, and static public I/O. The source tensor function is removed from the emitted device module.

The lowering emits GPU dialect IR and the execution plan, not SPIR-V bytes or a product host dispatch wrapper. The test-only path in `test/Vulkan/` runs MLIR’s `--test-vulkan-runner-pipeline`, extracts the generated `gpu.binary`, validates the SPIR-V, and dispatches it through the test Vulkan runner. That runner and its Conv2D fixtures are not the model-instance runtime or a `ncnn-compile` product backend.

## Model-instance C ABI

`include/ncnn-mlir/Vulkan/ModelABI.h` is a C-compatible declaration header. It defines ABI version 1, a fixed-width 32-bit `ncnn_vk_status`, `ncnn_vk_options` (`struct_size`, `abi_version`, `device_index`, and `flags`), `NCNN_VK_DEVICE_AUTO`, and the options initializer. The options structure is 16 bytes with four consecutive 32-bit fields; v1 requires a full v1-sized structure, accepts larger structures with the same ABI version while ignoring trailing fields, and requires `flags == 0`.

A model-specific header declares an opaque `<model>_vk_context` and these signatures:

```c
ncnn_vk_status <model>_vk_create(
    const ncnn_vk_options *options,
    <model>_vk_context **context,
    char *diagnostic,
    size_t diagnostic_capacity);

ncnn_vk_status <model>_vk_run(
    <model>_vk_context *context,
    const float *input1, size_t input1_elements,
    float *output1, size_t output1_elements,
    char *diagnostic, size_t diagnostic_capacity);

ncnn_vk_status <model>_vk_destroy(<model>_vk_context *context);
```

The model-specific header contract uses one typed pointer/count pair per input and output. Static shape/rank/element-count macros describe continuous CHW host buffers; callers provide exact element counts to `run`, which has no dynamic shape parameters. The CPU ABI remains a separate stateless API and is unchanged.

`create` requires a non-null options pointer and context output, clears `*context` before work, and on failure releases partial resources and leaves it null. `device_index == NCNN_VK_DEVICE_AUTO` selects the first compatible enumerated physical device; an explicit index selects that enumerated device or fails with `UNSUPPORTED_DEVICE`. One context uses one device; multiple devices are selectable by separate contexts, not combined for one graph. `create` owns per-context device/pipeline/resources and uploads model constants once. Each successful `run` performs the complete host-to-device, dispatch, and device-to-host operation. Repeated runs are supported. `destroy(NULL)` returns `SUCCESS`; after successful destruction the context pointer is invalid and must not be reused. Callers must externally synchronize destruction with all other uses of that context. Concurrent `run` calls on one live context are serialized; distinct contexts are independent. No context is hidden in a process-global singleton.

A nonzero diagnostic capacity requires a writable non-null buffer. On return, such a buffer is NUL-terminated and may contain a truncated UTF-8 diagnostic; capacity zero disables diagnostics. A failed `run` does not promise output-buffer contents. Status values are stable: `SUCCESS=0`, `INVALID_ARGUMENT=1`, `UNSUPPORTED_DEVICE=2`, `OUT_OF_MEMORY=3`, `DEVICE_LOST=4`, `API_ERROR=5`, and `BUFFER_SIZE_MISMATCH=6`.

The CTest `vulkan-abi-c-contract` builds a generated-header-shaped C fixture against the CPU `relu_scalar.h`, checks that their public I/O counts/shapes agree, and exercises options/status/null-destroy/buffer-count behavior with a test-only contract stub. It does not exercise Vulkan runtime behavior. The current `ncnn-compile` product path does not emit model-specific Vulkan headers or define these lifecycle symbols in a Vulkan shared library.

## Vulkan execution-plan schema

`ncnn-mlir/Support/VulkanExecutionPlan.hpp` defines the separate typed schema, serializer, deserializer, and verifier. It is not the CPU `EmitModelPlan` JSON or the CPU C ABI manifest. The current CPU execution-plan pipeline does not emit this Vulkan schema.

The version-1 JSON kind is `ncnn.vulkan_execution_plan`. Its top-level fields are `schema_version`, `kind`, `model`, `build_identity`, `host_triple`, `device_target`, `resources`, `kernels`, `inputs`, `outputs`, and `plan_hash`. Resources describe a unique ID, role, FP32 static shape/byte size, logical and physical layouts, optional host layout and an optional structured layout transform (`from_layout`, `to_layout`, `operation`; host-to-device for inputs and device-to-host for outputs), optional weight SHA-256 identity, and first/last kernel indices. Kernels are stored in execution order and describe unique ID, operation, SPIR-V entry point, workgroup size, resource reads/writes, descriptor set/binding/access records, and push-constant bytes by offset. Public I/O maps to resource IDs.

The verifier rejects unknown schema versions/JSON fields, duplicate IDs or descriptor locations, malformed weight identities, unchecked/mismatched byte sizes, sizes over the Vulkan 1.1 minimum storage-buffer range, missing host/device layout mappings, invalid workgroup/push-constant bounds, read-before-write, writes to inputs/weights, incomplete descriptor coverage, invalid public I/O roles, and lifetimes that do not exactly cover ordered uses. JSON serialization is deterministic; `plan_hash` is the 16-digit lowercase hexadecimal `stableHash64` of the canonical plan JSON with the `plan_hash` field omitted. Deserialization validates the stored hash. Kernel schedule records are verified with synthetic unit fixtures; this schema alone does not create executable shaders, Vulkan resources, or dispatch code.
