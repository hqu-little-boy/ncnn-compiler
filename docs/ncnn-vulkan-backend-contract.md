# Vulkan semantic, ABI, and execution-plan contracts

This document describes the Vulkan semantic subset, plan schema, C ABI, standalone host runtime, and the current opt-in `ncnn-compile` product artifact path. CPU remains the default backend; Vulkan is selected explicitly with `--backend=vulkan` and is fail-closed.

## Semantic pipeline and legality

`ncnn-mlir-opt --ncnn-vulkan-semantic-pipeline` consumes the strict NCNN-to-TOSA boundary: a model function with no remaining `ncnn.*` operations (an identity graph may contain no TOSA operations). The pipeline checks that boundary, runs the common TOSA-to-Linalg lowering, rejects residual TOSA, then runs `verify-vulkan-supported-model`.

The common lowering sequence is the upstream TOSA-to-Linalg pipeline, Linalg constant-transpose folding, nested TOSA-to-tensor and TOSA-to-arith conversion, canonicalization, and CSE. It uses `preferConv2DKernelLayoutHWCF=true`; this describes an intermediate Linalg layout, not a Vulkan storage-buffer layout. The CPU `ncnn-tosa-to-linalg-pipeline` uses the same helper and retains its CPU fusion and convolution-strategy passes in their established order.

The V1 verifier accepts only static FP32 copy/identity, elementwise ReLU (`max(input, 0.0)`), and same-shape elementwise Add. The lowered elementwise forms must have identity indexing maps and parallel iteration. Layout-only `linalg.transpose`, `tensor.expand_shape`, `tensor.collapse_shape`, `tensor.reshape`, `tensor.empty`, and `linalg.copy` are accepted when their tensor types satisfy the same static constraints. `tensor.reshape` additionally requires a constant rank-one index shape whose extents exactly match the static result type and whose element count preserves the source byte size. Every data tensor must be ranked FP32 with rank 1–4, positive static extents, checked byte size, and no more than the Vulkan 1.1 minimum storage-buffer range (128 MiB); the constant index shape operand is metadata, not a data buffer. Other operations, types, ranks, parameters, dynamic/zero extents, overflow, or oversized tensors fail; there is no CPU fallback. Device-specific limits remain a runtime responsibility and are not inferred by this semantic verifier.

The verifier reports `ncnn.name` and `ncnn.source_layer` recovered from the importer’s `NameLoc` carrier (`FileLineColLoc` filename `ncnn-layer`) when available, and falls back to operation attributes. Importing `.param/.bin` is not part of this pipeline command.

## GPU operation lowering

`ncnn-mlir-opt --ncnn-vulkan-v2-prepare-pipeline` consumes an ncnn-dialect model, runs NCNN-to-TOSA and the common TOSA-to-Linalg lowering, and rejects residual TOSA. It intentionally does not run the V1-only legality verifier because V2 adds convolution forms. `ncnn-mlir-opt --ncnn-vulkan-v2-pipeline` appends `lower-ncnn-vulkan-model`; for artifact-producing tests, the preparation pipeline and lowering pass can be run separately with `plan-path` set on the pass.

The lowering accepts exactly one model function. Its pointwise/layout path handles at most one compute operation per fixture: identity, `linalg.copy`, ReLU, same-shape Add, `linalg.transpose`, or a static layout-only reshape. Same-element-count reshape views may surround that operation. The Conv2D path handles one `linalg.conv_2d_nhwc_hwcf` plus the compiler-generated static input pad/layout views, broadcast bias initializer, and output crop/transpose scaffolding. Conv2D is restricted to batch one, groups one, static FP32 CHW input/output, constant FP32 weights, dilation one, 1x1 or 3x3 kernels, per-axis stride one or two, and zero padding of at most one element per side; 1x1 requires zero padding. Bias may be absent. Unsupported graph operations, multiple convolutions, dynamic extents/weights, unsupported convolution parameters, and multiple pointwise compute operations fail closed.

Each emitted `gpu.module` uses a Vulkan 1.1 / SPIR-V 1.0 target environment, a single `gpu.func` entry point, a `[64,1,1]` workgroup, and an explicit global-ID bounds guard. Pointwise and layout kernels use flattened FP32 StorageBuffers; transpose calculates row-major remapping and reshape preserves linear element order. The naive Conv2D kernel assigns each output element to one work item, accumulates in fixed kernel/input-channel order, applies zero padding and bias, reads OIHW source weights in HWCF physical order, and writes continuous CHW output. The schema-v1 plan records the weight and bias SHA-256 identities, descriptor bindings, layouts, workgroup, and static public I/O. The source tensor function is removed from the emitted device module.

The single-operation lowering emits GPU dialect IR and the execution plan, not SPIR-V bytes or a product host wrapper. Both the opt-in device tests and the product driver run MLIR’s `--test-vulkan-runner-pipeline`, extract `gpu.binary`, and validate SPIR-V. `lib/VulkanRuntime/` supplies the reusable host context; the V2 one-shot `vulkan_add_runner` remains only a test reference. The product compiler uses the separate static graph path below, rather than applying the single-operation Linalg restrictions to an entire model.

## Static graph lowering and product package

`--ncnn-vulkan-graph-pipeline` runs the strict shared NCNN-to-TOSA pipeline followed by `lower-ncnn-vulkan-graph`. The product driver runs these two steps separately to supply `plan-path` and `weights-path`. This fork is at the strict TOSA boundary: it preserves explicit convolution/pooling/axis semantics and never runs CPU strategy, packing, vectorization, bufferization, or OpenMP passes. CPU pipeline order is unchanged.

The graph pass analyzes one straight-line tensor function before emitting artifacts. Data tensors are static FP32, rank 1–4, positive extents, and at most 128 MiB each. A checked SSA schedule is the single source for operation legality, resource IDs, descriptor bindings, and shader generation. The supported graph forms are ReLU (`tosa.clamp` with `[0,+inf]` and propagating NaNs), same-shape Add, identity/copy, exact linear reshape, static transpose/slice, Concat with a valid axis and matching non-axis dimensions, and batch-one Conv2D with constant FP32 filter/bias. Conv2D supports 1x1/3x3, stride one/two, dilation one; 1x1 allows at most one semantic pad element per side, and 3x3 allows at most one semantic pad element before each edge plus statically inferred bottom/right stride-tail padding. The shared conversion's extra trailing stride padding and static output crop are executed, not silently dropped. MaxPool is restricted to batch-one NHWC FP32 3x3 stride-2 with propagating NaNs, zero top/left padding and bottom/right static tail padding of zero or one. AveragePool is restricted to a full-spatial global FP32 reduction with kernel equal to the input spatial extent, stride one, zero padding/zero-points, and FP32 accumulation; regular and adaptive AveragePool are unsupported. Stable Softmax remains six separate reduction/elementwise kernels (ReduceMax, Sub, Exp, ReduceSum, Reciprocal, Mul) with statically validated equal-rank Sub/Mul operands, at most one singleton-broadcast axis across both operands, and zero TOSA Mul shift. Other forms fail with source-layer diagnostics; there is no CPU fallback.

Each compute/layout operation has an independent bounded GPU kernel. Linear reshape and importer-proven Split/scale-one inference Dropout are read-only SSA aliases, not tensor slicing. Transpose and slice materialize their row-major indexing. Every public output has a distinct Output resource and an explicit copy, including identity graphs and repeated SSA returns. Public rank-three host tensors are continuous CHW; other ranks retain their continuous row-major order. Internal NHWC values are converted by actual shader dispatches; layout metadata does not ask the runtime to perform an implicit transpose. Resources are independently allocated without lifetime-based slot reuse.

Reduction semantics are explicit: MaxPool starts each window at negative infinity and excludes padded coordinates; AveragePool accumulates row-major with `+0.0` in H-then-W order and divides by the exact spatial element count; ReduceMax/ReduceSum use increasing axis indices with `-infinity`/`+0.0` initial values. Softmax keeps the six separate operations in NCNN order; the `Exp` maps to the validated GLSL `Exp` SPIR-V instruction. These operation boundaries are not fused.

GPU module symbols are `<model>_vulkan_k<kernel-id>` and entry points are `kernel_<kernel-id>`, with local size `[64,1,1]`. Every shader is serialized and validated, and its entry point, local size, descriptor set/bindings and residual IR are checked against the corresponding plan kernel. The driver rejects missing, duplicate, or extra binary IDs. The model DSO embeds all shaders and constant payloads keyed by kernel/resource ID; it has no dependency on temporary shader or weight files.

The intermediate `weights-path` JSON has kind `ncnn.vulkan_weight_payloads`, schema version 1, and `weights` entries containing `resource_id` and lowercase hexadecimal `bytes`. Deserialization rejects unknown fields, malformed/truncated byte text and duplicate IDs; package validation checks exact plan coverage, byte sizes and logical weight identities. Conv filters have logical OIHW shape/identity and physical OHWI payload matching the TOSA kernel; the standalone Linalg fixture retains HWCF. The shared hash mapping is used by the compiler package and runtime. Other constants use their recorded row-major order.

The product manifest's `shaders` array records `kernel_id`, `binary_symbol`, `entry_point`, `sha256`, and `size_bytes` in plan order. Its `weights` array records `resource_id`, physical-payload `sha256`, and `size_bytes` in resource order. `shader_sha256s`/`weight_sha256s` cover all payloads; the legacy scalar shader fields describe the first shader. The plan records logical weight identities, which need not equal physical-payload hashes after a layout transform. All payloads, model bytes, host/device targets and plan identity participate in the build identity.

## Compiler-only host program

Product packaging derives a typed `VulkanHostProgram` from the verified plan,
exact packaged shader words, and verified weight payloads. The separate
`NCNNVulkanHostProgramSupport` target is linked by the compiler and its tests,
not by `NCNNVulkanRuntime` or the model DSO. Its SPIR-V deserialization/MLIR
requirements are compiler dependencies, not additional model deployment inputs.
The execution-plan JSON schema and public ABI remain version 1.

The program declares tensor/device/staging resources, ordered public ports,
shader interfaces and pipelines, descriptor bindings, push constants, checked
one-dimensional dispatch groups, initialization/destruction ownership, and
ordered typed operations. Device buffers remain independent; read-only SSA views
use their existing resource ID, while repeated public outputs have separate
resources and readbacks. Weight staging is create-upload scoped; public I/O
staging and device storage are context scoped. The program contains no Vulkan
handles, memory-type/queue-family indices, driver identity, actual allocation
requirements, or pipeline cache. Those decisions and actual device-limit checks
remain late-bound. A dispatch count greater than a Vulkan minimum guaranteed
limit is not rejected merely because of that minimum.

Shader reflection consumes the actual SPIR-V words, checks their typed compute
entry/local size, FP32 storage-buffer representation and declared range, exact
descriptor coverage/access, and push-constant layout/ranges. The supported binary
environment is SPIR-V 1.0, Logical GLSL450, Shader capability and the storage-buffer
storage-class extension. Only the selected entry-point function is accepted;
calls, additional functions, Switch and unsupported pointer/control-flow forms
fail closed. Access analysis visits reachable CFG blocks, selecting literal
constant true/false branches and conservatively following both successors for
other conditions. Actual MLIR SSA storage-store types are checked, without
assuming raw instruction and MLIR walk order coincide. The type proof covers
both StorageBuffer and legacy Uniform/BufferBlock storage-buffer encodings.
This interface proof is
not a proof of arbitrary shader mathematics or every dynamic indexing predicate.
The existing offline `spirv-val`, residual-IR and numerical gates remain required;
diagnostic disassembly alone is not the typed interface proof.

Static operations preserve the current conservative recorder semantics: weight
and input host→transfer→compute transitions; before each kernel, compute
write→read/write barriers for its read/write resources that have any earlier
shader write; compute→transfer→host output readback. The source compute stage
also orders earlier reads before a rewrite of a previously produced resource.
Barrier offsets are zero and ranges cover the logical buffer, equivalent to the
runtime's whole-buffer range; non-coherent flush/invalidate still use the whole
actual mapped allocation. Fence completion, every output invalidate and mapping
check precede any caller output publication. This is a compiler description, not
a new inference execution path.

The manifest's `host_program` object is a compiler-only diagnostic with
`host_program_usage = "compiler-only-diagnostic"` and a matching
`host_program_identity`. Its internal domains are
`ncnn-vulkan-host-program/v1`, `ncnn-vulkan-host-codegen/v1`,
`ncnn-vulkan-device-helper-contract/v1`, and `ncnn-vulkan-host-sync/v1`.
The helper-contract domain records the compiler representation's requirement.
`NCNNVulkanDeviceHelper` supplies private Vulkan resource primitives, not an
interpreter for this program. Neither the domain nor linking the helper proves
that the generic model executor has been compiled away.

Program identity is SHA-256 over the length-prefixed canonical JSON encoding:
`u64le(domain-byte-count)`, UTF-8
`ncnn-vulkan-host-program-canonical/v1`, `u64le(json-byte-count)`, then compact
UTF-8 JSON with sorted object keys and ordered arrays. The identity field itself
and both source-plan hashes are excluded. Version domains, ABI/host/device target,
fixed codegen options, resource/I/O/lifecycle/operation declarations, reflected
shader interfaces and exact little-endian shader-byte hashes, and logical and
physical weight identities are included. Invalid UTF-8 and invalid enum values
are rejected before JSON construction. Generated source/object/link hashes are
separate evidence, not recursive program-identity inputs.

Both package factories bind program identity and a canonical semantic-plan hash
in length-prefixed build-identity inputs: `ncnn-vulkan-static-plan-v3` for the
general path and `ncnn-vulkan-product-v2` for the narrow ReLU factory. JSON
whitespace/key order and incoming plan build/hash fields do not substitute for
the checked host-program semantics. Changes to build identity and manifest/plan
bytes are expected; shader bytes and the public header are not changed by host
planning. The program diagnostic is not read by the deployed wrapper, which
still parses its embedded execution plan and calls the generic runtime.

## Product static-graph support matrix (V5.4)

This matrix is for the product `--ncnn-vulkan-graph-pipeline` and explicit `ncnn-compile --backend=vulkan` path. It does **not** widen either historical single-compute-op contract: `--ncnn-vulkan-semantic-pipeline` plus `verify-vulkan-supported-model` is the V1 copy/identity/ReLU/same-shape-Add Linalg legality path, while `--ncnn-vulkan-v2-pipeline` / `lower-ncnn-vulkan-model` lowers one V2 compute op per fixture (including its separately bounded Conv2D path). Their fixtures under `test/Pipelines/` and `test/Conversion/NCNNToVulkan/lower-*.mlir` are compatibility tests, not evidence that a whole graph uses those limits or that those commands support V5 product graphs. Product graph support is independently enforced against the TOSA-level op schedule below; no model-name dispatch or CPU fallback is used.

All product graph data tensors, including intermediate tensors, must be ranked static FP32, rank 1–4, positive in every dimension, checked for byte-count overflow, and no larger than 128 MiB per storage buffer. Public rank-three tensors use continuous CHW; public tensors of other ranks use row-major order. Internal Conv/Pool tensors are NHWC and layout changes are materialized by graph kernels, not implicit runtime transposes. Constant weights/bias are embedded resources; Conv filter identity is logical OIHW and the graph payload is physical OHWI. The following test references name lit fixture functions or Python fixture IDs; `ncnn-mlir-lit` executes the lit files in the default CTest suite.

| Product graph operation / input family | Accepted parameters, rank, dtype, layout and axis | Positive evidence | Negative / boundary evidence |
|---|---|---|---|
| Model boundary and storage | One straight-line function with at least one static tensor input and output; FP32 data rank 1–4; positive static extents; per-tensor bytes ≤128 MiB. Static multi-input, multi-output, fan-out and repeated returned SSA values are supported. | `graph-add-relu.mlir::add_relu`, `graph-conv-chain.mlir::conv_chain`, V5.1 `copy_identity`; device CTests `vulkan-v5-multiop-{nvidia,llvmpipe}` and `vulkan-v5-squeezenet-{nvidia,llvmpipe}`. | `graph-reject.mlir::{dynamic,fp16,rank5,zero_extent,oversized,overflow,external,first/second}`. Every failure occurs before plan/package publication; an existing output is not replaced. |
| ReLU | `tosa.clamp` with exact FP32 `min=+0`, `max=+inf`, identical input/output shape, NaN propagation; rank 1–4. | `graph-add-relu.mlir::add_relu`; full `vulkan-v5-squeezenet-{nvidia,llvmpipe}`. | `graph-reject.mlir::bounded_clamp` rejects bounded `[0,6]` activation; unsupported/NaN semantics fail closed. |
| Add | `tosa.add` on exactly identical static FP32 shapes; rank 1–4. Broadcasting is not accepted for Add. | `graph-add-relu.mlir::add_relu`; V5.1 device fixture `add_relu` checks both sum and ReLU public outputs. | `graph-reject.mlir::broadcast_add`. |
| Copy / identity | Same-shape static FP32 copy/identity, rank 1–4. Every public output is separately materialized, even for aliases or repeated SSA returns. | `graph-layout-alias.mlir::aliases`; V5.1 fixtures `copy_identity` and `repeated_input_alias` on both `vulkan-v5-multiop-{nvidia,llvmpipe}` CTests. | `graph-reject.mlir::{uninitialized,unsupported}` exercise uninitialized and non-subset values; unsupported producers, invalid function boundaries and non-data tensor types fail closed. |
| Reshape / Split / inference Dropout | Reshape is a **linear view only**: static constant output shape and unchanged element count/byte size. Split and inference `Dropout(scale=1)` are aliases only; they add no slicing, copy, or stochastic behavior. Rank 1–4 FP32. | `graph-layout-alias.mlir::aliases`; V5.1 `split_concat`, `copy_identity`, and `repeated_input_alias`; complete SqueezeNet `vulkan-v5-squeezenet-{nvidia,llvmpipe}` verifies its Split and inference Dropout topology. | Non-static, changed-size, or unsupported reshape/producer forms are not admitted. Dynamic shapes and unsupported operators are covered by `graph-reject.mlir::{dynamic,unsupported}`. |
| Transpose / Slice | Static rank-preserving permutation with every axis exactly once and output extents matching the permutation; Slice uses constant rank-length start/size, stays in bounds, and preserves rank. Both operate on FP32 rank 1–4 and materialize row-major indexing. | `graph-layout-alias.mlir::layout` exercises transpose then static slice; V2 single-op transpose fixtures are separate historical tests. | `graph-reject.mlir::duplicate_transpose_axis` is rejected by TOSA verification; `graph-reject.mlir::slice_out_of_bounds` is rejected by graph legality. Dynamic slice metadata is outside the subset. |
| Conv2D | Batch-one NHWC FP32 `tosa.conv2d`; constant OHWI filter and constant channel bias; FP32 accumulation; zero input/weight zero-points; only 1×1 or 3×3, dilation 1, each spatial stride 1 or 2. Semantic padding is at most one element per edge; for 3×3, shared TOSA lowering may add up to `stride-1` trailing bottom/right tail pad before the explicit output crop. Output extents must match the exact TOSA padding/stride formula. No grouped/depthwise or dynamic weights. | `graph-conv-chain.mlir::conv_chain` exercises 3×3 and 1×1 in one multi-op graph; `graph-v52.mlir::conv1x1_pad1`; V5.1 `conv_relu`, `double_conv`, `split_concat`; complete SqueezeNet CTests exercise 26 actual convolutions. | `graph-reject-conv.mlir::{kernel5,dilation2,stride3,padding2,dynamic_weight,resnet18_first_blocked_conv}`; V5.4's ResNet18 probe records its exact first computational convolution below. |
| Concat | FP32 static tensors of the same rank 1–4; any valid axis; every non-axis extent matches, and output axis extent is exactly the sum of input axis extents. | `graph-v52.mlir::concat_spatial`; V5.2 device fixtures `concat_height_axis`, `concat_width_axis`; full SqueezeNet tests exercise channel concatenations. | `graph-reject.mlir::{concat_bad_axis,concat_mismatched_non_axis}`. |
| MaxPool | Rank-four NHWC FP32, batch 1, preserve channels; exact 3×3 kernel, 2×2 stride, propagating NaNs; top/left pad zero; bottom/right TOSA tail pad independently 0 or 1 with output shape checked. Padded coordinates contribute `-inf`. | `graph-v52.mlir::max_pool_tail`; V5.2 device fixture `max_pool_tail`; complete SqueezeNet `pool1/pool3/pool5`. | `graph-reject.mlir::{max_pool_unsupported_padding,max_pool_ignore_nan,max_pool_bad_kernel,max_pool_bad_stride}`. Other MaxPool settings are unsupported. |
| AveragePool | Only global spatial average: rank-four NHWC FP32, batch 1, kernel equals full H×W, output spatial extent 1×1, stride 1, no padding, zero input/output zero-points, FP32 accumulation; accumulation is row-major H then W. | `graph-v52.mlir::global_average`; V5.2 `global_avg_pool`; full SqueezeNet `pool10`. | `graph-reject.mlir::{regular_average_pool,average_pool_nonzero_zero_point}`; regular and adaptive AveragePool are unsupported. |
| Stable Softmax primitives | No fused Softmax kernel. Stable Softmax is the explicit six-op sequence ReduceMax → Sub → Exp → ReduceSum → Reciprocal → Mul. Reductions are same-rank FP32 with one valid axis and reduced extent set to 1; Sub/Mul operands have equal rank and may broadcast across at most one singleton axis total; Exp/Reciprocal preserve the exact tensor type; Mul shift must be scalar zero. NaNs in ReduceMax propagate. | `graph-v52.mlir::stable_softmax`; V5.2 device fixtures `softmax_rank1` (axis 0) and `softmax_rank3_axis2`; complete SqueezeNet rank-one probability Softmax, each cross-checked against ncnn and CPU. | `graph-reject.mlir::{sub_crossed_broadcast,mul_crossed_broadcast,sub_one_operand_multiaxis,mul_nonzero_shift}`; other layouts/axes have no end-to-end model claim unless covered by the same component legality and a numerical fixture. |
| Integer/index metadata | Only scalar zero integer metadata in the specifically consumed Conv/Pool zero-points or Mul shift is accepted. Dense nonzero index shape constants are currently rejected by graph planning, even when used only as reshape metadata. | Zero-point/shift metadata is covered by `graph-v52.mlir` positive fixtures. | `graph-reject.mlir::nonzero_index_shape_metadata` pins the current fail-closed diagnostic. This is a known graph-path limitation, not an implemented shape operation. |
| All other ops / graph structure | No control-flow regions, dynamic extents, non-FP32 tensor data, unsupported TOSA operations, arbitrary rank/layout conversion, grouped convolution, regular/adaptive AvgPool, or unlisted parameter combination. | Exact model end-to-end claims only for the CTest fixtures named above; currently the full model product verification is SqueezeNet v1.1. | `graph-reject.mlir` and `graph-reject-conv.mlir`; diagnostics include the source layer/name whenever the source operation carries an NCNN location. There is no silent fallback. |

**ResNet18 next-model probe (observed 2026-10-06; no implementation added).** The unmodified `/mnt/ncnn-compiler/yolov5_v7.0_models/resnet18.ncnn.param` does not declare static Input dimensions; actual product CLI invocation therefore first fails closed with `func.func ... requires static tensor dimensions`. With an equivalent temporary param carrying the standard static `224×224×3` Input metadata, the product graph pass next encounters a hoisted classifier `arith.constant dense<[512,1,1]> : tensor<3xindex>` (unknown source location) and currently rejects it as `non-FP32 constants must be scalar zero metadata`; the graph pass runs before any Vulkan artifact is published. Independently, the ordinary importer plus strict NCNN-to-TOSA pipeline exposes the first compute layer as `convrelu_0`, source layer 1: NHWC input `[1,224,224,3]`, filter `[64,7,7,3]`, stride `[2,2]`, TOSA padding `[3,4,3,4]` (explicit pad 3 plus stride-tail pad 1), output `[1,113,113,64]` before the explicit crop. The isolated exact TOSA convolution is rejected with `conv2d supports only 1x1 and 3x3 kernels`; this proves the first **computational Conv2D** outside the current subset only after static-input and shape-metadata barriers are accounted for. The boundary is pinned by `graph-reject-conv.mlir::resnet18_first_blocked_conv`; this is diagnostic evidence only, not a ResNet support claim or model-name special case.

CPU remains the default backend and its pipeline is unchanged. Vulkan product support remains explicit opt-in (`--backend=vulkan`) and every unsupported graph must fail without CPU fallback.

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

The generated model-specific header uses one typed pointer/count pair per input and output. Static shape/rank/element-count macros describe continuous FP32 host buffers; callers provide exact element counts to `run`, which has no dynamic shape parameters. The generated signature lists all inputs in plan order, then all outputs in plan order, followed by the diagnostic pointer/capacity. Each port has its own rank/dimension/element-count macros; repeated SSA outputs still have separate public buffers. Single-input/output signatures retain ABI version 1 and the same argument order. The CPU ABI remains a separate stateless API and is unchanged.

The frozen model ABI contract requires a non-null options pointer and context output, clears `*context` before work, and on create failure releases partial resources and leaves it null. `device_index == NCNN_VK_DEVICE_AUTO` selects the first compatible enumerated physical device; an explicit index selects that enumerated device or fails with `UNSUPPORTED_DEVICE`. This index is an enumeration ordinal, not a stable device identity across ICD installations or enumeration changes. One context uses one device; multiple devices are selectable by separate contexts, not combined for one graph. Create owns per-context device/pipeline/resources and uploads model constants once. Each successful run performs the complete host-to-device, dispatch, and device-to-host operation; repeated runs are supported. `destroy(NULL)` returns `SUCCESS`; explicit destroy reports cleanup/device errors through the status return, and after destruction the context pointer is invalid and must not be reused. Callers must externally synchronize destruction with all other uses of that context. Concurrent runs on one live context are serialized; distinct contexts are independent. No context is hidden in a process-global singleton.

A nonzero diagnostic capacity requires a writable non-null buffer separate from tensor buffers and options/context storage. Callers provide non-overlapping buffers for distinct public outputs and synchronize every use of their tensor storage until `run` returns. On return, such a buffer is NUL-terminated and may contain a truncated UTF-8 diagnostic; capacity zero disables diagnostics. With valid caller buffers, a reported `run` failure leaves all caller output bytes unchanged: all device readback synchronization and staging checks finish before any output is copied. Host argument errors are retryable after correction; staging, execution or readback failures mark the context unusable and require destroy/create before another inference. Status values are stable: `SUCCESS=0`, `INVALID_ARGUMENT=1`, `UNSUPPORTED_DEVICE=2`, `OUT_OF_MEMORY=3`, `DEVICE_LOST=4`, `API_ERROR=5`, and `BUFFER_SIZE_MISMATCH=6`.

The CTest `vulkan-abi-c-contract` remains a generated-header-shaped C contract test using a test stub. The V3 device suite retains its model-shaped test adapter. `ncnn-compile-vulkan-product` is registered when both `NCNN_ENABLE_VULKAN_PRODUCT=ON` and `NCNN_ENABLE_VULKAN_V0_TESTS=ON`; it exercises the generated model header and lifecycle symbols in the packaged DSO and requires the configured device/ICD test environment. `ncnn-compile-vulkan-offline` and the pure product/runtime/evidence unit tests are registered with product support even when the device suite is disabled; they do not count as GPU numerical evidence. `NCNN_VULKAN_TEST_DEVICES` selects `nvidia`, `llvmpipe`, or their default pair for the device suite, and only selected ICD dependencies are required. The standalone product support can be built without enabling that device-test suite. `ncnn-compile` emits the distinct `<model>_vulkan.h`, `<model>_vulkan.json`, `<model>_vulkan.plan.json`, and `lib<model>_vulkan.so` artifacts.

## Product model library and deployment

`--backend=cpu` remains the default. A compiler configured with `NCNN_ENABLE_VULKAN_PRODUCT=ON` accepts explicit `--backend=vulkan`; CPU tuning/thread/vector/ISA flags, dynamic input-shape options, custom clang/linker arguments, CPU-stage emission, and profiling are rejected for this backend. The host toolchain must target 64-bit x86 Linux ELF; the device target remains the independent `vulkan1.1-spirv1.0` target. The compiler backend requires MLIR 21's Vulkan runner pipeline plus `spirv-val` and `spirv-dis` during model compilation. A deployed model DSO does not JIT, invoke the compiler or require SPIR-V command-line tools. The default CPU-only CMake configuration does not enable Vulkan product dependencies.

The product path accepts complete static FP32 graphs in the explicitly checked graph subset above, including multiple compute operations, fan-out/Concat, constant-weight convolution, and multiple public inputs/outputs. It does not select kernels by model name. Ranks 1–4, positive static extents, layouts, operation parameters and the 128 MiB per-buffer limit are checked. Unsupported operations, dynamic extents, non-FP32 data and unsupported layouts fail before publication. There is no CPU fallback.

The driver embeds the validated SPIR-V and deterministic schema-v1 plan in the model DSO. The manifest records plan/shader/`.param`/`.bin` hashes, host/device targets, public I/O and the DSO's actual direct `DT_NEEDED` entries. It also pins `host_implementation = "generic-plan-runtime"`, integer `host_codegen_version = 1`, and `execution_plan_usage = "embedded-runtime-required"`; host implementation and codegen version participate in both package build-identity domains. Public ABI version 1 is unchanged.

The generated wrapper parses its embedded plan during `create` and calls the generic `VulkanRuntimeContext` executor during `run`; the runtime walks the fixed kernel schedule and derives inter-kernel hazards from recorded reads/writes. The graph schedule and shader/resource decisions are already compiler-produced, but the host executor has not been compiled away. `NCNNVulkanRuntime`, `NCNNVulkanDeviceHelper` and `NCNNVulkanPlanSupport` are static archives included in the model library; static linkage or stripped symbols do not establish absence of an executor or LLVM code. The plan/payload/hash path uses LLVM Support. Shared-LLVM builds require their actual LLVM SONAME, while static-LLVM builds can carry LLVM code without a direct LLVM `DT_NEEDED` entry. The manifest's `dependencies` array is not a transitive-closure or static-source audit. The published plan/manifest are build diagnostics; the model reads its embedded plan rather than a deployment-side plan file.

The library exports only `<model>_vk_create`, `<model>_vk_run`, and `<model>_vk_destroy`; it has no RPATH/RUNPATH to the compiler build or temporary directory. Deployment must provide the manifest-listed runtime dependencies (including the configured LLVM shared runtime when that build links LLVM dynamically), Vulkan loader, system C/C++ runtime, and a compatible Vulkan 1.1 device/ICD. The generated header includes the public `ModelABI.h` contract installed by the compiler package.

`--verify-execution` makes `ncnn-compile` `dlopen` the staged DSO, create a Vulkan context, run deterministic buffers for every static input, check all output values are finite, and destroy the context before publication. This is a general ABI/lifecycle smoke check, not a mathematical oracle; independent CPU/Vulkan/golden tests establish numerical correctness. Device absence or runtime failure fails the requested verification and leaves prior output intact. Without this flag, compilation performs offline SPIR-V validation but does not require creating a device. CPU and Vulkan default output directories are separate; explicit output directories containing artifacts from the other backend are refused. Publication of header, plan, manifest and library uses the same staged sibling-directory replacement and rollback transaction as CPU artifacts.

## Standalone Linux host runtime

`include/ncnn-mlir/Vulkan/VulkanRuntime.hpp` and `lib/VulkanRuntime/` implement the V3 host execution core. `VulkanRuntimeContext::create` takes a typed schema-v1 plan, exactly one `(kernel ID, SPIR-V words)` payload per planned kernel, weight payloads keyed by resource ID, and runtime options. The runtime verifies the plan, binary IDs/magic, resource sizes, weight IDs, and plan weight SHA-256 identities before dispatch setup. Shader and weight payload spans are consumed synchronously by `create`; their backing storage may be released after it returns. Physical weight payloads must match the plan's physical layout; Conv2D weight verification maps either OHWI graph payloads or HWCF single-op fixture payloads back to logical OIHW before checking identity. Bias identity is checked directly. SPIR-V semantic validation remains an offline `spirv-val` responsibility.

The current host runtime accepts only `device_target = "vulkan1.1-spirv1.0"`, static FP32 storage buffers, and one-dimensional workgroups `[x,1,1]`; device-specific workgroup, descriptor, push-constant and storage-buffer limits are checked during create. Each storage buffer is limited by both the plan's 128 MiB minimum-range contract and the selected device's actual `maxStorageBufferRange`. The default aggregate context allocation cap is 512 MiB. It counts actual Vulkan memory requirement sizes for every device resource, persistent public-I/O staging buffer, and temporary weight staging buffer during upload; exceeding the cap fails create with `OUT_OF_MEMORY` before Vulkan memory allocation. This is a per-context cap, not a process-wide GPU-memory reservation; multiple contexts allocate independently and can still fail when the device lacks free memory. A plan's sum of logical buffer bytes is not a measured peak allocation. No resource aliasing is performed. `get_device_info()` reports the selected physical-device name plus Vulkan API/driver versions and vendor/device IDs; the optional device-name match filter is test-only and is not part of ABI v1.

Each context owns its Vulkan 1.1 instance, selected physical/logical device, compute queue, one reusable command pool/buffer/fence, per-resource device buffers, descriptor sets/layouts, shader modules and compute pipelines. Create is transactional: `*context` remains null on failure, and all successfully created Vulkan handles are released in reverse dependency order. Static weight and bias resources are staged and uploaded once during create. Every run copies caller input bytes to mapped host-visible staging, flushes mapped writes when required, records host→transfer and transfer→compute barriers, dispatches kernels in plan order, inserts compute write→compute read/write barriers for inter-kernel resource hazards, copies outputs to readback staging, applies transfer→host synchronization, waits on the reusable fence, invalidates mapped reads when required, and only then copies bytes to caller outputs. Buffer copy offsets are zero and all FP32 byte counts are four-byte aligned. Staging mappings cover the full allocation and flush/invalidate use `VK_WHOLE_SIZE`.

`run` requires exactly one non-null byte span of exact `byte_size` for each public input and output resource; caller memory must remain valid until `run` returns. All output synchronization and staging lookups finish before caller output publication, so reported failures leave valid caller output buffers unchanged. Errors map to the stable ABI status set: host/device/descriptor allocation failure→`OUT_OF_MEMORY`, device lost→`DEVICE_LOST`, unsupported device/target/limits→`UNSUPPORTED_DEVICE`, size mismatch→`BUFFER_SIZE_MISMATCH`, malformed package/options→`INVALID_ARGUMENT`, and other Vulkan failures→`API_ERROR`; diagnostics are bounded and NUL-terminated at the C ABI fixture boundary. Host tensor validation failures occur before execution and are retryable after correction. A context whose input staging, flush, command submission/wait, output invalidation or publication preparation fails is marked unusable; destroy it and create a new context before retrying. The existing failure injector includes mapped flush/invalidate operations only for tests, not as product ABI controls. Explicit destroy reports device-idle or validation-layer cleanup errors before releasing the context. Calls to one live context are serialized, including command-pool and queue access; separate contexts own independent mutable resources. Callers must synchronize destroy against every other context use.

The runtime library is configured when either `NCNN_ENABLE_VULKAN_PRODUCT=ON` or `NCNN_ENABLE_VULKAN_V0_TESTS=ON`; Vulkan build dependencies remain absent from the default CPU build. The device-test option requires a Vulkan loader/headers, SPIR-V tools, `vulkaninfo`, the Khronos validation layer, and the explicitly selected ICDs. `NCNN_VULKAN_TEST_DEVICES=llvmpipe` needs no NVIDIA ICD/EGL library; `nvidia` needs no software ICD; the default device list requests both. Unknown, duplicate or empty requested device lists and missing dependencies fail configuration. A requested but unusable device fails its test, rather than becoming a skip. Product-only offline tests deliberately use an unusable ICD and still require CPU/golden and validated package/failure checks. They also load the generated Vulkan DSO with its real header, require explicit failed create with a null context and bounded diagnostics (including capacity one), and check `destroy(NULL)`; no GPU run/output is claimed. `test/Vulkan/vulkan_runtime_abi_fixture.*` remains a test-only adapter; its fault injector is an in-process test hook and is not a product runtime control.

NVIDIA device CTests use a build-local manifest for the vendor-supported headless EGL frontend (`libEGL_nvidia.so.0`), retaining the installed manifest's API version and the same physical device/driver. The system manifest is not modified. Both Release and sanitizer jobs use this frontend with normal driver unloading: the GLX frontend in driver 615.71.09 leaks a libdrm device at unload, whereas the tested EGL path releases its resources. Tests do not pin the ICD or disable LeakSanitizer. This test environment selection does not add a product runtime dependency or change the application's ICD selection. See [NVIDIA's installed-components documentation](https://download.nvidia.com/XFree86/Linux-x86_64/550.40.07/README/installedcomponents.html) for the supported EGL/GLX Vulkan ICD frontends.

## Private device and resource helper

`NCNNVulkanDeviceHelper` is a separate static PIC archive implemented in
`lib/VulkanDeviceHelper/`. Its private C++ interface is
`include/ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp`; the shared error, operation and
device-info declarations in `VulkanDeviceTypes.hpp` require only standard headers
and `ModelABI.h`. `VulkanRuntime.hpp` does not transitively include Vulkan SDK
headers. None of these changes adds a public model ABI symbol or option.

The helper depends on Vulkan and standard/platform C++ facilities, not LLVM,
MLIR, GTest, execution-plan parsing or payload hashing. It owns device/queue,
buffer/memory/mapping, descriptor/pipeline and command/fence primitives. Its
inputs are explicit sizes, Vulkan bindings/ranges, shader words and entry points;
it accepts no execution plan, host program, model-resource ID or kernel schedule.
Copy/barrier/bind/dispatch order and hazard analysis remain in the generic
`VulkanRuntimeContext` executor. The product manifest therefore still says
`generic-plan-runtime` and `embedded-runtime-required`.

Buffer setup is two-phase. Preparation creates a `VkBuffer`, queries its actual
memory requirements and selects a late-bound memory type, without allocating
`VkDeviceMemory`. The runtime prepares every device, persistent I/O staging and
temporary weight staging buffer, checks the checked sum of requirement sizes and
allocation count, and only then allocates, binds and maps. Default context budget
is 512 MiB; storage-buffer size remains bounded by 128 MiB and the selected
device's actual range. Whole-allocation mappings and `VK_WHOLE_SIZE`
flush/invalidate preserve non-coherent atom-range correctness. Requirement-size
accounting is not a measurement of driver-resident peak memory.

Device selection preserves the existing enumeration policy: AUTO selects the
first Vulkan 1.1 compute-capable device matching the private test filter, then
the caller checks model limits on that device. A later limit failure does not
retry another device. Explicit indices are enumeration ordinals. No device UUID,
queue family, memory type or allocation size is embedded from the build machine.

Resources are move-only RAII objects. The owner waits for device work and
releases child resources before the parent device/instance; partial construction
and repeated cleanup release only acquired handles. The helper reports stable
Vulkan status errors; the runtime retains run serialization, retryable host
argument errors, execution poison and all-output publication protection. Fault
hooks and fake Vulkan entry points are private test mechanisms, not C ABI or
environment controls.

A helper-only native harness and standalone CMake subproject do not link the
runtime, plan support, LLVM/MLIR or GTest. Default no-device tests cover synthetic
limits, actual-requirement budgets, allocation/cleanup failures and mapped-memory
ranges; explicitly selected llvmpipe/NVIDIA tests execute a bounded Add shader
through the helper's resource/pipeline/command primitives. Independent evidence
checks actual header dependencies, compile arguments, target interfaces, archive
members and native link inputs. This helper proof is separate from the model
DSO's still-present compiler/plan dependencies and from driver-internal LLVM.

## Qualification and evidence boundary

The runtime's debug-utils callback counts ERROR messages only when its internal validation option is enabled. Product ABI v1 does not expose that option or a device-identity/telemetry query. Product device tests instead force `VK_LAYER_KHRONOS_validation` and reject validation errors in the captured process logs. A default callback count of zero is not evidence that validation was enabled. Physical-device records include API/driver/vendor/device/type and raw `vulkaninfo` output; layer names or software devices cannot substitute for hardware evidence.

The qualification tools and tier-specific CTests are documented in [the Vulkan test README](../test/Vulkan/README.md). `run_gate.py` always creates a fresh stage, rebuilds CPU/Vulkan fixtures, runs global format/tidy and unfiltered CTest, and preserves raw results. The evidence checker validates scope and identities before producing a release-eligibility decision; software-only or offline-only success is not sufficient. Existing model-specific numerical budgets remain unchanged. Eligibility is restricted to the static FP32 support matrix and devices/ICDs/drivers actually present in the evidence, not a general Vulkan compatibility certification. `--verify-execution` remains a lifecycle smoke and cannot replace the independent CPU/Vulkan/golden checks. Historical committed evidence snapshots are not current-stage qualification.

## Vulkan execution-plan schema

`ncnn-mlir/Support/VulkanExecutionPlan.hpp` defines the separate typed schema, serializer, deserializer, and verifier. It is not the CPU `EmitModelPlan` JSON or the CPU C ABI manifest. The current CPU execution-plan pipeline does not emit this Vulkan schema.

The version-1 JSON kind is `ncnn.vulkan_execution_plan`. Its top-level fields are `schema_version`, `kind`, `model`, `build_identity`, `host_triple`, `device_target`, `resources`, `kernels`, `inputs`, `outputs`, and `plan_hash`. Resources describe a unique ID, role, FP32 static shape/byte size, logical and physical layouts, optional host layout and an optional structured layout transform (`from_layout`, `to_layout`, `operation`; host-to-device for inputs and device-to-host for outputs), optional weight SHA-256 identity, and first/last kernel indices. Kernels are stored in execution order and describe unique ID, operation, SPIR-V entry point, workgroup size, resource reads/writes, descriptor set/binding/access records, and push-constant bytes by offset. Public I/O maps to resource IDs.

The verifier rejects unknown schema versions/JSON fields, duplicate IDs or descriptor locations, malformed weight identities, unchecked/mismatched byte sizes, sizes over the Vulkan 1.1 minimum storage-buffer range, missing host/device layout mappings, invalid workgroup/push-constant bounds, read-before-write, writes to inputs/weights, incomplete descriptor coverage, invalid public I/O roles, and lifetimes that do not exactly cover ordered uses. An unused public input is the sole exception: it has no kernel accesses and uses `UINT32_MAX` for both lifetime indices; all other resources must be used by at least one ordered kernel. JSON serialization is deterministic; `plan_hash` is the 16-digit lowercase hexadecimal `stableHash64` of the canonical plan JSON with the `plan_hash` field omitted. Deserialization validates the stored hash. Kernel schedule records are verified with synthetic unit fixtures; this schema alone does not create executable shaders, Vulkan resources, or dispatch code.
