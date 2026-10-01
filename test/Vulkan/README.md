# Vulkan device and runtime tests

This is an opt-in Vulkan test suite, not a product backend. It covers V0 feasibility, V2 single-operation lowering, and the standalone V3 host runtime. The compiler's default CPU pipeline and C ABI do not depend on Vulkan. V0 lowers the fixed FP32 `Add` GPU fixture with MLIR's Vulkan runner pipeline, extracts and validates the SPIR-V, dispatches it, and compares output against the CPU AOT result and an independent scalar golden.

## Enable and run

The opt-in configuration requires MLIR/LLVM 21, Vulkan headers/loader, NVIDIA and llvmpipe ICD manifests, `spirv-val`, `spirv-dis`, `vulkaninfo`, and the Khronos validation layer:

```sh
cmake -S compiler -B /tmp/ncnn-v0-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DNCNN_ENABLE_VULKAN_V0_TESTS=ON \
  -DCOMPILER_ENABLE_FORMAT_TARGETS=OFF \
  -DCOMPILER_INSTALL_RPATH=ON
cmake --build /tmp/ncnn-v0-build --target vulkan-v0-build --parallel
ctest --test-dir /tmp/ncnn-v0-build \
  -R '^vulkan-v0-add-(nvidia|llvmpipe)$' --output-on-failure
```

The NVIDIA test is the real-device gate and runs negative cases; the llvmpipe test forces the software ICD and is only functional/numerical coverage. The host runner has no CPU fallback. If a required tool or device is absent, the opt-in test fails with a diagnostic rather than silently skipping.

## GPU-dialect kernel and lowering

`gpu_func_add_probe.mlir` is the positive V0 kernel source: a static 257-element FP32 Add with three `StorageBuffer` memrefs, 64×1×1 workgroup, and explicit tail bounds check. Its module target environment is Vulkan 1.1 with SPIR-V 1.0 plus `SPV_KHR_storage_buffer_storage_class`.

The opt-in harness runs MLIR 21 `--test-vulkan-runner-pipeline`, which lowers the `gpu.func` and emits a `gpu.binary` object. `vulkan_v0_spirv_extract` writes the embedded SPIR-V bytes for `spirv-val`, checks/disassembly, and the existing Vulkan host runner. The test requires the converted IR to contain `gpu.binary @v0_add` and no `gpu.func`, `memref.load/store`, or unrealized conversion casts; SPIR-V must expose the `add` GLCompute entry point, local size `[64,1,1]`, and set 0 bindings 0–2. Lowering or extraction failure is a hard test failure—there is no fallback to the older hand-authored shader.

`add.mlir` is retained only as the first direct-SPIR-V-dialect experiment and historical comparison. It is not used to satisfy the current GPU-dialect provenance gate.

## Evidence

CTest writes generated artifacts to `<build>/test/Vulkan/evidence/gpu-func/<device>/`: GPU-binary MLIR, extracted `.spv`, disassembly, validator and tool logs, fixed input/output bytes, CPU library identity, device properties, tolerance and per-element comparison summaries. The earlier `evidence/nvidia/` and `evidence/llvmpipe/` snapshots are the 2026-09-28 direct-SPIR-V-dialect run; preserve them as historical evidence. The new GPU-dialect provenance run is archived separately under `evidence/gpu-func/{nvidia,llvmpipe}/`. It uses the same 257-element input contract; the device runs reported zero validation errors and zero CPU/GPU/golden difference under `atol=1e-6, rtol=1e-6`.

## V2 isolated operation tests

With `NCNN_ENABLE_VULKAN_V0_TESTS=ON`, the default build also creates the V2 CPU AOT fixtures and registers `vulkan-v2-*` tests. `run_v2.py` imports the `.param/.bin` fixture with `ncnn-mlir-driver`, runs the Vulkan preparation/lowering pipeline, writes and verifies the execution plan, lowers the resulting GPU module to `gpu.binary`, extracts SPIR-V, runs `spirv-val --target-env vulkan1.1`, and dispatches each operation on NVIDIA and llvmpipe. Each device case runs fixed edge values plus seeds `0,1,42`, comparing GPU results against the matching CPU AOT artifact and an independent scalar golden; evidence is written under `evidence/v2/<case>/`.

The test matrix covers static FP32 Add, ReLU, copy, identity, transpose, reshape, and restricted 1x1/3x3 Conv2D with stride/padding/bias variants. Conv tests hash-check the schema-v1 plan’s HWCF weights and bias against the ncnn binary stream; the runner converts source OIHW weights to the device buffer order and checks CHW outputs against CPU AOT/golden. V2 tests retain the one-shot test runner and do not exercise persistent model contexts.

## V3 persistent host runtime

V3 builds the opt-in `NCNNVulkanRuntime` library and two runtime-backed integration fixtures. `run_v3.py` first generates and validates the existing V2 Add, ReLU, and Conv2D SPIR-V/plan packages, then calls the plan-driven V3 runtime directly. The Add→ReLU fixture dispatches two kernels in one context, exercises a shared intermediate/barrier, repeated changing inputs, serialized concurrent calls, alternating independent contexts, bounded diagnostics/status validation, injected create/submit/wait failures, failed-output preservation, and post-failure recovery. It compares each tested run against a freshly built CPU AOT model for the same `.param/.bin` graph and a scalar golden.

The Conv2D fixture builds a persistent context with one V2 kernel and two static FP32 weight resources. It checks OIHW-to-HWCF weight identity, verifies each weight payload is uploaded exactly once during create, repeats inference with changed inputs, and compares each output against the same fixture's CPU AOT and scalar golden. Runtime test options force the mapped-memory flush/invalidate call path so it is exercised even on coherent host memory; actual non-coherent memory is used when selected by the device. Both fixtures require validation-layer error count zero and run on explicitly selected NVIDIA and llvmpipe ICDs; evidence records the physical device/driver identity, execution-plan and SPIR-V hashes, CPU library identity, input/output hashes, numerical errors, upload count, and peak context memory. Absence of either required device is a test failure, not a skip.

The V3 runtime build is still test-option gated and is not linked into `ncnn-compile` products. A typical focused development invocation is:

```sh
cmake --build /tmp/ncnn-vulkan-build --target vulkan-v3-build --parallel
ctest --test-dir /tmp/ncnn-vulkan-build \
  -R '^vulkan-v3-runtime-' --output-on-failure
```

The V3 device evidence (plans, SPIR-V, inputs/outputs, hashes, CPU/GPU/golden errors, and runtime resource counters) is written under `<build>/test/Vulkan/evidence/v3/<device>/`.
