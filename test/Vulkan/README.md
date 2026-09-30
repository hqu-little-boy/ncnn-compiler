# Vulkan V0 device spike

This is an opt-in feasibility test, not a product backend. The compiler's default CPU pipeline and C ABI do not depend on Vulkan. The test creates a fixed FP32 `Add` CPU AOT artifact with `ncnn-compile`, lowers the `gpu.func` fixture with MLIR's Vulkan runner pipeline to a serialized SPIR-V object, extracts and validates that object, then dispatches it through the Vulkan loader and compares every output against both the CPU AOT result and an independent scalar golden.

## Enable and run

The opt-in configuration requires MLIR/LLVM 21, Vulkan headers/loader, a Vulkan ICD, `spirv-val`, `vulkaninfo`, and the Khronos validation layer:

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

The test matrix covers static FP32 Add, ReLU, copy, identity, transpose, reshape, and restricted 1x1/3x3 Conv2D with stride/padding/bias variants. Conv tests hash-check the schema-v1 plan’s HWCF weights and bias against the ncnn binary stream; the runner converts source OIHW weights to the device buffer order and checks CHW outputs against CPU AOT/golden. This is test-only code generation and dispatch: the current product compiler still does not emit a Vulkan model library or runtime wrapper.
