# Vulkan V0 device spike

This is an opt-in feasibility test, not a product backend. The compiler's default CPU pipeline and C ABI do not depend on Vulkan. The test creates a fixed FP32 `Add` CPU AOT artifact with `ncnn-compile`, lowers an explicit SPIR-V dialect MLIR module to SPIR-V, validates it, then dispatches it through the Vulkan loader and compares every output against both the CPU AOT result and an independent scalar golden.

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

## Kernel and current lowering boundary

`add.mlir` defines 257 FP32 inputs per operand, three storage-buffer bindings, a 64-invocation workgroup, and an explicit bounds check for the five dispatched workgroups. It targets SPIR-V 1.3 / Vulkan 1.1 and uses only core Shader, StorageBuffer, and FP32 functionality.

The executable kernel is currently authored in the SPIR-V dialect. `gpu_func_add_probe.mlir` separately reproduces the MLIR 21 GPU-dialect lowering attempt: `mlir-opt-21 gpu_func_add_probe.mlir '--spirv-attach-target=module=v0_add ver=v1.1 caps=Shader client_api=Vulkan' --convert-gpu-to-spirv --verify-each` currently fails with `failed to legalize operation 'memref.load'`. The opt-in CTest records this probe's result and diagnostic with each evidence run. Therefore this spike proves the MLIR SPIR-V serialization and Vulkan host/device execution path, **not** that the project's Linalg or a resource-using `gpu.func` can already be lowered. That remains a V0/V1 boundary to resolve before claiming an ncnn Vulkan backend.

## Evidence

CTest writes generated artifacts to `<build>/test/Vulkan/evidence/<device>/`: lowered MLIR, `.spv`, validator and tool logs, fixed input/output bytes, CPU library identity, device properties, tolerance and per-element comparison summaries. The `evidence/nvidia/` and `evidence/llvmpipe/` directories in this source tree are the archived 2026-09-28 run. The input has 257 elements; both devices reported zero validation errors and zero CPU/GPU/golden difference under `atol=1e-6, rtol=1e-6`.
