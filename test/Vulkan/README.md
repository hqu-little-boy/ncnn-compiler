# Vulkan qualification tests

The suite tests the explicitly enabled Vulkan product backend as well as its
isolated shader and runtime components. CPU remains the default backend and its
ABI has no Vulkan dependency. The authoritative model, layout, ABI and deployment
subset is the [Vulkan backend contract](../../docs/ncnn-vulkan-backend-contract.md).
There is no transparent CPU fallback.

## Independent test tiers

| Tier | CMake options | Execution and dependencies |
|---|---|---|
| Offline | `NCNN_ENABLE_VULKAN_PRODUCT=ON`, `NCNN_ENABLE_VULKAN_V0_TESTS=OFF` | CPU AOT/golden regression, legality/lowering, SPIR-V validation, package/ELF checks and failure paths. Requires compiler dependencies and Vulkan headers/loader, not a working ICD or GPU. |
| Software | Both options `ON`, `NCNN_VULKAN_TEST_DEVICES=llvmpipe` | The offline tests plus explicitly selected llvmpipe ABI/lifecycle/numerical tests. Requires its ICD, `vulkaninfo` and validation layer; does not require NVIDIA libraries. |
| Hardware | Both options `ON`, `NCNN_VULKAN_TEST_DEVICES=nvidia` | The offline tests plus the same device chain on the selected NVIDIA ICD. Requires its manifest, EGL frontend, actual GPU and validation layer. |
| Full device matrix | Both options `ON`, `NCNN_VULKAN_TEST_DEVICES=nvidia;llvmpipe` | Both device tiers. This is the default device list when the device-test option is enabled. |

Both product and device-test options default to `OFF`. Device selection is
explicit: an invalid, duplicate or empty requested device list is an error.
Missing requested dependencies or devices fail instead of silently dropping or
skipping tests. Software results do not establish hardware compatibility or
performance. Tests retain the `vulkan` label and add `offline`, `software` or
`hardware` labels where applicable.

From the parent directory containing `compiler/`, a focused software setup is:

```sh
cmake -S compiler -B /tmp/ncnn-vulkan-software-dev \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DNCNN_ENABLE_VULKAN_PRODUCT=ON -DNCNN_ENABLE_VULKAN_V0_TESTS=ON \
  -DNCNN_VULKAN_TEST_DEVICES=llvmpipe -DCOMPILER_INSTALL_RPATH=ON
cmake --build /tmp/ncnn-vulkan-software-dev --parallel 4
cmake --build /tmp/ncnn-vulkan-software-dev \
  --target numerical_tests numerical_dynamic_operator_tests numerical_dynamic_tests \
  --parallel 4
ctest --test-dir /tmp/ncnn-vulkan-software-dev --output-on-failure
```

Filtered runs are useful for development but are not full engineering gates.
For qualification, create a new build directory; never count an old generated
model `.so` as evidence for a changed compiler.

## Shader, runtime and product coverage

- `VulkanRuntimeTest.*`: all five device-independent plan/package/budget,
  injected instance-creation and status-name cases run whenever the product or
  device tests are enabled, including product-only offline configurations.
  Every gate profile and the V6.1/V6.2/V6.4 evidence mappings require them.
- `vulkan-v0-add-*`: `gpu_func_add_probe.mlir` supplies the 257-element FP32 Add
  kernel with StorageBuffers, local size `[64,1,1]` and tail bounds protection.
  MLIR `--test-vulkan-runner-pipeline` must produce `gpu.binary`; extraction,
  residual IR checks and `spirv-val --target-env vulkan1.1` precede dispatch.
  CPU AOT, GPU and independent scalar golden use identical inputs. Negative
  cases require the expected diagnostic, not just any nonzero exit, followed by
  recovery. `add.mlir` is only a historical direct-SPIR-V experiment.
- `vulkan-v2-*`: single-op Add, ReLU, copy, identity, transpose, reshape and
  restricted Conv2D variants. Seeds `0,1,42`, boundary inputs, plan/binding/layout
  checks and logical OIHW/device HWCF weight identities are recorded.
- `vulkan-v3-runtime-*`: persistent Add→ReLU and weighted Conv2D contexts;
  changing inputs, alternating contexts, same-context concurrent calls, ABI
  errors, create/budget/submit/wait failure injection and cleanup. The direct
  Add→Split→ReLU dual-output fixture additionally compares sum and ReLU against
  its CPU AOT and golden, injects input flush and second-output invalidate
  failures, checks every sentinel, and verifies poison/isolation/recreation.
  Forced mapped-memory sync exercises flush/invalidate even on coherent memory;
  it is not evidence that non-coherent physical memory was selected.
- `ncnn-compile-vulkan-offline`: an unusable ICD is imposed in subprocesses.
  CPU AOT runs in a CPU-only native loader, not inside Python; sanitizer builds
  instrument the loader and model with the same shared ASan/UBSan configuration
  and disable recovery after reports without suppressing leak detection. Initial
  and post-failure CPU calls retain separate command, result and raw-stream
  records. Vulkan compilation without `--verify-execution` still validates and
  audits the package, and shader/tool/link/audit/device verification failures
  preserve successful output. It has no GPU numerical claim.
- `ncnn-compile-vulkan-product`: real generated-header/DSO lifecycle,
  CPU/Vulkan/golden ReLU comparison, deterministic identities, export/dependency
  audit and transactional failure paths. The configured ICD is explicit.
- `vulkan-v5-{multiop,pooling-softmax,squeezenet}-*`: generated multi-I/O product
  ABI and complete static-graph execution. Every trial compares newly compiled
  CPU AOT and Vulkan products against vendored ncnn and, for small fixtures,
  independent goldens; multiple contexts and destroy/recreate are checked.
  SqueezeNet retains its fixed elementwise, finite/probability-sum and top-k
  budgets.

The C ABI stub test is not a substitute for loading a product DSO. Likewise,
`--verify-execution` is a finite-output lifecycle smoke, not a numerical oracle.
NVIDIA tests use the vendor-supported headless EGL ICD in a build-local manifest;
no system manifest is modified and normal library unloading is retained.

## Fresh engineering gates and evidence

`run_gate.py` creates a unique build directory on every invocation and runs
configure, global `format_check`, complete build, all three numerical targets,
global `tidy`, and unfiltered CTest. All build commands use `--parallel`. It records
raw logs, source content identity (including dirty/new files and submodules),
configuration/cache/tool identities, pre-run CTest inventory, JUnit results and
a preserved `LastTest.log`. Compiler/tidy project warnings and sanitizer reports
are failures. It does not apply source formatting automatically.

```sh
python3 compiler/test/Vulkan/run_gate.py --profile release --archive
python3 compiler/test/Vulkan/run_gate.py --profile offline --archive
python3 compiler/test/Vulkan/run_gate.py --profile software --archive
```

The `hardware` profile selects only NVIDIA; `release` selects both ICDs. The
`sanitize` profile also selects both and requires a matching static PIC
ASan+UBSan LLVM/MLIR installation, its actual Vulkan runner test tools and lit,
and a leak-clean isolated OpenMP runtime:

```sh
python3 compiler/test/Vulkan/run_gate.py --profile sanitize --jobs 1 --archive \
  --sanitizer-install /path/to/static-asan-llvm/install \
  --lit /path/to/llvm-source/llvm/utils/lit/lit.py \
  --openmp-dir /path/to/libomp-only \
  --toolchain-identity /path/to/toolchain-identity.json
```

The sanitizer gate audits host executables for shared ASan and absence of core
LLVM/MLIR DSOs. Inherited sanitizer suppression/cleanup overrides are rejected.
The fixture compiler must not recover after sanitizer reports. A budget
exception is accepted only for the two named PP OCR v5 fixtures, at the existing
450-second authorization ceiling, and requires `--budget-evidence FIXTURE=JSON`
with a same-source fingerprint and the hash-matched raw 300-second hard-timeout
log. The record and log are copied into the gate stage/archive. Release budgets,
mathematical oracles, CTest, leak detection and sanitizer checks remain
unchanged; an unrelated failure or a stale/different-source timeout is not a
budget waiver.

Stage evidence is generated under:

- `test/Native/vulkan-v1-cpu-baseline/` for CPU non-regression;
- `test/Native/vulkan-product-{offline,cli}/` for product tests;
- `test/Vulkan/evidence/gpu-func/<device>/`, `v2/<case>/`, `v3/<device>/`, and
  `v5/<suite>-<device>/` for device chains.

Device jobs preserve physical-device API/driver/vendor/device/type and raw
`vulkaninfo` summaries, validation outputs, model/plan/shader identities, fixed
inputs, seeds, per-output numerical results and failure records. Runtime-specific
measured counters are not public product ABI telemetry. A model's plan byte sum
is not a measured allocation peak; absent measurements must remain marked as
such. Committed `evidence/` snapshots are historical, not fresh qualification.

After CTest, `check_release_evidence.py` validates stage scope and associates
step IDs with actual tests and CPU/Vulkan/golden records. Its synthetic regression
suite is registered independently in CTest; the complete collector is not a
CTest dependency on itself. A combined decision can be requested with:

```sh
python3 compiler/test/Vulkan/check_release_evidence.py \
  --stage /tmp/current-release-stage --stage /tmp/current-sanitize-stage \
  --stage /tmp/current-offline-stage --stage /tmp/current-software-stage \
  --require-release --output /tmp/vulkan-release-evidence.json
```

Missing, modified, stale or unexecuted evidence, a numerical/validation error,
false hardware identity, mismatched source identities or an unexpected skip
blocks qualification. Existing configured PerformanceModel skips are explicitly
listed, not counted as numerical or device passes. A single offline/software/
hardware report is not a release decision. The supported model/device scope
remains bounded even when all gates pass.

`--archive` writes a stage-adjacent evidence tarball on success or failure with a
file-hash manifest, raw gate records and model evidence. A sanitizer archive also
contains the verified toolchain identity, its configuration/build/smoke logs and
the patch records used for the isolated OpenMP runtime. It excludes unrelated
full numerical build trees. Logs are preserved before any CTest enumeration can
rewrite `LastTest.log`; moving a bundle must not silently reuse absolute paths
from a previous build. `.github/workflows/vulkan.yml` uses the same fresh runner
for hosted offline/software jobs and an explicitly requested self-hosted NVIDIA
job. Unrequested hardware coverage is absent, not passed; these CI tier jobs do
not by themselves supply the separate complete sanitizer/release decision.
