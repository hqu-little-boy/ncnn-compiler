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

- `vulkan-device-helper-unit`, `vulkan-device-helper-no-device`,
  `vulkan-device-helper-independence` and
  `vulkan-device-helper-evidence-tests` are registered whenever product or device
  support is enabled, including product-only/offline configurations. The native
  harness and standalone subproject link the private helper and Vulkan/platform
  libraries, not Runtime, execution-plan support, LLVM/MLIR or GTest. Synthetic
  properties and test-only Vulkan failures cover limits, actual-requirement
  budgets/counts, partial cleanup and whole-allocation mapped synchronization.
  A product-only configure/inventory/execution check guards against registration
  omissions; an unavailable ICD requires explicit create failure, not a skip.
- `vulkan-device-helper-add-{nvidia,llvmpipe}` execute the 257-element bounded
  Add fixture through the helper's buffer, descriptor/pipeline and command/fence
  primitives with validation enabled. Shader generation and offline `spirv-val`
  remain build-time compiler operations, outside the native helper's dependency
  boundary. Same-source CPU AOT/scalar golden, raw tensors and actual device
  identity accompany helper results. These are primitive tests, not evidence
  that a generated model has switched away from the generic executor.
- `VulkanRuntimeBudgetTest.*` is a separate executor-plus-fake-Vulkan GTest
  target, not the helper-only harness. It verifies the runtime's real preparation
  path checks all actual requirement sizes, persistent I/O staging, temporary
  weight staging and allocation count before the first `vkAllocateMemory`, and
  releases partial allocations after bind failure. Its legal plan/LLVM/GTest
  dependencies are not part of the helper-independence proof.
- `VulkanRuntimeTest.*`: all five device-independent plan/package/budget,
  injected instance-creation and status-name cases run whenever the product or
  device tests are enabled, including product-only offline configurations.
  Every gate profile requires them, the ten `VulkanProductTest.*` package/host
  contract cases and ten `VulkanExecutionPlanTest.*` cases. Required inventories
  use test identities rather than a total-count threshold; a product-only omission
  is not a passed gate.
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
  preserve successful output. A second real-header native loader also opens the
  generated Vulkan DSO under the unavailable ICD, checks explicit create failure,
  null context, bounded diagnostics (including capacity one), and `destroy(NULL)`.
  It publishes no GPU output and has no GPU numerical claim.
- `ncnn-compile-vulkan-product`: real generated-header/DSO lifecycle,
  CPU/Vulkan/golden ReLU comparison, deterministic identities, export/dependency
  audit and transactional failure paths. The configured ICD is explicit. Release
  repeated product artifacts must match byte-for-byte; sanitizer-instrumented DSO
  comparison may canonicalize only the unique `ncnn-compile-*/vulkan_model_wrapper.bc`
  path recorded by Clang's sanitizer module metadata. Headers, manifests, plans,
  and every other DSO byte remain hash-checked; any difference beyond that
  compiler-generated temporary path blocks the test.
- `vulkan-v5-{multiop,pooling-softmax,squeezenet}-*`: generated multi-I/O product
  ABI and complete static-graph execution. Every trial compares newly compiled
  CPU AOT and Vulkan products against vendored ncnn and, for small fixtures,
  independent goldens; multiple contexts and destroy/recreate are checked.
  SqueezeNet retains its fixed elementwise, finite/probability-sum and top-k
  budgets. When release-package and instrumented SAN toolchains emit different
  SPIR-V bytes for a checked graph, VA0 preserves both raw hashes and compares the
  disassembled instructions. Only order/operand permutations of the pure
  `OpULessThan`, `OpUGreaterThanEqual`, and commutative `OpLogicalAnd` guard
  expressions are canonicalized; control flow, resource interfaces, other
  instructions, and actual CPU/Vulkan/golden outputs remain exact/fully checked.
  Any other shader semantic difference blocks the baseline.

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

To qualify a frozen host-runtime baseline, add
`--legacy-baseline-stage /path/to/frozen-release-stage` to each profile invocation.
The snapshot preserves the old gate/source identity and the ReLU, Add→ReLU,
Conv→ReLU and SqueezeNet product fixtures for both ICDs; old results are labelled
`legacy-reference`, not counted as fresh current execution. Current products
remain `generic-plan-runtime` with ABI v1. Each profile validates its own baseline
scope; after all four fresh stages, use `--require-va0-baseline` on the composite
checker command to require the complete baseline/coverage matrix. Without a
reference, ordinary CI retains its existing functional-profile eligibility and
reports the host-baseline evidence as incomplete. Functional Vulkan eligibility
never means model-specific host AOT: `host_aot_release_eligible` is false.

Compiler-only host-program qualification is explicit and separate from runtime
qualification. `--stage-id VA1` retains `host_execution_role="legacy-product"`
and `host_aot_candidate=false`: GPU results still execute the generic runtime,
not the diagnostic host program. The product-only offline configuration also
registers `VulkanHostProgramTest.*`, `VulkanShaderInterfaceTest.*`, and
`vulkan-host-program-evidence-tests`; enabling device tests is not a prerequisite
for their registration. Their canonical identity, typed interface, lifetime,
RAW/WAW/WAR, repeated-output, tail/overflow and tamper tests are default CTests.

```sh
python3 compiler/test/Vulkan/run_gate.py --stage-id VA1 \
  --profile release --archive --legacy-baseline-stage /path/to/frozen-release-stage
python3 compiler/test/Vulkan/run_gate.py --stage-id VA1 \
  --profile sanitize --jobs 1 --archive \
  --sanitizer-install /path/to/static-asan-llvm/install \
  --lit /path/to/llvm-source/llvm/utils/lit/lit.py \
  --openmp-dir /path/to/libomp-only \
  --toolchain-identity /path/to/toolchain-identity.json \
  --legacy-baseline-stage /path/to/frozen-release-stage
python3 compiler/test/Vulkan/check_release_evidence.py \
  --stage /tmp/current-va1-release-stage --stage /tmp/current-va1-sanitize-stage \
  --require-va1-planner --output /tmp/va1-planner-evidence.json
```

These gates use distinct fresh `ncnn-compiler-stage-va1-<profile>-*` directories.
`va1_planner_eligible` requires both complete Release and matching static SAN
profiles, equal source fingerprints, all required executed tests, validated
native program evidence, and hash-revalidated archives. It does not require the
four-profile host-AOT product-switch matrix and cannot make
`host_aot_release_eligible` true. The separate `--require-release` flag retains
its four-profile legacy VA0 release meaning; it is not the two-profile VA1
planner qualifier, and omitting it here does not skip any Release build/test
step. Missing, skipped, stale, altered or unexecuted planner evidence fails closed. A final source change requires new full gates.
Native product evidence includes `host-program-evidence.json`, raw shader/weight
bytes and a frozen plan-derived recorder trace. The independent checker binds
those bytes and the manifest diagnostic to the same plan, reproduces canonical
identity and fixed commands, and normalizes only ordering inside independent
barrier batches; dispatch/copy/publication order remains checked. See the
[host-program format contract](../../docs/ncnn-vulkan-backend-contract.md#compiler-only-host-program).

Device-helper qualification uses `--stage-id VA2` and the same fresh full
Release and matching static sanitizer profiles. It requires the helper-only
native tests, two explicitly selected ICD positives, no-device/product-only
registration checks, and independent header/target/archive/link provenance.
The root stage's production helper archive and the standalone build are both
checked; a successful GTest or old runtime result cannot replace helper evidence.
The default independence CTest also runs in an ordinary CMake/CTest tree without
`gate-state.json`: it performs the same technical artifact checks but records an
unqualified, unbound result. Existing non-VA2 stages and offline/software/hardware
profiles retain their own identities; they do not acquire VA2 qualification just
by executing the artifact audit. The post-CTest stage collector, not the
independence test itself, requires the complete current test-result matrix.

```sh
python3 compiler/test/Vulkan/run_gate.py --stage-id VA2 \
  --profile release --archive --legacy-baseline-stage /path/to/frozen-release-stage
python3 compiler/test/Vulkan/run_gate.py --stage-id VA2 \
  --profile sanitize --jobs 1 --archive \
  --sanitizer-install /path/to/static-asan-llvm/install \
  --lit /path/to/llvm-source/llvm/utils/lit/lit.py \
  --openmp-dir /path/to/libomp-only \
  --toolchain-identity /path/to/toolchain-identity.json \
  --legacy-baseline-stage /path/to/frozen-release-stage
python3 compiler/test/Vulkan/check_release_evidence.py \
  --stage /tmp/current-va2-release-stage --stage /tmp/current-va2-sanitize-stage \
  --require-va2-helper --output /tmp/va2-helper-evidence.json
```

`va2_helper_eligible` requires both same-source complete profiles and
hash-revalidated archives. It does not change the four-profile `--require-release`
meaning or make `host_aot_release_eligible` true. Model execution remains
`legacy-product` / `generic-plan-runtime`; helper independence does not establish
absence of model-side LLVM or an executor. Per-test raw records and the
independent audit are under `test/Vulkan/evidence/va2/`; missing, skipped,
contaminated or tampered records fail closed.

### Private lifecycle fixtures and VA3 qualification

The compiler-only `makeVulkanLifecycleSource` API emits private model-specific
initialization/destruction fixtures; it does not replace the public
`--backend=vulkan` product. `run` keeps its ABI declaration but rejects a valid
call with `API_ERROR` and leaves caller output unchanged. No legacy executor is
used for candidate computation, and these tests produce no candidate inference
results. See the [lifecycle contract](../../docs/ncnn-vulkan-backend-contract.md#private-model-lifecycle-code-generation).

With `BUILD_TESTING=ON` and `NCNN_ENABLE_VULKAN_PRODUCT=ON`, the build emits
unweighted and weighted fixture sources/headers under
`test/Vulkan/native/lifecycle/{unweighted,weighted}/`, validates their
`shader_0.spv`, and builds ordinary and validation-enabled DSOs. The compiler-side
emitter may link LLVM/MLIR/GTest; the candidate libraries instead link their
emitted object, the device helper, Vulkan and standard/platform libraries. They
are test artifacts, not installed or transactionally published products.

Default non-device coverage includes every `VulkanLifecycleCodegenTest.*` case,
`vulkan-lifecycle-fake-unit`, `vulkan-lifecycle-no-device`,
`vulkan-lifecycle-independence`, and `vulkan-lifecycle-evidence-tests`. Product-only
configurations retain these tests when `NCNN_ENABLE_VULKAN_V0_TESTS=OFF`. The
selected device matrix additionally registers `vulkan-lifecycle-nvidia` and/or
`vulkan-lifecycle-llvmpipe`. A C11 loader uses the generated header and only the
three C ABI symbols. The same validation-enabled candidate DSO bytes are used on
both selected ICDs within a profile. The C ABI selects AUTO or an enumeration
ordinal, not a device-name string. The harness's `--device-match` value records an
expectation only; the capture runner binds the selected ICD, actual `vulkaninfo`
identity and ordinal passed to the harness, and checks its recorded
`requested_device_index`. A label alone is not device-selection evidence.

Codegen cases cover deterministic source/header identity, sparse slots and
bindings, exact embedded weight bytes, string/UTF-8 escaping, multiple ports and
weights, standard zero-length arrays, budget/cleanup ordering, and malformed
program/header/shader/weight rejection. Fake Vulkan records allocation-before-
budget failures, partial-create cleanup, actual copy recording and completed
upload counts, temporary staging release, and safe cleanup after a recoverable
idle-wait failure. Requirement-byte accounting is not device-resident memory
telemetry. The real C harness records `candidate_execution="not_run"` and
`upload_count="not_measured"`; fake counters and legacy numerical outputs must
not be relabelled as real candidate inference or device telemetry.

```sh
python3 compiler/test/Vulkan/run_gate.py --stage-id VA3 \
  --profile release --build-root /tmp --jobs 4 --tidy-jobs 4 --archive \
  --legacy-baseline-stage /path/to/frozen-release-stage
python3 compiler/test/Vulkan/run_gate.py --stage-id VA3 \
  --profile sanitize --build-root /tmp --jobs 1 --tidy-jobs 4 --archive \
  --sanitizer-install /path/to/static-asan-llvm/install \
  --lit /path/to/llvm-source/llvm/utils/lit/lit.py \
  --openmp-dir /path/to/libomp-only \
  --toolchain-identity /path/to/toolchain-identity.json \
  --legacy-baseline-stage /path/to/frozen-release-stage
python3 compiler/test/Vulkan/check_release_evidence.py \
  --stage /tmp/current-va3-release-stage --stage /tmp/current-va3-sanitize-stage \
  --require-va3-lifecycle --output /tmp/va3-lifecycle-evidence.json
```

`--stage-id VA3` uses fresh `ncnn-compiler-stage-va3-<profile>-*` directories and
retains the full configure/format/build/three-numerical-target/tidy/unfiltered-
CTest gate. The composite lifecycle qualifier requires same-source complete
Release and matching static ASan+UBSan profiles, both ICD lifecycle positives,
all source-declared codegen cases and native tests, actual source/object/archive/
compile/link/ELF evidence, and hash-revalidated archives. Per-test records and the
post-CTest aggregate live under `test/Vulkan/evidence/va3/`. Missing, skipped,
stale or altered observations fail closed. Ordinary CMake/CTest artifact audits
remain usable without `gate-state.json`, but do not by themselves qualify a
stage.

The qualifier covers VA3.1–VA3.4, not model execution or product release.
`host_aot_release_eligible` remains false, and the four-profile legacy
`--require-release` meaning is unchanged. Any final source change requires new
full gates rather than promotion of a failed or superseded stage by rerunning
only an individual test or audit.

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
LLVM/MLIR DSOs. Toolchain identity `ncnn-san-toolchain-identity/v2` binds the full
compile database, source archive, cache, installed LLVM/MLIR configs and exact
export/manifest/archive sets. Every C/C++ command is checked for address+undefined,
no-recover and PIC; assembly is listed separately rather than counted as
instrumented C/C++. The staged audit revalidates the content-bound inventory and
all copied archive/input records. Historical v1 identities retain their original
fixed-inventory rules; v2 does not pad files or inherit a previous build's counts.
Actual affinity-leak/unsigned-mask OpenMP patches, application records and clean
LSan/UBSan smoke evidence are hashed and preserved. Inherited sanitizer
suppression/cleanup overrides are rejected.
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
file-hash manifest, raw gate records and model evidence. Every generated tarball
is reopened and its exact regular-file inventory and member bytes are rehashed
before success is printed. `gate-archive-receipt.json` records the verified archive
hash outside the archive itself to avoid recursive identity. A composite
`--require-va0-baseline` audit requires and revalidates all four archive receipts;
missing, altered, unlisted or duplicate members fail. A sanitizer archive also
contains the verified toolchain identity, its configuration/build/smoke logs and
the patch records used for the isolated OpenMP runtime. It excludes unrelated
full numerical build trees. Logs are preserved before any CTest enumeration can
rewrite `LastTest.log`; moving a bundle must not silently reuse absolute paths
from a previous build. `.github/workflows/vulkan.yml` uses the same fresh runner
for hosted offline/software jobs and an explicitly requested self-hosted NVIDIA
job. Unrequested hardware coverage is absent, not passed; these CI tier jobs do
not by themselves supply the separate complete sanitizer/release decision.
