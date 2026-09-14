# Numerical Golden Tests

完整的适配注意事项见
[`docs/operator-numerical-validation-guide.md`](../../docs/operator-numerical-validation-guide.md)。
已发现但尚未由 upstream 确认的问题见
[`docs/ncnn-suspected-issues.md`](../../docs/ncnn-suspected-issues.md)。

该测试工程使用 GTest，将编译器生成的裸指针动态库与 upstream ncnn 的优化 FP32 CPU 路径
结果进行比较。ncnn 按构建平台和当前 CPU 自动选择 runtime dispatch、SIMD、packing、
线程、Winograd、SGEMM 等优化；关闭 Vulkan 和 FP16/BF16/INT8 降精度路径，确保比较双方都
使用 FP32 CPU 计算。默认固定种子随机输入来自标准正态分布，不人为设置有限范围；这使输入
在浮点数的有效域内覆盖不同数量级。只有需要验证定义域或边界行为的用例才使用显式范围。通用比较器使用
`abs(actual - expected) <= atol + rtol * abs(expected)`，默认 `atol=1e-6`，避免通过缩小输入
或只设置较大的绝对误差预算来掩盖随数值尺度增长的问题。精确搬运路径使用 `rtol=0`，低精度
和量化路径可显式设置独立的 `atol`。

## 运行前必须重建 generated fixture

> **不要直接运行 CTest。CTest 只执行已有测试，不会构建测试 target，也不会重新生成模型
> fixture。每一次测试前都必须先构建包含目标用例的正确 target。**

```bash
cmake --build compiler/build --target numerical_tests --parallel
cmake --build compiler/build \
  --target numerical_dynamic_operator_tests numerical_dynamic_tests --parallel
```

静态标签只需构建 `numerical_tests`；`numerical-dynamic` 标签同时包含动态算子和动态模型，
运行整个标签前必须同时构建上面的两个动态 target。对应 target 会通过 CMake 依赖重建其
`generated/` 下的模型 `.so`、头文件、manifest 和 execution-plan sidecar，然后才能运行相应的 `ctest -R ...` 或
`ctest -L ...`。尤其在修改
`ncnn-compile`、driver、opt、pipeline、模型 `.param/.bin` 或 fixture 参数后，直接运行
CTest 会静默复用陈旧产物，可能得到完全错误的数值、动态依赖和性能结果。完整说明及单个
fixture 的命令见
[`docs/ncnn-suspected-issues.md`](../../docs/ncnn-suspected-issues.md) 的“CTest 不会重建
fixture”一节。

现有分层：

- `operators/`：小型单算子 `.param/.bin` fixture 的精确比较。
- `models/`：完整模型的误差、softmax 和 top-k 验证。
- `support/`：固定种子随机输入、动态库加载、ncnn reference 和通用比较器。

新增用例时：

1. 在 `fixtures/` 添加模型，或使用 ncnn submodule 中已有模型。
2. 在 `CMakeLists.txt` 调用 `ncnn_add_compiled_fixture` 注册编译产物。
3. 在 `operators/` 或 `models/` 添加薄 GTest，复用 `run_ncnn_reference`、
   `compare_values` 和 `check_softmax`。

当前单算子层覆盖编译器支持的全部计算层，包括 `ConvolutionDepthWise`、`HardSigmoid`、
`HardSwish`、`Reshape`、`BinaryOp` 和 `InnerProduct`。参数矩阵覆盖卷积的 dilation、无 bias、
非对称 stride/padding 和 SAME padding，Depthwise 的非对称空间参数和 SAME padding，
Pooling 的非对称参数、global 参数忽略、average/SAME/tail，Reshape 的 `-1`/`0` 语义，
BinaryOp 的双向广播，以及 Concat/Softmax 的 rank-3 正负 axis。

当前明确不属于 native numerical 覆盖范围：通用 group convolution、regular average pooling
`include_pad=1` 和 adaptive pooling。这些配置在严格 lowering 中保留 ncnn operation 并被
residual gate 拒绝。

SqueezeNet 第一版验收要求：所有输出 finite、softmax sum 误差不超过 `1e-5`、top-1
一致、top-5 集合一致、相对误差不超过 `1e-4`（近零绝对误差不超过 `1e-6`）。Release 构建用于验证真实优化
产物；ASan/LSan 构建用于检查测试进程。

## 性能基准（performance 标签）

`performance_tests` executable 将编译产物与 upstream ncnn 的运行时墙钟耗时对比，
与数值金标复用同一批 `compile_<model>` fixture（不重复编译模型）。首轮全量
基线数据与热点结论见
[`docs/ncnn-performance-baseline-report.md`](../../docs/ncnn-performance-baseline-report.md)。
运行前必须先构建：

```bash
cmake --build compiler/build --target performance_tests --parallel
ctest --test-dir compiler/build -L performance -V   # ctest 默认隐藏通过用例输出，需 -V
```

**有意义的对比必须在 Release 构建下进行**：Debug 构建中 vendored ncnn 参考以 `-O0`
编译，加速比会严重虚高。sanitizer 构建下所有性能测试自动 `GTEST_SKIP`（计时无意义，
fixture 本身也带插桩）。

普通模型 fixture 的 P0 公平口径已写入 `test/Numerical/CMakeLists.txt`：x86 使用
`--vector-mode=fixed-width`、`+avx2`、`+fma` 和 `-march=x86-64-v3`，与本机 ncnn
runtime dispatch 的 AVX2/FMA 路径对齐；attention/大归约模型因显式行向量化会触发
LLVM 小时级编译爆炸，关闭该层 vector mode 但仍统一传入 `-march=x86-64-v3`，避免
退回 SSE2。若目标机器不支持 x86-64-v3，不应运行这组预编译产物。P0 后首次代表
测量（Release，本机 6 大核）为 resnet18 13.4→53.2 ms（ratio 3.96）、yolov5n det
48.7→107.6 ms（ratio 2.21）；完整报告中的旧表值来自 P0 前 SSE2 口径，供历史参考，
不应直接与新值作差分。

### 方法论

按 vendored ncnn `benchmark/benchncnn.cpp` 的惯例：线程数取物理大核数，
`set_cpu_powersave(2)` + `set_omp_dynamic(0)`，ncnn 侧经 `opt.num_threads` 与
`set_omp_num_threads` 固定；编译产物侧链接同一 libomp 实例，主线程上的设置对其
同样生效（另设 `OMP_NUM_THREADS` 兜底独立 OpenMP 运行时副本）。每个模型先预热
再计时（`steady_clock`），报告均值/最小/中位/变异系数。ncnn 参考的 opt 块与数值
金标逐字一致（FP32 CPU 路径），end-to-end 模式每次迭代新建 Extractor——该开销计入
上游运行时成本，与 benchncnn 计时体一致。`prepared`/`allocation_audit` 模式在计时前
加载 Net、创建并探测 Extractor、准备输入和输出；计时调用通过干净 Extractor 状态的
复制赋值复用已加载对象。`allocation_audit` 额外以 ncnn 专用 `CountingAllocator` 统计
准备后的分配/释放和峰值 live bytes；这些诊断结果不参与正式 ratio 门禁。所有测试
`RUN_SERIAL` 独占 CPU。

### 输出格式

每模型一行机器可 grep 的报告：

```
PERF model=resnet18 mode=end_to_end status=measured threads=8 warmup=10 iters=20 ncnn_ms=11.230 compiled_ms=9.874 ratio=0.879 ncnn_min_ms=11.012 compiled_min_ms=9.701 cv=0.031 setup=ncnn_extractor_and_io_per_iteration
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `NCNN_PERF_THREADS` | 物理大核数 | 双侧统一线程数 |
| `NCNN_PERF_WARMUP` | 轻 10 / 重（输入 ≥ 640x640x3）5 | 预热次数 |
| `NCNN_PERF_ITERS` | 轻 20 / 重 10 | 计时迭代次数 |
| `NCNN_PERF_MAX_RATIO` | per-class 默认门禁 | 默认走 `performance_test.cpp` 的逐类阈值表（2026-09-03 基线 + 头寸：常规 6.0 / server_rec 36 / formula 48 / int8 14 / medium_rec_int8 34），随追平里程碑收紧；显式设置正数 = 全局覆盖，`0` = 显式关闭全部门禁，非数字直接失败；`NCNN_PERF_THREADS` 被 pin（单线程/自定线程实验）且未设本变量时不判定，保持纯报告 |
| `NCNN_PERF_MODE` | `end_to_end` | `end_to_end` 每轮新建 ncnn Extractor 并计入 I/O；`prepared` 复用已加载 Net/Extractor，`allocation_audit` 在相同 prepared 口径下额外输出 ncnn 分配/释放/峰值 live bytes；后两者均为 measured 诊断且不参与门禁 |
| `NCNN_PERF_JSON` | 未设置 = 关 | 逐模型追加一行带 mode/status/setup/diagnostics 的 NDJSON；已测量行还带 `target`、`plan_revision`、`plan_hash` 和 `build_identity`，其中 build identity 是包含 code-generation identity 的 plan hash；旧记录缺少 mode 时按 `end_to_end` 汇总 |
| `NCNN_PERF_SKIP_SANITY` | 关 | 跳过每模型一次的宽松数值 sanity（rtol=atol=5e-3） |

门禁用法示例（per-class 阈值随追平计划收紧，临时放宽/收紧用全局覆盖）：

```bash
NCNN_PERF_MAX_RATIO=1.5 ctest --test-dir compiler/build -R PerformanceModel.ResNet18
NCNN_PERF_MAX_RATIO=0 NCNN_PERF_THREADS=1 ctest --test-dir compiler/build -R PerformanceModel.ResNet18
NCNN_PERF_JSON=/tmp/perf.ndjson ctest --test-dir compiler/build -L performance
python3 tools/perf_json_summary.py /tmp/perf.ndjson --gate 1.5
```

`perf_json_summary.py` 按 `(model, mode)` 汇总；旧 NDJSON 没有 `mode` 时按
`end_to_end` 兼容。同一输入文件中若同一 `(model, mode)` 出现多个完整且冲突的
`target`/`plan_hash`/`build_identity`，工具会拒绝静默覆盖。`--gate 0` 关闭汇总门禁，
正数只检查 `status=measured` 且 `gate_eligible=true` 的 `end_to_end` 记录。

### 覆盖范围

当前覆盖 46 个静态形状模型（其中 44 个可实际完成双侧计时，2 个 upstream
参考崩溃而显式 `GTEST_SKIP`）：squeezenet_v1_1、resnet18/34/50/101、yolov5n/s/m/l/x_cls、
yolov5n/s/m/l/x 检测（640x640）、yolov5n/s/m/l/x_seg（双输出）、efficientnet_b0-b3、
PP-LCNet doc_ori/textline_ori（FP32+Int8）、Chineseocr-Lite AngleNet、PP-OCRv5/v6
rec 全系（tiny/mobile/server/medium/small，FP32+Int8）、PP-OCRv5/v6 det 静态头
（FP32 640x640 与 medium_det_int8）、PP-StructrureV2 SLANet CNN、PP-FormulaNet_plus_S
encoder。

明确不覆盖：多输入模型（SLAHead、FormulaNet embed/decoder）、dynamic 系 fixture
（含 `pp_uvdoc_dynamic`、`pp_ocrv5_server_det_int8`——后者无模型资产，本就不存在）、
fp16/bf16 精度孪生产物、operators 单算子（执行时间微秒级，计时噪声占比过大）。
det_int8 中的 tiny/small/mobile 三行缺失：其上游 ncnn 参考在独立最小复现下即段
错误（FP32/Int8 均然，见 `docs/ncnn-suspected-issues.md`），无法提供对比侧；
PP-LCNet 两个 Int8 参考行同样因 upstream ncnn 崩溃而显式跳过。上游修复后按
`ModelSpec` 补回计时行即可。新增单入单出模型时同样扩行。

int8 行的计时在 `ReferenceInferenceMode::Int8` 下双侧同模式进行；由于 int8 金标
契约只做稳定性校验、无跨厂商交叉对比，这些行的 sanity 以有限域检查替代宽松
`compare_values`。

`../yolov5_v7.0_models` 或 `../ncnn_modelzoo/liteocr` 资产缺失时构建 `compile_<model>`
会失败——与数值测试行为一致（仅 squeezenet 自带于 third_party）。

注意：顶层的静态 llvm-mca 检查 `fp16-arithmetic-codegen` 也挂历史 `performance`
标签，`ctest -L performance` 会连它一起跑；其 fixture 属于 `numerical_tests` 构建链，
未构建时该用例会因缺产物失败，先 `cmake --build compiler/build --target numerical_tests`
再查标签即可。
