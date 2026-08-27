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
`generated/` 下的模型 `.so`、头文件和 manifest，然后才能运行相应的 `ctest -R ...` 或
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
与数值金标复用同一批 `compile_<model>` fixture（不重复编译模型）。运行前必须先构建：

```bash
cmake --build compiler/build --target performance_tests --parallel
ctest --test-dir compiler/build -L performance -V   # ctest 默认隐藏通过用例输出，需 -V
```

**有意义的对比必须在 Release 构建下进行**：Debug 构建中 vendored ncnn 参考以 `-O0`
编译，加速比会严重虚高。sanitizer 构建下所有性能测试自动 `GTEST_SKIP`（计时无意义，
fixture 本身也带插桩）。

### 方法论

按 vendored ncnn `benchmark/benchncnn.cpp` 的惯例：线程数取物理大核数，
`set_cpu_powersave(2)` + `set_omp_dynamic(0)`，ncnn 侧经 `opt.num_threads` 与
`set_omp_num_threads` 固定；编译产物侧链接同一 libomp 实例，主线程上的设置对其
同样生效（另设 `OMP_NUM_THREADS` 兜底独立 OpenMP 运行时副本）。每个模型先预热
再计时（`steady_clock`），报告均值/最小/中位/变异系数。ncnn 参考的 opt 块与数值
金标逐字一致（FP32 CPU 路径），每次迭代新建 Extractor——该开销计入上游运行时成本，
与 benchncnn 计时体一致。所有测试 `RUN_SERIAL` 独占 CPU。

### 输出格式

每模型一行机器可 grep 的报告：

```
PERF model=resnet18 threads=8 warmup=10 iters=20 ncnn_ms=11.230 compiled_ms=9.874 ratio=0.879 ncnn_min_ms=11.012 compiled_min_ms=9.701 cv=0.031
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `NCNN_PERF_THREADS` | 物理大核数 | 双侧统一线程数 |
| `NCNN_PERF_WARMUP` | 轻 10 / 重（输入 ≥ 640x640x3）5 | 预热次数 |
| `NCNN_PERF_ITERS` | 轻 20 / 重 10 | 计时迭代次数 |
| `NCNN_PERF_MAX_RATIO` | 未设置 = 纯报告 | 设置后 `compiled_mean > x * ncnn_mean` 即失败，消息携带实测数字 |
| `NCNN_PERF_JSON` | 未设置 = 关 | 逐模型追加一行 NDJSON 到该路径 |
| `NCNN_PERF_SKIP_SANITY` | 关 | 跳过每模型一次的宽松数值 sanity（rtol=atol=5e-3） |

门禁用法示例（建议先在代表机型采集基线后再定阈值，起点可参考 `1.5`）：

```bash
NCNN_PERF_MAX_RATIO=1.5 ctest --test-dir compiler/build -R PerformanceModel.ResNet18
NCNN_PERF_JSON=/tmp/perf.ndjson ctest --test-dir compiler/build -L performance
```

### 覆盖范围

当前覆盖 27 个静态形状模型：squeezenet_v1_1、resnet18/34/50/101、yolov5n/s/m/l/x_cls、
yolov5n/s/m/l/x 检测（640x640）、yolov5n/s/m/l/x_seg（双输出）、efficientnet_b0-b3、
PP-LCNet doc_ori/textline_ori、Chineseocr-Lite AngleNet。

明确不覆盖：多输入模型（SLAHead、FormulaNet embed/decoder）、dynamic OCR 族、operators
单算子（执行时间微秒级，计时噪声占比过大）。新增 OCR rec/det 等单入单出模型时在
`models/performance_test.cpp` 按 `ModelSpec` 扩行即可。

`../yolov5_v7.0_models` 或 `../ncnn_modelzoo/liteocr` 资产缺失时构建 `compile_<model>`
会失败——与数值测试行为一致（仅 squeezenet 自带于 third_party）。

注意：顶层的静态 llvm-mca 检查 `fp16-arithmetic-codegen` 也挂历史 `performance`
标签，`ctest -L performance` 会连它一起跑；其 fixture 属于 `numerical_tests` 构建链，
未构建时该用例会因缺产物失败，先 `cmake --build compiler/build --target numerical_tests`
再查标签即可。
