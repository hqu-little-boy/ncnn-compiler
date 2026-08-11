# Numerical Golden Tests

完整的适配注意事项见
[`docs/operator-numerical-validation-guide.md`](../../docs/operator-numerical-validation-guide.md)。
已发现但尚未由 upstream 确认的问题见
[`docs/ncnn-suspected-issues.md`](../../docs/ncnn-suspected-issues.md)。

该测试工程使用 GTest，将编译器生成的裸指针动态库与 upstream ncnn 的优化 FP32 CPU 路径
结果进行比较。ncnn 按构建平台和当前 CPU 自动选择 runtime dispatch、SIMD、packing、
线程、Winograd、SGEMM 等优化；关闭 Vulkan 和 FP16/BF16/INT8 降精度路径，确保比较双方都
使用 FP32 CPU 计算。固定种子输入采用 `[-0.01, 0.01]` 范围，避免随机大幅值在深层网络中
放大不同累加顺序的舍入差异。

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
一致、top-5 集合一致、最大绝对误差不超过 `1e-4`。Release 构建用于验证真实优化
产物；ASan/LSan 构建用于检查测试进程。
