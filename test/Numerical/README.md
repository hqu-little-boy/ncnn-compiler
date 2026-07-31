# Numerical Golden Tests

该测试工程使用 GTest，将编译器生成的裸指针动态库与 upstream ncnn 的纯标量 CPU
结果进行比较。CMake 和运行时 `ncnn::Option` 都会关闭 SIMD、runtime CPU dispatch、
packing、线程、Vulkan、Winograd、SGEMM、fp16、bf16 和 int8 路径。固定种子输入采用
`[-0.01, 0.01]` 范围，避免随机大幅值在深层网络中放大不同累加顺序的舍入差异。

现有分层：

- `operators/`：小型单算子 `.param/.bin` fixture 的精确比较。
- `models/`：完整模型的误差、softmax 和 top-k 验证。
- `support/`：固定种子随机输入、动态库加载、ncnn reference 和通用比较器。

新增用例时：

1. 在 `fixtures/` 添加模型，或使用 ncnn submodule 中已有模型。
2. 在 `CMakeLists.txt` 调用 `ncnn_add_compiled_fixture` 注册编译产物。
3. 在 `operators/` 或 `models/` 添加薄 GTest，复用 `run_ncnn_reference`、
   `compare_values` 和 `check_softmax`。

当前单算子层覆盖编译器支持的全部计算层：`Convolution`、`ReLU`、`Pooling`、
`Split`、`Concat`、`Dropout` 和 `Softmax`。`Input`、`Const`、`Output` 是模型边界或
权重表示，由这些计算层 fixture 的编译和执行一并覆盖。

SqueezeNet 第一版验收要求：所有输出 finite、softmax sum 误差不超过 `1e-5`、top-1
一致、top-5 集合一致、最大绝对误差不超过 `1e-4`。Release 构建用于验证真实优化
产物；ASan/LSan 构建用于检查测试进程。
