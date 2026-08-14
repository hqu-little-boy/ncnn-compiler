# ncnn 算子适配与数值验收避坑指南

本文记录为 ncnn-mlir 增加算子 lowering 和 numerical golden 测试时已经实际遇到的
问题。目标不是描述某一个算子的实现，而是给后续适配者一套能避免“模型可编译但语义不对”、
“reference 本身不稳定”以及“测试看似覆盖、实际未覆盖”的工作方法。

相关位置：

- 数值测试工程：`test/Numerical/`
- 测试支持层：`test/Numerical/support/numerical_test_support.cpp`
- fixture 注册：`test/Numerical/CMakeLists.txt`
- 稳定且唯一的产品模型编译入口：`tools/ncnn-compile.cpp`
- Python 流水线调试入口：`tools/compile_ncnn_model.py`
- 方言和 lowering：`include/ncnn-mlir/Dialect/NCNN/IR/`、`lib/Conversion/NCNNToTosa/`
- ncnn 疑似问题与上游报告前检查项：[`docs/ncnn-suspected-issues.md`](ncnn-suspected-issues.md)

## 1. “支持一个层”不等于支持该层的全部参数组合

应以 importer、verifier、normalize 和 lowering 共同接受的参数集合定义“当前支持”，不能只看
`ImportContext` 的 `kImporters` 注册表里是否出现层名。例如 Pooling 至少包含 max/average、
regular/global/adaptive、四种 `pad_mode` 和 `include_pad` 等互相影响的语义；Convolution 还包含
非对称 padding、stride、dilation、bias、SAME sentinel 和量化相关参数。

新增算子时应建立支持矩阵，逐项回答：

1. `.param` 中允许哪些参数 ID、默认值和数据类型。
2. parsed graph 和 ncnn dialect 如何保存原始语义。
3. normalize 是否需要把 sentinel、负 axis 或融合属性变成显式形式。
4. 目标 dialect 是否能直接表达该组合，还是需要拆解。
5. 单算子 fixture 覆盖了哪一个组合，哪些组合仍依赖模型级测试或尚未支持。

单个 happy-path golden 只能证明该参数组合正确，不能自动把整个 ncnn 层标记为完整支持。

## 2. `.param` 的词法形式会影响参数类型

本项目 parser 根据 token 的词法形式区分整数和浮点数。`0=0` 是整数，`0=0.000000` 才是
浮点数。ReLU 的 `negative_slope`、Dropout 的 `scale` 等参数要求浮点类型；写成整数零会在
importer 中得到“参数必须为 float”，即使数学值相同。

编写 fixture 时应遵循 ncnn 参数本身的类型，而不是只保证值相等：

```text
# 错误：negative_slope 被解析为整数
ReLU relu 1 1 data output 0=0

# 正确：negative_slope 是 float
ReLU relu 1 1 data output 0=0.000000
```

测试非默认参数尤其重要，否则 parser/default path 可能掩盖类型错误。

## 3. `.bin` 不是裸 float 数组

有权重的 fixture 必须遵循 ncnn model bin 编码。普通 float32 权重前有 32-bit little-endian
storage flag，之后才是 payload；bias 等后续 tensor 按层定义继续排列。直接写裸 float 会使
ncnn 和本项目 parser 从不同偏移读取数据。

当前 Convolution fixture 用 `fixtures/generate_bins.py` 显式生成：

```text
u32 little-endian storage flag = 0
kernel float32 payload
bias float32 payload
```

注意事项：

- 明确使用 little-endian `struct.pack`，不要依赖宿主 native endian。
- 参数中的 `weight_data_size` 必须和 kernel 元素数一致。
- bias term 为 1 时必须追加 `num_output` 个 bias。
- 无权重模型仍传入真实的空 `.bin` 文件，使两侧都走“已提供并验证模型文件”的路径。
- 不要把构建生成的二进制 fixture 提交到源码树；提交生成脚本和可读 `.param`。

## 4. 当前 reference 使用 ncnn 优化 CPU 路径

`run_ncnn_reference()` 没有调用 `register_custom_layer()` 或 `create_layer_naive()`，也没有关闭
packing、线程、Winograd 或 SGEMM。ncnn 会按构建平台和当前 CPU 选择正常的 CPU Layer、runtime
dispatch 和优化 kernel；这与实际部署路径更接近，但不同架构的累加顺序可能产生合法浮点差异。

test support 当前显式设置：

- `lightmode=false`、`use_vulkan_compute=false`；
- FP16/BF16 packed、storage 和 arithmetic 均关闭；
- `use_int8_inference` 仅在 INT8 reference case 开启；
- `flush_denormals=0`。

新增 numerical test 时无需维护 naive creator 注册表。应使用固定输入、与算子语义匹配的容差和
模型不变量，并在失败信息中记录最大误差及索引。若未来需要纯标量 reference，应先在 test support
实现独立模式并增加验证，不能仅通过文档声称已经关闭优化。

## 5. `ncnn::Mat` 的逻辑连续不代表 channel 之间无 padding

三维 `ncnn::Mat(w, h, c)` 的 channel stride 是 `cstep`，可能大于 `w * h`。以下操作对
多 channel tensor 是错误的：

```cpp
std::memcpy(mat.data, chw.data(), chw.size_bytes());
std::vector<float>(static_cast<float*>(mat.data),
                   static_cast<float*>(mat.data) + mat.total());
```

它们会把下一个 channel 写入或读自对齐 padding，而不是下一个逻辑 channel。本项目曾因此只在
Pooling 的最后一个 channel 出现错误，容易被误判为 pooling padding lowering 问题。

正确做法是逐 channel 复制和展平：

```cpp
for (int c = 0; c < channels; ++c) {
  std::memcpy(mat.channel(c),
              chw.data() + c * width * height,
              width * height * sizeof(float));
}
```

输出同样按 `output.channel(c)` 逐 channel 收集。只有一维/二维且确认无额外 stride 时，才可直接
复制裸区间。

## 6. 不要把调用方数组直接包装成 ncnn reference 输入

该行为的源码证据、复现边界和上游报告前检查项见
[`ncnn-suspected-issues.md`](ncnn-suspected-issues.md) 的第 1 节。

`ncnn::Mat(w, h, c, external_data)` 不拥有内存，也不会为架构 kernel 的安全预读提供尾部空间。
即使 packing 已关闭，ASan 仍可能在 ncnn convolution kernel 中报告输入数组末尾越界读取。

reference 应由 ncnn 自行分配 `Mat`，再逐 channel 复制输入。这样 allocator 会提供 ncnn 预期的
alignment 和尾部空间。该问题发生在 reference 侧，不代表生成的动态库越界；如果 reference
和被测库在同一进程，必须先根据调用栈区分责任方。

## 7. CHW、HWC、NHWC 与 ncnn `Mat` 维度不要混用

编译器公共 ABI 使用连续 CHW 数组；ncnn 三维 `Mat` 构造参数顺序为 `(w, h, c)`。因此
`tensor<3x227x227xf32>` 应映射为 `Mat(227, 227, 3)`，而不是按 tensor shape 原顺序构造。

还需注意：

- 编译器在 ncnn->TOSA lowering 中通过 `TypeConverter` 的双向 materialization 按需完成
  CHW/NHWC 转换；函数签名和 C ABI 仍是 CHW。合法非 ncnn op 或残留 ncnn op 与已转换
  TOSA 数据流交界时也会自动插入布局转换。
- Concat 的 ncnn axis 0 是 channel axis；转换到 NHWC 后目标 axis 必须相应转换。
- Global Pooling 可能把 `[C,H,W]` 降成一维 `[C]`，reference 输出不再是三维 Mat。
- 多输入测试的每个输入必须独立记录 shape 和 blob name，不能假设所有输入同 shape。

## 8. Pooling 是最容易产生边界语义差异的算子

ncnn `pad_mode=0` 表示 full padding，并会根据 `(input + explicit_pad - kernel) % stride`
在 bottom/right 追加 tail padding；`pad_mode=1` 才是只使用显式 padding的 valid 模式。
TOSA pool 还要求 padded extent、kernel 和 stride 满足其 verifier 约束。

适配 Pooling 时至少检查：

1. shape inference 使用 floor、ceil 还是 tail padding。
2. tail padding 只加在 bottom/right，还是按 SAME 规则分到两侧。
3. max pooling 的 padding value 必须等价于负无穷，不能用 0。
4. average pooling 是否包含 padding 元素。
5. global pooling 是否忽略 regular kernel/stride。
6. TOSA 的输出 shape 和传入 pad 数组是否完全一致。

小型单算子 fixture 建议先使用无歧义、可整除的 valid shape，分别增加专门 fixture 覆盖
full-tail、SAME_UPPER、SAME_LOWER 和 average exclude-pad。当前 average `include_pad=1`
会保留 `ncnn.pooling`，因此严格 native pipeline 会拒绝该组合；它应作为明确的未支持
参数边界，而不是 numerical golden。类似地，`pad_mode=0` 在某些 `stride > kernel` 的
组合下可能产生超过 TOSA kernel 约束的 padding，需要单独修复 lowering 后再纳入矩阵。
真实模型测试不能替代这些分支，但可继续覆盖 SqueezeNet 的实际 `pad_mode=0` 路径。

当前 Numerical 参数矩阵已覆盖：Convolution 的无 bias、dilation、非对称显式 padding、
SAME_UPPER/LOWER；ReLU 的普通、零/负输入和 leaky slope；Pooling 的 max/average、global、
SAME_UPPER/LOWER 与 tail；Concat 的 rank-3 正负 axis；Softmax 的 rank-3 正负 axis；以及
三路 Split 输出经过多级 consumer 的拓扑。`ConvolutionDepthWise` 当前支持纯 depthwise 的
FP32、FP16/BF16 边界、INT8 scale-term 和融合 ReLU；这不等于支持通用 group convolution、
动态权重或其他融合激活。

完整 34 个 source 计算层，以及包含 `Input` 在内的 35 个 importer source layer type，能力矩阵以 [`ncnn-compile-support-status.md`](ncnn-compile-support-status.md)
为准；本节只列 numerical fixture 已覆盖的参数组合。

## 9. 多输入、多输出 ABI 必须实际调用，不能只检查 manifest

公共 C ABI 的顺序固定为全部输入 tensor 参数组、全部输出数据指针、shape-only 动态输出的 data
capacity，然后是数据依赖输出的 shape/capacity/rank。当前静态 fixture 的输入组只有数据指针；固定-rank 动态输入还会在数据
指针后携带 shape，动态 rank 再携带 rank。内部 wrapper 再恢复模型
函数原始参数顺序。Concat 和 Split 是最小的静态 ABI 回归用例：

```c
int concat_scalar(const float *input1, const float *input2, float *output1);
int split_scalar(const float *input1, float *output1, float *output2);
```

测试应同时验证：

- 两个输入分别绑定到正确 ncnn blob；
- 每个输出独立比较，不只比较第一个输出；
- 输出元素数来自实际 manifest/shape，不通过总字节数猜测；
- `dlsym` 得到的 `void *` 通过 `std::bit_cast` 转为与该 case 完全匹配的函数指针类型。

断言按类别报告，不能用一个总断言掩盖失败原因：安全性（返回码、finite、空指针）、逐元素
数值（每个输出的最大绝对/相对误差和索引）、模型语义（softmax sum、top-1、top-5）以及
ABI 结构（manifest/头文件的数量、顺序、shape、元素宏）。

不要用一个固定的一入一出函数指针调用所有模型。C++ 调错函数指针签名是未定义行为，即使在某个
ABI 上暂时能工作。

数据依赖输出测试必须按 `MAX_ELEMENTS` 分配 data buffer，检查执行入口返回的 actual shape/rank，
并只比较逻辑前缀。`shape_capacity` 的单位是可写 extent 数量，不是 data buffer 元素数量；这类
输出不应调用普通 shape-only inference。

shape-only 动态输出则相反：测试先调用 inference 入口得到 shape，再按元素数分配 data buffer，
并把该元素数作为执行入口的 `uint64_t output_capacity`。少一个元素应稳定返回
`NCNN_STATUS_OUTPUT_CAPACITY_INSUFFICIENT`。

## 10. 简单 routing op 也可能引入新的系统符号

Split lowering 最终可能被优化成 `memcpy`。因此它虽然没有数学计算，生成动态库却新增了未定义
符号 `memcpy`。严格链接后的 undefined-symbol allowlist 必须根据实际 lowering 审核更新，不能
假设所有新算子只会使用既有的 `malloc`、`free`、`expf`。

更新 allowlist 时应满足：

- 只加入 libc/libm 或已启用 OpenMP 路径中预期且稳定的符号；
- 普通动态库通过 `-z defs` 和 `--no-undefined`；sanitizer 产物跳过这两个链接选项，但继续执行
  undefined-symbol 和 `DT_NEEDED` 审计；
- `DT_NEEDED` 仍仅包含允许的系统库；默认并行产物可包含匹配的 `libomp`；
- 不允许 `memrefCopy`、MLIR runner utils 或项目私有 runtime 偷渡进来。

## 11. 随机输入范围本身是测试契约

固定 seed 只解决可重复性，不能解决数值条件。深层卷积网络在不同合法累加顺序下会放大 float32
舍入差异。本项目曾在 `[-1,1]` 输入下观察到 SqueezeNet softmax 最大绝对差约 `3.1e-3`，而
固定预处理范围 `[-0.01,0.01]` 可满足当前 `1e-4` 门槛。

不能简单通过不断缩小输入或放宽阈值让失败消失。应先判断：

1. top-1、top-5 和 softmax sum 是否仍一致。
2. 单算子 golden 是否一致。
3. 第一层 Conv 从何处开始偏离。
4. 偏差是否随层数平滑累积，还是在 Pooling/Concat/padding 处突然跳变。
5. 输入范围是否对应产品真实预处理输出。

模型 golden 必须记录 seed、分布、范围、预处理和阈值；改变任一项都属于测试契约变更。

## 12. Sanitizer 覆盖范围容易被高估

在 sanitizer CMake build 中运行 numerical test（fixture 通过 `ncnn-compile` 生成并加载 `.so`），可以覆盖：

- GTest harness；
- 本地 test support；
- 通过 `add_subdirectory` 构建的 ncnn reference；
- 动态加载和输入/输出桥接；
- 由 numerical CMake fixture 自动传递 sanitizer 编译/链接参数后生成的模型 `.so`。

直接调用 `ncnn-compile` 时，生成模型不会仅因 driver 自身位于 sanitizer build 中就自动插桩，
调用方仍须通过可重复的 `--clang-arg` 和 `--linker-arg` 传递 sanitizer 参数。当前 numerical
CMake fixture 只要在 `CMAKE_CXX_FLAGS` 中看到 `fsanitize`，就固定传递
`-fsanitize=address,undefined`、`-lasan` 和 `-lubsan`。它不是从实际 flag 推导 sanitizer 集合；
因此支持的测试配置是 ASan+UBSan 组合，其他组合需要先修正 CMake 逻辑。

若要求 sanitizer 覆盖生成模型，编译与链接都必须传递匹配的 `-fsanitize=` 选项，并验证最终
`.so` 确实包含 instrumentation。`ncnn-compile` 只允许与所选 address/undefined sanitizer
匹配的 `__asan_*`、`__ubsan_*`、`__sanitizer_*` 符号族及 `libasan`/`libubsan`（或对应
Clang runtime）`DT_NEEDED`；若存在 sanitizer 符号却缺少对应 runtime 依赖，产物审计失败。

## 13. 把 ncnn 作为 submodule 接入时的 CMake 问题

ncnn 子工程声明 C 和 C++。顶层项目若只有 `LANGUAGES CXX`，在已有 build 目录中由嵌套
`project()` 首次启用 C 可能触发 CMake 平台文件生成错误。因此顶层项目应声明：

```cmake
project(ncnn_compiler LANGUAGES C CXX)
```

此外：

- CI checkout 必须启用 `submodules: true`。
- ncnn 自己的 glslang/pybind11 submodule 在 Vulkan、Python 和相关工具关闭时不需要初始化。
- 项目 `format`/`tidy` 文件 glob 必须排除 `test/third_party/`，不能格式化或审查第三方源码。
- ncnn 的全局 cache options 必须在 `add_subdirectory` 之前统一设置。

## 14. 运行 CTest 前必须重建对应 fixture

CTest 是测试执行器，不是构建工具。它不会触发 fixture 的 CMake custom command。每次运行
numerical CTest 前，必须先构建包含目标用例的测试 executable target：

```bash
cmake --build compiler/build --target numerical_tests --parallel
cmake --build compiler/build \
  --target numerical_dynamic_operator_tests numerical_dynamic_tests --parallel
```

静态标签只需构建 `numerical_tests`。`numerical-dynamic` 标签同时包含动态算子和动态模型，
运行整个标签前必须同时构建两个动态 target。只运行单个 fixture 时可以先构建对应的
`compile_<fixture-name>` target，但测试程序本身也有变化时必须构建测试 executable target，
并使用精确 `ctest -R`，不能运行包含未重建类别的整个 label。
修改 compiler、driver、opt、lowering pipeline、模型或 fixture 参数后直接执行 `ctest`，会
复用旧的 generated `.so`，从而使数值、符号依赖和性能结果失真。详细坑点和命令见
[`ncnn-suspected-issues.md`](ncnn-suspected-issues.md) 的“CTest 不会重建 fixture”一节。

## 15. 并行构建下生成 fixture 要避免相互踩产物

多个模型 fixture 会同时运行 driver、opt、translate、clang、nm 和 readelf。本环境曾在并行构建
时出现以下瞬时问题：

- `readelf` 读到尚未完成的 `.so`；
- GTest discovery 执行到正在替换的 executable，返回 `text file busy`；
- `ar`/`ranlib` 在目标替换阶段报告文件不存在。

工程仍使用规定的 `cmake --build build --parallel`，但 fixture 生成 target 通过依赖链串行化。
新增 fixture 时应把它接入该链，或改造生成脚本使用临时文件加原子 rename。不要通过全局关闭
并行构建掩盖产物声明和原子性问题，也不要让测试与会重链接同一 executable 的 target 并发运行。

## 16. 推荐的新增算子流程

1. 明确支持矩阵和拒绝路径，不从层名推断完整支持。
2. 添加 parser/importer/verifier 测试，检查参数类型、默认值、shape 和错误上下文。
3. 添加 normalize/lowering lit，检查 axis、layout、padding 和无残留算子。
4. 添加最小 `.param/.bin` numerical fixture，使用固定 seed 和显式输入范围。
5. 根据输入/输出数量使用精确的 C 函数指针签名。
6. 比较全部输出，报告最大误差及索引，并检查 finite。
7. 审核新增 undefined symbols、动态依赖和导出符号。
8. 在 Release 下验证真实优化产物。
9. 在 ASan+UBSan 下验证 harness、reference 和桥接，并确认生成 `.so` 已插桩。
10. 最后先重建全部测试 target，再运行全部 CTest、format、tidy 和 `git diff --check`。

若模型级结果不一致，优先从第一处中间结果偏差开始定位，而不是直接调整最终阈值。对
SqueezeNet，建议依次检查第一层 Conv、Pooling 尾部 padding、Fire Concat channel 顺序、
conv10 四边 padding、Global Average Pooling 和 Softmax axis。
