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

## 4. ncnn 的 CMake 开关关闭后，仍可能选择架构专用 Layer

仅设置以下选项为 `OFF` 不足以保证 `ncnn::Net` 使用通用实现：

```text
NCNN_SSE2 / NCNN_AVX* / NCNN_FMA / NCNN_XOP / NCNN_F16C
NCNN_RUNTIME_CPU / NCNN_GNU_INLINE_ASM
NCNN_OPENMP / NCNN_THREADS / NCNN_VULKAN
```

ncnn 仍会根据 `NCNN_TARGET_ARCH=x86` 注册 x86 Layer 类，其中可能包含 packed kernel。运行时
`use_packing_layout=false` 也不等同于“使用 naive Layer”。本项目因此在加载 param 前，通过
`Net::register_custom_layer()` 覆盖当前测试涉及的内建层，并由
`create_layer_naive()` 创建 reference Layer。

新增算子 numerical test 时，必须把该层加入 naive creator 注册表。否则测试结果可能依赖宿主
CPU、ncnn 架构实现或 sanitizer build，违背 reference 可重复性的目标。

编译期和运行期仍需同时关闭：

- packing、fp16、bf16、int8；
- Winograd 和 SGEMM；
- Vulkan、线程和 runtime CPU dispatch；
- DAZ/FTZ，当前 reference 将 `flush_denormals` 设为 0。

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
三路 Split 输出经过多级 consumer 的拓扑。`ConvolutionDepthWise` 当前支持独立的纯 depthwise
静态 FP32 子集；这不等于支持通用 group convolution、动态权重、量化或融合激活。

完整 26 个计算 op 的集合，以及包含 `Input` 在内的 27 个 importer source layer type，能力矩阵以 [`ncnn-compile-support-status.md`](ncnn-compile-support-status.md)
为准；本节只列 numerical fixture 已覆盖的参数组合。

## 9. 多输入、多输出 ABI 必须实际调用，不能只检查 manifest

公共 C ABI 的顺序固定为全部输入 tensor 参数组，然后全部输出数据指针。当前静态 fixture 的
输入组只有数据指针；固定-rank动态输入还会在数据指针后携带 shape。内部 wrapper 再恢复模型
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

## 10. 简单 routing op 也可能引入新的系统符号

Split lowering 最终可能被优化成 `memcpy`。因此它虽然没有数学计算，生成动态库却新增了未定义
符号 `memcpy`。严格链接后的 undefined-symbol allowlist 必须根据实际 lowering 审核更新，不能
假设所有新算子只会使用既有的 `malloc`、`free`、`expf`。

更新 allowlist 时应满足：

- 只加入 libc/libm 中预期且稳定的符号；
- 动态库仍通过 `-z defs` 和 `--no-undefined`；
- `DT_NEEDED` 仍仅包含允许的系统库；
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
- 动态加载和输入/输出桥接。

但当前 `ncnn-compile` 直接调用 clang，源码没有 sanitizer 专用转发选项；生成的模型 `.so`
本身并未被 sanitizer instrument。测试进程带 ASan 不等于动态库内部
每个访问都已插桩。

若要求 sanitizer 覆盖生成模型，必须给 `ncnn-compile` 增加可重复的 compile/link flag 参数，并验证
最终 `.so` 确实包含 sanitizer runtime/instrumentation；不能只看 CTest 运行在
`build-sanitize` 目录。

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

## 14. 并行构建下生成 fixture 要避免相互踩产物

多个模型 fixture 会同时运行 driver、opt、translate、clang、nm 和 readelf。本环境曾在并行构建
时出现以下瞬时问题：

- `readelf` 读到尚未完成的 `.so`；
- GTest discovery 执行到正在替换的 executable，返回 `text file busy`；
- `ar`/`ranlib` 在目标替换阶段报告文件不存在。

工程仍使用规定的 `cmake --build build --parallel`，但 fixture 生成 target 通过依赖链串行化。
新增 fixture 时应把它接入该链，或改造生成脚本使用临时文件加原子 rename。不要通过全局关闭
并行构建掩盖产物声明和原子性问题，也不要让测试与会重链接同一 executable 的 target 并发运行。

## 15. 推荐的新增算子流程

1. 明确支持矩阵和拒绝路径，不从层名推断完整支持。
2. 添加 parser/importer/verifier 测试，检查参数类型、默认值、shape 和错误上下文。
3. 添加 normalize/lowering lit，检查 axis、layout、padding 和无残留算子。
4. 添加最小 `.param/.bin` numerical fixture，使用固定 seed 和显式输入范围。
5. 将新层加入 ncnn naive creator 注册表。
6. 根据输入/输出数量使用精确的 C 函数指针签名。
7. 比较全部输出，报告最大误差及索引，并检查 finite。
8. 审核新增 undefined symbols、动态依赖和导出符号。
9. 在 Release 下验证真实优化产物。
10. 在 sanitizer 下验证 harness、reference 和桥接，并明确生成 `.so` 是否已插桩。
11. 最后运行完整构建、全部 CTest、format、tidy 和 `git diff --check`。

若模型级结果不一致，优先从第一处中间结果偏差开始定位，而不是直接调整最终阈值。对
SqueezeNet，建议依次检查第一层 Conv、Pooling 尾部 padding、Fire Concat channel 顺序、
conv10 四边 padding、Global Average Pooling 和 Softmax axis。
