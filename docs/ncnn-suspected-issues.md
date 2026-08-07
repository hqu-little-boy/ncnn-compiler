# ncnn 疑似问题与 API 契约记录

本文记录在 ncnn-mlir 数值验收过程中发现、但尚未由 ncnn upstream 确认的问题。
“疑似问题”表示当前观察到的行为与公开 API 的通常预期存在冲突，不表示已经证明是
ncnn 的实现缺陷。本文对应的 ncnn submodule 提交为
`a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`。

上游报告前，应使用独立于本项目的最小 ncnn 程序复现，并记录编译器、目标架构、编译
选项、输入 shape、内存分配方式和完整 sanitizer 调用栈。当前 numerical 测试由 CMake
fixture 通过唯一产品入口 `ncnn-compile` 生成并加载模型 `.so`；sanitizer 默认覆盖 harness、
reference 和 ABI 桥接，不会自动插桩该 `.so`。当前数值测试的规避方法见
[`operator-numerical-validation-guide.md`](operator-numerical-validation-guide.md)。

## 1. 外部 Mat 与优化 kernel 的尾部预读

**分类：高可信疑似 API 契约缺口，尚未确认是实现 bug。**

### 现象

使用 `ncnn::Mat(w, h, c, external_data)` 包装一个大小恰好为
`w * h * c * sizeof(float)` 的调用方数组，并将其送入 ncnn convolution，在
AddressSanitizer 下可能报告输入数组末尾越界读取。使用同样的 shape、但让 ncnn
自行分配 `Mat` 后，将数据逐 channel 复制进去，报告消失。

这里存在两个必须分开验证的潜在原因：external Mat 仍使用对齐后的 `cstep`，所以当
`cstep > w * h` 时，`w * h * c` 个元素连 Mat 的逻辑 channel stride 都不足以容纳；
即使 `cstep == w * h`，优化 kernel 仍可能依赖 ncnn allocator 提供的额外 overread
空间。当前项目内观察尚未用独立复现将两者完全隔离。

### 源码证据

- 外部三维 Mat 构造函数只保存调用方指针，并按 16 字节计算 `cstep`：
  `test/third_party/ncnn/src/mat.h:81-88`、`test/third_party/ncnn/src/mat.h:1073-1085`。
- ncnn 自有 allocator 明确为优化 kernel 额外分配 `NCNN_MALLOC_OVERREAD`：
  `test/third_party/ncnn/src/allocator.h:33-36`、`test/third_party/ncnn/src/allocator.h:56-68`。
- x86 convolution 的 packed 路径使用固定宽度 SIMD load，例如
  `test/third_party/ncnn/src/layer/x86/convolution_packed.h:2352-2368`。

这说明 ncnn 自有分配内存和外部内存的安全边界不同。公开的 external Mat 构造函数
签名没有在调用点体现调用方需要按 `cstep` 分配各 channel、以及是否必须提供额外尾部
空间；如果这些确实是外部输入的责任，至少需要在 API 文档中明确说明所需 stride、
alignment、padding 和 overread 空间。

### 为什么还不能定性为 ncnn bug

ncnn 的 allocator 注释已经明确承认优化 kernel 可能预读，并且这可能是有意的性能
设计。当前观察来自本项目的 x86 sanitizer 测试，尚未证明所有架构、所有 kernel 或
所有 ncnn 版本都具有相同要求，也没有确认 external Mat API 是否在其他官方文档中
规定了调用方必须提供 guard space。原始触发 shape 还可能同时包含 `cstep` padding；
在排除 stride 导致的容量不足前，不能把 ASan 报告单独归因于优化 kernel overread。

### 当前规避

reference 不直接包装调用方数组。先构造由 ncnn 分配的 `Mat`，再逐 channel 复制逻辑
CHW 数据；输出也逐 channel 展平。实现位于
`test/Numerical/support/numerical_test_support.cpp:199-224`，适配原则见
[`operator-numerical-validation-guide.md`](operator-numerical-validation-guide.md) 的
“不要把调用方数组直接包装成 ncnn reference 输入”一节。

### 上游报告前的最小复现

应准备一个只依赖 ncnn 的程序，包含：

1. 先选择满足 `cstep == w * h` 的 shape，排除 channel padding 导致的容量不足；
2. 分别测试逻辑精确大小、按 `cstep * c` 分配和额外增加 guard space 的外部数组；
3. 使用最小 convolution `.param/.bin`，明确 input shape、kernel、stride 和输出 shape；
4. 对照 external Mat 路径与 ncnn-owned Mat 路径；
5. 记录 ncnn 编译选项、CPU 型号、ASan 版本和完整报告；
6. 明确越界发生在读取哪个 tensor、哪个 kernel 和哪个元素之后。

## 2. 关闭优化选项后仍默认选择 CPU Layer

**分类：可复现的配置语义疑点，可能是文档/API 能力缺口。**

### 现象

关闭 SIMD、runtime CPU dispatch、packing 等 CMake 和运行时选项，不能仅凭这些选项
断言 `ncnn::Net` 使用 naive Layer。当前数值测试仍需显式调用
`register_custom_layer()`，将被测内建层覆盖为 `create_layer_naive()` 创建的 Layer。

### 源码证据

`Net::load_param()` 的选择顺序是 overwrite builtin、可选 Vulkan、`create_layer_cpu()`、
custom layer：`test/third_party/ncnn/src/net.cpp:1399-1413`。而 `create_layer_cpu()` 会
选择架构 registry，找不到时才回退到普通 registry：
`test/third_party/ncnn/src/layer.cpp:442-522`。`create_layer_naive()` 则直接使用普通
registry creator：`test/third_party/ncnn/src/layer.cpp:428-439`。

因此，“关闭硬件优化”与“强制使用 naive reference”是两个不同的契约。当前 workaround
利用 `Net::register_custom_layer()` 对内建类型建立 overwrite registry；该行为由
`test/third_party/ncnn/src/net.cpp:1213-1255` 和 `:2625-2672` 支持。

### 为什么还不能定性为 ncnn bug

`create_layer_cpu()` 很可能就是 ncnn 面向正常部署的预期入口，CMake 选项也不一定承诺
提供一个可复现的纯标量模式。当前缺口主要是：公开的配置命名容易让测试作者把
“没有 SIMD”理解成“没有架构专用 Layer”，而 ncnn 没有一个显式的 `use_naive_layers`
式选项供 reference 测试使用。

### 当前规避

在 `load_param()` 前为每个测试覆盖层注册 naive creator，并保持 packing、fp16、bf16、
int8、线程、Vulkan、Winograd 和 SGEMM 等路径关闭。新增 numerical fixture 时必须将
该层加入 `register_naive_layers()`。

### 上游报告前的检查项

应先确认官方文档是否定义这些选项的精确语义，并提供同一模型在
`create_layer_cpu()`、`create_layer_naive()` 和 custom overwrite 三条路径下的 layer
类型或结果对照。报告中不要把当前测试 harness 的 workaround 描述成 ncnn 必须采用的
部署配置。

## 3. 已确认的布局设计，不应误报为 bug

### `Mat::cstep` 的 channel padding

三维 Mat 的 `cstep` 按 16 字节对齐，源码为
`test/third_party/ncnn/src/mat.cpp:755-780`。因此 `cstep` 可能大于 `w * h`；连续
CHW 数组不能通过一次整块 `memcpy` 映射到多 channel Mat。这个行为是 ncnn 的存储布局
设计，不是已确认的越界或数值 bug。调用方应按 channel 复制，并按 channel 读取输出。

### Pooling 的 full padding

当前观察到 `pad_mode=0` 会根据输入、显式 padding、kernel 和 stride 在 bottom/right
追加 tail padding，而 `pad_mode=1` 只使用显式 padding。这是 ncnn Pooling 的语义，
不能在 lowering 中未经验证地改写成 valid 或另一种 SAME 规则。应使用专门 fixture
验证 shape、padding value 和 average pooling 的 include-pad 行为。

### 深层模型的 float32 累加差异

在 `[-1, 1]` 输入下，SqueezeNet 曾观察到约 `3.1e-3` 的 softmax 最大绝对差；将输入
范围固定为 `[-0.01, 0.01]` 后满足当前 `1e-4` 门槛。仅凭这一现象不能判定 ncnn bug，
因为不同合法实现的 float32 累加顺序会产生舍入差异。应先比较单算子、第一处中间结果、
top-k 和 softmax sum，再决定是否存在语义错误。

## 4. 结论与维护规则

- 当前只有 external Mat 的 `cstep`/尾部空间契约值得作为高优先级 API 文档问题向
  upstream 进一步确认；是否存在独立的 kernel 越界缺陷仍需最小复现隔离。
- naive Layer 选择问题应作为 reference 测试可复现性和配置文档问题跟踪。
- `cstep`、Pooling full padding 和浮点累加差异目前按 ncnn 正常设计或证据不足处理。
- 在获得独立 reproducer 或 upstream 明确回复前，禁止在其他文档中写成“ncnn 已确认的
  bug”。
- 若升级 ncnn submodule，应重新运行 external Mat sanitizer 对照、naive Layer 选择检查
  和所有 numerical golden tests，并更新本文提交号及源码行号。
