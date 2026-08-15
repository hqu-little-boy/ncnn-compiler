# ncnn-compile 命令行参考

`ncnn-compile` 是稳定的 C++ 模型编译入口。它将 ncnn 的 `.param/.bin` 模型依次下降到
MLIR、LLVM IR 和本地目标代码，最终生成可供 C/C++ 程序调用的头文件和 Linux 动态库。

> 本文中的相对路径均以 `compiler/` 为基准，构建目录使用 `build/`。

## 1. 基本用法

```text
ncnn-compile [options] <input .param file>
```

最简单的调用方式：

```bash
./build/tools/ncnn-compile model.param
```

如果未指定其他选项，driver 会：

1. 读取 `model.param`。
2. 自动读取同目录下的 `model.bin`。
3. 从文件名推导模型名 `model`。
4. 创建名为 `model/` 的输出目录。
5. 使用默认优化等级 `-O3` 编译模型。

默认输出目录只包含稳定的公共产物：

```text
model/
├── model.h
└── libmodel.so
```

静态模型生成最简 C ABI 入口，并为每个 fixed-rank tensor 定义 rank、维度和动态维 mask；仅当
元素数可静态确定时生成 `ELEMENTS` 宏。例如：

```c
#define MODEL_INPUT1_RANK 3
#define MODEL_INPUT1_DIM0 INT64_C(3)
#define MODEL_INPUT1_DIM1 INT64_C(227)
#define MODEL_INPUT1_DIM2 INT64_C(227)
#define MODEL_INPUT1_DYNAMIC_DIM_MASK UINT32_C(0x0)
#define MODEL_INPUT1_ELEMENTS UINT64_C(154587)

#define MODEL_OUTPUT1_RANK 1
#define MODEL_OUTPUT1_DIM0 INT64_C(1000)
#define MODEL_OUTPUT1_DYNAMIC_DIM_MASK UINT32_C(0x0)
#define MODEL_OUTPUT1_ELEMENTS UINT64_C(1000)
#define MODEL_OUTPUT1_SHAPE_DEPENDS_ON_DATA 0

int model(const float *input1, float *output1);
```

输入参数排在输出参数之前。输入声明为 `const float *`，输出声明为 `float *`。调用成功返回
`0`；任一输入或输出为空指针时返回非零错误码。

ABI pass 和头文件生成器使用模型专用 typed 参数。固定 rank 动态 extent 追加 shape，shape-only
动态输出通过 `<model>_infer_output_shapes` 查询；DetectionOutput 等有界数据依赖输出按最大容量
分配，并由执行入口返回实际 shape/rank。`--input-shape=*` 还支持 rank 1 至 4 的受限动态 rank
identity/ReLU 模型。该能力不代表任意算子、多输入或多输出动态图均受支持。完整契约见
[`dynamic-rank-c-abi.md`](../../docs/dynamic-rank-c-abi.md)。

生成头文件统一定义 `NCNN_STATUS_SUCCESS` 到
`NCNN_STATUS_OUTPUT_CAPACITY_INSUFFICIENT` 六种状态（值 `0..5`）。shape-only 动态输出的执行
入口还接收 `uint64_t <output>_capacity`，单位是 output data buffer 的元素数；数据依赖输出的
`uint32_t <output>_shape_capacity` 则是 shape 数组可写的 extent 数量。

## 2. 输入和输出

### 2.1 `<input .param file>`

位置参数，指定 ncnn 模型结构文件：

```bash
ncnn-compile models/squeezenet.param
```

默认权重路径通过将末尾 `.param` 替换为 `.bin` 得到，因此上述命令会读取：

```text
models/squeezenet.bin
```

输入 `.param` 和对应 `.bin` 都必须存在。

### 2.2 `--bin=<path>`

显式指定权重文件。适用于 `.param` 和 `.bin` 不同名或位于不同目录的情况：

```bash
ncnn-compile model.param --bin=weights/model-fp32.bin
```

也可以使用空格分隔：

```bash
ncnn-compile model.param --bin weights/model-fp32.bin
```

### 2.3 `--model-name=<name>`

覆盖从 `.param` 文件名推导的模型名：

```bash
ncnn-compile model.param --model-name=image_classifier
```

模型名同时决定：

- 导出的 C 函数名 `image_classifier`。
- 头文件名 `image_classifier.h`。
- 动态库名 `libimage_classifier.so`。
- 默认输出目录名 `image_classifier/`。
- 使用 `--emit-manifest` 时的文件名 `image_classifier.json`。

模型名会被转换为合法的 C 标识符。例如：

```text
resnet-50  -> resnet_50
123model   -> ncnn_123model
```

转换按 ASCII C23 标识符规则进行：非 ASCII 字节和连续的非标识符字符替换为下划线；原名称中
合法的连续下划线保持不变。以数字或下划线开头的名称、C23 关键字以及 `main` 会增加 `ncnn_`
前缀。ABI manifest 中的参数名采用
相同规则；如果两个参数转换后重名，编译会失败而不会发布产物。

### 2.4 `--output-dir=<path>` 和 `-o <path>`

覆盖默认输出目录：

```bash
ncnn-compile model.param --output-dir=dist/model
```

短选项写法：

```bash
ncnn-compile model.param -o dist/model
```

`--model-name` 决定产物名称和导出函数名，`--output-dir` 只决定存放位置。例如：

```bash
ncnn-compile model.param \
  --model-name=resnet50 \
  --output-dir=dist/resnet
```

生成：

```text
dist/resnet/
├── resnet50.h
└── libresnet50.so
```

编译、链接和审计首先在系统临时目录完成，完整产物集也会先准备到输出目录同一文件系统中的
相邻替换目录。发布时 driver 原子地将旧输出目录改名为备份，再将替换目录改名为最终目录；第二步
失败时会回滚旧目录。因此任何编译、审计、准备或发布失败都不会暴露部分产物，并尽可能保留上一份
有效产物。成功重编译会以完整的新产物集替换旧目录，因此本次不再请求的中间 MLIR 文件会消失。
如果输出目录含有不属于 `ncnn-compile` 的文件，driver 会在编译前拒绝执行，且不会删除该文件。

### 2.5 `--input-shape=<CxHxW|*>`

固定 rank 形式恰好包含三个 extent，分隔符为 `x` 或 `X`，顺序为 ncnn 原生的
`C x H x W`。extent 可以是正整数，也可以是 `?`；`?` 表示该维在执行时由 C ABI shape
参数提供。

未提供任何 `--input-shape` 时，尺寸完全省略（`w/h/c/d` 均缺省或为 0）的 Input 会自动建立为
`[C,?,?]`。若该 Input 直接连接到具有有效四维权重 `[O,I,H,W]` 的 Convolution，则从 `I`
推导 `C`；否则 `C` 也保持动态。

```bash
ncnn-compile model.param --input-shape=3x224x224
```

动态空间 extent：

```bash
ncnn-compile model.param --input-shape=3x?x?
```

该选项可重复。任意显式 fixed-rank override 会关闭上述自动推导。override 数量必须等于尺寸完全
省略的 Input 数量，并按这些 Input 的 source-layer 顺序匹配；为兼容既有调用，也接受为全部
Input 各提供一个 override，已有静态尺寸的 Input 不会被改写。不能只覆盖无尺寸 Input 的任意子集：

```bash
ncnn-compile multi_input.param \
  --input-shape=3x?x? \
  --input-shape=1x32x?
```

真正动态 rank 使用 `*`：

```bash
ncnn-compile relu.param --input-shape=*
```

`*` 必须是唯一一个 `--input-shape`，模型必须只有一个未声明尺寸的 Input 和一个输出，且当前
计算图只允许 shape-preserving identity/ReLU。编译器生成 rank 1、2、3、4 四个 ranked
specialization 和公共 rank dispatcher；不使用 unranked memref。

### 2.6 `--input-dim-constraint=<INPUT:DIM:min=N,multiple=N>`

为 fixed-rank 动态输入维增加运行时约束。该选项可重复，例如：

```bash
ncnn-compile model.param \
  --input-shape=3x?x? \
  --input-dim-constraint=0:1:min=32,multiple=32 \
  --input-dim-constraint=0:2:min=32,multiple=32
```

`INPUT` 是按 source-layer 顺序排列的 Input 索引，`DIM` 使用 CHW 维序。约束只能指向 `?`
动态维；索引越界、重复约束、非正 minimum/multiple 或静态维约束都会在编译阶段拒绝。生成的
执行入口和 `<model>_infer_output_shapes` 都会在调用模型前检查 minimum 和整除条件。

Importer 还会为可追溯到模型输入的动态 `Slice` 轴自动推导 minimum。当前要求最后一片为
`-233`，其他片为正数或 `-233`；minimum 是显式 size 之和，每个 `-233` 至少按 1 计，
`multiple_of` 默认为 1。追溯可穿过受支持的 BinaryOp 广播和 Squeeze 轴映射；同一维同时有
显式约束时取更严格的 minimum，并保留显式 `multiple_of`。

### 2.7 当前 LiteOCR 动态输入配置

下表对应默认数值回归中的产品实例。它描述已经由源码和测试证明的 fixed-rank 动态范围，
不是对同族任意模型的自动承诺：

| 模型族 | 输入 override 与约束 | 输出 ABI |
|---|---|---|
| PP-LCNet doc/textline、AngleNet | `3x?x?`，H/W `min=1,multiple=1` | 固定分类向量；无 shape inference、无 output capacity |
| PP-OCRv5 mobile rec、PP-OCRv6 tiny rec | `3x48x?`，W `min=5,multiple=1` | shape-only 动态序列；先 inference，再传 data capacity |
| PP-OCRv5 mobile/server det、PP-OCRv6 tiny/small/medium det | `3x?x?`，H/W `min=32,multiple=32` | shape-only 动态概率图 `[1,H,W]` |

PP-LCNet 文档方向模型示例：

```bash
./build/tools/ncnn-compile \
  ../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.param \
  --input-shape=3x?x? \
  --input-dim-constraint=0:1:min=1,multiple=1 \
  --input-dim-constraint=0:2:min=1,multiple=1 \
  --emit-manifest --output-dir=/tmp/pp_lcnet_doc_dynamic
```

OCR 识别模型把高度固定为 48，仅动态化宽度；检测模型则对 H/W 同时增加 32 对齐约束：

```bash
# recognition
./build/tools/ncnn-compile rec.param --input-shape=3x48x? \
  --input-dim-constraint=0:2:min=5,multiple=1

# detection
./build/tools/ncnn-compile det.param --input-shape=3x?x? \
  --input-dim-constraint=0:1:min=32,multiple=32 \
  --input-dim-constraint=0:2:min=32,multiple=32
```

识别输出 sequence extent 为 `(W + 3) / 8`；PP-OCRv6 tiny rec 类别数为 6906，
PP-OCRv5 mobile rec 类别数为 18385。PP-LCNet doc/textline 的固定输出分别为 4 和 2，
AngleNet 固定输出为 2。

## 3. 优化和调试信息

### 3.1 `-O0`、`-O1`、`-O2`、`-O3`

设置 Clang 编译 LLVM IR 和链接动态库时使用的优化等级。帮助文本将该选项显示为
`-O <string>`，实际应使用 Clang 风格的连写形式：

```bash
ncnn-compile model.param -O0
ncnn-compile model.param -O2
ncnn-compile model.param -O3
```

| 选项 | 含义 | 典型场景 |
|---|---|---|
| `-O0` | 不进行目标代码优化 | 调试代码生成问题、缩短调试编译时间 |
| `-O1` | 基础优化 | 快速开发验证 |
| `-O2` | 常规优化 | 平衡性能和编译时间 |
| `-O3` | 更积极的优化 | 发布构建和性能测试 |

默认值是 `-O3`。该选项主要控制最终 LLVM IR 到目标代码的优化；流水线仍会执行 MLIR lowering
所必需的 canonicalization、CSE、bufferization 和 dialect conversion。

### 3.2 `--threads=<count>`

启用 Linalg 并行循环 lowering 和 OpenMP 运行时。默认值 `0` 生成 OpenMP worksharing loop，
线程团队大小由 OpenMP 运行时按当前可用 CPU 决定；`1` 明确关闭并行；大于 `1` 时把固定线程数
写入生成代码：

```bash
ncnn-compile model.param --threads=8
```

多线程产物依赖 `libomp`。交叉编译时，目标 sysroot 必须提供匹配的 OpenMP 运行时。

### 3.3 `--vector-width=<bits>`

设置 SIMD 宽度，默认值为 `256` 位。`0` 不设置宽度偏好；非零值必须是 64 的倍数。当前张量
计算以 32 位 lane 计算，例如串行模式下 `256` 生成 8-lane vector IR。
该选项通常与目标 CPU 选项配合使用：

```bash
ncnn-compile model.param --march=native --vector-width=256
```

显式向量化适用于 `--threads=1` 的串行 lowering。OpenMP 模式下，编译器把该值作为每个 worker
内 LLVM 自动向量化的宽度偏好。使用 `--threads=1 --vector-width=0` 可同时关闭并行和显式 SIMD。

### 3.4 `-g` 和 `--debug-info`

让 Clang 在目标文件和动态库中生成调试信息：

```bash
ncnn-compile model.param -O0 -g
```

长选项与 `-g` 等价：

```bash
ncnn-compile model.param -O0 --debug-info
```

`-g` 不会自动保留中间 MLIR。需要同时检查 lowering 结果时，应增加 `--emit`：

```bash
ncnn-compile model.param -O0 -g --emit=all -o debug/model
```

## 4. 中间产物和 ABI manifest

### 4.1 `--emit=<stage>`

保留指定的 MLIR 中间阶段。该选项可以重复使用，也可以用逗号分隔多个阶段：

```bash
# 保留一个阶段
ncnn-compile model.param --emit=tosa

# 逗号分隔多个阶段
ncnn-compile model.param --emit=ncnn,tosa,llvm

# 重复指定
ncnn-compile model.param --emit=ncnn,tosa --emit=memref --emit=llvm

# 保留全部公共阶段
ncnn-compile model.param --emit=all
```

支持的阶段如下：

| 阶段 | 输出文件 | 含义 |
|---|---|---|
| `ncnn` | `model.ncnn.mlir` | importer 生成的 ncnn dialect IR |
| `tosa` | `model.tosa.mlir` | ncnn lowering 到 TOSA 后的 IR |
| `linalg` | `model.linalg.mlir` | TOSA lowering 到 Linalg 后的 IR |
| `memref` | `model.memref.mlir` | bufferization 和输出参数转换后的 IR |
| `capi` | `model.capi.mlir` | 准备 C ABI wrapper 后的 IR |
| `llvm` | `model.llvm.mlir` | lowering 到 LLVM dialect 后的 MLIR |
| `llvm-ir` | `model.ll` | 翻译后的 LLVM IR |
| `object` | `model.o` | 目标文件 |
| `assembly` | `model.s` | 目标汇编 |
| `all` | 上述全部文件 | 保留所有公共阶段 |

`--emit=llvm` 表示 LLVM dialect MLIR；LLVM IR、目标文件和汇编分别由 `llvm-ir`、`object`、
`assembly` 请求。未请求的代码生成中间文件仍只存在于临时构建目录，不会发布。

`--emit` 只控制成功后保留哪些中间文件，不会让流水线提前停止。即使只指定 `--emit=ncnn`，
driver 仍会继续完成动态库编译、链接和产物审计。

如果只需要运行某一个早期阶段而不生成动态库，可以直接使用开发工具：

```bash
./build/tools/ncnn-mlir-driver model.param \
  --bin model.bin \
  -o model.ncnn.mlir

./build/bin/ncnn-mlir-opt \
  --ncnn-to-tosa-pipeline \
  model.ncnn.mlir \
  -o model.tosa.mlir
```

### 4.2 `--emit-manifest`

将 JSON ABI manifest 发布到输出目录：

```bash
ncnn-compile model.param --emit-manifest
```

输出目录会额外包含：

```text
model/
├── model.h
├── model.json
└── libmodel.so
```

manifest 描述导出函数以及输入、输出的名称、shape、元素类型和动态维。例如：

```json
{
  "function": "model",
  "target": {
    "triple": "x86_64-pc-linux-gnu",
    "cpu": "",
    "march": "",
    "tune": "",
    "features": [],
    "execution_profile": "x86-64-auto"
  },
  "precision_policy": {
    "storage": "f32",
    "complex_math": "f32",
    "complex_accumulator": "f32"
  },
  "inputs": [
    {
      "name": "input1",
      "shape": [3, 227, 227],
      "element_type": "f32",
      "dynamic_dim_mask": 0
    }
  ],
  "outputs": [
    {
      "name": "output1",
      "shape": [1000],
      "element_type": "f32",
      "dynamic_dim_mask": 0
    }
  ]
}
```

固定 rank 动态 extent 用 `-1` 和 `dynamic_dim_mask` 表示。数据依赖输出还包含
`maximum_shape` 和 `shape_depends_on_data: true`；受限动态 rank 参数使用 `dynamic_rank`、
`rank_min` 和 `rank_max`。值为 false 的 `shape_depends_on_data` 可省略。

带约束的动态输入还包含 `dimension_constraints`，每项提供 `dimension`、`minimum` 和
`multiple_of`。生成头文件同时提供 `<MODEL>_<INPUT>_DIM<N>_MINIMUM` 与
`<MODEL>_<INPUT>_DIM<N>_MULTIPLE_OF` 宏。

`target` 记录最终代码生成使用的 target provenance：Clang target triple、`mcpu`、`march`、`mtune`、
显式 target features，以及解析后的 `execution_profile`。典型 profile 包括
`x86-64-avx512-fp16`、`x86-64-avx512-bf16`、`aarch64-fp16`、`riscv-rvv-fp16` 和
`x86-64-fp16-storage-fp32`。后者明确表示 FP16 storage 但 FP32 accumulation，不表示原生 FP16
arithmetic。

`precision_policy` 记录实际存储和复杂计算边界；FP16 策略还会记录 `fp16_accumulator`，发生显式
回退时记录 `fallback`，避免把 storage 类型误解为全部算术的执行类型。

manifest 默认只是生成头文件所需的内部临时产物，不会发布。以下情况适合显式开启：

- ABI 自动化测试。
- 检查模型输入输出 shape。
- 自动生成调用代码或语言绑定。
- 调试 C API 生成过程。
- 数值测试读取模型接口信息。

## 5. 目标平台和 CPU

### 5.0 精度策略

`--precision` 选择模型的存储和计算策略：

| 值 | 语义 |
|---|---|
| `auto` | 保持模型和算子默认策略；Manifest profile 使用 `*-auto`，不宣称纯 FP32 |
| `f32` | FP32 storage 和 FP32 arithmetic |
| `fp16` | FP16 storage；卷积 accumulator 由 `--fp16-accumulator` 选择 |
| `bf16` | BF16 storage boundary，复杂算子按已实现的 FP32 boundary 规则处理 |
| `int8` | 验证并标记模型已有的受限 INT8 路径；带 INT8 权重/scale-term 的 Convolution、ConvolutionDepthWise、InnerProduct、Gemm 使用 I32 累加和 FP32 边界，并支持受限 Quantize、Dequantize、Requantize、Cast 链路 |

`--precision=int8` 不会把 FP32 权重或算子自动量化。INT8 计算仍由模型中已有的 INT8 权重、
`int8_scale_term` 和量化边界层决定；该选项用于选择/验证 precision policy 和记录 manifest
provenance，超出支持子集的已量化模型仍会编译失败。

```bash
# 需要目标具备原生 FP16 arithmetic；不满足时编译失败
ncnn-compile model.param --precision=fp16 --fp16-accumulator=f16

# 只使用 FP16 storage，使用 FP32 accumulator
ncnn-compile model.param --precision=fp16 --fp16-accumulator=f32

# 显式允许不具备原生 FP16 arithmetic 的目标回退
ncnn-compile model.param --precision=fp16 --fp16-accumulator=f16 \
  --target-feature=+f16c --allow-fallback
```

目标 capability 会同时根据 triple、march、mcpu 和 target features 检查。缺少 capability 时，
编译器会报告目标和缺失的原生能力；`--allow-fallback` 只允许 FP16 arithmetic 使用 FP32
accumulator，不会把 fallback 报告为原生 FP16。

### 5.1 `--target-triple=<triple>`

指定 Clang target triple：

```bash
ncnn-compile model.param \
  --target-triple=x86_64-unknown-linux-gnu
```

CLI 只接受 64 位 Linux ELF triple。宿主 Linux x86-64 已通过完整 Release、数值模型和动态库执行验证；
AArch64 FP16 与 RISC-V Zfh/Zvfh FP16 已通过编译器生成 LLVM IR 的静态 assembly 检查，但不等同于
目标硬件运行验证。
Windows DLL、macOS dylib 和 32 位 ELF 不属于当前产物契约。

AArch64 交叉编译示例：

```bash
ncnn-compile model.param \
  --target-triple=aarch64-unknown-linux-gnu \
  --sysroot=/opt/aarch64-sysroot
```

交叉编译时，仅指定 triple 通常不够，还必须提供匹配的 Clang 工具链和 target sysroot。
`--verify-execution` 只能用于生成库可由当前宿主执行的情况。

### 5.2 `--march=<architecture>`

指定允许使用的目标指令集架构，相当于传递 Clang 的 `-march=<architecture>`：

```bash
ncnn-compile model.param --march=x86-64
```

针对当前构建机器：

```bash
ncnn-compile model.param --march=native
```

`--march=native` 适用于本机部署，不适合分发到未知 CPU，因为生成的动态库可能使用其他机器
不支持的指令。

### 5.3 `--mcpu=<cpu>`

指定具体目标 CPU，让 Clang 根据该 CPU 的能力生成代码：

```bash
ncnn-compile model.param --mcpu=znver4
```

ARM64 示例：

```bash
ncnn-compile model.param \
  --target-triple=aarch64-unknown-linux-gnu \
  --mcpu=cortex-a76 \
  --sysroot=/opt/aarch64-sysroot
```

可用 CPU 名称取决于 Clang 版本和目标架构。

### 5.4 `--mtune=<cpu>`

针对指定 CPU 调整指令选择和调度，但通常不用于开启新的指令集能力：

```bash
ncnn-compile model.param \
  --march=x86-64 \
  --mtune=generic
```

三个选项的职责可以概括为：

- `--march` 决定允许使用哪些指令。
- `--mcpu` 选择具体 CPU 及其默认能力。
- `--mtune` 决定针对哪种 CPU 优化指令调度。

### 5.5 `--target-feature=<+feature|-feature>`

单独启用或禁用 Clang target feature，可重复指定：

```bash
ncnn-compile model.param \
  --target-feature=+sse4.2 \
  --target-feature=-avx
```

也可以使用空格形式。即使 feature 以 `-` 开头，driver 也会把它识别为该选项的值：

```bash
ncnn-compile model.param \
  --target-feature -avx \
  --target-feature -avx2
```

这是较底层的代码生成控制。错误组合可能导致 Clang 拒绝编译、与 `--march`/`--mcpu` 冲突，
或者生成当前主机无法执行的指令。常规使用应优先选择 `--march` 或 `--mcpu`。

### 5.6 `--sysroot=<path>`

指定目标系统根目录：

```bash
ncnn-compile model.param \
  --target-triple=aarch64-unknown-linux-gnu \
  --sysroot=/opt/aarch64-linux-gnu/sysroot
```

sysroot 通常需要包含目标平台的 libc、libm、动态链接器、头文件和库文件。当前动态库链接使用
`-lc` 和 `-lm`，因此交叉编译时 sysroot 必须提供对应目标架构的 libc 和 libm。

## 6. Clang 和 linker 参数透传

### 6.1 `--clang-arg=<argument>`

向 LLVM IR 编译为目标文件的 Clang 命令额外传递一个参数：

```bash
ncnn-compile model.param --clang-arg=-ffast-math
```

可以重复使用：

```bash
ncnn-compile model.param \
  --clang-arg=-ffast-math \
  --clang-arg=-fno-math-errno
```

也支持空格形式，包括以 `-` 开头的参数值：

```bash
ncnn-compile model.param --clang-arg -ffast-math
```

这些参数只加入 LLVM IR 到目标文件的编译命令，不会自动加入动态库链接命令。该选项适合
实验性浮点优化、Clang 诊断控制和 driver 尚未提供专用选项的代码生成功能。

### 6.2 `--linker-arg=<argument>`

向 `clang -shared` 动态库链接阶段额外传递一个参数：

```bash
ncnn-compile model.param --linker-arg=-Wl,--hash-style=gnu
```

可以重复使用：

```bash
ncnn-compile model.param \
  --linker-arg=-fuse-ld=lld \
  --linker-arg=-Wl,--hash-style=gnu
```

需要向实际 linker 传参时，通常使用 `-Wl,` 前缀。普通产物使用 `-z defs` 和
`--no-undefined`；显式启用 address/undefined sanitizer 时跳过这两个链接选项，以允许 sanitizer
runtime 在最终进程中解析其符号，但仍执行未定义符号、导出和 `DT_NEEDED` 审计。driver 确保：

- 没有未解析的非预期符号。
- 只导出模型执行入口，以及存在 shape-only 动态输出时的 `<model>_infer_output_shapes`。
- 不生成非确定性的 build ID。
- 最终库只依赖允许的系统库和函数。

高级透传参数可能破坏这些约束。如果链接结果违反导出符号、未定义符号或依赖库契约，driver
会拒绝发布产物。

## 7. 日志和执行验证

### 7.1 `-v`

打印 driver 执行的所有外部命令：

```bash
ncnn-compile model.param -v
```

输出包括：

```text
ncnn-mlir-driver ...
ncnn-mlir-opt --ncnn-to-tosa-pipeline ...
ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline ...
mlir-translate --mlir-to-llvmir ...
clang -x ir ...
clang -shared ...
llvm-nm ...
llvm-readelf ...
```

适用于检查工具发现结果、确认代码生成参数、调试交叉编译，或者复制某条命令独立复现失败阶段。

### 7.2 `--verify-execution`

生成一个临时 C harness，链接刚生成的动态库并执行 ABI smoke test：

```bash
ncnn-compile model.param --verify-execution
```

该 smoke test 会检查：

- 生成的头文件可以被 C 编译器使用。
- 动态库可以成功链接和加载。
- 模型入口在零初始化输入上返回成功。
- 输出元素是有限浮点数。
- 任意输入或输出为空指针时，接口返回错误。

动态模型还会按 manifest 构造运行时 shape；shape-only 输出先调用 inference 再分配，数据依赖
输出按 `MAX_ELEMENTS` 分配并读取 actual shape/rank，受限动态 rank 路径还检查合法和非法 rank。

该选项只适用于生成库可以在当前宿主执行的情况。交叉编译到不同架构时不要启用。例如，在
x86-64 主机上生成 AArch64 动态库时，driver 可以完成编译和静态审计，但无法直接运行 AArch64
harness。

`--verify-execution` 只是 ABI smoke test，不会将结果与原始 ncnn runtime 做完整数值比较。
验证通过时，driver 会在终端打印 `ncnn-compile: ABI execution verification passed`。harness
源码、测试可执行文件和零初始化输入都位于系统临时目录，执行后自动删除，不会发布到输出目录；
该测试也不会打印或保存输出张量数值。验证失败时，harness 会指出失败的检查项，例如内存分配
失败、模型返回非零、某个输出元素不是有限值，或者某个空指针参数被错误接受；driver 同时保留
harness 的非零退出码。

## 8. 通用帮助选项

### 8.1 `-h` 和 `--help`

显示公共命令行选项：

```bash
ncnn-compile --help
```

### 8.2 `--help-hidden`

显示公共选项以及供 CTest、工具链开发和问题定位使用的隐藏选项：

```bash
ncnn-compile --help-hidden
```

隐藏选项主要用于覆盖 `ncnn-mlir-driver`、`ncnn-mlir-opt`、Clang 和 LLVM 工具路径，不属于
推荐的稳定用户接口。

### 8.3 `--help-list` 和 `--help-list-hidden`

以列表形式显示公共选项或包括隐藏选项的完整列表：

```bash
ncnn-compile --help-list
ncnn-compile --help-list-hidden
```

### 8.4 `--version`

显示 LLVM command-line 层提供的版本信息：

```bash
ncnn-compile --version
```

## 9. 常见命令组合

### 9.1 发布编译

```bash
ncnn-compile model.param -O3
```

### 9.2 指定模型名和输出目录

```bash
ncnn-compile models/squeezenet.param \
  --bin=models/squeezenet.bin \
  --model-name=squeezenet_v1_1 \
  --output-dir=dist/squeezenet \
  -O3
```

### 9.3 调试全部 lowering 阶段

```bash
ncnn-compile model.param \
  -O0 \
  -g \
  -v \
  --emit=all \
  --emit-manifest \
  --output-dir=debug/model
```

### 9.4 生成并执行 ABI smoke test

```bash
ncnn-compile model.param \
  -O2 \
  --emit-manifest \
  --verify-execution
```

### 9.5 生成可移植的基础 x86-64 动态库

```bash
ncnn-compile model.param \
  -O3 \
  --target-triple=x86_64-unknown-linux-gnu \
  --march=x86-64 \
  --mtune=generic
```

### 9.6 针对当前机器优化

```bash
ncnn-compile model.param \
  -O3 \
  --march=native
```

该方式适合本机部署，不适合将动态库分发到 CPU 能力未知的机器。

### 9.7 AArch64 交叉编译

```bash
ncnn-compile model.param \
  -O3 \
  --target-triple=aarch64-unknown-linux-gnu \
  --mcpu=cortex-a76 \
  --sysroot=/opt/aarch64-sysroot \
  --output-dir=dist/aarch64/model
```

### 9.8 验证 fixed-rank 动态模型 ABI

```bash
ncnn-compile model.param \
  --input-shape=3x?x? \
  --input-dim-constraint=0:1:min=32,multiple=32 \
  --input-dim-constraint=0:2:min=32,multiple=32 \
  --emit=all --emit-manifest --verify-execution \
  --output-dir=/tmp/model_dynamic
```

检查生成头文件和 manifest，而不是仅凭输入含 `?` 推断调用形式：固定分类输出不会生成
`_infer_output_shapes` 或 capacity 参数；动态概率图/序列输出会生成 inference 入口并要求 data
capacity；DetectionOutput 的数据依赖输出则返回 actual shape/rank。

## 10. Python 调试流水线

`tools/compile_ncnn_model.py` 保留用于开发调试、快速验证完整流水线和观察中间阶段。它不是稳定
生产入口，也不是 C++ CLI 的等价实现：

```bash
python3 tools/compile_ncnn_model.py \
  model.param \
  --bin=model.bin \
  --emit=all \
  --emit-manifest \
  -O0 \
  -v
```

稳定编译和发布应使用 C++ executable：

```bash
./build/tools/ncnn-compile model.param
```

当前 Python 脚本缺少 `--precision`、`--fp16-accumulator`、`--allow-fallback`、`--threads` 和
`--vector-width`；其 `--emit=all` 只发布 ncnn/tosa/linalg/memref/capi/llvm 六个 MLIR 阶段，
不发布 `.ll`、`.o` 或 `.s`。这些能力的权威接口是 C++ `ncnn-compile --help`。
