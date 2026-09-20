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
- 使用 `--emit-execution-plan` 时的文件名 `image_classifier.plan.json`。

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
| PP-OCRv5 mobile/server rec、PP-OCRv6 tiny/small/medium rec、PP-OCRv6 small/medium rec INT8 | `3x48x?`，W `min=5,multiple=1` | shape-only 动态序列；先 inference，再传 data capacity |
| PP-OCRv5 mobile/server det、PP-OCRv5 mobile det INT8、PP-OCRv6 tiny/small/medium det | `3x?x?`，H/W `min=32,multiple=32` | shape-only 动态概率图 `[1,H,W]` |

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
PP-OCRv5 mobile/server rec 类别数为 18385，PP-OCRv6 small/medium rec 类别数为 18710。
PP-LCNet doc/textline 的固定输出分别为 4 和 2，
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

### 3.4 `--conv-strategy=<mode>` 和 `--conv-gemm-l2-bytes=<bytes>`

控制算子形态策略层（`strategy-ncnn`）的卷积改写行为。该层在向量化之前把卷积
改写为 matmul 形态，使计算主体进入投影映射的收缩主干：

```bash
ncnn-compile model.param --conv-strategy=auto
```

`--conv-strategy` 取值：

| 取值 | 行为 |
|---|---|
| `auto`（默认） | 1×1 s1 无条件折叠为 matmul；其余 k×k 仅静态空间维且命中 ncnn `prefer_sgemm` 启发式时走 im2col+matmul |
| `gemm` | 静态实例强制走 GEMM 形态（跳过阈值判断） |
| `conv` | 关闭全部改写，保留直接卷积 |
| `winograd` | P7：3×3 s1 d1 且 IC>8 或 OC>8 的静态批 1 实例改写为 F(6,3)（输入/权重/输出变换 + 中心 batch_matmul，ncnn conv3x3s1_winograd63 同款矩阵）；判据外实例回退常规 dispatch。数值契约：f32 相对误差 ~1e-3 量级（有理插值点），对拍容差 rtol=1e-3/atol=1e-4（operator 级守护 `ConvolutionWinogradMatchesReference`），模型级启用须逐模型对账 golden 预算；`auto` 默认不启用（预算验证后另行翻默认） |

`--conv-gemm-l2-bytes` 覆盖启发式中的 L2 缓存字节预算（默认 `524288`，即 512 KiB）。
数值上 im2col 路径改变累加顺序（ULP 级差异），由全量数值黄金测试按既定预算验收。

### 3.5 `--vector-math=<mode>`

控制超越函数（expf/tanhf/erff/logf/powf）在向量 IR 中的下降后端，默认
`auto`：

```bash
ncnn-compile model.param --vector-mode=auto --vector-math=auto   # 默认
ncnn-compile model.param --vector-mode=auto --vector-math=libmvec
ncnn-compile model.param --vector-mode=auto --vector-math=sleef
ncnn-compile model.param --vector-math=none                      # 历史标量路径
```

取值语义：

| 取值 | 行为 |
|---|---|
| `auto`（默认） | 探测目标 sysroot 的 libmvec；可用则直连，否则静默降级 vendored SLEEF 静态档案，再退 `none` |
| `libmvec` | 显式 glibc libmvec。探测不可用时**报错退出**——点名即兑现 |
| `sleef` | 显式 vendored SLEEF（`third_party/sleef` 构建出的静态档案）；档案缺失时报错退出 |
| `none` | 保持历史行为：MathToLibm 标量 libcall |

探测以 `-Wl,-z,defs -lm` 编译引用全部五个 `_ZGV` 符号的探针：glibc 的
libm linker script 会按需带入 `libmvec.so.1`（产物自动获得该 DT_NEEDED），
musl/旧 glibc 直接失败。libmvec 变体绑定编译期 ISA（显式 feature/march
提示推导 `_ZGV{b,c,d,e}N{4,8,16}` 与 AArch64 `_ZGVnN4`）；SLEEF 入口
（如 `Sleef_expf8_u10`）内部自带运行时 ISA 分发。

部署契约：产物携带编译期解析的符号版本引用。"目标支持但部署机更旧"由构
建方以 `--sysroot` 指向对应环境重编解决，编译器不补偿。libmvec 的
tanh/erf 向量变体要求 x86-64 glibc ≥ 2.35、AArch64 ≥ 2.40；exp 类要求
x86-64 ≥ 2.22、AArch64 ≥ 2.38。musl 无 libmvec。SLEEF 路径为静态编入，
无运行时库要求。erfcf 无向量变体，任何后端下保持标量。

### 3.6 `-g` 和 `--debug-info`

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

### 3.7 INT8 内核 opt-in 与目标能力

`--int8-kernel=portable|auto|vnni` 控制静态 INT8 row-dot 内核，**不等同于**
`--precision=int8` 或目标 CPU 的 ISA 能力声明：

| 策略 | 行为 |
|---|---|
| `portable`（默认） | 保持既有 row-dot 路径，不显式发射 VNNI；最终 Clang 仍可按目标 ISA 自动向量化 |
| `auto` | 探测并记录能力，但当前仍保持 portable，等待性能默认化验收；不自动选 VNNI |
| `vnni` | 显式 opt-in；目标不支持则报错。仅合格静态、连续 K 且 K≥32 的 row-dot 使用 256-bit VNNI，其他算子形态仍可 fallback |

探针与最终编译共享 `--target-triple`、`--sysroot`、`--march`、`--mcpu` 和有序
`--target-feature` 参数。AVX-VNNI 要求 x86-64、AVX2、AVX-VNNI；AVX512-VNNI
要求 x86-64、AVX2、AVX512F/BW/VL/VNNI。`x86-64-v4` 本身不包含 VNNI。
重复 feature 由 Clang 按实际顺序和依赖关系解析，最后禁用 VNNI 会使强制策略失败：

```bash
# AVX-VNNI（部署 CPU 必须支持；不会生成运行时 dispatcher）
ncnn-compile model.param --int8-kernel=vnni --march=x86-64-v3 \
  --target-feature=+avxvnni --threads=6

# 串行必须显式开启固定宽 MLIR 向量化
ncnn-compile model.param --int8-kernel=vnni --march=x86-64-v3 \
  --target-feature=+avxvnni --threads=1 --vector-mode=fixed-width

# 清晰拒绝：最后的 disable 生效，不可根据前面的 enable 选中 VNNI
ncnn-compile model.param --int8-kernel=vnni --march=x86-64-v3 \
  --target-feature=+avxvnni --target-feature=-avxvnni
```

`vnni` 与有效 `threads=1`、`vector-mode=off/scalable` 的组合会报错，避免旧串行流水线
静默跳过内核或留下未下降的并行 IR。OpenMP 探测失败回退到单线程时也执行此校验。
`--int8-depthwise` 默认关闭；启用要求固定宽、非零 lane 的 MLIR 向量化
（`--vector-mode=fixed-width`，或解析为固定宽的 `auto`），否则报错。它是独立的
逐通道 signed-i8 扩宽乘加路径，不要求 VNNI，也不跨通道做 dot reduction。
`--int8-cast-chain` 默认关闭，启用只消除可证明安全的中间物化，保留有损 cast 算术。
三项优化均保持 opt-in，不根据仅有的 capability 宣称实际算子已改写。

为避免探针与最终 codegen 不一致，`auto/vnni` **拒绝所有 `--clang-arg`**，包括
`-mno-avxvnni`、`-Xclang`、响应文件、配置文件以及 `-D/-U` 能力宏覆盖；请改用专用
目标选项。`portable` 保留原有透传行为。`auto/vnni` 也不接受 `--march=native` 或
`--mcpu=native`，必须写明 CPU/features；任意策略下显式 `--target-triple` 与
`--march/--mcpu/--mtune=native` 的组合均拒绝，避免交叉目标偷用 host ISA。

`--emit-execution-plan` 的 codegen identity 包含请求策略及解析后的 `int8-target`。
`low_precision` 分别记录 capability、请求策略状态与实际改写记录；普通 `auto` 在有能力时标记
`pending_defaultization`。AVX512 编译成功不等于在对应硬件完成运行或性能验收。

### 3.8 `--tuning-profile` 和有限 Matmul 参数

P17 增加了可审计的有限编译期调优入口。默认
`--tuning-profile=stable` 保持既有 portable INT8 行为和稳定的 Matmul 参数；
`--tuning-profile=p16-int8` 只在目标探测确认 VNNI、固定宽向量和 OpenMP worker
条件满足时自动选择 P16 已配对验证的 INT8 row-dot、静态 depthwise 和 cast-chain
组合，否则记录 fallback 并回到 stable portable 路径。该 profile 不引入运行时
dispatcher，亦不改变公共 C ABI。

```bash
ncnn-compile model.param --tuning-profile=p16-int8 \
  --march=x86-64-v3 --target-feature=+avxvnni \
  --vector-mode=fixed-width --threads=6
```

可用的有限 tile 覆盖用于复现实验或 A/B，不进行无界搜索：

```text
--matmul-m-rows=<positive>
--matmul-acc-columns=<positive>
--row-chunk-lanes=<non-negative>
--matmul-i8-rows=<positive>
--matmul-i8-acc-columns=<positive>
```

未显式提供时使用 profile 的 bounded 值 `4/16/8/2/4`。最终 execution plan 的
`tuning` 对象记录 profile、status、五个 resolved 值和 `tuning-v1`；这些字段以及
profile 名称进入 plan hash/build identity，避免不同编译策略静默合并性能结果。
显式 `--int8-kernel/--int8-depthwise/--int8-cast-chain` 优先于 profile；因此可用
`--int8-kernel=portable` 审计 profile 的 tile 选择而不启用 VNNI。

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

### 4.2 `--emit-execution-plan`

显式请求一个只读的、确定性的执行计划 manifest。该文件独立于 ABI manifest，描述最终
bufferized IR 中可直接观察到的 operation、region、内部 allocation、静态字节数以及
copy/transpose/matmul/vector/parallel 摘要：

```bash
ncnn-compile model.param \
  --emit-manifest \
  --emit-execution-plan \
  --output-dir=dist/model
```

输出目录额外包含：

```text
dist/model/
├── model.h
├── model.json          # typed C ABI manifest
├── model.plan.json     # execution-plan manifest
└── libmodel.so
```

`model.plan.json` 的 `schema_version` 当前为 `1`，`kind` 为
`ncnn.model_execution_plan`。`plan_hash` 是静态 plan 与实际 code-generation identity 的
确定性摘要，`build_identity` 当前等于该摘要；`codegen_identity` 以 hex 编码保存原始
code-generation 参数，避免通过 pipeline 选项传输时被空格拆分。动态 shape、未知 element
size、整数溢出以及无法从静态 IR 证明的 peak workspace 均使用 `null` 和 diagnostics 记录，
绝不填充为零。静态 plan 的
`runtime_counters` 始终是 `not_collected`，`prepared_runner` 标记为
`available_in_performance_harness`；本选项不会合并 allocation、重写 alias、改变
layout/kernel 选择、注入 runtime，也不会修改公共 C ABI。数值性能 harness 可另行以
`NCNN_PERF_MODE=prepared` 复用已加载 Net/Extractor，或以
`NCNN_PERF_MODE=allocation_audit` 输出 ncnn 专用分配审计；两者均不进入 end-to-end
ratio 门禁。

不指定该选项时不会生成 plan 文件，现有产物、ABI manifest、导出符号和运行时行为保持不变。

### 4.3 `--profile`

显式构建带诊断回调的动态库，并自动发布与本次生成代码匹配的 execution plan：

```bash
ncnn-compile model.param \\
  --profile \\
  --emit-manifest \\
  --output-dir=dist/model
```

输出目录除头文件、ABI manifest 和动态库外，还包含 `model.plan.json`。运行时只有设置
`NCNN_PROFILE_PATH` 才会写出 profile sidecar；`NCNN_PROFILE_MODEL`、
`NCNN_PROFILE_PLAN_HASH`、`NCNN_PROFILE_TARGET` 和 `NCNN_PROFILE_THREADS` 可用于覆盖
sidecar 的环境元数据，未设置时使用编译期默认值；`NCNN_PROFILE_BUILD_IDENTITY` 可覆盖
build identity。`NCNN_PROFILE_MODE` 必须与待 join 的 perf 行一致（例如 prepared 示例需设置
`NCNN_PROFILE_MODE=prepared`）。profile 文件的 `kind` 为
`ncnn.model_execution_profile`，并带有版本化的 `attribution_revision=attribution-v1`。
默认 schema 1 保持 `process-cumulative` 聚合；设置 `NCNN_PROFILE_SCHEMA=2` 时输出一条
compact NDJSON 记录/次 invocation，记录 `invocation_id`、`complete` 和
`aggregation=per-invocation`。两种 schema 都记录
operation/parallel/allocation/copy/movement 事件、峰值 live bytes、顶层 wall-time、事件不匹配和
容量溢出；当前 movement ABI 已区分 transpose、pack 和 unpack，但 pack/unpack 未被所有
producer 覆盖。插桩只覆盖可安全识别的显式回调点；未覆盖、并发归属无法证明或无法证明的
时间保持 `null`/unknown，不能解读为零开销、完整硬件计数器或完整 reentrant attribution。

`--profile` 不改变 typed bare-pointer 公共 C ABI，但会改变内部动态库内容并增加诊断开销。
instrumented 时间只能用于 `perf_attribution_report.py` 的诊断归因，不能写入或替代正式
end-to-end ratio 门禁。归因工具严格校验 model、plan hash、build identity、attribution
revision、target、threads、mode 和 schema；其中 build identity 是包含 code-generation
identity 的 plan hash。报告包含 top-20 runtime cost、top-10 allocation/workspace、copy/layout
bytes 和 unknown/incomplete reasons；未知值保持 null，不以零代替。典型用法为：

```bash
python3 tools/perf_attribution_report.py \\
  --perf=perf.ndjson \\
  --plan=dist/model/model.plan.json \\
  --profile=profile.json \\
  --mode=prepared \\
  --output=dist/model/attribution.json
```

不指定 `--profile` 时，默认流水线不插入回调、不链接 profile runtime，也不会额外发布 plan
文件。

### 4.4 `--emit-manifest`

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
llvm-as ...
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

---

## 11. 面试问题与答案（精度策略 / 目标代码生成 / Manifest）

### 11.1 `--precision` 和 `--target-triple` 怎样联动？

答案：精度策略并不是只看模型 dtype，而是同时校验目标是否具备对应的硬件能力。
`compiler/tools/ncnn-mlir-driver.cpp` 在解析 `--precision`、`--fp16-accumulator`、
`--allow-fallback`、`--target-triple`、`--march`、`--mcpu`、`--target-feature` 之后构造
`ncnn_mlir::TargetSpec`，调用 `resolve_precision_policy`：

- 先用 `infer_target_capabilities` 推断 `TargetCapabilities`（FP16 storage、FP16
  arithmetic、BF16、INT8 四项布尔）；
- 再用 `validate_precision_target` 校验当前 `--precision` 是否被目标支持；
- 最后，如果 `--precision=fp16` 且要求 FP16 accumulator，会再次校验
  `fp16_arithmetic`；缺失时如果没有 `--allow-fallback` 就直接报错。

需要特别说明的是，这一组校验早于 `ncnn-graph→ncnn-dialect` 导入，所以一旦不支持
就会在 driver 阶段拒绝，不会留下一个“看起来能跑但运行时崩溃”的产物。

### 11.2 阶段七“目标特化”到底特化在哪里？

答案：落地在三个相互独立但必须一致的层次。

1. **能力层**：`compiler/lib/Support/Precision.cpp` 的 `infer_target_capabilities`
   按 triple + march + mcpu + features 推断能力，x86-64/AArch64/RISC-V 各自有不同的
   启发式 fallback。例如 RISC-V 需要显式 `zvfh` 才会被认为是 vector FP16 arithmetic，
   而 `zfh` 只算 scalar FP16 arithmetic。
2. **代码生成层**：`compiler/tools/ncnn-compile.cpp` 把 `--target-triple`、
   `--march`、`--mcpu`、`--mtune`、`--target-feature` 透传给 `clang`（含 LLVM
   `TargetMachine` 的 `target=` 和 `-Xclang -target-feature`）。
3. **契约层**：编译完成后 `Manifest::Target` 写回 JSON，告诉调用方这次产物的
   target provenance 和 `execution_profile`，下游不能把 `x86-64-fp16-storage-fp32`
   误报成原生 FP16。

早期实现是仅靠 `--target-feature` 透传给 Clang，不做能力校验，结果出现
`--precision=int8` 在没有 VNNI 的目标上能“成功”编译但跑出完全不同的数值。
阶段七把能力校验提前到 driver，消除了“能编但不算”这个语义漏洞。

### 11.3 `feature_enabled(target, names, fallback)` 的语义是什么？

答案：它把"用户显式开关"和"架构默认值"合在一起：

```text
enabled = fallback  // 架构默认
for feature in target.features:
    if feature matches one of names:
        enabled = not feature.startswith('-')
```

实现位置在 `compiler/lib/Support/Precision.cpp`，对应 `compiler/include/ncnn-mlir/Support/Precision.hpp`
的同名 `TargetSpec`。`fallback` 表示“当前架构（如 x86-64）默认是否具备该能力”，
而 `target.features` 表示“用户是否用 `+`/`-` 覆盖”。注意 `fallback` 不是 boolean
而是 `initializer_list` 里那些“架构默认就开”的判断结果。

### 11.4 `validate_precision_target` 抛出的错误为什么比阶段六更长？

答案：阶段六的错误只说“目标不支持 fp16”。阶段七的诊断会包含具体目标描述和缺失能力：

```text
precision fp16 is not supported by target 'aarch64-unknown-linux-gnu/armv8.2-a';
missing native FP16 arithmetic support; specify a matching --march, --mcpu,
or --target-feature
```

由 `target_description()` 输出 triple + `mcpu`（或 `march` 退化），错误信息再明确指出
FP16 storage 还是 FP16 arithmetic，方便使用者立刻知道是 march 缺了 `+fp16` 还是 mcpu
选错了。

### 11.5 为什么 Manifest 要额外写 `target` 而不是把 triple 塞进 `precision_policy`？

答案：`precision_policy` 描述的是“这次编译采用什么数值规则”，而 `target` 描述的是
“目标硬件的 provenance”。两者解耦的原因有两个：

1. **诊断稳定性**：`precision_policy` 可能在不同 host 上相同（都是 `f16` accumulator），
   但 `target` 反映具体代码生成参数（`mcpu=sapphirerapids`、`+avx512fp16`）。当数值偏差
   出问题时，先看 `target` 再看 `precision_policy`。
2. **profile 命名**：`precision_execution_profile` 依赖目标架构和 accumulator：FP16
   arithmetic 在 x86-64 上叫 `x86-64-avx512-fp16`，在 aarch64 上叫 `aarch64-fp16`，
   在 RISC-V RVV 上叫 `riscv-rvv-fp16`。profile 和 capability 必须能同时索引。

### 11.6 `x86-64-fp16-storage-fp32` 这个 profile 的含义是什么？

答案：明确是 fallback，不能被算作原生 FP16。规则如下：

- `--precision=fp16`、`--fp16-accumulator=f16` 同时设置，但目标只有 `+f16c` 没有
  `+avx512fp16`；
- 用户给了 `--allow-fallback`；
- `resolve_precision_policy` 把 accumulator 强制改为 `f32`，`used_fallback = true`；
- manifest profile 写 `x86-64-fp16-storage-fp32`；
- MLIR 上的 `ncnn.fp16_accumulator` 也被改为 `"f32"`，下游 Linalg 会插
  `arith.extf`/`arith.truncf` 完成 FP16 storage、FP32 accumulation。

这条 profile 的意义在于：避免出现“产物名为 fp16，权重也是 fp16，但实际是按 FP32
accumulation 算”的报告偏差。

### 11.7 阶段七遇到的坑：build 目录被误复用

答案：阶段六 commit 之后，有人直接在旧的 build 目录里继续构建。`ncnn-mlir-opt` 仍然指向
旧目标的 `.mlir` 通过而驱动层 `--precision=fp16` 报错 “`feature +avx512fp16` 不可识别”。
因为 `clang-21 -x ir -c` 的目标参数和驱动参数在两个不同的层生效，旧 build 会保留旧
`model.ll`，而驱动层会按新参数重新生成 capability 报告。**修复**：门禁脚本里强制
`/tmp/ncnn-compiler-stage-<name>` 作为新 build 目录并把 `--parallel` 写进所有
`cmake --build` 调用。

### 11.8 阶段七遇到的坑：manifest 字段被重复写入

答案：旧实现里 `generate-ncnn-c-api` pass 写一次 manifest，C++ driver 再写一次，
导致 `target` 字段出现两次，第二组覆盖第一组。**修复**：在 driver 端只在 manifest 不存在
`target` 字段时写入，避免重复；并把 `precision_execution_profile` 写进 IR 上的
`ncnn.precision` 字符串，避免下游 IR 通过器误读。

### 11.9 阶段七遇到的坑：x86-64 “原生 FP16” 在 RISC-V 上“看上去也成功”

答案：因为 `feature_enabled` 在 RISC-V 上默认把 `zvfh` 标为开，如果用户
`--target-feature=+zfh -zvfh` 写错顺序，capability 会变成“FP16 arithmetic 不开”
但 `--precision=fp16` 还能继续。**修复**：`feature_enabled` 改成按出现顺序覆盖
（不是按 `+`/`-` 一次性取最后），并要求同一特征多次出现时给出 `compiler warning`。

### 11.10 阶段七的回归测试为何没加 INT8 + AVX-VNNI 的跑分？

答案：当前 CI 宿主只有 x86-64 sapphire rapids，AVX-VNNI 存在但 AVX512-VNNI
不存在。`ncnn-compile --precision=int8` 在这两个 host 上都生成 INT32 accumulation，
差异主要来自 loop unroll 和 L1 cache。P4 已落地受限的 int8 row-dot/requant
路径；仍未落地的是面向原生 `vpdpbusd` 的 VNNI 选型与指令生成，继续作为 `P8`
路线图审计项，不在阶段七硬塞跑分；阶段七的“目标代码生成”以**指令级静态检查**
和 **IR 端 capability**为验收点。

### 11.11 一次跨架构的 FP16 静态验证具体跑什么？

答案：`compiler/test/Native/check_fp16_arithmetic.py`：

- `clang-21 -x ir -S model.ll -target aarch64-unknown-linux-gnu -march=armv8.2-a+fp16`
  生成 `aarch64-armv8.2-fp16.s`，正则检查 `fmul|add|mla h` 至少出现一次。
- 同样以 `-target riscv64-unknown-linux-gnu -march=rv64gcv_zfh_zvfh` 检查
  `fmul.h/fadd.h/fmadd.h`。
- 同时检查 `model.ll` 包含 `fmul half` / `fadd half`，并通过 `llvm-mca` 跑出
  `Block RThroughput` 写入 `fp16-performance.json`。
- manifest 校验 `target.execution_profile == "x86-64-avx512-fp16"` 且
  `+avx512fp16` 在 `features` 中。

这里**不**执行 foreign binary（没有 AArch64 / RISC-V 硬件），所以测试报告必须明确
“静态指令验证”而不是“目标硬件运行验证”。
