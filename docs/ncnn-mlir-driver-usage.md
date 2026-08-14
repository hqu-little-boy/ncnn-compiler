# ncnn-mlir-driver 使用文档

> 编译器**前端入口**：把 ncnn 模型（`.param` + `.bin`）解析并提升为 **MLIR `ncnn` 方言模块**，
> 是唯一产品入口 `ncnn-compile` 所使用的固定 ncnn→TOSA→Linalg→MemRef→LLVM 管线的起点。
> `ncnn-mlir-opt` 和 Python 脚本只用于开发调试，不是另一条产品 pipeline。
> 参数解析用 LLVM 自带的 `llvm::cl`（`mlir-opt` 同款 CommandLine 库）。

> 注意：项目根目录是 `/mnt/ncnn-compiler`。本文中相对路径均以 `compiler/` 为基准。
> 构建目录名可自定（下文用 `build`）。

---

## 1. 构建

```bash
cd /mnt/ncnn-compiler/compiler
cmake -S . -B build -G "Unix Makefiles"
cmake --build build --parallel
```

完整构建必须使用上述 `cmake --build build --parallel` 命令，统一通过 CMake 的
`--parallel` 选项启用并行构建。

产物：
- `build/tools/ncnn-mlir-driver` —— `.param/.bin → MLIR` 前端驱动（本文主角）。
- `build/bin/ncnn-mlir-opt` —— 基于 `MlirOptMain` 的 `mlir-opt` 克隆，注册了 ncnn 方言
  及**全部上游方言/扩展/pass**，因此支持标准 mlir-opt 选项（`-o`、`--mlir-print-op-generic`、
  `--verify-diagnostics`、`--allow-unregistered-dialect`）与所有 pass（`--canonicalize`、
  `--tosa-to-linalg`、`--convert-arith-to-llvm` …）。做 MLIR 的解析/校验/round-trip 与
  下降调试（lit 测试用它），**不消费 ncnn 模型**。

依赖：LLVM/MLIR 21（Debian 包 `llvm-21-dev` + `libmlir-21-dev`）、CMake 3.20+、C++23
编译器、Python 3（lit 和 fixture 生成）、GTest 开发包以及 `clang-21`、`mlir-translate-21`、
`llvm-nm-21`、`llvm-readelf-21`、`FileCheck-21`。CMake 通过
`find_package(MLIR CONFIG)` 定位（config 目录 `/usr/lib/llvm-21/lib/cmake/mlir`），
链接 `MLIRIR / MLIRSupport / MLIRFuncDialect / MLIRArithDialect / MLIRParser` 等。

两个工具的区别：

| | `ncnn-mlir-driver` | `ncnn-mlir-opt` |
|---|---|---|
| 输入 | ncnn 模型 `.param`(+`.bin`) | MLIR 文本 `.mlir` |
| 输出 | MLIR 模块（或原始 parsed-graph） | MLIR 文本（原样重印） |
| 职责 | ncnn 模型 → ncnn 方言 IR | MLIR 解析/校验/pass 管线 |
| 依赖 | NCNNGraph + NCNNImporter + NCNNDialect | NCNNDialect + NCNNToFunc + NormalizeNCNN + NCNNToTosa + VerifyNoNCNNOps + NCNNPipelines + 上游方言/pass |

---

## 2. 命令行接口

```
ncnn-mlir-driver [options] <input .param file>
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `<input>`（位置参数，必填） | ncnn 网络结构文件 `.param` | — |
| `--bin=<path>` | 权重文件 `.bin` | 由 `<input>` 把末尾 `.param` 换成 `.bin` 推导 |
| `--input-shape=<CxHxW>` | 可重复；extent 为正整数或 `?`，按 Input source order 匹配 | 无 |
| `--input-shape=*` | 受限动态 rank 1..4；必须是唯一 occurrence | 无 |
| `--input-dim-constraint=<INPUT:DIM:min=N,multiple=N>` | 可重复；约束 fixed-rank 动态输入维 | 无 |
| `--precision=<mode>` | `auto`、`f32`、`fp16`、`bf16` 或 `int8` | `auto` |
| `--fp16-accumulator=<mode>` | FP16 accumulator：`f16` 或 `f32` | `f16` |
| `--allow-fallback` | 允许目标缺少原生 FP16 arithmetic 时使用 FP32 accumulator | 关闭 |
| `--target-triple=<triple>` | 目标 Linux ELF triple，用于 capability 检查 | 空 |
| `--march=<arch>` / `--mcpu=<cpu>` | 目标 ISA 或具体 CPU | 空 |
| `--target-feature=<feature>` | 可重复的 `+feature`/`-feature` | 无 |
| `-o <path>` | 产物输出路径，`-` 写到 stdout | `-`（stdout） |
| `--emit=<stage>` | 选择要发出的阶段（见下） | `mlir` |
| `--verify` | 对导入的 MLIR 模块跑校验器 | 开 |

`--emit` 的取值：

- `mlir`（默认）：**MLIR `ncnn` 方言模块**——类型化 SSA DAG，权重为 `ncnn.const`，
  每个值带推断出的 `RankedTensorType`，是后续下降到 tosa/linalg 的起点。
- `parsed-graph`：原始解析出的 ncnn 计算图（layer/blob/参数 + 绑定的权重形状），
  最贴近 `.param`/`.bin` 原貌，适合排查解析问题。

> `ncnn-mlir-driver` 当前只输出 `parsed-graph` 或 ncnn 方言 MLIR。后续产品阶段由
> `ncnn-compile` 调用固定 pipeline；这里的 `--emit` 不是多路径产品接口。

精度策略在导入的 `ncnn.model` 上保存为 `ncnn.precision`、`ncnn.fp16_accumulator`，以及必要时的
`ncnn.precision_fallback`。目标参数必须和最终 Clang 代码生成使用的参数一致；通常应通过
`ncnn-compile` 传递，而不是单独调用 driver。

查看帮助（`--help` 已用 `OptionCategory` 收窄，不会被链接 libLLVM 引入的海量
codegen option 淹没）：

```bash
./build/tools/ncnn-mlir-driver --help
```

```
OVERVIEW: ncnn-mlir-driver -- compile ncnn .param/.bin toward MLIR/native code

USAGE: ncnn-mlir-driver [options] <input .param file>

OPTIONS:

Generic Options:

  --help                  - Display available options (--help-hidden for more)
  --help-list             - Display list of available options (--help-list-hidden for more)
  --version               - Display the version of this program

ncnn-mlir-driver options:

  --bin=<path>            - Weight file (.bin). Defaults to <input> with .param replaced by .bin
  --emit=<value>          - Select the stage to emit:
    =parsed-graph         -   Raw parsed ncnn graph (param + bound weights)
    =mlir                 -   MLIR module in the ncnn dialect (default)
  --input-shape=<shape|*> - Input shape override as CxHxW; '?' is a dynamic extent, and '*' requests dynamic rank 1..4
  -o <path>               - Output file. '-' writes to stdout (default)
  --verify                - Re-verify the imported MLIR module (default: on)
```

---

## 3. 执行示例

以自带的 SqueezeNet v1.1 为例（`ncnn/examples/squeezenet_v1.1.{param,bin}`）。

### 3.1 默认：产 ncnn 方言 MLIR 模块（自动推导 .bin）

```bash
cd /mnt/ncnn-compiler/compiler
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param
```

`.bin` 自动由 `.param` 推导，输出（节选）：

```mlir
module {
  ncnn.model @model {
    %input = ncnn.input {blob_name = "data", layer_name = "data"} : tensor<3x227x227xf32>
    %cst = ncnn.const {name = "conv1.weight.0", value = dense<...>} : tensor<64x3x3x3xf32>
    %cst_0 = ncnn.const {name = "conv1.weight.1", value = dense<...>} : tensor<64xf32>
    %0 = ncnn.convolution %input, %cst, %cst_0 {dilation_h = 1 : i64, dilation_w = 1 : i64,
         has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, ncnn.name = "conv1",
         ncnn.source_layer = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64,
         pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64}
         : (tensor<3x227x227xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
         -> tensor<64x113x113xf32>
    %1 = ncnn.relu %0 {ncnn.name = "relu_conv1", ncnn.source_layer = 2 : i64}
         : (tensor<64x113x113xf32>) -> tensor<64x113x113xf32>
    ...
    ncnn.output %N {blob_name = "prob"} : tensor<1000xf32>
  }
}
```

要点：
- **输入**是 `ncnn.input`，形状 `[C,H,W]`（ncnn 原生 CHW）。
- **权重**被抬成 `ncnn.const`（`%cst`=卷积核 `[O,I,H,W]`、`%cst_0`=bias `[O]`），
  作为 `ncnn.convolution` 的操作数。
- **ncnn 数字参数**变成具名强类型属性（`kernel_h`、`stride_h`、`pad_*`、`has_bias`…）。
- **每个值带推断出的 ranked shape/元素类型**。传统模型通常生成静态类型；`?` 可保留固定
  rank 动态 extent，`*` 为受限模型生成 rank 1..4 specialization。具体算子仍须有对应动态
  shape inference 和 lowering，不能由 ABI 支持推导为任意动态图均可编译。
- **来源溯源**保留在 `ncnn.name` / `ncnn.source_layer` discardable 属性里。

MLIR 模块格式详见 [ncnn-ir-format.md](ncnn-ir-format.md)。

### 3.2 显式指定 .bin

```bash
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param \
  --bin=test/third_party/ncnn/examples/squeezenet_v1.1.bin
```

### 3.3 指定动态或省略的 Input shape

固定 shape 或固定 rank 动态 extent：

```bash
./build/tools/ncnn-mlir-driver model.param --input-shape=3x224x224
./build/tools/ncnn-mlir-driver model.param --input-shape=3x?x?
./build/tools/ncnn-mlir-driver model.param \
  --input-shape=3x?x? \
  --input-dim-constraint=0:1:min=32,multiple=32 \
  --input-dim-constraint=0:2:min=32,multiple=32
```

多输入时重复指定，数量必须等于全部 Input 数量，并按 source-layer 顺序绑定。已有静态尺寸的
Input 不会被改写，但仍占一个条目。`--input-shape=*` 必须独占，只支持一个未声明尺寸的 Input、
一个输出以及 identity/ReLU shape-preserving 图，并生成 rank 1..4 specialization。

维度约束以 `#ncnn.dim_constraint` 结构化属性保存在 `ncnn.shape_constraints` 中，只允许指向
动态维。`convert-ncnn-model-to-func` 会将该属性传播到入口函数，后续 C ABI wrapper 和
shape inference wrapper 使用同一约束。

### 3.4 查看原始解析图

```bash
./build/tools/ncnn-mlir-driver --emit=parsed-graph test/third_party/ncnn/examples/squeezenet_v1.1.param
```

```
ncnn_graph: 75 layers, 83 blobs
inputs: 1
  - data
outputs: 1
  - prob
layers:
  [  0] Input          data                   in=[] out=[data] {0=227 1=227 2=3}
  [  1] Convolution    conv1                  in=[data] out=[conv1] {0=64 1=3 2=1 3=2 4=0 5=1 6=1728} w=[[64,3,3,3:f32:6912B],[64:f32:256B]]
  [  2] ReLU           relu_conv1             in=[conv1] out=[conv1_relu_conv1] {0=0}
  ...
  [ 74] Softmax        prob                   in=[pool10] out=[prob] {0=0}
```

参数按 ncnn 原生 key（`{0=227 1=227 2=3}` 即 Input 的 `w=227 h=227 c=3`），
`w=[...]` 是绑定的权重张量形状。格式详见 [parsed-graph-format.md](parsed-graph-format.md)。

### 3.5 写到文件 / 管道

```bash
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param -o /tmp/squeezenet.mlir
# 诊断走 stderr、产物走 stdout，可直接管道：
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param 2>/dev/null \
  | ./build/bin/ncnn-mlir-opt   # 再用 opt 做一次 round-trip 校验
```

### 3.6 关闭校验

```bash
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param --verify=false
```

---

## 4. 退出码与错误

- 成功：退出码 `0`，产物写到 stdout 或 `-o`。
- 失败：退出码 `1`，错误信息写到 stderr。

文件不存在示例：

```bash
$ ./build/tools/ncnn-mlir-driver /tmp/nope.param
error: failed to load model: cannot open param file: /tmp/nope.param
$ echo $?
1
```

其他失败点及对应信息：

| 阶段 | stderr 前缀 |
|------|-------------|
| 加载 `.param`/`.bin` | `error: failed to load model: ...` |
| 导入为 MLIR 模块 | `error: failed to import graph: layer N (Type, name): ...` |
| MLIR 校验（`--verify` 开时） | `error: MLIR module verification failed`（具体诊断另起一行） |
| 打开/写出 `-o` 文件 | `error: cannot open output file ...` / `error: failed while writing ...` |

---

## 5. 内部流水线

```
.param/.bin
  └─ ncnn_graph::Graph::load          →  原始计算图 (--emit=parsed-graph 到此为止)
       └─ ncnn_importer::import_graph →  MLIR ncnn 模型（model/input/const/output + 计算算子）
             └─ mlir::verify            (--verify)
                  └─ ModuleOp::print     →  MLIR 文本 (--emit=mlir，默认)
```

Importer 输出后必须先运行 `--convert-ncnn-model-to-func`，一次性把输入、输出和权重
转移到 `func.func` 参数/结果与 `arith.constant`，并删除模型边界算子。该 pass 是
`--normalize-ncnn` 和各目标 conversion 的固定前置。`--convert-ncnn-to-tosa` 采用部分
转换契约：只转换明确支持的算子，其他 ncnn op 可在开发阶段暂时残留；产品严格 pipeline
的最终 gate 会拒绝任何未消除的 ncnn op。
模型函数化使用 MLIR full dialect conversion：`ModelOp` pattern 移动原 region 并建立函数
边界，`ConstOp` pattern 转换权重，conversion driver 负责 SSA remap、合法性和 rewrite
失败处理，不会复制整个模型计算图；这不是整 module 的事务回滚。
`--normalize-ncnn` 会先只读验证并预计算 SAME padding，再原地提交全部规范化修改；验证失败
不会产生部分结果。`--convert-ncnn-to-tosa` 使用 MLIR dialect conversion patterns，借助
`TypeConverter` 的 source/target materialization 在 CHW 和 NHWC 间转换，并由 conversion
driver 管理 SSA remap 和 rewrite 失败处理。两者均不 clone 整个 module，也不提供整 module
事务回滚语义。

所有项目 pass 的 CLI 名称、选项、dependent dialect、factory 和注册均由统一的
`ncnn-mlir/Passes.td` 生成，`ncnn-mlir-opt` 通过一次 `registerNCNNPasses()` 完成注册。

```bash
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param 2>/dev/null \
  | ./build/bin/ncnn-mlir-opt --convert-ncnn-model-to-func

# 严格 SqueezeNet → TOSA：
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param 2>/dev/null \
  | ./build/bin/ncnn-mlir-opt --ncnn-to-tosa-pipeline

# 已经是 Linalg IR 时，转换为调用方提供输出缓冲区的 memref IR：
./build/bin/ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline model.linalg.mlir

# 从 ncnn 模型生成经过严格链接和符号检查的动态库：
./build/tools/ncnn-compile \
  test/third_party/ncnn/examples/squeezenet_v1.1.param
```

`--ncnn-to-tosa-pipeline` 内部固定运行：

```text
convert-ncnn-model-to-func
→ normalize-ncnn
→ convert-ncnn-to-tosa
→ canonicalize → cse
→ verify-no-ncnn-ops
```

产品 pipeline 还固定继续执行 `ncnn-tosa-to-linalg-pipeline`、
`ncnn-linalg-to-memref-pipeline`、`generate-ncnn-c-api` 和
`ncnn-memref-to-llvm-pipeline`，随后完成 LLVM IR 翻译、目标代码编译、严格链接、符号/依赖
审计和头文件生成。单独运行一个 pass 或 pipeline 不等于产品编译成功。

单独运行 `--convert-ncnn-to-tosa` 用于开发和调试某条 conversion，不表示模型已完整
lowering。可转换的算子实例在 conversion target 中是 illegal，明确不支持的实例和未知 ncnn
op 则允许残留；`--ncnn-to-tosa-pipeline` 以最终“无 ncnn op 残留”为成功标准。函数签名
保持 CHW，布局 materialization 只出现在函数体内部和不同合法性区域的值边界。

`--ncnn-linalg-to-memref-pipeline` 固定执行 function-boundary One-Shot Bufferize、
`buffer-results-to-out-params` 和标准 deallocation pipeline。对于 SqueezeNet，入口契约为：

```mlir
func.func @model(%input: memref<3x227x227xf32>,
                 %output: memref<1000xf32> {bufferize.result})
```

函数无返回值；`%input` 和 `%output` 由调用方拥有，不会被函数释放。
完整 SqueezeNet 的重复调用验收需要产品 LLVM lowering 和最终 C ABI。

`--ncnn-memref-to-llvm-pipeline` 消除 Linalg/Affine/SCF/Math/Arith/MemRef/Func/CF，并将
`math.exp` 映射到系统 `expf`。C++ `ncnn-compile` driver 直接调用
`mlir-translate-21`、`clang-21 -fPIC` 和严格共享库链接；最终 `.so` 只依赖 libc/libm，
未定义符号只允许 `malloc`、`free`、`erfcf`、`erff`、`expf`、`powf`、`memcpy`、`memset`，且禁止 `memrefCopy` 和
runner/project runtime。

`ncnn-compile model.param` 默认从文件名推导模型名和同名输出目录，并自动使用同目录的
`model.bin`。默认目录只包含用户需要的两个文件：

```text
model/
├── model.h
└── libmodel.so
```

可用 `--model-name` 覆盖模型名，使用 `-o/--output-dir` 覆盖输出目录，使用 `--bin` 覆盖权重
路径。公共函数参数顺序是全部输入 tensor 参数组、全部输出数据指针、shape-only 动态输出 data
capacity、数据依赖输出元数据。
元素类型由 C 指针类型表达；当前 ncnn 模型数据主路径为 f32。静态 tensor 只传 data，固定 rank
动态输入追加 shape，受限动态 rank 再追加 rank。只有静态元素数可知时才生成 `ELEMENTS`；
shape-only 动态输出的执行入口追加以 data 元素数计的 `uint64_t capacity`，数据依赖输出生成
`MAX_DIMn/MAX_ELEMENTS` 并追加以 extent 数量计的 shape capacity。状态码 `0..5` 分别表示成功、
空指针、非法 shape、约束违反、shape 算术溢出和输出容量不足。

`--emit` 用于保留调试所需的中间 MLIR，可重复指定或使用逗号分隔：

```bash
# 只保留 ncnn、TOSA 和 LLVM dialect MLIR
ncnn-compile model.param --emit ncnn,tosa --emit llvm

# 保留全部中间 MLIR
ncnn-compile model.param --emit all
```

支持的阶段为 `ncnn`、`tosa`、`linalg`、`memref`、`capi` 和 `llvm`。`llvm` 指 LLVM dialect
MLIR，不是临时 LLVM IR `.ll`；内部 `.ll` 和 object 不属于公共产物。

ABI manifest 默认是内部临时产物。需要检查 ABI 或供内部测试使用时，显式增加
`--emit-manifest`，输出目录中会额外生成 `<model>.json`。

代码生成选项包括：

```text
-O0 / -O1 / -O2 / -O3
--target-triple <triple>
--march <architecture>
--mcpu <cpu>
--mtune <cpu>
--target-feature <+feature|-feature>  # 可重复
--sysroot <path>
-g / --debug-info
--clang-arg <argument>                # 高级编译参数，可重复
--linker-arg <argument>               # 高级链接参数，可重复
-v / --verbose
```

默认优化等级为 `-O3`。交叉编译时调用方必须提供与 target triple 匹配的 clang 工具链和
sysroot；当前产物和链接审计只支持 64 位 Linux ELF target。`--verify-execution` 只适用于
生成库可在当前宿主执行的情况。以 `-` 开头的高级参数既可写成
`--clang-arg=-ffast-math`，也可写成 `--clang-arg -ffast-math`；target feature 和 linker
参数同理。

输出先在系统临时目录完成编译、严格链接和符号审计，成功后才发布。编译失败不会覆盖
上一份有效 `.h/.so`。成功重编译会移除该输出目录中的旧编译产物和不再请求的 MLIR；如果目录
含有不属于 `ncnn-compile` 的文件，命令会拒绝执行，避免误删用户数据。

CMake 构建后入口位于 `build/tools/ncnn-compile`。执行 `cmake --install build` 会把
`ncnn-compile`、Python 调试脚本、`ncnn-mlir-driver` 和 `ncnn-mlir-opt` 安装到同一个 `bin` 目录，
安装后的 `ncnn-compile` 会自动发现这些内部工具。

SqueezeNet 默认接口为：

```c
#define SQUEEZENET_V1_1_INPUT1_ELEMENTS UINT64_C(154587)
#define SQUEEZENET_V1_1_OUTPUT1_ELEMENTS UINT64_C(1000)
int squeezenet_v1_1(const float *input1, float *output1);
```

动态库导出模型执行函数；存在 shape-only 动态输出时还导出 `<model>_infer_output_shapes`。内部
模型为 private，不生成 `_mlir_ciface_*`。输入输出由调用方持有，采用 native-endian 连续数组；
错误码由生成头文件中的 `NCNN_STATUS_*` 宏定义，而不是统一返回 1。

---

## 6. 源码位置

- 前端驱动：`compiler/tools/ncnn-mlir-driver.cpp`
- opt 工具：`compiler/bin/ncnn-mlir-opt.cpp`（+ `compiler/bin/RegisterNCNNDialects.h`）
- importer：`compiler/lib/Importer/NCNNImporter.cpp`
- ncnn 方言：`compiler/lib/Dialect/NCNN/IR/`（+ `compiler/include/ncnn-mlir/Dialect/NCNN/IR/`）
- 转换/规范化：`compiler/lib/{Conversion,Transforms}/`
- lowering pipeline：`compiler/lib/Pipelines/NCNNPipelines.cpp`
- 稳定 C++ compiler driver：`compiler/tools/ncnn-compile.cpp`
- Python 调试流水线：`compiler/tools/compile_ncnn_model.py`（快速验证阶段和观察中间过程）
- 算子数值适配避坑指南：`compiler/docs/operator-numerical-validation-guide.md`
- SqueezeNet pipeline 测试：`compiler/test/Pipelines/squeezenet-m2.mlir`
- 解析器/数据模型：`lib/Graph/`（`ncnn_graph` 库）
- 构建配置：各目录 `CMakeLists.txt`（顶层 `compiler/CMakeLists.txt`）
