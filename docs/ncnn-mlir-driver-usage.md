# ncnn-mlir-driver 使用文档

> 编译器**前端入口**：把 ncnn 模型（`.param` + `.bin`）解析并提升为 **MLIR `ncnn` 方言模块**，
> 是后续 ncnn→TOSA、ncnn→Linalg/SCF、ncnn→Host 多路径下降管线的起点。
> 参数解析用 LLVM 自带的 `llvm::cl`（`mlir-opt` 同款 CommandLine 库）。

> 注意：项目根目录是 `/mnt/ncnn-compiler`。本文中相对路径均以 `compiler/` 为基准。
> 构建目录名可自定（下文用 `build`）。

---

## 1. 构建

```bash
cd /mnt/ncnn-compiler/compiler
cmake -S . -B build -G "Unix Makefiles"
cmake --build build --target ncnn-mlir-driver ncnn-mlir-opt -j
```

产物：
- `build/tools/ncnn-mlir-driver` —— `.param/.bin → MLIR` 前端驱动（本文主角）。
- `build/bin/ncnn-mlir-opt` —— 基于 `MlirOptMain` 的 `mlir-opt` 克隆，注册了 ncnn 方言
  及**全部上游方言/扩展/pass**，因此支持标准 mlir-opt 选项（`-o`、`--mlir-print-op-generic`、
  `--verify-diagnostics`、`--allow-unregistered-dialect`）与所有 pass（`--canonicalize`、
  `--tosa-to-linalg`、`--convert-arith-to-llvm` …）。做 MLIR 的解析/校验/round-trip 与
  下降调试（lit 测试用它），**不消费 ncnn 模型**。

依赖：LLVM/MLIR 21（Debian 包 `llvm-21-dev` + `libmlir-21-dev`）。CMake 通过
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
| `-o <path>` | 产物输出路径，`-` 写到 stdout | `-`（stdout） |
| `--emit=<stage>` | 选择要发出的阶段（见下） | `mlir` |
| `--verify` | 对导入的 MLIR 模块跑校验器 | 开 |

`--emit` 的取值：

- `mlir`（默认）：**MLIR `ncnn` 方言模块**——类型化 SSA DAG，权重为 `ncnn.const`，
  每个值带推断出的 `RankedTensorType`，是后续下降到 tosa/linalg 的起点。
- `parsed-graph`：原始解析出的 ncnn 计算图（layer/blob/参数 + 绑定的权重形状），
  最贴近 `.param`/`.bin` 原貌，适合排查解析问题。

> `--emit` 是驱动最关键的设计点：后续 `tosa`、`linalg`、`llvm`、`library`
> 等下降阶段会加入到这个同一个枚举里，CLI 表面保持稳定。

查看帮助（`--help` 已用 `OptionCategory` 收窄，不会被链接 libLLVM 引入的海量
codegen option 淹没）：

```bash
./build/tools/ncnn-mlir-driver --help
```

```
OVERVIEW: ncnn-mlir-driver -- compile ncnn .param/.bin toward MLIR/native code

USAGE: ncnn-mlir-driver [options] <input .param file>

OPTIONS:

ncnn-mlir-driver options:
  --bin=<path>    - Weight file (.bin). Defaults to <input> with .param replaced by .bin
  --emit=<value>  - Select the stage to emit:
    =parsed-graph -   Raw parsed ncnn graph (param + bound weights)
    =mlir         -   MLIR module in the ncnn dialect (default)
  -o <path>       - Output file. '-' writes to stdout (default)
  --verify        - Re-verify the imported MLIR module (default: on)
```

---

## 3. 执行示例

以自带的 SqueezeNet v1.1 为例（`ncnn/examples/squeezenet_v1.1.{param,bin}`）。

### 3.1 默认：产 ncnn 方言 MLIR 模块（自动推导 .bin）

```bash
cd /mnt/ncnn-compiler/compiler
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param
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
- **每个值带推断出的静态形状/元素类型**（`tensor<64x113x113xf32>`）。
- **来源溯源**保留在 `ncnn.name` / `ncnn.source_layer` discardable 属性里。

MLIR 模块格式详见 [ncnn-ir-format.md](ncnn-ir-format.md)。

### 3.2 显式指定 .bin

```bash
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param \
  --bin=../ncnn/examples/squeezenet_v1.1.bin
```

### 3.3 查看原始解析图

```bash
./build/tools/ncnn-mlir-driver --emit=parsed-graph ../ncnn/examples/squeezenet_v1.1.param
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

### 3.4 写到文件 / 管道

```bash
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param -o /tmp/squeezenet.mlir
# 诊断走 stderr、产物走 stdout，可直接管道：
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param 2>/dev/null \
  | ./build/bin/ncnn-mlir-opt   # 再用 opt 做一次 round-trip 校验
```

### 3.5 关闭校验

```bash
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param --verify=false
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
`--normalize-ncnn` 和各目标 conversion 的固定前置。`--convert-ncnn-to-tosa` 采用长期
部分转换契约：只转换明确支持的算子，其他 ncnn op 可留给后续 Linalg/SCF 或 Host 路径。

```bash
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param 2>/dev/null \
  | ./build/bin/ncnn-mlir-opt --convert-ncnn-model-to-func

# 严格 SqueezeNet → TOSA：
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param 2>/dev/null \
  | ./build/bin/ncnn-mlir-opt --ncnn-to-tosa-pipeline
```

`--ncnn-to-tosa-pipeline` 内部固定运行：

```text
convert-ncnn-model-to-func
→ normalize-ncnn
→ convert-ncnn-to-tosa
→ canonicalize → cse
→ verify-no-ncnn-ops
```

单独运行 `--convert-ncnn-to-tosa` 用于开发和调试某条 conversion，不表示模型已完整
lowering。`--ncnn-to-tosa-pipeline` 以最终“无 ncnn op 残留”为成功标准。

---

## 6. 源码位置

- 前端驱动：`compiler/tools/ncnn-mlir-driver.cpp`
- opt 工具：`compiler/bin/ncnn-mlir-opt.cpp`（+ `compiler/bin/RegisterNCNNDialects.h`）
- importer：`compiler/lib/Importer/NCNNImporter.cpp`
- ncnn 方言：`compiler/lib/Dialect/NCNN/IR/`（+ `compiler/include/ncnn-mlir/Dialect/NCNN/IR/`）
- 转换/规范化：`compiler/lib/{Conversion,Transforms}/`
- TOSA pipeline：`compiler/lib/Pipelines/NCNNPipelines.cpp`
- SqueezeNet pipeline 测试：`compiler/test/Pipelines/squeezenet-m2.mlir`
- 解析器/数据模型：`compiler/lib/Graph/`（`ncnn_graph` 库）
- 构建配置：各目录 `CMakeLists.txt`（顶层 `compiler/CMakeLists.txt`）
