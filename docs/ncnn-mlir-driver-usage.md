# ncnn-mlir-driver 使用文档

> 编译器端到端驱动的命令行入口。当前处于 M1 阶段：可把 ncnn 模型
> (`.param` + `.bin`) 解析并转成类型化前端 IR，尚未接入 MLIR pass 下降管线。
> 参数解析用 LLVM 自带的 `llvm::cl`（`mlir-opt` 同款 CommandLine 库），
> 后续接入 tosa/linalg/llvm 下降阶段时复用同一套 option 基础设施。

> 注意：项目实际根目录是 `/mnt/ncnn-compiler`（不是 `/home/zeng/debian/...`）。
> 本文中的相对路径均以 `compiler/` 为基准。

---

## 1. 构建

本机没有 Ninja；仓库里 `cmake-build-debug/` 是 CLion 用 Ninja 配的，命令行请另建
一个用 Unix Makefiles 的构建目录：

```bash
cd /mnt/ncnn-compiler/compiler
cmake -S . -B build-make -G "Unix Makefiles"
cmake --build build-make --target ncnn-mlir-driver -j
```

产物：`build-make/ncnn-mlir-driver`。

依赖：LLVM/MLIR 21（Debian 包 `llvm-21-dev` + `libmlir-21-dev`）。CMake 通过
`find_package(LLVM CONFIG)` 定位（config 目录 `/usr/lib/llvm-21/lib/cmake/llvm`），
只链接 LLVM `support` 组件（含 `CommandLine` 与 `raw_ostream`）。

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
| `--emit=<stage>` | 选择要发出的阶段（见下） | `ncnn-ir` |
| `--verify` | 对类型化 IR 跑校验器 | 开 |

`--emit` 的取值：

- `parsed-graph`：原始解析出的 ncnn 计算图（layer/blob/参数 + 绑定的权重形状），
  最贴近 `.param`/`.bin` 原貌，适合排查解析问题。
- `ncnn-ir`（默认）：类型化前端 IR（typed DAG），已推导张量类型与 use-def 关系，
  是后续下降到 tosa/linalg 的起点。

> `--emit` 是驱动最关键的设计点：后续 `tosa`、`linalg`、`llvm`、`library`
> 等下降阶段会加入到这个同一个枚举里，CLI 表面保持稳定。

查看帮助（`--help` 已用 `OptionCategory` 收窄，不会被链接 libLLVM 引入的海量
codegen option 淹没）：

```bash
./build-make/ncnn-mlir-driver --help
```

```
OVERVIEW: ncnn-mlir-driver -- compile ncnn .param/.bin toward MLIR/native code

USAGE: ncnn-mlir-driver [options] <input .param file>

OPTIONS:

Generic Options:
  --help          - Display available options (--help-hidden for more)
  --help-list     - Display list of available options (--help-list-hidden for more)
  --version       - Display the version of this program

ncnn-mlir-driver options:
  --bin=<path>    - Weight file (.bin). Defaults to <input> with .param replaced by .bin
  --emit=<value>  - Select the stage to emit:
    =parsed-graph -   Raw parsed ncnn graph (param + bound weights)
    =ncnn-ir      -   Typed ncnn frontend IR (default)
  -o <path>       - Output file. '-' writes to stdout (default)
  --verify        - Run the IR verifier on the typed ncnn IR (default: on)
```

---

## 3. 执行示例

以自带的 SqueezeNet v1.1 为例（`ncnn/examples/squeezenet_v1.1.{param,bin}`）。

### 3.1 默认：产类型化 IR（自动推导 .bin）

```bash
cd /mnt/ncnn-compiler/compiler
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param
```

`.bin` 自动由 `.param` 推导（`squeezenet_v1.1.param` → `squeezenet_v1.1.bin`），
输出前几行：

```
ncnn_frontend.typed_dag_dump version=1
operations 126
op 0 {kind=const,attrs={literal_type={shape=[64,3,3,3],element=f32,layout=oihw,elements=1728,bytes=6912},payload_bytes=6912,fnv1a64=0x3c505b732fd566d8},name="conv1.weight",source_layer=1,operands=[],results=[v1]}
op 1 {kind=const,attrs={literal_type={shape=[64],element=f32,layout=ncnn_w,elements=64,bytes=256},payload_bytes=256,fnv1a64=0xd30558885e110c61},name="conv1.bias",source_layer=1,operands=[],results=[v2]}
op 2 {kind=conv2d,attrs={kernel=[3,3],stride=[2,2],dilation=[1,1],pad=[0,0,0,0],has_bias=true,int8_scale_term=0,quantization=none},name="conv1",source_layer=1,operands=[v0,v1,v2],results=[v3]}
op 3 {kind=relu,attrs={negative_slope=0},name="relu_conv1",source_layer=2,operands=[v3],results=[v4]}
...
```

权重被抬成 `const` op（`conv1.weight` 布局 `oihw`、`conv1.bias` 布局 `ncnn_w`），
`conv2d` 通过 `operands=[v0,v1,v2]` 引用输入激活 + 权重 + bias。

### 3.2 显式指定 .bin

```bash
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param \
  --bin=../ncnn/examples/squeezenet_v1.1.bin
```

### 3.3 查看原始解析图

```bash
./build-make/ncnn-mlir-driver --emit=parsed-graph ../ncnn/examples/squeezenet_v1.1.param
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
  [  3] Pooling        pool1                  in=[conv1_relu_conv1] out=[pool1] {0=0 1=3 2=2 3=0 4=0}
  ...
  [ 73] Pooling        pool10                 in=[conv10_relu_conv10] out=[pool10] {0=1 1=0 2=1 3=0 4=1}
  [ 74] Softmax        prob                   in=[pool10] out=[prob] {0=0}
```

参数按 ncnn 原生 key（`{0=227 1=227 2=3}` 即 Input 的 `w=227 h=227 c=3`），
`w=[...]` 是绑定的权重张量形状（`[64,3,3,3:f32:6912B]` = 卷积核 + 字节数）。

### 3.4 写到文件

```bash
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param -o /tmp/squeezenet.ncnn-ir
```

诊断信息走 stderr，产物走 stdout/`-o`，因此可以直接管道：

```bash
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param 2>/dev/null | head
```

### 3.5 关闭校验

```bash
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param --verify=false
```

---

## 4. 退出码与错误

- 成功：退出码 `0`，产物写到 stdout 或 `-o`。
- 失败：退出码 `1`，错误信息写到 stderr。

文件不存在示例：

```bash
$ ./build-make/ncnn-mlir-driver /tmp/nope.param
error: failed to load model: cannot open param file: /tmp/nope.param
$ echo $?
1
```

其他失败点及对应信息：

| 阶段 | stderr 前缀 |
|------|-------------|
| 加载 `.param`/`.bin` | `error: failed to load model: ...` |
| 转类型化 IR | `error: failed to import graph: ...` |
| IR 校验（`--verify` 开时） | `error: IR verification failed: ...` |
| 打开/写出 `-o` 文件 | `error: cannot open output file ...` / `error: failed while writing ...` |

---

## 5. 内部流水线

```
.param/.bin
  └─ ncnn_graph::Graph::load        →  原始计算图 (--emit=parsed-graph 到此为止)
       └─ ncnn_frontend::import_graph → 类型化前端 IR
            └─ ncnn_frontend::verify_graph  (--verify)
                 └─ dump              →  文本产物 (--emit=ncnn-ir，默认)
```

后续阶段（tosa → linalg → llvm → 原生库）会作为新的 `--emit` 取值接入，
届时会链接对应的 MLIR target。当前版本仅链接 LLVM `support`。

---

## 6. 源码位置

- 驱动入口：`compiler/tools/ncnn-mlir-driver.cpp`
- 构建配置：`compiler/CMakeLists.txt`（`ncnn-mlir-driver` target）
- 依赖库：`ncnn_graph`（解析 + 数据模型）、`ncnn_frontend`（类型化 IR + 校验）
