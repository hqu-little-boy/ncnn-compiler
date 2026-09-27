# ncnn-compiler

**面向 ncnn 模型的 MLIR 编译器 / An MLIR compiler for ncnn models**

[简体中文](#简体中文) · [English](#english)

---

## 简体中文

### 项目简介

`ncnn-compiler` 将 ncnn 的网络结构文件（`.param`）和权重文件（`.bin`）编译为可由 C/C++ 程序调用的原生共享库。项目基于 LLVM/MLIR，定义了 ncnn 方言及一组逐级 lowering、验证和优化 pass。

产品编译主流程为：

```text
ncnn .param/.bin
    → ncnn 方言 MLIR
    → TOSA
    → Linalg
    → MemRef
    → LLVM IR
    → Linux ELF 共享库（.so）+ C 头文件
```

当前产品目标为 **64 位 Linux ELF**。并非所有 ncnn layer 或参数组合都已实现端到端 lowering；严格产品 pipeline 会在存在未支持的 ncnn 操作时报告失败，而不会把部分 lowering 当作成功编译。

### 工具

| 工具 | 用途 |
| --- | --- |
| `ncnn-compile` | 稳定的产品入口；运行固定编译 pipeline，生成 C ABI 头文件和共享库。 |
| `ncnn-mlir-driver` | 前端调试工具；读取 `.param/.bin`，输出 parsed graph 或 ncnn 方言 MLIR。 |
| `ncnn-mlir-opt` | MLIR 开发工具；解析、验证并运行已注册的项目及上游 MLIR pass/pipeline。 |

完整编译以 `ncnn-compile` 为准。单独运行前端或某个 MLIR pass 只检查相应阶段，不代表模型已能生成最终库。

### 依赖

- CMake 3.20 或更新版本
- 支持 C++23 的 C/C++ 编译器
- LLVM/MLIR 21 开发包及 CMake 配置文件
- Python 3、lit、GoogleTest
- 默认配置所需的 LLVM/MLIR 工具（包括 `mlir-translate-21`、`FileCheck-21` 等），以及 `clang-21`、`clang-format-21`、`clang-tidy-21` 和 `clang-apply-replacements-21`

CMake 默认在 `/usr/lib/llvm-21/lib/cmake/llvm` 和 `/usr/lib/llvm-21/lib/cmake/mlir` 查找 LLVM/MLIR。若安装位置不同，请在配置时设置 `LLVM_DIR` 和 `MLIR_DIR`。默认启用测试和格式/静态分析辅助目标，因此需要相应依赖。

### 构建

在本目录执行：

```bash
git submodule update --init --recursive
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

主要可执行文件位于 `build/tools/` 和 `build/bin/`。运行测试：

```bash
ctest --test-dir build --output-on-failure -j"$(nproc)"
```

### 快速开始

使用仓库中的 SqueezeNet 示例，将模型编译为 C ABI 头文件和共享库：

```bash
./build/tools/ncnn-compile \
  test/third_party/ncnn/examples/squeezenet_v1.1.param \
  --output-dir=build/squeezenet
```

若未指定 `--bin`，工具会从 `.param` 路径推导对应的 `.bin`。默认模型名来自 `.param` 文件名；输出目录中主要包含：

```text
build/squeezenet/
├── squeezenet_v1.1.h
└── libsqueezenet_v1.1.so
```

头文件提供模型专用的 C ABI 函数及输入/输出 tensor 信息。也可先单独检查导入后的 ncnn 方言 MLIR：

```bash
./build/tools/ncnn-mlir-driver \
  test/third_party/ncnn/examples/squeezenet_v1.1.param \
  -o build/squeezenet.ncnn.mlir
```

`ncnn-mlir-driver --help` 和 `ncnn-compile --help` 可查看当前版本支持的选项。编译器支持受限的动态 shape；这不等于支持任意动态计算图。INT8 模式也要求输入模型具有相应量化权重/边界，并不会自动把浮点模型量化。

### 代码结构

```text
include/   MLIR 方言、类型及 pass 的公共声明
lib/       图解析与导入、方言、转换 pass、pipeline 等实现
bin/       ncnn-mlir-opt
tools/     ncnn-compile、ncnn-mlir-driver 及辅助脚本
test/      单元、MLIR lit、数值和集成测试
docs/      命令行、IR 格式、开发及验证文档
```

### 文档

- [ncnn-compile 命令行参考](docs/ncnn-compile-command-line.md)
- [CLI 选项索引](docs/cli-options.md)
- [ncnn-mlir-driver 使用说明](docs/ncnn-mlir-driver-usage.md)
- [ncnn 方言 IR 格式](docs/ncnn-ir-format.md)
- [Parsed graph 格式](docs/parsed-graph-format.md)
- [算子数值验证指南](docs/operator-numerical-validation-guide.md)
- [已知/待调查问题](docs/ncnn-suspected-issues.md)

---

## English

### Overview

`ncnn-compiler` compiles an ncnn network definition (`.param`) and its weights (`.bin`) into a native shared library callable from C/C++ programs. Built on LLVM/MLIR, the project defines an ncnn dialect and a set of staged lowering, verification, and optimization passes.

The product compilation path is:

```text
ncnn .param/.bin
    → ncnn-dialect MLIR
    → TOSA
    → Linalg
    → MemRef
    → LLVM IR
    → Linux ELF shared library (.so) + C header
```

The current product target is **64-bit Linux ELF**. Not every ncnn layer or parameter configuration has an end-to-end lowering implementation. The strict product pipeline reports failure when unsupported ncnn operations remain instead of treating a partial lowering as a successful compile.

### Tools

| Tool | Purpose |
| --- | --- |
| `ncnn-compile` | Stable product entry point. Runs the fixed compilation pipeline and emits a C ABI header and shared library. |
| `ncnn-mlir-driver` | Front-end/debugging tool. Reads `.param/.bin` and emits either a parsed graph or ncnn-dialect MLIR. |
| `ncnn-mlir-opt` | MLIR development tool for parsing, verification, and running registered project and upstream MLIR passes/pipelines. |

Use `ncnn-compile` for a complete build. Running only the frontend or an individual MLIR pass validates that stage only; it does not mean the model can produce a final library.

### Requirements

- CMake 3.20 or newer
- A C/C++ compiler with C++23 support
- LLVM/MLIR 21 development packages and CMake config files
- Python 3, lit, and GoogleTest
- LLVM/MLIR tools required by the default configuration (including `mlir-translate-21` and `FileCheck-21`), plus `clang-21`, `clang-format-21`, `clang-tidy-21`, and `clang-apply-replacements-21`

By default, CMake looks for LLVM and MLIR under `/usr/lib/llvm-21/lib/cmake/llvm` and `/usr/lib/llvm-21/lib/cmake/mlir`. Set `LLVM_DIR` and `MLIR_DIR` when they are installed elsewhere. Tests and formatting/static-analysis helper targets are enabled by default, so their dependencies are required as well.

### Build

Run from this directory:

```bash
git submodule update --init --recursive
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The main executables are placed under `build/tools/` and `build/bin/`. Run the test suite with:

```bash
ctest --test-dir build --output-on-failure -j"$(nproc)"
```

### Quick start

Compile the bundled SqueezeNet example into a C ABI header and shared library:

```bash
./build/tools/ncnn-compile \
  test/third_party/ncnn/examples/squeezenet_v1.1.param \
  --output-dir=build/squeezenet
```

Unless `--bin` is provided, the compiler derives the `.bin` path from the `.param` path. The model name defaults to the `.param` filename; the output directory contains, among other optional artifacts:

```text
build/squeezenet/
├── squeezenet_v1.1.h
└── libsqueezenet_v1.1.so
```

The header exposes a model-specific C ABI function and input/output tensor information. To inspect the imported ncnn-dialect MLIR separately:

```bash
./build/tools/ncnn-mlir-driver \
  test/third_party/ncnn/examples/squeezenet_v1.1.param \
  -o build/squeezenet.ncnn.mlir
```

Run `ncnn-mlir-driver --help` or `ncnn-compile --help` for options available in the current build. Dynamic-shape support is constrained and does not imply support for arbitrary dynamic graphs. INT8 mode also requires appropriate quantized weights/boundaries in the input model; it does not automatically quantize a floating-point model.

### Source layout

```text
include/   Public declarations for MLIR dialects, types, and passes
lib/       Graph parsing/import, dialects, conversion passes, and pipelines
bin/       ncnn-mlir-opt
tools/     ncnn-compile, ncnn-mlir-driver, and helper scripts
test/      Unit, MLIR lit, numerical, and integration tests
docs/      CLI, IR format, development, and validation documentation
```

### Documentation

- [ncnn-compile command-line reference](docs/ncnn-compile-command-line.md)
- [CLI options index](docs/cli-options.md)
- [ncnn-mlir-driver usage](docs/ncnn-mlir-driver-usage.md)
- [ncnn dialect IR format](docs/ncnn-ir-format.md)
- [Parsed graph format](docs/parsed-graph-format.md)
- [Operator numerical validation guide](docs/operator-numerical-validation-guide.md)
- [Known and suspected issues](docs/ncnn-suspected-issues.md)
