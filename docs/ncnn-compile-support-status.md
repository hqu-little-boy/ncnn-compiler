# ncnn 编译支持现状

> 本文档记录编译器对 ncnn 模型的当前支持范围、编译流水线、目标平台及已知限制。
> 相关文档：[ncnn-compile-command-line.md](ncnn-compile-command-line.md)、
> [ncnn-ir-format.md](ncnn-ir-format.md)、
> [operator-numerical-validation-guide.md](operator-numerical-validation-guide.md)、
> [ncnn-suspected-issues.md](ncnn-suspected-issues.md)。

---

## 1. 项目概述

ncnn compiler 是一个基于 MLIR 的 ahead-of-time 编译器，将 ncnn 模型（`.param` + `.bin`）
编译为具有稳定 C ABI 的 Linux 共享库（`.so`）。技术栈为 LLVM/MLIR 21、C++23，当前以
**SqueezeNet v1.1** 的算子集作为端到端验证目标。

---

## 2. 支持的算子

### 2.1 模型边界 op

| Op | 用途 |
|---|---|
| `ncnn.model` | 顶层网络容器 |
| `ncnn.input` | ncnn `Input` 层，产生 `tensor<CxHxWxf32>` |
| `ncnn.const` | 来自 `.bin` 的权重常量 |
| `ncnn.output` | 标记导出输出 blob |

### 2.2 计算 op（完整支持、可端到端编译）

| ncnn 层类型 | 方言 op | 关键属性 | 备注 |
|---|---|---|---|
| `Convolution` | `ncnn.convolution` | kernel_h/w, stride_h/w, dilation_h/w, pad_top/bottom/left/right, has_bias, int8_scale_term | 仅支持 fp32 静态权重；交叉相关 |
| `ReLU` | `ncnn.relu` | negative_slope（0=ReLU，≠0=LeakyReLU） | |
| `Pooling` | `ncnn.pooling` | kind(0=max/1=avg), mode(0=regular/1=global/2=adaptive), kernel/stride/pad_*, pad_mode(0–3), include_pad | 见 §2.3 限制 |
| `Split` | `ncnn.split` | 无 | 纯路由，≥2 输出；SSA 化后被消除 |
| `Concat` | `ncnn.concat` | axis | 沿 axis 拼接 |
| `Dropout` | `ncnn.dropout` | scale（默认 1.0） | 推理时 scale=1.0 为恒等 |
| `Softmax` | `ncnn.softmax` | axis | |

导入器（`lib/Importer/NCNNImporter.cpp`）对上述以外的层类型返回 `unsupported layer type` 错误。

### 2.3 算子级限制

- **Convolution**：动态权重（param 19）、融合激活（param 9/10）、非零 `pad_value`（param 18）
  在导入时拒绝。`int8_scale_term` 可解析但 **int8 量化卷积不会被 lowering**——残留 ncnn op
  被严格流水线拒绝。
- **Pooling**：`mode=Adaptive` 和 `Average + include_pad=1` **不会被 lowering**（残留 → 拒绝）。
  `pad_mode=0`（full + tail padding）通过尾部填充计算正常支持。
- **Softmax**：旧版 `fixbug0=0` 仅允许 `axis=0`。

---

## 3. 编译流水线

```
.param/.bin
 └─ ncnn-mlir-driver ──────────────────▶ model.ncnn.mlir   (ncnn 方言)
     └─ ncnn-mlir-opt --ncnn-to-tosa-pipeline ─────▶ model.tosa.mlir
         └─ ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline ─▶ model.linalg.mlir
             └─ ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline ─▶ model.memref.mlir
                 └─ ncnn-mlir-opt --generate-ncnn-c-api ─▶ model.capi.mlir (+ JSON manifest)
                     └─ ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline ─▶ model.llvm.mlir
                         └─ mlir-translate --mlir-to-llvmir ─▶ model.ll
                             └─ clang -x ir -fPIC -O<n> -c ─▶ model.o
                                 └─ clang -shared -nostdlib ─▶ libmodel.so
                                     └─ 符号/依赖审计 + 头文件生成
                                         └─ 原子发布: model.h + libmodel.so
```

### 3.1 各阶段说明

| 阶段 | Pass / 工具 | 作用 |
|---|---|---|
| 解析 | `ncnn_graph::Graph::load` | 解析 `.param`（magic `7767517`）和 `.bin` 为 parsed-graph |
| 导入 | `ncnn_importer::import_graph` | 提升为类型化 MLIR SSA DAG，执行静态形状推断 |
| 模型→函数 | `convert-ncnn-model-to-func` | 消除模型边界，建立 `func.func` + `arith.constant` |
| 规范化 | `normalize-ncnn` | 解析 SAME padding 哨兵值、归一化负 axis、消除 Split |
| ncnn→TOSA | `convert-ncnn-to-tosa` | 逐 op 转换，处理 CHW→NHWC 布局转置 |
| TOSA→Linalg | 上游 `addTosaToLinalgPasses` | 标准 TOSA-to-Linalg + TosaToTensor + TosaToArith |
| Linalg→MemRef | `bufferize-ncnn` + `buffer-results-to-out-params` | One-Shot Bufferize，输出提升为 caller-owned 参数 |
| C API 生成 | `generate-ncnn-c-api` | 准备 bare-pointer ABI 元数据 |
| MemRef→LLVM | `convert-linalg-to-loops` → ... → `finalize-ncnn-c-api` | 完整下降到 LLVM 方言 |
| 代码生成 | `mlir-translate` + `clang` | LLVM IR → 目标文件 → 共享库 |

### 3.2 布局处理

ncnn 原生使用 **CHW** 布局（卷积权重 `[O,I,H,W]`）。C ABI 保持 CHW，但内部
`convert-ncnn-to-tosa` 会转置为 **NHWC/OHWI**（TOSA conv/pool 要求），计算完成后再转回。

---

## 4. IR 格式

| 阶段 | 文件格式 | 说明 |
|---|---|---|
| parsed-graph | 自定义文本 | 调试用，不被下游消费 |
| ncnn 方言 | MLIR 文本 | `ncnn.model` + 类型化 SSA 值 |
| TOSA | MLIR 文本 | 纯上游 TOSA 方言 |
| Linalg | MLIR 文本 | tensor + Linalg/Arith/Math |
| MemRef | MLIR 文本 | bufferized，caller-owned output，void return |
| C API | MLIR 文本 | 附加 bare-pointer ABI 元数据 |
| LLVM 方言 | MLIR 文本 | 纯 LLVM dialect |
| LLVM IR | `.ll` | 内部产物 |
| **最终产物** | `model.h` + `libmodel.so` | C 头文件 + Linux 共享库 |
| Manifest | `model.json` | JSON ABI 清单（`--emit-manifest` 时发布） |

---

## 5. 目标平台与 ABI

### 5.1 支持的目标

- **Linux x86-64 ELF**（主要，宿主测试）
- **Linux AArch64 ELF**（交叉编译，需配套 Clang 工具链 + sysroot）

不支持：Windows DLL、macOS dylib、32-bit ELF。

### 5.2 C ABI 契约

```c
int <model_name>(const float *input1, ..., float *output1, ...);
```

- 所有输入参数在前，输出参数在后。
- 输入为 `const float *`，输出为 `float *`。
- 调用方拥有所有 buffer， contiguous、native-endian、f32。
- 返回 0 表示成功；任一指针为 NULL 返回 1。
- 仅支持静态形状。

### 5.3 链接约束

- 使用 `-nostdlib`、`-Wl,-z,defs`、`--no-undefined`、版本脚本。
- 允许的未定义符号仅限：`expf`、`free`、`malloc`、`memcpy`、`memset`。
- 仅导出模型入口函数符号。
- 禁止出现：`memrefCopy`、`runner_utils`、`RunnerUtils`、`ncnn_runtime`。

---

## 6. 优化支持

| 层级 | 内容 |
|---|---|
| MLIR 级（始终执行） | canonicalize、CSE、One-Shot Bufferize、buffer-results-to-out-params、deallocation、linalg-to-loops、math-to-libm |
| 代码生成级 | Clang `-O0`/`-O1`/`-O2`/`-O3`（默认 `-O3`） |
| 目标调优 | `--target-triple`、`--march`（含 `native`）、`--mcpu`、`--mtune`、`--target-feature`、`--sysroot` |
| 图级优化 | 无（无算子融合、无常量折叠、无量化优化） |

隐式优化：Dropout scale=1.0 折叠为恒等；Split 通过 SSA 消除。

---

## 7. 测试覆盖

### 7.1 lit / FileCheck 测试

- `test/Dialect/NCNN/`：op 解析/打印/验证
- `test/Conversion/NCNNToFunc/`、`test/Conversion/NCNNToTosa/`：转换正确性
- `test/Transforms/`：NormalizeNCNN、GenerateCAPI、各 verify gate
- `test/Pipelines/`：完整流水线组合、SqueezeNet 端到端、严格失败路径

### 7.2 GTest 单元测试

- `test/Unit/parser_test.cpp`、`malformed_graph_test.cpp`：解析器健壮性
- `test/Unit/load_squeezenet*.cpp`：集成（层数、拓扑、权重形状/类型）
- `test/Unit/ncnn_importer_test.cpp`：导入器正常/异常覆盖

### 7.3 数值黄金测试

`test/Numerical/`：将编译产物 `.so` 的输出与上游 ncnn **naive 标量 CPU 参考**对比。

- `operators/supported_ops_test.cpp`：覆盖全部 7 个计算 op 的参数矩阵：
  - Convolution：basic、no-bias、dilated、asymmetric padding、SAME_UPPER/LOWER
  - ReLU：standard、leaky、zero/negative inputs
  - Pooling：max、average(exclude-pad)、global max/avg、SAME_UPPER/LOWER、tail window
  - Softmax：channel/height/width axis + negative-axis
  - Concat：channel/height/width + negative-axis
  - Split：2-way、3-way consumer topology
  - Dropout：identity
- `models/squeezenet_test.cpp`：完整 SqueezeNet v1.1 端到端
  - 全有限输出、softmax 求和误差 ≤1e-5、top-1 匹配、top-5 集合匹配、最大绝对误差 ≤1e-4

### 7.4 运行时测试

- `squeezenet-shared-library`：完整编译 + `--verify-execution`
- `ncnn-compile-cli`：CLI 契约测试

### 7.5 未覆盖（明确排除）

- `ConvolutionDepthWise` / 分组卷积（不导入）
- Average pooling `include_pad=1`（严格流水线拒绝）
- Adaptive pooling（严格流水线拒绝）

---

## 8. 已知限制总结

| 类别 | 限制 |
|---|---|
| 算子覆盖 | 仅 SqueezeNet 级别（7 个计算 op）；无 depthwise/group conv、InnerProduct、BinaryOp、Interp、Padding、BN、RNN |
| 量化 | int8 参数可解析但不被 lowering；f16 权重可解析但端到端路径仅 f32 |
| 形状 | 仅静态形状 |
| 数据类型 | ABI 仅 f32 |
| 入口 | 单入口点，单组输入/输出 |
| 平台 | 仅 Linux 64-bit ELF |
| GPU/NPU | 无，仅 CPU（LLVM 后端） |
| 图优化 | 无算子融合、无常量折叠 |

---

## 9. 关键源文件索引

| 关注点 | 文件 |
|---|---|
| Op 定义（TableGen） | `include/ncnn-mlir/Dialect/NCNN/IR/NCNNOps.td` |
| Op 形状推断/验证 | `lib/Dialect/NCNN/IR/NCNNOps.cpp` |
| 图解析/数据模型 | `lib/Graph/graph.cpp`、`lib/Graph/parser.cpp` |
| 导入器（层接受） | `lib/Importer/NCNNImporter.cpp` |
| 模型→函数转换 | `lib/Conversion/NCNNToFunc/NCNNToFunc.cpp` |
| 规范化 | `lib/Transforms/NormalizeNCNN/NormalizeNCNN.cpp` |
| ncnn→TOSA 转换 | `lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` |
| C API 生成 | `lib/Transforms/GenerateCAPI/GenerateCAPI.cpp` |
| 流水线组合 | `lib/Pipelines/NCNNPipelines.cpp` |
| 前端驱动 | `tools/ncnn-mlir-driver.cpp` |
| 编译驱动 | `tools/ncnn-compile.cpp` |
| mlir-opt 克隆 | `bin/ncnn-mlir-opt.cpp` |
