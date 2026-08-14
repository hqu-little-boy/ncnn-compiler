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
**SqueezeNet v1.1** 以及静态 FP32 的 `PP-LCNet_x1_0_doc_ori`、
`PP-LCNet_x1_0_textline_ori`、`Chineseocr_Lite_AngleNet`、
`PP-OCRv6_tiny_rec`、`PP-OCRv6_tiny_det`、`PP-OCRv6_small_det`、
`PP-OCRv6_medium_det`、`PP-OCRv5_mobile_det`、`PP-OCRv5_server_det`、
`PP-StructrureV2_SLANet_plus_cnn`、`PP-FormulaNet_plus_S_encoder` 作为端到端验证目标。

---

## 2. 支持的算子

### 2.1 模型边界 op

| Op | 用途 |
|---|---|
| `ncnn.model` | 顶层网络容器 |
| `ncnn.input` | ncnn `Input` 层，产生 `tensor<CxHxWxf32>` |
| `ncnn.const` | 来自 `.bin` 的权重常量 |
| `ncnn.output` | 标记导出输出 blob |

### 2.2 计算 op（31 个 source 计算层，32 个方言计算 op）

Importer 当前注册 32 个 source layer type，包含 `Input` 和下表 31 个计算层。方言另有一个
lowering 内部使用的 `ncnn.zero_point_cast`，因此共有 32 个计算 op。表中“支持”只表示对应受限实例可导入并通过产品
strict pipeline，不表示该 ncnn 层的所有参数组合都接受。方言权威定义是
[`NCNNOps.td`](../include/ncnn-mlir/Dialect/NCNN/IR/NCNNOps.td)，导入权威是
`lib/Importer/NCNNImporter.cpp` 的 `importers()`；最终能力还须满足
`lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` 的 conversion pattern 和严格残留 op gate。

| ncnn 层类型 | 方言 op | 关键属性 | 备注 |
|---|---|---|---|
| `Convolution` | `ncnn.convolution` | kernel_h/w, stride_h/w, dilation_h/w, pad_top/bottom/left/right, has_bias, int8_scale_term | 支持静态 FP32/FP16-storage 权重、INT8 scale-term 和 FP16/BF16 混合边界；融合 ReLU/Sigmoid 拆为显式 op；交叉相关 |
| `ReLU` | `ncnn.relu` | negative_slope（0=ReLU，≠0=LeakyReLU） | |
| `Pooling` | `ncnn.pooling` | kind(0=max/1=avg), mode(0=regular/1=global/2=adaptive), kernel/stride/pad_*, pad_mode(0–3), include_pad | 见 §2.3 限制 |
| `Split` | `ncnn.split` | 无 | 纯路由，≥2 输出；SSA 化后被消除 |
| `Concat` | `ncnn.concat` | axis | 沿 axis 拼接 |
| `Dropout` | `ncnn.dropout` | scale（默认 1.0） | 推理时 scale=1.0 为恒等 |
| `Softmax` | `ncnn.softmax` | axis | 支持 fixed-rank 动态 extent；FP16/BF16 使用低精度边界，复杂计算保持 f32 |
| `ConvolutionDepthWise` | `ncnn.convolution_depthwise` | kernel_h/w, stride_h/w, dilation_h/w, pad_top/bottom/left/right, has_bias | 支持纯 depthwise、静态 FP32/FP16-storage 权重、INT8 scale-term 和融合 ReLU；不是通用 group conv |
| `Deconvolution` | `ncnn.deconvolution` | kernel_h/w, stride_h/w, dilation_h/w, pad_top/bottom/left/right, output_pad_bottom/right, has_bias | FP32 2x2 stride-2 子集；权重和参数静态，输入 H/W 可动态；可选 bias 和融合 ReLU |
| `Padding` | `ncnn.padding` | top, bottom, left, right, value | FP32 constant spatial padding；支持动态 H/W |
| `Interp` | `ncnn.interp` | height_scale, width_scale, output_h/w | FP32 rank-3 nearest；支持正整数倍或显式静态目标，scale 模式支持动态 H/W |
| `Sigmoid` | `ncnn.sigmoid` | 无 | 支持 fixed-rank 动态 extent；FP16/BF16 使用低精度边界，复杂计算保持 f32 |
| `HardSigmoid` | `ncnn.hard_sigmoid` | alpha, beta | 支持 FP16/BF16 mixed boundary |
| `HardSwish` | `ncnn.hard_swish` | alpha, beta | 支持 FP16/BF16 mixed boundary |
| `Reshape` | `ncnn.reshape` | shape、shape_spec、shape_zero_sources、shape_sources、shape_expression | 支持静态 shape、单个 `-1`、`0` 复制维度，并保留原始 shape 语义；也支持引用单一 shape 输入并保持维序的表达式（如 `1w,1h,1c`） |
| `BinaryOp` | `ncnn.binary` | op_type, with_scalar, scalar | 加法/乘法/最大值；支持标量和同 rank 双向广播，动态 extent 保守推断 |
| `InnerProduct` | `ncnn.inner_product` | has_bias, int8_scale_term | 支持 FP32 和受限 INT8 scale-term；静态输入按元素展平，另支持 rank-2 动态 M、静态 K 的 `[M,K] -> [M,O]` |
| `ShuffleChannel` | `ncnn.shuffle_channel` | group, reverse | 静态 FP32；group 必须整除通道数 |
| `Slice` | `ncnn.slice` | slices, axis | FP32；支持显式 sizes 和 `-233` 按序切分；动态切分轴上的显式 size 保持精确，`-233` 结果为动态 |
| `Reduction` | `ncnn.reduction` | kind, reduce_all, coeff, axes, keepdims | 静态 FP32 mean 子集 |
| `GELU` | `ncnn.gelu` | fast | 当前支持标准 erfc 形式（`fast=0`） |
| `Squeeze` | `ncnn.squeeze` | axes | 显式静态 axes |
| `BatchNorm` | `ncnn.batch_norm` | epsilon | 静态 FP32，按首维归一化 |
| `ExpandDims` | `ncnn.expand_dims` | axes | 显式静态 axes |
| `Permute` | `ncnn.permute` | permutation | 当前 importer 支持 rank-2 |
| `Gemm` | `ncnn.gemm` | alpha, beta, int8_scale_term | 动态 A、转置常量 B、行偏置；支持 FP32 和受限 INT8 term 1/2 |
| `DetectionOutput` | `ncnn.detection_output` | num_class, nms_threshold, nms_top_k, keep_top_k, confidence_threshold, variance_x/y/w/h | Caffe SSD 三输入子集；bounded 数据依赖输出，支持 FP16/BF16 mixed boundary |
| `Quantize` | `ncnn.quantize` | scale | 受限 FP32 到 INT8 量化边界 |
| `Dequantize` | `ncnn.dequantize` | scale, bias | 受限 INT32 到 FP32 反量化边界 |
| `Requantize` | `ncnn.requantize` | scale_in, scale_out, bias, activation | 受限 INT32 到 INT8 重定标边界 |
| `Cast` | `ncnn.cast` | type_from, type_to | 支持已注册的低精度边界类型转换 |

导入器（`lib/Importer/NCNNImporter.cpp`）对上述以外的层类型返回 `unsupported layer type` 错误。

### 2.3 算子级限制

- **Convolution**：支持 FP16/BF16 storage boundary、FP16 原生算术或 FP32 accumulator fallback，以及
  `int8_scale_term` 的 INT8→INT32→FP32 重定标路径。动态权重（param 19）、除 ReLU/Sigmoid 外的融合激活、
  非零 `pad_value`（param 18）在导入时拒绝。
- **ConvolutionDepthWise**：只接受纯 depthwise；支持 FP16/BF16 storage boundary、INT8 重定标、无激活或融合
  ReLU。通用 group convolution、动态权重和其他融合激活仍然拒绝；显式 padding 和 SAME_UPPER/LOWER 均支持。
- **Deconvolution**：当前只支持 FP32、静态 2x2 kernel、stride 2、dilation 1、零 crop/output
  padding、无显式输出 shape override；融合激活仅支持无激活或 ReLU。直接 NCNN IR lowering
  会拒绝超出 TOSA `transpose_conv2d` `out_pad` 表示范围的 crop。无激活/ReLU 携带的
  activation params 按 ncnn 语义忽略，`weight_data_size` 必须与实际权重元素数一致。输入空间
  extent 可以动态，输出 H/W 由运行时 shape program 推导。
- **Padding**：仅支持 spatial constant padding，不支持 per-channel padding data 或其他 padding mode。
- **Interp**：仅支持 `resize_type=1` 的 nearest 模式和 `align_corner=0`。支持正整数倍 scale，
  以及显式静态 `output_h/output_w`；后者按 ncnn 的 `floor(dst * input / output)` 采样语义直接
  lower 为 Linalg。整数 scale 的静态实例可走 TOSA resize，动态 H/W 直接 lower 为
  `tensor.dim + tensor.empty + linalg.generic`。
- **Pooling**：NCNN IR 类型推断允许安全的动态通道，并在 regular/adaptive 结果中保留通道、global
  结果为 `[C]`。动态 H/W 的 global 和固定 target adaptive 直接 lower 为运行时 Linalg/SCF 归约；
  adaptive 窗口按 `floor(input * output_index / output)` 与
  `ceil(input * (output_index + 1) / output)` 计算。regular `Average + include_pad=1` 仍明确拒绝。
  `pad_mode=0`（full + tail padding）通过尾部填充计算正常支持；global 模式按 ncnn 语义忽略
  regular-only 的 padding 和 `include_pad` 参数。
- **Softmax**：旧版 `fixbug0=0` 仅允许 `axis=0`。
- **Reshape**：非 expression 形式通过 `shape_spec` 保留原始 `0/-1`，并以
  `shape_zero_sources` 明确每个 `0` 映射的输入维；只允许单个 `-1`，缺失维的 `0` 明确拒绝。
  shape expression 当前支持直接引用单一输入 blob 的 `w/h/d/c` 维度，并要求输出
  保持该输入的维序；算术、维度置换、跨输入混合、min/max 等通用表达式仍会明确拒绝。动态 expression 直接下降为
  `tensor.dim + tensor.from_elements + tensor.reshape`。M09 的固定 `3x640x640` 和受 minimum/
  multiple-of 约束的 `3x?x?` 产品实例均已完成跨 shape 数值闭环；这不表示任意 expression
  语法、任意模型或无约束动态尺寸均受支持。
- **Slice**：当前支持 `slices` 参数，不支持 `indices` 参数形式。切分轴动态时要求最后一片为
  `-233`，显式片大小仍静态精确，按序遇到的 `-233` 片保守推断为动态；静态切分轴仍严格校验
  所有片完整消费该轴。
- **BinaryOp**：Importer/NCNN IR 支持 `op_type=0/2/4`（add/multiply/max）的标量或同 rank
  广播；动态维仅在一侧静态为 `1` 或两侧符号 shape 可证明等价时放行，不可由当前 ABI 验证的
  运行时析取广播条件继续拒绝。不支持的 op_type 和 rank 组合继续拒绝。
- **InnerProduct**：权重及 K/O 必须静态；rank-2 输入允许动态 M，并推断 `[M,O]`。动态 K、
  动态权重和其他非静态输入形态继续拒绝。
- **Reduction**：当前只支持 `operation=3`（mean）；显式 axes 要求新版 `fixbug0=1`。
- **Gemm**：仅支持 `constantA=0, constantB=1, constantC=1, transA=0, transB=1`、
  `broadcast_type_C=4` 的 FP32 或 `int8_scale_term=1/2` 子集；不支持 packing 和输出转置。
- **DetectionOutput**：仅支持 Caffe-style SSD，要求三个静态 FP32 输入且满足 location、confidence、
  priorbox 的元素数关系；拒绝 MXNet `num_class=-233`。每行是
  `[label, confidence, xmin, ymin, xmax, ymax]`，最大 storage 为
  `[maximum_detections, 6]`，实际 shape 为 `[count, 6]`：

  ```text
  maximum_detections =
    min(keep_top_k, (num_class - 1) * min(nms_top_k, num_prior))
  ```

  `PriorBox`、`Proposal`、`YoloDetectionOutput` 和 `Yolov3DetectionOutput` 尚未支持。
  当前 Caffe lowering 从 priorbox 输入第二行开头读取 4 个 variance，并对所有 prior 复用；
  op 上保留的 `variance_x/y/w/h` 属性仅参与合法性校验，不覆盖该输入数据。

### 2.4 算子扩展一致性契约

新增一个 source layer 时，算子能力分散在多个阶段；以下注册表/描述符必须同步，
否则会出现“某阶段已支持、另一阶段漏实现”的问题：

| 阶段 | 位置 | 责任 |
|---|---|---|
| 层类型 → importer handler | `lib/Importer/NCNNImporter.cpp` 的 `importers()` | 层类型到 `ImportContext` handler 的分派 |
| 算子族实现 | `lib/Importer/Import{Input,Convolution,Pooling,Activation,Tensor,Ops,Detection}.cpp` | 参数校验、SSA 构造、blob 绑定 |
| 方言 schema/实现 | `include/.../NCNNOps.td`、`lib/Dialect/.../NCNNOps.cpp` | op、属性、verifier 和 shape inference |
| 权重字节消费 | `lib/Graph/graph.cpp` 的 `weight_loaders()` | 带权重 layer 必须注册 loader 消费 `.bin` |
| 共享参数解码 | `lib/Graph/graph.cpp` 的 `decode_convolution_params()` | 参数 ID/默认值/合法性的单一来源 |
| 能力声明 | `include/ncnn-mlir/Support/LayerCapabilities.hpp` | `HasWeights`/`NeedsNormalization`/`Lowerable` |
| 规范化 | `lib/Transforms/NormalizeNCNN/NormalizeNCNN.cpp` | 需要归一化属性的 op 增加 typed pattern |
| Lowering | `lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` | 必须提供 conversion pattern，否则严格流水线拒绝残留 |
| 构建与声明 | 对应 `CMakeLists.txt`、Importer internal 声明 | 新源码必须进入 target，handler 声明与定义一致 |
| 合法性与 gate | conversion target、`VerifyNoNCNNOps`/`VerifyNoTosaOps` | op 必须被正确标为 illegal/dynamically legal 并最终消除 |
| 测试 | Dialect/Conversion/Unit/Numerical/Native | 覆盖接受、拒绝、shape、数值、ABI 和产物审计 |

`test/Unit/layer_registry_test.cpp` 校验 descriptor 与真实 importer/weight-loader
注册表的一致性；新增算子时应保持该测试通过。

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
| 导入 | `ncnn_importer::import_graph` | 提升为类型化 MLIR SSA DAG，通过 op 的 `InferTypeOpInterface` 执行 ranked shape inference |
| 模型→函数 | `convert-ncnn-model-to-func` | Full dialect conversion：移动模型 region，建立 `func.func` + `arith.constant`，不 clone SSA 图 |
| 规范化 | `normalize-ncnn` | 两阶段校验/提交：SAME padding 校验由 `TypeSwitch` 分派，提交由 typed `OpRewritePattern` 完成；Split 由标准 folder/canonicalize 消除 |
| ncnn→目标 IR | `convert-ncnn-to-tosa` | 大多数算子生成 TOSA；动态 Interp 和 DetectionOutput 等白名单实例直接生成 Linalg/SCF/Tensor/Arith |
| TOSA→Linalg | 上游 `addTosaToLinalgPasses` + `verify-no-tosa-ops` | 标准 TOSA-to-Linalg + TosaToTensor + TosaToArith，并拒绝残留 TOSA |
| Linalg→MemRef | `bufferize-ncnn` + `buffer-results-to-out-params` + 两个 verify gate | One-Shot Bufferize，输出提升为 caller-owned 参数，并验证 buffer ownership 与 shape contract |
| C API 生成 | `generate-ncnn-c-api` | 准备 bare-pointer ABI 元数据，通过 MLIR `SymbolTable` 重命名内部函数并更新全部符号引用 |
| MemRef→LLVM | OpenMP 或 Affine vector/serial loops → ... → `finalize-ncnn-c-api` | 按 threads/vector 选项完整下降到 LLVM 方言 |
| 代码生成 | `mlir-translate` + `clang` | LLVM IR → 目标文件 → 共享库 |

项目 pass 的参数、说明、dependent dialect、factory 和统一注册由
`include/ncnn-mlir/Passes.td` 生成；各实现只继承生成的 pass base 并提供核心逻辑。

### 3.2 布局处理

ncnn 原生使用 **CHW** 布局（卷积权重 `[O,I,H,W]`）。C ABI 保持 CHW，但内部
`convert-ncnn-to-tosa` 的 `TypeConverter` 会把 rank-3 CHW 映射为 rank-4 **NHWC**，权重
由 convolution pattern 转为 **OHWI**。source/target materialization 在函数、合法非 ncnn op
或残留 ncnn op 的边界按需转回 CHW，不改变函数签名。

该转换采用部分转换契约：可 lowering 的 ncnn op 必须由 pattern 消除；regular average
include-pad 和尚未分配目标路径的 ncnn op 可以残留。单 pass 成功
不表示已经得到纯 TOSA，严格 pipeline 最后通过 `verify-no-ncnn-ops` 拒绝任何残留。转换失败时
由 MLIR conversion driver 处理 rewrite 失败，不通过 clone 整个 module 实现事务性。

---

## 4. IR 格式

| 阶段 | 文件格式 | 说明 |
|---|---|---|
| parsed-graph | 自定义文本 | 调试用，不被下游消费 |
| ncnn 方言 | MLIR 文本 | `ncnn.model` + 类型化 SSA 值 |
| TOSA | MLIR 文本 | 上游 TOSA 为主；动态 Interp、DetectionOutput、标准 GELU 可同时含 Linalg/SCF/Tensor/Arith/Math |
| Linalg | MLIR 文本 | tensor + Linalg/Arith/Math |
| MemRef | MLIR 文本 | bufferized，caller-owned output，void return |
| C API | MLIR 文本 | 附加 bare-pointer ABI 元数据 |
| LLVM 方言 | MLIR 文本 | 纯 LLVM dialect |
| LLVM IR | `.ll` | 内部产物 |
| **最终产物** | `model.h` + `libmodel.so` | C 头文件 + Linux 共享库 |
| Manifest | `model.json` | JSON ABI 清单（`--emit-manifest` 时发布） |

---

## 5. 目标平台与 ABI

### 5.1 CLI 接受与已验证目标

- CLI 接受 64 位 Linux ELF triple。当前已配置的 `build` 树在宿主 Linux x86-64 注册 215 项 CTest，
  数值模型和动态库执行验证。
- AArch64 FP16 和 RISC-V Zfh/Zvfh FP16 已使用编译器生成的 LLVM IR 做静态 assembly 指令验证；
  这不等同于在目标硬件上运行。交叉编译仍需匹配的 Clang 工具链和 sysroot，`--verify-execution` 不能
  用于当前宿主无法执行的目标产物。

不支持：Windows DLL、macOS dylib、32-bit ELF。

### 5.2 C ABI 契约

```c
int <model_name>(const <input_type> *input1, ..., <output_type> *output1, ...);
```

- 所有输入 tensor 参数组在前，输出数据指针在后；输入参数组为 `data`、可选 `shape`、可选 `rank`。
- 元素类型由 C 指针类型表达。ABI 层支持 f16、bf16、f32、f64 以及 8/16/32/64 位有符号、
  无符号整数；当前 ncnn 产品模型路径仍只产生 f32 tensor。
- 完全静态 tensor 省略 shape/rank；固定-rank 动态输入增加 `const int64_t shape[RANK]`；动态
  rank 输入增加 shape 和 rank。
- 调用方拥有所有 buffer；数据为 contiguous、native-endian。动态输入 shape 的静态维必须匹配
  头文件宏，动态维必须大于 0。
- 生成头文件定义六种状态：`0` success、`1` null pointer、`2` invalid shape、`3` constraint
  violation、`4` shape arithmetic overflow、`5` output capacity insufficient。
- `--input-shape=CxHxW` 可重复，按 Input source-layer 顺序绑定；extent 可为正整数或 `?`。
- 未提供 override 时，尺寸完全省略的 Input 自动成为 `[C,?,?]`；直接连接到 Convolution 时可从
  `[O,I,H,W]` 权重推导通道。显式 override 必须覆盖全部无尺寸 Input，或兼容性地覆盖全部 Input。
- `--input-dim-constraint=INPUT:DIM:min=N,multiple=N` 可重复，为 fixed-rank 动态输入维添加
  结构化 minimum/multiple 约束；执行和 shape inference 入口都会校验，manifest/header 同步公开。
- 动态 Slice 的 minimum 可从切片 sizes 自动推导并合并到同一约束集合。
- 阶段二已支持 regular Pooling、纯 Depthwise Convolution、Deconvolution、Sigmoid 的动态
  H/W shape inference 和 NCNN-to-TOSA lowering；动态 `SAME` 仅支持对应 stride 为 1。
- Concat 非拼接动态维与 Binary 非静态-1 广播维必须由输入约束下的符号 shape program 证明
  等价，否则编译失败；Concat 拼接轴使用 V2 加法表达式，Binary 输出广播维可使用 V2 Max。
  完整动态 M09 已通过约束 shape、容量、溢出、重复调用和交替 shape 测试；其他模型仍须逐算子
  验证下游动态 lowering 能力。
- shape-only 动态输出由 `<model>_infer_output_shapes` 返回，调用方据此分配 output buffer；执行
  入口为每个此类输出接收 `uint64_t capacity`，单位是 data buffer 元素数。
- 数据依赖输出由执行入口返回 actual shape/rank，并接收 shape metadata capacity；调用方按
  `MAX_DIMn`/`MAX_ELEMENTS` 分配最大 data buffer。shape capacity 是元数据数组容量，不是 data
  buffer 容量。
- `--input-shape=*` 支持 rank 1 至 4 的受限 specialization，当前仅单输入、单输出、
  shape-preserving identity/ReLU；不表示一般多输入/多输出动态 rank 已实现。完整契约见
  [`dynamic-rank-c-abi.md`](../../docs/dynamic-rank-c-abi.md)。

### 5.3 链接约束

- 始终使用 `-nostdlib` 和版本脚本。普通产物还使用 `-Wl,-z,defs`、`--no-undefined`；显式
  sanitizer 链接参数存在时跳过这两个选项，改由后续未定义符号和 `DT_NEEDED` 审计约束产物。
- 允许的 libc/libm 未定义符号包括：`ceilf`、`floorf`、`erfcf`、`erff`、`expf`、`powf`、
  `free`、`malloc`、`memcpy`、`memset`。默认 OpenMP 产物还允许审计过的 `__kmpc_*` 并依赖
  `libomp`；`--threads=1` 不引入 OpenMP 依赖。目标链接还显式加入匹配的 compiler-rt builtins。
- 仅导出模型执行入口，以及存在 shape-only 动态输出时的 `<model>_infer_output_shapes`。
- 禁止出现：`memrefCopy`、`runner_utils`、`RunnerUtils`、`ncnn_runtime`。

---

## 6. 优化支持

| 层级 | 内容 |
|---|---|
| MLIR 级（始终执行） | canonicalize、CSE、LICM、One-Shot Bufferize、buffer-results-to-out-params、deallocation、linalg-to-loops、math-to-libm |
| 代码生成级 | Clang `-O0`/`-O1`/`-O2`/`-O3`（默认 `-O3`） |
| SIMD | 默认使用 256-bit LLVM 向量宽度偏好；`--threads=1` 时使用 Affine Super Vectorizer 和 Vector-to-LLVM |
| 多线程 | 默认将 Linalg 并行维 lowering 为 OpenMP，并由运行时使用可用 CPU；`--threads=1` 可关闭 |
| 目标调优 | `--target-triple`、`--march`（含 `native`）、`--mcpu`、`--mtune`、`--target-feature`、`--sysroot` |
| 图级优化 | 无（无算子融合、无常量折叠、无量化优化） |

隐式优化：Dropout scale=1.0 折叠为恒等；Split 通过 SSA 消除。

---

## 7. 测试覆盖

### 7.1 lit / FileCheck 测试

- `test/Dialect/NCNN/`：op 解析/打印/验证
- `test/Conversion/NCNNToFunc/`、`test/Conversion/NCNNToTosa/`：转换正确性、合法 op 边界物化和失败回滚
- `test/Transforms/`：NormalizeNCNN 两阶段提交、GenerateCAPI、各 verify gate；bufferized
  ownership 检查覆盖 view、CFG block argument 和 region yield alias；缺少显式所有权契约的
  call result 按保守策略拒绝
- `test/Pipelines/`：完整流水线组合、SqueezeNet 端到端、严格失败路径

### 7.2 GTest 单元测试

- `test/Unit/parser_test.cpp`、`malformed_graph_test.cpp`：解析器健壮性
- `test/Unit/load_squeezenet*.cpp`：集成（层数、拓扑、权重形状/类型）
- `test/Unit/ncnn_importer_test.cpp`：导入器正常/异常覆盖

### 7.3 数值黄金测试

`test/Numerical/`：将编译产物 `.so` 的输出与 vendored upstream ncnn runtime 对比。当前
reference 使用 ncnn 的优化 CPU 路径，允许其按平台和 CPU 选择 runtime dispatch、SIMD、packing、
线程、Winograd/SGEMM；test support 显式关闭 Vulkan 和 FP16/BF16 降精度，并按 case 选择 INT8。
测试没有注册 custom layer，也没有调用 `create_layer_naive()`。

- `operators/supported_ops_test.cpp`：覆盖支持算子的参数矩阵，包括：
  - Convolution：basic、no-bias、dilated、asymmetric padding/stride、SAME_UPPER/LOWER
  - ReLU：standard、leaky、zero/negative inputs
  - Pooling：max、average(exclude-pad)、非对称空间参数、global 参数忽略、SAME_UPPER/LOWER、tail window
  - Softmax：channel/height/width axis + negative-axis
  - Concat：channel/height/width + negative-axis
  - Split：2-way、3-way consumer topology
  - Dropout：identity
  - ConvolutionDepthWise：basic、非对称 stride/dilation/padding、SAME_UPPER/LOWER，及
    PP-OCRv6 small/medium 使用的 `C=96/K=7/pad=3`、`C=256/K=9/pad=4` 大核实例
  - Reshape：静态 shape、`-1` 推断、`0` 复制输入维度，以及 `1w,1h,1c` 引用输入动态 shape expression
  - BinaryOp：标量、channel broadcast、反向 broadcast
  - PP-OCRv6 rec 新增算子：GELU（含负尾部）、Squeeze、BatchNorm 零方差保护、
    ExpandDims 负轴、rank-2 Permute、非默认 alpha/beta Gemm、BinaryOp add、
    rank-3 输入融合 ReLU InnerProduct
  - PP-OCRv6 det 新增算子：恒等及四边非对称 constant Padding；恒等、2 倍、H/W 非对称
    3/4 倍和 8 倍 nearest Interp；带 bias/融合 ReLU、无 bias 及真实 tiny head `I=16/O=1`
    的 Deconvolution；Sigmoid 概率范围、极值截断和 NaN 传播语义；权重加载另覆盖正式 FP32 tag
  - DetectionOutput：Caffe SSD decode、confidence filtering、逐类 top-k、IoU NMS、全局 top-k，
    同时验证最大 storage 与实际 `[count,6]` shape
- `models/squeezenet_test.cpp`：完整 SqueezeNet v1.1 端到端
- `models/pp_lcnet_test.cpp`：PP-LCNet doc ori、textline ori、ChineseOCR Lite AngleNet、PP-OCRv6
  tiny rec、tiny det、small det、medium det 和 PP-OCRv5 mobile/server det 与 upstream ncnn 数值对齐；
  五个 det 模型均使用 `3x640x640` 输入并验证 `1x640x640` 概率图及重复调用一致性。tiny、
  medium、v5 mobile 和 v5 server 使用 `1e-4` 预算；small 使用独立 `3e-4` 预算，固定输入下实测最大
  绝对误差约 `2.256e-4`
  - 全有限输出、softmax 求和误差 ≤1e-5（PP-OCRv6 的 6906 类输出为 ≤2e-5）、top-1 匹配、top-5 集合匹配、最大绝对误差 ≤1e-4

### 7.4 运行时测试

- `squeezenet-shared-library`：完整编译 + `--verify-execution`
- `ncnn-compile-cli`：CLI 契约测试
- 静态数值基线通过 `ctest -L numerical-static-baseline` 独立执行，覆盖 31 个 source 计算层的静态
  lowering/数值测试、SqueezeNet、PP-LCNet、PP-OCR，以及固定 target 的导出符号、
  manifest、header 和共享库产物大小。静态产物不得导出 `_infer_output_shapes`；原生 artifact/CLI
  基线另使用 `static-baseline` label。
- 动态数值测试编译为独立的 `numerical_dynamic_tests`，通过 `ctest -L numerical-dynamic`
  执行；另有动态算子测试覆盖 global/adaptive pooling、Reshape `0/-1`、Slice `-233` 和
  InnerProduct 动态 M，并在多个运行时 shape 上与 ncnn 对齐。静态与动态测试不共享 expected。

### 7.5 未覆盖（明确排除）

- 通用 `ConvolutionDepthWise` / 分组卷积；当前仅支持纯 depthwise 子集
- Average pooling `include_pad=1`（严格流水线拒绝）
- DetectionOutput 动态 prior 数量、动态权重，以及需要跨输入运行时 shape 等式的动态通道实例

---

## 8. 已知限制总结

| 类别 | 限制 |
|---|---|
| 算子覆盖 | 31 个 source 计算层、32 个方言计算 op 的受限实例；DetectionOutput 仅 Caffe SSD；无 PriorBox/Proposal/Yolo、通用 group conv、RNN |
| 量化 | 支持已覆盖的 INT8 scale-term Convolution/Depthwise/InnerProduct/Gemm 与 Quantize/Dequantize/Requantize/Cast 链；未覆盖的量化层仍由严格 lowering gate 拒绝；FP16/BF16 storage boundary 按精度策略处理 |
| 形状 | 静态 shape 广泛覆盖；fixed-rank 动态 extent 覆盖 Convolution、Depthwise、Deconvolution、Padding、Interp、Sigmoid/Hard activation/GELU/Dropout/Softmax/BatchNorm、空间 Concat、可证明的 Binary 广播、Reshape `0/-1`、Slice、Reduction、轴变换、regular/global/adaptive Pooling、Gemm/InnerProduct 动态 M。V2 shape program 支持多输入 Add/Multiply/FloorDiv/CeilDiv/Max。动态 rank 仅一入一出 identity/ReLU rank 1..4；DetectionOutput 动态 prior、动态权重及一般动态图仍不支持 |
| 数据类型 | GenerateCAPI 支持 typed f16/bf16/f32/f64 与整数 ABI；当前 ncnn 模型数据主路径为 f32 |
| 入口 | 一个执行入口；shape-only 动态输出另有 inference 入口；执行 ABI 按输入组、输出 data、数据依赖元数据排列 |
| 平台 | Linux 64-bit ELF；x86-64 原生运行验证，AArch64/RISC-V 低精度指令静态验证 |
| GPU/NPU | 无，仅 CPU（LLVM 后端） |
| 图优化 | 无算子融合、无常量折叠 |

---

## 9. 关键源文件索引

| 关注点 | 文件 |
|---|---|
| Op 定义（TableGen） | `include/ncnn-mlir/Dialect/NCNN/IR/NCNNOps.td` |
| Op 形状推断/验证 | `lib/Dialect/NCNN/IR/NCNNOps.cpp` |
| 图解析/数据模型 | `lib/Graph/graph.cpp`、`lib/Graph/parser.cpp` |
| 导入器（层接受） | `lib/Importer/NCNNImporter.cpp`、`lib/Importer/ImporterInternal.hpp` |
| 导入器（算子族实现） | `lib/Importer/Import{Input,Convolution,Pooling,Activation,Tensor,Ops,Detection}.cpp` |
| 模型→函数转换 | `lib/Conversion/NCNNToFunc/NCNNToFunc.cpp` |
| 规范化 | `lib/Transforms/NormalizeNCNN/NormalizeNCNN.cpp` |
| ncnn→TOSA 转换 | `lib/Conversion/NCNNToTosa/NCNNToTosa.cpp` |
| C API 生成 | `lib/Transforms/GenerateCAPI/GenerateCAPI.cpp` |
| 流水线组合 | `lib/Pipelines/NCNNPipelines.cpp` |
| 前端驱动 | `tools/ncnn-mlir-driver.cpp` |
| 编译驱动 | `tools/ncnn-compile.cpp` |
| mlir-opt 克隆 | `bin/ncnn-mlir-opt.cpp` |
