# ncnn 方言 IR 格式说明

> 对应源码：`include/ncnn-mlir/Dialect/NCNN/IR/`（NCNNDialect.td/.hpp、NCNNOps.td/.hpp）；
> `lib/Dialect/NCNN/IR/`（NCNNDialect.cpp、NCNNOps.cpp）。
> 生成工具：`ncnn-mlir-driver --emit=mlir`（默认阶段）。
> 相关文档：[parsed-graph-format.md](parsed-graph-format.md)、[ncnn-mlir-driver-usage.md](ncnn-mlir-driver-usage.md)、
> [operator-numerical-validation-guide.md](operator-numerical-validation-guide.md)。

`ncnn` 方言 IR 是编译器**类型化前端 IR**，是一个标准的 **MLIR 模块**（`builtin.module`
里一个 `ncnn.model`）。它由 `ncnn_importer::import_graph()` 从原始
[parsed-graph](parsed-graph-format.md) 提升而来，是下游唯一产品 lowering
（TOSA→Linalg→MemRef→LLVM）真正消费的输入。

> 本方言取代了早期的自定义 C++ 前端 IR（`ncnn_frontend`，已删除）。同样的
> 「线性层列表 → 类型化 SSA DAG」提升现在直接落到 MLIR：SSA 值、use-def 链、
> 强类型属性、形状推断都由 MLIR 基础设施承载。

---

## 1. 这是什么阶段

流水线中的位置：

```
.param/.bin
  ──ncnn_graph::Graph::load──▶  parsed-graph        (原始层列表，1:1 镜像 ncnn 文件)
  ──ncnn_importer::import_graph──▶  ncnn 方言 MLIR 模块   ← 本文档
  ──convert-ncnn-model-to-func──▶  func.func + arith.constant + ncnn 计算 op
  ──normalize-ncnn──▶  func.func + normalized ncnn 计算 op
  ──convert-ncnn-to-tosa──▶  func.func + tosa + 未分配路径的 ncnn op
  ──ncnn-to-tosa-pipeline/verify-no-ncnn-ops──▶  无 ncnn op 的目标 IR
  ──ncnn-tosa-to-linalg-pipeline──▶  tensor + Linalg/Arith/Math IR
  ──ncnn-linalg-to-memref-pipeline──▶  caller-owned output memref + void return
  ──generate-ncnn-c-api──▶  private model + prepared bare-pointer ABI metadata
  ──ncnn-memref-to-llvm-pipeline──▶  纯 LLVM dialect IR
```

`convert-ncnn-model-to-func` 是 normalize 和所有目标 conversion 的固定前置 pass，而不是
后端 ABI 收尾步骤。
它使用 MLIR full dialect conversion 消除模型边界并建立标准函数形态：`ModelOp` pattern
创建函数并把原 region 内容移动到函数 block，input/output 映射为参数和单个 return，
`ConstOp` pattern 转为 `arith.constant`。conversion driver 管理 SSA remap、合法性和 rewrite
失败处理，不 clone 整个模型 SSA 图，也不提供整 module 事务回滚。该 pass 不转换计算语义或 CHW/OIHW 布局；后续
`normalize-ncnn` 保持 CHW/OIHW，收敛 axis、padding、融合属性等目标无关语义。该 pass
先只读验证所有 ncnn op 的函数边界并预计算可能失败的 SAME padding，全部成功后才原地
提交属性更新；失败不会留下部分规范化结果，也不需要 clone 整个模块。Split 的 SSA fan-out
语义由 op folder 表达，交给标准 canonicalize 或目标 conversion 消除。
`convert-ncnn-to-tosa` 只处理函数体内明确属于 TOSA 支持集合的计算算子；单独运行时允许
其他 ncnn op 留给其他 conversion。组合的 `ncnn-to-tosa-pipeline` 通过最终残留检查要求
输入模型严格消除 ncnn op。大多数算子转换为 TOSA；标准 GELU 为保持 ncnn 的 `erfc`
负尾部数值语义，转换为可 bufferize 的 `linalg.map + math.erfc`。

`convert-ncnn-to-tosa` 基于 MLIR dialect conversion：每种受支持算子由独立 conversion
pattern 转换，`TypeConverter` 描述 rank-3 CHW 到 rank-4 NHWC 的类型映射，source/target
materialization 自动在函数边界、合法非 ncnn op 和残留 ncnn op 两侧插入 transpose + reshape。
函数签名不参与类型转换，所以公共函数 ABI 始终保持 CHW。支持的 ncnn op 被标为 illegal，
明确不支持的实例和未知 ncnn op 保持 legal；conversion driver 负责 SSA 值 remap、替换、删除
和 rewrite 失败处理，不使用手写值映射或模块 clone。
`ncnn-linalg-to-memref-pipeline` 在 Linalg 阶段之后执行 One-Shot Bufferize，随后立即将
memref result 提升为带 `bufferize.result` 的输出参数，再执行 deallocation。入口函数不返回
tensor/memref，调用方拥有 input/output buffer，函数只释放内部临时分配。最终 gate 禁止
`tensor.*`、`bufferization.*`、`ncnn.*`、`tosa.*` 和 unrealized cast 残留，并验证每个
`memref.alloc` 有唯一且后支配所有使用的释放。
`ncnn-memref-to-llvm-pipeline` 将 Linalg copy 和计算统一降为 loops，再完成
Affine/SCF/Math/Arith/MemRef/Func/CF 到 LLVM dialect 的转换，因此不会引入外部
`memrefCopy` runtime 符号。若模块经过 `generate-ncnn-c-api`，该 pipeline 还会在 Func→LLVM
后生成模型专用 typed wrapper：公共参数顺序固定为全部输入参数组、全部输出数据指针、shape-only
动态输出 data capacity、数据依赖输出元数据。wrapper 为静态或动态 ranked tensor 构造连续
memref descriptor，并可执行 rank
1..4 dispatch 或返回 actual shape。模型实现被改为 private；公共 wrapper 用状态码 `0..5`
区分成功、空指针、非法 shape、约束违反、shape 算术溢出和输出容量不足。

与 parsed-graph 的关键区别：**parsed-graph 是保留 ncnn 文件语义的线性层视图，ncnn 方言模块
是类型化的 SSA DAG**。parsed-graph 不是下游 IR，也不提供完整的中间值 shape inference。
`import_graph` 做了这些提升：

- **blob 名 → SSA 值**：每条 ncnn blob 变成一个带 `RankedTensorType` 的 MLIR `Value`
  （`%0`、`%1`…，输入由 `ncnn.input` 定义）。唯一定义点与 use-def 链由 MLIR 自动维护；
  blob 名保留在定义它的算子的 `ncnn.name` 属性里（溯源用）。
- **权重 → 常量算子**：conv 的 weight/bias 从"挂在层上的张量"变成独立的 `ncnn.const`
  算子，产出各自的值，作为 conv 的操作数。
- **参数字典 → 强类型属性**：ncnn 的 `{0=64 1=3 …}` 数字 key 解析成
  `ncnn.convolution {kernel_h=3, kernel_w=3, stride_h=2, …}` 这样的具名强类型属性
  （`I64Attr` / `BoolAttr` / `F32Attr`）。
- **shape inference**：每个值都带推断出的 ranked shape 与元素类型。传统模型通常为静态 extent；
  无 override 时，尺寸完全省略的 Input 自动成为 `[C,?,?]`，直接连接到 Convolution 时可从
  `[O,I,H,W]` 权重推导 `C=I`。`--input-shape` 的 `?` 可保留动态 extent，`*` 为受限模型生成 rank 1..4 ranked specialization。
  动态值能否继续下降取决于具体算子支持。
  conv/pool/concat 实现 `InferTensorTypeAdaptor`，importer 通过标准 `inferReturnTypes` 获取结果，
  `InferTypeOpInterface` 用相同实现精确复核声明类型。

---

## 2. 生成方式

```bash
cd /mnt/ncnn-compiler/compiler

# 默认就是 MLIR（--emit 可省略）
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param

# 显式写法 + 输出到文件
./build/tools/ncnn-mlir-driver --emit=mlir \
  test/third_party/ncnn/examples/squeezenet_v1.1.param -o squeezenet.mlir

# 用 ncnn-mlir-opt 做 round-trip 校验（parse -> verify -> print）
./build/bin/ncnn-mlir-opt squeezenet.mlir
```

`.bin` 权重路径默认由 `.param` 推导（`.param` → `.bin`），也可用 `--bin=<path>` 显式指定。

---

## 3. 整体结构

输出是标准 MLIR 文本：一个 `module`，内含一个 `ncnn.model @model`。

```mlir
module {
  ncnn.model @model {
    %input = ncnn.input {blob_name = "data", layer_name = "data"} : tensor<3x227x227xf32>
    %cst = ncnn.const {name = "conv1.weight.0", value = dense<...>} : tensor<64x3x3x3xf32>
    %cst_0 = ncnn.const {name = "conv1.weight.1", value = dense<...>} : tensor<64xf32>
    %0 = ncnn.convolution %input, %cst, %cst_0 {...} : (...) -> tensor<64x113x113xf32>
    %1 = ncnn.relu %0 {...} : (tensor<64x113x113xf32>) -> tensor<64x113x113xf32>
    ...
    ncnn.output %N {blob_name = "prob"} : tensor<1000xf32>
  }
}
```

- **`ncnn.input`**对应 ncnn `Input` 层，形状 `[C,H,W]`。
- fixed-rank 动态输入可以在 `ncnn.model` 上携带 `ncnn.shape_constraints`，元素为结构化
  `#ncnn.dim_constraint<input, dim, min, multiple_of>` 属性；约束只允许指向动态维，可由 CLI
  提供，也可由可追溯到模型输入的动态 Slice 自动推导 minimum。
- 函数化后，动态输出维的 provenance 存为 `ncnn.shape_program`。简单单来源表达式使用 V1
  opcode/operand 对；复合或多输入表达式使用 V2 前缀树，支持 Constant、InputDim、Add、
  Multiply、FloorDiv、CeilDiv 和 Max。V2 节点已携带输入索引，不设置 `shape_source_input`。
- **`ncnn.output`**对应 `graph.output_blob_names` 选择的导出 blob。
- 计算层和权重都由 `ncnn.*` 算子表示，尚未建立函数 ABI。

---

## 4. 算子与属性

模型边界由 `ncnn.model`、`ncnn.input`、`ncnn.const`、`ncnn.output` 表示。当前计算 op 共
32 个：31 个对应 source 计算层，`ncnn.zero_point_cast` 是 lowering 使用的内部 op。权威集合是 [`NCNNOps.td`](../include/ncnn-mlir/Dialect/NCNN/IR/NCNNOps.td) 的
TableGen 定义，完整能力矩阵见 [ncnn-compile-support-status.md](ncnn-compile-support-status.md)。
下表 7 项只是 SqueezeNet 示例实际使用的计算 op：

| 算子 | 操作数 | 属性 | 结果 |
|------|--------|------|------|
| `ncnn.convolution` | input, weight, [bias, scales…] | `kernel_h/kernel_w`、`stride_h/stride_w`、`dilation_h/dilation_w`、`pad_top/pad_bottom/pad_left/pad_right`、`has_bias`、`int8_scale_term` | 1 个张量 |
| `ncnn.relu` | input | `negative_slope`（默认 0.0；非 0 = LeakyReLU） | 1 个（同类型） |
| `ncnn.pooling` | input | `kind`(0=max,1=average)、`mode`(0=regular,1=global,2=adaptive)、`kernel_h/kernel_w`、`stride_h/stride_w`、`pad_*`、`pad_mode`(0..3)、`include_pad` | 1 个张量 |
| `ncnn.split` | input | 无 | ≥2 个（都与输入同类型） |
| `ncnn.concat` | ≥2 个输入 | `axis` | 1 个张量 |
| `ncnn.dropout` | input | `scale`（默认 1.0；推理期恒等/缩放） | 1 个（同类型） |
| `ncnn.softmax` | input | `axis` | 1 个（同类型） |

`ncnn.detection_output` 是特殊的双结果计算 op：第一个结果是
`tensor<maximum_detections x 6 x f32>` bounded storage，第二个 `tensor<2xi64>` 携带实际
`[count,6]` shape。第二个结果不对应额外的 source top blob。

每个算子还带两个 **discardable 属性**用于溯源：`ncnn.name`（ncnn 层名）、
`ncnn.source_layer`（来自 parsed-graph 的第几层）。

convolution 真实样例：

```mlir
%0 = ncnn.convolution %arg0, %cst, %cst_0 {dilation_h = 1 : i64, dilation_w = 1 : i64,
     has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, ncnn.name = "conv1",
     ncnn.source_layer = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64,
     pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64}
     : (tensor<3x227x227xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>)
     -> tensor<64x113x113xf32>
```

SAME padding 用 ncnn 哨兵表达：`pad_*` 全为 `-233` = SAME_UPPER，全为 `-234` = SAME_LOWER。

---

## 5. 张量类型与布局约定

值的类型是无 encoding 的 MLIR `RankedTensorType`（`tensor<形状 x 元素类型>`），
**布局靠维度顺序约定**（不是单独的属性字段）。`ncnn` op 不接受 sparse、layout 或其他
tensor encoding；带 encoding 的 tensor 会在方言校验阶段被明确拒绝：

| 张量 | 维序约定 | 例 |
|------|---------|----|
| 特征图 | `[C, H, W]`（ncnn 原生 CHW） | `tensor<64x113x113xf32>` |
| conv 权重 | `[O, I, H, W]` | `tensor<64x3x3x3xf32>` |
| conv bias / 1-D 常量 | `[N]` | `tensor<64xf32>` |

NCNN dialect 当前会出现 `f32`（主路径）、`f16`、`bf16`、`i8`，量化累加和边界 op 还会使用
`i32`。后续 GenerateCAPI 的可接受 ABI 类型范围更宽：`f16/bf16/f32/f64` 以及
8/16/32/64 位有符号或无符号整数；这不表示 importer 能生成所有这些类型。

> **布局是隐式约定，不是显式 layout 字段**。verifier 与形状推断按此约定校验
> （conv 要求 input rank3、weight rank4）。ncnn→tosa 下降通过 `TypeConverter` 和双向
> materialization 按需完成 CHW↔NHWC 转换，函数 ABI 仍保持 CHW。

---

## 6. 形状推断

- `relu` / `dropout` / `softmax`：`SameOperandsAndResultType`（结果类型 = 输入类型）。
- `split`：所有结果 = 输入类型（个数 = 输出 blob 数，≥2）。
- `convolution` / `pooling` / `concat` / `reshape` / `detection_output` 等通过
  `InferTensorTypeAdaptor` 实现
  `inferReturnTypeComponents`，在构建时计算结果形状（conv 输出 `[O, H_out, W_out]`、pool
  按 regular/global/adaptive、concat 沿 `axis` 求和）。标准 `InferTypeOpInterface` verifier
  会重算并与声明的结果类型精确比对，不一致即报错；importer 不维护额外的公开推断入口。

conv 输出尺寸公式（显式 pad）：
`extent = dilation*(kernel-1)+1`；`out = 1 + (in + pad_before + pad_after - extent)/stride`。
SAME（`-233`/`-234`）：`out = 1 + (in-1)/stride`。

显式非负 padding 的 Convolution 可传播动态 H/W；动态 nearest Interp 以运行时 H/W 乘静态
scale；DetectionOutput 根据输入和 top-k 参数推导最大 storage，实际行数由执行时 shape carrier
返回。动态 SAME 在对应 stride 为 1 时可直接化为零显式 padding；stride 大于 1 时仍无法在
静态 `pad` 属性中表达运行时分配，编译器会拒绝。

动态 Slice 轴在最后一片为 `-233`、其他片为正数或 `-233` 时，可自动生成输入 minimum：显式
size 求和，每个 `-233` 至少按 1 计。追溯当前可穿过受支持的 BinaryOp 广播和 Squeeze 轴映射；
无法追溯到模型输入时仍由 verifier 拒绝。

---

## 7. 完整示例（SqueezeNet v1.1 节选）

```bash
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param 2>/dev/null | head
```

```mlir
module {
  ncnn.model @model {
    %input = ncnn.input {blob_name = "data", layer_name = "data"} : tensor<3x227x227xf32>
    %cst = ncnn.const {name = "conv1.weight.0", value = dense<...>} : tensor<64x3x3x3xf32>
    %0 = ncnn.convolution %input, %cst, %cst_0 {...} : (...) -> tensor<64x113x113xf32>
    %1 = ncnn.relu %0 {...} : (tensor<64x113x113xf32>) -> tensor<64x113x113xf32>
    %2 = ncnn.pooling %1 {kind = 0 : i64, mode = 0 : i64, ...} : (tensor<64x113x113xf32>) -> tensor<64x56x56xf32>
    ...
    ncnn.output %N {blob_name = "prob"} : tensor<1000xf32>
  }
}
```

统计（squeezenet_v1.1，实测）：

- **52 个 `ncnn.const`**（26 个 conv 各带 weight + bias）。
- 计算算子：`ncnn.convolution` 26、`ncnn.relu` 26、`ncnn.split` 8、`ncnn.concat` 8、
  `ncnn.pooling` 4、`ncnn.softmax` 1、`ncnn.dropout` 1。
- `ncnn.input` 1 个；`ncnn.output` 1 个。函数化后分别成为函数参数和返回值。

---

## 8. 相关源码

| 内容 | 文件 |
|------|------|
| 方言/算子定义（TableGen） | `include/ncnn-mlir/Dialect/NCNN/IR/NCNNOps.td` |
| 算子形状推断 / verifier | `lib/Dialect/NCNN/IR/NCNNOps.cpp` |
| parsed-graph → MLIR 提升 | `lib/Importer/NCNNImporter.cpp` |
| 文本由 MLIR 通用 printer 产生 | （`ModuleOp::print`） |
