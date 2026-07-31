# ncnn 方言 IR 格式说明

> 对应源码：`include/ncnn-mlir/Dialect/NCNN/IR/`（NCNNDialect.td/.hpp、NCNNOps.td/.hpp）；
> `lib/Dialect/NCNN/IR/`（NCNNDialect.cpp、NCNNOps.cpp）。
> 生成工具：`ncnn-mlir-driver --emit=mlir`（默认阶段）。
> 相关文档：[parsed-graph-format.md](parsed-graph-format.md)、[ncnn-mlir-driver-usage.md](ncnn-mlir-driver-usage.md)。

`ncnn` 方言 IR 是编译器**类型化前端 IR**，是一个标准的 **MLIR 模块**（`builtin.module`
里一个 `ncnn.model`）。它由 `ncnn_importer::import_graph()` 从原始
[parsed-graph](parsed-graph-format.md) 提升而来，是下游多路径 lowering（TOSA、Linalg/SCF、Host）
真正消费的输入。

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
  ──ncnn-to-tosa-pipeline/verify-no-ncnn-ops──▶  严格纯 TOSA IR
  ──ncnn-tosa-to-linalg-pipeline──▶  tensor + Linalg/Arith/Math IR
  ──ncnn-linalg-to-memref-pipeline──▶  caller-owned output memref + void return
```

`convert-ncnn-model-to-func` 是 normalize 和所有目标 conversion 的固定前置 pass，而不是
后端 ABI 收尾步骤。
它只消除模型边界并建立标准函数形态，不转换计算语义或 CHW/OIHW 布局；后续
`normalize-ncnn` 保持 CHW/OIHW，收敛 axis、padding、融合属性等目标无关语义。
`convert-ncnn-to-tosa` 只处理函数体内明确属于 TOSA 支持集合的计算算子；单独运行时允许
其他 ncnn op 留给其他 conversion。组合的 `ncnn-to-tosa-pipeline` 通过最终残留检查要求
输入模型严格转换为纯 TOSA。
`ncnn-linalg-to-memref-pipeline` 在 Linalg 阶段之后执行 One-Shot Bufferize，随后立即将
memref result 提升为带 `bufferize.result` 的输出参数，再执行 deallocation。入口函数不返回
tensor/memref，调用方拥有 input/output buffer，函数只释放内部临时分配。

与 parsed-graph 的关键区别：**parsed-graph 是层的线性列表，ncnn 方言模块是类型化的 SSA DAG**。
`import_graph` 做了这些提升：

- **blob 名 → SSA 值**：每条 ncnn blob 变成一个带 `RankedTensorType` 的 MLIR `Value`
  （`%0`、`%1`…，输入由 `ncnn.input` 定义）。唯一定义点与 use-def 链由 MLIR 自动维护；
  blob 名保留在定义它的算子的 `ncnn.name` 属性里（溯源用）。
- **权重 → 常量算子**：conv 的 weight/bias 从"挂在层上的张量"变成独立的 `ncnn.const`
  算子，产出各自的值，作为 conv 的操作数。
- **参数字典 → 强类型属性**：ncnn 的 `{0=64 1=3 …}` 数字 key 解析成
  `ncnn.convolution {kernel_h=3, kernel_w=3, stride_h=2, …}` 这样的具名强类型属性
  （`I64Attr` / `BoolAttr` / `F32Attr`）。
- **shape inference**：每个值都带推断出的静态形状与元素类型（`tensor<64x113x113xf32>`）。
  conv/pool/concat 实现了 `InferShapedTypeOpInterface`，结果类型在构建时推断、verifier 复核。

---

## 2. 生成方式

```bash
cd /mnt/ncnn-compiler/compiler

# 默认就是 MLIR（--emit 可省略）
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param

# 显式写法 + 输出到文件
./build/tools/ncnn-mlir-driver --emit=mlir \
  ../ncnn/examples/squeezenet_v1.1.param -o squeezenet.mlir

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
- **`ncnn.output`**对应 `graph.output_blob_names` 选择的导出 blob。
- 计算层和权重都由 `ncnn.*` 算子表示，尚未建立函数 ABI。

---

## 4. 算子与属性

模型边界由 `ncnn.model`、`ncnn.input`、`ncnn.const`、`ncnn.output` 表示，另有 7 个计算算子：

| 算子 | 操作数 | 属性 | 结果 |
|------|--------|------|------|
| `ncnn.convolution` | input, weight, [bias, scales…] | `kernel_h/kernel_w`、`stride_h/stride_w`、`dilation_h/dilation_w`、`pad_top/pad_bottom/pad_left/pad_right`、`has_bias`、`int8_scale_term` | 1 个张量 |
| `ncnn.relu` | input | `negative_slope`（默认 0.0；非 0 = LeakyReLU） | 1 个（同类型） |
| `ncnn.pooling` | input | `kind`(0=max,1=average)、`mode`(0=regular,1=global,2=adaptive)、`kernel_h/kernel_w`、`stride_h/stride_w`、`pad_*`、`pad_mode`(0..3)、`include_pad` | 1 个张量 |
| `ncnn.split` | input | 无 | ≥2 个（都与输入同类型） |
| `ncnn.concat` | ≥2 个输入 | `axis` | 1 个张量 |
| `ncnn.dropout` | input | `scale`（默认 1.0；推理期恒等/缩放） | 1 个（同类型） |
| `ncnn.softmax` | input | `axis` | 1 个（同类型） |

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

值的类型是 MLIR `RankedTensorType`（`tensor<形状 x 元素类型>`），**布局靠维度顺序约定**
（不是单独的属性字段）：

| 张量 | 维序约定 | 例 |
|------|---------|----|
| 特征图 | `[C, H, W]`（ncnn 原生 CHW） | `tensor<64x113x113xf32>` |
| conv 权重 | `[O, I, H, W]` | `tensor<64x3x3x3xf32>` |
| conv bias / 1-D 常量 | `[N]` | `tensor<64xf32>` |

元素类型：`f32`（主路径）、`f16`、`i8`（量化权重/结果）。

> **布局是隐式约定，不是显式 layout 字段**。verifier 与形状推断按此约定校验
> （conv 要求 input rank3、weight rank4）。ncnn→tosa 下降时再按需做 CHW→NHWC 转换
> （见开发指南的布局决策）。

---

## 6. 形状推断

- `relu` / `dropout` / `softmax`：`SameOperandsAndResultType`（结果类型 = 输入类型）。
- `split`：所有结果 = 输入类型（个数 = 输出 blob 数，≥2）。
- `convolution` / `pooling` / `concat`：实现 `InferShapedTypeOpInterface::inferReturnTypeComponents`，
  在构建时计算结果形状（conv 输出 `[O, H_out, W_out]`、pool 按 regular/global/adaptive、
  concat 沿 `axis` 求和）。verifier 会重算并与声明的结果类型比对，不一致即报错。

conv 输出尺寸公式（显式 pad）：
`extent = dilation*(kernel-1)+1`；`out = 1 + (in + pad_before + pad_after - extent)/stride`。
SAME（`-233`/`-234`）：`out = 1 + (in-1)/stride`。

---

## 7. 完整示例（SqueezeNet v1.1 节选）

```bash
./build/tools/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param 2>/dev/null | head
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
