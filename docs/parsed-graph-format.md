# parsed-graph 产物格式说明

> 对应 `ncnn-mlir-driver --emit=parsed-graph` 的文本产物。
> 源码：`lib/Graph/graph.cpp` 的 `ncnn_graph::Graph::dump()`（相对 `compiler/`）。
> 上游流水线见 [ncnn-mlir-driver-usage.md](ncnn-mlir-driver-usage.md)。

## 1. 这是什么阶段

`parsed-graph` 是编译流水线的**最前端语义视图**：`.param`/`.bin` 刚被解析成
`ncnn_graph::Graph` 数据模型，尚未建立 MLIR 类型化 SSA 或 lowering。它保留 ncnn 原始模型
的层顺序、blob 连线、参数字典
（`key=value`）、以及从 `.bin` 按算子顺序读出并绑定到各层的权重张量。

它的用途：

- **交叉验证解析器**：和 netron 打开同一模型对照，确认层数/blob 数/参数值一致。
- **排查导入 bug**：ncnn 方言 IR（类型化 IR）出问题时，先看 `parsed-graph`
  确认是解析层错了还是 import 层错了。
- **理解模型结构**：一屏看清整张图的拓扑与权重规模。

它**不是**给下游 pass 吃的 IR，也不是完整的中间值 shape inference——那是 ncnn 方言 IR（`--emit=mlir`，见
[ncnn-ir-format.md](ncnn-ir-format.md)）及之后的阶段。`parsed-graph`
是纯人读的调试快照。

## 2. 生成方式

```bash
cd /mnt/ncnn-compiler/compiler
./build/tools/ncnn-mlir-driver test/third_party/ncnn/examples/squeezenet_v1.1.param \
    --emit=parsed-graph
```

`.bin` 默认由 `.param` 路径推导（`.param` → `.bin`）。若权重文件名不匹配，
用 `--bin=<path>` 显式指定。写文件用 `-o <path>`，默认写 stdout。

## 3. 整体结构

产物分两部分：**图头**（层数/blob 数 + 输入输出 blob 名）和**层列表**。

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
  [ 74] Softmax        prob                   in=[pool10] out=[prob] {0=0}
```

### 3.1 图头

| 行 | 含义 |
|----|------|
| `ncnn_graph: N layers, M blobs` | 层总数与 blob（张量边）总数。对应 `.param` 第二行的两个整数。 |
| `inputs: K` + 缩进列表 | 图输入 blob 名（来自 `Input` 层 / ncnn 的输入声明）。 |
| `outputs: K` + 缩进列表 | 图输出 blob 名。 |

### 3.2 层列表

每层一行，格式（源码 `dump()` 用固定宽度对齐）：

```
  [idx] <type>        <name>                 in=[...] out=[...] {params} w=[weights]
```

- `[idx]`：层序号，右对齐 3 宽。就是层在 `.param` 里的出现顺序。
- `<type>`：ncnn 算子类型名，左对齐 14 宽，如 `Convolution`/`ReLU`/`Pooling`。
- `<name>`：层名，左对齐 22 宽。
- `in=[...]`：输入 blob 名列表，逗号分隔。`Input` 层为空 `in=[]`。
- `out=[...]`：输出 blob 名列表。`Split` 层会有多个输出（复制成 n 份）。
- `{params}`：参数字典，`key=value` 空格分隔。**仅当该层有参数时出现**。
- `w=[...]`：绑定的权重张量列表。**仅当该层有权重时出现**。

## 4. 参数字典 `{...}` 的读法

key 是 ncnn 的参数 id（整数），value 按类型打印。参数语义**因算子而异**，
需对照 ncnn 源码 `src/layer/<op>.cpp` 的 `load_param`。四种取值形态：

| 形态 | 打印样式 | 例 |
|------|---------|-----|
| 整数 | `id=123` | `0=64` |
| 浮点 | `id=1.5` | `0=0.1` |
| 字符串 | `id="text"` | `0="relu"` |
| 整数数组 | `id=[1,2,3]` | `-23310` 类数组重映射后 |
| 浮点数组 | `id=[0.1,0.2]` | 激活参数等 |

> 数组参数在 ncnn 原始文件里用负 key（`id <= -23300`，偏移 `-23300`）编码，
> 解析器已按 ncnn/netron 约定重映射为正 key。详见
> [../../docs/netron-ncnn-parsing.md](../../docs/netron-ncnn-parsing.md)。

**示例（`Convolution` conv1）**：`{0=64 1=3 2=1 3=2 4=0 5=1 6=1728}`

对照 `convolution.cpp` 的 `load_param`：

| key | 语义 | 值 |
|-----|------|-----|
| `0` | num_output | 64 |
| `1` | kernel_w | 3 |
| `2` | dilation_w | 1 |
| `3` | stride_w | 2 |
| `4` | pad_left | 0 |
| `5` | bias_term | 1（有 bias） |
| `6` | weight_data_size | 1728（= 64×3×3×3 个 float） |

**示例（`Pooling` pool1）**：`{0=0 1=3 2=2 3=0 4=0}` → pool_type=0(max),
kernel=3, stride=2, pad=0, global_pooling=0。`pool10` 是 `{0=1 ... 4=1}`，
`0=1`(avg) + `4=1`(global) 即全局平均池化。

**示例（`ReLU`）**：`{0=0}` → negative_slope=0，即标准 ReLU。

> 未打印的 key 表示该层用该参数的**默认值**，不代表值为 0。

## 5. 权重 `w=[...]` 的读法

每个权重张量打印为 `[shape:dtype:bytesB]`，多个张量逗号分隔。张量的**顺序**
就是 ncnn `load_model` 从 `.bin` 读出的顺序。

**示例（conv1）**：`w=[[64,3,3,3:f32:6912B],[64:f32:256B]]`

- 第一个 `[64,3,3,3:f32:6912B]`：卷积权重，shape `[out=64, in=3, kh=3, kw=3]`
  （OIHW 布局），float32，6912 字节（= 1728 × 4）。
- 第二个 `[64:f32:256B]`：bias，shape `[64]`，float32，256 字节（= 64 × 4）。

顺序 weight → bias 严格对应 `convolution.cpp` `load_model` 里的读取次序。
kernel 读取 4 字节类型 flag，`load_weight()` 支持 f32/f16/i8；bias 和 scale
固定按无 flag 的 f32 读取。端到端数值验证路径当前以 f32 为主，但参数解析与
权重加载已能消费 f16/i8 张量。

## 6. 完整示例：SqueezeNet v1.1

75 层的完整 `parsed-graph` 展开可直接跑上面的命令查看。结构要点：

- `[0]` `Input`：`{0=227 1=227 2=3}` = w=227, h=227, c=3。
- `[1..]` 主干由 `Convolution` + `ReLU` + `Pooling` 构成。
- `Split`（如 `[6]`）把一个 blob 复制成两份（`out=[...splitncnn_0,...splitncnn_1]`），
  喂给 fire 模块的 expand1x1 和 expand3x3 两条支路。
- `Concat`（如 `[11]`）把两条 expand 支路沿 channel 拼回。
- `[70]` `Dropout`：推理期恒等，无参数无权重。
- `[73]` `Pooling` `{0=1 ... 4=1}`：全局平均池化，把 `[1000,16,16]` 收成 `[1000]`。
- `[74]` `Softmax` `{0=0}`：沿 channel（axis 0）归一化，输出 `prob`。

数出来 26 个 `Convolution`、26 个 `ReLU`、8 个 `Split`、8 个 `Concat`、
4 个 `Pooling`、1 个 `Dropout`、1 个 `Softmax`、1 个 `Input`，正好覆盖
SqueezeNet v1.1 使用的 8 种算子。

## 7. 与 `ncnn-ir` 的区别

| 维度 | `parsed-graph` | `ncnn-ir` |
|------|---------------|-----------|
| 抽象层次 | ncnn 原始模型的忠实镜像 | 类型化的有向图 IR（SSA-ish） |
| 参数 | 原始 `key=value` 字典 | 解析成具名属性（`kernel`/`stride`/…） |
| 权重 | 按层绑定的张量 | 独立的 `const` 算子，带 `TensorType` |
| shape | 主要保留 Input 参数和权重 shape；不提供中间值类型推断 | 每个 value 都有推导出的 shape/dtype，布局由维序约定 |
| 用途 | 调试 / 交叉验证解析器 | 下游 lowering（→ tosa → linalg → …）的输入 |

排查问题的顺序：先 `parsed-graph` 确认解析对，再 `ncnn-ir` 确认导入对。

## 8. 相关文档

- [ncnn-mlir-driver-usage.md](ncnn-mlir-driver-usage.md) — 驱动的完整命令行接口。
- [../../docs/netron-ncnn-parsing.md](../../docs/netron-ncnn-parsing.md) — ncnn
  `.param`/`.bin` 格式与 `-23300` 数组编码的权威解析原理。
- [../../docs/ncnn-mlir-compiler-plan.md](../../docs/ncnn-mlir-compiler-plan.md) —
  整体架构与项目计划。
