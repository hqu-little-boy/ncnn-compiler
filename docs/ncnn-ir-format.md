# ncnn-ir 文本格式说明

> 对应源码：`include/ncnn_frontend/`（Types.hpp、OperationKind.hpp、Operations.hpp、Ops/*.hpp）；
> `src/frontend/`（ir.cpp、ir_dump.cpp、Types.cpp、Ops/*.cpp）
> 生成工具：`ncnn-mlir-driver --emit=ncnn-ir`（默认阶段）
> 相关文档：[parsed-graph-format.md](parsed-graph-format.md)、[ncnn-mlir-driver-usage.md](ncnn-mlir-driver-usage.md)

`ncnn-ir` 是编译器**类型化前端 IR** 的文本转储。它由 `ncnn_frontend::import_graph()` 从原始
[parsed-graph](parsed-graph-format.md) 提升而来，是下游 lowering（ncnn → tosa → linalg → …）真正消费的输入。

---

## 1. 这是什么阶段

流水线中的位置：

```
.param/.bin
  ──ncnn_graph::Graph::load──▶  parsed-graph      (原始层列表，1:1 镜像 ncnn 文件)
  ──ncnn_frontend::import_graph──▶  ncnn-ir       ← 本文档
  ──（后续）convert-ncnn-to-tosa──▶  tosa …
```

与 parsed-graph 的关键区别：**parsed-graph 是层的线性列表，ncnn-ir 是类型化的有向无环图（DAG）**。
`import_graph` 做了这些提升：

- **blob 名 → SSA 值**：每条 ncnn blob 变成一个带类型的 `Value`（`v0`、`v1`…），有唯一定义点和显式使用点（use）列表。
- **权重 → 常量算子**：conv 的 weight/bias 从"挂在层上的张量"变成独立的 `const` 算子，产出各自的值。
- **参数字典 → 强类型属性**：ncnn 的 `{0=64 1=3 …}` 数字 key 解析成 `conv2d{kernel=[3,3],stride=[2,2],pad=[…]}` 这样的具名属性。
- **shape inference**：每个值都带推导出的静态 shape、元素类型、布局。
- **激活融合展开**：ncnn 把 activation 折进 conv 的写法被拆成显式的 relu 算子（SqueezeNet 里本就是独立 ReLU 层）。

因此 ncnn-ir 比 parsed-graph 多出**类型、SSA 连接、use-def 关系**，是"编译器视角"的表示。

---

## 2. 生成方式

```bash
cd /mnt/ncnn-compiler/compiler

# 默认就是 ncnn-ir（--emit 可省略）
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param

# 显式写法 + 输出到文件
./build-make/ncnn-mlir-driver --emit=ncnn-ir \
  ../ncnn/examples/squeezenet_v1.1.param -o squeezenet.ncnn-ir
```

`.bin` 权重路径默认由 `.param` 推导（`.param` → `.bin`），也可用 `--bin=<path>` 显式指定。

---

## 3. 整体结构

转储是纯文本，分四段，顺序固定：

```
ncnn_frontend.typed_dag_dump version=1     ← 头（格式版本）
operations 126                             ← 算子段：先声明数量，再逐行
op 0 {...}
op 1 {...}
...
values 130                                 ← 值段：先声明数量，再逐行
value 0 {...}
value 1 {...}
...
inputs [v0]                                ← 图输入（值 id 列表）
outputs [v129]                             ← 图输出（值 id 列表）
```

> 头里的 `version=1` 是格式版本号，供解析器做兼容判断。

---

## 4. 算子行（`op`）

每个算子一行，格式：

```
op <序号> {<属性>,name=<名>,source_layer=<原层号>,operands=<值列表>,results=<值列表>}
```

真实样例（SqueezeNet 前四个算子）：

```
op 0 {kind=const,attrs={literal_type={shape=[64,3,3,3],element=f32,layout=oihw,elements=1728,bytes=6912},payload_bytes=6912,fnv1a64=0x3c505b732fd566d8},name="conv1.weight",source_layer=1,operands=[],results=[v1]}
op 1 {kind=const,attrs={literal_type={shape=[64],element=f32,layout=ncnn_w,elements=64,bytes=256},payload_bytes=256,fnv1a64=0xd30558885e110c61},name="conv1.bias",source_layer=1,operands=[],results=[v2]}
op 2 {kind=conv2d,attrs={kernel=[3,3],stride=[2,2],dilation=[1,1],pad=[0,0,0,0],has_bias=true,int8_scale_term=0,quantization=none},name="conv1",source_layer=1,operands=[v0,v1,v2],results=[v3]}
op 3 {kind=relu,attrs={negative_slope=0},name="relu_conv1",source_layer=2,operands=[v3],results=[v4]}
```

字段含义：

| 字段 | 说明 |
|------|------|
| `op <序号>` | 算子在算子段中的下标（从 0） |
| `kind=…` | 算子种类，见下表 |
| `attrs={…}` | 该 kind 特有的强类型属性 |
| `name="…"` | 算子名（转义字符串，来源见下）。const 算子名形如 `conv1.weight` / `conv1.bias`；计算算子沿用 ncnn 层名 |
| `source_layer=<n>` | 该算子来自 parsed-graph 的第几层，便于回溯定位 |
| `operands=[…]` | 输入值 id 列表，按顺序 |
| `results=[…]` | 输出值 id 列表，按顺序 |

> `operands` 里出现的 `v0,v1,v2` 依次是 conv 的**输入特征图、weight 常量、bias 常量**——权重被提升为常量算子后，作为普通操作数接进 conv。这是 ncnn-ir 相对 parsed-graph 最直观的结构变化。
>
> 值 id 越界时会带 `!out_of_range` 后缀（如 `v999!out_of_range`），正常转储不应出现。

---

## 5. 算子种类与属性

`kind` 取值对应 `OperationKind` 枚举，SqueezeNet 用到 8 种：

| kind | 属性（`attrs={...}`） | 来源 ncnn 层 |
|------|----------------------|-------------|
| `const` | `literal_type={...},payload_bytes=N,fnv1a64=0x…` | 权重（conv 的 weight/bias） |
| `conv2d` | `kernel=[h,w],stride=[h,w],dilation=[h,w],pad=[top,bottom,left,right],has_bias=<bool>,int8_scale_term=N,quantization=<mode>` | Convolution |
| `relu` | `negative_slope=<float>`（0 = 标准 ReLU，非 0 = LeakyReLU） | ReLU |
| `pool2d` | `kind=<max\|average>,mode=<regular\|global\|adaptive>,kernel=[h,w],stride=[h,w],pad=[t,b,l,r],pad_mode=N,include_pad=<bool>` | Pooling |
| `split` | `{}`（无属性；把一个值复制成多份） | Split |
| `concat` | `axis=<int>` | Concat |
| `dropout` | `scale=<float>`（推理期恒等/缩放） | Dropout |
| `softmax` | `axis=<int>` | Softmax |

### const 的属性细节

```
kind=const,attrs={literal_type={shape=[64,3,3,3],element=f32,layout=oihw,elements=1728,bytes=6912},payload_bytes=6912,fnv1a64=0x3c505b732fd566d8}
```

- `literal_type` 是这块常量的张量类型（见 §7 类型格式）。
- `payload_bytes` 是权重数据字节数。
- `fnv1a64` 是权重数据的 FNV-1a 64 位哈希——**转储里不展开原始字节**，用哈希代替，既能做交叉验证（同一权重哈希应一致）又不让转储爆炸。

> conv 的 weight 布局是 `oihw`（out/in/h/w），bias 布局是 `ncnn_w`（一维）。

### quantization 取值

`conv2d` 的 `quantization` 字段：`none` / `dequantize` / `requantize`，由 `int8_scale_term` 推导，Phase-1 全 float32 路径下恒为 `none`。

---

## 6. 值行（`value`）

每个 SSA 值一行，格式：

```
value <序号> {name=<名>,type=<类型>,def=<定义点>,uses=<使用点列表>}
```

真实样例：

```
value 0 {name="data",type={shape=[3,227,227],element=f32,layout=ncnn_chw,elements=154587,bytes=618348},def=graph_input(0),uses=[{user=op2,operand=0}]}
value 3 {name="conv1",type={shape=[64,113,113],element=f32,layout=ncnn_chw,...},def=op_result(op2,0),uses=[{user=op3,operand=0}]}
```

字段含义：

| 字段 | 说明 |
|------|------|
| `value <序号>` | 值下标（从 0），即 `v<序号>` 里的编号 |
| `name="…"` | 值名（转义字符串），通常沿用 ncnn blob 名 |
| `type={…}` | 该值的张量类型，见 §7 |
| `def=…` | 定义点：`graph_input(<i>)` 表示第 i 个图输入；`op_result(op<n>,<r>)` 表示第 n 个算子的第 r 个结果 |
| `uses=[…]` | 使用点列表，每项 `{user=op<n>,operand=<k>}` 表示"被第 n 个算子当作第 k 个操作数使用" |

`def` 与 `uses` 构成完整的 **use-def 链**：任一算子的 `operands` 里引用的值，都能在对应 `value` 的 `uses`
里找到反向记录（driver 的单元测试 `load_squeezenet_typed` 就断言了这种双向一致性）。

---

## 7. 张量类型格式

值和 const 字面量的类型都用同一格式：

```
{shape=[..],element=<e>,layout=<l>,elements=<n>,bytes=<b>}
```

| 子字段 | 说明 |
|--------|------|
| `shape=[..]` | 各维大小，逗号分隔 |
| `element` | 元素类型：`f32` / `f16` / `i8` |
| `layout` | 布局：`scalar` / `ncnn_w` / `ncnn_hw` / `ncnn_chw` / `ncnn_cdhw` / `oihw` |
| `elements` | 元素总数（各维乘积） |
| `bytes` | 字节大小（`elements × 元素字节`） |

> **布局说明**：当前前端 IR 保留 ncnn 原生布局——特征图是 `ncnn_chw`（C×H×W），conv 权重是 `oihw`，
> bias 是 `ncnn_w`。计划中的 NHWC 统一发生在后续 ncnn→tosa 边界（见 plan §3 布局策略），本阶段尚未转换。

---

## 8. 完整示例（SqueezeNet v1.1）

```bash
./build-make/ncnn-mlir-driver ../ncnn/examples/squeezenet_v1.1.param | head
```

输出头部：

```
ncnn_frontend.typed_dag_dump version=1
operations 126
op 0 {kind=const,attrs={literal_type={shape=[64,3,3,3],element=f32,layout=oihw,elements=1728,bytes=6912},payload_bytes=6912,fnv1a64=0x3c505b732fd566d8},name="conv1.weight",source_layer=1,operands=[],results=[v1]}
op 1 {kind=const,attrs={literal_type={shape=[64],element=f32,layout=ncnn_w,elements=64,bytes=256},payload_bytes=256,fnv1a64=0xd30558885e110c61},name="conv1.bias",source_layer=1,operands=[],results=[v2]}
op 2 {kind=conv2d,attrs={kernel=[3,3],stride=[2,2],dilation=[1,1],pad=[0,0,0,0],has_bias=true,int8_scale_term=0,quantization=none},name="conv1",source_layer=1,operands=[v0,v1,v2],results=[v3]}
op 3 {kind=relu,attrs={negative_slope=0},name="relu_conv1",source_layer=2,operands=[v3],results=[v4]}
```

统计（squeezenet_v1.1，实测）：

- **126 个算子** = 74 个计算算子 + 52 个 const（26 个 conv 各带 weight+bias）
- 计算算子分布：Convolution 26、ReLU 26、Split 8、Concat 8、Pooling 4、Softmax 1、Dropout 1、加图输入
- 图输入 1 个：`data`，`[3,227,227] f32 ncnn_chw`
- 图输出 1 个：`prob`，`[1000] f32`

---

## 9. 与其它阶段对比

| 维度 | parsed-graph | **ncnn-ir** | tosa（后续） |
|------|-------------|-------------|-------------|
| 结构 | 层的线性列表 | 类型化 SSA DAG | MLIR 方言 |
| 权重 | 挂在层上（`w=[...]`） | 独立 `const` 算子 | 常量/attr |
| 参数 | 原始数字 key（`0=64`） | 强类型属性（`kernel=[3,3]`） | tosa 属性 |
| 类型/shape | 无（仅权重 shape） | 每个值带完整类型 | tensor 类型 |
| use-def | 靠 blob 名隐式连接 | 显式 def + uses | SSA |
| 布局 | ncnn 原生 | ncnn 原生（chw/oihw） | NHWC |
| 用途 | 调试、交叉验证 | **下游 lowering 输入** | 接 MLIR 官方链 |

---

## 10. 相关源码

| 内容 | 文件 |
|------|------|
| IR 数据结构（Operation/Value/TensorType…） | `include/ncnn_frontend/ir.hpp` |
| parsed-graph → ncnn-ir 提升 | `src/frontend/importer.cpp` |
| shape inference / op schema | `src/frontend/op_schema.cpp` |
| 转储文本格式（本文档的权威定义） | `src/frontend/ir_dump.cpp` |
| 校验（`--verify`） | `src/frontend/verifier.cpp` |
| 驱动入口 | `tools/ncnn-mlir-driver.cpp` |
