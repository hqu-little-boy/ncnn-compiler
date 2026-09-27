# `ncnn-compile` CLI 选项对照表

> **用途**：查「这个选项归哪个 pass 管、默认什么、取值范围」的**索引**。
> 语义细节、示例与产物布局看同目录的散文参考
> [`ncnn-compile-command-line.md`](ncnn-compile-command-line.md)。
>
> **性质**：选项名是**公开契约**，不改名（见 roadmap §8）。
> 本表按代码实测生成，选项全集 = `tools/ncnn-compile.cpp` 的全部
> `llvm::cl::opt` / `llvm::cl::list`。
>
> 更新日期：2026-09-27（P5 / T-M3）

---

## 怎么读这张表

* **所属 pass**：该选项最终落到哪个 pass / pipeline。标 *ncnn-compile* 的是
  driver 自身行为（工具链、产物、诊断），不进任何 pass。
* **默认**：`cl::init` 的值；空表示默认靠上下文推导。
* 选项影响产物字节（因而进 identity 串）时在备注里标 ⚠️。
  `--warnings-as-errors` **不**进 identity 串——它只改诊断，不改产物。

---

## 1. 输入与输出

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `<input .param>` | 路径（位置参数） | — | ncnn-compile | ncnn 模型结构文件 |
| `--param` | 路径 | 同位置参数 | ncnn-compile | 显式指定 `.param` |
| `--bin` | 路径 | `<input>.bin` | ncnn-compile | 权重文件 |
| `--model-name` | 标识符 | 取输入文件名 | ncnn-compile | 导出函数名，进 C ABI |
| `--input-shape` | `CxHxW`，`?` 为动态 | 从模型推导 | ncnn-compile | 可重复；动态 extent 用 `?` |
| `--input-dim-constraint` | 约束表达式 | — | ncnn-compile | 可重复；动态维约束 |
| `--output-dir` | 路径 | `<model>/` | ncnn-compile | 目录须只含本工具产物 |
| `--emit` | `ncnn`/`tosa`/`linalg`/`memref`/`capi`/`llvm`/`llvm-ir`/`object`/`assembly`/`all` | — | ncnn-compile | 可重复、逗号分隔；保留中间产物 |
| `--emit-manifest` | flag | off | GenerateCAPI | 出 `.json` ABI manifest |
| `--emit-execution-plan` | flag | off | EmitModelPlan | 出 `.plan.json`；`--profile` 隐含开启 |

## 2. 精度与累加

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--precision` | `auto`/`f32`/`fp16`/`bf16`/`int8` | `auto` | 精度策略（`Support/Precision`） | ⚠️ 影响产物 |
| `--fp16-accumulator` | `f16`/`f32` | `f16` | 同上 | FP16 卷积累加器类型 |
| `--allow-fallback` | flag | off | 同上 | 允许不支持的 FP16 运算回退 FP32 累加 |

## 3. 目标与工具链

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--target-triple` | 三元组 | 取 clang `-dumpmachine` | ncnn-compile | 仅 64-bit Linux ELF |
| `--march` / `--mcpu` / `--mtune` | 目标标识 | 空 | ncnn-compile | 直传 clang |
| `--target-feature` | 特性串 | — | ncnn-compile | 可重复；⚠️ 进 identity |
| `--sysroot` | 路径 | 空 | ncnn-compile | 直传 clang |
| `-O` | `0`/`1`/`2`/`3` | `3` | ncnn-compile | 传给 clang 代码生成 |
| `-g` | flag | off | ncnn-compile | 调试信息 |
| `-v` | flag | off | ncnn-compile | 打印执行的命令 |
| `--clang-arg` | 任意 | — | ncnn-compile | 可重复；⚠️ 进 identity |
| `--linker-arg` | 任意 | — | ncnn-compile | 可重复；⚠️ 进 identity |

## 4. 并行与向量

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--threads` | 正整数 | `0`（取 OMP_NUM_THREADS） | SCF→OpenMP（`ConvertSCFToOpenMP`） | `1` = 串行路径 |
| `--vector-width` | 位宽 | `256` | `VectorizeNCNN` / `affine-super-vectorize` | `0` 禁用偏好 |
| `--vector-mode` | `off`/`avx2`/`avx512`/`auto` | `off` | `VectorizeNCNN` | 张量级行向量化的总开关 |

## 5. 卷积与矩阵乘策略

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--conv-strategy` | `auto`/`gemm`/`conv`/`winograd` | `auto` | `StrategyNCNN` | 分派卷积实现 |
| `--conv-gemm-l2-bytes` | 字节数 | `524288` | `StrategyNCNN` | GEMM 分块的 L2 预算 |
| `--packed-conv-depthwise` | flag | **off** | `StrategyNCNN` | P21 opt-in，f32 packed Conv/Depthwise 布局岛 |
| `--matmul-packing` | `off`/`auto` | `auto` | `PackStaticMatmulNCNN` | 静态 f32 RHS 打包；**96 MiB 预算** |
| `--matmul-m-rows` | 整数 | `0`（取 tuning profile） | `MatmulKernelNCNN` | M 方向寄存器 tile 行数 |
| `--matmul-acc-columns` | 整数 | `0`（取 tuning profile） | `MatmulKernelNCNN` | 累加器列数 |
| `--row-chunk-lanes` | 整数 | `0`（取 tuning profile） | `MatmulKernelNCNN` | 行块 lane 数 |

## 6. INT8

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--int8-kernel` | `portable`/`vnni`/`avx512` | `portable` | `MatmulKernelNCNN` | INT8 MAC 内核形态 |
| `--int8-depthwise` | flag | off | `MatmulKernelNCNN` | 静态 INT8 depthwise SIMD |
| `--int8-cast-chain` | flag | off | `FuseQuantChainNCNN` | 融合已证的静态 cast map 消费者 |
| `--matmul-i8-rows` | 整数 | `0`（取 tuning profile） | `MatmulKernelNCNN` | INT8 M 方向 tile 行 |
| `--matmul-i8-acc-columns` | 整数 | `0`（取 tuning profile） | `MatmulKernelNCNN` | INT8 累加器列 |

## 7. 跨算子融合

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--selective-fusion` | flag | **on** | `FuseLinalgEpilogue` | 静态生产者→epilogue 融合总开关 |
| `--layout-aware-fusion` | flag | **on** | `FuseLinalgEpilogue` | 要求生产者/消费者布局相容 |
| `--selective-fusion-broadcast` | flag | **on** | `FuseLinalgEpilogue` | 允许静态证投影广播输入 |
| `--selective-fusion-cast-chain` | flag | **on** | `FuseLinalgEpilogue` | 允许有序算术 cast 链进 epilogue |
| `--selective-fusion-max-chain` | 正整数 | `8` | `FuseLinalgEpilogue` | 一次融合的逐元素算子数上限 |

## 8. 向量数学库

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--vector-math` | `auto`/`libm`/`sleef`/`off` | `auto` | `LowerVectorMathNCNN` | 超越函数的向量实现 |
| `--sleef-path` | 目录 | 空 | ncnn-compile | vendored SLEEF 静态档案目录 |

## 9. 调优与诊断

| 选项 | 取值 | 默认 | 所属 pass | 备注 |
|---|---|---|---|---|
| `--tuning-profile` | `stable`/`p16-int8`/`native-int8` | `stable` | 调优层（`StrategyNCNN`/`MatmulKernelNCNN` 读） | ⚠️ 进 identity |
| `--profile` | flag | off | `InstrumentNCNNProfile`（三个站点 pass） | 诊断 profile，隐含 `--emit-execution-plan` |
| `--warnings-as-errors` | flag | off | `Diagnostics`（`Support/Diagnostics.hpp`） | **不进 identity**：把 ncnn 警告升为 error 并让 pass 失败 |
| `--verify-execution` | flag | off | ncnn-compile | 产物自检 |

`--warnings-as-errors` 的下沉方式：`ncnn-compile` 自己不跑 pass，它 exec
`ncnn-mlir-opt` 子进程，所以开关经环境变量 `NCNN_WARNINGS_AS_ERRORS`
传给子进程。也可直接对 `ncnn-mlir-opt` 设该变量，效果一致。

## 10. 工具路径覆盖（调试用）

| 选项 | 默认 | 备注 |
|---|---|---|
| `--driver` | `ncnn-mlir-driver` | 导入器 |
| `--opt` | `ncnn-mlir-opt` | pass runner |
| `--translate` | `mlir-translate-21` | MLIR→LLVM IR |
| `--clang` | `clang-21` | 代码生成与链接 |
| `--nm` / `--readelf` / `--llvm-as` | 对应 `-21` 工具 | 产物审计 |
| `--expected-undefined` | 空 | 未定义符号白名单扩展 |

---

## 与散文参考的分工

| 需求 | 看哪里 |
|---|---|
| 这个选项归谁管、默认值 | **本文件** |
| 选项的语义、示例命令、产物长什么样 | [`ncnn-compile-command-line.md`](ncnn-compile-command-line.md) |
| 支持/不支持的算子与模型 | [`ncnn-compile-support-status.md`](ncnn-compile-support-status.md) |
| 新增选项时的纪律 | [`code-quality-refactor-roadmap.md`](code-quality-refactor-roadmap.md) §8（不改公开契约）+ 本文件补一行 |
