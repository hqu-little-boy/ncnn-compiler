# ncnn 性能差距归因分析（2026-09-03）

本文是 [`ncnn-performance-baseline-report.md`](ncnn-performance-baseline-report.md) 的姊妹篇。
后者回答"差距多大"，本文回答"为什么慢"。数据来源：2026-09-03 全量 44 模型
Release 实测（`NCNN_PERF_JSON`）、同日 6 代表模型单线程分解实验、
四个代表 fixture 的 objdump 指令审计与代码路径核对。

## 1. 结论速览

1. **差距全部来自单核内核效率，不来自并行**。单线程分解实验（2 warmup +
   5 timed，`NCNN_PERF_THREADS=1`）中所有模型 ratio 都比 6 线程更差：
   编译产物 6T/1T 加速 2.9–5.5×，全面优于 vendored ncnn 的 1.5–4.2×。
   多线程目前在掩盖内核差距，而非制造差距。
2. **向量算术从未合成 FMA**（全产物 0 条 `vfmadd*`），每个 MAC 以
   `vmulps`+`vaddps` 两条指令完成：FP 端口占用 ×2、K 循环累加依赖链延迟
   ×2，微内核上限直接砍半以上。
3. **数据搬运指令压倒算术指令**（resnet18 ≈19:1、formula encoder ≈80:1）：
   im2col 物化拷贝、逐 op 布局转换、部分 lane 标量脚手架（`vmovss`/
   `vinsertps`）是代码主体。
4. **int8 路径是独立的最差分支**：量化卷积不经过 strategy/matmul-kernel/
   向量化任何 pass，落到标量 i32-MAC 循环 + 逐层激活量化/反量化全量
   elementwise pass，产物无任何 VNNI。
5. **最差的 attention 系模型其实劣在卷积**：formula encoder 无任何
   attention（53 Conv + 27 DepthWiseConv），因"编译爆炸名单"被整体关闭
   MLIR 显式向量化，纯卷积就打出 42.9×（1T 83.6×）。

## 2. 证据

### 2.1 单线程分解（内核质量 vs 并行效率）

ratio（compiled/ncnn）；右侧两列为 6T 时间 ÷ 1T 时间的倒数（即加速比）。

| 模型 | 1T ratio | 6T ratio | ncnn 加速 | compiled 加速 |
|---|---:|---:|---:|---:|
| resnet18 | 6.30 | 4.54 | 2.2× | 3.0× |
| yolov5s | 3.88 | 2.70 | 1.9× | 2.8× |
| pp_ocrv5_server_rec | 42.82 | 31.16 | 4.2× | 5.5× |
| pp_ocrv6_medium_rec | 7.04 | 4.67 | 2.6× | 3.9× |
| pp_ocrv6_medium_rec_int8 | 54.11 | 29.84 | 2.2× | 4.1× |
| pp_ocrv6_small_rec_int8 | 21.55 | 11.36 | 1.5× | 2.9× |
| pp_formulanet_plus_s_encoder | 83.58 | 42.93 | 2.3× | 4.6× |

以标准 FLOPs 折算单核算力：resnet18 ncnn ≈64 GFLOP/s vs 产物 ≈10
（6.3× 缺口）；yolov5s ≈92 vs ≈24（3.9×）。与 A1b 隔离基准测得的
matmul 内核 18–26 GFLOP/s 相互印证。

### 2.2 产物指令审计（objdump，`--no-show-raw-insn` 全量统计）

| fixture | 总指令 | 向量 FMA | VNNI | 主要构成 |
|---|---:|---:|---:|---|
| resnet18（8-lane 显式向量化） | 45.8k | **0** | 0 | vmovaps/vmovups/vmovss ≈24k；vaddps 809 + vmulps 444 |
| pp_formulanet_plus_s_encoder（scalar MLIR 模式） | 221.6k | **0** | 0 | 向量"指令"10.4 万条里几乎全是搬运；vmulps 仅 564 |
| pp_ocrv6_medium_rec_int8 | 171.4k | **0** | **0** | vmulps 5434（多为反量化乘法）、vdivps 3684（GELU 类）、vinsertps 2772 |

### 2.3 代码路径核对

- `MatmulKernelNCNN.cpp` A1b 内核：`arith::MulFOp`+`arith::AddFOp` 不带
  fastmath；全工具链（tools/ncnn-compile.cpp、lib/Pipelines）无任何
  fastmath/contract 处理 → LLVM `fmul`+`fadd` 无 contract 标志，-O3 也
  不会合成 FMA。
- A1b 内核形态为"每行 C 读入寄存器 + K 内层 broadcast(A)·B 行累加"：
  M 方向无寄存器分块，B 按行反复重读，算术强度 ~0.5 flop/byte，远低于
  本机机器平衡点，内存带宽受限。
- `StrategyNCNN.cpp` 仅改写浮点 `linalg::Conv2DNhwcHwcfOp`；量化
  （i8×i8→i32）卷积与 depthwise 均不进 im2col+GEMM，也不在
  `MatmulKernelNCNN`/`VectorizeNCNN` 覆盖范围内（grep 零命中）。
- int8 下降（`NCNNToTosa.cpp`）：每层卷积前 `quantizeSignedI8` 对激活做
  全量 linalg generic 量化，卷积后 `dequantizeAccumulator`——比 fp32 路径
  多出两个全张量 pass，且卷积本体是标量整数循环。
- ncnn 侧同条件运行 AVX2/AVX512 dispatch 手写内核：Winograd（3×3 s1）、
  NC4HW4 打包 + 寄存器分块 sgemm、int8 avxvnni `vpdpbusd`，激活函数
  融合进卷积出口。

### 2.4 编译爆炸名单与最差 ratio 重合

`test/Numerical/CMakeLists.txt` 的 `ncnn_model_no_vector_overrides`
（11 个 PP rec/SLANet/Formula 模型因 clang -O3 合法化编译爆炸而关闭
MLIR 显式行向量化）与 ratio 表 6×–43× 的全部 11 行一一重合；名单外
模型全部落在 2.2–5.0×。显式向量化的有无是当前模型间最大分层因子。

## 3. 根因清单（按影响排序）

| # | 根因 | 影响面 | 指向修复 |
|---|---|---|---|
| 1 | ~~无 fastmath/contract → 全产物零 FMA~~ **已修复（2026-09-03 P1）**：A1b 直发 `vector.fma`，全表 p50 4.00→3.35、44/44 改善（见 parity-plan 执行状态） | 全部模型 | 已落地；VectorizeNCNN/FuseLinalgEpilogue 经核实不自产累加 op，无 contract 注入落点 |
| 2 | 11 模型关闭显式向量化（编译爆炸） | 全部 rec/attention 系（6×–43×） | 已通过 lanes 分块 + 标量尾解决超宽向量合法化爆炸；vector.contract/tile 化路线经 spike 否决，当前不作为默认路径 |
| 3 | ~~int8 无 VNNI、标量 i32-MAC、逐层量化/反量化 pass~~ **已修复（2026-09-06 P4）**：Q-conv 接入 strategy（im2col+`matmul_transpose_b`，权重常量编译期重排 [N,K]）、i8 row-dot 内核（标量 MAC 多链，LV 生成 `vpmovsxbw`+`vpmaddwd`/`vpmulld`——本机无 `avx_vnni_int8`，s8×s8 无 byte-VNNI 自动下降，`vpdpbusd` 需 intrinsics 直发不符合 ISA 口径）、`fuse-quant-chain-ncnn` 将逐层 requant/dequant 链融合为单一 generic、行级向量化扩展整数/混合类型。int8 行 ratio 19.9×–28.8× → 1.9×–4.8×（medium_rec_int8 19.94→4.82） | 全部 int8 行 | 已落地；K<8 标量尾、动态空间维与 dw-Q 未覆盖（见 parity-plan §3-P4 附注） |
| 4 | ~~matmul 微内核无 M 维寄存器分块，B 反复重读~~ **已修复（2026-09-06 P3）**：A1b M×N 寄存器分块（4×16，K 循环 4 条独立 FMA 链），隔离内核 3.06×/3.48×（117.7/60.6 GFLOP/s）；端到端受非 GEMM 开销占比限制（resnet18 1T 1.61×，GEMM 分块后仅占 ~25%） | 全部走 GEMM 的层 | 已落地（选项 `--matmul-m-rows`/`--matmul-acc-columns`）；无打包深 K 上限 ~56 GFLOP/s，≥ 60 复评联动 P6 打包 |
| 5 | 无默认 Winograd；3×3 s1 主路径走 im2col+GEMM 或直接卷积 | conv 主导模型（resnet/yolo/det 系 2–5×） | `--conv-strategy=winograd` 已落地且数值 golden 通过；性能门禁未达，继续 opt-in/default-off，后续 pack 化为 audit-only |
| 6 | ~~depthwise conv 纯标量 generic~~ **已修复（2026-09-06 P5）**：`vectorizeDepthwiseConvRows` 按 C 行 rank-1 transfer + vector.fma 改写 multiplier=1 形态，全表 p50 3.42→2.84、43/44 改善，det 系 1.06–1.55×（见 parity-plan 执行状态） | 含 depthwise 的全部模型（27–28 层/attention 系） | 已落地；multiplier≠1 保持 P8 audit-only/no-go，需专门布局与数值证明后再评估 |
| 7 | 数据搬运占比过高（im2col 物化、逐 op 布局转换、部分 lane 脚手架） | 全部模型 | im2col 融合进 matmul 主循环（gather-free 化）；MHA/attention 的非常量 transpose 运行时路径检查 |

注：#1 是全局一行级修复、预期全表收益；#2 是当前最大单项；#3 独立成线。
修复路线的阶段化展开见
[`ncnn-performance-parity-plan.md`](ncnn-performance-parity-plan.md)。

## 4. 复现

```bash
# 6T 正式口径
NCNN_PERF_JSON=/tmp/perf6.json ctest --test-dir build -L performance
# 1T 分解（重模型务必降迭代，否则 server_rec 单用例 ~10 分钟）
NCNN_PERF_THREADS=1 NCNN_PERF_SKIP_SANITY=1 NCNN_PERF_WARMUP=2 \
NCNN_PERF_ITERS=5 NCNN_PERF_JSON=/tmp/perf1.json \
  ./build/test/Numerical/performance_tests --gtest_filter='<...>'
# 指令审计
objdump -d --no-show-raw-insn build/test/Numerical/generated/<m>/lib<m>.so
```

1T 实验注意：gtest 按**注册序**执行 filter 内用例（server_rec #354 先于
medium_rec #355），估算剩余时长时勿按 filter 顺序推算。
