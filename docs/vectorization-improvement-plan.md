# 向量化路径改进计划（v3，架构通用版）

> **执行状态（T0+A2 已落地）**：`TargetVectorInfo`（triple/march→Fixed/Scalable 推导）、
> `vectorize-ncnn` 行级向量化 pass、`lower-vector-transfers-ncnn` 规范化 pass、
> `--vector-mode` CLI、epilogue tile 参数化已实现并通过全量门禁
> （387/387 ctest；SqueezeNet 向量模式 bit-exact 且 ~5× 加速；
> NEON/SVE/RVV 静态 asm 验证出现 SIMD 指令）。卷积/matmul 接入与 OpenMP 组合
> 属 A1/A3 后续阶段。

> v3 修订：向量化目标从"x86 性能追赶"改为**面向 LLVM 全部后端架构的通用路径**
> （x86-64 SSE/AVX/AVX512、AArch64 NEON/SVE、RISC-V RV64V）。v2 已落地的四个
> 优化提交不变（常量折叠、INT8 预量化、BN 折叠、epilogue 融合），本版重构剩余
> 路线的目标抽象与验证矩阵。参考基线仍为 vendored ncnn x86 实现。

---

## 1. 现状中的架构偏置（必须消除）

| # | 偏置点 | 位置 | 问题 |
|---|---|---|---|
| 1 | `kConvTileWidth/kMatmulTileColumns = 16` 硬编码 | FuseLinalgEpilogue.cpp:26-27 | 按 AVX512 f32 lane 数设定；NEON(4)/RVV(VL 可变) 下只是巧合可用，无法表达意图 |
| 2 | `--vector-width` 默认 256bit 且仅喂给 clang `-mprefer-vector-width` | ncnn-compile.cpp:115-119、1921-1923 | `-mprefer-vector-width` 是 x86 专属概念；对 AArch64/RISC-V 无意义 |
| 3 | Affine Super Vectorizer 仅生成固定宽度 1-D 向量 | NCNNPipelines.cpp:103-109 | 无法表达 SVE/RVV 的运行时可变 VL，天然不可移植（删除它的又一理由） |
| 4 | SIMD 判定依赖"产物跑在宿主" | 测试/验收 | 项目已支持 AArch64 FP16 与 RISC-V Zfh/Zvfh 的静态 asm 验证（support-status §5.1），向量化验收矩阵必须对齐该流程 |

## 2. 架构通用性设计原则

1. **单一参数源 `TargetVectorInfo`**：由 `--target-triple/--march/--mcpu/
   --target-feature` 解析出每个浮点类型的向量描述：
   `{ Fixed(lanes) | Scalable(minLanes) }` × 对齐宽度。
   - x86：SSE=4 / AVX2=8 / AVX512=16（按 feature 推导，可被 `--vector-width` 覆盖）；
   - AArch64：NEON=4（fixed）；`+sve` 时 Scalable(≥4)，`+sme` 暂不启用；
   - RISC-V：`v` 扩展时 Scalable(≥4，LMUL 由 mcpu/features 决定)；无 v 则标量。
   实现放 `lib/Support/TargetVectorInfo.{hpp,cpp}`，CLI 显式覆盖优先于推导。
2. **固定宽度与可伸缩双模**：
   - Fixed 目标：`vector<8xf32>` 等，走完整 VV→VV 清理链；
   - Scalable 目标（SVE/RVV）：`vector<[4]xf32>`。上游 `vectorize()` 已带
     `inputScalableVecDims` 形参（`Linalg/Transforms/Transforms.h:879-884`），
     `vector.contract`/transfer/mask 及 VectorToLLVM 均支持 scalable；
     尾部用 runtime mask（`lower-vector-mask` 对两种模式通用）；
   - Spike 必查项：unroll/full-partial/drop-unit-dims 等 VV→VV pattern 对
     scalable 的覆盖度不足时按 op 类型降级（contract+elementwise 先行，
     其余留给 LLVM 循环向量化兜底），不得阻塞主路径。
3. **tile 与 lane 解耦**：`kConvTileWidth/kMatmulTileColumns` 从 TargetVectorInfo
   推导（Fixed: lanes×2 夹在 [8,32]；Scalable: 固定 16 作为列分块即可，与 VL
   正交），并暴露 `--conv-tile-width` 覆盖。tile 只影响融合粒度与局部性，
   不承担 ISA 语义。
4. **代码生成层去 x86 化**：`-mprefer-vector-width` 仅在 triple 为 x86 且用户
   显式设置时传递；SVE 目标不传 `-msve-vector-bits`（保持 scalable）；RVV 依赖
   `-march` 透传 vscale_range。MLIR 级不再有任何 ISA 名出现。
5. **数值契约跨架构一致**：不同 lane 数/VL 本就改变累加顺序（ncnn 各架构内核
   同样如此），既有逐模型容差预算已吸收该差异；禁止 fast-math/reassociate 以外
   的捷径。每阶段验收跑全部三目标的 golden 数值回归。
6. **验证矩阵（每阶段强制）**：
   - lit：同一输入 IR 分别断言 Fixed 模式（CHECK `vector<8xf32>` 类）与
     Scalable 模式（CHECK `vector<[4]xf32>` 类）；
   - 交叉产物静态验证：x86-64(native objdump)、AArch64(SDOUT/SVE 指令抽查)、
     RISC-V(vsetvl/vf 指令抽查)，复用 support-status §5.1 的静态验证管线；
   - 数值：native 全量 ctest；交叉目标以现有 static-asm 流程 + 抽样模型
     qemu/user-mode 执行对比（如环境允许）。

## 3. 阶段规划

### Phase T0：TargetVectorInfo 基础设施（1-1.5 天）

- 新增 `lib/Support/TargetVectorInfo`，解析 triple/features → 双模描述；
- FuseLinalgEpilogue 的 tile 常量改由其推导（默认行为不变，先消偏置 #1）；
- 三目标冒烟：现有一个小模型分别以
  `x86-64 / aarch64+neon / aarch64+sve / riscv64+v` 编译到 `.so`/`.o` 并通过
  现有符号审计。
- 验收：lit 覆盖 feature→lanes 推导表；冒烟产物链接审计绿。

### Phase A1：算子形态策略层（3-5 天，与 v2 相同 + 架构项）

- 1x1 s1 → matmul；通用 k×k → im2col+matmul；Winograd 留接口默认 off；
- **架构项**：im2col 阈值里的 L2/通道启发式不假设 x86 缓存层次，默认取保守
  常数 + `TargetVectorInfo` 提供的寄存器宽度参与 tile 选择；CLI 可覆盖。
- 验收同 v2，外加：三目标交叉编译均通过、matmul 形态在 SVE/RVV 目标同样成立。

### Phase A2：串行 Linalg 向量化（4-6 天）

插入点不变（bufferize 前 tensor 层），但按 §2-2 实现双模：

1. contraction 路径（conv/matmul → `vector.contract`）：Fixed 与 Scalable
   都优先落地——这是三架构收益最大、pattern 最成熟的部分；
2. elementwise/reduction `vectorize()`：Fixed 全量；Scalable 按 spike 结论
   启用子集；
3. 清理链按模式裁剪（scalable 跳过依赖固定 shape 的 unroll 类 pattern）；
4. MemRef→LLVM 条件修正（含 vector op 即追加 ConvertVectorToLLVM 序列）；
5. 删除 affine super-vectorizer 路径（偏置 #3 随之消除）。

验收：
1. lit 双模式 CHECK（见 §2-6）；
2. native 数值全绿；serial 相对标量基线 ≥1.5×（x86 AVX2 配置）；
3. AArch64+SVE / RV64V 产物静态 asm 验证出现向量指令，且 `vector.contract`
   未退化为标量循环；
4. 无 v 扩展的 RISC-V / SSE-only x86 自动回退标量或窄向量，编译不断失败。

### Phase A3：OpenMP + SIMD 组合（5-8 天，含 spike 1 天）

- FuseLinalgEpilogue W-tile 循环 → `scf.forall(shared_outs)` 模式 +
  内层双模向量化；通用 op 走 `scf::tileUsingForall`；
- spike 不变：LLVM 21 `convert-scf-to-openmp` 对 forall 支持；fallback
  forall→scf.parallel；
- **架构项**：各目标 sysroot 的 libomp 可得性审计（x86/AArch64 通常具备；
  RISC-V 裸 sysroot 可能缺失）→ 缺失时自动等价 `--threads=1` 并在 manifest
  标注；OpenMP 符号白名单审计按目标条件化。

验收：x86/AArch64 多核近线性 + SIMD 共存；RISC-V 无 OpenMP 时优雅回退；
数值回归全绿。

### Phase A4：固化（2-3 天）

- CLI 收敛：`--vector-width` 变为目标无关的"lane 数覆盖"（bits 语义废弃，
  迁移期兼容）；新增 `--vector-mode={auto,fixed-width,scalable,off}`、
  `--conv-tile-width`；
- 文档：support-status §6 补登 epilogue 融合/strategy/双模向量化与各目标
  默认值表（triple×feature→模式）；本文档沉淀三目标 perf 表；
- 删除死路径。

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| VV→VV pattern 对 scalable 支持不全（§2-2） | A2 前置 spike；按 op 子集启用，剩余交 LLVM 循环向量化；绝不阻塞 contraction 主线 |
| vscale_range 未随 `-march` 正确进入 LLVM IR | T0 冒烟即检查 IR 属性；必要时显式添加 |
| 各目标 libomp/编译器_rt 差异 | A3 sysroot 审计 + 自动 threads=1 回退 |
| NEON/RVV 窄寄存器上 tile=16 过大导致寄存器溢出 | tile 已解耦并可 CLI 调；A4 给出 per-target 默认表 |
| FP16/BF16 边界路径与向量化交互 | 低精度 boundary 逻辑位于 NCNN 层，向量化在其下游，理论正交；数值回归覆盖 int8/FP16-storage 产物 |
| 陈旧 fixture 假绿/假红 | 每次 ctest 前重建 target（suspected-issues 强制要求） |

## 5. 度量记录模板（三目标）

| 模型 | 目标/配置 | wall (ms) | SIMD% (perf/native) | asm 抽查 | 数值最大误差 |
|---|---|---|---|---|---|
| | x86-64 AVX2 scalar 基线 | | | | |
| | x86-64 +A2 fixed | | | | |
| | aarch64 neon / sve +A2 | | n/a（静态） | SDOT/FMLA… | |
| | riscv64 v +A2 scalable | | n/a（静态） | vsetvl/vfmacc… | |
| | 各目标 +A3 openmp | | | | |
