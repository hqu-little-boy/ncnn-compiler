# ncnn 性能追平计划（v2）

> **v2 初版修订（2026-09-05，计划评审后；执行状态补记至 2026-09-13）**：
> ① §1 增补 **M3 降级口径**（P7 Winograd 预算否决命中主力重模型时的
>    退路），消除「≤1.1 依赖 Winograd + 预算一票否决」的内在张力；
> ② P2 范围扩至名单外慢件（efficientnet_b1/b2/b3 实测 1–2h/件），
>    spike 扩围定界 + 编译耗时门禁硬化 + 超 300s 件数收敛目标；
> ③ P3 验收首轮下调至 ≥45 GFLOP/s（无显式打包），P2/P3/P4 工期校准
>    （3–5/3–4/5–8 → 4–7/4–6/6–10），总预估 22–31 → 30–45 人天；
> ④ 验收三件套增补第四件：1T 抽测复核并行扩展——P1 实测校准内核收益
>    向 6T ratio 的传导折扣约 0.8，6T 预测不得线性外推；
> ⑤ P1 未执行的「隔离 bench ≥1.6×」验收项作废，基线补测移入 P3 开场；
> ⑥ P5 建议提前与 P2 并行穿插（独立无依赖、成本最低，且 depthwise
>    直接拖累 P2 验收模型 formula_encoder）。

> **执行状态（2026-09-03）**：P0、P1 已落地（同日验证）。
> - P1：A1b K 内层改发射 `vector::FMAOp`（MatmulKernelNCNN.cpp），lit
>   `forall-kernel.mlir` 断言 vector.fma / 禁 mulf；产物 asm 复核
>   resnet18 444 / resnet34 780 / yolov5s 540 / formula 564 / server_rec 640
>   / medium_rec 312 / yolov5n 652 条 vfmadd（此前全表为 0）；int8 产物
>   保持 0（Q 路径不进 A1b，P4 处理）。原计划中"VectorizeNCNN/FuseLinalg
>   加 contract fastmath"经核实**无落点**——两 pass 不自产累加 op（clone
>   继承源 flags），全工具链唯一自产 MAC 链就是 A1b；linalg named op 的
>   body 由 linalg 自动生成；contract 注入未在 P3/P5 落地，当前不作为既有能力。
> - P0：per-class 门禁表（performance_test.cpp `default_ratio_gate`，
>   `NCNN_PERF_MAX_RATIO` 未设时生效、显式设置全局覆盖、0=显式关闭、
>   NCNN_PERF_THREADS pin 时让位）；fixture 编译计时包装
>   `test/Numerical/support/timed_compile.cmake`（STATUS 常态打印 +
>   超 300s WARNING）；汇总脚本 `tools/perf_json_summary.py`
>   （p50/p90 + --baseline 逐模型 Δratio）。
> - **P1 实测收益（2026-09-03 全量 44 模型重测，47/47 过门禁）**：全表
>   p50 4.00→3.35、max 42.93→38.77，44/44 全线改善。绝对值大点：
>   server_det_static 5.59×→3.17×（25.99s→14.09s，−46%）、server_rec
>   31.2×→22.7×（14.40s→10.03s，−30%）；ratio 大点：server_det_static
>   −2.42、server_rec −8.47、resnet34 −1.10、tiny_rec −1.18、
>   mobile_rec_int8 −1.49。int8 系仅小幅变化（medium_rec_int8
>   29.84→27.15，Q 路径未进 A1b，符合预期）。
> - P0 附带发现：efficientnet_b1/b2/b3 单 fixture `clang -x ir -O3`
>   实测 60 分钟–2 小时级（RSS ~340MB 缓慢爬升，慢性非爆炸；17 个
>   fixture 超 300s 预算触发 WARNING），是 P2 编译爆炸根除的现行样本；
>   当时待排查的“FMA 是否为诱因”已由 P2 spike 排除，爆炸根因是无界
>   行向量化 IR 及其 math lowering。

> **执行状态（2026-09-05）**：P2 已落地（spike 定界 → 修复 → 全量验证，
> 详见 §3-P2 执行状态附注）。
> - **spike 结论（机制定案）**：编译爆炸 = 两条行向量化路径都以"最内维
>   整行"为向量宽度（VectorizeNCNN 张量级、A1b memref 级行级 generic，
>   后者同时是 vector-mode=off 应急路径的唯一向量化来源），行宽无界
>   （server_rec CTC logits 行 18385、formula 特征行 37249、slanet
>   59536）；math op（exp/sigmoid/tanh/erf/pow）一旦落在超宽向量上，
>   就超出 libmvec 等宽 ABI（8 lanes）走不到 LowerVectorMathNCNN，被
>   标量 convert-math-to-libm 逐 lane 拆成 extract/call/insert 风暴。
>   实测：server_rec 向量化 IR 含 vector<18385xf32>×36,774、标量
>   expf×21,745，clang -O3 600s 不完成；efficientnet 系的慢性编译同
>   机制（C 宽行 × sigmoid/swish + A1b 路径 672 宽行）。v1 假设
>   "FMA 是否为诱因"**排除**——爆炸在向量化 IR 形态本身，与 FMA 无关。
> - **主修路线变更（spike 驱动，等价于 v1 备选路线"收敛行宽"）**：
>   matmul 侧 A1b 内核行宽本就有界（N tile ≤128，非瓶颈），不引入
>   vector.contract 重写；两条行向量化路径统一改为 lanes 分块 +
>   标量尾兜底。分块宽取 lanes 有一箭双雕之效：math op 恰好命中
>   libmvec 等宽 ABI（此前任何行宽都不等于 lanes，libmvec 从未真正
>   启用），且 LLVM 代码量与行宽彻底解耦。
> - A/B 实证（PP-OCRv5_server_rec）：旧整行形态 clang -O3 600s 超时
>   未完成 → 分块后 22s；efficientnet_b1 无 lanes 路径 8.7min → 9s。
> - 落地清单：`VectorizeNCNN.cpp`（行分块 + 标量尾，张量级）；
>   `MatmulKernelNCNN.cpp` 新增 `row-chunk-lanes`（默认 8，行级 generic
>   分块）；`test/Numerical/CMakeLists.txt` 清空 override 名单（机制
>   保留）+ `timed_compile.cmake` 预算硬化（模型级 fixture 超 300s
>   FATAL_ERROR）；lit 更新 3 份 + 新增尾/整行用例（135/135）。
> - 分块预算二轮校准（perf A/B 驱动）：初始 lanes 等宽对 i8 行退化为
>   8 字节级 SIMD、f32 行循环粒度偏碎，int8 rec/small_rec 相对 ncnn
>   退化 9–29%；修正为按元素位宽分档——≤32 位元素 4×lanes（仍是
>   LowerVectorMathNCNN 可拆分上限，math op 拆 ≤4 段向量调用）、64 位
>   元素 lanes。A/B：small_rec 4.16→3.58、medium_rec_int8 29.65→28.18。
> - golden 交付两轮实测拦截，修复后全绿：①分块循环步长误用分块宽
>   （步长应为 1、偏移 = 下标 × 分块宽，否则整行只写第一个分块）；
>   ②私有行张量写入索引误用网格索引（行张量形状 [1,…,1,W]，外层写
>   索引必须全零，否则大规模越界写 → 堆损坏/OpenMP 死锁）。lit 只断言
>   op 存在性拦不住这两类语义 bug，golden 才是语义闸门。
> - **runtime 实测账目（修正 v1 预测）**：P2 后 rec/attention 系
>   runtime 与 P1 基本持平（formula −13%，server_rec/medium ±2%，
>   tiny/mobile_int8 残余 +15–23% 相对退化）——v1 §7 "P2 后 ≤5×/≤8×"
>   的预测基于"名单内模型卷积/GEMM 全标量"的错误前提，spike 已证伪
>   （A1b 自 P1 起就向量化名单内模型的 GEMM；其余 elementwise 有 A1b
>   行级向量化 + clang auto-vec 兜底）。名单内模型与 ncnn 的真实差距
>   在 im2col 标量 gather、depthwise 标量、matmul 微内核质量、attention
>   转置——分别是 P6/P5/P3/P6 的根因，P2 的价值是把向量化基建接回
>   （P3-P6 的前提）并根除编译爆炸。tiny_rec/mobile_rec_int8 残余
>   退化记入 P3 输入。
> - **遗留慢件定性（常量池下限）**：yolov5 l/x 系 5 件单 fixture
>   470–620s，其 .ll 文本 99% 以上是权重常量（yolov5x：174MB 参数 →
>   2.26GB 文本，代码本体仅 7.7MB），与代码生成形态无关（P2 前后同
>   量级，此前被 efficientnet 小时级爆炸掩盖）。对策：该 5 件给 900s
>   硬预算覆盖（CMakeLists `ncnn_model_compile_budget_*`），其余维持
>   300s；常量池成本的根除（常量外置/.incbin 链接）另立课题，不阻塞
>   P2 验收——验收口径相应修正为「无小时级爆炸；>300s 件数 ≤ 5 且
>   全部 ≤ 700s（常量池主导）、其余全部 ≤ 300s」。

> **执行状态（2026-09-06）**：P5 已落地（v2 建议的 P2 后穿插时点）。
> - 实现落点：`VectorizeNCNN` 新增 `vectorizeDepthwiseConvRows`——
>   multiplier=1 的 `linalg.depthwise_conv_2d_nhwc_hwcm`（全部 26 个
>   fixture / 434 处 depthwise 的唯一形态）改写为 forall(n,oh,ow) 网格
>   + C 维分块 rank-1 transfer：权重 [KH,KW,C,1] 折叠 [KH*KW,C] 行视图
>   （常量权重由 canonicalizer 直接折叠）、init 折叠 4D 作累加种子，
>   kh 外 kw 内逐窗口 `vector.fma` 累加（named op body
>   acc=addf(mulf(in,w),acc) 的归约序；单舍入差异由数值预算吸收，P1
>   同款取舍；标量尾逐位复刻 mulf+addf）；替换值 expand 回 5D 与下游
>   collapse_shape 消费者在 canonicalizer 对消。分块预算沿用逐元素
>   路径（f32 → 4×lanes），非 2 幂退 lanes、仍非 2 幂保持标量
>   （vector<3> 宽度教训）；multiplier≠1 / 动态 shape 保持标量，P8 仅作
>   audit-only 评估，未进入默认生产路径。
> - asm 断言：formula_encoder vfmadd 564（P1 后）→ **3072**、
>   vmovups 23965（行向量拷贝），server_det_static 3808；对照
>   resnet18（无 depthwise）vfmadd 444 不变——无附带代码形变。
> - lit：新增 `depthwise-vectorize.mlir`（2 正向 + multiplier≠1 /
>   窄通道 / 动态 shape 3 负向守护），136/136。
> - **全量实测（全新 /tmp 构建目录，437/437 过含 per-class 门禁；该次
>   默认路径 stage 的已测 fixture 均在 300s 内）**：这次结果不覆盖 P2 登记
>   的常量池 900s 例外件，也不覆盖 P7 opt-in Winograd 的约 8 分钟 fixture；
>   对 P2 基线 **43/44 模型 ratio
>   改善**（仅 server_rec +0.54 在噪声带内），全表 p50 3.42→2.84、
>   max 33.41→31.93。det 系收益最大：medium_det 2.93→1.89（1.55×）、
>   mobile_det_static 2.09→1.50（1.39×）、small_det 2.95→2.20、
>   tiny_det 3.21→2.48；int8 rec 系同步受益（medium_rec_int8
>   28.75→19.94、tiny_int8 8.31→5.08、small_int8 12.33→9.34——int8
>   模型的 dw 层保持 f32，被本次改写覆盖）。公式系 33.41→31.93
>   （1.05×）：formula_encoder 的 dw 仅占算力小头，v2 §3-P5
>   「det/公式系 1.2–1.5×」的公式系预期未达、det 系达成——其单核
>   差距主源仍是 matmul 微内核/im2col（P3/P6）。
> - 1T 抽测（resnet18 / medium_det / mobile_det_static / formula）：
>   compiled 多核扩展 2.7–5.9× 全面优于 ncnn 1.6–2.2×，「并行无罪」
>   结论不变；formula 单核差距归 P3/P6。
> - 构建注记：`cmake --build --parallel` 不带并行度时 make 走无界
>   并行（实测 89 个并发编译），fixture 墙钟被超订挤爆 300s 预算
>   门禁（resnet101 单独实测 118s）——全量构建须显式 `--parallel 16`。

> **执行状态（2026-09-06，P3）**：matmul M×N 寄存器分块已落地。
> - **内核改动**（`MatmulKernelNCNN.cpp`）：A1b 内核 M 方向按
>   `tileRows=4` 行一组驻留向量 accumulator（K 循环 4 条独立 FMA 链，
>   破除单链把发射率钉死在 FMA 延迟的瓶颈；B 行每轮 K 读一次复用 4
>   次），列侧按 `accColumns=16` 分块扫 N（余数列块窄向量），寄存器
>   预算 tileRows×accColumns ≤ 64 浮点（8 ymm，防溢出）。M 满块 +
>   余数行退单行形态；transfer 全标 inBounds（动态 n 块偏移下也无掩
>   码 load/store）。新增选项 `--matmul-m-rows`（=1 回 legacy 单行内
>   核，A/B 口径）与 `--matmul-acc-columns`。
> - **开场动作（隔离 bench 补测）**：FMA 后 legacy 单行内核实测
>   36.7（K=576）/ 16.2（K=2304）GFLOP/s 1T——v1 §3-P1 的「18–26」
>   是 FMA 前口径。参数扫描（9 组 M×N 组合，两组形状）确认 4×16 最
>   优；checksum 全组合与 legacy 逐位一致。
> - **实测（同机同时段 A/B，powersave 口径）**：隔离 bench 1T
>   **117.7 / 60.6 GFLOP/s**（3.06×/3.48×）；resnet18 1T 端到端
>   257.1→159.4ms（1.61×）、6T 72.2→51.3ms（ratio 3.41→2.65）；
>   6T 抽测 vs 9/3 基线（机器状态漂移，方向参考）：server_rec
>   31.16→22.60、efficientnet_b1 4.11→1.92、medium_det 3.26→1.75、
>   resnet18 4.54→2.87。golden 抽查 4/4。验收 ① 达成、② 未达（前
>   提被证伪，GEMM 占 1T ~25%，剩余在 P6），详见 §3-P3 附注。
> - **全量表重测（2026-09-06 全新 /tmp 构建，438/438 ctest 全过，含
>   per-class 门禁与新增 bench 门禁；47 perf = 44 实测 + 2 自跳 +
>   1 陈旧注册）**：全表 p50 2.84→**2.09**、p90 6.92、max 31.93→
>   **23.01**。大点：formula_encoder 31.93→**17.57**（GEMM 主导的
>   编码器吃到 P3 全部收益，−45%）、server_rec →23.01、
>   server_det_static →2.59；重模型（ncnn≥100ms）p50 **1.73**，
>   M2 口径（≤1.5/中位 ≤1.3）逼近；yolov5n_seg 1.37、
>   mobile_det_static 1.35、yolov5n 1.45 等已进 1.5 以内。int8 行
>   原地（Q 路径不进 A1b，P4 范畴）。
> - **bench 沉淀**：`test/Numerical/bench/matmul_bench.mlir`（浅 K
>   576 / 深 K 2304 两形状）+ `support/matmul_bench_main.c` +
>   ctest `MatmulBench.Gflops`（RUN_SERIAL，门禁 60/40 GFLOP/s，
>   env 可覆盖）；P6 打包后复评 ≥ 60 用同一门禁。

> **执行状态（2026-09-06，P4）**：int8 VNNI 线已落地（spike 定档 →
> 改造 → 全量验证，详见 §3-P4 执行状态附注）。Q-conv 全链接入
> strategy（im2col+`matmul_transpose_b`，权重常量编译期重排 [N,K]）；
> i8 row-dot 内核落 tier ②/③（标量 MAC 多链由 LV 生成
> `vpmovsxbw`+`vpmaddwd`/`vpmulld`，`vpdpbusd` 因 ISA 与 s8×s8 语义
> 不可自动达成）；`fuse-quant-chain-ncnn` 融合逐层 requant/dequant
> 链；行级向量化与 im2col gather 扩展整数形态。int8 行
> 19.9×–28.8× → **1.9×–4.8×**，全表 p50 2.09→**1.92**、max 23.01→
> **19.02**；验收 4/5（medium_rec_int8 4.82 vs ≤FP32×1.5 界 4.47，
> 差 8%，运行方差 ±15% 同量级；≤4× 预期未达，下一杠杆为内核内
> requant epilogue 融合，联动 P6）。

> **执行状态（2026-09-11，P6）**：搬运削减三线已落地（热点基线
> SIGUSR2 采样器实测驱动：resnet18 im2col gather 22.4%、medium_rec
> MHA batch_matmul 标量链 29%）。
> - **P6-A im2col 物化消除**：`MatmulKernelNCNN` 新增 `probeIm2colSource`
>   （gather→collapse→alloc 链探测，穿透 M 向 subview 与 tile-and-fuse
>   多副本共享 alloc；源图支配 matmul 位置的 SSA 守卫——yolov5 系在
>   无守卫时炸 dominance，即取到别层 pad 缓冲的教训）；`kernelize` K
>   循环改外层 kp（窗口位置 kh·KW+kw）×内层 ic（通道）双层直取源图窗
>   口，k = kp·IC + ic 与折叠 [[0,1],[2,3,4]] 展平严格一致（FMA 累加
>   链数值逐位不变）；gather/collapse/alloc 链判死删除。K 序不变 +
>   单舍入 FMA 同 P1 语义。非 2 幂通道（IC=3 首层）同样融合（宽容匹
>   配，整行向量化 2 幂宽度限制对直取无意义）——B5 遗留的非 2 幂
>   gather 标量形态根除。
> - **P6-C MHA 批量收缩内核化**：batch=1 的 `linalg.batch_matmul`
>   canonical 化（StrategyNCNN `rewriteUnitBatchMatmul`，[[0,1],[2]]
>   折叠成普通 matmul 进 TileMatmulForall+A1b 路径，medium_rec 44 个
>   全中）；静态 batch>1（heads=8 scores/context）新增
>   `kernelizeBatchMatmul`：forall(b) 网格 + 逐批 [b] 面板 rank-reduced
>   subview + A1b 形态内核（替代通用下降的标量 mul+add 逐 k 读写回 C
>   链）。静态 MHA 的 QKV/scores/context 进入向量内核；动态序列维仍走
>   generic contraction。
> - **实测**（stage 全新 /tmp 构建 438/438 ctest 全过 + perf 全表）：
>   全表 p50 2.84→**1.92**、p90 **3.77**、max 31.93→**19.19**；重模型
>   （ncnn≥100ms）p50 **1.72**——**M2 口径（≤1.5/中位 ≤1.3）已逼
>   近**。大点：server_rec 23.01→19.19、formula_encoder 17.57→14.61
>   （−17%）、squeezenet 3.61→2.76、medium_rec 4.67→2.25（compiled
>   133→80ms，−40%，vfmadd 1386→1930、标量 MAC 站点 329→117）、
>   resnet18 2.87→2.40、resnet50 ~1.81、efficientnet_b1 1.92→1.95、
>   yolov5m 1.45、yolov5s 1.28、mobile_det 1.09、mobile_rec 1.16——
>   8 个模型进 1.5 以内。medium_rec 单测 4.67→2.59/全表 2.25（机器
>   漂移带内）；golden 全绿（含 attention pipeline 覆盖）；SIGUSR2
>   复测 im2col gather 热点 **0%**（验收 <10% 达成）。MHA
>   splitHeads 转置物化（copy 类残余 ~16%）未消解，批量内核已接管
>   收缩热点，转置消解评估移入 P8 收尾审计。
> - **范围修正**：部分 yolov5 层的 im2col 链源图（pad）不支配 matmul
>   位置（拓扑序），融合按支配守卫正确跳过、保持原向量化 gather 路
>   径——这些层留给后续 pass 重排研究，不阻塞验收。

> **执行状态（2026-09-12，P7 + P4 遗留 requant 融合）**：
> - **P7 Winograd F(6,3)**：机制与数值契约落地（`--conv-strategy=winograd`
>   opt-in，operator 级 + resnet18 全主干双 golden 通过，rtol 1e-3/atol
>   1e-4）；性能验收**未达**（resnet18 变体实测 69.8ms vs 基线 35.5ms，
>   1.97× 劣化）——通用 tile 形态的 3 次 transpose 搬运吃掉算术强度
>   收益，且 ncnn 对该批模型实际不选 F(6,3)（`test_prefer_winograd63`
>   对 IC≥64 false）。按 §3-P7 v2 失败退路：保持 opt-in 默认关闭，
>   conv 系 M3 走降级口径（例外清单 ≤1.25）。详见 §3-P7 执行状态附注。
> - **P4 遗留 requant epilogue 融合**（medium_rec_int8 ≤4 目标的下一
>   杠杆）：`kernelizeInt8RowDot` 的写回段内联 requant generic（acc
>   唯一用户、恒等主值 + 常量广播 scale/bias 的 FuseQuantChain 产物形
>   态）——i32 中间物化与独立全量 pass 同时消失，asm 见 vcvtdq2ps
>   692 处 + vcvttps2dq 217 处进入向量代码。数值 golden（Int8 完整链
>   + Int8Codegen）绿灯。medium_rec_int8 实测 5.04（4.82 漂移带内，
>   该模型残余在 dw 层与通道侧而非 requant——P4 附注的"联动 P6"杠杆
>   落地，瓶颈判定更新为 dw-int8/通道侧）。
> - **P7 后全量基线（同日全新 stage 树 48/48 ctest 全过 + JSON 重测）**：
>   全表 p50 **1.72**、max 19.41（server_rec）；重模型（ncnn≥100ms）
>   p50 **1.63**；11/44 进 1.5 以内、4/44 进 1.25 以内、1/44 追平
>   （mobile_det_static **0.83× 反超**）。剩余差距的分层归因与对应
>   办法见 §8。

> 目标：44 模型 performance_tests 套件编译产物全面追平 vendored ncnn
> （x86-64 AVX2/FMA 口径）。根因依据见姊妹篇
> [`ncnn-performance-gap-analysis.md`](ncnn-performance-gap-analysis.md)
> （2026-09-03 历史基线定案：并行无罪、内核有罪——零 FMA、向量化名单=最差
> 名单、当时尚无原生 VNNI/pass 覆盖、搬运:算术 19:1–80:1；P4 后受限
> int8 row-dot/requant 已落地，但原生 `vpdpbusd` 选择仍不可达）。
> 本文按根因给出阶段化路线、验收口径与风险。

---

## 1. 目标与验收口径

**"追上"的定义（分级）**，ratio = compiled_mean / ncnn_mean（6 线程正式口径）：

| 里程碑 | 覆盖 | 门禁 |
|---|---|---|
| M1（P1–P3 后） | 常规 conv 网 | 全部 ≤ 2.0，中位数 ≤ 1.5 |
| M2（P4–P6 后） | 全表 | 重模型（ncnn ≥ 100ms）≤ 1.5，中位数 ≤ 1.3 |
| M3（P7–P8 后） | 全表 | 重模型 ≤ 1.1，轻模型 ≤ 1.25（含计时噪声带），中位数 ≤ 1.0 |
| M3 降级口径 | 全表 | 仅当 P7 的性能或编译预算门禁未能覆盖主力重模型时启用：Winograd 继续 opt-in/default-off，不把未通过性能门禁的变体算作默认覆盖；已通过主力路径门禁的重模型 ≤ 1.1，未过门禁的重模型 ≤ 1.25 并进例外清单逐模型对账，中位数 ≤ 1.0 不变。M3 宣告时必须声明走标准口径还是降级口径 |

- 门禁落地：`NCNN_PERF_MAX_RATIO` 升级为按类阈值表（P0），ctest 内强制，
  轻模型阈值宽、重模型严（与 baseline-report §4 的 cv 观察一致）。
- **编译时长预算**：普通 fixture 的 `clang -x ir -O3` 段 ≤ 5 分钟；
  已明确登记的例外（包括 P2 常量池主导件和 P7 opt-in Winograd fixture）
  可单独使用 900 秒预算，整包重建不劣于当前（防止修复劣化重蹈"编译爆炸
  名单"覆辙）。P0 起在 fixture 构建记录编译耗时并在超出对应预算时告警或
  失败。
- 每阶段验收固定三件套：① 全量数值黄金 ctest 绿（预算不放宽）；
  ② 声称性能收益的生产/性能阶段执行 `NCNN_PERF_JSON` 全量表重测并更新
  baseline-report；仅改测试或文档、且明确声明 regression-only 的阶段须记录
  不做 ratio 重测；③ 静态 asm 抽查（objdump 断言目标指令出现/消失）。
- 第四件（v2 增补）：验收时对 3 个代表模型补 1T 抽测复核并行扩展。
  P1 实测校准：内核收益向 6T ratio 的传导折扣约 0.8（resnet34 预测
  1.3–2×、实测 1.28×）——后续里程碑的 6T 预测与门禁评估一律先过该
  折扣，不得把单核收益线性外推到墙钟。

## 2. 历史基线锚点（2026-09-03；当前状态见顶部执行块与 §8）

| 分层 | 现值 | 主根因（对应 §3 阶段） |
|---|---|---|
| 常规 conv 网（名单外 33 模型） | 2.2–5.0× | 零 FMA、无 M×N 分块、无 Winograd、depthwise 标量（P1/P3/P5/P6/P7） |
| rec/attention 系（名单内 11 模型） | 4.7–42.9× | 关闭显式向量化（编译爆炸），标量卷积/GEMM（P2） |
| int8 全系 | 4.9–29.8× | 标量 i32-MAC、无 VNNI、逐层量化/反量化 pass（P4） |
| 单核算力锚点 | resnet18 10 vs 64 GFLOP/s；yolov5s 24 vs 92 | 同上 |

## 3. 阶段规划

### P0 门禁与度量基建（0.5–1 天）

- per-class ratio 阈值表进 `test/Numerical/CMakeLists.txt`（模型清单已知，
  按上表 M1 口径先设宽松值，随里程碑收紧）；
- fixture 构建脚本记录各模型编译耗时，超 5 分钟/个即 ctest 告警（防爆炸回归）；
- perf JSON 增加 p50/p90 汇总脚本（复用现有 NDJSON）。

验收：门禁生效（人为构造超阈值用例失败）；耗时记录产出当前基线表。

### P1 FMA 落地（1–2 天）——全局一行级收益

根因：A1b/VectorizeNCNN 发射的 `MulFOp`+`AddFOp` 无 fastmath，工具链无
contract 处理 → 全产物 0 条 `vfmadd*`，MAC 指令数与依赖链延迟双倍。

- `MatmulKernelNCNN`：K 内层 `mul+add` 直接发射 **`vector::FMAOp`**
  （单舍入，与 ncnn FMA 内核舍入语义一致，数值预算可吸收）；
- 原计划曾要求 `VectorizeNCNN`/`FuseLinalgEpilogue` 生成的累加型 arith op
  链设置 arith fastmath `<contract>`；执行核实两 pass 不自产累加 op，故该项
  未落地，当前不把它写成既有能力；仅 A1b 的 `vector::FMAOp` 路径已落地；
- lit：断言 A1b 产物出现 `vector.fma`；asm 抽查：resnet18/公式产物
  `vfmadd*` 计数 > 0、`vmulps` 显著下降。

验收：golden 全绿 + 隔离 matmul bench（现状 18–26 GFLOP/s）≥ 1.6×；
全量表重测，GEMM 主导模型（resnet/yolo/公式系）预期 1.3–2× 收益。

> v2 补记（2026-09-05）：隔离 bench 项未随 P1 执行，**作废**——全量
> 重测 + asm 断言已覆盖验收意图；「隔离 bench 现值补测」移入 P3 开场
> 动作（P3 验收需要该基线作对照）。

### P2 编译爆炸根除，11 模型回归向量化（4–7 天）——最大单项

根因：`ncnn_model_no_vector_overrides` 11 模型在任意 lane 宽度下
`clang -x ir -O3` 数十分钟至小时级编译（LLVM 合法化期），被迫整模关闭
MLIR 显式向量化；名单与 ratio ≥ 6× 全部重合。

**v2 范围修正**：编译爆炸面大于 11 模型名单——名单外、向量化开着的
efficientnet_b1/b2/b3 实测单 fixture 1–2 小时（P0 重建实测，RSS 稳定
呈慢性）；P0 告警共抓到 17 个 fixture 超 300s。本阶段对爆炸源的定界
必须同时覆盖这两类样本，否则主修路线只治 matmul 行向量化、不治
efficientnet 类深窄 conv 网，编译耗时预算约束将持续空转。

- **Spike（1–1.5 天）**：binary-search 定位爆炸源——候选：①
  VectorizeNCNN 行向量化对大 K matmul-邻接 generic 产生的超长直线
  vector 代码；② `vector<128xf32>` 整行类型在 x86 合法化的拆分代价；
  ③ A1b 显式内核落地后（名单制定在先）爆炸是否已自然缓解；
  ④ efficientnet 系慢性编译与上述机制是否同源（同时回答 v1 遗留的
  一行回退 A/B：FMA 是否为诱因）。
- **原计划主修路线（后经 spike 否决）**：matmul 形态改走
  **canonical tiling + `vector.contract`**（`scf::tileUsingSCF` 切 M/N
  tile → vector.contract）；行向量化保留给逐元素 generic。
- **原计划备选路线（实际落地）**：保留 A1b 手写内核 + 收敛行宽/unroll
  上界（按 spike 结论定界），P1 的 vector.fma 直接受益。
- 名单收敛：按模型逐个移出 override，编译耗时预算内放行。
- **预算门禁硬化（v2）**：`timed_compile.cmake` 对已定界慢件（17 个
  超时件 + 11 override 模型）改为硬失败；新慢件先软告警观察一轮再
  硬化，避免首次接触即打断整包重建。

验收：11 模型全量编译 ≤ 预算；超 300s fixture 数从 17 收敛至 ≤ 5
（且全部为 yolov5 l/x 系常量池下限件、≤ 700s，见执行状态）；
全量表重测——rec/attention 系从 4.7–42.9× 收敛至与名单外同量级
（预期 ≤ 5×，**实测未达**：名单内模型 runtime 与 P1 基本持平，
v1 预测前提"卷积/GEMM 全标量"被 spike 证伪，runtime 收敛依赖
P3/P4/P5/P6，见执行状态 runtime 账目）；golden 全绿。**传递关系**：
验收后 11 模型并入常规路径，P3/P5/P6 的收益预期同样适用于它们，
后续里程碑对账不得把这批模型当独立分支。

> **P2 执行状态附注（2026-09-05）**：主修路线按 spike 结论调整——
> 爆炸机制是两条行向量化路径（VectorizeNCNN 张量级、A1b memref 级
> 行级 generic）的"最内维整行"无界向量 + math op 超出 libmvec 等宽
> ABI 后的标量 libcall 风暴，修复为统一 lanes 分块 + 标量尾（即 v1
> 备选路线"收敛行宽"），不引入 vector.contract 重写（matmul 侧 A1b
> 行宽本就有界，非瓶颈）。A/B：server_rec 旧形态 clang 600s 超时 →
> 22s；efficientnet_b1 无 lanes 路径 8.7min → 9s；efficientnet_b1
> /b2/b3 全 fixture 编译 1–2h → 30/35/44s；名单内 11 模型全部回到
> 向量化且编译 ≤ 106s。"FMA 是否为爆炸诱因"排除（爆炸在向量化 IR
> 形态本身）。

### P3 matmul 微内核 M×N 寄存器分块（4–6 天）

根因：A1b 每行单 accumulator、B 按行反复重读，算术强度 ~0.5 flop/byte，
带宽受限；ncnn sgemm 为打包面板 + 6×16 寄存器分块。

- A1b 内核改 M×N tile：M 方向 4–6 行 accumulator 驻留寄存器，A tile
  一次读入、B 行复用 4–6 次；行宽由 TargetVectorInfo 推导（延续 tile/lane
  解耦原则）；
- 打包评估：先做零拷贝（kernel 内 A 子面板 hoist 到寄存器/栈），K·N
  工作集仍超 L2 时再评估显式 B 打包（P6 联动）；
- 线程：外层 M（或 M×N 网格）保持 scf.parallel→OpenMP。
- 开场动作（v2）：补测 P1 后隔离 matmul bench 现值作为对照基线
  （该项在 P1 作废，见 P1 补记）。

验收：隔离 bench 1T 首轮 ≥ 45 GFLOP/s（无显式打包，约 AVX2+FMA 峰值
的 40%；ncnn sgemm 同形状对照 ≥ 100 作为差距标尺不变），P6 打包
落地后复评 ≥ 60；resnet18 1T 端到端 ≥ 2× 于 P1 后水平（6T 口径按
~0.8 传导折扣预期 1.6–2.2×，不作线性外推）；golden 全绿。

> **P3 执行状态附注（2026-09-06）**：已落地（M×N 寄存器分块 + 隔离
> bench 沉淀，见顶部执行状态块）。验收对账：①隔离 bench ≥ 45
> GFLOP/s **达成**（浅 K 117.7 / 深 K 60.6，两形状对 legacy 单行内核
> 3.06×/3.48×，checksum 与 legacy 逐位一致——K 循环累加序未动）；
> ②resnet18 1T 端到端 ≥ 2× **未达**（同机同时段 A/B 实测 1.61×：
> legacy 257.1ms → 分块 159.4ms）——其前提「GEMM 主导 resnet18 1T」
> 被实测证伪：分块后 GEMM 仅占 1T 墙钟 ~25%（1.82 GFLOP ÷ ~60–112
> GFLOP/s ≈ 25–30ms），其余在 im2col 物化/epilogue/池化（P6 范畴），
> 与 P2 runtime 账目修正同因。深 K（K≥2304，B tile 超 L2）实测 ~56
> GFLOP/s 即本阶段无打包上限，≥ 60 复评留 P6。resnet18 6T ratio
> 3.41→2.65（6T 预期 1.6–2.2× 的**绝对值口径**落空、方向达成）。
> 传导折扣实测 ~0.68（1T 1.61× → 6T 1.41×），略低于 0.8 校准值。

### P4 int8 VNNI 线（6–10 天）——独立最差分支

根因：量化卷积不进 strategy/matmul-kernel/向量化任何 pass，标量
i32-MAC + 逐层激活量化/反量化全量 pass；产物 0 VNNI（ncnn 为
avxvnni `vpdpbusd` + LUT requant）。

- **Spike（1 天）**：LLVM 21 对 i8i8→i32 `vector.contract` 的 x86 下降
  能力——依次验证 ① 直出 `vpdpbusd`（avxvnni）；② i8→i16 ext +
  i16i16→i32 contract 出 `vpmaddwd` 链；③ 均不可则 widened 自动向量化
  兜底。以 spike 结论定内核形态，三档均可接受（收益递减）。
- 量化卷积接入 `strategy-ncnn`：Q-conv（含 depthwise-Q 变体）与浮点同
  路径改写 im2col+matmul；i8 数据布局复用浮点 matmul 微内核骨架；
- 语义精确性：requant（scale-term 乘加 + round-half-away + clamp）、
  激活量化舍入语义逐位复刻（沿用 INT8 预量化的既有对账方法），golden
  int8 稳定性契约不放宽；
- 量化/反量化融合：producer epilogue 内联激活量化（省全量 pass），
  consumer 侧 dequant 融进 requant 尾部。
- 语义-形态冲突预案（v2）：若「逐位复刻 requant」与某一下降档位冲突，
  语义优先、降档实现，并同步下调该档预期收益——golden 契约先于
  融合强度与收益目标。

验收：int8 行 ratio ≤ 自身 FP32 行 ratio × 1.5；asm 抽查出现
`vpdpbusd`（或 spike 定档的对应指令）；medium_rec_int8 预期 29.8× → ≤ 4×。

> **P4 执行状态附注（2026-09-06）**：已落地，详见顶部执行状态块。
> - **spike 定档（与 v1 预期路线不同）**：tier ① `vpdpbusd` 不可达——
>   本机 CPU 具备 `avx_vnni` 但无 `avx_vnni_int8`（`vpdpbssd` 需要），
>   且 byte-VNNI 原生组合是 u8×s8，s8×s8 数据需 u8 技巧 + intrinsics
>   直发（违背 ISA 口径与语义优先原则）。tier ② `vpmaddwd` 达成，但
>   路径 ≠ 计划预期的 vector.contract 下降：**标量 i32-MAC 循环经
>   LLVM 循环向量化器稳定生成 `vpmovsxbw`+`vpmaddwd`**（ncnn AVX2
>   int8 内核同款 MAC）；显式向量内核的 mul+add 配对树依赖中端展开+
>   重关联的偶然建树，实测稳定落回 `vpmulld`（tier ③）。最终内核
>   形态：**刻意发射标量 i32-MAC 多链**（tileRows×accColumns 条独立
>   链，预算 8），由 LV 完成向量化——整型结合律保证与串行逐位一致。
> - **落地清单**：① `quantizeSignedI8` 改纯 arith trunc 形态
>   （round-half-away 逐位等价证明：`x≥0 ? trunc(x+.5) : trunc(x−.5)`，
>   f32 侧 ±128 收拢保证 fptosi 无 UB）——消除向量 floor/ceil 依赖；
>   ② 新 pass `fuse-quant-chain-ncnn`：逐层 requant/dequant 尾部
>   （sitofp→scale mul→bias→[act]→quantize 3–4 个全量 pass）融合为
>   单一混合类型 generic，op 序/常量不变、数值逐位一致；③
>   `strategy-ncnn`：int8 k×k 静态恒走 im2col+GEMM（无标量直接卷积
>   内核可用）、权重常量直接重排 [N,K]（ncnn transpose_pack_B 的编译
>   期化）、1×1 与 InnerProduct/Gemm 同走 `matmul_transpose_b`；④
>   `TileMatmulForall` 模板化支持 transpose_b；⑤ `MatmulKernelNCNN`
>   新增 i8 row-dot 内核（`--matmul-i8-rows`/`--matmul-i8-acc-columns`，
>   默认 2×4）；⑥ 行级向量化扩展整数/混合类型 + 广播映射输入
>   （requant 尾部与激活量化向量化）；⑦ im2col gather i8（非 2 幂
>   通道 32-lane 分块 + 标量尾）；⑧ VectorizeNCNN 提升修复为逐结果
>   类型（quantize 的 cmpf i1 结果曾统一赋 f32 向量型，非法 IR）。
> - **范围修正**：dw-Q 未做——int8 模型 zoo 的 dw 层本身保持 f32
>   （被 P5 覆盖），ncnn 自身 dw-int8 也不走 im2col+gemm；动态空间维
>   i8 conv 不进 GEMM（perf 表均为静态口径）；K<8 的 GEMM 走标量尾。
> - **实测（2026-09-06 全新 /tmp stage 构建 438/438 ctest 全过后的
>   空闲机器全量表）**：int8 行 19.9×–28.8× → **1.9×–4.8×**——
>   medium_rec_int8 19.94→4.82、small_rec_int8 9.34→4.18、
>   tiny_rec_int8 8.31→3.66、mobile_rec_int8 4.9×→2.25、
>   medium_det_int8 1.9×→1.92（该行 P5 后已达标，本次持平）；compiled
>   墙钟较显式向量内核再降 12–32%（LV pmaddwd 收益）。全表
>   p50 2.09→**1.92**、max 23.01→**19.02**。asm：GEMM 内核 `imull`
>   →0，`vpmaddwd`（196 处/medium_rec）+`vpmulld` 接管。
> - **验收对账**：① int8 ≤ FP32×1.5——**4/5 过**
>   （medium_rec_int8 4.82 vs 界 4.47，差 8%；该模型三次全表
>   4.31/4.50/4.82、一次抽测 3.74，运行间方差 ±15% 与界限同量级，
>   未判定为稳定未达）；② asm 抽查——`vpdpbusd` 不可达（本机无
>   `avx_vnni_int8`，s8×s8 无自动 byte-VNNI），定档
>   `vpmaddwd`/`vpmulld` 出现 ✓；③ medium_rec_int8 ≤ 4×——4.82，
>   **未达**（同上方差说明），残余主源：requant 仍为独立全量 pass
>   （i32 中间物化 + 逐层访存），内核内 requant epilogue 融合是下一
>   杠杆（联动 P6 打包）；④ 1T 抽测（同机同时段 6T 对照）——
>   medium_rec_int8 1T 7.04 / 6T 3.74（compiled 多核扩展 4.67×，fp32
>   同系 4.42×）、small_rec_int8 3.32×（fp32 3.42×）——int8 并行扩展
>   与 fp32 相当，「并行无罪」结论在 int8 上成立。
> - **tile 参数**：`--matmul-i8-rows`/`--matmul-i8-acc-columns` 默认
>   2×4；（1,4）扫描实测 pmulld 站点 512→2464 显著劣化，2×4 定档。

### P5 depthwise 行向量化（2–3 天）

根因：DepthwiseConv2D lower 为独立 generic，窗口 gather 不满足逐元素
匹配条件，纯标量（attention 系每模型 27–28 层；仅 formula_encoder 即
27 层，直接拖累 P2 验收模型）。

**v2 排序调整**：建议提前——本阶段独立无依赖、成本最低，宜与 P2
并行穿插或紧随其后执行，使 P2 验收时 rec/attention 系的实测更接近
真实内核水平，并为 P6 搬运削减提供参照点。

- `VectorizeNCNN` 增补 depthwise 形态：multiplier=1 时窗口读在 C 维连续
  （`x[oh+kh][ow+kw][c0..c0+VL]`），按 C 行 rank-1 transfer 改写；
  多通道 multiplier 变体降级标量留待 P8 评估；
- 复用 matmul-kernel pass 的"通道为 2 的幂"门控经验（vector<3> 非法
  宽度教训）。

验收：formula_encoder 1T 单测中 depthwise 段 asm 出现行向量拷贝+FMA；
全量表 det/公式系 1.2–1.5× 收益；golden 全绿。

> **P5 执行状态附注（2026-09-06）**：已按本节路线落地
> （`vectorizeDepthwiseConvRows`，见顶部执行状态块）。实测修正：det 系
> 1.06–1.55×（medium_det 1.55× 最高）、int8 rec 系同步受益；公式系仅
> 1.05×——formula_encoder 的 dw 占算力小头，1.2–1.5× 的公式系预期
> 高估，其单核差距主源在 P3/P6。asm 验收达成：formula_encoder
> vfmadd 564→3072 + 行向量拷贝，对照 resnet18 不变。

### P6 搬运削减：im2col 融合与布局折腾（3–5 天）

根因：搬运:算术 19:1–80:1——im2col 物化拷贝、逐 op 布局转换、
`vmovss`/`vinsertps` 部分 lane 脚手架。

- im2col 与 matmul 微内核融合（gather-free：kernel 内按窗口索引直取
  A 面板），优先做 1×1 已是视图无损耗、k×k 窗口融合分档启用
  （工作集阈值沿用 `--conv-gemm-l2-bytes` 启发式）；
- 非 2 幂窄通道 gather 行向量化补齐（B5 遗留：IC 非 2 幂仍标量）；
- MHA/attention 非常量 transpose 运行时路径审计：QKV 投影后的逐帧
  转置按 consumer 形态消解或并入相邻 matmul 的映射。

验收：重模型 perf 表中 im2col 热点段（SIGUSR2 采样器复用）占比 < 10%；
全量表 + golden 三件套绿。

### P7 Winograd（4–6 天，含数值预算验证）

根因（原计划假设，已由 P7 实测修正）：曾假定 ncnn 对 3×3 s1 默认
Winograd（resnet/yolo/det 类主力），本工具链仅留开关位；P7 实测显示本机
ncnn 对 IC≥64 的主力实例并不选择 F(6,3)，因此该假设不能作为当前事实。
M3 目标下 conv 系从 1.3–2× 再往 1.1× 走仍需另有可证明的内核收益。

- **原计划（后经 P7 性能实测修正）**：`strategy-ncnn` 落地
  `winograd`：F(6×6,3×3)（ncnn 同款）——输入/权重变换为 generic（可
  向量化），中心 matmul 复用 P3 微内核，逆变换同；
- **原计划 dispatch 假设（未翻为当前默认）**：3×3 s1 且 OC·IC 超阈值
  （复刻 ncnn convolution.winograd 判据），CLI `--conv-strategy=winograd`
  保持默认 off；原计划的“预算验证通过后翻 auto 默认”因 P7 性能未达而作废；
- **数值预算先行**：F(6,3) 误差上界逐模型对账（复用 golden 预算机制，
  参考 ExpandStridedMetadata 教训——预算过不了就只对宽松预算模型启用，
  不做全局默认）。
- **失败退路（v2）**：若性能或编译预算门禁未覆盖 resnet/yolo/det
  主力重模型，M3 转降级口径（§1）——该批模型进例外清单（≤ 1.25）+
  逐模型对账报告作为本阶段产出；不得为过预算放宽 golden 或降低变换精度。

原计划验收（P7 性能门禁未达）：预算内模型 auto 走 Winograd，
resnet18/yolov5 系 3×3 段 ≥ 1.5×于 P3 后水平；golden 全绿且误差在已
对账预算内。

> **P7 执行状态附注（2026-09-12）**：机制与数值契约落地，性能验收
> **未达**——按 v2 失败退路处理，Winograd 保持 opt-in 默认关闭，conv
> 系走 M3 降级口径（例外清单对账，见下）。
> - **机制落地**：`--conv-strategy=winograd`（显式 opt-in）对 3×3 s1
>   d1 且 IC>8||OC>8 的静态批 1 实例改写为：pad 到 tile 网格 → 输入
>   变换两段 8-tap 全展开 generic（B/Bᵀ）→ 编译期权重变换常量
>   [64,OC,IC]（G·w·Gᵀ 两段折叠）→ 中心 `linalg.batch_matmul`
>   [64,OC,IC]×[64,IC,T]（命中 P6-C kernelizeBatchMatmul 的
>   forall(b)+A1b 内核，复用 P3 微内核）→ 输出变换两段（Aᵀ/A）→
>   裁剪 + bias 补加（conv init 逐元素）。判据外实例回退常规 dispatch
>   （显式策略不比 auto 更少优化）。
> - **数值契约达成**：operator 级 `ConvolutionWinogradMatchesReference`
>   （rtol 1e-3/atol 1e-4）与 resnet18 全主干
>   `ResNet18WinogradMatchesNcnn` 双 golden 通过；f32 相对误差 ~1e-3
>   量级（有理插值点 ±1/±2/±1/2/∞，spike f64 验证矩阵恒等式 2.5e-13）。
> - **性能验收未达（关键实测）**：resnet18 winograd 变体（全主干改写）
>   实测 69.8ms vs 基线（im2col 路径）35.5ms——**1.97× 劣化**，与
>   计划预期的 ≥1.5× 收益相反。归因：(1) 变换段虽已向量化（直线展开
>   形态，产物 19544 vmulps/23856 vaddps vs 归约形态的 52 vmulss），
>   但 F(6,3) 通用 tile 形态的 3 次数据搬运 transpose（Y 布局重排）
>   未消解，算术强度提升被搬运吃掉；(2) ncnn 的 Winograd 优势来自
>   pack 面板 + 专用 AVX2 内核 + MHA 式 L2 分块调度，其 dispatch 判据
>   `test_prefer_winograd63` 对 IC≥64 返回 **false**（本机 ncnn 对
>   resnet18 主干实际不选 F(6,3)，走 sgemm + 打包）——即"ncnn 默认
>   Winograd"的前提在这批主力模型上本就不成立（计划根因描述过强）。
> - **编译预算**：直线展开形态 IR 体积大（.so 282MB vs 基线 47MB），
>   resnet18_winograd 单 fixture 编译 ~8 分钟，预算覆盖表单列 900s。
> - **决策**：按 v2 退路——`--conv-strategy=winograd` 保持 opt-in（机制
>   与数值契约已沉淀，供后续 pack 化/专用内核研究复用），auto 不翻
>   默认；conv 系 M3 口径走降级路径（例外清单 ≤1.25 逐模型对账）。
>   后续若要复活 Winograd 收益，需 ncnn 同款 pack-A/pack-B 物化 +
>   专用变换内核（P3/P6 微内核架构内的专项，预估 4–6 人天）。

### P8 收尾状态（截至 2026-09-13，尚未宣告 M3）

P8 按“先低风险可证明收益，再做隔离审计”的顺序执行；没有把未经
数值、生命周期或性能门禁证明的路径写成默认优化。

- **P8-1 wide-N 矩阵内核（已落地，提交 `16abe60`）**：
  `MatmulKernelNCNN` 不再把逐元素行宽上限 1024 错用于浮点
  `linalg.matmul` 的 RHS/输出 N 维（本轮以 f32 fixture 验证），也不再用于
  int8 row-dot accumulator
  的 N 维；`accColumns`、寄存器预算、尾块路径保持有界。新增 f32
  `N=2049`、int8 `N=2051` lit，确认不会生成完整 N 宽向量。
- **P8-2 静态/动态 MHA 守护（已落地，提交 `9720c4a`）**：
  新增 standalone raw memref `N=2049` 的 `linalg.batch_matmul` bounded-panel
  回归；另为 `N=3` 的静态 self-attention fixture 增加
  TOSA→Linalg→buffer pipeline smoke guard，确认该 fixture 进入向量化
  batch-kernel 形态。检查没有逐个证明 score/context 两个 op，也没有覆盖
  长序列 MHA；动态序列仍保留 generic contraction。没有删除
  `splitHeads`/key/context transpose，也不宣称动态 MHA 已内核化。
- **P8-3 f32 depthwise multiplier>1（audit-only）**：继续保留
  multiplier=1 的生产向量路径和 multiplier=2 的负向 lit。现有改写把
  `[KH,KW,C,1]`、`[N,OH,OW,C,1]` 折叠为 C 连续行；multiplier>1 的
  HWCM 逻辑布局要求 multiplier 轴专门向量化或显式重排，不能只放宽
  matcher，否则会产生错误的输入 lane 或越界访问。当前 importer/端到端
  契约也不足以宣称 multiplier>1 的 NCNN 支持。
- **P8-4 int8 depthwise/cast（audit-only）**：未满足 i8×i8→i32
  named-op 类型契约、逐位 rounding/zero-point/bias 顺序和端到端 importer
  证明前不改默认 lowering；继续沿用 P4 的 row-dot/requant 语义。
- **P8-5 小模型 arena/entry hoist（audit-only）**：One-Shot Bufferize、
  `BufferResultsToOutParams(hoistStaticAllocs=true)`、deallocation pipeline
  和 verifier 已存在；在没有线程安全、生命周期、并发调用和收益证据前，
  不引入改变 public C ABI 的全局 arena。
- **P8-6 pack-A/pack-B 与 Winograd（audit-only）**：P7 Winograd 继续
  opt-in/default-off；没有新的 pack 实验达到代码尺寸、编译时间、checksum
  和代表模型性能的联合门禁，因此不扩大 dispatch。

P8-1/2 的最终独立提交门禁均在全新 `/tmp` Release 树完成：
format_check、全局 tidy、完整并行构建、三个 numerical target 和完整 ctest
均通过；最终各为 441 个 ctest、0 失败，2 个已知 int8 性能项按既有条件
跳过。P8-1 初次 stage0 兼容性尝试使用旧 binary 运行新增 wide-N lit，曾因旧
FileCheck 类型期望失败；随后修正为 strided subview 类型，并在全新 stage 树
重建、重跑通过。该兼容性修复过程不作为最终通过次数隐藏。验收证明回归和
安全边界成立，不等同于新的全量 ratio 基线；
M3 仍按 P7 附注的降级口径处理，待后续性能基线重测后再宣告。

后续收尾工作：更新 support-status §6、baseline-report 与 gap-analysis
的 P8 对照，并在新的性能实测完成后决定是否形成 M3 例外清单；在此之前
不提交预测收益。

## 4. 依赖与排序

```
P0 ─→ P1 ─→ P2 ─→ P3 ─→ P4（复用 P3 骨架）
         └→ P5（与 P2 并行穿插；备选时点 P2 后）
              │      └─→ P6
                     P3 ─→ P7（复用微内核）
              全部 ─→ P8
```

- P1 最先：所有后续内核路径直接受益，且改动最小；
- P2 先于 P3-P7 的度量解读：否则 11 个模型（含全部极值点）不可见；
- P5 独立无依赖，v2 建议提前与 P2 并行穿插（最迟不晚于 P3，见 P5 节）；
  P4 依赖 P3 的微内核骨架与 P2 的 tiling 路线；
- 串行预估：**30–45 人天**（v2 校准：P0 0.5–1 / P1 1–2（已完）/
  P2 4–7 / P3 4–6 / P4 6–10 / P5 2–3 / P6 3–5 / P7 4–6 / P8 2–3）。
  校准依据：P2–P4 属 LLVM 下降调优与语义复刻类工作，spike 类定界按
  可能吃掉一半预算计；P5 提前不影响总盘，并行执行还会压缩串行总长。

## 5. 数值契约与回归策略

- 所有内核改动（FMA 单舍入、Winograd 变换、int8 requant）走既有
  golden 预算验收，**预算不因性能放宽**；int8 舍入语义逐位复刻；
- 每阶段三件套（golden / perf 全量表 / asm 抽查）+ 编译耗时预算；
- ratio 门禁随里程碑单调收紧，历史表按 baseline-report 惯例留档。

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| ~~vector.contract 路线在 clang -O3 仍爆炸（P2 主路线失效）~~ **已由 P2 spike 否决** | 实际采用 lanes 分块 + 标量尾；继续用编译耗时预算门禁防止无界向量回归 |
| 编译爆炸面大于 11 模型名单（efficientnet 系名单外 1–2h/件实测） | P2 spike 扩围定界（含 FMA 一行回退 A/B）；普通超 300s fixture 数收敛目标 ≤ 5，登记的常量池与 opt-in Winograd 例外按 900s 单独对账 |
| 内核收益向 6T ratio 传导打折（多核下内存带宽共享上限） | 每阶段验收附 1T 抽测复核并行扩展；6T 预测一律过 ~0.8 传导折扣（P1 校准） |
| M3 依赖 Winograd，而 P7 性能或编译预算门禁未覆盖主力重模型 | §1 M3 降级口径：例外清单 ≤ 1.25 + 逐模型对账；不为过预算放宽 golden |
| i8 contract 无 VNNI 下降（LLVM 21 不识别） | 三档 spike（vpdpbusd / pmaddwd / widened auto-vec），档位收益递减但均可验收 |
| Winograd 误差超模型 golden 预算 | 预算逐模型对账，过不了不改全局默认（保持 opt-in） |
| int8 requant 语义偏差 | 复用 INT8 预量化对账方法；逐位复刻优先于融合强度 |
| M×N 分块寄存器溢出（tile 过大） | tile 由 TargetVectorInfo 推导 + `--conv-tile-width` CLI 覆盖（既有解耦机制） |
| perf 波动掩盖里程碑判定 | 重模型 cv≤0.1 已实证；门禁以重模型为主、轻模型放宽（baseline-report §4 结论） |
| 双 OpenMP runtime（libgomp+libomp）线程扰动 | 既有两条腿线程控制已实证有效，不新增依赖 |

## 7. 里程碑指标预测（按当前根因折算，验收时以实测为准）

| 阶段完成 | 名单外 conv 网 | 名单内 rec/attention | int8 |
|---|---|---|---|
| P1 | 5× → 2.5–3.5× | 不变（标量） | 不变 |
| P2 | — | 42.9× → ≤ 5× | 30× → ≤ 8× |
| P3 | → 1.6–2.2× | → 3–4× | — |
| P4 | — | — | → ≤ 3×（≈自身 FP32） |
| P5+P6 | → 1.2–1.6× | → 2–3× | → ≤ 2.5× |
| P7+P8（M3） | ≤ 1.1（降级口径 ≤ 1.25） | ≤ 1.5（attention 尾部） | ≤ 1.5 |

v2 校准注：P1 实测显示内核收益向 6T ratio 的传导折扣约 0.8（resnet34
预测 1.3–2×、实测 1.28×），P3 行据此放宽；P5 提前后其收益将部分并入
P2/P3 验收时的实测值。本表为预测，验收以实测为准。

## 8. P7 后剩余差距分层归因与对策（2026-09-12 基线；P8 状态补记至 2026-09-13）

数据口径：P7 + requant 融合落地后、全新 stage 树全量重测（44 模型，
6 线程正式口径，JSON 留档）。当前全表 p50 **1.72**、重模型 p50 **1.63**、
11/44 进 1.5 以内、1/44 反超（mobile_det_static 0.83×）。

### 8.1 分布带与里程碑对账

| 分布带 | 模型数 | 代表 | 对应里程碑口径 |
|---|---|---|---|
| 已追平/反超（≤1.1） | 1 | mobile_det_static 0.83× | M3 单点达标 |
| 1.1–1.5 | 10 | yolov5m_seg 1.19、yolov5s 1.25、mobile_rec 1.14 | M2 达标带 |
| 1.6–2.3 | ~20 | resnet 1.61–2.22、efficientnet 1.67–1.92、yolov5 大件 1.63–1.74、det 系 1.66–2.07 | M2 未达的主因（重模型 p50 1.63 vs 界 1.5） |
| 2.3–3.0 / 3.8–10 | ~5 | 其余中间带模型 | 非主力尾部，需按模型单独归因 |
| 3.0–3.8 | 6 | int8 rec 三件 3.46–3.70、tiny_rec 3.57、anglenet 3.01 | 独立战线（int8/小开销） |
| >10 | 2 | server_rec 19.41、formula_encoder 13.81 | 表尾巨兽（attention 链） |

M2 口径对账：13 个重模型 6 个达标（≤1.5），缺口 ≈ 重模型 p50 从 1.63
压到 1.5（−8%）。M3 口径：远未达，走 P7 附注的降级路径。

### 8.2 差距源 → 对策（按优先级）

**A. conv 系 1.6–2.3 带（约 20 模型，M2 收尾的关键面）**
- 归因：GEMM 本体已由 P3 微内核接管（60+ GFLOP/s），残余在 im2col
  gather 残留层（P6 支配守卫跳过的 yolov5 层）、epilogue 未融合段与
  池化/激活的中间物化；ncnn 侧的对照优势是其 pack 面板 sgemm。
- 对策（两选一，代价/收益对账后定）：
  1. **接受降级口径**（P7 附注路线）：例外清单 ≤1.25 逐模型对账
     （resnet/yolo 系大部分已落在 1.2–1.75，对账即收尾）；
  2. **Winograd pack 化复活（候选，当前未实施）**（4–6 人天）：
     pack-A/pack-B 物化 + 专用变换内核（ncnn conv3x3s1_winograd63 同款
     结构）；预期把 conv 段从 ~2× 拉向 1.3× 量级，但必须先通过代码尺寸、
     编译时间、checksum 和代表模型性能的联合门禁；P7 的变换/矩阵基建可
     复用，不确定性集中在 pack 后的 L2 行为。

**B. server_rec / formula_encoder 巨兽（19.4×/13.8×）**
- 归因：attention 链的 MHA 转置物化（P6 残余 ~16% copy）+ CTC/解码
  段的长序列 matmul + 非 conv 段（LSTM/reshape 链）未被任何内核 pass
  覆盖；ncnn 侧这批模型同样不是 F(6,3) 受益者，差距主体在 IR 形态。
- 对策：P6 遗留的 MHA splitHeads 转置消解（consumer 形态并入相邻
  matmul 映射或 fused softmax 输出直产转置布局）；CTC 段 SIGUSR2
  采样定位后按热点逐个接入内核 pass。预估 3–5 人天（不含验证）。

**C. int8 rec 三件（3.46–3.70×）**
- 归因：requant 融合已落地（本表 int8 行较 P4 收敛），残余在
  **dw-int8 层**（ncnn 的 dw-int8 走专用内核；我们的 dw 层保持 f32
  被 P5 覆盖，但 int8 模型的激活量化链在 dw 前后的 cast 未折叠）与
  通道侧 cast/gather。
- 候选审计项（当前未实施）：dw-Q 变体接入 P5 的
  `vectorizeDepthwiseConvRows`（i8 输入、i8×i8→i32 行向量 MAC，沿用
  P4 的 tier② 定档）；激活量化 cast 的 producer 折叠进 dw 输出。必须先
  补齐 named-op 类型契约、逐位 golden 和 importer 端到端证明。

**D. 小模型固定开销（anglenet 3.01×、tiny_rec 3.57×）**
- 归因：ncnn 0.8ms 级模型的 ratio 被固定开销放大——每 run 的
  malloc/dealloc 链（One-Shot Bufferize 逐 tensor 分配）+ 多 pass
  的中间物化。
- 对策（候选审计项，当前 audit-only）：评估 arena 化 / entry 级 hoist
  静态分配；在生命周期、线程安全、并发调用和收益均得到证明前不改默认
  ownership。tiny_rec 的 3.57× 还含 conv 段（并入 A 线）。

### 8.3 原计划收益预测（未作为验收结论）

下表保留原 P8 立项时的假设，P8-1/2 完成后尚未重新测量，不能视为
实际收益或 M3 结论。

| 对策线完成后 | 预期 |
|---|---|
| A1 降级口径对账 | M3 降级宣告（conv 系例外清单 ≤1.25） |
| A2 pack 化 + B + C + D | 重模型 p50 → ~1.3、中位 → ~1.2；server_rec/formula → ≤3× 量级 |

预测过 §1 的 0.8 传导折扣；验收以实测为准。建议排序：B（单点收益
最大）→ C（独立且便宜）→ D → A2（可选，决定 M3 走标准还是降级）。

### 8.4 P8 实际结论（2026-09-13）

P8-1 的生产改动只放宽了错误的 eligibility 门控，未改变分块内核的
寄存器预算；P8-2 为 raw memref batch matmul 增加 N=2049 的 bounded-panel 回归，
并为 N=3 的静态 self-attention pipeline 增加 batch-kernel smoke guard；
dynamic self-attention 仍按设计保留 generic fallback。该阶段没有新增
server_rec 长 logits 数值覆盖、splitHeads/key/context transpose 对照或
6T ratio 重测，因此只构成 pipeline/regression guard，不是长序列 MHA 的
端到端性能验收。两阶段均通过完整回归，但不能从 441/441 ctest 的通过率
推导性能收益。

实际达成项是：长 N 不再被 1024 门控误拒、静态 MHA 的既有 batch kernel
有明确守护、动态 MHA/layout transpose 没有被未经证明地改写。未达成项是
M3 标准口径、int8 depthwise、multiplier>1 depthwise、arena 复用以及
pack 化；它们均保留为 audit/no-go，而不是默认生产特性。后续若重新开启
其中任一项，必须先补齐数值 golden、类型/生命周期证明、编译预算和新的
全量 ratio 基线。
