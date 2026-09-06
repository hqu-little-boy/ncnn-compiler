# ncnn 性能追平计划（v2）

> **v2 修订（2026-09-05，计划评审后）**：
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
>   body 由 linalg 自动生成，contract 注入留待 P3/P5 内核工作时一并处理。
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
>   FMA 是否为诱因待 P2 一行回退 A/B 排查。

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
>   （vector<3> 宽度教训）；multiplier≠1 / 动态 shape 保持标量留 P8。
> - asm 断言：formula_encoder vfmadd 564（P1 后）→ **3072**、
>   vmovups 23965（行向量拷贝），server_det_static 3808；对照
>   resnet18（无 depthwise）vfmadd 444 不变——无附带代码形变。
> - lit：新增 `depthwise-vectorize.mlir`（2 正向 + multiplier≠1 /
>   窄通道 / 动态 shape 3 负向守护），136/136。
> - **全量实测（全新 /tmp 构建目录，437/437 过含 per-class 门禁，
>   fixture 编译 ≤ 300s 零告警）**：对 P2 基线 **43/44 模型 ratio
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

> 目标：44 模型 performance_tests 套件编译产物全面追平 vendored ncnn
> （x86-64 AVX2/FMA 口径）。根因依据见姊妹篇
> [`ncnn-performance-gap-analysis.md`](ncnn-performance-gap-analysis.md)
> （2026-09-03 定案：并行无罪、内核有罪——零 FMA、向量化名单=最差名单、
> int8 无 VNNI/pass 覆盖、搬运:算术 19:1–80:1）。
> 本文按根因给出阶段化路线、验收口径与风险。

---

## 1. 目标与验收口径

**"追上"的定义（分级）**，ratio = compiled_mean / ncnn_mean（6 线程正式口径）：

| 里程碑 | 覆盖 | 门禁 |
|---|---|---|
| M1（P1–P3 后） | 常规 conv 网 | 全部 ≤ 2.0，中位数 ≤ 1.5 |
| M2（P4–P6 后） | 全表 | 重模型（ncnn ≥ 100ms）≤ 1.5，中位数 ≤ 1.3 |
| M3（P7–P8 后） | 全表 | 重模型 ≤ 1.1，轻模型 ≤ 1.25（含计时噪声带），中位数 ≤ 1.0 |
| M3 降级口径 | 全表 | 仅当 P7 预算否决命中主力重模型时启用：启用 Winograd 的重模型 ≤ 1.1，预算未过的重模型 ≤ 1.25 并进例外清单逐模型对账，中位数 ≤ 1.0 不变。M3 宣告时必须声明走标准口径还是降级口径 |

- 门禁落地：`NCNN_PERF_MAX_RATIO` 升级为按类阈值表（P0），ctest 内强制，
  轻模型阈值宽、重模型严（与 baseline-report §4 的 cv 观察一致）。
- **编译时长预算**：单 fixture `clang -x ir -O3` 段 ≤ 5 分钟、整包重建
  不劣于当前（防止修复劣化重蹈"编译爆炸名单"覆辙）。P0 起在 fixture
  构建记录编译耗时并在超预算时告警。
- 每阶段验收固定三件套：① 全量数值黄金 ctest 绿（预算不放宽）；
  ② `NCNN_PERF_JSON` 全量表重测并更新 baseline-report；③ 静态 asm
  抽查（objdump 断言目标指令出现/消失）。
- 第四件（v2 增补）：验收时对 3 个代表模型补 1T 抽测复核并行扩展。
  P1 实测校准：内核收益向 6T ratio 的传导折扣约 0.8（resnet34 预测
  1.3–2×、实测 1.28×）——后续里程碑的 6T 预测与门禁评估一律先过该
  折扣，不得把单核收益线性外推到墙钟。

## 2. 现状锚点（2026-09-03）

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
- `VectorizeNCNN`/`FuseLinalgEpilogue` 生成的累加型 arith op 链：设置
  arith fastmath `<contract>`（MLIR 21 arith 原生支持），保留 LLVM 合约
  自由度；仅限累加路径，逐元素单舍入 op 不动；
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
- **主修路线**：matmul 形态改走 **canonical tiling + `vector.contract`**
  （`scf::tileUsingSCF` 切 M/N tile → vector.contract；A3 的
  TileMatmulForall/ForallizeDisjointTileLoops 基建已就绪），其 LLVM
  下降产出规整紧凑代码，绕开行向量化直线爆炸；行向量化保留给逐元素
  generic。
- **备选路线**：保留 A1b 手写内核 + 收敛行宽/unroll 上界（按 spike 结论
  定界），P1 的 vector.fma 直接受益。
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

根因：ncnn 对 3×3 s1 默认 Winograd（resnet/yolo/det 类主力），本工具链
仅留开关位。M3 目标下 conv 系从 1.3–2× 再往 1.1× 走基本绕不开。

- `strategy-ncnn` 落地 `winograd`：F(6×6,3×3)（ncnn 同款）——输入/权重
  变换为 generic（可向量化），中心 matmul 复用 P3 微内核，逆变换同；
- dispatch 启发式编译期化：3×3 s1 且 OC·IC 超阈值（复刻 ncnn
  convolution.winograd 判据），CLI `--conv-strategy=winograd` 保持默认
  off，预算验证通过后翻 auto 默认；
- **数值预算先行**：F(6,3) 误差上界逐模型对账（复用 golden 预算机制，
  参考 ExpandStridedMetadata 教训——预算过不了就只对宽松预算模型启用，
  不做全局默认）。
- **失败退路（v2）**：若预算否决命中 resnet/yolo/det 主力重模型，
  M3 转降级口径（§1）——该批模型进例外清单（≤ 1.25）+ 逐模型对账
  报告作为本阶段产出；不得为过预算放宽 golden 或降低变换精度。

验收：预算内模型 auto 走 Winograd，resnet18/yolov5 系 3×3 段 ≥ 1.5×
于 P3 后水平；golden 全绿且误差在已对账预算内。

### P8 收尾与追平宣告（2–3 天）

- 小模型固定开销审计：每 run 的 malloc/free 链（One-Shot Bufferize +
  deallocation 逐 tensor 分配）——arena 化/entry 级 hoist 评估，
  anglenet 级亚 5ms 模型轻模型门禁的最后一公里；
- per-class 阈值收紧到 M3 口径（或 §1 降级口径），门禁转强制；
- baseline-report 全量重测更新、gap-analysis 附修复后对照；
- 文档沉淀：support-status §6 优化矩阵更新，本计划标注完成态。

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
| vector.contract 路线在 clang -O3 仍爆炸（P2 主路线失效） | Spike 先行定界；备选路线 A1b+unroll 上界；编译耗时预算门禁兜底（v2 对定界慢件硬化为硬失败） |
| 编译爆炸面大于 11 模型名单（efficientnet 系名单外 1–2h/件实测） | P2 spike 扩围定界（含 FMA 一行回退 A/B）；超 300s fixture 数收敛目标 ≤ 3 |
| 内核收益向 6T ratio 传导打折（多核下内存带宽共享上限） | 每阶段验收附 1T 抽测复核并行扩展；6T 预测一律过 ~0.8 传导折扣（P1 校准） |
| M3 依赖 Winograd 而预算否决命中主力重模型 | §1 M3 降级口径：例外清单 ≤ 1.25 + 逐模型对账；不为过预算放宽 golden |
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
