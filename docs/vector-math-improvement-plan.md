# 向量化数学后端改造计划（libmvec / SLEEF）

> 状态：v1 执行中。目标：消除 `convert-math-to-libm` 对向量 math op 的逐 lane
> 标量化，使行级向量化产物中的超越函数真正获得向量库调用。
>
> 修订：v1 —— 新增 `--vector-math={auto,libmvec,sleef,none}`（默认 auto）与
> `lower-vector-math-ncnn` pass；SLEEF 以 vendored 静态档案接入。

## 1. 现状问题

| # | 偏置点 | 位置 | 问题 |
|---|---|---|---|
| 1 | MathToLibm 逐 lane 标量化 | 上游 `convert-math-to-lim` | `vector<227xf32>` exp → 227 个内联标量调用；`vector<8xf32>` → 8 组 extract/call/insert |
| 2 | 行向量化提升 math op | lib/Transforms/VectorizeNCNN/VectorizeNCNN.cpp | `isLiftableElementwise` 将 math op 提升为向量语义后，伪 SIMD 静默产生 |
| 3 | 白名单不设防 | tools/ncnn-compile.cpp 未定义符号审计 | expf/tanhf 本就在 allowlist，标量化损失完全不可见 |

## 2. 设计原则

- **编译期选择，无运行时分发**。部署环境错位由构建方以 `--sysroot`
  声明并重编解决；编译器不为"目标支持但部署机更旧"兜底。
- **点名即兑现**。显式 `--vector-math=libmvec` 而探测不可用 → 报错退出；
  `auto` 探测失败才静默降级 SLEEF，再退 none。
- **导出面零增量**。libmvec 走未定义符号白名单条件放行（仿 `__kmpc_`）；
  SLEEF 静态档案经 version script 保持 local，导出审计不变。
- **覆盖面诚实**。erfcf 无任何向量变体、scalable/f64/bf16/奇数尾宽保持
  标量 MathToLibm 兜底；文档发布覆盖表而非声称全覆盖。

## 3. 后端能力矩阵（实测基线）

| 后端 | 函数覆盖 | ISA 变体 | 目标机门槛 |
|---|---|---|---|
| glibc libmvec | expf/logf/powf/sinf/cosf/tanhf/erff/expm1f… | `_ZGV{b,c,d,e}N{4,8,8,16}v_*`（x86）；`_ZGVnN4v_*`（AArch64 NEON） | tanh/erf x86≥GLIBC_2.35、ARM≥2.40；musl 无 |
| vendored SLEEF 3.6.1（静态检入 third_party/sleef） | expf/logf/powf/tanhf/erff × 宽度 {1,4,8,16} | `Sleef_<基名><宽度>_u10`，入口内部自带运行时 ISA 分发（SIGILL 探测+首调用缓存） | 无（静态编入产物；分发器需 libc 的 clock_gettime/posix_memalign，审计已放行） |

## 4. 阶段规划

### Phase M0：测量（0.5 天）
dump squeezenet 与算子夹具的 memref/llvm 层 IR，统计 math op 向量宽度分布
与标量化 call site 数量。

验收：
1. 宽度分布表落入 §6 度量记录；
2. 据此裁定超限整行（>4×lanes）循环分块是否进入 v1。

### Phase M1+M2：`lower-vector-math-ncnn` pass 与管线选项（2.5 天）
新 pass：ABI 等宽 f32 向量整体替换为库调用；lanes 整数倍且 ≤4×lanes 展开
slice/call/insert 链；其余形态不动。管线选项 `vector-math/vector-math-abi/
vector-math-lanes` 贯通，插入点在 SCFToControlFlow 之前。

验收：
1. lit：exact-width/binary-powf/unroll-chunks/untouched/disabled/sleef 全绿；
2. 端到端管线测试显示 mangled call 且无残留向量 math；
3. 默认（none）路径产物与历史行为一致。

### Phase M3：CLI 解析 + 五符号探针 + 门禁联动（1.5 天）
`--vector-math` 解析与 libmvec 探针（引用全部五个符号，`-Wl,-z,defs -lm`
强制解析）；manifest 增 `vector_math` 字段；NEEDED/未定义白名单按模式启用；
SLEEF 档案定位链与链接注入。

验收：
1. 显式模式对缺失 sysroot 报错退出；auto 静默降级有 info 输出；
2. ncnn-compile-cli 与 squeezenet-shared-library 测试双模式绿；
3. static_artifact_baseline.json 与 manifest 字段同提交更新。

### Phase M4：SLEEF 源码检入与链接集成（已完成，按用户决策改检入制）
GitHub git 协议在开发环境不可达，经确认改为源码检入：third_party/sleef
（3.6.1 精简树）。CMake 裁剪集成（共享库/测试/DFT/GNUABI 全关），单静态档
案 `libsleef.a`（分发器内置）经 `$<TARGET_FILE_DIR:sleef>` 注入定位链，
并随 install 发布至 `<prefix>/lib`。

验收：
1. auto 在无 libmvec 目标上自动选中 SLEEF；
2. sleef 产物导出面/DT_NEEDED 与现状一致；
3. 抽查 u10 精度 ≤1 ulp。

### Phase M5：数值回归 + 文档（1.5 天）
Numerical golden 扩展 exp/tanh/erf/log/pow × {none,auto,libmvec,sleef}；
本文档定稿；ncnn-compile-command-line.md 与 support-status 矩阵更新。

验收：
1. golden 全量通过（libmvec ~4 ulp、SLEEF u10 ≤1 ulp 均在容差内）;
2. 文档包含部署下限表（libmvec tanh/erf x86≥2.35 / ARM≥2.40；musl 无）。

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 整行宽主导真实模型 → v1 展开链不命中 | M0 先测；证实则升格循环分块 |
| 旧 glibc 主机运行时缺 libmvec | 探针覆盖最新符号；部署责任归构建方并写入文档 |
| SLEEF 交叉编译脆弱 | v1 仅 native；交叉显式报错引导换后端 |
| 基线漂移 | manifest 字段与基线 JSON 同提交 |
| erfcf 等残留标量化 | 文档覆盖表明示 |

## 6. 度量记录

M0 测量（x86_64 基线变体 b4/lanes=4，--threads=4 --vector-mode=auto）：

| 模型/夹具 | math op | 到达 pass 的向量宽度 | 处理路径 |
|---|---|---|---|
| sigmoid（1×2×2 夹具） | exp ×1 | 行宽 2 < lanes | 标量兜底（现状保持） |
| sigmoid8（1×8×8 自制） | exp ×1 | 行宽 8 = 2×lanes | 展开链 → `U _ZGVbN4v_expf@GLIBC_2.22`，DT_NEEDED 自动挂 libmvec.so.1 |
| softmax（3×4×3 夹具） | exp ×1（归约 generic 内） | 不被行向量化 | 标量兜底（既有行为，非本次回归） |

结论：命中条件 = 纯逐元层且内维宽与 lanes 成整数倍；归约型（softmax/logsum）
与大整行（>4×lanes）维持现状。循环分块与归约体向量化列为后续迭代。

