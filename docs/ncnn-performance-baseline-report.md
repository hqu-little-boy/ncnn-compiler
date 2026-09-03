# ncnn 性能对比基线报告（2026-08-28）

编译产物与 upstream ncnn 的运行时墙钟对比基线。测量设施为 `test/Numerical/`
的 `performance_tests`（performance 标签，提交 `e38a823`/`0a21650`），方法论、
环境变量与运行方式见 [`test/Numerical/README.md`](../test/Numerical/README.md)
的性能基准一节；本文记录首轮全量基线数据与结论，供后续优化回归对照。

## 1. 测量环境与口径

- 构建：Release（`compiler/build`，Unix Makefiles）。**Debug 构建下 vendored
  ncnn 参考为 `-O0`，加速比严重虚高，本文数据不适用于 Debug**。
- 机型：x86_64，`ncnn::get_physical_big_cpu_count()` = 6（物理大核），
  双侧统一该线程数（`set_cpu_powersave(2)` + `set_omp_dynamic(0)`）。
- OpenMP runtime 事实：vendored ncnn（g++ 构建）链 **libgomp**，编译产物
  （clang-21）链 **libomp**，进程内两个独立 runtime；线程控制依赖
  `ncnn::set_omp_num_threads`（ncnn 侧）+ `OMP_NUM_THREADS` 环境变量
  （libomp 在 dlopen 初始化时读取）两条路径。`NCNN_PERF_THREADS=1` 与默认
  的双侧缩放均已实证有效。
- 参考侧 opt 与数值金标逐字一致（FP32 或 Int8 模式），每次迭代新建
  Extractor，开销计入上游运行时成本（与 vendored `benchncnn.cpp` 计时体
  一致）。迭代策略：重模型（输入 ≥ 640x640x3）5 次预热 + 10 次计时，其余
  10 + 20；报告均值，`cv` 为两侧较大变异系数。
- vendored ncnn：`a4d2ea1d`，x86_64 fma dispatch。
- 数据来自两轮测量：27 个模型（2026-08-27，ctest + NDJSON）与 19 个模型
  （2026-08-28，直跑二进制）；这两轮是 **P0 之前的参考基线**（模型 fixture
  默认 `--vector-mode=off`，未声明目标 ISA，编译产物主要为 SSE2 宽度），跨轮
  同模型波动约 ±10%（如 squeezenet 4.50 → 4.01），下表为各轮实测值。
- P0 已在 `test/Numerical/CMakeLists.txt` 落地并成功重建：普通模型 fixture 统一
  使用 `--vector-mode=fixed-width --target-feature=+avx2,+fma --clang-arg=-march=x86-64-v3`
  （8-lane，匹配本机 i5-12600KF 的 AVX2/FMA 能力）。`pp_ocr` rec、SLANet CNN、
  FormulaNet encoder 因显式行向量化触发 clang -O3 小时级编译爆炸，关闭 MLIR 显式
  vector mode 但仍传入相同 x86-64-v3 ISA；这些模型的旧表值仍有效但不代表 P0 后
  统一行向量化口径。
- P0 完成后的快速全量重测（1 次预热 + 1 次计时，非正式稳定统计）已覆盖 44 个
  可执行模型，全部通过；另外 2 个 PP-LCNet Int8 用例因 upstream ncnn 参考在
  `gemm_transB_packed_tile_int8_avxvnni` 崩溃而显式 SKIP（独立复现与四个 det_int8
  的情况一致，详见 `ncnn-suspected-issues.md` §4）。普通模型使用 fixed-width+v3，
  11 个编译爆炸例外使用 scalar MLIR+v3。代表值：resnet18 12.7→48.8 ms（ratio
  3.83），yolov5n 42.4→95.5 ms（2.25），yolov5x 481.5→1590.6 ms（3.30）；例外
  中的 formula encoder 63.3→5754.7 ms（90.9），server_rec 464.9→20812.1 ms
  （44.8）。1/1 样本只确认可运行，不用于稳定性或门禁结论；后续需按默认 10/20
  或 5/10 迭代重新采正式表。

## 2. 覆盖范围

46 个静态形状模型，全部单输入单入/单出（yolov5_seg 为双输出）。
明确不在册：多输入模型（SLAHead、FormulaNet embed/decoder）、dynamic 系
fixture、fp16/bf16 精度孪生产物、operators 单算子；det_int8 的
tiny/small/mobile 三行因 upstream ncnn 参考崩溃缺席（见
[`ncnn-suspected-issues.md`](ncnn-suspected-issues.md) 第 4 节，独立复现工程
`../ncnn-det-int8-crash-repro/`）。int8 行双侧同 Int8 模式计时。

## 3. 基线数据

ratio = compiled_mean / ncnn_mean，旧基线中**全部 > 1（46/46 慢于 ncnn）**。
P0 后普通模型已改为 x86-64-v3 公平口径；下表旧数字仅用于保存 P0 前的完整
可复现历史，不能与 P0 后单次重测直接作严格前后差值。

### 3.1 分类网络（FP32）

| 模型 | 输入 | ncnn ms | compiled ms | ratio | cv |
|---|---|---:|---:|---:|---:|
| squeezenet_v1_1 | 227x227x3 | 4.9 | 21.8 | 4.50 | 0.20 |
| resnet18 | 224x224x3 | 10.7 | 49.5 | 4.64 | 0.16 |
| resnet34 | 224x224x3 | 20.8 | 88.5 | 4.25 | 0.12 |
| resnet50 | 224x224x3 | 24.8 | 93.8 | 3.78 | 0.11 |
| resnet101 | 224x224x3 | 43.7 | 167.0 | 3.82 | 0.13 |
| yolov5n_cls | 224x224x3 | 3.0 | 9.0 | 3.00 | 0.06 |
| yolov5s_cls | 224x224x3 | 7.1 | 18.7 | 2.63 | 0.09 |
| yolov5m_cls | 224x224x3 | 14.0 | 54.7 | 3.90 | 0.18 |
| yolov5l_cls | 224x224x3 | 23.3 | 116.6 | 5.02 | 0.12 |
| yolov5x_cls | 224x224x3 | 40.5 | 184.9 | 4.56 | 0.09 |
| efficientnet_b0 | 224x224x3 | 8.1 | 27.4 | 3.40 | 0.13 |
| efficientnet_b1 | 240x240x3 | 16.3 | 58.5 | 3.60 | 0.14 |
| efficientnet_b2 | 260x260x3 | 16.9 | 72.2 | 4.28 | 0.15 |
| efficientnet_b3 | 300x300x3 | 29.0 | 126.0 | 4.35 | 0.15 |
| pp_lcnet_x1_0_doc_ori | 224x224x3 | 3.0 | 9.9 | 3.24 | 0.11 |
| pp_lcnet_x1_0_textline_ori | 160x80x3 | 3.4 | 10.7 | 3.15 | 0.08 |
| chineseocr_lite_anglenet | 192x32x3 | 0.74 | 3.1 | 4.18 | 0.05 |

### 3.2 检测/分割（FP32，640x640）

| 模型 | 输出 | ncnn ms | compiled ms | ratio | cv |
|---|---|---:|---:|---:|---:|
| yolov5n | 25200x85 | 44.9 | 81.8 | **1.82** | 0.06 |
| yolov5s | 25200x85 | 79.0 | 205.8 | 2.61 | 0.12 |
| yolov5m | 25200x85 | 153.3 | 513.3 | 3.35 | 0.11 |
| yolov5l | 25200x85 | 266.3 | 1239.1 | 4.65 | 0.07 |
| yolov5x | 25200x85 | 435.1 | 1904.0 | 4.38 | 0.06 |
| yolov5n_seg | proto+dets | 57.1 | 123.4 | 2.16 | 0.10 |
| yolov5s_seg | proto+dets | 114.9 | 288.1 | 2.51 | 0.08 |
| yolov5m_seg | proto+dets | 204.3 | 662.5 | 3.24 | 0.09 |
| yolov5l_seg | proto+dets | 351.4 | 1608.6 | 4.58 | 0.08 |
| yolov5x_seg | proto+dets | 539.5 | 2328.5 | 4.32 | 0.03 |

### 3.3 PP-OCR 识别（320x48x3 → 40 帧 x 类别）

| 模型 | 类别数 | ncnn ms | compiled ms | ratio | cv |
|---|---:|---:|---:|---:|---:|
| pp_ocrv6_tiny_rec | 6906 | 2.9 | 11.7 | 4.08 | 0.19 |
| pp_ocrv6_tiny_rec_int8 | 6906 | 2.6 | 27.6 | **10.8** | 0.13 |
| pp_ocrv5_mobile_rec | 18385 | 21.9 | 55.6 | 2.54 | 0.14 |
| pp_ocrv5_mobile_rec_int8 | 18385 | 20.0 | 85.8 | **4.28** | 0.18 |
| pp_ocrv5_server_rec（attention） | 18385 | 446.2 | 19545.9 | **43.8** | 0.01 |
| pp_ocrv6_medium_rec | 18710 | 34.7 | 168.0 | 4.84 | 0.15 |
| pp_ocrv6_medium_rec_int8 | 18710 | 21.4 | 607.7 | **28.4** | 0.15 |
| pp_ocrv6_small_rec | 18710 | 13.1 | 43.9 | 3.35 | 0.16 |
| pp_ocrv6_small_rec_int8 | 18710 | 8.5 | 94.6 | **11.1** | 0.07 |

### 3.4 PP-OCR 检测 / 结构 / 公式

| 模型 | 输入 | ncnn ms | compiled ms | ratio | cv |
|---|---|---:|---:|---:|---:|
| pp_ocrv6_tiny_det | 640x640x3 | 46.8 | 125.9 | 2.69 | 0.10 |
| pp_ocrv6_small_det | 640x640x3 | 101.7 | 297.0 | 2.92 | 0.10 |
| pp_ocrv6_medium_det | 640x640x3 | 422.6 | 1266.9 | 3.00 | 0.09 |
| pp_ocrv6_medium_det_int8（Int8 双侧） | 32x32x3 | 3.7 | 27.4 | **7.38** | 0.08 |
| pp_ocrv5_mobile_det_static | 640x640x3 | 112.8 | 252.8 | 2.24 | 0.10 |
| pp_ocrv5_server_det_static | 640x640x3 | 4654.6 | 42461.5 | **9.12** | 0.01 |
| pp_structrurev2_slanet_plus_cnn | 488x488x3 | 42.8 | 158.6 | 3.70 | 0.16 |
| pp_formulanet_plus_s_encoder | 384x384x1 | 64.3 | 5591.8 | **86.9** | 0.10 |
| pp_lcnet_x1_0_doc_ori_int8（Int8 双侧） | 224x224x3 | 6.3 | 29.6 | 4.69 | 0.13 |
| pp_lcnet_x1_0_textline_ori_int8（Int8 双侧） | 160x80x3 | 13.6 | 42.9 | 3.15 | 0.14 |

## 4. 结论

1. **46/46 全部慢于 ncnn**。常规卷积网整体落在 1.8–5.0 区间；yolov5n 检测
   最接近（1.82），yolov5l_cls / resnet18 一类在 4.6–5.0。
2. **劣化热点按严重度分层**（与 A3 向量化/OpenMP 落地后对本机 ctest 时长
   的既有观察方向一致，见 `perf(ncnn)` 提交序列背景）：
   - attention 类：formula_encoder 86.9x、server_rec 43.8x —— 首要优化对象；
   - int8 内核：medium_rec_int8 28.4x、small/tiny_rec_int8 与
     medium_det_int8 7.4–11.1x；
   - 大检测头：server_det_static 9.1x（单次 42.5s）；
   - 常规卷积：2–5x。
3. **int8 产物普遍比自身 FP32 版更慢**（tiny_rec 4.08→10.8、medium_rec
   4.84→28.4、small_rec 3.35→11.1），量化内核未兑现应有收益，是独立于
   attention 问题的第二热点。
4. 重模型（yolov5x/seg、server 系）的 `cv` 多在 0.01–0.10，数据稳定；
   亚 5ms 模型 `cv` 可到 0.2，门禁判定应以重模型为主、轻模型阈值放宽。

## 5. 后续工作

- 差距根因的分层归因与修复指向见姊妹篇
  [`ncnn-performance-gap-analysis.md`](ncnn-performance-gap-analysis.md)
  （2026-09-03：FMA 缺失、向量化名单重合、int8 无 VNNI 等证据）；
  追平路线图见 [`ncnn-performance-parity-plan.md`](ncnn-performance-parity-plan.md)。
- 以本文为基线，按 `NCNN_PERF_MAX_RATIO` 门禁做优化回归（阈值待各代表机型
  采集后定，建议轻模型从 6.0 起、重模型从 2.0 起试点）。
- attention 路径与 int8 计算内核的优化立项；优化后重跑本报告更新基线。
- upstream 修复 det_int8 参考崩溃后，补齐 tiny/small/mobile 三行（README
  覆盖一节有补行指引）。
- 单次会话 fresh 全量重测：`NCNN_PERF_JSON=<path> ctest --test-dir build
  -L performance -j1`（约 25–30 分钟，RUN_SERIAL）。
