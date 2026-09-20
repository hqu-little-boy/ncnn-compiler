# P18 perf stat 解释性证据

采集对象：P18 absolute-excess top five（`pp_ocrv5_server_rec`、`pp_ocrv5_server_det_static`、`pp_formulanet_plus_s_encoder`、`yolov5x_seg`、`yolov5x`）和两个对照（`resnet18`、`squeezenet_v1_1`）。每个对象使用 fresh stage 的单模型 CTest，`NCNN_PERF_THREADS=6`、warmup=2、iterations=3；这是解释性采样，不是正式 ratio 门禁。

事件请求：`cycles,instructions,branches,branch-misses,cache-references,cache-misses,dTLB-load-misses,context-switches,page-faults`。

本机是 hybrid CPU；`perf stat` 将部分事件展开为 `cpu_atom/*` 与 `cpu_core/*`，未计数或不可用项保留原始 `<not counted>`/空值语义，不能跨 CPU 汇总或解释为完整硬件流量。原始 CSV 和每个模型的 CTest 日志均保留在本目录。正式性能结论仍以 `candidate-end-to-end.ndjson` 为准。
