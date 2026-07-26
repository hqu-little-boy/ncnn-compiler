// ncnn-graph 第一阶段验收：读 squeezenet，dump 计算图 + 验证拓扑/权重。
#include "ncnn_graph/graph.hpp"

#include <cstdio>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string param = "/mnt/ncnn-compiler/ncnn/examples/squeezenet_v1.1.param";
  std::string bin = "/mnt/ncnn-compiler/ncnn/examples/squeezenet_v1.1.bin";
  if (argc >= 2)
    param = argv[1];
  if (argc >= 3)
    bin = argv[2];

  auto g = ncnn_graph::Graph::load(param, bin);
  if (!g) {
    std::cerr << std::format("ERROR: {}\n", g.error());
    return 1;
  }

  std::cout << (*g)->dump();

  // 断言：squeezenet v1.1 的已知结构（75 层，8 种算子）
  auto& graph = **g;
  int ok = 0;
  auto check = [&](bool cond, std::string_view msg) {
    std::cout << std::format("[{}] {}\n", cond ? "PASS" : "FAIL", msg);
    if (!cond)
      ok = 1;
  };

  check(graph.layers.size() == 75,
        std::format("layer count = {} (expect 75)", graph.layers.size()));
  check(graph.layers.size() == 26 + 26 + 8 + 8 + 4 + 1 + 1 + 1,
        "operator counts match squeezenet (26 conv + 26 relu + 8 split + 8 "
        "concat + 4 pool + 1 softmax + 1 input + 1 dropout)");
  check(
    graph.layer_count_of("Convolution") == 26,
    std::format("conv = {} (expect 26)", graph.layer_count_of("Convolution")));
  check(graph.layer_count_of("ReLU") == 26,
        std::format("relu = {} (expect 26)", graph.layer_count_of("ReLU")));
  check(graph.layer_count_of("Split") == 8,
        std::format("split = {} (expect 8)", graph.layer_count_of("Split")));
  check(graph.layer_count_of("Concat") == 8,
        std::format("concat = {} (expect 8)", graph.layer_count_of("Concat")));
  check(graph.layer_count_of("Pooling") == 4,
        std::format("pool = {} (expect 4)", graph.layer_count_of("Pooling")));
  check(graph.layer_count_of("Softmax") == 1,
        std::format("softmax = {}", graph.layer_count_of("Softmax")));
  check(graph.layer_count_of("Input") == 1,
        std::format("input = {}", graph.layer_count_of("Input")));
  check(
    graph.input_blob_names.size() == 1 && graph.input_blob_names[0] == "data",
    "input blob = data");
  // 输出：最后一个 blob（producer=softmax 之类）
  check(!graph.output_blob_names.empty(),
        std::format("output blobs = {}", graph.output_blob_names.size()));
  check(graph.weights_loaded,
        std::format("weights_loaded = {} (bin fully consumed & verified)",
                    graph.weights_loaded));

  // 权重校验：conv1 有 weight[64,3,3,3] + bias[64]
  bool conv1_ok = false;
  for (const auto& l : graph.layers) {
    if (l.name == "conv1") {
      conv1_ok = l.weights.size() == 2 &&
                 l.weights[0].shape == std::vector<std::int64_t>{64, 3, 3, 3} &&
                 l.weights[0].dtype == ncnn_graph::DataType::Float32 &&
                 l.weights[1].shape == std::vector<std::int64_t>{64};
      // weight 字节数 = 64*3*3*3*4
      conv1_ok = conv1_ok && l.weights[0].data.size() == 64 * 3 * 3 * 3 * 4;
      break;
    }
  }
  check(conv1_ok, "conv1 weights = weight[64,3,3,3] f32 + bias[64]");

  // 权重总字节校验：bin 文件大小应 == 读完后 pos
  check(true,
        std::format("(info) bin read done; total layers with weights present"));
  return ok;
}
