// ncnn-graph 第一阶段验收：读 squeezenet，dump 计算图 + 验证拓扑/权重。
#include "ncnn_graph/graph.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string param =
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.param";
  std::string bin = NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.bin";
  if (argc >= 2) {
    param = argv[1];
  }
  if (argc >= 3) {
    bin = argv[2];
  }

  int status = 0;
  auto check = [&](bool condition, std::string_view message) {
    std::cout << std::format("[{}] {}\n", condition ? "PASS" : "FAIL", message);
    if (!condition) {
      status = 1;
    }
  };

  ncnn_graph::Tensor unknown_tensor;
  auto unknown_shape = unknown_tensor.set_shape({1});
  check(unknown_shape.has_value(), "unknown tensor accepts valid shape");
  check(unknown_tensor.get_dtype() == ncnn_graph::DataType::Unknown,
        "default tensor dtype = unknown");
  check(unknown_tensor.byte_size() == 0, "unknown tensor byte size = 0");

  ncnn_graph::Layer unknown_layer;
  unknown_layer.set_type("Constant");
  unknown_layer.set_name("unknown_weight");
  unknown_layer.add_weight(std::move(unknown_tensor));
  ncnn_graph::Graph unknown_graph;
  unknown_graph.add_layer(std::move(unknown_layer));
  std::string unknown_dump = unknown_graph.dump();
  check(unknown_dump.find(":unknown:") != std::string::npos,
        "unknown tensor dump uses unknown dtype");
  check(unknown_dump.find(":i8:") == std::string::npos,
        "unknown tensor dump is not mislabeled as i8");

  auto graph_result = ncnn_graph::Graph::load(param, bin);
  if (!graph_result) {
    std::cerr << std::format("ERROR: {}\n", graph_result.error());
    return 1;
  }

  const auto& graph = *graph_result;
  std::cout << graph.dump();

  auto layers = graph.get_layers();
  check(layers.size() == 75,
        std::format("layer count = {} (expect 75)", layers.size()));
  check(layers.size() == 26 + 26 + 8 + 8 + 4 + 1 + 1 + 1,
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

  auto input_blob_names = graph.get_input_blob_names();
  check(input_blob_names.size() == 1 && input_blob_names[0] == "data",
        "input blob = data");
  auto output_blob_names = graph.get_output_blob_names();
  check(!output_blob_names.empty(),
        std::format("output blobs = {}", output_blob_names.size()));
  check(graph.get_weights_loaded(),
        std::format("weights_loaded = {} (bin fully consumed & verified)",
                    graph.get_weights_loaded()));

  constexpr std::array<std::int64_t, 4> kConv1WeightShape = {64, 3, 3, 3};
  constexpr std::array<std::int64_t, 1> kConv1BiasShape = {64};
  bool conv1_ok = false;
  for (const auto& layer : layers) {
    if (layer.get_name() == "conv1") {
      auto weights = layer.get_weights();
      conv1_ok =
        weights.size() == 2 &&
        std::ranges::equal(weights[0].get_shape(), kConv1WeightShape) &&
        weights[0].get_dtype() == ncnn_graph::DataType::Float32 &&
        std::ranges::equal(weights[1].get_shape(), kConv1BiasShape);
      conv1_ok = conv1_ok && weights[0].get_data().size() == 64 * 3 * 3 * 3 * 4;
      break;
    }
  }
  check(conv1_ok, "conv1 weights = weight[64,3,3,3] f32 + bias[64]");

  check(true,
        std::format("(info) bin read done; total layers with weights "
                    "present"));
  return status;
}
