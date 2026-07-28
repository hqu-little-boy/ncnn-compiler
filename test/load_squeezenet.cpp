// ncnn-graph 第一阶段验收：读 squeezenet，dump 计算图 + 验证拓扑/权重。
#include "ncnn_graph/graph.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <format>
#include <iostream>
#include <string>

TEST(LoadSqueezenet, UnknownTensorDump) {
  ncnn_graph::Tensor unknown_tensor;
  auto unknown_shape = unknown_tensor.set_shape({1});
  EXPECT_TRUE(unknown_shape.has_value())
    << "unknown tensor accepts valid shape";
  EXPECT_EQ(unknown_tensor.get_dtype(), ncnn_graph::DataType::Unknown)
    << "default tensor dtype = unknown";
  EXPECT_EQ(unknown_tensor.byte_size(), 0u) << "unknown tensor byte size = 0";

  ncnn_graph::Layer unknown_layer;
  unknown_layer.set_type("Constant");
  unknown_layer.set_name("unknown_weight");
  unknown_layer.add_weight(std::move(unknown_tensor));
  ncnn_graph::Graph unknown_graph;
  unknown_graph.add_layer(std::move(unknown_layer));
  std::string unknown_dump = unknown_graph.dump();
  EXPECT_NE(unknown_dump.find(":unknown:"), std::string::npos)
    << "unknown tensor dump uses unknown dtype";
  EXPECT_EQ(unknown_dump.find(":i8:"), std::string::npos)
    << "unknown tensor dump is not mislabeled as i8";
}

TEST(LoadSqueezenet, LoadsAndValidates) {
  const std::string param =
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.param";
  const std::string bin =
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.bin";

  auto graph_result = ncnn_graph::Graph::load(param, bin);
  ASSERT_TRUE(graph_result) << graph_result.error();

  const auto& graph = *graph_result;
  std::cout << graph.dump();

  auto layers = graph.get_layers();
  EXPECT_EQ(layers.size(), 75u)
    << std::format("layer count = {} (expect 75)", layers.size());
  EXPECT_EQ(layers.size(), 26u + 26 + 8 + 8 + 4 + 1 + 1 + 1)
    << "operator counts match squeezenet (26 conv + 26 relu + 8 split + 8 "
       "concat + 4 pool + 1 softmax + 1 input + 1 dropout)";
  EXPECT_EQ(graph.layer_count_of("Convolution"), 26u)
    << std::format("conv = {} (expect 26)",
                   graph.layer_count_of("Convolution"));
  EXPECT_EQ(graph.layer_count_of("ReLU"), 26u)
    << std::format("relu = {} (expect 26)", graph.layer_count_of("ReLU"));
  EXPECT_EQ(graph.layer_count_of("Split"), 8u)
    << std::format("split = {} (expect 8)", graph.layer_count_of("Split"));
  EXPECT_EQ(graph.layer_count_of("Concat"), 8u)
    << std::format("concat = {} (expect 8)", graph.layer_count_of("Concat"));
  EXPECT_EQ(graph.layer_count_of("Pooling"), 4u)
    << std::format("pool = {} (expect 4)", graph.layer_count_of("Pooling"));
  EXPECT_EQ(graph.layer_count_of("Softmax"), 1u)
    << std::format("softmax = {}", graph.layer_count_of("Softmax"));
  EXPECT_EQ(graph.layer_count_of("Input"), 1u)
    << std::format("input = {}", graph.layer_count_of("Input"));

  auto input_blob_names = graph.get_input_blob_names();
  EXPECT_TRUE(input_blob_names.size() == 1 && input_blob_names[0] == "data")
    << "input blob = data";
  auto output_blob_names = graph.get_output_blob_names();
  EXPECT_FALSE(output_blob_names.empty())
    << std::format("output blobs = {}", output_blob_names.size());
  EXPECT_TRUE(graph.get_weights_loaded())
    << std::format("weights_loaded = {} (bin fully consumed & verified)",
                   graph.get_weights_loaded());

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
  EXPECT_TRUE(conv1_ok) << "conv1 weights = weight[64,3,3,3] f32 + bias[64]";
}
