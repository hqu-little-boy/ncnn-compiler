#include "ncnn_graph/graph.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/importer.hpp"
#include "ncnn_frontend/verifier.hpp"

namespace {

ncnn_graph::Tensor make_tensor(std::vector<std::int64_t> shape,
                               ncnn_graph::DataType type) {
  ncnn_graph::Tensor tensor;
  auto shaped = tensor.set_shape(std::move(shape));
  if (!shaped) {
    std::cerr << shaped.error() << '\n';
    std::terminate();
  }
  tensor.set_dtype(type);
  tensor.set_data(std::vector<std::byte>(tensor.byte_size()));
  return tensor;
}

ncnn_graph::Layer make_layer(std::string type,
                             std::string name,
                             std::vector<std::string> inputs,
                             std::vector<std::string> outputs) {
  ncnn_graph::Layer layer;
  layer.set_type(std::move(type));
  layer.set_name(std::move(name));
  layer.set_inputs(std::move(inputs));
  layer.set_outputs(std::move(outputs));
  return layer;
}

ncnn_graph::Graph make_supported_graph() {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(5));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(5));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));

  auto convolution = make_layer("Convolution", "conv", {"data"}, {"conv"});
  ncnn_graph::ParamDict convolution_params;
  convolution_params.set_value(0, ncnn_graph::ParamValue::make_int(2));
  convolution_params.set_value(1, ncnn_graph::ParamValue::make_int(3));
  convolution_params.set_value(3, ncnn_graph::ParamValue::make_int(1));
  convolution_params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  convolution_params.set_value(5, ncnn_graph::ParamValue::make_int(1));
  convolution_params.set_value(6, ncnn_graph::ParamValue::make_int(18));
  convolution.add_weight(
    make_tensor({2, 1, 3, 3}, ncnn_graph::DataType::Float16));
  convolution.add_weight(make_tensor({2}, ncnn_graph::DataType::Float32));
  convolution.set_params(std::move(convolution_params));
  graph.add_layer(std::move(convolution));

  graph.add_layer(make_layer("ReLU", "relu", {"conv"}, {"relu"}));
  graph.add_layer(make_layer("Split", "split", {"relu"}, {"left", "right"}));
  graph.add_layer(
    make_layer("Concat", "concat", {"left", "right"}, {"concat"}));

  auto pool = make_layer("Pooling", "pool", {"concat"}, {"pool"});
  ncnn_graph::ParamDict pool_params;
  pool_params.set_value(0, ncnn_graph::ParamValue::make_int(0));
  pool_params.set_value(1, ncnn_graph::ParamValue::make_int(2));
  pool_params.set_value(2, ncnn_graph::ParamValue::make_int(2));
  pool_params.set_value(5, ncnn_graph::ParamValue::make_int(0));
  pool.set_params(std::move(pool_params));
  graph.add_layer(std::move(pool));

  graph.add_layer(make_layer("Dropout", "drop", {"pool"}, {"drop"}));
  auto global = make_layer("Pooling", "global", {"drop"}, {"global"});
  ncnn_graph::ParamDict global_params;
  global_params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  global_params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  global.set_params(std::move(global_params));
  graph.add_layer(std::move(global));
  graph.add_layer(make_layer("Softmax", "prob", {"global"}, {"prob"}));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"prob"});
  graph.set_weights_loaded(true);
  return graph;
}

ncnn_graph::Graph make_int8_graph() {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(1));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));
  auto convolution = make_layer("Convolution", "conv", {"data"}, {"out"});
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  params.set_value(1, ncnn_graph::ParamValue::make_int(1));
  params.set_value(6, ncnn_graph::ParamValue::make_int(1));
  params.set_value(8, ncnn_graph::ParamValue::make_int(101));
  convolution.set_params(std::move(params));
  convolution.add_weight(make_tensor({1, 1, 1, 1}, ncnn_graph::DataType::Int8));
  convolution.add_weight(make_tensor({1}, ncnn_graph::DataType::Float32));
  convolution.add_weight(make_tensor({1}, ncnn_graph::DataType::Float32));
  convolution.add_weight(make_tensor({1}, ncnn_graph::DataType::Float32));
  graph.add_layer(std::move(convolution));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"out"});
  graph.set_weights_loaded(true);
  return graph;
}

ncnn_graph::Graph make_int8_chain_graph() {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(1));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));

  auto add_convolution = [&](std::string name,
                             std::string input_name,
                             std::string output_name,
                             std::int64_t scale_term) {
    auto convolution = make_layer("Convolution",
                                  std::move(name),
                                  {std::move(input_name)},
                                  {std::move(output_name)});
    ncnn_graph::ParamDict params;
    params.set_value(0, ncnn_graph::ParamValue::make_int(1));
    params.set_value(1, ncnn_graph::ParamValue::make_int(1));
    params.set_value(6, ncnn_graph::ParamValue::make_int(1));
    params.set_value(8, ncnn_graph::ParamValue::make_int(scale_term));
    convolution.set_params(std::move(params));
    convolution.add_weight(
      make_tensor({1, 1, 1, 1}, ncnn_graph::DataType::Int8));
    convolution.add_weight(make_tensor({1}, ncnn_graph::DataType::Float32));
    convolution.add_weight(make_tensor({1}, ncnn_graph::DataType::Float32));
    if (scale_term > 100) {
      convolution.add_weight(make_tensor({1}, ncnn_graph::DataType::Float32));
    }
    graph.add_layer(std::move(convolution));
  };
  add_convolution("requant", "data", "int8", 102);
  add_convolution("dequant", "int8", "out", 2);
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"out"});
  graph.set_weights_loaded(true);
  return graph;
}

ncnn_graph::Graph make_pool_graph(std::int64_t input_size,
                                  std::int64_t kernel,
                                  std::int64_t stride) {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(input_size));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(input_size));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));
  auto pool = make_layer("Pooling", "pool", {"data"}, {"out"});
  ncnn_graph::ParamDict params;
  params.set_value(1, ncnn_graph::ParamValue::make_int(kernel));
  params.set_value(2, ncnn_graph::ParamValue::make_int(stride));
  params.set_value(5, ncnn_graph::ParamValue::make_int(0));
  pool.set_params(std::move(params));
  graph.add_layer(std::move(pool));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"out"});
  graph.set_weights_loaded(true);
  return graph;
}

bool shape_equals(const ncnn_frontend::TensorType& type,
                  std::span<const std::int64_t> expected) {
  return std::ranges::equal(type.get_shape(), expected);
}

}  // namespace

TEST(NcnnImporterTest, ImportsSupportedGraph) {
  using ncnn_frontend::import_graph;
  using ncnn_frontend::Operation;
  using ncnn_frontend::OperationKind;
  using ncnn_frontend::verify_graph;

  auto imported = import_graph(make_supported_graph());
  ASSERT_TRUE(imported.has_value())
    << "all eight supported source layer types import";

  constexpr std::array<std::int64_t, 3> kPoolShape = {4, 3, 3};
  constexpr std::array<std::int64_t, 1> kOutputShape = {4};
  EXPECT_EQ(imported->get_operations().size(), 10u)
    << "input is graph argument and two convolution constants are ops";
  EXPECT_EQ(imported->operation_count_of(OperationKind::Constant), 2u)
    << "f16 weight and f32 bias become constants";
  const auto pool =
    std::ranges::find(imported->get_operations(), "pool", &Operation::get_name);
  EXPECT_TRUE(pool != imported->get_operations().end() &&
              shape_equals(
                imported->get_value(pool->get_results()[0]).get_type(),
                kPoolShape))
    << "pad_mode 0 uses ceil/full-padding output shape";
  EXPECT_TRUE(shape_equals(
    imported->get_value(imported->get_outputs()[0]).get_type(), kOutputShape))
    << "global pool and softmax produce rank-one output";
  EXPECT_TRUE(verify_graph(*imported).has_value())
    << "imported graph verifies";
}

TEST(NcnnImporterTest, ImportsInt8Quantization) {
  using ncnn_frontend::import_graph;
  using ncnn_frontend::Operation;
  using ncnn_frontend::OperationKind;

  auto int8 = import_graph(make_int8_graph());
  EXPECT_TRUE(int8 && int8->operation_count_of(OperationKind::Constant) == 4)
    << "int8 kernel and three quantization scales become constants";

  auto int8_chain = import_graph(make_int8_chain_graph());
  EXPECT_TRUE(int8_chain &&
              int8_chain->get_value(int8_chain->get_outputs()[0])
                  .get_type()
                  .get_element_type() == ncnn_frontend::ElementType::Float32)
    << "102 i8 output feeds term 2 input and dequantizes to f32";
  if (int8_chain) {
    const auto requant = std::ranges::find(
      int8_chain->get_operations(), "requant", &Operation::get_name);
    EXPECT_TRUE(requant != int8_chain->get_operations().end() &&
                int8_chain->get_value(requant->get_results()[0])
                    .get_type()
                    .get_element_type() == ncnn_frontend::ElementType::Int8)
      << "scale term above 100 produces i8 result";
  }

  auto unquantized_i8 = make_int8_graph();
  std::vector<ncnn_graph::Layer> unquantized_layers(
    unquantized_i8.get_layers().begin(), unquantized_i8.get_layers().end());
  ncnn_graph::ParamDict unquantized_params;
  unquantized_params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  unquantized_params.set_value(1, ncnn_graph::ParamValue::make_int(1));
  unquantized_params.set_value(6, ncnn_graph::ParamValue::make_int(1));
  unquantized_layers[1].set_params(std::move(unquantized_params));
  unquantized_layers[1].set_weights(
    {make_tensor({1, 1, 1, 1}, ncnn_graph::DataType::Int8)});
  unquantized_i8.set_layers(std::move(unquantized_layers));
  EXPECT_FALSE(import_graph(unquantized_i8))
    << "unquantized convolution rejects i8 kernel";
}

TEST(NcnnImporterTest, ShapeHints) {
  using ncnn_frontend::import_graph;

  auto hinted = make_supported_graph();
  std::vector<ncnn_graph::Layer> hinted_layers(hinted.get_layers().begin(),
                                               hinted.get_layers().end());
  ncnn_graph::ParamDict hinted_relu_params;
  hinted_relu_params.set_value(
    30, ncnn_graph::ParamValue::make_int_array({3, 5, 5, 2}));
  hinted_relu_params.set_value(31, ncnn_graph::ParamValue::make_int(1));
  hinted_layers[2].set_params(std::move(hinted_relu_params));
  hinted.set_layers(std::move(hinted_layers));
  EXPECT_TRUE(import_graph(hinted).has_value())
    << "valid param 30 shape hint and param 31 feature mask are accepted";

  auto bad_hint = make_supported_graph();
  hinted_layers.assign(bad_hint.get_layers().begin(),
                       bad_hint.get_layers().end());
  ncnn_graph::ParamDict bad_hint_params;
  bad_hint_params.set_value(
    30, ncnn_graph::ParamValue::make_int_array({3, 4, 5, 2}));
  hinted_layers[2].set_params(std::move(bad_hint_params));
  bad_hint.set_layers(std::move(hinted_layers));
  EXPECT_FALSE(import_graph(bad_hint))
    << "inconsistent param 30 shape hint is rejected";
}

TEST(NcnnImporterTest, PoolingShapes) {
  using ncnn_frontend::import_graph;

  auto large_kernel_pool = import_graph(make_pool_graph(2, 3, 2));
  constexpr std::array<std::int64_t, 3> kLargeKernelShape = {1, 2, 2};
  EXPECT_TRUE(
    large_kernel_pool &&
    shape_equals(
      large_kernel_pool->get_value(large_kernel_pool->get_outputs()[0])
        .get_type(),
      kLargeKernelShape))
    << "pad_mode 0 follows ncnn tail padding when kernel exceeds input";

  auto adaptive_graph = make_pool_graph(5, 1, 1);
  auto adaptive_layers = std::vector<ncnn_graph::Layer>(
    adaptive_graph.get_layers().begin(), adaptive_graph.get_layers().end());
  ncnn_graph::ParamDict adaptive_params;
  adaptive_params.set_value(7, ncnn_graph::ParamValue::make_int(1));
  adaptive_params.set_value(8, ncnn_graph::ParamValue::make_int(-233));
  adaptive_params.set_value(18, ncnn_graph::ParamValue::make_int(2));
  adaptive_layers[1].set_params(std::move(adaptive_params));
  adaptive_graph.set_layers(std::move(adaptive_layers));
  auto adaptive_imported = import_graph(adaptive_graph);
  constexpr std::array<std::int64_t, 3> kAdaptiveShape = {1, 2, 5};
  EXPECT_TRUE(
    adaptive_imported &&
    shape_equals(
      adaptive_imported->get_value(adaptive_imported->get_outputs()[0])
        .get_type(),
      kAdaptiveShape))
    << "adaptive -233 preserves the corresponding input dimension";

  auto global_adaptive = make_pool_graph(5, 0, 1);
  adaptive_layers.assign(global_adaptive.get_layers().begin(),
                         global_adaptive.get_layers().end());
  ncnn_graph::ParamDict global_adaptive_params;
  global_adaptive_params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  global_adaptive_params.set_value(7, ncnn_graph::ParamValue::make_int(1));
  adaptive_layers[1].set_params(std::move(global_adaptive_params));
  global_adaptive.set_layers(std::move(adaptive_layers));
  auto global_adaptive_imported = import_graph(global_adaptive);
  constexpr std::array<std::int64_t, 1> kGlobalShape = {1};
  EXPECT_TRUE(global_adaptive_imported &&
              shape_equals(
                global_adaptive_imported
                  ->get_value(global_adaptive_imported->get_outputs()[0])
                  .get_type(),
                kGlobalShape))
    << "global pooling takes priority over adaptive pooling";

  auto zero_adaptive_width = make_pool_graph(5, 1, 1);
  adaptive_layers.assign(zero_adaptive_width.get_layers().begin(),
                         zero_adaptive_width.get_layers().end());
  ncnn_graph::ParamDict zero_width_params;
  zero_width_params.set_value(7, ncnn_graph::ParamValue::make_int(1));
  zero_width_params.set_value(8, ncnn_graph::ParamValue::make_int(0));
  zero_width_params.set_value(18, ncnn_graph::ParamValue::make_int(2));
  adaptive_layers[1].set_params(std::move(zero_width_params));
  zero_adaptive_width.set_layers(std::move(adaptive_layers));
  EXPECT_FALSE(import_graph(zero_adaptive_width))
    << "adaptive pooling rejects zero output width";

  auto zero_adaptive_height = make_pool_graph(5, 1, 1);
  adaptive_layers.assign(zero_adaptive_height.get_layers().begin(),
                         zero_adaptive_height.get_layers().end());
  ncnn_graph::ParamDict zero_height_params;
  zero_height_params.set_value(7, ncnn_graph::ParamValue::make_int(1));
  zero_height_params.set_value(8, ncnn_graph::ParamValue::make_int(2));
  zero_height_params.set_value(18, ncnn_graph::ParamValue::make_int(0));
  adaptive_layers[1].set_params(std::move(zero_height_params));
  zero_adaptive_height.set_layers(std::move(adaptive_layers));
  EXPECT_FALSE(import_graph(zero_adaptive_height))
    << "adaptive pooling rejects zero output height";
}

TEST(NcnnImporterTest, RejectsInvalidGraphs) {
  using ncnn_frontend::import_graph;

  ncnn_graph::Graph unknown_graph;
  unknown_graph.add_layer(make_layer("Mystery", "bad", {}, {"out"}));
  unknown_graph.set_output_blob_names({"out"});
  EXPECT_FALSE(import_graph(unknown_graph)) << "unknown layer type is rejected";

  auto bad_kind = make_supported_graph();
  std::vector<ncnn_graph::Layer> layers(bad_kind.get_layers().begin(),
                                        bad_kind.get_layers().end());
  ncnn_graph::ParamDict bad_params;
  bad_params.set_value(0, ncnn_graph::ParamValue::make_int(5));
  bad_params.set_value(1, ncnn_graph::ParamValue::make_int(5));
  bad_params.set_value(2, ncnn_graph::ParamValue::make_float(1.0f));
  layers[0].set_params(std::move(bad_params));
  bad_kind.set_layers(std::move(layers));
  EXPECT_FALSE(import_graph(bad_kind)) << "wrong parameter kind is rejected";

  auto missing_weight = make_supported_graph();
  layers.assign(missing_weight.get_layers().begin(),
                missing_weight.get_layers().end());
  layers[1].set_weights({});
  missing_weight.set_layers(std::move(layers));
  EXPECT_FALSE(import_graph(missing_weight))
    << "missing convolution weights are rejected";

  auto bad_channels = make_supported_graph();
  layers.assign(bad_channels.get_layers().begin(),
                bad_channels.get_layers().end());
  layers[1].set_weights(
    {make_tensor({2, 2, 3, 3}, ncnn_graph::DataType::Float32),
     make_tensor({2}, ncnn_graph::DataType::Float32)});
  ncnn_graph::ParamDict channel_params;
  channel_params.set_value(0, ncnn_graph::ParamValue::make_int(2));
  channel_params.set_value(1, ncnn_graph::ParamValue::make_int(3));
  channel_params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  channel_params.set_value(5, ncnn_graph::ParamValue::make_int(1));
  channel_params.set_value(6, ncnn_graph::ParamValue::make_int(36));
  layers[1].set_params(std::move(channel_params));
  bad_channels.set_layers(std::move(layers));
  EXPECT_FALSE(import_graph(bad_channels))
    << "input and weight channel mismatch is rejected";
}
