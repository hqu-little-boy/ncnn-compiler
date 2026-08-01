#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"
#include "ncnn-mlir/Graph/graph.hpp"
#include "ncnn-mlir/Importer/NCNNImporter.hpp"
#include <gtest/gtest.h>

namespace {

void register_dialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::ncnn::NCNNDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

template <typename OpT>
int count_ops(mlir::ModuleOp module) {
  int count = 0;
  module->walk([&](OpT) { ++count; });
  return count;
}

mlir::RankedTensorType result_type_by_name(mlir::ModuleOp module,
                                           mlir::StringRef name) {
  mlir::RankedTensorType result;
  module->walk([&](mlir::Operation* op) {
    auto attr = op->getAttrOfType<mlir::StringAttr>("ncnn.name");
    if (attr != nullptr && attr.getValue() == name &&
        op->getNumResults() == 1) {
      result =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    }
  });
  return result;
}

mlir::RankedTensorType output_type(mlir::ModuleOp module) {
  mlir::RankedTensorType result;
  module->walk([&](mlir::ncnn::OutputOp output) {
    result =
      mlir::dyn_cast<mlir::RankedTensorType>(output.getInput().getType());
  });
  return result;
}

bool shape_is(mlir::RankedTensorType type,
              std::initializer_list<std::int64_t> expected) {
  return type != nullptr && std::ranges::equal(type.getShape(), expected);
}

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

class NcnnImporterTest : public ::testing::Test {
 protected:
  NcnnImporterTest() { register_dialects(context_); }

  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ncnn_importer::ImportError>
  import(const ncnn_graph::Graph& graph) {
    return ncnn_importer::import_graph(graph, &context_);
  }

  mlir::MLIRContext context_;
};

}  // namespace

TEST_F(NcnnImporterTest, ImportsSupportedGraph) {
  auto imported = import(make_supported_graph());
  ASSERT_TRUE(imported.has_value())
    << "all eight supported source layer types import";
  mlir::ModuleOp module = imported->get();
  EXPECT_TRUE(mlir::succeeded(mlir::verify(module.getOperation())));

  EXPECT_EQ(count_ops<mlir::ncnn::ModelOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::InputOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::OutputOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::ConstOp>(module), 2)
    << "f16 weight and f32 bias become ncnn model constants";
  EXPECT_EQ(count_ops<mlir::func::FuncOp>(module), 0)
    << "import does not establish the function ABI";
  EXPECT_EQ(count_ops<mlir::ncnn::ConvolutionOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::ReluOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::SplitOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::ConcatOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::PoolingOp>(module), 2);
  EXPECT_EQ(count_ops<mlir::ncnn::DropoutOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::SoftmaxOp>(module), 1);

  EXPECT_TRUE(shape_is(result_type_by_name(module, "pool"), {4, 3, 3}))
    << "pad_mode 0 uses ceil/full-padding output shape";
  EXPECT_TRUE(shape_is(output_type(module), {4}))
    << "global pool and softmax produce rank-one output";
}

TEST_F(NcnnImporterTest, ImportsInt8Quantization) {
  auto int8 = import(make_int8_graph());
  ASSERT_TRUE(int8.has_value()) << int8.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::ConstOp>(int8->get()), 4)
    << "int8 kernel and three quantization scales become constants";

  auto int8_chain = import(make_int8_chain_graph());
  ASSERT_TRUE(int8_chain.has_value()) << int8_chain.error().to_string();
  auto out_type = output_type(int8_chain->get());
  EXPECT_TRUE(out_type != nullptr && out_type.getElementType().isF32())
    << "102 i8 output feeds term 2 input and dequantizes to f32";
  auto requant_type = result_type_by_name(int8_chain->get(), "requant");
  EXPECT_TRUE(requant_type != nullptr &&
              requant_type.getElementType().isInteger(8))
    << "scale term above 100 produces i8 result";

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
  EXPECT_FALSE(import(unquantized_i8))
    << "unquantized convolution rejects i8 kernel";
}

TEST_F(NcnnImporterTest, PoolingShapes) {
  auto large_kernel_pool = import(make_pool_graph(2, 3, 2));
  ASSERT_TRUE(large_kernel_pool.has_value());
  EXPECT_TRUE(shape_is(output_type(large_kernel_pool->get()), {1, 2, 2}))
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
  auto adaptive_imported = import(adaptive_graph);
  ASSERT_TRUE(adaptive_imported.has_value());
  EXPECT_TRUE(shape_is(output_type(adaptive_imported->get()), {1, 2, 5}))
    << "adaptive -233 preserves the corresponding input dimension";

  auto global_adaptive = make_pool_graph(5, 0, 1);
  adaptive_layers.assign(global_adaptive.get_layers().begin(),
                         global_adaptive.get_layers().end());
  ncnn_graph::ParamDict global_adaptive_params;
  global_adaptive_params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  global_adaptive_params.set_value(7, ncnn_graph::ParamValue::make_int(1));
  adaptive_layers[1].set_params(std::move(global_adaptive_params));
  global_adaptive.set_layers(std::move(adaptive_layers));
  auto global_adaptive_imported = import(global_adaptive);
  ASSERT_TRUE(global_adaptive_imported.has_value());
  EXPECT_TRUE(shape_is(output_type(global_adaptive_imported->get()), {1}))
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
  EXPECT_FALSE(import(zero_adaptive_width))
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
  EXPECT_FALSE(import(zero_adaptive_height))
    << "adaptive pooling rejects zero output height";
}

TEST_F(NcnnImporterTest, RejectsInvalidGraphs) {
  ncnn_graph::Graph unknown_graph;
  unknown_graph.add_layer(make_layer("Mystery", "bad", {}, {"out"}));
  unknown_graph.set_output_blob_names({"out"});
  EXPECT_FALSE(import(unknown_graph)) << "unknown layer type is rejected";

  ncnn_graph::Graph grouped_convolution;
  grouped_convolution.add_layer(
    make_layer("ConvolutionDepthWise", "grouped", {}, {"out"}));
  grouped_convolution.set_output_blob_names({"out"});
  EXPECT_FALSE(import(grouped_convolution))
    << "group convolution is a distinct unsupported source layer";

  auto bad_kind = make_supported_graph();
  std::vector<ncnn_graph::Layer> layers(bad_kind.get_layers().begin(),
                                        bad_kind.get_layers().end());
  ncnn_graph::ParamDict bad_params;
  bad_params.set_value(0, ncnn_graph::ParamValue::make_int(5));
  bad_params.set_value(1, ncnn_graph::ParamValue::make_int(5));
  bad_params.set_value(2, ncnn_graph::ParamValue::make_float(1.0f));
  layers[0].set_params(std::move(bad_params));
  bad_kind.set_layers(std::move(layers));
  EXPECT_FALSE(import(bad_kind)) << "wrong parameter kind is rejected";

  auto missing_weight = make_supported_graph();
  layers.assign(missing_weight.get_layers().begin(),
                missing_weight.get_layers().end());
  layers[1].set_weights({});
  missing_weight.set_layers(std::move(layers));
  EXPECT_FALSE(import(missing_weight))
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
  EXPECT_FALSE(import(bad_channels))
    << "input and weight channel mismatch is rejected";
}
