#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
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
  std::size_t element_width = 0;
  switch (type) {
    case ncnn_graph::DataType::Unknown:
      break;
    case ncnn_graph::DataType::Float32:
      element_width = sizeof(float);
      break;
    case ncnn_graph::DataType::Float16:
    case ncnn_graph::DataType::BFloat16:
      element_width = 2;
      break;
    case ncnn_graph::DataType::Int8:
      element_width = 1;
      break;
  }
  std::vector<std::int64_t> contents_shape(tensor.get_shape().begin(),
                                           tensor.get_shape().end());
  auto contents = tensor.set_contents(
    std::move(contents_shape),
    type,
    std::vector<std::byte>(tensor.element_count() * element_width));
  if (!contents) {
    std::cerr << contents.error() << '\n';
    std::terminate();
  }
  return tensor;
}

ncnn_graph::Tensor make_float_tensor(std::vector<std::int64_t> shape,
                                     float value) {
  ncnn_graph::Tensor tensor =
    make_tensor(std::move(shape), ncnn_graph::DataType::Float32);
  std::vector<std::byte> data(tensor.byte_size());
  for (std::size_t offset = 0; offset < data.size(); offset += sizeof(value)) {
    std::memcpy(data.data() + offset, &value, sizeof(value));
  }
  std::vector<std::int64_t> tensor_shape(tensor.get_shape().begin(),
                                         tensor.get_shape().end());
  auto contents = tensor.set_contents(
    std::move(tensor_shape), ncnn_graph::DataType::Float32, std::move(data));
  if (!contents) {
    std::cerr << contents.error() << '\n';
    std::terminate();
  }
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
  convolution.add_weight(make_float_tensor({1}, 2.0F));
  convolution.add_weight(make_float_tensor({1}, 4.0F));
  convolution.add_weight(make_float_tensor({1}, 8.0F));
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
    convolution.add_weight(make_float_tensor({1}, 2.0F));
    convolution.add_weight(make_float_tensor({1}, 4.0F));
    if (scale_term > 100) {
      convolution.add_weight(make_float_tensor({1}, 8.0F));
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

ncnn_graph::Graph make_deconvolution_graph(std::int64_t weight_count) {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(2));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(2));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(16));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));

  auto deconvolution = make_layer("Deconvolution", "deconv", {"data"}, {"out"});
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  params.set_value(1, ncnn_graph::ParamValue::make_int(2));
  params.set_value(3, ncnn_graph::ParamValue::make_int(2));
  params.set_value(6, ncnn_graph::ParamValue::make_int(weight_count));
  params.set_value(9, ncnn_graph::ParamValue::make_int(1));
  params.set_value(10,
                   ncnn_graph::ParamValue::make_float_array({0.125F, -0.25F}));
  deconvolution.set_params(std::move(params));
  deconvolution.add_weight(
    make_tensor({1, 16, 2, 2}, ncnn_graph::DataType::Float32));
  graph.add_layer(std::move(deconvolution));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"out"});
  graph.set_weights_loaded(true);
  return graph;
}

ncnn_graph::Graph make_detection_output_graph() {
  ncnn_graph::Graph graph;
  auto add_input = [&](std::string name,
                       std::string blob,
                       std::int64_t width,
                       std::int64_t height) {
    auto input = make_layer("Input", std::move(name), {}, {std::move(blob)});
    ncnn_graph::ParamDict params;
    params.set_value(0, ncnn_graph::ParamValue::make_int(width));
    params.set_value(1, ncnn_graph::ParamValue::make_int(height));
    params.set_value(2, ncnn_graph::ParamValue::make_int(1));
    input.set_params(std::move(params));
    graph.add_layer(std::move(input));
  };
  add_input("location", "location", 12, 1);
  add_input("confidence", "confidence", 9, 1);
  add_input("priorbox", "priorbox", 12, 2);
  auto detection = make_layer("DetectionOutput",
                              "detection",
                              {"location", "confidence", "priorbox"},
                              {"detections"});
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(3));
  params.set_value(2, ncnn_graph::ParamValue::make_int(3));
  params.set_value(3, ncnn_graph::ParamValue::make_int(2));
  detection.set_params(std::move(params));
  graph.add_layer(std::move(detection));
  graph.set_input_blob_names({"location", "confidence", "priorbox"});
  graph.set_output_blob_names({"detections"});
  graph.set_weights_loaded(true);
  return graph;
}

class NcnnImporterTest : public ::testing::Test {
 protected:
  NcnnImporterTest() { register_dialects(context_); }

  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ncnn_importer::ImportError>
  import(const ncnn_graph::Graph& graph) {
    return ncnn_importer::import_graph(graph, context_);
  }

  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ncnn_importer::ImportError>
  import(const ncnn_graph::Graph& graph,
         const ncnn_importer::ImportOptions& options) {
    return ncnn_importer::import_graph(graph, context_, options);
  }

  mlir::MLIRContext context_;
};

}  // namespace

TEST_F(NcnnImporterTest, ImportsDynamicInputShapeOverride) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "input", {}, {"data"}));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"data"});
  graph.set_weights_loaded(true);
  ncnn_importer::ImportOptions options;
  options.input_shape = ncnn_importer::InputShape{4, -1, 8};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  auto type = result_type_by_name(imported->get(), "input");
  ASSERT_TRUE(type);
  EXPECT_TRUE(shape_is(type, {4, mlir::ShapedType::kDynamic, 8}));

  options.input_dim_constraints = {
    {.input = 0, .dimension = 1, .minimum = 32, .multiple_of = 32}};
  imported = import(graph, options);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  auto models = imported->get().getOps<mlir::ncnn::ModelOp>();
  ASSERT_NE(models.begin(), models.end());
  mlir::ncnn::ModelOp model = *models.begin();
  auto constraints =
    model->getAttrOfType<mlir::ArrayAttr>("ncnn.shape_constraints");
  ASSERT_TRUE(constraints);
  ASSERT_EQ(constraints.size(), 1U);
  auto constraint = mlir::cast<mlir::ncnn::DimConstraintAttr>(constraints[0]);
  EXPECT_EQ(constraint.getInput(), 0U);
  EXPECT_EQ(constraint.getDim(), 1U);
  EXPECT_EQ(constraint.getMin(), 32);
  EXPECT_EQ(constraint.getMultipleOf(), 32);
}

TEST_F(NcnnImporterTest, InfersOmittedInputChannelsFromConvolutionWeights) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "input", {}, {"data"}));
  auto convolution = make_layer("Convolution", "conv", {"data"}, {"out"});
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(4));
  params.set_value(1, ncnn_graph::ParamValue::make_int(1));
  params.set_value(6, ncnn_graph::ParamValue::make_int(12));
  convolution.set_params(std::move(params));
  convolution.add_weight(
    make_tensor({4, 3, 1, 1}, ncnn_graph::DataType::Float32));
  graph.add_layer(std::move(convolution));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"out"});
  graph.set_weights_loaded(true);

  auto imported = import(graph);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  EXPECT_TRUE(
    shape_is(result_type_by_name(imported->get(), "input"),
             {3, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}));
}

TEST_F(NcnnImporterTest, ImportsDynamicRankSpecializations) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "input", {}, {"data"}));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"data"});
  graph.set_weights_loaded(true);
  ncnn_importer::ImportOptions options;
  options.dynamic_rank = true;
  auto imported = import(graph, options);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  std::array<bool, 4> ranks{};
  imported->get().walk([&](mlir::ncnn::ModelOp model) {
    auto rank = model->getAttrOfType<mlir::IntegerAttr>("ncnn.rank_variant");
    ASSERT_TRUE(rank);
    ASSERT_GE(rank.getInt(), 1);
    ASSERT_LE(rank.getInt(), 4);
    ranks[rank.getInt() - 1] = true;
  });
  EXPECT_TRUE(std::ranges::all_of(ranks, std::identity{}));
}

TEST_F(NcnnImporterTest, RejectsUnsupportedDynamicRankModel) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "input", {}, {"data"}));
  graph.add_layer(
    make_layer("Convolution", "convolution", {"data"}, {"output"}));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"output"});
  graph.set_weights_loaded(true);
  ncnn_importer::ImportOptions options;
  options.dynamic_rank = true;
  auto imported = import(graph, options);
  ASSERT_FALSE(imported.has_value());
  EXPECT_NE(imported.error().get_message().find("identity/shape-preserving"),
            std::string_view::npos);
}

TEST_F(NcnnImporterTest, DoesNotOverrideExplicitInputShape) {
  auto graph = make_supported_graph();
  ncnn_importer::ImportOptions options;
  options.input_shape = ncnn_importer::InputShape{4, -1, 8};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  EXPECT_TRUE(
    shape_is(result_type_by_name(imported->get(), "input"), {1, 5, 5}));
}

TEST_F(NcnnImporterTest, AppliesMultipleInputShapesInSourceOrder) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "first", {}, {"first_blob"}));
  graph.add_layer(make_layer("Input", "second", {}, {"second_blob"}));
  graph.set_input_blob_names({"first_blob", "second_blob"});
  graph.set_output_blob_names({"first_blob", "second_blob"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions options;
  options.input_shapes = {{2, -1, 4}, {1, 3, -1}};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  EXPECT_TRUE(shape_is(result_type_by_name(imported->get(), "first"),
                       {2, mlir::ShapedType::kDynamic, 4}));
  EXPECT_TRUE(shape_is(result_type_by_name(imported->get(), "second"),
                       {1, 3, mlir::ShapedType::kDynamic}));
}

TEST_F(NcnnImporterTest, PreservesStaticAndDynamicInputsInSourceOrder) {
  ncnn_graph::Graph graph;
  auto first = make_layer("Input", "first", {}, {"first_blob"});
  ncnn_graph::ParamDict firstParams;
  firstParams.set_value(0, ncnn_graph::ParamValue::make_int(5));
  firstParams.set_value(1, ncnn_graph::ParamValue::make_int(4));
  firstParams.set_value(2, ncnn_graph::ParamValue::make_int(3));
  first.set_params(std::move(firstParams));
  graph.add_layer(std::move(first));
  graph.add_layer(make_layer("Input", "second", {}, {"second_blob"}));
  graph.set_input_blob_names({"first_blob", "second_blob"});
  graph.set_output_blob_names({"first_blob", "second_blob"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions options;
  options.input_shapes = {{9, -1, 9}, {1, -1, 8}};
  options.input_dim_constraints = {
    {.input = 1, .dimension = 1, .minimum = 4, .multiple_of = 2}};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  EXPECT_TRUE(
    shape_is(result_type_by_name(imported->get(), "first"), {3, 4, 5}));
  EXPECT_TRUE(shape_is(result_type_by_name(imported->get(), "second"),
                       {1, mlir::ShapedType::kDynamic, 8}));
}

TEST_F(NcnnImporterTest, ImportsReshapeShapeExpressionReferences) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "data", {}, {"data_blob"}));
  graph.add_layer(make_layer("Input", "reference", {}, {"reference_blob"}));
  auto reshape = make_layer(
    "Reshape", "reshape_as", {"data_blob", "reference_blob"}, {"output"});
  ncnn_graph::ParamDict params;
  params.set_value(6, ncnn_graph::ParamValue::make_string("1w,1h,1c"));
  reshape.set_params(std::move(params));
  graph.add_layer(std::move(reshape));
  graph.set_input_blob_names({"data_blob", "reference_blob"});
  graph.set_output_blob_names({"output"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions options;
  options.input_shapes = {{1, -1, -1}, {1, -1, -1}};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported) << imported.error().to_string();
  auto models = imported->get().getOps<mlir::ncnn::ModelOp>();
  ASSERT_NE(models.begin(), models.end());
  mlir::ncnn::ModelOp model = *models.begin();
  auto reshapes = model.getOps<mlir::ncnn::ReshapeOp>();
  auto operation = *reshapes.begin();
  EXPECT_EQ(operation->getNumOperands(), 2U);
  ASSERT_TRUE(operation.getShapeExpression());
  EXPECT_EQ(*operation.getShapeExpression(), "1w,1h,1c");
  ASSERT_TRUE(operation.getShapeSources());
  EXPECT_EQ(*operation.getShapeSources(),
            (llvm::ArrayRef<int64_t>{1, 0, 1, 1, 1, 2}));
  EXPECT_TRUE(
    shape_is(operation.getOutput().getType(),
             {1, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}));

  auto layers = std::vector<ncnn_graph::Layer>(graph.get_layers().begin(),
                                               graph.get_layers().end());
  ncnn_graph::ParamDict reorderedParams;
  reorderedParams.set_value(6, ncnn_graph::ParamValue::make_string("1h,1w,1c"));
  layers[2].set_params(std::move(reorderedParams));
  graph.set_layers(std::move(layers));
  auto reordered = import(graph, options);
  ASSERT_FALSE(reordered);
  EXPECT_NE(reordered.error().get_message().find("preserve"),
            std::string_view::npos);

  layers = std::vector<ncnn_graph::Layer>(graph.get_layers().begin(),
                                          graph.get_layers().end());
  ncnn_graph::ParamDict invalidMask;
  invalidMask.set_value(31, ncnn_graph::ParamValue::make_string("invalid"));
  layers[2].set_params(std::move(invalidMask));
  graph.set_layers(std::move(layers));
  EXPECT_FALSE(import(graph, options));
}

TEST_F(NcnnImporterTest, PreservesReshapeZeroAndInferSemantics) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "input", {}, {"data"}));
  auto reshape = make_layer("Reshape", "reshape", {"data"}, {"output"});
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(-1));
  params.set_value(1, ncnn_graph::ParamValue::make_int(0));
  reshape.set_params(std::move(params));
  graph.add_layer(std::move(reshape));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"output"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions options;
  options.input_shape = ncnn_importer::InputShape{3, 4, 5};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported) << imported.error().to_string();
  imported->get().walk([&](mlir::ncnn::ReshapeOp op) {
    ASSERT_TRUE(op.getShapeSpec());
    EXPECT_EQ(*op.getShapeSpec(), (llvm::ArrayRef<int64_t>{0, -1}));
    ASSERT_TRUE(op.getShapeZeroSources());
    EXPECT_EQ(*op.getShapeZeroSources(), (llvm::ArrayRef<int64_t>{1, -1}));
    EXPECT_TRUE(shape_is(op.getOutput().getType(), {4, 15}));
  });

  auto layers = std::vector<ncnn_graph::Layer>(graph.get_layers().begin(),
                                               graph.get_layers().end());
  ncnn_graph::ParamDict multipleUnknown;
  multipleUnknown.set_value(0, ncnn_graph::ParamValue::make_int(-1));
  multipleUnknown.set_value(1, ncnn_graph::ParamValue::make_int(-1));
  layers[1].set_params(std::move(multipleUnknown));
  graph.set_layers(std::move(layers));
  EXPECT_FALSE(import(graph, options));
}

TEST_F(NcnnImporterTest, ImportsDynamicBinarySlicePoolingAndInnerProduct) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "left", {}, {"left"}));
  graph.add_layer(make_layer("Input", "right", {}, {"right"}));
  auto binary = make_layer("BinaryOp", "max3d", {"left", "right"}, {"max3d"});
  ncnn_graph::ParamDict binaryParams;
  binaryParams.set_value(0, ncnn_graph::ParamValue::make_int(4));
  binary.set_params(std::move(binaryParams));
  graph.add_layer(std::move(binary));
  auto squeeze = make_layer("Squeeze", "max", {"max3d"}, {"max"});
  ncnn_graph::ParamDict squeezeParams;
  squeezeParams.set_value(3, ncnn_graph::ParamValue::make_int_array({1}));
  squeeze.set_params(std::move(squeezeParams));
  graph.add_layer(std::move(squeeze));
  auto slice = make_layer("Slice", "slice", {"max"}, {"first", "rest"});
  ncnn_graph::ParamDict sliceParams;
  sliceParams.set_value(0, ncnn_graph::ParamValue::make_int_array({2, -233}));
  sliceParams.set_value(1, ncnn_graph::ParamValue::make_int(0));
  slice.set_params(std::move(sliceParams));
  graph.add_layer(std::move(slice));
  auto product = make_layer("InnerProduct", "product", {"max"}, {"product"});
  ncnn_graph::ParamDict productParams;
  productParams.set_value(0, ncnn_graph::ParamValue::make_int(3));
  productParams.set_value(2, ncnn_graph::ParamValue::make_int(12));
  product.set_params(std::move(productParams));
  product.add_weight(make_tensor({3, 4}, ncnn_graph::DataType::Float32));
  graph.add_layer(std::move(product));
  graph.set_input_blob_names({"left", "right"});
  graph.set_output_blob_names({"first", "rest", "product"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions options;
  options.input_shapes = {{-1, 1, 4}, {1, 1, 4}};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported) << imported.error().to_string();
  auto model = *imported->get().getOps<mlir::ncnn::ModelOp>().begin();
  auto constraints =
    model->getAttrOfType<mlir::ArrayAttr>("ncnn.shape_constraints");
  ASSERT_TRUE(constraints);
  ASSERT_EQ(constraints.size(), 1U);
  auto constraint = mlir::cast<mlir::ncnn::DimConstraintAttr>(constraints[0]);
  EXPECT_EQ(constraint.getInput(), 0U);
  EXPECT_EQ(constraint.getDim(), 0U);
  EXPECT_EQ(constraint.getMin(), 3);
  EXPECT_EQ(constraint.getMultipleOf(), 1);
  EXPECT_TRUE(shape_is(result_type_by_name(imported->get(), "max"),
                       {mlir::ShapedType::kDynamic, 4}));
  EXPECT_TRUE(shape_is(result_type_by_name(imported->get(), "product"),
                       {mlir::ShapedType::kDynamic, 3}));
  imported->get().walk([&](mlir::ncnn::SliceOp op) {
    EXPECT_TRUE(shape_is(
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()), {2, 4}));
    EXPECT_TRUE(
      shape_is(mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType()),
               {mlir::ShapedType::kDynamic, 4}));
  });

  ncnn_graph::Graph pooling;
  pooling.add_layer(make_layer("Input", "input", {}, {"data"}));
  auto global = make_layer("Pooling", "global", {"data"}, {"output"});
  ncnn_graph::ParamDict globalParams;
  globalParams.set_value(4, ncnn_graph::ParamValue::make_int(1));
  global.set_params(std::move(globalParams));
  pooling.add_layer(std::move(global));
  pooling.set_input_blob_names({"data"});
  pooling.set_output_blob_names({"output"});
  pooling.set_weights_loaded(true);
  options.input_shapes = {{-1, -1, -1}};
  auto pooled = import(pooling, options);
  ASSERT_TRUE(pooled) << pooled.error().to_string();
  EXPECT_TRUE(
    shape_is(output_type(pooled->get()), {mlir::ShapedType::kDynamic}));
}

TEST_F(NcnnImporterTest, RejectsUnsafeDynamicDetectionAndWeights) {
  auto detection = make_detection_output_graph();
  auto layers = std::vector<ncnn_graph::Layer>(detection.get_layers().begin(),
                                               detection.get_layers().end());
  layers[2].set_params({});
  detection.set_layers(std::move(layers));
  ncnn_importer::ImportOptions options;
  options.input_shapes = {{1, 1, 12}, {1, 1, 9}, {1, 2, -1}};
  auto dynamicPrior = import(detection, options);
  ASSERT_FALSE(dynamicPrior);
  EXPECT_NE(dynamicPrior.error().get_message().find("static FP32 inputs"),
            std::string_view::npos);
}

TEST_F(NcnnImporterTest, RejectsInputShapeContractErrors) {
  ncnn_graph::Graph graph;
  graph.add_layer(make_layer("Input", "input", {}, {"data"}));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"data"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions mismatch;
  mismatch.input_shapes = {{1, 2, 3}, {4, 5, 6}};
  auto mismatched = import(graph, mismatch);
  ASSERT_FALSE(mismatched.has_value());
  EXPECT_NE(mismatched.error().get_message().find("count 2 matches neither 1"),
            std::string_view::npos);

  ncnn_importer::ImportOptions conflict;
  conflict.input_shape = ncnn_importer::InputShape{1, 2, 3};
  conflict.input_shapes = {{1, 2, 3}};
  auto conflicted = import(graph, conflict);
  ASSERT_FALSE(conflicted.has_value());
  EXPECT_NE(conflicted.error().get_message().find("cannot both be specified"),
            std::string_view::npos);

  ncnn_importer::ImportOptions invalid;
  invalid.input_shape = ncnn_importer::InputShape{1, 0, 3};
  auto invalid_shape = import(graph, invalid);
  ASSERT_FALSE(invalid_shape.has_value());
  EXPECT_NE(invalid_shape.error().get_message().find("positive or dynamic"),
            std::string_view::npos);
}

TEST_F(NcnnImporterTest, AppliesShapesOnlyToDimensionlessInputs) {
  ncnn_graph::Graph graph;
  auto declared = make_layer("Input", "declared", {}, {"left"});
  ncnn_graph::ParamDict declared_params;
  declared_params.set_value(0, ncnn_graph::ParamValue::make_int(5));
  declared_params.set_value(1, ncnn_graph::ParamValue::make_int(4));
  declared_params.set_value(2, ncnn_graph::ParamValue::make_int(3));
  declared.set_params(std::move(declared_params));
  graph.add_layer(std::move(declared));
  graph.add_layer(make_layer("Input", "omitted", {}, {"right"}));
  graph.set_input_blob_names({"left", "right"});
  graph.set_output_blob_names({"left", "right"});
  graph.set_weights_loaded(true);

  ncnn_importer::ImportOptions options;
  options.input_shapes = {{7, 8, 9}};
  auto imported = import(graph, options);
  ASSERT_TRUE(imported) << imported.error().to_string();
  EXPECT_TRUE(
    shape_is(result_type_by_name(imported->get(), "declared"), {3, 4, 5}));
  EXPECT_TRUE(
    shape_is(result_type_by_name(imported->get(), "omitted"), {7, 8, 9}));
}

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
    << "f16-storage weight and f32 bias become ncnn model constants";
  module->walk([](mlir::ncnn::ConstOp constant) {
    EXPECT_TRUE(constant.getOutput().getType().getElementType().isF32())
      << "f16 model storage is promoted to f32 computation";
  });
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

TEST_F(NcnnImporterTest, PreservesFloat16StorageForFloat16Policy) {
  ncnn_importer::ImportOptions options;
  options.precision.mode = ncnn_mlir::PrecisionMode::Float16;
  auto imported = import(make_supported_graph(), options);
  ASSERT_TRUE(imported) << imported.error().to_string();

  mlir::ModuleOp module = imported->get();
  auto model = *module.getOps<mlir::ncnn::ModelOp>().begin();
  EXPECT_EQ(model->getAttrOfType<mlir::StringAttr>("ncnn.precision").getValue(),
            "fp16");
  int float16_constants = 0;
  int float32_constants = 0;
  module->walk([&](mlir::ncnn::ConstOp constant) {
    mlir::Type element = constant.getOutput().getType().getElementType();
    float16_constants += element.isF16();
    float32_constants += element.isF32();
  });
  EXPECT_EQ(float16_constants, 1);
  EXPECT_EQ(float32_constants, 1);
}

TEST_F(NcnnImporterTest, ImportsBoundedDetectionOutputContract) {
  auto imported = import(make_detection_output_graph());
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::DetectionOutputOp>(imported->get()), 1);
  EXPECT_TRUE(shape_is(output_type(imported->get()), {2, 6}));

  auto invalid = make_detection_output_graph();
  auto layers = std::vector<ncnn_graph::Layer>(invalid.get_layers().begin(),
                                               invalid.get_layers().end());
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(-233));
  layers.back().set_params(std::move(params));
  invalid.set_layers(std::move(layers));
  auto rejected = import(invalid);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_NE(rejected.error().get_message().find("Caffe-style"),
            std::string_view::npos);
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

TEST_F(NcnnImporterTest, ImportsAsymmetricDepthwiseSpatialParameters) {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(9));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(11));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(2));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));

  auto depthwise =
    make_layer("ConvolutionDepthWise", "depthwise", {"data"}, {"out"});
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(2));
  params.set_value(1, ncnn_graph::ParamValue::make_int(3));
  params.set_value(11, ncnn_graph::ParamValue::make_int(2));
  params.set_value(2, ncnn_graph::ParamValue::make_int(2));
  params.set_value(12, ncnn_graph::ParamValue::make_int(1));
  params.set_value(3, ncnn_graph::ParamValue::make_int(2));
  params.set_value(13, ncnn_graph::ParamValue::make_int(3));
  params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  params.set_value(15, ncnn_graph::ParamValue::make_int(2));
  params.set_value(14, ncnn_graph::ParamValue::make_int(0));
  params.set_value(16, ncnn_graph::ParamValue::make_int(1));
  params.set_value(6, ncnn_graph::ParamValue::make_int(12));
  params.set_value(7, ncnn_graph::ParamValue::make_int(2));
  depthwise.set_params(std::move(params));
  depthwise.add_weight(
    make_tensor({2, 1, 2, 3}, ncnn_graph::DataType::Float32));
  graph.add_layer(std::move(depthwise));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"out"});
  graph.set_weights_loaded(true);

  auto imported = import(graph);
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  mlir::ModuleOp module = imported->get();
  EXPECT_TRUE(shape_is(output_type(module), {2, 4, 4}));
  module->walk([&](mlir::ncnn::ConvolutionDepthWiseOp op) {
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("stride_h").getInt(), 3);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("stride_w").getInt(), 2);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("dilation_h").getInt(), 1);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("dilation_w").getInt(), 2);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("pad_top").getInt(), 0);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("pad_bottom").getInt(), 1);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("pad_left").getInt(), 1);
    EXPECT_EQ(op->getAttrOfType<mlir::IntegerAttr>("pad_right").getInt(), 2);
  });
}

TEST_F(NcnnImporterTest, ExpandsSupportedFusedConvolutionActivations) {
  auto convolutionGraph = make_supported_graph();
  auto layers = std::vector<ncnn_graph::Layer>(
    convolutionGraph.get_layers().begin(), convolutionGraph.get_layers().end());
  ncnn_graph::ParamDict convolutionParams = layers[1].get_params();
  convolutionParams.set_value(9, ncnn_graph::ParamValue::make_int(4));
  layers[1].set_params(std::move(convolutionParams));
  convolutionGraph.set_layers(std::move(layers));
  auto convolution = import(convolutionGraph);
  ASSERT_TRUE(convolution) << convolution.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::SigmoidOp>(convolution->get()), 1);

  ncnn_graph::Graph depthwiseGraph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict inputParams;
  inputParams.set_value(0, ncnn_graph::ParamValue::make_int(5));
  inputParams.set_value(1, ncnn_graph::ParamValue::make_int(5));
  inputParams.set_value(2, ncnn_graph::ParamValue::make_int(2));
  input.set_params(std::move(inputParams));
  depthwiseGraph.add_layer(std::move(input));
  auto depthwise =
    make_layer("ConvolutionDepthWise", "depthwise", {"data"}, {"output"});
  ncnn_graph::ParamDict depthwiseParams;
  depthwiseParams.set_value(0, ncnn_graph::ParamValue::make_int(2));
  depthwiseParams.set_value(1, ncnn_graph::ParamValue::make_int(3));
  depthwiseParams.set_value(4, ncnn_graph::ParamValue::make_int(1));
  depthwiseParams.set_value(6, ncnn_graph::ParamValue::make_int(18));
  depthwiseParams.set_value(7, ncnn_graph::ParamValue::make_int(2));
  depthwiseParams.set_value(9, ncnn_graph::ParamValue::make_int(1));
  depthwise.set_params(std::move(depthwiseParams));
  depthwise.add_weight(
    make_tensor({2, 1, 3, 3}, ncnn_graph::DataType::Float32));
  depthwiseGraph.add_layer(std::move(depthwise));
  depthwiseGraph.set_input_blob_names({"data"});
  depthwiseGraph.set_output_blob_names({"output"});
  depthwiseGraph.set_weights_loaded(true);
  auto importedDepthwise = import(depthwiseGraph);
  ASSERT_TRUE(importedDepthwise) << importedDepthwise.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::ReluOp>(importedDepthwise->get()), 1);

  auto depthwiseLayers = std::vector<ncnn_graph::Layer>(
    depthwiseGraph.get_layers().begin(), depthwiseGraph.get_layers().end());
  auto invalidActivation = depthwiseLayers[1].get_params();
  invalidActivation.set_value(9, ncnn_graph::ParamValue::make_float(1.0F));
  depthwiseLayers[1].set_params(std::move(invalidActivation));
  depthwiseGraph.set_layers(std::move(depthwiseLayers));
  EXPECT_FALSE(import(depthwiseGraph));

  depthwiseLayers = std::vector<ncnn_graph::Layer>(
    depthwiseGraph.get_layers().begin(), depthwiseGraph.get_layers().end());
  auto invalidPadValue = depthwiseLayers[1].get_params();
  invalidPadValue.set_value(9, ncnn_graph::ParamValue::make_int(1));
  invalidPadValue.set_value(18, ncnn_graph::ParamValue::make_int(1));
  depthwiseLayers[1].set_params(std::move(invalidPadValue));
  depthwiseGraph.set_layers(std::move(depthwiseLayers));
  EXPECT_FALSE(import(depthwiseGraph));
}

TEST_F(NcnnImporterTest, DeconvolutionIgnoresReluActivationParameters) {
  auto imported = import(make_deconvolution_graph(64));
  ASSERT_TRUE(imported.has_value()) << imported.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::DeconvolutionOp>(imported->get()), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::ReluOp>(imported->get()), 1);
}

TEST_F(NcnnImporterTest, ImportsQuantizeAndCastBoundaries) {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(3));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(2));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));

  auto quantize = make_layer("Quantize", "quantize", {"data"}, {"quantized"});
  ncnn_graph::ParamDict quantize_params;
  quantize_params.set_value(0, ncnn_graph::ParamValue::make_int(1));
  quantize.set_params(std::move(quantize_params));
  quantize.add_weight(make_float_tensor({1}, 2.0F));
  graph.add_layer(std::move(quantize));

  auto to_float = make_layer("Cast", "to_float", {"quantized"}, {"restored"});
  ncnn_graph::ParamDict cast_params;
  cast_params.set_value(0, ncnn_graph::ParamValue::make_int(3));
  cast_params.set_value(1, ncnn_graph::ParamValue::make_int(1));
  to_float.set_params(std::move(cast_params));
  graph.add_layer(std::move(to_float));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"restored"});
  graph.set_weights_loaded(true);

  auto imported = import(graph);
  ASSERT_TRUE(imported) << imported.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::QuantizeOp>(imported->get()), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::CastOp>(imported->get()), 1);
  EXPECT_TRUE(output_type(imported->get()).getElementType().isF32());
}

TEST_F(NcnnImporterTest, ImportsQuantizedGemmWithSignedConstantB) {
  ncnn_graph::Graph graph;
  auto input = make_layer("Input", "input", {}, {"data"});
  ncnn_graph::ParamDict input_params;
  input_params.set_value(0, ncnn_graph::ParamValue::make_int(4));
  input_params.set_value(1, ncnn_graph::ParamValue::make_int(2));
  input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
  input.set_params(std::move(input_params));
  graph.add_layer(std::move(input));

  auto squeeze = make_layer("Squeeze", "squeeze", {"data"}, {"matrix"});
  ncnn_graph::ParamDict squeeze_params;
  squeeze_params.set_value(3, ncnn_graph::ParamValue::make_int_array({0}));
  squeeze.set_params(std::move(squeeze_params));
  graph.add_layer(std::move(squeeze));

  auto gemm = make_layer("Gemm", "gemm", {"matrix"}, {"output"});
  ncnn_graph::ParamDict gemm_params;
  gemm_params.set_value(3, ncnn_graph::ParamValue::make_int(1));
  gemm_params.set_value(5, ncnn_graph::ParamValue::make_int(1));
  gemm_params.set_value(6, ncnn_graph::ParamValue::make_int(1));
  gemm_params.set_value(8, ncnn_graph::ParamValue::make_int(3));
  gemm_params.set_value(9, ncnn_graph::ParamValue::make_int(4));
  gemm_params.set_value(10, ncnn_graph::ParamValue::make_int(4));
  gemm_params.set_value(18, ncnn_graph::ParamValue::make_int(2));
  gemm.set_params(std::move(gemm_params));
  gemm.add_weight(make_tensor({3, 4}, ncnn_graph::DataType::Int8));
  gemm.add_weight(make_float_tensor({3}, 0.0F));
  gemm.add_weight(make_float_tensor({1}, 4.0F));
  graph.add_layer(std::move(gemm));
  graph.set_input_blob_names({"data"});
  graph.set_output_blob_names({"output"});
  graph.set_weights_loaded(true);

  auto imported = import(graph);
  ASSERT_TRUE(imported) << imported.error().to_string();
  EXPECT_EQ(count_ops<mlir::ncnn::GemmOp>(imported->get()), 1);
  EXPECT_TRUE(shape_is(output_type(imported->get()), {2, 3}));
}

TEST_F(NcnnImporterTest, RejectsDeconvolutionWeightCountMismatch) {
  auto imported = import(make_deconvolution_graph(68));
  ASSERT_FALSE(imported.has_value());
  EXPECT_NE(imported.error().to_string().find("weight_data_size"),
            std::string::npos);
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

  auto make_interp_graph = [](float scale, std::int64_t input_size) {
    ncnn_graph::Graph graph;
    auto input = make_layer("Input", "input", {}, {"data"});
    ncnn_graph::ParamDict input_params;
    input_params.set_value(0, ncnn_graph::ParamValue::make_int(input_size));
    input_params.set_value(1, ncnn_graph::ParamValue::make_int(input_size));
    input_params.set_value(2, ncnn_graph::ParamValue::make_int(1));
    input.set_params(std::move(input_params));
    graph.add_layer(std::move(input));
    auto interp = make_layer("Interp", "interp", {"data"}, {"out"});
    ncnn_graph::ParamDict params;
    params.set_value(0, ncnn_graph::ParamValue::make_int(1));
    params.set_value(1, ncnn_graph::ParamValue::make_float(scale));
    params.set_value(2, ncnn_graph::ParamValue::make_float(scale));
    interp.set_params(std::move(params));
    graph.add_layer(std::move(interp));
    graph.set_input_blob_names({"data"});
    graph.set_output_blob_names({"out"});
    graph.set_weights_loaded(true);
    return graph;
  };
  EXPECT_FALSE(
    import(make_interp_graph(std::numeric_limits<float>::infinity(), 2)))
    << "non-finite Interp scale is rejected";
  EXPECT_TRUE(import(make_interp_graph(2049.0F, 2)))
    << "direct Linalg Interp is not restricted by TOSA scale limits";
  EXPECT_TRUE(import(make_interp_graph(2.0F, 8192)))
    << "direct Linalg Interp is not restricted by TOSA dimension limits";
}
