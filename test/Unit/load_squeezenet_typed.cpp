#include <cstdint>
#include <initializer_list>
#include <ranges>
#include <string>

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

bool shape_is(mlir::RankedTensorType type,
              std::initializer_list<std::int64_t> expected) {
  return type != nullptr && std::ranges::equal(type.getShape(), expected);
}

}  // namespace

TEST(LoadSqueezenetTyped, ImportsAndVerifies) {
  mlir::MLIRContext context;
  register_dialects(context);

  auto decoded = ncnn_graph::Graph::load(
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.param",
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.bin");
  ASSERT_TRUE(decoded.has_value()) << (decoded ? "" : decoded.error());

  auto imported = ncnn_importer::import_graph(*decoded, context);
  ASSERT_TRUE(imported.has_value())
    << (imported ? "" : imported.error().to_string());

  mlir::ModuleOp module = imported->get();
  EXPECT_TRUE(mlir::succeeded(mlir::verify(module.getOperation())))
    << "imported module verifies";

  ASSERT_EQ(count_ops<mlir::ncnn::ModelOp>(module), 1);
  ASSERT_EQ(count_ops<mlir::ncnn::InputOp>(module), 1);
  ASSERT_EQ(count_ops<mlir::ncnn::OutputOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::func::FuncOp>(module), 0);
  mlir::RankedTensorType input_type;
  mlir::RankedTensorType output_type;
  module->walk([&](mlir::ncnn::InputOp input) {
    input_type =
      mlir::dyn_cast<mlir::RankedTensorType>(input.getOutput().getType());
  });
  module->walk([&](mlir::ncnn::OutputOp output) {
    output_type =
      mlir::dyn_cast<mlir::RankedTensorType>(output.getInput().getType());
  });
  EXPECT_TRUE(shape_is(input_type, {3, 227, 227}))
    << "input is one [3,227,227] f32 graph argument";
  EXPECT_TRUE(input_type != nullptr && input_type.getElementType().isF32());
  EXPECT_TRUE(shape_is(output_type, {1000})) << "output is one [1000] value";

  EXPECT_EQ(count_ops<mlir::ncnn::ConstOp>(module), 52)
    << "weight constants: 26 conv * (weight + bias)";
  EXPECT_EQ(count_ops<mlir::ncnn::ConvolutionOp>(module), 26);
  EXPECT_EQ(count_ops<mlir::ncnn::ReluOp>(module), 26);
  EXPECT_EQ(count_ops<mlir::ncnn::SplitOp>(module), 8);
  EXPECT_EQ(count_ops<mlir::ncnn::ConcatOp>(module), 8);
  EXPECT_EQ(count_ops<mlir::ncnn::PoolingOp>(module), 4);
  EXPECT_EQ(count_ops<mlir::ncnn::SoftmaxOp>(module), 1);
  EXPECT_EQ(count_ops<mlir::ncnn::DropoutOp>(module), 1);

  EXPECT_TRUE(shape_is(result_type_by_name(module, "conv1"), {64, 113, 113}))
    << "conv1 shape is [64,113,113]";
  EXPECT_TRUE(shape_is(result_type_by_name(module, "pool1"), {64, 56, 56}))
    << "pool1 shape is [64,56,56]";
  EXPECT_TRUE(shape_is(result_type_by_name(module, "pool3"), {128, 28, 28}))
    << "pool3 shape is [128,28,28]";
  EXPECT_TRUE(shape_is(result_type_by_name(module, "pool5"), {256, 14, 14}))
    << "pool5 shape is [256,14,14]";
  EXPECT_TRUE(shape_is(result_type_by_name(module, "conv10"), {1000, 16, 16}))
    << "conv10 shape is [1000,16,16]";
  EXPECT_TRUE(shape_is(result_type_by_name(module, "pool10"), {1000}))
    << "global pool shape is [1000]";

  bool split_results = true;
  module->walk([&](mlir::ncnn::SplitOp split) {
    split_results = split_results && split.getNumResults() == 2;
  });
  EXPECT_TRUE(split_results) << "all Split operations have two results";
}
