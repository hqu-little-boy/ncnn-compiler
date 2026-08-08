#include <cstdint>
#include <limits>
#include <ranges>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/ShapeProgram.hpp"

namespace {

TEST(ShapeProgramTest, NormalizesAndComparesExpressions) {
  mlir::ncnn::DimensionExpr first(0, 1);
  first.append(mlir::ncnn::ShapeOpcode::Add, 2);
  first.append(mlir::ncnn::ShapeOpcode::Add, 3);
  first.append(mlir::ncnn::ShapeOpcode::Multiply, 1);

  mlir::ncnn::DimensionExpr second(0, 1);
  second.append(mlir::ncnn::ShapeOpcode::Add, 5);

  EXPECT_EQ(first, second);
  EXPECT_FALSE(first.isIdentity());
  EXPECT_TRUE(
    std::ranges::equal(first.serialize(), llvm::ArrayRef<int64_t>{0, 5}));
}

TEST(ShapeProgramTest, EvaluatesAndRejectsOverflow) {
  mlir::ncnn::DimensionExpr expression(0, 2);
  expression.append(mlir::ncnn::ShapeOpcode::Add, 4);
  expression.append(mlir::ncnn::ShapeOpcode::Multiply, 2);
  expression.append(mlir::ncnn::ShapeOpcode::Divide, 4);
  auto result = expression.evaluateChecked(28);
  ASSERT_TRUE(result) << result.error();
  EXPECT_EQ(*result, 16);

  mlir::ncnn::DimensionExpr overflow(0, 2);
  overflow.append(mlir::ncnn::ShapeOpcode::Add, 1);
  EXPECT_FALSE(overflow.evaluateChecked(std::numeric_limits<int64_t>::max()));
}

TEST(ShapeProgramTest, DeserializesValidatedInstructions) {
  auto expression = mlir::ncnn::DimensionExpr::deserialize(
    1, 2, llvm::ArrayRef<int64_t>{0, 3, 1, 2, 2, 4});
  ASSERT_TRUE(expression) << expression.error();
  EXPECT_EQ(expression->getInputIndex(), 1U);
  EXPECT_EQ(expression->getInputDimension(), 2U);
  EXPECT_EQ(expression->evaluateChecked(13), 8);

  EXPECT_FALSE(mlir::ncnn::DimensionExpr::deserialize(
    0, 0, llvm::ArrayRef<int64_t>{2, 0}));
  EXPECT_FALSE(mlir::ncnn::DimensionExpr::deserialize(
    0, 0, llvm::ArrayRef<int64_t>{4, 1}));
}

}  // namespace
