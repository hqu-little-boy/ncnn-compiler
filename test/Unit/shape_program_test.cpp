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

TEST(ShapeProgramTest, ProvesEquivalentConstrainedExpressions) {
  mlir::ncnn::DimensionExpr pooled(0, 1);
  pooled.append(mlir::ncnn::ShapeOpcode::Add, 1);
  pooled.append(mlir::ncnn::ShapeOpcode::Add, -2);
  pooled.append(mlir::ncnn::ShapeOpcode::Divide, 1);
  pooled.append(mlir::ncnn::ShapeOpcode::Add, 1);

  mlir::ncnn::DimensionExpr identity(0, 1);
  EXPECT_TRUE(pooled.equivalentUnder({}, identity));

  mlir::ncnn::DimensionExpr quarterThenDouble(0, 1);
  quarterThenDouble.append(mlir::ncnn::ShapeOpcode::Divide, 4);
  quarterThenDouble.append(mlir::ncnn::ShapeOpcode::Multiply, 2);
  mlir::ncnn::DimensionExpr half(0, 1);
  half.append(mlir::ncnn::ShapeOpcode::Divide, 2);

  llvm::SmallVector<mlir::ncnn::ShapeConstraint> constraints{
    {.inputIndex = 0, .inputDimension = 1, .minimum = 32, .multipleOf = 32}};
  EXPECT_TRUE(quarterThenDouble.equivalentUnder(constraints, half));
  EXPECT_FALSE(quarterThenDouble.equivalentUnder({}, half));
}

TEST(ShapeProgramTest, ConsumesPowerOfTwoInputConstraint) {
  mlir::ncnn::DimensionExpr fiveHalvesThenDouble(0, 1);
  for (int step = 0; step < 5; ++step) {
    fiveHalvesThenDouble.append(mlir::ncnn::ShapeOpcode::Divide, 2);
  }
  fiveHalvesThenDouble.append(mlir::ncnn::ShapeOpcode::Multiply, 2);

  mlir::ncnn::DimensionExpr fourHalves(0, 1);
  for (int step = 0; step < 4; ++step) {
    fourHalves.append(mlir::ncnn::ShapeOpcode::Divide, 2);
  }

  llvm::SmallVector<mlir::ncnn::ShapeConstraint> constraints{
    {.inputIndex = 0, .inputDimension = 1, .minimum = 32, .multipleOf = 32}};
  EXPECT_TRUE(fiveHalvesThenDouble.equivalentUnder(constraints, fourHalves));
  EXPECT_FALSE(fiveHalvesThenDouble.equivalentUnder({}, fourHalves));
}

TEST(ShapeProgramTest, EvaluatesV2MultiInputExpression) {
  using mlir::ncnn::ShapeExpr;
  using mlir::ncnn::ShapeExprOpcode;
  auto expression =
    ShapeExpr::binary(ShapeExprOpcode::Max,
                      ShapeExpr::binary(ShapeExprOpcode::Add,
                                        ShapeExpr::inputDimension(0, 1),
                                        ShapeExpr::constant(3)),
                      ShapeExpr::binary(ShapeExprOpcode::Multiply,
                                        ShapeExpr::inputDimension(1, 0),
                                        ShapeExpr::constant(2)));
  llvm::SmallVector<int64_t> firstShape{8, 10};
  llvm::SmallVector<int64_t> secondShape{7};
  llvm::SmallVector<llvm::ArrayRef<int64_t>> shapes{firstShape, secondShape};
  auto result = expression.evaluateChecked(shapes);
  ASSERT_TRUE(result) << result.error();
  EXPECT_EQ(*result, 14);

  auto roundTrip = ShapeExpr::deserialize(expression.serialize());
  ASSERT_TRUE(roundTrip) << roundTrip.error();
  EXPECT_EQ(roundTrip->serialize(), expression.serialize());
}

TEST(ShapeProgramTest, EvaluatesV2FloorAndCeilDivision) {
  using mlir::ncnn::ShapeExpr;
  using mlir::ncnn::ShapeExprOpcode;
  llvm::SmallVector<llvm::ArrayRef<int64_t>> shapes;
  auto floor = ShapeExpr::binary(ShapeExprOpcode::FloorDivide,
                                 ShapeExpr::constant(-7),
                                 ShapeExpr::constant(3));
  auto ceil = ShapeExpr::binary(ShapeExprOpcode::CeilDivide,
                                ShapeExpr::constant(-7),
                                ShapeExpr::constant(3));
  EXPECT_EQ(floor.evaluateChecked(shapes), -3);
  EXPECT_EQ(ceil.evaluateChecked(shapes), -2);
  EXPECT_FALSE(ShapeExpr::binary(ShapeExprOpcode::FloorDivide,
                                 ShapeExpr::constant(1),
                                 ShapeExpr::constant(0))
                 .evaluateChecked(shapes));
}

TEST(ShapeProgramTest, RejectsInvalidV2Expression) {
  EXPECT_FALSE(
    mlir::ncnn::ShapeExpr::deserialize(llvm::ArrayRef<int64_t>{2, 0, 1}));
  EXPECT_FALSE(
    mlir::ncnn::ShapeExpr::deserialize(llvm::ArrayRef<int64_t>{1, 0, 3, 0}));
  EXPECT_FALSE(mlir::ncnn::ShapeExpr::deserialize(
    llvm::ArrayRef<int64_t>{1, std::numeric_limits<int64_t>::max(), 0}));
  llvm::SmallVector<int64_t> deep;
  for (int index = 0; index < 300; ++index) {
    deep.push_back(2);
    deep.append({0, 1});
  }
  deep.append({0, 1});
  EXPECT_FALSE(mlir::ncnn::ShapeExpr::deserialize(deep));
}

}  // namespace
