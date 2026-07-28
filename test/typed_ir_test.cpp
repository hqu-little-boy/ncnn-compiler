#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string_view>
#include <vector>

#include "ncnn_frontend/ir.hpp"
#include "ncnn_frontend/verifier.hpp"

namespace {

ncnn_frontend::TensorType make_type(std::vector<std::int64_t> shape,
                                    ncnn_frontend::TensorLayout layout) {
  auto type = ncnn_frontend::TensorType::create(
    std::move(shape), ncnn_frontend::ElementType::Float32, layout);
  if (!type) {
    std::cerr << type.error() << '\n';
    std::terminate();
  }
  return std::move(*type);
}

}  // namespace

TEST(TypedIrTest, ValidGraphVerifies) {
  using namespace ncnn_frontend;

  TensorType type = make_type({2}, TensorLayout::NcnnW);
  std::vector<Value> values;
  values.emplace_back(
    "input", type, GraphInputDef(0), std::vector<Use>{Use(OpId(0), 0)});
  values.emplace_back(
    "left", type, OpResultDef(OpId(0), 0), std::vector<Use>{Use(OpId(1), 0)});
  values.emplace_back(
    "right", type, OpResultDef(OpId(0), 1), std::vector<Use>{Use(OpId(1), 1)});
  values.emplace_back("output",
                      make_type({4}, TensorLayout::NcnnW),
                      OpResultDef(OpId(1), 0),
                      std::vector<Use>());
  std::vector<Operation> operations;
  operations.emplace_back("split",
                          SplitOp(),
                          std::vector<ValueId>{ValueId(0)},
                          std::vector<ValueId>{ValueId(1), ValueId(2)},
                          0);
  operations.emplace_back("concat",
                          ConcatOp(0),
                          std::vector<ValueId>{ValueId(1), ValueId(2)},
                          std::vector<ValueId>{ValueId(3)},
                          1);
  Graph valid(std::move(operations),
              std::move(values),
              std::vector<ValueId>{ValueId(0)},
              std::vector<ValueId>{ValueId(3)});
  EXPECT_TRUE(verify_graph(valid).has_value())
    << "valid graph argument, multi-result, and uses verify";
}

TEST(TypedIrTest, RepeatedUseVerifies) {
  using namespace ncnn_frontend;
  TensorType type = make_type({2}, TensorLayout::NcnnW);

  std::vector<Value> repeated_values;
  repeated_values.emplace_back(
    "input",
    type,
    GraphInputDef(0),
    std::vector<Use>{Use(OpId(0), 0), Use(OpId(0), 1)});
  repeated_values.emplace_back("output",
                               make_type({4}, TensorLayout::NcnnW),
                               OpResultDef(OpId(0), 0),
                               std::vector<Use>());
  Graph repeated(
    std::vector<Operation>{Operation(
      "concat", ConcatOp(0), {ValueId(0), ValueId(0)}, {ValueId(1)}, 0)},
    std::move(repeated_values),
    {ValueId(0)},
    {ValueId(1)});
  EXPECT_TRUE(verify_graph(repeated).has_value())
    << "same value used twice by one operation verifies";
}

TEST(TypedIrTest, MissingRepeatedUseRejected) {
  using namespace ncnn_frontend;
  TensorType type = make_type({2}, TensorLayout::NcnnW);

  std::vector<Value> missing_use_values;
  missing_use_values.emplace_back(
    "input", type, GraphInputDef(0), std::vector<Use>{Use(OpId(0), 0)});
  missing_use_values.emplace_back("output",
                                  make_type({4}, TensorLayout::NcnnW),
                                  OpResultDef(OpId(0), 0),
                                  std::vector<Use>());
  Graph missing_use(
    std::vector<Operation>{Operation(
      "concat", ConcatOp(0), {ValueId(0), ValueId(0)}, {ValueId(1)}, 0)},
    std::move(missing_use_values),
    {ValueId(0)},
    {ValueId(1)});
  EXPECT_FALSE(verify_graph(missing_use))
    << "missing repeated use is rejected";
}

TEST(TypedIrTest, OutOfRangeResultRejected) {
  using namespace ncnn_frontend;
  TensorType type = make_type({2}, TensorLayout::NcnnW);

  std::vector<Value> bad_result_values;
  bad_result_values.emplace_back(
    "input", type, GraphInputDef(0), std::vector<Use>{Use(OpId(0), 0)});
  bad_result_values.emplace_back(
    "output", type, OpResultDef(OpId(1), 0), std::vector<Use>());
  Graph bad_result(std::vector<Operation>{Operation(
                     "relu", ReluOp(0.0f), {ValueId(0)}, {ValueId(1)}, 0)},
                   std::move(bad_result_values),
                   {ValueId(0)},
                   {ValueId(1)});
  EXPECT_FALSE(verify_graph(bad_result))
    << "out-of-range result definition is rejected";
}

TEST(TypedIrTest, CycleRejected) {
  using namespace ncnn_frontend;
  TensorType type = make_type({2}, TensorLayout::NcnnW);

  std::vector<Value> backward_values;
  backward_values.emplace_back(
    "later", type, OpResultDef(OpId(1), 0), std::vector<Use>{Use(OpId(0), 0)});
  backward_values.emplace_back(
    "early", type, OpResultDef(OpId(0), 0), std::vector<Use>{Use(OpId(1), 0)});
  Graph cycle(
    std::vector<Operation>{
      Operation("first", ReluOp(0.0f), {ValueId(0)}, {ValueId(1)}, 0),
      Operation("second", ReluOp(0.0f), {ValueId(1)}, {ValueId(0)}, 1)},
    std::move(backward_values),
    {},
    {ValueId(0)});
  EXPECT_FALSE(verify_graph(cycle))
    << "backward dependency and cycle are rejected";
}

TEST(TypedIrTest, InvalidEnumsRejected) {
  using namespace ncnn_frontend;
  EXPECT_FALSE(TensorType::create(
    {1}, static_cast<ElementType>(255), TensorLayout::NcnnW))
    << "invalid element type enum is rejected";
  EXPECT_FALSE(TensorType::create(
    {1}, ElementType::Float32, static_cast<TensorLayout>(255)))
    << "invalid tensor layout enum is rejected";
}

TEST(TypedIrTest, NegativeReluSlopeAccepted) {
  using namespace ncnn_frontend;
  TensorType type = make_type({2}, TensorLayout::NcnnW);

  std::vector<Value> negative_relu_values;
  negative_relu_values.emplace_back(
    "input", type, GraphInputDef(0), std::vector<Use>{Use(OpId(0), 0)});
  negative_relu_values.emplace_back(
    "output", type, OpResultDef(OpId(0), 0), std::vector<Use>());
  Graph negative_relu(
    {Operation("relu", ReluOp(-0.25f), {ValueId(0)}, {ValueId(1)}, 0)},
    std::move(negative_relu_values),
    {ValueId(0)},
    {ValueId(1)});
  EXPECT_TRUE(verify_graph(negative_relu).has_value())
    << "finite negative ReLU slope is accepted";
}

TEST(TypedIrTest, MovedFromConstantRejected) {
  using namespace ncnn_frontend;

  TensorType scalar_type = make_type({}, TensorLayout::Scalar);
  auto literal =
    TensorLiteral::create(scalar_type, std::vector<std::byte>(sizeof(float)));
  ASSERT_TRUE(literal.has_value()) << (literal ? "" : literal.error());
  ConstOp moved_from(std::move(*literal));
  ConstOp owner(std::move(moved_from));
  static_cast<void>(owner);
  Graph bad_constant(
    {Operation("bad_const", std::move(moved_from), {}, {ValueId(0)}, 0)},
    {Value(
      "constant", scalar_type, OpResultDef(OpId(0), 0), std::vector<Use>())},
    {},
    {ValueId(0)});
  EXPECT_FALSE(verify_graph(bad_constant))
    << "moved-from constant payload is rejected by verifier";
}

TEST(TypedIrTest, LongChainVerifies) {
  using namespace ncnn_frontend;
  TensorType type = make_type({2}, TensorLayout::NcnnW);

  constexpr std::size_t kLongChainLength = 4096;
  std::vector<Operation> long_operations;
  std::vector<Value> long_values;
  long_operations.reserve(kLongChainLength);
  long_values.reserve(kLongChainLength + 1);
  long_values.emplace_back(
    "input", type, GraphInputDef(0), std::vector<Use>{Use(OpId(0), 0)});
  for (std::size_t index = 0; index < kLongChainLength; ++index) {
    const bool last = index + 1 == kLongChainLength;
    std::vector<Use> uses;
    if (!last) {
      uses.emplace_back(OpId(index + 1), 0);
    }
    long_values.emplace_back(std::format("value_{}", index),
                             type,
                             OpResultDef(OpId(index), 0),
                             std::move(uses));
    long_operations.emplace_back(std::format("relu_{}", index),
                                 ReluOp(0.0f),
                                 std::vector<ValueId>{ValueId(index)},
                                 std::vector<ValueId>{ValueId(index + 1)},
                                 index);
  }
  Graph long_chain(std::move(long_operations),
                   std::move(long_values),
                   {ValueId(0)},
                   {ValueId(kLongChainLength)});
  EXPECT_TRUE(verify_graph(long_chain).has_value())
    << "long operation chain verifies without recursive traversal";
}
