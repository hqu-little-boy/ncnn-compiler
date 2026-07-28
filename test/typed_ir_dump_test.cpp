#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn_frontend/ir.hpp"

namespace {

ncnn_frontend::TensorType make_type(std::vector<std::int64_t> shape,
                                    ncnn_frontend::ElementType element_type,
                                    ncnn_frontend::TensorLayout layout) {
  auto type =
    ncnn_frontend::TensorType::create(std::move(shape), element_type, layout);
  if (!type) {
    std::cerr << type.error() << '\n';
    std::terminate();
  }
  return std::move(*type);
}

ncnn_frontend::TensorLiteral make_hello_literal() {
  auto type = make_type(
    {5}, ncnn_frontend::ElementType::Int8, ncnn_frontend::TensorLayout::NcnnW);
  std::vector<std::byte> data = {std::byte{'h'},
                                 std::byte{'e'},
                                 std::byte{'l'},
                                 std::byte{'l'},
                                 std::byte{'o'}};
  auto literal =
    ncnn_frontend::TensorLiteral::create(std::move(type), std::move(data));
  if (!literal) {
    std::cerr << literal.error() << '\n';
    std::terminate();
  }
  return std::move(*literal);
}

}  // namespace

TEST(TypedIrDumpTest, CanonicalDumpMatchesGolden) {
  using namespace ncnn_frontend;

  auto byte_type = make_type({5}, ElementType::Int8, TensorLayout::NcnnW);
  std::vector<Value> values;
  values.emplace_back("const\"\\\n",
                      byte_type,
                      OpResultDef(OpId(0), 0),
                      std::vector<Use>{Use(OpId(1), 0)});
  values.emplace_back(
    "output", byte_type, OpResultDef(OpId(1), 0), std::vector<Use>());
  std::vector<Operation> operations;
  operations.emplace_back("constant",
                          ConstOp(make_hello_literal()),
                          std::vector<ValueId>(),
                          std::vector<ValueId>{ValueId(0)},
                          4);
  operations.emplace_back("relu",
                          ReluOp(-0.25f),
                          std::vector<ValueId>{ValueId(0)},
                          std::vector<ValueId>{ValueId(1)},
                          5);
  Graph graph(std::move(operations),
              std::move(values),
              std::vector<ValueId>(),
              std::vector<ValueId>{ValueId(1)});

  constexpr std::string_view kExpected =
    "ncnn_frontend.typed_dag_dump version=1\n"
    "operations 2\n"
    "op 0 {kind=const,attrs={literal_type={shape=[5],element=i8,layout="
    "ncnn_w,elements=5,bytes=5},payload_bytes=5,fnv1a64="
    "0xa430d84680aabd0b},name=\"constant\",source_layer=4,operands=[],"
    "results=[v0]}\n"
    "op 1 {kind=relu,attrs={negative_slope=-0.25},name=\"relu\","
    "source_layer=5,operands=[v0],results=[v1]}\n"
    "values 2\n"
    "value 0 {name=\"const\\\"\\\\\\n\",type={shape=[5],element=i8,"
    "layout=ncnn_w,elements=5,bytes=5},def=op_result(op0,0),"
    "uses=[{user=op1,operand=0}]}\n"
    "value 1 {name=\"output\",type={shape=[5],element=i8,layout=ncnn_w,"
    "elements=5,bytes=5},def=op_result(op1,0),uses=[]}\n"
    "inputs []\n"
    "outputs [v1]\n";
  const std::string dumped = graph.dump();
  EXPECT_EQ(dumped, kExpected)
    << "canonical dump matches the version 1 golden";
  EXPECT_EQ(graph.dump(), dumped)
    << "repeated dump is byte-for-byte stable";
  EXPECT_EQ(dumped.find("hello"), std::string::npos)
    << "constant payload is summarized rather than expanded";
}

TEST(TypedIrDumpTest, MalformedReferencesMarkedSafely) {
  using namespace ncnn_frontend;
  auto byte_type = make_type({5}, ElementType::Int8, TensorLayout::NcnnW);

  Graph malformed(
    {Operation("bad", ReluOp(-0.0f), {ValueId(99)}, {ValueId(98)}, 0)},
    {Value(
      "bad_value", byte_type, OpResultDef(OpId(88), 7), {Use(OpId(77), 6)})},
    {ValueId(66)},
    {ValueId(55)});
  const std::string malformed_dump = malformed.dump();
  EXPECT_TRUE(malformed_dump.find("v99!out_of_range") != std::string::npos &&
              malformed_dump.find("v98!out_of_range") != std::string::npos &&
              malformed_dump.find("op88!out_of_range") != std::string::npos &&
              malformed_dump.find("op77!out_of_range") != std::string::npos &&
              malformed_dump.find("v66!out_of_range") != std::string::npos &&
              malformed_dump.find("v55!out_of_range") != std::string::npos)
    << "unverified graph references are preserved and marked safely";
  EXPECT_NE(malformed_dump.find("negative_slope=-0"), std::string::npos)
    << "negative zero has a stable spelling";
}

TEST(TypedIrDumpTest, AttributeVariantsHaveStableForms) {
  using namespace ncnn_frontend;

  std::vector<Operation> attribute_operations;
  attribute_operations.emplace_back("const",
                                    ConstOp(make_hello_literal()),
                                    std::vector<ValueId>(),
                                    std::vector<ValueId>(),
                                    0);
  attribute_operations.emplace_back(
    "conv",
    Conv2DOp(3, 5, 2, 4, 1, 2, 1, 2, 3, 4, true, 102),
    std::vector<ValueId>(),
    std::vector<ValueId>(),
    1);
  attribute_operations.emplace_back(
    "relu", ReluOp(0.0f), std::vector<ValueId>(), std::vector<ValueId>(), 2);
  attribute_operations.emplace_back("pool",
                                    Pool2DOp(static_cast<PoolKind>(9),
                                             static_cast<PoolMode>(8),
                                             3,
                                             3,
                                             2,
                                             2,
                                             0,
                                             1,
                                             2,
                                             3,
                                             0,
                                             false),
                                    std::vector<ValueId>(),
                                    std::vector<ValueId>(),
                                    3);
  attribute_operations.emplace_back(
    "split", SplitOp(), std::vector<ValueId>(), std::vector<ValueId>(), 4);
  attribute_operations.emplace_back(
    "concat", ConcatOp(-1), std::vector<ValueId>(), std::vector<ValueId>(), 5);
  attribute_operations.emplace_back("dropout",
                                    DropoutOp(0.5f),
                                    std::vector<ValueId>(),
                                    std::vector<ValueId>(),
                                    6);
  attribute_operations.emplace_back(
    "softmax", SoftmaxOp(2), std::vector<ValueId>(), std::vector<ValueId>(), 7);
  Graph attributes(std::move(attribute_operations), {}, {}, {});
  const std::string attributes_dump = attributes.dump();
  EXPECT_TRUE(
    attributes_dump.find("kind=conv2d") != std::string::npos &&
    attributes_dump.find("quantization=requantize") != std::string::npos &&
    attributes_dump.find("kind=split,attrs={}") != std::string::npos &&
    attributes_dump.find("kind=concat,attrs={axis=-1}") != std::string::npos &&
    attributes_dump.find("kind=dropout,attrs={scale=0.5}") !=
      std::string::npos &&
    attributes_dump.find("kind=softmax,attrs={axis=2}") != std::string::npos)
    << "all operation attribute variants have stable dump forms";
  EXPECT_TRUE(attributes_dump.find("kind=invalid(9)") != std::string::npos &&
              attributes_dump.find("mode=invalid(8)") != std::string::npos)
    << "invalid enum values are printed without undefined behavior";
}

TEST(TypedIrDumpTest, EmptyGraphDump) {
  using namespace ncnn_frontend;
  Graph empty({}, {}, {}, {});
  EXPECT_EQ(empty.dump(),
            "ncnn_frontend.typed_dag_dump version=1\noperations 0\nvalues 0\n"
            "inputs []\noutputs []\n")
    << "empty graph has a complete canonical dump";
}
