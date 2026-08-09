#include "numerical_test_support.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

constexpr int kSuccess = 0;

struct Case {
  std::int64_t height;
  std::int64_t width;
};

constexpr std::array<Case, 2> kCases{
  {Case{.height = 3, .width = 5}, Case{.height = 7, .width = 11}}};

std::string fixture_path(std::string_view name) {
  return std::string(NUMERICAL_FIXTURE_DIR) + "/" + std::string(name) +
         ".param";
}

void expect_dynamic_identity(std::string_view fixture,
                             std::string_view library,
                             std::string_view symbol,
                             int channels) {
  CompiledModel compiled(library, symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(library, std::string(symbol) + "_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{
      channels, shape.height, shape.width};
    const std::size_t elements = static_cast<std::size_t>(channels) *
                                 static_cast<std::size_t>(shape.height) *
                                 static_cast<std::size_t>(shape.width);
    const std::vector<float> input =
      make_random_input(elements, 0xD1A000U, -2.0F, 2.0F);
    std::array<std::int64_t, 3> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, dimensions);

    const ReferenceModel reference(
      fixture_path(fixture),
      NUMERICAL_EMPTY_BIN_PATH,
      "data",
      "output",
      TensorShape(shape.width, shape.height, channels));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), elements);
    std::vector<float> actual(elements);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  }
}

TEST(NumericalDynamicOperator, ReluMatchesNcnnAcrossShapes) {
  expect_dynamic_identity("relu", RELU_DYNAMIC_LIBRARY_PATH, "relu_dynamic", 3);
}

TEST(NumericalDynamicOperator, SigmoidMatchesNcnnAcrossShapes) {
  expect_dynamic_identity(
    "sigmoid", SIGMOID_DYNAMIC_LIBRARY_PATH, "sigmoid_dynamic", 3);
}

TEST(NumericalDynamicOperator, PaddingIdentityMatchesNcnnAcrossShapes) {
  expect_dynamic_identity(
    "padding_identity", PADDING_DYNAMIC_LIBRARY_PATH, "padding_dynamic", 2);
}

TEST(NumericalDynamicOperator, InterpMatchesNcnnAcrossShapes) {
  CompiledModel compiled(INTERP_DYNAMIC_LIBRARY_PATH, "interp_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(INTERP_DYNAMIC_LIBRARY_PATH,
                      "interp_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{4, shape.height, shape.width};
    const std::array<std::int64_t, 3> expected_shape{
      4, shape.height * 2, shape.width * 2};
    const std::size_t input_elements = 4U * shape.height * shape.width;
    const std::size_t output_elements =
      4U * expected_shape[1] * expected_shape[2];
    const std::vector<float> input =
      make_random_input(input_elements, 0x1A7E2U, -1.0F, 1.0F);
    std::array<std::int64_t, 3> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, expected_shape);
    const ReferenceModel reference(fixture_path("interp"),
                                   NUMERICAL_EMPTY_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(shape.width, shape.height, 4));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), output_elements);
    std::vector<float> actual(output_elements);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
  }
}

TEST(NumericalDynamicOperator, HardActivationsMatchNcnnAcrossShapes) {
  expect_dynamic_identity("hard_sigmoid_dynamic",
                          HARD_SIGMOID_DYNAMIC_LIBRARY_PATH,
                          "hard_sigmoid_dynamic",
                          3);
  expect_dynamic_identity("hard_swish_dynamic",
                          HARD_SWISH_DYNAMIC_LIBRARY_PATH,
                          "hard_swish_dynamic",
                          3);
}

TEST(NumericalDynamicOperator, GeluDropoutAndSoftmaxMatchNcnnAcrossShapes) {
  expect_dynamic_identity(
    "gelu_dynamic", GELU_DYNAMIC_LIBRARY_PATH, "gelu_dynamic", 3);
  expect_dynamic_identity(
    "dropout_dynamic", DROPOUT_DYNAMIC_LIBRARY_PATH, "dropout_dynamic", 3);
  expect_dynamic_identity(
    "softmax_dynamic", SOFTMAX_DYNAMIC_LIBRARY_PATH, "softmax_dynamic", 3);
}

TEST(NumericalDynamicOperator, BatchNormMatchesNcnnAcrossShapes) {
  CompiledModel compiled(BATCH_NORM_DYNAMIC_LIBRARY_PATH, "batch_norm_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(BATCH_NORM_DYNAMIC_LIBRARY_PATH,
                      "batch_norm_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{3, shape.height, shape.width};
    const std::size_t elements = 3U * shape.height * shape.width;
    const std::vector<float> input =
      make_random_input(elements, 0xBADD00U, -2.0F, 2.0F);
    std::array<std::int64_t, 3> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, dimensions);
    const ReferenceModel reference(fixture_path("batch_norm_dynamic"),
                                   BATCH_NORM_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(shape.width, shape.height, 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> actual(elements);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-5F));
  }
}

TEST(NumericalDynamicOperator, ShuffleChannelMatchesNcnnAcrossShapes) {
  expect_dynamic_identity("shuffle_channel",
                          SHUFFLE_CHANNEL_DYNAMIC_LIBRARY_PATH,
                          "shuffle_channel_dynamic",
                          4);
}

TEST(NumericalDynamicOperator, RejectsInsufficientCapacity) {
  CompiledModel compiled(RELU_DYNAMIC_LIBRARY_PATH, "relu_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  const std::array<std::int64_t, 3> dimensions{3, 3, 5};
  const std::vector<float> input(45, 0.0F);
  std::vector<float> output(44, 0.0F);
  EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()), 5);
}

}  // namespace
}  // namespace ncnn_compiler::test
