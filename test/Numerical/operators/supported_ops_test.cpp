#include "numerical_test_support.hpp"

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

std::string fixture_path(std::string_view name) {
  return std::string(NUMERICAL_FIXTURE_DIR) + "/" + std::string(name) +
         ".param";
}

void expect_single_input_operator(std::string_view name,
                                  std::string_view library_path,
                                  TensorShape input_shape,
                                  std::size_t output_elements,
                                  float tolerance,
                                  std::uint32_t seed) {
  const std::vector<float> input =
    make_random_input(input_shape.element_count(), seed, -2.0F, 2.0F);
  const ReferenceModel reference{.param_path = fixture_path(name),
                                 .bin_path = name == "convolution"
                                               ? CONVOLUTION_BIN_PATH
                                               : NUMERICAL_EMPTY_BIN_PATH,
                                 .input_blob = "data",
                                 .output_blob = "output",
                                 .input_shape = input_shape};
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), output_elements);

  const std::string symbol = std::string(name) + "_scalar";
  CompiledModel compiled(library_path, symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(output_elements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, tolerance));
}

TEST(NumericalOperator, ConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution",
                               CONVOLUTION_LIBRARY_PATH,
                               {.width = 4, .height = 4, .channels = 1},
                               32,
                               1.0e-5F,
                               0x434F4E56U);
}

TEST(NumericalOperator, ReluMatchesNcnn) {
  expect_single_input_operator("relu",
                               RELU_LIBRARY_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               0.0F,
                               0x52454C55U);
}

TEST(NumericalOperator, PoolingMatchesNcnn) {
  expect_single_input_operator("pooling",
                               POOLING_LIBRARY_PATH,
                               {.width = 5, .height = 5, .channels = 2},
                               8,
                               0.0F,
                               0x504F4F4CU);
}

TEST(NumericalOperator, DropoutMatchesNcnn) {
  expect_single_input_operator("dropout",
                               DROPOUT_LIBRARY_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               0.0F,
                               0x44524F50U);
}

TEST(NumericalOperator, SoftmaxMatchesNcnn) {
  expect_single_input_operator("softmax",
                               SOFTMAX_LIBRARY_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               1.0e-6F,
                               0x534F4654U);
}

TEST(NumericalOperator, SplitMatchesNcnn) {
  constexpr TensorShape kShape{.width = 5, .height = 4, .channels = 3};
  const std::vector<float> input =
    make_random_input(kShape.element_count(), 0x53504C49U, -2.0F, 2.0F);
  const ReferenceInput reference_input{
    .blob_name = "data", .shape = kShape, .values = input};
  constexpr std::array<std::string_view, 2> kOutputs{"left", "right"};
  const auto expected = run_ncnn_reference(fixture_path("split"),
                                           NUMERICAL_EMPTY_BIN_PATH,
                                           std::span(&reference_input, 1),
                                           kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 2);

  CompiledModel compiled(SPLIT_LIBRARY_PATH, "split_scalar");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> left(kShape.element_count());
  std::vector<float> right(kShape.element_count());
  ASSERT_EQ(compiled.run_two_outputs(input, left, right), 0);
  EXPECT_TRUE(compare_values(left, (*expected)[0], 0.0F));
  EXPECT_TRUE(compare_values(right, (*expected)[1], 0.0F));
}

TEST(NumericalOperator, ConcatMatchesNcnn) {
  constexpr TensorShape kFirstShape{.width = 4, .height = 3, .channels = 1};
  constexpr TensorShape kSecondShape{.width = 4, .height = 3, .channels = 2};
  const std::vector<float> first =
    make_random_input(kFirstShape.element_count(), 0x43415431U, -2.0F, 2.0F);
  const std::vector<float> second =
    make_random_input(kSecondShape.element_count(), 0x43415432U, -2.0F, 2.0F);
  const std::array<ReferenceInput, 2> inputs{
    ReferenceInput{.blob_name = "first", .shape = kFirstShape, .values = first},
    ReferenceInput{
      .blob_name = "second", .shape = kSecondShape, .values = second}};
  constexpr std::array<std::string_view, 1> kOutputs{"output"};
  const auto expected = run_ncnn_reference(
    fixture_path("concat"), NUMERICAL_EMPTY_BIN_PATH, inputs, kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 1);

  CompiledModel compiled(CONCAT_LIBRARY_PATH, "concat_scalar");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kFirstShape.element_count() +
                            kSecondShape.element_count());
  ASSERT_EQ(compiled.run_two_inputs(first, second, actual), 0);
  EXPECT_TRUE(compare_values(actual, expected->front(), 0.0F));
}

}  // namespace
}  // namespace ncnn_compiler::test
