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
                                  std::string_view bin_path,
                                  TensorShape input_shape,
                                  std::size_t output_elements,
                                  float tolerance,
                                  std::uint32_t seed) {
  const std::vector<float> input =
    make_random_input(input_shape.element_count(), seed, -2.0F, 2.0F);
  const ReferenceModel reference{.param_path = fixture_path(name),
                                 .bin_path = std::string(bin_path),
                                 .input_blob = "data",
                                 .output_blob = "output",
                                 .input_shape = input_shape};
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), output_elements);

  const std::string symbol = name == "convolution" || name == "relu" ||
                                 name == "pooling" || name == "dropout" ||
                                 name == "softmax"
                               ? std::string(name) + "_scalar"
                               : std::string(name);
  CompiledModel compiled(library_path, symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(output_elements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, tolerance));
}

void expect_concat(std::string_view name,
                   std::string_view library_path,
                   TensorShape first_shape,
                   TensorShape second_shape,
                   std::uint32_t first_seed,
                   std::uint32_t second_seed) {
  const std::vector<float> first =
    make_random_input(first_shape.element_count(), first_seed, -2.0F, 2.0F);
  const std::vector<float> second =
    make_random_input(second_shape.element_count(), second_seed, -2.0F, 2.0F);
  const std::array<ReferenceInput, 2> inputs{
    ReferenceInput{.blob_name = "first", .shape = first_shape, .values = first},
    ReferenceInput{
      .blob_name = "second", .shape = second_shape, .values = second}};
  constexpr std::array<std::string_view, 1> kOutputs{"output"};
  const auto expected = run_ncnn_reference(
    fixture_path(name), NUMERICAL_EMPTY_BIN_PATH, inputs, kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 1);

  const std::string symbol =
    name == "concat" ? "concat_scalar" : std::string(name);
  CompiledModel compiled(library_path, symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(first_shape.element_count() +
                            second_shape.element_count());
  ASSERT_EQ(compiled.run_two_inputs(first, second, actual), 0);
  EXPECT_TRUE(compare_values(actual, expected->front(), 0.0F));
}

TEST(NumericalOperator, ConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution",
                               CONVOLUTION_LIBRARY_PATH,
                               CONVOLUTION_BIN_PATH,
                               {.width = 4, .height = 4, .channels = 1},
                               32,
                               1.0e-5F,
                               0x434F4E56U);
}

TEST(NumericalOperator, ConvolutionWithoutBiasMatchesNcnn) {
  expect_single_input_operator("convolution_no_bias",
                               CONVOLUTION_NO_BIAS_LIBRARY_PATH,
                               CONVOLUTION_NO_BIAS_BIN_PATH,
                               {.width = 4, .height = 4, .channels = 1},
                               32,
                               1.0e-5F,
                               0x434E4249U);
}

TEST(NumericalOperator, DilatedConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution_dilated",
                               CONVOLUTION_DILATED_LIBRARY_PATH,
                               CONVOLUTION_DILATED_BIN_PATH,
                               {.width = 7, .height = 7, .channels = 1},
                               128,
                               1.0e-5F,
                               0x4344494CU);
}

TEST(NumericalOperator, AsymmetricConvolutionPaddingMatchesNcnn) {
  expect_single_input_operator("convolution_asymmetric",
                               CONVOLUTION_ASYMMETRIC_LIBRARY_PATH,
                               CONVOLUTION_ASYMMETRIC_BIN_PATH,
                               {.width = 4, .height = 4, .channels = 1},
                               48,
                               1.0e-5F,
                               0x43415359U);
}

TEST(NumericalOperator, ConvolutionSameUpperMatchesNcnn) {
  expect_single_input_operator("convolution_same_upper",
                               CONVOLUTION_SAME_UPPER_LIBRARY_PATH,
                               CONVOLUTION_SAME_UPPER_BIN_PATH,
                               {.width = 5, .height = 5, .channels = 1},
                               18,
                               1.0e-5F,
                               0x43535550U);
}

TEST(NumericalOperator, ConvolutionSameLowerMatchesNcnn) {
  expect_single_input_operator("convolution_same_lower",
                               CONVOLUTION_SAME_LOWER_LIBRARY_PATH,
                               CONVOLUTION_SAME_LOWER_BIN_PATH,
                               {.width = 5, .height = 5, .channels = 1},
                               18,
                               1.0e-5F,
                               0x43534C4FU);
}

TEST(NumericalOperator, ReluMatchesNcnn) {
  expect_single_input_operator("relu",
                               RELU_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               0.0F,
                               0x52454C55U);
}

TEST(NumericalOperator, ReluHandlesZeroAndNegativeValues) {
  constexpr TensorShape kShape{.width = 5, .height = 4, .channels = 3};
  std::vector<float> input(kShape.element_count());
  for (std::size_t index = 0; index < input.size(); ++index) {
    constexpr std::array<float, 5> kValues{-2.0F, -0.0F, 0.0F, 0.5F, 2.0F};
    input[index] = kValues[index % kValues.size()];
  }
  const ReferenceModel reference{.param_path = fixture_path("relu"),
                                 .bin_path = NUMERICAL_EMPTY_BIN_PATH,
                                 .input_blob = "data",
                                 .output_blob = "output",
                                 .input_shape = kShape};
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(RELU_LIBRARY_PATH, "relu_scalar");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kShape.element_count());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
}

TEST(NumericalOperator, LeakyReluMatchesNcnn) {
  expect_single_input_operator("relu_leaky",
                               RELU_LEAKY_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               0.0F,
                               0x4C52454CU);
}

TEST(NumericalOperator, PoolingMatchesNcnn) {
  expect_single_input_operator("pooling",
                               POOLING_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 5, .channels = 2},
                               8,
                               0.0F,
                               0x504F4F4CU);
}

TEST(NumericalOperator, AveragePoolingExcludingPadMatchesNcnn) {
  expect_single_input_operator("pooling_average",
                               POOLING_AVERAGE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 5, .channels = 2},
                               18,
                               1.0e-6F,
                               0x50415647U);
}

TEST(NumericalOperator, GlobalMaxPoolingMatchesNcnn) {
  expect_single_input_operator("pooling_global_max",
                               POOLING_GLOBAL_MAX_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 5, .channels = 2},
                               2,
                               0.0F,
                               0x50474D58U);
}

TEST(NumericalOperator, GlobalAveragePoolingMatchesNcnn) {
  expect_single_input_operator("pooling_global_average",
                               POOLING_GLOBAL_AVERAGE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 5, .channels = 2},
                               2,
                               1.0e-6F,
                               0x50474156U);
}

TEST(NumericalOperator, PoolingSameUpperMatchesNcnn) {
  expect_single_input_operator("pooling_same_upper",
                               POOLING_SAME_UPPER_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 2},
                               12,
                               0.0F,
                               0x50535550U);
}

TEST(NumericalOperator, PoolingSameLowerMatchesNcnn) {
  expect_single_input_operator("pooling_same_lower",
                               POOLING_SAME_LOWER_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 2},
                               12,
                               0.0F,
                               0x50534C4FU);
}

TEST(NumericalOperator, PoolingTailWindowMatchesNcnn) {
  expect_single_input_operator("pooling_tail",
                               POOLING_TAIL_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 4, .height = 4, .channels = 2},
                               8,
                               0.0F,
                               0x50544149U);
}

TEST(NumericalOperator, DropoutMatchesNcnn) {
  expect_single_input_operator("dropout",
                               DROPOUT_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               0.0F,
                               0x44524F50U);
}

TEST(NumericalOperator, SoftmaxMatchesNcnn) {
  expect_single_input_operator("softmax",
                               SOFTMAX_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               60,
                               1.0e-6F,
                               0x534F4654U);
}

#define NCNN_SOFTMAX_AXIS_TEST(test_name, fixture_name, library_path, seed) \
  TEST(NumericalOperator, test_name) {                                      \
    expect_single_input_operator(fixture_name,                              \
                                 library_path,                              \
                                 NUMERICAL_EMPTY_BIN_PATH,                  \
                                 {.width = 5, .height = 4, .channels = 3},  \
                                 60,                                        \
                                 1.0e-6F,                                   \
                                 seed);                                     \
  }

NCNN_SOFTMAX_AXIS_TEST(SoftmaxChannelAxisMatchesNcnn,
                       "softmax_channel",
                       SOFTMAX_CHANNEL_LIBRARY_PATH,
                       0x534D4348U)
NCNN_SOFTMAX_AXIS_TEST(SoftmaxHeightAxisMatchesNcnn,
                       "softmax_height",
                       SOFTMAX_HEIGHT_LIBRARY_PATH,
                       0x534D4847U)
NCNN_SOFTMAX_AXIS_TEST(SoftmaxWidthAxisMatchesNcnn,
                       "softmax_width",
                       SOFTMAX_WIDTH_LIBRARY_PATH,
                       0x534D5744U)
NCNN_SOFTMAX_AXIS_TEST(SoftmaxNegativeChannelAxisMatchesNcnn,
                       "softmax_negative_channel",
                       SOFTMAX_NEGATIVE_CHANNEL_LIBRARY_PATH,
                       0x534E4348U)
NCNN_SOFTMAX_AXIS_TEST(SoftmaxNegativeHeightAxisMatchesNcnn,
                       "softmax_negative_height",
                       SOFTMAX_NEGATIVE_HEIGHT_LIBRARY_PATH,
                       0x534E4847U)
NCNN_SOFTMAX_AXIS_TEST(SoftmaxNegativeWidthAxisMatchesNcnn,
                       "softmax_negative_width",
                       SOFTMAX_NEGATIVE_WIDTH_LIBRARY_PATH,
                       0x534E5744U)

#undef NCNN_SOFTMAX_AXIS_TEST

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
  expect_concat("concat",
                CONCAT_LIBRARY_PATH,
                {.width = 4, .height = 3, .channels = 1},
                {.width = 4, .height = 3, .channels = 2},
                0x43415431U,
                0x43415432U);
}

#define NCNN_CONCAT_AXIS_TEST(                                                 \
  test_name, fixture_name, library_path, first_shape, second_shape, seed)      \
  TEST(NumericalOperator, test_name) {                                         \
    expect_concat(                                                             \
      fixture_name, library_path, first_shape, second_shape, seed, seed + 1U); \
  }

NCNN_CONCAT_AXIS_TEST(ConcatHeightAxisMatchesNcnn,
                      "concat_height",
                      CONCAT_HEIGHT_LIBRARY_PATH,
                      (TensorShape{.width = 4, .height = 3, .channels = 2}),
                      (TensorShape{.width = 4, .height = 2, .channels = 2}),
                      0x43414854U)
NCNN_CONCAT_AXIS_TEST(ConcatWidthAxisMatchesNcnn,
                      "concat_width",
                      CONCAT_WIDTH_LIBRARY_PATH,
                      (TensorShape{.width = 4, .height = 3, .channels = 2}),
                      (TensorShape{.width = 2, .height = 3, .channels = 2}),
                      0x43415744U)
NCNN_CONCAT_AXIS_TEST(ConcatNegativeChannelAxisMatchesNcnn,
                      "concat_negative_channel",
                      CONCAT_NEGATIVE_CHANNEL_LIBRARY_PATH,
                      (TensorShape{.width = 4, .height = 3, .channels = 1}),
                      (TensorShape{.width = 4, .height = 3, .channels = 2}),
                      0x434E4348U)
NCNN_CONCAT_AXIS_TEST(ConcatNegativeHeightAxisMatchesNcnn,
                      "concat_negative_height",
                      CONCAT_NEGATIVE_HEIGHT_LIBRARY_PATH,
                      (TensorShape{.width = 4, .height = 3, .channels = 2}),
                      (TensorShape{.width = 4, .height = 2, .channels = 2}),
                      0x434E4854U)
NCNN_CONCAT_AXIS_TEST(ConcatNegativeWidthAxisMatchesNcnn,
                      "concat_negative_width",
                      CONCAT_NEGATIVE_WIDTH_LIBRARY_PATH,
                      (TensorShape{.width = 4, .height = 3, .channels = 2}),
                      (TensorShape{.width = 2, .height = 3, .channels = 2}),
                      0x434E5744U)

#undef NCNN_CONCAT_AXIS_TEST

TEST(NumericalOperator, SplitThreeWayConsumerTopologyMatchesNcnn) {
  expect_single_input_operator("split_three",
                               SPLIT_THREE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               {.width = 5, .height = 4, .channels = 3},
                               180,
                               0.0F,
                               0x53504C33U);
}

}  // namespace
}  // namespace ncnn_compiler::test
