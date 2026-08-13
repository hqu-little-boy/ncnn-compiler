#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
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

TEST(SupportedOps, DetectionOutputCaffeSsd) {
  const std::vector<float> location{
    1.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> confidence{
    0.1F, 0.9F, 0.2F, 0.1F, 0.8F, 0.95F, 0.6F, 0.4F, 0.4F};
  const std::vector<float> prior{
    0.0F, 0.0F, 0.4F, 0.4F, 0.05F, 0.05F, 0.45F, 0.45F, 0.6F, 0.6F, 1.0F, 1.0F,
    0.1F, 0.1F, 0.2F, 0.2F, 0.1F,  0.1F,  0.2F,  0.2F,  0.1F, 0.1F, 0.2F, 0.2F};

  const std::array<ReferenceInput, 3> inputs{
    ReferenceInput("location", TensorShape(12, 1, 1), location),
    ReferenceInput("confidence", TensorShape(9, 1, 1), confidence),
    ReferenceInput("prior", TensorShape(12, 2, 1), prior)};
  constexpr std::array<std::string_view, 1> outputs{"output"};
  const auto reference = run_ncnn_reference(fixture_path("detection_output"),
                                            NUMERICAL_EMPTY_BIN_PATH,
                                            inputs,
                                            outputs);
  ASSERT_TRUE(reference.has_value()) << reference.error();
  ASSERT_EQ(reference->front().size(), 12);

  CompiledModel compiled(DETECTION_OUTPUT_LIBRARY_PATH, "detection_output");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> output(18, 0.0F);
  std::array<std::int64_t, 2> actualShape{};
  ASSERT_EQ(compiled.run_three_inputs_two_outputs(
              location, confidence, prior, output, actualShape),
            0);

  EXPECT_EQ(actualShape, (std::array<std::int64_t, 2>{2, 6}));
  const std::array<float, 12> expected{2.0F,
                                       0.95F,
                                       0.05F,
                                       0.05F,
                                       0.45F,
                                       0.45F,
                                       1.0F,
                                       0.9F,
                                       0.04F,
                                       -0.04F,
                                       0.44F,
                                       0.36F};
  EXPECT_TRUE(compare_values(
    std::span<const float>(output.data(), expected.size()), expected, 1.0e-6F));
  EXPECT_TRUE(compare_values(
    std::span<const float>(output.data(), 12), reference->front(), 1.0e-6F));
}

void expect_single_input_operator(std::string_view name,
                                  std::string_view library_path,
                                  std::string_view bin_path,
                                  TensorShape input_shape,
                                  std::size_t output_elements,
                                  float tolerance,
                                  std::uint32_t seed) {
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  ASSERT_TRUE(input_shape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*input_elements, seed, -2.0F, 2.0F);
  std::string_view reference_name = name;
  if (name == "convolution_mixed_fp16" ||
      name == "convolution_arithmetic_fp16" ||
      name == "convolution_mixed_bf16") {
    reference_name = "convolution";
  } else if (name == "convolution_depthwise_mixed_fp16" ||
             name == "convolution_depthwise_arithmetic_fp16" ||
             name == "convolution_depthwise_mixed_bf16") {
    reference_name = "convolution_depthwise";
  }
  const ReferenceModel reference(fixture_path(reference_name),
                                 std::string(bin_path),
                                 "data",
                                 "output",
                                 input_shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), output_elements);

  const std::string symbol = name == "convolution" || name == "relu" ||
                                 name == "pooling" || name == "dropout" ||
                                 name == "softmax"
                               ? std::format("{}_scalar", name)
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
  const auto first_elements = first_shape.element_count();
  ASSERT_TRUE(first_elements.has_value()) << first_elements.error();
  ASSERT_TRUE(first_shape.byte_count(sizeof(float)).has_value());
  const auto second_elements = second_shape.element_count();
  ASSERT_TRUE(second_elements.has_value()) << second_elements.error();
  ASSERT_TRUE(second_shape.byte_count(sizeof(float)).has_value());
  ASSERT_LE(*first_elements,
            std::numeric_limits<std::size_t>::max() - *second_elements);
  const std::vector<float> first =
    make_random_input(*first_elements, first_seed, -2.0F, 2.0F);
  const std::vector<float> second =
    make_random_input(*second_elements, second_seed, -2.0F, 2.0F);
  const std::array<ReferenceInput, 2> inputs{
    ReferenceInput("first", first_shape, first),
    ReferenceInput("second", second_shape, second)};
  constexpr std::array<std::string_view, 1> kOutputs{"output"};
  const auto expected = run_ncnn_reference(
    fixture_path(name), NUMERICAL_EMPTY_BIN_PATH, inputs, kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 1);

  const std::string symbol =
    name == "concat" ? "concat_scalar" : std::string(name);
  CompiledModel compiled(library_path, symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(*first_elements + *second_elements);
  ASSERT_EQ(compiled.run_two_inputs(first, second, actual), 0);
  EXPECT_TRUE(compare_values(actual, expected->front(), 0.0F));
}

void expect_shuffle_channel(std::string_view fixture,
                            std::string_view libraryPath,
                            std::span<const std::size_t> sourceChannels) {
  const TensorShape shape(4, 3, 6);
  constexpr std::size_t kChannelElements = 12;
  std::vector<float> input(6 * kChannelElements);
  for (std::size_t channel = 0; channel < 6; ++channel) {
    std::fill_n(
      input.begin() + static_cast<std::ptrdiff_t>(channel * kChannelElements),
      kChannelElements,
      static_cast<float>(channel));
  }
  const ReferenceModel reference(
    fixture_path(fixture), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(libraryPath, fixture);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
  ASSERT_EQ(sourceChannels.size(), 6U);
  for (std::size_t outputChannel = 0; outputChannel < 6; ++outputChannel) {
    const auto begin = actual.begin() + static_cast<std::ptrdiff_t>(
                                          outputChannel * kChannelElements);
    EXPECT_TRUE(std::all_of(begin, begin + kChannelElements, [&](float value) {
      return value == static_cast<float>(sourceChannels[outputChannel]);
    }));
  }
}

TEST(NumericalOperator, ConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution",
                               CONVOLUTION_LIBRARY_PATH,
                               CONVOLUTION_BIN_PATH,
                               TensorShape(4, 4, 1),
                               32,
                               1.0e-5F,
                               0x434F4E56U);
}

TEST(NumericalOperator, PaddingAsymmetricNegativeMaxMatchesNcnn) {
  expect_single_input_operator("padding",
                               PADDING_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(3, 2, 2),
                               24,
                               0.0F,
                               0x50414444U);
}

TEST(NumericalOperator, PaddingAllFourSidesMatchNcnn) {
  expect_single_input_operator("padding_four_sides",
                               PADDING_FOUR_SIDES_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(3, 2, 2),
                               100,
                               0.0F,
                               0x50414434U);
}

TEST(NumericalOperator, PaddingIdentityMatchesNcnnAndInput) {
  const TensorShape shape(4, 3, 2);
  const std::vector<float> input =
    make_random_input(24, 0x50494445U, -2.0F, 2.0F);
  const ReferenceModel reference(fixture_path("padding_identity"),
                                 NUMERICAL_EMPTY_BIN_PATH,
                                 "data",
                                 "output",
                                 shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(PADDING_IDENTITY_LIBRARY_PATH, "padding_identity");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
  EXPECT_EQ(actual, input);
}

TEST(NumericalOperator, InterpNearestTwofoldMatchesNcnn) {
  expect_single_input_operator("interp",
                               INTERP_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(3, 2, 2),
                               48,
                               0.0F,
                               0x494E5450U);
}

TEST(NumericalOperator, InterpNearestAsymmetricOddEvenScalesMatchNcnn) {
  expect_single_input_operator("interp_asymmetric",
                               INTERP_ASYMMETRIC_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(3, 2, 2),
                               144,
                               0.0F,
                               0x494E5434U);
}

TEST(NumericalOperator, InterpNearestExplicitTargetMatchesNcnn) {
  expect_single_input_operator("interp_explicit_target",
                               INTERP_EXPLICIT_TARGET_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(16, 16, 2),
                               31 * 31 * 2,
                               0.0F,
                               0x494E5431U);
}

TEST(NumericalOperator, InterpIdentityMatchesNcnnAndInput) {
  const TensorShape shape(4, 3, 2);
  const std::vector<float> input =
    make_random_input(24, 0x49494445U, -2.0F, 2.0F);
  const ReferenceModel reference(fixture_path("interp_identity"),
                                 NUMERICAL_EMPTY_BIN_PATH,
                                 "data",
                                 "output",
                                 shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(INTERP_IDENTITY_LIBRARY_PATH, "interp_identity");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
  EXPECT_EQ(actual, input);
}

TEST(NumericalOperator, InterpNearestEightfoldMatchesNcnn) {
  expect_single_input_operator("interp_eightfold",
                               INTERP_EIGHTFOLD_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(2, 3, 2),
                               768,
                               0.0F,
                               0x494E5438U);
}

TEST(NumericalOperator, DeconvolutionLayoutAndFusedReluMatchNcnn) {
  const TensorShape shape(3, 2, 2);
  const auto elements = shape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  std::vector<float> input(*elements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
      static_cast<float>(index + 1) * (index % 2 == 0 ? 0.25F : -0.20F);
  }
  const ReferenceModel reference(fixture_path("deconvolution"),
                                 DECONVOLUTION_BIN_PATH,
                                 "data",
                                 "output",
                                 shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 48U);

  CompiledModel compiled(DECONVOLUTION_LIBRARY_PATH, "deconvolution");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(expected->size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  EXPECT_TRUE(
    std::ranges::all_of(actual, [](float value) { return value >= 0.0F; }));
  EXPECT_TRUE(
    std::ranges::any_of(actual, [](float value) { return value == 0.0F; }));
  EXPECT_TRUE(
    std::ranges::any_of(actual, [](float value) { return value > 0.0F; }));
}

TEST(NumericalOperator, DeconvolutionWithoutBiasMatchesNcnn) {
  expect_single_input_operator("deconvolution_no_bias",
                               DECONVOLUTION_NO_BIAS_LIBRARY_PATH,
                               DECONVOLUTION_NO_BIAS_BIN_PATH,
                               TensorShape(3, 2, 2),
                               72,
                               1.0e-6F,
                               0x4445434EU);
}

TEST(NumericalOperator, DeconvolutionTinyHeadMatchesNcnn) {
  expect_single_input_operator("deconvolution_tiny_head",
                               DECONVOLUTION_TINY_HEAD_LIBRARY_PATH,
                               DECONVOLUTION_TINY_HEAD_BIN_PATH,
                               TensorShape(2, 2, 16),
                               16,
                               1.0e-6F,
                               0x44544844U);
}

TEST(NumericalOperator, SigmoidMatchesNcnnAndProducesProbabilities) {
  const TensorShape shape(5, 2, 2);
  constexpr std::array<float, 20> input{
    -20.0F, -10.0F, -6.0F, -3.0F, -1.0F, -0.5F, -0.1F, 0.0F,  0.1F,  0.5F,
    1.0F,   2.0F,   3.0F,  4.0F,  5.0F,  6.0F,  8.0F,  10.0F, 15.0F, 20.0F};
  const ReferenceModel reference(
    fixture_path("sigmoid"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(SIGMOID_LIBRARY_PATH, "sigmoid");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));
  EXPECT_NEAR(actual[7], 0.5F, 1.0e-7F);
  EXPECT_LT(actual.front(), 1.0e-8F);
  EXPECT_GT(actual.back(), 1.0F - 1.0e-7F);
}

TEST(NumericalOperator, SigmoidExtremeValuesMatchNcnnClamping) {
  const TensorShape shape(5, 2, 2);
  constexpr std::array<float, 20> input{-std::numeric_limits<float>::infinity(),
                                        -std::numeric_limits<float>::max(),
                                        -100.0F,
                                        -88.3762626647949F,
                                        -20.0F,
                                        -1.0F,
                                        -0.0F,
                                        0.0F,
                                        1.0F,
                                        20.0F,
                                        88.3762626647949F,
                                        100.0F,
                                        std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::infinity(),
                                        -10.0F,
                                        -5.0F,
                                        5.0F,
                                        10.0F,
                                        -0.5F,
                                        0.5F};
  const ReferenceModel reference(
    fixture_path("sigmoid"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(SIGMOID_LIBRARY_PATH, "sigmoid");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  EXPECT_GT(actual[0], 0.0F);
  EXPECT_EQ(actual[0], actual[1]);
  EXPECT_EQ(actual[1], actual[2]);
  EXPECT_EQ(actual[11], 1.0F);
  EXPECT_EQ(actual[11], actual[12]);
  EXPECT_EQ(actual[12], actual[13]);
}

TEST(NumericalOperator, SigmoidNaNBehaviorIsStable) {
  const TensorShape shape(5, 2, 2);
  constexpr std::size_t kNanIndex = 3;
  const std::array<float, 20> input{
    -2.0F, -0.0F, 0.0F,   std::numeric_limits<float>::quiet_NaN(),
    2.0F,  -1.0F, 1.0F,   0.5F,
    -0.5F, 3.0F,  -3.0F,  4.0F,
    -4.0F, 0.25F, -0.25F, 5.0F,
    -5.0F, 6.0F,  -6.0F,  1.5F};
  const ReferenceModel reference(
    fixture_path("sigmoid"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(SIGMOID_LIBRARY_PATH, "sigmoid");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(std::isfinite((*expected)[kNanIndex]));
  EXPECT_TRUE(std::isnan(actual[kNanIndex]));
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (index != kNanIndex) {
      EXPECT_NEAR(actual[index], (*expected)[index], 1.0e-6F) << index;
    }
  }
}

TEST(NumericalOperator, ConvolutionWithoutBiasMatchesNcnn) {
  expect_single_input_operator("convolution_no_bias",
                               CONVOLUTION_NO_BIAS_LIBRARY_PATH,
                               CONVOLUTION_NO_BIAS_BIN_PATH,
                               TensorShape(4, 4, 1),
                               32,
                               1.0e-5F,
                               0x434E4249U);
}

TEST(NumericalOperator, DilatedConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution_dilated",
                               CONVOLUTION_DILATED_LIBRARY_PATH,
                               CONVOLUTION_DILATED_BIN_PATH,
                               TensorShape(7, 7, 1),
                               128,
                               1.0e-5F,
                               0x4344494CU);
}

TEST(NumericalOperator, AsymmetricConvolutionPaddingMatchesNcnn) {
  expect_single_input_operator("convolution_asymmetric",
                               CONVOLUTION_ASYMMETRIC_LIBRARY_PATH,
                               CONVOLUTION_ASYMMETRIC_BIN_PATH,
                               TensorShape(4, 4, 1),
                               48,
                               1.0e-5F,
                               0x43415359U);
}

TEST(NumericalOperator, AsymmetricConvolutionStrideMatchesNcnn) {
  expect_single_input_operator("convolution_asymmetric_stride",
                               CONVOLUTION_ASYMMETRIC_STRIDE_LIBRARY_PATH,
                               CONVOLUTION_ASYMMETRIC_STRIDE_BIN_PATH,
                               TensorShape(8, 9, 1),
                               32,
                               1.0e-5F,
                               0x43415354U);
}

TEST(NumericalOperator, ConvolutionSameUpperMatchesNcnn) {
  expect_single_input_operator("convolution_same_upper",
                               CONVOLUTION_SAME_UPPER_LIBRARY_PATH,
                               CONVOLUTION_SAME_UPPER_BIN_PATH,
                               TensorShape(5, 5, 1),
                               18,
                               1.0e-5F,
                               0x43535550U);
}

TEST(NumericalOperator, ConvolutionSameLowerMatchesNcnn) {
  expect_single_input_operator("convolution_same_lower",
                               CONVOLUTION_SAME_LOWER_LIBRARY_PATH,
                               CONVOLUTION_SAME_LOWER_BIN_PATH,
                               TensorShape(5, 5, 1),
                               18,
                               1.0e-5F,
                               0x43534C4FU);
}

TEST(NumericalOperator, ConvolutionFp16StorageMatchesNcnn) {
  expect_single_input_operator("convolution_fp16_storage",
                               CONVOLUTION_FP16_STORAGE_LIBRARY_PATH,
                               CONVOLUTION_FP16_STORAGE_BIN_PATH,
                               TensorShape(4, 4, 1),
                               32,
                               1.0e-5F,
                               0x46313643U);
}

TEST(NumericalOperator, ConvolutionMixedFp16MatchesNcnn) {
  expect_single_input_operator("convolution_mixed_fp16",
                               CONVOLUTION_MIXED_FP16_LIBRARY_PATH,
                               CONVOLUTION_BIN_PATH,
                               TensorShape(4, 4, 1),
                               32,
                               2.0e-3F,
                               0x4D463136U);
}

TEST(NumericalOperator, ConvolutionFp16AccumulatorMatchesNcnn) {
  expect_single_input_operator("convolution_arithmetic_fp16",
                               CONVOLUTION_ARITHMETIC_FP16_LIBRARY_PATH,
                               CONVOLUTION_BIN_PATH,
                               TensorShape(4, 4, 1),
                               32,
                               8.0e-3F,
                               0x41463136U);
}

TEST(NumericalOperator, ConvolutionMixedBf16MatchesNcnn) {
  expect_single_input_operator("convolution_mixed_bf16",
                               CONVOLUTION_MIXED_BF16_LIBRARY_PATH,
                               CONVOLUTION_BIN_PATH,
                               TensorShape(4, 4, 1),
                               32,
                               2.0e-2F,
                               0x4D424631U);
}

TEST(NumericalOperator, DepthwiseConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution_depthwise",
                               CONVOLUTION_DEPTHWISE_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_BIN_PATH,
                               TensorShape(5, 5, 2),
                               50,
                               1.0e-5F,
                               0x44574356U);
}

TEST(NumericalOperator, DepthwiseConvolutionFp16StorageMatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_fp16_storage",
                               CONVOLUTION_DEPTHWISE_FP16_STORAGE_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_FP16_STORAGE_BIN_PATH,
                               TensorShape(5, 5, 2),
                               50,
                               1.0e-5F,
                               0x46313644U);
}

TEST(NumericalOperator, DepthwiseConvolutionMixedFp16MatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_mixed_fp16",
                               CONVOLUTION_DEPTHWISE_MIXED_FP16_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_BIN_PATH,
                               TensorShape(5, 5, 2),
                               50,
                               2.0e-3F,
                               0x444D4631U);
}

TEST(NumericalOperator, DepthwiseConvolutionFp16AccumulatorMatchesNcnn) {
  expect_single_input_operator(
    "convolution_depthwise_arithmetic_fp16",
    CONVOLUTION_DEPTHWISE_ARITHMETIC_FP16_LIBRARY_PATH,
    CONVOLUTION_DEPTHWISE_BIN_PATH,
    TensorShape(5, 5, 2),
    50,
    8.0e-3F,
    0x44414631U);
}

TEST(NumericalOperator, DepthwiseConvolutionMixedBf16MatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_mixed_bf16",
                               CONVOLUTION_DEPTHWISE_MIXED_BF16_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_BIN_PATH,
                               TensorShape(5, 5, 2),
                               50,
                               2.0e-2F,
                               0x444D4246U);
}

TEST(NumericalOperator, DepthwiseConvolution7x7MatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_7x7",
                               CONVOLUTION_DEPTHWISE_7X7_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_7X7_BIN_PATH,
                               TensorShape(5, 4, 96),
                               5 * 4 * 96,
                               1.0e-5F,
                               0x44573758U);
}

TEST(NumericalOperator, DepthwiseConvolution9x9MatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_9x9",
                               CONVOLUTION_DEPTHWISE_9X9_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_9X9_BIN_PATH,
                               TensorShape(5, 4, 256),
                               5 * 4 * 256,
                               1.0e-5F,
                               0x44573958U);
}

TEST(NumericalOperator, AsymmetricDepthwiseConvolutionMatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_asymmetric",
                               CONVOLUTION_DEPTHWISE_ASYMMETRIC_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_ASYMMETRIC_BIN_PATH,
                               TensorShape(9, 11, 2),
                               32,
                               1.0e-5F,
                               0x44574153U);
}

TEST(NumericalOperator, DepthwiseConvolutionSameUpperMatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_same_upper",
                               CONVOLUTION_DEPTHWISE_SAME_UPPER_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_SAME_UPPER_BIN_PATH,
                               TensorShape(5, 4, 2),
                               12,
                               1.0e-5F,
                               0x44535550U);
}

TEST(NumericalOperator, DepthwiseConvolutionSameLowerMatchesNcnn) {
  expect_single_input_operator("convolution_depthwise_same_lower",
                               CONVOLUTION_DEPTHWISE_SAME_LOWER_LIBRARY_PATH,
                               CONVOLUTION_DEPTHWISE_SAME_LOWER_BIN_PATH,
                               TensorShape(5, 4, 2),
                               12,
                               1.0e-5F,
                               0x44534C4FU);
}

TEST(NumericalOperator, HardSigmoidMatchesNcnn) {
  expect_single_input_operator("hard_sigmoid",
                               HARD_SIGMOID_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               1.0e-6F,
                               0x48534947U);
}

TEST(NumericalOperator, HardSwishMatchesNcnn) {
  expect_single_input_operator("hard_swish",
                               HARD_SWISH_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               1.0e-6F,
                               0x48535749U);
}

TEST(NumericalOperator, GELUMatchesNcnn) {
  expect_single_input_operator("gelu",
                               GELU_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 2, 2),
                               20,
                               1.0e-6F,
                               0x47454C55U);
}

TEST(NumericalOperator, GELUNegativeTailMatchesNcnn) {
  const TensorShape shape(5, 2, 2);
  const std::array<float, 20> input{
    -10.0F, -8.0F, -6.0F, -5.0F, -4.0F, -3.0F, -2.0F, -1.0F, 0.0F,  1.0F,
    2.0F,   3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,  9.0F,  10.0F, 0.5F};
  const ReferenceModel reference(
    fixture_path("gelu"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(GELU_LIBRARY_PATH, "gelu");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(input.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
}

TEST(NumericalOperator, ReshapeMatchesNcnn) {
  expect_single_input_operator("reshape",
                               RESHAPE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(2, 2, 2),
                               8,
                               0.0F,
                               0x52455348U);
}

TEST(NumericalOperator, ReshapeCopyDimensionMatchesNcnn) {
  expect_single_input_operator("reshape_copy_dimension",
                               RESHAPE_COPY_DIMENSION_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(4, 3, 2),
                               24,
                               0.0F,
                               0x52435059U);
}

TEST(NumericalOperator, SqueezeMatchesNcnn) {
  expect_single_input_operator("squeeze",
                               SQUEEZE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(4, 1, 3),
                               12,
                               0.0F,
                               0x5351555AU);
}

TEST(NumericalOperator, BatchNormZeroVarianceMatchesNcnn) {
  const TensorShape shape(4, 1, 3);
  const auto elements = shape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  const std::vector<float> input =
    make_random_input(*elements, 0x424E4F52U, -2.0F, 2.0F);
  const ReferenceModel reference(
    fixture_path("batch_norm"), BATCH_NORM_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(BATCH_NORM_LIBRARY_PATH, "batch_norm");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(*elements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  EXPECT_TRUE(std::ranges::all_of(
    actual, [](float value) { return std::isfinite(value); }));
}

TEST(NumericalOperator, ExpandDimsNegativeAxisMatchesNcnn) {
  expect_single_input_operator("expand_dims",
                               EXPAND_DIMS_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(4, 1, 3),
                               12,
                               0.0F,
                               0x45585044U);
}

TEST(NumericalOperator, PermuteRankTwoMatchesNcnn) {
  expect_single_input_operator("permute",
                               PERMUTE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(4, 1, 3),
                               12,
                               0.0F,
                               0x5045524DU);
}

TEST(NumericalOperator, GemmScaledBiasMatchesNcnn) {
  expect_single_input_operator("gemm",
                               GEMM_LIBRARY_PATH,
                               GEMM_BIN_PATH,
                               TensorShape(3, 1, 2),
                               12,
                               1.0e-6F,
                               0x47454D4DU);
}

TEST(NumericalOperator, BinaryMultiplyScalarMatchesNcnn) {
  expect_single_input_operator("binary_mul_scalar",
                               BINARY_MUL_SCALAR_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               0.0F,
                               0x424D5343U);
}

TEST(NumericalOperator, BinaryMultiplyChannelBroadcastMatchesNcnn) {
  const TensorShape first_shape(4, 3, 2);
  const TensorShape second_shape(1, 1, 2);
  const auto first_elements = first_shape.element_count();
  const auto second_elements = second_shape.element_count();
  ASSERT_TRUE(first_elements.has_value()) << first_elements.error();
  ASSERT_TRUE(second_elements.has_value()) << second_elements.error();
  const std::vector<float> first =
    make_random_input(*first_elements, 0x424D4331U, -2.0F, 2.0F);
  const std::vector<float> second =
    make_random_input(*second_elements, 0x424D4332U, -2.0F, 2.0F);
  const std::array<ReferenceInput, 2> inputs{
    ReferenceInput("first", first_shape, first),
    ReferenceInput("second", second_shape, second)};
  constexpr std::array<std::string_view, 1> outputs{"output"};
  const auto expected = run_ncnn_reference(fixture_path("binary_mul_channel"),
                                           NUMERICAL_EMPTY_BIN_PATH,
                                           inputs,
                                           outputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(BINARY_MUL_CHANNEL_LIBRARY_PATH, "binary_mul_channel");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(*first_elements);
  ASSERT_EQ(compiled.run_two_inputs(first, second, actual), 0);
  EXPECT_TRUE(compare_values(actual, expected->front(), 0.0F));
}

TEST(NumericalOperator, BinaryMultiplyReverseBroadcastMatchesNcnn) {
  const TensorShape first_shape(1, 1, 2);
  const TensorShape second_shape(4, 3, 2);
  const auto first_elements = first_shape.element_count();
  const auto second_elements = second_shape.element_count();
  ASSERT_TRUE(first_elements.has_value()) << first_elements.error();
  ASSERT_TRUE(second_elements.has_value()) << second_elements.error();
  const std::vector<float> first =
    make_random_input(*first_elements, 0x42524231U, -2.0F, 2.0F);
  const std::vector<float> second =
    make_random_input(*second_elements, 0x42524232U, -2.0F, 2.0F);
  const std::array<ReferenceInput, 2> inputs{
    ReferenceInput("first", first_shape, first),
    ReferenceInput("second", second_shape, second)};
  constexpr std::array<std::string_view, 1> outputs{"output"};
  const auto expected =
    run_ncnn_reference(fixture_path("binary_mul_reverse_broadcast"),
                       NUMERICAL_EMPTY_BIN_PATH,
                       inputs,
                       outputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(BINARY_MUL_REVERSE_BROADCAST_LIBRARY_PATH,
                         "binary_mul_reverse_broadcast");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(*second_elements);
  ASSERT_EQ(compiled.run_two_inputs(first, second, actual), 0);
  EXPECT_TRUE(compare_values(actual, expected->front(), 0.0F));
}

TEST(NumericalOperator, BinaryAddMatchesNcnn) {
  const TensorShape shape(4, 3, 2);
  const auto elements = shape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  const std::vector<float> first =
    make_random_input(*elements, 0x42414431U, -2.0F, 2.0F);
  const std::vector<float> second =
    make_random_input(*elements, 0x42414432U, -2.0F, 2.0F);
  const std::array<ReferenceInput, 2> inputs{
    ReferenceInput("first", shape, first),
    ReferenceInput("second", shape, second)};
  constexpr std::array<std::string_view, 1> outputs{"output"};
  const auto expected = run_ncnn_reference(
    fixture_path("binary_add"), NUMERICAL_EMPTY_BIN_PATH, inputs, outputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(BINARY_ADD_LIBRARY_PATH, "binary_add");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(*elements);
  ASSERT_EQ(compiled.run_two_inputs(first, second, actual), 0);
  EXPECT_TRUE(compare_values(actual, expected->front(), 0.0F));
}

TEST(NumericalOperator, InnerProductMatchesNcnn) {
  expect_single_input_operator("inner_product",
                               INNER_PRODUCT_LIBRARY_PATH,
                               INNER_PRODUCT_BIN_PATH,
                               TensorShape(8, 1, 1),
                               3,
                               1.0e-6F,
                               0x49505054U);
}

TEST(NumericalOperator, InnerProductRankThreeFusedReluMatchesNcnn) {
  const TensorShape shape(3, 1, 2);
  const auto elements = shape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  std::vector<float> input(*elements, -1.0F);
  const ReferenceModel reference(fixture_path("inner_product_fused_relu"),
                                 INNER_PRODUCT_FUSED_RELU_BIN_PATH,
                                 "data",
                                 "output",
                                 shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(INNER_PRODUCT_FUSED_RELU_LIBRARY_PATH,
                         "inner_product_fused_relu");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(4);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  EXPECT_TRUE(
    std::ranges::all_of(actual, [](float value) { return value >= 0.0F; }));
}

TEST(NumericalOperator, ReluMatchesNcnn) {
  expect_single_input_operator("relu",
                               RELU_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               0.0F,
                               0x52454C55U);
}

TEST(NumericalOperator, ReluHandlesZeroAndNegativeValues) {
  const TensorShape kShape(5, 4, 3);
  const auto elements = kShape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  ASSERT_TRUE(kShape.byte_count(sizeof(float)).has_value());
  std::vector<float> input(*elements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    constexpr std::array<float, 5> kValues{-2.0F, -0.0F, 0.0F, 0.5F, 2.0F};
    input[index] = kValues[index % kValues.size()];
  }
  const ReferenceModel reference(
    fixture_path("relu"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", kShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(RELU_LIBRARY_PATH, "relu_scalar");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(*elements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
}

TEST(NumericalOperator, LeakyReluMatchesNcnn) {
  expect_single_input_operator("relu_leaky",
                               RELU_LEAKY_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               0.0F,
                               0x4C52454CU);
}

TEST(NumericalOperator, PoolingMatchesNcnn) {
  expect_single_input_operator("pooling",
                               POOLING_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 5, 2),
                               8,
                               0.0F,
                               0x504F4F4CU);
}

TEST(NumericalOperator, AveragePoolingExcludingPadMatchesNcnn) {
  expect_single_input_operator("pooling_average",
                               POOLING_AVERAGE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 5, 2),
                               18,
                               1.0e-6F,
                               0x50415647U);
}

TEST(NumericalOperator, AsymmetricPoolingParametersMatchNcnn) {
  expect_single_input_operator("pooling_asymmetric",
                               POOLING_ASYMMETRIC_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(7, 8, 2),
                               24,
                               0.0F,
                               0x50415359U);
}

TEST(NumericalOperator, GlobalMaxPoolingMatchesNcnn) {
  expect_single_input_operator("pooling_global_max",
                               POOLING_GLOBAL_MAX_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 5, 2),
                               2,
                               0.0F,
                               0x50474D58U);
}

TEST(NumericalOperator, GlobalAveragePoolingMatchesNcnn) {
  expect_single_input_operator("pooling_global_average",
                               POOLING_GLOBAL_AVERAGE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 5, 2),
                               2,
                               1.0e-6F,
                               0x50474156U);
}

TEST(NumericalOperator, GlobalMaxPoolingIgnoresRegularParameters) {
  expect_single_input_operator("pooling_global_max_ignored_params",
                               POOLING_GLOBAL_MAX_IGNORED_PARAMS_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 2),
                               2,
                               0.0F,
                               0x5047494DU);
}

TEST(NumericalOperator, GlobalAveragePoolingIgnoresRegularParameters) {
  expect_single_input_operator(
    "pooling_global_average_ignored_params",
    POOLING_GLOBAL_AVERAGE_IGNORED_PARAMS_LIBRARY_PATH,
    NUMERICAL_EMPTY_BIN_PATH,
    TensorShape(5, 4, 2),
    2,
    1.0e-6F,
    0x50474941U);
}

TEST(NumericalOperator, PoolingSameUpperMatchesNcnn) {
  expect_single_input_operator("pooling_same_upper",
                               POOLING_SAME_UPPER_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 2),
                               12,
                               0.0F,
                               0x50535550U);
}

TEST(NumericalOperator, PoolingSameLowerMatchesNcnn) {
  expect_single_input_operator("pooling_same_lower",
                               POOLING_SAME_LOWER_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 2),
                               12,
                               0.0F,
                               0x50534C4FU);
}

TEST(NumericalOperator, PoolingTailWindowMatchesNcnn) {
  expect_single_input_operator("pooling_tail",
                               POOLING_TAIL_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(4, 4, 2),
                               8,
                               0.0F,
                               0x50544149U);
}

TEST(NumericalOperator, DropoutMatchesNcnn) {
  expect_single_input_operator("dropout",
                               DROPOUT_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               0.0F,
                               0x44524F50U);
}

TEST(NumericalOperator, SoftmaxMatchesNcnn) {
  expect_single_input_operator("softmax",
                               SOFTMAX_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               60,
                               1.0e-6F,
                               0x534F4654U);
}

#define NCNN_SOFTMAX_AXIS_TEST(test_name, fixture_name, library_path, seed) \
  TEST(NumericalOperator, test_name) {                                      \
    expect_single_input_operator(fixture_name,                              \
                                 library_path,                              \
                                 NUMERICAL_EMPTY_BIN_PATH,                  \
                                 TensorShape(5, 4, 3),                      \
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
  const TensorShape kShape(5, 4, 3);
  const auto elements = kShape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  ASSERT_TRUE(kShape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*elements, 0x53504C49U, -2.0F, 2.0F);
  const ReferenceInput reference_input("data", kShape, input);
  constexpr std::array<std::string_view, 2> kOutputs{"left", "right"};
  const auto expected = run_ncnn_reference(fixture_path("split"),
                                           NUMERICAL_EMPTY_BIN_PATH,
                                           std::span(&reference_input, 1),
                                           kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 2);

  CompiledModel compiled(SPLIT_LIBRARY_PATH, "split_scalar");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> left(*elements);
  std::vector<float> right(*elements);
  ASSERT_EQ(compiled.run_two_outputs(input, left, right), 0);
  EXPECT_TRUE(compare_values(left, (*expected)[0], 0.0F));
  EXPECT_TRUE(compare_values(right, (*expected)[1], 0.0F));
}

TEST(NumericalOperator, ConcatMatchesNcnn) {
  expect_concat("concat",
                CONCAT_LIBRARY_PATH,
                TensorShape(4, 3, 1),
                TensorShape(4, 3, 2),
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
                      (TensorShape(4, 3, 2)),
                      (TensorShape(4, 2, 2)),
                      0x43414854U)
NCNN_CONCAT_AXIS_TEST(ConcatWidthAxisMatchesNcnn,
                      "concat_width",
                      CONCAT_WIDTH_LIBRARY_PATH,
                      (TensorShape(4, 3, 2)),
                      (TensorShape(2, 3, 2)),
                      0x43415744U)
NCNN_CONCAT_AXIS_TEST(ConcatNegativeChannelAxisMatchesNcnn,
                      "concat_negative_channel",
                      CONCAT_NEGATIVE_CHANNEL_LIBRARY_PATH,
                      (TensorShape(4, 3, 1)),
                      (TensorShape(4, 3, 2)),
                      0x434E4348U)
NCNN_CONCAT_AXIS_TEST(ConcatNegativeHeightAxisMatchesNcnn,
                      "concat_negative_height",
                      CONCAT_NEGATIVE_HEIGHT_LIBRARY_PATH,
                      (TensorShape(4, 3, 2)),
                      (TensorShape(4, 2, 2)),
                      0x434E4854U)
NCNN_CONCAT_AXIS_TEST(ConcatNegativeWidthAxisMatchesNcnn,
                      "concat_negative_width",
                      CONCAT_NEGATIVE_WIDTH_LIBRARY_PATH,
                      (TensorShape(4, 3, 2)),
                      (TensorShape(2, 3, 2)),
                      0x434E5744U)

#undef NCNN_CONCAT_AXIS_TEST

TEST(NumericalOperator, SplitThreeWayConsumerTopologyMatchesNcnn) {
  expect_single_input_operator("split_three",
                               SPLIT_THREE_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(5, 4, 3),
                               180,
                               0.0F,
                               0x53504C33U);
}

TEST(NumericalOperator, ShuffleChannelMatchesNcnn) {
  constexpr std::array<std::size_t, 6> kSourceChannels{0, 3, 1, 4, 2, 5};
  expect_shuffle_channel(
    "shuffle_channel", SHUFFLE_CHANNEL_LIBRARY_PATH, kSourceChannels);
}

TEST(NumericalOperator, ShuffleChannelReverseMatchesNcnn) {
  constexpr std::array<std::size_t, 6> kSourceChannels{0, 2, 4, 1, 3, 5};
  expect_shuffle_channel("shuffle_channel_reverse",
                         SHUFFLE_CHANNEL_REVERSE_LIBRARY_PATH,
                         kSourceChannels);
}

TEST(NumericalOperator, SliceMatchesNcnn) {
  const TensorShape shape(4, 3, 5);
  const auto elements = shape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  const std::vector<float> input =
    make_random_input(*elements, 0x534C4943U, -2.0F, 2.0F);
  const ReferenceInput referenceInput("data", shape, input);
  constexpr std::array<std::string_view, 2> kOutputs{"left", "right"};
  const auto expected = run_ncnn_reference(fixture_path("slice"),
                                           NUMERICAL_EMPTY_BIN_PATH,
                                           std::span(&referenceInput, 1),
                                           kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 2);
  CompiledModel compiled(SLICE_LIBRARY_PATH, "slice");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> left(24);
  std::vector<float> right(36);
  ASSERT_EQ(compiled.run_two_outputs(input, left, right), 0);
  EXPECT_TRUE(compare_values(left, (*expected)[0], 0.0F));
  EXPECT_TRUE(compare_values(right, (*expected)[1], 0.0F));
  EXPECT_TRUE(std::equal(left.begin(), left.end(), input.begin()));
  EXPECT_TRUE(std::equal(right.begin(), right.end(), input.begin() + 24));
}

TEST(NumericalOperator, ReductionMeanMatchesNcnn) {
  expect_single_input_operator("reduction_mean",
                               REDUCTION_MEAN_LIBRARY_PATH,
                               NUMERICAL_EMPTY_BIN_PATH,
                               TensorShape(4, 3, 4),
                               4,
                               1.0e-6F,
                               0x52454455U);
}

TEST(NumericalOperator, ReductionMeanKeepdimsCoeffMatchesNcnn) {
  const TensorShape shape(4, 3, 4);
  const auto elements = shape.element_count();
  ASSERT_TRUE(elements.has_value()) << elements.error();
  std::vector<float> input(*elements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(index + 1);
  }
  const ReferenceModel reference(fixture_path("reduction_mean_keepdims_coeff"),
                                 NUMERICAL_EMPTY_BIN_PATH,
                                 "data",
                                 "output",
                                 shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 4U);

  CompiledModel compiled(REDUCTION_MEAN_KEEPDIMS_COEFF_LIBRARY_PATH,
                         "reduction_mean_keepdims_coeff");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(4);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  for (std::size_t channel = 0; channel < actual.size(); ++channel) {
    const auto first = static_cast<float>((channel * 12) + 1);
    const float last = first + 11.0F;
    EXPECT_NEAR(actual[channel], 0.25F * (first + last), 1.0e-6F);
  }
}

TEST(NumericalSupport, TensorShapeRejectsNegativeAndOverflowingSizes) {
  const TensorShape negative(-1, 2, 3);
  const auto negative_elements = negative.element_count();
  const auto negative_bytes = negative.byte_count(sizeof(float));
  ASSERT_FALSE(negative_elements.has_value());
  ASSERT_FALSE(negative_bytes.has_value());
  EXPECT_NE(negative_elements.error().find("non-negative"), std::string::npos);

  const TensorShape large(std::numeric_limits<int>::max(),
                          std::numeric_limits<int>::max(),
                          std::numeric_limits<int>::max());
  const auto large_elements = large.element_count();
  const auto large_bytes = large.byte_count(sizeof(float));
  EXPECT_TRUE(!large_elements.has_value() || !large_bytes.has_value());
}

TEST(NumericalSupport, ReferenceInputErrorIncludesBlobName) {
  const ReferenceInput input("bad_input", TensorShape(-1, 2, 3), {});
  constexpr std::array<std::string_view, 1> kOutputs{"output"};
  const auto result = run_ncnn_reference(fixture_path("relu"),
                                         NUMERICAL_EMPTY_BIN_PATH,
                                         std::span(&input, 1),
                                         kOutputs);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("bad_input"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_compiler::test
