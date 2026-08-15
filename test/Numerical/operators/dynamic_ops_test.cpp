#include "numerical_test_support.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
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

TEST(NumericalDynamicOperator, AxisTransformsMatchNcnnAcrossShapes) {
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{1, shape.height, shape.width};
    const std::size_t elements = shape.height * shape.width;
    const std::vector<float> input =
      make_random_input(elements, 0xA815U, -2.0F, 2.0F);
    const ReferenceModel squeezeReference(
      fixture_path("squeeze_dynamic"),
      NUMERICAL_EMPTY_BIN_PATH,
      "data",
      "output",
      TensorShape(shape.width, shape.height, 1));
    const auto squeezed = run_ncnn_reference(squeezeReference, input);
    ASSERT_TRUE(squeezed.has_value()) << squeezed.error();

    for (const auto& [library, symbol, expectedShape] :
         {std::tuple{SQUEEZE_DYNAMIC_LIBRARY_PATH,
                     "squeeze_dynamic",
                     std::array<std::int64_t, 2>{shape.height, shape.width}},
          std::tuple{PERMUTE_DYNAMIC_LIBRARY_PATH,
                     "permute_dynamic",
                     std::array<std::int64_t, 2>{shape.width, shape.height}}}) {
      CompiledModel compiled(library, symbol);
      ASSERT_TRUE(compiled.valid()) << compiled.error();
      CompiledModel infer(library,
                          std::string(symbol) + "_infer_output_shapes");
      ASSERT_TRUE(infer.valid()) << infer.error();
      std::array<std::int64_t, 2> inferred{};
      ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
      EXPECT_EQ(inferred, expectedShape);
      std::vector<float> actual(elements);
      ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
                kSuccess);
      if (std::string_view(symbol) == "squeeze_dynamic") {
        EXPECT_TRUE(compare_values(actual, *squeezed, 0.0F));
      } else {
        const ReferenceModel reference(
          fixture_path("permute_dynamic"),
          NUMERICAL_EMPTY_BIN_PATH,
          "data",
          "output",
          TensorShape(shape.width, shape.height, 1));
        const auto expected = run_ncnn_reference(reference, input);
        ASSERT_TRUE(expected.has_value()) << expected.error();
        EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
      }
    }

    CompiledModel expanded(EXPAND_DIMS_DYNAMIC_LIBRARY_PATH,
                           "expand_dims_dynamic");
    ASSERT_TRUE(expanded.valid()) << expanded.error();
    CompiledModel infer(EXPAND_DIMS_DYNAMIC_LIBRARY_PATH,
                        "expand_dims_dynamic_infer_output_shapes");
    ASSERT_TRUE(infer.valid()) << infer.error();
    std::array<std::int64_t, 3> expandedShape{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, expandedShape), kSuccess);
    EXPECT_EQ(expandedShape, dimensions);
    std::vector<float> actual(elements);
    ASSERT_EQ(expanded.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *squeezed, 0.0F));
  }
}

TEST(NumericalDynamicOperator, ReductionMatchesNcnnAcrossShapes) {
  CompiledModel compiled(REDUCTION_DYNAMIC_LIBRARY_PATH, "reduction_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(REDUCTION_DYNAMIC_LIBRARY_PATH,
                      "reduction_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{shape.height, 4, shape.width};
    const std::array<std::int64_t, 2> expectedShape{shape.height, shape.width};
    const std::vector<float> input =
      make_random_input(4U * shape.height * shape.width, 0x5EDU, -2.0F, 2.0F);
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, expectedShape);
    const ReferenceModel reference(fixture_path("reduction_dynamic"),
                                   NUMERICAL_EMPTY_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(shape.width, 4, shape.height));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> actual(shape.height * shape.width);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  }
}

TEST(NumericalDynamicOperator, ReductionMeanUsesRuntimeElementCount) {
  CompiledModel compiled(REDUCTION_DYNAMIC_MEAN_LIBRARY_PATH,
                         "reduction_dynamic_mean");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{4, shape.height, shape.width};
    const std::vector<float> input =
      make_random_input(4U * shape.height * shape.width, 0xC0A17U, -2.0F, 2.0F);
    const ReferenceModel reference(fixture_path("reduction_dynamic_mean"),
                                   NUMERICAL_EMPTY_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(shape.width, shape.height, 4));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), 4U);
    std::vector<float> actual(4);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  }
}

TEST(NumericalDynamicOperator, GemmDynamicMMatchesNcnnAcrossShapes) {
  CompiledModel compiled(GEMM_DYNAMIC_LIBRARY_PATH, "gemm_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(GEMM_DYNAMIC_LIBRARY_PATH,
                      "gemm_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{2, 1, shape.height};
    const std::array<std::int64_t, 2> expectedShape{shape.height, 4};
    const std::vector<float> input =
      make_random_input(2U * shape.height, 0x6E44U, -2.0F, 2.0F);
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, expectedShape);
    const ReferenceModel reference(fixture_path("gemm_dynamic"),
                                   GEMM_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(shape.height, 1, 2));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> actual(4U * shape.height);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-6F));
  }
}

TEST(NumericalDynamicOperator, FixedSliceMatchesNcnnAcrossShapes) {
  CompiledModel compiled(SLICE_DYNAMIC_LIBRARY_PATH, "slice_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(SLICE_DYNAMIC_LIBRARY_PATH,
                      "slice_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const Case shape : kCases) {
    const std::array<std::int64_t, 3> dimensions{4, shape.height, shape.width};
    const std::array<std::int64_t, 3> expectedShape{
      2, shape.height, shape.width};
    const std::vector<float> input =
      make_random_input(4U * shape.height * shape.width, 0x511CEU, -2.0F, 2.0F);
    std::array<std::int64_t, 3> leftShape{};
    std::array<std::int64_t, 3> rightShape{};
    ASSERT_EQ(
      infer.infer_dynamic_two_outputs(dimensions, leftShape, rightShape),
      kSuccess);
    EXPECT_EQ(leftShape, expectedShape);
    EXPECT_EQ(rightShape, expectedShape);
    const ReferenceInput referenceInput(
      "data", TensorShape(shape.width, shape.height, 4), input);
    constexpr std::array<std::string_view, 2> kOutputs{"left", "right"};
    const auto expected = run_ncnn_reference(fixture_path("slice_dynamic"),
                                             NUMERICAL_EMPTY_BIN_PATH,
                                             std::span(&referenceInput, 1),
                                             kOutputs);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    const std::size_t elements = 2U * shape.height * shape.width;
    std::vector<float> left(elements);
    std::vector<float> right(elements);
    ASSERT_EQ(compiled.run_dynamic_two_outputs(input, dimensions, left, right),
              kSuccess);
    EXPECT_TRUE(compare_values(left, (*expected)[0], 0.0F));
    EXPECT_TRUE(compare_values(right, (*expected)[1], 0.0F));
  }
}

TEST(NumericalDynamicOperator, GlobalPoolingMatchesNcnnAcrossShapes) {
  for (const auto& [fixture, library, symbol, tolerance] :
       {std::tuple{"pooling_global_max_dynamic",
                   POOLING_GLOBAL_MAX_DYNAMIC_LIBRARY_PATH,
                   "pooling_global_max_dynamic",
                   0.0F},
        std::tuple{"pooling_global_average_dynamic",
                   POOLING_GLOBAL_AVERAGE_DYNAMIC_LIBRARY_PATH,
                   "pooling_global_average_dynamic",
                   1.0e-6F}}) {
    CompiledModel compiled(library, symbol);
    ASSERT_TRUE(compiled.valid()) << compiled.error();
    for (const Case shape : kCases) {
      const std::array<std::int64_t, 3> dimensions{
        3, shape.height, shape.width};
      const std::vector<float> input = make_random_input(
        3U * shape.height * shape.width, 0x610BA1U, -2.0F, 2.0F);
      const ReferenceModel reference(fixture_path(fixture),
                                     NUMERICAL_EMPTY_BIN_PATH,
                                     "data",
                                     "output",
                                     TensorShape(shape.width, shape.height, 3));
      const auto expected = run_ncnn_reference(reference, input);
      ASSERT_TRUE(expected.has_value()) << expected.error();
      std::vector<float> actual(3);
      ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
                kSuccess);
      EXPECT_TRUE(compare_values(actual, *expected, tolerance));
    }
  }
}

TEST(NumericalDynamicOperator, PaddedRegularPoolingMatchesNcnnAcrossShapes) {
  constexpr std::array<Case, 4> kPaddedCases{{
    Case{.height = 1, .width = 1},
    Case{.height = 3, .width = 5},
    Case{.height = 4, .width = 6},
    Case{.height = 7, .width = 11},
  }};
  for (const auto& [fixture, library, symbol, tolerance] :
       {std::tuple{"pooling_padded_max_dynamic",
                   POOLING_PADDED_MAX_DYNAMIC_LIBRARY_PATH,
                   "pooling_padded_max_dynamic",
                   0.0F},
        std::tuple{"pooling_padded_average_dynamic",
                   POOLING_PADDED_AVERAGE_DYNAMIC_LIBRARY_PATH,
                   "pooling_padded_average_dynamic",
                   1.0e-6F},
        std::tuple{"pooling_padded_average_include_dynamic",
                   POOLING_PADDED_AVERAGE_INCLUDE_DYNAMIC_LIBRARY_PATH,
                   "pooling_padded_average_include_dynamic",
                   1.0e-6F}}) {
    CompiledModel compiled(library, symbol);
    ASSERT_TRUE(compiled.valid()) << compiled.error();
    CompiledModel infer(library, std::string(symbol) + "_infer_output_shapes");
    ASSERT_TRUE(infer.valid()) << infer.error();

    for (const Case shape : kPaddedCases) {
      const std::array<std::int64_t, 3> dimensions{
        3, shape.height, shape.width};
      const std::array<std::int64_t, 3> expectedShape{
        3, (shape.height + 1) / 2, (shape.width + 1) / 2};
      const std::vector<float> input = make_random_input(
        3U * shape.height * shape.width, 0x50AD00U, -2.0F, 2.0F);
      std::array<std::int64_t, 3> inferred{};
      ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
      EXPECT_EQ(inferred, expectedShape);

      const ReferenceModel reference(fixture_path(fixture),
                                     NUMERICAL_EMPTY_BIN_PATH,
                                     "data",
                                     "output",
                                     TensorShape(shape.width, shape.height, 3));
      const auto expected = run_ncnn_reference(reference, input);
      ASSERT_TRUE(expected.has_value()) << expected.error();
      const std::size_t outputElements =
        3U * static_cast<std::size_t>(expectedShape[1]) *
        static_cast<std::size_t>(expectedShape[2]);
      ASSERT_EQ(expected->size(), outputElements);

      std::vector<float> actual(outputElements);
      ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
                kSuccess);
      EXPECT_TRUE(compare_values(actual, *expected, tolerance))
        << fixture << " at " << shape.height << "x" << shape.width;
    }
  }
}

TEST(NumericalDynamicOperator, AdaptivePoolingMatchesNcnnAcrossShapes) {
  for (const auto& [fixture, library, symbol, tolerance] :
       {std::tuple{"pooling_adaptive_max_dynamic",
                   POOLING_ADAPTIVE_MAX_DYNAMIC_LIBRARY_PATH,
                   "pooling_adaptive_max_dynamic",
                   0.0F},
        std::tuple{"pooling_adaptive_average_dynamic",
                   POOLING_ADAPTIVE_AVERAGE_DYNAMIC_LIBRARY_PATH,
                   "pooling_adaptive_average_dynamic",
                   1.0e-6F}}) {
    CompiledModel compiled(library, symbol);
    ASSERT_TRUE(compiled.valid()) << compiled.error();
    for (const Case shape : kCases) {
      const std::array<std::int64_t, 3> dimensions{
        3, shape.height, shape.width};
      const std::vector<float> input = make_random_input(
        3U * shape.height * shape.width, 0xADA771U, -2.0F, 2.0F);
      const ReferenceModel reference(fixture_path(fixture),
                                     NUMERICAL_EMPTY_BIN_PATH,
                                     "data",
                                     "output",
                                     TensorShape(shape.width, shape.height, 3));
      const auto expected = run_ncnn_reference(reference, input);
      ASSERT_TRUE(expected.has_value()) << expected.error();
      const std::size_t outputElements =
        std::string_view(fixture) == "pooling_adaptive_max_dynamic"
          ? 6U * shape.width
          : 18U;
      std::vector<float> actual(outputElements);
      ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
                kSuccess);
      EXPECT_TRUE(compare_values(actual, *expected, tolerance));
    }
  }
}

TEST(NumericalDynamicOperator, ReshapeSpecMatchesNcnnAcrossShapes) {
  CompiledModel compiled(RESHAPE_SPEC_DYNAMIC_LIBRARY_PATH,
                         "reshape_spec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(RESHAPE_SPEC_DYNAMIC_LIBRARY_PATH,
                      "reshape_spec_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const std::int64_t rows : {3, 7}) {
    const std::array<std::int64_t, 3> dimensions{1, rows, 6};
    const std::array<std::int64_t, 2> expectedShape{rows * 2, 3};
    const std::vector<float> input =
      make_random_input(rows * 6U, 0x2E5A9EU, -2.0F, 2.0F);
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, expectedShape);
    const ReferenceModel reference(fixture_path("reshape_spec_dynamic"),
                                   NUMERICAL_EMPTY_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(6, rows, 1));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> actual(input.size());
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
  }
}

TEST(NumericalDynamicOperator, OrderedSliceMatchesNcnnAcrossShapes) {
  CompiledModel compiled(SLICE_ORDERED_DYNAMIC_LIBRARY_PATH,
                         "slice_ordered_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(SLICE_ORDERED_DYNAMIC_LIBRARY_PATH,
                      "slice_ordered_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const std::int64_t rows : {8, 11}) {
    const std::array<std::int64_t, 3> dimensions{1, rows, 4};
    const std::vector<float> input =
      make_random_input(rows * 4U, 0x511CEDU, -2.0F, 2.0F);
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, (std::array<std::int64_t, 2>{rows, 4}));
    const ReferenceModel reference(fixture_path("slice_ordered_dynamic"),
                                   NUMERICAL_EMPTY_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(4, rows, 1));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> actual(input.size());
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 0.0F));
  }
}

TEST(NumericalDynamicOperator, InnerProductDynamicMMatchesNcnnAcrossShapes) {
  CompiledModel compiled(INNER_PRODUCT_DYNAMIC_M_LIBRARY_PATH,
                         "inner_product_dynamic_m");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(INNER_PRODUCT_DYNAMIC_M_LIBRARY_PATH,
                      "inner_product_dynamic_m_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  for (const std::int64_t rows : {3, 7}) {
    const std::array<std::int64_t, 3> dimensions{1, rows, 4};
    const std::array<std::int64_t, 2> expectedShape{rows, 3};
    const std::vector<float> input =
      make_random_input(rows * 4U, 0x1AAE2U, -2.0F, 2.0F);
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess);
    EXPECT_EQ(inferred, expectedShape);
    const ReferenceModel reference(fixture_path("inner_product_dynamic_m"),
                                   INNER_PRODUCT_DYNAMIC_M_BIN_PATH,
                                   "data",
                                   "output",
                                   TensorShape(4, rows, 1));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> actual(rows * 3U);
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-5F));
  }
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
