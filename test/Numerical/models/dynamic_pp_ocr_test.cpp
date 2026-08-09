#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <sys/resource.h>

namespace ncnn_compiler::test {
namespace {

constexpr int kSuccess = 0;
constexpr int kInvalidShape = 2;
constexpr int kConstraintViolation = 3;
constexpr int kShapeArithmeticOverflow = 4;
constexpr int kOutputCapacityInsufficient = 5;

struct DynamicCase {
  std::int64_t height;
  std::int64_t width;
};

constexpr std::array<DynamicCase, 5> kCases = {{
  DynamicCase{.height = 32, .width = 32},
  DynamicCase{.height = 320, .width = 320},
  DynamicCase{.height = 320, .width = 960},
  DynamicCase{.height = 736, .width = 1280},
  DynamicCase{.height = 640, .width = 640},
}};

std::size_t element_count(const DynamicCase& shape) {
  return static_cast<std::size_t>(shape.height) *
         static_cast<std::size_t>(shape.width) * 3U;
}

std::array<std::int64_t, 3> input_shape(const DynamicCase& shape) {
  return {3, shape.height, shape.width};
}

std::array<std::int64_t, 3> expected_output_shape(const DynamicCase& shape) {
  return {1, shape.height, shape.width};
}

void record_peak_rss() {
  struct rusage usage{};
  ASSERT_EQ(getrusage(RUSAGE_SELF, &usage), 0);
  ::testing::Test::RecordProperty("peak_rss_kib",
                                  std::to_string(usage.ru_maxrss));
}

TEST(NumericalDynamicModel, PPOCRv5ServerDetMatchesNcnnAcrossShapes) {
  CompiledModel compiled(PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv5_server_det_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const DynamicCase& shape : kCases) {
    const auto input_dimensions = input_shape(shape);
    const auto output_dimensions = expected_output_shape(shape);
    const std::size_t input_elements = element_count(shape);
    const std::size_t output_elements = static_cast<std::size_t>(shape.height) *
                                        static_cast<std::size_t>(shape.width);
    const std::vector<float> input = make_random_input(
      input_elements,
      static_cast<std::uint32_t>((shape.height * 131) + shape.width),
      -0.01F,
      0.01F);
    std::array<std::int64_t, 3> inferred_shape{};
    ASSERT_EQ(infer.infer_dynamic(input_dimensions, inferred_shape), kSuccess);
    EXPECT_EQ(inferred_shape, output_dimensions);

    const ReferenceModel reference(
      PP_OCRV5_SERVER_DET_DYNAMIC_PARAM_PATH,
      PP_OCRV5_SERVER_DET_DYNAMIC_BIN_PATH,
      "in0",
      "out0",
      TensorShape(shape.width, static_cast<int>(shape.height), 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), output_elements);

    std::vector<float> actual(output_elements);
    ASSERT_EQ(
      compiled.run_dynamic(input, input_dimensions, actual, output_elements),
      kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
    EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
      return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    }));

    std::vector<float> repeated(output_elements);
    ASSERT_EQ(
      compiled.run_dynamic(input, input_dimensions, repeated, output_elements),
      kSuccess);
    EXPECT_EQ(repeated, actual);
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5ServerDetRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normal_shape = {3, 32, 32};
  const std::vector<float> input(3 * 32 * 32, 0.0F);
  std::vector<float> output(32 * 32, 0.0F);
  EXPECT_EQ(compiled.run_dynamic(input, normal_shape, output, output.size()),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 5>{{
         {3, 31, 32},
         {3, 32, 31},
         {3, 33, 32},
         {3, 320, 321},
         {3, 0, 320},
       }}) {
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, 1),
              dimensions[1] == 0 ? kInvalidShape : kConstraintViolation);
  }

  std::vector<float> output_capacity_failure((32 * 32) - 1);
  EXPECT_EQ(compiled.run_dynamic(input,
                                 normal_shape,
                                 output_capacity_failure,
                                 output_capacity_failure.size()),
            kOutputCapacityInsufficient);

  const std::array<std::int64_t, 3> overflow_shape = {
    3,
    std::numeric_limits<std::int64_t>::max() - 31,
    std::numeric_limits<std::int64_t>::max() - 31,
  };
  EXPECT_EQ(compiled.run_dynamic(input, overflow_shape, output, 1),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5ServerDetSupportsAlternatingShapes) {
  CompiledModel compiled(PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  for (const DynamicCase& shape : std::array<DynamicCase, 3>{{
         DynamicCase{.height = 320, .width = 960},
         DynamicCase{.height = 640, .width = 640},
         DynamicCase{.height = 736, .width = 1280},
       }}) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input(element_count(shape), 0.001F);
    const std::size_t output_elements = static_cast<std::size_t>(shape.height) *
                                        static_cast<std::size_t>(shape.width);
    std::vector<float> output(output_elements);
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              kSuccess);
  }
  record_peak_rss();
}

}  // namespace
}  // namespace ncnn_compiler::test
