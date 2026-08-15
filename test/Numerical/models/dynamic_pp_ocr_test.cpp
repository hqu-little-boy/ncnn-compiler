#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
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

constexpr std::size_t kTinyRecClasses = 6906;
constexpr std::size_t kRecClasses = 18385;

struct DetectionCase {
  std::int64_t height;
  std::int64_t width;
};

constexpr std::array<DetectionCase, 5> kDetectionCases = {{
  DetectionCase{.height = 32, .width = 32},
  DetectionCase{.height = 320, .width = 320},
  DetectionCase{.height = 320, .width = 960},
  DetectionCase{.height = 736, .width = 1280},
  DetectionCase{.height = 640, .width = 640},
}};

std::size_t element_count(const DetectionCase& shape) {
  return static_cast<std::size_t>(shape.height) *
         static_cast<std::size_t>(shape.width) * 3U;
}

std::array<std::int64_t, 3> input_shape(const DetectionCase& shape) {
  return {3, shape.height, shape.width};
}

std::array<std::int64_t, 3> expected_output_shape(const DetectionCase& shape) {
  return {1, shape.height, shape.width};
}

std::size_t recognition_input_elements(std::int64_t width) {
  return 3U * 48U * static_cast<std::size_t>(width);
}

std::int64_t recognition_sequence_length(std::int64_t width) {
  return (width + 3) / 8;
}

std::size_t recognition_output_elements(std::int64_t width) {
  return static_cast<std::size_t>(recognition_sequence_length(width)) *
         kRecClasses;
}

std::size_t tiny_recognition_output_elements(std::int64_t width) {
  return static_cast<std::size_t>(recognition_sequence_length(width)) *
         kTinyRecClasses;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

void record_peak_rss() {
  struct rusage usage{};
  ASSERT_EQ(getrusage(RUSAGE_SELF, &usage), 0);
  ::testing::Test::RecordProperty("peak_rss_kib",
                                  std::to_string(usage.ru_maxrss));
}

TEST(NumericalDynamicModel, PPOCRv5MobileDetMatchesNcnnAcrossShapes) {
  CompiledModel compiled(PP_OCRV5_MOBILE_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(PP_OCRV5_MOBILE_DET_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv5_mobile_det_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const DetectionCase& shape : kDetectionCases) {
    const auto inputDimensions = input_shape(shape);
    const auto outputDimensions = expected_output_shape(shape);
    const std::size_t outputElements = static_cast<std::size_t>(shape.height) *
                                       static_cast<std::size_t>(shape.width);
    const std::vector<float> input = make_random_input(
      element_count(shape),
      static_cast<std::uint32_t>((shape.height * 137) + shape.width),
      -0.01F,
      0.01F);
    std::array<std::int64_t, 3> inferredShape{};
    ASSERT_EQ(infer.infer_dynamic(inputDimensions, inferredShape), kSuccess);
    EXPECT_EQ(inferredShape, outputDimensions);

    const ReferenceModel reference(
      PP_OCRV5_MOBILE_DET_DYNAMIC_PARAM_PATH,
      PP_OCRV5_MOBILE_DET_DYNAMIC_BIN_PATH,
      "in0",
      "out0",
      TensorShape(shape.width, static_cast<int>(shape.height), 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), outputElements);

    std::vector<float> actual(outputElements);
    ASSERT_EQ(
      compiled.run_dynamic(input, inputDimensions, actual, outputElements),
      kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
    EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
      return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    }));

    std::vector<float> repeated(outputElements);
    ASSERT_EQ(
      compiled.run_dynamic(input, inputDimensions, repeated, outputElements),
      kSuccess);
    EXPECT_EQ(repeated, actual);
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5MobileDetRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_OCRV5_MOBILE_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normalShape = {3, 32, 32};
  const std::vector<float> input(3 * 32 * 32, 0.0F);
  std::vector<float> output(32 * 32, 0.0F);
  EXPECT_EQ(compiled.run_dynamic(input, normalShape, output, output.size()),
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

  std::vector<float> shortOutput((32 * 32) - 1);
  EXPECT_EQ(
    compiled.run_dynamic(input, normalShape, shortOutput, shortOutput.size()),
    kOutputCapacityInsufficient);

  const std::array<std::int64_t, 3> overflowShape = {
    3,
    std::numeric_limits<std::int64_t>::max() - 31,
    std::numeric_limits<std::int64_t>::max() - 31,
  };
  EXPECT_EQ(compiled.run_dynamic(input, overflowShape, output, 1),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5MobileDetSupportsAlternatingShapes) {
  CompiledModel compiled(PP_OCRV5_MOBILE_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  for (const DetectionCase& shape : std::array<DetectionCase, 3>{{
         DetectionCase{.height = 320, .width = 960},
         DetectionCase{.height = 640, .width = 640},
         DetectionCase{.height = 736, .width = 1280},
       }}) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input(element_count(shape), 0.001F);
    const std::size_t outputElements = static_cast<std::size_t>(shape.height) *
                                       static_cast<std::size_t>(shape.width);
    std::vector<float> output(outputElements);
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              kSuccess);
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5ServerDetMatchesNcnnAcrossShapes) {
  CompiledModel compiled(PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_det_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv5_server_det_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const DetectionCase& shape : kDetectionCases) {
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
  for (const DetectionCase& shape : std::array<DetectionCase, 3>{{
         DetectionCase{.height = 320, .width = 960},
         DetectionCase{.height = 640, .width = 640},
         DetectionCase{.height = 736, .width = 1280},
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

TEST(NumericalDynamicModel, PPOCRv6TinyRecInfersSequenceAcrossWidths) {
  CompiledModel infer(PP_OCRV6_TINY_REC_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv6_tiny_rec_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const std::int64_t width :
       std::array<std::int64_t, 9>{5, 8, 12, 13, 20, 21, 64, 319, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess)
      << "width " << width;
    EXPECT_EQ(
      inferred,
      (std::array<std::int64_t, 2>{recognition_sequence_length(width),
                                   static_cast<std::int64_t>(kTinyRecClasses)}))
      << "width " << width;
  }
}

TEST(NumericalDynamicModel, PPOCRv6TinyRecMatchesNcnnAcrossWidths) {
  CompiledModel compiled(PP_OCRV6_TINY_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv6_tiny_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{5, 13, 64, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input =
      make_random_input(recognition_input_elements(width),
                        static_cast<std::uint32_t>(0x36540000U + width),
                        -1.0F,
                        1.0F);
    const ReferenceModel reference(PP_OCRV6_TINY_REC_DYNAMIC_PARAM_PATH,
                                   PP_OCRV6_TINY_REC_DYNAMIC_BIN_PATH,
                                   "in0",
                                   "out0",
                                   TensorShape(static_cast<int>(width), 48, 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), tiny_recognition_output_elements(width));

    std::vector<float> actual(expected->size());
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess)
      << "width " << width;
    EXPECT_TRUE(compare_values(actual, *expected, 5.0e-4F))
      << "width " << width;
    for (std::int64_t row = 0; row < recognition_sequence_length(width);
         ++row) {
      const std::size_t offset =
        static_cast<std::size_t>(row) * kTinyRecClasses;
      EXPECT_TRUE(check_softmax(
        std::span<const float>(actual).subspan(offset, kTinyRecClasses),
        std::span<const float>(*expected).subspan(offset, kTinyRecClasses),
        5.0e-4))
        << "width " << width << ", row " << row;
    }
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv6TinyRecSupportsAlternatingWidths) {
  CompiledModel compiled(PP_OCRV6_TINY_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv6_tiny_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{320, 5, 64, 13}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input(recognition_input_elements(width), 0.001F);
    std::vector<float> output(tiny_recognition_output_elements(width));
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              kSuccess)
      << "width " << width;
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv6TinyRecRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_OCRV6_TINY_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv6_tiny_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normalShape = {3, 48, 5};
  const std::vector<float> input(recognition_input_elements(5), 0.0F);
  std::vector<float> output(tiny_recognition_output_elements(5));
  ASSERT_EQ(compiled.run_dynamic(input, normalShape, output, output.size()),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 4>{
         {{3, 48, 0}, {3, 48, 4}, {3, 47, 5}, {1, 48, 5}}}) {
    const int expectedStatus =
      dimensions[2] == 4 ? kConstraintViolation : kInvalidShape;
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              expectedStatus);
  }

  std::vector<float> shortOutput(tiny_recognition_output_elements(5) - 1);
  EXPECT_EQ(
    compiled.run_dynamic(input, normalShape, shortOutput, shortOutput.size()),
    kOutputCapacityInsufficient);

  const std::array<std::int64_t, 3> overflowShape = {
    3, 48, std::numeric_limits<std::int64_t>::max()};
  EXPECT_EQ(compiled.run_dynamic(input, overflowShape, output, output.size()),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv6TinyRecArtifactsDescribeDynamicAbi) {
  EXPECT_GT(std::filesystem::file_size(PP_OCRV6_TINY_REC_DYNAMIC_LIBRARY_PATH),
            0U);

  const std::string manifest =
    read_text(PP_OCRV6_TINY_REC_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv6_tiny_rec_dynamic"), std::string::npos);
  EXPECT_NE(manifest.find("6906"), std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 5"), std::string::npos);
  EXPECT_NE(manifest.find("\"multiple_of\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"shape_program\""), std::string::npos);

  const std::string header = read_text(PP_OCRV6_TINY_REC_DYNAMIC_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_OCRV6_TINY_REC_DYNAMIC_INPUT1_RANK 3",
         "#define PP_OCRV6_TINY_REC_DYNAMIC_INPUT1_DIM2 NCNN_DYNAMIC_DIM",
         "#define PP_OCRV6_TINY_REC_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x4)",
         "#define PP_OCRV6_TINY_REC_DYNAMIC_INPUT1_DIM2_MINIMUM "
         "INT64_C(5)",
         "#define PP_OCRV6_TINY_REC_DYNAMIC_OUTPUT1_RANK 2",
         "#define PP_OCRV6_TINY_REC_DYNAMIC_OUTPUT1_DIM1 INT64_C(6906)",
         "pp_ocrv6_tiny_rec_dynamic_infer_output_shapes",
         "uint64_t output1_capacity",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }

  const std::string linalgIr =
    read_text(PP_OCRV6_TINY_REC_DYNAMIC_LINALG_IR_PATH);
  EXPECT_EQ(linalgIr.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalgIr.find("math.exp"), std::string::npos);
  EXPECT_NE(linalgIr.find("tensor.expand_shape"), std::string::npos);
}

TEST(NumericalDynamicModel, PPOCRv5MobileRecInfersSequenceAcrossWidths) {
  CompiledModel infer(PP_OCRV5_MOBILE_REC_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv5_mobile_rec_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const std::int64_t width :
       std::array<std::int64_t, 9>{5, 8, 12, 13, 20, 21, 64, 319, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess)
      << "width " << width;
    EXPECT_EQ(
      inferred,
      (std::array<std::int64_t, 2>{recognition_sequence_length(width),
                                   static_cast<std::int64_t>(kRecClasses)}))
      << "width " << width;
  }
}

TEST(NumericalDynamicModel, PPOCRv5MobileRecMatchesNcnnAcrossWidths) {
  CompiledModel compiled(PP_OCRV5_MOBILE_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{5, 13, 64, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input =
      make_random_input(recognition_input_elements(width),
                        static_cast<std::uint32_t>(0x354D0000U + width),
                        -1.0F,
                        1.0F);
    const ReferenceModel reference(PP_OCRV5_MOBILE_REC_DYNAMIC_PARAM_PATH,
                                   PP_OCRV5_MOBILE_REC_DYNAMIC_BIN_PATH,
                                   "in0",
                                   "out0",
                                   TensorShape(static_cast<int>(width), 48, 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), recognition_output_elements(width));

    std::vector<float> actual(expected->size());
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess)
      << "width " << width;
    EXPECT_TRUE(compare_values(actual, *expected, 5.0e-4F))
      << "width " << width;
    for (std::int64_t row = 0; row < recognition_sequence_length(width);
         ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * kRecClasses;
      EXPECT_TRUE(check_softmax(
        std::span<const float>(actual).subspan(offset, kRecClasses),
        std::span<const float>(*expected).subspan(offset, kRecClasses),
        5.0e-4))
        << "width " << width << ", row " << row;
    }
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5MobileRecSupportsAlternatingWidths) {
  CompiledModel compiled(PP_OCRV5_MOBILE_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{320, 5, 64, 13}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input(recognition_input_elements(width), 0.001F);
    std::vector<float> output(recognition_output_elements(width));
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              kSuccess)
      << "width " << width;
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5MobileRecRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_OCRV5_MOBILE_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normalShape = {3, 48, 5};
  const std::vector<float> input(recognition_input_elements(5), 0.0F);
  std::vector<float> output(recognition_output_elements(5));
  ASSERT_EQ(compiled.run_dynamic(input, normalShape, output, output.size()),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 4>{
         {{3, 48, 0}, {3, 48, 4}, {3, 47, 5}, {1, 48, 5}}}) {
    const int expectedStatus =
      dimensions[2] == 4 ? kConstraintViolation : kInvalidShape;
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              expectedStatus);
  }

  std::vector<float> shortOutput(recognition_output_elements(5) - 1);
  EXPECT_EQ(
    compiled.run_dynamic(input, normalShape, shortOutput, shortOutput.size()),
    kOutputCapacityInsufficient);

  const std::array<std::int64_t, 3> overflowShape = {
    3, 48, std::numeric_limits<std::int64_t>::max()};
  EXPECT_EQ(compiled.run_dynamic(input, overflowShape, output, output.size()),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5MobileRecArtifactsDescribeDynamicAbi) {
  EXPECT_GT(
    std::filesystem::file_size(PP_OCRV5_MOBILE_REC_DYNAMIC_LIBRARY_PATH), 0U);

  const std::string manifest =
    read_text(PP_OCRV5_MOBILE_REC_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv5_mobile_rec_dynamic"), std::string::npos);
  EXPECT_NE(manifest.find("18385"), std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 5"), std::string::npos);
  EXPECT_NE(manifest.find("\"multiple_of\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"shape_program\""), std::string::npos);

  const std::string header = read_text(PP_OCRV5_MOBILE_REC_DYNAMIC_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_OCRV5_MOBILE_REC_DYNAMIC_INPUT1_RANK 3",
         "#define PP_OCRV5_MOBILE_REC_DYNAMIC_INPUT1_DIM2 NCNN_DYNAMIC_DIM",
         "#define PP_OCRV5_MOBILE_REC_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x4)",
         "#define PP_OCRV5_MOBILE_REC_DYNAMIC_INPUT1_DIM2_MINIMUM "
         "INT64_C(5)",
         "#define PP_OCRV5_MOBILE_REC_DYNAMIC_OUTPUT1_RANK 2",
         "#define PP_OCRV5_MOBILE_REC_DYNAMIC_OUTPUT1_DIM1 INT64_C(18385)",
         "pp_ocrv5_mobile_rec_dynamic_infer_output_shapes",
         "uint64_t output1_capacity",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }

  const std::string linalgIr =
    read_text(PP_OCRV5_MOBILE_REC_DYNAMIC_LINALG_IR_PATH);
  EXPECT_EQ(linalgIr.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalgIr.find("math.exp"), std::string::npos);
  EXPECT_NE(linalgIr.find("tensor.expand_shape"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_compiler::test
