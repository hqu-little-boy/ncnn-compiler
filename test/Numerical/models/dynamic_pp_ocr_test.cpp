#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
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
constexpr std::size_t kMediumRecClasses = 18710;

struct DetectionCase {
  std::int64_t height;
  std::int64_t width;
};

struct DynamicDetectionModel {
  const char* test_name;
  const char* library_path;
  const char* param_path;
  const char* bin_path;
  const char* symbol;
  const char* infer_symbol;
  std::array<DetectionCase, 3> numerical_cases;
  float tolerance;
  std::uint32_t seed;
};

struct DynamicLCNetModel {
  const char* test_name;
  const char* library_path;
  const char* param_path;
  const char* bin_path;
  const char* symbol;
  const char* manifest_path;
  const char* header_path;
  const char* linalg_ir_path;
  const char* header_prefix;
  std::size_t output_elements;
  std::uint32_t seed;
};

constexpr std::array<DetectionCase, 5> kDetectionCases = {{
  DetectionCase{.height = 32, .width = 32},
  DetectionCase{.height = 320, .width = 320},
  DetectionCase{.height = 320, .width = 960},
  DetectionCase{.height = 736, .width = 1280},
  DetectionCase{.height = 640, .width = 640},
}};

const std::array<DynamicDetectionModel, 3> kV6DynamicDetectionModels = {{
  DynamicDetectionModel{
    .test_name = "Tiny",
    .library_path = PP_OCRV6_TINY_DET_DYNAMIC_LIBRARY_PATH,
    .param_path = PP_OCRV6_TINY_DET_DYNAMIC_PARAM_PATH,
    .bin_path = PP_OCRV6_TINY_DET_DYNAMIC_BIN_PATH,
    .symbol = "pp_ocrv6_tiny_det_dynamic",
    .infer_symbol = "pp_ocrv6_tiny_det_dynamic_infer_output_shapes",
    .numerical_cases = {{
      DetectionCase{.height = 32, .width = 32},
      DetectionCase{.height = 320, .width = 960},
      DetectionCase{.height = 960, .width = 320},
    }},
    .tolerance = 1.0e-4F,
    .seed = 0x3654494EU,
  },
  DynamicDetectionModel{
    .test_name = "Small",
    .library_path = PP_OCRV6_SMALL_DET_DYNAMIC_LIBRARY_PATH,
    .param_path = PP_OCRV6_SMALL_DET_DYNAMIC_PARAM_PATH,
    .bin_path = PP_OCRV6_SMALL_DET_DYNAMIC_BIN_PATH,
    .symbol = "pp_ocrv6_small_det_dynamic",
    .infer_symbol = "pp_ocrv6_small_det_dynamic_infer_output_shapes",
    .numerical_cases = {{
      DetectionCase{.height = 32, .width = 32},
      DetectionCase{.height = 320, .width = 640},
      DetectionCase{.height = 640, .width = 320},
    }},
    .tolerance = 3.0e-4F,
    .seed = 0x36534D4CU,
  },
  DynamicDetectionModel{
    .test_name = "Medium",
    .library_path = PP_OCRV6_MEDIUM_DET_DYNAMIC_LIBRARY_PATH,
    .param_path = PP_OCRV6_MEDIUM_DET_DYNAMIC_PARAM_PATH,
    .bin_path = PP_OCRV6_MEDIUM_DET_DYNAMIC_BIN_PATH,
    .symbol = "pp_ocrv6_medium_det_dynamic",
    .infer_symbol = "pp_ocrv6_medium_det_dynamic_infer_output_shapes",
    .numerical_cases = {{
      DetectionCase{.height = 32, .width = 32},
      DetectionCase{.height = 256, .width = 640},
      DetectionCase{.height = 640, .width = 256},
    }},
    .tolerance = 1.0e-4F,
    .seed = 0x364D4544U,
  },
}};

const std::array<DynamicLCNetModel, 2> kDynamicLCNetModels = {{
  DynamicLCNetModel{
    .test_name = "DocOri",
    .library_path = PP_LCNET_DOC_ORI_DYNAMIC_LIBRARY_PATH,
    .param_path = PP_LCNET_DOC_ORI_DYNAMIC_PARAM_PATH,
    .bin_path = PP_LCNET_DOC_ORI_DYNAMIC_BIN_PATH,
    .symbol = "pp_lcnet_x1_0_doc_ori_dynamic",
    .manifest_path = PP_LCNET_DOC_ORI_DYNAMIC_MANIFEST_PATH,
    .header_path = PP_LCNET_DOC_ORI_DYNAMIC_HEADER_PATH,
    .linalg_ir_path = PP_LCNET_DOC_ORI_DYNAMIC_LINALG_IR_PATH,
    .header_prefix = "PP_LCNET_X1_0_DOC_ORI_DYNAMIC",
    .output_elements = 4,
    .seed = 0x4C43444FU,
  },
  DynamicLCNetModel{
    .test_name = "TextlineOri",
    .library_path = PP_LCNET_TEXTLINE_ORI_DYNAMIC_LIBRARY_PATH,
    .param_path = PP_LCNET_TEXTLINE_ORI_DYNAMIC_PARAM_PATH,
    .bin_path = PP_LCNET_TEXTLINE_ORI_DYNAMIC_BIN_PATH,
    .symbol = "pp_lcnet_x1_0_textline_ori_dynamic",
    .manifest_path = PP_LCNET_TEXTLINE_ORI_DYNAMIC_MANIFEST_PATH,
    .header_path = PP_LCNET_TEXTLINE_ORI_DYNAMIC_HEADER_PATH,
    .linalg_ir_path = PP_LCNET_TEXTLINE_ORI_DYNAMIC_LINALG_IR_PATH,
    .header_prefix = "PP_LCNET_X1_0_TEXTLINE_ORI_DYNAMIC",
    .output_elements = 2,
    .seed = 0x4C43544CU,
  },
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

std::size_t medium_recognition_output_elements(std::int64_t width) {
  return static_cast<std::size_t>(recognition_sequence_length(width)) *
         kMediumRecClasses;
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

class PPOCRv6DynamicDetTest
  : public ::testing::TestWithParam<DynamicDetectionModel> {};

class PPLCNetDynamicTest : public ::testing::TestWithParam<DynamicLCNetModel> {
};

TEST_P(PPOCRv6DynamicDetTest, MatchesNcnnAcrossShapes) {
  const DynamicDetectionModel& model = GetParam();
  CompiledModel compiled(model.library_path, model.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  CompiledModel infer(model.library_path, model.infer_symbol);
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const DetectionCase& shape : model.numerical_cases) {
    const auto input_dimensions = input_shape(shape);
    const auto output_dimensions = expected_output_shape(shape);
    const std::size_t output_elements = static_cast<std::size_t>(shape.height) *
                                        static_cast<std::size_t>(shape.width);
    const std::vector<float> input = make_random_input(
      element_count(shape),
      model.seed +
        static_cast<std::uint32_t>((shape.height * 131) + shape.width),
      -0.01F,
      0.01F);
    std::array<std::int64_t, 3> inferred_shape{};
    ASSERT_EQ(infer.infer_dynamic(input_dimensions, inferred_shape), kSuccess);
    EXPECT_EQ(inferred_shape, output_dimensions);

    const ReferenceModel reference(
      model.param_path,
      model.bin_path,
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
    EXPECT_TRUE(compare_values(actual, *expected, model.tolerance));
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

TEST_P(PPOCRv6DynamicDetTest, RejectsInvalidShapesAndCapacity) {
  const DynamicDetectionModel& model = GetParam();
  CompiledModel compiled(model.library_path, model.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normal_shape = {3, 32, 32};
  const std::vector<float> input(3 * 32 * 32, 0.0F);
  std::vector<float> output(32 * 32, 0.0F);
  ASSERT_EQ(compiled.run_dynamic(input, normal_shape, output, output.size()),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 6>{{
         {3, 31, 32},
         {3, 32, 31},
         {3, 33, 32},
         {3, 320, 321},
         {3, 0, 320},
         {1, 32, 32},
       }}) {
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, 1),
              dimensions[1] == 0 || dimensions[0] != 3 ? kInvalidShape
                                                       : kConstraintViolation);
  }

  std::vector<float> short_output((32 * 32) - 1);
  EXPECT_EQ(compiled.run_dynamic(
              input, normal_shape, short_output, short_output.size()),
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

TEST_P(PPOCRv6DynamicDetTest, SupportsAlternatingShapes) {
  const DynamicDetectionModel& model = GetParam();
  CompiledModel compiled(model.library_path, model.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const DetectionCase& shape : std::array<DetectionCase, 4>{
         DetectionCase{.height = 64, .width = 96},
         DetectionCase{.height = 96, .width = 64},
         DetectionCase{.height = 32, .width = 32},
         DetectionCase{.height = 64, .width = 96},
       }) {
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

INSTANTIATE_TEST_SUITE_P(
  PPOCRv6Det,
  PPOCRv6DynamicDetTest,
  ::testing::ValuesIn(kV6DynamicDetectionModels),
  [](const ::testing::TestParamInfo<DynamicDetectionModel>& info) {
    return info.param.test_name;
  });

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

TEST_P(PPLCNetDynamicTest, MatchesNcnnAcrossShapes) {
  const DynamicLCNetModel& model = GetParam();
  CompiledModel compiled(model.library_path, model.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const DetectionCase& shape : std::array<DetectionCase, 4>{
         DetectionCase{.height = 1, .width = 1},
         DetectionCase{.height = 79, .width = 159},
         DetectionCase{.height = 80, .width = 160},
         DetectionCase{.height = 224, .width = 224},
       }) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input = make_random_input(
      element_count(shape),
      model.seed +
        static_cast<std::uint32_t>((shape.height * 131) + shape.width),
      -1.0F,
      1.0F);
    const ReferenceModel reference(
      model.param_path,
      model.bin_path,
      "in0",
      "out0",
      TensorShape(
        static_cast<int>(shape.width), static_cast<int>(shape.height), 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), model.output_elements);

    std::vector<float> actual(model.output_elements);
    ASSERT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, actual),
              kSuccess)
      << "shape " << shape.height << 'x' << shape.width;
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F))
      << "shape " << shape.height << 'x' << shape.width;
    EXPECT_TRUE(check_softmax(actual, *expected));
  }
  record_peak_rss();
}

TEST_P(PPLCNetDynamicTest, SupportsAlternatingShapes) {
  const DynamicLCNetModel& model = GetParam();
  CompiledModel compiled(model.library_path, model.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const DetectionCase& shape : std::array<DetectionCase, 4>{
         DetectionCase{.height = 81, .width = 161},
         DetectionCase{.height = 1, .width = 1},
         DetectionCase{.height = 80, .width = 160},
         DetectionCase{.height = 81, .width = 161},
       }) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input(element_count(shape), 0.001F);
    std::vector<float> output(model.output_elements);
    EXPECT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, output),
              kSuccess)
      << "shape " << shape.height << 'x' << shape.width;
  }
  record_peak_rss();
}

TEST_P(PPLCNetDynamicTest, RejectsInvalidShapes) {
  const DynamicLCNetModel& model = GetParam();
  CompiledModel compiled(model.library_path, model.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normal_shape = {3, 80, 160};
  const std::vector<float> input(3 * 80 * 160, 0.0F);
  std::vector<float> output(model.output_elements);
  ASSERT_EQ(compiled.run_dynamic_fixed_output(input, normal_shape, output),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 3>{
         {{3, 0, 160}, {3, 80, 0}, {1, 80, 160}}}) {
    EXPECT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, output),
              kInvalidShape);
  }

  const std::array<std::int64_t, 3> overflow_shape = {
    3,
    std::numeric_limits<std::int64_t>::max(),
    std::numeric_limits<std::int64_t>::max(),
  };
  EXPECT_EQ(compiled.run_dynamic_fixed_output(input, overflow_shape, output),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST_P(PPLCNetDynamicTest, ArtifactsDescribeDynamicInputAndFixedOutput) {
  const DynamicLCNetModel& model = GetParam();
  EXPECT_GT(std::filesystem::file_size(model.library_path), 0U);

  const std::string manifest = read_text(model.manifest_path);
  EXPECT_NE(manifest.find(model.symbol), std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"multiple_of\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"dynamic_dim_mask\": 6"), std::string::npos);

  const std::string prefix = model.header_prefix;
  const std::string header = read_text(model.header_path);
  for (const std::string& required : {
         "#define " + prefix + "_INPUT1_RANK 3",
         "#define " + prefix + "_INPUT1_DIM1 NCNN_DYNAMIC_DIM",
         "#define " + prefix + "_INPUT1_DIM2 NCNN_DYNAMIC_DIM",
         "#define " + prefix + "_INPUT1_DYNAMIC_DIM_MASK UINT32_C(0x6)",
         "#define " + prefix + "_INPUT1_DIM1_MINIMUM INT64_C(1)",
         "#define " + prefix + "_INPUT1_DIM2_MINIMUM INT64_C(1)",
         "#define " + prefix + "_OUTPUT1_RANK 1",
         "#define " + prefix + "_OUTPUT1_DIM0 INT64_C(" +
           std::to_string(model.output_elements) + ")",
         "#define " + prefix + "_OUTPUT1_DYNAMIC_DIM_MASK UINT32_C(0x0)",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }
  EXPECT_EQ(header.find("_infer_output_shapes"), std::string::npos);
  EXPECT_EQ(header.find("output1_capacity"), std::string::npos);

  const std::string linalg_ir = read_text(model.linalg_ir_path);
  EXPECT_NE(linalg_ir.find("tensor<3x?x?xf32>"), std::string::npos);
  EXPECT_NE(linalg_ir.find("linalg.reduce"), std::string::npos);
  EXPECT_NE(linalg_ir.find("math.exp"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("ncnn.pooling"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
  PPLCNet,
  PPLCNetDynamicTest,
  ::testing::ValuesIn(kDynamicLCNetModels),
  [](const ::testing::TestParamInfo<DynamicLCNetModel>& info) {
    return info.param.test_name;
  });

TEST(NumericalDynamicModel, PPLCNetDocOriInt8BackboneMatchesNcnnAcrossShapes) {
  CompiledModel compiled(PP_LCNET_DOC_ORI_INT8_BACKBONE_LIBRARY_PATH,
                         "pp_lcnet_x1_0_doc_ori_int8_backbone_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const DetectionCase& shape : std::array<DetectionCase, 3>{
         DetectionCase{.height = 160, .width = 160},
         DetectionCase{.height = 224, .width = 224},
         DetectionCase{.height = 256, .width = 192},
       }) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input = make_random_input(
      element_count(shape),
      static_cast<std::uint32_t>(0x49384242U + shape.height + shape.width),
      -1.0F,
      1.0F);
    const ReferenceModel reference(
      PP_LCNET_DOC_ORI_INT8_BACKBONE_PARAM_PATH,
      PP_LCNET_DOC_ORI_INT8_BACKBONE_BIN_PATH,
      "in0",
      "out0",
      TensorShape(
        static_cast<int>(shape.width), static_cast<int>(shape.height), 3));
    const auto expected =
      run_ncnn_reference(reference, input, ReferenceInferenceMode::Int8);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), 256U);

    std::vector<float> actual(256);
    ASSERT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, actual),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 2.5e-1F))
      << shape.height << 'x' << shape.width;
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel,
     PPLCNetDocOriInt8StaticAndDynamicExecutionsAreConsistent) {
  CompiledModel static_compiled(PP_LCNET_DOC_ORI_INT8_STATIC_LIBRARY_PATH,
                                "pp_lcnet_x1_0_doc_ori_int8");
  CompiledModel dynamic_compiled(PP_LCNET_DOC_ORI_INT8_DYNAMIC_LIBRARY_PATH,
                                 "pp_lcnet_x1_0_doc_ori_int8_dynamic");
  ASSERT_TRUE(static_compiled.valid()) << static_compiled.error();
  ASSERT_TRUE(dynamic_compiled.valid()) << dynamic_compiled.error();

  const DetectionCase standard{.height = 224, .width = 224};
  const std::vector<float> input =
    make_random_input(element_count(standard), 0x49384655U, -1.0F, 1.0F);
  std::vector<float> static_output(4);
  std::vector<float> dynamic_output(4);
  ASSERT_EQ(static_compiled.run(input, static_output), kSuccess);
  ASSERT_EQ(dynamic_compiled.run_dynamic_fixed_output(
              input, input_shape(standard), dynamic_output),
            kSuccess);
  EXPECT_TRUE(compare_values(dynamic_output, static_output, 1.0e-5F));

  for (const DetectionCase& shape : std::array<DetectionCase, 3>{
         DetectionCase{.height = 192, .width = 224},
         DetectionCase{.height = 256, .width = 192},
         DetectionCase{.height = 192, .width = 224},
       }) {
    const std::vector<float> dynamic_input(element_count(shape), 0.001F);
    std::vector<float> output(4);
    ASSERT_EQ(dynamic_compiled.run_dynamic_fixed_output(
                dynamic_input, input_shape(shape), output),
              kSuccess);
    EXPECT_TRUE(std::ranges::all_of(output, [](float value) {
      return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    }));
    EXPECT_NEAR(
      std::accumulate(output.begin(), output.end(), 0.0F), 1.0F, 1.0e-5F);
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPLCNetDocOriInt8RejectsInvalidShapes) {
  CompiledModel compiled(PP_LCNET_DOC_ORI_INT8_DYNAMIC_LIBRARY_PATH,
                         "pp_lcnet_x1_0_doc_ori_int8_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  const std::array<std::int64_t, 3> normal_shape{3, 80, 160};
  const std::vector<float> input(3 * 80 * 160, 0.0F);
  std::vector<float> output(4);
  ASSERT_EQ(compiled.run_dynamic_fixed_output(input, normal_shape, output),
            kSuccess);
  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 3>{
         {{3, 0, 160}, {3, 80, 0}, {1, 80, 160}}}) {
    EXPECT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, output),
              kInvalidShape);
  }
  const std::array<std::int64_t, 3> overflow_shape = {
    3,
    std::numeric_limits<std::int64_t>::max(),
    std::numeric_limits<std::int64_t>::max(),
  };
  EXPECT_EQ(compiled.run_dynamic_fixed_output(input, overflow_shape, output),
            kShapeArithmeticOverflow);
}

TEST(NumericalDynamicModel, PPLCNetDocOriInt8ArtifactsDescribeQuantizedAbi) {
  EXPECT_GT(
    std::filesystem::file_size(PP_LCNET_DOC_ORI_INT8_DYNAMIC_LIBRARY_PATH), 0U);
  const std::string manifest =
    read_text(PP_LCNET_DOC_ORI_INT8_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_lcnet_x1_0_doc_ori_int8_dynamic"),
            std::string::npos);
  EXPECT_NE(manifest.find("\"dynamic_dim_mask\": 6"), std::string::npos);
  const std::string header =
    read_text(PP_LCNET_DOC_ORI_INT8_DYNAMIC_HEADER_PATH);
  EXPECT_NE(
    header.find("PP_LCNET_X1_0_DOC_ORI_INT8_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK "
                "UINT32_C(0x6)"),
    std::string::npos);
  EXPECT_NE(
    header.find("PP_LCNET_X1_0_DOC_ORI_INT8_DYNAMIC_OUTPUT1_DIM0 INT64_C(4)"),
    std::string::npos);
  const std::string ncnn_ir =
    read_text(PP_LCNET_DOC_ORI_INT8_DYNAMIC_NCNN_IR_PATH);
  EXPECT_NE(ncnn_ir.find("int8_scale_term = 1"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("int8_scale_term = 102"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("xi8>"), std::string::npos);
  const std::string linalg_ir =
    read_text(PP_LCNET_DOC_ORI_INT8_DYNAMIC_LINALG_IR_PATH);
  EXPECT_NE(linalg_ir.find("arith.fptosi"), std::string::npos);
  EXPECT_NE(linalg_ir.find("arith.sitofp"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("ncnn.convolution"), std::string::npos);
}

TEST(NumericalDynamicModel, ChineseOCRLiteAngleNetMatchesNcnnAcrossShapes) {
  CompiledModel compiled(CHINESEOCR_LITE_ANGLENET_DYNAMIC_LIBRARY_PATH,
                         "chineseocr_lite_anglenet_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const DetectionCase& shape : std::array<DetectionCase, 4>{
         DetectionCase{.height = 1, .width = 1},
         DetectionCase{.height = 31, .width = 191},
         DetectionCase{.height = 32, .width = 192},
         DetectionCase{.height = 33, .width = 193},
       }) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input =
      make_random_input(element_count(shape),
                        static_cast<std::uint32_t>(
                          0x414E0000U + (shape.height * 131) + shape.width),
                        -1.0F,
                        1.0F);
    const ReferenceModel reference(
      CHINESEOCR_LITE_ANGLENET_DYNAMIC_PARAM_PATH,
      CHINESEOCR_LITE_ANGLENET_DYNAMIC_BIN_PATH,
      "in0",
      "out0",
      TensorShape(
        static_cast<int>(shape.width), static_cast<int>(shape.height), 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), 2U);

    std::vector<float> actual(2);
    ASSERT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, actual),
              kSuccess)
      << "shape " << shape.height << 'x' << shape.width;
    EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F))
      << "shape " << shape.height << 'x' << shape.width;
    EXPECT_TRUE(check_softmax(actual, *expected));
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, ChineseOCRLiteAngleNetSupportsAlternatingShapes) {
  CompiledModel compiled(CHINESEOCR_LITE_ANGLENET_DYNAMIC_LIBRARY_PATH,
                         "chineseocr_lite_anglenet_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const DetectionCase& shape : std::array<DetectionCase, 4>{
         DetectionCase{.height = 33, .width = 193},
         DetectionCase{.height = 1, .width = 1},
         DetectionCase{.height = 32, .width = 192},
         DetectionCase{.height = 33, .width = 193},
       }) {
    const auto dimensions = input_shape(shape);
    const std::vector<float> input(element_count(shape), 0.001F);
    std::vector<float> output(2);
    EXPECT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, output),
              kSuccess)
      << "shape " << shape.height << 'x' << shape.width;
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, ChineseOCRLiteAngleNetRejectsInvalidShapes) {
  CompiledModel compiled(CHINESEOCR_LITE_ANGLENET_DYNAMIC_LIBRARY_PATH,
                         "chineseocr_lite_anglenet_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normal_shape = {3, 32, 192};
  const std::vector<float> input(3 * 32 * 192, 0.0F);
  std::vector<float> output(2);
  ASSERT_EQ(compiled.run_dynamic_fixed_output(input, normal_shape, output),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 3>{
         {{3, 0, 192}, {3, 32, 0}, {1, 32, 192}}}) {
    EXPECT_EQ(compiled.run_dynamic_fixed_output(input, dimensions, output),
              kInvalidShape);
  }

  const std::array<std::int64_t, 3> overflow_shape = {
    3,
    std::numeric_limits<std::int64_t>::max(),
    std::numeric_limits<std::int64_t>::max(),
  };
  EXPECT_EQ(compiled.run_dynamic_fixed_output(input, overflow_shape, output),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, ChineseOCRLiteAngleNetArtifactsDescribeDynamicAbi) {
  EXPECT_GT(
    std::filesystem::file_size(CHINESEOCR_LITE_ANGLENET_DYNAMIC_LIBRARY_PATH),
    0U);

  const std::string manifest =
    read_text(CHINESEOCR_LITE_ANGLENET_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("chineseocr_lite_anglenet_dynamic"),
            std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"multiple_of\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"dynamic_dim_mask\": 6"), std::string::npos);
  EXPECT_NE(manifest.find("\"shape\": [\n        2\n      ]"),
            std::string::npos);

  const std::string header =
    read_text(CHINESEOCR_LITE_ANGLENET_DYNAMIC_HEADER_PATH);
  for (const std::string_view required : {
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_INPUT1_RANK 3",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_INPUT1_DIM1 "
         "NCNN_DYNAMIC_DIM",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_INPUT1_DIM2 "
         "NCNN_DYNAMIC_DIM",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x6)",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_INPUT1_DIM1_MINIMUM "
         "INT64_C(1)",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_INPUT1_DIM2_MINIMUM "
         "INT64_C(1)",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_OUTPUT1_RANK 1",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_OUTPUT1_DIM0 INT64_C(2)",
         "#define CHINESEOCR_LITE_ANGLENET_DYNAMIC_OUTPUT1_ELEMENTS "
         "UINT64_C(2)",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }

  const std::string linalg_ir =
    read_text(CHINESEOCR_LITE_ANGLENET_DYNAMIC_LINALG_IR_PATH);
  EXPECT_EQ(linalg_ir.find("ncnn.pooling"), std::string::npos);
  EXPECT_NE(linalg_ir.find("math.exp"), std::string::npos);
  EXPECT_NE(linalg_ir.find("linalg.reduce"), std::string::npos);
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

TEST(NumericalDynamicModel, PPOCRv5ServerRecInfersSequenceAcrossWidths) {
  CompiledModel infer(PP_OCRV5_SERVER_REC_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv5_server_rec_dynamic_infer_output_shapes");
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

TEST(NumericalDynamicModel, PPOCRv5ServerRecMatchesNcnnAcrossWidths) {
  CompiledModel compiled(PP_OCRV5_SERVER_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{5, 13, 64, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input =
      make_random_input(recognition_input_elements(width),
                        static_cast<std::uint32_t>(0x35530000U + width),
                        -1.0F,
                        1.0F);
    const ReferenceModel reference(PP_OCRV5_SERVER_REC_DYNAMIC_PARAM_PATH,
                                   PP_OCRV5_SERVER_REC_DYNAMIC_BIN_PATH,
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
        2.0e-4))
        << "width " << width << ", row " << row;
    }
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5ServerRecSupportsAlternatingWidths) {
  CompiledModel compiled(PP_OCRV5_SERVER_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_rec_dynamic");
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

TEST(NumericalDynamicModel, PPOCRv5ServerRecRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_OCRV5_SERVER_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv5_server_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normal_shape = {3, 48, 5};
  const std::vector<float> input(recognition_input_elements(5), 0.0F);
  std::vector<float> output(recognition_output_elements(5));
  ASSERT_EQ(compiled.run_dynamic(input, normal_shape, output, output.size()),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 4>{
         {{3, 48, 0}, {3, 48, 4}, {3, 47, 5}, {1, 48, 5}}}) {
    const int expected_status =
      dimensions[2] == 4 ? kConstraintViolation : kInvalidShape;
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              expected_status);
  }

  std::vector<float> short_output(recognition_output_elements(5) - 1);
  EXPECT_EQ(compiled.run_dynamic(
              input, normal_shape, short_output, short_output.size()),
            kOutputCapacityInsufficient);

  const std::array<std::int64_t, 3> overflow_shape = {
    3, 48, std::numeric_limits<std::int64_t>::max()};
  EXPECT_EQ(compiled.run_dynamic(input, overflow_shape, output, output.size()),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv5ServerRecArtifactsDescribeDynamicAbi) {
  EXPECT_GT(
    std::filesystem::file_size(PP_OCRV5_SERVER_REC_DYNAMIC_LIBRARY_PATH), 0U);

  const std::string manifest =
    read_text(PP_OCRV5_SERVER_REC_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv5_server_rec_dynamic"), std::string::npos);
  EXPECT_NE(manifest.find("18385"), std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 5"), std::string::npos);
  EXPECT_NE(manifest.find("\"multiple_of\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"shape_program\""), std::string::npos);

  const std::string header = read_text(PP_OCRV5_SERVER_REC_DYNAMIC_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_OCRV5_SERVER_REC_DYNAMIC_INPUT1_RANK 3",
         "#define PP_OCRV5_SERVER_REC_DYNAMIC_INPUT1_DIM2 NCNN_DYNAMIC_DIM",
         "#define PP_OCRV5_SERVER_REC_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x4)",
         "#define PP_OCRV5_SERVER_REC_DYNAMIC_INPUT1_DIM2_MINIMUM "
         "INT64_C(5)",
         "#define PP_OCRV5_SERVER_REC_DYNAMIC_OUTPUT1_RANK 2",
         "#define PP_OCRV5_SERVER_REC_DYNAMIC_OUTPUT1_DIM1 INT64_C(18385)",
         "pp_ocrv5_server_rec_dynamic_infer_output_shapes",
         "uint64_t output1_capacity",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }

  const std::string linalg_ir =
    read_text(PP_OCRV5_SERVER_REC_DYNAMIC_LINALG_IR_PATH);
  EXPECT_EQ(linalg_ir.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalg_ir.find("math.exp"), std::string::npos);
  EXPECT_NE(linalg_ir.find("tensor.expand_shape"), std::string::npos);
}

TEST(NumericalDynamicModel, PPOCRv6MediumRecInfersSequenceAcrossWidths) {
  CompiledModel infer(PP_OCRV6_MEDIUM_REC_DYNAMIC_LIBRARY_PATH,
                      "pp_ocrv6_medium_rec_dynamic_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();

  for (const std::int64_t width :
       std::array<std::int64_t, 9>{5, 8, 12, 13, 20, 21, 64, 319, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    std::array<std::int64_t, 2> inferred{};
    ASSERT_EQ(infer.infer_dynamic(dimensions, inferred), kSuccess)
      << "width " << width;
    EXPECT_EQ(inferred,
              (std::array<std::int64_t, 2>{
                recognition_sequence_length(width),
                static_cast<std::int64_t>(kMediumRecClasses)}))
      << "width " << width;
  }
}

TEST(NumericalDynamicModel, PPOCRv6MediumRecMatchesNcnnAcrossWidths) {
  CompiledModel compiled(PP_OCRV6_MEDIUM_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv6_medium_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{5, 13, 64, 320}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input =
      make_random_input(recognition_input_elements(width),
                        static_cast<std::uint32_t>(0x364D0000U + width),
                        -1.0F,
                        1.0F);
    const ReferenceModel reference(PP_OCRV6_MEDIUM_REC_DYNAMIC_PARAM_PATH,
                                   PP_OCRV6_MEDIUM_REC_DYNAMIC_BIN_PATH,
                                   "in0",
                                   "out0",
                                   TensorShape(static_cast<int>(width), 48, 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), medium_recognition_output_elements(width));

    std::vector<float> actual(expected->size());
    ASSERT_EQ(compiled.run_dynamic(input, dimensions, actual, actual.size()),
              kSuccess)
      << "width " << width;
    EXPECT_TRUE(compare_values(actual, *expected, 5.0e-4F))
      << "width " << width;
    for (std::int64_t row = 0; row < recognition_sequence_length(width);
         ++row) {
      const std::size_t offset =
        static_cast<std::size_t>(row) * kMediumRecClasses;
      EXPECT_TRUE(check_softmax(
        std::span<const float>(actual).subspan(offset, kMediumRecClasses),
        std::span<const float>(*expected).subspan(offset, kMediumRecClasses),
        2.0e-4))
        << "width " << width << ", row " << row;
    }
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv6MediumRecSupportsAlternatingWidths) {
  CompiledModel compiled(PP_OCRV6_MEDIUM_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv6_medium_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  for (const std::int64_t width : std::array<std::int64_t, 4>{320, 5, 64, 13}) {
    const std::array<std::int64_t, 3> dimensions = {3, 48, width};
    const std::vector<float> input(recognition_input_elements(width), 0.001F);
    std::vector<float> output(medium_recognition_output_elements(width));
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              kSuccess)
      << "width " << width;
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv6MediumRecRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_OCRV6_MEDIUM_REC_DYNAMIC_LIBRARY_PATH,
                         "pp_ocrv6_medium_rec_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const std::array<std::int64_t, 3> normal_shape = {3, 48, 5};
  const std::vector<float> input(recognition_input_elements(5), 0.0F);
  std::vector<float> output(medium_recognition_output_elements(5));
  ASSERT_EQ(compiled.run_dynamic(input, normal_shape, output, output.size()),
            kSuccess);

  for (const auto dimensions : std::array<std::array<std::int64_t, 3>, 4>{
         {{3, 48, 0}, {3, 48, 4}, {3, 47, 5}, {1, 48, 5}}}) {
    const int expected_status =
      dimensions[2] == 4 ? kConstraintViolation : kInvalidShape;
    EXPECT_EQ(compiled.run_dynamic(input, dimensions, output, output.size()),
              expected_status);
  }

  std::vector<float> short_output(medium_recognition_output_elements(5) - 1);
  EXPECT_EQ(compiled.run_dynamic(
              input, normal_shape, short_output, short_output.size()),
            kOutputCapacityInsufficient);

  const std::array<std::int64_t, 3> overflow_shape = {
    3, 48, std::numeric_limits<std::int64_t>::max()};
  EXPECT_EQ(compiled.run_dynamic(input, overflow_shape, output, output.size()),
            kShapeArithmeticOverflow);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPOCRv6MediumRecArtifactsDescribeDynamicAbi) {
  EXPECT_GT(
    std::filesystem::file_size(PP_OCRV6_MEDIUM_REC_DYNAMIC_LIBRARY_PATH), 0U);

  const std::string manifest =
    read_text(PP_OCRV6_MEDIUM_REC_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv6_medium_rec_dynamic"), std::string::npos);
  EXPECT_NE(manifest.find("18710"), std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 5"), std::string::npos);
  EXPECT_NE(manifest.find("\"multiple_of\": 1"), std::string::npos);
  EXPECT_NE(manifest.find("\"shape_program\""), std::string::npos);

  const std::string header = read_text(PP_OCRV6_MEDIUM_REC_DYNAMIC_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_OCRV6_MEDIUM_REC_DYNAMIC_INPUT1_RANK 3",
         "#define PP_OCRV6_MEDIUM_REC_DYNAMIC_INPUT1_DIM2 NCNN_DYNAMIC_DIM",
         "#define PP_OCRV6_MEDIUM_REC_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x4)",
         "#define PP_OCRV6_MEDIUM_REC_DYNAMIC_INPUT1_DIM2_MINIMUM "
         "INT64_C(5)",
         "#define PP_OCRV6_MEDIUM_REC_DYNAMIC_OUTPUT1_RANK 2",
         "#define PP_OCRV6_MEDIUM_REC_DYNAMIC_OUTPUT1_DIM1 INT64_C(18710)",
         "pp_ocrv6_medium_rec_dynamic_infer_output_shapes",
         "uint64_t output1_capacity",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }

  const std::string linalg_ir =
    read_text(PP_OCRV6_MEDIUM_REC_DYNAMIC_LINALG_IR_PATH);
  EXPECT_EQ(linalg_ir.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalg_ir.find("math.exp"), std::string::npos);
  EXPECT_NE(linalg_ir.find("tensor.expand_shape"), std::string::npos);
}

TEST(NumericalDynamicModel, PPUVDocMatchesNcnnAcrossShapes) {
  CompiledModel compiled(PP_UVDOC_DYNAMIC_LIBRARY_PATH, "pp_uvdoc_dynamic");
  CompiledModel infer(PP_UVDOC_DYNAMIC_LIBRARY_PATH,
                      "pp_uvdoc_dynamic_infer_output_shapes");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  ASSERT_TRUE(infer.valid()) << infer.error();

  constexpr std::array<DetectionCase, 2> cases{{
    DetectionCase{.height = 12, .width = 16},
    DetectionCase{.height = 19, .width = 23},
  }};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const DetectionCase shape = cases[index];
    const auto runtime_shape = input_shape(shape);
    std::array<std::int64_t, 3> output_shape{};
    ASSERT_EQ(infer.infer_dynamic(runtime_shape, output_shape), kSuccess);
    EXPECT_EQ(output_shape,
              (std::array<std::int64_t, 3>{3, shape.height, shape.width}));

    const std::vector<float> input = make_random_input(
      element_count(shape), 0x5556444FU + index, -0.25F, 0.25F);
    const ReferenceModel reference(
      PP_UVDOC_DYNAMIC_PARAM_PATH,
      PP_UVDOC_DYNAMIC_BIN_PATH,
      "in0",
      "out0",
      TensorShape(
        static_cast<int>(shape.width), static_cast<int>(shape.height), 3));
    const auto expected = run_ncnn_reference(reference, input);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ(expected->size(), element_count(shape));
    std::vector<float> actual(expected->size());
    ASSERT_EQ(compiled.run_dynamic(input, runtime_shape, actual, actual.size()),
              kSuccess);
    EXPECT_TRUE(compare_values(actual, *expected, 3.0e-3F))
      << shape.height << 'x' << shape.width;
    EXPECT_TRUE(std::ranges::all_of(
      actual, [](float value) { return std::isfinite(value); }));
  }
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPUVDocSupportsAlternatingShapes) {
  CompiledModel compiled(PP_UVDOC_DYNAMIC_LIBRARY_PATH, "pp_uvdoc_dynamic");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  const DetectionCase first{.height = 10, .width = 14};
  const DetectionCase second{.height = 17, .width = 11};
  const std::vector<float> first_input =
    make_random_input(element_count(first), 0x55564131U, -0.2F, 0.2F);
  const std::vector<float> second_input =
    make_random_input(element_count(second), 0x55564132U, -0.2F, 0.2F);
  std::vector<float> first_output(element_count(first));
  std::vector<float> second_output(element_count(second));
  std::vector<float> repeated_output(element_count(first));
  ASSERT_EQ(
    compiled.run_dynamic(
      first_input, input_shape(first), first_output, first_output.size()),
    kSuccess);
  ASSERT_EQ(
    compiled.run_dynamic(
      second_input, input_shape(second), second_output, second_output.size()),
    kSuccess);
  ASSERT_EQ(
    compiled.run_dynamic(
      first_input, input_shape(first), repeated_output, repeated_output.size()),
    kSuccess);
  EXPECT_EQ(repeated_output, first_output);
  record_peak_rss();
}

TEST(NumericalDynamicModel, PPUVDocRejectsInvalidShapesAndCapacity) {
  CompiledModel compiled(PP_UVDOC_DYNAMIC_LIBRARY_PATH, "pp_uvdoc_dynamic");
  CompiledModel infer(PP_UVDOC_DYNAMIC_LIBRARY_PATH,
                      "pp_uvdoc_dynamic_infer_output_shapes");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  ASSERT_TRUE(infer.valid()) << infer.error();
  const DetectionCase valid{.height = 8, .width = 9};
  const std::vector<float> input(element_count(valid), 0.0F);
  std::vector<float> output(element_count(valid));
  const std::array<std::int64_t, 3> invalid_shape{3, 1, 9};
  std::array<std::int64_t, 3> output_shape{};
  EXPECT_EQ(infer.infer_dynamic(invalid_shape, output_shape),
            kConstraintViolation);
  EXPECT_EQ(compiled.run_dynamic(input, invalid_shape, output, output.size()),
            kConstraintViolation);
  output.pop_back();
  EXPECT_EQ(
    compiled.run_dynamic(input, input_shape(valid), output, output.size()),
    kOutputCapacityInsufficient);
}

TEST(NumericalDynamicModel, PPUVDocArtifactsDescribeDynamicAbi) {
  EXPECT_GT(std::filesystem::file_size(PP_UVDOC_DYNAMIC_LIBRARY_PATH), 0U);
  const std::string manifest = read_text(PP_UVDOC_DYNAMIC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_uvdoc_dynamic"), std::string::npos);
  EXPECT_NE(manifest.find("\"minimum\": 2"), std::string::npos);
  EXPECT_NE(manifest.find("\"shape_source_input\": 0"), std::string::npos);
  const std::string header = read_text(PP_UVDOC_DYNAMIC_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_UVDOC_DYNAMIC_INPUT1_DYNAMIC_DIM_MASK UINT32_C(0x6)",
         "#define PP_UVDOC_DYNAMIC_OUTPUT1_DYNAMIC_DIM_MASK UINT32_C(0x6)",
         "pp_uvdoc_dynamic_infer_output_shapes",
         "uint64_t output1_capacity",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }
  const std::string ncnn_ir = read_text(PP_UVDOC_DYNAMIC_NCNN_IR_PATH);
  EXPECT_NE(ncnn_ir.find("ncnn.grid_sample"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("padding_type = 2"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("negative_slope"), std::string::npos);
  const std::string linalg_ir = read_text(PP_UVDOC_DYNAMIC_LINALG_IR_PATH);
  EXPECT_EQ(linalg_ir.find("ncnn.grid_sample"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("ncnn.interp"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("ncnn.padding"), std::string::npos);
  EXPECT_NE(linalg_ir.find("math.floor"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_compiler::test
