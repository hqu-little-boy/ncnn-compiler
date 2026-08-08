#include "numerical_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

const TensorShape kInputShape(224, 224, 3);
constexpr std::size_t kOutputElements = 4;
const ReferenceModel kReference(PP_LCNET_DOC_ORI_PARAM_PATH,
                                PP_LCNET_DOC_ORI_BIN_PATH,
                                "in0",
                                "out0",
                                kInputShape);

TEST(NumericalModel, PPLCNetDocOriMatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x4C434E45U, -1.0F, 1.0F);
  const auto expected = run_ncnn_reference(kReference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_LCNET_DOC_ORI_LIBRARY_PATH,
                         "pp_lcnet_x1_0_doc_ori");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_NEAR(
    std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1.0e-5F);
  EXPECT_TRUE(check_softmax(actual, *expected));
}

TEST(NumericalModel, PPLCNetTextlineOriMatchesNcnn) {
  const TensorShape input_shape(160, 80, 3);
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x54455854U, -1.0F, 1.0F);
  const ReferenceModel reference(PP_LCNET_TEXTLINE_ORI_PARAM_PATH,
                                 PP_LCNET_TEXTLINE_ORI_BIN_PATH,
                                 "in0",
                                 "out0",
                                 input_shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 2U);

  CompiledModel compiled(PP_LCNET_TEXTLINE_ORI_LIBRARY_PATH,
                         "pp_lcnet_x1_0_textline_ori");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(2);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_NEAR(
    std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1.0e-5F);
  EXPECT_TRUE(check_softmax(actual, *expected));
}

TEST(NumericalModel, ChineseOCRLiteAngleNetMatchesNcnn) {
  const TensorShape inputShape(192, 32, 3);
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x414E474CU, -1.0F, 1.0F);
  const ReferenceModel reference(CHINESEOCR_LITE_ANGLENET_PARAM_PATH,
                                 CHINESEOCR_LITE_ANGLENET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 2U);

  CompiledModel compiled(CHINESEOCR_LITE_ANGLENET_LIBRARY_PATH,
                         "chineseocr_lite_anglenet");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(2);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_NEAR(
    std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1.0e-5F);
  EXPECT_TRUE(check_softmax(actual, *expected));
}

TEST(NumericalModel, PPOCRv6TinyRecMatchesNcnn) {
  const TensorShape inputShape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 6906;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x4F435236U, -1.0F, 1.0F);
  const ReferenceModel reference(PP_OCRV6_TINY_REC_PARAM_PATH,
                                 PP_OCRV6_TINY_REC_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kSequenceLength * kClasses);

  CompiledModel compiled(PP_OCRV6_TINY_REC_LIBRARY_PATH, "pp_ocrv6_tiny_rec");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(expected->size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  for (std::size_t index = 0; index < kSequenceLength; ++index) {
    const std::span<const float> actualRow(actual.data() + (index * kClasses),
                                           kClasses);
    const std::span<const float> expectedRow(
      expected->data() + (index * kClasses), kClasses);
    EXPECT_TRUE(check_softmax(actualRow, expectedRow, 2.0e-5))
      << "row " << index;
  }
}

TEST(NumericalModel, PPOCRv6TinyDetMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x44455436U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_OCRV6_TINY_DET_PARAM_PATH,
                                 PP_OCRV6_TINY_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_OCRV6_TINY_DET_LIBRARY_PATH, "pp_ocrv6_tiny_det");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv6SmallDetMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  constexpr float kTolerance = 3.0e-4F;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x53444554U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_OCRV6_SMALL_DET_PARAM_PATH,
                                 PP_OCRV6_SMALL_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_OCRV6_SMALL_DET_LIBRARY_PATH, "pp_ocrv6_small_det");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, kTolerance));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv6MediumDetMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x4D444554U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_OCRV6_MEDIUM_DET_PARAM_PATH,
                                 PP_OCRV6_MEDIUM_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_OCRV6_MEDIUM_DET_LIBRARY_PATH,
                         "pp_ocrv6_medium_det");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv5MobileDetMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x354D4445U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_OCRV5_MOBILE_DET_PARAM_PATH,
                                 PP_OCRV5_MOBILE_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_OCRV5_MOBILE_DET_LIBRARY_PATH,
                         "pp_ocrv5_mobile_det");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv5ServerDetMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x35534445U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_OCRV5_SERVER_DET_PARAM_PATH,
                                 PP_OCRV5_SERVER_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_OCRV5_SERVER_DET_LIBRARY_PATH,
                         "pp_ocrv5_server_det");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

}  // namespace
}  // namespace ncnn_compiler::test
