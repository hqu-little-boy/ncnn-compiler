#include "numerical_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
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

std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

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

TEST(NumericalModel, PPOCRv5MobileRecMatchesNcnn) {
  const TensorShape inputShape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 18385;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x354D5245U, -1.0F, 1.0F);
  const ReferenceModel reference(PP_OCRV5_MOBILE_REC_PARAM_PATH,
                                 PP_OCRV5_MOBILE_REC_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kSequenceLength * kClasses);

  CompiledModel compiled(PP_OCRV5_MOBILE_REC_LIBRARY_PATH,
                         "pp_ocrv5_mobile_rec");
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
    EXPECT_TRUE(check_softmax(actualRow, expectedRow, 2.0e-4))
      << "row " << index;
  }
}

TEST(NumericalModel, PPOCRv5MobileRecArtifactsCoverAttentionPipeline) {
  EXPECT_GT(std::filesystem::file_size(PP_OCRV5_MOBILE_REC_LIBRARY_PATH), 0U);

  const std::string manifest = read_text(PP_OCRV5_MOBILE_REC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv5_mobile_rec"), std::string::npos);
  EXPECT_NE(manifest.find("18385"), std::string::npos);

  const std::string ncnnIr = read_text(PP_OCRV5_MOBILE_REC_NCNN_IR_PATH);
  EXPECT_NE(ncnnIr.find("ncnn.layer_norm"), std::string::npos);
  EXPECT_NE(ncnnIr.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(ncnnIr.find("num_heads = 8"), std::string::npos);
  EXPECT_NE(ncnnIr.find("scale = 0.258198"), std::string::npos);

  const std::string linalgIr = read_text(PP_OCRV5_MOBILE_REC_LINALG_IR_PATH);
  EXPECT_EQ(linalgIr.find("ncnn.layer_norm"), std::string::npos);
  EXPECT_EQ(linalgIr.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalgIr.find("linalg.batch_matmul"), std::string::npos);
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

TEST(NumericalModel, PPOCRv6TinyDetFp16StorageMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x46503644U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_OCRV6_TINY_DET_PARAM_PATH,
                                 PP_OCRV6_TINY_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(PP_OCRV6_TINY_DET_FP16_LIBRARY_PATH,
                         "pp_ocrv6_tiny_det_fp16");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 2.0e-2F));
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

  CompiledModel compiled(PP_OCRV5_SERVER_DET_STATIC_LIBRARY_PATH,
                         "pp_ocrv5_server_det_static");
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

TEST(NumericalModel, PPStructureV2SLANetPlusCNNMatchesNcnn) {
  const TensorShape inputShape(488, 488, 3);
  constexpr std::size_t kOutputElements = 256 * 96;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x534C414EU, -0.1F, 0.1F);
  const ReferenceModel reference(PP_STRUCTRUREV2_SLANET_PLUS_CNN_PARAM_PATH,
                                 PP_STRUCTRUREV2_SLANET_PLUS_CNN_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_STRUCTRUREV2_SLANET_PLUS_CNN_LIBRARY_PATH,
                         "pp_structrurev2_slanet_plus_cnn");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPStructureV2SLANetPlusCNNFp16StorageMatchesNcnn) {
  const TensorShape inputShape(488, 488, 3);
  constexpr std::size_t kOutputElements = 256 * 96;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x534C4631U, -0.1F, 0.1F);
  const ReferenceModel reference(PP_STRUCTRUREV2_SLANET_PLUS_CNN_PARAM_PATH,
                                 PP_STRUCTRUREV2_SLANET_PLUS_CNN_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(PP_STRUCTRUREV2_SLANET_PLUS_CNN_FP16_LIBRARY_PATH,
                         "pp_structrurev2_slanet_plus_cnn_fp16");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 4.0e-2F));
  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPFormulaNetPlusSEncoderMatchesNcnn) {
  const TensorShape inputShape(384, 384, 1);
  constexpr std::size_t kOutputElements = 144 * 2048;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x464F524DU, -0.01F, 0.01F);
  const ReferenceModel reference(PP_FORMULANET_PLUS_S_ENCODER_PARAM_PATH,
                                 PP_FORMULANET_PLUS_S_ENCODER_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_FORMULANET_PLUS_S_ENCODER_LIBRARY_PATH,
                         "pp_formulanet_plus_s_encoder");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(std::ranges::all_of(
    actual, [](float value) { return std::isfinite(value); }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPFormulaNetPlusSEncoderFp16StorageMatchesNcnn) {
  const TensorShape inputShape(384, 384, 1);
  constexpr std::size_t kOutputElements = 144 * 2048;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x464D4631U, -0.01F, 0.01F);
  const ReferenceModel reference(PP_FORMULANET_PLUS_S_ENCODER_PARAM_PATH,
                                 PP_FORMULANET_PLUS_S_ENCODER_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel compiled(PP_FORMULANET_PLUS_S_ENCODER_FP16_LIBRARY_PATH,
                         "pp_formulanet_plus_s_encoder_fp16");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 2.0e-2F));
  EXPECT_TRUE(std::ranges::all_of(
    actual, [](float value) { return std::isfinite(value); }));
  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, FormulaNetFp16ArtifactsPreserveStorageAndFp32Abi) {
  const auto baselineSize =
    std::filesystem::file_size(PP_FORMULANET_PLUS_S_ENCODER_LIBRARY_PATH);
  const auto fp16Size =
    std::filesystem::file_size(PP_FORMULANET_PLUS_S_ENCODER_FP16_LIBRARY_PATH);
  EXPECT_LT(fp16Size, baselineSize * 4 / 5);

  const std::string manifest =
    read_text(PP_FORMULANET_PLUS_S_ENCODER_FP16_MANIFEST_PATH);
  EXPECT_NE(manifest.find("\"element_type\": \"f32\""), std::string::npos);
  EXPECT_EQ(manifest.find("\"element_type\": \"f16\""), std::string::npos);

  const std::string ncnnIr =
    read_text(PP_FORMULANET_PLUS_S_ENCODER_FP16_NCNN_IR_PATH);
  EXPECT_NE(ncnnIr.find("xf16>"), std::string::npos);
  EXPECT_NE(ncnnIr.find("ncnn.precision = \"fp16\""), std::string::npos);
  const std::string linalgIr =
    read_text(PP_FORMULANET_PLUS_S_ENCODER_FP16_LINALG_IR_PATH);
  EXPECT_NE(linalgIr.find("arith.extf"), std::string::npos);
  EXPECT_NE(linalgIr.find("arith.truncf"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_compiler::test
