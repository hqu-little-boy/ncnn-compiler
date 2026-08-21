#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <numeric>
#include <ranges>
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

constexpr std::size_t kFormulaMemoryElements = 144 * 2048;
constexpr std::size_t kFormulaEmbeddingElements = 384;
constexpr std::size_t kFormulaCacheStride = 16 * 24;
constexpr std::size_t kFormulaClasses = 50000;
constexpr std::array<std::string_view, 5> kFormulaDecoderOutputs{
  "out_cache_k0", "out_cache_v0", "out_cache_k1", "out_cache_v1", "out0"};

struct FormulaDecoderResult {
  std::array<std::vector<float>, 4> caches;
  std::vector<float> logits;
};

FormulaDecoderResult make_decoder_result(std::size_t sequence_length) {
  FormulaDecoderResult result;
  for (auto& cache : result.caches) {
    cache.resize(sequence_length * kFormulaCacheStride);
  }
  result.logits.resize(kFormulaClasses);
  return result;
}

std::expected<std::vector<std::vector<float>>, std::string>
run_formula_decoder_reference(std::span<const float> memory,
                              std::span<const float> embedding,
                              std::span<const float> mask,
                              const std::array<std::vector<float>, 4>& caches,
                              int cache_length) {
  const std::array inputs{
    ReferenceInput("in0", TensorShape(2048, 144), memory),
    ReferenceInput("in1", TensorShape(384, 1), embedding),
    ReferenceInput("in2", TensorShape(cache_length + 1, 1), mask),
    ReferenceInput("cache_k0", TensorShape(24, cache_length, 16), caches[0]),
    ReferenceInput("cache_v0", TensorShape(24, cache_length, 16), caches[1]),
    ReferenceInput("cache_k1", TensorShape(24, cache_length, 16), caches[2]),
    ReferenceInput("cache_v1", TensorShape(24, cache_length, 16), caches[3]),
  };
  return run_ncnn_reference(PP_FORMULANET_PLUS_S_DECODER_PARAM_PATH,
                            PP_FORMULANET_PLUS_S_DECODER_BIN_PATH,
                            inputs,
                            kFormulaDecoderOutputs);
}

int run_formula_decoder_compiled(
  const CompiledModel& compiled,
  std::span<const float> memory,
  std::span<const float> embedding,
  std::span<const float> mask,
  const std::array<std::vector<float>, 4>& caches,
  int cache_length,
  FormulaDecoderResult& result) {
  const std::array<std::int64_t, 2> mask_shape{1, cache_length + 1};
  const std::array<std::int64_t, 3> cache_shape{16, cache_length, 24};
  return compiled.run_formula_decoder(memory,
                                      embedding,
                                      mask,
                                      mask_shape,
                                      caches[0],
                                      cache_shape,
                                      caches[1],
                                      cache_shape,
                                      caches[2],
                                      cache_shape,
                                      caches[3],
                                      cache_shape,
                                      result.caches[0],
                                      result.caches[1],
                                      result.caches[2],
                                      result.caches[3],
                                      result.logits);
}

std::string without_whitespace(std::string text) {
  std::erase_if(text,
                [](unsigned char value) { return std::isspace(value) != 0; });
  return text;
}

std::size_t count_substring(std::string_view text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t offset = 0;
       (offset = text.find(needle, offset)) != std::string_view::npos;
       offset += needle.size()) {
    ++count;
  }
  return count;
}

TEST(NumericalModel, PPLCNetDocOriMatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x4C434E45U);
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

TEST(NumericalModel, PPLCNetDocOriInt8ProducesStableSoftmax) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x4C434938U);
  CompiledModel compiled(PP_LCNET_DOC_ORI_INT8_LIBRARY_PATH,
                         "pp_lcnet_x1_0_doc_ori_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));
  EXPECT_NEAR(
    std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1.0e-5F);
}

TEST(NumericalModel, PPLCNetTextlineOriMatchesNcnn) {
  const TensorShape input_shape(160, 80, 3);
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x54455854U);
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

TEST(NumericalModel, PPLCNetTextlineOriInt8ProducesStableSoftmax) {
  const TensorShape input_shape(160, 80, 3);
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x544C4938U);
  CompiledModel compiled(PP_LCNET_TEXTLINE_ORI_INT8_LIBRARY_PATH,
                         "pp_lcnet_x1_0_textline_ori_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(2);
  std::vector<float> repeated(2);
  ASSERT_EQ(compiled.run(input, actual), 0);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));
  EXPECT_NEAR(
    std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1.0e-5F);
}

TEST(NumericalModel, PPLCNetTextlineOriInt8PreservesIntegerTargetCode) {
  const std::string llvm_ir =
    read_text(PP_LCNET_TEXTLINE_ORI_INT8_LLVM_IR_PATH);
  EXPECT_GE(count_substring(llvm_ir, "sext i8"), 20U);
  EXPECT_GE(count_substring(llvm_ir, "mul i32"), 20U);
  EXPECT_GE(count_substring(llvm_ir, "fptosi float"), 20U);
  EXPECT_GE(count_substring(llvm_ir, "sitofp i32"), 20U);

  const std::string assembly =
    read_text(PP_LCNET_TEXTLINE_ORI_INT8_ASSEMBLY_PATH);
  EXPECT_GE(count_substring(assembly, "movsbl"), 20U);
  EXPECT_GE(count_substring(assembly, "imull"), 20U);
}

TEST(NumericalModel, ChineseOCRLiteAngleNetMatchesNcnn) {
  const TensorShape inputShape(192, 32, 3);
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x414E474CU);
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
    make_random_input(*inputElements, 0x4F435236U);
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
    EXPECT_TRUE(check_softmax(actualRow, expectedRow, 2.0e-4))
      << "row " << index;
  }
}

TEST(NumericalModel, PPOCRv6TinyRecInt8ProducesStableSoftmax) {
  const TensorShape inputShape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 6906;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x36544938U);
  CompiledModel compiled(PP_OCRV6_TINY_REC_INT8_LIBRARY_PATH,
                         "pp_ocrv6_tiny_rec_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kSequenceLength * kClasses);
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  for (std::size_t index = 0; index < kSequenceLength; ++index) {
    const std::size_t offset = index * kClasses;
    const auto row = std::span<const float>(actual).subspan(offset, kClasses);
    EXPECT_TRUE(std::ranges::all_of(row,
                                    [](float value) {
                                      return std::isfinite(value) &&
                                             value >= 0.0F && value <= 1.0F;
                                    }))
      << "row " << index;
    EXPECT_NEAR(std::accumulate(row.begin(), row.end(), 0.0F), 1.0F, 2.0e-4F)
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
    make_random_input(*inputElements, 0x354D5245U);
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
  EXPECT_TRUE(compare_values(actual, *expected, 5.0e-4F, 2.0e-5F));
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

TEST(NumericalModel, PPOCRv5MobileRecInt8ProducesStableSoftmax) {
  const TensorShape inputShape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 18385;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x354D4938U);
  CompiledModel compiled(PP_OCRV5_MOBILE_REC_INT8_LIBRARY_PATH,
                         "pp_ocrv5_mobile_rec_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kSequenceLength * kClasses);
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  for (std::size_t index = 0; index < kSequenceLength; ++index) {
    const auto row =
      std::span<const float>(actual).subspan(index * kClasses, kClasses);
    EXPECT_TRUE(std::ranges::all_of(row,
                                    [](float value) {
                                      return std::isfinite(value) &&
                                             value >= 0.0F && value <= 1.0F;
                                    }))
      << "row " << index;
    EXPECT_NEAR(std::accumulate(row.begin(), row.end(), 0.0F), 1.0F, 3.0e-4F)
      << "row " << index;
  }
}

TEST(NumericalModel, PPOCRv6MediumRecInt8ProducesStableSoftmax) {
  const TensorShape input_shape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 18710;
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x364D4938U);
  CompiledModel compiled(PP_OCRV6_MEDIUM_REC_INT8_LIBRARY_PATH,
                         "pp_ocrv6_medium_rec_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kSequenceLength * kClasses);
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  for (std::size_t index = 0; index < kSequenceLength; ++index) {
    const auto row =
      std::span<const float>(actual).subspan(index * kClasses, kClasses);
    EXPECT_TRUE(std::ranges::all_of(row, [](float value) {
      return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    }));
    EXPECT_NEAR(std::accumulate(row.begin(), row.end(), 0.0F), 1.0F, 1.0e-3F);
  }
}

TEST(NumericalModel, PPOCRv6SmallRecProducesStableSoftmax) {
  const TensorShape input_shape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 18710;
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x36534D45U);
  const ReferenceModel reference(PP_OCRV6_SMALL_REC_PARAM_PATH,
                                 PP_OCRV6_SMALL_REC_BIN_PATH,
                                 "in0",
                                 "out0",
                                 input_shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  CompiledModel compiled(PP_OCRV6_SMALL_REC_LIBRARY_PATH, "pp_ocrv6_small_rec");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kSequenceLength * kClasses);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 5.0e-4F, 5.0e-5F));
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv5MobileDetInt8ProducesStableOutput) {
  const TensorShape input_shape(32, 32, 3);
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x35444938U);
  CompiledModel compiled(PP_OCRV5_MOBILE_DET_INT8_LIBRARY_PATH,
                         "pp_ocrv5_mobile_det_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(32 * 32);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv5ServerRecMatchesNcnn) {
  const TensorShape input_shape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 18385;
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x35535245U);
  const ReferenceModel reference(PP_OCRV5_SERVER_REC_PARAM_PATH,
                                 PP_OCRV5_SERVER_REC_BIN_PATH,
                                 "in0",
                                 "out0",
                                 input_shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kSequenceLength * kClasses);

  CompiledModel compiled(PP_OCRV5_SERVER_REC_LIBRARY_PATH,
                         "pp_ocrv5_server_rec");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(expected->size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 5.0e-4F, 2.0e-5F));
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  for (std::size_t index = 0; index < kSequenceLength; ++index) {
    const std::span<const float> actual_row(actual.data() + (index * kClasses),
                                            kClasses);
    const std::span<const float> expected_row(
      expected->data() + (index * kClasses), kClasses);
    EXPECT_TRUE(check_softmax(actual_row, expected_row, 2.0e-4))
      << "row " << index;
  }
}

TEST(NumericalModel, PPOCRv5ServerRecArtifactsCoverAttentionPipeline) {
  EXPECT_GT(std::filesystem::file_size(PP_OCRV5_SERVER_REC_LIBRARY_PATH), 0U);

  const std::string manifest = read_text(PP_OCRV5_SERVER_REC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv5_server_rec"), std::string::npos);
  EXPECT_NE(manifest.find("18385"), std::string::npos);

  const std::string ncnn_ir = read_text(PP_OCRV5_SERVER_REC_NCNN_IR_PATH);
  EXPECT_NE(ncnn_ir.find("ncnn.layer_norm"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("num_heads = 8"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("scale = 0.258198"), std::string::npos);

  const std::string linalg_ir = read_text(PP_OCRV5_SERVER_REC_LINALG_IR_PATH);
  EXPECT_EQ(linalg_ir.find("ncnn.layer_norm"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalg_ir.find("linalg.batch_matmul"), std::string::npos);
}

TEST(NumericalModel, PPOCRv6MediumRecMatchesNcnn) {
  const TensorShape input_shape(320, 48, 3);
  constexpr std::size_t kSequenceLength = 40;
  constexpr std::size_t kClasses = 18710;
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, 0x364D5245U);
  const ReferenceModel reference(PP_OCRV6_MEDIUM_REC_PARAM_PATH,
                                 PP_OCRV6_MEDIUM_REC_BIN_PATH,
                                 "in0",
                                 "out0",
                                 input_shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kSequenceLength * kClasses);

  CompiledModel compiled(PP_OCRV6_MEDIUM_REC_LIBRARY_PATH,
                         "pp_ocrv6_medium_rec");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(expected->size());
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 3.0e-4F, 2.0e-4F));
  std::vector<float> repeated(actual.size());
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
  for (std::size_t index = 0; index < kSequenceLength; ++index) {
    const std::span<const float> actual_row(actual.data() + (index * kClasses),
                                            kClasses);
    const std::span<const float> expected_row(
      expected->data() + (index * kClasses), kClasses);
    EXPECT_TRUE(check_softmax(actual_row, expected_row, 1.0e-3))
      << "row " << index;
  }
}

TEST(NumericalModel, PPOCRv6MediumRecArtifactsCoverAttentionPipeline) {
  EXPECT_GT(std::filesystem::file_size(PP_OCRV6_MEDIUM_REC_LIBRARY_PATH), 0U);

  const std::string manifest = read_text(PP_OCRV6_MEDIUM_REC_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_ocrv6_medium_rec"), std::string::npos);
  EXPECT_NE(manifest.find("18710"), std::string::npos);

  const std::string ncnn_ir = read_text(PP_OCRV6_MEDIUM_REC_NCNN_IR_PATH);
  EXPECT_NE(ncnn_ir.find("ncnn.layer_norm"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("num_heads = 8"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("scale = 0.204124"), std::string::npos);

  const std::string linalg_ir = read_text(PP_OCRV6_MEDIUM_REC_LINALG_IR_PATH);
  EXPECT_EQ(linalg_ir.find("ncnn.layer_norm"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("ncnn.multi_head_attention"), std::string::npos);
  EXPECT_NE(linalg_ir.find("linalg.batch_matmul"), std::string::npos);
}

TEST(NumericalModel, PPOCRv6TinyDetMatchesNcnn) {
  const TensorShape inputShape(640, 640, 3);
  constexpr std::size_t kOutputElements = 640 * 640;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x44455436U);
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
  EXPECT_TRUE(compare_values(actual, *expected, 2.0e-3F, 2.0e-3F));
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv6TinyDetInt8ProducesStableOutput) {
  const TensorShape inputShape(32, 32, 3);
  constexpr std::size_t kOutputElements = 32 * 32;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x49384454U);
  CompiledModel compiled(PP_OCRV6_TINY_DET_INT8_LIBRARY_PATH,
                         "pp_ocrv6_tiny_det_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
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
    make_random_input(*inputElements, 0x46503644U);
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
  EXPECT_TRUE(compare_values(actual, *expected, 5.0e-2F, 5.0e-2F));
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
    make_random_input(*inputElements, 0x53444554U);
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

TEST(NumericalModel, PPOCRv6SmallDetInt8ProducesStableOutput) {
  const TensorShape inputShape(32, 32, 3);
  constexpr std::size_t kOutputElements = 32 * 32;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x49385344U);
  CompiledModel compiled(PP_OCRV6_SMALL_DET_INT8_LIBRARY_PATH,
                         "pp_ocrv6_small_det_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(std::ranges::all_of(actual, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPOCRv6MediumDetInt8ProducesStableOutput) {
  const TensorShape inputShape(32, 32, 3);
  constexpr std::size_t kOutputElements = 32 * 32;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x49384D44U);
  CompiledModel compiled(PP_OCRV6_MEDIUM_DET_INT8_LIBRARY_PATH,
                         "pp_ocrv6_medium_det_int8");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
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
    make_random_input(*inputElements, 0x4D444554U);
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
    make_random_input(*inputElements, 0x354D4445U);
  const ReferenceModel reference(PP_OCRV5_MOBILE_DET_PARAM_PATH,
                                 PP_OCRV5_MOBILE_DET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 inputShape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(PP_OCRV5_MOBILE_DET_LIBRARY_PATH,
                         "pp_ocrv5_mobile_det_static");
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
    make_random_input(*inputElements, 0x35534445U);
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
    make_random_input(*inputElements, 0x534C414EU);
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

TEST(NumericalModel, PPStructureV2SLANetPlusSLAHeadMatchesNcnn) {
  std::vector<float> hidden = make_random_input(256, 0x534C4844U);
  const std::vector<float> memory = make_random_input(256 * 96, 0x534C4D45U);
  const std::vector<float> token = make_random_input(50, 0x534C544BU);
  const std::array inputs{
    ReferenceInput("in0", TensorShape(256, 1), hidden),
    ReferenceInput("in1", TensorShape(96, 256), memory),
    ReferenceInput("in2", TensorShape(50, 1), token),
  };
  constexpr std::array<std::string_view, 3> kOutputs = {"out0", "out1", "out2"};
  const auto expected =
    run_ncnn_reference(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_PARAM_PATH,
                       PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_BIN_PATH,
                       inputs,
                       kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ((*expected)[0].size(), 256U);
  ASSERT_EQ((*expected)[1].size(), 50U);
  ASSERT_EQ((*expected)[2].size(), 8U);

  CompiledModel compiled(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_LIBRARY_PATH,
                         "pp_structrurev2_slanet_plus_slahead");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual_hidden(256);
  std::vector<float> actual_token(50);
  std::vector<float> actual_location(8);
  ASSERT_EQ(
    compiled.run_three_inputs_three_outputs(
      hidden, memory, token, actual_hidden, actual_token, actual_location),
    0);
  EXPECT_TRUE(compare_values(actual_hidden, (*expected)[0], 1.0e-4F));
  EXPECT_TRUE(compare_values(actual_token, (*expected)[1], 1.0e-4F));
  EXPECT_TRUE(compare_values(actual_location, (*expected)[2], 1.0e-4F));
  EXPECT_TRUE(check_softmax(actual_token, (*expected)[1]));
}

TEST(NumericalModel, PPStructureV2SLANetCNNAndHeadDecodeMultipleSteps) {
  const TensorShape imageShape(488, 488, 3);
  const auto imageElements = imageShape.element_count();
  ASSERT_TRUE(imageElements.has_value()) << imageElements.error();
  const std::vector<float> image =
    make_random_input(*imageElements, 0x534C4443U);
  CompiledModel cnn(PP_STRUCTRUREV2_SLANET_PLUS_CNN_LIBRARY_PATH,
                    "pp_structrurev2_slanet_plus_cnn");
  CompiledModel head(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_LIBRARY_PATH,
                     "pp_structrurev2_slanet_plus_slahead");
  ASSERT_TRUE(cnn.valid()) << cnn.error();
  ASSERT_TRUE(head.valid()) << head.error();

  std::vector<float> memory(256 * 96);
  ASSERT_EQ(cnn.run(image, memory), 0);
  std::vector<float> actualHidden(256, 0.0F);
  std::vector<float> expectedHidden(256, 0.0F);
  std::vector<float> token(50, 0.0F);
  token[0] = 1.0F;
  constexpr std::array<std::string_view, 3> kOutputs = {"out0", "out1", "out2"};
  for (int step = 0; step < 3; ++step) {
    const std::array inputs{
      ReferenceInput("in0", TensorShape(256, 1), expectedHidden),
      ReferenceInput("in1", TensorShape(96, 256), memory),
      ReferenceInput("in2", TensorShape(50, 1), token),
    };
    const auto expected =
      run_ncnn_reference(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_PARAM_PATH,
                         PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_BIN_PATH,
                         inputs,
                         kOutputs);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    std::vector<float> nextHidden(256);
    std::vector<float> structure(50);
    std::vector<float> location(8);
    ASSERT_EQ(head.run_three_inputs_three_outputs(
                actualHidden, memory, token, nextHidden, structure, location),
              0);
    EXPECT_TRUE(compare_values(nextHidden, (*expected)[0], 2.0e-4F));
    EXPECT_TRUE(compare_values(structure, (*expected)[1], 2.0e-4F));
    EXPECT_TRUE(compare_values(location, (*expected)[2], 2.0e-4F));
    actualHidden = std::move(nextHidden);
    expectedHidden = (*expected)[0];
    const auto nextToken = static_cast<std::size_t>(std::distance(
      (*expected)[1].begin(), std::ranges::max_element((*expected)[1])));
    std::ranges::fill(token, 0.0F);
    token[nextToken] = 1.0F;
  }
}

TEST(NumericalModel, PPStructureV2SLANetSLAHeadArtifactsRemainStatic) {
  const std::string manifest =
    read_text(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_structrurev2_slanet_plus_slahead"),
            std::string::npos);
  EXPECT_EQ(manifest.find("-1"), std::string::npos);

  const std::string header =
    read_text(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_INPUT1_DIM0 INT64_C(1)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_INPUT1_DIM1 INT64_C(256)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_INPUT2_DIM0 INT64_C(256)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_INPUT2_DIM1 INT64_C(96)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_INPUT3_DIM1 INT64_C(50)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_OUTPUT1_DIM1 "
         "INT64_C(256)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_OUTPUT2_DIM0 INT64_C(50)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_OUTPUT3_DIM0 INT64_C(8)",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }
  EXPECT_EQ(header.find("_infer_output_shapes"), std::string::npos);

  const std::string linalgIr =
    read_text(PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_LINALG_IR_PATH);
  EXPECT_NE(linalgIr.find("tensor<256x96xf32>"), std::string::npos);
  EXPECT_NE(linalgIr.find("tensor<1x256xf32>"), std::string::npos);
  EXPECT_EQ(linalgIr.find("?"), std::string::npos);
}

TEST(NumericalModel, PPStructureV2SLANetArtifactsRemainStaticSpecialization) {
  const std::string manifest =
    read_text(PP_STRUCTRUREV2_SLANET_PLUS_CNN_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_structrurev2_slanet_plus_cnn"),
            std::string::npos);
  EXPECT_NE(manifest.find("\"dynamic_dim_mask\": 0"), std::string::npos);
  EXPECT_EQ(manifest.find("-1"), std::string::npos);

  const std::string header =
    read_text(PP_STRUCTRUREV2_SLANET_PLUS_CNN_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_STRUCTRUREV2_SLANET_PLUS_CNN_INPUT1_DIM0 INT64_C(3)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_CNN_INPUT1_DIM1 INT64_C(488)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_CNN_INPUT1_DIM2 INT64_C(488)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_CNN_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x0)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_CNN_OUTPUT1_DIM0 "
         "INT64_C(256)",
         "#define PP_STRUCTRUREV2_SLANET_PLUS_CNN_OUTPUT1_DIM1 INT64_C(96)",
         "int pp_structrurev2_slanet_plus_cnn(const float *input1, float "
         "*output1);",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }
  EXPECT_EQ(header.find("input1_shape"), std::string::npos);
  EXPECT_EQ(header.find("_infer_output_shapes"), std::string::npos);

  const std::string linalg_ir =
    read_text(PP_STRUCTRUREV2_SLANET_PLUS_CNN_LINALG_IR_PATH);
  EXPECT_NE(linalg_ir.find("tensor<3x488x488xf32>"), std::string::npos);
  EXPECT_NE(linalg_ir.find("tensor<256x96xf32>"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("?"), std::string::npos);
}

TEST(NumericalModel, PPStructureV2SLANetPlusCNNFp16StorageMatchesNcnn) {
  const TensorShape inputShape(488, 488, 3);
  constexpr std::size_t kOutputElements = 256 * 96;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x534C4631U);
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
  EXPECT_TRUE(compare_values(actual, *expected, 4.0e-2F, 4.0e-2F));
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
    make_random_input(*inputElements, 0x464F524DU);
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
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F, 5.0e-5F));
  EXPECT_TRUE(std::ranges::all_of(
    actual, [](float value) { return std::isfinite(value); }));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_EQ(repeated, actual);
}

TEST(NumericalModel, PPFormulaNetPlusSEmbedMatchesNcnnAndClampsIndices) {
  CompiledModel compiled(PP_FORMULANET_PLUS_S_EMBED_LIBRARY_PATH,
                         "pp_formulanet_plus_s_embed");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  constexpr std::array<std::pair<std::int32_t, std::int32_t>, 5> kCases{{
    {1, 0},
    {0, 0},
    {49999, 1028},
    {-7, -3},
    {50017, 2048},
  }};
  constexpr std::array<std::string_view, 1> kOutputs{"out0"};
  for (const auto& [tokenIndex, positionIndex] : kCases) {
    const std::array<std::int32_t, 1> token{tokenIndex};
    const std::array<std::int32_t, 1> position{positionIndex};
    const std::array inputs{
      ReferenceInput(
        "in0", TensorShape(1), std::span<const std::int32_t>(token)),
      ReferenceInput(
        "in1", TensorShape(1), std::span<const std::int32_t>(position)),
    };
    const auto expected =
      run_ncnn_reference(PP_FORMULANET_PLUS_S_EMBED_PARAM_PATH,
                         PP_FORMULANET_PLUS_S_EMBED_BIN_PATH,
                         inputs,
                         kOutputs);
    ASSERT_TRUE(expected.has_value()) << expected.error();
    ASSERT_EQ((*expected)[0].size(), 384U);

    std::vector<float> actual(384);
    ASSERT_EQ(compiled.run_two_integer_inputs(token, position, actual), 0);
    EXPECT_TRUE(compare_values(actual, (*expected)[0], 1.0e-4F))
      << "token=" << tokenIndex << ", position=" << positionIndex;
    EXPECT_TRUE(std::ranges::all_of(
      actual, [](float value) { return std::isfinite(value); }));
    std::vector<float> repeated(384);
    ASSERT_EQ(compiled.run_two_integer_inputs(token, position, repeated), 0);
    EXPECT_EQ(repeated, actual);
  }
}

TEST(NumericalModel, PPFormulaNetPlusSEmbedArtifactsUseStaticIntegerAbi) {
  const std::string manifest =
    read_text(PP_FORMULANET_PLUS_S_EMBED_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_formulanet_plus_s_embed"), std::string::npos);
  EXPECT_EQ(std::ranges::count(manifest, '?'), 0);
  EXPECT_EQ(manifest.find("-1"), std::string::npos);
  EXPECT_NE(manifest.find("\"element_type\": \"i32\""),
            manifest.rfind("\"element_type\": \"i32\""));
  EXPECT_NE(manifest.find("\"element_type\": \"f32\""), std::string::npos);

  const std::string header = read_text(PP_FORMULANET_PLUS_S_EMBED_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_FORMULANET_PLUS_S_EMBED_INPUT1_DIM0 INT64_C(1)",
         "#define PP_FORMULANET_PLUS_S_EMBED_INPUT2_DIM0 INT64_C(1)",
         "#define PP_FORMULANET_PLUS_S_EMBED_OUTPUT1_DIM0 INT64_C(1)",
         "#define PP_FORMULANET_PLUS_S_EMBED_OUTPUT1_DIM1 INT64_C(384)",
         "int pp_formulanet_plus_s_embed(const int32_t *input1, const int32_t "
         "*input2, float *output1);",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }
  EXPECT_EQ(header.find("input1_shape"), std::string::npos);
  EXPECT_EQ(header.find("_infer_output_shapes"), std::string::npos);
}

TEST(NumericalModel, PPFormulaNetArtifactsRemainStaticSpecialization) {
  const std::string manifest =
    read_text(PP_FORMULANET_PLUS_S_ENCODER_MANIFEST_PATH);
  EXPECT_NE(manifest.find("pp_formulanet_plus_s_encoder"), std::string::npos);
  EXPECT_NE(manifest.find("\"dynamic_dim_mask\": 0"), std::string::npos);
  EXPECT_EQ(manifest.find("-1"), std::string::npos);

  const std::string header =
    read_text(PP_FORMULANET_PLUS_S_ENCODER_HEADER_PATH);
  for (const std::string_view required : {
         "#define PP_FORMULANET_PLUS_S_ENCODER_INPUT1_DIM0 INT64_C(1)",
         "#define PP_FORMULANET_PLUS_S_ENCODER_INPUT1_DIM1 INT64_C(384)",
         "#define PP_FORMULANET_PLUS_S_ENCODER_INPUT1_DIM2 INT64_C(384)",
         "#define PP_FORMULANET_PLUS_S_ENCODER_INPUT1_DYNAMIC_DIM_MASK "
         "UINT32_C(0x0)",
         "#define PP_FORMULANET_PLUS_S_ENCODER_OUTPUT1_DIM0 INT64_C(144)",
         "#define PP_FORMULANET_PLUS_S_ENCODER_OUTPUT1_DIM1 INT64_C(2048)",
         "int pp_formulanet_plus_s_encoder(const float *input1, float "
         "*output1);",
       }) {
    EXPECT_NE(header.find(required), std::string::npos) << required;
  }
  EXPECT_EQ(header.find("input1_shape"), std::string::npos);
  EXPECT_EQ(header.find("_infer_output_shapes"), std::string::npos);

  const std::string linalg_ir =
    read_text(PP_FORMULANET_PLUS_S_ENCODER_LINALG_IR_PATH);
  EXPECT_NE(linalg_ir.find("tensor<1x384x384xf32>"), std::string::npos);
  EXPECT_NE(linalg_ir.find("tensor<144x2048xf32>"), std::string::npos);
  EXPECT_EQ(linalg_ir.find("?"), std::string::npos);
}

TEST(NumericalModel, PPFormulaNetPlusSDecoderSingleStepMatchesNcnn) {
  const std::vector<float> memory =
    make_random_input(kFormulaMemoryElements, 0x4445434FU);
  const std::vector<float> embedding =
    make_random_input(kFormulaEmbeddingElements, 0x454D4244U);
  const std::vector<float> mask{-8.0F, 0.0F};
  std::array<std::vector<float>, 4> caches;
  for (std::size_t index = 0; index < caches.size(); ++index) {
    caches[index] = make_random_input(kFormulaCacheStride, 0x43414348U + index);
  }

  CompiledModel infer(PP_FORMULANET_PLUS_S_DECODER_LIBRARY_PATH,
                      "pp_formulanet_plus_s_decoder_infer_output_shapes");
  ASSERT_TRUE(infer.valid()) << infer.error();
  const std::array<std::int64_t, 2> mask_shape{1, 2};
  const std::array<std::int64_t, 3> cache_shape{16, 1, 24};
  std::array<std::array<std::int64_t, 3>, 4> inferred{};
  ASSERT_EQ(infer.infer_formula_decoder(mask_shape,
                                        cache_shape,
                                        cache_shape,
                                        cache_shape,
                                        cache_shape,
                                        inferred[0],
                                        inferred[1],
                                        inferred[2],
                                        inferred[3]),
            0);
  for (const auto& shape : inferred) {
    EXPECT_EQ(shape, (std::array<std::int64_t, 3>{16, 2, 24}));
  }

  CompiledModel compiled(PP_FORMULANET_PLUS_S_DECODER_LIBRARY_PATH,
                         "pp_formulanet_plus_s_decoder");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  FormulaDecoderResult actual = make_decoder_result(2);
  ASSERT_EQ(run_formula_decoder_compiled(
              compiled, memory, embedding, mask, caches, 1, actual),
            0);

  const std::vector<float> mismatched_mask(3, 0.0F);
  const std::array<std::int64_t, 2> mismatched_mask_shape{1, 3};
  FormulaDecoderResult rejected = make_decoder_result(2);
  for (auto& cache : rejected.caches) {
    std::ranges::fill(cache, 123.0F);
  }
  std::ranges::fill(rejected.logits, 123.0F);
  EXPECT_EQ(compiled.run_formula_decoder(memory,
                                         embedding,
                                         mismatched_mask,
                                         mismatched_mask_shape,
                                         caches[0],
                                         cache_shape,
                                         caches[1],
                                         cache_shape,
                                         caches[2],
                                         cache_shape,
                                         caches[3],
                                         cache_shape,
                                         rejected.caches[0],
                                         rejected.caches[1],
                                         rejected.caches[2],
                                         rejected.caches[3],
                                         rejected.logits),
            2);
  for (const auto& cache : rejected.caches) {
    EXPECT_TRUE(
      std::ranges::all_of(cache, [](float value) { return value == 123.0F; }));
  }
  EXPECT_TRUE(std::ranges::all_of(rejected.logits,
                                  [](float value) { return value == 123.0F; }));
  EXPECT_EQ(infer.infer_formula_decoder(mismatched_mask_shape,
                                        cache_shape,
                                        cache_shape,
                                        cache_shape,
                                        cache_shape,
                                        inferred[0],
                                        inferred[1],
                                        inferred[2],
                                        inferred[3]),
            2);

  const std::array<std::int64_t, 3> mismatched_cache_shape{16, 2, 24};
  const std::vector<float> mismatched_cache(2U * kFormulaCacheStride, 0.0F);
  EXPECT_EQ(compiled.run_formula_decoder(memory,
                                         embedding,
                                         mask,
                                         mask_shape,
                                         caches[0],
                                         cache_shape,
                                         mismatched_cache,
                                         mismatched_cache_shape,
                                         caches[2],
                                         cache_shape,
                                         caches[3],
                                         cache_shape,
                                         rejected.caches[0],
                                         rejected.caches[1],
                                         rejected.caches[2],
                                         rejected.caches[3],
                                         rejected.logits),
            2);
  for (const auto& cache : rejected.caches) {
    EXPECT_TRUE(
      std::ranges::all_of(cache, [](float value) { return value == 123.0F; }));
  }
  EXPECT_EQ(infer.infer_formula_decoder(mask_shape,
                                        cache_shape,
                                        mismatched_cache_shape,
                                        cache_shape,
                                        cache_shape,
                                        inferred[0],
                                        inferred[1],
                                        inferred[2],
                                        inferred[3]),
            2);

  FormulaDecoderResult short_output = make_decoder_result(2);
  short_output.caches[0].pop_back();
  EXPECT_EQ(run_formula_decoder_compiled(
              compiled, memory, embedding, mask, caches, 1, short_output),
            5);

  const auto expected =
    run_formula_decoder_reference(memory, embedding, mask, caches, 1);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 5U);
  for (std::size_t index = 0; index < 4; ++index) {
    ASSERT_EQ((*expected)[index].size(), 2U * kFormulaCacheStride);
  }
  ASSERT_EQ((*expected)[4].size(), kFormulaClasses);
  for (std::size_t index = 0; index < 4; ++index) {
    EXPECT_TRUE(
      compare_values(actual.caches[index], (*expected)[index], 2.0e-3F))
      << kFormulaDecoderOutputs[index];
  }
  EXPECT_TRUE(compare_values(actual.logits, (*expected)[4], 2.0e-3F));
  EXPECT_TRUE(std::ranges::all_of(
    actual.logits, [](float value) { return std::isfinite(value); }));

  FormulaDecoderResult repeated = make_decoder_result(2);
  ASSERT_EQ(run_formula_decoder_compiled(
              compiled, memory, embedding, mask, caches, 1, repeated),
            0);
  EXPECT_EQ(repeated.caches, actual.caches);
  EXPECT_EQ(repeated.logits, actual.logits);
}

TEST(NumericalModel, PPFormulaNetPlusSM11M17M18ContinuousDecodeMatchesNcnn) {
  const TensorShape image_shape(384, 384, 1);
  const std::vector<float> image = make_random_input(384U * 384U, 0x4D313131U);
  const ReferenceModel encoder_reference(
    PP_FORMULANET_PLUS_S_ENCODER_PARAM_PATH,
    PP_FORMULANET_PLUS_S_ENCODER_BIN_PATH,
    "in0",
    "out0",
    image_shape);
  const auto reference_memory = run_ncnn_reference(encoder_reference, image);
  ASSERT_TRUE(reference_memory.has_value()) << reference_memory.error();

  CompiledModel encoder(PP_FORMULANET_PLUS_S_ENCODER_LIBRARY_PATH,
                        "pp_formulanet_plus_s_encoder");
  CompiledModel embed(PP_FORMULANET_PLUS_S_EMBED_LIBRARY_PATH,
                      "pp_formulanet_plus_s_embed");
  CompiledModel decoder(PP_FORMULANET_PLUS_S_DECODER_LIBRARY_PATH,
                        "pp_formulanet_plus_s_decoder");
  CompiledModel infer(PP_FORMULANET_PLUS_S_DECODER_LIBRARY_PATH,
                      "pp_formulanet_plus_s_decoder_infer_output_shapes");
  ASSERT_TRUE(encoder.valid()) << encoder.error();
  ASSERT_TRUE(embed.valid()) << embed.error();
  ASSERT_TRUE(decoder.valid()) << decoder.error();
  ASSERT_TRUE(infer.valid()) << infer.error();

  std::vector<float> compiled_memory(kFormulaMemoryElements);
  ASSERT_EQ(encoder.run(image, compiled_memory), 0);
  ASSERT_TRUE(
    compare_values(compiled_memory, *reference_memory, 1.0e-4F, 5.0e-5F));

  std::array<std::vector<float>, 4> compiled_caches;
  std::array<std::vector<float>, 4> reference_caches;
  for (std::size_t index = 0; index < compiled_caches.size(); ++index) {
    compiled_caches[index].assign(kFormulaCacheStride, 0.0F);
    reference_caches[index].assign(kFormulaCacheStride, 0.0F);
  }

  std::int32_t token = 1;
  constexpr std::array<std::string_view, 1> kEmbedOutputs{"out0"};
  for (int step = 0; step < 2; ++step) {
    const int cache_length = step + 1;
    const std::array<std::int32_t, 1> token_input{token};
    const std::array<std::int32_t, 1> position_input{step};
    const std::array embed_inputs{
      ReferenceInput("in0", TensorShape(1), token_input),
      ReferenceInput("in1", TensorShape(1), position_input),
    };
    const auto reference_embedding =
      run_ncnn_reference(PP_FORMULANET_PLUS_S_EMBED_PARAM_PATH,
                         PP_FORMULANET_PLUS_S_EMBED_BIN_PATH,
                         embed_inputs,
                         kEmbedOutputs);
    ASSERT_TRUE(reference_embedding.has_value()) << reference_embedding.error();
    std::vector<float> compiled_embedding(kFormulaEmbeddingElements);
    ASSERT_EQ(embed.run_two_integer_inputs(
                token_input, position_input, compiled_embedding),
              0);
    ASSERT_TRUE(
      compare_values(compiled_embedding, (*reference_embedding)[0], 1.0e-4F));

    const std::vector<float> mask(cache_length + 1, 0.0F);
    const auto expected =
      run_formula_decoder_reference(*reference_memory,
                                    (*reference_embedding)[0],
                                    mask,
                                    reference_caches,
                                    cache_length);
    ASSERT_TRUE(expected.has_value()) << expected.error();

    const std::array<std::int64_t, 2> mask_shape{1, cache_length + 1};
    const std::array<std::int64_t, 3> cache_shape{16, cache_length, 24};
    std::array<std::array<std::int64_t, 3>, 4> inferred{};
    ASSERT_EQ(infer.infer_formula_decoder(mask_shape,
                                          cache_shape,
                                          cache_shape,
                                          cache_shape,
                                          cache_shape,
                                          inferred[0],
                                          inferred[1],
                                          inferred[2],
                                          inferred[3]),
              0);
    for (const auto& shape : inferred) {
      EXPECT_EQ(shape, (std::array<std::int64_t, 3>{16, cache_length + 1, 24}));
    }

    FormulaDecoderResult actual = make_decoder_result(cache_length + 1);
    ASSERT_EQ(run_formula_decoder_compiled(decoder,
                                           compiled_memory,
                                           compiled_embedding,
                                           mask,
                                           compiled_caches,
                                           cache_length,
                                           actual),
              0);
    // Encoder, embedding, and recurrent cache differences accumulate here.
    constexpr float kContinuousTolerance = 5.0e-3F;
    for (std::size_t index = 0; index < 4; ++index) {
      ASSERT_TRUE(compare_values(
        actual.caches[index], (*expected)[index], kContinuousTolerance))
        << "step " << step << ' ' << kFormulaDecoderOutputs[index];
      compiled_caches[index] = std::move(actual.caches[index]);
      reference_caches[index] = (*expected)[index];
    }
    EXPECT_TRUE(
      compare_values(actual.logits, (*expected)[4], kContinuousTolerance))
      << "step " << step;
    token = static_cast<std::int32_t>(std::ranges::max_element((*expected)[4]) -
                                      (*expected)[4].begin());
  }
}

TEST(NumericalModel, PPFormulaNetPlusSDecoderArtifactsDescribeDynamicCacheAbi) {
  EXPECT_GT(
    std::filesystem::file_size(PP_FORMULANET_PLUS_S_DECODER_LIBRARY_PATH), 0U);
  const std::string manifest =
    without_whitespace(read_text(PP_FORMULANET_PLUS_S_DECODER_MANIFEST_PATH));
  EXPECT_EQ(count_substring(manifest, "\"name\":\"input"), 7U);
  EXPECT_EQ(count_substring(manifest, "\"name\":\"output"), 5U);
  EXPECT_EQ(count_substring(manifest, "\"dynamic_dim_mask\":2"), 9U);
  EXPECT_EQ(count_substring(manifest, "\"shape_program_version\":2"), 4U);
  EXPECT_EQ(count_substring(manifest, "\"lhs_input\":"), 4U);
  EXPECT_NE(manifest.find("\"input_dimension_relations\":"), std::string::npos);
  constexpr std::array<std::array<int, 3>, 4> kRelations{
    std::array{2, 3, 1},
    std::array{3, 4, 0},
    std::array{2, 5, 1},
    std::array{5, 6, 0},
  };
  for (const auto& [lhs, rhs, offset] : kRelations) {
    const std::string relation = std::format(
      "{{\"lhs_dimension\":1,\"lhs_input\":{},\"offset\":{},"
      "\"rhs_dimension\":1,\"rhs_input\":{}}}",
      lhs,
      offset,
      rhs);
    EXPECT_NE(manifest.find(relation), std::string::npos) << relation;
  }
  for (int input = 3; input <= 6; ++input) {
    const std::string program =
      std::format("\"shape_program\":[[0,16],[2,1,{},1,0,1],[0,24]]", input);
    EXPECT_NE(manifest.find(program), std::string::npos) << program;
  }

  const std::string header =
    read_text(PP_FORMULANET_PLUS_S_DECODER_HEADER_PATH);
  for (int input = 3; input <= 7; ++input) {
    EXPECT_NE(header.find(std::format(
                "PP_FORMULANET_PLUS_S_DECODER_INPUT{}_DYNAMIC_DIM_MASK "
                "UINT32_C(0x2)",
                input)),
              std::string::npos);
  }
  for (int output = 1; output <= 4; ++output) {
    EXPECT_NE(header.find(std::format(
                "PP_FORMULANET_PLUS_S_DECODER_OUTPUT{}_DYNAMIC_DIM_MASK "
                "UINT32_C(0x2)",
                output)),
              std::string::npos);
  }
  EXPECT_NE(header.find("pp_formulanet_plus_s_decoder_infer_output_shapes"),
            std::string::npos);
  EXPECT_NE(
    header.find("PP_FORMULANET_PLUS_S_DECODER_INPUT_DIM_RELATION_COUNT 4"),
    std::string::npos);
  for (int input = 3; input <= 7; ++input) {
    EXPECT_NE(header.find(std::format("input{}_shape", input)),
              std::string::npos);
  }
  EXPECT_NE(header.find("uint64_t output1_capacity"), std::string::npos);
  EXPECT_NE(header.find("uint64_t output4_capacity"), std::string::npos);

  const std::string ncnn_ir =
    read_text(PP_FORMULANET_PLUS_S_DECODER_NCNN_IR_PATH);
  EXPECT_NE(ncnn_ir.find("ncnn.sdpa"), std::string::npos);
  EXPECT_NE(ncnn_ir.find("ncnn.input_dim_relations"), std::string::npos);
  const std::string tosa_ir =
    read_text(PP_FORMULANET_PLUS_S_DECODER_TOSA_IR_PATH);
  EXPECT_EQ(tosa_ir.find("ncnn.sdpa"), std::string::npos);
  const std::string linalg_ir =
    read_text(PP_FORMULANET_PLUS_S_DECODER_LINALG_IR_PATH);
  for (const std::string_view operation : {"ncnn.binary",
                                           "ncnn.gelu",
                                           "ncnn.gemm",
                                           "ncnn.layer_norm",
                                           "ncnn.permute",
                                           "ncnn.reshape",
                                           "ncnn.sdpa",
                                           "ncnn.split"}) {
    EXPECT_EQ(linalg_ir.find(operation), std::string::npos) << operation;
  }
}

TEST(NumericalModel, PPFormulaNetPlusSEncoderFp16StorageMatchesNcnn) {
  const TensorShape inputShape(384, 384, 1);
  constexpr std::size_t kOutputElements = 144 * 2048;
  const auto inputElements = inputShape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x464D4631U);
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
  EXPECT_TRUE(compare_values(actual, *expected, 2.0e-2F, 2.0e-2F));
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
