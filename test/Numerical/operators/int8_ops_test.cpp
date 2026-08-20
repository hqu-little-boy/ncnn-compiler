#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

std::string fixture_path(std::string_view name, std::string_view extension) {
  return std::string(NUMERICAL_FIXTURE_DIR) + "/" + std::string(name) + "." +
         std::string(extension);
}

void expect_int8_fixture(std::string_view fixture,
                         std::string_view bin_path,
                         std::string_view library_path,
                         TensorShape input_shape,
                         std::size_t output_elements,
                         float tolerance,
                         std::uint32_t seed) {
  const auto input_elements = input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const auto input = make_random_input(*input_elements, seed, -1.5F, 1.5F);
  const ReferenceModel reference(fixture_path(fixture, "param"),
                                 std::string(bin_path),
                                 "data",
                                 "output",
                                 input_shape);
  const auto expected =
    run_ncnn_reference(reference, input, ReferenceInferenceMode::Int8);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), output_elements);

  CompiledModel compiled(library_path, fixture);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(output_elements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, tolerance, tolerance));
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

struct Int8Case {
  const char* fixture;
  const char* bin;
  const char* library;
  TensorShape shape;
  std::size_t outputs;
  float tolerance;
  std::uint32_t seed;
};

class Int8OperatorTest : public ::testing::TestWithParam<Int8Case> {};

std::string int8_case_name(const ::testing::TestParamInfo<Int8Case>& info) {
  constexpr std::array names{"ConvTerm1",
                             "ConvTerm2",
                             "ConvTerm101",
                             "ConvTerm102",
                             "DepthwiseTerm1",
                             "DepthwiseTerm2",
                             "DepthwiseTerm101",
                             "DepthwiseTerm102",
                             "InnerProductTerm1",
                             "InnerProductTerm2",
                             "GemmTerm2"};
  return names[info.index];
}

TEST_P(Int8OperatorTest, MatchesInt8NcnnReferenceWithF32Boundary) {
  const auto& test = GetParam();
  expect_int8_fixture(test.fixture,
                      test.bin,
                      test.library,
                      test.shape,
                      test.outputs,
                      test.tolerance,
                      test.seed);
}

INSTANTIATE_TEST_SUITE_P(
  Contractions,
  Int8OperatorTest,
  ::testing::Values(Int8Case{"convolution_int8_term1",
                             CONVOLUTION_INT8_TERM1_BIN_PATH,
                             CONVOLUTION_INT8_TERM1_LIBRARY_PATH,
                             TensorShape(4, 3, 2),
                             36,
                             1.0e-5F,
                             0xC001U},
                    Int8Case{"convolution_int8_term2",
                             CONVOLUTION_INT8_TERM2_BIN_PATH,
                             CONVOLUTION_INT8_TERM2_LIBRARY_PATH,
                             TensorShape(4, 3, 2),
                             36,
                             1.0e-5F,
                             0xC002U},
                    Int8Case{"convolution_int8_term101",
                             CONVOLUTION_INT8_TERM101_BIN_PATH,
                             CONVOLUTION_INT8_TERM101_LIBRARY_PATH,
                             TensorShape(4, 3, 2),
                             24,
                             1.0e-5F,
                             0xC101U},
                    Int8Case{"convolution_int8_term102",
                             CONVOLUTION_INT8_TERM102_BIN_PATH,
                             CONVOLUTION_INT8_TERM102_LIBRARY_PATH,
                             TensorShape(4, 3, 2),
                             24,
                             1.0e-5F,
                             0xC102U},
                    Int8Case{"depthwise_int8_term1",
                             DEPTHWISE_INT8_TERM1_BIN_PATH,
                             DEPTHWISE_INT8_TERM1_LIBRARY_PATH,
                             TensorShape(4, 3, 3),
                             36,
                             1.0e-5F,
                             0xD001U},
                    Int8Case{"depthwise_int8_term2",
                             DEPTHWISE_INT8_TERM2_BIN_PATH,
                             DEPTHWISE_INT8_TERM2_LIBRARY_PATH,
                             TensorShape(4, 3, 3),
                             36,
                             1.0e-5F,
                             0xD002U},
                    Int8Case{"depthwise_int8_term101",
                             DEPTHWISE_INT8_TERM101_BIN_PATH,
                             DEPTHWISE_INT8_TERM101_LIBRARY_PATH,
                             TensorShape(4, 3, 3),
                             36,
                             1.0e-5F,
                             0xD101U},
                    Int8Case{"depthwise_int8_term102",
                             DEPTHWISE_INT8_TERM102_BIN_PATH,
                             DEPTHWISE_INT8_TERM102_LIBRARY_PATH,
                             TensorShape(4, 3, 3),
                             36,
                             1.0e-5F,
                             0xD102U},
                    Int8Case{"inner_product_int8_term1",
                             INNER_PRODUCT_INT8_TERM1_BIN_PATH,
                             INNER_PRODUCT_INT8_TERM1_LIBRARY_PATH,
                             TensorShape(4, 1, 1),
                             3,
                             1.0e-5F,
                             0x1F01U},
                    Int8Case{"inner_product_int8_term2",
                             INNER_PRODUCT_INT8_TERM2_BIN_PATH,
                             INNER_PRODUCT_INT8_TERM2_LIBRARY_PATH,
                             TensorShape(4, 1, 1),
                             3,
                             1.0e-5F,
                             0x1F02U},
                    Int8Case{"gemm_int8_term2",
                             GEMM_INT8_TERM2_BIN_PATH,
                             GEMM_INT8_TERM2_LIBRARY_PATH,
                             TensorShape(3, 1, 2),
                             12,
                             1.0e-5F,
                             0x6E22U}),
  int8_case_name);

TEST(Int8Model, CompleteRequantizedChainMatchesNcnn) {
  expect_int8_fixture("int8_complete_chain",
                      INT8_COMPLETE_CHAIN_BIN_PATH,
                      INT8_COMPLETE_CHAIN_LIBRARY_PATH,
                      TensorShape(4, 3, 2),
                      24,
                      1.0e-5F,
                      0x8C11U);
}

TEST(Int8Codegen, UsesI32AccumulationBroadcastScalesAndF32Abi) {
  const std::string tosa = read_text(INT8_COMPLETE_CHAIN_TOSA_IR_PATH);
  EXPECT_NE(tosa.find("acc_type = i32"), std::string::npos);
  EXPECT_NE(tosa.find("tensor<1x1x1x3xf32>"), std::string::npos);
  EXPECT_NE(tosa.find("arith.fptosi"), std::string::npos);
  EXPECT_NE(tosa.find("tensor<1x3x4x3xi8>"), std::string::npos);

  const std::string manifest = read_text(INT8_COMPLETE_CHAIN_MANIFEST_PATH);
  EXPECT_EQ(manifest.find("\"element_type\": \"i8\""), std::string::npos);
  EXPECT_NE(manifest.find("\"element_type\": \"f32\""), std::string::npos);

  const std::string header = read_text(INT8_COMPLETE_CHAIN_HEADER_PATH);
  EXPECT_NE(header.find("const float *input1, float *output1"),
            std::string::npos);
  EXPECT_EQ(header.find("int8_t *"), std::string::npos);
}

}  // namespace
}  // namespace ncnn_compiler::test
