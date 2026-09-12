#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

// A3 守护用例：同一 sigmoid 算子，标量路径（threads=1 等价的历史形态）
// 与 MLIR 向量化 + OpenMP 多线程组合路径必须逐位一致。夹具
// sigmoid_vector_fixed 经 --vector-mode=fixed-width 编译，默认
// threads=0 → OpenMP，覆盖 scf.forall(shared_outs) → OpenMP 发射与
// 内层 SIMD 的完整链路。
TEST(NumericalOperator, SigmoidVectorizedOpenMPMatchesScalarBitwise) {
  const TensorShape shape(5, 2, 2);
  constexpr std::array<float, 20> input{
    -20.0F, -10.0F, -6.0F, -3.0F, -1.0F, -0.5F, -0.1F, 0.0F,  0.1F,  0.5F,
    1.0F,   2.0F,   3.0F,  4.0F,  5.0F,  6.0F,  8.0F,  10.0F, 15.0F, 20.0F};

  const ReferenceModel reference(
    fixture_path("sigmoid"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel scalar(SIGMOID_LIBRARY_PATH, "sigmoid");
  ASSERT_TRUE(scalar.valid()) << scalar.error();
  std::vector<float> scalar_output(input.size());
  ASSERT_EQ(scalar.run(input, scalar_output), 0);

  CompiledModel vectorized(SIGMOID_VECTOR_FIXED_LIBRARY_PATH,
                           "sigmoid_vector_fixed");
  ASSERT_TRUE(vectorized.valid()) << vectorized.error();
  std::vector<float> vectorized_output(input.size());
  ASSERT_EQ(vectorized.run(input, vectorized_output), 0);

  // 数值契约：与 ncnn 参考实现一致，且与标量产物逐位相同。
  EXPECT_TRUE(compare_values(vectorized_output, *expected, 1.0e-6F));
  EXPECT_TRUE(compare_values(vectorized_output, scalar_output, 0.0F));
}

// conv→matmul 策略路径的守护：卷积经 strategy-ncnn 折叠为 matmul 后，
// tile-matmul-forall 的分块融合与 forallize-disjoint-tile-loops 的
// epilogue 并行化必须保持与标量产物逐位一致（K 维归约序不变）。
TEST(NumericalOperator, ConvolutionVectorizedOpenMPMatchesScalarBitwise) {
  const TensorShape shape(4, 4, 1);
  const ReferenceModel reference(
    fixture_path("convolution"), CONVOLUTION_BIN_PATH, "data", "output", shape);
  const auto inputElements = shape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x434F4E56U);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel scalar(CONVOLUTION_LIBRARY_PATH, "convolution_scalar");
  ASSERT_TRUE(scalar.valid()) << scalar.error();
  std::vector<float> scalar_output(32);
  ASSERT_EQ(scalar.run(input, scalar_output), 0);

  CompiledModel vectorized(CONVOLUTION_VECTOR_FIXED_LIBRARY_PATH,
                           "convolution_vector_fixed");
  ASSERT_TRUE(vectorized.valid()) << vectorized.error();
  std::vector<float> vectorized_output(32);
  ASSERT_EQ(vectorized.run(input, vectorized_output), 0);

  EXPECT_TRUE(compare_values(vectorized_output, *expected, 1.0e-5F));
  EXPECT_TRUE(compare_values(vectorized_output, scalar_output, 0.0F));
}

// P7 Winograd F(6,3) 守护：3×3 s1、IC=16/OC=32（判据内）经 strategy-ncnn
// 改写为 变换→batch_matmul→逆变换。变换改变累加结构，f32 相对误差
// ~1e-3 量级（有理插值点 ±1/±2/±1/2/∞ 的系数上界，spike 实测定档），
// 数值契约对齐 ncnn FP32 参考 rtol=1e-3 / atol=1e-4，不要求与标量
// 产物逐位一致（对照 conv→matmul 路径的 0.0F 口径）。
TEST(NumericalOperator, ConvolutionWinogradMatchesReference) {
  const TensorShape shape(9, 9, 16);
  const ReferenceModel reference(fixture_path("convolution_winograd"),
                                 CONVOLUTION_WINOGRAD_BIN_PATH,
                                 "data",
                                 "output",
                                 shape);
  const auto inputElements = shape.element_count();
  ASSERT_TRUE(inputElements.has_value()) << inputElements.error();
  const std::vector<float> input =
    make_random_input(*inputElements, 0x574E4F47U);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel winograd(CONVOLUTION_WINOGRAD_LIBRARY_PATH,
                         "convolution_winograd");
  ASSERT_TRUE(winograd.valid()) << winograd.error();
  std::vector<float> output(9 * 9 * 32);
  ASSERT_EQ(winograd.run(input, output), 0);

  EXPECT_TRUE(compare_values(output, *expected, 1.0e-3F, 1.0e-4F));
}

// 向量数学后端守护（--vector-math）：sigmoid 的 exp 经 libmvec /
// vendored SLEEF 向量调用，属近似实现。数值契约改为对齐 ncnn FP32 参考
// 容差（1e-6），不再要求与标量产物逐位一致；两后端之间也只比对参考。
TEST(NumericalOperator, SigmoidVectorMathLibmvecMatchesReference) {
  const TensorShape shape(5, 2, 2);
  constexpr std::array<float, 20> input{
    -20.0F, -10.0F, -6.0F, -3.0F, -1.0F, -0.5F, -0.1F, 0.0F,  0.1F,  0.5F,
    1.0F,   2.0F,   3.0F,  4.0F,  5.0F,  6.0F,  8.0F,  10.0F, 15.0F, 20.0F};

  const ReferenceModel reference(
    fixture_path("sigmoid"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel vectorized(SIGMOID_VECTOR_LMV_LIBRARY_PATH,
                           "sigmoid_vector_lmv");
  ASSERT_TRUE(vectorized.valid()) << vectorized.error();
  std::vector<float> vectorized_output(input.size());
  ASSERT_EQ(vectorized.run(input, vectorized_output), 0);

  EXPECT_TRUE(compare_values(vectorized_output, *expected, 1.0e-6F));
}

TEST(NumericalOperator, SigmoidVectorMathSleefMatchesReference) {
  const TensorShape shape(5, 2, 2);
  constexpr std::array<float, 20> input{
    -20.0F, -10.0F, -6.0F, -3.0F, -1.0F, -0.5F, -0.1F, 0.0F,  0.1F,  0.5F,
    1.0F,   2.0F,   3.0F,  4.0F,  5.0F,  6.0F,  8.0F,  10.0F, 15.0F, 20.0F};

  const ReferenceModel reference(
    fixture_path("sigmoid"), NUMERICAL_EMPTY_BIN_PATH, "data", "output", shape);
  const auto expected = run_ncnn_reference(reference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();

  CompiledModel vectorized(SIGMOID_VECTOR_SLEEF_LIBRARY_PATH,
                           "sigmoid_vector_sleef");
  ASSERT_TRUE(vectorized.valid()) << vectorized.error();
  std::vector<float> vectorized_output(input.size());
  ASSERT_EQ(vectorized.run(input, vectorized_output), 0);

  EXPECT_TRUE(compare_values(vectorized_output, *expected, 1.0e-6F));
}

}  // namespace

}  // namespace ncnn_compiler::test
