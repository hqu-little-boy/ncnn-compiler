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

}  // namespace

}  // namespace ncnn_compiler::test
