#include "numerical_test_support.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

// P7 Winograd F(6,3) 数值预算（resnet18 全主干）：变换改变累加结构，
// f32 相对误差按层 ~1e-3 量级、跨 20+ conv 层累计后由端到端对拍实测
// 定档（见 docs/ncnn-performance-parity-plan.md §3-P7 的预算先行原则；
// 阈值 = 实测最大误差 + 头寸，不放宽常规路径的 1e-4 口径）。
const TensorShape kInputShape(224, 224, 3);
constexpr std::size_t kOutputElements = 1000;
const ReferenceModel kReference(
  RESNET18_PARAM_PATH, RESNET18_BIN_PATH, "in0", "out0", kInputShape);

TEST(NumericalModel, ResNet18WinogradMatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  ASSERT_TRUE(kInputShape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*input_elements, 0x52183518U);
  const auto expected = run_ncnn_reference(kReference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(RESNET18_WINOGRAD_LIBRARY_PATH, "resnet18_winograd");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-3F, 1.0e-4F));
}

}  // namespace
}  // namespace ncnn_compiler::test
