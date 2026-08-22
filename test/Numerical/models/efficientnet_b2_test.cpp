#include "numerical_test_support.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

const TensorShape kInputShape(260, 260, 3);
constexpr std::size_t kOutputElements = 1000;
const ReferenceModel kReference(EFFICIENTNET_B2_PARAM_PATH,
                                EFFICIENTNET_B2_BIN_PATH,
                                "in0",
                                "out0",
                                kInputShape);

TEST(NumericalModel, EfficientNetB2MatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  ASSERT_TRUE(kInputShape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*input_elements, 0x454e4232U);
  const auto expected = run_ncnn_reference(kReference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(EFFICIENTNET_B2_LIBRARY_PATH, "efficientnet_b2");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F, 2.0e-5F));
}

}  // namespace
}  // namespace ncnn_compiler::test
