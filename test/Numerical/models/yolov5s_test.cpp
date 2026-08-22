#include "numerical_test_support.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

const TensorShape kInputShape(640, 640, 3);
constexpr std::size_t kGridSum = (80 * 80) + (40 * 40) + (20 * 20);
constexpr std::size_t kAnchors = 3;
constexpr std::size_t kAttributes = 85;
constexpr std::size_t kOutputElements = kGridSum * kAnchors * kAttributes;
const ReferenceModel kReference(
  YOLOV5S_PARAM_PATH, YOLOV5S_BIN_PATH, "in0", "out0", kInputShape);

TEST(NumericalModel, Yolov5sDetectionMatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  ASSERT_TRUE(kInputShape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*input_elements, 0x59357373U);
  const auto expected = run_ncnn_reference(kReference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(YOLOV5S_LIBRARY_PATH, "yolov5s");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F, 2.0e-5F));

  std::vector<float> repeated(kOutputElements);
  ASSERT_EQ(compiled.run(input, repeated), 0);
  EXPECT_TRUE(compare_values(repeated, *expected, 1.0e-4F, 2.0e-5F));
}

}  // namespace
}  // namespace ncnn_compiler::test
