#include "numerical_test_support.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

const TensorShape kInputShape(224, 224, 3);
constexpr std::size_t kOutputElements = 1000;
const ReferenceModel kReference(
  YOLOV5S_CLS_PARAM_PATH, YOLOV5S_CLS_BIN_PATH, "in0", "out0", kInputShape);

TEST(NumericalModel, Yolov5SClsMatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  ASSERT_TRUE(kInputShape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*input_elements, 0x59357363U);
  const auto expected = run_ncnn_reference(kReference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(YOLOV5S_CLS_LIBRARY_PATH, "yolov5s_cls");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F, 2.0e-5F));
}

}  // namespace
}  // namespace ncnn_compiler::test
