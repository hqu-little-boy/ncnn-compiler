#include "numerical_test_support.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

constexpr TensorShape kInputShape{.width = 227, .height = 227, .channels = 3};
constexpr std::size_t kOutputElements = 1000;
const ReferenceModel kReference{.param_path = SQUEEZENET_PARAM_PATH,
                                .bin_path = SQUEEZENET_BIN_PATH,
                                .input_blob = "data",
                                .output_blob = "prob",
                                .input_shape = kInputShape};

TEST(NumericalModel, SqueezeNetMatchesNcnn) {
  const std::vector<float> input =
    make_random_input(kInputShape.element_count(), 0x53515545U, -0.01F, 0.01F);
  const auto expected = run_ncnn_reference(kReference, input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), kOutputElements);

  CompiledModel compiled(SQUEEZENET_LIBRARY_PATH, "squeezenet_v1_1");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> actual(kOutputElements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(check_softmax(actual, *expected));
}

}  // namespace
}  // namespace ncnn_compiler::test
