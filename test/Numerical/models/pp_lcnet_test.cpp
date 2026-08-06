#include "numerical_test_support.hpp"

#include <numeric>
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

}  // namespace
}  // namespace ncnn_compiler::test
