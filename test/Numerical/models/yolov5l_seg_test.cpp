#include "numerical_test_support.hpp"

#include <array>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

const TensorShape kInputShape(640, 640, 3);
constexpr std::size_t kProtoChannels = 32;
constexpr std::size_t kProtoExtent = 160;
constexpr std::size_t kProtoElements =
  kProtoChannels * kProtoExtent * kProtoExtent;
constexpr std::size_t kGridSum = (80 * 80) + (40 * 40) + (20 * 20);
constexpr std::size_t kAnchors = 3;
constexpr std::size_t kAttributes = 117;
constexpr std::size_t kDetectionElements = kGridSum * kAnchors * kAttributes;

TEST(NumericalModel, Yolov5lSegMatchesNcnn) {
  const auto input_elements = kInputShape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  ASSERT_TRUE(kInputShape.byte_count(sizeof(float)).has_value());
  const std::vector<float> input =
    make_random_input(*input_elements, 0x53454734U);

  constexpr std::array<std::string_view, 2> kOutputs{"out1", "out0"};
  const std::array<ReferenceInput, 1> kInputs{
    ReferenceInput("in0", kInputShape, input)};
  const auto expected = run_ncnn_reference(
    YOLOV5L_SEG_PARAM_PATH, YOLOV5L_SEG_BIN_PATH, kInputs, kOutputs);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_EQ(expected->size(), 2U);
  ASSERT_EQ(expected->front().size(), kProtoElements);
  ASSERT_EQ(expected->back().size(), kDetectionElements);

  CompiledModel compiled(YOLOV5L_SEG_LIBRARY_PATH, "yolov5l_seg");
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  std::vector<float> proto(kProtoElements);
  std::vector<float> detections(kDetectionElements);
  ASSERT_EQ(compiled.run_two_outputs(input, proto, detections), 0);
  EXPECT_TRUE(compare_values(proto, expected->front(), 1.0e-4F, 2.0e-5F));
  EXPECT_TRUE(compare_values(detections, expected->back(), 1.0e-4F, 2.0e-5F));

  std::vector<float> repeated_proto(kProtoElements);
  std::vector<float> repeated_detections(kDetectionElements);
  ASSERT_EQ(
    compiled.run_two_outputs(input, repeated_proto, repeated_detections), 0);
  EXPECT_TRUE(
    compare_values(repeated_proto, expected->front(), 1.0e-4F, 2.0e-5F));
  EXPECT_TRUE(
    compare_values(repeated_detections, expected->back(), 1.0e-4F, 2.0e-5F));
}

}  // namespace
}  // namespace ncnn_compiler::test
