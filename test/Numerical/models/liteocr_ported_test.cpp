#include "numerical_test_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "mat.h"

namespace ncnn_compiler::test {
namespace {

struct BgrImage {
  int width;
  int height;
  std::vector<std::uint8_t> pixels;
};

BgrImage make_document_image(int width, int height) {
  BgrImage image{
    .width = width,
    .height = height,
    .pixels = std::vector<std::uint8_t>(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U,
      245)};
  const auto fill = [&](int left,
                        int top,
                        int right,
                        int bottom,
                        std::array<std::uint8_t, 3> color) {
    for (int y = std::max(0, top); y < std::min(height, bottom); ++y) {
      for (int x = std::max(0, left); x < std::min(width, right); ++x) {
        const std::size_t offset =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)) *
          3U;
        std::ranges::copy(color, image.pixels.begin() + offset);
      }
    }
  };

  fill(width / 12, height / 12, width / 3, height / 5, {40, 70, 170});
  for (int row = 0; row < 7; ++row) {
    const int top = (height / 3) + (row * std::max(4, height / 14));
    const int right = width - (width / (row % 3 + 5));
    fill(width / 10, top, right, top + std::max(2, height / 45), {25, 25, 25});
  }
  fill(width * 3 / 4,
       height * 4 / 5,
       width * 9 / 10,
       height * 9 / 10,
       {180, 80, 30});
  return image;
}

BgrImage make_textline_image(int width, int height) {
  BgrImage image{
    .width = width,
    .height = height,
    .pixels = std::vector<std::uint8_t>(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U,
      255)};
  const auto glyph = [](char character) -> std::array<std::uint8_t, 7> {
    switch (character) {
      case 'L':
        return {16, 16, 16, 16, 16, 16, 31};
      case 'I':
        return {31, 4, 4, 4, 4, 4, 31};
      case 'T':
        return {31, 4, 4, 4, 4, 4, 4};
      case 'E':
        return {31, 16, 16, 30, 16, 16, 31};
      case 'O':
        return {14, 17, 17, 17, 17, 17, 14};
      case 'C':
        return {15, 16, 16, 16, 16, 16, 15};
      case 'R':
        return {30, 17, 17, 30, 20, 18, 17};
      case 'S':
        return {15, 16, 16, 14, 1, 1, 30};
      case '2':
        return {14, 17, 1, 2, 4, 8, 31};
      case '0':
        return {14, 17, 19, 21, 25, 17, 14};
      case '6':
        return {6, 8, 16, 30, 17, 17, 14};
      default:
        return {};
    }
  };
  constexpr std::string_view kText = "LITEOCR TEST 2026";
  const int scale = std::max(2, std::min(6, height / 10));
  const int text_width = (static_cast<int>(kText.size()) * 6 * scale) - scale;
  const int left = std::max(scale, (width - text_width) / 2);
  const int top = std::max(scale, (height - 7 * scale) / 2);
  for (std::size_t index = 0; index < kText.size(); ++index) {
    const auto rows = glyph(kText[index]);
    for (int row = 0; row < 7; ++row) {
      for (int column = 0; column < 5; ++column) {
        if ((rows[row] & (1U << (4 - column))) == 0) {
          continue;
        }
        for (int dy = 0; dy < scale; ++dy) {
          for (int dx = 0; dx < scale; ++dx) {
            const int x = left + (static_cast<int>(index) * 6 * scale) +
                          (column * scale) + dx;
            const int y = top + (row * scale) + dy;
            if (x < 0 || x >= width || y < 0 || y >= height) {
              continue;
            }
            const std::size_t offset =
              (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
               static_cast<std::size_t>(x)) *
              3U;
            image.pixels[offset] = 20;
            image.pixels[offset + 1] = 20;
            image.pixels[offset + 2] = 20;
          }
        }
      }
    }
  }
  return image;
}

BgrImage rotate_180(const BgrImage& source) {
  BgrImage result{
    .width = source.width, .height = source.height, .pixels = source.pixels};
  ncnn::kanna_rotate_c3(source.pixels.data(),
                        source.width,
                        source.height,
                        source.width * 3,
                        result.pixels.data(),
                        result.width,
                        result.height,
                        result.width * 3,
                        2);
  return result;
}

std::vector<float> flatten(const ncnn::Mat& input) {
  std::vector<float> result(static_cast<std::size_t>(input.w) *
                            static_cast<std::size_t>(input.h) *
                            static_cast<std::size_t>(input.c));
  const std::size_t channel_elements =
    static_cast<std::size_t>(input.w) * static_cast<std::size_t>(input.h);
  for (int channel = 0; channel < input.c; ++channel) {
    const float* channel_data = input.channel(channel);
    std::copy_n(
      channel_data,
      channel_elements,
      result.begin() + static_cast<std::size_t>(channel) * channel_elements);
  }
  return result;
}

std::vector<float> preprocess_doc_orientation(const BgrImage& image) {
  constexpr int kResizeShortSide = 256;
  constexpr int kTargetSize = 224;
  int resized_width = kResizeShortSide;
  int resized_height = kResizeShortSide;
  if (image.width < image.height) {
    resized_height = image.height * kResizeShortSide / image.width;
  } else {
    resized_width = image.width * kResizeShortSide / image.height;
  }

  std::vector<std::uint8_t> resized(static_cast<std::size_t>(resized_width) *
                                    static_cast<std::size_t>(resized_height) *
                                    3U);
  ncnn::resize_bilinear_c3(image.pixels.data(),
                           image.width,
                           image.height,
                           image.width * 3,
                           resized.data(),
                           resized_width,
                           resized_height,
                           resized_width * 3);
  const int x = (resized_width - kTargetSize) / 2;
  const int y = (resized_height - kTargetSize) / 2;
  ncnn::Mat input = ncnn::Mat::from_pixels_roi(resized.data(),
                                               ncnn::Mat::PIXEL_BGR,
                                               resized_width,
                                               resized_height,
                                               x,
                                               y,
                                               kTargetSize,
                                               kTargetSize);
  constexpr std::array<float, 3> kMean{127.5F, 127.5F, 127.5F};
  constexpr std::array<float, 3> kNorm{
    1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F};
  input.substract_mean_normalize(kMean.data(), kNorm.data());
  return flatten(input);
}

std::vector<float> preprocess_textline_orientation(const BgrImage& image,
                                                   bool angle_net) {
  const int target_width = angle_net ? 192 : 160;
  const int target_height = angle_net ? 32 : 80;
  const float ratio = static_cast<float>(target_height) / image.height;
  const int aspect_width = static_cast<int>(image.width * ratio);
  int source_width = image.width;
  int resized_width = aspect_width;
  if (!angle_net && aspect_width >= target_width &&
      aspect_width < target_width * 3) {
    resized_width = target_width;
  } else if (!angle_net && aspect_width >= target_width * 3) {
    source_width =
      std::min(image.width, static_cast<int>(3.0F * target_width / ratio));
    resized_width = target_width;
  }

  std::vector<std::uint8_t> resized(static_cast<std::size_t>(resized_width) *
                                    static_cast<std::size_t>(target_height) *
                                    3U);
  ncnn::resize_bilinear_c3(image.pixels.data(),
                           source_width,
                           image.height,
                           image.width * 3,
                           resized.data(),
                           resized_width,
                           target_height,
                           resized_width * 3);

  std::vector<std::uint8_t> final_pixels(
    static_cast<std::size_t>(target_width) *
      static_cast<std::size_t>(target_height) * 3U,
    angle_net ? 255 : 114);
  const int copied_width = std::min(resized_width, target_width);
  for (int y = 0; y < target_height; ++y) {
    std::copy_n(
      resized.data() + (static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(resized_width) * 3U),
      static_cast<std::size_t>(copied_width) * 3U,
      final_pixels.data() + (static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(target_width) * 3U));
  }

  ncnn::Mat input = ncnn::Mat::from_pixels(
    final_pixels.data(),
    angle_net ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR,
    target_width,
    target_height);
  constexpr std::array<float, 3> kMean{127.5F, 127.5F, 127.5F};
  constexpr std::array<float, 3> kNorm{
    1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F};
  input.substract_mean_normalize(kMean.data(), kNorm.data());
  return flatten(input);
}

std::size_t argmax(std::span<const float> values) {
  return static_cast<std::size_t>(
    std::distance(values.begin(), std::ranges::max_element(values)));
}

void expect_compiled_matches_liteocr_flow(const ReferenceModel& reference,
                                          std::string_view library,
                                          std::string_view symbol,
                                          std::span<const float> input,
                                          std::span<const float> rotated_input,
                                          std::size_t output_elements,
                                          bool invert_binary_label) {
  CompiledModel compiled(library, symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();

  const auto expected = run_ncnn_reference(reference, input);
  const auto expected_rotated = run_ncnn_reference(reference, rotated_input);
  ASSERT_TRUE(expected.has_value()) << expected.error();
  ASSERT_TRUE(expected_rotated.has_value()) << expected_rotated.error();
  ASSERT_EQ(expected->size(), output_elements);
  ASSERT_EQ(expected_rotated->size(), output_elements);

  std::vector<float> actual(output_elements);
  std::vector<float> actual_rotated(output_elements);
  ASSERT_EQ(compiled.run(input, actual), 0);
  ASSERT_EQ(compiled.run(rotated_input, actual_rotated), 0);
  EXPECT_TRUE(compare_values(actual, *expected, 1.0e-4F));
  EXPECT_TRUE(compare_values(actual_rotated, *expected_rotated, 1.0e-4F));

  const auto liteocr_label = [&](std::span<const float> output) {
    const std::size_t prediction = argmax(output);
    return invert_binary_label ? (prediction == 0 ? 1U : 0U) : prediction;
  };
  const std::size_t actual_label = liteocr_label(actual);
  const std::size_t rotated_label = liteocr_label(actual_rotated);
  EXPECT_EQ(actual_label, liteocr_label(*expected));
  EXPECT_EQ(rotated_label, liteocr_label(*expected_rotated));
  EXPECT_NE(actual_label, rotated_label);
}

TEST(LiteOCRPorted, DocOrientationUsesCompiledLibrary) {
  const BgrImage image = make_document_image(480, 640);
  const BgrImage rotated = rotate_180(image);
  const std::vector<float> input = preprocess_doc_orientation(image);
  const std::vector<float> rotated_input = preprocess_doc_orientation(rotated);
  const ReferenceModel reference(PP_LCNET_DOC_ORI_PARAM_PATH,
                                 PP_LCNET_DOC_ORI_BIN_PATH,
                                 "in0",
                                 "out0",
                                 TensorShape(224, 224, 3));
  expect_compiled_matches_liteocr_flow(reference,
                                       PP_LCNET_DOC_ORI_LIBRARY_PATH,
                                       "pp_lcnet_x1_0_doc_ori",
                                       input,
                                       rotated_input,
                                       4,
                                       false);
}

TEST(LiteOCRPorted, TextlineOrientationUsesCompiledLibrary) {
  const BgrImage image = make_textline_image(320, 80);
  const BgrImage rotated = rotate_180(image);
  const std::vector<float> input =
    preprocess_textline_orientation(image, false);
  const std::vector<float> rotated_input =
    preprocess_textline_orientation(rotated, false);
  const ReferenceModel reference(PP_LCNET_TEXTLINE_ORI_PARAM_PATH,
                                 PP_LCNET_TEXTLINE_ORI_BIN_PATH,
                                 "in0",
                                 "out0",
                                 TensorShape(160, 80, 3));
  expect_compiled_matches_liteocr_flow(reference,
                                       PP_LCNET_TEXTLINE_ORI_LIBRARY_PATH,
                                       "pp_lcnet_x1_0_textline_ori",
                                       input,
                                       rotated_input,
                                       2,
                                       false);
}

TEST(LiteOCRPorted, AngleNetOrientationUsesCompiledLibrary) {
  const BgrImage image = make_textline_image(240, 64);
  const BgrImage rotated = rotate_180(image);
  const std::vector<float> input = preprocess_textline_orientation(image, true);
  const std::vector<float> rotated_input =
    preprocess_textline_orientation(rotated, true);
  const ReferenceModel reference(CHINESEOCR_LITE_ANGLENET_PARAM_PATH,
                                 CHINESEOCR_LITE_ANGLENET_BIN_PATH,
                                 "in0",
                                 "out0",
                                 TensorShape(192, 32, 3));
  expect_compiled_matches_liteocr_flow(reference,
                                       CHINESEOCR_LITE_ANGLENET_LIBRARY_PATH,
                                       "chineseocr_lite_anglenet",
                                       input,
                                       rotated_input,
                                       2,
                                       true);
}

}  // namespace
}  // namespace ncnn_compiler::test
