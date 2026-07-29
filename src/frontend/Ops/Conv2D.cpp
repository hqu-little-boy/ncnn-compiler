#include "ncnn_frontend/Ops/Conv2D.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {
namespace {

std::string quantization_name(ConvQuantizationMode mode) {
  switch (mode) {
    case ConvQuantizationMode::None:
      return "none";
    case ConvQuantizationMode::Dequantize:
      return "dequantize";
    case ConvQuantizationMode::Requantize:
      return "requantize";
  }
  return invalid_enum(mode);
}

}  // namespace

Conv2DOp::Conv2DOp(std::int64_t kernel_height,
                   std::int64_t kernel_width,
                   std::int64_t stride_height,
                   std::int64_t stride_width,
                   std::int64_t dilation_height,
                   std::int64_t dilation_width,
                   std::int64_t pad_top,
                   std::int64_t pad_bottom,
                   std::int64_t pad_left,
                   std::int64_t pad_right,
                   bool has_bias,
                   std::int64_t int8_scale_term) noexcept
  : kernel_height_(kernel_height),
    kernel_width_(kernel_width),
    stride_height_(stride_height),
    stride_width_(stride_width),
    dilation_height_(dilation_height),
    dilation_width_(dilation_width),
    pad_top_(pad_top),
    pad_bottom_(pad_bottom),
    pad_left_(pad_left),
    pad_right_(pad_right),
    has_bias_(has_bias),
    int8_scale_term_(int8_scale_term) {}

std::int64_t Conv2DOp::get_kernel_height() const noexcept {
  return kernel_height_;
}
std::int64_t Conv2DOp::get_kernel_width() const noexcept {
  return kernel_width_;
}
std::int64_t Conv2DOp::get_stride_height() const noexcept {
  return stride_height_;
}
std::int64_t Conv2DOp::get_stride_width() const noexcept {
  return stride_width_;
}
std::int64_t Conv2DOp::get_dilation_height() const noexcept {
  return dilation_height_;
}
std::int64_t Conv2DOp::get_dilation_width() const noexcept {
  return dilation_width_;
}
std::int64_t Conv2DOp::get_pad_top() const noexcept {
  return pad_top_;
}
std::int64_t Conv2DOp::get_pad_bottom() const noexcept {
  return pad_bottom_;
}
std::int64_t Conv2DOp::get_pad_left() const noexcept {
  return pad_left_;
}
std::int64_t Conv2DOp::get_pad_right() const noexcept {
  return pad_right_;
}
bool Conv2DOp::get_has_bias() const noexcept {
  return has_bias_;
}

std::int64_t Conv2DOp::get_int8_scale_term() const noexcept {
  return int8_scale_term_;
}

ConvQuantizationMode Conv2DOp::get_quantization_mode() const noexcept {
  if (int8_scale_term_ == 0) {
    return ConvQuantizationMode::None;
  }
  if (int8_scale_term_ > 100) {
    return ConvQuantizationMode::Requantize;
  }
  return ConvQuantizationMode::Dequantize;
}

std::expected<std::vector<TensorType>, std::string> infer_result_types(
  const Conv2DOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  const std::size_t minimum_operands = operation.get_has_bias() ? 3 : 2;
  const auto term = operation.get_int8_scale_term();
  if (term != 0 && term != 1 && term != 2 && term != 101 && term != 102) {
    return std::unexpected("Convolution has unsupported int8_scale_term");
  }
  const bool quantized = term != 0;
  const bool requantized = term > 100;
  const std::size_t expected_operands =
    minimum_operands + (quantized ? 2 : 0) + (requantized ? 1 : 0);
  if (operands.size() != expected_operands || result_count != 1) {
    return std::unexpected(
      "Convolution operands do not match bias and quantization mode");
  }
  const auto& input = operands[0];
  if (quantized) {
    if (input.get_element_type() != ElementType::Float32 &&
        input.get_element_type() != ElementType::Int8) {
      return std::unexpected("quantized convolution input must be f32 or i8");
    }
  } else if (input.get_element_type() != ElementType::Float32) {
    return std::unexpected("non-quantized convolution input must be f32");
  }
  auto input_layout = expect_chw(input, "convolution input");
  if (!input_layout) {
    return std::unexpected(input_layout.error());
  }
  const auto& weight = operands[1];
  if (weight.get_layout() != TensorLayout::OIHW ||
      weight.get_shape().size() != 4) {
    return std::unexpected("convolution weight must have [O,I,H,W] layout");
  }
  if (!quantized && weight.get_element_type() == ElementType::Int8) {
    return std::unexpected("non-quantized convolution weight cannot be i8");
  }
  if (quantized && weight.get_element_type() == ElementType::Float16) {
    return std::unexpected("quantized convolution weight must be f32 or i8");
  }
  if (weight.get_element_type() != ElementType::Float32 &&
      weight.get_element_type() != ElementType::Float16 &&
      weight.get_element_type() != ElementType::Int8) {
    return std::unexpected("convolution weight has unsupported element type");
  }
  const auto input_shape = operands[0].get_shape();
  const auto weight_shape = weight.get_shape();
  if (input_shape[0] <= 0 || input_shape[1] <= 0 || input_shape[2] <= 0 ||
      weight_shape[0] <= 0 || weight_shape[1] <= 0 || weight_shape[2] <= 0 ||
      weight_shape[3] <= 0) {
    return std::unexpected(
      "convolution input and weight dimensions must be positive");
  }
  if (weight_shape[1] != input_shape[0]) {
    return std::unexpected(std::format(
      "convolution input channels {} do not match weight channels {}",
      input_shape[0],
      weight_shape[1]));
  }
  if (weight_shape[2] != operation.get_kernel_height() ||
      weight_shape[3] != operation.get_kernel_width()) {
    return std::unexpected("convolution kernel attributes do not match weight");
  }
  if (operation.get_has_bias()) {
    const auto& bias = operands[2];
    if (bias.get_layout() != TensorLayout::NcnnW ||
        bias.get_shape().size() != 1 ||
        bias.get_shape()[0] != weight_shape[0]) {
      return std::unexpected("convolution bias must have shape [O]");
    }
    if (bias.get_element_type() != ElementType::Float32) {
      return std::unexpected("convolution bias must be f32");
    }
  }
  for (std::size_t index = minimum_operands; index < operands.size(); ++index) {
    const auto& scale = operands[index];
    const std::size_t scale_index = index - minimum_operands;
    const std::int64_t expected_size = scale_index == 0 ? weight_shape[0] : 1;
    if (scale.get_layout() != TensorLayout::NcnnW ||
        scale.get_element_type() != ElementType::Float32 ||
        scale.get_shape().size() != 1 ||
        scale.get_shape()[0] != expected_size) {
      return std::unexpected("convolution scale has the wrong role or shape");
    }
  }
  for (const auto [value, name] :
       {std::pair{operation.get_kernel_height(), "convolution kernel height"},
        std::pair{operation.get_kernel_width(), "convolution kernel width"},
        std::pair{operation.get_stride_height(), "convolution stride height"},
        std::pair{operation.get_stride_width(), "convolution stride width"},
        std::pair{operation.get_dilation_height(),
                  "convolution dilation height"},
        std::pair{operation.get_dilation_width(),
                  "convolution dilation width"}}) {
    auto valid = expect_positive(value, name);
    if (!valid) {
      return std::unexpected(valid.error());
    }
  }
  const std::int64_t pads[] = {operation.get_pad_top(),
                               operation.get_pad_bottom(),
                               operation.get_pad_left(),
                               operation.get_pad_right()};
  bool same_upper = true;
  bool same_lower = true;
  for (const std::int64_t pad : pads) {
    if (pad < 0 && pad != -233 && pad != -234) {
      return std::unexpected("convolution padding has unsupported value");
    }
    same_upper = same_upper && pad == -233;
    same_lower = same_lower && pad == -234;
  }
  if ((!same_upper && !same_lower) &&
      (operation.get_pad_top() < 0 || operation.get_pad_bottom() < 0 ||
       operation.get_pad_left() < 0 || operation.get_pad_right() < 0)) {
    return std::unexpected("convolution SAME padding must use one pad mode");
  }
  auto extent_height = checked_multiply(operation.get_dilation_height(),
                                        operation.get_kernel_height() - 1,
                                        "convolution kernel height extent");
  if (!extent_height) {
    return std::unexpected(extent_height.error());
  }
  extent_height = checked_add(*extent_height, 1, "convolution kernel height");
  if (!extent_height) {
    return std::unexpected(extent_height.error());
  }
  auto extent_width = checked_multiply(operation.get_dilation_width(),
                                       operation.get_kernel_width() - 1,
                                       "convolution kernel width extent");
  if (!extent_width) {
    return std::unexpected(extent_width.error());
  }
  extent_width = checked_add(*extent_width, 1, "convolution kernel width");
  if (!extent_width) {
    return std::unexpected(extent_width.error());
  }
  std::int64_t output_height = 0;
  std::int64_t output_width = 0;
  if (same_upper || same_lower) {
    output_height = 1 + ((input_shape[1] - 1) / operation.get_stride_height());
    output_width = 1 + ((input_shape[2] - 1) / operation.get_stride_width());
  } else {
    auto height = checked_add(
      input_shape[1], operation.get_pad_top(), "convolution padded height");
    if (!height) {
      return std::unexpected(height.error());
    }
    height = checked_add(
      *height, operation.get_pad_bottom(), "convolution padded height");
    if (!height) {
      return std::unexpected(height.error());
    }
    auto width = checked_add(
      input_shape[2], operation.get_pad_left(), "convolution padded width");
    if (!width) {
      return std::unexpected(width.error());
    }
    width = checked_add(
      *width, operation.get_pad_right(), "convolution padded width");
    if (!width) {
      return std::unexpected(width.error());
    }
    if (*height < *extent_height || *width < *extent_width) {
      return std::unexpected("convolution kernel exceeds padded input");
    }
    output_height =
      1 + ((*height - *extent_height) / operation.get_stride_height());
    output_width =
      1 + ((*width - *extent_width) / operation.get_stride_width());
  }
  auto result =
    create_type({weight_shape[0], output_height, output_width},
                requantized ? ElementType::Int8 : ElementType::Float32,
                TensorLayout::NcnnCHW,
                "convolution");
  if (!result) {
    return std::unexpected(result.error());
  }
  return std::vector<TensorType>{std::move(*result)};
}

std::string format_attributes(const Conv2DOp& operation) {
  return std::format(
    "kind=conv2d,attrs={{kernel=[{},{}],stride=[{},{}],dilation=[{},{}],"
    "pad=[{},{},{},{}],has_bias={},int8_scale_term={},quantization={}}}",
    operation.get_kernel_height(),
    operation.get_kernel_width(),
    operation.get_stride_height(),
    operation.get_stride_width(),
    operation.get_dilation_height(),
    operation.get_dilation_width(),
    operation.get_pad_top(),
    operation.get_pad_bottom(),
    operation.get_pad_left(),
    operation.get_pad_right(),
    operation.get_has_bias(),
    operation.get_int8_scale_term(),
    quantization_name(operation.get_quantization_mode()));
}

OperationKind operation_kind(const Conv2DOp&) noexcept {
  return OperationKind::Convolution;
}

}  // namespace ncnn_frontend
