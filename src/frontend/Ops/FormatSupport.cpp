#include "FormatSupport.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

std::string format_float(float value) {
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? "-inf" : "inf";
  }
  if (value == 0.0f) {
    return std::signbit(value) ? "-0" : "0";
  }

  char buffer[64];
  const auto [end, error] =
    std::to_chars(buffer,
                  buffer + sizeof(buffer),
                  value,
                  std::chars_format::general,
                  std::numeric_limits<float>::max_digits10);
  if (error != std::errc()) {
    return "<float-format-error>";
  }
  return {buffer, end};
}

std::string element_type_name(ElementType type) {
  switch (type) {
    case ElementType::Float32:
      return "f32";
    case ElementType::Float16:
      return "f16";
    case ElementType::Int8:
      return "i8";
  }
  return invalid_enum(type);
}

std::string layout_name(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::Scalar:
      return "scalar";
    case TensorLayout::NcnnW:
      return "ncnn_w";
    case TensorLayout::NcnnHW:
      return "ncnn_hw";
    case TensorLayout::NcnnCHW:
      return "ncnn_chw";
    case TensorLayout::NcnnCDHW:
      return "ncnn_cdhw";
    case TensorLayout::OIHW:
      return "oihw";
  }
  return invalid_enum(layout);
}

std::string format_shape(std::span<const std::int64_t> shape) {
  std::string result = "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      result.push_back(',');
    }
    result += std::to_string(shape[index]);
  }
  result.push_back(']');
  return result;
}

std::string format_type(const TensorType& type) {
  return std::format("{{shape={},element={},layout={},elements={},bytes={}}}",
                     format_shape(type.get_shape()),
                     element_type_name(type.get_element_type()),
                     layout_name(type.get_layout()),
                     type.get_element_count(),
                     type.get_byte_size());
}

}  // namespace ncnn_frontend
