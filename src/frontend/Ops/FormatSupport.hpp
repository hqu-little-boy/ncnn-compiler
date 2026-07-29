#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

// 模板函数可内联在头文件（风格 §3 例外）。
template <typename Enum>
std::string invalid_enum(Enum value) {
  return std::format("invalid({})",
                     static_cast<std::underlying_type_t<Enum>>(value));
}

std::string format_float(float value);

std::string element_type_name(ElementType type);

std::string layout_name(TensorLayout layout);

std::string format_shape(std::span<const std::int64_t> shape);

std::string format_type(const TensorType& type);

}  // namespace ncnn_frontend
