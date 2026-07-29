#include "ncnn_frontend/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ncnn_frontend {
namespace {

std::size_t element_size(ElementType type) noexcept {
  switch (type) {
    case ElementType::Float32:
      return 4;
    case ElementType::Float16:
      return 2;
    case ElementType::Int8:
      return 1;
  }
  return 0;
}

bool valid_element_type(ElementType type) noexcept {
  switch (type) {
    case ElementType::Float32:
    case ElementType::Float16:
    case ElementType::Int8:
      return true;
  }
  return false;
}

std::size_t expected_rank(TensorLayout layout) noexcept {
  switch (layout) {
    case TensorLayout::Scalar:
      return 0;
    case TensorLayout::NcnnW:
      return 1;
    case TensorLayout::NcnnHW:
      return 2;
    case TensorLayout::NcnnCHW:
      return 3;
    case TensorLayout::NcnnCDHW:
    case TensorLayout::OIHW:
      return 4;
  }
  return 0;
}

bool valid_layout(TensorLayout layout) noexcept {
  switch (layout) {
    case TensorLayout::Scalar:
    case TensorLayout::NcnnW:
    case TensorLayout::NcnnHW:
    case TensorLayout::NcnnCHW:
    case TensorLayout::NcnnCDHW:
    case TensorLayout::OIHW:
      return true;
  }
  return false;
}

}  // namespace

OpId::OpId(std::size_t index) noexcept : index_(index) {}

std::size_t OpId::get_index() const noexcept {
  return index_;
}

ValueId::ValueId(std::size_t index) noexcept : index_(index) {}

std::size_t ValueId::get_index() const noexcept {
  return index_;
}

TensorType::TensorType(std::vector<std::int64_t> shape,
                       ElementType element_type,
                       TensorLayout layout,
                       std::size_t element_count,
                       std::size_t byte_size)
  : shape_(std::move(shape)),
    element_type_(element_type),
    layout_(layout),
    element_count_(element_count),
    byte_size_(byte_size) {}

std::expected<TensorType, std::string> TensorType::create(
  std::vector<std::int64_t> shape,
  ElementType element_type,
  TensorLayout layout) {
  if (!valid_element_type(element_type)) {
    return std::unexpected("tensor element type enum is invalid");
  }
  if (!valid_layout(layout)) {
    return std::unexpected("tensor layout enum is invalid");
  }
  if (shape.size() != expected_rank(layout)) {
    return std::unexpected(
      std::format("layout rank {} does not match shape rank {}",
                  expected_rank(layout),
                  shape.size()));
  }

  std::size_t count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      return std::unexpected("tensor dimensions must be non-negative");
    }
    const auto converted = static_cast<std::uint64_t>(dimension);
    if (converted > std::numeric_limits<std::size_t>::max()) {
      return std::unexpected("tensor dimension does not fit size_t");
    }
    const auto size = static_cast<std::size_t>(converted);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) {
      return std::unexpected("tensor element count overflows size_t");
    }
    count *= size;
  }

  const std::size_t width = element_size(element_type);
  if (count != 0 && width > std::numeric_limits<std::size_t>::max() / count) {
    return std::unexpected("tensor byte size overflows size_t");
  }
  return TensorType(
    std::move(shape), element_type, layout, count, count * width);
}

std::span<const std::int64_t> TensorType::get_shape() const noexcept {
  return shape_;
}

ElementType TensorType::get_element_type() const noexcept {
  return element_type_;
}

TensorLayout TensorType::get_layout() const noexcept {
  return layout_;
}

std::size_t TensorType::get_element_count() const noexcept {
  return element_count_;
}

std::size_t TensorType::get_byte_size() const noexcept {
  return byte_size_;
}

TensorLiteral::TensorLiteral(TensorType type, std::vector<std::byte> data)
  : type_(std::move(type)), data_(std::move(data)) {}

std::expected<TensorLiteral, std::string> TensorLiteral::create(
  TensorType type, std::vector<std::byte> data) {
  if (data.size() != type.get_byte_size()) {
    return std::unexpected(
      std::format("constant payload has {} bytes, expected {}",
                  data.size(),
                  type.get_byte_size()));
  }
  return TensorLiteral(std::move(type), std::move(data));
}

const TensorType& TensorLiteral::get_type() const noexcept {
  return type_;
}

std::span<const std::byte> TensorLiteral::get_data() const noexcept {
  return data_;
}

}  // namespace ncnn_frontend
