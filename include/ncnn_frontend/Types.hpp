#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace ncnn_frontend {

class OpId {
 public:
  explicit OpId(std::size_t index) noexcept;
  std::size_t get_index() const noexcept;
  auto operator<=>(const OpId&) const = default;

 private:
  std::size_t index_;
};

class ValueId {
 public:
  explicit ValueId(std::size_t index) noexcept;
  std::size_t get_index() const noexcept;
  auto operator<=>(const ValueId&) const = default;

 private:
  std::size_t index_;
};

enum class ElementType { Float32, Float16, Int8 };
enum class TensorLayout { Scalar, NcnnW, NcnnHW, NcnnCHW, NcnnCDHW, OIHW };

class TensorType {
 public:
  [[nodiscard]] static std::expected<TensorType, std::string> create(
    std::vector<std::int64_t> shape,
    ElementType element_type,
    TensorLayout layout);

  std::span<const std::int64_t> get_shape() const noexcept;
  ElementType get_element_type() const noexcept;
  TensorLayout get_layout() const noexcept;
  std::size_t get_element_count() const noexcept;
  std::size_t get_byte_size() const noexcept;
  bool operator==(const TensorType&) const = default;

 private:
  TensorType(std::vector<std::int64_t> shape,
             ElementType element_type,
             TensorLayout layout,
             std::size_t element_count,
             std::size_t byte_size);

  std::vector<std::int64_t> shape_;
  ElementType element_type_;
  TensorLayout layout_;
  std::size_t element_count_;
  std::size_t byte_size_;
};

class TensorLiteral {
 public:
  [[nodiscard]] static std::expected<TensorLiteral, std::string> create(
    TensorType type, std::vector<std::byte> data);

  const TensorType& get_type() const noexcept;
  std::span<const std::byte> get_data() const noexcept;
  bool operator==(const TensorLiteral&) const = default;

 private:
  TensorLiteral(TensorType type, std::vector<std::byte> data);

  TensorType type_;
  std::vector<std::byte> data_;
};

}  // namespace ncnn_frontend
