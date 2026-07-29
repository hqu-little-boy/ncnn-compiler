#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

std::expected<void, std::string> expect_arity(
  std::span<const TensorType> operands,
  std::size_t expected,
  std::size_t results,
  std::size_t expected_results,
  std::string_view operation) {
  if (operands.size() != expected) {
    return std::unexpected(std::format(
      "{} requires {} operands, got {}", operation, expected, operands.size()));
  }
  if (results != expected_results) {
    return std::unexpected(std::format(
      "{} requires {} results, got {}", operation, expected_results, results));
  }
  return {};
}

std::expected<void, std::string> expect_data_type(const TensorType& type,
                                                  std::string_view role) {
  if (type.get_element_type() != ElementType::Float32) {
    return std::unexpected(std::format("{} must be f32", role));
  }
  return {};
}

std::expected<void, std::string> expect_chw(const TensorType& type,
                                            std::string_view role) {
  if (type.get_layout() != TensorLayout::NcnnCHW ||
      type.get_shape().size() != 3) {
    return std::unexpected(std::format("{} must have [C,H,W] layout", role));
  }
  return {};
}

std::expected<void, std::string> expect_positive(std::int64_t value,
                                                 std::string_view name) {
  if (value <= 0) {
    return std::unexpected(std::format("{} must be positive", name));
  }
  return {};
}

std::expected<std::int64_t, std::string> checked_add(
  std::int64_t left, std::int64_t right, std::string_view description) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    return std::unexpected(std::format("{} overflows int64", description));
  }
  return left + right;
}

std::expected<std::int64_t, std::string> checked_multiply(
  std::int64_t left, std::int64_t right, std::string_view description) {
  if (left < 0 || right < 0) {
    return std::unexpected(std::format("{} must not be negative", description));
  }
  if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left) {
    return std::unexpected(std::format("{} overflows int64", description));
  }
  return left * right;
}

std::expected<TensorType, std::string> create_type(
  std::vector<std::int64_t> shape,
  ElementType element_type,
  TensorLayout layout,
  std::string_view operation) {
  auto result = TensorType::create(std::move(shape), element_type, layout);
  if (!result) {
    return std::unexpected(
      std::format("{} result type: {}", operation, result.error()));
  }
  return result;
}

}  // namespace ncnn_frontend
