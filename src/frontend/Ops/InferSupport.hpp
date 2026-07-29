#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

std::expected<void, std::string> expect_arity(
  std::span<const TensorType> operands,
  std::size_t expected,
  std::size_t results,
  std::size_t expected_results,
  std::string_view operation);

std::expected<void, std::string> expect_data_type(const TensorType& type,
                                                   std::string_view role);

std::expected<void, std::string> expect_chw(const TensorType& type,
                                             std::string_view role);

std::expected<void, std::string> expect_positive(std::int64_t value,
                                                  std::string_view name);

std::expected<std::int64_t, std::string> checked_add(
  std::int64_t left, std::int64_t right, std::string_view description);

std::expected<std::int64_t, std::string> checked_multiply(
  std::int64_t left, std::int64_t right, std::string_view description);

std::expected<TensorType, std::string> create_type(
  std::vector<std::int64_t> shape,
  ElementType element_type,
  TensorLayout layout,
  std::string_view operation);

}  // namespace ncnn_frontend
