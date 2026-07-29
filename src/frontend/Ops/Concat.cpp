#include "ncnn_frontend/Ops/Concat.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

ConcatOp::ConcatOp(std::int64_t axis) noexcept : axis_(axis) {}

std::int64_t ConcatOp::get_axis() const noexcept {
  return axis_;
}

std::expected<std::vector<TensorType>, std::string> infer_result_types(
  const ConcatOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  if (operands.size() < 2 || result_count != 1) {
    return std::unexpected(
      "Concat requires at least two operands and one result");
  }
  const auto& first = operands[0];
  const auto first_shape = first.get_shape();
  const auto rank = static_cast<std::int64_t>(first_shape.size());
  std::int64_t axis = operation.get_axis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return std::unexpected("Concat axis is outside operand rank");
  }
  std::vector<std::int64_t> shape(first_shape.begin(), first_shape.end());
  for (std::size_t operand_index = 1; operand_index < operands.size();
       ++operand_index) {
    const auto& operand = operands[operand_index];
    if (operand.get_element_type() != first.get_element_type() ||
        operand.get_layout() != first.get_layout() ||
        operand.get_shape().size() != first_shape.size()) {
      return std::unexpected(
        "Concat operands must have matching types and rank");
    }
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
      if (std::cmp_equal(dimension, axis)) {
        continue;
      }
      if (operand.get_shape()[dimension] != first_shape[dimension]) {
        return std::unexpected("Concat non-axis dimensions must match");
      }
    }
    auto sum = checked_add(shape[static_cast<std::size_t>(axis)],
                           operand.get_shape()[static_cast<std::size_t>(axis)],
                           "Concat axis dimension");
    if (!sum) {
      return std::unexpected(sum.error());
    }
    shape[static_cast<std::size_t>(axis)] = *sum;
  }
  auto result = create_type(
    std::move(shape), first.get_element_type(), first.get_layout(), "Concat");
  if (!result) {
    return std::unexpected(result.error());
  }
  return std::vector<TensorType>{std::move(*result)};
}

std::string format_attributes(const ConcatOp& operation) {
  return std::format("kind=concat,attrs={{axis={}}}", operation.get_axis());
}

OperationKind operation_kind(const ConcatOp&) noexcept {
  return OperationKind::Concat;
}

}  // namespace ncnn_frontend
