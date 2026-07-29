#include "ncnn_frontend/Ops/Softmax.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

SoftmaxOp::SoftmaxOp(std::int64_t axis) noexcept : axis_(axis) {}

std::int64_t SoftmaxOp::get_axis() const noexcept {
  return axis_;
}

std::expected<std::vector<TensorType>, std::string> infer_result_types(
  const SoftmaxOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "Softmax");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  const std::int64_t rank =
    static_cast<std::int64_t>(operands[0].get_shape().size());
  std::int64_t axis = operation.get_axis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return std::unexpected("Softmax axis is outside operand rank");
  }
  return std::vector<TensorType>{operands[0]};
}

std::string format_attributes(const SoftmaxOp& operation) {
  return std::format("kind=softmax,attrs={{axis={}}}", operation.get_axis());
}

OperationKind operation_kind(const SoftmaxOp&) noexcept {
  return OperationKind::Softmax;
}

}  // namespace ncnn_frontend
