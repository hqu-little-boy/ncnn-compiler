#include "ncnn_frontend/Ops/Softmax.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

SoftmaxOp::SoftmaxOp(std::int64_t axis) noexcept : axis_(axis) {}

std::int64_t SoftmaxOp::get_axis() const noexcept {
  return axis_;
}

std::expected<std::vector<TensorType>, std::string>
SoftmaxOp::infer_result_types(
  std::span<const TensorType> operands,
  std::size_t result_count) const {
  auto arity = expect_arity(operands, 1, result_count, 1, "Softmax");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  const std::int64_t rank =
    static_cast<std::int64_t>(operands[0].get_shape().size());
  std::int64_t axis = get_axis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return std::unexpected("Softmax axis is outside operand rank");
  }
  return std::vector<TensorType>{operands[0]};
}

std::string SoftmaxOp::format_attributes() const {
  return std::format("kind=softmax,attrs={{axis={}}}", get_axis());
}


}  // namespace ncnn_frontend
