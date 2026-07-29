#include "ncnn_frontend/Ops/Relu.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

ReluOp::ReluOp(float negative_slope) noexcept
  : negative_slope_(negative_slope) {}

float ReluOp::get_negative_slope() const noexcept {
  return negative_slope_;
}

std::expected<std::vector<TensorType>, std::string> infer_result_types(
  const ReluOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "ReLU");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  if (!std::isfinite(operation.get_negative_slope())) {
    return std::unexpected("ReLU negative slope must be finite");
  }
  return std::vector<TensorType>{operands[0]};
}

std::string format_attributes(const ReluOp& operation) {
  return std::format("kind=relu,attrs={{negative_slope={}}}",
                     format_float(operation.get_negative_slope()));
}

OperationKind operation_kind(const ReluOp&) noexcept {
  return OperationKind::Relu;
}

}  // namespace ncnn_frontend
