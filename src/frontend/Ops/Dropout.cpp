#include "ncnn_frontend/Ops/Dropout.hpp"

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

DropoutOp::DropoutOp(float scale) noexcept : scale_(scale) {}

float DropoutOp::get_scale() const noexcept {
  return scale_;
}

std::expected<std::vector<TensorType>, std::string> infer_result_types(
  const DropoutOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "Dropout");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  if (!std::isfinite(operation.get_scale())) {
    return std::unexpected("Dropout scale must be finite");
  }
  return std::vector<TensorType>{operands[0]};
}

std::string format_attributes(const DropoutOp& operation) {
  return std::format("kind=dropout,attrs={{scale={}}}",
                     format_float(operation.get_scale()));
}

OperationKind operation_kind(const DropoutOp&) noexcept {
  return OperationKind::Dropout;
}

}  // namespace ncnn_frontend
