#include "ncnn_frontend/Ops/Dropout.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

DropoutOp::DropoutOp(float scale) noexcept : scale_(scale) {}

float DropoutOp::get_scale() const noexcept {
  return scale_;
}

std::expected<std::vector<TensorType>, std::string>
DropoutOp::infer_result_types(
  std::span<const TensorType> operands,
  std::size_t result_count) const {
  auto arity = expect_arity(operands, 1, result_count, 1, "Dropout");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  if (!std::isfinite(get_scale())) {
    return std::unexpected("Dropout scale must be finite");
  }
  return std::vector<TensorType>{operands[0]};
}

std::string DropoutOp::format_attributes() const {
  return std::format("kind=dropout,attrs={{scale={}}}",
                     format_float(get_scale()));
}


}  // namespace ncnn_frontend
