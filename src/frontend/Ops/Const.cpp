#include "ncnn_frontend/Ops/Const.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {
namespace {

std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const std::byte byte : data) {
    hash ^= std::to_integer<std::uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

ConstOp::ConstOp(TensorLiteral literal) : literal_(std::move(literal)) {}

const TensorLiteral& ConstOp::get_literal() const noexcept {
  return literal_;
}

std::expected<std::vector<TensorType>, std::string>
ConstOp::infer_result_types(std::span<const TensorType> operands,
                            std::size_t result_count) const {
  auto arity = expect_arity(operands, 0, result_count, 1, "Const");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  return std::vector<TensorType>{literal_.get_type()};
}

std::string ConstOp::format_attributes() const {
  return std::format(
    "kind=const,attrs={{literal_type={},payload_bytes={},fnv1a64=0x{:"
    "016x}}}",
    format_type(literal_.get_type()),
    literal_.get_data().size(),
    fnv1a64(literal_.get_data()));
}

}  // namespace ncnn_frontend
