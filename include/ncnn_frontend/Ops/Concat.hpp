#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class ConcatOp {
 public:
  explicit ConcatOp(std::int64_t axis) noexcept;
  std::int64_t get_axis() const noexcept;

 private:
  std::int64_t axis_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const ConcatOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const ConcatOp& operation);

OperationKind operation_kind(const ConcatOp&) noexcept;

}  // namespace ncnn_frontend
