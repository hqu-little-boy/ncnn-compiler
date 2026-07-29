#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class ConstOp {
 public:
  explicit ConstOp(TensorLiteral literal);
  const TensorLiteral& get_literal() const noexcept;

 private:
  TensorLiteral literal_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const ConstOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const ConstOp& operation);

OperationKind operation_kind(const ConstOp&) noexcept;

}  // namespace ncnn_frontend
