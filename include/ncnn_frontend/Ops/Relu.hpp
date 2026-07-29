#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class ReluOp {
 public:
  explicit ReluOp(float negative_slope) noexcept;
  float get_negative_slope() const noexcept;

 private:
  float negative_slope_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const ReluOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const ReluOp& operation);

OperationKind operation_kind(const ReluOp&) noexcept;

}  // namespace ncnn_frontend
