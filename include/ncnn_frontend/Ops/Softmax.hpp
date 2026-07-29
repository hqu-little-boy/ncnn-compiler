#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class SoftmaxOp {
 public:
  explicit SoftmaxOp(std::int64_t axis) noexcept;
  std::int64_t get_axis() const noexcept;

 private:
  std::int64_t axis_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const SoftmaxOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const SoftmaxOp& operation);

OperationKind operation_kind(const SoftmaxOp&) noexcept;

}  // namespace ncnn_frontend
