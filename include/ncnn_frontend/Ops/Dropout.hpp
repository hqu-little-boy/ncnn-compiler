#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class DropoutOp {
 public:
  explicit DropoutOp(float scale) noexcept;
  float get_scale() const noexcept;

 private:
  float scale_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const DropoutOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const DropoutOp& operation);

OperationKind operation_kind(const DropoutOp&) noexcept;

}  // namespace ncnn_frontend
