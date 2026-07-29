#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class SplitOp {};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const SplitOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const SplitOp& operation);

OperationKind operation_kind(const SplitOp&) noexcept;

}  // namespace ncnn_frontend
