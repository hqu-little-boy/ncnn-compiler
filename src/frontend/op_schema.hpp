#pragma once

#include <expected>
#include <string>
#include <vector>

#include "ncnn_frontend/ir.hpp"

namespace ncnn_frontend {

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_and_verify_operation(const OperationAttributes& attributes,
                           const std::vector<TensorType>& operand_types,
                           std::size_t result_count);

}  // namespace ncnn_frontend
