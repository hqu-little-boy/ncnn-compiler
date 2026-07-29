#include "op_schema.hpp"

#include <span>
#include <variant>
#include <vector>

#include "ncnn_frontend/Ops/Concat.hpp"
#include "ncnn_frontend/Ops/Const.hpp"
#include "ncnn_frontend/Ops/Conv2D.hpp"
#include "ncnn_frontend/Ops/Dropout.hpp"
#include "ncnn_frontend/Ops/Pool2D.hpp"
#include "ncnn_frontend/Ops/Relu.hpp"
#include "ncnn_frontend/Ops/Softmax.hpp"
#include "ncnn_frontend/Ops/Split.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

std::expected<std::vector<TensorType>, std::string> infer_and_verify_operation(
  const OperationAttributes& attributes,
  const std::vector<TensorType>& operand_types,
  std::size_t result_count) {
  const auto operands = std::span<const TensorType>(operand_types);
  return std::visit(
    [&](const auto& operation)
      -> std::expected<std::vector<TensorType>, std::string> {
      return infer_result_types(operation, operands, result_count);
    },
    attributes);
}

}  // namespace ncnn_frontend
