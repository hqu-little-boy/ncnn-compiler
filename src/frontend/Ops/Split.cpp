#include "ncnn_frontend/Ops/Split.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

std::expected<std::vector<TensorType>, std::string>
SplitOp::infer_result_types(

  std::span<const TensorType> operands,
  std::size_t result_count) const {
  if (operands.size() != 1 || result_count < 2) {
    return std::unexpected(
      "Split requires one operand and at least two results");
  }
  return std::vector<TensorType>(result_count, operands[0]);
}

std::string SplitOp::format_attributes() const {
  return "kind=split,attrs={}";
}


}  // namespace ncnn_frontend
