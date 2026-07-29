#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

// CRTP base for all ncnn ops.
//
// Derived must define:
//   static constexpr OperationKind kind_v;
//   std::string format_attributes() const;
//   std::expected<std::vector<TensorType>, std::string>
//     infer_result_types(std::span<const TensorType>, std::size_t) const;
//
// Provides:
//   operation_kind() — auto-derived from Derived::kind_v
//   elementwise_infer() — protected helper for pass-through shape inference
template <typename Derived>
class OpBase {
 public:
  OperationKind operation_kind() const noexcept {
    return Derived::kind_v;
  }

 protected:
  static std::expected<std::vector<TensorType>, std::string>
  elementwise_infer(std::span<const TensorType> operands,
                    std::size_t result_count) {
    if (operands.size() != 1 || result_count != 1) {
      return std::unexpected(
        "elementwise op requires 1 operand and 1 result");
    }
    return std::vector<TensorType>{operands[0]};
  }
};

}  // namespace ncnn_frontend
