#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/Ops/OpBase.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class ConstOp : public OpBase<ConstOp> {
 public:
  static constexpr OperationKind kind_v = OperationKind::Constant;

  explicit ConstOp(TensorLiteral literal);
  const TensorLiteral& get_literal() const noexcept;

  [[nodiscard]] std::expected<std::vector<TensorType>, std::string>
  infer_result_types(std::span<const TensorType> operands,
                     std::size_t result_count) const;

  std::string format_attributes() const;

 private:
  TensorLiteral literal_;
};

}  // namespace ncnn_frontend
