#pragma once

#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/Ops/OpBase.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class ReluOp : public OpBase<ReluOp> {
 public:
  static constexpr OperationKind kind_v = OperationKind::Relu;

  explicit ReluOp(float negative_slope) noexcept;
  float get_negative_slope() const noexcept;

  [[nodiscard]] std::expected<std::vector<TensorType>, std::string>
  infer_result_types(std::span<const TensorType> operands,
                     std::size_t result_count) const;

  std::string format_attributes() const;

 private:
  float negative_slope_;
};

}  // namespace ncnn_frontend
