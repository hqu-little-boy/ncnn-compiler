#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/Ops/OpBase.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class ConcatOp : public OpBase<ConcatOp> {
 public:
  static constexpr OperationKind kind_v = OperationKind::Concat;

  explicit ConcatOp(std::int64_t axis) noexcept;
  std::int64_t get_axis() const noexcept;

  [[nodiscard]] std::expected<std::vector<TensorType>, std::string>
  infer_result_types(std::span<const TensorType> operands,
                     std::size_t result_count) const;

  std::string format_attributes() const;

 private:
  std::int64_t axis_;
};

}  // namespace ncnn_frontend
