#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

enum class PoolKind { Maximum, Average };
enum class PoolMode { Regular, Global, Adaptive };

class Pool2DOp {
 public:
  Pool2DOp(PoolKind kind,
           PoolMode mode,
           std::int64_t kernel_height,
           std::int64_t kernel_width,
           std::int64_t stride_height,
           std::int64_t stride_width,
           std::int64_t pad_top,
           std::int64_t pad_bottom,
           std::int64_t pad_left,
           std::int64_t pad_right,
           int pad_mode,
           bool include_pad) noexcept;

  PoolKind get_kind() const noexcept;
  PoolMode get_mode() const noexcept;
  std::int64_t get_kernel_height() const noexcept;
  std::int64_t get_kernel_width() const noexcept;
  std::int64_t get_stride_height() const noexcept;
  std::int64_t get_stride_width() const noexcept;
  std::int64_t get_pad_top() const noexcept;
  std::int64_t get_pad_bottom() const noexcept;
  std::int64_t get_pad_left() const noexcept;
  std::int64_t get_pad_right() const noexcept;
  int get_pad_mode() const noexcept;
  bool get_include_pad() const noexcept;

 private:
  PoolKind kind_;
  PoolMode mode_;
  std::int64_t kernel_height_;
  std::int64_t kernel_width_;
  std::int64_t stride_height_;
  std::int64_t stride_width_;
  std::int64_t pad_top_;
  std::int64_t pad_bottom_;
  std::int64_t pad_left_;
  std::int64_t pad_right_;
  int pad_mode_;
  bool include_pad_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const Pool2DOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const Pool2DOp& operation);

OperationKind operation_kind(const Pool2DOp&) noexcept;

}  // namespace ncnn_frontend
