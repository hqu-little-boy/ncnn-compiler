#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

enum class ConvQuantizationMode { None, Dequantize, Requantize };

class Conv2DOp {
 public:
  Conv2DOp(std::int64_t kernel_height,
           std::int64_t kernel_width,
           std::int64_t stride_height,
           std::int64_t stride_width,
           std::int64_t dilation_height,
           std::int64_t dilation_width,
           std::int64_t pad_top,
           std::int64_t pad_bottom,
           std::int64_t pad_left,
           std::int64_t pad_right,
           bool has_bias,
           std::int64_t int8_scale_term = 0) noexcept;

  std::int64_t get_kernel_height() const noexcept;
  std::int64_t get_kernel_width() const noexcept;
  std::int64_t get_stride_height() const noexcept;
  std::int64_t get_stride_width() const noexcept;
  std::int64_t get_dilation_height() const noexcept;
  std::int64_t get_dilation_width() const noexcept;
  std::int64_t get_pad_top() const noexcept;
  std::int64_t get_pad_bottom() const noexcept;
  std::int64_t get_pad_left() const noexcept;
  std::int64_t get_pad_right() const noexcept;
  bool get_has_bias() const noexcept;
  std::int64_t get_int8_scale_term() const noexcept;
  ConvQuantizationMode get_quantization_mode() const noexcept;

 private:
  std::int64_t kernel_height_;
  std::int64_t kernel_width_;
  std::int64_t stride_height_;
  std::int64_t stride_width_;
  std::int64_t dilation_height_;
  std::int64_t dilation_width_;
  std::int64_t pad_top_;
  std::int64_t pad_bottom_;
  std::int64_t pad_left_;
  std::int64_t pad_right_;
  bool has_bias_;
  std::int64_t int8_scale_term_;
};

[[nodiscard]] std::expected<std::vector<TensorType>, std::string>
infer_result_types(const Conv2DOp& operation,
                   std::span<const TensorType> operands,
                   std::size_t result_count);

std::string format_attributes(const Conv2DOp& operation);

OperationKind operation_kind(const Conv2DOp&) noexcept;

}  // namespace ncnn_frontend
