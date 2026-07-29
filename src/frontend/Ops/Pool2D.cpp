#include "ncnn_frontend/Ops/Pool2D.hpp"

#include "FormatSupport.hpp"
#include "InferSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {
namespace {

std::string pool_kind_name(PoolKind kind) {
  switch (kind) {
    case PoolKind::Maximum:
      return "max";
    case PoolKind::Average:
      return "average";
  }
  return invalid_enum(kind);
}

std::string pool_mode_name(PoolMode mode) {
  switch (mode) {
    case PoolMode::Regular:
      return "regular";
    case PoolMode::Global:
      return "global";
    case PoolMode::Adaptive:
      return "adaptive";
  }
  return invalid_enum(mode);
}

std::expected<std::int64_t, std::string> infer_regular_dimension(
  std::int64_t input,
  std::int64_t kernel,
  std::int64_t stride,
  std::int64_t pad_before,
  std::int64_t pad_after,
  int pad_mode,
  std::string_view name) {
  auto valid_input = expect_positive(input, std::format("{} input", name));
  if (!valid_input) {
    return std::unexpected(valid_input.error());
  }
  auto valid_kernel = expect_positive(kernel, std::format("{} kernel", name));
  if (!valid_kernel) {
    return std::unexpected(valid_kernel.error());
  }
  auto valid_stride = expect_positive(stride, std::format("{} stride", name));
  if (!valid_stride) {
    return std::unexpected(valid_stride.error());
  }
  if (pad_before < 0 || pad_after < 0) {
    return std::unexpected(
      std::format("{} padding must be non-negative", name));
  }
  if (pad_mode == 2 || pad_mode == 3) {
    return 1 + ((input - 1) / stride);
  }
  if (pad_mode == 0) {
    auto numerator = checked_add(input, pad_before, "pool padded dimension");
    if (!numerator) {
      return std::unexpected(numerator.error());
    }
    numerator = checked_add(*numerator, pad_after, "pool padded dimension");
    if (!numerator) {
      return std::unexpected(numerator.error());
    }
    const std::int64_t difference = *numerator - kernel;
    const std::int64_t quotient = difference / stride;
    const std::int64_t remainder = difference % stride;
    return (remainder == 0 ? 1 : 2) + quotient;
  }
  auto numerator = checked_add(input, pad_before, "pool padded dimension");
  if (!numerator) {
    return std::unexpected(numerator.error());
  }
  numerator = checked_add(*numerator, pad_after, "pool padded dimension");
  if (!numerator) {
    return std::unexpected(numerator.error());
  }
  if (*numerator < kernel) {
    return std::unexpected(std::format("{} kernel exceeds padded input", name));
  }
  return 1 + ((*numerator - kernel) / stride);
}

}  // namespace

Pool2DOp::Pool2DOp(PoolKind kind,
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
                   bool include_pad) noexcept
  : kind_(kind),
    mode_(mode),
    kernel_height_(kernel_height),
    kernel_width_(kernel_width),
    stride_height_(stride_height),
    stride_width_(stride_width),
    pad_top_(pad_top),
    pad_bottom_(pad_bottom),
    pad_left_(pad_left),
    pad_right_(pad_right),
    pad_mode_(pad_mode),
    include_pad_(include_pad) {}

PoolKind Pool2DOp::get_kind() const noexcept {
  return kind_;
}
PoolMode Pool2DOp::get_mode() const noexcept {
  return mode_;
}
std::int64_t Pool2DOp::get_kernel_height() const noexcept {
  return kernel_height_;
}
std::int64_t Pool2DOp::get_kernel_width() const noexcept {
  return kernel_width_;
}
std::int64_t Pool2DOp::get_stride_height() const noexcept {
  return stride_height_;
}
std::int64_t Pool2DOp::get_stride_width() const noexcept {
  return stride_width_;
}
std::int64_t Pool2DOp::get_pad_top() const noexcept {
  return pad_top_;
}
std::int64_t Pool2DOp::get_pad_bottom() const noexcept {
  return pad_bottom_;
}
std::int64_t Pool2DOp::get_pad_left() const noexcept {
  return pad_left_;
}
std::int64_t Pool2DOp::get_pad_right() const noexcept {
  return pad_right_;
}
int Pool2DOp::get_pad_mode() const noexcept {
  return pad_mode_;
}
bool Pool2DOp::get_include_pad() const noexcept {
  return include_pad_;
}

std::expected<std::vector<TensorType>, std::string> infer_result_types(
  const Pool2DOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "Pooling");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  auto input = expect_chw(operands[0], "pooling input");
  if (!input) {
    return std::unexpected(input.error());
  }
  auto data_type = expect_data_type(operands[0], "pooling input");
  if (!data_type) {
    return std::unexpected(data_type.error());
  }
  if (operation.get_kind() != PoolKind::Maximum &&
      operation.get_kind() != PoolKind::Average) {
    return std::unexpected("pooling kind is invalid");
  }
  const auto input_shape = operands[0].get_shape();
  if (input_shape[0] <= 0 || input_shape[1] <= 0 || input_shape[2] <= 0) {
    return std::unexpected("pooling input dimensions must be positive");
  }
  if (operation.get_mode() == PoolMode::Global) {
    auto result = create_type({input_shape[0]},
                              operands[0].get_element_type(),
                              TensorLayout::NcnnW,
                              "global pooling");
    if (!result) {
      return std::unexpected(result.error());
    }
    return std::vector<TensorType>{std::move(*result)};
  }
  if (operation.get_mode() == PoolMode::Adaptive) {
    const std::int64_t output_height = operation.get_kernel_height() == -233
                                         ? input_shape[1]
                                         : operation.get_kernel_height();
    const std::int64_t output_width = operation.get_kernel_width() == -233
                                        ? input_shape[2]
                                        : operation.get_kernel_width();
    auto height =
      expect_positive(output_height, "adaptive pooling output height");
    if (!height) {
      return std::unexpected(height.error());
    }
    auto width = expect_positive(output_width, "adaptive pooling output width");
    if (!width) {
      return std::unexpected(width.error());
    }
    auto result = create_type({input_shape[0], output_height, output_width},
                              operands[0].get_element_type(),
                              TensorLayout::NcnnCHW,
                              "adaptive pooling");
    if (!result) {
      return std::unexpected(result.error());
    }
    return std::vector<TensorType>{std::move(*result)};
  }
  if (operation.get_mode() != PoolMode::Regular ||
      operation.get_pad_mode() < 0 || operation.get_pad_mode() > 3) {
    return std::unexpected("pooling mode or pad mode is invalid");
  }
  auto output_height = infer_regular_dimension(input_shape[1],
                                               operation.get_kernel_height(),
                                               operation.get_stride_height(),
                                               operation.get_pad_top(),
                                               operation.get_pad_bottom(),
                                               operation.get_pad_mode(),
                                               "pooling height");
  if (!output_height) {
    return std::unexpected(output_height.error());
  }
  auto output_width = infer_regular_dimension(input_shape[2],
                                              operation.get_kernel_width(),
                                              operation.get_stride_width(),
                                              operation.get_pad_left(),
                                              operation.get_pad_right(),
                                              operation.get_pad_mode(),
                                              "pooling width");
  if (!output_width) {
    return std::unexpected(output_width.error());
  }
  auto result = create_type({input_shape[0], *output_height, *output_width},
                            operands[0].get_element_type(),
                            TensorLayout::NcnnCHW,
                            "pooling");
  if (!result) {
    return std::unexpected(result.error());
  }
  return std::vector<TensorType>{std::move(*result)};
}

std::string format_attributes(const Pool2DOp& operation) {
  return std::format(
    "kind=pool2d,attrs={{kind={},mode={},kernel=[{},{}],stride=[{},{}],"
    "pad=[{},{},{},{}],pad_mode={},include_pad={}}}",
    pool_kind_name(operation.get_kind()),
    pool_mode_name(operation.get_mode()),
    operation.get_kernel_height(),
    operation.get_kernel_width(),
    operation.get_stride_height(),
    operation.get_stride_width(),
    operation.get_pad_top(),
    operation.get_pad_bottom(),
    operation.get_pad_left(),
    operation.get_pad_right(),
    operation.get_pad_mode(),
    operation.get_include_pad());
}

OperationKind operation_kind(const Pool2DOp&) noexcept {
  return OperationKind::Pooling;
}

}  // namespace ncnn_frontend
