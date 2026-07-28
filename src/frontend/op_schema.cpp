#include "op_schema.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ncnn_frontend {
namespace {

std::expected<void, std::string> expect_arity(
  std::span<const TensorType> operands,
  std::size_t expected,
  std::size_t results,
  std::size_t expected_results,
  std::string_view operation) {
  if (operands.size() != expected) {
    return std::unexpected(std::format(
      "{} requires {} operands, got {}", operation, expected, operands.size()));
  }
  if (results != expected_results) {
    return std::unexpected(std::format(
      "{} requires {} results, got {}", operation, expected_results, results));
  }
  return {};
}

std::expected<void, std::string> expect_data_type(const TensorType& type,
                                                  std::string_view role) {
  if (type.get_element_type() != ElementType::Float32) {
    return std::unexpected(std::format("{} must be f32", role));
  }
  return {};
}

std::expected<void, std::string> expect_chw(const TensorType& type,
                                            std::string_view role) {
  if (type.get_layout() != TensorLayout::NcnnCHW ||
      type.get_shape().size() != 3) {
    return std::unexpected(std::format("{} must have [C,H,W] layout", role));
  }
  return {};
}

std::expected<void, std::string> expect_positive(std::int64_t value,
                                                 std::string_view name) {
  if (value <= 0) {
    return std::unexpected(std::format("{} must be positive", name));
  }
  return {};
}

std::expected<std::int64_t, std::string> checked_add(
  std::int64_t left, std::int64_t right, std::string_view description) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    return std::unexpected(std::format("{} overflows int64", description));
  }
  return left + right;
}

std::expected<std::int64_t, std::string> checked_multiply(
  std::int64_t left, std::int64_t right, std::string_view description) {
  if (left < 0 || right < 0) {
    return std::unexpected(std::format("{} must not be negative", description));
  }
  if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left) {
    return std::unexpected(std::format("{} overflows int64", description));
  }
  return left * right;
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

std::expected<TensorType, std::string> create_type(
  std::vector<std::int64_t> shape,
  ElementType element_type,
  TensorLayout layout,
  std::string_view operation) {
  auto result = TensorType::create(std::move(shape), element_type, layout);
  if (!result) {
    return std::unexpected(
      std::format("{} result type: {}", operation, result.error()));
  }
  return result;
}

std::expected<std::vector<TensorType>, std::string> infer_constant(
  const ConstOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 0, result_count, 1, "Const");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  return std::vector<TensorType>{operation.get_literal().get_type()};
}

std::expected<std::vector<TensorType>, std::string> infer_convolution(
  const Conv2DOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  const std::size_t minimum_operands = operation.get_has_bias() ? 3 : 2;
  const auto term = operation.get_int8_scale_term();
  if (term != 0 && term != 1 && term != 2 && term != 101 && term != 102) {
    return std::unexpected("Convolution has unsupported int8_scale_term");
  }
  const bool quantized = term != 0;
  const bool requantized = term > 100;
  const std::size_t expected_operands =
    minimum_operands + (quantized ? 2 : 0) + (requantized ? 1 : 0);
  if (operands.size() != expected_operands || result_count != 1) {
    return std::unexpected(
      "Convolution operands do not match bias and quantization mode");
  }
  const auto& input = operands[0];
  if (quantized) {
    if (input.get_element_type() != ElementType::Float32 &&
        input.get_element_type() != ElementType::Int8) {
      return std::unexpected("quantized convolution input must be f32 or i8");
    }
  } else if (input.get_element_type() != ElementType::Float32) {
    return std::unexpected("non-quantized convolution input must be f32");
  }
  auto input_layout = expect_chw(input, "convolution input");
  if (!input_layout) {
    return std::unexpected(input_layout.error());
  }
  const auto& weight = operands[1];
  if (weight.get_layout() != TensorLayout::OIHW ||
      weight.get_shape().size() != 4) {
    return std::unexpected("convolution weight must have [O,I,H,W] layout");
  }
  if (!quantized && weight.get_element_type() == ElementType::Int8) {
    return std::unexpected("non-quantized convolution weight cannot be i8");
  }
  if (quantized && weight.get_element_type() == ElementType::Float16) {
    return std::unexpected("quantized convolution weight must be f32 or i8");
  }
  if (weight.get_element_type() != ElementType::Float32 &&
      weight.get_element_type() != ElementType::Float16 &&
      weight.get_element_type() != ElementType::Int8) {
    return std::unexpected("convolution weight has unsupported element type");
  }
  const auto input_shape = operands[0].get_shape();
  const auto weight_shape = weight.get_shape();
  if (input_shape[0] <= 0 || input_shape[1] <= 0 || input_shape[2] <= 0 ||
      weight_shape[0] <= 0 || weight_shape[1] <= 0 || weight_shape[2] <= 0 ||
      weight_shape[3] <= 0) {
    return std::unexpected(
      "convolution input and weight dimensions must be positive");
  }
  if (weight_shape[1] != input_shape[0]) {
    return std::unexpected(std::format(
      "convolution input channels {} do not match weight channels {}",
      input_shape[0],
      weight_shape[1]));
  }
  if (weight_shape[2] != operation.get_kernel_height() ||
      weight_shape[3] != operation.get_kernel_width()) {
    return std::unexpected("convolution kernel attributes do not match weight");
  }
  if (operation.get_has_bias()) {
    const auto& bias = operands[2];
    if (bias.get_layout() != TensorLayout::NcnnW ||
        bias.get_shape().size() != 1 ||
        bias.get_shape()[0] != weight_shape[0]) {
      return std::unexpected("convolution bias must have shape [O]");
    }
    if (bias.get_element_type() != ElementType::Float32) {
      return std::unexpected("convolution bias must be f32");
    }
  }
  for (std::size_t index = minimum_operands; index < operands.size(); ++index) {
    const auto& scale = operands[index];
    const std::size_t scale_index = index - minimum_operands;
    const std::int64_t expected_size = scale_index == 0 ? weight_shape[0] : 1;
    if (scale.get_layout() != TensorLayout::NcnnW ||
        scale.get_element_type() != ElementType::Float32 ||
        scale.get_shape().size() != 1 ||
        scale.get_shape()[0] != expected_size) {
      return std::unexpected("convolution scale has the wrong role or shape");
    }
  }
  for (const auto [value, name] :
       {std::pair{operation.get_kernel_height(), "convolution kernel height"},
        std::pair{operation.get_kernel_width(), "convolution kernel width"},
        std::pair{operation.get_stride_height(), "convolution stride height"},
        std::pair{operation.get_stride_width(), "convolution stride width"},
        std::pair{operation.get_dilation_height(),
                  "convolution dilation height"},
        std::pair{operation.get_dilation_width(),
                  "convolution dilation width"}}) {
    auto valid = expect_positive(value, name);
    if (!valid) {
      return std::unexpected(valid.error());
    }
  }
  const std::int64_t pads[] = {operation.get_pad_top(),
                               operation.get_pad_bottom(),
                               operation.get_pad_left(),
                               operation.get_pad_right()};
  bool same_upper = true;
  bool same_lower = true;
  for (const std::int64_t pad : pads) {
    if (pad < 0 && pad != -233 && pad != -234) {
      return std::unexpected("convolution padding has unsupported value");
    }
    same_upper = same_upper && pad == -233;
    same_lower = same_lower && pad == -234;
  }
  if ((!same_upper && !same_lower) &&
      (operation.get_pad_top() < 0 || operation.get_pad_bottom() < 0 ||
       operation.get_pad_left() < 0 || operation.get_pad_right() < 0)) {
    return std::unexpected("convolution SAME padding must use one pad mode");
  }
  auto extent_height = checked_multiply(operation.get_dilation_height(),
                                        operation.get_kernel_height() - 1,
                                        "convolution kernel height extent");
  if (!extent_height) {
    return std::unexpected(extent_height.error());
  }
  extent_height = checked_add(*extent_height, 1, "convolution kernel height");
  if (!extent_height) {
    return std::unexpected(extent_height.error());
  }
  auto extent_width = checked_multiply(operation.get_dilation_width(),
                                       operation.get_kernel_width() - 1,
                                       "convolution kernel width extent");
  if (!extent_width) {
    return std::unexpected(extent_width.error());
  }
  extent_width = checked_add(*extent_width, 1, "convolution kernel width");
  if (!extent_width) {
    return std::unexpected(extent_width.error());
  }
  std::int64_t output_height = 0;
  std::int64_t output_width = 0;
  if (same_upper || same_lower) {
    output_height = 1 + ((input_shape[1] - 1) / operation.get_stride_height());
    output_width = 1 + ((input_shape[2] - 1) / operation.get_stride_width());
  } else {
    auto height = checked_add(
      input_shape[1], operation.get_pad_top(), "convolution padded height");
    if (!height) {
      return std::unexpected(height.error());
    }
    height = checked_add(
      *height, operation.get_pad_bottom(), "convolution padded height");
    if (!height) {
      return std::unexpected(height.error());
    }
    auto width = checked_add(
      input_shape[2], operation.get_pad_left(), "convolution padded width");
    if (!width) {
      return std::unexpected(width.error());
    }
    width = checked_add(
      *width, operation.get_pad_right(), "convolution padded width");
    if (!width) {
      return std::unexpected(width.error());
    }
    if (*height < *extent_height || *width < *extent_width) {
      return std::unexpected("convolution kernel exceeds padded input");
    }
    output_height =
      1 + ((*height - *extent_height) / operation.get_stride_height());
    output_width =
      1 + ((*width - *extent_width) / operation.get_stride_width());
  }
  auto result =
    create_type({weight_shape[0], output_height, output_width},
                requantized ? ElementType::Int8 : ElementType::Float32,
                TensorLayout::NcnnCHW,
                "convolution");
  if (!result) {
    return std::unexpected(result.error());
  }
  return std::vector<TensorType>{std::move(*result)};
}

std::expected<std::vector<TensorType>, std::string> infer_relu(
  const ReluOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "ReLU");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  if (!std::isfinite(operation.get_negative_slope())) {
    return std::unexpected("ReLU negative slope must be finite");
  }
  return std::vector<TensorType>{operands[0]};
}

std::expected<std::vector<TensorType>, std::string> infer_pooling(
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

std::expected<std::vector<TensorType>, std::string> infer_split(
  std::span<const TensorType> operands, std::size_t result_count) {
  if (operands.size() != 1 || result_count < 2) {
    return std::unexpected(
      "Split requires one operand and at least two results");
  }
  return std::vector<TensorType>(result_count, operands[0]);
}

std::expected<std::vector<TensorType>, std::string> infer_concat(
  const ConcatOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  if (operands.size() < 2 || result_count != 1) {
    return std::unexpected(
      "Concat requires at least two operands and one result");
  }
  const auto& first = operands[0];
  const auto first_shape = first.get_shape();
  const auto rank = static_cast<std::int64_t>(first_shape.size());
  std::int64_t axis = operation.get_axis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return std::unexpected("Concat axis is outside operand rank");
  }
  std::vector<std::int64_t> shape(first_shape.begin(), first_shape.end());
  for (std::size_t operand_index = 1; operand_index < operands.size();
       ++operand_index) {
    const auto& operand = operands[operand_index];
    if (operand.get_element_type() != first.get_element_type() ||
        operand.get_layout() != first.get_layout() ||
        operand.get_shape().size() != first_shape.size()) {
      return std::unexpected(
        "Concat operands must have matching types and rank");
    }
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
      if (std::cmp_equal(dimension, axis)) {
        continue;
      }
      if (operand.get_shape()[dimension] != first_shape[dimension]) {
        return std::unexpected("Concat non-axis dimensions must match");
      }
    }
    auto sum = checked_add(shape[static_cast<std::size_t>(axis)],
                           operand.get_shape()[static_cast<std::size_t>(axis)],
                           "Concat axis dimension");
    if (!sum) {
      return std::unexpected(sum.error());
    }
    shape[static_cast<std::size_t>(axis)] = *sum;
  }
  auto result = create_type(
    std::move(shape), first.get_element_type(), first.get_layout(), "Concat");
  if (!result) {
    return std::unexpected(result.error());
  }
  return std::vector<TensorType>{std::move(*result)};
}

std::expected<std::vector<TensorType>, std::string> infer_dropout(
  const DropoutOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "Dropout");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  if (!std::isfinite(operation.get_scale())) {
    return std::unexpected("Dropout scale must be finite");
  }
  return std::vector<TensorType>{operands[0]};
}

std::expected<std::vector<TensorType>, std::string> infer_softmax(
  const SoftmaxOp& operation,
  std::span<const TensorType> operands,
  std::size_t result_count) {
  auto arity = expect_arity(operands, 1, result_count, 1, "Softmax");
  if (!arity) {
    return std::unexpected(arity.error());
  }
  const std::int64_t rank =
    static_cast<std::int64_t>(operands[0].get_shape().size());
  std::int64_t axis = operation.get_axis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return std::unexpected("Softmax axis is outside operand rank");
  }
  return std::vector<TensorType>{operands[0]};
}

}  // namespace

std::expected<std::vector<TensorType>, std::string> infer_and_verify_operation(
  const OperationAttributes& attributes,
  const std::vector<TensorType>& operand_types,
  std::size_t result_count) {
  const auto operands = std::span<const TensorType>(operand_types);
  return std::visit(
    [&](const auto& operation)
      -> std::expected<std::vector<TensorType>, std::string> {
      using Operation = std::decay_t<decltype(operation)>;
      if constexpr (std::is_same_v<Operation, ConstOp>) {
        return infer_constant(operation, operands, result_count);
      } else if constexpr (std::is_same_v<Operation, Conv2DOp>) {
        return infer_convolution(operation, operands, result_count);
      } else if constexpr (std::is_same_v<Operation, ReluOp>) {
        return infer_relu(operation, operands, result_count);
      } else if constexpr (std::is_same_v<Operation, Pool2DOp>) {
        return infer_pooling(operation, operands, result_count);
      } else if constexpr (std::is_same_v<Operation, SplitOp>) {
        return infer_split(operands, result_count);
      } else if constexpr (std::is_same_v<Operation, ConcatOp>) {
        return infer_concat(operation, operands, result_count);
      } else if constexpr (std::is_same_v<Operation, DropoutOp>) {
        return infer_dropout(operation, operands, result_count);
      } else {
        return infer_softmax(operation, operands, result_count);
      }
    },
    attributes);
}

}  // namespace ncnn_frontend
