#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <type_traits>

#include "ncnn_frontend/ir.hpp"

namespace ncnn_frontend {
namespace {

template <typename Enum>
std::string invalid_enum(Enum value) {
  return std::format("invalid({})",
                     static_cast<std::underlying_type_t<Enum>>(value));
}

std::string escape_string(std::string_view text) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string result;
  result.reserve(text.size() + 2);
  result.push_back('"');
  for (const unsigned char byte : text) {
    switch (byte) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (byte >= 0x20 && byte <= 0x7E) {
          result.push_back(static_cast<char>(byte));
        } else {
          result += "\\x";
          result.push_back(kHex[byte >> 4]);
          result.push_back(kHex[byte & 0x0F]);
        }
        break;
    }
  }
  result.push_back('"');
  return result;
}

std::string format_float(float value) {
  if (std::isnan(value)) {
    return "nan";
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? "-inf" : "inf";
  }
  if (value == 0.0f) {
    return std::signbit(value) ? "-0" : "0";
  }

  char buffer[64];
  const auto [end, error] =
    std::to_chars(buffer,
                  buffer + sizeof(buffer),
                  value,
                  std::chars_format::general,
                  std::numeric_limits<float>::max_digits10);
  if (error != std::errc()) {
    return "<float-format-error>";
  }
  return {buffer, end};
}

std::string element_type_name(ElementType type) {
  switch (type) {
    case ElementType::Float32:
      return "f32";
    case ElementType::Float16:
      return "f16";
    case ElementType::Int8:
      return "i8";
  }
  return invalid_enum(type);
}

std::string layout_name(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::Scalar:
      return "scalar";
    case TensorLayout::NcnnW:
      return "ncnn_w";
    case TensorLayout::NcnnHW:
      return "ncnn_hw";
    case TensorLayout::NcnnCHW:
      return "ncnn_chw";
    case TensorLayout::NcnnCDHW:
      return "ncnn_cdhw";
    case TensorLayout::OIHW:
      return "oihw";
  }
  return invalid_enum(layout);
}

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

std::string quantization_name(ConvQuantizationMode mode) {
  switch (mode) {
    case ConvQuantizationMode::None:
      return "none";
    case ConvQuantizationMode::Dequantize:
      return "dequantize";
    case ConvQuantizationMode::Requantize:
      return "requantize";
  }
  return invalid_enum(mode);
}

std::string format_shape(std::span<const std::int64_t> shape) {
  std::string result = "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      result.push_back(',');
    }
    result += std::to_string(shape[index]);
  }
  result.push_back(']');
  return result;
}

std::string format_type(const TensorType& type) {
  return std::format("{{shape={},element={},layout={},elements={},bytes={}}}",
                     format_shape(type.get_shape()),
                     element_type_name(type.get_element_type()),
                     layout_name(type.get_layout()),
                     type.get_element_count(),
                     type.get_byte_size());
}

std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const std::byte byte : data) {
    hash ^= std::to_integer<std::uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string format_value_id(ValueId id, std::size_t value_count) {
  const std::string suffix =
    id.get_index() < value_count ? "" : "!out_of_range";
  return std::format("v{}{}", id.get_index(), suffix);
}

std::string format_op_id(OpId id, std::size_t operation_count) {
  const std::string suffix =
    id.get_index() < operation_count ? "" : "!out_of_range";
  return std::format("op{}{}", id.get_index(), suffix);
}

std::string format_value_ids(std::span<const ValueId> ids,
                             std::size_t value_count) {
  std::string result = "[";
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) {
      result.push_back(',');
    }
    result += format_value_id(ids[index], value_count);
  }
  result.push_back(']');
  return result;
}

std::string format_attributes(const OperationAttributes& attributes) {
  if (attributes.valueless_by_exception()) {
    return "kind=invalid,attrs={variant=valueless}";
  }
  return std::visit(
    [](const auto& operation) -> std::string {
      using T = std::decay_t<decltype(operation)>;
      if constexpr (std::is_same_v<T, ConstOp>) {
        const TensorLiteral& literal = operation.get_literal();
        return std::format(
          "kind=const,attrs={{literal_type={},payload_bytes={},fnv1a64=0x{:"
          "016x}}}",
          format_type(literal.get_type()),
          literal.get_data().size(),
          fnv1a64(literal.get_data()));
      } else if constexpr (std::is_same_v<T, Conv2DOp>) {
        return std::format(
          "kind=conv2d,attrs={{kernel=[{},{}],stride=[{},{}],dilation=[{},{}],"
          "pad=[{},{},{},{}],has_bias={},int8_scale_term={},quantization={}}}",
          operation.get_kernel_height(),
          operation.get_kernel_width(),
          operation.get_stride_height(),
          operation.get_stride_width(),
          operation.get_dilation_height(),
          operation.get_dilation_width(),
          operation.get_pad_top(),
          operation.get_pad_bottom(),
          operation.get_pad_left(),
          operation.get_pad_right(),
          operation.get_has_bias(),
          operation.get_int8_scale_term(),
          quantization_name(operation.get_quantization_mode()));
      } else if constexpr (std::is_same_v<T, ReluOp>) {
        return std::format("kind=relu,attrs={{negative_slope={}}}",
                           format_float(operation.get_negative_slope()));
      } else if constexpr (std::is_same_v<T, Pool2DOp>) {
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
      } else if constexpr (std::is_same_v<T, SplitOp>) {
        return "kind=split,attrs={}";
      } else if constexpr (std::is_same_v<T, ConcatOp>) {
        return std::format("kind=concat,attrs={{axis={}}}",
                           operation.get_axis());
      } else if constexpr (std::is_same_v<T, DropoutOp>) {
        return std::format("kind=dropout,attrs={{scale={}}}",
                           format_float(operation.get_scale()));
      } else {
        return std::format("kind=softmax,attrs={{axis={}}}",
                           operation.get_axis());
      }
    },
    attributes);
}

}  // namespace

std::string Graph::dump() const {
  std::string result = "ncnn_frontend.typed_dag_dump version=1\n";
  result += std::format("operations {}\n", operations_.size());
  for (std::size_t index = 0; index < operations_.size(); ++index) {
    const Operation& operation = operations_[index];
    result += std::format(
      "op {} {{{},name={},source_layer={},operands={},results={}}}\n",
      index,
      format_attributes(operation.get_attributes()),
      escape_string(operation.get_name()),
      operation.get_source_layer_index(),
      format_value_ids(operation.get_operands(), values_.size()),
      format_value_ids(operation.get_results(), values_.size()));
  }

  result += std::format("values {}\n", values_.size());
  for (std::size_t index = 0; index < values_.size(); ++index) {
    const Value& value = values_[index];
    std::string definition;
    if (const auto* input =
          std::get_if<GraphInputDef>(&value.get_definition())) {
      definition = std::format("graph_input({})", input->get_input_index());
    } else {
      const auto& op_result = std::get<OpResultDef>(value.get_definition());
      definition =
        std::format("op_result({},{})",
                    format_op_id(op_result.get_op(), operations_.size()),
                    op_result.get_result_index());
    }

    std::string uses = "[";
    for (std::size_t use_index = 0; use_index < value.get_uses().size();
         ++use_index) {
      if (use_index != 0) {
        uses.push_back(',');
      }
      const Use& use = value.get_uses()[use_index];
      uses += std::format("{{user={},operand={}}}",
                          format_op_id(use.get_user(), operations_.size()),
                          use.get_operand_index());
    }
    uses.push_back(']');
    result += std::format("value {} {{name={},type={},def={},uses={}}}\n",
                          index,
                          escape_string(value.get_name()),
                          format_type(value.get_type()),
                          definition,
                          uses);
  }

  result +=
    std::format("inputs {}\n", format_value_ids(inputs_, values_.size()));
  result +=
    std::format("outputs {}\n", format_value_ids(outputs_, values_.size()));
  return result;
}

}  // namespace ncnn_frontend
