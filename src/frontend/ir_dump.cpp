#include <format>
#include <string>

#include "Ops/FormatSupport.hpp"
#include "ncnn_frontend/Ops/Concat.hpp"
#include "ncnn_frontend/Ops/Const.hpp"
#include "ncnn_frontend/Ops/Conv2D.hpp"
#include "ncnn_frontend/Ops/Dropout.hpp"
#include "ncnn_frontend/Ops/Pool2D.hpp"
#include "ncnn_frontend/Ops/Relu.hpp"
#include "ncnn_frontend/Ops/Softmax.hpp"
#include "ncnn_frontend/Ops/Split.hpp"
#include "ncnn_frontend/ir.hpp"

namespace ncnn_frontend {
namespace {

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

std::string format_operation_attributes(const OperationAttributes& attributes) {
  if (attributes.valueless_by_exception()) {
    return "kind=invalid,attrs={variant=valueless}";
  }
  return std::visit(
    [](const auto& operation) -> std::string {
      return operation.format_attributes();
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
      format_operation_attributes(operation.get_attributes()),
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
