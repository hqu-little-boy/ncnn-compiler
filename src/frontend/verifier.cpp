#include "ncnn_frontend/verifier.hpp"

#include "op_schema.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ncnn_frontend {
namespace {

std::expected<void, std::string> verify_tensor_type(const TensorType& type,
                                                    std::string_view context) {
  auto recreated = TensorType::create(
    std::vector<std::int64_t>(type.get_shape().begin(), type.get_shape().end()),
    type.get_element_type(),
    type.get_layout());
  if (!recreated) {
    return std::unexpected(std::format(
      "{} has invalid tensor type: {}", context, recreated.error()));
  }
  if (*recreated != type) {
    return std::unexpected(
      std::format("{} has inconsistent cached tensor size", context));
  }
  return {};
}

std::expected<void, std::string> verify_graph_io(const Graph& graph) {
  const auto values = graph.get_values();
  std::vector<bool> listed_inputs(values.size(), false);
  for (std::size_t index = 0; index < graph.get_inputs().size(); ++index) {
    const ValueId value_id = graph.get_inputs()[index];
    if (value_id.get_index() >= values.size()) {
      return std::unexpected(
        std::format("graph input {} has out-of-range ValueId {}",
                    index,
                    value_id.get_index()));
    }
    if (listed_inputs[value_id.get_index()]) {
      return std::unexpected("graph input list contains a duplicate ValueId");
    }
    listed_inputs[value_id.get_index()] = true;
    const auto* definition = std::get_if<GraphInputDef>(
      &values[value_id.get_index()].get_definition());
    if (definition == nullptr || definition->get_input_index() != index) {
      return std::unexpected(
        std::format("graph input {} definition is inconsistent", index));
    }
  }
  for (std::size_t value_index = 0; value_index < values.size();
       ++value_index) {
    if (std::holds_alternative<GraphInputDef>(
          values[value_index].get_definition()) != listed_inputs[value_index]) {
      return std::unexpected(
        std::format("value {} graph-input definition and input list disagree",
                    value_index));
    }
  }
  std::vector<bool> listed_outputs(values.size(), false);
  for (std::size_t index = 0; index < graph.get_outputs().size(); ++index) {
    const ValueId value_id = graph.get_outputs()[index];
    if (value_id.get_index() >= values.size()) {
      return std::unexpected(
        std::format("graph output {} has out-of-range ValueId {}",
                    index,
                    value_id.get_index()));
    }
    if (listed_outputs[value_id.get_index()]) {
      return std::unexpected("graph output list contains a duplicate ValueId");
    }
    listed_outputs[value_id.get_index()] = true;
  }
  return {};
}

std::expected<void, std::string> verify_results_and_schema(
  const Graph& graph, std::vector<std::vector<Use>>& expected_uses) {
  const auto operations = graph.get_operations();
  const auto values = graph.get_values();
  for (std::size_t op_index = 0; op_index < operations.size(); ++op_index) {
    const auto& operation = operations[op_index];
    std::vector<TensorType> operand_types;
    operand_types.reserve(operation.get_operands().size());
    for (std::size_t operand_index = 0;
         operand_index < operation.get_operands().size();
         ++operand_index) {
      const ValueId value_id = operation.get_operands()[operand_index];
      if (value_id.get_index() >= values.size()) {
        return std::unexpected(
          std::format("operation {} operand {} has out-of-range ValueId {}",
                      op_index,
                      operand_index,
                      value_id.get_index()));
      }
      expected_uses[value_id.get_index()].emplace_back(OpId(op_index),
                                                       operand_index);
      operand_types.push_back(values[value_id.get_index()].get_type());
      if (const auto* definition = std::get_if<OpResultDef>(
            &values[value_id.get_index()].get_definition())) {
        const std::size_t producer = definition->get_op().get_index();
        if (producer >= operations.size()) {
          return std::unexpected(
            std::format("operation {} operand {} has invalid producer {}",
                        op_index,
                        operand_index,
                        producer));
        }
        if (producer >= op_index) {
          return std::unexpected(std::format(
            "operation {} has non-topological dependency on operation {}",
            op_index,
            producer));
        }
      }
    }
    if (const auto* constant =
          std::get_if<ConstOp>(&operation.get_attributes())) {
      const auto& literal = constant->get_literal();
      if (literal.get_data().size() != literal.get_type().get_byte_size()) {
        return std::unexpected(std::format(
          "operation {} constant payload length does not match its type",
          op_index));
      }
    }
    auto inferred = infer_and_verify_operation(operation.get_attributes(),
                                               operand_types,
                                               operation.get_results().size());
    if (!inferred) {
      return std::unexpected(std::format("operation {} ({}): {}",
                                         op_index,
                                         operation.get_name(),
                                         inferred.error()));
    }
    for (std::size_t result_index = 0;
         result_index < operation.get_results().size();
         ++result_index) {
      const ValueId value_id = operation.get_results()[result_index];
      if (value_id.get_index() >= values.size()) {
        return std::unexpected(
          std::format("operation {} result {} has out-of-range ValueId {}",
                      op_index,
                      result_index,
                      value_id.get_index()));
      }
      const auto* definition = std::get_if<OpResultDef>(
        &values[value_id.get_index()].get_definition());
      if (definition == nullptr ||
          definition->get_op().get_index() != op_index ||
          definition->get_result_index() != result_index) {
        return std::unexpected(
          std::format("operation {} result {} definition is inconsistent",
                      op_index,
                      result_index));
      }
      if (values[value_id.get_index()].get_type() !=
          (*inferred)[result_index]) {
        return std::unexpected(std::format(
          "operation {} result {} type does not match schema inference",
          op_index,
          result_index));
      }
    }
  }
  return {};
}

std::expected<void, std::string> verify_unique_definitions(const Graph& graph) {
  const auto operations = graph.get_operations();
  const auto values = graph.get_values();
  std::vector<std::size_t> definitions(values.size(), 0);
  for (const auto& operation : operations) {
    for (const ValueId result : operation.get_results()) {
      if (result.get_index() < definitions.size()) {
        ++definitions[result.get_index()];
      }
    }
  }
  for (std::size_t value_index = 0; value_index < values.size();
       ++value_index) {
    const auto& definition = values[value_index].get_definition();
    if (const auto* result = std::get_if<OpResultDef>(&definition)) {
      if (result->get_op().get_index() >= operations.size()) {
        return std::unexpected(
          std::format("value {} definition has out-of-range OpId {}",
                      value_index,
                      result->get_op().get_index()));
      }
      const auto results =
        operations[result->get_op().get_index()].get_results();
      if (result->get_result_index() >= results.size() ||
          results[result->get_result_index()].get_index() != value_index ||
          definitions[value_index] != 1) {
        return std::unexpected(std::format(
          "value {} does not have exactly one result definition", value_index));
      }
    } else if (definitions[value_index] != 0) {
      return std::unexpected(
        std::format("graph input value {} is also an op result", value_index));
    }
  }
  return {};
}

std::expected<void, std::string> verify_uses(
  const Graph& graph, const std::vector<std::vector<Use>>& expected_uses) {
  for (std::size_t value_index = 0; value_index < graph.get_values().size();
       ++value_index) {
    const auto actual = graph.get_values()[value_index].get_uses();
    if (actual.size() != expected_uses[value_index].size()) {
      return std::unexpected(
        std::format("value {} use-list does not match operands", value_index));
    }
    std::vector<bool> matched(expected_uses[value_index].size(), false);
    for (const auto& use : actual) {
      bool found = false;
      for (std::size_t expected_index = 0;
           expected_index < expected_uses[value_index].size();
           ++expected_index) {
        if (!matched[expected_index] &&
            use == expected_uses[value_index][expected_index]) {
          matched[expected_index] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        return std::unexpected(std::format(
          "value {} use-list does not match operands", value_index));
      }
    }
    for (const auto& use : actual) {
      if (use.get_user().get_index() >= graph.get_operations().size()) {
        return std::unexpected(
          std::format("value {} use has out-of-range user", value_index));
      }
      const auto operands =
        graph.get_operations()[use.get_user().get_index()].get_operands();
      if (use.get_operand_index() >= operands.size() ||
          operands[use.get_operand_index()].get_index() != value_index) {
        return std::unexpected(std::format(
          "value {} use does not point back to operand", value_index));
      }
    }
  }
  return {};
}

}  // namespace

std::expected<void, std::string> verify_graph(const Graph& graph) {
  for (std::size_t value_index = 0; value_index < graph.get_values().size();
       ++value_index) {
    auto type = verify_tensor_type(graph.get_values()[value_index].get_type(),
                                   std::format("value {}", value_index));
    if (!type) {
      return type;
    }
  }
  auto graph_io = verify_graph_io(graph);
  if (!graph_io) {
    return graph_io;
  }
  std::vector<std::vector<Use>> expected_uses(graph.get_values().size());
  auto operations = verify_results_and_schema(graph, expected_uses);
  if (!operations) {
    return operations;
  }
  auto definitions = verify_unique_definitions(graph);
  if (!definitions) {
    return definitions;
  }
  return verify_uses(graph, expected_uses);
}

}  // namespace ncnn_frontend
