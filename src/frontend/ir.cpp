#include "ncnn_frontend/ir.hpp"

#include <utility>
#include <variant>

#include "ncnn_frontend/Ops/Concat.hpp"
#include "ncnn_frontend/Ops/Const.hpp"
#include "ncnn_frontend/Ops/Conv2D.hpp"
#include "ncnn_frontend/Ops/Dropout.hpp"
#include "ncnn_frontend/Ops/Pool2D.hpp"
#include "ncnn_frontend/Ops/Relu.hpp"
#include "ncnn_frontend/Ops/Softmax.hpp"
#include "ncnn_frontend/Ops/Split.hpp"

namespace ncnn_frontend {

GraphInputDef::GraphInputDef(std::size_t input_index) noexcept
  : input_index_(input_index) {}

std::size_t GraphInputDef::get_input_index() const noexcept {
  return input_index_;
}

OpResultDef::OpResultDef(OpId op, std::size_t result_index) noexcept
  : op_(op), result_index_(result_index) {}

OpId OpResultDef::get_op() const noexcept {
  return op_;
}

std::size_t OpResultDef::get_result_index() const noexcept {
  return result_index_;
}

Use::Use(OpId user, std::size_t operand_index) noexcept
  : user_(user), operand_index_(operand_index) {}

OpId Use::get_user() const noexcept {
  return user_;
}

std::size_t Use::get_operand_index() const noexcept {
  return operand_index_;
}

Value::Value(std::string name,
             TensorType type,
             ValueDef definition,
             std::vector<Use> uses)
  : name_(std::move(name)),
    type_(std::move(type)),
    definition_(std::move(definition)),
    uses_(std::move(uses)) {}

std::string_view Value::get_name() const noexcept {
  return name_;
}

const TensorType& Value::get_type() const noexcept {
  return type_;
}

const ValueDef& Value::get_definition() const noexcept {
  return definition_;
}

std::span<const Use> Value::get_uses() const noexcept {
  return uses_;
}

Operation::Operation(std::string name,
                     OperationAttributes attributes,
                     std::vector<ValueId> operands,
                     std::vector<ValueId> results,
                     std::size_t source_layer_index)
  : name_(std::move(name)),
    attributes_(std::move(attributes)),
    operands_(std::move(operands)),
    results_(std::move(results)),
    source_layer_index_(source_layer_index) {}

std::string_view Operation::get_name() const noexcept {
  return name_;
}

OperationKind Operation::get_kind() const noexcept {
  return std::visit(
    [](const auto& operation) -> OperationKind {
      return operation.operation_kind();
    },
    attributes_);
}

const OperationAttributes& Operation::get_attributes() const noexcept {
  return attributes_;
}

std::span<const ValueId> Operation::get_operands() const noexcept {
  return operands_;
}

std::span<const ValueId> Operation::get_results() const noexcept {
  return results_;
}

std::size_t Operation::get_source_layer_index() const noexcept {
  return source_layer_index_;
}

Graph::Graph(std::vector<Operation> operations,
             std::vector<Value> values,
             std::vector<ValueId> inputs,
             std::vector<ValueId> outputs)
  : operations_(std::move(operations)),
    values_(std::move(values)),
    inputs_(std::move(inputs)),
    outputs_(std::move(outputs)) {}

std::span<const Operation> Graph::get_operations() const noexcept {
  return operations_;
}

std::span<const Value> Graph::get_values() const noexcept {
  return values_;
}

std::span<const ValueId> Graph::get_inputs() const noexcept {
  return inputs_;
}

std::span<const ValueId> Graph::get_outputs() const noexcept {
  return outputs_;
}

const Operation& Graph::get_operation(OpId id) const {
  return operations_.at(id.get_index());
}

const Value& Graph::get_value(ValueId id) const {
  return values_.at(id.get_index());
}

std::size_t Graph::operation_count_of(OperationKind kind) const noexcept {
  std::size_t count = 0;
  for (const auto& operation : operations_) {
    if (operation.get_kind() == kind) {
      ++count;
    }
  }
  return count;
}

}  // namespace ncnn_frontend
