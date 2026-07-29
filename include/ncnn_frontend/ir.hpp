#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Operations.hpp"
#include "ncnn_frontend/Types.hpp"

namespace ncnn_frontend {

class GraphInputDef {
 public:
  explicit GraphInputDef(std::size_t input_index) noexcept;
  std::size_t get_input_index() const noexcept;
  bool operator==(const GraphInputDef&) const = default;

 private:
  std::size_t input_index_;
};

class OpResultDef {
 public:
  OpResultDef(OpId op, std::size_t result_index) noexcept;
  OpId get_op() const noexcept;
  std::size_t get_result_index() const noexcept;
  bool operator==(const OpResultDef&) const = default;

 private:
  OpId op_;
  std::size_t result_index_;
};

using ValueDef = std::variant<GraphInputDef, OpResultDef>;

class Use {
 public:
  Use(OpId user, std::size_t operand_index) noexcept;
  OpId get_user() const noexcept;
  std::size_t get_operand_index() const noexcept;
  bool operator==(const Use&) const = default;

 private:
  OpId user_;
  std::size_t operand_index_;
};

class Value {
 public:
  Value(std::string name,
        TensorType type,
        ValueDef definition,
        std::vector<Use> uses);

  std::string_view get_name() const noexcept;
  const TensorType& get_type() const noexcept;
  const ValueDef& get_definition() const noexcept;
  std::span<const Use> get_uses() const noexcept;

 private:
  std::string name_;
  TensorType type_;
  ValueDef definition_;
  std::vector<Use> uses_;
};

class Operation {
 public:
  Operation(std::string name,
            OperationAttributes attributes,
            std::vector<ValueId> operands,
            std::vector<ValueId> results,
            std::size_t source_layer_index);

  std::string_view get_name() const noexcept;
  OperationKind get_kind() const noexcept;
  const OperationAttributes& get_attributes() const noexcept;
  std::span<const ValueId> get_operands() const noexcept;
  std::span<const ValueId> get_results() const noexcept;
  std::size_t get_source_layer_index() const noexcept;

 private:
  std::string name_;
  OperationAttributes attributes_;
  std::vector<ValueId> operands_;
  std::vector<ValueId> results_;
  std::size_t source_layer_index_;
};

class Graph {
 public:
  Graph(std::vector<Operation> operations,
        std::vector<Value> values,
        std::vector<ValueId> inputs,
        std::vector<ValueId> outputs);

  std::span<const Operation> get_operations() const noexcept;
  std::span<const Value> get_values() const noexcept;
  std::span<const ValueId> get_inputs() const noexcept;
  std::span<const ValueId> get_outputs() const noexcept;
  const Operation& get_operation(OpId id) const;
  const Value& get_value(ValueId id) const;
  std::size_t operation_count_of(OperationKind kind) const noexcept;
  std::string dump() const;

 private:
  std::vector<Operation> operations_;
  std::vector<Value> values_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
};

}  // namespace ncnn_frontend
