#include "ncnn_graph/graph.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <string_view>

#include "ncnn_frontend/importer.hpp"
#include "ncnn_frontend/verifier.hpp"

namespace {

bool shape_equals(const ncnn_frontend::TensorType& type,
                  std::span<const std::int64_t> expected) {
  return std::ranges::equal(type.get_shape(), expected);
}

const ncnn_frontend::Operation* find_operation(
  const ncnn_frontend::Graph& graph, std::string_view name) {
  auto iterator = std::ranges::find(
    graph.get_operations(), name, &ncnn_frontend::Operation::get_name);
  return iterator == graph.get_operations().end() ? nullptr : &*iterator;
}

}  // namespace

int main() {
  using ncnn_frontend::import_graph;
  using ncnn_frontend::Operation;
  using ncnn_frontend::OperationKind;
  using ncnn_frontend::OpId;
  using ncnn_frontend::Use;
  using ncnn_frontend::verify_graph;
  int status = 0;
  auto check = [&](bool condition, std::string_view message) {
    std::cout << std::format("[{}] {}\n", condition ? "PASS" : "FAIL", message);
    if (!condition) {
      status = 1;
    }
  };

  auto decoded = ncnn_graph::Graph::load(
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.param",
    NCNN_GRAPH_SOURCE_ROOT "/ncnn/examples/squeezenet_v1.1.bin");
  if (!decoded) {
    std::cerr << decoded.error() << '\n';
    return 1;
  }
  auto imported = import_graph(*decoded);
  if (!imported) {
    std::cerr << imported.error().to_string() << '\n';
    return 1;
  }

  constexpr std::array<std::int64_t, 3> kInputShape = {3, 227, 227};
  constexpr std::array<std::int64_t, 1> kOutputShape = {1000};
  check(
    imported->get_inputs().size() == 1 &&
      shape_equals(imported->get_value(imported->get_inputs()[0]).get_type(),
                   kInputShape),
    "input is one [3,227,227] f32 graph argument");
  check(
    imported->get_outputs().size() == 1 &&
      shape_equals(imported->get_value(imported->get_outputs()[0]).get_type(),
                   kOutputShape),
    "output is one [1000] f32 value");
  check(imported->get_operations().size() == 126,
        "total operation count is 126");
  check(imported->operation_count_of(OperationKind::Constant) == 52,
        "constant operation count is 52");
  check(imported->get_operations().size() -
            imported->operation_count_of(OperationKind::Constant) ==
          74,
        "non-input computation operation count is 74");

  const auto* conv1 = find_operation(*imported, "conv1");
  const auto* pool1 = find_operation(*imported, "pool1");
  const auto* pool3 = find_operation(*imported, "pool3");
  const auto* pool5 = find_operation(*imported, "pool5");
  const auto* conv10 = find_operation(*imported, "conv10");
  const auto* pool10 = find_operation(*imported, "pool10");
  constexpr std::array<std::int64_t, 3> kConv1Shape = {64, 113, 113};
  constexpr std::array<std::int64_t, 3> kPool1Shape = {64, 56, 56};
  constexpr std::array<std::int64_t, 3> kPool3Shape = {128, 28, 28};
  constexpr std::array<std::int64_t, 3> kPool5Shape = {256, 14, 14};
  constexpr std::array<std::int64_t, 3> kConv10Shape = {1000, 16, 16};
  auto result_has_shape = [&](const Operation* operation,
                              std::span<const std::int64_t> shape) {
    return operation != nullptr && operation->get_results().size() == 1 &&
           shape_equals(
             imported->get_value(operation->get_results()[0]).get_type(),
             shape);
  };
  check(result_has_shape(conv1, kConv1Shape), "conv1 shape is [64,113,113]");
  check(result_has_shape(pool1, kPool1Shape), "pool1 shape is [64,56,56]");
  check(result_has_shape(pool3, kPool3Shape), "pool3 shape is [128,28,28]");
  check(result_has_shape(pool5, kPool5Shape), "pool5 shape is [256,14,14]");
  check(result_has_shape(conv10, kConv10Shape),
        "conv10 K1 P1 shape is [1000,16,16]");
  check(result_has_shape(pool10, kOutputShape), "global pool shape is [1000]");

  bool split_results = true;
  bool uses_complete = true;
  for (std::size_t op_index = 0; op_index < imported->get_operations().size();
       ++op_index) {
    const auto& operation = imported->get_operations()[op_index];
    if (operation.get_kind() == OperationKind::Split) {
      split_results = split_results && operation.get_results().size() == 2;
    }
    for (std::size_t operand_index = 0;
         operand_index < operation.get_operands().size();
         ++operand_index) {
      const auto uses =
        imported->get_value(operation.get_operands()[operand_index]).get_uses();
      uses_complete = uses_complete &&
                      std::ranges::find(
                        uses, Use(OpId(op_index), operand_index)) != uses.end();
    }
  }
  check(split_results, "all Split operations preserve two typed results");
  check(uses_complete, "every operand has an explicit matching use");
  check(verify_graph(*imported).has_value(), "full typed graph verifies");

  const std::string dumped = imported->dump();
  check(dumped.starts_with(
          "ncnn_frontend.typed_dag_dump version=1\noperations 126\n"),
        "real model dump has the canonical header and operation count");
  check(dumped.find("\ninputs [v") != std::string::npos &&
          dumped.find("\noutputs [v") != std::string::npos &&
          dumped.ends_with("\n"),
        "real model dump includes graph inputs, outputs, and final newline");
  check(dumped.size() < 1024 * 1024 && dumped.find('\0') == std::string::npos,
        "real model dump is bounded text without expanded payload bytes");

  return status;
}
