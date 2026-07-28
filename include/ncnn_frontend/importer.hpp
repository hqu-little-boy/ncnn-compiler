#pragma once

#include "ncnn_graph/graph.hpp"

#include <cstddef>
#include <expected>
#include <string>
#include <string_view>

#include "ncnn_frontend/ir.hpp"

namespace ncnn_frontend {

class ImportError {
 public:
  ImportError(std::size_t layer_index,
              std::string layer_type,
              std::string layer_name,
              std::string message);

  std::size_t get_layer_index() const noexcept;
  std::string_view get_layer_type() const noexcept;
  std::string_view get_layer_name() const noexcept;
  std::string_view get_message() const noexcept;
  std::string to_string() const;

 private:
  std::size_t layer_index_;
  std::string layer_type_;
  std::string layer_name_;
  std::string message_;
};

[[nodiscard]] std::expected<Graph, ImportError> import_graph(
  const ncnn_graph::Graph& graph);

}  // namespace ncnn_frontend
