#pragma once

#include <expected>
#include <string>

#include "ncnn_frontend/ir.hpp"

namespace ncnn_frontend {

[[nodiscard]] std::expected<void, std::string> verify_graph(const Graph& graph);

}  // namespace ncnn_frontend
