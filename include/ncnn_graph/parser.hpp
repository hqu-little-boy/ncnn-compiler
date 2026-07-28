// 参数解析器：移植 ncnn/src/paramdict.cpp 的 ParamDict::load_param 文本逻辑。
// 参考 netron/source/ncnn.js TextParamReader 的 -23300 数组约定。
#pragma once

#include "ncnn_graph/graph.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace ncnn_graph {

[[nodiscard]] std::expected<ParamDict, std::string> parse_layer_params(
  std::string_view tail);

}  // namespace ncnn_graph
