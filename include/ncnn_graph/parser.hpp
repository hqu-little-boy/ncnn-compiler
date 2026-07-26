// 参数解析器：移植 ncnn/src/paramdict.cpp 的 ParamDict::load_param 文本逻辑。
// 参考 netron/source/ncnn.js TextParamReader 的 -23300 数组约定。
#pragma once

#include <string>
#include <string_view>

namespace ncnn_graph {

// 把一行层参数串（如 "0=64 1=3 5=1 6=1728" 或 "-23303=5,0.1,0.2,..."）解析进
// ParamDict。 返回错误描述字符串（成功为空）。
[[nodiscard]] std::string parse_layer_params(std::string_view tail,
                                             class ParamDict& out);

}  // namespace ncnn_graph
