#include "ncnn_graph/parser.hpp"

#include "ncnn_graph/graph.hpp"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <string>
#include <string_view>

namespace ncnn_graph {

// 判断 token 是否含小数点/e（ncnn vstr_is_float 的同款逻辑）
static bool token_is_float(std::string_view s) {
  for (char c : s) {
    if (c == '.' || c == 'e' || c == 'E')
      return true;
  }
  return false;
}

static bool token_is_string(std::string_view s) {
  return !s.empty() &&
         (std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '"');
}

static std::int64_t parse_int(std::string_view s, bool& ok) {
  std::int64_t v{};
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  ok = (ec == std::errc{});
  return v;
}

static float parse_float(std::string_view s) {
  // ncnn 用自写 vstr_to_float；这里用 strtod，够用且标准
  std::string tmp(s);
  return static_cast<float>(std::strtod(tmp.c_str(), nullptr));
}

// 切分 "k=v,k=v" 或 "v,v,v" 的数组尾
// 移植自 ncnn ParamDict::load_param：单值后若跟逗号，则整个变成数组
std::string parse_layer_params(std::string_view tail, ParamDict& out) {
  // 逐 token 处理：先按空格切 "id=value"
  std::size_t i = 0;
  while (i < tail.size()) {
    // 跳过空白
    while (i < tail.size() && std::isspace(static_cast<unsigned char>(tail[i])))
      ++i;
    if (i >= tail.size())
      break;

    // 读 id（直到 '='）
    std::size_t eq = tail.find('=', i);
    if (eq == std::string_view::npos) {
      // 无 '='：非法
      return std::string("missing '=' in param token at: ") +
             std::string(tail.substr(i, 16));
    }
    std::string_view id_tok = tail.substr(i, eq - i);
    // trim id_tok
    while (!id_tok.empty() &&
           std::isspace(static_cast<unsigned char>(id_tok.front())))
      id_tok.remove_prefix(1);
    while (!id_tok.empty() &&
           std::isspace(static_cast<unsigned char>(id_tok.back())))
      id_tok.remove_suffix(1);

    bool ok = false;
    std::int64_t id = parse_int(id_tok, ok);
    if (!ok)
      return std::string("bad param id: ") + std::string(id_tok);

    // 负 id 表示数组：ncnn 约定 id <= -23300 → is_array, id = -id - 23300
    bool is_array_neg = (id <= -23300);
    int real_id =
      is_array_neg ? static_cast<int>(-id - 23300) : static_cast<int>(id);

    // 读 value：从 eq+1 开始，到下一个空格或行尾
    std::size_t vstart = eq + 1;
    std::size_t vend = vstart;
    while (vend < tail.size() &&
           !std::isspace(static_cast<unsigned char>(tail[vend])))
      ++vend;
    std::string_view first_val = tail.substr(vstart, vend - vstart);
    i = vend;

    // 看第一个值后面紧跟的是不是逗号（数组展开）。ncnn 文本里数组有两种写法：
    //   (a) -23303=5,0.1,0.2  （显式 length 前缀）
    //   (b) 3=0.1,0.2,0.4     （隐式，逗号续接）
    // 两者值部分都用逗号分隔，且都在同一 token 内（无空格）。所以直接对
    // first_val 做逗号切分。 但注意：ncnn 解析 (b) 时是先读一个无逗号值，再
    // scan 逗号续接——值中无空格， 所以
    // first_val（到空格为止）已包含整段逗号序列。
    bool has_comma = first_val.find(',') != std::string_view::npos;

    if (is_array_neg) {
      // 显式数组：第一个元素是长度
      // 切分 first_val 按逗号
      std::vector<std::string_view> parts;
      std::size_t p = 0;
      while (p < first_val.size()) {
        std::size_t c = first_val.find(',', p);
        if (c == std::string_view::npos) {
          parts.push_back(first_val.substr(p));
          break;
        }
        parts.push_back(first_val.substr(p, c - p));
        p = c + 1;
      }
      if (parts.empty())
        return "empty negative-key array";
      // parts[0] 是长度（ncnn 旧式数组带长度前缀）。但 netron 的处理是：
      //   若 key 为负且 value 是数组，shift 出长度后余下是数据。
      // 这里 parts[0]=len, parts[1..] 是数据。
      std::size_t len = parts.size() > 1
                          ? static_cast<std::size_t>(parse_int(parts[0], ok))
                          : 0;
      // 数据从 parts[1] 开始
      std::vector<float> farr;
      std::vector<std::int64_t> iarr;
      bool as_float = false;
      for (std::size_t k = 1; k < parts.size(); ++k) {
        if (token_is_float(parts[k])) {
          as_float = true;
          break;
        }
      }
      for (std::size_t k = 1;
           k < parts.size() && (len == 0 || farr.size() < len);
           ++k) {
        if (as_float)
          farr.push_back(parse_float(parts[k]));
        else {
          bool ok2 = false;
          iarr.push_back(parse_int(parts[k], ok2));
        }
      }
      if (as_float)
        out.set(real_id, ParamValue::make_float_array(std::move(farr)));
      else
        out.set(real_id, ParamValue::make_int_array(std::move(iarr)));
    } else if (has_comma) {
      // 隐式数组 (b)
      std::vector<std::string_view> parts;
      std::size_t p = 0;
      while (p < first_val.size()) {
        std::size_t c = first_val.find(',', p);
        if (c == std::string_view::npos) {
          parts.push_back(first_val.substr(p));
          break;
        }
        parts.push_back(first_val.substr(p, c - p));
        p = c + 1;
      }
      bool as_float = false;
      for (auto pv : parts)
        if (token_is_float(pv)) {
          as_float = true;
          break;
        }
      std::vector<float> farr;
      std::vector<std::int64_t> iarr;
      for (auto pv : parts) {
        if (as_float)
          farr.push_back(parse_float(pv));
        else {
          bool ok2 = false;
          iarr.push_back(parse_int(pv, ok2));
        }
      }
      if (as_float)
        out.set(real_id, ParamValue::make_float_array(std::move(farr)));
      else
        out.set(real_id, ParamValue::make_int_array(std::move(iarr)));
    } else {
      // 单值
      if (token_is_string(first_val)) {
        out.set(real_id, ParamValue::make_string(std::string(first_val)));
      } else if (token_is_float(first_val)) {
        out.set(real_id, ParamValue::make_float(parse_float(first_val)));
      } else {
        bool ok2 = false;
        std::int64_t v = parse_int(first_val, ok2);
        if (!ok2)
          return std::string("bad int param value: ") + std::string(first_val);
        out.set(real_id, ParamValue::make_int(v));
      }
    }
  }
  return {};
}

}  // namespace ncnn_graph
