#include "ncnn_graph/parser.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ncnn_graph {
namespace {

constexpr std::int64_t kArrayIdBase = 23300;
constexpr int kMaximumParamId = 31;

bool token_is_nonfinite_float(std::string_view token) {
  if (!token.empty() && (token.front() == '-' || token.front() == '+')) {
    token.remove_prefix(1);
  }
  std::string lowercase(token);
  std::ranges::transform(lowercase, lowercase.begin(), [](char character) {
    return static_cast<char>(
      std::tolower(static_cast<unsigned char>(character)));
  });
  return lowercase == "nan" || lowercase == "inf" || lowercase == "infinity";
}

bool token_is_float(std::string_view token) {
  return token_is_nonfinite_float(token) ||
         std::ranges::any_of(token, [](char character) {
           return character == '.' || character == 'e' || character == 'E';
         });
}

bool token_is_string(std::string_view token) {
  return !token.empty() &&
         (std::isalpha(static_cast<unsigned char>(token.front())) ||
          token.front() == '"');
}

template <typename Value>
std::expected<Value, std::string> parse_number(std::string_view token,
                                               std::string_view description) {
  if (token.empty()) {
    return std::unexpected(std::format("empty {}", description));
  }
  Value value{};
  auto [end, error] =
    std::from_chars(token.data(), token.data() + token.size(), value);
  if (error == std::errc::result_out_of_range) {
    return std::unexpected(
      std::format("{} out of range: {}", description, token));
  }
  if (error != std::errc{} || end != token.data() + token.size()) {
    return std::unexpected(std::format("bad {}: {}", description, token));
  }
  if constexpr (std::is_floating_point_v<Value>) {
    if (!std::isfinite(value)) {
      return std::unexpected(
        std::format("non-finite {}: {}", description, token));
    }
  }
  return value;
}

std::expected<std::vector<std::string_view>, std::string> split_array(
  std::string_view text) {
  std::vector<std::string_view> elements;
  std::size_t begin = 0;
  while (true) {
    std::size_t comma = text.find(',', begin);
    std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    std::string_view element = text.substr(begin, end - begin);
    if (element.empty()) {
      return std::unexpected("array contains an empty element");
    }
    elements.push_back(element);
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  return elements;
}

std::expected<int, std::string> decode_param_id(std::string_view token,
                                                bool& explicit_array) {
  auto parsed = parse_number<std::int64_t>(token, "param id");
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  if (*parsed >= 0 && *parsed <= kMaximumParamId) {
    explicit_array = false;
    return static_cast<int>(*parsed);
  }
  constexpr std::int64_t kMinimumArrayId = -kArrayIdBase - kMaximumParamId;
  constexpr std::int64_t kMaximumArrayId = -kArrayIdBase;
  if (*parsed >= kMinimumArrayId && *parsed <= kMaximumArrayId) {
    explicit_array = true;
    return static_cast<int>(-*parsed - kArrayIdBase);
  }
  return std::unexpected(std::format("param id out of range: {}", token));
}

std::expected<ParamValue, std::string> parse_array_value(
  std::span<const std::string_view> elements) {
  bool as_float = std::ranges::any_of(elements, token_is_float);
  if (as_float) {
    std::vector<float> values;
    values.reserve(elements.size());
    for (std::string_view element : elements) {
      auto value = parse_number<float>(element, "float array element");
      if (!value) {
        return std::unexpected(value.error());
      }
      values.push_back(*value);
    }
    return ParamValue::make_float_array(std::move(values));
  }

  std::vector<std::int64_t> values;
  values.reserve(elements.size());
  for (std::string_view element : elements) {
    auto value = parse_number<std::int64_t>(element, "integer array element");
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(*value);
  }
  return ParamValue::make_int_array(std::move(values));
}

std::expected<ParamValue, std::string> parse_value(std::string_view token,
                                                   bool explicit_array) {
  if (token.empty()) {
    return std::unexpected("empty param value");
  }

  auto elements = split_array(token);
  if (!elements) {
    return std::unexpected(elements.error());
  }
  if (explicit_array) {
    auto length = parse_number<std::int64_t>((*elements)[0], "array length");
    if (!length) {
      return std::unexpected(length.error());
    }
    if (*length < 0) {
      return std::unexpected("array length must be non-negative");
    }
    std::size_t data_count = elements->size() - 1;
    if (std::cmp_not_equal(*length, data_count)) {
      return std::unexpected(std::format(
        "array length mismatch: declared {}, got {}", *length, data_count));
    }
    return parse_array_value(
      std::span<const std::string_view>(*elements).subspan(1));
  }

  if (elements->size() > 1) {
    return parse_array_value(*elements);
  }
  if (token_is_nonfinite_float(token)) {
    auto value = parse_number<float>(token, "float param value");
    if (!value) {
      return std::unexpected(value.error());
    }
    return ParamValue::make_float(*value);
  }
  if (token_is_string(token)) {
    return ParamValue::make_string(std::string(token));
  }
  if (token_is_float(token)) {
    auto value = parse_number<float>(token, "float param value");
    if (!value) {
      return std::unexpected(value.error());
    }
    return ParamValue::make_float(*value);
  }
  auto value = parse_number<std::int64_t>(token, "integer param value");
  if (!value) {
    return std::unexpected(value.error());
  }
  return ParamValue::make_int(*value);
}

}  // namespace

std::expected<ParamDict, std::string> parse_layer_params(
  std::string_view tail) {
  ParamDict params;
  std::size_t position = 0;
  while (position < tail.size()) {
    while (position < tail.size() &&
           std::isspace(static_cast<unsigned char>(tail[position]))) {
      ++position;
    }
    if (position >= tail.size()) {
      break;
    }

    std::size_t end = position;
    while (end < tail.size() &&
           !std::isspace(static_cast<unsigned char>(tail[end]))) {
      ++end;
    }
    std::string_view assignment = tail.substr(position, end - position);
    std::size_t equals = assignment.find('=');
    if (equals == std::string_view::npos) {
      return std::unexpected(
        std::format("missing '=' in param: {}", assignment));
    }
    if (assignment.find('=', equals + 1) != std::string_view::npos) {
      return std::unexpected(
        std::format("multiple '=' in param: {}", assignment));
    }

    bool explicit_array = false;
    auto id = decode_param_id(assignment.substr(0, equals), explicit_array);
    if (!id) {
      return std::unexpected(id.error());
    }
    auto value = parse_value(assignment.substr(equals + 1), explicit_array);
    if (!value) {
      return std::unexpected(std::format("param {}: {}", *id, value.error()));
    }
    params.set_value(*id, std::move(*value));
    position = end;
  }
  return params;
}

}  // namespace ncnn_graph
