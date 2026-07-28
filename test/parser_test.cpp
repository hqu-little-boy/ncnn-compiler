#include "ncnn_graph/parser.hpp"

#include <cmath>
#include <format>
#include <iostream>
#include <string_view>

namespace {

bool contains(std::string_view text, std::string_view fragment) {
  return text.find(fragment) != std::string_view::npos;
}

}  // namespace

int main() {
  int status = 0;
  auto check = [&](bool condition, std::string_view message) {
    std::cout << std::format("[{}] {}\n", condition ? "PASS" : "FAIL", message);
    if (!condition) {
      status = 1;
    }
  };

  auto params = ncnn_graph::parse_layer_params(
    "0=42 1=-1.25e2 2=relu 3=1,2,3 4=1.0,2,3e-1 "
    "-23305=3,7,8,9 -23306=2,0.5,1.5 0=99");
  check(params.has_value(), "valid scalar and array parameters parse");
  if (params) {
    check(params->get_int(0) == 99, "duplicate IDs use the last value");
    check(std::abs(params->get_float(1) + 125.0f) < 0.001f,
          "exponent float parses completely");
    auto string_value = params->get_string(2);
    check(string_value && string_value->get() == "relu",
          "string value remains owned by ParamDict");

    auto implicit_ints = params->get_int_array(3);
    check(implicit_ints.size() == 3 && implicit_ints[0] == 1 &&
            implicit_ints[2] == 3,
          "implicit integer array parses");
    auto implicit_floats = params->get_float_array(4);
    check(implicit_floats.size() == 3 &&
            std::abs(implicit_floats[2] - 0.3f) < 0.001f,
          "implicit mixed numeric array promotes to float");
    auto explicit_ints = params->get_int_array(5);
    check(explicit_ints.size() == 3 && explicit_ints[1] == 8,
          "encoded integer array ID and length parse");
    auto explicit_floats = params->get_float_array(6);
    check(explicit_floats.size() == 2 && explicit_floats[0] == 0.5f,
          "encoded float array parses");

    check(params->get_float(0, 7.0f) == 7.0f,
          "wrong-kind float lookup returns caller default");
    check(params->get_int(1, 13) == 13,
          "wrong-kind integer lookup returns caller default");
    check(!params->get_string(31), "missing string lookup returns nullopt");
  }

  auto empty_array = ncnn_graph::parse_layer_params("-23300=0");
  check(empty_array && empty_array->get_int_array(0).empty(),
        "zero-length explicit array parses");

  auto expect_failure = [&](std::string_view text,
                            std::string_view error_fragment,
                            std::string_view description) {
    auto result = ncnn_graph::parse_layer_params(text);
    check(!result && contains(result.error(), error_fragment), description);
  };

  expect_failure("0", "missing '='", "missing assignment separator rejected");
  expect_failure(
    "0=1=2", "multiple '='", "multiple assignment separators rejected");
  expect_failure("0=", "empty param value", "empty scalar value rejected");
  expect_failure("0=12junk", "bad integer", "integer suffix garbage rejected");
  expect_failure("0=1.0junk", "bad float", "float suffix garbage rejected");
  expect_failure(
    "0=1,,2", "empty element", "consecutive array commas rejected");
  expect_failure("0=,1", "empty element", "leading array comma rejected");
  expect_failure("0=1,", "empty element", "trailing array comma rejected");
  expect_failure(
    "-23300=2,1", "length mismatch", "explicit array length mismatch rejected");
  expect_failure(
    "-23300=-1", "non-negative", "negative explicit array length rejected");
  expect_failure(
    "32=1", "param id out of range", "ordinary parameter ID above 31 rejected");
  expect_failure("-23332=1,0",
                 "param id out of range",
                 "encoded parameter ID below range rejected");
  expect_failure("999999999999999999999999=1",
                 "out of range",
                 "overflowing parameter ID rejected");
  expect_failure("0=999999999999999999999999",
                 "out of range",
                 "overflowing integer value rejected");
  expect_failure(
    "0=1e9999", "out of range", "overflowing float value rejected");
  expect_failure("0=nan", "non-finite", "non-finite float token rejected");

  return status;
}
