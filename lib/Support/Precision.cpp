#include "ncnn-mlir/Support/Precision.hpp"

#include <algorithm>
#include <cctype>
#include <expected>
#include <format>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ncnn_mlir {
namespace {

std::string lowercase(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

bool contains_any(std::string_view value,
                  std::initializer_list<std::string_view> fragments) {
  return std::ranges::any_of(fragments, [value](std::string_view fragment) {
    return value.contains(fragment);
  });
}

bool feature_enabled(const TargetSpec& target,
                     std::initializer_list<std::string_view> names,
                     bool fallback) {
  bool enabled = fallback;
  for (const std::string& feature : target.features) {
    const std::string normalized = lowercase(feature);
    const bool matches = std::ranges::any_of(names, [&](std::string_view name) {
      return normalized == name || normalized == "+" + std::string(name) ||
             normalized == "-" + std::string(name);
    });
    if (matches) {
      enabled = !normalized.starts_with('-');
    }
  }
  return enabled;
}

}  // namespace

std::expected<PrecisionMode, std::string> parse_precision_mode(
  std::string_view value) {
  if (value == "auto") {
    return PrecisionMode::Auto;
  }
  if (value == "f32" || value == "fp32") {
    return PrecisionMode::Float32;
  }
  if (value == "f16" || value == "fp16") {
    return PrecisionMode::Float16;
  }
  if (value == "bf16") {
    return PrecisionMode::BFloat16;
  }
  if (value == "i8" || value == "int8") {
    return PrecisionMode::Int8;
  }
  return std::unexpected(std::format(
    "invalid precision '{}'; expected auto, f32, fp16, bf16, or int8", value));
}

std::string_view precision_mode_name(PrecisionMode mode) noexcept {
  switch (mode) {
    case PrecisionMode::Auto:
      return "auto";
    case PrecisionMode::Float32:
      return "f32";
    case PrecisionMode::Float16:
      return "fp16";
    case PrecisionMode::BFloat16:
      return "bf16";
    case PrecisionMode::Int8:
      return "int8";
  }
  return "auto";
}

TargetCapabilities infer_target_capabilities(const TargetSpec& target) {
  const std::string triple = lowercase(target.triple);
  std::string attributes = lowercase(target.march + " " + target.mcpu);
  for (const std::string& feature : target.features) {
    attributes += " " + lowercase(feature);
  }
  const bool arm = contains_any(triple, {"aarch64", "arm64"}) ||
                   contains_any(attributes, {"armv8", "armv9", "aarch64"});
  const bool x86 = contains_any(triple, {"x86_64", "amd64"}) ||
                   contains_any(attributes, {"x86-64", "x86_64"});

  TargetCapabilities capabilities;
  capabilities.fp16 = feature_enabled(
    target,
    {"fullfp16", "fp16", "avx512fp16"},
    (arm && contains_any(attributes, {"fullfp16", "fp16", "armv8.2"})) ||
      (x86 && attributes.contains("avx512fp16")));
  capabilities.bf16 =
    feature_enabled(target,
                    {"bf16", "avx512bf16"},
                    (arm && attributes.contains("bf16")) ||
                      (x86 && attributes.contains("avx512bf16")));
  capabilities.int8 = feature_enabled(
    target,
    {"dotprod", "i8mm", "avx512vnni", "avxvnni"},
    (arm && contains_any(attributes, {"dotprod", "i8mm"})) ||
      (x86 && contains_any(attributes, {"avx512vnni", "avxvnni"})));
  return capabilities;
}

std::expected<void, std::string> validate_precision_target(
  PrecisionMode mode, const TargetSpec& target) {
  if (mode == PrecisionMode::Auto || mode == PrecisionMode::Float32) {
    return {};
  }
  const TargetCapabilities capabilities = infer_target_capabilities(target);
  const bool supported = mode == PrecisionMode::Float16    ? capabilities.fp16
                         : mode == PrecisionMode::BFloat16 ? capabilities.bf16
                                                           : capabilities.int8;
  if (supported) {
    return {};
  }
  return std::unexpected(std::format(
    "precision {} is not supported by the selected target; specify a matching "
    "--march, --mcpu, or --target-feature",
    precision_mode_name(mode)));
}

}  // namespace ncnn_mlir
