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

std::expected<FP16AccumulatorMode, std::string> parse_fp16_accumulator_mode(
  std::string_view value) {
  if (value == "f16" || value == "fp16") {
    return FP16AccumulatorMode::Float16;
  }
  if (value == "f32" || value == "fp32") {
    return FP16AccumulatorMode::Float32;
  }
  return std::unexpected(
    std::format("invalid FP16 accumulator '{}'; expected f16 or f32", value));
}

std::string_view fp16_accumulator_mode_name(FP16AccumulatorMode mode) noexcept {
  return mode == FP16AccumulatorMode::Float16 ? "f16" : "f32";
}

OperatorPrecisionCapability operator_precision_capability(
  std::string_view operation) noexcept {
  if (operation == "ncnn.convolution" ||
      operation == "ncnn.convolution_depthwise") {
    return OperatorPrecisionCapability::FP16Arithmetic;
  }
  if (operation == "ncnn.sigmoid" || operation == "ncnn.hard_sigmoid" ||
      operation == "ncnn.hard_swish" || operation == "ncnn.gelu" ||
      operation == "ncnn.softmax" || operation == "ncnn.batch_norm" ||
      operation == "ncnn.detection_output" ||
      operation == "ncnn.deconvolution" || operation == "ncnn.gemm" ||
      operation == "ncnn.binary" || operation == "ncnn.pooling" ||
      operation == "ncnn.concat") {
    return OperatorPrecisionCapability::LowPrecisionBoundary;
  }
  return OperatorPrecisionCapability::Float32Only;
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
  capabilities.fp16_storage = feature_enabled(
    target,
    {"fullfp16", "fp16", "f16c", "avx512fp16"},
    (arm && contains_any(attributes, {"fullfp16", "fp16", "armv8.2"})) ||
      (x86 && contains_any(attributes, {"f16c", "avx512fp16"})));
  capabilities.fp16_arithmetic = feature_enabled(
    target,
    {"fullfp16", "asimdhp", "avx512fp16", "zvfh", "zfh"},
    (arm && contains_any(attributes, {"fullfp16", "asimdhp", "armv8.2"})) ||
      (x86 && attributes.contains("avx512fp16")) ||
      contains_any(attributes, {"zvfh", "zfh"}));
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
  const bool supported = mode == PrecisionMode::Float16
                           ? capabilities.fp16_storage
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

std::expected<PrecisionPolicy, std::string> resolve_precision_policy(
  PrecisionMode mode,
  FP16AccumulatorMode accumulator,
  bool allow_fallback,
  const TargetSpec& target) {
  if (auto supported = validate_precision_target(mode, target); !supported) {
    return std::unexpected(supported.error());
  }
  PrecisionPolicy policy{.mode = mode, .fp16_accumulator = accumulator};
  if (mode != PrecisionMode::Float16 ||
      accumulator != FP16AccumulatorMode::Float16) {
    return policy;
  }
  if (infer_target_capabilities(target).fp16_arithmetic) {
    return policy;
  }
  if (!allow_fallback) {
    return std::unexpected(
      "FP16 arithmetic is not supported by the selected target; require "
      "AVX512-FP16, ARM ASIMDHP/fullfp16, or RISC-V Zfh/Zvfh, or specify "
      "--allow-fallback");
  }
  policy.fp16_accumulator = FP16AccumulatorMode::Float32;
  policy.used_fallback = true;
  return policy;
}

}  // namespace ncnn_mlir
