#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace ncnn_mlir {

enum class PrecisionMode { Auto, Float32, Float16, BFloat16, Int8 };
enum class FP16AccumulatorMode { Float16, Float32 };

enum class OperatorPrecisionCapability {
  Float32Only,
  FP16Arithmetic,
  LowPrecisionBoundary,
};

struct PrecisionPolicy {
  PrecisionMode mode = PrecisionMode::Auto;
  FP16AccumulatorMode fp16_accumulator = FP16AccumulatorMode::Float16;
  bool used_fallback = false;

  bool preserve_float16_storage() const noexcept {
    return mode == PrecisionMode::Float16;
  }

  bool uses_low_precision_storage() const noexcept {
    return mode == PrecisionMode::Float16 || mode == PrecisionMode::BFloat16;
  }
};

[[nodiscard]] std::expected<PrecisionMode, std::string> parse_precision_mode(
  std::string_view value);
std::string_view precision_mode_name(PrecisionMode mode) noexcept;
[[nodiscard]] std::expected<FP16AccumulatorMode, std::string>
parse_fp16_accumulator_mode(std::string_view value);
std::string_view fp16_accumulator_mode_name(FP16AccumulatorMode mode) noexcept;
OperatorPrecisionCapability operator_precision_capability(
  std::string_view operation) noexcept;

struct TargetSpec {
  std::string triple;
  std::string march;
  std::string mcpu;
  std::vector<std::string> features;
};

struct TargetCapabilities {
  bool fp16_storage = false;
  bool fp16_arithmetic = false;
  bool bf16 = false;
  bool int8 = false;
};

TargetCapabilities infer_target_capabilities(const TargetSpec& target);
std::string target_architecture_name(const TargetSpec& target);
std::string precision_execution_profile(const PrecisionPolicy& policy,
                                        const TargetSpec& target);
[[nodiscard]] std::expected<void, std::string> validate_precision_target(
  PrecisionMode mode, const TargetSpec& target);
[[nodiscard]] std::expected<PrecisionPolicy, std::string>
resolve_precision_policy(PrecisionMode mode,
                         FP16AccumulatorMode accumulator,
                         bool allow_fallback,
                         const TargetSpec& target);

}  // namespace ncnn_mlir
