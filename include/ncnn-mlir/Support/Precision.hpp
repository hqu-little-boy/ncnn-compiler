#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace ncnn_mlir {

enum class PrecisionMode { Auto, Float32, Float16, BFloat16, Int8 };

struct PrecisionPolicy {
  PrecisionMode mode = PrecisionMode::Auto;

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

struct TargetSpec {
  std::string triple;
  std::string march;
  std::string mcpu;
  std::vector<std::string> features;
};

struct TargetCapabilities {
  bool fp16 = false;
  bool bf16 = false;
  bool int8 = false;
};

TargetCapabilities infer_target_capabilities(const TargetSpec& target);
[[nodiscard]] std::expected<void, std::string> validate_precision_target(
  PrecisionMode mode, const TargetSpec& target);

}  // namespace ncnn_mlir
