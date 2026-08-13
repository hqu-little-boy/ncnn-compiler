#include "ncnn-mlir/Support/Precision.hpp"

#include <gtest/gtest.h>

TEST(PrecisionTest, ParsesSupportedModes) {
  using ncnn_mlir::PrecisionMode;
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("auto"), PrecisionMode::Auto);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("f32"), PrecisionMode::Float32);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("fp16"), PrecisionMode::Float16);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("bf16"), PrecisionMode::BFloat16);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("int8"), PrecisionMode::Int8);
  EXPECT_FALSE(ncnn_mlir::parse_precision_mode("half"));
}

TEST(PrecisionTest, ChecksTargetCapabilities) {
  ncnn_mlir::TargetSpec target{.triple = "x86_64-unknown-linux-gnu",
                               .march = "x86-64-v4",
                               .mcpu = "",
                               .features = {"+avx512fp16", "+avx512vnni"}};
  const auto capabilities = ncnn_mlir::infer_target_capabilities(target);
  EXPECT_TRUE(capabilities.fp16);
  EXPECT_FALSE(capabilities.bf16);
  EXPECT_TRUE(capabilities.int8);
  EXPECT_TRUE(ncnn_mlir::validate_precision_target(
    ncnn_mlir::PrecisionMode::Float16, target));
  EXPECT_FALSE(ncnn_mlir::validate_precision_target(
    ncnn_mlir::PrecisionMode::BFloat16, target));

  target.features.emplace_back("-avx512fp16");
  EXPECT_FALSE(ncnn_mlir::infer_target_capabilities(target).fp16);
}
